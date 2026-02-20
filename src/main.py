"""
PropFirmPilot — Main orchestrator for fully automated FX trading.

Coordinates the daily trading cycle:
1. Fetch FX data → 2. Run scanner → 3. Multi-agent decision →
4. Compliance check → 5. Execute trade → 6. Monitor equity

Supports three modes:
- Daily cycle (default): sequential scan → decide → execute
- Monitor-only: watch existing positions without opening new ones
- Scheduler: async 24/7 pipeline via SQLite Decision Store (Hybrid EA+LLM)

Usage:
    # Run daily cycle
    python -m src.main --config config/e8_trial_5k.yaml

    # Run with custom date (backtesting mode)
    python -m src.main --config config/e8_trial_5k.yaml --date 2026-02-12

    # Monitor-only mode (no new trades, watch existing positions)
    python -m src.main --config config/e8_trial_5k.yaml --monitor-only

    # Scheduler mode (24/7 async pipeline)
    python -m src.main --config config/e8_trial_5k.yaml --scheduler
"""

import argparse
import asyncio
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from loguru import logger

from src.compliance.prop_firm_guard import AccountSnapshot, PropFirmGuard, TradePlan
from src.config import AppConfig, load_config
from src.decision.agent_bridge import AgentBridge
from src.execution.matchtrader_client import MatchTraderClient
from src.execution.order_manager import OrderManager, TradeSignal
from src.execution.position_sizer import PositionSizer
from src.monitor.alert_service import AlertService
from src.monitor.equity_monitor import EquityMonitor
from src.monitor.memory_journal import MemoryJournal
from src.monitor.telegram_bot import TelegramBotHandler
from src.monitor.trade_journal import TradeJournal
from src.signal.scanner_bridge import ScannerBridge


class PropFirmPilot:
    """Main orchestrator for the prop-firm-pilot trading system.

    Coordinates all subsystems for each daily trading cycle:
    - ScannerBridge: runs qlib_market_scanner for FX signal generation
    - AgentBridge: runs TradingAgents for multi-agent BUY/SELL/HOLD decisions
    - PropFirmGuard: validates compliance with E8 Markets rules
    - MatchTraderClient: executes trades via REST API
    - EquityMonitor: real-time drawdown monitoring
    - TradeJournal: persistent trade logging
    - AlertService: Telegram notifications
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config

        # ── Subsystems ──────────────────────────────────────────────────
        self.scanner = ScannerBridge(
            scanner_path=config.scanner.project_path,
            topk=config.scanner.topk,
            profile="fx",  # Explicitly use FX profile
        )
        self.agents = AgentBridge(
            agents_path=config.agents.project_path,
            selected_analysts=config.agents.selected_analysts,
            config={
                "deep_think_llm": config.agents.deep_think_llm,
                "quick_think_llm": config.agents.quick_think_llm,
                "output_language": config.agents.output_language,
            },
        )
        self.journal = TradeJournal(config.monitor.trade_journal_path)
        self.memory_journal = MemoryJournal(config.monitor.memory_dir)
        self.equity_monitor = EquityMonitor(
            check_interval=config.monitor.equity_check_interval,
            drawdown_alert_pct=config.monitor.drawdown_alert_pct,
            auto_close_pct=config.monitor.auto_close_pct,
        )
        self.alert_service = AlertService(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            account_id=os.getenv("MATCHTRADER_ACCOUNT_ID", ""),
            initial_balance=config.account.initial_balance,
            profit_target_pct=config.compliance.profit_target,
            daily_loss_pct=config.compliance.daily_drawdown_limit,
            max_drawdown_pct=config.compliance.max_drawdown_limit,
        )

        # Build instruments dict for order manager
        instruments_dict: dict[str, dict[str, Any]] = {}
        for symbol, inst_config in config.instruments.items():
            instruments_dict[symbol] = inst_config.model_dump()
        self.order_manager = OrderManager(instruments_dict)

        self.guard = PropFirmGuard(
            config=config.compliance,
            execution_config=config.execution,
            instruments=config.instruments,
        )
        self.sizer = PositionSizer(
            config=config.execution,
            instruments=config.instruments,
        )

        # ── State ───────────────────────────────────────────────────────
        self._matchtrader: MatchTraderClient | None = None

    async def run_daily_cycle(self, date_override: str | None = None) -> None:
        """Execute the full daily trading cycle.

        Args:
            date_override: Override date (YYYY-MM-DD) for testing. None = today.
        """
        today = date_override or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        logger.info("=" * 60)
        logger.info("PropFirmPilot: daily cycle starting for {}", today)
        logger.info("=" * 60)

        async with MatchTraderClient(
            base_url=os.getenv("MATCHTRADER_API_URL", ""),
            email=os.getenv("MATCHTRADER_USERNAME", ""),
            password=os.getenv("MATCHTRADER_PASSWORD", ""),
            broker_id=os.getenv("MATCHTRADER_BROKER_ID", "2"),
            account_id=os.getenv("MATCHTRADER_ACCOUNT_ID"),
            daily_request_limit=self.config.compliance.daily_api_request_limit,
        ) as client:
            self._matchtrader = client

            # Step 1: Authenticate
            try:
                await client.login()
            except Exception as e:
                logger.critical("PropFirmPilot: login failed: {}", e)
                await self.alert_service.system_error(f"Login failed: {e}")
                return

            # Step 2: Check account status
            balance_info = await client.get_balance()
            logger.info(
                "Account: balance=${:,.2f}, equity=${:,.2f}, margin=${:,.2f}",
                balance_info.balance,
                balance_info.equity,
                balance_info.margin,
            )

            # Step 3: Run scanner pipeline
            logger.info("PropFirmPilot: running scanner pipeline...")
            signals = self.scanner.run_pipeline(
                date=today,
                tickers=self.config.symbols,
            )
            if not signals:
                logger.warning("PropFirmPilot: no signals from scanner — skipping trades")
                return

            logger.info("PropFirmPilot: received {} signals", len(signals))

            # Step 4: Run TradingAgents decisions on top signals
            top_signals = signals[: self.config.scanner.topk]
            for signal in top_signals:
                qlib_data = signal.to_qlib_data()
                decision = self.agents.decide(
                    symbol=signal.instrument,
                    trade_date=today,
                    qlib_data=qlib_data,
                )

                if not decision.is_actionable:
                    logger.info("PropFirmPilot: {} → HOLD, skipping", signal.instrument)
                    continue

                # Type narrowing: is_actionable guarantees BUY or SELL
                trade_side: Literal["BUY", "SELL"] = "BUY" if decision.decision == "BUY" else "SELL"

                # Step 5: Execute trade
                await self._execute_trade(
                    client=client,
                    signal=signal,
                    side=trade_side,
                    balance_info=balance_info,
                    agent_decision=decision,
                )

            # Step 6: Log daily summary
            open_positions = await client.get_open_positions()
            self.journal.log_equity_snapshot(
                balance=balance_info.balance,
                equity=balance_info.equity,
                daily_pnl=balance_info.equity - balance_info.balance,
                open_positions=len(open_positions),
            )

            logger.info(
                "PropFirmPilot: daily cycle complete. Open positions: {}",
                len(open_positions),
            )

        self._matchtrader = None

    async def _execute_trade(
        self,
        client: MatchTraderClient,
        signal: Any,
        side: Literal["BUY", "SELL"],
        balance_info: Any,
        agent_decision: Any,
    ) -> None:
        """Execute a single trade with compliance checks."""
        symbol = signal.instrument

        # Calculate SL/TP (using default pip distances for now)
        default_sl_pips = 50.0
        default_tp_pips = 100.0

        # Calculate volume via PositionSizer
        volume = self.sizer.calculate_volume(
            symbol=symbol,
            account_equity=balance_info.equity,
            stop_loss_pips=default_sl_pips,
        )
        risk_amount = self.sizer.calculate_risk_amount(
            symbol=symbol,
            volume=volume,
            stop_loss_pips=default_sl_pips,
        )

        # Build TradePlan
        trade_plan = TradePlan(
            symbol=symbol,
            side=side,
            volume=volume,
            stop_loss=default_sl_pips,
            take_profit=default_tp_pips,
            risk_amount=risk_amount,
        )

        # Build AccountSnapshot
        account = AccountSnapshot(
            balance=balance_info.balance,
            equity=balance_info.equity,
            margin=balance_info.margin,
            free_margin=balance_info.free_margin,
            day_start_balance=balance_info.balance,  # Fallback to current balance
            initial_balance=self.config.account.initial_balance,
            open_positions=self.order_manager.active_count,
            daily_pnl=balance_info.equity - balance_info.balance,
            total_pnl=balance_info.equity - self.config.account.initial_balance,
            equity_high_water_mark=self.config.account.initial_balance,
        )

        # Run compliance checks
        comp_result = self.guard.check_all(trade_plan, account)
        if not comp_result.passed:
            logger.warning(
                "PropFirmPilot: trade rejected for {}: {} ({})",
                symbol,
                comp_result.reason,
                comp_result.rule_name,
            )
            return

        # Anti-duplicate-strategy delay
        delay = self.guard.add_random_delay()
        await asyncio.sleep(delay)

        # Execute
        logger.info(
            "PropFirmPilot: executing {} {} {} lots (risk=${:.2f})",
            side,
            symbol,
            volume,
            risk_amount,
        )

        # Record API call budget usage
        self.guard.record_api_call()
        order = await client.open_position(
            symbol=symbol,
            side=side,
            volume=volume,
        )

        if order.success:
            trade_signal = TradeSignal(
                symbol=symbol,
                side=side,
                score=signal.score,
                confidence=signal.confidence,
                score_gap=signal.score_gap,
            )
            record = self.order_manager.record_open(
                signal=trade_signal,
                position_id=order.position_id,
                volume=volume,
                entry_price=0.0,  # Will be filled from position query
                stop_loss=0.0,
                take_profit=0.0,
                risk_amount=risk_amount,
            )
            self.journal.log_trade(record.model_dump())
            self.memory_journal.log_trade_decision(
                trade_plan=trade_plan,
                signal=signal,
                agent_decision=agent_decision,
            )
            await self.alert_service.trade_opened(symbol, side, volume, 0.0)
        else:
            logger.error("PropFirmPilot: trade failed for {}: {}", symbol, order.message)
            await self.alert_service.system_error(
                f"Trade failed: {side} {symbol} — {order.message}"
            )

    async def run_monitor_only(self) -> None:
        """Monitor-only mode — watch existing positions without opening new ones."""
        logger.info("PropFirmPilot: starting monitor-only mode")

        async with MatchTraderClient(
            base_url=os.getenv("MATCHTRADER_API_URL", ""),
            email=os.getenv("MATCHTRADER_USERNAME", ""),
            password=os.getenv("MATCHTRADER_PASSWORD", ""),
            broker_id=os.getenv("MATCHTRADER_BROKER_ID", "2"),
            account_id=os.getenv("MATCHTRADER_ACCOUNT_ID"),
        ) as client:
            await client.login()
            balance = await client.get_balance()

            async def get_equity() -> float:
                b = await client.get_balance()
                return b.equity

            await self.equity_monitor.start(
                get_equity=get_equity,
                on_alert=self.alert_service.drawdown_warning,
                on_emergency_close=client.close_all_positions,
                day_start_balance=balance.balance,
                initial_balance=self.config.account.initial_balance,
                daily_drawdown_limit=self.config.compliance.daily_drawdown_limit,
                max_drawdown_limit=self.config.compliance.max_drawdown_limit,
            )


async def _run_scheduler(config: AppConfig) -> None:
    """Run the Hybrid EA+LLM async scheduler pipeline.

    Launches all async workers (scanner, LLM, execution, janitor, equity monitor)
    connected via a SQLite Decision Store for 24/7 operation.
    """
    from src.compliance.prop_firm_guard import PropFirmGuard
    from src.decision_store.sqlite_store import DecisionStore
    from src.execution.engine import ExecutionEngine
    from src.execution.instrument_registry import InstrumentRegistry
    from src.execution.position_sizer import PositionSizer
    from src.scheduler.scheduler import Scheduler

    logger.info("PropFirmPilot: starting in SCHEDULER mode (24/7 async pipeline)")

    # Initialize subsystems
    store = DecisionStore(db_path=config.decision_store.db_path)
    scanner = ScannerBridge(
        scanner_path=config.scanner.project_path,
        topk=config.scanner.topk,
        profile="fx",
    )
    agents = AgentBridge(
        agents_path=config.agents.project_path,
        selected_analysts=config.agents.selected_analysts,
        config={
            "deep_think_llm": config.agents.deep_think_llm,
            "quick_think_llm": config.agents.quick_think_llm,
            "output_language": config.agents.output_language,
        },
    )
    alert_service = AlertService(
        bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        account_id=os.getenv("MATCHTRADER_ACCOUNT_ID", ""),
        initial_balance=config.account.initial_balance,
        profit_target_pct=config.compliance.profit_target,
        daily_loss_pct=config.compliance.daily_drawdown_limit,
        max_drawdown_pct=config.compliance.max_drawdown_limit,
    )

    async with MatchTraderClient(
        base_url=os.getenv("MATCHTRADER_API_URL", ""),
        email=os.getenv("MATCHTRADER_USERNAME", ""),
        password=os.getenv("MATCHTRADER_PASSWORD", ""),
        broker_id=os.getenv("MATCHTRADER_BROKER_ID", "2"),
        account_id=os.getenv("MATCHTRADER_ACCOUNT_ID"),
        daily_request_limit=config.compliance.daily_api_request_limit,
    ) as client:
        await client.login()

        # ── Build InstrumentRegistry from effective instruments ────────
        registry = await InstrumentRegistry.from_matchtrader(client, config.symbols)
        logger.info(
            "InstrumentRegistry: {} tradeable, {} untradeable",
            len(registry.tradeable_symbols),
            len(registry.untradeable_symbols),
        )
        await alert_service.send(
            f"🔧 <b>Instrument Registry</b>\n"
            f"• Tradeable: {', '.join(registry.tradeable_symbols) or 'none'}\n"
            f"• Untradeable: {', '.join(registry.untradeable_symbols) or 'none'}"
        )

        guard = PropFirmGuard(config.compliance, config.execution, config.instruments)
        sizer = PositionSizer(config.execution, config.instruments)
        engine = ExecutionEngine(
            store=store,
            guard=guard,
            matchtrader=client,
            sizer=sizer,
            config=config,
            alert_service=alert_service,
            instrument_registry=registry,
        )

        scheduler = Scheduler(
            config=config,
            store=store,
            scanner=scanner,
            agents=agents,
            engine=engine,
            matchtrader=client,
            alert_service=alert_service,
            instrument_registry=registry,
        )

        # ── Telegram bot command handler ───────────────────────────────
        journal = TradeJournal(config.monitor.trade_journal_path)
        bot_handler = TelegramBotHandler(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            alert_service=alert_service,
            trading_client=client,
            trade_journal=journal,
        )

        # ── Startup recovery ───────────────────────────────────────────
        recovered = await scheduler.recover_stale_claims()
        if recovered > 0:
            logger.info("PropFirmPilot: recovered {} stale claims", recovered)

        # ── Signal handlers for graceful shutdown ──────────────────────
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(
                    sig,
                    lambda: asyncio.create_task(scheduler.stop()),
                )
            except NotImplementedError:
                # Windows does not support add_signal_handler for SIGTERM;
                # SIGINT is handled via KeyboardInterrupt fallback below.
                pass

        try:
            # Run scheduler and bot handler concurrently
            await asyncio.gather(
                scheduler.start(),
                bot_handler.start(),
            )
        except KeyboardInterrupt:
            logger.info("PropFirmPilot: KeyboardInterrupt received")
        finally:
            bot_handler.stop()
            await scheduler.stop()
            store.close()
            logger.info("PropFirmPilot: scheduler stopped cleanly")


def setup_logging(config: AppConfig) -> None:
    """Configure loguru logging from config."""
    log_dir = Path(config.logging.file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=config.logging.level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | {message}",
    )
    logger.add(
        config.logging.file,
        level=config.logging.level,
        rotation=config.logging.rotation,
        retention=config.logging.retention,
        encoding="utf-8",
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="PropFirmPilot — Automated FX Trading")
    parser.add_argument(
        "--config",
        default="config/e8_trial_5k.yaml",
        help="Path to account config YAML",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Override date for the trading cycle (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--monitor-only",
        action="store_true",
        help="Monitor-only mode (no new trades)",
    )
    parser.add_argument(
        "--scheduler",
        action="store_true",
        help="Run in scheduler mode (async 24/7 pipeline with SQLite decision store)",
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Load config
    config = load_config(args.config)
    setup_logging(config)

    logger.info("PropFirmPilot v0.1.0 starting")
    logger.info("Config: {}", args.config)
    logger.info("Symbols: {}", config.symbols)

    # Create and run pilot
    pilot = PropFirmPilot(config)

    if args.scheduler:
        asyncio.run(_run_scheduler(config))
    elif args.monitor_only:
        asyncio.run(pilot.run_monitor_only())
    else:
        asyncio.run(pilot.run_daily_cycle(date_override=args.date))


if __name__ == "__main__":
    main()
