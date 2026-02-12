"""
PropFirmPilot — Main orchestrator for fully automated FX trading.

Coordinates the daily trading cycle:
1. Fetch FX data → 2. Run scanner → 3. Multi-agent decision →
4. Compliance check → 5. Execute trade → 6. Monitor equity

Usage:
    # Run daily cycle
    python -m src.main --config config/e8_signature_50k.yaml

    # Run with custom date (backtesting mode)
    python -m src.main --config config/e8_signature_50k.yaml --date 2026-02-12

    # Monitor-only mode (no new trades, watch existing positions)
    python -m src.main --config config/e8_signature_50k.yaml --monitor-only
"""

import argparse
import asyncio
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal

from dotenv import load_dotenv
from loguru import logger

from src.config import AppConfig, load_config
from src.decision.agent_bridge import AgentBridge
from src.execution.matchtrader_client import MatchTraderClient
from src.execution.order_manager import OrderManager, TradeSignal
from src.monitor.alert_service import AlertService
from src.monitor.equity_monitor import EquityMonitor
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
        self.equity_monitor = EquityMonitor(
            check_interval=config.monitor.equity_check_interval,
            drawdown_alert_pct=config.monitor.drawdown_alert_pct,
            auto_close_pct=config.monitor.auto_close_pct,
        )
        self.alert_service = AlertService(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        )

        # Build instruments dict for order manager
        instruments_dict: Dict[str, Dict[str, Any]] = {}
        for symbol, inst_config in config.instruments.items():
            instruments_dict[symbol] = inst_config.model_dump()
        self.order_manager = OrderManager(instruments_dict)

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
            broker_id="e8markets",
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
    ) -> None:
        """Execute a single trade with compliance checks."""
        symbol = signal.instrument

        # Calculate SL/TP (using default pip distances for now)
        default_sl_pips = 50.0
        default_tp_pips = 100.0

        # Simple position sizing: risk 1% of equity
        risk_pct = self.config.execution.default_risk_pct
        risk_amount = balance_info.equity * risk_pct

        # Get instrument config
        inst_config = self.config.instruments.get(symbol)
        if inst_config is None:
            logger.warning("PropFirmPilot: no instrument config for {}, skipping", symbol)
            return

        # Calculate volume
        pip_value = inst_config.pip_value
        volume = risk_amount / (default_sl_pips * pip_value)

        # Apply random offset (anti-duplicate-strategy)
        offset = random.uniform(
            -self.config.execution.position_offset_pct,
            self.config.execution.position_offset_pct,
        )
        volume *= 1 + offset

        # Clamp to lot limits
        volume = max(inst_config.min_lot, min(inst_config.max_lot, volume))
        volume = round(volume, 2)

        # Check position limit
        if self.order_manager.active_count >= self.config.execution.max_positions:
            logger.warning(
                "PropFirmPilot: max positions ({}) reached, skipping {}",
                self.config.execution.max_positions,
                symbol,
            )
            return

        # Anti-duplicate-strategy delay
        delay = random.uniform(
            self.config.execution.random_delay_min,
            self.config.execution.random_delay_max,
        )
        logger.debug("PropFirmPilot: applying random delay {:.1f}s", delay)
        await asyncio.sleep(delay)

        # Execute
        logger.info(
            "PropFirmPilot: executing {} {} {} lots (risk=${:.2f})",
            side,
            symbol,
            volume,
            risk_amount,
        )

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
            broker_id="e8markets",
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
        default="config/e8_signature_50k.yaml",
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

    if args.monitor_only:
        asyncio.run(pilot.run_monitor_only())
    else:
        asyncio.run(pilot.run_daily_cycle(date_override=args.date))


if __name__ == "__main__":
    main()
