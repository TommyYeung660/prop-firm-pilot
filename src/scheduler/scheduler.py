"""
Async multi-cycle orchestrator for the Hybrid EA+LLM pipeline.

Manages concurrent async loops for scanner, LLM workers, execution engine,
janitor, equity monitor, position monitor, and daily summary. Replaces
PropFirmPilot.run_daily_cycle() as the top-level entry point for 24/7 operation.

Includes:
- Graceful shutdown via stop() (triggered by SIGINT/SIGTERM in main.py)
- Startup recovery for stale claims from crashed sessions
- Instrument validation and symbol mapping via InstrumentRegistry
- Position close detection (SL/TP hit monitoring)
- Automated daily summary at configurable UTC hour
- Telegram alert integration for key lifecycle events

Usage:
    scheduler = Scheduler(config, store, scanner, agents, engine, matchtrader)
    await scheduler.recover_stale_claims()
    await scheduler.start()  # Runs until interrupted
"""

import asyncio
import json
from collections.abc import Coroutine
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from src.config import AppConfig
from src.decision.agent_bridge import AgentBridge
from src.decision.decision_formatter import format_decision
from src.decision.schemas import TradeIntent
from src.decision_store.janitor import Janitor
from src.decision_store.sqlite_store import DecisionStore
from src.execution.engine import ExecutionEngine
from src.execution.instrument_registry import InstrumentRegistry
from src.execution.matchtrader_client import MatchTraderClient
from src.monitor.alert_service import AlertService
from src.monitor.equity_monitor import EquityMonitor
from src.signal.scanner_bridge import ScannerBridge


class Scheduler:
    """Async orchestrator managing scanner, LLM workers, and execution engine.

    Runs 5 concurrent async loops on different cadences:
    - Scanner loop (every 4h): generates TradeIntents from market signals
    - LLM worker(s) (continuous, poll 30s): evaluates intents via TradingAgents
    - Execution loop (every 10s): executes approved intents via MatchTrader
    - Janitor loop (every 10min): recycles expired claims, cleans old intents
    - Equity monitor (every 60s): monitors drawdown, triggers emergency close

    Usage:
        scheduler = Scheduler(config, store, scanner, agents, engine, matchtrader)
        await scheduler.recover_stale_claims()
        await scheduler.start()  # Runs until interrupted
    """

    def __init__(
        self,
        config: AppConfig,
        store: DecisionStore,
        scanner: ScannerBridge,
        agents: AgentBridge,
        engine: ExecutionEngine,
        matchtrader: MatchTraderClient,
        alert_service: AlertService | None = None,
        instrument_registry: InstrumentRegistry | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._scanner = scanner
        self._agents = agents
        self._engine = engine
        self._matchtrader = matchtrader
        self._alert_service = alert_service
        self._registry = instrument_registry

        # Internal subsystems
        self._janitor = Janitor(store, config.decision_store.intent_retention_days)
        self._equity_monitor = EquityMonitor(
            check_interval=config.scheduler.equity_poll_interval_seconds,
            drawdown_alert_pct=config.monitor.drawdown_alert_pct,
            auto_close_pct=config.monitor.auto_close_pct,
        )
        self._running = False
        self._daily_summary_sent_date: str = ""  # Track last daily summary date

    # ── Public API ──────────────────────────────────────────────────────

    async def start(self) -> None:
        """Launch all workers as concurrent asyncio tasks."""
        self._running = True
        logger.info("Scheduler: starting all workers")

        tasks: list[Coroutine[Any, Any, None]] = [
            self._scanner_loop(),
            self._execution_loop(),
            self._janitor_loop(),
            self._equity_monitor_loop(),
            self._position_monitor_loop(),
            self._daily_summary_loop(),
        ]
        # Spawn configurable number of LLM workers
        for i in range(self._config.scheduler.llm_worker_count):
            tasks.append(self._llm_worker_loop(worker_id=f"llm-{i}"))

        await asyncio.gather(*tasks)

    async def stop(self) -> None:
        """Signal all workers to stop gracefully."""
        logger.info("Scheduler: stopping all workers")
        self._running = False
        self._equity_monitor.stop()

    async def recover_stale_claims(self) -> int:
        """Recover stale claimed intents from a previous crashed session.

        On startup, any intents stuck in 'claimed' state are from a worker
        that crashed. Recycle them back to timed_out so the Janitor can
        re-queue them or they can be manually reviewed.

        Returns:
            Number of stale claims recovered.
        """
        recycled = await asyncio.to_thread(self._store.recycle_expired_claims)
        if recycled > 0:
            logger.warning("Scheduler: recovered {} stale claims from previous session", recycled)
            await self._send_alert(
                f"🔄 <b>Startup Recovery</b>\n"
                f"• Recovered {recycled} stale claim(s) from previous session"
            )
        else:
            logger.info("Scheduler: no stale claims found — clean startup")
        return recycled

    # ── Worker Loops ────────────────────────────────────────────────────

    async def _scanner_loop(self) -> None:
        """Periodically run the scanner pipeline and create TradeIntents."""
        logger.info("Scanner loop: started")
        while self._running:
            try:
                today = self._today_str()
                logger.info("Scanner loop: starting scan for {}", today)

                signals = await asyncio.to_thread(
                    self._scanner.run_pipeline,
                    date=today,
                    tickers=self._config.symbols,
                )

                for signal in signals[: self._config.scanner.topk]:
                    # Idempotency: skip if intent already exists
                    exists = await asyncio.to_thread(
                        self._store.intent_exists,
                        signal.instrument,
                        today,
                        "scanner",
                    )
                    if exists:
                        logger.debug(
                            "Scanner loop: intent already exists for {}, skipping",
                            signal.instrument,
                        )
                        continue

                    intent = TradeIntent(
                        trade_date=today,
                        symbol=signal.instrument,
                        scanner_score=signal.score,
                        scanner_confidence=signal.confidence,
                        scanner_score_gap=signal.score_gap,
                        scanner_drop_distance=signal.drop_distance,
                        scanner_topk_spread=signal.topk_spread,
                        source="scanner",
                        expires_at=self._now_utc() + timedelta(hours=4),
                    )
                    await asyncio.to_thread(self._store.insert_intent, intent)
                    logger.info("Scanner loop: created intent for {}", signal.instrument)
                    await self._send_alert(
                        f"🔍 <b>Intent Created</b>\n"
                        f"• {signal.instrument} (score={signal.score:.2f}, "
                        f"conf={signal.confidence})"
                    )

            except Exception as e:
                logger.error("Scanner loop error: {}", e)
                await self._send_alert(f"⚠️ <b>Scanner Error</b>\n<code>{e}</code>")

            await asyncio.sleep(self._config.scheduler.scanner_interval_seconds)

    async def _llm_worker_loop(self, worker_id: str) -> None:
        """Continuously claim pending intents and evaluate via LLM agents."""
        logger.info("LLM worker {}: started", worker_id)
        while self._running:
            intent: TradeIntent | None = None
            try:
                intent = await asyncio.to_thread(self._store.claim_next_pending, worker_id)
                if intent is None:
                    await asyncio.sleep(self._config.scheduler.llm_poll_interval_seconds)
                    continue

                logger.info(
                    "LLM worker {}: processing intent {} ({})",
                    worker_id,
                    intent.id,
                    intent.symbol,
                )
                await self._process_claimed_intent(worker_id, intent)

            except Exception as e:
                intent_id = intent.id if intent is not None else "unknown"
                logger.error(
                    "LLM worker {}: error on intent {}: {}",
                    worker_id,
                    intent_id,
                    e,
                )
                # Intent is in "claimed" state — valid transitions are:
                # ready_for_exec, cancelled, timed_out (NOT failed)
                if intent is not None:
                    try:
                        await asyncio.to_thread(
                            self._store.mark_cancelled,
                            intent.id,
                            f"LLM error: {e}",
                        )
                    except Exception:
                        logger.error(
                            "LLM worker {}: failed to cancel intent {}",
                            worker_id,
                            intent.id,
                        )
                await self._send_alert(
                    f"⚠️ <b>LLM Worker Error</b>\n"
                    f"• Worker: {worker_id}\n"
                    f"• Intent: {intent_id}\n"
                    f"• Error: <code>{e}</code>"
                )

    async def _process_claimed_intent(self, worker_id: str, intent: TradeIntent) -> None:
        """Evaluate a claimed intent via LLM agents and update the store."""
        # Build qlib_data from scanner fields
        qlib_data = {
            "score": intent.scanner_score,
            "signal_strength": intent.scanner_confidence,
            "confidence": intent.scanner_confidence,
            "score_gap": intent.scanner_score_gap,
            "drop_distance": intent.scanner_drop_distance,
            "topk_spread": intent.scanner_topk_spread,
        }

        decision = await asyncio.to_thread(
            self._agents.decide,
            symbol=intent.symbol,
            trade_date=intent.trade_date,
            qlib_data=qlib_data,
        )

        if decision.is_actionable:
            # Use format_decision for proper SL/TP calculation
            formatted = format_decision(
                symbol=intent.symbol,
                decision=decision.decision,
                scanner_score=intent.scanner_score,
                scanner_confidence=intent.scanner_confidence,
                agent_state=decision.final_state,
            )
            await asyncio.to_thread(
                self._store.update_intent_decision,
                intent.id,
                decision.decision,
                sl_pips=formatted.suggested_sl_pips,
                tp_pips=formatted.suggested_tp_pips,
                risk_report=decision.risk_report,
                state_json=json.dumps(decision.final_state, default=str),
            )
            await asyncio.to_thread(self._store.mark_ready_for_exec, intent.id)
            logger.info(
                "LLM worker {}: intent {} → {} (ready for execution)",
                worker_id,
                intent.id,
                decision.decision,
            )
        else:
            await asyncio.to_thread(
                self._store.mark_cancelled,
                intent.id,
                f"LLM decided {decision.decision}",
            )
            logger.info(
                "LLM worker {}: intent {} → HOLD (cancelled)",
                worker_id,
                intent.id,
            )

    async def _execution_loop(self) -> None:
        """Periodically process ready_for_exec intents through execution."""
        logger.info("Execution loop: started")
        while self._running:
            try:
                processed = await self._engine.execute_ready_intents()
                if processed > 0:
                    logger.info("Execution loop: processed {} intents", processed)
            except Exception as e:
                logger.error("Execution loop error: {}", e)
                await self._send_alert(f"⚠️ <b>Execution Loop Error</b>\n<code>{e}</code>")

            await asyncio.sleep(self._config.scheduler.execution_poll_interval_seconds)

    async def _janitor_loop(self) -> None:
        """Periodically recycle expired claims and clean old intents."""
        logger.info("Janitor loop: started")
        while self._running:
            try:
                recycled, cleaned = await asyncio.to_thread(self._janitor.run_cycle)
                if recycled > 0 or cleaned > 0:
                    logger.info(
                        "Janitor loop: recycled={}, cleaned={}",
                        recycled,
                        cleaned,
                    )
            except Exception as e:
                logger.error("Janitor loop error: {}", e)

            await asyncio.sleep(self._config.scheduler.janitor_interval_seconds)

    async def _equity_monitor_loop(self) -> None:
        """Start equity monitoring with drawdown alerts."""
        logger.info("Equity monitor loop: started")
        try:

            async def get_equity() -> float:
                balance = await self._matchtrader.get_balance()
                return balance.equity

            balance = await self._matchtrader.get_balance()
            await self._equity_monitor.start(
                get_equity=get_equity,
                day_start_balance=balance.balance,
                initial_balance=self._config.account.initial_balance,
                daily_drawdown_limit=self._config.compliance.daily_drawdown_limit,
                max_drawdown_limit=self._config.compliance.max_drawdown_limit,
            )
        except Exception as e:
            logger.error("Equity monitor loop error: {}", e)

    async def _position_monitor_loop(self) -> None:
        """Detect positions closed by SL/TP/manual and update store + send alerts.

        Polls every position_monitor_interval_seconds. Compares opened intents
        in the store against currently open positions from MatchTrader. When an
        intent's position_id is no longer in the open positions list, the position
        has been closed (SL/TP hit or manual close).
        """
        logger.info("Position monitor loop: started")
        while self._running:
            try:
                # Get intents that are in "opened" state
                opened_intents = await asyncio.to_thread(self._store.get_active_positions)
                if opened_intents:
                    # Get currently open positions from broker
                    open_positions = await self._matchtrader.get_open_positions()
                    open_position_ids = {str(p.position_id) for p in open_positions}

                    for intent in opened_intents:
                        if intent.position_id and intent.position_id not in open_position_ids:
                            # Position was closed (SL/TP/manual)
                            await self._handle_position_closed(intent)
            except Exception as e:
                logger.error("Position monitor loop error: {}", e)
                await self._send_alert(f"⚠️ <b>Position Monitor Error</b>\n<code>{e}</code>")

            await asyncio.sleep(self._config.scheduler.position_monitor_interval_seconds)

    async def _handle_position_closed(self, intent: TradeIntent) -> None:
        """Process a detected position closure — update store and send alert.

        Attempts to fetch the closed position details from MatchTrader for
        PnL information. Falls back gracefully if details are unavailable.

        Args:
            intent: The opened intent whose position is no longer active.
        """
        symbol = intent.symbol
        side = intent.suggested_side or "?"
        position_id = intent.position_id or ""

        logger.info(
            "Position monitor: position {} ({}) closed externally",
            position_id,
            symbol,
        )

        # Mark closed in store
        try:
            await asyncio.to_thread(self._store.mark_closed, intent.id)
        except Exception as e:
            logger.error(
                "Position monitor: failed to mark intent {} closed: {}",
                intent.id,
                e,
            )
            return

        # Try to fetch closed position details for PnL
        pnl = 0.0
        close_price = 0.0
        open_price = 0.0
        volume = 0.0
        hit_type = "manual"  # Default — could be SL, TP, or manual

        try:
            # Search last 24h of closed positions
            now_ms = int(self._now_utc().timestamp() * 1000)
            day_ago_ms = now_ms - 86_400_000
            closed_positions = await self._matchtrader.get_closed_positions(
                from_ts=day_ago_ms, to_ts=now_ms
            )
            # Find matching closed position
            for closed in closed_positions:
                if str(closed.position_id) == position_id:
                    pnl = closed.profit
                    close_price = closed.close_price
                    open_price = closed.open_price
                    volume = closed.volume
                    # Infer hit type from PnL direction vs side
                    # (crude heuristic — TP if profitable, SL if loss)
                    if pnl > 0:
                        hit_type = "TP"
                    elif pnl < 0:
                        hit_type = "SL"
                    break
        except Exception as e:
            logger.warning(
                "Position monitor: could not fetch closed position details for {}: {}",
                position_id,
                e,
            )

        # Convert broker symbol to config symbol for display
        display_symbol = symbol
        if self._registry is not None:
            display_symbol = self._registry.to_config_safe(symbol)

        # Get current equity for alert
        equity: float | None = None
        try:
            balance = await self._matchtrader.get_balance()
            equity = balance.equity
        except Exception:
            pass

        # Send appropriate alert
        if self._alert_service is not None:
            try:
                if hit_type in ("SL", "TP"):
                    await self._alert_service.sl_tp_hit(
                        symbol=display_symbol,
                        side=side,
                        volume=volume,
                        pnl=pnl,
                        hit_type=hit_type,
                        trigger_price=close_price,
                        equity=equity,
                        position_id=position_id,
                    )
                else:
                    await self._alert_service.trade_closed(
                        symbol=display_symbol,
                        side=side,
                        pnl=pnl,
                        reason="Position closed externally",
                        volume=volume,
                        open_price=open_price,
                        close_price=close_price,
                        equity=equity,
                        position_id=position_id,
                    )
            except Exception as e:
                logger.error("Position monitor: alert failed for {}: {}", position_id, e)

    async def _daily_summary_loop(self) -> None:
        """Send a daily summary at the configured UTC hour.

        Checks every 60 seconds whether the current UTC hour matches
        daily_summary_hour_utc and the summary hasn't been sent today yet.
        """
        logger.info("Daily summary loop: started")
        while self._running:
            try:
                now = self._now_utc()
                today_str = now.strftime("%Y-%m-%d")
                target_hour = self._config.scheduler.daily_summary_hour_utc

                if now.hour == target_hour and self._daily_summary_sent_date != today_str:
                    await self._send_daily_summary(today_str)
                    self._daily_summary_sent_date = today_str
            except Exception as e:
                logger.error("Daily summary loop error: {}", e)

            await asyncio.sleep(60)

    # ── Alert Helper ────────────────────────────────────────────────────

    async def _send_daily_summary(self, date_str: str) -> None:
        """Gather account data and send the daily summary alert.

        Args:
            date_str: Today's date in YYYY-MM-DD format.
        """
        if self._alert_service is None:
            return

        try:
            balance_info = await self._matchtrader.get_balance()
            open_positions = await self._matchtrader.get_open_positions()

            # Get today's intents to count trades
            today_intents = await asyncio.to_thread(self._store.get_intents_by_date, date_str)
            trades_today = sum(1 for i in today_intents if i.status in ("opened", "closed"))
            daily_pnl = sum(p.profit for p in open_positions)

            # Estimate day-start balance
            day_start_balance = balance_info.balance - daily_pnl
            daily_dd_pct = (
                abs(daily_pnl) / day_start_balance
                if daily_pnl < 0 and day_start_balance > 0
                else 0.0
            )

            await self._alert_service.daily_summary(
                date=date_str,
                trades=trades_today,
                pnl=daily_pnl,
                equity=balance_info.equity,
                daily_dd_pct=daily_dd_pct,
                open_positions=len(open_positions),
                day_start_balance=day_start_balance,
            )
            logger.info("Daily summary sent for {}", date_str)
        except Exception as e:
            logger.error("Failed to send daily summary: {}", e)
            await self._send_alert(f"⚠️ <b>Daily Summary Error</b>\n<code>{e}</code>")

    async def _send_alert(self, message: str) -> None:
        """Send a Telegram alert if AlertService is configured."""
        if self._alert_service is not None:
            try:
                await self._alert_service.send(message)
            except Exception as e:
                logger.error("Scheduler: failed to send alert: {}", e)

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _today_str() -> str:
        """Return today's date in UTC as YYYY-MM-DD."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _now_utc() -> datetime:
        """Return current UTC datetime."""
        return datetime.now(timezone.utc)
