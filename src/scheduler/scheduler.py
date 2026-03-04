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
from numbers import Real
from typing import Any

from loguru import logger

from src.compliance.best_day_tracker import BestDayTracker
from src.compliance.hwm_tracker import HighWaterMarkTracker
from src.config import AppConfig
from src.decision.agent_bridge import AgentBridge
from src.decision.decision_formatter import format_decision
from src.decision.schemas import TradeIntent
from src.decision_store.janitor import Janitor
from src.decision_store.sqlite_store import DecisionStore, InvalidTransitionError
from src.execution.engine import ExecutionEngine
from src.execution.instrument_registry import InstrumentRegistry
from src.execution.matchtrader_client import MatchTraderClient
from src.monitor.alert_service import AlertService
from src.monitor.equity_monitor import EquityMonitor
from src.monitor.memory_journal import MemoryJournal
from src.monitor.trade_journal import TradeJournal
from src.optimize.optimization_engine import OptimizationEngine
from src.optimize.optimization_state import OptimizationState, Thresholds
from src.scheduler.market_hours import MarketHoursChecker
from src.scheduler.session_cadence import SessionCadence
from src.scheduler.volatility_monitor import VolatilityMonitor
from src.signal.scanner_bridge import ScannerBridge

# ── Constants ──────────────────────────────────────────────────────────────

CONFIDENCE_MAP: dict[str, float] = {"high": 0.9, "medium": 0.6, "low": 0.3}


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
        best_day_tracker: BestDayTracker | None = None,
        optimization_engine: OptimizationEngine | None = None,
        memory_journal: MemoryJournal | None = None,
        trade_journal: TradeJournal | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._scanner = scanner
        self._agents = agents
        self._engine = engine
        self._matchtrader = matchtrader
        self._alert_service = alert_service
        self._registry = instrument_registry
        self._best_day_tracker = best_day_tracker or BestDayTracker(
            best_day_limit=config.compliance.best_day_limit,
            stop_ratio=config.compliance.best_day_stop,
        )

        # Internal subsystems
        self._janitor = Janitor(store, config.decision_store.intent_retention_days)
        self._equity_monitor = EquityMonitor(
            check_interval=config.scheduler.equity_poll_interval_seconds,
            drawdown_alert_pct=config.monitor.drawdown_alert_pct,
            auto_close_pct=config.monitor.auto_close_pct,
        )
        self._running = False
        self._daily_summary_sent_date: str = ""  # Track last daily summary date
        self._best_day_close_positions: dict[str, float] = {}  # pos_id -> unrealized PnL
        self._best_day_tracker_date: str = self._today_str()  # UTC date for daily reset
        self._optimization_engine = optimization_engine
        self._optimization_state: OptimizationState | None = None
        self._memory_journal = memory_journal
        self._trade_journal = trade_journal

        # Dynamic drawdown HWM tracking
        self._hwm_tracker: HighWaterMarkTracker | None = None
        if config.compliance.drawdown_type == "dynamic":
            self._hwm_tracker = HighWaterMarkTracker(
                initial_balance=config.account.initial_balance,
                drawdown_pct=config.compliance.max_drawdown_limit,
                state_path=config.compliance.hwm_state_path,
            )

        # Weekend market closure handling
        self._market_hours = MarketHoursChecker(config.scheduler.market_hours)
        self._weekend_force_close_done = False  # Reset each weekend

        # Phase 2.5: Trailing stop / breakeven tracking
        self._breakeven_applied: set[str] = set()  # position IDs where SL moved to BE

        # Phase 2.6: Re-evaluation tracking
        self._last_reevaluation: dict[str, datetime] = {}  # position_id -> last eval time
        self._reevaluation_close_positions: dict[str, float] = {}  # pos_id -> unrealized PnL
        self._last_known_profit: dict[str, float] = {}  # pos_id -> last polled profit

        # v1.2.0: Event-driven re-scan when a position closes (frees a slot)
        self._rescan_event = asyncio.Event()

        # v1.2.0: Session-aware scanner cadence
        self._session_cadence = SessionCadence(config.scheduler)

        # v1.2.0: Volatility-triggered re-scans
        self._volatility_monitor = VolatilityMonitor(config.scheduler, config.symbols)

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

        # v1.2.0: Volatility monitor loop (if enabled)
        if self._config.scheduler.volatility_trigger_enabled:
            tasks.append(self._volatility_monitor_loop())

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
                # Weekend check — pause during market closure
                await self._wait_for_market_open("Scanner loop")
                today = self._today_str()
                if self._should_pause_new_entries():
                    logger.warning(
                        "Scanner loop: Best Day protection active ({}), pausing new intents",
                        self._best_day_tracker.summary(),
                    )
                    await asyncio.sleep(self._session_cadence.get_scanner_interval(self._now_utc()))
                    continue
                logger.info("Scanner loop: starting scan for {}", today)

                signals = await asyncio.to_thread(
                    self._scanner.run_pipeline,
                    date=today,
                    tickers=self._config.symbols,
                    max_signal_age_days=self._config.scanner.max_signal_age_days,
                )

                # v1.3.0: Early exit when no fresh signals available
                if not signals:
                    logger.warning(
                        "Scanner loop: no signals returned for {} (may be stale or unavailable)",
                        today,
                    )
                    await self._send_alert(
                        f"\u26a0\ufe0f <b>Scanner: No Signals</b>\n"
                        f"No fresh signals for {today}. "
                        f"Skipping intent creation this cycle."
                    )
                    await asyncio.sleep(self._session_cadence.get_scanner_interval(self._now_utc()))
                    continue
                # Per-symbol topk: pick the best signal per symbol, then take topk
                best_per_symbol: dict[str, Any] = {}
                for signal in signals:
                    sym = signal.instrument
                    if sym not in best_per_symbol or signal.score > best_per_symbol[sym].score:
                        best_per_symbol[sym] = signal
                candidates = sorted(best_per_symbol.values(), key=lambda s: s.score, reverse=True)
                topk_signals = candidates[: self._config.scanner.topk]
                logger.info(
                    "Scanner loop: {} signals -> {} symbols -> {} candidates",
                    len(signals),
                    len(best_per_symbol),
                    len(topk_signals),
                )

                # ── Capacity check: avoid creating intents beyond max_positions ──
                max_pos = self._config.execution.max_positions
                open_count = len(await asyncio.to_thread(self._store.get_active_positions))
                pipeline_count = await asyncio.to_thread(self._store.count_pipeline_intents)
                total_occupied = open_count + pipeline_count
                available_slots = max_pos - total_occupied
                if available_slots <= 0:
                    logger.info(
                        "Scanner loop: at capacity ({} open + {} pipeline >= {} max), "
                        "skipping intent creation",
                        open_count,
                        pipeline_count,
                        max_pos,
                    )
                else:
                    created_count = 0
                    for signal in topk_signals:
                        if created_count >= available_slots:
                            logger.info(
                                "Scanner loop: reached available slot limit ({}/{}), "
                                "stopping intent creation",
                                created_count,
                                available_slots,
                            )
                            break
                        # Idempotency: skip if an in-progress intent already exists
                        exists = await asyncio.to_thread(
                            self._store.intent_exists,
                            signal.instrument,
                            today,
                            "scanner",
                        )
                        if exists:
                            logger.info(
                                "Scanner loop: in-progress intent exists for {}, skipping",
                                signal.instrument,
                            )
                            continue

                        # C3 fix: Skip symbols with recent compliance rejection (cooldown)
                        rejection_cooldown = getattr(
                            self._config.scheduler, "rejection_cooldown_minutes", 120
                        )
                        recently_rejected = await asyncio.to_thread(
                            self._store.has_recent_rejection,
                            signal.instrument,
                            today,
                            cooldown_minutes=rejection_cooldown,
                        )
                        if recently_rejected:
                            logger.warning(
                                "Scanner loop: {} was rejected within {}min cooldown, "
                                "skipping to avoid retry loop",
                                signal.instrument,
                                rejection_cooldown,
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
                        created_count += 1
                        self._log_trade_event(
                            "INTENT_CREATED",
                            {
                                "intent_id": intent.id,
                                "symbol": intent.symbol,
                                "trade_date": intent.trade_date,
                                "scanner_score": intent.scanner_score,
                                "scanner_confidence": intent.scanner_confidence,
                            },
                        )
                        logger.info(
                            "Scanner loop: created intent for {} ({}/{})",
                            signal.instrument,
                            created_count,
                            available_slots,
                        )
                        await self._send_alert(
                            f"\U0001f50d <b>Intent Created</b>\n"
                            f"\u2022 {signal.instrument} (score={signal.score:.2f}, "
                            f"conf={signal.confidence})"
                        )

                # v1.2.0: Multi-timeframe — run intraday scan to confirm entry timing
                if self._config.scheduler.multi_timeframe_enabled and topk_signals:
                    try:
                        await self._run_intraday_scan(topk_signals, today)
                    except Exception as e:
                        logger.warning(
                            "Multi-timeframe scan failed (proceeding with daily-only): {}", e
                        )
            except asyncio.CancelledError:
                logger.info("Scanner loop: cancelled")
                return
            except Exception as e:
                logger.error("Scanner loop error: {}", e)
                await self._send_alert(f"⚠️ <b>Scanner Error</b>\n<code>{e}</code>")
            try:
                # v1.2.0: Dynamic interval based on session + rescan event
                scan_interval = self._session_cadence.get_scanner_interval(self._now_utc())
                session_name = self._session_cadence.current_session_name(self._now_utc())
                logger.debug(
                    "Scanner loop: next scan in {}s (session: {})",
                    scan_interval,
                    session_name,
                )
                await asyncio.wait_for(
                    self._rescan_event.wait(),
                    timeout=scan_interval,
                )
                self._rescan_event.clear()
                logger.info("Scanner loop: rescan event received — running early scan")
            except asyncio.TimeoutError:
                pass  # Normal timeout — proceed with scheduled scan
            except asyncio.CancelledError:
                logger.info("Scanner loop: cancelled during sleep")
                return

        logger.info("Scanner loop: stopped")

    async def _llm_worker_loop(self, worker_id: str) -> None:
        """Continuously claim pending intents and evaluate via LLM agents."""
        logger.info("LLM worker {}: started", worker_id)
        while self._running:
            # Weekend check — pause during market closure
            await self._wait_for_market_open("LLM worker")
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

            except asyncio.CancelledError:
                logger.info("LLM worker {}: cancelled", worker_id)
                return

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
                    await self._cancel_intent_safe(
                        worker_id=worker_id,
                        intent_id=intent.id,
                        reason=f"LLM error: {e}",
                        context="worker_error_recovery",
                    )
                await self._send_alert(
                    f"⚠️ <b>LLM Worker Error</b>\n"
                    f"• Worker: {worker_id}\n"
                    f"• Intent: {intent_id}\n"
                    f"• Error: <code>{e}</code>"
                )

        logger.info("LLM worker {}: stopped", worker_id)

    async def _process_claimed_intent(self, worker_id: str, intent: TradeIntent) -> None:
        """Evaluate a claimed intent via LLM agents and update the store."""
        # Block mock-based decisions from executing real trades
        if self._agents.using_mock:
            logger.critical(
                "LLM worker {}: BLOCKING intent {} — AgentBridge is using MockTradingGraph. "
                "Real TradingAgents must be loaded for live trading.",
                worker_id,
                intent.id,
            )
            await self._cancel_intent_safe(
                worker_id=worker_id,
                intent_id=intent.id,
                reason="Mock LLM fallback active — refusing to trade with random decisions",
                context="mock_llm_guard",
            )
            await self._send_alert(
                f"🚫 <b>Trade BLOCKED</b>\n"
                f"• Intent: {intent.symbol}\n"
                f"• Reason: Mock LLM fallback active — TradingAgents import failed"
            )
            return

        # Build qlib_data from scanner fields
        qlib_data = {
            "score": intent.scanner_score,
            "signal_strength": intent.scanner_confidence,
            "confidence": intent.scanner_confidence,
            "score_gap": intent.scanner_score_gap,
            "drop_distance": intent.scanner_drop_distance,
            "topk_spread": intent.scanner_topk_spread,
        }

        thresholds = self._get_thresholds_for_symbol(intent.symbol)
        pre_blended = self._blend_confidence(intent.scanner_confidence, intent.scanner_score)
        if not self._passes_threshold(intent.scanner_confidence, pre_blended, thresholds):
            cancelled = await self._cancel_intent_safe(
                worker_id=worker_id,
                intent_id=intent.id,
                reason="LLM pre-filter: low confidence",
                context="llm_pre_filter",
            )
            if cancelled:
                self._log_trade_event(
                    "INTENT_CANCELLED",
                    {
                        "intent_id": intent.id,
                        "symbol": intent.symbol,
                        "reason": "LLM pre-filter: low confidence",
                    },
                )
                logger.info(
                    "LLM worker {}: intent {} pre-filtered (conf={}, blended={:.2f})",
                    worker_id,
                    intent.id,
                    intent.scanner_confidence,
                    pre_blended,
                )
            return

        decision = await asyncio.to_thread(
            self._agents.decide,
            symbol=intent.symbol,
            trade_date=intent.trade_date,
            qlib_data=qlib_data,
        )
        self._log_trade_event(
            "LLM_DECISION",
            {
                "intent_id": intent.id,
                "symbol": intent.symbol,
                "decision": decision.decision,
                "risk_report": decision.risk_report,
            },
        )
        if self._memory_journal is not None:
            try:
                context = {
                    "intent_id": intent.id,
                    "scanner_score": intent.scanner_score,
                    "scanner_confidence": intent.scanner_confidence,
                    "score_gap": intent.scanner_score_gap,
                    "drop_distance": intent.scanner_drop_distance,
                    "topk_spread": intent.scanner_topk_spread,
                }
                if decision.risk_report:
                    context["risk_report"] = decision.risk_report
                if decision.final_state:
                    context["final_state"] = decision.final_state
                self._memory_journal.log_decision(
                    symbol=intent.symbol,
                    side=decision.decision,
                    decision=decision.decision,
                    context=context,
                )
            except Exception as e:
                logger.warning(
                    "MemoryJournal: failed to log decision for {}: {}",
                    intent.symbol,
                    e,
                )

        if decision.is_actionable:
            if self._should_pause_new_entries():
                await self._cancel_intent_safe(
                    worker_id=worker_id,
                    intent_id=intent.id,
                    reason="Best Day protection active — pausing new entries",
                    context="best_day_pause",
                )
                self._log_trade_event(
                    "INTENT_CANCELLED",
                    {
                        "intent_id": intent.id,
                        "symbol": intent.symbol,
                        "reason": "Best Day protection active — pausing new entries",
                    },
                )
                logger.warning(
                    "LLM worker {}: cancelled intent {} ({}) due to Best Day protection",
                    worker_id,
                    intent.id,
                    intent.symbol,
                )
                return

            # Use format_decision for proper SL/TP calculation
            formatted = format_decision(
                symbol=intent.symbol,
                decision=decision.decision,
                scanner_score=intent.scanner_score,
                scanner_confidence=intent.scanner_confidence,
                agent_state=decision.final_state,
            )
            if not self._passes_threshold(
                intent.scanner_confidence,
                formatted.confidence_score,
                thresholds,
            ):
                cancelled = await self._cancel_intent_safe(
                    worker_id=worker_id,
                    intent_id=intent.id,
                    reason="LLM post-filter: low confidence",
                    context="llm_post_filter",
                )
                if cancelled:
                    self._log_trade_event(
                        "INTENT_CANCELLED",
                        {
                            "intent_id": intent.id,
                            "symbol": intent.symbol,
                            "reason": "LLM post-filter: low confidence",
                        },
                    )
                    logger.info(
                        "LLM worker {}: intent {} post-filtered (conf={}, blended={:.2f})",
                        worker_id,
                        intent.id,
                        intent.scanner_confidence,
                        formatted.confidence_score,
                    )
                return
            try:
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
            except InvalidTransitionError as e:
                latest = await asyncio.to_thread(self._store.get_intent, intent.id)
                latest_status = latest.status if latest is not None else "missing"
                if latest_status != "claimed":
                    logger.warning(
                        "LLM worker {}: stale claim for intent {} (status={}, reason={})",
                        worker_id,
                        intent.id,
                        latest_status,
                        e,
                    )
                    return
                raise
        else:
            cancelled = await self._cancel_intent_safe(
                worker_id=worker_id,
                intent_id=intent.id,
                reason=f"LLM decided {decision.decision}",
                context="hold_decision",
            )
            if cancelled:
                self._log_trade_event(
                    "INTENT_CANCELLED",
                    {
                        "intent_id": intent.id,
                        "symbol": intent.symbol,
                        "reason": f"LLM decided {decision.decision}",
                    },
                )
                logger.info(
                    "LLM worker {}: intent {} → HOLD (cancelled)",
                    worker_id,
                    intent.id,
                )

    async def _cancel_intent_safe(
        self,
        *,
        worker_id: str,
        intent_id: str,
        reason: str,
        context: str,
    ) -> bool:
        """Attempt intent cancellation and tolerate state races.

        Returns:
            True when cancellation succeeded, False when it was skipped/failed.
        """
        try:
            await asyncio.to_thread(self._store.mark_cancelled, intent_id, reason)
            return True
        except InvalidTransitionError as e:
            latest = await asyncio.to_thread(self._store.get_intent, intent_id)
            latest_status = latest.status if latest is not None else "missing"
            logger.warning(
                "LLM worker {}: skip cancel for intent {} during {} (status={}, reason={})",
                worker_id,
                intent_id,
                context,
                latest_status,
                e,
            )
            return False
        except Exception as e:
            logger.error(
                "LLM worker {}: failed to cancel intent {} during {}: {}",
                worker_id,
                intent_id,
                context,
                e,
            )
            return False

    async def _execution_loop(self) -> None:
        """Periodically process ready_for_exec intents through execution."""
        logger.info("Execution loop: started")
        while self._running:
            try:
                # Weekend check — pause during market closure
                await self._wait_for_market_open("Execution loop")
                processed = await self._engine.execute_ready_intents()
                if processed > 0:
                    logger.info("Execution loop: processed {} intents", processed)
            except asyncio.CancelledError:
                logger.info("Execution loop: cancelled")
                return
            except Exception as e:
                logger.error("Execution loop error: {}", e)
                await self._send_alert(f"⚠️ <b>Execution Loop Error</b>\n<code>{e}</code>")
            try:
                await asyncio.sleep(self._config.scheduler.execution_poll_interval_seconds)
            except asyncio.CancelledError:
                logger.info("Execution loop: cancelled during sleep")
                return

        logger.info("Execution loop: stopped")

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
            except asyncio.CancelledError:
                logger.info("Janitor loop: cancelled")
                return
            except Exception as e:
                logger.error("Janitor loop error: {}", e)
            try:
                await asyncio.sleep(self._config.scheduler.janitor_interval_seconds)
            except asyncio.CancelledError:
                logger.info("Janitor loop: cancelled during sleep")
                return

        logger.info("Janitor loop: stopped")

    async def _equity_monitor_loop(self) -> None:
        """Start equity monitoring with drawdown alerts."""
        logger.info("Equity monitor loop: started")
        try:

            async def get_equity() -> float:
                balance = await self._matchtrader.get_balance()
                return balance.equity

            balance = await self._matchtrader.get_balance()
            # For dynamic drawdown, use HWM as the reference for max drawdown
            max_dd_reference = self._config.account.initial_balance
            if self._hwm_tracker is not None:
                max_dd_reference = self._hwm_tracker.high_water_mark

            await self._equity_monitor.start(
                get_equity=get_equity,
                day_start_balance=balance.balance,
                initial_balance=max_dd_reference,  # HWM for dynamic, initial for balance-based
                daily_drawdown_limit=self._config.compliance.daily_drawdown_limit,
                max_drawdown_limit=self._config.compliance.max_drawdown_limit,
            )
        except asyncio.CancelledError:
            logger.info("Equity monitor loop: cancelled")
            return

        except Exception as e:
            logger.error("Equity monitor loop error: {}", e)

    async def _position_monitor_loop(self) -> None:
        """Detect positions closed by SL/TP/manual and update store + send alerts.

        Also monitors the Best Day Rule — if daily PnL approaches the limit,
        proactively closes all winning positions to avoid breaching the rule.

        Polls every position_monitor_interval_seconds. Compares opened intents
        in the store against currently open positions from MatchTrader. When an
        intent's position_id is no longer in the open positions list, the position
        has been closed (SL/TP hit or manual close).
        """
        logger.info("Position monitor loop: started")
        while self._running:
            try:
                # Weekend force-close check
                if (
                    self._market_hours.should_force_close(self._now_utc())
                    and not self._weekend_force_close_done
                ):
                    await self._force_close_for_weekend()
                self._maybe_rollover_best_day_tracker()
                # Get intents that are in "opened" state
                opened_intents = await asyncio.to_thread(self._store.get_active_positions)
                if opened_intents:
                    # Get currently open positions from broker
                    open_positions = await self._matchtrader.get_open_positions()
                    open_position_ids = {str(p.position_id) for p in open_positions}

                    # Update BestDayTracker with current unrealized PnL
                    total_unrealized = sum(p.profit for p in open_positions)
                    self._best_day_tracker.update_unrealized(total_unrealized)

                    # Record last-known profit for each open position (manual_close fallback)
                    for p in open_positions:
                        self._last_known_profit[str(p.position_id)] = p.profit

                    # Check for closed positions (SL/TP/manual)
                    for intent in opened_intents:
                        if intent.position_id and intent.position_id not in open_position_ids:
                            # Position was closed (SL/TP/manual)
                            await self._handle_position_closed(intent)

                    # Best Day Rule: proactively close winners if approaching limit
                    if self._best_day_tracker.should_close_winners() and open_positions:
                        await self._close_winning_positions(open_positions)

                    # Phase 2.5: Trailing stop — move SL to breakeven
                    if open_positions:
                        await self._apply_breakeven_stops(open_positions, opened_intents)

                    # Phase 2.6: Re-evaluate open positions via LLM
                    if open_positions:
                        await self._reevaluate_open_positions(open_positions, opened_intents)

            except asyncio.CancelledError:
                logger.info("Position monitor loop: cancelled")
                return
            except Exception as e:
                logger.error("Position monitor loop error: {}", e)
                await self._send_alert(f"⚠️ <b>Position Monitor Error</b>\n<code>{e}</code>")

            try:
                # Auto-throttle: increase sleep when API budget is low
                base_interval = self._config.scheduler.position_monitor_interval_seconds
                limiter = self._matchtrader._rate_limiter
                remaining = self._coerce_numeric(
                    getattr(limiter, "write_remaining", getattr(limiter, "remaining", 0)),
                    fallback=0.0,
                )
                daily_limit = self._coerce_numeric(
                    getattr(limiter, "daily_write_limit", getattr(limiter, "_daily_limit", 1)),
                    fallback=1.0,
                )
                if daily_limit <= 0:
                    daily_limit = 1.0
                if remaining < daily_limit * 0.15:
                    sleep_interval = base_interval * 4
                    logger.warning(
                        "Position monitor: API budget critical ({}/{} remaining)"
                        " — throttling to {}s interval",
                        remaining,
                        daily_limit,
                        sleep_interval,
                    )
                elif remaining < daily_limit * 0.30:
                    sleep_interval = base_interval * 2
                    logger.info(
                        "Position monitor: API budget low ({}/{} remaining)"
                        " — throttling to {}s interval",
                        remaining,
                        daily_limit,
                        sleep_interval,
                    )
                else:
                    sleep_interval = base_interval
                # During market closure, reduce polling frequency
                if not self._market_hours.is_market_open(self._now_utc()):
                    await asyncio.sleep(base_interval * 10)  # 20min instead of 2min
                else:
                    await asyncio.sleep(sleep_interval)
            except asyncio.CancelledError:
                logger.info("Position monitor loop: cancelled during sleep")
                return

        logger.info("Position monitor loop: stopped")

    async def _handle_position_closed(self, intent: TradeIntent) -> None:
        """Process a detected position closure — update store and send alert.
        Fetches closed position details from MatchTrader for PnL, persists
        PnL/exit data to the store, calls LLM reflect, and sends alerts.

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
        # Try to fetch closed position details for PnL
        pnl = 0.0
        close_price = 0.0
        open_price = 0.0
        volume = 0.0
        exit_reason = "manual_close"  # Default — could be tp_hit, sl_hit, etc.
        try:
            # Retry with increasing delays to let broker update closed positions list
            now_ms = int(self._now_utc().timestamp() * 1000)
            day_ago_ms = now_ms - 86_400_000
            matched = False
            for attempt, delay in enumerate((2.0, 4.0, 8.0), start=1):
                await asyncio.sleep(delay)
                closed_positions = await self._matchtrader.get_closed_positions(
                    from_ts=day_ago_ms, to_ts=now_ms
                )
                for closed in closed_positions:
                    if str(closed.position_id) == position_id:
                        pnl = closed.profit
                        close_price = closed.close_price
                        open_price = closed.open_price
                        volume = closed.volume
                        # Infer exit reason from PnL direction
                        if pnl > 0:
                            exit_reason = "tp_hit"
                        elif pnl < 0:
                            exit_reason = "sl_hit"
                        matched = True
                        break
                if matched:
                    break
                logger.debug(
                    "Position monitor: closed position {} not found in broker API"
                    " (attempt {}/3, waited {}s)",
                    position_id,
                    attempt,
                    delay,
                )
        except Exception as e:
            logger.warning(
                "Position monitor: could not fetch closed position details for {}: {}",
                position_id,
                e,
            )
        # Override exit_reason if Best Day Rule triggered this close
        # and use recorded unrealized PnL as fallback when broker query returned 0
        if position_id in self._best_day_close_positions:
            exit_reason = "best_day_close"
            if pnl == 0.0:
                pnl = self._best_day_close_positions[position_id]
                logger.info(
                    "Position monitor: using recorded unrealized PnL ${:+.2f} for {}",
                    pnl,
                    position_id,
                )
            self._best_day_close_positions.pop(position_id, None)
        # Override exit_reason if re-evaluation triggered this close
        # and use recorded unrealized PnL as fallback when broker query returned 0
        if position_id in self._reevaluation_close_positions:
            exit_reason = "reeval_close"
            if pnl == 0.0:
                pnl = self._reevaluation_close_positions[position_id]
                logger.info(
                    "Position monitor: using recorded unrealized PnL ${:+.2f} for {}",
                    pnl,
                    position_id,
                )
            self._reevaluation_close_positions.pop(position_id, None)
        # Fallback for unknown close: use last-known unrealized PnL from position monitor
        if pnl == 0.0 and position_id in self._last_known_profit:
            pnl = self._last_known_profit[position_id]
            logger.info(
                "Position monitor: using last-known polled PnL ${:+.2f} for {}",
                pnl,
                position_id,
            )
        # Re-infer exit_reason from final PnL if still classified as manual_close.
        # This handles cases where broker API didn't return the closed position
        # but we recovered PnL from _last_known_profit or other fallbacks.
        if exit_reason == "manual_close" and pnl != 0.0:
            exit_reason = "tp_hit" if pnl > 0 else "sl_hit"
            logger.info(
                "Position monitor: re-inferred exit_reason={} from fallback PnL for {}",
                exit_reason,
                position_id,
            )

        # Clean up reevaluation tracking
        self._last_reevaluation.pop(position_id, None)
        self._last_known_profit.pop(position_id, None)
        # Calculate hold duration
        hold_duration_seconds: int | None = None
        if intent.executed_at is not None:
            delta = self._now_utc() - intent.executed_at
            hold_duration_seconds = int(delta.total_seconds())

        # Mark closed in store with PnL data
        try:
            await asyncio.to_thread(
                self._store.mark_closed,
                intent.id,
                realized_pnl=pnl,
                exit_price=close_price,
                exit_reason=exit_reason,
                hold_duration_seconds=hold_duration_seconds,
            )
        except Exception as e:
            logger.error(
                "Position monitor: failed to mark intent {} closed: {}",
                intent.id,
                e,
            )
            return
        if self._memory_journal is not None:
            try:
                self._memory_journal.append_trade_result(
                    symbol=symbol,
                    pnl=pnl,
                    reason=exit_reason,
                )
            except Exception as e:
                logger.warning(
                    "MemoryJournal: failed to append trade result for {}: {}",
                    symbol,
                    e,
                )
        self._log_trade_event(
            "TRADE_CLOSED",
            {
                "intent_id": intent.id,
                "symbol": symbol,
                "position_id": position_id,
                "pnl": pnl,
                "reason": exit_reason,
            },
        )
        # Call LLM reflect for learning
        if pnl != 0.0:
            try:
                await asyncio.to_thread(
                    self._agents.reflect,
                    {symbol: pnl},
                )
                logger.info("LLM reflect called for {} PnL={}", symbol, pnl)
            except Exception as e:
                logger.warning("LLM reflect failed for {}: {}", symbol, e)

        # Update BestDayTracker with realized PnL
        self._best_day_tracker.record_trade_pnl(pnl)
        # Convert broker symbol to config symbol for display
        display_symbol = symbol
        if self._registry is not None:
            display_symbol = self._registry.to_config_safe(symbol)
        equity: float | None = None
        try:
            balance = await self._matchtrader.get_balance()
            equity = balance.equity
        except Exception:
            pass

        # Update dynamic drawdown HWM tracker with new closed balance
        if self._hwm_tracker is not None and equity is not None:
            try:
                self._hwm_tracker.update_balance(balance.balance)
                self._hwm_tracker.save()
                logger.info(
                    "HWM updated: balance=${:.2f}, hwm=${:.2f}, loss_level=${:.2f}, locked={}",
                    balance.balance,
                    self._hwm_tracker.high_water_mark,
                    self._hwm_tracker.loss_level,
                    self._hwm_tracker.is_locked,
                )
            except Exception as e:
                logger.error("Failed to update HWM tracker: {}", e)

        # Map exit_reason to alert hit_type for backward compatibility
        hit_type = {"tp_hit": "TP", "sl_hit": "SL"}.get(exit_reason, "manual")
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
                        reason=f"Position closed ({exit_reason})",
                        volume=volume,
                        open_price=open_price,
                        close_price=close_price,
                        equity=equity,
                        position_id=position_id,
                    )
            except Exception as e:
                logger.error(
                    "Position monitor: alert failed for {}: {}",
                    position_id,
                    e,
                )

        # v1.2.0: Signal scanner to re-scan immediately (slot freed)
        self._rescan_event.set()
        logger.info("Position closed → rescan event set for {}", symbol)

    async def _close_winning_positions(self, open_positions: list[Any]) -> None:
        """Close all winning (profitable) positions to protect Best Day Rule.

        Called when BestDayTracker detects daily PnL is approaching the limit.
        Only closes positions with positive profit to lock in gains.

        Args:
            open_positions: List of PositionInfo from MatchTrader.
        """
        winners = [p for p in open_positions if p.profit > 0]
        if not winners:
            logger.info("Position monitor: should_close_winners triggered but no winning positions")
            return

        logger.warning(
            "Position monitor: Best Day Rule — closing {} winning position(s) ({})",
            len(winners),
            self._best_day_tracker.summary(),
        )

        closed_count = 0
        total_pnl = 0.0
        for pos in winners:
            try:
                result = await self._matchtrader.close_position(
                    position_id=str(pos.position_id),
                    symbol=pos.symbol,
                    side=pos.side,
                    volume=pos.volume,
                )
                if result.success:
                    closed_count += 1
                    total_pnl += pos.profit
                    self._best_day_close_positions[str(pos.position_id)] = pos.profit
                    logger.info(
                        "Position monitor: closed winning position {} ({}) PnL=${:+.2f}",
                        pos.position_id,
                        pos.symbol,
                        pos.profit,
                    )
                else:
                    logger.error(
                        "Position monitor: failed to close position {}: {}",
                        pos.position_id,
                        result.message,
                    )
            except Exception as e:
                logger.error(
                    "Position monitor: error closing position {}: {}",
                    pos.position_id,
                    e,
                )

        # Send alert about Best Day protection
        await self._send_alert(
            f"🛡️ <b>Best Day Protection</b>\n"
            f"• Closed {closed_count}/{len(winners)} winning position(s)\n"
            f"• Total PnL locked: ${total_pnl:+.2f}\n"
            f"• {self._best_day_tracker.summary()}"
        )

    async def _apply_breakeven_stops(
        self, open_positions: list[Any], opened_intents: list[TradeIntent]
    ) -> None:
        """Move SL to breakeven when profit reaches configured fraction of TP distance.

        For each open position with a matching intent that has suggested_tp_pips,
        calculate if profit has reached breakeven_activation_pct of TP distance.
        If so, modify the position's SL to the entry price (breakeven).

        Args:
            open_positions: Currently open positions from MatchTrader.
            opened_intents: Intents in "opened" state from the store.
        """
        # Build intent lookup by position_id
        intent_lookup = {
            intent.position_id: intent
            for intent in opened_intents
            if intent.position_id is not None
        }

        for pos in open_positions:
            try:
                pos_id = str(pos.position_id)

                # Skip if breakeven already applied
                if pos_id in self._breakeven_applied:
                    continue

                # Find matching intent
                intent = intent_lookup.get(pos_id)
                if intent is None:
                    continue

                # Skip if no suggested_tp_pips
                if intent.suggested_tp_pips is None:
                    continue

                # Resolve config symbol
                if self._registry is not None:
                    config_symbol = self._registry.to_config_safe(pos.symbol)
                else:
                    config_symbol = pos.symbol.rstrip(".")

                # Get pip_size from instruments
                instrument = self._config.instruments.get(config_symbol)
                if instrument is None:
                    continue

                pip_size = instrument.pip_size

                # Calculate TP distance and profit distance
                tp_distance = intent.suggested_tp_pips * pip_size
                profit_distance = abs(pos.current_price - pos.open_price)

                # Check if reached breakeven activation threshold
                if profit_distance >= tp_distance * self._config.scheduler.breakeven_activation_pct:
                    result = await self._matchtrader.modify_position(
                        position_id=pos_id,
                        symbol=pos.symbol,
                        side=pos.side,
                        volume=pos.volume,
                        sl=pos.open_price,
                        tp=pos.tp_price,
                    )
                    if result.success:
                        self._breakeven_applied.add(pos_id)
                        logger.info(
                            "Breakeven stop applied for position {} ({}) - SL moved to entry price",
                            pos_id,
                            pos.symbol,
                        )
                        await self._send_alert(
                            f"🛡️ <b>Breakeven Stop Applied</b>\n"
                            f"• Position: {pos_id} ({pos.symbol})\n"
                            f"• SL moved to entry price: {pos.open_price}"
                        )
                    else:
                        logger.warning(
                            "Failed to apply breakeven stop for position {}: {}",
                            pos_id,
                            result.message,
                        )
            except Exception as e:
                logger.error(
                    "Error applying breakeven stop for position {}: {}",
                    str(pos.position_id),
                    e,
                )

    async def _reevaluate_open_positions(
        self, open_positions: list[Any], opened_intents: list[TradeIntent]
    ) -> None:
        """Re-evaluate open positions via LLM agents on a configurable interval.
        Only closes a position when the LLM returns a signal that contradicts
        the current position direction (e.g. BUY position + LLM SELL → close).
        HOLD means 'keep the position open, do nothing'. Same-direction signals
        (e.g. BUY position + LLM BUY) confirm the position and keep it open.

        A minimum hold time (reeval_min_hold_seconds) is enforced before the
        first re-evaluation to prevent premature exits.

        Args:
            open_positions: Currently open positions from MatchTrader.
            opened_intents: Intents in "opened" state from the store.
        """
        # Skip entirely if using mock agents
        if self._agents.using_mock:
            logger.debug("Skipping re-evaluation — using mock agents")
            return

        # Build position lookups
        open_position_ids = {str(p.position_id) for p in open_positions}
        position_lookup = {str(p.position_id): p for p in open_positions}
        for intent in opened_intents:
            try:
                if intent.position_id is None:
                    continue
                if intent.position_id not in open_position_ids:
                    continue
                # Skip if already closed by re-evaluation
                if intent.position_id in self._reevaluation_close_positions:
                    continue

                # Check timing — enforce minimum hold time and reeval interval
                now = self._now_utc()
                last_eval = self._last_reevaluation.get(intent.position_id)
                if last_eval is None:
                    # First reeval check — enforce minimum hold time from position open
                    opened_at = intent.executed_at or intent.created_at
                    hold_seconds = (now - opened_at).total_seconds()
                    if hold_seconds < self._config.scheduler.reeval_min_hold_seconds:
                        continue
                else:
                    time_since = (now - last_eval).total_seconds()
                    if time_since < self._config.scheduler.reeval_interval_seconds:
                        continue
                # Update last evaluation time
                self._last_reevaluation[intent.position_id] = now

                # Build qlib_data with position context for LLM
                pos = position_lookup[intent.position_id]
                hold_duration = None
                if intent.executed_at is not None:
                    hold_duration = int((now - intent.executed_at).total_seconds())
                qlib_data = {
                    "score": intent.scanner_score,
                    "signal_strength": intent.scanner_confidence,
                    "confidence": intent.scanner_confidence,
                    "score_gap": intent.scanner_score_gap,
                    "drop_distance": intent.scanner_drop_distance,
                    "topk_spread": intent.scanner_topk_spread,
                    # Position context for re-evaluation
                    "position_side": pos.side,
                    "unrealized_pnl": pos.profit,
                    "entry_price": pos.open_price,
                    "current_price": pos.current_price,
                    "hold_duration_seconds": hold_duration,
                }
                # Get LLM decision
                decision = await asyncio.to_thread(
                    self._agents.decide,
                    symbol=intent.symbol,
                    trade_date=intent.trade_date,
                    qlib_data=qlib_data,
                )

                # Determine if the signal is a reversal of the current position
                is_reversal = (pos.side == "BUY" and decision.decision == "SELL") or (
                    pos.side == "SELL" and decision.decision == "BUY"
                )

                if is_reversal:
                    # Reverse signal — close the position
                    result = await self._matchtrader.close_position(
                        position_id=intent.position_id,
                        symbol=pos.symbol,
                        side=pos.side,
                        volume=pos.volume,
                    )
                    if result.success:
                        self._reevaluation_close_positions[intent.position_id] = pos.profit
                        logger.info(
                            "Re-evaluation closed position {} ({}) - reverse signal {} vs {}",
                            intent.position_id,
                            intent.symbol,
                            decision.decision,
                            pos.side,
                        )
                        await self._send_alert(
                            f"🔄 <b>Re-evaluation Close</b>\n"
                            f"• Position: {intent.position_id} ({intent.symbol})\n"
                            f"• Side: {pos.side} → LLM signal: {decision.decision}\n"
                            f"• Reason: Reverse signal detected"
                        )
                elif decision.decision == "HOLD":
                    logger.info(
                        "Re-evaluation: HOLD for position {} ({}) - keeping position open",
                        intent.position_id,
                        intent.symbol,
                    )
                else:
                    logger.info(
                        "Re-evaluation confirms position {} ({}) - decision: {}",
                        intent.position_id,
                        intent.symbol,
                        decision.decision,
                    )
            except Exception as e:
                logger.error(
                    "Error re-evaluating position {}: {}",
                    intent.position_id if intent.position_id else "unknown",
                    e,
                )

    async def _run_intraday_scan(self, daily_signals: list, today: str) -> None:
        """Run intraday scanner on symbols that daily scan identified.

        This provides entry timing — the daily scan sets direction,
        the intraday scan confirms the entry point is favorable.
        """
        entry_tf = self._config.scheduler.entry_timeframe
        symbols = [s.instrument for s in daily_signals]
        logger.info(
            "Multi-timeframe: running {} scan for {} symbols: {}",
            entry_tf,
            len(symbols),
            symbols,
        )

        intraday_signals = await asyncio.to_thread(
            self._scanner.run_pipeline,
            date=today,
            tickers=symbols,
            interval=entry_tf,
        )

        # Log results — intents were already created by daily scan
        # Intraday scan results are used for confidence boosting (Phase 2)
        if intraday_signals:
            for signal in intraday_signals:
                logger.info(
                    "Multi-timeframe {}: {} score={:.4f} conf={}",
                    entry_tf,
                    signal.instrument,
                    signal.score,
                    signal.confidence,
                )
        else:
            logger.info("Multi-timeframe: no intraday signals generated")

    async def _volatility_monitor_loop(self) -> None:
        """Poll quotes and trigger re-scan on significant price moves."""
        logger.info("Volatility monitor loop: started")
        while self._running:
            try:
                await self._wait_for_market_open("Volatility monitor")
                now = self._now_utc()

                for symbol in self._config.symbols:
                    try:
                        # Map config symbol to broker symbol if needed
                        broker_symbol = symbol
                        if self._registry is not None:
                            broker_symbol = self._registry.to_broker(symbol)
                        quote = await self._matchtrader.get_quote(broker_symbol)
                        mid_price = (quote.bid + quote.ask) / 2
                        self._volatility_monitor.record_quote(symbol, mid_price, now)
                    except Exception as e:
                        logger.debug("Volatility monitor: quote failed for {}: {}", symbol, e)

                triggered, symbol, pct = self._volatility_monitor.check_triggers(now)
                if triggered:
                    self._rescan_event.set()
                    await self._send_alert(
                        f"\U0001f4c8 <b>Volatility Trigger</b>\n"
                        f"\u2022 {symbol} moved {pct:+.2f}% in "
                        f"{self._config.scheduler.volatility_window_minutes}min\n"
                        f"\u2022 Triggering early scan"
                    )

            except asyncio.CancelledError:
                logger.info("Volatility monitor loop: cancelled")
                return
            except Exception as e:
                logger.error("Volatility monitor loop error: {}", e)

            try:
                await asyncio.sleep(self._config.scheduler.volatility_poll_interval_seconds)
            except asyncio.CancelledError:
                logger.info("Volatility monitor loop: cancelled during sleep")
                return

        logger.info("Volatility monitor loop: stopped")

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
            except asyncio.CancelledError:
                logger.info("Daily summary loop: cancelled")
                return
            except Exception as e:
                logger.error("Daily summary loop error: {}", e)
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                logger.info("Daily summary loop: cancelled during sleep")
                return

        logger.info("Daily summary loop: stopped")

    # ── Alert Helper ────────────────────────────────────────────────────

    async def _send_daily_summary(self, date_str: str) -> None:
        """Gather account data and send the daily summary alert.

        Args:
            date_str: Today's date in YYYY-MM-DD format.
        """
        if self._optimization_engine is not None:
            try:
                self._optimization_state = await asyncio.to_thread(
                    self._optimization_engine.refresh_state
                )
            except Exception as e:
                logger.warning("Optimization refresh failed: {}", e)

        if self._alert_service is None:
            return

        try:
            balance_info = await self._matchtrader.get_balance()
            open_positions = await self._matchtrader.get_open_positions()

            # Get today's intents to count trades
            today_intents = await asyncio.to_thread(self._store.get_intents_by_date, date_str)
            trades_today = sum(1 for i in today_intents if i.status in ("opened", "closed"))
            realized_pnl = sum(
                (i.realized_pnl or 0.0) for i in today_intents if i.status == "closed"
            )
            unrealized_pnl = sum(p.profit for p in open_positions)
            daily_pnl = realized_pnl + unrealized_pnl

            # Estimate day-start balance from realized PnL only.
            # balance already includes realized results, not floating PnL.
            day_start_balance = balance_info.balance - realized_pnl
            daily_dd_pct = (
                abs(daily_pnl) / day_start_balance
                if daily_pnl < 0 and day_start_balance > 0
                else 0.0
            )

            max_dd_ref = self._hwm_tracker.high_water_mark if self._hwm_tracker else None
            await self._alert_service.daily_summary(
                date=date_str,
                trades=trades_today,
                pnl=daily_pnl,
                equity=balance_info.equity,
                daily_dd_pct=daily_dd_pct,
                open_positions=len(open_positions),
                day_start_balance=day_start_balance,
                max_dd_reference=max_dd_ref,
            )
            logger.info("Daily summary sent for {}", date_str)
        except Exception as e:
            logger.error("Failed to send daily summary: {}", e)
            await self._send_alert(f"⚠️ <b>Daily Summary Error</b>\n<code>{e}</code>")

    # ── Weekend Market Closure ──────────────────────────────────────────

    async def _wait_for_market_open(self, loop_name: str) -> None:
        """Sleep until market opens. Logs once and sleeps in chunks."""
        now = self._now_utc()
        if self._market_hours.is_market_open(now):
            return

        wait_seconds = self._market_hours.seconds_until_open(now)
        logger.info(
            "{}: market closed — sleeping {:.0f}s ({:.1f}h) until open",
            loop_name,
            wait_seconds,
            wait_seconds / 3600,
        )
        await self._send_alert(
            f"💤 <b>{loop_name}</b>: market closed, sleeping until open "
            f"({wait_seconds / 3600:.1f}h)"
        )

        # Sleep in 5-minute chunks to allow graceful shutdown
        while not self._market_hours.is_market_open(self._now_utc()) and self._running:
            await asyncio.sleep(min(300, wait_seconds))
            wait_seconds = self._market_hours.seconds_until_open(self._now_utc())

        if self._running:
            self._weekend_force_close_done = False  # Reset for next weekend
            logger.info("{}: market open — resuming", loop_name)
            await self._send_alert(f"☀️ <b>{loop_name}</b>: market open, resuming operations")

    async def _force_close_for_weekend(self) -> None:
        """Force-close all open positions before weekend market closure."""
        logger.warning("Weekend force-close: closing all positions before market close")
        try:
            open_positions = await self._matchtrader.get_open_positions()
            if not open_positions:
                logger.info("Weekend force-close: no open positions")
                self._weekend_force_close_done = True
                return

            closed_count = 0
            total_pnl = 0.0
            for pos in open_positions:
                try:
                    result = await self._matchtrader.close_position(
                        position_id=str(pos.position_id),
                        symbol=pos.symbol,
                        side=pos.side,
                        volume=pos.volume,
                    )
                    if result.success:
                        closed_count += 1
                        total_pnl += pos.profit
                except Exception as e:
                    logger.error(
                        "Weekend force-close: failed to close {}: {}",
                        pos.position_id,
                        e,
                    )

            self._weekend_force_close_done = True
            await self._send_alert(
                f"🌙 <b>Weekend Force-Close</b>\n"
                f"• Closed {closed_count}/{len(open_positions)} positions\n"
                f"• Estimated PnL: ${total_pnl:+.2f}"
            )
        except Exception as e:
            logger.error("Weekend force-close failed: {}", e)
            await self._send_alert(f"⚠️ <b>Weekend Force-Close FAILED</b>\n<code>{e}</code>")

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

    @staticmethod
    def _coerce_numeric(value: Any, fallback: float) -> float:
        """Convert optional numeric-like values to float; fallback on mocks/invalid."""
        if isinstance(value, bool):
            return fallback
        if isinstance(value, Real):
            return float(value)
        return fallback

    def _should_pause_new_entries(self) -> bool:
        """Return True when Best Day protection says we should avoid new entries."""
        try:
            return self._best_day_tracker.should_close_winners()
        except Exception as e:
            logger.warning("Best Day protection check failed, defaulting to allow entries: {}", e)
            return False

    def _log_trade_event(self, event_type: str, details: dict[str, Any]) -> None:
        """Safely append an event to TradeJournal if configured."""
        if self._trade_journal is None:
            return
        try:
            self._trade_journal.log_event(event_type, details)
        except Exception as e:
            logger.warning("TradeJournal: failed to log {}: {}", event_type, e)

    def _get_thresholds_for_symbol(self, symbol: str) -> Thresholds:
        """Return thresholds for a symbol, falling back to global defaults."""
        if self._optimization_state is None:
            return Thresholds()
        if symbol in self._optimization_state.symbol_thresholds:
            return self._optimization_state.symbol_thresholds[symbol]
        return self._optimization_state.global_thresholds

    @staticmethod
    def _confidence_score(confidence: str) -> float:
        """Map confidence label to numeric score."""
        return CONFIDENCE_MAP.get(confidence, 0.5)

    @classmethod
    def _blend_confidence(cls, confidence: str, score: float) -> float:
        """Blend confidence label score with scanner score."""
        return 0.6 * cls._confidence_score(confidence) + 0.4 * score

    @classmethod
    def _passes_threshold(cls, confidence: str, blended: float, thresholds: Thresholds) -> bool:
        """Check whether confidence meets configured thresholds."""
        current = cls._confidence_score(confidence)
        required = cls._confidence_score(thresholds.min_confidence)
        if current < required:
            return False
        return blended >= thresholds.min_blended_confidence

    def _maybe_rollover_best_day_tracker(self) -> None:
        """Reset BestDayTracker at UTC day rollover to avoid cross-day carryover."""
        today = self._today_str()
        if today == self._best_day_tracker_date:
            return
        logger.info(
            "Scheduler: new UTC day {} detected, resetting BestDayTracker (prev={})",
            today,
            self._best_day_tracker_date,
        )
        self._best_day_tracker.reset()
        self._best_day_tracker_date = today
