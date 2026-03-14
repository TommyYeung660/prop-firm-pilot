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
import os
import random
from collections.abc import Coroutine
from datetime import datetime, timedelta, timezone
from numbers import Real
from typing import Any

import httpx
from loguru import logger

from src.compliance.best_day_tracker import BestDayTracker
from src.compliance.hwm_tracker import HighWaterMarkTracker
from src.config import AppConfig
from src.data.fx_data_fetcher import EodhdProvider
from src.data.fx_tick_aggregator import FXTickAggregator
from src.data.fx_websocket_client import EODHDFXWebSocketClient
from src.data.market_data_hub import MarketDataHub
from src.decision.agent_bridge import AgentBridge, AgentDecision, RiskMeta
from src.decision.decision_formatter import format_decision
from src.decision.schemas import TradeIntent
from src.decision.tactical_exit_manager import (
    TacticalExitEvaluation,
    TacticalExitManager,
    WriteBudgetSnapshot,
)
from src.decision.tactical_exit_rules import TacticalExitSnapshot
from src.decision.tactical_validator import TacticalData, TacticalResult, TacticalValidator
from src.decision_store.janitor import Janitor
from src.decision_store.sqlite_store import DecisionStore, InvalidTransitionError
from src.execution.engine import ExecutionEngine
from src.execution.instrument_registry import InstrumentRegistry
from src.execution.matchtrader_client import MatchTraderClient
from src.monitor.alert_service import AlertService
from src.monitor.equity_monitor import EquityMonitor
from src.monitor.memory_journal import MemoryJournal
from src.monitor.operational_metrics import OperationalMetrics
from src.monitor.trade_journal import TradeJournal
from src.optimize.optimization_engine import OptimizationEngine
from src.optimize.optimization_state import OptimizationState, Thresholds
from src.scheduler.close_resolution import determine_resolution_path
from src.scheduler.decision_cache import StrategicDecisionCache
from src.scheduler.low_confidence_cooldown import LowConfidenceCooldown
from src.scheduler.market_hours import MarketHoursChecker
from src.scheduler.news_event_trigger import NewsEventTrigger
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
        tactical_validator: TacticalValidator | None = None,
        decision_cache: StrategicDecisionCache | None = None,
        metrics: OperationalMetrics | None = None,
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
        self._best_day_paused_today: str | None = None  # Date when Best Day pause was activated
        self._optimization_engine = optimization_engine
        self._optimization_state: OptimizationState | None = None
        self._memory_journal = memory_journal
        self._trade_journal = trade_journal
        self._latest_market_event_context: str = ""
        self._tick_aggregator: FXTickAggregator | None = None
        self._websocket_client: EODHDFXWebSocketClient | None = None
        self._market_data_hub: MarketDataHub | None = None
        self._market_data_task: asyncio.Task[None] | None = None
        self._market_data_ready = False
        self._metrics = metrics or OperationalMetrics()

        # v1.3.7: Tactical execution module (shadow mode)
        self._tactical_validator = tactical_validator or TacticalValidator(config.tactical)
        self._tactical_exit_manager = TacticalExitManager(config.tactical.exit)
        self._decision_cache = decision_cache or StrategicDecisionCache(
            ttl_seconds=config.tactical.decision_cache.ttl_seconds
        )

        # v1.4.0: EODHD intraday provider for tactical bar data
        eodhd_key = os.getenv("EODHD_API_KEY", "")
        self._eodhd: EodhdProvider | None = EodhdProvider(api_key=eodhd_key) if eodhd_key else None
        if not eodhd_key:
            logger.warning("EODHD_API_KEY not set — tactical bar gates will be pass-through")

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
        self._news_trigger: NewsEventTrigger | None = None
        if config.scheduler.news_trigger_enabled:
            alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
            if alpha_vantage_key:
                self._news_trigger = NewsEventTrigger(
                    api_key=alpha_vantage_key,
                    keywords=config.scheduler.news_keywords,
                    lookback_minutes=config.scheduler.news_lookback_minutes,
                    max_headlines=config.scheduler.news_max_headlines,
                    cooldown_seconds=config.scheduler.news_cooldown_seconds,
                )
            else:
                logger.warning("ALPHA_VANTAGE_API_KEY not set — news trigger disabled")

        # v1.3.9: Low-confidence scanner cooldown (P3.10)
        self._low_confidence_cooldown = LowConfidenceCooldown(
            cooldown_minutes=config.scheduler.low_confidence_cooldown_minutes,
            threshold=config.scheduler.low_confidence_threshold,
        )

    # ── Public API ──────────────────────────────────────────────────────

    async def start(self) -> None:
        """Launch all workers as concurrent asyncio tasks."""
        self._running = True
        logger.info("Scheduler: starting all workers")
        await self._refresh_optimization_state()
        await self._initialize_market_data_hub()

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
        if self._news_trigger is not None:
            tasks.append(self._news_event_loop())

        await asyncio.gather(*tasks)

    async def stop(self) -> None:
        """Signal all workers to stop gracefully."""
        logger.info("Scheduler: stopping all workers")
        self._running = False
        self._equity_monitor.stop()
        if self._websocket_client is not None:
            await self._websocket_client.stop()
        if self._market_data_task is not None:
            self._market_data_task.cancel()
            self._market_data_task = None

    def _build_metrics_snapshot(self) -> dict[str, Any]:
        """Build the current operational metrics snapshot with feed status."""
        snapshot: dict[str, Any] = dict(self._metrics.get_summary())
        if self._market_data_ready and self._market_data_hub is not None:
            snapshot["market_data"] = self._market_data_hub.feed_status()
        return snapshot

    async def _initialize_market_data_hub(self) -> None:
        """Warm up and start the WebSocket-first market-data sidecar."""
        self._market_data_ready = False
        if not self._config.websocket.enabled:
            return
        if self._eodhd is None:
            logger.warning(
                "Scheduler: websocket market data requested but EODHD provider unavailable"
            )
            return
        eodhd_key = os.getenv("EODHD_API_KEY", "").strip()
        if not eodhd_key:
            logger.warning("Scheduler: websocket market data disabled — EODHD_API_KEY missing")
            return
        symbols = self._config.websocket.symbols or list(self._config.symbols)
        self._tick_aggregator = FXTickAggregator()
        self._websocket_client = EODHDFXWebSocketClient(
            api_token=eodhd_key,
            symbols=symbols,
            reconnect_base_seconds=self._config.websocket.reconnect_base_seconds,
            reconnect_max_seconds=self._config.websocket.reconnect_max_seconds,
            stale_after_seconds=self._config.websocket.stale_after_seconds,
        )
        self._websocket_client.register_tick_callback(self._tick_aggregator.add_tick)
        self._market_data_hub = MarketDataHub(
            aggregator=self._tick_aggregator,
            websocket_client=self._websocket_client,
            rest_provider=self._eodhd,
            symbols=symbols,
            quote_ttl_seconds=self._config.websocket.quote_ttl_seconds,
            operational_metrics=self._metrics,
        )
        self._volatility_monitor.set_market_data_hub(self._market_data_hub)
        await self._market_data_hub.warmup()
        self._market_data_task = asyncio.create_task(self._websocket_client.run())
        self._market_data_ready = True

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
                # Best Day daily stop: once activated for today, skip scanning entirely
                if self._best_day_paused_today == today:
                    await asyncio.sleep(self._session_cadence.get_scanner_interval(self._now_utc()))
                    continue
                if self._should_pause_new_entries():
                    logger.warning(
                        "Scanner loop: Best Day protection active ({}), "
                        "pausing new intents FOR THE REST OF THE DAY",
                        self._best_day_tracker.summary(),
                    )
                    self._best_day_paused_today = today
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

                        # P0: Position-aware scanner — skip symbols with active position
                        # This prevents wasted LLM calls for symbols already held.
                        # The duplicate_entry_guard in _process_claimed_intent is a
                        # backup; this early check avoids ~50% of unnecessary LLM spend.
                        has_active = await asyncio.to_thread(
                            self._store.has_active_position_for_symbol,
                            signal.instrument,
                        )
                        if has_active:
                            logger.info(
                                "Scanner loop: {} already has active position, skipping intent",
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
                            self._log_trade_event(
                                "SCANNER_SKIP",
                                {
                                    "symbol": signal.instrument,
                                    "reason": "recent_rejection_cooldown",
                                    "cooldown_minutes": rejection_cooldown,
                                },
                            )
                            logger.warning(
                                "Scanner loop: {} was rejected within {}min cooldown, "
                                "skipping to avoid retry loop",
                                signal.instrument,
                                rejection_cooldown,
                            )
                            continue

                        # P3.10: Low-confidence scanner cooldown
                        if self._low_confidence_cooldown.is_cooled_down(
                            signal.instrument, self._now_utc()
                        ):
                            count = self._low_confidence_cooldown.get_count(signal.instrument)
                            self._log_trade_event(
                                "SCANNER_SKIP",
                                {
                                    "symbol": signal.instrument,
                                    "reason": "low_confidence_cooldown",
                                    "consecutive_cancels": count,
                                },
                            )
                            logger.warning(
                                "Scanner loop: {} in low-confidence cooldown "
                                "({} consecutive cancels), skipping",
                                signal.instrument,
                                count,
                            )
                            continue

                        # P2.5: Circuit breaker — daily SL hit limit
                        daily_sl_limit = self._config.scheduler.daily_sl_hit_limit
                        if daily_sl_limit > 0:
                            daily_sl_count = await asyncio.to_thread(
                                self._store.count_sl_hits_today, today
                            )
                            if daily_sl_count >= daily_sl_limit:
                                logger.warning(
                                    "Scanner loop: circuit breaker — {} daily SL hits >= {} "
                                    "limit, blocking ALL new entries for today",
                                    daily_sl_count,
                                    daily_sl_limit,
                                )
                                await self._send_alert(
                                    f"\U0001f6d1 <b>Circuit Breaker</b>\n"
                                    f"\u2022 {daily_sl_count} SL hits today \u2265 "
                                    f"{daily_sl_limit} limit\n"
                                    f"\u2022 Blocking all new entries"
                                )
                                break  # Exit the entire signal loop

                        # P2.5: Circuit breaker — per-symbol SL limit
                        symbol_sl_limit = self._config.scheduler.symbol_loss_limit
                        if symbol_sl_limit > 0:
                            symbol_sl_count = await asyncio.to_thread(
                                self._store.count_symbol_losses_today,
                                signal.instrument,
                                today,
                            )
                            if symbol_sl_count >= symbol_sl_limit:
                                logger.warning(
                                    "Scanner loop: circuit breaker — {} has {} SL hits "
                                    "today >= {} limit, locking symbol",
                                    signal.instrument,
                                    symbol_sl_count,
                                    symbol_sl_limit,
                                )
                                continue  # Skip this symbol, try next

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
                    timeout_reason = f"LLM timeout: {e}"
                    if isinstance(e, TimeoutError | asyncio.TimeoutError):
                        await self._timeout_intent_safe(
                            worker_id=worker_id,
                            intent_id=intent.id,
                            reason=timeout_reason,
                            context="worker_timeout_recovery",
                        )
                    else:
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
        historical_pnl_context = self._build_historical_pnl_context(intent.symbol)
        if historical_pnl_context:
            qlib_data["historical_pnl_context"] = historical_pnl_context
        if self._latest_market_event_context:
            qlib_data["market_event_context"] = self._latest_market_event_context

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
                self._low_confidence_cooldown.record_low_confidence(intent.symbol, self._now_utc())
                self._metrics.record_llm_result("cancel")
            return

        cache_key = self._decision_cache_key(intent)
        cached_decision = self._decision_cache.get_cached(intent.symbol, cache_key)
        if cached_decision is not None:
            decision = self._hydrate_cached_decision(intent.symbol, cached_decision)
            self._log_trade_event(
                "LLM_DECISION_CACHE_HIT",
                {
                    "intent_id": intent.id,
                    "symbol": intent.symbol,
                    "cache_key": cache_key,
                    "decision": decision.decision,
                },
            )
        else:
            decision = await asyncio.to_thread(
                self._agents.decide,
                symbol=intent.symbol,
                trade_date=intent.trade_date,
                qlib_data=qlib_data,
                intent_id=intent.id,
            )
            self._decision_cache.store(
                intent.symbol,
                cache_key,
                self._serialize_decision_for_cache(decision),
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
            self._metrics.record_llm_result("success")
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

            # P2.6: Same-symbol active position check
            has_active = await asyncio.to_thread(
                self._store.has_active_position_for_symbol,
                intent.symbol,
            )
            if has_active:
                await self._cancel_intent_safe(
                    worker_id=worker_id,
                    intent_id=intent.id,
                    reason=f"Duplicate entry blocked: active {intent.symbol} position exists",
                    context="duplicate_entry_guard",
                )
                logger.warning(
                    "LLM worker {}: blocked {} — active position already exists",
                    worker_id,
                    intent.symbol,
                )
                return

            # P2.6: Same-direction daily limit
            max_same_dir = self._config.scheduler.max_same_direction_per_day
            if max_same_dir > 0:
                same_dir_count = await asyncio.to_thread(
                    self._store.count_same_direction_today,
                    intent.symbol,
                    decision.decision,
                    intent.trade_date,
                )
                if same_dir_count >= max_same_dir:
                    await self._cancel_intent_safe(
                        worker_id=worker_id,
                        intent_id=intent.id,
                        reason=(
                            f"Same-direction limit: {intent.symbol} {decision.decision} "
                            f"already attempted {same_dir_count}x today "
                            f"(limit={max_same_dir})"
                        ),
                        context="same_direction_limit",
                    )
                    logger.warning(
                        "LLM worker {}: blocked {} {} — already attempted {}x today",
                        worker_id,
                        intent.symbol,
                        decision.decision,
                        same_dir_count,
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
                self._low_confidence_cooldown.record_low_confidence(intent.symbol, self._now_utc())
                self._metrics.record_llm_result("cancel")
                return
            try:
                # v1.3.9: Store AB model_id in final_state for engine.py
                if decision.model_id:
                    decision.final_state["_model_id"] = decision.model_id
                await asyncio.to_thread(
                    self._store.update_intent_decision,
                    intent.id,
                    decision.decision,
                    sl_pips=formatted.suggested_sl_pips,
                    tp_pips=formatted.suggested_tp_pips,
                    risk_report=decision.risk_report,
                    state_json=json.dumps(decision.final_state, default=str),
                )
                # ── v1.3.7: Tactical validation (Shadow Mode) ──
                if self._config.tactical.enabled:
                    tactical_result = await self._run_tactical_validation(
                        intent, side=decision.decision
                    )
                    await self._log_tactical_result(intent, decision.decision, tactical_result)

                    if not self._config.tactical.shadow_mode:
                        if tactical_result.action == "WAIT":
                            progressed = await self._retry_tactical_pending(
                                worker_id=worker_id,
                                intent=intent,
                                side=decision.decision,
                                initial_result=tactical_result,
                            )
                            self._metrics.record_tactical_result(passed=progressed)
                            return
                        if tactical_result.action == "REJECT":
                            await self._cancel_intent_safe(
                                worker_id=worker_id,
                                intent_id=intent.id,
                                reason=(
                                    f"Tactical gate {tactical_result.action}: "
                                    f"{tactical_result.detail}"
                                ),
                                context="tactical_gate_reject",
                            )
                            self._log_trade_event(
                                "TACTICAL_BLOCKED",
                                {
                                    "intent_id": intent.id,
                                    "symbol": intent.symbol,
                                    "side": decision.decision,
                                    "action": tactical_result.action,
                                    "detail": tactical_result.detail,
                                },
                            )
                            logger.warning(
                                "LLM worker {}: intent {} tactical {} — blocked from execution",
                                worker_id,
                                intent.id,
                                tactical_result.action,
                            )
                            self._metrics.record_tactical_result(passed=False)
                            return

                self._metrics.record_tactical_result(passed=True)
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
            self._metrics.record_llm_result("cancel")
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

            # v1.3.9: Cancel any stale ready_for_exec intents for the same symbol
            # This prevents the race where an older BUY intent executes after a newer HOLD
            stale_intents = await asyncio.to_thread(self._store.get_ready_intents)
            for stale in stale_intents:
                if stale.symbol == intent.symbol and stale.id != intent.id:
                    await self._cancel_intent_safe(
                        worker_id=worker_id,
                        intent_id=stale.id,
                        reason="superseded_by_hold",
                        context=f"Newer HOLD decision for {intent.symbol} cancels stale intent",
                    )
                    logger.info(
                        "Cancelled stale ready_for_exec intent {} for {} (superseded by HOLD)",
                        stale.id,
                        intent.symbol,
                    )

    def _decision_cache_key(self, intent: TradeIntent) -> str:
        """Build a stable cache fingerprint from strategic inputs only."""
        return (
            f"{intent.scanner_score:.6f}|{intent.scanner_confidence}|"
            f"{intent.scanner_score_gap:.6f}|{intent.scanner_drop_distance:.6f}|"
            f"{intent.scanner_topk_spread:.6f}|{self._latest_market_event_context}"
        )

    def _serialize_decision_for_cache(self, decision: AgentDecision) -> dict[str, Any]:
        """Convert AgentDecision into a cache-friendly payload."""
        return {
            "decision": decision.decision,
            "final_state": decision.final_state,
            "risk_report": decision.risk_report,
            "model_id": decision.model_id,
            "risk_meta": decision.risk_meta.model_dump(),
        }

    def _hydrate_cached_decision(self, symbol: str, payload: dict[str, Any]) -> AgentDecision:
        """Rebuild AgentDecision from cached strategic output."""
        return AgentDecision(
            symbol=symbol,
            decision=payload.get("decision", "HOLD"),
            final_state=payload.get("final_state", {}),
            risk_report=payload.get("risk_report", ""),
            model_id=payload.get("model_id", ""),
            risk_meta=RiskMeta(**payload.get("risk_meta", {})),
        )

    def _build_historical_pnl_context(self, symbol: str) -> str:
        """Summarize recent PnL feedback for LLM prompts."""
        symbol_pnl: float | None = None
        portfolio_summary = ""
        if self._optimization_state is not None:
            symbol_pnl = self._optimization_state.feedback_pnl.get(symbol)
            if self._optimization_state.feedback_pnl:
                ordered = ", ".join(
                    f"{sym}={pnl:.2f}"
                    for sym, pnl in sorted(self._optimization_state.feedback_pnl.items())
                )
                portfolio_summary = f"7d feedback pnl by symbol: {ordered}."

        recent_symbol_trades: list[str] = []
        if self._trade_journal is not None:
            try:
                closed_trades = self._trade_journal.get_closed_trades(days=7)
                for trade in reversed(closed_trades):
                    if trade.get("symbol") != symbol:
                        continue
                    timestamp = str(trade.get("timestamp", ""))[:10]
                    pnl = float(trade.get("pnl", trade.get("realized_pnl", 0.0)) or 0.0)
                    reason = trade.get("reason", trade.get("exit_reason", "unknown"))
                    recent_symbol_trades.append(f"{timestamp}: pnl={pnl:.2f} ({reason})")
                    if len(recent_symbol_trades) >= 3:
                        break
            except Exception as e:
                logger.debug("Historical pnl context build failed for {}: {}", symbol, e)

        parts: list[str] = []
        if symbol_pnl is not None:
            parts.append(f"7d realized pnl for {symbol}: {symbol_pnl:.2f}.")
        if portfolio_summary:
            parts.append(portfolio_summary)
        if recent_symbol_trades:
            parts.append("Recent closed trades: " + " | ".join(recent_symbol_trades))
        return " ".join(parts)

    def _build_reflection_payload(
        self,
        *,
        intent: TradeIntent,
        pnl: float,
        exit_reason: str,
        position_id: str,
        resolution_path: str,
        hold_duration_seconds: int | None,
        decision: Any | None,
    ) -> dict[str, Any]:
        """Build a structured post-trade reflection payload for TradingAgents."""
        final_state: dict[str, Any] = {}
        if getattr(decision, "final_state", None):
            final_state = dict(decision.final_state)
        elif intent.agent_state_json:
            try:
                final_state = json.loads(intent.agent_state_json)
            except Exception:
                final_state = {}

        risk_report = getattr(decision, "risk_report", "") or intent.agent_risk_report
        model_id = getattr(decision, "model_id", "")
        return {
            "symbol": intent.symbol,
            "trade_date": intent.trade_date,
            "closed_at": self._now_utc().isoformat(),
            "position_id": position_id,
            "side": intent.suggested_side or "",
            "realized_pnl": pnl,
            "close_reason": exit_reason,
            "resolution_path": resolution_path,
            "hold_duration_seconds": hold_duration_seconds,
            "scanner_score": intent.scanner_score,
            "scanner_confidence": intent.scanner_confidence,
            "historical_pnl_context": self._build_historical_pnl_context(intent.symbol),
            "market_event_context": self._latest_market_event_context,
            "decision_summary": intent.suggested_side or "",
            "risk_report": risk_report,
            "model_id": model_id,
            "final_state": final_state,
        }

    async def _log_tactical_result(
        self,
        intent: TradeIntent,
        side: str,
        tactical_result: TacticalResult,
        retry_count: int = 0,
    ) -> None:
        """Log and alert tactical validation results."""
        self._log_trade_event(
            "TACTICAL_RESULT",
            {
                "intent_id": intent.id,
                "symbol": intent.symbol,
                "side": side,
                "action": tactical_result.action,
                "retry_count": retry_count,
                "hard_gates": [
                    {"gate": r.gate_name, "passed": r.passed, "detail": r.detail}
                    for r in tactical_result.hard_gates
                ],
                "soft_score": tactical_result.soft_score,
                "shadow_mode": self._config.tactical.shadow_mode,
            },
        )

        if self._alert_service:
            hard_summary = " ".join(
                f"{'✅' if r.passed else '❌'}{r.gate_name}" for r in tactical_result.hard_gates
            )
            soft_summary = " ".join(
                f"{'✅' if r.passed else '❌'}{r.gate_name}" for r in tactical_result.soft_gates
            )
            shadow = "(Shadow)" if self._config.tactical.shadow_mode else ""
            retry_suffix = f" [retry {retry_count}]" if retry_count > 0 else ""
            await self._send_alert(
                f"🔬 <b>Tactical Gate {shadow}</b>\n"
                f"• {intent.symbol} {side}{retry_suffix}\n"
                f"• Action: <b>{tactical_result.action}</b>\n"
                f"• Hard: {hard_summary}\n"
                f"• Soft: {soft_summary}"
                f" ({tactical_result.soft_score}/{tactical_result.soft_required})\n"
                f"• {tactical_result.detail}"
            )

    async def _retry_tactical_pending(
        self,
        *,
        worker_id: str,
        intent: TradeIntent,
        side: str,
        initial_result: TacticalResult,
    ) -> bool:
        """Retry tactical WAIT decisions before expiring them."""
        await asyncio.to_thread(self._store.mark_tactical_pending, intent.id)
        retry_cfg = self._config.tactical.retry
        self._log_trade_event(
            "TACTICAL_PENDING",
            {
                "intent_id": intent.id,
                "symbol": intent.symbol,
                "side": side,
                "detail": initial_result.detail,
                "max_retries": retry_cfg.max_retries,
            },
        )

        for retry_count in range(1, retry_cfg.max_retries + 1):
            jitter = (
                random.uniform(-retry_cfg.jitter_seconds, retry_cfg.jitter_seconds)
                if retry_cfg.jitter_seconds > 0
                else 0.0
            )
            await asyncio.sleep(max(0.0, retry_cfg.interval_seconds + jitter))
            latest = await asyncio.to_thread(self._store.get_intent, intent.id)
            if latest is None or latest.status != "tactical_pending":
                logger.warning(
                    "LLM worker {}: tactical retry aborted for {} (status={})",
                    worker_id,
                    intent.id,
                    latest.status if latest else "missing",
                )
                return False

            tactical_result = await self._run_tactical_validation(latest, side=side)
            await self._log_tactical_result(
                latest,
                side,
                tactical_result,
                retry_count=retry_count,
            )

            if tactical_result.action == "PASS":
                await asyncio.to_thread(self._store.mark_ready_for_exec_from_tactical, intent.id)
                logger.info(
                    "LLM worker {}: intent {} tactical retry passed on attempt {}",
                    worker_id,
                    intent.id,
                    retry_count,
                )
                return True

            if tactical_result.action == "REJECT":
                await self._cancel_intent_safe(
                    worker_id=worker_id,
                    intent_id=intent.id,
                    reason=f"Tactical gate REJECT: {tactical_result.detail}",
                    context="tactical_retry_reject",
                )
                return False

        if retry_cfg.expire_action == "degrade":
            await asyncio.to_thread(self._store.mark_ready_for_exec_from_tactical, intent.id)
            self._log_trade_event(
                "TACTICAL_DEGRADED",
                {
                    "intent_id": intent.id,
                    "symbol": intent.symbol,
                    "side": side,
                    "retries": retry_cfg.max_retries,
                },
            )
            logger.warning(
                "LLM worker {}: intent {} tactical retries exhausted — degrading to execution",
                worker_id,
                intent.id,
            )
            return True

        await self._timeout_intent_safe(
            worker_id=worker_id,
            intent_id=intent.id,
            reason=f"Tactical gate WAIT: {initial_result.detail}",
            context="tactical_retry_expired",
        )
        return False

    async def _run_tactical_validation(self, intent: TradeIntent, side: str) -> TacticalResult:
        """Fetch tactical data and run Hard/Soft gate validation.

        Returns TacticalResult. In shadow mode, result is logged but not acted on.
        """
        try:
            if side not in ("BUY", "SELL"):
                return TacticalResult(
                    action="PASS",
                    detail=f"Unsupported side for tactical validation: {side}",
                )
            data = await self._fetch_tactical_data(intent.symbol)
            result = self._tactical_validator.evaluate(side=side, data=data)
        except Exception as e:
            logger.warning(
                "Tactical validation error for {} {}: {}",
                intent.symbol,
                side,
                e,
            )
            result = TacticalResult(
                action="WAIT",
                detail=f"Tactical validation error (blocked, retry later): {e}",
            )
        return result

    async def _fetch_tactical_data(self, symbol: str) -> TacticalData:
        """Fetch spread and intraday bars for tactical validation.

        Uses MatchTrader get_quote() for spread and EODHD intraday API for
        5-min and 1-hour OHLCV bars required by ATR, EMA, RSI, and candle gates.
        """
        data = TacticalData()
        hub_supplied_data = False
        hub_quote_has_timestamp = False
        instrument = self._config.instruments.get(symbol)
        if instrument:
            data.typical_spread = instrument.avg_spread_pips * instrument.pip_size

        # ── WebSocket-first market data hub ────────────────────────────────
        if self._market_data_ready and self._market_data_hub is not None:
            try:
                quote_result = await self._market_data_hub.get_quote(symbol)
                data.quote_source = quote_result.source
                quote = quote_result.quote or {}
                ask = quote.get("ask", 0)
                bid = quote.get("bid", 0)
                ts_ms = quote.get("timestamp_ms", 0) or quote.get("timestampMs", 0)
                data.current_spread = abs(ask - bid)
                if ts_ms:
                    data.latest_bar_time = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                    hub_quote_has_timestamp = True
            except Exception as e:
                logger.warning("Failed to fetch hub quote for {}: {}", symbol, e)

            try:
                bars_5min_result, bars_1h_result = await asyncio.gather(
                    self._market_data_hub.get_bars(
                        symbol,
                        "5m",
                        self._config.websocket.warmup_5m_bars,
                    ),
                    self._market_data_hub.get_bars(
                        symbol,
                        "1h",
                        self._config.websocket.warmup_1h_bars,
                    ),
                )
                if not bars_5min_result.bars.empty:
                    data.bars_5min = bars_5min_result.bars
                    data.bars_5min_source = bars_5min_result.source
                if not bars_1h_result.bars.empty:
                    data.bars_1h = bars_1h_result.bars
                    data.bars_1h_source = bars_1h_result.source
                sources = {
                    source
                    for source in (
                        data.quote_source,
                        data.bars_5min_source,
                        data.bars_1h_source,
                    )
                    if source
                }
                if len(sources) == 1:
                    data.data_source = next(iter(sources))
                elif sources:
                    data.data_source = "mixed"
                hub_supplied_data = (
                    data.current_spread > 0 or not data.bars_5min.empty or not data.bars_1h.empty
                )
                if hub_supplied_data and hub_quote_has_timestamp:
                    return data
            except Exception as e:
                logger.warning("Failed to fetch hub intraday bars for {}: {}", symbol, e)

        # ── Spread from MatchTrader ──────────────────────────────────────────
        # Map config symbol (e.g. "EURUSD") to broker symbol (e.g. "EURUSD.")
        broker_symbol = symbol
        if self._registry is not None:
            broker_symbol = self._registry.to_broker(symbol)
        try:
            quote = await self._matchtrader.get_quote(broker_symbol)
            if quote:
                if isinstance(quote, dict):
                    ask = quote.get("ask", 0)
                    bid = quote.get("bid", 0)
                    ts_ms = quote.get("timestampMs", 0)
                else:
                    ask = getattr(quote, "ask", 0)
                    bid = getattr(quote, "bid", 0)
                    ts_ms = getattr(quote, "timestamp_ms", 0)
                data.current_spread = abs(ask - bid)
                # v1.3.9-fix: Use quote timestamp for data_freshness gate instead
                # of EODHD bar time.  EODHD intraday bars can lag 10+ hours during
                # DST transitions; MatchTrader quotes are real-time (<1 min delay).
                if ts_ms:
                    data.latest_bar_time = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                    logger.debug(
                        "Tactical: quote timestamp for {} = {} (age {:.0f}s)",
                        symbol,
                        data.latest_bar_time,
                        (datetime.now(timezone.utc) - data.latest_bar_time).total_seconds(),
                    )
        except Exception as e:
            logger.debug("Failed to fetch quote for {}: {}", symbol, e)

        if hub_supplied_data:
            return data

        # ── Intraday bars from EODHD ────────────────────────────────────────
        if self._eodhd:
            now = datetime.now(timezone.utc)
            # 5min bars: need ~50 bars for EMA(21) + RSI(14) ≈ 5 hours lookback
            start_5min = (now - timedelta(hours=6)).date()
            # 1h bars: need ~20 bars for ATR(14) ≈ 24 hours lookback
            start_1h = (now - timedelta(hours=30)).date()
            end_date = now.date()

            try:
                async with httpx.AsyncClient() as client:
                    bars_5min, bars_1h = await asyncio.gather(
                        self._eodhd.fetch_bars(
                            symbol, start_5min, end_date, client, interval="5min"
                        ),
                        self._eodhd.fetch_bars(symbol, start_1h, end_date, client, interval="1h"),
                    )
                if not bars_5min.empty:
                    data.bars_5min = bars_5min
                    data.bars_5min_source = "rest_fallback"
                    # Note: latest_bar_time is now set from MatchTrader quote above.
                    # EODHD bar timestamps are NOT used for data_freshness due to
                    # potential multi-hour delay during DST transitions.
                    logger.debug(
                        "Tactical: fetched {} 5min bars for {}",
                        len(bars_5min),
                        symbol,
                    )
                if not bars_1h.empty:
                    data.bars_1h = bars_1h
                    data.bars_1h_source = "rest_fallback"
                    logger.debug(
                        "Tactical: fetched {} 1h bars for {}",
                        len(bars_1h),
                        symbol,
                    )
                if data.bars_5min_source or data.bars_1h_source:
                    data.data_source = "rest_fallback"
            except Exception as e:
                logger.warning("Failed to fetch EODHD intraday bars for {}: {}", symbol, e)

        # v1.3.9-fix: Do NOT fallback to now() — let data_freshness gate fail
        # when no actual bar data was retrieved (EODHD fetch failure or empty response).

        return data

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

    async def _timeout_intent_safe(
        self,
        *,
        worker_id: str,
        intent_id: str,
        reason: str,
        context: str,
    ) -> bool:
        """Attempt intent timeout transition and tolerate state races.

        Returns:
            True when timeout succeeded, False when it was skipped/failed.
        """
        try:
            await asyncio.to_thread(self._store.mark_timed_out, intent_id, reason)
            return True
        except InvalidTransitionError as e:
            latest = await asyncio.to_thread(self._store.get_intent, intent_id)
            latest_status = latest.status if latest is not None else "missing"
            logger.warning(
                "LLM worker {}: skip timeout for intent {} during {} (status={}, reason={})",
                worker_id,
                intent_id,
                context,
                latest_status,
                e,
            )
            return False
        except Exception as e:
            logger.error(
                "LLM worker {}: failed to time out intent {} during {}: {}",
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
            balance = await self._matchtrader.get_balance()
            # For dynamic drawdown, use HWM as the reference for max drawdown
            max_dd_reference = self._config.account.initial_balance
            if self._hwm_tracker is not None:
                max_dd_reference = self._hwm_tracker.high_water_mark

            await self._equity_monitor.start(
                get_equity=self._get_equity,
                on_alert=self._handle_drawdown_alert,
                on_reduce_exposure=self._reduce_exposure_on_drawdown,
                on_emergency_close=self._handle_emergency_close,
                on_equity_snapshot=self._record_equity_snapshot,
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

    async def _get_equity(self) -> float:
        """Fetch current account equity from MatchTrader."""
        balance = await self._matchtrader.get_balance()
        return balance.equity

    async def _handle_drawdown_alert(
        self,
        level: str,
        daily_dd_pct: float,
        max_dd_pct: float,
        equity: float,
    ) -> None:
        """Forward drawdown warnings to the alert service and journal."""
        self._log_trade_event(
            "EQUITY_ALERT",
            {
                "level": level,
                "daily_dd_pct": daily_dd_pct,
                "max_dd_pct": max_dd_pct,
                "equity": equity,
            },
        )
        if self._alert_service is not None:
            await self._alert_service.drawdown_warning(level, daily_dd_pct, max_dd_pct, equity)

    async def _reduce_exposure_on_drawdown(
        self,
        level: str,
        daily_dd_pct: float,
        max_dd_pct: float,
        equity: float,
    ) -> None:
        """Reduce open exposure when drawdown reaches DANGER level."""
        if level != "DANGER":
            return

        positions = await self._matchtrader.get_open_positions()
        if not positions:
            return

        results: list[tuple[str, float, bool]] = []
        for pos in positions:
            reduce_volume = self._compute_reduction_volume(pos.symbol, pos.volume)
            if reduce_volume <= 0:
                continue
            result = await self._matchtrader.close_position(
                position_id=pos.position_id,
                symbol=pos.symbol,
                side=pos.side,
                volume=reduce_volume,
            )
            results.append((pos.position_id, reduce_volume, result.success))

        if results:
            self._log_trade_event(
                "EQUITY_REDUCE_EXPOSURE",
                {
                    "level": level,
                    "daily_dd_pct": daily_dd_pct,
                    "max_dd_pct": max_dd_pct,
                    "equity": equity,
                    "results": [
                        {"position_id": pid, "volume": volume, "success": success}
                        for pid, volume, success in results
                    ],
                },
            )
            await self._send_alert(
                f"⚠️ <b>Drawdown De-Risk</b>\n"
                f"• Level: {level}\n"
                f"• Equity: {equity:.2f}\n"
                f"• Daily DD: {daily_dd_pct:.1%}\n"
                f"• Max DD: {max_dd_pct:.1%}\n"
                f"• Reduced {len(results)} position(s)"
            )

    def _compute_reduction_volume(self, symbol: str, volume: float) -> float:
        """Calculate partial close volume for de-risking."""
        if volume <= 0:
            return 0.0
        instrument = self._config.instruments.get(symbol)
        min_lot = instrument.min_lot if instrument is not None else 0.01
        reduce_volume = round(volume * self._config.monitor.reduce_exposure_pct, 2)
        if reduce_volume < min_lot:
            reduce_volume = min(volume, min_lot)
        return min(volume, reduce_volume)

    async def _handle_emergency_close(self) -> None:
        """Close all positions and emit emergency audit events."""
        results = await self._matchtrader.close_all_positions()
        closed_count = sum(1 for r in results if r.success)
        self._log_trade_event(
            "EQUITY_EMERGENCY_CLOSE",
            {
                "closed_count": closed_count,
                "results": [result.model_dump() for result in results],
            },
        )
        await self._send_alert(
            "🛑 <b>Emergency Close Executed</b>\n"
            f"• Closed {closed_count}/{len(results)} position(s)"
        )

    async def _record_equity_snapshot(
        self,
        equity: float,
        daily_dd_pct: float,
        max_dd_pct: float,
    ) -> None:
        """Persist the latest equity snapshot to store and journal."""
        open_positions = 0
        try:
            positions = await self._matchtrader.get_open_positions()
            open_positions = len(positions)
        except Exception as e:
            logger.debug("Equity snapshot: failed to fetch open positions: {}", e)

        await asyncio.to_thread(
            self._store.insert_equity_snapshot,
            equity,
            daily_dd_pct,
            max_dd_pct,
            None,
            open_positions,
        )
        if self._trade_journal is not None:
            try:
                self._trade_journal.log_equity_snapshot(
                    balance=equity,
                    equity=equity,
                    daily_pnl=0.0,
                    open_positions=open_positions,
                )
            except Exception as e:
                logger.debug("TradeJournal equity snapshot failed: {}", e)

    async def _run_equity_check_once(self, reason: str) -> None:
        """Force an immediate one-shot equity check outside the polling loop."""
        balance = await self._matchtrader.get_balance()
        max_dd_reference = self._config.account.initial_balance
        if self._hwm_tracker is not None:
            max_dd_reference = self._hwm_tracker.high_water_mark

        result = await self._equity_monitor.check_once(
            get_equity=self._get_equity,
            on_alert=self._handle_drawdown_alert,
            on_reduce_exposure=self._reduce_exposure_on_drawdown,
            on_emergency_close=self._handle_emergency_close,
            on_equity_snapshot=self._record_equity_snapshot,
            day_start_balance=balance.balance,
            initial_balance=max_dd_reference,
            daily_drawdown_limit=self._config.compliance.daily_drawdown_limit,
            max_drawdown_limit=self._config.compliance.max_drawdown_limit,
        )
        self._log_trade_event(
            "EQUITY_CHECK_ON_DEMAND",
            {
                "reason": reason,
                "level": result["level"],
                "worst_pct": result["worst_pct"],
                "equity": result["equity"],
            },
        )

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
                if self._market_hours.is_market_open(self._now_utc()):
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

                        if open_positions:
                            await self._run_tactical_exit_cycle(open_positions, opened_intents)

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
            for attempt, delay in enumerate((2.0, 4.0, 8.0, 12.0, 16.0), start=1):
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
                    " (attempt {}/5, waited {}s)",
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
        decision = None  # May be loaded by execution_meta fallback below
        # ── v1.3.9: execution_meta fallback for volume/prices ──────────
        # When broker API didn't return data, read from execution_meta
        # (persisted at trade open time by engine.py)
        used_execution_meta = False
        if volume == 0.0 or close_price == 0.0:
            try:
                decision = await asyncio.to_thread(self._store.get_decision, intent.id)
                if decision and decision.execution_meta:
                    meta = json.loads(decision.execution_meta)
                    if volume == 0.0 and meta.get("volume"):
                        volume = meta["volume"]
                        used_execution_meta = True
                        logger.info(
                            "Position monitor: using execution_meta volume={} for {}",
                            volume,
                            position_id,
                        )
                    if open_price == 0.0 and meta.get("fill_price"):
                        open_price = meta["fill_price"]
                    # For close_price fallback: use SL/TP price based on exit_reason
                    if close_price == 0.0:
                        if exit_reason == "sl_hit" and meta.get("sl_price"):
                            close_price = meta["sl_price"]
                        elif exit_reason == "tp_hit" and meta.get("tp_price"):
                            close_price = meta["tp_price"]
                        if close_price != 0.0:
                            logger.info(
                                "Position monitor: using execution_meta {} price={} for {}",
                                exit_reason,
                                close_price,
                                position_id,
                            )
            except Exception as e:
                logger.debug(
                    "Position monitor: could not read execution_meta for {}: {}",
                    intent.id,
                    e,
                )
        used_best_day = False
        # Override exit_reason if Best Day Rule triggered this close
        # and use recorded unrealized PnL as fallback when broker query returned 0
        if position_id in self._best_day_close_positions:
            exit_reason = "best_day_close"
            used_best_day = True
            if pnl == 0.0:
                pnl = self._best_day_close_positions[position_id]
                logger.info(
                    "Position monitor: using recorded unrealized PnL ${:+.2f} for {}",
                    pnl,
                    position_id,
                )
            self._best_day_close_positions.pop(position_id, None)
        used_reeval = False
        # Override exit_reason if re-evaluation triggered this close
        # and use recorded unrealized PnL as fallback when broker query returned 0
        if position_id in self._reevaluation_close_positions:
            exit_reason = "reeval_close"
            used_reeval = True
            if pnl == 0.0:
                pnl = self._reevaluation_close_positions[position_id]
                logger.info(
                    "Position monitor: using recorded unrealized PnL ${:+.2f} for {}",
                    pnl,
                    position_id,
                )
            self._reevaluation_close_positions.pop(position_id, None)
        # Fallback for unknown close: use last-known unrealized PnL from position monitor
        used_last_known = False
        if pnl == 0.0 and position_id in self._last_known_profit:
            pnl = self._last_known_profit[position_id]
            used_last_known = True
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
        # v1.3.9: Improved exit_reason — compare close_price vs SL/TP prices
        # More accurate than PnL sign alone (handles manual close in profit)
        if close_price != 0.0 and exit_reason in ("tp_hit", "sl_hit", "manual_close"):
            try:
                if decision is None:
                    decision = await asyncio.to_thread(self._store.get_decision, intent.id)
                if decision and decision.execution_meta:
                    meta = json.loads(decision.execution_meta)
                    sl_price = meta.get("breakeven_sl") or meta.get("sl_price", 0.0)
                    tp_price = meta.get("tp_price", 0.0)
                    pip_size = self._get_pip_size(symbol)
                    tolerance = pip_size * 3  # 3-pip tolerance for slippage

                    if tp_price and abs(close_price - tp_price) <= tolerance:
                        exit_reason = "tp_hit"
                    elif sl_price and abs(close_price - sl_price) <= tolerance:
                        exit_reason = "sl_hit"
            except Exception:
                pass  # Keep existing exit_reason; decision stays as loaded

        # v1.3.9: Resolution path tracking for trade retrospection quality
        resolution_path = determine_resolution_path(
            matched=matched,
            used_execution_meta=used_execution_meta,
            used_best_day=used_best_day,
            used_reeval=used_reeval,
            used_last_known=used_last_known,
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
                    intent_id=intent.id,
                    position_id=position_id,
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
                "resolution_path": resolution_path,
            },
        )
        self._decision_cache.invalidate(symbol)
        # v1.3.9: Operational metrics — record trade close (P3.11)
        self._metrics.record_trade_close(exit_reason)
        # v1.3.9: Reset low-confidence cooldown on successful close (P3.10)
        self._low_confidence_cooldown.reset_symbol(symbol)
        # Call LLM reflect for learning
        if pnl != 0.0:
            try:
                if decision is None:
                    decision = await asyncio.to_thread(self._store.get_decision, intent.id)
                reflection_payload = self._build_reflection_payload(
                    intent=intent,
                    pnl=pnl,
                    exit_reason=exit_reason,
                    position_id=position_id,
                    resolution_path=resolution_path,
                    hold_duration_seconds=hold_duration_seconds,
                    decision=decision,
                )
                await asyncio.to_thread(
                    self._agents.reflect,
                    reflection_payload,
                )
                logger.info(
                    "LLM reflect called for {} PnL={} reason={} resolution={}",
                    symbol,
                    pnl,
                    exit_reason,
                    resolution_path,
                )
            except Exception as e:
                logger.warning("LLM reflect failed for {}: {}", symbol, e)

        # Update BestDayTracker with realized PnL
        self._best_day_tracker.record_trade_pnl(pnl)

        # v1.3.9: Update AB test stats
        if self._optimization_state is not None and pnl != 0.0:
            try:
                ab_decision = decision
                if ab_decision is None:
                    ab_decision = await asyncio.to_thread(self._store.get_decision, intent.id)
                if ab_decision and ab_decision.execution_meta:
                    ab_meta = json.loads(ab_decision.execution_meta)
                    ab_model_id = ab_meta.get("model_id", "")
                    if ab_model_id and self._optimization_state.ab_test:
                        from src.optimize.ab_testing import update_ab_stats

                        update_ab_stats(self._optimization_state.ab_test, ab_model_id, pnl)
                        logger.info("AB test: recorded pnl={:.2f} for model={}", pnl, ab_model_id)
            except Exception as e:
                logger.debug("AB test stats update failed: {}", e)

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

        # v1.3.8: Removed auto-rescan on position close to reduce excessive rescans.
        # Volatility monitor already handles re-entry timing based on market conditions.
        logger.debug("Position closed for {} — skipping auto-rescan (v1.3.8)", symbol)

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
                        # Read-back verification: confirm SL was actually updated
                        await asyncio.sleep(0.5)  # Brief delay for propagation
                        verified = await self._matchtrader.verify_sl_tp(
                            position_id=pos_id, expected_sl=pos.open_price
                        )
                        if not verified:
                            # Retry once: re-send modify + verify
                            logger.warning(
                                "Breakeven SL verify failed for {}, retrying modify...",
                                pos_id,
                            )
                            retry_result = await self._matchtrader.modify_position(
                                position_id=pos_id,
                                symbol=pos.symbol,
                                side=pos.side,
                                volume=pos.volume,
                                sl=pos.open_price,
                                tp=pos.tp_price,
                            )
                            if retry_result.success:
                                await asyncio.sleep(0.5)
                                verified = await self._matchtrader.verify_sl_tp(
                                    position_id=pos_id, expected_sl=pos.open_price
                                )
                        if verified:
                            self._breakeven_applied.add(pos_id)
                            # Store the breakeven SL in execution_meta for exit_reason
                            self._update_breakeven_sl_in_meta(intent, pos.open_price)
                            logger.info(
                                "Breakeven stop applied and verified for {} ({}) — SL to entry",
                                pos_id,
                                pos.symbol,
                            )
                            await self._send_alert(
                                f"\U0001f6e1\ufe0f <b>Breakeven Stop Applied</b>\n"
                                f"\u2022 Position: {pos_id} ({pos.symbol})\n"
                                f"\u2022 SL moved to entry price: {pos.open_price}"
                            )
                        else:
                            logger.error(
                                "Breakeven SL UNVERIFIED for {} after retry!",
                                pos_id,
                            )
                            await self._send_alert(
                                f"\u26a0\ufe0f <b>Breakeven SL UNVERIFIED</b>\n"
                                f"\u2022 Position: {pos_id} ({pos.symbol})\n"
                                f"\u2022 Modify reported success but read-back failed"
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

    def _update_breakeven_sl_in_meta(self, intent: TradeIntent, breakeven_sl: float) -> None:
        """Persist breakeven SL into execution_meta so exit_reason can reference it."""
        try:
            decision = self._store.get_decision(intent.id)
            if decision and decision.execution_meta:
                meta = json.loads(decision.execution_meta)
            else:
                meta = {}
            meta["breakeven_sl"] = breakeven_sl
            self._store.update_execution_meta(intent.id, json.dumps(meta))
        except Exception as e:
            logger.warning(
                "Failed to update breakeven_sl in meta for {}: {}",
                intent.id,
                e,
            )

    def _load_execution_meta_dict(self, intent: TradeIntent) -> dict[str, Any]:
        """Best-effort read of execution metadata from the intent or decision store."""
        meta_json = getattr(intent, "execution_meta", "") or ""
        if not meta_json:
            try:
                decision = self._store.get_decision(intent.id)
                if decision and decision.execution_meta:
                    meta_json = decision.execution_meta
            except Exception as e:
                logger.debug(
                    "Tactical exit: could not load execution_meta for {}: {}",
                    intent.id,
                    e,
                )

        if not meta_json:
            return {}

        try:
            return json.loads(meta_json)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Tactical exit: invalid execution_meta for {}: {}", intent.id, e)
            return {}

    @staticmethod
    def _parse_meta_datetime(value: Any) -> datetime | None:
        """Parse metadata timestamps stored as ISO strings."""
        if isinstance(value, datetime):
            return value
        if not isinstance(value, str) or not value:
            return None
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            return None

    def _compute_tactical_exit_unrealized_r(self, pos: Any, meta: dict[str, Any]) -> float:
        """Convert floating PnL distance into R using original stop distance."""
        entry_price = self._coerce_numeric(meta.get("fill_price"), fallback=pos.open_price)
        original_sl = self._coerce_numeric(meta.get("sl_price"), fallback=pos.sl_price or 0.0)
        risk_distance = abs(entry_price - original_sl)
        if risk_distance <= 0:
            return 0.0

        if pos.side == "BUY":
            profit_distance = pos.current_price - entry_price
        else:
            profit_distance = entry_price - pos.current_price
        return profit_distance / risk_distance

    def _build_tactical_exit_snapshot(
        self,
        pos: Any,
        intent: TradeIntent,
        tactical_data: TacticalData,
    ) -> TacticalExitSnapshot:
        """Build the pure tactical-exit snapshot for one open position."""
        meta = self._load_execution_meta_dict(intent)
        original_sl_price = (
            self._coerce_numeric(meta.get("sl_price"), fallback=pos.sl_price or 0.0) or None
        )
        original_tp_price = (
            self._coerce_numeric(meta.get("tp_price"), fallback=pos.tp_price or 0.0) or None
        )
        return TacticalExitSnapshot(
            position_id=str(pos.position_id),
            symbol=intent.symbol,
            side=pos.side,
            open_price=pos.open_price,
            current_price=pos.current_price,
            volume=pos.volume,
            sl_price=pos.sl_price,
            tp_price=pos.tp_price,
            original_sl_price=original_sl_price,
            original_tp_price=original_tp_price,
            unrealized_r=self._compute_tactical_exit_unrealized_r(pos, meta),
            partial_close_done=bool(meta.get("partial_close_done", False)),
            bars_5min=tactical_data.bars_5min,
            bars_1h=tactical_data.bars_1h,
            prior_trailing_sl=self._coerce_numeric(
                meta.get("trailing_sl"),
                fallback=0.0,
            )
            or None,
            last_tactical_exit_action=str(meta.get("last_tactical_exit_action", "")),
            last_tactical_exit_at=self._parse_meta_datetime(meta.get("last_tactical_exit_at")),
        )

    def _get_tactical_exit_budget_snapshot(self) -> WriteBudgetSnapshot:
        """Read current broker write-budget snapshot for tactical exit throttling."""
        limiter = getattr(
            self._matchtrader,
            "rate_limiter",
            getattr(self._matchtrader, "_rate_limiter", None),
        )
        write_remaining = int(
            self._coerce_numeric(
                getattr(limiter, "write_remaining", getattr(limiter, "remaining", 0)),
                fallback=0.0,
            )
        )
        daily_write_limit = int(
            self._coerce_numeric(
                getattr(limiter, "daily_write_limit", getattr(limiter, "_daily_limit", 0)),
                fallback=0.0,
            )
        )
        return WriteBudgetSnapshot(
            write_remaining=write_remaining,
            daily_write_limit=daily_write_limit,
        )

    async def _handle_tactical_exit_evaluation(
        self,
        pos: Any,
        intent: TradeIntent,
        evaluation: TacticalExitEvaluation,
    ) -> None:
        """Handle non-broker side effects of a tactical exit evaluation."""
        if evaluation.decision.action != "HOLD":
            await self._execute_tactical_exit_action(pos, intent, evaluation)
            return

        if evaluation.skip_reason:
            event_type = (
                "TACTICAL_EXIT_BUDGET_BLOCKED"
                if evaluation.skip_reason == "write_budget_blocked"
                else "TACTICAL_EXIT_SKIPPED"
            )
            self._log_trade_event(
                event_type,
                {
                    "position_id": str(pos.position_id),
                    "intent_id": intent.id,
                    "symbol": intent.symbol,
                    "reason": evaluation.skip_reason,
                },
            )

        if evaluation.requires_llm_exception_review:
            await self._reevaluate_open_positions([pos], [intent])

    def _price_precision_for_symbol(self, symbol: str) -> int | None:
        """Resolve broker price precision for a symbol using registry or config fallback."""
        config_symbol = symbol
        if self._registry is not None:
            config_symbol = self._registry.to_config_safe(symbol)
            info = self._registry.get_info(config_symbol)
            if info is not None:
                return info.price_precision
        else:
            config_symbol = symbol.rstrip(".")

        instrument = self._config.instruments.get(config_symbol)
        if instrument is None:
            return None

        pip_size_text = f"{instrument.pip_size:.10f}".rstrip("0")
        decimals = len(pip_size_text.split(".")[-1]) if "." in pip_size_text else 0
        if instrument.pip_size < 1:
            return decimals + 1
        return decimals

    def _normalize_tactical_price(self, symbol: str, price: float | None) -> float | None:
        """Round tactical price levels to broker precision before writes or verification."""
        if price is None:
            return None
        precision = self._price_precision_for_symbol(symbol)
        if precision is None:
            return float(price)
        return round(float(price), precision)

    def _calculate_partial_close_volume(
        self,
        symbol: str,
        current_volume: float,
        ratio: float,
    ) -> float | None:
        """Calculate a valid partial-close volume respecting broker minimum lot size."""
        config_symbol = symbol
        if self._registry is not None:
            config_symbol = self._registry.to_config_safe(symbol)
        else:
            config_symbol = symbol.rstrip(".")

        instrument = self._config.instruments.get(config_symbol)
        min_lot = instrument.min_lot if instrument is not None else 0.01
        step = min_lot
        precision = max(0, len(f"{step:.8f}".rstrip("0").split(".")[-1]))

        raw_close_volume = current_volume * ratio
        close_volume = int(raw_close_volume / step) * step
        close_volume = round(close_volume, precision)
        remaining_volume = round(current_volume - close_volume, precision)

        if close_volume < min_lot:
            return None
        if remaining_volume < min_lot:
            return None
        return close_volume

    def _update_tactical_exit_meta(
        self,
        intent: TradeIntent,
        decision: Any,
        *,
        partial_close_volume: float | None = None,
    ) -> None:
        """Persist tactical exit metadata back into execution_meta."""
        meta = self._load_execution_meta_dict(intent)
        meta["tactical_exit_state"] = decision.state
        meta["last_tactical_exit_action"] = decision.action
        meta["last_tactical_exit_at"] = self._now_utc().isoformat()
        meta["tactical_exit_reason"] = decision.reason

        if decision.action == "MOVE_TO_BREAKEVEN" and decision.new_sl is not None:
            meta["breakeven_sl"] = decision.new_sl
        if decision.action == "TRAIL_SL" and decision.new_sl is not None:
            meta["trailing_sl"] = decision.new_sl
        if decision.action == "REPRICE_TP" and decision.new_tp is not None:
            meta["dynamic_tp"] = decision.new_tp
        if decision.action == "PARTIAL_CLOSE":
            meta["partial_close_done"] = True
            meta["partial_close_ratio"] = decision.partial_close_ratio
            meta["partial_close_volume"] = partial_close_volume
            meta["partial_close_at"] = self._now_utc().isoformat()

        meta_json = json.dumps(meta, default=str)
        self._store.update_execution_meta(intent.id, meta_json)

    async def _execute_tactical_exit_action(
        self,
        pos: Any,
        intent: TradeIntent,
        evaluation: TacticalExitEvaluation,
    ) -> None:
        """Execute a tactical exit action and persist the result on success."""
        decision = evaluation.decision
        position_id = str(pos.position_id)

        if decision.action == "PARTIAL_CLOSE":
            close_volume = self._calculate_partial_close_volume(
                intent.symbol,
                current_volume=pos.volume,
                ratio=decision.partial_close_ratio or 0.0,
            )
            if close_volume is None:
                self._log_trade_event(
                    "TACTICAL_EXIT_SKIPPED",
                    {
                        "position_id": position_id,
                        "intent_id": intent.id,
                        "symbol": intent.symbol,
                        "reason": "invalid_partial_close_volume",
                    },
                )
                return

            result = await self._matchtrader.close_position(
                position_id=position_id,
                symbol=pos.symbol,
                side=pos.side,
                volume=close_volume,
            )
            if not result.success:
                self._log_trade_event(
                    "TACTICAL_EXIT_SKIPPED",
                    {
                        "position_id": position_id,
                        "intent_id": intent.id,
                        "symbol": intent.symbol,
                        "reason": result.message,
                    },
                )
                return

            self._update_tactical_exit_meta(
                intent,
                decision,
                partial_close_volume=close_volume,
            )
            self._log_trade_event(
                "TACTICAL_EXIT_ACTION",
                {
                    "position_id": position_id,
                    "intent_id": intent.id,
                    "symbol": intent.symbol,
                    "action": decision.action,
                    "volume": close_volume,
                },
            )
            return

        if decision.action == "EXIT_NOW":
            result = await self._matchtrader.close_position(
                position_id=position_id,
                symbol=pos.symbol,
                side=pos.side,
                volume=pos.volume,
            )
            if not result.success:
                self._log_trade_event(
                    "TACTICAL_EXIT_SKIPPED",
                    {
                        "position_id": position_id,
                        "intent_id": intent.id,
                        "symbol": intent.symbol,
                        "reason": result.message,
                    },
                )
                return

            self._update_tactical_exit_meta(intent, decision)
            self._log_trade_event(
                "TACTICAL_EXIT_ACTION",
                {
                    "position_id": position_id,
                    "intent_id": intent.id,
                    "symbol": intent.symbol,
                    "action": decision.action,
                    "volume": pos.volume,
                },
            )
            return

        expected_sl = self._normalize_tactical_price(
            intent.symbol,
            decision.new_sl if decision.new_sl is not None else pos.sl_price,
        )
        expected_tp = self._normalize_tactical_price(
            intent.symbol,
            decision.new_tp if decision.new_tp is not None else pos.tp_price,
        )
        if decision.new_sl is not None:
            decision.new_sl = expected_sl
        if decision.new_tp is not None:
            decision.new_tp = expected_tp

        result = await self._matchtrader.modify_position(
            position_id=position_id,
            symbol=pos.symbol,
            side=pos.side,
            volume=pos.volume,
            sl=expected_sl,
            tp=expected_tp,
        )
        if not result.success:
            self._log_trade_event(
                "TACTICAL_EXIT_SKIPPED",
                {
                    "position_id": position_id,
                    "intent_id": intent.id,
                    "symbol": intent.symbol,
                    "reason": result.message,
                },
            )
            return

        verified = await self._matchtrader.verify_sl_tp(
            position_id=position_id,
            expected_sl=expected_sl,
            expected_tp=expected_tp,
            price_precision=self._price_precision_for_symbol(intent.symbol),
        )
        if not verified:
            self._log_trade_event(
                "TACTICAL_EXIT_SKIPPED",
                {
                    "position_id": position_id,
                    "intent_id": intent.id,
                    "symbol": intent.symbol,
                    "reason": "verify_failed",
                },
            )
            return

        if decision.action == "MOVE_TO_BREAKEVEN":
            self._breakeven_applied.add(position_id)

        self._update_tactical_exit_meta(intent, decision)
        self._log_trade_event(
            "TACTICAL_EXIT_ACTION",
            {
                "position_id": position_id,
                "intent_id": intent.id,
                "symbol": intent.symbol,
                "action": decision.action,
            },
        )

    async def _run_tactical_exit_cycle(
        self,
        open_positions: list[Any],
        opened_intents: list[TradeIntent],
    ) -> None:
        """Evaluate all open positions through the tactical exit manager."""
        if not self._config.tactical.exit.enabled:
            return

        intent_lookup = {
            intent.position_id: intent
            for intent in opened_intents
            if intent.position_id is not None
        }
        budget = self._get_tactical_exit_budget_snapshot()

        for pos in open_positions:
            intent = intent_lookup.get(str(pos.position_id))
            if intent is None:
                continue

            tactical_data = await self._fetch_tactical_data(intent.symbol)
            snapshot = self._build_tactical_exit_snapshot(pos, intent, tactical_data)
            evaluation = self._tactical_exit_manager.evaluate_position(
                snapshot=snapshot,
                budget=budget,
                now=self._now_utc(),
            )
            await self._handle_tactical_exit_evaluation(pos, intent, evaluation)

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
                    intent_id=intent.id,
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
        # v1.3.9: Skip intraday rescan for daily models — scores won't change
        # until candle close, so rescanning is wasted compute + confusing logs
        if self._config.scheduler.scanner_timeframe == "1d":
            logger.info(
                "Skipping intraday rescan — daily model scores unchanged until candle close"
                " ({} symbols)",
                len(daily_signals),
            )
            return
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

                if self._market_data_ready and self._market_data_hub is not None:
                    triggered, symbol, pct = await self._volatility_monitor.check_once(now)
                else:
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
                    self._latest_market_event_context = (
                        f"Volatility trigger: {symbol} moved {pct:+.2f}% in "
                        f"{self._config.scheduler.volatility_window_minutes} minutes."
                    )
                    self._rescan_event.set()
                    await self._run_equity_check_once(reason=f"volatility:{symbol}")
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

    async def _news_event_loop(self) -> None:
        """Poll macro headlines and trigger rescans when fresh events arrive."""
        if self._news_trigger is None:
            return

        logger.info("News event loop: started")
        async with httpx.AsyncClient(timeout=15.0) as client:
            while self._running:
                try:
                    await self._wait_for_market_open("News event loop")
                    now = self._now_utc()
                    triggered, headlines = await self._news_trigger.check_once(
                        client=client,
                        now=now,
                    )
                    if triggered:
                        self._latest_market_event_context = self._format_headline_context(headlines)
                        self._rescan_event.set()
                        await self._run_equity_check_once(reason="news_event")
                        lead = headlines[0]
                        await self._send_alert(
                            f"📰 <b>News Trigger</b>\n"
                            f"• {lead['title']}\n"
                            f"• Triggering early scan"
                        )
                except asyncio.CancelledError:
                    logger.info("News event loop: cancelled")
                    return
                except Exception as e:
                    logger.error("News event loop error: {}", e)

                try:
                    await asyncio.sleep(self._config.scheduler.news_poll_interval_seconds)
                except asyncio.CancelledError:
                    logger.info("News event loop: cancelled during sleep")
                    return

        logger.info("News event loop: stopped")

    def _format_headline_context(self, headlines: list[dict[str, str]]) -> str:
        """Convert fresh headlines into prompt-ready event context."""
        return "Fresh macro/news events: " + " | ".join(
            f"{headline['published_at']}: {headline['title']}" for headline in headlines[:3]
        )

    async def _refresh_optimization_state(self) -> None:
        """Load the latest optimization state and propagate AB routing."""
        if self._optimization_engine is None:
            return
        try:
            self._optimization_state = await asyncio.to_thread(
                self._optimization_engine.refresh_state
            )
            if self._optimization_state:
                self._agents.set_ab_state(self._optimization_state.ab_test)
        except Exception as e:
            logger.warning("Optimization refresh failed: {}", e)

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
        await self._refresh_optimization_state()

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

            # v1.3.9: Emit operational metrics snapshot (P3.11 + P3.12)
            self._log_trade_event("METRICS_SNAPSHOT", self._build_metrics_snapshot())

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

    def _get_pip_size(self, symbol: str) -> float:
        """Look up pip size from instrument config, with safe default."""
        instrument = self._config.instruments.get(symbol)
        if instrument:
            return instrument.pip_size
        return 0.0001  # Default for major FX pairs

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
        self._best_day_paused_today = None
        # v1.3.9: Daily reset for operational metrics and low-confidence cooldown
        self._metrics.reset()
        self._low_confidence_cooldown.reset_all()
