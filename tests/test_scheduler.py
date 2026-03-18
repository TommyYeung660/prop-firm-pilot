"""
Tests for src/scheduler/scheduler.py — Async multi-cycle orchestrator.

Uses mocked ScannerBridge, AgentBridge, ExecutionEngine, and MatchTraderClient
with a real DecisionStore (in-memory SQLite). Tests cover all worker loops:
scanner, LLM worker, execution, janitor, and equity monitor.
"""

import asyncio
import json
import time
import unittest.mock
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.config import (
    AccountConfig,
    AppConfig,
    ComplianceConfig,
    DecisionStoreConfig,
    ExecutionConfig,
    MonitorConfig,
    SchedulerConfig,
)
from src.decision.agent_bridge import AgentDecision
from src.decision.schemas import TradeIntent
from src.decision_store.sqlite_store import DecisionStore, InvalidTransitionError
from src.optimize.optimization_state import OptimizationState, Thresholds
from src.scheduler.scheduler import Scheduler

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: object) -> DecisionStore:
    """Create a fresh DecisionStore with a temporary database."""
    db_path = f"{tmp_path}/test_scheduler.db"
    s = DecisionStore(db_path=db_path)
    yield s  # type: ignore[misc]
    s.close()


@pytest.fixture
def config() -> AppConfig:
    """Minimal AppConfig with short intervals for testing."""
    return AppConfig(
        account=AccountConfig(initial_balance=50000),
        compliance=ComplianceConfig(),
        scheduler=SchedulerConfig(
            scanner_interval_seconds=0,
            llm_poll_interval_seconds=0,
            execution_poll_interval_seconds=0,
            janitor_interval_seconds=0,
            llm_worker_count=1,
            equity_poll_interval_seconds=0,
            position_monitor_interval_seconds=0,
            daily_summary_hour_utc=22,
        ),
        decision_store=DecisionStoreConfig(),
        monitor=MonitorConfig(),
    )


@pytest.fixture
def mock_scanner() -> MagicMock:
    """Mock ScannerBridge that returns no signals by default."""
    scanner = MagicMock()
    scanner.run_pipeline.return_value = []
    return scanner


@pytest.fixture
def mock_agents() -> MagicMock:
    """Mock AgentBridge that returns BUY by default."""
    agents = MagicMock()
    agents.decide.return_value = AgentDecision(
        symbol="EURUSD",
        decision="BUY",
        final_state={"test": True},
        risk_report="test risk report",
    )
    agents.using_mock = False  # Default to NOT using mock for existing tests
    return agents


@pytest.fixture
def mock_engine() -> AsyncMock:
    """Mock ExecutionEngine that processes zero intents by default."""
    engine = AsyncMock()
    engine.execute_ready_intents.return_value = 0
    return engine


@pytest.fixture
def mock_matchtrader() -> AsyncMock:
    """Mock MatchTraderClient with default balance."""
    client = AsyncMock()
    client.get_balance.return_value = MagicMock(
        balance=50000.0,
        equity=50000.0,
        margin=0.0,
        free_margin=50000.0,
    )
    client.get_open_positions.return_value = []
    client.get_quote.return_value = {"ask": 1.0850, "bid": 1.0848}
    # Mock rate limiter for auto-throttle code in _position_monitor_loop
    rate_limiter = MagicMock()
    rate_limiter.remaining = 1800
    rate_limiter._daily_limit = 2000
    client._rate_limiter = rate_limiter
    return client


@pytest.fixture
def scheduler(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
) -> Scheduler:
    """Create a Scheduler with all mocked dependencies."""
    return Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )


def _make_mock_signal(
    instrument: str = "EURUSD",
    score: float = 0.85,
    confidence: str = "high",
    market_date: str = "2026-02-16",
    side: str | None = None,
    schema_version: str = "fx_signal_v1",
) -> MagicMock:
    """Create a mock ScannerSignal."""
    signal = MagicMock()
    signal.instrument = instrument
    signal.score = score
    signal.confidence = confidence
    signal.score_gap = 0.1
    signal.drop_distance = 0.05
    signal.topk_spread = 0.02
    signal.scanner_version = "v1.5.0_beta"
    signal.schema_version = schema_version
    signal.market_date = market_date
    signal.label_version = "cost_aware_directional_return_v1"
    signal.side = side
    return signal


async def _run_loop_once(scheduler: Scheduler, loop_coro) -> None:
    """Run a scheduler loop for exactly one iteration then stop.

    Patches asyncio.sleep so the loop body runs once, then _running is set
    to False causing the while-loop to exit.  Also patches asyncio.wait_for
    for the scanner loop's rescan-event wait (v1.2.0).
    """
    call_count = 0

    async def fake_sleep(seconds: float) -> None:
        nonlocal call_count
        call_count += 1
        if call_count >= 1:
            scheduler._running = False

    _orig_wait_for = asyncio.wait_for

    async def fake_wait_for(coro, *, timeout=None):
        nonlocal call_count
        call_count += 1
        if call_count >= 1:
            scheduler._running = False
        # Cancel the coroutine to avoid 'was never awaited' warnings
        coro.close()
        return None

    with (
        unittest.mock.patch("asyncio.sleep", fake_sleep),
        unittest.mock.patch("asyncio.wait_for", fake_wait_for),
    ):
        scheduler._running = True
        await loop_coro


# ── Scanner Loop Tests ──────────────────────────────────────────────────────


class TestScannerLoop:
    """Tests for Scheduler._scanner_loop()."""

    async def test_skips_intent_creation_when_best_day_headroom_is_exhausted(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
        tmp_path,
    ) -> None:
        """Scanner should skip candidates before intent creation when Best Day headroom is gone."""
        from src.monitor.trade_journal import TradeJournal

        journal = TradeJournal(tmp_path / "trade_journal.jsonl")
        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            trade_journal=journal,
        )
        today = Scheduler._today_str()
        safe_limit = config.compliance.best_day_limit * config.compliance.best_day_stop
        _advance_intent_to_closed(
            store,
            trade_date=today,
            symbol="GBPUSD",
            realized_pnl=safe_limit,
        )
        mock_matchtrader.get_balance.return_value = MagicMock(
            balance=50000.0 + safe_limit,
            equity=50000.0 + safe_limit,
            margin=0.0,
            free_margin=50000.0 + safe_limit,
        )
        mock_scanner.run_pipeline.return_value = [_make_mock_signal("EURUSD", market_date=today)]

        await _run_loop_once(sched, sched._scanner_loop())

        pending = store.claim_next_pending("llm-0")
        assert pending is None

        lines = journal._path.read_text(encoding="utf-8").strip().splitlines()
        events = [json.loads(line) for line in lines]
        skip_event = next(e for e in events if e["type"] == "SCANNER_SKIP")
        assert skip_event["symbol"] == "EURUSD"
        assert skip_event["reason"] == "compliance_headroom"
        assert skip_event["rule_name"] == "BEST_DAY_RULE"

        new_intents = [
            intent
            for intent in store.get_intents_by_date(today)
            if not (intent.status == "closed" and intent.symbol == "GBPUSD")
        ]
        assert new_intents == []

    async def test_creates_intents_from_signals(
        self,
        scheduler: Scheduler,
        mock_scanner: MagicMock,
        store: DecisionStore,
    ) -> None:
        """Should create TradeIntents in store from scanner signals."""
        mock_scanner.run_pipeline.return_value = [_make_mock_signal("EURUSD")]

        await _run_loop_once(scheduler, scheduler._scanner_loop())

        today = Scheduler._today_str()
        intents = store.get_intents_by_date(today)
        assert len(intents) == 1
        assert intents[0].symbol == "EURUSD"
        assert intents[0].source == "scanner"
        assert intents[0].status == "pending"

    async def test_creates_multiple_intents(
        self,
        scheduler: Scheduler,
        mock_scanner: MagicMock,
        store: DecisionStore,
    ) -> None:
        """Should create intents for each signal up to topk."""
        mock_scanner.run_pipeline.return_value = [
            _make_mock_signal("EURUSD"),
            _make_mock_signal("GBPUSD"),
        ]

        await _run_loop_once(scheduler, scheduler._scanner_loop())

        today = Scheduler._today_str()
        intents = store.get_intents_by_date(today)
        symbols = {i.symbol for i in intents}
        assert symbols == {"EURUSD", "GBPUSD"}

    async def test_persists_scanner_contract_metadata_in_intent(
        self,
        scheduler: Scheduler,
        mock_scanner: MagicMock,
        store: DecisionStore,
    ) -> None:
        """Scanner loop should persist versioned scanner metadata onto TradeIntent."""
        mock_scanner.run_pipeline.return_value = [
            _make_mock_signal("EURUSD", market_date=Scheduler._today_str())
        ]

        await _run_loop_once(scheduler, scheduler._scanner_loop())

        intents = store.get_intents_by_date(Scheduler._today_str())
        assert len(intents) == 1
        intent = intents[0]
        assert intent.scanner_version == "v1.5.0_beta"
        assert intent.scanner_schema_version == "fx_signal_v1"
        assert intent.scanner_market_date == Scheduler._today_str()
        assert intent.scanner_label_version == "cost_aware_directional_return_v1"

    async def test_scanner_loop_creates_long_and_short_intents_with_side(
        self,
        scheduler: Scheduler,
        mock_scanner: MagicMock,
        store: DecisionStore,
    ) -> None:
        mock_scanner.run_pipeline.return_value = [
            _make_mock_signal("EURUSD", score=0.83, side="long", schema_version="fx_signal_v2"),
            _make_mock_signal("USDCHF", score=0.14, side="short", schema_version="fx_signal_v2"),
        ]

        await _run_loop_once(scheduler, scheduler._scanner_loop())

        intents = store.get_intents_by_date(Scheduler._today_str())
        assert len(intents) == 2
        assert {intent.scanner_side for intent in intents} == {"long", "short"}

    async def test_scanner_loop_sorts_v2_candidates_by_directional_quality(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        config.scanner.topk = 1
        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
        )
        mock_scanner.run_pipeline.return_value = [
            _make_mock_signal("EURUSD", score=0.81, side="long", schema_version="fx_signal_v2"),
            _make_mock_signal("USDCHF", score=0.12, side="short", schema_version="fx_signal_v2"),
        ]

        await _run_loop_once(sched, sched._scanner_loop())

        intents = store.get_intents_by_date(Scheduler._today_str())
        assert len(intents) == 1
        assert intents[0].symbol == "USDCHF"
        assert intents[0].scanner_side == "short"

    async def test_scanner_loop_keeps_only_best_side_per_symbol_for_v2(
        self,
        scheduler: Scheduler,
        mock_scanner: MagicMock,
        store: DecisionStore,
    ) -> None:
        """A side-aware bundle must not create both long and short intents for one symbol."""
        mock_scanner.run_pipeline.return_value = [
            _make_mock_signal("USDCAD", score=0.62, side="long", schema_version="fx_signal_v2"),
            _make_mock_signal("USDCAD", score=0.62, side="short", schema_version="fx_signal_v2"),
        ]

        await _run_loop_once(scheduler, scheduler._scanner_loop())

        intents = store.get_intents_by_date(Scheduler._today_str())
        assert len(intents) == 1
        assert intents[0].symbol == "USDCAD"
        assert intents[0].scanner_side == "long"

    async def test_scanner_loop_duplicate_guard_blocks_opposite_side_when_symbol_pending(
        self,
        scheduler: Scheduler,
        mock_scanner: MagicMock,
        store: DecisionStore,
    ) -> None:
        store.insert_intent(
            TradeIntent(
                trade_date=Scheduler._today_str(),
                symbol="USDCHF",
                source="scanner",
                scanner_schema_version="fx_signal_v2",
                scanner_side="short",
            )
        )
        mock_scanner.run_pipeline.return_value = [
            _make_mock_signal("USDCHF", score=0.15, side="short", schema_version="fx_signal_v2"),
            _make_mock_signal("USDCHF", score=0.84, side="long", schema_version="fx_signal_v2"),
        ]

        await _run_loop_once(scheduler, scheduler._scanner_loop())

        intents = store.get_intents_by_date(Scheduler._today_str())
        assert len(intents) == 1
        assert intents[0].scanner_side == "short"

    async def test_logs_structured_cooldown_skip_event(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
        tmp_path,
    ) -> None:
        from src.monitor.trade_journal import TradeJournal

        journal = TradeJournal(tmp_path / "trade_journal.jsonl")
        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            trade_journal=journal,
        )
        mock_scanner.run_pipeline.return_value = [_make_mock_signal("EURUSD")]
        now = sched._now_utc()
        sched._low_confidence_cooldown.record_low_confidence("EURUSD", now)
        sched._low_confidence_cooldown.record_low_confidence("EURUSD", now)

        await _run_loop_once(sched, sched._scanner_loop())

        lines = journal._path.read_text(encoding="utf-8").strip().splitlines()
        events = [json.loads(line) for line in lines]
        skip_event = next(e for e in events if e["type"] == "SCANNER_SKIP")
        assert skip_event["symbol"] == "EURUSD"
        assert skip_event["reason"] == "low_confidence_cooldown"
        assert skip_event["consecutive_cancels"] == 2

    async def test_blocks_intent_creation_when_market_data_entry_not_safe(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
        tmp_path,
    ) -> None:
        from src.monitor.trade_journal import TradeJournal

        journal = TradeJournal(tmp_path / "trade_journal.jsonl")
        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            trade_journal=journal,
        )
        sched._market_data_ready = True
        sched._market_data_hub = MagicMock()
        sched._market_data_hub.get_entry_readiness = AsyncMock(
            return_value=MagicMock(
                entry_safe=False,
                block_reason="market_data.quote_unavailable",
                websocket_state="degraded",
                ws_last_error="keepalive ping timeout",
                quote_source="rest_fallback",
                bars_5m_source="rest_fallback",
                bars_1h_source="rest_fallback",
            )
        )
        sched._market_data_hub.feed_status.return_value = {
            "initialized_at": "2026-03-17T03:00:00+00:00",
            "uptime_seconds": 42,
            "websocket_closed_bar_counts": {
                "EURUSD": {"1m": 6, "5m": 1, "1h": 0},
                "USDCHF": {"1m": 0, "5m": 0, "1h": 0},
            },
        }
        mock_scanner.run_pipeline.return_value = [_make_mock_signal("EURUSD")]

        await _run_loop_once(sched, sched._scanner_loop())

        intents = store.get_intents_by_date(Scheduler._today_str())
        assert intents == []
        sched._market_data_hub.get_entry_readiness.assert_awaited_once_with("EURUSD")

        lines = journal._path.read_text(encoding="utf-8").strip().splitlines()
        events = [json.loads(line) for line in lines]
        skip_event = next(e for e in events if e["type"] == "SCANNER_SKIP")
        assert skip_event["symbol"] == "EURUSD"
        assert skip_event["reason"] == "market_data_entry_block"
        assert skip_event["entry_block_reason"] == "market_data.quote_unavailable"
        assert skip_event["feed_state"] == "degraded"
        assert skip_event["market_data_initialized_at"] == "2026-03-17T03:00:00+00:00"
        assert skip_event["market_data_uptime_seconds"] == 42
        assert skip_event["websocket_closed_bar_counts"]["EURUSD"] == {
            "1m": 6,
            "5m": 1,
            "1h": 0,
        }

    async def test_creates_intent_when_market_data_gap_is_retryable(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
        tmp_path,
    ) -> None:
        from src.monitor.trade_journal import TradeJournal

        journal = TradeJournal(tmp_path / "trade_journal.jsonl")
        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            trade_journal=journal,
        )
        sched._market_data_ready = True
        sched._market_data_hub = MagicMock()
        sched._market_data_hub.get_entry_readiness = AsyncMock(
            return_value=MagicMock(
                entry_safe=True,
                requires_tactical_retry=True,
                pending_reason="market_data.startup_5m_bar_pending",
                block_reason="",
                websocket_state="healthy",
                ws_last_error=None,
                quote_source="websocket_cache",
                bars_5m_source="rest_fallback",
                bars_1h_source="warmup_cache",
            )
        )
        sched._market_data_hub.feed_status.return_value = {
            "initialized_at": "2026-03-17T03:00:00+00:00",
            "uptime_seconds": 42,
            "websocket_closed_bar_counts": {
                "EURUSD": {"1m": 0, "5m": 0, "1h": 0},
            },
        }
        mock_scanner.run_pipeline.return_value = [_make_mock_signal("EURUSD")]

        await _run_loop_once(sched, sched._scanner_loop())

        intents = store.get_intents_by_date(Scheduler._today_str())
        assert len(intents) == 1
        sched._market_data_hub.get_entry_readiness.assert_awaited_once_with("EURUSD")

        lines = journal._path.read_text(encoding="utf-8").strip().splitlines()
        events = [json.loads(line) for line in lines]
        assert not any(e["type"] == "SCANNER_SKIP" for e in events)
        admitted_event = next(e for e in events if e["type"] == "SCANNER_ADMITTED")
        assert admitted_event["symbol"] == "EURUSD"
        assert admitted_event["reason"] == "market_data_startup_retryable"
        assert admitted_event["pending_reason"] == "market_data.startup_5m_bar_pending"
        assert admitted_event["quote_source"] == "websocket_cache"
        assert admitted_event["bars_5m_source"] == "rest_fallback"
        assert admitted_event["bars_1h_source"] == "warmup_cache"

    async def test_logs_scanner_bundle_rejection_reason_code(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
        tmp_path,
    ) -> None:
        """Rejected scanner bundles should surface explicit reason codes downstream."""
        from src.monitor.trade_journal import TradeJournal

        journal = TradeJournal(tmp_path / "trade_journal.jsonl")
        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            trade_journal=journal,
        )
        mock_scanner.run_pipeline.return_value = []
        mock_scanner.get_last_rejection_reason_code.return_value = "scanner.contract.invalid"

        await _run_loop_once(sched, sched._scanner_loop())

        lines = journal._path.read_text(encoding="utf-8").strip().splitlines()
        events = [json.loads(line) for line in lines]
        reject_event = next(e for e in events if e["type"] == "SCANNER_BUNDLE_REJECTED")
        assert reject_event["reason_code"] == "scanner.contract.invalid"

    async def test_skips_duplicate_intents(
        self,
        scheduler: Scheduler,
        mock_scanner: MagicMock,
        store: DecisionStore,
    ) -> None:
        """Should not create duplicate intents for same symbol+date+source."""
        mock_scanner.run_pipeline.return_value = [_make_mock_signal("EURUSD")]

        # Run twice — second time should skip
        await _run_loop_once(scheduler, scheduler._scanner_loop())

        # Manually re-enable and run again
        call_count = 0

        async def fake_sleep(seconds: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                scheduler._running = False

        async def fake_wait_for(coro, *, timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                scheduler._running = False
            coro.close()
            return None

        with (
            unittest.mock.patch("asyncio.sleep", fake_sleep),
            unittest.mock.patch("asyncio.wait_for", fake_wait_for),
        ):
            scheduler._running = True
            await scheduler._scanner_loop()
        today = Scheduler._today_str()
        intents = store.get_intents_by_date(today)
        assert len(intents) == 1  # Still only 1, not 2

    async def test_handles_pipeline_error(
        self,
        scheduler: Scheduler,
        mock_scanner: MagicMock,
        store: DecisionStore,
    ) -> None:
        """Should catch scanner errors without crashing the loop."""
        mock_scanner.run_pipeline.side_effect = RuntimeError("Scanner crashed")

        await _run_loop_once(scheduler, scheduler._scanner_loop())

        today = Scheduler._today_str()
        intents = store.get_intents_by_date(today)
        assert len(intents) == 0  # No intents created

    async def test_respects_topk(
        self,
        scheduler: Scheduler,
        mock_scanner: MagicMock,
        store: DecisionStore,
        config: AppConfig,
    ) -> None:
        """Should only create intents for top-K signals."""
        # Config scanner.topk defaults to 3
        signals = [_make_mock_signal(f"PAIR{i}") for i in range(5)]
        mock_scanner.run_pipeline.return_value = signals

        await _run_loop_once(scheduler, scheduler._scanner_loop())

        today = Scheduler._today_str()
        intents = store.get_intents_by_date(today)
        assert len(intents) == config.scanner.topk

    async def test_per_symbol_topk_deduplicates_same_symbol(
        self,
        scheduler: Scheduler,
        mock_scanner: MagicMock,
        store: DecisionStore,
    ) -> None:
        """When all signals are the same symbol, only 1 intent is created."""
        mock_scanner.run_pipeline.return_value = [
            _make_mock_signal("GBPUSD", score=0.95),
            _make_mock_signal("GBPUSD", score=0.90),
            _make_mock_signal("GBPUSD", score=0.85),
            _make_mock_signal("GBPUSD", score=0.80),
            _make_mock_signal("GBPUSD", score=0.75),
        ]

        await _run_loop_once(scheduler, scheduler._scanner_loop())

        today = Scheduler._today_str()
        intents = store.get_intents_by_date(today)
        assert len(intents) == 1
        assert intents[0].symbol == "GBPUSD"

    async def test_per_symbol_topk_picks_best_score(
        self,
        scheduler: Scheduler,
        mock_scanner: MagicMock,
        store: DecisionStore,
    ) -> None:
        """Should pick the highest-scoring signal per symbol."""
        mock_scanner.run_pipeline.return_value = [
            _make_mock_signal("EURUSD", score=0.7),
            _make_mock_signal("EURUSD", score=0.9),
            _make_mock_signal("EURUSD", score=0.8),
            _make_mock_signal("GBPUSD", score=0.6),
            _make_mock_signal("GBPUSD", score=0.85),
        ]

        await _run_loop_once(scheduler, scheduler._scanner_loop())

        today = Scheduler._today_str()
        intents = store.get_intents_by_date(today)
        symbols = {i.symbol for i in intents}
        assert symbols == {"EURUSD", "GBPUSD"}
        eur = [i for i in intents if i.symbol == "EURUSD"][0]
        gbp = [i for i in intents if i.symbol == "GBPUSD"][0]
        assert eur.scanner_score == 0.9
        assert gbp.scanner_score == 0.85

    async def test_per_symbol_topk_respects_topk_limit(
        self,
        scheduler: Scheduler,
        mock_scanner: MagicMock,
        store: DecisionStore,
        config: AppConfig,
    ) -> None:
        """With 5 symbols, only topk=3 intents created (highest scores)."""
        mock_scanner.run_pipeline.return_value = [
            _make_mock_signal("PAIR0", score=0.95),
            _make_mock_signal("PAIR0", score=0.80),
            _make_mock_signal("PAIR1", score=0.90),
            _make_mock_signal("PAIR1", score=0.70),
            _make_mock_signal("PAIR2", score=0.85),
            _make_mock_signal("PAIR2", score=0.60),
            _make_mock_signal("PAIR3", score=0.50),
            _make_mock_signal("PAIR3", score=0.40),
            _make_mock_signal("PAIR4", score=0.30),
            _make_mock_signal("PAIR4", score=0.20),
        ]

        await _run_loop_once(scheduler, scheduler._scanner_loop())

        today = Scheduler._today_str()
        intents = store.get_intents_by_date(today)
        assert len(intents) == config.scanner.topk  # 3
        symbols = {i.symbol for i in intents}
        # Top 3 best-per-symbol scores: PAIR0(0.95), PAIR1(0.90), PAIR2(0.85)
        assert symbols == {"PAIR0", "PAIR1", "PAIR2"}

    async def test_per_symbol_topk_diverse_over_monopoly(
        self,
        scheduler: Scheduler,
        mock_scanner: MagicMock,
        store: DecisionStore,
    ) -> None:
        """Even when one symbol dominates, diversity is preserved."""
        mock_scanner.run_pipeline.return_value = [
            _make_mock_signal("GBPUSD", score=0.95),
            _make_mock_signal("GBPUSD", score=0.94),
            _make_mock_signal("GBPUSD", score=0.93),
            _make_mock_signal("EURUSD", score=0.80),
            _make_mock_signal("USDJPY", score=0.75),
        ]

        await _run_loop_once(scheduler, scheduler._scanner_loop())

        today = Scheduler._today_str()
        intents = store.get_intents_by_date(today)
        symbols = {i.symbol for i in intents}
        # All 3 symbols should get intents, not 3x GBPUSD
        assert symbols == {"GBPUSD", "EURUSD", "USDJPY"}


# ── Scanner Capacity Check Tests ────────────────────────────────────────────


class TestScannerCapacityCheck:
    """Tests for BUG #4 fix: scanner loop respects max_positions capacity."""

    async def test_skips_all_when_positions_at_max(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """When open positions == max_positions, no new intents are created."""
        # Create a scheduler with max_positions=1
        config_1 = config.model_copy(update={"execution": ExecutionConfig(max_positions=1)})
        sched = Scheduler(
            config=config_1,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
        )

        # Insert an opened position to fill the single slot
        intent = TradeIntent(trade_date=Scheduler._today_str(), symbol="EURUSD")
        store.insert_intent(intent)
        store.claim_next_pending("llm-0")
        store.update_intent_decision(
            intent.id,
            side="BUY",
            sl_pips=20,
            tp_pips=40,
            risk_report="test",
            state_json="{}",
        )
        store.mark_ready_for_exec(intent.id)
        store.mark_executing(intent.id)
        store.mark_opened(intent.id, position_id="POS-001")

        # Scanner returns a new signal
        mock_scanner.run_pipeline.return_value = [_make_mock_signal("GBPUSD")]

        await _run_loop_once(sched, sched._scanner_loop())

        # No new intents should be created (only the old opened one)
        today = Scheduler._today_str()
        intents = store.get_intents_by_date(today)
        assert len(intents) == 1
        assert intents[0].symbol == "EURUSD"

    async def test_skips_all_when_pipeline_at_max(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """When pipeline intents fill max_positions, no new intents are created."""
        config_2 = config.model_copy(update={"execution": ExecutionConfig(max_positions=2)})
        sched = Scheduler(
            config=config_2,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
        )

        # Insert 2 pending intents (pipeline, not yet opened)
        store.insert_intent(
            TradeIntent(
                trade_date=Scheduler._today_str(),
                symbol="EURUSD",
                source="scanner",
            )
        )
        store.insert_intent(
            TradeIntent(
                trade_date=Scheduler._today_str(),
                symbol="GBPUSD",
                source="scanner",
            )
        )

        # Scanner returns another signal
        mock_scanner.run_pipeline.return_value = [_make_mock_signal("USDJPY")]

        await _run_loop_once(sched, sched._scanner_loop())

        today = Scheduler._today_str()
        intents = store.get_intents_by_date(today)
        symbols = {i.symbol for i in intents}
        # Only the 2 original intents, no USDJPY
        assert len(intents) == 2
        assert "USDJPY" not in symbols

    async def test_limits_new_intents_to_available_slots(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """When 1 slot is used, only create enough intents to fill remaining."""
        config_2 = config.model_copy(update={"execution": ExecutionConfig(max_positions=2)})
        sched = Scheduler(
            config=config_2,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
        )

        # 1 pending intent already in pipeline
        store.insert_intent(
            TradeIntent(
                trade_date=Scheduler._today_str(),
                symbol="EURUSD",
                source="scanner",
            )
        )

        # Scanner returns 3 signals — only 1 slot available
        mock_scanner.run_pipeline.return_value = [
            _make_mock_signal("GBPUSD", score=0.9),
            _make_mock_signal("USDJPY", score=0.8),
            _make_mock_signal("AUDUSD", score=0.7),
        ]

        await _run_loop_once(sched, sched._scanner_loop())

        today = Scheduler._today_str()
        intents = store.get_intents_by_date(today)
        # 1 original + 1 new = 2 total (max_positions=2)
        assert len(intents) == 2
        symbols = {i.symbol for i in intents}
        assert "EURUSD" in symbols  # original
        assert "GBPUSD" in symbols  # highest score new signal
        assert "USDJPY" not in symbols  # would exceed capacity

    async def test_mixed_open_and_pipeline_capacity(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Open positions + pipeline intents both count toward max_positions."""
        config_3 = config.model_copy(update={"execution": ExecutionConfig(max_positions=3)})
        sched = Scheduler(
            config=config_3,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
        )

        # 1 opened position
        opened = TradeIntent(trade_date=Scheduler._today_str(), symbol="EURUSD")
        store.insert_intent(opened)
        store.claim_next_pending("llm-0")
        store.update_intent_decision(
            opened.id,
            side="BUY",
            sl_pips=20,
            tp_pips=40,
            risk_report="test",
            state_json="{}",
        )
        store.mark_ready_for_exec(opened.id)
        store.mark_executing(opened.id)
        store.mark_opened(opened.id, position_id="POS-001")

        # 1 pending intent in pipeline
        store.insert_intent(
            TradeIntent(
                trade_date=Scheduler._today_str(),
                symbol="GBPUSD",
                source="scanner",
            )
        )

        # Scanner returns 2 signals — only 1 slot available
        mock_scanner.run_pipeline.return_value = [
            _make_mock_signal("USDJPY", score=0.9),
            _make_mock_signal("AUDUSD", score=0.8),
        ]

        await _run_loop_once(sched, sched._scanner_loop())

        today = Scheduler._today_str()
        intents = store.get_intents_by_date(today)
        # 1 opened + 1 pending + 1 new = 3 total (max_positions=3)
        assert len(intents) == 3
        symbols = {i.symbol for i in intents}
        assert "USDJPY" in symbols  # got the 1 available slot
        assert "AUDUSD" not in symbols  # would exceed capacity

    async def test_idempotency_check_still_works_with_capacity(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Idempotency (intent_exists) check still skips duplicates within capacity."""
        config_3 = config.model_copy(update={"execution": ExecutionConfig(max_positions=3)})
        sched = Scheduler(
            config=config_3,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
        )

        # Insert a pending intent for EURUSD
        store.insert_intent(
            TradeIntent(
                trade_date=Scheduler._today_str(),
                symbol="EURUSD",
                source="scanner",
            )
        )

        # Scanner returns EURUSD again + a new symbol
        mock_scanner.run_pipeline.return_value = [
            _make_mock_signal("EURUSD", score=0.9),  # duplicate — should be skipped
            _make_mock_signal("GBPUSD", score=0.8),  # new — should be created
        ]

        await _run_loop_once(sched, sched._scanner_loop())

        today = Scheduler._today_str()
        intents = store.get_intents_by_date(today)
        # 1 original EURUSD + 1 new GBPUSD (EURUSD not duplicated)
        assert len(intents) == 2
        symbols = {i.symbol for i in intents}
        assert symbols == {"EURUSD", "GBPUSD"}


# ── LLM Worker Loop Tests ──────────────────────────────────────────────────


class TestLLMWorkerLoop:
    """Tests for Scheduler._llm_worker_loop()."""

    async def test_claims_and_processes_pending(
        self,
        scheduler: Scheduler,
        mock_agents: MagicMock,
        store: DecisionStore,
    ) -> None:
        """Should claim a pending intent and call agents.decide."""
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.85,
            scanner_confidence="high",
        )
        store.insert_intent(intent)

        await _run_loop_once(scheduler, scheduler._llm_worker_loop("llm-0"))

        mock_agents.decide.assert_called_once()
        call_kwargs = mock_agents.decide.call_args
        assert call_kwargs[1]["symbol"] == "EURUSD"

    async def test_marks_actionable_ready_for_exec(
        self,
        scheduler: Scheduler,
        mock_agents: MagicMock,
        store: DecisionStore,
    ) -> None:
        """Should mark BUY/SELL intents as ready_for_exec."""
        mock_agents.decide.return_value = AgentDecision(
            symbol="EURUSD",
            decision="BUY",
            final_state={"test": True},
            risk_report="test",
        )
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.85,
            scanner_confidence="high",
        )
        store.insert_intent(intent)

        await _run_loop_once(scheduler, scheduler._llm_worker_loop("llm-0"))

        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "ready_for_exec"
        assert updated.suggested_side == "BUY"

    async def test_cancels_hold_decision(
        self,
        scheduler: Scheduler,
        mock_agents: MagicMock,
        store: DecisionStore,
    ) -> None:
        """Should cancel intents when LLM decides HOLD."""
        mock_agents.decide.return_value = AgentDecision(
            symbol="EURUSD",
            decision="HOLD",
            final_state={},
            risk_report="",
        )
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.85,
            scanner_confidence="high",
        )
        store.insert_intent(intent)

        await _run_loop_once(scheduler, scheduler._llm_worker_loop("llm-0"))

        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "cancelled"

    async def test_cancels_on_agent_error(
        self,
        scheduler: Scheduler,
        mock_agents: MagicMock,
        store: DecisionStore,
    ) -> None:
        """Should cancel (NOT fail) intent when agent raises exception."""
        mock_agents.decide.side_effect = RuntimeError("LLM API timeout")
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.85,
            scanner_confidence="high",
        )
        store.insert_intent(intent)

        await _run_loop_once(scheduler, scheduler._llm_worker_loop("llm-0"))

        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "cancelled"  # NOT "failed"

    async def test_marks_timed_out_on_agent_timeout(
        self,
        scheduler: Scheduler,
        mock_agents: MagicMock,
        store: DecisionStore,
    ) -> None:
        """LLM timeout should end in timed_out rather than generic cancellation."""
        mock_agents.decide.side_effect = TimeoutError("LLM API timeout")
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.85,
            scanner_confidence="high",
        )
        store.insert_intent(intent)

        await _run_loop_once(scheduler, scheduler._llm_worker_loop("llm-0"))

        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "timed_out"
        assert "LLM API timeout" in (updated.execution_error or "")

    async def test_sleeps_when_no_pending(
        self,
        scheduler: Scheduler,
        mock_agents: MagicMock,
    ) -> None:
        """Should sleep when no pending intents exist (not crash)."""
        # No intents inserted — claim_next_pending returns None
        sleep_calls = []

        async def track_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)
            scheduler._running = False

        with unittest.mock.patch("asyncio.sleep", track_sleep):
            scheduler._running = True
            await scheduler._llm_worker_loop("llm-0")

        assert len(sleep_calls) == 1
        mock_agents.decide.assert_not_called()

    async def test_scanner_tactical_mode_skips_llm_worker(
        self,
        scheduler: Scheduler,
        mock_agents: MagicMock,
        store: DecisionStore,
    ) -> None:
        """scanner_tactical should bypass LLM and derive side from scanner signal."""
        scheduler._config.scheduler.entry_funnel_mode = "scanner_tactical"
        scheduler._config.tactical.enabled = False
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.82,
            scanner_confidence="high",
            scanner_schema_version="fx_signal_v2",
            scanner_side="long",
        )
        store.insert_intent(intent)

        await _run_loop_once(scheduler, scheduler._llm_worker_loop("llm-0"))

        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "ready_for_exec"
        assert updated.suggested_side == "BUY"
        mock_agents.decide.assert_not_called()

    async def test_scanner_tactical_mode_cancels_when_scanner_side_missing(
        self,
        scheduler: Scheduler,
        mock_agents: MagicMock,
        store: DecisionStore,
    ) -> None:
        """scanner_tactical should cancel safely when side is absent/invalid."""
        scheduler._config.scheduler.entry_funnel_mode = "scanner_tactical"
        scheduler._config.tactical.enabled = False
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.82,
            scanner_confidence="high",
            scanner_schema_version="fx_signal_v2",
            scanner_side=None,
        )
        store.insert_intent(intent)

        await _run_loop_once(scheduler, scheduler._llm_worker_loop("llm-0"))

        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "cancelled"
        assert updated.execution_error is not None
        assert "scanner_side" in updated.execution_error
        mock_agents.decide.assert_not_called()

    async def test_scanner_tactical_mode_ignores_llm_pre_filter(
        self,
        scheduler: Scheduler,
        mock_agents: MagicMock,
        store: DecisionStore,
    ) -> None:
        """scanner_tactical should not be cancelled by LLM pre-filter thresholds."""
        scheduler._config.scheduler.entry_funnel_mode = "scanner_tactical"
        scheduler._config.tactical.enabled = False
        scheduler._config.scheduler.llm_threshold_override.enabled = True
        scheduler._config.scheduler.llm_threshold_override.min_confidence = "high"
        scheduler._config.scheduler.llm_threshold_override.min_blended_confidence = 0.95
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.20,
            scanner_confidence="low",
            scanner_schema_version="fx_signal_v2",
            scanner_side="long",
        )
        store.insert_intent(intent)

        await _run_loop_once(scheduler, scheduler._llm_worker_loop("llm-0"))

        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "ready_for_exec"
        assert updated.suggested_side == "BUY"
        assert updated.execution_error != "LLM pre-filter: low confidence"
        mock_agents.decide.assert_not_called()

    async def test_scanner_tactical_mode_ignores_llm_post_filter(
        self,
        scheduler: Scheduler,
        mock_agents: MagicMock,
        store: DecisionStore,
    ) -> None:
        """scanner_tactical should not be cancelled by LLM post-filter thresholds."""
        scheduler._config.scheduler.entry_funnel_mode = "scanner_tactical"
        scheduler._config.tactical.enabled = False
        scheduler._config.scheduler.llm_threshold_override.enabled = True
        scheduler._config.scheduler.llm_threshold_override.min_confidence = "low"
        scheduler._config.scheduler.llm_threshold_override.min_blended_confidence = 0.9
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.95,
            scanner_confidence="high",
            scanner_schema_version="fx_signal_v2",
            scanner_side="long",
        )
        store.insert_intent(intent)

        with patch("src.scheduler.scheduler.format_decision") as mock_format:
            mock_format.return_value = MagicMock(
                confidence_score=0.1,
                suggested_sl_pips=25.0,
                suggested_tp_pips=50.0,
            )
            await _run_loop_once(scheduler, scheduler._llm_worker_loop("llm-0"))

        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "ready_for_exec"
        assert updated.suggested_side == "BUY"
        assert updated.execution_error != "LLM post-filter: low confidence"
        mock_agents.decide.assert_not_called()

    async def test_scanner_llm_tactical_mode_keeps_current_path(
        self,
        scheduler: Scheduler,
        mock_agents: MagicMock,
        store: DecisionStore,
    ) -> None:
        """scanner_llm_tactical should keep existing LLM decision flow."""
        scheduler._config.scheduler.entry_funnel_mode = "scanner_llm_tactical"
        scheduler._config.tactical.enabled = False
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.82,
            scanner_confidence="high",
            scanner_schema_version="fx_signal_v2",
            scanner_side="long",
        )
        store.insert_intent(intent)

        await _run_loop_once(scheduler, scheduler._llm_worker_loop("llm-0"))

        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "ready_for_exec"
        mock_agents.decide.assert_called_once()

    async def test_no_trade_mode_never_marks_ready_for_exec(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
    ) -> None:
        """no_trade should keep evidence but never transition to ready_for_exec."""
        scheduler._config.scheduler.entry_funnel_mode = "no_trade"
        scheduler._config.tactical.enabled = False
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.82,
            scanner_confidence="high",
            scanner_schema_version="fx_signal_v2",
            scanner_side="long",
        )
        store.insert_intent(intent)

        with patch.object(
            store, "mark_ready_for_exec", wraps=store.mark_ready_for_exec
        ) as mark_ready:
            await _run_loop_once(scheduler, scheduler._llm_worker_loop("llm-0"))

        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "cancelled"
        assert mark_ready.call_count == 0

    async def test_no_trade_mode_skips_llm_decide(
        self,
        scheduler: Scheduler,
        mock_agents: MagicMock,
        store: DecisionStore,
    ) -> None:
        """no_trade should not call AgentBridge.decide at all."""
        scheduler._config.scheduler.entry_funnel_mode = "no_trade"
        scheduler._config.tactical.enabled = False
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.82,
            scanner_confidence="high",
            scanner_schema_version="fx_signal_v2",
            scanner_side="long",
        )
        store.insert_intent(intent)

        await _run_loop_once(scheduler, scheduler._llm_worker_loop("llm-0"))

        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "cancelled"
        assert updated.execution_error is not None
        assert "no_trade mode" in updated.execution_error
        mock_agents.decide.assert_not_called()

    async def test_tactical_only_mode_rejects_scanner_intents(
        self,
        scheduler: Scheduler,
        mock_agents: MagicMock,
        store: DecisionStore,
    ) -> None:
        """tactical_only should be conservatively gated until tactical source exists."""
        scheduler._config.scheduler.entry_funnel_mode = "tactical_only"
        scheduler._config.tactical.enabled = False
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.82,
            scanner_confidence="high",
            scanner_schema_version="fx_signal_v2",
            scanner_side="long",
        )
        store.insert_intent(intent)

        await _run_loop_once(scheduler, scheduler._llm_worker_loop("llm-0"))

        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "cancelled"
        assert updated.execution_error is not None
        assert "tactical_only" in updated.execution_error
        mock_agents.decide.assert_not_called()


# ── Process Claimed Intent Tests ────────────────────────────────────────────


class TestProcessClaimedIntent:
    """Tests for Scheduler._process_claimed_intent()."""

    async def test_buy_decision_flow(
        self,
        scheduler: Scheduler,
        mock_agents: MagicMock,
        store: DecisionStore,
    ) -> None:
        """BUY decision → update_intent_decision + mark_ready_for_exec."""
        mock_agents.decide.return_value = AgentDecision(
            symbol="EURUSD",
            decision="BUY",
            final_state={"risk_report": "moderate"},
            risk_report="moderate risk",
        )
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.85,
            scanner_confidence="high",
        )
        store.insert_intent(intent)
        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None

        await scheduler._process_claimed_intent("llm-0", claimed)

        updated = store.get_intent(intent.id)
        assert updated.status == "ready_for_exec"
        assert updated.suggested_side == "BUY"
        assert updated.suggested_sl_pips is not None
        assert updated.suggested_tp_pips is not None

    async def test_sell_decision_flow(
        self,
        scheduler: Scheduler,
        mock_agents: MagicMock,
        store: DecisionStore,
    ) -> None:
        """SELL decision → same ready_for_exec flow."""
        mock_agents.decide.return_value = AgentDecision(
            symbol="GBPUSD",
            decision="SELL",
            final_state={"risk_report": "low"},
            risk_report="low risk",
        )
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="GBPUSD",
            scanner_score=0.75,
            scanner_confidence="medium",
        )
        store.insert_intent(intent)
        claimed = store.claim_next_pending("llm-0")

        await scheduler._process_claimed_intent("llm-0", claimed)

        updated = store.get_intent(intent.id)
        assert updated.status == "ready_for_exec"
        assert updated.suggested_side == "SELL"

    async def test_hold_decision_cancels(
        self,
        scheduler: Scheduler,
        mock_agents: MagicMock,
        store: DecisionStore,
    ) -> None:
        """HOLD decision → mark_cancelled."""
        mock_agents.decide.return_value = AgentDecision(
            symbol="EURUSD",
            decision="HOLD",
            final_state={},
            risk_report="",
        )
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.60,
            scanner_confidence="low",
        )
        store.insert_intent(intent)
        claimed = store.claim_next_pending("llm-0")

        await scheduler._process_claimed_intent("llm-0", claimed)

        updated = store.get_intent(intent.id)
        assert updated.status == "cancelled"

    async def test_actionable_decision_cancelled_when_best_day_pause_active(
        self,
        scheduler: Scheduler,
        mock_agents: MagicMock,
        store: DecisionStore,
    ) -> None:
        """Actionable decision should be cancelled if Best Day pause is active."""
        mock_agents.decide.return_value = AgentDecision(
            symbol="EURUSD",
            decision="BUY",
            final_state={"risk_report": "pause active"},
            risk_report="pause active",
        )
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.88,
            scanner_confidence="high",
        )
        store.insert_intent(intent)
        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None

        with patch.object(scheduler, "_should_pause_new_entries", return_value=True):
            await scheduler._process_claimed_intent("llm-0", claimed)

        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "cancelled"
        assert updated.execution_error is not None
        assert "Best Day protection active" in updated.execution_error

    async def test_hold_cancel_state_race_is_tolerated(
        self,
        scheduler: Scheduler,
        mock_agents: MagicMock,
        store: DecisionStore,
    ) -> None:
        """HOLD path should not raise if cancellation loses a state race."""
        mock_agents.decide.return_value = AgentDecision(
            symbol="EURUSD",
            decision="HOLD",
            final_state={},
            risk_report="",
        )
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.60,
            scanner_confidence="low",
        )
        store.insert_intent(intent)
        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None

        with patch.object(
            store,
            "mark_cancelled",
            side_effect=InvalidTransitionError(
                f"Cannot cancel {claimed.id}: not in 'pending' or 'claimed' state"
            ),
        ):
            await scheduler._process_claimed_intent("llm-0", claimed)

    async def test_actionable_stale_claim_race_is_tolerated(
        self,
        scheduler: Scheduler,
        mock_agents: MagicMock,
        store: DecisionStore,
    ) -> None:
        """Actionable path should not raise when claim expires before mark_ready."""
        mock_agents.decide.return_value = AgentDecision(
            symbol="GBPUSD",
            decision="SELL",
            final_state={},
            risk_report="",
        )
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="GBPUSD",
            scanner_score=0.80,
            scanner_confidence="medium",
            claim_ttl_minutes=0,
        )
        store.insert_intent(intent)
        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None
        store.recycle_expired_claims()

        await scheduler._process_claimed_intent("llm-0", claimed)

        latest = store.get_intent(intent.id)
        assert latest is not None
        assert latest.status == "timed_out"
        assert latest.suggested_side is None

    async def test_short_candidate_buy_decision_is_cancelled_as_direction_mismatch(
        self,
        scheduler: Scheduler,
        mock_agents: MagicMock,
        store: DecisionStore,
    ) -> None:
        scheduler._config.tactical.enabled = False
        mock_agents.decide.return_value = AgentDecision(
            symbol="USDCHF",
            decision="BUY",
            final_state={"risk_report": "countertrend buy"},
            risk_report="countertrend buy",
        )
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="USDCHF",
            scanner_score=0.12,
            scanner_confidence="high",
            scanner_schema_version="fx_signal_v2",
            scanner_side="short",
        )
        store.insert_intent(intent)
        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None

        await scheduler._process_claimed_intent("llm-0", claimed)

        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "cancelled"
        assert updated.execution_error == "direction_mismatch"

    async def test_long_candidate_sell_decision_is_cancelled_as_direction_mismatch(
        self,
        scheduler: Scheduler,
        mock_agents: MagicMock,
        store: DecisionStore,
    ) -> None:
        scheduler._config.tactical.enabled = False
        mock_agents.decide.return_value = AgentDecision(
            symbol="EURUSD",
            decision="SELL",
            final_state={"risk_report": "countertrend sell"},
            risk_report="countertrend sell",
        )
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.88,
            scanner_confidence="high",
            scanner_schema_version="fx_signal_v2",
            scanner_side="long",
        )
        store.insert_intent(intent)
        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None

        await scheduler._process_claimed_intent("llm-0", claimed)

        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "cancelled"
        assert updated.execution_error == "direction_mismatch"

    def test_decision_cache_key_includes_scanner_side(self, scheduler: Scheduler) -> None:
        intent_long = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="USDCHF",
            scanner_score=0.12,
            scanner_confidence="high",
            scanner_schema_version="fx_signal_v2",
            scanner_side="long",
        )
        intent_short = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="USDCHF",
            scanner_score=0.12,
            scanner_confidence="high",
            scanner_schema_version="fx_signal_v2",
            scanner_side="short",
        )

        assert scheduler._decision_cache_key(intent_long) != scheduler._decision_cache_key(
            intent_short
        )

    async def test_pre_filter_logs_threshold_source_and_values(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
        tmp_path,
    ) -> None:
        """Pre-filter cancellation event should include threshold source and values."""
        from src.monitor.trade_journal import TradeJournal

        journal = TradeJournal(tmp_path / "trade_journal.jsonl")
        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            trade_journal=journal,
        )
        sched._optimization_state = OptimizationState(
            global_thresholds=Thresholds(min_confidence="low", min_blended_confidence=0.1),
            symbol_thresholds={
                "EURUSD": Thresholds(min_confidence="low", min_blended_confidence=0.1)
            },
        )
        sched._config.scheduler.llm_threshold_override.enabled = True
        sched._config.scheduler.llm_threshold_override.min_confidence = "high"
        sched._config.scheduler.llm_threshold_override.min_blended_confidence = 0.9

        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.35,
            scanner_confidence="low",
        )
        store.insert_intent(intent)
        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None

        await sched._process_claimed_intent("llm-0", claimed)

        lines = journal._path.read_text(encoding="utf-8").strip().splitlines()
        events = [json.loads(line) for line in lines]
        cancel_event = next(
            event
            for event in events
            if event["type"] == "INTENT_CANCELLED"
            and event["reason"] == "LLM pre-filter: low confidence"
        )
        assert cancel_event["threshold_source"] == "override"
        assert cancel_event["threshold_min_confidence"] == "high"
        assert cancel_event["threshold_min_blended_confidence"] == 0.9

    async def test_post_filter_logs_threshold_source_and_values(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
        tmp_path,
    ) -> None:
        """Post-filter decision/cancel events should include threshold source and values."""
        from src.monitor.trade_journal import TradeJournal

        journal = TradeJournal(tmp_path / "trade_journal.jsonl")
        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            trade_journal=journal,
        )
        sched._optimization_state = OptimizationState(
            global_thresholds=Thresholds(min_confidence="low", min_blended_confidence=0.8)
        )
        sched._config.scheduler.llm_threshold_override.enabled = False

        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.95,
            scanner_confidence="high",
        )
        store.insert_intent(intent)
        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None

        with patch("src.scheduler.scheduler.format_decision") as mock_format:
            mock_format.return_value = MagicMock(confidence_score=0.4)
            await sched._process_claimed_intent("llm-0", claimed)

        lines = journal._path.read_text(encoding="utf-8").strip().splitlines()
        events = [json.loads(line) for line in lines]
        decision_event = next(event for event in events if event["type"] == "LLM_DECISION")
        cancel_event = next(
            event
            for event in events
            if event["type"] == "INTENT_CANCELLED"
            and event["reason"] == "LLM post-filter: low confidence"
        )

        assert decision_event["threshold_source"] == "dynamic"
        assert decision_event["threshold_min_confidence"] == "low"
        assert decision_event["threshold_min_blended_confidence"] == 0.8
        assert cancel_event["threshold_source"] == "dynamic"
        assert cancel_event["threshold_min_confidence"] == "low"
        assert cancel_event["threshold_min_blended_confidence"] == 0.8

    async def test_memory_journal_context_includes_threshold_fields(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
    ) -> None:
        """Memory journal decision context should include threshold metadata."""
        scheduler._memory_journal = MagicMock()
        scheduler._optimization_state = OptimizationState(
            global_thresholds=Thresholds(min_confidence="high", min_blended_confidence=0.95)
        )
        scheduler._config.scheduler.llm_threshold_override.enabled = True
        scheduler._config.scheduler.llm_threshold_override.min_confidence = "medium"
        scheduler._config.scheduler.llm_threshold_override.min_blended_confidence = 0.62

        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.85,
            scanner_confidence="high",
        )
        store.insert_intent(intent)
        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None

        await scheduler._process_claimed_intent("llm-0", claimed)

        scheduler._memory_journal.log_decision.assert_called_once()
        context = scheduler._memory_journal.log_decision.call_args.kwargs["context"]
        assert context["threshold_source"] == "override"
        assert context["threshold_min_confidence"] == "medium"
        assert context["threshold_min_blended_confidence"] == 0.62


# ── Execution Loop Tests ────────────────────────────────────────────────────


class TestExecutionLoop:
    """Tests for Scheduler._execution_loop()."""

    async def test_delegates_to_engine(
        self,
        scheduler: Scheduler,
        mock_engine: AsyncMock,
    ) -> None:
        """Should call engine.execute_ready_intents() each iteration."""
        await _run_loop_once(scheduler, scheduler._execution_loop())

        mock_engine.execute_ready_intents.assert_called_once()

    async def test_handles_engine_error(
        self,
        scheduler: Scheduler,
        mock_engine: AsyncMock,
    ) -> None:
        """Should catch engine errors without crashing the loop."""
        mock_engine.execute_ready_intents.side_effect = RuntimeError("DB locked")

        await _run_loop_once(scheduler, scheduler._execution_loop())

        # Loop completed without raising
        mock_engine.execute_ready_intents.assert_called_once()


# ── Janitor Loop Tests ──────────────────────────────────────────────────────


class TestJanitorLoop:
    """Tests for Scheduler._janitor_loop()."""

    async def test_runs_cleanup_cycle(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
    ) -> None:
        """Should call janitor.run_cycle() without error."""
        await _run_loop_once(scheduler, scheduler._janitor_loop())
        # If we get here, janitor ran successfully (real store, no expired claims)

    async def test_handles_janitor_error(
        self,
        scheduler: Scheduler,
    ) -> None:
        """Should catch janitor errors without crashing the loop."""
        scheduler._janitor = MagicMock()
        scheduler._janitor.run_cycle.side_effect = RuntimeError("Cleanup failed")

        await _run_loop_once(scheduler, scheduler._janitor_loop())
        # Loop completed without raising


# ── Equity Monitor Loop Tests ───────────────────────────────────────────────


class TestEquityMonitorLoop:
    """Tests for Scheduler._equity_monitor_loop()."""

    async def test_calls_get_balance(
        self,
        scheduler: Scheduler,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should fetch initial balance from MatchTrader."""
        # EquityMonitor.start is a long-running loop — we need to mock it
        scheduler._equity_monitor = MagicMock()
        scheduler._equity_monitor.start = AsyncMock()

        await scheduler._equity_monitor_loop()

        mock_matchtrader.get_balance.assert_called()

    async def test_passes_alert_and_emergency_callbacks(
        self,
        scheduler: Scheduler,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should wire alert/emergency/equity callbacks into EquityMonitor.start()."""
        scheduler._equity_monitor = MagicMock()
        scheduler._equity_monitor.start = AsyncMock()
        scheduler._alert_service = MagicMock()

        await scheduler._equity_monitor_loop()

        call_kwargs = scheduler._equity_monitor.start.call_args.kwargs
        assert callable(call_kwargs["on_alert"])
        assert callable(call_kwargs["on_emergency_close"])
        assert callable(call_kwargs["on_equity_snapshot"])

    async def test_handles_balance_error(
        self,
        scheduler: Scheduler,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should catch balance fetch errors without crashing."""
        mock_matchtrader.get_balance.side_effect = RuntimeError("Auth expired")

        await scheduler._equity_monitor_loop()
        # Loop completed without raising


# ── Start/Stop Tests ────────────────────────────────────────────────────────


class TestStartStop:
    """Tests for Scheduler.start() and stop()."""

    async def test_stop_sets_running_false(self, scheduler: Scheduler) -> None:
        """Should set _running to False and stop equity monitor."""
        scheduler._running = True
        scheduler._equity_monitor = MagicMock()

        await scheduler.stop()

        assert scheduler._running is False
        scheduler._equity_monitor.stop.assert_called_once()

    async def test_start_sets_running_true(self, scheduler: Scheduler) -> None:
        """start() should set _running = True before launching workers."""
        # We'll mock asyncio.gather to avoid actually running loops
        with patch("asyncio.gather", new_callable=AsyncMock) as mock_gather:
            await scheduler.start()

        assert mock_gather.called
        # Should have 7 coroutines: scanner + execution + janitor + equity
        #   + position_monitor + daily_summary + 1 LLM worker
        args = mock_gather.call_args[0]
        assert len(args) == 7
        # Clean up unawaited coroutines / scheduled tasks to suppress warnings
        for item in args:
            if hasattr(item, "close"):
                item.close()
            elif hasattr(item, "cancel"):
                item.cancel()

    async def test_start_initializes_market_data_hub_before_workers(
        self,
        scheduler: Scheduler,
    ) -> None:
        """start() should warm up market data before worker loops begin."""

        async def fake_initialize() -> None:
            scheduler._market_data_hub = object()
            scheduler._market_data_ready = True

        async def fake_gather(*args, **kwargs) -> None:
            assert scheduler._market_data_hub is not None
            assert scheduler._market_data_ready is True

        scheduler._initialize_market_data_hub = AsyncMock(side_effect=fake_initialize)

        with patch("asyncio.gather", new=AsyncMock(side_effect=fake_gather)) as mock_gather:
            await scheduler.start()

        scheduler._initialize_market_data_hub.assert_awaited_once()
        assert mock_gather.called
        args = mock_gather.call_args[0]
        for item in args:
            if hasattr(item, "close"):
                item.close()
            elif hasattr(item, "cancel"):
                item.cancel()

    async def test_start_with_multiple_llm_workers(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should spawn extra LLM workers based on config."""
        config.scheduler.llm_worker_count = 3
        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
        )

        with patch("asyncio.gather", new_callable=AsyncMock) as mock_gather:
            await sched.start()

        # 6 base loops + 3 LLM workers = 9
        args = mock_gather.call_args[0]
        assert len(args) == 9
        # Clean up unawaited coroutines / scheduled tasks to suppress warnings
        for item in args:
            if hasattr(item, "close"):
                item.close()
            elif hasattr(item, "cancel"):
                item.cancel()

    async def test_initialize_market_data_hub_injects_broker_quote_provider(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
        monkeypatch,
    ) -> None:
        """MarketDataHub must receive an async broker-quote callback at initialization."""
        monkeypatch.setenv("EODHD_API_KEY", "test-key")
        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
        )

        mock_hub = MagicMock()
        mock_hub.warmup = AsyncMock()
        mock_ws = MagicMock()
        mock_ws.register_tick_callback = MagicMock()
        mock_ws.run = AsyncMock(return_value=None)

        with (
            patch("src.scheduler.scheduler.EODHDFXWebSocketClient", return_value=mock_ws),
            patch("src.scheduler.scheduler.MarketDataHub", return_value=mock_hub) as hub_cls,
            patch("src.scheduler.scheduler.asyncio.create_task", return_value=MagicMock()),
        ):
            await sched._initialize_market_data_hub()

        kwargs = hub_cls.call_args.kwargs
        assert "broker_quote_provider" in kwargs
        assert callable(kwargs["broker_quote_provider"])
        await kwargs["broker_quote_provider"]("EURUSD")
        mock_matchtrader.get_quote.assert_awaited_once_with("EURUSD")


# ── Helper Method Tests ─────────────────────────────────────────────────────


class TestHelpers:
    """Tests for static helper methods."""

    def test_today_str_format(self) -> None:
        """Should return date in YYYY-MM-DD format."""
        today = Scheduler._today_str()
        assert len(today) == 10
        assert today[4] == "-"
        assert today[7] == "-"

    def test_now_utc_is_aware(self) -> None:
        """Should return timezone-aware datetime in UTC."""
        now = Scheduler._now_utc()
        assert now.tzinfo is not None


# ── Phase 2C: Startup Recovery Tests ────────────────────────────────────────


class TestRecoverStaleClaims:
    """Tests for Scheduler.recover_stale_claims()."""

    async def test_recovers_expired_claimed_intents(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
    ) -> None:
        """Should recycle stale claimed intents on startup."""
        from datetime import timedelta

        # Insert an intent, claim it, and backdate expires_at to make it stale
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.85,
            scanner_confidence="high",
        )
        store.insert_intent(intent)
        claimed = store.claim_next_pending("old-worker")
        assert claimed is not None

        # Backdate expires_at to make it expired
        store._conn.execute(
            "UPDATE intents SET expires_at = ? WHERE id = ?",
            (
                (Scheduler._now_utc() - timedelta(hours=1)).isoformat(),
                intent.id,
            ),
        )
        store._conn.commit()

        recovered = await scheduler.recover_stale_claims()
        assert recovered == 1

        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "timed_out"

    async def test_returns_zero_when_no_stale_claims(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
    ) -> None:
        """Should return 0 when there are no stale claims."""
        recovered = await scheduler.recover_stale_claims()
        assert recovered == 0

    async def test_sends_alert_on_recovery(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should send a Telegram alert when stale claims are recovered."""
        from datetime import timedelta

        mock_alert = AsyncMock()
        mock_alert.send = AsyncMock(return_value=True)

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            alert_service=mock_alert,
        )

        # Insert and claim an intent, then backdate
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="GBPUSD",
            scanner_score=0.70,
            scanner_confidence="medium",
        )
        store.insert_intent(intent)
        store.claim_next_pending("dead-worker")
        store._conn.execute(
            "UPDATE intents SET expires_at = ? WHERE id = ?",
            (
                (Scheduler._now_utc() - timedelta(hours=1)).isoformat(),
                intent.id,
            ),
        )
        store._conn.commit()

        recovered = await sched.recover_stale_claims()
        assert recovered == 1
        mock_alert.send.assert_called_once()
        alert_msg = mock_alert.send.call_args[0][0]
        assert "Recovery" in alert_msg
        assert "1" in alert_msg

    async def test_no_alert_when_clean_startup(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should NOT send alert when no stale claims exist."""
        mock_alert = AsyncMock()
        mock_alert.send = AsyncMock(return_value=True)

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            alert_service=mock_alert,
        )

        recovered = await sched.recover_stale_claims()
        assert recovered == 0
        mock_alert.send.assert_not_called()


# ── Phase 2C: Alert Integration Tests ──────────────────────────────────────


class TestAlertIntegration:
    """Tests for _send_alert() and alert calls in worker loops."""

    async def test_send_alert_when_service_configured(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should call alert_service.send() when configured."""
        mock_alert = AsyncMock()
        mock_alert.send = AsyncMock(return_value=True)

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            alert_service=mock_alert,
        )

        await sched._send_alert("Test message")
        mock_alert.send.assert_called_once_with("Test message")

    async def test_send_alert_skips_when_no_service(
        self,
        scheduler: Scheduler,
    ) -> None:
        """Should silently skip when alert_service is None."""
        assert scheduler._alert_service is None
        # Should not raise
        await scheduler._send_alert("This should not crash")

    async def test_send_alert_catches_alert_errors(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should catch errors from alert_service.send() without crashing."""
        mock_alert = AsyncMock()
        mock_alert.send = AsyncMock(side_effect=RuntimeError("Telegram down"))

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            alert_service=mock_alert,
        )

        # Should not raise
        await sched._send_alert("This alert will fail")
        mock_alert.send.assert_called_once()

    async def test_scanner_loop_sends_alert_on_intent_creation(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Scanner loop should send alert when creating a new intent."""
        mock_alert = AsyncMock()
        mock_alert.send = AsyncMock(return_value=True)

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            alert_service=mock_alert,
        )
        mock_scanner.run_pipeline.return_value = [_make_mock_signal("EURUSD")]

        await _run_loop_once(sched, sched._scanner_loop())

        # Should have sent an alert for the created intent
        mock_alert.send.assert_called()
        alert_msg = mock_alert.send.call_args[0][0]
        assert "EURUSD" in alert_msg
        assert "Intent" in alert_msg

    async def test_scanner_loop_sends_alert_on_error(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Scanner loop should send alert when scanner errors."""
        mock_alert = AsyncMock()
        mock_alert.send = AsyncMock(return_value=True)

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            alert_service=mock_alert,
        )
        mock_scanner.run_pipeline.side_effect = RuntimeError("Scanner crashed")

        await _run_loop_once(sched, sched._scanner_loop())

        mock_alert.send.assert_called()
        alert_msg = mock_alert.send.call_args[0][0]
        assert "Scanner" in alert_msg or "Error" in alert_msg

    async def test_execution_loop_sends_alert_on_error(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Execution loop should send alert on engine error."""
        mock_alert = AsyncMock()
        mock_alert.send = AsyncMock(return_value=True)

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            alert_service=mock_alert,
        )
        mock_engine.execute_ready_intents.side_effect = RuntimeError("DB locked")

        await _run_loop_once(sched, sched._execution_loop())

        mock_alert.send.assert_called()
        alert_msg = mock_alert.send.call_args[0][0]
        assert "Execution" in alert_msg or "Error" in alert_msg

    async def test_llm_worker_sends_alert_on_error(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """LLM worker loop should send alert on agent error."""
        mock_alert = AsyncMock()
        mock_alert.send = AsyncMock(return_value=True)

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            alert_service=mock_alert,
        )
        mock_agents.decide.side_effect = RuntimeError("LLM API timeout")

        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.85,
            scanner_confidence="high",
        )
        store.insert_intent(intent)

        await _run_loop_once(sched, sched._llm_worker_loop("llm-0"))

        mock_alert.send.assert_called()
        alert_msg = mock_alert.send.call_args[0][0]
        assert "LLM" in alert_msg or "Worker" in alert_msg


# ── Position Monitor Loop Tests ────────────────────────────────────────────


def _advance_intent_to_opened(store: DecisionStore, symbol: str = "EURUSD") -> TradeIntent:
    """Insert an intent and advance it through the state machine to 'opened'.

    Returns the intent with position_id set.
    """
    intent = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol=symbol,
        scanner_score=0.85,
        scanner_confidence="high",
    )
    store.insert_intent(intent)
    store.claim_next_pending("test-worker")
    store.update_intent_decision(
        intent.id,
        side="BUY",
        sl_pips=30.0,
        tp_pips=50.0,
        risk_report="test risk",
        state_json="{}",
    )
    store.mark_ready_for_exec(intent.id)
    store.mark_executing(intent.id)
    store.mark_opened(intent.id, position_id="pos_123")
    return store.get_intent(intent.id)


def _advance_intent_to_closed(
    store: DecisionStore,
    trade_date: str,
    symbol: str = "EURUSD",
    realized_pnl: float = 0.0,
) -> TradeIntent:
    """Insert and fully close an intent with realized PnL."""
    intent = TradeIntent(
        trade_date=trade_date,
        symbol=symbol,
        scanner_score=0.85,
        scanner_confidence="high",
    )
    store.insert_intent(intent)
    store.claim_next_pending("test-worker")
    store.update_intent_decision(
        intent.id,
        side="BUY",
        sl_pips=30.0,
        tp_pips=50.0,
        risk_report="test risk",
        state_json="{}",
    )
    store.mark_ready_for_exec(intent.id)
    store.mark_executing(intent.id)
    store.mark_opened(intent.id, position_id=f"closed-{symbol}")
    store.mark_closed(intent.id, realized_pnl=realized_pnl, exit_reason="tp_hit")
    closed = store.get_intent(intent.id)
    assert closed is not None
    return closed


class TestPositionMonitorLoop:
    """Tests for Scheduler._position_monitor_loop()."""

    async def test_detects_closed_position(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should detect when a position is no longer in open positions."""
        mock_alert = AsyncMock()
        mock_alert.trade_closed = AsyncMock()
        mock_alert.sl_tp_hit = AsyncMock()
        mock_alert.send = AsyncMock()

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            alert_service=mock_alert,
        )

        # Create an opened intent
        opened = _advance_intent_to_opened(store, "EURUSD")

        # Broker returns no open positions → position was closed
        mock_matchtrader.get_open_positions.return_value = []
        # Closed positions endpoint returns the closed trade
        closed_pos = MagicMock()
        closed_pos.position_id = "pos_123"
        closed_pos.profit = 0.0
        closed_pos.close_price = 1.1050
        closed_pos.open_price = 1.1000
        closed_pos.volume = 0.01
        mock_matchtrader.get_closed_positions.return_value = [closed_pos]
        mock_matchtrader.get_balance.return_value = MagicMock(equity=50100.0)

        await _run_loop_once(sched, sched._position_monitor_loop())

        # Intent should be marked closed
        updated = store.get_intent(opened.id)
        assert updated.status == "closed"

    async def test_ignores_still_open_position(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should not close intents whose positions are still open."""
        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
        )

        opened = _advance_intent_to_opened(store, "EURUSD")

        # Broker still has the position open
        pos = MagicMock()
        pos.position_id = "pos_123"
        mock_matchtrader.get_open_positions.return_value = [pos]

        await _run_loop_once(sched, sched._position_monitor_loop())

        updated = store.get_intent(opened.id)
        assert updated.status == "opened"  # Unchanged

    async def test_skips_when_no_opened_intents(
        self,
        scheduler: Scheduler,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should not call get_open_positions when no opened intents exist."""
        await _run_loop_once(scheduler, scheduler._position_monitor_loop())

        # No opened intents → no broker API call
        mock_matchtrader.get_open_positions.assert_not_called()

    async def test_skips_position_management_when_market_closed(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Market-closed loop should not poll broker or run tactical management."""
        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
        )
        _advance_intent_to_opened(store, "EURUSD")
        sched._market_hours = MagicMock()
        sched._market_hours.should_force_close.return_value = False
        sched._market_hours.is_market_open.return_value = False
        sched._run_tactical_exit_cycle = AsyncMock()
        sched._reevaluate_open_positions = AsyncMock()

        await _run_loop_once(sched, sched._position_monitor_loop())

        mock_matchtrader.get_open_positions.assert_not_called()
        sched._run_tactical_exit_cycle.assert_not_called()
        sched._reevaluate_open_positions.assert_not_called()

    async def test_sends_sl_tp_alert_on_loss(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should send SL alert when closed position has negative PnL."""
        mock_alert = AsyncMock()
        mock_alert.sl_tp_hit = AsyncMock()
        mock_alert.trade_closed = AsyncMock()
        mock_alert.send = AsyncMock()

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            alert_service=mock_alert,
        )

        _advance_intent_to_opened(store, "EURUSD")

        mock_matchtrader.get_open_positions.return_value = []
        closed_pos = MagicMock()
        closed_pos.position_id = "pos_123"
        closed_pos.profit = -50.0  # Loss → SL
        closed_pos.close_price = 1.0950
        closed_pos.open_price = 1.1000
        closed_pos.volume = 0.01
        mock_matchtrader.get_closed_positions.return_value = [closed_pos]
        mock_matchtrader.get_balance.return_value = MagicMock(equity=49950.0)

        await _run_loop_once(sched, sched._position_monitor_loop())

        mock_alert.sl_tp_hit.assert_called_once()
        call_kwargs = mock_alert.sl_tp_hit.call_args[1]
        assert call_kwargs["hit_type"] == "SL"
        assert call_kwargs["pnl"] == -50.0

    async def test_sends_tp_alert_on_profit(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should send TP alert when closed position has positive PnL."""
        mock_alert = AsyncMock()
        mock_alert.sl_tp_hit = AsyncMock()
        mock_alert.trade_closed = AsyncMock()
        mock_alert.send = AsyncMock()

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            alert_service=mock_alert,
        )

        _advance_intent_to_opened(store, "GBPUSD")

        mock_matchtrader.get_open_positions.return_value = []
        closed_pos = MagicMock()
        closed_pos.position_id = "pos_123"
        closed_pos.profit = 75.0  # Profit → TP
        closed_pos.close_price = 1.2650
        closed_pos.open_price = 1.2600
        closed_pos.volume = 0.01
        mock_matchtrader.get_closed_positions.return_value = [closed_pos]
        mock_matchtrader.get_balance.return_value = MagicMock(equity=50075.0)

        await _run_loop_once(sched, sched._position_monitor_loop())

        mock_alert.sl_tp_hit.assert_called_once()
        call_kwargs = mock_alert.sl_tp_hit.call_args[1]
        assert call_kwargs["hit_type"] == "TP"
        assert call_kwargs["pnl"] == 75.0

    async def test_sends_manual_close_alert(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should send trade_closed alert when PnL is zero (manual close)."""
        mock_alert = AsyncMock()
        mock_alert.sl_tp_hit = AsyncMock()
        mock_alert.trade_closed = AsyncMock()
        mock_alert.send = AsyncMock()

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            alert_service=mock_alert,
        )

        _advance_intent_to_opened(store, "USDJPY")

        mock_matchtrader.get_open_positions.return_value = []
        closed_pos = MagicMock()
        closed_pos.position_id = "pos_123"
        closed_pos.profit = 0.0  # Breakeven → manual
        closed_pos.close_price = 150.00
        closed_pos.open_price = 150.00
        closed_pos.volume = 0.01
        mock_matchtrader.get_closed_positions.return_value = [closed_pos]
        mock_matchtrader.get_balance.return_value = MagicMock(equity=50000.0)

        await _run_loop_once(sched, sched._position_monitor_loop())

        mock_alert.trade_closed.assert_called_once()
        mock_alert.sl_tp_hit.assert_not_called()

    async def test_handles_api_error_gracefully(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should catch API errors without crashing the loop."""
        mock_alert = AsyncMock()
        mock_alert.send = AsyncMock()

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            alert_service=mock_alert,
        )

        _advance_intent_to_opened(store, "EURUSD")
        mock_matchtrader.get_open_positions.side_effect = RuntimeError("API down")

        await _run_loop_once(sched, sched._position_monitor_loop())

        # Loop should complete without raising
        mock_alert.send.assert_called()  # Error alert sent


# ── Daily Summary Loop Tests ───────────────────────────────────────────────


class TestDailySummaryLoop:
    """Tests for Scheduler._daily_summary_loop()."""

    async def test_sends_at_target_hour(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should send daily summary when UTC hour matches target."""
        from datetime import datetime, timezone

        mock_alert = AsyncMock()
        mock_alert.daily_summary = AsyncMock()
        mock_alert.send = AsyncMock()

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            alert_service=mock_alert,
        )

        # Mock time to be at the target hour (22 UTC)
        fake_now = datetime(2026, 2, 16, 22, 5, 0, tzinfo=timezone.utc)
        with patch.object(Scheduler, "_now_utc", return_value=fake_now):
            mock_matchtrader.get_balance.return_value = MagicMock(balance=50100.0, equity=50100.0)
            mock_matchtrader.get_open_positions.return_value = []

            await _run_loop_once(sched, sched._daily_summary_loop())

        mock_alert.daily_summary.assert_called_once()

    async def test_skips_wrong_hour(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should not send summary when UTC hour does not match target."""
        from datetime import datetime, timezone

        mock_alert = AsyncMock()
        mock_alert.daily_summary = AsyncMock()
        mock_alert.send = AsyncMock()

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            alert_service=mock_alert,
        )

        # Mock time to be at non-target hour (15 UTC, target is 22)
        fake_now = datetime(2026, 2, 16, 15, 30, 0, tzinfo=timezone.utc)
        with patch.object(Scheduler, "_now_utc", return_value=fake_now):
            await _run_loop_once(sched, sched._daily_summary_loop())

        mock_alert.daily_summary.assert_not_called()

    async def test_sends_only_once_per_day(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should not re-send summary if already sent today."""
        from datetime import datetime, timezone

        mock_alert = AsyncMock()
        mock_alert.daily_summary = AsyncMock()
        mock_alert.send = AsyncMock()

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            alert_service=mock_alert,
        )

        # Simulate already sent today
        sched._daily_summary_sent_date = "2026-02-16"

        fake_now = datetime(2026, 2, 16, 22, 5, 0, tzinfo=timezone.utc)
        with patch.object(Scheduler, "_now_utc", return_value=fake_now):
            await _run_loop_once(sched, sched._daily_summary_loop())

        mock_alert.daily_summary.assert_not_called()

    async def test_handles_summary_error(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should catch errors during summary generation without crashing."""
        from datetime import datetime, timezone

        mock_alert = AsyncMock()
        mock_alert.daily_summary = AsyncMock(side_effect=RuntimeError("DB error"))
        mock_alert.send = AsyncMock()

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            alert_service=mock_alert,
        )

        fake_now = datetime(2026, 2, 16, 22, 5, 0, tzinfo=timezone.utc)
        with patch.object(Scheduler, "_now_utc", return_value=fake_now):
            mock_matchtrader.get_balance.return_value = MagicMock(balance=50000.0, equity=50000.0)
            mock_matchtrader.get_open_positions.return_value = []

            await _run_loop_once(sched, sched._daily_summary_loop())

        # Loop should complete without raising

    async def test_summary_pnl_uses_realized_when_no_open_positions(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Daily summary PnL should match realized PnL when no positions remain open."""
        mock_alert = AsyncMock()
        mock_alert.daily_summary = AsyncMock()
        mock_alert.send = AsyncMock()

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            alert_service=mock_alert,
        )

        date_str = "2026-02-16"
        _advance_intent_to_closed(store, trade_date=date_str, symbol="EURUSD", realized_pnl=163.68)
        mock_matchtrader.get_balance.return_value = MagicMock(
            balance=50163.68,
            equity=50163.68,
            margin=0.0,
            free_margin=50163.68,
        )
        mock_matchtrader.get_open_positions.return_value = []

        await sched._send_daily_summary(date_str)

        mock_alert.daily_summary.assert_called_once()
        kwargs = mock_alert.daily_summary.call_args.kwargs
        assert kwargs["pnl"] == pytest.approx(163.68)
        assert kwargs["open_positions"] == 0
        assert kwargs["day_start_balance"] == pytest.approx(50000.0)

    async def test_daily_summary_logs_tactical_entry_calibration_snapshot(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
        tmp_path,
    ) -> None:
        from src.monitor.trade_journal import TradeJournal

        mock_alert = AsyncMock()
        mock_alert.daily_summary = AsyncMock()
        mock_alert.send = AsyncMock()

        journal = TradeJournal(tmp_path / "trade_journal.jsonl")
        journal.log_event(
            "TACTICAL_RESULT",
            {
                "timestamp": "2026-02-16T08:00:00+00:00",
                "symbol": "EURUSD",
                "resolution": "RETRY_PENDING",
                "summary_reason_code": "spread.fail.ratio_too_wide",
                "context": {"session_label": "london", "regime_label": "normal"},
                "provenance": {"data_source": "rest_fallback"},
            },
        )

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            alert_service=mock_alert,
            trade_journal=journal,
        )

        mock_matchtrader.get_balance.return_value = MagicMock(
            balance=50000.0,
            equity=50000.0,
            margin=0.0,
            free_margin=50000.0,
        )
        mock_matchtrader.get_open_positions.return_value = []

        await sched._send_daily_summary("2026-02-16")

        lines = journal._path.read_text(encoding="utf-8").strip().splitlines()
        events = [json.loads(line) for line in lines]
        snapshot = next(e for e in events if e["type"] == "TACTICAL_ENTRY_CALIBRATION_SNAPSHOT")
        assert snapshot["date"] == "2026-02-16"
        assert snapshot["groups"][0]["symbol"] == "EURUSD"
        assert snapshot["entry_funnel_mode"] == "scanner_tactical"
        assert snapshot["scanner_candidates"] == 3
        assert snapshot["intents_created"] == 2
        assert snapshot["opened_count"] == 1
        assert snapshot["llm_vetoes"] == 1
        assert snapshot["tactical_waits"] == 1
        assert snapshot["tactical_expires"] == 1
        assert snapshot["llm_veto_rate"] == 0.3333


# ── Mock LLM Blocking Tests ──────────────────────────────────────────────────────


class TestMockLLMBlocking:
    """Tests for mock LLM blocking in _process_claimed_intent()."""

    async def test_blocks_intent_when_using_mock(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """When agents.using_mock is True, intent should be cancelled without LLM call."""
        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
        )
        mock_agents.using_mock = True

        # Insert a pending intent
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.85,
            scanner_confidence="high",
        )
        store.insert_intent(intent)

        # Run one iteration of LLM worker loop
        await _run_loop_once(sched, sched._llm_worker_loop("llm-0"))

        # Verify: intent status == cancelled
        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "cancelled"

        # Verify: agents.decide was NOT called (blocked before LLM call)
        mock_agents.decide.assert_not_called()

        # Verify: cancellation reason contains "Mock LLM fallback"
        assert updated.execution_error is not None
        assert "Mock LLM fallback" in updated.execution_error

    async def test_sends_alert_when_blocking_mock(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should send alert when blocking intent due to mock LLM."""
        mock_alert = AsyncMock()
        mock_alert.send = AsyncMock(return_value=True)

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            alert_service=mock_alert,
        )
        mock_agents.using_mock = True

        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="GBPUSD",
            scanner_score=0.75,
            scanner_confidence="medium",
        )
        store.insert_intent(intent)

        await _run_loop_once(sched, sched._llm_worker_loop("llm-0"))

        # Verify alert was sent with BLOCKED or Mock in the message
        mock_alert.send.assert_called_once()
        alert_msg = mock_alert.send.call_args[0][0]
        assert "BLOCKED" in alert_msg
        assert "Mock" in alert_msg

    async def test_allows_intent_when_not_using_mock(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """When agents.using_mock is False, normal flow should proceed."""
        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
        )
        mock_agents.using_mock = False

        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.85,
            scanner_confidence="high",
        )
        store.insert_intent(intent)

        await _run_loop_once(sched, sched._llm_worker_loop("llm-0"))

        # Verify: agents.decide WAS called (normal flow)
        mock_agents.decide.assert_called_once()

        # Verify: intent goes to ready_for_exec (since default mock returns BUY)
        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "ready_for_exec"
        assert updated.suggested_side == "BUY"

    async def test_blocks_multiple_intents_with_mock(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Multiple intents should all be blocked when using_mock is True."""
        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
        )
        mock_agents.using_mock = True

        # Insert multiple pending intents
        intents = []
        for symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
            intent = TradeIntent(
                trade_date=Scheduler._today_str(),
                symbol=symbol,
                scanner_score=0.80,
                scanner_confidence="high",
            )
            store.insert_intent(intent)
            intents.append(intent)

        # Run worker loop multiple times to process all intents
        for _ in range(len(intents)):
            await _run_loop_once(sched, sched._llm_worker_loop("llm-0"))

        # Verify: all intents are cancelled
        for intent in intents:
            updated = store.get_intent(intent.id)
            assert updated is not None
            assert updated.status == "cancelled"
            assert updated.execution_error is not None
            assert "Mock LLM fallback" in updated.execution_error

        # Verify: agents.decide was never called
        mock_agents.decide.assert_not_called()


class TestBestDayIntegration:
    """Tests for BestDayTracker integration in position monitor loop."""

    async def test_updates_unrealized_pnl_on_best_day_tracker(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should call update_unrealized() with sum of open position profits."""
        # Create mock BestDayTracker
        mock_tracker = MagicMock()
        mock_tracker.should_close_winners.return_value = False

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            best_day_tracker=mock_tracker,
        )

        # Create an opened intent (required for monitor loop to query open positions)
        _advance_intent_to_opened(store, "EURUSD")

        # Set up open positions with known profits
        pos1 = MagicMock()
        pos1.position_id = "pos_1"
        pos1.profit = 100.0
        pos2 = MagicMock()
        pos2.position_id = "pos_2"
        pos2.profit = 50.0
        mock_matchtrader.get_open_positions.return_value = [pos1, pos2]

        await _run_loop_once(sched, sched._position_monitor_loop())

        # Should have called update_unrealized with total profit
        mock_tracker.update_unrealized.assert_called_once_with(150.0)

    async def test_closes_winners_when_threshold_reached(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should close profitable positions when should_close_winners() returns True."""
        # Create mock BestDayTracker that says we need to close winners
        mock_tracker = MagicMock()
        mock_tracker.should_close_winners.return_value = True
        mock_tracker.summary.return_value = "test summary"

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            best_day_tracker=mock_tracker,
        )

        # Create opened intent to trigger position monitoring
        _advance_intent_to_opened(store, "EURUSD")

        # Set up open positions with some profitable ones
        pos1 = MagicMock()
        pos1.position_id = "pos_1"
        pos1.symbol = "EURUSD"
        pos1.side = "BUY"
        pos1.volume = 0.01
        pos1.profit = 200.0  # Profitable

        pos2 = MagicMock()
        pos2.position_id = "pos_2"
        pos2.symbol = "GBPUSD"
        pos2.side = "SELL"
        pos2.volume = 0.02
        pos2.profit = 50.0  # Profitable

        mock_matchtrader.get_open_positions.return_value = [pos1, pos2]
        mock_matchtrader.close_position.return_value = MagicMock(success=True)

        await _run_loop_once(sched, sched._position_monitor_loop())

        # Should have called close_position for both profitable positions
        assert mock_matchtrader.close_position.call_count == 2
        # Verify best_day_close_positions set tracks closed position IDs
        assert "pos_1" in sched._best_day_close_positions
        assert "pos_2" in sched._best_day_close_positions
        assert sched._pending_close_outcomes["pos_1"].trigger_source == "best_day_close"
        assert sched._pending_close_outcomes["pos_2"].trigger_source == "best_day_close"
        assert sched._pending_close_outcomes["pos_1"].action_kind == "full_close"

    async def test_does_not_close_when_threshold_not_reached(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should NOT close positions when should_close_winners() returns False."""
        mock_tracker = MagicMock()
        mock_tracker.should_close_winners.return_value = False

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            best_day_tracker=mock_tracker,
        )

        _advance_intent_to_opened(store, "EURUSD")

        # Set up open positions
        pos = MagicMock()
        pos.position_id = "pos_1"
        pos.profit = 200.0
        mock_matchtrader.get_open_positions.return_value = [pos]

        await _run_loop_once(sched, sched._position_monitor_loop())

        # Should NOT have called close_position
        mock_matchtrader.close_position.assert_not_called()

    async def test_only_closes_profitable_positions(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should only close positions with profit > 0."""
        mock_tracker = MagicMock()
        mock_tracker.should_close_winners.return_value = True
        mock_tracker.summary.return_value = "test summary"

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            best_day_tracker=mock_tracker,
        )

        _advance_intent_to_opened(store, "EURUSD")

        # Set up positions: one profitable, one losing
        pos_profit = MagicMock()
        pos_profit.position_id = "pos_win"
        pos_profit.symbol = "EURUSD"
        pos_profit.side = "BUY"
        pos_profit.volume = 0.01
        pos_profit.profit = 200.0  # Profitable

        pos_loss = MagicMock()
        pos_loss.position_id = "pos_loss"
        pos_loss.symbol = "GBPUSD"
        pos_loss.side = "SELL"
        pos_loss.volume = 0.02
        pos_loss.profit = -50.0  # Losing

        mock_matchtrader.get_open_positions.return_value = [pos_profit, pos_loss]
        mock_matchtrader.close_position.return_value = MagicMock(success=True)

        await _run_loop_once(sched, sched._position_monitor_loop())

        # Should only have closed the profitable position
        assert mock_matchtrader.close_position.call_count == 1
        close_call = mock_matchtrader.close_position.call_args
        assert close_call[1]["position_id"] == "pos_win"

        # Verify only the profitable position was tracked for best day close
        assert "pos_win" in sched._best_day_close_positions
        assert "pos_loss" not in sched._best_day_close_positions
        assert sched._pending_close_outcomes["pos_win"].trigger_source == "best_day_close"

    async def test_reduce_exposure_registers_partial_close_outcomes(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Drawdown de-risk should flow through close control as partial_close."""
        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
        )

        pos = MagicMock()
        pos.position_id = "pos_reduce"
        pos.symbol = "EURUSD"
        pos.side = "BUY"
        pos.volume = 0.10
        mock_matchtrader.get_open_positions.return_value = [pos]
        mock_matchtrader.close_position.return_value = MagicMock(success=True)

        await sched._reduce_exposure_on_drawdown("DANGER", 0.041, 0.031, 4875.0)

        assert "pos_reduce" in sched._pending_close_outcomes
        outcome = sched._pending_close_outcomes["pos_reduce"]
        assert outcome.trigger_source == "reduce_exposure"
        assert outcome.action_kind == "partial_close"

    async def test_emergency_close_registers_full_close_outcomes(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Emergency close should register one full-close outcome per position."""
        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
        )

        pos1 = MagicMock()
        pos1.position_id = "pos_em_1"
        pos1.symbol = "EURUSD"
        pos1.side = "BUY"
        pos1.volume = 0.10

        pos2 = MagicMock()
        pos2.position_id = "pos_em_2"
        pos2.symbol = "GBPUSD"
        pos2.side = "SELL"
        pos2.volume = 0.20

        mock_matchtrader.get_open_positions.return_value = [pos1, pos2]
        mock_matchtrader.close_position.return_value = MagicMock(success=True)

        await sched._handle_emergency_close()

        assert sched._pending_close_outcomes["pos_em_1"].trigger_source == "emergency_close"
        assert sched._pending_close_outcomes["pos_em_2"].trigger_source == "emergency_close"
        assert sched._pending_close_outcomes["pos_em_1"].action_kind == "full_close"

    async def test_sends_best_day_protection_alert(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should send alert when closing winning positions."""
        mock_tracker = MagicMock()
        mock_tracker.should_close_winners.return_value = True
        mock_tracker.summary.return_value = "BestDay: realized=+$200.00"

        mock_alert = AsyncMock()
        mock_alert.send = AsyncMock(return_value=True)

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            best_day_tracker=mock_tracker,
            alert_service=mock_alert,
        )

        _advance_intent_to_opened(store, "EURUSD")

        pos = MagicMock()
        pos.position_id = "pos_1"
        pos.symbol = "EURUSD"
        pos.side = "BUY"
        pos.volume = 0.01
        pos.profit = 200.0
        mock_matchtrader.get_open_positions.return_value = [pos]
        mock_matchtrader.close_position.return_value = MagicMock(success=True)

        await _run_loop_once(sched, sched._position_monitor_loop())

        # Should have sent alert with Best Day Protection message
        mock_alert.send.assert_called()
        alert_msg = mock_alert.send.call_args[0][0]
        assert "Best Day" in alert_msg or "Protection" in alert_msg

    async def test_best_day_tracker_auto_created_from_config(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should auto-create BestDayTracker from config when not provided."""
        # Create scheduler without explicit best_day_tracker
        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            # best_day_tracker not passed - should be auto-created
        )

        # Verify tracker was created
        assert sched._best_day_tracker is not None
        # Verify it has correct config values
        assert sched._best_day_tracker._best_day_limit == config.compliance.best_day_limit
        assert sched._best_day_tracker._stop_ratio == config.compliance.best_day_stop


# ── Manual Close PnL Fallback Tests ────────────────────────────────────────


class TestManualClosePnlFallback:
    """Tests for manual_close PnL fallback via _last_known_profit."""

    async def test_manual_close_uses_last_known_profit_when_broker_returns_zero(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """When broker get_closed_positions returns pnl=0.0 for a manual close,
        the scheduler should fall back to the last-known polled profit."""
        mock_alert = AsyncMock()
        mock_alert.trade_closed = AsyncMock()
        mock_alert.sl_tp_hit = AsyncMock()
        mock_alert.send = AsyncMock()

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            alert_service=mock_alert,
        )

        opened = _advance_intent_to_opened(store, "GBPUSD")

        # Simulate: position was previously polled with profit=12.50
        sched._last_known_profit["pos_123"] = 12.50

        # Broker returns no open positions → position was closed
        mock_matchtrader.get_open_positions.return_value = []
        # Closed positions endpoint returns pnl=0.0 (broker hasn't updated yet)
        closed_pos = MagicMock()
        closed_pos.position_id = "pos_123"
        closed_pos.profit = 0.0
        closed_pos.close_price = 0.0
        closed_pos.open_price = 0.0
        closed_pos.volume = 0.09
        mock_matchtrader.get_closed_positions.return_value = [closed_pos]
        mock_matchtrader.get_balance.return_value = MagicMock(equity=5012.50)

        await _run_loop_once(sched, sched._position_monitor_loop())
        # With PnL=0.0 from broker, fallback to last_known=12.50 > 0 → re-inferred as tp_hit
        # So sl_tp_hit alert should fire (not trade_closed)
        mock_alert.sl_tp_hit.assert_called_once()
        call_kwargs = mock_alert.sl_tp_hit.call_args[1]
        assert call_kwargs["pnl"] == 12.50
        assert call_kwargs["hit_type"] == "TP"

        # Intent should be stored with correct realized PnL
        updated = store.get_intent(opened.id)
        assert updated.status == "closed"
        assert updated.realized_pnl == 12.50

    async def test_manual_close_no_fallback_when_broker_has_pnl(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """When broker returns real PnL, last_known_profit should NOT override it."""
        mock_alert = AsyncMock()
        mock_alert.trade_closed = AsyncMock()
        mock_alert.sl_tp_hit = AsyncMock()
        mock_alert.send = AsyncMock()

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            alert_service=mock_alert,
        )

        opened = _advance_intent_to_opened(store, "EURUSD")

        # Stale fallback value (should NOT be used)
        sched._last_known_profit["pos_123"] = 5.00

        mock_matchtrader.get_open_positions.return_value = []
        # Broker returns actual PnL this time
        closed_pos = MagicMock()
        closed_pos.position_id = "pos_123"
        closed_pos.profit = -15.30  # Real loss
        closed_pos.close_price = 1.0985
        closed_pos.open_price = 1.1000
        closed_pos.volume = 0.01
        mock_matchtrader.get_closed_positions.return_value = [closed_pos]
        mock_matchtrader.get_balance.return_value = MagicMock(equity=49984.70)

        await _run_loop_once(sched, sched._position_monitor_loop())

        # SL hit alert with real PnL, not fallback
        mock_alert.sl_tp_hit.assert_called_once()
        call_kwargs = mock_alert.sl_tp_hit.call_args[1]
        assert call_kwargs["pnl"] == -15.30

        updated = store.get_intent(opened.id)
        assert updated.realized_pnl == -15.30

    async def test_manual_close_cleans_up_last_known_profit(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """After handling a closed position, _last_known_profit entry should be removed."""
        mock_alert = AsyncMock()
        mock_alert.trade_closed = AsyncMock()
        mock_alert.sl_tp_hit = AsyncMock()
        mock_alert.send = AsyncMock()

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            alert_service=mock_alert,
        )

        _advance_intent_to_opened(store, "USDJPY")

        sched._last_known_profit["pos_123"] = 8.75
        sched._last_known_profit["pos_other"] = 3.00  # Another position, should remain

        mock_matchtrader.get_open_positions.return_value = []
        closed_pos = MagicMock()
        closed_pos.position_id = "pos_123"
        closed_pos.profit = 0.0
        closed_pos.close_price = 0.0
        closed_pos.open_price = 0.0
        closed_pos.volume = 0.01
        mock_matchtrader.get_closed_positions.return_value = [closed_pos]
        mock_matchtrader.get_balance.return_value = MagicMock(equity=50008.75)

        await _run_loop_once(sched, sched._position_monitor_loop())

        # pos_123 should be cleaned up
        assert "pos_123" not in sched._last_known_profit
        # pos_other should still be there
        assert sched._last_known_profit["pos_other"] == 3.00

    async def test_position_monitor_records_last_known_profit(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Position monitor loop should record profit for each open position."""
        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
        )

        _advance_intent_to_opened(store, "GBPUSD")

        # Broker shows position still open with profit
        pos = MagicMock()
        pos.position_id = "pos_123"
        pos.profit = 7.25
        pos.symbol = "GBPUSD.pro"
        pos.side = "BUY"
        pos.volume = 0.09
        pos.open_price = 1.34900
        pos.current_price = 1.34980
        mock_matchtrader.get_open_positions.return_value = [pos]

        await _run_loop_once(sched, sched._position_monitor_loop())

        # Should have recorded the profit
        assert sched._last_known_profit["pos_123"] == 7.25

    async def test_manual_close_fallback_not_found_in_closed_positions(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """When position not found in closed positions AND last_known_profit exists,
        should use fallback PnL."""
        mock_alert = AsyncMock()
        mock_alert.trade_closed = AsyncMock()
        mock_alert.sl_tp_hit = AsyncMock()
        mock_alert.send = AsyncMock()

        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
            alert_service=mock_alert,
        )

        opened = _advance_intent_to_opened(store, "GBPUSD")

        # Position was tracked with profit
        sched._last_known_profit["pos_123"] = 22.10

        mock_matchtrader.get_open_positions.return_value = []
        # Broker returns empty closed positions (position not found in 24h window)
        mock_matchtrader.get_closed_positions.return_value = []
        mock_matchtrader.get_balance.return_value = MagicMock(equity=5022.10)

        await _run_loop_once(sched, sched._position_monitor_loop())

        # Position not found in broker → fallback PnL=22.10 > 0 → re-inferred as tp_hit
        mock_alert.sl_tp_hit.assert_called_once()
        call_kwargs = mock_alert.sl_tp_hit.call_args[1]
        assert call_kwargs["pnl"] == 22.10
        assert call_kwargs["hit_type"] == "TP"

        updated = store.get_intent(opened.id)
        assert updated.realized_pnl == 22.10


# ── v1.3.7: Tactical Integration ────────────────────────────────────────


class TestTacticalIntegration:
    """v1.3.7: Verify tactical validation in _process_claimed_intent."""

    async def test_shadow_mode_always_marks_ready(
        self,
        scheduler: Scheduler,
        mock_agents: MagicMock,
        store: DecisionStore,
    ) -> None:
        """In shadow mode (default), tactical validation runs but intent always proceeds."""
        mock_agents.decide.return_value = AgentDecision(
            symbol="EURUSD",
            decision="BUY",
            final_state={"test": True},
            risk_report="test risk",
        )
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.85,
            scanner_confidence="high",
        )
        store.insert_intent(intent)
        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None

        await scheduler._process_claimed_intent("llm-0", claimed)

        updated = store.get_intent(intent.id)
        assert updated.status == "ready_for_exec"

    async def test_tactical_enabled_fetches_quote(
        self,
        scheduler: Scheduler,
        mock_agents: MagicMock,
        mock_matchtrader: AsyncMock,
        store: DecisionStore,
    ) -> None:
        """When tactical is enabled, get_quote should be awaited for spread data."""
        mock_agents.decide.return_value = AgentDecision(
            symbol="EURUSD",
            decision="BUY",
            final_state={"test": True},
            risk_report="test risk",
        )
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.85,
            scanner_confidence="high",
        )
        store.insert_intent(intent)
        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None

        await scheduler._process_claimed_intent("llm-0", claimed)

        mock_matchtrader.get_quote.assert_awaited_once_with("EURUSD")

    async def test_tactical_disabled_skips_validation(
        self,
        config: AppConfig,
        store: DecisionStore,
        mock_scanner: MagicMock,
        mock_agents: MagicMock,
        mock_engine: AsyncMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """When tactical.enabled=False, no tactical validation is run."""
        config.tactical.enabled = False
        sched = Scheduler(
            config=config,
            store=store,
            scanner=mock_scanner,
            agents=mock_agents,
            engine=mock_engine,
            matchtrader=mock_matchtrader,
        )
        mock_agents.decide.return_value = AgentDecision(
            symbol="EURUSD",
            decision="SELL",
            final_state={"test": True},
            risk_report="test risk",
        )
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.85,
            scanner_confidence="high",
        )
        store.insert_intent(intent)
        claimed = store.claim_next_pending("llm-0")

        await sched._process_claimed_intent("llm-0", claimed)

        updated = store.get_intent(intent.id)
        assert updated.status == "ready_for_exec"


# ── Task 2: Intraday Rescan Skip for Daily Model (#10, P2) ──────────────


async def test_intraday_rescan_skipped_for_daily_model(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """Intraday rescan should be skipped when scanner_timeframe is '1d'."""
    config.scheduler.scanner_timeframe = "1d"
    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    signals = [_make_mock_signal("EURUSD"), _make_mock_signal("GBPUSD")]
    mock_scanner.run_pipeline.reset_mock()

    await sched._run_intraday_scan(signals, "2026-03-09")

    # Scanner should NOT have been called for intraday rescan
    mock_scanner.run_pipeline.assert_not_called()


async def test_intraday_rescan_runs_for_non_daily_model(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """Intraday rescan should proceed when scanner_timeframe is not '1d'."""
    config.scheduler.scanner_timeframe = "4h"
    config.scheduler.entry_timeframe = "1h"
    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    signals = [_make_mock_signal("EURUSD")]
    mock_scanner.run_pipeline.reset_mock()
    mock_scanner.run_pipeline.return_value = []

    await sched._run_intraday_scan(signals, "2026-03-09")

    # Scanner SHOULD have been called for non-daily model
    mock_scanner.run_pipeline.assert_called_once()


# ── Task 4: HOLD→BUY Stale Intent Race Fix (#9, P2) ────────────────────────


async def test_hold_cancels_stale_ready_intents_for_same_symbol(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """When LLM decides HOLD, cancel all ready_for_exec intents for the same symbol."""
    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    # Create stale intent A — pending → claimed → ready_for_exec (BUY)
    intent_a = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="EURUSD",
        scanner_score=0.85,
        scanner_confidence="high",
    )
    store.insert_intent(intent_a)
    claimed_a = store.claim_next_pending("llm-0")
    assert claimed_a is not None
    store.update_intent_decision(
        claimed_a.id, "BUY", sl_pips=30.0, tp_pips=50.0, risk_report="test", state_json="{}"
    )
    store.mark_ready_for_exec(claimed_a.id)

    # Verify intent A is ready_for_exec
    assert store.get_intent(intent_a.id).status == "ready_for_exec"

    # Create newer intent B for same symbol — pending → claimed
    intent_b = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="EURUSD",
        scanner_score=0.80,
        scanner_confidence="medium",
    )
    store.insert_intent(intent_b)
    claimed_b = store.claim_next_pending("llm-1")
    assert claimed_b is not None

    # Simulate LLM returning HOLD for intent B
    mock_agents.decide.return_value = AgentDecision(
        symbol="EURUSD",
        decision="HOLD",
        final_state={},
        risk_report="no action",
    )

    await sched._process_claimed_intent("llm-1", claimed_b)

    # Verify: stale intent A should also be cancelled (superseded by HOLD)
    stale = store.get_intent(intent_a.id)
    assert stale.status == "cancelled", (
        f"Stale BUY intent should be cancelled after HOLD, got: {stale.status}"
    )

    # Verify: intent B is also cancelled (direct HOLD handling)
    newer = store.get_intent(intent_b.id)
    assert newer.status == "cancelled"


async def test_hold_does_not_cancel_intents_for_different_symbol(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """When LLM decides HOLD for EURUSD, don't cancel GBPUSD ready intents."""
    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    # Create intent for GBPUSD — ready_for_exec
    intent_gbp = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="GBPUSD",
        scanner_score=0.90,
        scanner_confidence="high",
    )
    store.insert_intent(intent_gbp)
    claimed_gbp = store.claim_next_pending("llm-0")
    assert claimed_gbp is not None
    store.update_intent_decision(
        claimed_gbp.id, "BUY", sl_pips=30.0, tp_pips=50.0, risk_report="test", state_json="{}"
    )
    store.mark_ready_for_exec(claimed_gbp.id)

    # Create intent for EURUSD — claimed (will get HOLD)
    intent_eur = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="EURUSD",
        scanner_score=0.75,
        scanner_confidence="medium",
    )
    store.insert_intent(intent_eur)
    claimed_eur = store.claim_next_pending("llm-1")
    assert claimed_eur is not None

    # LLM returns HOLD for EURUSD
    mock_agents.decide.return_value = AgentDecision(
        symbol="EURUSD",
        decision="HOLD",
        final_state={},
        risk_report="no action",
    )

    await sched._process_claimed_intent("llm-1", claimed_eur)

    # GBPUSD intent should still be ready_for_exec
    gbp = store.get_intent(intent_gbp.id)
    assert gbp.status == "ready_for_exec", (
        f"GBPUSD intent should NOT be cancelled by EURUSD HOLD, got: {gbp.status}"
    )


# ── v1.3.9: execution_meta fallback tests ────────────────────────────────


async def test_handle_position_closed_uses_execution_meta_fallback(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """When broker API returns 0 for volume/close_price, fall back to execution_meta."""
    import json

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )
    sched._alert_service = AsyncMock()

    # Create and advance an intent to 'opened' state
    intent = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="EURUSD",
        scanner_score=0.85,
        scanner_confidence="high",
    )
    store.insert_intent(intent)
    claimed = store.claim_next_pending("llm-0")
    assert claimed is not None
    store.update_intent_decision(
        claimed.id, "BUY", sl_pips=50.0, tp_pips=100.0, risk_report="test", state_json="{}"
    )
    store.mark_ready_for_exec(claimed.id)
    store.mark_executing(claimed.id)
    store.mark_opened(claimed.id, position_id="pos-1")

    # Save execution_meta to the decision record
    meta = {
        "fill_price": 1.085,
        "volume": 0.05,
        "side": "BUY",
        "sl_price": 1.080,
        "tp_price": 1.095,
        "sl_pips": 50,
        "tp_pips": 100,
    }
    store.update_execution_meta(claimed.id, json.dumps(meta))

    # Broker API returns NO closed positions (simulates 0 values)
    mock_matchtrader.get_closed_positions = AsyncMock(return_value=[])
    mock_matchtrader.get_balance.return_value = MagicMock(
        balance=50000.0,
        equity=50000.0,
        margin=0.0,
        free_margin=50000.0,
    )

    # Reload intent from store (now in 'opened' state with position_id)
    opened_intent = store.get_intent(claimed.id)

    await sched._handle_position_closed(opened_intent)

    # Verify: store.mark_closed was called - check the intent is now 'closed'
    closed_intent = store.get_intent(claimed.id)
    assert closed_intent.status == "closed"

    # Verify: alert was sent with execution_meta volume (0.05, not 0.0)
    sched._alert_service.trade_closed.assert_called_once()
    call_kwargs = sched._alert_service.trade_closed.call_args[1]
    assert call_kwargs["volume"] == 0.05, (
        f"Expected volume=0.05 from execution_meta, got {call_kwargs['volume']}"
    )


async def test_handle_position_closed_exit_reason_from_close_price(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """When close_price matches TP price within tolerance, exit_reason should be tp_hit."""
    import json

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )
    sched._alert_service = AsyncMock()

    # Create and advance intent to 'opened' state
    intent = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="EURUSD",
        scanner_score=0.85,
        scanner_confidence="high",
    )
    store.insert_intent(intent)
    claimed = store.claim_next_pending("llm-0")
    assert claimed is not None
    store.update_intent_decision(
        claimed.id, "BUY", sl_pips=50.0, tp_pips=100.0, risk_report="test", state_json="{}"
    )
    store.mark_ready_for_exec(claimed.id)
    store.mark_executing(claimed.id)
    store.mark_opened(claimed.id, position_id="pos-2")

    meta = {
        "fill_price": 1.085,
        "volume": 0.05,
        "side": "BUY",
        "sl_price": 1.080,
        "tp_price": 1.095,
    }
    store.update_execution_meta(claimed.id, json.dumps(meta))

    # Broker returns closed position with profit (close near TP)
    closed_pos = MagicMock(
        position_id="pos-2",
        profit=50.0,
        close_price=1.0951,  # Within 3-pip tolerance of TP=1.095
        open_price=1.085,
        volume=0.05,
    )
    mock_matchtrader.get_closed_positions = AsyncMock(return_value=[closed_pos])
    mock_matchtrader.get_balance.return_value = MagicMock(
        balance=50050.0,
        equity=50050.0,
        margin=0.0,
        free_margin=50050.0,
    )

    opened_intent = store.get_intent(claimed.id)
    await sched._handle_position_closed(opened_intent)

    # Verify: exit_reason should be tp_hit (confirmed by close_price proximity)
    closed_intent = store.get_intent(claimed.id)
    assert closed_intent.status == "closed"
    assert closed_intent.exit_reason == "tp_hit"

    # Verify: sl_tp_hit alert was sent (not trade_closed)
    sched._alert_service.sl_tp_hit.assert_called_once()
    call_kwargs = sched._alert_service.sl_tp_hit.call_args[1]
    assert call_kwargs["hit_type"] == "TP"
    assert call_kwargs["trigger_price"] == 1.0951


async def test_handle_position_closed_logs_unified_trade_closed_payload(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
    tmp_path,
):
    """TRADE_CLOSED journal event should include canonical close-control fields."""
    import json

    from src.monitor.trade_journal import TradeJournal

    journal = TradeJournal(tmp_path / "trade_journal.jsonl")
    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
        trade_journal=journal,
    )
    sched._alert_service = AsyncMock()

    intent = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="EURUSD",
        scanner_score=0.85,
        scanner_confidence="high",
    )
    store.insert_intent(intent)
    claimed = store.claim_next_pending("llm-0")
    assert claimed is not None
    store.update_intent_decision(
        claimed.id, "BUY", sl_pips=50.0, tp_pips=100.0, risk_report="test", state_json="{}"
    )
    store.mark_ready_for_exec(claimed.id)
    store.mark_executing(claimed.id)
    store.mark_opened(claimed.id, position_id="pos-trade-closed")
    store.update_execution_meta(
        claimed.id,
        json.dumps(
            {
                "fill_price": 1.085,
                "volume": 0.05,
                "side": "BUY",
                "sl_price": 1.080,
                "tp_price": 1.095,
            }
        ),
    )

    closed_pos = MagicMock(
        position_id="pos-trade-closed",
        profit=35.0,
        close_price=1.0950,
        open_price=1.085,
        volume=0.05,
        close_reason="",
    )
    mock_matchtrader.get_closed_positions = AsyncMock(return_value=[closed_pos])
    mock_matchtrader.get_balance.return_value = MagicMock(
        balance=50035.0,
        equity=50035.0,
        margin=0.0,
        free_margin=50035.0,
    )

    opened_intent = store.get_intent(claimed.id)
    await sched._handle_position_closed(opened_intent)

    entries = [json.loads(line) for line in journal._path.read_text(encoding="utf-8").splitlines()]
    trade_closed = next(entry for entry in entries if entry["type"] == "TRADE_CLOSED")
    assert trade_closed["trigger_source"] == "manual_or_broker"
    assert trade_closed["action_kind"] == "external_detected_close"
    assert trade_closed["final_close_reason"] == "tp_hit"
    assert trade_closed["resolution_path"] == "broker_api"


def test_build_reflection_payload_includes_trade_outcome_context(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """v1.4.0: reflection payload should carry outcome and context fields."""
    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )
    sched._latest_market_event_context = "Volatility trigger: EURUSD moved +0.42% in 30 minutes."

    intent = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="EURUSD",
        scanner_score=0.85,
        scanner_confidence="high",
    )
    intent.suggested_side = "BUY"
    payload = sched._build_reflection_payload(
        intent=intent,
        pnl=-12.5,
        exit_reason="sl_hit",
        position_id="pos-3",
        resolution_path="broker_api",
        hold_duration_seconds=300,
        decision=MagicMock(risk_report="Avoid fading CPI spike", model_id="model-a"),
    )

    assert payload["symbol"] == "EURUSD"
    assert payload["realized_pnl"] == -12.5
    assert payload["close_reason"] == "sl_hit"
    assert payload["position_id"] == "pos-3"
    assert payload["market_event_context"] == sched._latest_market_event_context
    assert payload["risk_report"] == "Avoid fading CPI spike"


async def test_handle_position_closed_reflect_failure_does_not_break_close_flow(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """Reflection errors should be best-effort and not block intent closure."""
    import json

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )
    sched._alert_service = AsyncMock()
    mock_agents.reflect.side_effect = RuntimeError("memory backend unavailable")

    intent = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="EURUSD",
        scanner_score=0.85,
        scanner_confidence="high",
    )
    store.insert_intent(intent)
    claimed = store.claim_next_pending("llm-0")
    assert claimed is not None
    store.update_intent_decision(
        claimed.id, "BUY", sl_pips=50.0, tp_pips=100.0, risk_report="test", state_json="{}"
    )
    store.mark_ready_for_exec(claimed.id)
    store.mark_executing(claimed.id)
    store.mark_opened(claimed.id, position_id="pos-4")
    store.update_execution_meta(
        claimed.id,
        json.dumps(
            {
                "fill_price": 1.085,
                "volume": 0.05,
                "side": "BUY",
                "sl_price": 1.080,
                "tp_price": 1.095,
            }
        ),
    )

    closed_pos = MagicMock(
        position_id="pos-4",
        profit=-25.0,
        close_price=1.0800,
        open_price=1.085,
        volume=0.05,
    )
    mock_matchtrader.get_closed_positions = AsyncMock(return_value=[closed_pos])
    mock_matchtrader.get_balance.return_value = MagicMock(
        balance=49975.0,
        equity=49975.0,
        margin=0.0,
        free_margin=49975.0,
    )

    opened_intent = store.get_intent(claimed.id)
    await sched._handle_position_closed(opened_intent)

    closed_intent = store.get_intent(claimed.id)
    assert closed_intent.status == "closed"
    mock_agents.reflect.assert_called_once()
    reflect_payload = mock_agents.reflect.call_args.args[0]
    assert reflect_payload["symbol"] == "EURUSD"
    assert reflect_payload["realized_pnl"] == -25.0
    assert reflect_payload["close_reason"] == "sl_hit"
    assert reflect_payload["risk_report"] == "test"


async def test_handle_position_closed_passes_identity_to_memory_journal(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """Closed trade journaling should pass intent_id and position_id to MemoryJournal."""
    import json

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )
    sched._alert_service = AsyncMock()
    sched._memory_journal = MagicMock()

    intent = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="EURUSD",
        scanner_score=0.85,
        scanner_confidence="high",
    )
    store.insert_intent(intent)
    claimed = store.claim_next_pending("llm-0")
    assert claimed is not None
    store.update_intent_decision(
        claimed.id, "BUY", sl_pips=50.0, tp_pips=100.0, risk_report="test", state_json="{}"
    )
    store.mark_ready_for_exec(claimed.id)
    store.mark_executing(claimed.id)
    store.mark_opened(claimed.id, position_id="pos-journal")
    store.update_execution_meta(
        claimed.id,
        json.dumps(
            {
                "fill_price": 1.085,
                "volume": 0.05,
                "side": "BUY",
                "sl_price": 1.080,
                "tp_price": 1.095,
            }
        ),
    )

    closed_pos = MagicMock(
        position_id="pos-journal",
        profit=18.0,
        close_price=1.0950,
        open_price=1.085,
        volume=0.05,
    )
    mock_matchtrader.get_closed_positions = AsyncMock(return_value=[closed_pos])
    mock_matchtrader.get_balance.return_value = MagicMock(
        balance=50018.0,
        equity=50018.0,
        margin=0.0,
        free_margin=50018.0,
    )

    opened_intent = store.get_intent(claimed.id)
    await sched._handle_position_closed(opened_intent)

    sched._memory_journal.append_trade_result.assert_called_once_with(
        intent_id=claimed.id,
        position_id="pos-journal",
        symbol="EURUSD",
        pnl=18.0,
        reason="tp_hit",
    )


# ── v1.3.9: Tactical Gate Enforcement Tests ──────────────────────────────


async def test_tactical_wait_retries_then_cancels_when_expire_action_cancel(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """WAIT should enter tactical_pending, retry, then cancel on retry expiry."""
    from src.decision.tactical_validator import TacticalResult

    config.tactical.enabled = True
    config.tactical.shadow_mode = False
    config.tactical.retry.max_retries = 1
    config.tactical.retry.interval_seconds = 0
    config.tactical.retry.jitter_seconds = 0
    config.tactical.retry.expire_action = "cancel"

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    intent = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="EURUSD",
        scanner_score=0.85,
        scanner_confidence="high",
    )
    store.insert_intent(intent)
    claimed = store.claim_next_pending("llm-0")
    assert claimed is not None

    tactical_wait = TacticalResult(
        action="WAIT",
        detail="Spread too wide (0.0005 > 0.0003)",
    )
    with (
        patch.object(sched, "_run_tactical_validation", new_callable=AsyncMock) as mock_tac,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_tac.side_effect = [tactical_wait, tactical_wait]
        await sched._process_claimed_intent("llm-0", claimed)

    final = store.get_intent(claimed.id)
    assert final is not None
    assert final.status == "timed_out"
    assert "Tactical gate WAIT" in (final.execution_error or "")
    assert mock_tac.await_count == 2


async def test_tactical_wait_retries_then_times_out_when_expire_action_cancel(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """Retry exhaustion should classify tactical WAIT expiry as timed_out."""
    from src.decision.tactical_validator import TacticalResult

    config.tactical.enabled = True
    config.tactical.shadow_mode = False
    config.tactical.retry.max_retries = 1
    config.tactical.retry.interval_seconds = 0
    config.tactical.retry.jitter_seconds = 0
    config.tactical.retry.expire_action = "cancel"

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    intent = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="EURUSD",
        scanner_score=0.85,
        scanner_confidence="high",
    )
    store.insert_intent(intent)
    claimed = store.claim_next_pending("llm-0")
    assert claimed is not None

    tactical_wait = TacticalResult(
        action="WAIT",
        detail="Spread too wide (0.0005 > 0.0003)",
    )
    with (
        patch.object(sched, "_run_tactical_validation", new_callable=AsyncMock) as mock_tac,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_tac.side_effect = [tactical_wait, tactical_wait]
        await sched._process_claimed_intent("llm-0", claimed)

    final = store.get_intent(claimed.id)
    assert final is not None
    assert final.status == "timed_out"
    assert "Tactical gate WAIT" in (final.execution_error or "")
    assert mock_tac.await_count == 2


async def test_tactical_gate_passes_in_shadow_mode(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """Shadow mode (default): tactical WAIT should NOT block — intent proceeds to ready_for_exec.

    This verifies the shadow mode behavior was preserved when adding enforcement.
    """
    from src.decision.tactical_validator import TacticalResult

    # Enable tactical gate with shadow_mode=true (default, shadow only)
    config.tactical.enabled = True
    config.tactical.shadow_mode = True

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    intent = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="EURUSD",
        scanner_score=0.85,
        scanner_confidence="high",
    )
    store.insert_intent(intent)
    claimed = store.claim_next_pending("llm-0")
    assert claimed is not None

    tactical_wait = TacticalResult(
        action="WAIT",
        detail="Spread too wide (shadow)",
    )
    with patch.object(sched, "_run_tactical_validation", new_callable=AsyncMock) as mock_tac:
        mock_tac.return_value = tactical_wait
        await sched._process_claimed_intent("llm-0", claimed)

    # Verify: intent should be ready_for_exec (NOT cancelled)
    final = store.get_intent(claimed.id)
    assert final.status == "ready_for_exec"


async def test_tactical_wait_retries_then_ready_for_exec_when_gate_passes(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """WAIT should pause in tactical_pending and resume to ready_for_exec on PASS."""
    from src.decision.tactical_validator import TacticalResult

    config.tactical.enabled = True
    config.tactical.shadow_mode = False
    config.tactical.retry.max_retries = 2
    config.tactical.retry.interval_seconds = 0
    config.tactical.retry.jitter_seconds = 0

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    intent = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="EURUSD",
        scanner_score=0.85,
        scanner_confidence="high",
    )
    store.insert_intent(intent)
    claimed = store.claim_next_pending("llm-0")
    assert claimed is not None

    tactical_wait = TacticalResult(action="WAIT", detail="Need another 5min bar")
    tactical_pass = TacticalResult(action="PASS", detail="Momentum aligned")
    with (
        patch.object(sched, "_run_tactical_validation", new_callable=AsyncMock) as mock_tac,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_tac.side_effect = [tactical_wait, tactical_pass]
        await sched._process_claimed_intent("llm-0", claimed)

    final = store.get_intent(claimed.id)
    assert final.status == "ready_for_exec"
    assert mock_tac.await_count == 2


async def test_tactical_retry_promotes_intent_after_first_5m_bar_arrives(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """Startup 5m warmup WAIT should resume to ready_for_exec on the next retry."""
    from src.decision.tactical_validator import TacticalResult

    config.tactical.enabled = True
    config.tactical.shadow_mode = False
    config.tactical.retry.max_retries = 2
    config.tactical.retry.interval_seconds = 0
    config.tactical.retry.jitter_seconds = 0

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    intent = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="EURUSD",
        scanner_score=0.85,
        scanner_confidence="high",
    )
    store.insert_intent(intent)
    claimed = store.claim_next_pending("llm-0")
    assert claimed is not None

    tactical_wait = TacticalResult(
        action="WAIT",
        resolution="RETRY_PENDING",
        detail="Awaiting first websocket 5m closed bar after startup",
        summary_reason_code="market_data.startup_5m_bar_pending",
    )
    tactical_pass = TacticalResult(
        action="PASS",
        resolution="EXECUTE_NOW",
        detail="Momentum aligned",
        summary_reason_code="tactical.pass.all_gates_aligned",
    )
    with (
        patch.object(sched, "_run_tactical_validation", new_callable=AsyncMock) as mock_tac,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_tac.side_effect = [tactical_wait, tactical_pass]
        await sched._process_claimed_intent("llm-0", claimed)

    final = store.get_intent(claimed.id)
    assert final.status == "ready_for_exec"
    assert mock_tac.await_count == 2


async def test_tactical_wait_extends_expiry_to_cover_retry_budget(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """Tactical WAIT should extend expires_at beyond the base claim TTL."""
    from src.decision.tactical_validator import TacticalResult

    config.tactical.enabled = True
    config.tactical.shadow_mode = False
    config.tactical.retry.max_retries = 12
    config.tactical.retry.interval_seconds = 300
    config.tactical.retry.jitter_seconds = 0
    config.tactical.retry.expire_action = "cancel"

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    intent = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="EURUSD",
        scanner_score=0.85,
        scanner_confidence="high",
        claim_ttl_minutes=30,
    )
    store.insert_intent(intent)
    claimed = store.claim_next_pending("llm-0")
    assert claimed is not None

    tactical_wait = TacticalResult(action="WAIT", detail="Need more confirmation")
    with (
        patch.object(sched, "_run_tactical_validation", new_callable=AsyncMock) as mock_tac,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_tac.side_effect = [tactical_wait] * 13
        await sched._process_claimed_intent("llm-0", claimed)

    final = store.get_intent(claimed.id)
    assert final is not None
    assert final.claim_ts is not None
    assert final.expires_at is not None
    assert (final.expires_at - final.claim_ts).total_seconds() >= 3600


async def test_tactical_wait_alerts_are_throttled_for_identical_retries(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """Identical tactical WAIT alerts should not spam Telegram on every retry."""
    from src.decision.tactical_validator import TacticalResult

    mock_alert = AsyncMock()
    mock_alert.send = AsyncMock(return_value=True)
    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
        alert_service=mock_alert,
    )
    intent = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="EURUSD",
        scanner_score=0.85,
        scanner_confidence="high",
        expires_at=datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
    )
    result = TacticalResult(action="WAIT", detail="Need another 5min bar")

    with patch.object(
        sched,
        "_now_utc",
        side_effect=[
            datetime(2026, 3, 17, 10, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 17, 10, 1, tzinfo=timezone.utc),
        ],
    ):
        await sched._log_tactical_result(intent, "BUY", result, retry_count=0)
        await sched._log_tactical_result(intent, "BUY", result, retry_count=1)

    mock_alert.send.assert_awaited_once()


async def test_tactical_result_event_includes_feed_and_signal_diagnostics(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
    tmp_path,
):
    """TACTICAL_RESULT should carry feed and scanner diagnostics for incident triage."""
    from src.decision.tactical_validator import GateResult, TacticalResult
    from src.monitor.trade_journal import TradeJournal

    journal = TradeJournal(tmp_path / "trade_journal.jsonl")
    mock_scanner.get_last_rejection_reason_code.return_value = "scanner.bundle.target_date_missing"
    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
        trade_journal=journal,
    )
    sched._market_data_ready = True
    sched._market_data_hub = MagicMock()
    sched._market_data_hub.feed_status.return_value = {
        "websocket": {
            "state": "degraded",
            "last_error": "keepalive ping timeout",
        }
    }
    intent = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="EURUSD",
        scanner_score=0.85,
        scanner_confidence="high",
        expires_at=datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
    )
    result = TacticalResult(
        action="WAIT",
        detail="Need another 5min bar",
        summary_reason_code="spread.fail.ratio_too_wide",
        hard_gates=[
            GateResult(
                gate_name="spread",
                passed=False,
                status="FAIL",
                reason_code="spread.fail.ratio_too_wide",
                detail="spread_ratio=3.33, limit=2.0x",
            ),
            GateResult(
                gate_name="atr_regime",
                passed=False,
                status="FAIL",
                reason_code="atr.fail.insufficient_1h_data",
                detail="Insufficient 1H data for ATR calculation",
            ),
            GateResult(
                gate_name="data_freshness",
                passed=True,
                status="PASS",
                reason_code="freshness.pass.quote_fresh",
                detail="quote_age=20s, max=600s",
            ),
        ],
        provenance={
            "data_source": "rest_fallback",
            "quote_source": "rest_fallback",
            "bars_5m_source": "rest_fallback",
            "bars_1h_source": "warmup_cache",
        },
    )

    await sched._log_tactical_result(intent, "BUY", result, retry_count=1)

    lines = journal._path.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line) for line in lines]
    tactical_event = next(e for e in events if e["type"] == "TACTICAL_RESULT")
    assert tactical_event["scanner_rejection_reason"] == "scanner.bundle.target_date_missing"
    assert tactical_event["feed_state"] == "degraded"
    assert tactical_event["ws_last_error"] == "keepalive ping timeout"
    assert tactical_event["quote_source"] == "rest_fallback"
    assert tactical_event["bars_5m_source"] == "rest_fallback"
    assert tactical_event["bars_1h_source"] == "warmup_cache"
    assert tactical_event["tactical_deadline_at"] == "2026-03-17T12:00:00+00:00"
    assert tactical_event["failed_hard_gate_names"] == ["spread", "atr_regime"]
    assert tactical_event["failed_hard_gate_reason_codes"] == [
        "spread.fail.ratio_too_wide",
        "atr.fail.insufficient_1h_data",
    ]
    assert tactical_event["hard_gate_reason_codes"] == [
        "spread.fail.ratio_too_wide",
        "atr.fail.insufficient_1h_data",
        "freshness.pass.quote_fresh",
    ]


async def test_tactical_wait_degrades_to_ready_for_exec_on_retry_expiry(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """Degrade mode should release intent to execution after retry budget is exhausted."""
    from src.decision.tactical_validator import TacticalResult

    config.tactical.enabled = True
    config.tactical.shadow_mode = False
    config.tactical.retry.max_retries = 1
    config.tactical.retry.interval_seconds = 0
    config.tactical.retry.jitter_seconds = 0
    config.tactical.retry.expire_action = "degrade"

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    intent = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="EURUSD",
        scanner_score=0.85,
        scanner_confidence="high",
    )
    store.insert_intent(intent)
    claimed = store.claim_next_pending("llm-0")
    assert claimed is not None

    tactical_wait = TacticalResult(action="WAIT", detail="Still waiting")
    with (
        patch.object(sched, "_run_tactical_validation", new_callable=AsyncMock) as mock_tac,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_tac.side_effect = [tactical_wait, tactical_wait]
        await sched._process_claimed_intent("llm-0", claimed)

    final = store.get_intent(claimed.id)
    assert final.status == "ready_for_exec"
    assert mock_tac.await_count == 2


async def test_tactical_wait_execute_degraded_releases_directly_to_ready_for_exec(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """Resolution=EXECUTE_DEGRADED should bypass retry flow and release immediately."""
    from src.decision.tactical_validator import TacticalResult

    config.tactical.enabled = True
    config.tactical.shadow_mode = False
    config.tactical.retry.max_retries = 2
    config.tactical.retry.interval_seconds = 0
    config.tactical.retry.jitter_seconds = 0

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    intent = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="EURUSD",
        scanner_score=0.85,
        scanner_confidence="high",
    )
    store.insert_intent(intent)
    claimed = store.claim_next_pending("llm-0")
    assert claimed is not None

    tactical_degrade = TacticalResult(
        action="WAIT",
        resolution="EXECUTE_DEGRADED",
        summary_reason_code="soft.wait.score_below_threshold",
        detail="Retry budget already exhausted upstream",
    )
    with (
        patch.object(sched, "_run_tactical_validation", new_callable=AsyncMock) as mock_tac,
        patch.object(sched, "_retry_tactical_pending", new_callable=AsyncMock) as mock_retry,
    ):
        mock_tac.return_value = tactical_degrade
        await sched._process_claimed_intent("llm-0", claimed)

    final = store.get_intent(claimed.id)
    assert final is not None
    assert final.status == "ready_for_exec"
    mock_retry.assert_not_awaited()


async def test_tactical_wait_with_timeout_resolution_marks_timed_out_immediately(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """Resolution=EXPIRE_TIMEOUT should time out immediately without retry."""
    from src.decision.tactical_validator import TacticalResult

    config.tactical.enabled = True
    config.tactical.shadow_mode = False
    config.tactical.retry.max_retries = 2
    config.tactical.retry.interval_seconds = 0
    config.tactical.retry.jitter_seconds = 0

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    intent = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="EURUSD",
        scanner_score=0.85,
        scanner_confidence="high",
    )
    store.insert_intent(intent)
    claimed = store.claim_next_pending("llm-0")
    assert claimed is not None

    tactical_timeout = TacticalResult(
        action="WAIT",
        resolution="EXPIRE_TIMEOUT",
        summary_reason_code="freshness.fail.timestamp_missing",
        detail="Tactical window already expired",
    )
    with (
        patch.object(sched, "_run_tactical_validation", new_callable=AsyncMock) as mock_tac,
        patch.object(sched, "_retry_tactical_pending", new_callable=AsyncMock) as mock_retry,
    ):
        mock_tac.return_value = tactical_timeout
        await sched._process_claimed_intent("llm-0", claimed)

    final = store.get_intent(claimed.id)
    assert final is not None
    assert final.status == "timed_out"
    assert "Tactical gate WAIT" in (final.execution_error or "")
    mock_retry.assert_not_awaited()


async def test_process_claimed_intent_uses_decision_cache_for_repeated_signal(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """Repeated strategic inputs should reuse the cached LLM decision."""
    config.tactical.enabled = False
    config.scheduler.max_same_direction_per_day = 0

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    intent1 = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="EURUSD",
        scanner_score=0.85,
        scanner_confidence="high",
    )
    intent2 = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="EURUSD",
        scanner_score=0.85,
        scanner_confidence="high",
    )
    store.insert_intent(intent1)
    store.insert_intent(intent2)

    claimed1 = store.claim_next_pending("llm-0")
    assert claimed1 is not None
    await sched._process_claimed_intent("llm-0", claimed1)

    claimed2 = store.claim_next_pending("llm-0")
    assert claimed2 is not None
    await sched._process_claimed_intent("llm-0", claimed2)

    assert mock_agents.decide.call_count == 1
    assert store.get_intent(claimed2.id).status == "ready_for_exec"


async def test_process_claimed_intent_injects_historical_pnl_context(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """LLM input should include historical PnL context for the current symbol."""
    config.tactical.enabled = False

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )
    sched._optimization_state = OptimizationState(feedback_pnl={"EURUSD": 12.5, "GBPUSD": -5.0})

    intent = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="EURUSD",
        scanner_score=0.85,
        scanner_confidence="high",
    )
    store.insert_intent(intent)
    claimed = store.claim_next_pending("llm-0")
    assert claimed is not None

    await sched._process_claimed_intent("llm-0", claimed)

    qlib_data = mock_agents.decide.call_args.kwargs["qlib_data"]
    assert "historical_pnl_context" in qlib_data
    assert "EURUSD" in qlib_data["historical_pnl_context"]
    assert "12.50" in qlib_data["historical_pnl_context"]


async def test_volatility_trigger_runs_immediate_equity_check(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """Volatility-triggered rescans should also force an immediate equity check."""
    config.scheduler.volatility_trigger_enabled = True
    config.scheduler.volatility_poll_interval_seconds = 0

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    sched._run_equity_check_once = AsyncMock()
    sched._volatility_monitor.check_triggers = MagicMock(return_value=(True, "EURUSD", 0.42))
    mock_matchtrader.get_quote.return_value = MagicMock(bid=1.0848, ask=1.0850)

    await _run_loop_once(sched, sched._volatility_monitor_loop())

    sched._run_equity_check_once.assert_awaited_once()


async def test_fetch_tactical_data_no_fallback_on_eodhd_failure(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """v1.3.9-fix: When EODHD fetch fails, latest_bar_time must remain None.

    Previously, a `datetime.now()` fallback was set, masking EODHD failures
    and causing the data_freshness hard gate to always pass.
    """

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    # Inject a mock EODHD provider that raises
    mock_eodhd = AsyncMock()
    mock_eodhd.fetch_bars = AsyncMock(side_effect=Exception("EODHD API timeout"))
    sched._eodhd = mock_eodhd

    data = await sched._fetch_tactical_data("EURUSD")

    # Verify: latest_bar_time must be None — no EODHD data AND mock_matchtrader
    # fixture returns a dict without timestampMs, so quote timestamp is also 0.
    assert data.latest_bar_time is None, (
        f"Expected latest_bar_time=None after EODHD failure, got {data.latest_bar_time}"
    )


async def test_fetch_tactical_data_prefers_market_data_hub_cache(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """v1.4.0: tactical reads should prefer websocket-derived hub data when healthy."""
    import pandas as pd

    now = datetime.now(timezone.utc)
    fresh_quote_ts_ms = int((now - timedelta(seconds=15)).timestamp() * 1000)

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    bars_5min = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp(now - timedelta(minutes=5)),
                "open": 1.10,
                "high": 1.11,
                "low": 1.09,
                "close": 1.105,
                "volume": 0,
            }
        ]
    )
    bars_1h = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp(now - timedelta(minutes=55)),
                "open": 1.09,
                "high": 1.12,
                "low": 1.08,
                "close": 1.105,
                "volume": 0,
            }
        ]
    )
    sched._market_data_ready = True
    sched._market_data_hub = MagicMock()
    sched._market_data_hub.get_quote = AsyncMock(
        return_value=MagicMock(
            source="websocket_cache",
            quote={"bid": 1.0848, "ask": 1.0850, "timestamp_ms": fresh_quote_ts_ms},
        )
    )
    sched._market_data_hub.get_bars = AsyncMock(
        side_effect=[
            MagicMock(source="websocket_cache", bars=bars_5min),
            MagicMock(source="websocket_cache", bars=bars_1h),
        ]
    )

    data = await sched._fetch_tactical_data("EURUSD")

    assert data.bars_5min.equals(bars_5min)
    assert data.bars_1h.equals(bars_1h)
    assert data.data_source == "websocket_cache"


async def test_fetch_tactical_data_uses_hub_rest_fallback_for_stale_symbol(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """v1.4.0: stale websocket symbols should still resolve through hub fallback."""
    import pandas as pd

    now = datetime.now(timezone.utc)
    fresh_quote_ts_ms = int((now - timedelta(seconds=15)).timestamp() * 1000)

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    bars_5min = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp(now - timedelta(minutes=5)),
                "open": 1.10,
                "high": 1.11,
                "low": 1.09,
                "close": 1.105,
                "volume": 0,
            }
        ]
    )
    bars_1h = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp(now - timedelta(minutes=55)),
                "open": 1.09,
                "high": 1.12,
                "low": 1.08,
                "close": 1.105,
                "volume": 0,
            }
        ]
    )
    sched._market_data_ready = True
    sched._market_data_hub = MagicMock()
    sched._market_data_hub.get_quote = AsyncMock(
        return_value=MagicMock(
            source="rest_fallback",
            quote={"bid": 1.0848, "ask": 1.0850, "timestamp_ms": fresh_quote_ts_ms},
        )
    )
    sched._market_data_hub.get_bars = AsyncMock(
        side_effect=[
            MagicMock(source="rest_fallback", bars=bars_5min),
            MagicMock(source="rest_fallback", bars=bars_1h),
        ]
    )

    data = await sched._fetch_tactical_data("EURUSD")

    assert data.bars_5min.equals(bars_5min)
    assert data.bars_1h.equals(bars_1h)
    assert data.data_source == "rest_fallback"


async def test_fetch_tactical_data_drops_stale_hub_1h_bars(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """Stale 1h hub bars must not be passed into tactical exit evaluation."""
    now = datetime.now(timezone.utc)
    fresh_quote_ts_ms = int((now - timedelta(seconds=15)).timestamp() * 1000)

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    bars_5min = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp(now - timedelta(minutes=5)),
                "open": 1.10,
                "high": 1.11,
                "low": 1.09,
                "close": 1.105,
                "volume": 0,
            }
        ]
    )
    bars_1h = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp(now - timedelta(days=2)),
                "open": 1.09,
                "high": 1.12,
                "low": 1.08,
                "close": 1.105,
                "volume": 0,
            }
        ]
    )
    sched._market_data_ready = True
    sched._market_data_hub = MagicMock()
    sched._market_data_hub.get_quote = AsyncMock(
        return_value=MagicMock(
            source="rest_fallback",
            quote={"bid": 1.0848, "ask": 1.0850, "timestamp_ms": fresh_quote_ts_ms},
        )
    )
    sched._market_data_hub.get_bars = AsyncMock(
        side_effect=[
            MagicMock(source="rest_fallback", bars=bars_5min),
            MagicMock(source="rest_fallback", bars=bars_1h),
        ]
    )

    with patch.object(sched, "_now_utc", return_value=now):
        data = await sched._fetch_tactical_data("EURUSD")

    assert data.bars_5min.equals(bars_5min)
    assert data.bars_1h.empty
    assert data.bars_1h_source == ""


async def test_fetch_tactical_data_drops_stale_hub_5min_bars(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """Stale 5m hub bars must not be passed into tactical exit evaluation."""
    now = datetime.now(timezone.utc)
    fresh_quote_ts_ms = int((now - timedelta(seconds=15)).timestamp() * 1000)

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    bars_5min = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp(now - timedelta(hours=2)),
                "open": 1.10,
                "high": 1.11,
                "low": 1.09,
                "close": 1.105,
                "volume": 0,
            }
        ]
    )
    bars_1h = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp(now - timedelta(minutes=55)),
                "open": 1.09,
                "high": 1.12,
                "low": 1.08,
                "close": 1.105,
                "volume": 0,
            }
        ]
    )
    sched._market_data_ready = True
    sched._market_data_hub = MagicMock()
    sched._market_data_hub.get_quote = AsyncMock(
        return_value=MagicMock(
            source="rest_fallback",
            quote={"bid": 1.0848, "ask": 1.0850, "timestamp_ms": fresh_quote_ts_ms},
        )
    )
    sched._market_data_hub.get_bars = AsyncMock(
        side_effect=[
            MagicMock(source="rest_fallback", bars=bars_5min),
            MagicMock(source="rest_fallback", bars=bars_1h),
        ]
    )

    with patch.object(sched, "_now_utc", return_value=now):
        data = await sched._fetch_tactical_data("EURUSD")

    assert data.bars_5min.empty
    assert data.bars_5min_source == ""
    assert data.bars_1h.equals(bars_1h)


async def test_fetch_tactical_data_keeps_1h_bar_when_close_time_is_still_fresh(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """A closed 1h bar should be judged by close time, not bucket open time."""
    now = datetime(2026, 3, 16, 6, 30, tzinfo=timezone.utc)
    fresh_quote_ts_ms = int((now - timedelta(seconds=15)).timestamp() * 1000)

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    bars_5min = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp(now - timedelta(minutes=5)),
                "open": 1.10,
                "high": 1.11,
                "low": 1.09,
                "close": 1.105,
                "volume": 0,
            }
        ]
    )
    bars_1h = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp(now - timedelta(hours=4, minutes=30)),
                "open": 1.09,
                "high": 1.12,
                "low": 1.08,
                "close": 1.105,
                "volume": 0,
            }
        ]
    )
    sched._market_data_ready = True
    sched._market_data_hub = MagicMock()
    sched._market_data_hub.get_quote = AsyncMock(
        return_value=MagicMock(
            source="rest_fallback",
            quote={"bid": 1.0848, "ask": 1.0850, "timestamp_ms": fresh_quote_ts_ms},
        )
    )
    sched._market_data_hub.get_bars = AsyncMock(
        side_effect=[
            MagicMock(source="rest_fallback", bars=bars_5min),
            MagicMock(source="rest_fallback", bars=bars_1h),
        ]
    )

    with patch.object(sched, "_now_utc", return_value=now):
        data = await sched._fetch_tactical_data("EURUSD")

    assert data.bars_1h.equals(bars_1h)
    assert data.bars_1h_source == "rest_fallback"


def test_sanitize_tactical_bars_suppresses_redundant_identical_stale_warnings(
    scheduler: Scheduler,
) -> None:
    """Repeated stale-bar warnings should be throttled until heartbeat/change."""
    now = datetime(2026, 3, 16, 14, 0, tzinfo=timezone.utc)
    stale_bars = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp(now - timedelta(hours=6)),
                "open": 1.09,
                "high": 1.12,
                "low": 1.08,
                "close": 1.105,
                "volume": 0,
            }
        ]
    )

    with (
        patch.object(scheduler, "_now_utc", return_value=now),
        patch("src.scheduler.scheduler.logger.warning") as mock_warning,
    ):
        scheduler._sanitize_tactical_bars(
            symbol="EURUSD",
            timeframe="1h",
            bars=stale_bars,
            source="rest_fallback",
        )
        scheduler._sanitize_tactical_bars(
            symbol="EURUSD",
            timeframe="1h",
            bars=stale_bars,
            source="rest_fallback",
        )

    assert mock_warning.call_count == 1
    warning_args = mock_warning.call_args.args
    assert "latest_open" in warning_args[0]
    assert "latest_close" in warning_args[0]


def test_sanitize_tactical_bars_relogs_after_heartbeat_window(
    scheduler: Scheduler,
) -> None:
    """Identical stale-bar warnings should reappear after the heartbeat window."""
    start = datetime(2026, 3, 16, 14, 0, tzinfo=timezone.utc)
    stale_bars = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp(start - timedelta(hours=6)),
                "open": 1.09,
                "high": 1.12,
                "low": 1.08,
                "close": 1.105,
                "volume": 0,
            }
        ]
    )

    with patch("src.scheduler.scheduler.logger.warning") as mock_warning:
        with patch.object(scheduler, "_now_utc", return_value=start):
            scheduler._sanitize_tactical_bars(
                symbol="EURUSD",
                timeframe="1h",
                bars=stale_bars,
                source="rest_fallback",
            )
        with patch.object(scheduler, "_now_utc", return_value=start + timedelta(minutes=16)):
            scheduler._sanitize_tactical_bars(
                symbol="EURUSD",
                timeframe="1h",
                bars=stale_bars,
                source="rest_fallback",
            )

    assert mock_warning.call_count == 2


async def test_fetch_tactical_data_uses_matchtrader_quote_when_hub_has_bars_only(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """When hub only has bars, scheduler should still fetch broker quote for freshness."""
    now_ms = int(time.time() * 1000)
    expected_quote_time = datetime.fromtimestamp((now_ms - 15_000) / 1000, tz=timezone.utc)
    now = datetime.now(timezone.utc)
    mock_matchtrader.get_quote.return_value = {
        "ask": 0.6012,
        "bid": 0.6009,
        "timestampMs": now_ms - 15_000,
    }

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    bars_5min = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp(now - timedelta(minutes=5)),
                "open": 0.6010,
                "high": 0.6020,
                "low": 0.6000,
                "close": 0.6015,
                "volume": 0,
            }
        ]
    )
    bars_1h = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp(now - timedelta(minutes=55)),
                "open": 0.5990,
                "high": 0.6030,
                "low": 0.5980,
                "close": 0.6015,
                "volume": 0,
            }
        ]
    )
    sched._market_data_ready = True
    sched._market_data_hub = MagicMock()
    sched._market_data_hub.get_quote = AsyncMock(
        return_value=MagicMock(source="rest_fallback", quote=None)
    )
    sched._market_data_hub.get_bars = AsyncMock(
        side_effect=[
            MagicMock(source="rest_fallback", bars=bars_5min),
            MagicMock(source="rest_fallback", bars=bars_1h),
        ]
    )

    data = await sched._fetch_tactical_data("NZDUSD")

    mock_matchtrader.get_quote.assert_awaited_once()
    assert data.bars_5min.equals(bars_5min)
    assert data.bars_1h.equals(bars_1h)
    assert data.current_spread == pytest.approx(0.0003)
    assert data.latest_bar_time == expected_quote_time


async def test_fetch_tactical_data_hub_bars_only_without_quote_timestamp_keeps_freshness_missing(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """Hub bars without any quote timestamp must not backfill freshness from bar time."""
    import pandas as pd

    now = datetime.now(timezone.utc)
    mock_matchtrader.get_quote.return_value = {"ask": 0.6012, "bid": 0.6009}

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    bars_5min = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp(now - timedelta(minutes=5)),
                "open": 0.6010,
                "high": 0.6020,
                "low": 0.6000,
                "close": 0.6015,
                "volume": 0,
            }
        ]
    )
    bars_1h = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp(now - timedelta(minutes=55)),
                "open": 0.5990,
                "high": 0.6030,
                "low": 0.5980,
                "close": 0.6015,
                "volume": 0,
            }
        ]
    )
    sched._market_data_ready = True
    sched._market_data_hub = MagicMock()
    sched._market_data_hub.get_quote = AsyncMock(
        return_value=MagicMock(source="rest_fallback", quote=None)
    )
    sched._market_data_hub.get_bars = AsyncMock(
        side_effect=[
            MagicMock(source="rest_fallback", bars=bars_5min),
            MagicMock(source="rest_fallback", bars=bars_1h),
        ]
    )

    data = await sched._fetch_tactical_data("NZDUSD")

    assert data.bars_5min.equals(bars_5min)
    assert data.bars_1h.equals(bars_1h)
    assert data.latest_bar_time is None


def test_build_metrics_snapshot_includes_market_data_feed_status(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )
    sched._config.scheduler.entry_funnel_mode = "scanner_tactical"
    sched._metrics.record_entry_funnel_event("scanner_candidate")
    sched._market_data_ready = True
    sched._market_data_hub = MagicMock()
    sched._market_data_hub.feed_status.return_value = {
        "websocket": {"state": "degraded", "last_error": "ping timeout"},
        "forced_stale_symbols": ["EURUSD"],
        "initialized_at": "2026-03-17T03:00:00+00:00",
        "uptime_seconds": 42,
        "websocket_closed_bar_counts": {"EURUSD": {"1m": 6, "5m": 1, "1h": 0}},
    }

    snapshot = sched._build_metrics_snapshot()

    assert snapshot["entry_funnel_mode"] == "scanner_tactical"
    assert snapshot["entry_funnel"]["scanner_candidates"] == 1
    assert snapshot["market_data"]["websocket"]["state"] == "degraded"
    assert snapshot["market_data"]["forced_stale_symbols"] == ["EURUSD"]
    assert snapshot["market_data"]["initialized_at"] == "2026-03-17T03:00:00+00:00"
    assert snapshot["market_data"]["uptime_seconds"] == 42
    assert snapshot["market_data"]["websocket_closed_bar_counts"]["EURUSD"]["5m"] == 1


async def test_fetch_tactical_data_uses_quote_timestamp(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """v1.3.9-fix: data_freshness gate should use MatchTrader quote timestamp.

    EODHD intraday bars can lag 10+ hours during DST transitions.
    The quote from MatchTrader is real-time (<1 min delay) and provides
    a reliable timestamp for the data_freshness hard gate.
    """
    import time
    from datetime import datetime, timezone

    import pandas as pd

    # Set up quote with a recent timestampMs (30 seconds ago)
    now_ms = int(time.time() * 1000)
    quote_ts_ms = now_ms - 30_000  # 30 seconds ago
    mock_matchtrader.get_quote.return_value = {
        "ask": 1.0850,
        "bid": 1.0848,
        "timestampMs": quote_ts_ms,
    }

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    # Inject a mock EODHD provider that returns EMPTY data (simulating DST lag)
    mock_eodhd = AsyncMock()
    empty_df = pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    mock_eodhd.fetch_bars = AsyncMock(return_value=empty_df)
    sched._eodhd = mock_eodhd

    data = await sched._fetch_tactical_data("EURUSD")

    # latest_bar_time should come from MatchTrader quote, NOT from EODHD bars
    assert data.latest_bar_time is not None, (
        "Expected latest_bar_time from MatchTrader quote, got None"
    )
    # Quote was 30s ago — age should be ~30s, well under data_max_age_seconds (600)
    age = (datetime.now(timezone.utc) - data.latest_bar_time).total_seconds()
    assert age < 120, f"Quote age {age:.0f}s too old — expected ~30s"
    assert age >= 25, f"Quote age {age:.0f}s too fresh — expected ~30s"


async def test_fetch_tactical_data_no_quote_timestamp_falls_through(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """When MatchTrader quote has no timestampMs AND EODHD fails, latest_bar_time stays None.

    This ensures the data_freshness gate correctly rejects when there is no
    reliable timestamp source at all.
    """

    # Quote without timestampMs (older mock format)
    mock_matchtrader.get_quote.return_value = {"ask": 1.0850, "bid": 1.0848}

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    # EODHD also fails
    mock_eodhd = AsyncMock()
    mock_eodhd.fetch_bars = AsyncMock(side_effect=Exception("EODHD down"))
    sched._eodhd = mock_eodhd

    data = await sched._fetch_tactical_data("EURUSD")

    assert data.latest_bar_time is None, (
        f"Expected None when no timestamp source available, got {data.latest_bar_time}"
    )
    assert data.latest_bar_time is None, (
        f"Expected latest_bar_time=None after EODHD failure, got {data.latest_bar_time}"
    )
