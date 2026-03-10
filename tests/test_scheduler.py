"""
Tests for src/scheduler/scheduler.py — Async multi-cycle orchestrator.

Uses mocked ScannerBridge, AgentBridge, ExecutionEngine, and MatchTraderClient
with a real DecisionStore (in-memory SQLite). Tests cover all worker loops:
scanner, LLM worker, execution, janitor, and equity monitor.
"""

import asyncio
import unittest.mock
from unittest.mock import AsyncMock, MagicMock, patch

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
) -> MagicMock:
    """Create a mock ScannerSignal."""
    signal = MagicMock()
    signal.instrument = instrument
    signal.score = score
    signal.confidence = confidence
    signal.score_gap = 0.1
    signal.drop_distance = 0.05
    signal.topk_spread = 0.02
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


# ── v1.3.9: Tactical Gate Enforcement Tests ──────────────────────────────


async def test_tactical_gate_blocks_intent_when_shadow_mode_off(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
):
    """v1.3.9-fix: When shadow_mode=False, tactical WAIT/REJECT must cancel the intent.

    Previously the tactical result was only logged/alerted but never used to block.
    """
    from src.decision.tactical_validator import TacticalResult

    # Enable tactical gate with shadow_mode=false
    config.tactical.enabled = True
    config.tactical.shadow_mode = False

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )

    # Create and claim an intent
    intent = TradeIntent(
        trade_date=Scheduler._today_str(),
        symbol="EURUSD",
        scanner_score=0.85,
        scanner_confidence="high",
    )
    store.insert_intent(intent)
    claimed = store.claim_next_pending("llm-0")
    assert claimed is not None

    # Mock _run_tactical_validation to return WAIT
    tactical_wait = TacticalResult(
        action="WAIT",
        detail="Spread too wide (0.0005 > 0.0003)",
    )
    with patch.object(sched, "_run_tactical_validation", new_callable=AsyncMock) as mock_tac:
        mock_tac.return_value = tactical_wait
        await sched._process_claimed_intent("llm-0", claimed)

    # Verify: intent should be cancelled, NOT ready_for_exec
    final = store.get_intent(claimed.id)
    assert final.status == "cancelled"
    assert "Tactical gate WAIT" in (final.execution_error or "")


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
