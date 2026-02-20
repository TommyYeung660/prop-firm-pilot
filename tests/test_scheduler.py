"""
Tests for src/scheduler/scheduler.py — Async multi-cycle orchestrator.

Uses mocked ScannerBridge, AgentBridge, ExecutionEngine, and MatchTraderClient
with a real DecisionStore (in-memory SQLite). Tests cover all worker loops:
scanner, LLM worker, execution, janitor, and equity monitor.
"""

import unittest.mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import (
    AccountConfig,
    AppConfig,
    ComplianceConfig,
    DecisionStoreConfig,
    MonitorConfig,
    SchedulerConfig,
)
from src.decision.agent_bridge import AgentDecision
from src.decision.schemas import TradeIntent
from src.decision_store.sqlite_store import DecisionStore
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
    to False causing the while-loop to exit.
    """
    call_count = 0

    async def fake_sleep(seconds: float) -> None:
        nonlocal call_count
        call_count += 1
        if call_count >= 1:
            scheduler._running = False

    with unittest.mock.patch("asyncio.sleep", fake_sleep):
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

        with unittest.mock.patch("asyncio.sleep", fake_sleep):
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
        # Close unawaited coroutines to suppress RuntimeWarning
        for coro in args:
            coro.close()

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
        # Close unawaited coroutines to suppress RuntimeWarning
        for coro in args:
            coro.close()


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
