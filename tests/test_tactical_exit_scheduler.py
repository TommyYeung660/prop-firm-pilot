"""Tests for scheduler integration with tactical exit manager."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import (
    AccountConfig,
    AppConfig,
    ComplianceConfig,
    DecisionStoreConfig,
    MonitorConfig,
    SchedulerConfig,
)
from src.decision.tactical_exit_manager import TacticalExitEvaluation
from src.decision.tactical_exit_rules import TacticalExitDecision
from src.decision.tactical_validator import TacticalData
from src.decision_store.sqlite_store import DecisionStore
from src.scheduler import scheduler as scheduler_module
from src.scheduler.scheduler import Scheduler


def _tactical_cycle_log_calls(mock_info: MagicMock) -> list:
    """Return only tactical-exit cycle summary log calls."""
    return [
        call
        for call in mock_info.call_args_list
        if call.args and "Tactical exit cycle" in call.args[0]
    ]


@pytest.fixture
def store(tmp_path) -> DecisionStore:
    """Create a temporary DecisionStore for scheduler tactical-exit tests."""
    db_path = f"{tmp_path}/test_tactical_exit_scheduler.db"
    decision_store = DecisionStore(db_path=db_path)
    yield decision_store
    decision_store.close()


@pytest.fixture
def config() -> AppConfig:
    """Create scheduler config with tactical exit enabled."""
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
            reeval_min_hold_seconds=0,
            reeval_interval_seconds=14400,
        ),
        decision_store=DecisionStoreConfig(),
        monitor=MonitorConfig(),
    )


@pytest.fixture
def mock_matchtrader() -> AsyncMock:
    """Create mock MatchTrader client with a non-critical write budget."""
    client = AsyncMock()
    client.get_balance.return_value = MagicMock(
        balance=50000.0,
        equity=50000.0,
        margin=0.0,
        free_margin=50000.0,
    )
    rate_limiter = MagicMock()
    rate_limiter.write_remaining = 400
    rate_limiter.daily_write_limit = 2000
    client.rate_limiter = rate_limiter
    client._rate_limiter = rate_limiter
    return client


@pytest.fixture
def scheduler(config: AppConfig, store: DecisionStore, mock_matchtrader: AsyncMock) -> Scheduler:
    """Create a scheduler with mocked dependencies for tactical exit wiring tests."""
    return Scheduler(
        config=config,
        store=store,
        scanner=MagicMock(),
        agents=MagicMock(),
        engine=AsyncMock(),
        matchtrader=mock_matchtrader,
    )


def _make_position() -> MagicMock:
    """Create a mock open position."""
    pos = MagicMock()
    pos.position_id = "POS-1"
    pos.symbol = "EURUSD."
    pos.side = "BUY"
    pos.volume = 0.10
    pos.open_price = 1.1000
    pos.current_price = 1.1035
    pos.profit = 35.0
    pos.sl_price = 1.0980
    pos.tp_price = 1.1080
    return pos


def _make_opened_intent() -> MagicMock:
    """Create a minimal opened intent with tactical-exit-compatible metadata."""
    intent = MagicMock()
    intent.id = "INT-1"
    intent.symbol = "EURUSD"
    intent.position_id = "POS-1"
    intent.execution_meta = "{}"
    intent.executed_at = None
    intent.created_at = None
    return intent


@pytest.mark.asyncio
async def test_run_tactical_exit_cycle_delegates_to_manager(scheduler: Scheduler) -> None:
    """Scheduler should delegate each open position to the tactical exit manager."""
    scheduler._fetch_tactical_data = AsyncMock(return_value=TacticalData())
    scheduler._handle_tactical_exit_evaluation = AsyncMock()
    scheduler._tactical_exit_manager = MagicMock()
    scheduler._tactical_exit_manager.evaluate_position.return_value = TacticalExitEvaluation(
        decision=TacticalExitDecision(
            action="HOLD",
            state="INITIAL_RISK",
            reason="no_tactical_exit_action",
        )
    )

    position = _make_position()
    intent = _make_opened_intent()

    await scheduler._run_tactical_exit_cycle([position], [intent])

    scheduler._tactical_exit_manager.evaluate_position.assert_called_once()
    scheduler._handle_tactical_exit_evaluation.assert_awaited_once()


@pytest.mark.asyncio
async def test_llm_reeval_only_runs_for_exception_cases(scheduler: Scheduler) -> None:
    """LLM re-evaluation should only be triggered when tactical exit flags an exception."""
    scheduler._fetch_tactical_data = AsyncMock(return_value=TacticalData())
    scheduler._tactical_exit_manager = MagicMock()
    scheduler._tactical_exit_manager.evaluate_position.return_value = TacticalExitEvaluation(
        decision=TacticalExitDecision(
            action="HOLD",
            state="PROFIT_PROTECTION",
            reason="severe_tactical_reversal",
        ),
        requires_llm_exception_review=True,
    )

    with (
        pytest.MonkeyPatch.context() as mp,
    ):
        mock_reeval = AsyncMock()
        mp.setattr(scheduler, "_reevaluate_open_positions", mock_reeval)
        await scheduler._run_tactical_exit_cycle([_make_position()], [_make_opened_intent()])
        mock_reeval.assert_awaited_once()


@pytest.mark.asyncio
async def test_non_exception_tactical_exit_does_not_call_llm_reeval(scheduler: Scheduler) -> None:
    """Normal tactical exit evaluations should not call the expensive LLM path."""
    scheduler._fetch_tactical_data = AsyncMock(return_value=TacticalData())
    scheduler._tactical_exit_manager = MagicMock()
    scheduler._tactical_exit_manager.evaluate_position.return_value = TacticalExitEvaluation(
        decision=TacticalExitDecision(
            action="HOLD",
            state="PROFIT_PROTECTION",
            reason="write_budget_blocked",
        ),
        requires_llm_exception_review=False,
    )

    with (
        pytest.MonkeyPatch.context() as mp,
    ):
        mock_reeval = AsyncMock()
        mp.setattr(scheduler, "_reevaluate_open_positions", mock_reeval)
        await scheduler._run_tactical_exit_cycle([_make_position()], [_make_opened_intent()])
        mock_reeval.assert_not_awaited()


def test_position_monitor_base_interval_uses_tactical_exit_cadence(scheduler: Scheduler) -> None:
    """Position monitor cadence should honor faster tactical exit evaluation settings."""
    scheduler._config.scheduler.position_monitor_interval_seconds = 120
    scheduler._config.tactical.exit.evaluation_interval_seconds = 60

    assert scheduler._position_monitor_base_interval_seconds() == 60


@pytest.mark.asyncio
async def test_run_tactical_exit_cycle_logs_hold_summary(
    scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hold-only tactical cycles should still emit an operator-visible summary log."""
    scheduler._fetch_tactical_data = AsyncMock(return_value=TacticalData())
    scheduler._handle_tactical_exit_evaluation = AsyncMock()
    scheduler._tactical_exit_manager = MagicMock()
    scheduler._tactical_exit_manager.evaluate_position.return_value = TacticalExitEvaluation(
        decision=TacticalExitDecision(
            action="HOLD",
            state="INITIAL_RISK",
            reason="no_tactical_exit_action",
        )
    )

    mock_info = MagicMock()
    monkeypatch.setattr(scheduler_module.logger, "info", mock_info)

    with monkeypatch.context() as mp:
        mp.setattr(scheduler, "_now_utc", MagicMock(return_value=datetime.now(timezone.utc)))
        await scheduler._run_tactical_exit_cycle([_make_position()], [_make_opened_intent()])

    assert _tactical_cycle_log_calls(mock_info)


@pytest.mark.asyncio
async def test_run_tactical_exit_cycle_suppresses_redundant_identical_hold_summary(
    scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated identical HOLD summaries should not be logged every cycle."""
    scheduler._fetch_tactical_data = AsyncMock(return_value=TacticalData())
    scheduler._handle_tactical_exit_evaluation = AsyncMock()
    scheduler._tactical_exit_manager = MagicMock()
    scheduler._tactical_exit_manager.evaluate_position.return_value = TacticalExitEvaluation(
        decision=TacticalExitDecision(
            action="HOLD",
            state="INITIAL_RISK",
            reason="no_tactical_exit_action",
        )
    )

    t0 = datetime(2026, 3, 16, 12, 0, tzinfo=timezone.utc)
    clock = MagicMock(return_value=t0)
    mock_info = MagicMock()
    monkeypatch.setattr(scheduler_module.logger, "info", mock_info)
    monkeypatch.setattr(scheduler, "_now_utc", clock)

    await scheduler._run_tactical_exit_cycle([_make_position()], [_make_opened_intent()])
    clock.return_value = t0 + timedelta(minutes=5)
    await scheduler._run_tactical_exit_cycle([_make_position()], [_make_opened_intent()])

    assert len(_tactical_cycle_log_calls(mock_info)) == 1


@pytest.mark.asyncio
async def test_run_tactical_exit_cycle_relogs_identical_summary_after_heartbeat(
    scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identical summaries should emit a low-frequency heartbeat log."""
    scheduler._fetch_tactical_data = AsyncMock(return_value=TacticalData())
    scheduler._handle_tactical_exit_evaluation = AsyncMock()
    scheduler._tactical_exit_manager = MagicMock()
    scheduler._tactical_exit_manager.evaluate_position.return_value = TacticalExitEvaluation(
        decision=TacticalExitDecision(
            action="HOLD",
            state="INITIAL_RISK",
            reason="no_tactical_exit_action",
        )
    )

    t0 = datetime(2026, 3, 16, 12, 0, tzinfo=timezone.utc)
    clock = MagicMock(return_value=t0)
    mock_info = MagicMock()
    monkeypatch.setattr(scheduler_module.logger, "info", mock_info)
    monkeypatch.setattr(scheduler, "_now_utc", clock)

    await scheduler._run_tactical_exit_cycle([_make_position()], [_make_opened_intent()])
    clock.return_value = t0 + timedelta(minutes=10)
    await scheduler._run_tactical_exit_cycle([_make_position()], [_make_opened_intent()])
    clock.return_value = t0 + timedelta(minutes=16)
    await scheduler._run_tactical_exit_cycle([_make_position()], [_make_opened_intent()])

    assert len(_tactical_cycle_log_calls(mock_info)) == 2


@pytest.mark.asyncio
async def test_run_tactical_exit_cycle_logs_immediately_when_summary_changes(
    scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Summary changes should be logged immediately without waiting for heartbeat."""
    scheduler._fetch_tactical_data = AsyncMock(return_value=TacticalData())
    scheduler._handle_tactical_exit_evaluation = AsyncMock()
    scheduler._tactical_exit_manager = MagicMock()
    scheduler._tactical_exit_manager.evaluate_position.side_effect = [
        TacticalExitEvaluation(
            decision=TacticalExitDecision(
                action="HOLD",
                state="INITIAL_RISK",
                reason="no_tactical_exit_action",
            )
        ),
        TacticalExitEvaluation(
            decision=TacticalExitDecision(
                action="HOLD",
                state="PROFIT_PROTECTION",
                reason="write_budget_blocked",
            ),
            skip_reason="write_budget_blocked",
        ),
    ]

    t0 = datetime(2026, 3, 16, 12, 0, tzinfo=timezone.utc)
    clock = MagicMock(return_value=t0)
    mock_info = MagicMock()
    monkeypatch.setattr(scheduler_module.logger, "info", mock_info)
    monkeypatch.setattr(scheduler, "_now_utc", clock)

    await scheduler._run_tactical_exit_cycle([_make_position()], [_make_opened_intent()])
    clock.return_value = t0 + timedelta(minutes=1)
    await scheduler._run_tactical_exit_cycle([_make_position()], [_make_opened_intent()])

    assert len(_tactical_cycle_log_calls(mock_info)) == 2
