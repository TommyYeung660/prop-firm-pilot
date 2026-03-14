"""Tests for tactical exit action execution and persistence."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import (
    AccountConfig,
    AppConfig,
    ComplianceConfig,
    DecisionStoreConfig,
    InstrumentConfig,
    MonitorConfig,
    SchedulerConfig,
)
from src.decision.schemas import TradeIntent
from src.decision.tactical_exit_manager import TacticalExitEvaluation
from src.decision.tactical_exit_rules import TacticalExitDecision
from src.decision_store.sqlite_store import DecisionStore
from src.execution.matchtrader_client import OrderResult
from src.monitor.trade_journal import TradeJournal
from src.scheduler.scheduler import Scheduler


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    """Create AppConfig with instrument sizing for partial-close tests."""
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
        monitor=MonitorConfig(trade_journal_path=str(tmp_path / "trade_journal.jsonl")),
        instruments={
            "EURUSD": InstrumentConfig(
                pip_value=10.0,
                pip_size=0.0001,
                min_lot=0.01,
                max_lot=5.0,
                avg_spread_pips=1.0,
            )
        },
    )


@pytest.fixture
def store(tmp_path: Path) -> DecisionStore:
    """Create a temporary DecisionStore."""
    decision_store = DecisionStore(str(tmp_path / "test_tactical_exit_execution.db"))
    yield decision_store
    decision_store.close()


@pytest.fixture
def trade_journal(tmp_path: Path) -> TradeJournal:
    """Create a temporary TradeJournal."""
    return TradeJournal(tmp_path / "trade_journal.jsonl")


@pytest.fixture
def mock_matchtrader() -> AsyncMock:
    """Create a mock MatchTrader client for tactical exit execution tests."""
    client = AsyncMock()
    client.close_position.return_value = OrderResult(
        success=True,
        position_id="POS-1",
        message="OK",
    )
    client.modify_position.return_value = OrderResult(
        success=True,
        position_id="POS-1",
        message="OK",
    )
    client.verify_sl_tp.return_value = True
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
def scheduler(
    config: AppConfig,
    store: DecisionStore,
    trade_journal: TradeJournal,
    mock_matchtrader: AsyncMock,
) -> Scheduler:
    """Create a scheduler with journal + mocked broker for tactical exit execution."""
    return Scheduler(
        config=config,
        store=store,
        scanner=MagicMock(),
        agents=MagicMock(),
        engine=AsyncMock(),
        matchtrader=mock_matchtrader,
        trade_journal=trade_journal,
    )


def _make_position() -> MagicMock:
    """Create a mock open position."""
    pos = MagicMock()
    pos.position_id = "POS-1"
    pos.symbol = "EURUSD."
    pos.side = "BUY"
    pos.volume = 0.10
    pos.open_price = 1.1000
    pos.current_price = 1.1060
    pos.profit = 60.0
    pos.sl_price = 1.0980
    pos.tp_price = 1.1080
    return pos


def _insert_opened_intent(store: DecisionStore) -> TradeIntent:
    """Insert an opened intent with execution metadata for tactical exit tests."""
    intent = TradeIntent(
        id="INT-1",
        trade_date="2026-03-12",
        symbol="EURUSD",
        scanner_score=0.85,
        scanner_confidence="high",
    )
    store.insert_intent(intent)
    store.claim_next_pending("test-worker")
    store.update_intent_decision(intent.id, "BUY", 20.0, 80.0, "risk", "{}")
    store.mark_ready_for_exec(intent.id)
    store.mark_executing(intent.id)
    store.mark_opened(intent.id, "POS-1")
    store.update_execution_meta(
        intent.id,
        json.dumps(
            {
                "fill_price": 1.1000,
                "sl_price": 1.0980,
                "tp_price": 1.1080,
                "volume": 0.10,
                "side": "BUY",
            }
        ),
    )
    return store.get_intent(intent.id)


@pytest.mark.asyncio
async def test_partial_close_executes_half_volume_and_marks_meta(
    scheduler: Scheduler,
    store: DecisionStore,
    trade_journal: TradeJournal,
    mock_matchtrader: AsyncMock,
) -> None:
    """Partial close should use 50% volume, persist metadata, and journal the action."""
    position = _make_position()
    intent = _insert_opened_intent(store)
    evaluation = TacticalExitEvaluation(
        decision=TacticalExitDecision(
            action="PARTIAL_CLOSE",
            state="PROFIT_PROTECTION",
            reason="profit_protection_partial_close",
            partial_close_ratio=0.5,
        )
    )

    await scheduler._execute_tactical_exit_action(position, intent, evaluation)

    call_kwargs = mock_matchtrader.close_position.await_args.kwargs
    assert call_kwargs["volume"] == pytest.approx(0.05)

    meta = json.loads(store.get_decision(intent.id).execution_meta)
    assert meta["partial_close_done"] is True
    assert meta["partial_close_volume"] == pytest.approx(0.05)
    assert meta["last_tactical_exit_action"] == "PARTIAL_CLOSE"
    assert meta["close_control"]["trigger_source"] == "tactical_exit"
    assert meta["close_control"]["action_kind"] == "partial_close"
    assert meta["close_control"]["execution_status"] == "submitted"

    entries = [
        json.loads(line)
        for line in trade_journal._path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(entry.get("type") == "TACTICAL_EXIT_ACTION" for entry in entries)
    assert any(
        entry.get("type") == "CLOSE_CONTROL_EVENT"
        and entry.get("execution_status") == "submitted"
        for entry in entries
    )


@pytest.mark.asyncio
async def test_modify_verification_failure_does_not_mark_action_complete(
    scheduler: Scheduler,
    store: DecisionStore,
    trade_journal: TradeJournal,
    mock_matchtrader: AsyncMock,
) -> None:
    """Read-back verification failure should not persist a successful tactical exit action."""
    position = _make_position()
    intent = _insert_opened_intent(store)
    mock_matchtrader.verify_sl_tp.return_value = False
    evaluation = TacticalExitEvaluation(
        decision=TacticalExitDecision(
            action="MOVE_TO_BREAKEVEN",
            state="PROTECTION",
            reason="breakeven_threshold_reached",
            new_sl=position.open_price,
        )
    )

    await scheduler._execute_tactical_exit_action(position, intent, evaluation)

    meta = json.loads(store.get_decision(intent.id).execution_meta)
    assert meta.get("last_tactical_exit_action") != "MOVE_TO_BREAKEVEN"
    assert meta.get("breakeven_sl") is None
    assert meta["close_control"]["execution_status"] == "verify_failed"
    assert meta["close_control"]["readback_status"] == "mismatch"

    entries = [
        json.loads(line)
        for line in trade_journal._path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        entry.get("type") == "CLOSE_CONTROL_EVENT"
        and entry.get("execution_status") == "verify_failed"
        for entry in entries
    )


@pytest.mark.asyncio
async def test_modify_action_normalizes_prices_before_verify_and_meta_update(
    scheduler: Scheduler,
    store: DecisionStore,
    mock_matchtrader: AsyncMock,
) -> None:
    """Tactical modify actions should round prices to broker precision before verify."""
    position = _make_position()
    intent = _insert_opened_intent(store)
    evaluation = TacticalExitEvaluation(
        decision=TacticalExitDecision(
            action="TRAIL_SL",
            state="TREND_EXTENSION",
            reason="atr_trailing_stop_improved",
            new_sl=1.1032149,
            new_tp=1.1098761,
        )
    )

    await scheduler._execute_tactical_exit_action(position, intent, evaluation)

    modify_kwargs = mock_matchtrader.modify_position.await_args.kwargs
    assert modify_kwargs["sl"] == pytest.approx(1.10321)
    assert modify_kwargs["tp"] == pytest.approx(1.10988)

    verify_kwargs = mock_matchtrader.verify_sl_tp.await_args.kwargs
    assert verify_kwargs["expected_sl"] == pytest.approx(1.10321)
    assert verify_kwargs["expected_tp"] == pytest.approx(1.10988)
    assert verify_kwargs["price_precision"] == 5

    meta = json.loads(store.get_decision(intent.id).execution_meta)
    assert meta["trailing_sl"] == pytest.approx(1.10321)
    assert meta["close_control"]["execution_status"] == "accepted"
    assert meta["close_control"]["readback_status"] == "verified"
