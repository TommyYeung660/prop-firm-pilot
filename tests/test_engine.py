"""
Tests for src/execution/engine.py — ExecutionEngine trade execution pipeline.

Uses mocked MatchTraderClient and PropFirmGuard with a real DecisionStore
(in-memory SQLite). Tests cover the full execution pipeline: compliance
checking, position sizing, trade execution, and state transitions.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.compliance.prop_firm_guard import AccountSnapshot, ComplianceResult, TradePlan
from src.config import (
    AccountConfig,
    AppConfig,
    ComplianceConfig,
    ExecutionConfig,
    InstrumentConfig,
)
from src.decision.schemas import TradeIntent
from src.decision_store.sqlite_store import DecisionStore
from src.execution.engine import ExecutionEngine

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: object) -> DecisionStore:
    """Create a fresh DecisionStore with a temporary database."""
    db_path = f"{tmp_path}/test_engine.db"
    s = DecisionStore(db_path=db_path)
    yield s  # type: ignore[misc]
    s.close()


@pytest.fixture
def config() -> AppConfig:
    """Minimal AppConfig for engine tests."""
    return AppConfig(
        account=AccountConfig(initial_balance=50000),
        compliance=ComplianceConfig(),
        execution=ExecutionConfig(
            max_positions=3,
            default_risk_pct=0.01,
            random_delay_min=0.0,
            random_delay_max=0.0,
        ),
        instruments={
            "EURUSD": InstrumentConfig(pip_value=10.0, pip_size=0.0001),
            "GBPUSD": InstrumentConfig(pip_value=10.0, pip_size=0.0001),
        },
    )


@pytest.fixture
def mock_guard() -> MagicMock:
    """Mock PropFirmGuard that passes all checks by default."""
    guard = MagicMock()
    guard.check_all.return_value = ComplianceResult(
        passed=True, rule_name="ALL", reason="All checks passed"
    )
    guard.add_random_delay.return_value = 0.0
    return guard


@pytest.fixture
def mock_matchtrader() -> AsyncMock:
    """Mock MatchTraderClient with default balance and empty positions."""
    client = AsyncMock()
    client.get_balance.return_value = MagicMock(
        balance=50000.0,
        equity=50000.0,
        margin=0.0,
        free_margin=50000.0,
    )
    client.get_open_positions.return_value = []
    client.open_position.return_value = MagicMock(
        success=True,
        position_id="pos_123",
        message="Position opened successfully",
    )
    return client


@pytest.fixture
def mock_sizer() -> MagicMock:
    """Mock PositionSizer with deterministic volume."""
    sizer = MagicMock()
    sizer.calculate_volume.return_value = 0.10
    sizer.calculate_risk_amount.return_value = 40.0
    return sizer


@pytest.fixture
def engine(
    store: DecisionStore,
    mock_guard: MagicMock,
    mock_matchtrader: AsyncMock,
    mock_sizer: MagicMock,
    config: AppConfig,
) -> ExecutionEngine:
    """Create an ExecutionEngine with all mocked dependencies."""
    return ExecutionEngine(
        store=store,
        guard=mock_guard,
        matchtrader=mock_matchtrader,
        sizer=mock_sizer,
        config=config,
    )


def _make_ready_intent(
    store: DecisionStore,
    symbol: str = "EURUSD",
    side: str = "BUY",
    sl_pips: float = 40.0,
    tp_pips: float = 80.0,
    scanner_confidence: str = "high",
) -> TradeIntent:
    """Create and advance an intent to ready_for_exec state."""
    intent = TradeIntent(
        trade_date="2026-02-16",
        symbol=symbol,
        scanner_score=0.85,
        scanner_confidence=scanner_confidence,
    )
    store.insert_intent(intent)

    # Advance to claimed
    claimed = store.claim_next_pending("llm-0")
    assert claimed is not None

    # Fill LLM decision
    store.update_intent_decision(
        intent_id=intent.id,
        side=side,
        sl_pips=sl_pips,
        tp_pips=tp_pips,
        risk_report="Test risk report",
        state_json='{"test": true}',
    )

    # Mark ready for execution
    store.mark_ready_for_exec(intent.id)
    return intent


def _make_closed_intent_today(
    store: DecisionStore,
    symbol: str,
    realized_pnl: float,
) -> TradeIntent:
    """Create an intent closed today with realized PnL."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    intent = TradeIntent(
        trade_date=today,
        symbol=symbol,
        scanner_score=0.80,
        scanner_confidence="high",
    )
    store.insert_intent(intent)
    claimed = store.claim_next_pending("llm-closed")
    assert claimed is not None
    store.update_intent_decision(
        intent.id,
        side="BUY",
        sl_pips=30.0,
        tp_pips=60.0,
        risk_report="closed test",
        state_json="{}",
    )
    store.mark_ready_for_exec(intent.id)
    store.mark_executing(intent.id)
    store.mark_opened(intent.id, position_id=f"closed-{symbol}")
    store.mark_closed(intent.id, realized_pnl=realized_pnl, exit_reason="tp_hit")
    closed = store.get_intent(intent.id)
    assert closed is not None
    return closed


def _make_opened_intent_with_execution_risk(
    store: DecisionStore,
    *,
    symbol: str,
    side: str,
    position_id: str,
    risk_pct: float,
) -> TradeIntent:
    """Create an opened intent and persist execution_meta.risk_pct for risk guard tests."""
    intent = TradeIntent(
        trade_date="2026-02-16",
        symbol=symbol,
        scanner_score=0.80,
        scanner_confidence="high",
    )
    store.insert_intent(intent)
    claimed = store.claim_next_pending("llm-opened")
    assert claimed is not None
    store.update_intent_decision(
        intent.id,
        side=side,
        sl_pips=40.0,
        tp_pips=80.0,
        risk_report="opened test",
        state_json="{}",
    )
    store.mark_ready_for_exec(intent.id)
    store.mark_executing(intent.id)
    store.mark_opened(intent.id, position_id=position_id)
    execution_meta = ExecutionEngine._build_execution_meta(
        fill_price=None,
        volume=0.10,
        side=side,
        sl_price=None,
        tp_price=None,
        sl_pips=40.0,
        tp_pips=80.0,
        pre_trade_bid=None,
        pre_trade_ask=None,
        slippage_pips=None,
        execution_latency_ms=None,
        random_delay_seconds=0.0,
        compliance_passed=True,
        order_raw_response={"positionId": position_id},
        risk_pct=risk_pct,
    )
    store.update_execution_meta(intent.id, execution_meta)
    opened = store.get_intent(intent.id)
    assert opened is not None
    return opened


def _get_execution_meta(store: DecisionStore, intent_id: str) -> dict[str, object]:
    """Read execution_meta JSON from decision row."""
    row = store._conn.execute(
        "SELECT execution_meta FROM decisions WHERE intent_id = ?",
        (intent_id,),
    ).fetchone()
    assert row is not None
    raw = row["execution_meta"] or "{}"
    return json.loads(raw)


# ── Execution Pipeline Tests ───────────────────────────────────────────────


class TestExecuteReadyIntents:
    """Tests for ExecutionEngine.execute_ready_intents()."""

    async def test_no_ready_intents_returns_zero(self, engine: ExecutionEngine) -> None:
        """Should return 0 when no intents are ready."""
        result = await engine.execute_ready_intents()
        assert result == 0

    async def test_successful_execution(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should execute a ready intent and mark it as opened."""
        intent = _make_ready_intent(store)

        result = await engine.execute_ready_intents()
        assert result == 1

        # Verify intent is now opened
        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "opened"
        assert updated.position_id == "pos_123"

        # Verify MatchTrader was called
        mock_matchtrader.open_position.assert_called_once_with(
            symbol="EURUSD",
            side="BUY",
            volume=0.10,
        )

    async def test_multiple_intents_processed(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
    ) -> None:
        """Should process all ready intents in one call."""
        intent1 = _make_ready_intent(store, symbol="EURUSD")
        intent2 = _make_ready_intent(store, symbol="GBPUSD")

        result = await engine.execute_ready_intents()
        assert result == 2

        assert store.get_intent(intent1.id).status == "opened"
        assert store.get_intent(intent2.id).status == "opened"

    async def test_sell_side_execution(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should correctly execute SELL trades."""
        _make_ready_intent(store, side="SELL")

        await engine.execute_ready_intents()

        mock_matchtrader.open_position.assert_called_once_with(
            symbol="EURUSD",
            side="SELL",
            volume=0.10,
        )


# ── Compliance Rejection Tests ──────────────────────────────────────────────


class TestComplianceGate:
    """Tests for compliance check integration."""

    async def test_compliance_rejection(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_guard: MagicMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should reject intent and NOT call open_position when compliance fails."""
        mock_guard.check_all.return_value = ComplianceResult(
            passed=False,
            rule_name="DAILY_DRAWDOWN",
            reason="Projected daily loss exceeds safety limit",
        )

        intent = _make_ready_intent(store)
        await engine.execute_ready_intents()

        # Intent should be rejected
        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "rejected"
        assert "daily loss" in updated.execution_error.lower()

        # MatchTrader should NOT be called
        mock_matchtrader.open_position.assert_not_called()

    async def test_compliance_snapshot_stored_on_rejection(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_guard: MagicMock,
    ) -> None:
        """Should persist compliance snapshot even when rejected."""
        mock_guard.check_all.return_value = ComplianceResult(
            passed=False,
            rule_name="MAX_DRAWDOWN",
            reason="Max drawdown exceeded",
            details={"current_loss": 3000.0},
        )

        intent = _make_ready_intent(store)
        await engine.execute_ready_intents()

        updated = store.get_intent(intent.id)
        assert updated.compliance_snapshot != ""

        snapshot_data = json.loads(updated.compliance_snapshot)
        assert snapshot_data["passed"] is False
        assert snapshot_data["rule_name"] == "MAX_DRAWDOWN"
        assert "account" in snapshot_data
        assert snapshot_data["account"]["balance"] == 50000.0

    async def test_compliance_snapshot_stored_on_success(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
    ) -> None:
        """Should persist compliance snapshot on successful execution."""
        intent = _make_ready_intent(store)
        await engine.execute_ready_intents()

        updated = store.get_intent(intent.id)
        assert updated.compliance_snapshot != ""

        snapshot_data = json.loads(updated.compliance_snapshot)
        assert snapshot_data["passed"] is True

    async def test_guard_receives_correct_trade_plan(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_guard: MagicMock,
        mock_sizer: MagicMock,
    ) -> None:
        """Should pass correctly built TradePlan to PropFirmGuard."""
        _make_ready_intent(store, symbol="EURUSD", side="BUY", sl_pips=40.0, tp_pips=80.0)
        await engine.execute_ready_intents()

        # Verify guard received a proper TradePlan
        call_args = mock_guard.check_all.call_args
        trade_plan = call_args[0][0]
        assert isinstance(trade_plan, TradePlan)
        assert trade_plan.symbol == "EURUSD"
        assert trade_plan.side == "BUY"
        assert trade_plan.volume == 0.10
        assert trade_plan.risk_amount == 40.0


# ── Failure Handling Tests ──────────────────────────────────────────────────


class TestFailureHandling:
    """Tests for execution failure scenarios."""

    async def test_api_error_marks_failed(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should mark intent as failed when MatchTrader API raises."""
        mock_matchtrader.open_position.side_effect = RuntimeError("Connection timeout")

        intent = _make_ready_intent(store)
        await engine.execute_ready_intents()

        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "failed"
        assert "Connection timeout" in updated.execution_error

    async def test_order_failure_marks_failed(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should mark intent as failed when order returns success=False."""
        mock_matchtrader.open_position.return_value = MagicMock(
            success=False,
            position_id="",
            message="Insufficient margin",
        )

        intent = _make_ready_intent(store)
        await engine.execute_ready_intents()

        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "failed"
        assert "Insufficient margin" in updated.execution_error

    async def test_account_snapshot_error_marks_failed(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should mark intent as failed when account snapshot fetch fails."""
        mock_matchtrader.get_balance.side_effect = RuntimeError("Auth expired")

        intent = _make_ready_intent(store)
        await engine.execute_ready_intents()

        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "failed"
        assert "Auth expired" in updated.execution_error

    async def test_invalid_side_skipped(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Should skip intents with HOLD or None side without marking failed."""
        intent = _make_ready_intent(store, side="HOLD")

        result = await engine.execute_ready_intents()
        assert result == 1

        # Intent should still be in ready_for_exec (skipped, not failed)
        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "ready_for_exec"

        # MatchTrader should NOT be called
        mock_matchtrader.open_position.assert_not_called()

    async def test_one_failure_does_not_block_others(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """If one intent fails, others should still be processed."""
        intent1 = _make_ready_intent(store, symbol="EURUSD")
        intent2 = _make_ready_intent(store, symbol="GBPUSD")

        # First call fails, second succeeds
        call_count = 0

        async def _side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Network error")
            return MagicMock(success=True, position_id="pos_456", message="OK")

        mock_matchtrader.open_position.side_effect = _side_effect

        result = await engine.execute_ready_intents()
        assert result == 2

        assert store.get_intent(intent1.id).status == "failed"
        assert store.get_intent(intent2.id).status == "opened"


# ── Position Sizing Tests ───────────────────────────────────────────────────


class TestPositionSizing:
    """Tests for trade plan building and position sizing."""

    async def test_uses_intent_sl_tp_when_available(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_sizer: MagicMock,
    ) -> None:
        """Should use intent's SL/TP pips when set by LLM."""
        _make_ready_intent(store, sl_pips=60.0, tp_pips=120.0)
        await engine.execute_ready_intents()

        # Verify sizer received the intent's SL pips
        mock_sizer.calculate_volume.assert_called_once_with(
            "EURUSD",
            50000.0,
            60.0,
            risk_pct_override=0.02,
        )
        mock_sizer.calculate_risk_amount.assert_called_once_with("EURUSD", 0.10, 60.0)

    async def test_falls_back_to_default_sl_tp(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_sizer: MagicMock,
    ) -> None:
        """Should fall back to DEFAULT_SL_TP when intent has no SL/TP."""
        _make_ready_intent(store, sl_pips=0.0, tp_pips=0.0)
        await engine.execute_ready_intents()

        # DEFAULT_SL_TP for EURUSD is sl_pips=40, tp_pips=80
        mock_sizer.calculate_volume.assert_called_once_with(
            "EURUSD",
            50000.0,
            40,
            risk_pct_override=0.02,
        )
        mock_sizer.calculate_risk_amount.assert_called_once_with("EURUSD", 0.10, 40)

    async def test_high_confidence_sparse_portfolio_uses_uplifted_risk_pct(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_sizer: MagicMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Sparse portfolios should pass an uplifted risk override to the sizer."""
        mock_matchtrader.get_open_positions.return_value = []

        _make_ready_intent(store, scanner_confidence="high")
        await engine.execute_ready_intents()

        mock_sizer.calculate_volume.assert_called_once_with(
            "EURUSD",
            50000.0,
            40.0,
            risk_pct_override=0.02,
        )

    async def test_low_confidence_keeps_default_risk_pct(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_sizer: MagicMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Low-confidence signals should keep the default risk sizing path."""
        mock_matchtrader.get_open_positions.return_value = []

        _make_ready_intent(store, scanner_confidence="low")
        await engine.execute_ready_intents()

        mock_sizer.calculate_volume.assert_called_once_with(
            "EURUSD",
            50000.0,
            40.0,
            risk_pct_override=0.01,
        )

    async def test_jpy_cross_uses_live_usdjpy_pip_value_override(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_sizer: MagicMock,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """JPY crosses should resolve a live pip-value override before sizing."""
        from src.execution.matchtrader_client import QuoteInfo

        engine._config.instruments["EURJPY"] = InstrumentConfig(pip_value=6.67, pip_size=0.01)
        mock_matchtrader.get_quote.side_effect = [
            QuoteInfo(symbol="USDJPY", bid=149.9, ask=150.1),
            QuoteInfo(symbol="EURJPY", bid=160.0, ask=160.02),
        ]
        mock_matchtrader.open_position.return_value = MagicMock(
            success=True,
            position_id="pos_jpy_1",
            message="OK",
            raw_response={"openPrice": "160.010"},
        )
        mock_matchtrader.modify_position = AsyncMock(
            return_value=MagicMock(success=True, position_id="pos_jpy_1", message="OK")
        )

        _make_ready_intent(store, symbol="EURJPY", side="BUY", sl_pips=40.0, tp_pips=80.0)
        await engine.execute_ready_intents()

        volume_call = mock_sizer.calculate_volume.call_args
        assert volume_call.args == ("EURJPY", 50000.0, 40.0)
        assert volume_call.kwargs["risk_pct_override"] == 0.02
        assert volume_call.kwargs["pip_value_override"] == pytest.approx(6.6667, abs=0.0001)

        risk_call = mock_sizer.calculate_risk_amount.call_args
        assert risk_call.args == ("EURJPY", 0.10, 40.0)
        assert risk_call.kwargs["pip_value_override"] == pytest.approx(6.6667, abs=0.0001)


class TestPortfolioRiskExecutionGate:
    """Tests for execution-side portfolio risk guard integration."""

    async def test_execute_ready_intents_rejects_when_portfolio_risk_guard_blocks(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Execution should reject when projected total open risk exceeds budget."""
        _make_opened_intent_with_execution_risk(
            store,
            symbol="GBPUSD",
            side="BUY",
            position_id="pos_existing_1",
            risk_pct=0.02,
        )
        ready = _make_ready_intent(store, symbol="EURUSD", side="BUY", scanner_confidence="high")

        mock_matchtrader.get_open_positions.return_value = [
            MagicMock(position_id="pos_existing_1", symbol="GBPUSD", side="BUY", profit=10.0),
        ]

        result = await engine.execute_ready_intents()
        assert result == 1

        updated = store.get_intent(ready.id)
        assert updated is not None
        assert updated.status == "rejected"
        assert updated.execution_error == "portfolio_risk.total_open_risk_exceeded"
        mock_matchtrader.open_position.assert_not_called()

    async def test_portfolio_risk_rejection_reason_and_meta_are_deterministic(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Rejection reason should be stable and metadata should include guard payload."""
        _make_opened_intent_with_execution_risk(
            store,
            symbol="GBPUSD",
            side="BUY",
            position_id="pos_existing_1",
            risk_pct=0.02,
        )
        ready = _make_ready_intent(store, symbol="EURUSD", side="BUY", scanner_confidence="high")
        mock_matchtrader.get_open_positions.return_value = [
            MagicMock(position_id="pos_existing_1", symbol="GBPUSD", side="BUY", profit=10.0),
        ]

        await engine.execute_ready_intents()

        updated = store.get_intent(ready.id)
        assert updated is not None
        assert updated.execution_error == "portfolio_risk.total_open_risk_exceeded"
        meta = _get_execution_meta(store, ready.id)
        assert meta["compliance_passed"] is False
        assert meta["portfolio_risk"]["allowed"] is False
        assert meta["portfolio_risk"]["reason_code"] == updated.execution_error

    async def test_open_risk_within_budget_allows_bounded_uplift_execution(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
        mock_sizer: MagicMock,
    ) -> None:
        """High-confidence entries should still execute when portfolio risk stays within budget."""
        _make_opened_intent_with_execution_risk(
            store,
            symbol="GBPUSD",
            side="SELL",
            position_id="pos_existing_1",
            risk_pct=0.005,
        )
        ready = _make_ready_intent(store, symbol="EURUSD", side="BUY", scanner_confidence="high")
        mock_matchtrader.get_open_positions.return_value = [
            MagicMock(position_id="pos_existing_1", symbol="GBPUSD", side="SELL", profit=5.0),
        ]

        result = await engine.execute_ready_intents()
        assert result == 1

        updated = store.get_intent(ready.id)
        assert updated is not None
        assert updated.status == "opened"
        mock_matchtrader.open_position.assert_called_once()
        mock_sizer.calculate_volume.assert_called_once_with(
            "EURUSD",
            50000.0,
            40.0,
            risk_pct_override=pytest.approx(0.015),
        )

        meta = _get_execution_meta(store, ready.id)
        assert meta["portfolio_risk"]["allowed"] is True
        assert meta["portfolio_risk"]["reason_code"] == "portfolio_risk.allowed"


# ── Account Snapshot Tests ──────────────────────────────────────────────────


class TestAccountSnapshot:
    """Tests for account snapshot construction."""

    async def test_snapshot_uses_config_initial_balance(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_guard: MagicMock,
    ) -> None:
        """Should use config's initial_balance in AccountSnapshot."""
        _make_ready_intent(store)
        await engine.execute_ready_intents()

        snapshot = mock_guard.check_all.call_args[0][1]
        assert isinstance(snapshot, AccountSnapshot)
        assert snapshot.initial_balance == 50000.0

    async def test_snapshot_includes_open_positions(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
        mock_guard: MagicMock,
    ) -> None:
        """Should count open positions in AccountSnapshot."""
        mock_matchtrader.get_open_positions.return_value = [
            MagicMock(profit=50.0),
            MagicMock(profit=-20.0),
        ]

        _make_ready_intent(store)
        await engine.execute_ready_intents()

        snapshot = mock_guard.check_all.call_args[0][1]
        assert snapshot.open_positions == 2
        assert snapshot.daily_pnl == 30.0  # 50 + (-20)

    async def test_snapshot_day_start_balance(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
        mock_guard: MagicMock,
    ) -> None:
        """Should estimate day_start_balance from balance minus realized PnL only."""
        mock_matchtrader.get_balance.return_value = MagicMock(
            balance=50100.0,
            equity=50100.0,
            margin=0.0,
            free_margin=50100.0,
        )
        mock_matchtrader.get_open_positions.return_value = [
            MagicMock(profit=100.0),
        ]

        _make_ready_intent(store)
        await engine.execute_ready_intents()

        snapshot = mock_guard.check_all.call_args[0][1]
        assert snapshot.day_start_balance == 50100.0  # no realized trades yet

    async def test_snapshot_uses_realized_plus_unrealized_pnl(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
        mock_guard: MagicMock,
    ) -> None:
        """daily_pnl should include realized closed trades + current unrealized PnL."""
        _make_closed_intent_today(store, symbol="GBPUSD", realized_pnl=150.0)
        _make_ready_intent(store)

        mock_matchtrader.get_balance.return_value = MagicMock(
            balance=50150.0,
            equity=50130.0,
            margin=0.0,
            free_margin=50130.0,
        )
        mock_matchtrader.get_open_positions.return_value = [MagicMock(profit=-20.0)]

        await engine.execute_ready_intents()

        snapshot = mock_guard.check_all.call_args[0][1]
        assert snapshot.daily_pnl == 130.0
        assert snapshot.day_start_balance == 50000.0


class TestBestDayExecutionGate:
    """Tests for execution-layer hard gate when Best Day protection is active."""

    async def test_best_day_allows_ready_intent_when_daily_pnl_is_zero(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
        mock_guard: MagicMock,
    ) -> None:
        """Execution should not reject a new entry when actual daily PnL is still zero."""
        ready = _make_ready_intent(store, tp_pips=5000.0)

        await engine.execute_ready_intents()

        updated = store.get_intent(ready.id)
        assert updated is not None
        assert updated.status == "opened"
        assert updated.execution_error is None
        mock_guard.check_all.assert_called_once()
        mock_matchtrader.open_position.assert_called_once()

    async def test_rejects_ready_intent_when_best_day_gate_active(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        config: AppConfig,
        mock_matchtrader: AsyncMock,
        mock_guard: MagicMock,
    ) -> None:
        """Execution should reject new entry before compliance check if gate is active."""
        pause_threshold = config.compliance.best_day_limit * config.compliance.best_day_stop * 0.90
        realized_pnl = pause_threshold + 5.0
        _make_closed_intent_today(store, symbol="GBPUSD", realized_pnl=realized_pnl)
        ready = _make_ready_intent(store)

        mock_matchtrader.get_balance.return_value = MagicMock(
            balance=50000.0 + realized_pnl,
            equity=50000.0 + realized_pnl,
            margin=0.0,
            free_margin=50000.0 + realized_pnl,
        )
        mock_matchtrader.get_open_positions.return_value = []

        await engine.execute_ready_intents()

        updated = store.get_intent(ready.id)
        assert updated is not None
        assert updated.status == "rejected"
        assert updated.execution_error is not None
        assert "Best Day entry gate" in updated.execution_error
        mock_guard.check_all.assert_not_called()
        mock_matchtrader.open_position.assert_not_called()


# ── Random Delay Tests ──────────────────────────────────────────────────────


class TestRandomDelay:
    """Tests for anti-duplicate-strategy delay."""

    async def test_random_delay_called(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_guard: MagicMock,
    ) -> None:
        """Should call guard.add_random_delay() for each executed intent."""
        _make_ready_intent(store)
        await engine.execute_ready_intents()

        mock_guard.add_random_delay.assert_called_once()

    async def test_random_delay_not_called_on_rejection(
        self,
        engine: ExecutionEngine,
        store: DecisionStore,
        mock_guard: MagicMock,
    ) -> None:
        """Should NOT apply delay when compliance rejects the trade."""
        mock_guard.check_all.return_value = ComplianceResult(
            passed=False, rule_name="MAX_DRAWDOWN", reason="Exceeded"
        )

        _make_ready_intent(store)
        await engine.execute_ready_intents()

        mock_guard.add_random_delay.assert_not_called()


# ── Serialization Tests ─────────────────────────────────────────────────────


class TestSerializeCompliance:
    """Tests for compliance snapshot serialization."""

    def test_serialize_compliance_result(self) -> None:
        """Should produce valid JSON with all required fields."""
        result = ComplianceResult(
            passed=True,
            rule_name="ALL",
            reason="All checks passed",
            details={"margin_used": 500.0},
        )
        snapshot = AccountSnapshot(
            balance=50000.0,
            equity=49500.0,
            margin=500.0,
            free_margin=49000.0,
            day_start_balance=50000.0,
            initial_balance=50000.0,
            open_positions=1,
            daily_pnl=-500.0,
            total_pnl=0.0,
        )

        json_str = ExecutionEngine._serialize_compliance(result, snapshot)
        data = json.loads(json_str)

        assert data["passed"] is True
        assert data["rule_name"] == "ALL"
        assert data["account"]["balance"] == 50000.0
        assert data["account"]["equity"] == 49500.0
        assert data["account"]["open_positions"] == 1
        assert data["details"]["margin_used"] == 500.0


# ── Instrument Registry Tests ──────────────────────────────────────────────


class TestInstrumentRegistry:
    """Tests for InstrumentRegistry integration in ExecutionEngine."""

    async def test_uses_registry_broker_symbol(
        self,
        store: DecisionStore,
        config: AppConfig,
        mock_guard: MagicMock,
        mock_matchtrader: AsyncMock,
        mock_sizer: MagicMock,
    ) -> None:
        """Should use registry to convert symbol to broker format before API call."""
        # Create a mock registry that maps EURUSD → EURUSD.
        mock_registry = MagicMock()
        mock_registry.to_broker.return_value = "EURUSD."

        # Create engine WITH the registry
        engine = ExecutionEngine(
            store=store,
            guard=mock_guard,
            matchtrader=mock_matchtrader,
            sizer=mock_sizer,
            config=config,
            instrument_registry=mock_registry,
        )

        # Execute a ready intent for EURUSD
        _make_ready_intent(store, symbol="EURUSD")
        await engine.execute_ready_intents()

        # Verify MatchTrader was called with the broker symbol (EURUSD.)
        mock_matchtrader.open_position.assert_called_once_with(
            symbol="EURUSD.",
            side="BUY",
            volume=0.10,
        )

        # Verify registry.to_broker() was called with the config symbol
        mock_registry.to_broker.assert_called_once_with("EURUSD")

    async def test_falls_back_without_registry(
        self,
        store: DecisionStore,
        config: AppConfig,
        mock_guard: MagicMock,
        mock_matchtrader: AsyncMock,
        mock_sizer: MagicMock,
    ) -> None:
        """Should use config symbol as-is when no registry is set."""
        # Create engine WITHOUT registry (None)
        engine = ExecutionEngine(
            store=store,
            guard=mock_guard,
            matchtrader=mock_matchtrader,
            sizer=mock_sizer,
            config=config,
            instrument_registry=None,
        )

        # Execute a ready intent for EURUSD
        _make_ready_intent(store, symbol="EURUSD")
        await engine.execute_ready_intents()

        # Verify MatchTrader was called with the config symbol unchanged
        mock_matchtrader.open_position.assert_called_once_with(
            symbol="EURUSD",
            side="BUY",
            volume=0.10,
        )

    async def test_registry_key_error_uses_original(
        self,
        store: DecisionStore,
        config: AppConfig,
        mock_guard: MagicMock,
        mock_matchtrader: AsyncMock,
        mock_sizer: MagicMock,
    ) -> None:
        """Should fall back to config symbol when registry raises KeyError."""
        # Create a mock registry that raises KeyError for unknown symbols
        mock_registry = MagicMock()
        mock_registry.to_broker.side_effect = KeyError("Symbol 'UNKNOWN' not in registry")

        # Update config to include UNKNOWN instrument for testing
        config.instruments["UNKNOWN"] = InstrumentConfig(pip_value=10.0, pip_size=0.0001)

        # Create engine WITH the registry that will raise
        engine = ExecutionEngine(
            store=store,
            guard=mock_guard,
            matchtrader=mock_matchtrader,
            sizer=mock_sizer,
            config=config,
            instrument_registry=mock_registry,
        )

        # Execute a ready intent for UNKNOWN
        _make_ready_intent(store, symbol="UNKNOWN")
        await engine.execute_ready_intents()

        # Verify MatchTrader was called with the original config symbol (fallback)
        mock_matchtrader.open_position.assert_called_once_with(
            symbol="UNKNOWN",
            side="BUY",
            volume=0.10,
        )

        # Verify registry.to_broker() was called (and raised)
        mock_registry.to_broker.assert_called_once_with("UNKNOWN")


# ── SL/TP Price Conversion Tests ─────────────────────────────────────────────


class TestSLTPPriceConversion:
    """Tests for SL/TP price conversion functionality."""

    def test_extract_open_price_from_open_price_key(self) -> None:
        """Should return float from from openPrice key."""
        raw = {"openPrice": "1.10000"}
        result = ExecutionEngine._extract_open_price(raw)
        assert result == 1.10000

    def test_extract_open_price_from_open_price(self) -> None:
        """Should return float from open_price_price key."""
        raw = {"open_price": 1.10000}
        result = ExecutionEngine._extract_open_price(raw)
        assert result == 1.10000

    def test_extract_open_price_from_price(self) -> None:
        """Should return float from price key."""
        raw = {"price": "1.10000"}
        result = ExecutionEngine._extract_open_price(raw)
        assert result == 1.10000

    def test_extract_open_price_from_fill_price_key(self) -> None:
        """Should return float from fillPrice key."""
        raw = {"fillPrice": 1.10000}
        result = ExecutionEngine._extract_open_price(raw)
        assert result == 1.10000

    def test_extract_open_price_from_open(self) -> None:
        """Should return float from open key."""
        raw = {"open": "1.10000"}
        result = ExecutionEngine._extract_open_price(raw)
        assert result == 1.10000

    def test_extract_open_price_empty_dict(self) -> None:
        """Should return None for empty dict."""
        result = ExecutionEngine._extract_open_price({})
        assert result is None

    def test_extract_open_price_non_numeric(self) -> None:
        """Should return None for non-numeric values."""
        raw = {"openPrice": "invalid"}
        result = ExecutionEngine._extract_open_price(raw)
        assert result is None

    def test_extract_open_price_prioritizes_keys(self) -> None:
        """Should prioritize keys in order (openPrice first)."""
        raw = {
            "openPrice": "1.10000",
            "open_price": "1.20000",
            "price": "1.30000",
        }
        result = ExecutionEngine._extract_open_price(raw)
        assert result == 1.10000

    async def test_fetch_position_open_price_found(self, engine: ExecutionEngine) -> None:
        """Should return price when position found."""
        mock_position = MagicMock()
        mock_position.position_id = "pos_123"
        mock_position.open_price = 1.10000
        engine._matchtrader.get_open_positions.return_value = [mock_position]

        result = await engine._fetch_position_open_price("pos_123")
        assert result == 1.10000

    async def test_fetch_position_open_price_not_found(self, engine: ExecutionEngine) -> None:
        """Should return None when position not found."""
        mock_position = MagicMock()
        mock_position.position_id = "pos_456"
        mock_position.open_price = 1.10000
        engine._matchtrader.get_open_positions.return_value = [mock_position]

        result = await engine._fetch_position_open_price("pos_123")
        assert result is None

    async def test_fetch_position_open_price_api_error(self, engine: ExecutionEngine) -> None:
        """Should return None on API error."""
        engine._matchtrader.get_open_positions.side_effect = RuntimeError("API error")

        result = await engine._fetch_position_open_price("pos_123")
        assert result is None

    async def test_set_sl_tp_on_position_buy(self, engine: ExecutionEngine) -> None:
        """BUY side: sl = open_price - sl_pips * pip_size, tp = open_price + tp_pips * pip_size."""
        raw_response = {"openPrice": "1.10000"}
        engine._matchtrader.modify_position.return_value = MagicMock(
            success=True, position_id="pos_123", message="OK"
        )

        sl_price, tp_price = await engine._set_sl_tp_on_position(
            position_id="pos_123",
            broker_symbol="EURUSD",
            config_symbol="EURUSD",
            side="BUY",
            volume=0.10,
            sl_pips=40.0,
            tp_pips=80.0,
            raw_response=raw_response,
        )

        # BUY: sl = 1.10000 - 40 * 0.0001 = 1.09600, tp = 1.10000 + 80 * 0.0001 = 1.10800
        assert sl_price == 1.09600
        assert tp_price == 1.10800

        engine._matchtrader.modify_position.assert_called_once_with(
            position_id="pos_123", symbol="EURUSD", side="BUY", volume=0.10, sl=1.09600, tp=1.10800
        )

    async def test_set_sl_tp_on_position_sell(self, engine: ExecutionEngine) -> None:
        """SELL side: sl = open_price + sl_pips * pip_size, tp = open_price - tp_pips * pip_size."""
        raw_response = {"openPrice": "1.10000"}
        engine._matchtrader.modify_position.return_value = MagicMock(
            success=True, position_id="pos_123", message="OK"
        )

        sl_price, tp_price = await engine._set_sl_tp_on_position(
            position_id="pos_123",
            broker_symbol="EURUSD",
            config_symbol="EURUSD",
            side="SELL",
            volume=0.10,
            sl_pips=40.0,
            tp_pips=80.0,
            raw_response=raw_response,
        )

        # SELL: sl = 1.10000 + 40 * 0.0001 = 1.10400, tp = 1.10000 - 80 * 0.0001 = 1.09200
        assert sl_price == 1.10400
        assert tp_price == 1.09200

        engine._matchtrader.modify_position.assert_called_once_with(
            position_id="pos_123", symbol="EURUSD", side="SELL", volume=0.10, sl=1.10400, tp=1.09200
        )

    async def test_set_sl_tp_on_position_no_price(self, engine: ExecutionEngine) -> None:
        """Returns (None, None) when open_price cannot be determined."""
        raw_response = {}
        engine._matchtrader.get_open_positions.return_value = []

        sl_price, tp_price = await engine._set_sl_tp_on_position(
            position_id="pos_123",
            broker_symbol="EURUSD",
            config_symbol="EURUSD",
            side="BUY",
            volume=0.10,
            sl_pips=40.0,
            tp_pips=80.0,
            raw_response=raw_response,
        )

        assert sl_price is None
        assert tp_price is None
        engine._matchtrader.modify_position.assert_not_called()

    async def test_set_sl_tp_on_position_invalid_price(self, engine: ExecutionEngine) -> None:
        """Returns (None, None) when open_price is invalid (zero or negative)."""
        raw_response = {"openPrice": "0.0"}

        sl_price, tp_price = await engine._set_sl_tp_on_position(
            position_id="pos_123",
            broker_symbol="EURUSD",
            config_symbol="EURUSD",
            side="BUY",
            sl_pips=40.0,
            volume=0.10,
            tp_pips=80.0,
            raw_response=raw_response,
        )

        assert sl_price is None
        assert tp_price is None
        engine._matchtrader.modify_position.assert_not_called()

    async def test_set_sl_tp_on_position_unknown_instrument(self, engine: ExecutionEngine) -> None:
        """Returns (None, None) when config_symbol not in instruments."""
        raw_response = {"openPrice": "1.10000"}

        sl_price, tp_price = await engine._set_sl_tp_on_position(
            position_id="pos_123",
            broker_symbol="UNKNOWN",
            config_symbol="UNKNOWN",
            side="BUY",
            sl_pips=40.0,
            volume=0.10,
            tp_pips=80.0,
            raw_response=raw_response,
        )

        assert sl_price is None
        assert tp_price is None
        engine._matchtrader.modify_position.assert_not_called()

    async def test_set_sl_tp_on_position_modify_fails(self, engine: ExecutionEngine) -> None:
        """Returns (None, None) when modify_position fails."""
        raw_response = {"openPrice": "1.10000"}
        engine._matchtrader.modify_position.return_value = MagicMock(
            success=False, position_id="", message="Failed"
        )

        sl_price, tp_price = await engine._set_sl_tp_on_position(
            position_id="pos_123",
            broker_symbol="EURUSD",
            config_symbol="EURUSD",
            side="BUY",
            sl_pips=40.0,
            volume=0.10,
            tp_pips=80.0,
            raw_response=raw_response,
        )

        assert sl_price is None
        assert tp_price is None

    async def test_set_sl_tp_on_position_modify_error(self, engine: ExecutionEngine) -> None:
        """Returns (None, None) when modify_position raises."""
        raw_response = {"openPrice": "1.10000"}
        engine._matchtrader.modify_position.side_effect = RuntimeError("Network error")

        sl_price, tp_price = await engine._set_sl_tp_on_position(
            position_id="pos_123",
            broker_symbol="EURUSD",
            config_symbol="EURUSD",
            side="BUY",
            sl_pips=40.0,
            volume=0.10,
            tp_pips=80.0,
            raw_response=raw_response,
        )

        assert sl_price is None
        assert tp_price is None

    async def test_set_sl_tp_on_position_with_registry(
        self,
        store: DecisionStore,
        config: AppConfig,
        mock_guard: MagicMock,
        mock_matchtrader: AsyncMock,
        mock_sizer: MagicMock,
    ) -> None:
        """Rounds to price_precision from registry."""
        # Mock registry with price_precision=3
        mock_registry = MagicMock()
        mock_info = MagicMock()
        mock_info.price_precision = 3
        mock_registry.get_info.return_value = mock_info

        engine = ExecutionEngine(
            store=store,
            guard=mock_guard,
            matchtrader=mock_matchtrader,
            sizer=mock_sizer,
            config=config,
            instrument_registry=mock_registry,
        )

        raw_response = {"openPrice": "1.100100"}
        mock_matchtrader.modify_position.return_value = MagicMock(
            success=True, position_id="pos_123", message="OK"
        )

        sl_price, tp_price = await engine._set_sl_tp_on_position(
            position_id="pos_123",
            broker_symbol="EURUSD",
            config_symbol="EURUSD",
            side="BUY",
            volume=0.10,
            sl_pips=40.0,
            tp_pips=80.0,
            raw_response=raw_response,
        )

        # With precision=3: sl=1.100100 - 0.00400 = 1.096100, tp=1.100100 + 0.00800 = 1.108100
        # Rounded to 3 decimals: sl=1.096, tp=1.108
        assert sl_price == 1.096
        assert tp_price == 1.108

        mock_registry.get_info.assert_called_once_with("EURUSD")

    async def test_set_sl_tp_on_position_fallback_price_no_registry(
        self, engine: ExecutionEngine
    ) -> None:
        """Uses fallback _fetch_position_open_price without registry."""
        # First, raw_response has no price
        raw_response = {}

        # Mock get_open_positions to return a position with open_price
        mock_position = MagicMock()
        mock_position.position_id = "pos_123"
        mock_position.open_price = 1.10000
        engine._matchtrader.get_open_positions.return_value = [mock_position]
        engine._matchtrader.modify_position.return_value = MagicMock(
            success=True, position_id="pos_123", message="OK"
        )

        sl_price, tp_price = await engine._set_sl_tp_on_position(
            position_id="pos_123",
            broker_symbol="EURUSD",
            config_symbol="EURUSD",
            side="BUY",
            volume=0.10,
            sl_pips=40.0,
            tp_pips=80.0,
            raw_response=raw_response,
        )

        assert sl_price == 1.09600
        assert tp_price == 1.10800
        engine._matchtrader.get_open_positions.assert_called_once()

    async def test_set_sl_tp_on_position_no_registry_default_precision(
        self, engine: ExecutionEngine
    ) -> None:
        """Falls back to precision=5 when no registry."""
        raw_response = {"openPrice": "1.10000"}
        engine._matchtrader.modify_position.return_value = MagicMock(
            success=True, position_id="pos_123", message="OK"
        )

        sl_price, tp_price = await engine._set_sl_tp_on_position(
            position_id="pos_123",
            broker_symbol="EURUSD",
            config_symbol="EURUSD",
            side="BUY",
            volume=0.10,
            sl_pips=40.0,
            tp_pips=80.0,
            raw_response=raw_response,
        )

        # Default precision=5
        assert sl_price == 1.09600
        assert tp_price == 1.10800

    async def test_integration_sl_tp_set_on_execution(
        self,
        store: DecisionStore,
        config: AppConfig,
        mock_guard: MagicMock,
        mock_matchtrader: AsyncMock,
        mock_sizer: MagicMock,
    ) -> None:
        """Successful execution calls modify_position with correct SL/TP prices."""
        # Setup mock_matchtrader.open_position with raw_response
        mock_matchtrader.open_position.return_value = MagicMock(
            success=True,
            position_id="pos_123",
            message="OK",
            raw_response={"openPrice": 1.10000},
        )
        mock_matchtrader.modify_position = AsyncMock(
            return_value=MagicMock(success=True, position_id="pos_123", message="OK")
        )

        engine = ExecutionEngine(
            store=store,
            guard=mock_guard,
            matchtrader=mock_matchtrader,
            sizer=mock_sizer,
            config=config,
        )

        _make_ready_intent(store, sl_pips=40.0, tp_pips=80.0)
        await engine.execute_ready_intents()

        # Verify modify_position was called with correct SL/TP
        mock_matchtrader.modify_position.assert_called_once_with(
            position_id="pos_123", symbol="EURUSD", side="BUY", volume=0.10, sl=1.09600, tp=1.10800
        )


# ── Slippage Detection Tests ────────────────────────────────────────────────


class TestSlippageDetection:
    """Tests for pre-trade quote validation and post-trade slippage alerting."""

    async def test_slippage_ok_no_alert(
        self,
        store: DecisionStore,
        config: AppConfig,
        mock_guard: MagicMock,
        mock_matchtrader: AsyncMock,
        mock_sizer: MagicMock,
    ) -> None:
        """No alert when fill price is within slippage tolerance."""
        from src.execution.matchtrader_client import QuoteInfo

        mock_matchtrader.get_quote.return_value = QuoteInfo(
            symbol="EURUSD.", bid=1.10873, ask=1.10877
        )
        mock_matchtrader.open_position.return_value = MagicMock(
            success=True,
            position_id="pos_123",
            message="OK",
            raw_response={"openPrice": "1.10878"},
        )
        mock_matchtrader.modify_position = AsyncMock(
            return_value=MagicMock(success=True, position_id="pos_123", message="OK")
        )

        engine = ExecutionEngine(
            store=store,
            guard=mock_guard,
            matchtrader=mock_matchtrader,
            sizer=mock_sizer,
            config=config,
        )
        _make_ready_intent(store, symbol="EURUSD", side="BUY")
        await engine.execute_ready_intents()

        for call in mock_matchtrader.method_calls:
            pass

    async def test_slippage_alert_when_exceeded(
        self,
        store: DecisionStore,
        config: AppConfig,
        mock_guard: MagicMock,
        mock_matchtrader: AsyncMock,
        mock_sizer: MagicMock,
    ) -> None:
        """Should log warning when fill price exceeds max slippage."""
        from src.execution.matchtrader_client import QuoteInfo

        config.execution.max_slippage_pips = 0.5

        mock_matchtrader.get_quote.return_value = QuoteInfo(
            symbol="EURUSD.", bid=1.10873, ask=1.10877
        )
        mock_matchtrader.open_position.return_value = MagicMock(
            success=True,
            position_id="pos_456",
            message="OK",
            raw_response={"openPrice": "1.10907"},
        )
        mock_matchtrader.modify_position = AsyncMock(
            return_value=MagicMock(success=True, position_id="pos_456", message="OK")
        )

        alert_service = AsyncMock()
        engine = ExecutionEngine(
            store=store,
            guard=mock_guard,
            matchtrader=mock_matchtrader,
            sizer=mock_sizer,
            config=config,
            alert_service=alert_service,
        )
        _make_ready_intent(store, symbol="EURUSD", side="BUY")
        await engine.execute_ready_intents()

        alert_service.system_error.assert_called_once()
        call_args = alert_service.system_error.call_args[0][0]
        assert "Slippage alert" in call_args

    async def test_quote_fetch_failure_proceeds(
        self,
        store: DecisionStore,
        config: AppConfig,
        mock_guard: MagicMock,
        mock_matchtrader: AsyncMock,
        mock_sizer: MagicMock,
    ) -> None:
        """Should proceed with trade when quote fetch fails."""
        mock_matchtrader.get_quote.side_effect = Exception("Network error")
        mock_matchtrader.open_position.return_value = MagicMock(
            success=True,
            position_id="pos_789",
            message="OK",
            raw_response={"openPrice": "1.10900"},
        )
        mock_matchtrader.modify_position = AsyncMock(
            return_value=MagicMock(success=True, position_id="pos_789", message="OK")
        )

        engine = ExecutionEngine(
            store=store,
            guard=mock_guard,
            matchtrader=mock_matchtrader,
            sizer=mock_sizer,
            config=config,
        )
        _make_ready_intent(store, symbol="EURUSD", side="BUY")
        result = await engine.execute_ready_intents()
        assert result == 1

        mock_matchtrader.open_position.assert_called_once()
        intents = store.get_ready_intents()
        assert len(intents) == 0

    async def test_sell_uses_bid_price(
        self,
        store: DecisionStore,
        config: AppConfig,
        mock_guard: MagicMock,
        mock_matchtrader: AsyncMock,
        mock_sizer: MagicMock,
    ) -> None:
        """For SELL orders, should use bid price as reference."""
        from src.execution.matchtrader_client import QuoteInfo

        config.execution.max_slippage_pips = 0.5

        mock_matchtrader.get_quote.return_value = QuoteInfo(
            symbol="EURUSD.", bid=1.10873, ask=1.10877
        )
        mock_matchtrader.open_position.return_value = MagicMock(
            success=True,
            position_id="pos_sell_1",
            message="OK",
            raw_response={"openPrice": "1.10843"},
        )
        mock_matchtrader.modify_position = AsyncMock(
            return_value=MagicMock(success=True, position_id="pos_sell_1", message="OK")
        )

        alert_service = AsyncMock()
        engine = ExecutionEngine(
            store=store,
            guard=mock_guard,
            matchtrader=mock_matchtrader,
            sizer=mock_sizer,
            config=config,
            alert_service=alert_service,
        )
        _make_ready_intent(store, symbol="EURUSD", side="SELL")
        await engine.execute_ready_intents()

        alert_service.system_error.assert_called_once()
        call_args = alert_service.system_error.call_args[0][0]
        assert "Slippage alert" in call_args


# ── BUG #1: Trade Opened price=0.0 fix tests ──────────────────────────────


class TestTradeOpenedPriceAlert:
    """Tests that trade_opened alert receives real fill price, not hardcoded 0.0."""

    async def test_alert_receives_fill_price_from_raw_response(
        self,
        store: DecisionStore,
        config: AppConfig,
        mock_guard: MagicMock,
        mock_matchtrader: AsyncMock,
        mock_sizer: MagicMock,
    ) -> None:
        """Alert should receive the fill price extracted from raw_response."""
        mock_matchtrader.open_position.return_value = MagicMock(
            success=True,
            position_id="pos_price_1",
            message="OK",
            raw_response={"openPrice": "1.08765"},
        )
        mock_matchtrader.modify_position = AsyncMock(
            return_value=MagicMock(success=True, position_id="pos_price_1", message="OK")
        )

        alert_service = AsyncMock()
        engine = ExecutionEngine(
            store=store,
            guard=mock_guard,
            matchtrader=mock_matchtrader,
            sizer=mock_sizer,
            config=config,
            alert_service=alert_service,
        )
        _make_ready_intent(store, symbol="EURUSD", side="BUY")
        await engine.execute_ready_intents()

        alert_service.trade_opened.assert_called_once()
        call_kwargs = alert_service.trade_opened.call_args.kwargs
        assert call_kwargs["price"] == pytest.approx(1.08765)
        assert call_kwargs["symbol"] == "EURUSD"
        assert call_kwargs["side"] == "BUY"

    async def test_alert_falls_back_to_position_query(
        self,
        store: DecisionStore,
        config: AppConfig,
        mock_guard: MagicMock,
        mock_matchtrader: AsyncMock,
        mock_sizer: MagicMock,
    ) -> None:
        """When raw_response has no price, should fallback to broker position query."""
        mock_matchtrader.open_position.return_value = MagicMock(
            success=True,
            position_id="pos_fallback_1",
            message="OK",
            raw_response={},  # No price keys
        )
        mock_matchtrader.modify_position = AsyncMock(
            return_value=MagicMock(success=True, position_id="pos_fallback_1", message="OK")
        )
        # Fallback: get_open_positions returns a position with open_price
        mock_matchtrader.get_open_positions.return_value = [
            MagicMock(position_id="pos_fallback_1", open_price=1.09234),
        ]

        alert_service = AsyncMock()
        engine = ExecutionEngine(
            store=store,
            guard=mock_guard,
            matchtrader=mock_matchtrader,
            sizer=mock_sizer,
            config=config,
            alert_service=alert_service,
        )
        _make_ready_intent(store, symbol="EURUSD", side="SELL")
        await engine.execute_ready_intents()

        alert_service.trade_opened.assert_called_once()
        call_kwargs = alert_service.trade_opened.call_args.kwargs
        assert call_kwargs["price"] == pytest.approx(1.09234)

    async def test_alert_defaults_to_zero_when_no_price_available(
        self,
        store: DecisionStore,
        config: AppConfig,
        mock_guard: MagicMock,
        mock_matchtrader: AsyncMock,
        mock_sizer: MagicMock,
    ) -> None:
        """When neither raw_response nor broker has price, should default to 0.0."""
        mock_matchtrader.open_position.return_value = MagicMock(
            success=True,
            position_id="pos_noprice_1",
            message="OK",
            raw_response={},  # No price keys
        )
        mock_matchtrader.modify_position = AsyncMock(
            return_value=MagicMock(success=True, position_id="pos_noprice_1", message="OK")
        )
        # Fallback also returns no matching position
        mock_matchtrader.get_open_positions.return_value = []

        alert_service = AsyncMock()
        engine = ExecutionEngine(
            store=store,
            guard=mock_guard,
            matchtrader=mock_matchtrader,
            sizer=mock_sizer,
            config=config,
            alert_service=alert_service,
        )
        _make_ready_intent(store, symbol="EURUSD", side="BUY")
        await engine.execute_ready_intents()

        alert_service.trade_opened.assert_called_once()
        call_kwargs = alert_service.trade_opened.call_args.kwargs
        assert call_kwargs["price"] == 0.0

    async def test_alert_receives_price_with_fill_price_key(
        self,
        store: DecisionStore,
        config: AppConfig,
        mock_guard: MagicMock,
        mock_matchtrader: AsyncMock,
        mock_sizer: MagicMock,
    ) -> None:
        """Should extract price from alternative key 'fillPrice' in raw_response."""
        mock_matchtrader.open_position.return_value = MagicMock(
            success=True,
            position_id="pos_alt_key",
            message="OK",
            raw_response={"fillPrice": 1.12345},
        )
        mock_matchtrader.modify_position = AsyncMock(
            return_value=MagicMock(success=True, position_id="pos_alt_key", message="OK")
        )

        alert_service = AsyncMock()
        engine = ExecutionEngine(
            store=store,
            guard=mock_guard,
            matchtrader=mock_matchtrader,
            sizer=mock_sizer,
            config=config,
            alert_service=alert_service,
        )
        _make_ready_intent(store, symbol="GBPUSD", side="BUY")
        await engine.execute_ready_intents()

        alert_service.trade_opened.assert_called_once()
        call_kwargs = alert_service.trade_opened.call_args.kwargs
        assert call_kwargs["price"] == pytest.approx(1.12345)
