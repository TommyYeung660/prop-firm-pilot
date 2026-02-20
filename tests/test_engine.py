"""
Tests for src/execution/engine.py — ExecutionEngine trade execution pipeline.

Uses mocked MatchTraderClient and PropFirmGuard with a real DecisionStore
(in-memory SQLite). Tests cover the full execution pipeline: compliance
checking, position sizing, trade execution, and state transitions.
"""

import json
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
) -> TradeIntent:
    """Create and advance an intent to ready_for_exec state."""
    intent = TradeIntent(
        trade_date="2026-02-16",
        symbol=symbol,
        scanner_score=0.85,
        scanner_confidence="high",
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
        mock_sizer.calculate_volume.assert_called_once_with("EURUSD", 50000.0, 60.0)
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
        mock_sizer.calculate_volume.assert_called_once_with("EURUSD", 50000.0, 40)
        mock_sizer.calculate_risk_amount.assert_called_once_with("EURUSD", 0.10, 40)


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
        """Should estimate day_start_balance from balance minus daily PnL."""
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
        assert snapshot.day_start_balance == 50000.0  # 50100 - 100


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
