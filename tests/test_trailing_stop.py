"""Tests for trailing stop / breakeven logic (Phase 2.5) — _apply_breakeven_stops()."""



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
from src.decision_store.sqlite_store import DecisionStore
from src.scheduler.scheduler import Scheduler

# ── Helper Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    """Create minimal AppConfig for testing."""
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
            breakeven_activation_pct=0.5,
        ),
        decision_store=DecisionStoreConfig(),
        monitor=MonitorConfig(),
        instruments={"EURUSD": InstrumentConfig(pip_value=10.0, pip_size=0.0001)},
    )


@pytest.fixture
def store(tmp_path: Path) -> DecisionStore:
    """Create DecisionStore for testing."""
    return DecisionStore(str(tmp_path / "test.db"))


@pytest.fixture
def mock_matchtrader() -> AsyncMock:
    """Create mock MatchTrader client."""
    client = AsyncMock()
    client.modify_position.return_value = MagicMock(success=True, message="OK")
    client.get_balance.return_value = MagicMock(
        balance=50000.0, equity=50000.0, margin=0.0, free_margin=50000.0
    )
    return client


@pytest.fixture
def scheduler(config: AppConfig, store: DecisionStore, mock_matchtrader: AsyncMock) -> Scheduler:
    """Create Scheduler instance for testing."""
    return Scheduler(
        config=config,
        store=store,
        scanner=MagicMock(),
        agents=MagicMock(),
        engine=AsyncMock(),
        matchtrader=mock_matchtrader,
    )


def _make_position(
    position_id: str = "POS-1",
    symbol: str = "EURUSD.",
    side: str = "BUY",
    volume: float = 0.01,
    open_price: float = 1.10000,
    current_price: float = 1.10600,
    profit: float = 6.0,
    sl_price: float | None = 1.09000,
    tp_price: float | None = 1.11000,
) -> MagicMock:
    """Create mock PositionInfo for testing."""
    pos = MagicMock()
    pos.position_id = position_id
    pos.symbol = symbol
    pos.side = side
    pos.volume = volume
    pos.open_price = open_price
    pos.current_price = current_price
    pos.profit = profit
    pos.sl_price = sl_price
    pos.tp_price = tp_price
    return pos


def _insert_opened_intent_for_breakeven(
    store: DecisionStore,
    intent_id: str = "INT-1",
    symbol: str = "EURUSD.",
    position_id: str = "POS-1",
    tp_pips: float = 100.0,
) -> TradeIntent:
    """Insert an intent and advance it to 'opened' status with TP pips."""
    intent = TradeIntent(
        trade_date="2026-02-16",
        symbol=symbol,
        scanner_score=0.85,
        scanner_confidence="high",
    )
    intent.id = intent_id
    store.insert_intent(intent)
    store.claim_next_pending("test-worker")
    store.update_intent_decision(intent_id, "BUY", 30.0, tp_pips, "test", "{}")
    store.mark_ready_for_exec(intent_id)
    store.mark_executing(intent_id)
    store.mark_opened(intent_id, position_id)
    return store.get_intent(intent_id)


# ── Test Class: _apply_breakeven_stops ────────────────────────────────────


class TestApplyBreakevenStops:
    """Test Scheduler._apply_breakeven_stops() method."""

    # ── BUY Position Scenarios ───────────────────────────────────────

    async def test_buy_at_60_percent_tp_modifies_to_breakeven(
        self, scheduler: Scheduler, mock_matchtrader: AsyncMock, store: DecisionStore
    ) -> None:
        """BUY position at 60% of TP should modify SL to breakeven."""
        # Setup: BUY position at 1.10600 (60% of TP from 1.10000 to 1.11000)
        # tp_pips=100 → tp_distance=0.01 → threshold=0.005
        # profit_distance=1.10600-1.10000=0.006 ≥ 0.005 ✓ triggers
        pos = _make_position(
            side="BUY",
            open_price=1.10000,
            current_price=1.10600,
        )
        intent = _insert_opened_intent_for_breakeven(store, "INT-1", "EURUSD.", "POS-1", 100.0)

        # Execute
        await scheduler._apply_breakeven_stops([pos], [intent])

        # Verify modify_position called with breakeven SL
        mock_matchtrader.modify_position.assert_called_once()
        call_kwargs = mock_matchtrader.modify_position.call_args.kwargs
        assert call_kwargs["position_id"] == "POS-1"
        assert call_kwargs["symbol"] == "EURUSD."
        assert call_kwargs["side"] == "BUY"
        assert call_kwargs["volume"] == 0.01
        assert call_kwargs["sl"] == 1.10000  # breakeven (open_price)

        # Verify position added to _breakeven_applied
        assert "POS-1" in scheduler._breakeven_applied

    async def test_breakeven_passes_tp_price_when_present(
        self, scheduler: Scheduler, mock_matchtrader: AsyncMock, store: DecisionStore
    ) -> None:
        """Breakeven modify should preserve existing TP price when present."""
        pos = _make_position(
            side="BUY",
            open_price=1.10000,
            current_price=1.10600,
            tp_price=1.11000,
        )
        intent = _insert_opened_intent_for_breakeven(store, "INT-1", "EURUSD.", "POS-1", 100.0)

        await scheduler._apply_breakeven_stops([pos], [intent])

        call_kwargs = mock_matchtrader.modify_position.call_args.kwargs
        assert call_kwargs["tp"] == 1.11000

    async def test_buy_below_50_percent_threshold_no_modify(
        self, scheduler: Scheduler, mock_matchtrader: AsyncMock, store: DecisionStore
    ) -> None:
        """BUY position below 50% threshold should NOT modify."""
        # Setup: BUY position at 1.10300 (30% of TP from 1.10000 to 1.11000)
        # tp_pips=100 → tp_distance=0.01 → threshold=0.005
        # profit_distance=1.10300-1.10000=0.003 < 0.005 ✗ doesn't trigger
        pos = _make_position(
            side="BUY",
            open_price=1.10000,
            current_price=1.10300,
        )
        intent = _insert_opened_intent_for_breakeven(store, "INT-1", "EURUSD.", "POS-1", 100.0)

        # Execute
        await scheduler._apply_breakeven_stops([pos], [intent])

        # Verify NOT called
        assert mock_matchtrader.modify_position.call_count == 0
        assert "POS-1" not in scheduler._breakeven_applied

    # ── SELL Position Scenarios ──────────────────────────────────────

    async def test_sell_at_60_percent_tp_modifies_to_breakeven(
        self, scheduler: Scheduler, mock_matchtrader: AsyncMock, store: DecisionStore
    ) -> None:
        """SELL position at 60% of TP should modify SL to breakeven."""
        # Setup: SELL position at 1.09400 (60% of TP from 1.10000 to 1.09000)
        # tp_pips=100 → tp_distance=0.01 → threshold=0.005
        # profit_distance=|1.09400-1.10000|=0.006 ≥ 0.005 ✓ triggers
        pos = _make_position(
            side="SELL",
            open_price=1.10000,
            current_price=1.09400,
        )
        intent = _insert_opened_intent_for_breakeven(store, "INT-1", "EURUSD.", "POS-1", 100.0)

        # Execute
        await scheduler._apply_breakeven_stops([pos], [intent])

        # Verify modify_position called with breakeven SL
        assert mock_matchtrader.modify_position.call_count == 1
        call_kwargs = mock_matchtrader.modify_position.call_args.kwargs
        assert call_kwargs["position_id"] == "POS-1"
        assert call_kwargs["side"] == "SELL"
        assert call_kwargs["sl"] == 1.10000  # breakeven (open_price)

        # Verify position added to _breakeven_applied
        assert "POS-1" in scheduler._breakeven_applied

    # ── Edge Cases ───────────────────────────────────────────────────

    async def test_position_already_in_breakeven_applied_skips(
        self, scheduler: Scheduler, mock_matchtrader: AsyncMock, store: DecisionStore
    ) -> None:
        """Position already in _breakeven_applied should skip."""
        # Setup: Pre-populate _breakeven_applied
        scheduler._breakeven_applied.add("POS-1")

        pos = _make_position(
            side="BUY",
            open_price=1.10000,
            current_price=1.10600,
        )
        intent = _insert_opened_intent_for_breakeven(store, "INT-1", "EURUSD.", "POS-1", 100.0)

        # Execute
        await scheduler._apply_breakeven_stops([pos], [intent])

        # Verify NOT called (already processed)
        assert mock_matchtrader.modify_position.call_count == 0

    async def test_missing_suggested_tp_pips_skips(
        self, scheduler: Scheduler, mock_matchtrader: AsyncMock, store: DecisionStore
    ) -> None:
        """Intent with missing suggested_tp_pips should skip without error."""
        # Setup: Intent with tp_pips=None
        pos = _make_position(
            side="BUY",
            open_price=1.10000,
            current_price=1.10600,
        )
        intent = _insert_opened_intent_for_breakeven(store, "INT-1", "EURUSD.", "POS-1", 0.0)
        intent.suggested_tp_pips = None

        # Execute (should not crash)
        await scheduler._apply_breakeven_stops([pos], [intent])

        # Verify NOT called
        assert mock_matchtrader.modify_position.call_count == 0

    async def test_no_matching_intent_skips(
        self, scheduler: Scheduler, mock_matchtrader: AsyncMock, store: DecisionStore
    ) -> None:
        """Position with no matching intent in opened_intents should skip."""
        # Setup: Create intent for different position_id
        pos = _make_position(position_id="POS-1")
        intent = _insert_opened_intent_for_breakeven(store, "INT-2", "EURUSD.", "POS-2", 100.0)

        # Execute (should not crash)
        await scheduler._apply_breakeven_stops([pos], [intent])

        # Verify NOT called (no matching intent for POS-1)
        assert mock_matchtrader.modify_position.call_count == 0

    async def test_instrument_not_in_config_skips(
        self, scheduler: Scheduler, mock_matchtrader: AsyncMock, store: DecisionStore
    ) -> None:
        """Position with instrument not in config.instruments should skip."""
        # Setup: Position for GBPUSD (not in config)
        pos = _make_position(symbol="GBPUSD.", position_id="POS-1")
        intent = _insert_opened_intent_for_breakeven(store, "INT-1", "GBPUSD.", "POS-1", 100.0)

        # Execute (should not crash)
        await scheduler._apply_breakeven_stops([pos], [intent])

        # Verify NOT called (GBPUSD not in config.instruments)
        assert mock_matchtrader.modify_position.call_count == 0

    async def test_modify_position_failure_does_not_add_to_set(
        self, scheduler: Scheduler, mock_matchtrader: AsyncMock, store: DecisionStore
    ) -> None:
        """When modify_position fails, should NOT add to _breakeven_applied."""
        # Setup: Mock failure response
        mock_matchtrader.modify_position.return_value = MagicMock(
            success=False, message="API error"
        )

        pos = _make_position(
            side="BUY",
            open_price=1.10000,
            current_price=1.10600,
        )
        intent = _insert_opened_intent_for_breakeven(store, "INT-1", "EURUSD.", "POS-1", 100.0)

        # Execute
        await scheduler._apply_breakeven_stops([pos], [intent])

        # Verify modify_position was called
        assert mock_matchtrader.modify_position.call_count == 1

        # Verify NOT added to _breakeven_applied (so retry possible)
        assert "POS-1" not in scheduler._breakeven_applied

    # ── Side Case ───────────────────────────────────────────────────

    async def test_side_passed_as_is_from_position(
        self, scheduler: Scheduler, mock_matchtrader: AsyncMock, store: DecisionStore
    ) -> None:
        """Side should be passed as-is from position.side to modify_position."""
        # Setup: Test lowercase "buy"
        pos = _make_position(side="BUY", open_price=1.10000, current_price=1.10600)
        intent = _insert_opened_intent_for_breakeven(store, "INT-1", "EURUSD.", "POS-1", 100.0)

        # Execute
        await scheduler._apply_breakeven_stops([pos], [intent])

        # Verify modify_position called
        assert mock_matchtrader.modify_position.call_count == 1
        call_kwargs = mock_matchtrader.modify_position.call_args.kwargs
        assert call_kwargs["side"] == "BUY"  # passed as-is from pos.side

    # ── Multiple Positions ───────────────────────────────────────────

    async def test_multiple_positions_processed_independently(
        self, scheduler: Scheduler, mock_matchtrader: AsyncMock, store: DecisionStore
    ) -> None:
        """Multiple positions should be processed independently."""
        # Setup: Two positions
        pos1 = _make_position(
            position_id="POS-1",
            side="BUY",
            open_price=1.10000,
            current_price=1.10600,  # 60% - should modify
        )
        pos2 = _make_position(
            position_id="POS-2",
            side="BUY",
            open_price=1.20000,
            current_price=1.20300,  # 30% - should NOT modify
        )
        intent1 = _insert_opened_intent_for_breakeven(
            store, "INT-1", "EURUSD.", "POS-1", 100.0
        )
        intent2 = _insert_opened_intent_for_breakeven(
            store, "INT-2", "EURUSD.", "POS-2", 100.0
        )

        # Execute
        await scheduler._apply_breakeven_stops([pos1, pos2], [intent1, intent2])

        # Verify only on pos1 modified
        assert mock_matchtrader.modify_position.call_count == 1
        assert "POS-1" in scheduler._breakeven_applied
        assert "POS-2" not in scheduler._breakeven_applied
