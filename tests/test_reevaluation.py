"""Tests for HOLD re-evaluation logic (Phase 2.6) — _reevaluate_open_positions()."""


from datetime import datetime, timedelta, timezone
from pathlib import Path
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
from src.decision.agent_bridge import AgentDecision
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
        ),
        decision_store=DecisionStoreConfig(),
        monitor=MonitorConfig(),
    )


@pytest.fixture
def store(tmp_path: Path) -> DecisionStore:
    """Create DecisionStore for testing."""
    return DecisionStore(str(tmp_path / "test.db"))


@pytest.fixture
def mock_matchtrader() -> AsyncMock:
    """Create mock MatchTrader client."""
    client = AsyncMock()
    client.close_position.return_value = MagicMock(success=True, message="OK")
    client.get_balance.return_value = MagicMock(
        balance=50000.0, equity=50000.0, margin=0.0, free_margin=50000.0
    )
    client.get_closed_positions.return_value = []
    return client


@pytest.fixture
def mock_agents() -> MagicMock:
    """Create mock AgentBridge with using_mock=False."""
    agents = MagicMock()
    agents.using_mock = False
    return agents


@pytest.fixture
def scheduler(
    config: AppConfig,
    store: DecisionStore,
    mock_matchtrader: AsyncMock,
    mock_agents: MagicMock,
) -> Scheduler:
    """Create Scheduler instance for testing."""
    return Scheduler(
        config=config,
        store=store,
        scanner=MagicMock(),
        agents=mock_agents,
        engine=AsyncMock(),
        matchtrader=mock_matchtrader,
    )


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_position(
    position_id: str = "POS-1",
    symbol: str = "EURUSD.",
    side: str = "BUY",
    volume: float = 0.01,
    open_price: float = 1.10000,
    current_price: float = 1.10200,
    profit: float = 2.0,
) -> MagicMock:
    """Create a mock PositionInfo."""
    pos = MagicMock()
    pos.position_id = position_id
    pos.symbol = symbol
    pos.side = side
    pos.volume = volume
    pos.open_price = open_price
    pos.current_price = current_price
    pos.profit = profit
    pos.sl_price = None
    pos.tp_price = None
    return pos


def _insert_opened_intent(
    store: DecisionStore,
    intent_id: str,
    symbol: str,
    position_id: str,
) -> TradeIntent:
    """Insert an intent and advance it to 'opened' status."""
    intent = TradeIntent(
        trade_date="2026-02-16",
        symbol=symbol,
        scanner_score=0.85,
        scanner_confidence="high",
    )
    intent.id = intent_id
    store.insert_intent(intent)
    store.claim_next_pending("test-worker")
    store.update_intent_decision(intent_id, "BUY", 50.0, 100.0, "test", "{}")
    store.mark_ready_for_exec(intent_id)
    store.mark_executing(intent_id)
    store.mark_opened(intent_id, position_id)
    return intent


# ── Test Class: _reevaluate_open_positions ──────────────────────────────


class TestReevaluateOpenPositions:
    """Tests for _reevaluate_open_positions() method."""

    async def test_hold_decision_closes_position(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
        mock_agents: MagicMock,
    ) -> None:
        """HOLD decision from LLM should trigger position close."""
        _insert_opened_intent(store, "INT-1", "EURUSD.", "POS-1")
        pos = _make_position(position_id="POS-1", symbol="EURUSD.", side="BUY")

        # LLM returns HOLD — not actionable
        mock_agents.decide.return_value = AgentDecision(
            symbol="EURUSD.",
            decision="HOLD",
            final_state={},
            risk_report="Market turning sideways",
        )

        opened_intents = store.get_active_positions()
        await scheduler._reevaluate_open_positions([pos], opened_intents)

        mock_matchtrader.close_position.assert_called_once_with(
            position_id="POS-1",
            symbol="EURUSD.",
            side="BUY",
            volume=0.01,
        )
        assert "POS-1" in scheduler._reevaluation_close_positions

    async def test_buy_decision_keeps_position_open(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
        mock_agents: MagicMock,
    ) -> None:
        """BUY decision from LLM should NOT close position."""
        _insert_opened_intent(store, "INT-1", "EURUSD.", "POS-1")
        pos = _make_position(position_id="POS-1", symbol="EURUSD.", side="BUY")

        mock_agents.decide.return_value = AgentDecision(
            symbol="EURUSD.",
            decision="BUY",
            final_state={},
            risk_report="Still bullish",
        )

        opened_intents = store.get_active_positions()
        await scheduler._reevaluate_open_positions([pos], opened_intents)

        mock_matchtrader.close_position.assert_not_called()
        assert "POS-1" not in scheduler._reevaluation_close_positions

    async def test_sell_decision_keeps_position_open(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
        mock_agents: MagicMock,
    ) -> None:
        """SELL decision from LLM should NOT close a SELL position (still actionable)."""
        _insert_opened_intent(store, "INT-1", "EURUSD.", "POS-1")
        pos = _make_position(position_id="POS-1", symbol="EURUSD.", side="SELL")

        mock_agents.decide.return_value = AgentDecision(
            symbol="EURUSD.",
            decision="SELL",
            final_state={},
            risk_report="Bearish momentum",
        )

        opened_intents = store.get_active_positions()
        await scheduler._reevaluate_open_positions([pos], opened_intents)

        mock_matchtrader.close_position.assert_not_called()
        assert "POS-1" not in scheduler._reevaluation_close_positions

    async def test_throttle_skips_recently_evaluated(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_agents: MagicMock,
    ) -> None:
        """Position evaluated within 4h should be skipped."""
        _insert_opened_intent(store, "INT-1", "EURUSD.", "POS-1")
        pos = _make_position(position_id="POS-1")

        # Pre-populate as recently evaluated
        scheduler._last_reevaluation["POS-1"] = datetime.now(timezone.utc)

        opened_intents = store.get_active_positions()
        await scheduler._reevaluate_open_positions([pos], opened_intents)

        mock_agents.decide.assert_not_called()

    async def test_throttle_allows_after_interval(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
        mock_agents: MagicMock,
    ) -> None:
        """Position evaluated >4h ago should be re-evaluated."""
        _insert_opened_intent(store, "INT-1", "EURUSD.", "POS-1")
        pos = _make_position(position_id="POS-1")

        # Set last eval to 5 hours ago
        scheduler._last_reevaluation["POS-1"] = datetime.now(timezone.utc) - timedelta(hours=5)

        mock_agents.decide.return_value = AgentDecision(
            symbol="EURUSD.",
            decision="BUY",
            final_state={},
            risk_report="Confirmed",
        )

        opened_intents = store.get_active_positions()
        await scheduler._reevaluate_open_positions([pos], opened_intents)

        mock_agents.decide.assert_called_once()

    async def test_mock_llm_skips_evaluation(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_agents: MagicMock,
    ) -> None:
        """When using_mock=True, skip all re-evaluation."""
        scheduler._agents.using_mock = True
        _insert_opened_intent(store, "INT-1", "EURUSD.", "POS-1")
        pos = _make_position(position_id="POS-1")

        opened_intents = store.get_active_positions()
        await scheduler._reevaluate_open_positions([pos], opened_intents)

        mock_agents.decide.assert_not_called()

    async def test_close_failure_does_not_add_to_set(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
        mock_agents: MagicMock,
    ) -> None:
        """Failed close_position should NOT add position to reevaluation set."""
        _insert_opened_intent(store, "INT-1", "EURUSD.", "POS-1")
        pos = _make_position(position_id="POS-1")

        mock_agents.decide.return_value = AgentDecision(
            symbol="EURUSD.",
            decision="HOLD",
            final_state={},
            risk_report="",
        )
        mock_matchtrader.close_position.return_value = MagicMock(
            success=False, message="Server error"
        )

        opened_intents = store.get_active_positions()
        await scheduler._reevaluate_open_positions([pos], opened_intents)

        mock_matchtrader.close_position.assert_called_once()
        assert "POS-1" not in scheduler._reevaluation_close_positions

    async def test_no_matching_intent_skips_position(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_agents: MagicMock,
    ) -> None:
        """Position with no matching opened intent in store should be skipped."""
        # Don't insert any intent — store is empty
        pos = _make_position(position_id="POS-UNKNOWN")

        opened_intents = store.get_active_positions()
        await scheduler._reevaluate_open_positions([pos], opened_intents)

        mock_agents.decide.assert_not_called()

    async def test_multiple_positions_evaluated(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
        mock_agents: MagicMock,
    ) -> None:
        """Multiple open positions should each be evaluated."""
        _insert_opened_intent(store, "INT-1", "EURUSD.", "POS-1")
        _insert_opened_intent(store, "INT-2", "GBPUSD.", "POS-2")
        pos1 = _make_position(position_id="POS-1", symbol="EURUSD.")
        pos2 = _make_position(position_id="POS-2", symbol="GBPUSD.")

        mock_agents.decide.return_value = AgentDecision(
            symbol="EURUSD.",
            decision="BUY",
            final_state={},
            risk_report="",
        )

        opened_intents = store.get_active_positions()
        await scheduler._reevaluate_open_positions([pos1, pos2], opened_intents)

        assert mock_agents.decide.call_count == 2

    async def test_last_reevaluation_updated_after_eval(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_agents: MagicMock,
    ) -> None:
        """_last_reevaluation dict should be updated after successful evaluation."""
        _insert_opened_intent(store, "INT-1", "EURUSD.", "POS-1")
        pos = _make_position(position_id="POS-1")

        mock_agents.decide.return_value = AgentDecision(
            symbol="EURUSD.",
            decision="BUY",
            final_state={},
            risk_report="",
        )

        assert "POS-1" not in scheduler._last_reevaluation

        opened_intents = store.get_active_positions()
        await scheduler._reevaluate_open_positions([pos], opened_intents)

        assert "POS-1" in scheduler._last_reevaluation
        assert isinstance(scheduler._last_reevaluation["POS-1"], datetime)

    async def test_decide_receives_correct_qlib_data(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_agents: MagicMock,
    ) -> None:
        """decide() should receive scanner data from the intent."""
        _insert_opened_intent(store, "INT-1", "EURUSD.", "POS-1")
        pos = _make_position(position_id="POS-1")

        mock_agents.decide.return_value = AgentDecision(
            symbol="EURUSD.",
            decision="BUY",
            final_state={},
            risk_report="",
        )

        opened_intents = store.get_active_positions()
        await scheduler._reevaluate_open_positions([pos], opened_intents)

        call_kwargs = mock_agents.decide.call_args.kwargs
        assert call_kwargs["symbol"] == "EURUSD."
        qlib_data = call_kwargs["qlib_data"]
        assert qlib_data["score"] == 0.85
        assert qlib_data["signal_strength"] == "high"
        assert qlib_data["confidence"] == "high"

    async def test_exception_during_decide_is_caught(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
        mock_agents: MagicMock,
    ) -> None:
        """Exception in agents.decide() should be caught, not propagate."""
        _insert_opened_intent(store, "INT-1", "EURUSD.", "POS-1")
        pos = _make_position(position_id="POS-1")

        mock_agents.decide.side_effect = RuntimeError("LLM crashed")

        # Should not raise
        opened_intents = store.get_active_positions()
        await scheduler._reevaluate_open_positions([pos], opened_intents)

        mock_matchtrader.close_position.assert_not_called()


# ── Test Class: Re-evaluation Exit Reason Override ──────────────────────


class TestReevaluationExitReason:
    """Tests for exit_reason='reeval_close' override in _handle_position_closed."""

    async def test_exit_reason_set_to_reevaluation_hold(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """When position_id is in _reevaluation_close_positions, exit_reason should override."""
        _insert_opened_intent(store, "INT-1", "EURUSD.", "POS-1")
        intent = store.get_intent("INT-1")

        # Pre-populate the reevaluation set
        scheduler._reevaluation_close_positions.add("POS-1")

        # Mock closed positions return empty (PnL defaults to 0)
        mock_matchtrader.get_closed_positions.return_value = []

        await scheduler._handle_position_closed(intent)

        # Verify exit_reason was overridden
        updated = store.get_intent("INT-1")
        assert updated is not None
        assert updated.exit_reason == "reeval_close"
        # Position should be removed from the set after handling
        assert "POS-1" not in scheduler._reevaluation_close_positions

    async def test_normal_close_not_overridden(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Position NOT in reevaluation set should keep original exit_reason."""
        _insert_opened_intent(store, "INT-1", "EURUSD.", "POS-1")
        intent = store.get_intent("INT-1")

        mock_matchtrader.get_closed_positions.return_value = []

        await scheduler._handle_position_closed(intent)

        updated = store.get_intent("INT-1")
        assert updated is not None
        # Default exit_reason when no closed position found and not in any override set
        assert updated.exit_reason == "manual_close"

    async def test_reevaluation_cleanup_on_close(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """_last_reevaluation should be cleaned up when position closes."""
        _insert_opened_intent(store, "INT-1", "EURUSD.", "POS-1")
        intent = store.get_intent("INT-1")

        scheduler._last_reevaluation["POS-1"] = datetime.now(timezone.utc)
        mock_matchtrader.get_closed_positions.return_value = []

        await scheduler._handle_position_closed(intent)

        # Cleanup removes from _last_reevaluation
        assert "POS-1" not in scheduler._last_reevaluation
