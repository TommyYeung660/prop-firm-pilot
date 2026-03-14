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
            reeval_min_hold_seconds=0,
            reeval_interval_seconds=14400,
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

    async def test_hold_decision_keeps_position_open(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
        mock_agents: MagicMock,
    ) -> None:
        """HOLD decision from LLM should keep position open (do nothing)."""
        _insert_opened_intent(store, "INT-1", "EURUSD.", "POS-1")
        pos = _make_position(position_id="POS-1", symbol="EURUSD.", side="BUY")
        # LLM returns HOLD — do nothing
        mock_agents.decide.return_value = AgentDecision(
            symbol="EURUSD.",
            decision="HOLD",
            final_state={},
            risk_report="Market turning sideways",
        )
        opened_intents = store.get_active_positions()
        await scheduler._reevaluate_open_positions([pos], opened_intents)
        mock_matchtrader.close_position.assert_not_called()
        assert "POS-1" not in scheduler._reevaluation_close_positions

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
            decision="SELL",
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
        # Verify position context fields are passed
        assert qlib_data["position_side"] == "BUY"
        assert qlib_data["unrealized_pnl"] == 2.0
        assert qlib_data["entry_price"] == 1.10000
        assert qlib_data["current_price"] == 1.10200
        assert "hold_duration_seconds" in qlib_data

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

    async def test_sell_signal_on_buy_position_closes(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
        mock_agents: MagicMock,
    ) -> None:
        """SELL signal on BUY position is a reversal — should close."""
        _insert_opened_intent(store, "INT-1", "EURUSD.", "POS-1")
        pos = _make_position(position_id="POS-1", symbol="EURUSD.", side="BUY")

        mock_agents.decide.return_value = AgentDecision(
            symbol="EURUSD.",
            decision="SELL",
            final_state={},
            risk_report="Bearish reversal",
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
        assert scheduler._pending_close_outcomes["POS-1"].trigger_source == "reeval_close"
        assert scheduler._pending_close_outcomes["POS-1"].action_kind == "full_close"

    async def test_buy_signal_on_sell_position_closes(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
        mock_agents: MagicMock,
    ) -> None:
        """BUY signal on SELL position is a reversal — should close."""
        _insert_opened_intent(store, "INT-1", "EURUSD.", "POS-1")
        pos = _make_position(position_id="POS-1", symbol="EURUSD.", side="SELL")

        mock_agents.decide.return_value = AgentDecision(
            symbol="EURUSD.",
            decision="BUY",
            final_state={},
            risk_report="Bullish reversal",
        )

        opened_intents = store.get_active_positions()
        await scheduler._reevaluate_open_positions([pos], opened_intents)

        mock_matchtrader.close_position.assert_called_once_with(
            position_id="POS-1",
            symbol="EURUSD.",
            side="SELL",
            volume=0.01,
        )
        assert "POS-1" in scheduler._reevaluation_close_positions
        assert scheduler._pending_close_outcomes["POS-1"].trigger_source == "reeval_close"
        assert scheduler._pending_close_outcomes["POS-1"].action_kind == "full_close"

    async def test_buy_signal_on_buy_position_keeps_open(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
        mock_agents: MagicMock,
    ) -> None:
        """BUY signal on BUY position is confirmation — should keep open."""
        _insert_opened_intent(store, "INT-1", "EURUSD.", "POS-1")
        pos = _make_position(position_id="POS-1", symbol="EURUSD.", side="BUY")

        mock_agents.decide.return_value = AgentDecision(
            symbol="EURUSD.",
            decision="BUY",
            final_state={},
            risk_report="Bullish continuation",
        )

        opened_intents = store.get_active_positions()
        await scheduler._reevaluate_open_positions([pos], opened_intents)

        mock_matchtrader.close_position.assert_not_called()
        assert "POS-1" not in scheduler._reevaluation_close_positions

    async def test_sell_signal_on_sell_position_keeps_open(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
        mock_agents: MagicMock,
    ) -> None:
        """SELL signal on SELL position is confirmation — should keep open."""
        _insert_opened_intent(store, "INT-1", "EURUSD.", "POS-1")
        pos = _make_position(position_id="POS-1", symbol="EURUSD.", side="SELL")

        mock_agents.decide.return_value = AgentDecision(
            symbol="EURUSD.",
            decision="SELL",
            final_state={},
            risk_report="Bearish continuation",
        )

        opened_intents = store.get_active_positions()
        await scheduler._reevaluate_open_positions([pos], opened_intents)

        mock_matchtrader.close_position.assert_not_called()
        assert "POS-1" not in scheduler._reevaluation_close_positions

    async def test_min_hold_time_skips_early_reeval(
        self,
        tmp_path: Path,
        mock_matchtrader: AsyncMock,
        mock_agents: MagicMock,
    ) -> None:
        """Position opened recently should skip reeval when min hold time is enforced."""
        cfg = AppConfig(
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
                reeval_min_hold_seconds=3600,
                reeval_interval_seconds=14400,
            ),
            decision_store=DecisionStoreConfig(),
            monitor=MonitorConfig(),
        )
        st = DecisionStore(str(tmp_path / "hold_skip.db"))
        sched = Scheduler(
            config=cfg,
            store=st,
            scanner=MagicMock(),
            agents=mock_agents,
            engine=AsyncMock(),
            matchtrader=mock_matchtrader,
        )
        # Intent created & opened just now → hold duration ≈ 0s < 3600s
        _insert_opened_intent(st, "INT-1", "EURUSD.", "POS-1")
        pos = _make_position(position_id="POS-1", symbol="EURUSD.", side="BUY")

        opened_intents = st.get_active_positions()
        await sched._reevaluate_open_positions([pos], opened_intents)

        mock_agents.decide.assert_not_called()

    async def test_min_hold_time_allows_after_threshold(
        self,
        tmp_path: Path,
        mock_matchtrader: AsyncMock,
        mock_agents: MagicMock,
    ) -> None:
        """Position opened >threshold ago should allow reeval."""
        cfg = AppConfig(
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
                reeval_min_hold_seconds=3600,
                reeval_interval_seconds=14400,
            ),
            decision_store=DecisionStoreConfig(),
            monitor=MonitorConfig(),
        )
        st = DecisionStore(str(tmp_path / "hold_allow.db"))
        sched = Scheduler(
            config=cfg,
            store=st,
            scanner=MagicMock(),
            agents=mock_agents,
            engine=AsyncMock(),
            matchtrader=mock_matchtrader,
        )
        _insert_opened_intent(st, "INT-1", "EURUSD.", "POS-1")
        # Backdate executed_at to 2 hours ago so hold_seconds > 3600
        two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)
        st._conn.execute(
            "UPDATE intents SET executed_at = ? WHERE id = ?",
            (two_hours_ago.isoformat(), "INT-1"),
        )
        st._conn.commit()

        pos = _make_position(position_id="POS-1", symbol="EURUSD.", side="BUY")

        mock_agents.decide.return_value = AgentDecision(
            symbol="EURUSD.",
            decision="BUY",
            final_state={},
            risk_report="Confirmed",
        )

        opened_intents = st.get_active_positions()
        await sched._reevaluate_open_positions([pos], opened_intents)

        mock_agents.decide.assert_called_once()


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
        scheduler._reevaluation_close_positions["POS-1"] = 0.0

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


# ── Additional Fixtures for Minimum Hold Time Tests ───────────────────────


@pytest.fixture
def config_with_hold_time(tmp_path: Path) -> AppConfig:
    """Config with minimum hold time enabled for testing grace period."""
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
            reeval_min_hold_seconds=3600,  # 1 hour minimum hold time
            reeval_interval_seconds=14400,
        ),
        decision_store=DecisionStoreConfig(),
        monitor=MonitorConfig(),
    )


@pytest.fixture
def scheduler_with_hold_time(
    config_with_hold_time: AppConfig,
    store: DecisionStore,
    mock_matchtrader: AsyncMock,
    mock_agents: MagicMock,
) -> Scheduler:
    """Scheduler with minimum hold time enabled."""
    return Scheduler(
        config=config_with_hold_time,
        store=store,
        scanner=MagicMock(),
        agents=mock_agents,
        engine=AsyncMock(),
        matchtrader=mock_matchtrader,
    )


# ── BUG #3: PnL Fallback Tests ──────────────────────────────────────────


class TestPnlFallback:
    """Tests for unrealized PnL fallback when broker query returns zero."""

    async def test_reeval_close_uses_unrealized_pnl_fallback(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """When broker returns no closed position, reeval close uses recorded unrealized PnL."""
        _insert_opened_intent(store, "INT-1", "EURUSD.", "POS-1")
        intent = store.get_intent("INT-1")

        # Record unrealized PnL at close time (simulating reeval close)
        scheduler._reevaluation_close_positions["POS-1"] = 42.50

        # Broker returns empty — position not yet in closed list
        mock_matchtrader.get_closed_positions.return_value = []

        await scheduler._handle_position_closed(intent)

        updated = store.get_intent("INT-1")
        assert updated is not None
        assert updated.exit_reason == "reeval_close"
        assert updated.realized_pnl == pytest.approx(42.50)

    async def test_reeval_close_prefers_broker_pnl_over_fallback(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """When broker returns real PnL, use it even if unrealized PnL was recorded."""
        _insert_opened_intent(store, "INT-1", "EURUSD.", "POS-1")
        intent = store.get_intent("INT-1")

        # Record unrealized PnL (should be overridden by broker data)
        scheduler._reevaluation_close_positions["POS-1"] = 42.50

        # Broker returns actual closed position with different PnL
        mock_matchtrader.get_closed_positions.return_value = [
            MagicMock(
                position_id="POS-1",
                profit=38.75,
                close_price=1.10500,
                open_price=1.10000,
                volume=0.10,
            ),
        ]

        await scheduler._handle_position_closed(intent)

        updated = store.get_intent("INT-1")
        assert updated is not None
        assert updated.exit_reason == "reeval_close"
        # Should use broker PnL (38.75), not the recorded unrealized PnL (42.50)
        assert updated.realized_pnl == pytest.approx(38.75)

    async def test_best_day_close_uses_unrealized_pnl_fallback(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """When broker returns no closed position, best_day close uses recorded unrealized PnL."""
        _insert_opened_intent(store, "INT-1", "EURUSD.", "POS-1")
        intent = store.get_intent("INT-1")

        # Record unrealized PnL at close time (simulating best day close)
        scheduler._best_day_close_positions["POS-1"] = 120.00

        mock_matchtrader.get_closed_positions.return_value = []

        await scheduler._handle_position_closed(intent)

        updated = store.get_intent("INT-1")
        assert updated is not None
        assert updated.exit_reason == "best_day_close"
        assert updated.realized_pnl == pytest.approx(120.00)

    async def test_reeval_close_negative_pnl_fallback(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """Negative unrealized PnL should also be used as fallback."""
        _insert_opened_intent(store, "INT-1", "EURUSD.", "POS-1")
        intent = store.get_intent("INT-1")

        scheduler._reevaluation_close_positions["POS-1"] = -15.30
        mock_matchtrader.get_closed_positions.return_value = []

        await scheduler._handle_position_closed(intent)

        updated = store.get_intent("INT-1")
        assert updated is not None
        assert updated.exit_reason == "reeval_close"
        assert updated.realized_pnl == pytest.approx(-15.30)
