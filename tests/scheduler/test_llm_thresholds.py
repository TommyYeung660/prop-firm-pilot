"""
Tests for LLM pre/post confidence threshold filtering in Scheduler.
"""

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
from src.decision.decision_formatter import FormattedDecision
from src.decision.schemas import TradeIntent
from src.decision_store.sqlite_store import DecisionStore
from src.optimize.optimization_state import OptimizationState, Thresholds
from src.scheduler.scheduler import Scheduler


@pytest.fixture
def store(tmp_path: object) -> DecisionStore:
    """Create a temporary DecisionStore for threshold tests."""
    db_path = f"{tmp_path}/test_llm_thresholds.db"
    s = DecisionStore(db_path=db_path)
    yield s  # type: ignore[misc]
    s.close()


@pytest.fixture
def config() -> AppConfig:
    """Minimal AppConfig for scheduler threshold tests."""
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
def mock_agents() -> MagicMock:
    """Mock AgentBridge for threshold tests."""
    agents = MagicMock()
    agents.using_mock = False
    agents.decide.return_value = AgentDecision(
        symbol="EURUSD",
        decision="BUY",
        final_state={"test": True},
        risk_report="test",
    )
    return agents


@pytest.fixture
def scheduler(
    config: AppConfig,
    store: DecisionStore,
    mock_agents: MagicMock,
) -> Scheduler:
    """Create a Scheduler with mocked dependencies."""
    mock_scanner = MagicMock()
    mock_engine = AsyncMock()
    mock_matchtrader = AsyncMock()
    return Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )


class TestThresholdHelpers:
    """Unit tests for threshold comparison logic."""

    def test_passes_threshold_respects_min_confidence(self) -> None:
        thresholds = Thresholds(min_confidence="high", min_blended_confidence=0.65)
        assert Scheduler._passes_threshold("low", 0.9, thresholds) is False
        assert Scheduler._passes_threshold("high", 0.7, thresholds) is True


class TestPrePostFiltering:
    """Integration tests for pre/post threshold filtering."""

    async def test_pre_filter_blocks_low_confidence(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_agents: MagicMock,
    ) -> None:
        scheduler._optimization_state = OptimizationState(
            global_thresholds=Thresholds(min_confidence="high", min_blended_confidence=0.8)
        )
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.35,
            scanner_confidence="low",
        )
        store.insert_intent(intent)
        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None

        await scheduler._process_claimed_intent("llm-0", claimed)

        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "cancelled"
        assert updated.execution_error is not None
        assert "pre-filter" in updated.execution_error
        mock_agents.decide.assert_not_called()

    async def test_post_filter_blocks_low_confidence_decision(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
        mock_agents: MagicMock,
    ) -> None:
        scheduler._optimization_state = OptimizationState(
            global_thresholds=Thresholds(min_confidence="low", min_blended_confidence=0.8)
        )
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.95,
            scanner_confidence="high",
        )
        store.insert_intent(intent)
        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None

        low_conf = FormattedDecision(
            symbol="EURUSD",
            side="BUY",
            confidence_score=0.4,
            suggested_sl_pips=20.0,
            suggested_tp_pips=40.0,
            risk_reward_ratio=2.0,
        )

        with patch("src.scheduler.scheduler.format_decision", return_value=low_conf):
            await scheduler._process_claimed_intent("llm-0", claimed)

        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.status == "cancelled"
        assert updated.execution_error is not None
        assert "post-filter" in updated.execution_error
        mock_agents.decide.assert_called_once()
