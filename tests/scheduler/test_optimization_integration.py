"""
Scheduler integration tests for optimization refresh.
"""

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
from src.decision_store.sqlite_store import DecisionStore
from src.optimize.optimization_state import ABTestState, OptimizationState
from src.scheduler.scheduler import Scheduler


@pytest.fixture
def store(tmp_path: object) -> DecisionStore:
    """Create a temporary DecisionStore for scheduler tests."""
    db_path = f"{tmp_path}/test_opt_integration.db"
    s = DecisionStore(db_path=db_path)
    yield s  # type: ignore[misc]
    s.close()


@pytest.fixture
def config() -> AppConfig:
    """Minimal AppConfig for scheduler optimization integration."""
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
def scheduler(config: AppConfig, store: DecisionStore) -> Scheduler:
    """Build a Scheduler with mocked dependencies."""
    mock_scanner = MagicMock()
    mock_agents = MagicMock()
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


class TestOptimizationIntegration:
    """Scheduler should refresh optimization state during daily summary."""

    async def test_daily_summary_triggers_optimization_refresh(self, scheduler: Scheduler) -> None:
        engine = MagicMock()
        scheduler._optimization_engine = engine

        await scheduler._send_daily_summary("2026-02-12")

        engine.refresh_state.assert_called_once()

    async def test_start_refreshes_optimization_state_before_workers(
        self,
        scheduler: Scheduler,
    ) -> None:
        """Startup should immediately load optimization + AB routing state."""
        state = OptimizationState(
            ab_test=ABTestState(
                model_a="rightcodes/gpt-5.4",
                model_b="volcengine/kimi-k2.5",
                ratio=0.5,
            )
        )
        engine = MagicMock()
        engine.refresh_state.return_value = state
        scheduler._optimization_engine = engine
        scheduler._agents = MagicMock()

        with pytest.raises(RuntimeError, match="stop after refresh"):
            with pytest.MonkeyPatch.context() as mp:
                async def fail_gather(*args, **kwargs):
                    del kwargs
                    for item in args:
                        if hasattr(item, "close"):
                            item.close()
                    raise RuntimeError("stop after refresh")

                mp.setattr("asyncio.gather", fail_gather)
                await scheduler.start()

        engine.refresh_state.assert_called_once()
        scheduler._agents.set_ab_state.assert_called_once_with(state.ab_test)
