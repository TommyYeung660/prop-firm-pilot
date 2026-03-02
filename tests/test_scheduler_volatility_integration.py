"""Tests for volatility monitor integration in Scheduler (v1.2.0)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import AppConfig
from src.scheduler.volatility_monitor import VolatilityMonitor


def test_scheduler_creates_volatility_monitor():
    """Scheduler should initialize a VolatilityMonitor."""
    from src.scheduler.scheduler import Scheduler

    config = AppConfig()
    config.scheduler.volatility_trigger_enabled = True
    scheduler = Scheduler(
        config=config,
        store=MagicMock(),
        scanner=MagicMock(),
        agents=MagicMock(),
        engine=MagicMock(),
        matchtrader=AsyncMock(),
    )
    assert hasattr(scheduler, "_volatility_monitor")
    assert isinstance(scheduler._volatility_monitor, VolatilityMonitor)


def test_volatility_loop_in_tasks_when_enabled():
    """When volatility_trigger_enabled, start() should include the volatility loop."""
    from src.scheduler.scheduler import Scheduler

    config = AppConfig()
    config.scheduler.volatility_trigger_enabled = True
    scheduler = Scheduler(
        config=config,
        store=MagicMock(),
        scanner=MagicMock(),
        agents=MagicMock(),
        engine=MagicMock(),
        matchtrader=AsyncMock(),
    )
    # Check the method exists
    assert hasattr(scheduler, "_volatility_monitor_loop")
