"""Tests for session cadence integration in Scheduler (v1.2.0)."""

from unittest.mock import AsyncMock, MagicMock

from src.config import AppConfig
from src.scheduler.session_cadence import SessionCadence


def test_scheduler_creates_session_cadence():
    """Scheduler should initialize a SessionCadence instance."""
    from src.scheduler.scheduler import Scheduler

    config = AppConfig()
    scheduler = Scheduler(
        config=config,
        store=MagicMock(),
        scanner=MagicMock(),
        agents=MagicMock(),
        engine=MagicMock(),
        matchtrader=AsyncMock(),
    )
    assert hasattr(scheduler, "_session_cadence")
    assert isinstance(scheduler._session_cadence, SessionCadence)
