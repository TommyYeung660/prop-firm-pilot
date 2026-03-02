"""Tests for session-aware cadence calculator (v1.2.0)."""

from datetime import datetime, timezone

import pytest

from src.config import SchedulerConfig
from src.scheduler.session_cadence import SessionCadence


def _utc(hour: int, minute: int = 0) -> datetime:
    """Create a UTC datetime on a Wednesday (weekday) at given hour."""
    return datetime(2026, 3, 4, hour, minute, tzinfo=timezone.utc)  # Wednesday


class TestSessionCadence:
    def test_disabled_returns_default_interval(self):
        config = SchedulerConfig(session_aware_enabled=False, scanner_interval_seconds=14400)
        cadence = SessionCadence(config)
        assert cadence.get_scanner_interval(_utc(8)) == 14400  # London hour, but disabled

    def test_london_session_returns_active_interval(self):
        config = SchedulerConfig(
            session_aware_enabled=True,
            active_session_interval_seconds=3600,
            quiet_session_interval_seconds=14400,
        )
        cadence = SessionCadence(config)
        assert cadence.get_scanner_interval(_utc(8)) == 3600  # 08:00 UTC = London

    def test_ny_session_returns_active_interval(self):
        config = SchedulerConfig(
            session_aware_enabled=True,
            active_session_interval_seconds=3600,
        )
        cadence = SessionCadence(config)
        assert cadence.get_scanner_interval(_utc(14)) == 3600  # 14:00 UTC = NY

    def test_london_ny_overlap_returns_active_interval(self):
        config = SchedulerConfig(
            session_aware_enabled=True,
            active_session_interval_seconds=3600,
        )
        cadence = SessionCadence(config)
        assert cadence.get_scanner_interval(_utc(13)) == 3600  # 13:00 = overlap

    def test_off_hours_returns_quiet_interval(self):
        config = SchedulerConfig(
            session_aware_enabled=True,
            active_session_interval_seconds=3600,
            quiet_session_interval_seconds=14400,
        )
        cadence = SessionCadence(config)
        assert cadence.get_scanner_interval(_utc(3)) == 14400  # 03:00 UTC = Asia

    def test_session_boundary_london_open(self):
        config = SchedulerConfig(session_aware_enabled=True, active_session_interval_seconds=3600)
        cadence = SessionCadence(config)
        assert cadence.is_active_session(_utc(7))  # 07:00 = London open
        assert not cadence.is_active_session(_utc(6, 59))  # 06:59 = not yet

    def test_session_boundary_ny_close(self):
        config = SchedulerConfig(session_aware_enabled=True, active_session_interval_seconds=3600)
        cadence = SessionCadence(config)
        assert cadence.is_active_session(_utc(20))  # 20:00 = still NY
        assert not cadence.is_active_session(_utc(21))  # 21:00 = NY closed

    def test_session_name_overlap(self):
        config = SchedulerConfig(session_aware_enabled=True)
        cadence = SessionCadence(config)
        assert cadence.current_session_name(_utc(13)) == "London/NY Overlap"

    def test_session_name_london_only(self):
        config = SchedulerConfig(session_aware_enabled=True)
        cadence = SessionCadence(config)
        assert cadence.current_session_name(_utc(8)) == "London"

    def test_session_name_ny_only(self):
        config = SchedulerConfig(session_aware_enabled=True)
        cadence = SessionCadence(config)
        assert cadence.current_session_name(_utc(17)) == "New York"

    def test_session_name_off_hours(self):
        config = SchedulerConfig(session_aware_enabled=True)
        cadence = SessionCadence(config)
        assert cadence.current_session_name(_utc(3)) == "Off-hours"
