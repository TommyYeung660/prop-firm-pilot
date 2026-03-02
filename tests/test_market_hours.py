"""Tests for MarketHoursConfig, MarketHoursChecker, DST utilities, and SessionCadence."""

from datetime import datetime, timezone

from src.config import AppConfig, MarketHoursConfig, SchedulerConfig
from src.scheduler.dst_utils import (
    dst_adjust_hour,
    get_utc_offset_hours,
    is_dst_active,
)
from src.scheduler.market_hours import MarketHoursChecker
from src.scheduler.session_cadence import SessionCadence


class TestMarketHoursConfig:
    """Tests for MarketHoursConfig defaults and loading."""

    def test_default_disabled(self) -> None:
        config = MarketHoursConfig()
        assert config.enabled is False

    def test_fx_defaults(self) -> None:
        config = MarketHoursConfig(enabled=True)
        assert config.close_day == "Friday"
        assert config.close_time_utc == "22:00"
        assert config.open_day == "Sunday"
        assert config.open_time_utc == "22:00"
        assert config.force_close_before_weekend is False

    def test_force_close_settings(self) -> None:
        config = MarketHoursConfig(
            enabled=True,
            force_close_before_weekend=True,
            force_close_minutes_before=30,
        )
        assert config.force_close_before_weekend is True
        assert config.force_close_minutes_before == 30

    def test_config_in_scheduler(self) -> None:
        sched = SchedulerConfig(market_hours=MarketHoursConfig(enabled=True))
        assert sched.market_hours.enabled is True

    def test_full_config_from_dict(self) -> None:
        config = AppConfig(
            scheduler=SchedulerConfig(
                market_hours={
                    "enabled": True,
                    "close_day": "Friday",
                    "close_time_utc": "22:00",
                    "open_day": "Sunday",
                    "open_time_utc": "22:00",
                    "force_close_before_weekend": True,
                    "force_close_minutes_before": 15,
                }
            )
        )
        mh = config.scheduler.market_hours
        assert mh.enabled is True
        assert mh.force_close_minutes_before == 15

    def test_dst_config_defaults(self) -> None:
        config = MarketHoursConfig()
        assert config.dst_auto is False
        assert config.server_timezone == "Europe/Athens"

    def test_dst_config_enabled(self) -> None:
        config = MarketHoursConfig(
            enabled=True,
            dst_auto=True,
            server_timezone="Europe/Athens",
        )
        assert config.dst_auto is True
        assert config.server_timezone == "Europe/Athens"


class TestMarketHoursChecker:
    """Tests for MarketHoursChecker.is_market_open() and force-close timing."""

    def test_disabled_always_open(self) -> None:
        checker = MarketHoursChecker(MarketHoursConfig(enabled=False))
        # Saturday should still show as open when disabled
        saturday = datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc)
        assert checker.is_market_open(saturday) is True

    def test_weekday_is_open(self) -> None:
        checker = MarketHoursChecker(MarketHoursConfig(enabled=True))
        wednesday = datetime(2026, 2, 25, 14, 0, tzinfo=timezone.utc)
        assert checker.is_market_open(wednesday) is True

    def test_saturday_is_closed(self) -> None:
        checker = MarketHoursChecker(MarketHoursConfig(enabled=True))
        saturday = datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc)
        assert checker.is_market_open(saturday) is False

    def test_friday_before_close_is_open(self) -> None:
        checker = MarketHoursChecker(
            MarketHoursConfig(enabled=True, close_day="Friday", close_time_utc="22:00")
        )
        friday_early = datetime(2026, 2, 27, 20, 0, tzinfo=timezone.utc)
        assert checker.is_market_open(friday_early) is True

    def test_friday_after_close_is_closed(self) -> None:
        checker = MarketHoursChecker(
            MarketHoursConfig(enabled=True, close_day="Friday", close_time_utc="22:00")
        )
        friday_late = datetime(2026, 2, 27, 23, 0, tzinfo=timezone.utc)
        assert checker.is_market_open(friday_late) is False

    def test_sunday_before_open_is_closed(self) -> None:
        checker = MarketHoursChecker(
            MarketHoursConfig(enabled=True, open_day="Sunday", open_time_utc="22:00")
        )
        sunday_early = datetime(2026, 3, 1, 18, 0, tzinfo=timezone.utc)
        assert checker.is_market_open(sunday_early) is False

    def test_sunday_after_open_is_open(self) -> None:
        checker = MarketHoursChecker(
            MarketHoursConfig(enabled=True, open_day="Sunday", open_time_utc="22:00")
        )
        sunday_late = datetime(2026, 3, 1, 23, 0, tzinfo=timezone.utc)
        assert checker.is_market_open(sunday_late) is True

    def test_should_force_close(self) -> None:
        checker = MarketHoursChecker(
            MarketHoursConfig(
                enabled=True,
                close_day="Friday",
                close_time_utc="22:00",
                force_close_before_weekend=True,
                force_close_minutes_before=15,
            )
        )
        # 14 minutes before close — should force close
        friday_2146 = datetime(2026, 2, 27, 21, 46, tzinfo=timezone.utc)
        assert checker.should_force_close(friday_2146) is True

        # 20 minutes before close — not yet
        friday_2140 = datetime(2026, 2, 27, 21, 40, tzinfo=timezone.utc)
        assert checker.should_force_close(friday_2140) is False

    def test_should_force_close_disabled(self) -> None:
        checker = MarketHoursChecker(
            MarketHoursConfig(enabled=True, force_close_before_weekend=False)
        )
        friday_2150 = datetime(2026, 2, 27, 21, 50, tzinfo=timezone.utc)
        assert checker.should_force_close(friday_2150) is False

    def test_seconds_until_open(self) -> None:
        checker = MarketHoursChecker(
            MarketHoursConfig(enabled=True, open_day="Sunday", open_time_utc="22:00")
        )
        # Saturday noon — should be ~34 hours until Sunday 22:00
        saturday_noon = datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc)
        seconds = checker.seconds_until_open(saturday_noon)
        assert 33 * 3600 < seconds < 35 * 3600


# ── DST Utility Tests ───────────────────────────────────────────────────


class TestDSTUtils:
    """Tests for DST auto-detection utilities."""

    def test_winter_time_athens_offset(self) -> None:
        """Athens is UTC+2 in winter (January)."""
        winter = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        offset = get_utc_offset_hours("Europe/Athens", winter)
        assert offset == 2.0

    def test_summer_time_athens_offset(self) -> None:
        """Athens is UTC+3 in summer (July)."""
        summer = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
        offset = get_utc_offset_hours("Europe/Athens", summer)
        assert offset == 3.0

    def test_winter_time_london_offset(self) -> None:
        """London is UTC+0 in winter."""
        winter = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        offset = get_utc_offset_hours("Europe/London", winter)
        assert offset == 0.0

    def test_summer_time_london_offset(self) -> None:
        """London is UTC+1 in summer (BST)."""
        summer = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
        offset = get_utc_offset_hours("Europe/London", summer)
        assert offset == 1.0

    def test_winter_time_ny_offset(self) -> None:
        """New York is UTC-5 in winter (EST)."""
        winter = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        offset = get_utc_offset_hours("America/New_York", winter)
        assert offset == -5.0

    def test_summer_time_ny_offset(self) -> None:
        """New York is UTC-4 in summer (EDT)."""
        summer = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
        offset = get_utc_offset_hours("America/New_York", summer)
        assert offset == -4.0

    def test_is_dst_active_winter(self) -> None:
        """No DST in January for Athens."""
        winter = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        assert is_dst_active("Europe/Athens", winter) is False

    def test_is_dst_active_summer(self) -> None:
        """DST active in July for Athens."""
        summer = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
        assert is_dst_active("Europe/Athens", summer) is True

    def test_dst_adjust_hour_winter(self) -> None:
        """In winter, no adjustment — hour stays the same."""
        winter = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        adjusted = dst_adjust_hour(22, "Europe/Athens", winter)
        assert adjusted == 22

    def test_dst_adjust_hour_summer_athens(self) -> None:
        """In summer, Athens DST=+1h → 22:00 winter → 21:00 actual UTC."""
        summer = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
        adjusted = dst_adjust_hour(22, "Europe/Athens", summer)
        assert adjusted == 21

    def test_dst_adjust_hour_summer_london(self) -> None:
        """In summer, London BST=+1h → 7:00 winter → 6:00 actual UTC."""
        summer = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
        adjusted = dst_adjust_hour(7, "Europe/London", summer)
        assert adjusted == 6

    def test_dst_adjust_hour_summer_ny(self) -> None:
        """In summer, NY EDT vs EST: DST offset=1h → 12:00 winter → 11:00 actual UTC."""
        summer = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
        adjusted = dst_adjust_hour(12, "America/New_York", summer)
        assert adjusted == 11

    def test_dst_adjust_hour_empty_timezone(self) -> None:
        """Empty timezone string returns base hour unchanged."""
        summer = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
        adjusted = dst_adjust_hour(22, "", summer)
        assert adjusted == 22

    def test_dst_transition_date_march(self) -> None:
        """European DST starts last Sunday of March 2026 = March 29.
        Before: UTC+2, After: UTC+3."""
        # March 28 (Saturday) — still winter
        before = datetime(2026, 3, 28, 12, 0, tzinfo=timezone.utc)
        assert is_dst_active("Europe/Athens", before) is False
        assert dst_adjust_hour(22, "Europe/Athens", before) == 22

        # March 30 (Monday) — now summer
        after = datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc)
        assert is_dst_active("Europe/Athens", after) is True
        assert dst_adjust_hour(22, "Europe/Athens", after) == 21

    def test_dst_transition_date_october(self) -> None:
        """European DST ends last Sunday of October 2026 = October 25.
        Before: UTC+3, After: UTC+2."""
        # October 24 (Saturday) — still summer
        before = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
        assert is_dst_active("Europe/Athens", before) is True
        assert dst_adjust_hour(22, "Europe/Athens", before) == 21

        # October 26 (Monday) — back to winter
        after = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
        assert is_dst_active("Europe/Athens", after) is False
        assert dst_adjust_hour(22, "Europe/Athens", after) == 22


# ── DST-Aware MarketHoursChecker Tests ──────────────────────────────────


class TestMarketHoursCheckerDST:
    """Tests for MarketHoursChecker with DST auto-adjustment enabled."""

    def _make_dst_config(self) -> MarketHoursConfig:
        return MarketHoursConfig(
            enabled=True,
            close_day="Friday",
            close_time_utc="22:00",
            open_day="Sunday",
            open_time_utc="22:00",
            dst_auto=True,
            server_timezone="Europe/Athens",
        )

    def test_winter_friday_close_at_22(self) -> None:
        """In winter, close time is 22:00 UTC (no adjustment)."""
        checker = MarketHoursChecker(self._make_dst_config())
        # Friday 21:59 — still open
        friday_before = datetime(2026, 1, 16, 21, 59, tzinfo=timezone.utc)
        assert checker.is_market_open(friday_before) is True
        # Friday 22:00 — closed
        friday_at = datetime(2026, 1, 16, 22, 0, tzinfo=timezone.utc)
        assert checker.is_market_open(friday_at) is False

    def test_summer_friday_close_at_21(self) -> None:
        """In summer, close time shifts to 21:00 UTC (Athens DST)."""
        checker = MarketHoursChecker(self._make_dst_config())
        # July 10, 2026 is a Friday
        # Friday 20:59 — still open
        friday_before = datetime(2026, 7, 10, 20, 59, tzinfo=timezone.utc)
        assert checker.is_market_open(friday_before) is True
        # Friday 21:00 — closed (DST-adjusted)
        friday_at = datetime(2026, 7, 10, 21, 0, tzinfo=timezone.utc)
        assert checker.is_market_open(friday_at) is False

    def test_summer_sunday_open_at_21(self) -> None:
        """In summer, open time shifts to 21:00 UTC (Athens DST)."""
        checker = MarketHoursChecker(self._make_dst_config())
        # July 12, 2026 is a Sunday
        # Sunday 20:59 — still closed
        sunday_before = datetime(2026, 7, 12, 20, 59, tzinfo=timezone.utc)
        assert checker.is_market_open(sunday_before) is False
        # Sunday 21:00 — open (DST-adjusted)
        sunday_at = datetime(2026, 7, 12, 21, 0, tzinfo=timezone.utc)
        assert checker.is_market_open(sunday_at) is True

    def test_summer_force_close_shifted(self) -> None:
        """Force close trigger also shifts with DST."""
        config = self._make_dst_config()
        config.force_close_before_weekend = True
        config.force_close_minutes_before = 15
        checker = MarketHoursChecker(config)

        # July 10, 2026 is a Friday. Summer close = 21:00 UTC.
        # Force close at 20:45 UTC
        friday_2044 = datetime(2026, 7, 10, 20, 44, tzinfo=timezone.utc)
        assert checker.should_force_close(friday_2044) is False

        friday_2046 = datetime(2026, 7, 10, 20, 46, tzinfo=timezone.utc)
        assert checker.should_force_close(friday_2046) is True

    def test_seconds_until_open_summer(self) -> None:
        """In summer, seconds_until_open uses DST-adjusted open time."""
        checker = MarketHoursChecker(self._make_dst_config())
        # Saturday July 11, 2026 noon. Next open = Sunday 21:00 UTC.
        saturday = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)
        seconds = checker.seconds_until_open(saturday)
        # ~33 hours from Sat 12:00 to Sun 21:00
        expected_approx = 33 * 3600
        assert abs(seconds - expected_approx) < 60

    def test_no_dst_when_disabled(self) -> None:
        """When dst_auto=False, summer times remain at winter baseline."""
        config = MarketHoursConfig(
            enabled=True,
            close_day="Friday",
            close_time_utc="22:00",
            dst_auto=False,
        )
        checker = MarketHoursChecker(config)
        # In summer, close should still be 22:00 (no DST adjustment)
        friday_2159 = datetime(2026, 7, 10, 21, 59, tzinfo=timezone.utc)
        assert checker.is_market_open(friday_2159) is True
        friday_2200 = datetime(2026, 7, 10, 22, 0, tzinfo=timezone.utc)
        assert checker.is_market_open(friday_2200) is False


# ── DST-Aware SessionCadence Tests ──────────────────────────────────────


class TestSessionCadenceDST:
    """Tests for SessionCadence with DST auto-adjustment."""

    def _make_dst_config(self) -> SchedulerConfig:
        return SchedulerConfig(
            session_aware_enabled=True,
            active_session_interval_seconds=3600,
            quiet_session_interval_seconds=14400,
            london_open_utc=7,
            london_close_utc=16,
            ny_open_utc=12,
            ny_close_utc=21,
            session_dst_auto=True,
            london_timezone="Europe/London",
            ny_timezone="America/New_York",
        )

    def test_winter_london_session(self) -> None:
        """In winter, London session is 07:00-16:00 UTC (no DST)."""
        cadence = SessionCadence(self._make_dst_config())
        # January Wednesday 08:00 UTC — London session
        winter_london = datetime(2026, 1, 14, 8, 0, tzinfo=timezone.utc)
        assert cadence.is_active_session(winter_london) is True
        assert cadence.current_session_name(winter_london) == "London"

    def test_summer_london_session_shifted(self) -> None:
        """In summer (BST), London opens at 06:00 UTC, closes at 15:00 UTC."""
        cadence = SessionCadence(self._make_dst_config())
        # July Wednesday 06:30 UTC — London session (shifted by BST)
        summer_london = datetime(2026, 7, 15, 6, 30, tzinfo=timezone.utc)
        assert cadence.is_active_session(summer_london) is True
        assert cadence.current_session_name(summer_london) == "London"

        # 05:30 UTC — before shifted London open
        before_london = datetime(2026, 7, 15, 5, 30, tzinfo=timezone.utc)
        assert cadence.current_session_name(before_london) == "Off-hours"

    def test_summer_ny_session_shifted(self) -> None:
        """In summer (EDT), NY opens at 11:00 UTC, closes at 20:00 UTC."""
        cadence = SessionCadence(self._make_dst_config())
        # July Wednesday 11:30 UTC — NY session (shifted by EDT)
        summer_ny = datetime(2026, 7, 15, 11, 30, tzinfo=timezone.utc)
        assert cadence.is_active_session(summer_ny) is True

        # 20:30 UTC — after shifted NY close
        after_ny = datetime(2026, 7, 15, 20, 30, tzinfo=timezone.utc)
        assert cadence.current_session_name(after_ny) == "Off-hours"

    def test_summer_overlap_shifted(self) -> None:
        """In summer, London/NY overlap shifts to ~11:00-15:00 UTC."""
        cadence = SessionCadence(self._make_dst_config())
        # July Wednesday 12:00 UTC — overlap zone
        overlap = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
        assert cadence.current_session_name(overlap) == "London/NY Overlap"

    def test_no_dst_session_when_disabled(self) -> None:
        """When session_dst_auto=False, session hours use winter baseline."""
        config = SchedulerConfig(
            session_aware_enabled=True,
            london_open_utc=7,
            london_close_utc=16,
            session_dst_auto=False,
        )
        cadence = SessionCadence(config)
        # In summer, 06:30 should be off-hours (no DST shift)
        summer = datetime(2026, 7, 15, 6, 30, tzinfo=timezone.utc)
        assert cadence.current_session_name(summer) == "Off-hours"

    def test_interval_active_session(self) -> None:
        """Active session returns active interval."""
        cadence = SessionCadence(self._make_dst_config())
        winter_london = datetime(2026, 1, 14, 10, 0, tzinfo=timezone.utc)
        assert cadence.get_scanner_interval(winter_london) == 3600

    def test_interval_quiet_session(self) -> None:
        """Off-hours returns quiet interval."""
        cadence = SessionCadence(self._make_dst_config())
        winter_offhours = datetime(2026, 1, 14, 3, 0, tzinfo=timezone.utc)
        assert cadence.get_scanner_interval(winter_offhours) == 14400
