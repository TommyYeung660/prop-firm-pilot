"""Tests for MarketHoursConfig and weekend market closure logic."""

from datetime import datetime, timezone

from src.config import AppConfig, MarketHoursConfig, SchedulerConfig
from src.scheduler.market_hours import MarketHoursChecker


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
