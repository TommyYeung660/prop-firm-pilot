"""Tests for MarketHoursConfig and weekend market closure logic."""

from src.config import AppConfig, MarketHoursConfig, SchedulerConfig


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
