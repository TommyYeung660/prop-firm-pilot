"""
Market hours checker — determines if FX market is open or closed.

Handles weekend closure for FX markets with configurable close/open times.
Supports DST auto-adjustment when dst_auto is enabled in config.
Used by the Scheduler to pause trading loops during weekends.

Usage:
    checker = MarketHoursChecker(config.scheduler.market_hours)
    if not checker.is_market_open(now_utc):
        sleep_seconds = checker.seconds_until_open(now_utc)
"""

from datetime import datetime, timedelta

from loguru import logger

from src.config import MarketHoursConfig
from src.scheduler.dst_utils import dst_adjust_hour, is_dst_active, log_dst_status

# Day name to weekday number (Monday=0, Sunday=6)
_DAY_MAP: dict[str, int] = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}


class MarketHoursChecker:
    """Checks whether FX market is currently open based on config.

    When dst_auto is enabled, close/open times are automatically adjusted
    for DST based on the configured server_timezone.

    Usage:
        checker = MarketHoursChecker(market_hours_config)
        if not checker.is_market_open(datetime.now(timezone.utc)):
            wait = checker.seconds_until_open(datetime.now(timezone.utc))
    """

    def __init__(self, config: MarketHoursConfig) -> None:
        self._config = config
        self._close_weekday = _DAY_MAP.get(config.close_day, 4)  # Default Friday
        self._open_weekday = _DAY_MAP.get(config.open_day, 6)  # Default Sunday

        # Parse winter-baseline hours from config
        close_parts = config.close_time_utc.split(":")
        self._base_close_hour = int(close_parts[0])
        self._base_close_minute = int(close_parts[1]) if len(close_parts) > 1 else 0
        open_parts = config.open_time_utc.split(":")
        self._base_open_hour = int(open_parts[0])
        self._base_open_minute = int(open_parts[1]) if len(open_parts) > 1 else 0

    def _get_close_hour(self, now: datetime) -> int:
        """Return DST-adjusted close hour for the given time."""
        if self._config.dst_auto:
            return dst_adjust_hour(self._base_close_hour, self._config.server_timezone, now)
        return self._base_close_hour

    def _get_open_hour(self, now: datetime) -> int:
        """Return DST-adjusted open hour for the given time."""
        if self._config.dst_auto:
            return dst_adjust_hour(self._base_open_hour, self._config.server_timezone, now)
        return self._base_open_hour

    def is_market_open(self, now: datetime) -> bool:
        """Return True if the FX market is currently open.

        Market is CLOSED from close_day close_time through open_day open_time.
        When dst_auto is enabled, times auto-adjust for DST.
        """
        if not self._config.enabled:
            return True

        close_hour = self._get_close_hour(now)
        open_hour = self._get_open_hour(now)
        close_minutes = close_hour * 60 + self._base_close_minute
        open_minutes = open_hour * 60 + self._base_open_minute

        weekday = now.weekday()  # Monday=0, Sunday=6
        time_minutes = now.hour * 60 + now.minute

        # Saturday is always closed
        if weekday == 5:  # Saturday
            return False

        # Friday after close time
        if weekday == self._close_weekday and time_minutes >= close_minutes:
            return False

        # Sunday before open time
        if weekday == self._open_weekday and time_minutes < open_minutes:
            return False

        return True

    def should_force_close(self, now: datetime) -> bool:
        """Return True if we should force-close all positions before weekend.

        Triggers `force_close_minutes_before` minutes before market close.
        Only on close_day, only if force_close_before_weekend is enabled.
        """
        if not self._config.enabled or not self._config.force_close_before_weekend:
            return False

        weekday = now.weekday()
        if weekday != self._close_weekday:
            return False

        close_hour = self._get_close_hour(now)
        close_minutes = close_hour * 60 + self._base_close_minute
        time_minutes = now.hour * 60 + now.minute
        trigger_minutes = close_minutes - self._config.force_close_minutes_before

        return time_minutes >= trigger_minutes

    def seconds_until_open(self, now: datetime) -> float:
        """Calculate seconds until the next market open time.

        Returns 0 if market is currently open.
        """
        if self.is_market_open(now):
            return 0.0

        open_hour = self._get_open_hour(now)
        # Find next open_day at open_time
        target = now.replace(hour=open_hour, minute=self._base_open_minute, second=0, microsecond=0)

        # Move to next open_day
        days_ahead = self._open_weekday - now.weekday()
        if days_ahead < 0:
            days_ahead += 7
        if days_ahead == 0 and now >= target:
            days_ahead += 7

        target = target + timedelta(days=days_ahead)
        return max(0.0, (target - now).total_seconds())

    def log_current_status(self, now: datetime) -> None:
        """Log current market hours status including DST info."""
        if self._config.dst_auto:
            log_dst_status(now)
            dst_active = is_dst_active(self._config.server_timezone, now)
            close_h = self._get_close_hour(now)
            open_h = self._get_open_hour(now)
            logger.info(
                "Market hours (DST {}): close {}:{:02d} UTC, open {}:{:02d} UTC",
                "active" if dst_active else "inactive",
                close_h,
                self._base_close_minute,
                open_h,
                self._base_open_minute,
            )
        else:
            logger.info(
                "Market hours (static): close {}:{:02d} UTC, open {}:{:02d} UTC",
                self._base_close_hour,
                self._base_close_minute,
                self._base_open_hour,
                self._base_open_minute,
            )
