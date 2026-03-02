"""
Session-aware cadence calculator — adjusts scanner interval by trading session.

FX markets have distinct sessions with varying liquidity:
- London (08:00–17:00 local): High liquidity
- New York (09:30–16:00 local): High liquidity
- London/NY overlap: Highest liquidity
- Asia/Off-hours: Lower liquidity

Config stores session hours as UTC winter-baseline. When session_dst_auto is
enabled, hours auto-adjust for DST using Europe/London and America/New_York
timezones.

During active sessions, the scanner runs more frequently to capture
opportunities. During quiet hours, it runs less frequently.

Usage:
    cadence = SessionCadence(scheduler_config)
    interval = cadence.get_scanner_interval(now_utc)
"""

from datetime import datetime

from loguru import logger

from src.config import SchedulerConfig
from src.scheduler.dst_utils import dst_adjust_hour


class SessionCadence:
    """Calculates scanner interval based on current FX trading session.

    When session_dst_auto is enabled, London and NY session hours
    auto-adjust for DST using their respective timezones.

    Usage:
        cadence = SessionCadence(config)
        interval_seconds = cadence.get_scanner_interval(datetime.now(timezone.utc))
    """

    def __init__(self, config: SchedulerConfig) -> None:
        self._config = config

    def _get_session_hours(self, now: datetime) -> tuple[int, int, int, int]:
        """Return (london_open, london_close, ny_open, ny_close) in UTC.

        If session_dst_auto is enabled, adjusts for DST.
        """
        if self._config.session_dst_auto:
            london_open = dst_adjust_hour(
                self._config.london_open_utc, self._config.london_timezone, now
            )
            london_close = dst_adjust_hour(
                self._config.london_close_utc, self._config.london_timezone, now
            )
            ny_open = dst_adjust_hour(self._config.ny_open_utc, self._config.ny_timezone, now)
            ny_close = dst_adjust_hour(self._config.ny_close_utc, self._config.ny_timezone, now)
            return london_open, london_close, ny_open, ny_close

        return (
            self._config.london_open_utc,
            self._config.london_close_utc,
            self._config.ny_open_utc,
            self._config.ny_close_utc,
        )

    def is_active_session(self, now: datetime) -> bool:
        """Return True if current time falls within London or NY session.

        Active = London session OR New York session (any overlap counts once).
        When session_dst_auto is enabled, session hours auto-adjust for DST.
        """
        hour = now.hour
        london_open, london_close, ny_open, ny_close = self._get_session_hours(now)

        in_london = london_open <= hour < london_close
        in_ny = ny_open <= hour < ny_close
        return in_london or in_ny

    def get_scanner_interval(self, now: datetime) -> int:
        """Return the appropriate scanner interval in seconds.

        If session-aware is disabled, returns the default scanner_interval_seconds.
        Otherwise, returns active or quiet interval based on session.
        """
        if not self._config.session_aware_enabled:
            return self._config.scanner_interval_seconds

        if self.is_active_session(now):
            return self._config.active_session_interval_seconds
        return self._config.quiet_session_interval_seconds

    def current_session_name(self, now: datetime) -> str:
        """Return human-readable session name for logging."""
        hour = now.hour
        london_open, london_close, ny_open, ny_close = self._get_session_hours(now)

        in_london = london_open <= hour < london_close
        in_ny = ny_open <= hour < ny_close

        if in_london and in_ny:
            return "London/NY Overlap"
        if in_london:
            return "London"
        if in_ny:
            return "New York"
        return "Off-hours"

    def log_session_hours(self, now: datetime) -> None:
        """Log current session hours including DST adjustments."""
        london_open, london_close, ny_open, ny_close = self._get_session_hours(now)
        dst_label = " (DST-adjusted)" if self._config.session_dst_auto else ""
        logger.info(
            "Session hours{}: London {:02d}:00-{:02d}:00 UTC, NY {:02d}:00-{:02d}:00 UTC",
            dst_label,
            london_open,
            london_close,
            ny_open,
            ny_close,
        )
