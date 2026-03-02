"""
Session-aware cadence calculator — adjusts scanner interval by trading session.

FX markets have distinct sessions with varying liquidity:
- London (07:00–16:00 UTC): High liquidity
- New York (12:00–21:00 UTC): High liquidity
- London/NY overlap (12:00–16:00 UTC): Highest liquidity
- Asia/Off-hours: Lower liquidity

During active sessions, the scanner runs more frequently to capture
opportunities. During quiet hours, it runs less frequently.

Usage:
    cadence = SessionCadence(scheduler_config)
    interval = cadence.get_scanner_interval(now_utc)
"""

from datetime import datetime

from src.config import SchedulerConfig


class SessionCadence:
    """Calculates scanner interval based on current FX trading session.

    Usage:
        cadence = SessionCadence(config)
        interval_seconds = cadence.get_scanner_interval(datetime.now(timezone.utc))
    """

    def __init__(self, config: SchedulerConfig) -> None:
        self._config = config

    def is_active_session(self, now: datetime) -> bool:
        """Return True if current time falls within London or NY session.

        Active = London session OR New York session (any overlap counts once).
        """
        hour = now.hour
        in_london = self._config.london_open_utc <= hour < self._config.london_close_utc
        in_ny = self._config.ny_open_utc <= hour < self._config.ny_close_utc
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
        in_london = self._config.london_open_utc <= hour < self._config.london_close_utc
        in_ny = self._config.ny_open_utc <= hour < self._config.ny_close_utc

        if in_london and in_ny:
            return "London/NY Overlap"
        if in_london:
            return "London"
        if in_ny:
            return "New York"
        return "Off-hours"
