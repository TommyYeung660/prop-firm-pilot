"""
DST (Daylight Saving Time) utilities — auto-detect timezone offsets for FX trading.

FX market hours shift with DST:
- E8 Markets server: Europe/Athens (UTC+2 winter, UTC+3 summer)
- London session: Europe/London (UTC+0 winter, UTC+1 summer)
- New York session: America/New_York (UTC-5 winter, UTC-4 summer)

European DST: Last Sunday of March → Last Sunday of October
US DST: Second Sunday of March → First Sunday of November

This module provides functions to calculate DST-aware UTC offsets for any
given datetime, so market hours and session boundaries auto-adjust.

Usage:
    from src.scheduler.dst_utils import get_utc_offset_hours, dst_adjusted_utc_hour

    # Get current UTC offset for a timezone
    offset = get_utc_offset_hours("Europe/London", now_utc)  # 0 or 1

    # Convert a "local nominal hour" to actual UTC hour
    utc_hour = dst_adjusted_utc_hour(22, "Europe/Athens", now_utc)  # 20 (summer) or 19 (winter)?
    # No — we do the inverse: given a winter-UTC reference, adjust for DST.
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from loguru import logger

# ── Well-known FX trading timezones ─────────────────────────────────────

# E8 Markets server timezone (MetaTrader server time)
E8_SERVER_TZ = "Europe/Athens"

# Trading session timezones
LONDON_TZ = "Europe/London"
NEW_YORK_TZ = "America/New_York"

# ── Core Functions ──────────────────────────────────────────────────────


def get_utc_offset_hours(tz_name: str, at_utc: datetime) -> float:
    """Return the UTC offset in hours for a timezone at a specific UTC time.

    Args:
        tz_name: IANA timezone name (e.g., "Europe/London", "America/New_York").
        at_utc: A UTC datetime to check DST status.

    Returns:
        UTC offset in hours (e.g., 0.0, 1.0, -5.0, -4.0).
    """
    tz = ZoneInfo(tz_name)
    # Create a timezone-aware datetime in the target zone
    if at_utc.tzinfo is None:
        at_utc = at_utc.replace(tzinfo=timezone.utc)
    local_dt = at_utc.astimezone(tz)
    offset = local_dt.utcoffset()
    if offset is None:
        return 0.0
    return offset.total_seconds() / 3600


def is_dst_active(tz_name: str, at_utc: datetime) -> bool:
    """Return True if DST is active for the given timezone at the specified UTC time.

    Args:
        tz_name: IANA timezone name.
        at_utc: A UTC datetime to check.

    Returns:
        True if DST is currently active.
    """
    tz = ZoneInfo(tz_name)
    if at_utc.tzinfo is None:
        at_utc = at_utc.replace(tzinfo=timezone.utc)
    local_dt = at_utc.astimezone(tz)
    dst_offset = local_dt.dst()
    if dst_offset is None:
        return False
    return dst_offset.total_seconds() > 0


def dst_adjust_hour(base_utc_hour: int, tz_name: str, at_utc: datetime) -> int:
    """Adjust a winter-time UTC hour for DST.

    Config files store market hours as "winter UTC" (no DST). When DST is active
    in the relevant timezone, the actual UTC hour shifts earlier by the DST offset.

    Example:
        - E8 server close = Friday 22:00 UTC (winter, Athens UTC+2 = midnight local)
        - In summer (Athens UTC+3), midnight local = 21:00 UTC
        - dst_adjust_hour(22, "Europe/Athens", summer_dt) → 21

    Args:
        base_utc_hour: The "winter time" UTC hour from config (0-23).
        tz_name: The timezone that determines DST (e.g., "Europe/Athens").
        at_utc: Current UTC datetime to check DST status.

    Returns:
        DST-adjusted UTC hour (0-23).
    """
    if not tz_name:
        return base_utc_hour

    tz = ZoneInfo(tz_name)
    if at_utc.tzinfo is None:
        at_utc = at_utc.replace(tzinfo=timezone.utc)

    local_dt = at_utc.astimezone(tz)
    dst_offset = local_dt.dst()
    if dst_offset is None or dst_offset.total_seconds() == 0:
        # No DST active — use base hour as-is
        return base_utc_hour

    # DST is active: shift UTC hour earlier by DST amount
    dst_hours = int(dst_offset.total_seconds() / 3600)
    adjusted = (base_utc_hour - dst_hours) % 24

    logger.debug(
        "DST adjust: {}:00 UTC (winter) → {}:00 UTC (DST active in {}, offset=+{}h)",
        base_utc_hour,
        adjusted,
        tz_name,
        dst_hours,
    )

    return adjusted


def get_session_hours_utc(
    tz_name: str,
    local_open_hour: int,
    local_close_hour: int,
    at_utc: datetime,
) -> tuple[int, int]:
    """Convert local session hours to UTC, accounting for DST.

    Trading sessions are defined in local time (e.g., London opens at 08:00 local).
    This function converts those to UTC hours based on current DST status.

    Args:
        tz_name: IANA timezone for the session (e.g., "Europe/London").
        local_open_hour: Session open hour in local time (0-23).
        local_close_hour: Session close hour in local time (0-23).
        at_utc: Current UTC datetime.

    Returns:
        Tuple of (open_utc_hour, close_utc_hour).
    """
    offset = get_utc_offset_hours(tz_name, at_utc)
    offset_int = int(offset)

    open_utc = (local_open_hour - offset_int) % 24
    close_utc = (local_close_hour - offset_int) % 24

    return open_utc, close_utc


def log_dst_status(at_utc: datetime) -> None:
    """Log current DST status for all relevant FX timezones."""
    for name, tz_name in [
        ("E8 Server (Athens)", E8_SERVER_TZ),
        ("London", LONDON_TZ),
        ("New York", NEW_YORK_TZ),
    ]:
        active = is_dst_active(tz_name, at_utc)
        offset = get_utc_offset_hours(tz_name, at_utc)
        logger.info(
            "DST status — {}: {} (UTC{:+.0f})",
            name,
            "Summer Time" if active else "Winter Time",
            offset,
        )
