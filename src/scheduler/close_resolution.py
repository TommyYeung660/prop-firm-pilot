"""
Close resolution path helper for trade retrospection quality.

Determines which data source provided the final PnL/close data
when a position is detected as closed.
"""


def determine_resolution_path(
    *,
    matched: bool,
    used_execution_meta: bool,
    used_best_day: bool,
    used_reeval: bool,
    used_last_known: bool,
) -> str:
    """Return a label for the data resolution path used.

    Args:
        matched: Broker API returned the closed position.
        used_execution_meta: execution_meta provided fallback data.
        used_best_day: Best Day Rule close provided PnL.
        used_reeval: Re-evaluation close provided PnL.
        used_last_known: Last-known polled PnL was used.

    Returns:
        String label identifying which source provided the data.
    """
    if matched:
        return "broker_api"
    if used_best_day:
        return "best_day_close"
    if used_reeval:
        return "reeval_close"
    if used_execution_meta:
        return "execution_meta"
    if used_last_known:
        return "last_known_profit"
    return "unknown"
