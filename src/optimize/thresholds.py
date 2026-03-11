"""
Dynamic confidence thresholds for LLM gating.

Computes stepwise thresholds based on win rates and
applies per-symbol adjustments around the global baseline.

Usage:
    thresholds = compute_thresholds(global_win_rate=0.5, symbol_win_rates={})
"""

from loguru import logger

from src.optimize.optimization_state import Thresholds

# ── Exceptions ──────────────────────────────────────────────────────────────


class ThresholdsError(Exception):
    """Base exception for threshold calculation."""


# ── Helpers ────────────────────────────────────────────────────────────────


def _stepwise_threshold(win_rate: float) -> Thresholds:
    # v1.3.8: Cold-start tier — relax thresholds for symbols with < 20% win rate
    # (insufficient data to judge). Prevents over-filtering new/low-trade symbols.
    if win_rate < 0.20:
        return Thresholds(min_confidence="medium", min_blended_confidence=0.48)
    # v1.3.9a: Relaxed losing-symbol tier (0.60→0.52) — previous value was too
    # restrictive: medium-confidence signals with score ≥0.3 produced blended ≈0.48
    # which never cleared 0.60.  Net effect: system opened only 1 position in 13.5h.
    if win_rate < 0.45:
        return Thresholds(min_confidence="medium", min_blended_confidence=0.52)
    if win_rate > 0.55:
        return Thresholds(min_confidence="low", min_blended_confidence=0.45)
    return Thresholds(min_confidence="medium", min_blended_confidence=0.50)


def _adjust_blended(base: float, delta: float) -> float:
    return round(max(0.30, min(0.80, base - delta)), 2)


# ── Public API ──────────────────────────────────────────────────────────────


def compute_thresholds(
    *,
    global_win_rate: float,
    symbol_win_rates: dict[str, float],
    inactive_days: dict[str, int] | None = None,
) -> dict[str, Thresholds]:
    """Compute global and per-symbol thresholds.

    Args:
        global_win_rate: Overall win rate (0.0-1.0).
        symbol_win_rates: Per-symbol win rates.

    Returns:
        Dict containing "global" and per-symbol Thresholds.
    """
    global_threshold = _stepwise_threshold(global_win_rate)
    result: dict[str, Thresholds] = {"global": global_threshold}

    for symbol, win_rate in symbol_win_rates.items():
        delta = win_rate - global_win_rate
        # v1.3.9a: Reduced per-symbol adjustment (0.05→0.03) to avoid over-penalizing
        adj = 0.03 if delta >= 0.05 else -0.03 if delta <= -0.05 else 0.0

        # H3: Decay per-symbol adjustment for inactive symbols.
        # After N inactive days, the harmful adjustment (positive adj value, which
        # raises thresholds via base - adj) decays toward zero.
        # Decay rate: reduce |adj| by 50% per inactive day, fully zeroed at 3+ days.
        if inactive_days and symbol in inactive_days:
            days_inactive = inactive_days[symbol]
            if days_inactive > 0 and adj < 0:
                decay_factor = max(0.0, 1.0 - (days_inactive / 3.0))
                adj = round(adj * decay_factor, 4)
                if days_inactive >= 3:
                    logger.info(
                        "Threshold decay: {} inactive {}d, adjustment zeroed",
                        symbol,
                        days_inactive,
                    )
        result[symbol] = Thresholds(
            min_confidence=global_threshold.min_confidence,
            min_blended_confidence=_adjust_blended(global_threshold.min_blended_confidence, adj),
        )

    logger.debug(
        "Thresholds: computed global={} with {} symbol overrides",
        global_threshold.min_confidence,
        len(result) - 1,
    )
    return result
