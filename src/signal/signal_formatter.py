"""
Signal formatter — transforms scanner signals for downstream consumers.

Primarily converts between scanner output format and TradingAgents input format.
Also handles signal filtering and enrichment.
"""

from typing import Any, Literal

from loguru import logger


def classify_signal_strength(
    score: float,
    score_gap: float,
    confidence: str,
) -> Literal["STRONG", "MODERATE", "WEAK"]:
    """Classify overall signal strength from scanner metrics.

    Args:
        score: Model prediction score.
        score_gap: Gap to next-ranked instrument.
        confidence: Scanner confidence level (high/medium/low).

    Returns:
        Signal strength classification.
    """
    if confidence == "high" and score_gap > 0.03:
        return "STRONG"
    if confidence in ("high", "medium") and score_gap > 0.01:
        return "MODERATE"
    return "WEAK"


def filter_actionable_signals(
    signals: list[dict[str, Any]],
    min_score: float = 0.0,
    min_confidence: str = "low",
    max_signals: int = 3,
) -> list[dict[str, Any]]:
    """Filter signals to only actionable ones.

    Args:
        signals: List of signal dicts from scanner.
        min_score: Minimum score threshold.
        min_confidence: Minimum confidence level.
        max_signals: Maximum number of signals to return.

    Returns:
        Filtered and sorted signal list.
    """
    confidence_order = {"high": 3, "medium": 2, "low": 1}
    min_conf_val = confidence_order.get(min_confidence, 1)

    filtered = []
    for signal in signals:
        score = signal.get("score", 0.0)
        conf = signal.get("confidence", "low")
        conf_val = confidence_order.get(conf, 1)

        if score >= min_score and conf_val >= min_conf_val:
            filtered.append(signal)

    # Sort by score descending
    filtered.sort(key=lambda s: s.get("score", 0.0), reverse=True)

    result = filtered[:max_signals]
    logger.debug(
        "SignalFormatter: filtered {}/{} signals (min_score={}, min_conf={})",
        len(result),
        len(signals),
        min_score,
        min_confidence,
    )
    return result


def enrich_with_side_suggestion(
    signal: dict[str, Any],
    score_threshold: float = 0.5,
) -> dict[str, Any]:
    """Add a suggested side (BUY/SELL) based on score.

    Scanner scores > threshold suggest BUY (model predicts upward movement).
    Scanner scores < (1 - threshold) suggest SELL.
    Otherwise HOLD.

    Note: TradingAgents makes the final BUY/SELL/HOLD decision.
    This is only a pre-filter hint.
    """
    score = signal.get("score", 0.5)
    enriched = dict(signal)

    if score > score_threshold:
        enriched["suggested_side"] = "BUY"
    elif score < (1 - score_threshold):
        enriched["suggested_side"] = "SELL"
    else:
        enriched["suggested_side"] = "HOLD"

    return enriched
