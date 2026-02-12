"""
Decision formatter — converts TradingAgents decisions into
actionable trade parameters for the execution layer.
"""

from typing import Any, Dict, Literal

from loguru import logger


class FormattedDecision:
    """Trade decision with execution parameters."""

    def __init__(
        self,
        symbol: str,
        side: Literal["BUY", "SELL", "HOLD"],
        confidence_score: float,
        suggested_sl_pips: float,
        suggested_tp_pips: float,
        risk_reward_ratio: float,
        reasoning_summary: str = "",
    ) -> None:
        self.symbol = symbol
        self.side = side
        self.confidence_score = confidence_score
        self.suggested_sl_pips = suggested_sl_pips
        self.suggested_tp_pips = suggested_tp_pips
        self.risk_reward_ratio = risk_reward_ratio
        self.reasoning_summary = reasoning_summary

    @property
    def is_actionable(self) -> bool:
        return self.side in ("BUY", "SELL")

    def __repr__(self) -> str:
        return (
            f"FormattedDecision({self.symbol}, {self.side}, "
            f"conf={self.confidence_score:.2f}, RR={self.risk_reward_ratio:.1f})"
        )


# Default SL/TP pips by instrument
DEFAULT_SL_TP: Dict[str, Dict[str, float]] = {
    "EURUSD": {"sl_pips": 40, "tp_pips": 80},
    "GBPUSD": {"sl_pips": 50, "tp_pips": 100},
    "USDJPY": {"sl_pips": 45, "tp_pips": 90},
    "AUDUSD": {"sl_pips": 35, "tp_pips": 70},
    "XAUUSD": {"sl_pips": 150, "tp_pips": 300},
}


def format_decision(
    symbol: str,
    decision: str,
    scanner_score: float,
    scanner_confidence: str,
    agent_state: Dict[str, Any] | None = None,
) -> FormattedDecision:
    """Convert raw decision into execution-ready format.

    Args:
        symbol: FX pair.
        decision: "BUY", "SELL", or "HOLD" from TradingAgents.
        scanner_score: Score from qlib_market_scanner.
        scanner_confidence: "high", "medium", or "low".
        agent_state: Full final_state from TradingAgents (optional).

    Returns:
        FormattedDecision with SL/TP suggestions.
    """
    # Get default SL/TP for this instrument
    defaults = DEFAULT_SL_TP.get(symbol, {"sl_pips": 50, "tp_pips": 100})
    sl_pips = defaults["sl_pips"]
    tp_pips = defaults["tp_pips"]

    # Calculate confidence score (0-1)
    conf_map = {"high": 0.9, "medium": 0.6, "low": 0.3}
    conf_score = conf_map.get(scanner_confidence, 0.5)

    # Blend with scanner score
    blended_confidence = 0.6 * conf_score + 0.4 * scanner_score

    # Extract reasoning from agent state
    reasoning = ""
    if agent_state:
        reasoning = agent_state.get("risk_report", "")
        if not reasoning:
            reasoning = agent_state.get("summary", "")

    # Adjust SL/TP based on confidence
    if blended_confidence < 0.5:
        # Low confidence → tighter SL, smaller TP
        sl_pips *= 0.8
        tp_pips *= 0.7
    elif blended_confidence > 0.8:
        # High confidence → wider SL, larger TP
        sl_pips *= 1.1
        tp_pips *= 1.2

    rr_ratio = tp_pips / sl_pips if sl_pips > 0 else 0.0

    result = FormattedDecision(
        symbol=symbol,
        side=decision if decision in ("BUY", "SELL") else "HOLD",
        confidence_score=blended_confidence,
        suggested_sl_pips=round(sl_pips, 1),
        suggested_tp_pips=round(tp_pips, 1),
        risk_reward_ratio=round(rr_ratio, 2),
        reasoning_summary=reasoning[:500],
    )

    logger.debug(
        "DecisionFormatter: {} → {} (conf={:.2f}, SL={:.0f}p, TP={:.0f}p, RR={:.1f})",
        symbol,
        decision,
        blended_confidence,
        sl_pips,
        tp_pips,
        rr_ratio,
    )
    return result
