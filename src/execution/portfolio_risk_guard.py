"""
Execution-side portfolio risk guard primitives.

Computes bounded portfolio risk admission checks for the next entry intent
without any broker I/O. This module is intentionally fail-closed.

Usage:
    guard = PortfolioRiskGuard(config.execution)
    decision = guard.evaluate_next_entry(
        next_symbol="EURUSD",
        next_side="BUY",
        next_risk_pct=0.01,
        open_positions=[],
    )
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from src.config import ExecutionConfig

REASON_ALLOWED = "portfolio_risk.allowed"
REASON_INVALID_INPUT = "portfolio_risk.invalid_input"
REASON_TOTAL_OPEN_RISK_EXCEEDED = "portfolio_risk.total_open_risk_exceeded"
REASON_SAME_DIRECTION_LIMIT_EXCEEDED = "portfolio_risk.same_direction_limit_exceeded"
REASON_CURRENCY_EXPOSURE_EXCEEDED = "portfolio_risk.currency_exposure_exceeded"


@dataclass(frozen=True)
class OpenPositionRiskSnapshot:
    """Minimal open-position risk payload used by the risk guard."""

    symbol: str
    side: Literal["BUY", "SELL"]
    open_risk_pct: float


@dataclass(frozen=True)
class PortfolioRiskDecision:
    """Result of evaluating whether the next entry is portfolio-safe."""

    allowed: bool
    reason_code: str
    projected_total_open_risk_pct: float
    projected_same_direction_positions: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable decision payload."""
        return asdict(self)


class PortfolioRiskGuard:
    """Fail-closed portfolio risk classifier for next-entry admission."""

    def __init__(self, config: ExecutionConfig) -> None:
        self._config = config

    def evaluate_next_entry(
        self,
        *,
        next_symbol: str,
        next_side: str,
        next_risk_pct: float,
        open_positions: list[OpenPositionRiskSnapshot],
    ) -> PortfolioRiskDecision:
        """Evaluate whether a new entry respects portfolio-level risk limits."""
        normalized_side = str(next_side or "").strip().upper()
        if normalized_side not in {"BUY", "SELL"}:
            return self._reject(
                reason_code=REASON_INVALID_INPUT,
                projected_total_open_risk_pct=0.0,
                projected_same_direction_positions=0,
                details={"error": "invalid_next_side", "next_side": next_side},
            )

        safe_next_risk_pct = float(next_risk_pct)
        if safe_next_risk_pct <= 0.0:
            return self._reject(
                reason_code=REASON_INVALID_INPUT,
                projected_total_open_risk_pct=0.0,
                projected_same_direction_positions=0,
                details={"error": "invalid_next_risk_pct", "next_risk_pct": safe_next_risk_pct},
            )

        next_ccy_pair = self._extract_currency_pair(next_symbol)
        if next_ccy_pair is None:
            return self._reject(
                reason_code=REASON_INVALID_INPUT,
                projected_total_open_risk_pct=0.0,
                projected_same_direction_positions=0,
                details={"error": "invalid_next_symbol", "next_symbol": next_symbol},
            )

        normalized_open_positions: list[OpenPositionRiskSnapshot] = []
        for position in open_positions:
            if position.side not in {"BUY", "SELL"}:
                return self._reject(
                    reason_code=REASON_INVALID_INPUT,
                    projected_total_open_risk_pct=0.0,
                    projected_same_direction_positions=0,
                    details={"error": "invalid_open_position_side", "side": position.side},
                )
            if position.open_risk_pct < 0.0:
                return self._reject(
                    reason_code=REASON_INVALID_INPUT,
                    projected_total_open_risk_pct=0.0,
                    projected_same_direction_positions=0,
                    details={
                        "error": "invalid_open_position_risk_pct",
                        "open_risk_pct": position.open_risk_pct,
                    },
                )
            if self._extract_currency_pair(position.symbol) is None:
                return self._reject(
                    reason_code=REASON_INVALID_INPUT,
                    projected_total_open_risk_pct=0.0,
                    projected_same_direction_positions=0,
                    details={"error": "invalid_open_position_symbol", "symbol": position.symbol},
                )
            normalized_open_positions.append(position)

        existing_open_risk_pct = (
            sum(position.open_risk_pct for position in normalized_open_positions)
            if self._config.reserve_risk_for_open_positions
            else 0.0
        )
        projected_total_open_risk_pct = existing_open_risk_pct + safe_next_risk_pct
        max_total_open_risk_pct = max(float(self._config.max_total_open_risk_pct), 0.0)
        if projected_total_open_risk_pct > max_total_open_risk_pct:
            return self._reject(
                reason_code=REASON_TOTAL_OPEN_RISK_EXCEEDED,
                projected_total_open_risk_pct=projected_total_open_risk_pct,
                projected_same_direction_positions=0,
                details={
                    "max_total_open_risk_pct": max_total_open_risk_pct,
                    "projected_total_open_risk_pct": projected_total_open_risk_pct,
                },
            )

        projected_same_direction_positions = (
            sum(1 for position in normalized_open_positions if position.side == normalized_side) + 1
        )
        max_same_direction_positions = max(int(self._config.max_same_direction_positions), 0)
        if projected_same_direction_positions > max_same_direction_positions:
            return self._reject(
                reason_code=REASON_SAME_DIRECTION_LIMIT_EXCEEDED,
                projected_total_open_risk_pct=projected_total_open_risk_pct,
                projected_same_direction_positions=projected_same_direction_positions,
                details={
                    "max_same_direction_positions": max_same_direction_positions,
                    "projected_same_direction_positions": projected_same_direction_positions,
                    "side": normalized_side,
                },
            )

        currency_counts: dict[str, int] = {}
        for position in normalized_open_positions:
            base, quote = self._extract_currency_pair(position.symbol) or ("", "")
            currency_counts[base] = currency_counts.get(base, 0) + 1
            currency_counts[quote] = currency_counts.get(quote, 0) + 1
        next_base, next_quote = next_ccy_pair
        currency_counts[next_base] = currency_counts.get(next_base, 0) + 1
        currency_counts[next_quote] = currency_counts.get(next_quote, 0) + 1

        max_currency_exposure_per_ccy = max(int(self._config.max_currency_exposure_per_ccy), 0)
        violating_ccys = sorted(
            ccy for ccy, count in currency_counts.items() if count > max_currency_exposure_per_ccy
        )
        if violating_ccys:
            return self._reject(
                reason_code=REASON_CURRENCY_EXPOSURE_EXCEEDED,
                projected_total_open_risk_pct=projected_total_open_risk_pct,
                projected_same_direction_positions=projected_same_direction_positions,
                details={
                    "max_currency_exposure_per_ccy": max_currency_exposure_per_ccy,
                    "violating_currency": violating_ccys[0],
                    "currency_counts": currency_counts,
                },
            )

        return PortfolioRiskDecision(
            allowed=True,
            reason_code=REASON_ALLOWED,
            projected_total_open_risk_pct=projected_total_open_risk_pct,
            projected_same_direction_positions=projected_same_direction_positions,
            details={},
        )

    @staticmethod
    def _extract_currency_pair(symbol: str) -> tuple[str, str] | None:
        sanitized = "".join(ch for ch in str(symbol or "").upper() if ch.isalpha())
        if len(sanitized) < 6:
            return None
        return sanitized[:3], sanitized[3:6]

    @staticmethod
    def _reject(
        *,
        reason_code: str,
        projected_total_open_risk_pct: float,
        projected_same_direction_positions: int,
        details: dict[str, Any],
    ) -> PortfolioRiskDecision:
        return PortfolioRiskDecision(
            allowed=False,
            reason_code=reason_code,
            projected_total_open_risk_pct=projected_total_open_risk_pct,
            projected_same_direction_positions=projected_same_direction_positions,
            details=details,
        )
