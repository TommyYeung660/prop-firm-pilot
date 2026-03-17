"""
Execution-side bounded capital allocation helpers.

Redistributes the existing nominal entry-risk budget across fewer open
positions without weakening compliance limits or requiring a full
portfolio-construction engine.

Usage:
    allocator = BoundedCapitalAllocator(config.execution)
    decision = allocator.allocate_entry_risk(open_positions=1, scanner_confidence="high")
"""

from dataclasses import asdict, dataclass
from typing import Any

from src.config import ExecutionConfig

CONFIDENCE_UPLIFT_FACTORS: dict[str, float] = {
    "high": 1.0,
    "medium": 0.5,
    "low": 0.0,
}


@dataclass(frozen=True)
class CapitalAllocationDecision:
    """Audit-friendly result of bounded entry-risk allocation."""

    effective_risk_pct: float
    default_risk_pct: float
    bounded_cap_pct: float
    portfolio_budget_pct: float
    slot_budget_pct: float
    confidence_factor: float
    uplift_applied: bool
    open_positions: int
    positions_after_entry: int
    scanner_confidence: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


class BoundedCapitalAllocator:
    """Compute bounded per-trade risk for sparse portfolios."""

    def __init__(self, config: ExecutionConfig) -> None:
        self._config = config

    def allocate_entry_risk(
        self,
        *,
        open_positions: int,
        scanner_confidence: str,
    ) -> CapitalAllocationDecision:
        """Return effective risk percent for the next entry.

        Rules:
        - Total nominal entry budget remains anchored to
          `default_risk_pct * max_positions`.
        - Sparse portfolios may concentrate that budget onto fewer trades.
        - Any single trade remains capped by `max_risk_pct`.
        - Confidence controls how much of the bounded uplift headroom is used.
        """
        default_risk_pct = max(float(self._config.default_risk_pct), 0.0)
        max_risk_pct = max(float(self._config.max_risk_pct), 0.0)
        max_positions = max(int(self._config.max_positions), 1)
        safe_open_positions = max(int(open_positions), 0)
        positions_after_entry = min(safe_open_positions + 1, max_positions)

        portfolio_budget_pct = default_risk_pct * max_positions
        slot_budget_pct = portfolio_budget_pct / positions_after_entry

        if max_risk_pct <= default_risk_pct:
            bounded_cap_pct = default_risk_pct
        else:
            bounded_cap_pct = max(default_risk_pct, min(slot_budget_pct, max_risk_pct))

        normalized_confidence = str(scanner_confidence or "").strip().lower()
        confidence_factor = CONFIDENCE_UPLIFT_FACTORS.get(normalized_confidence, 0.0)
        uplift_headroom = max(bounded_cap_pct - default_risk_pct, 0.0)
        effective_risk_pct = default_risk_pct + (confidence_factor * uplift_headroom)

        return CapitalAllocationDecision(
            effective_risk_pct=effective_risk_pct,
            default_risk_pct=default_risk_pct,
            bounded_cap_pct=bounded_cap_pct,
            portfolio_budget_pct=portfolio_budget_pct,
            slot_budget_pct=slot_budget_pct,
            confidence_factor=confidence_factor,
            uplift_applied=effective_risk_pct > default_risk_pct,
            open_positions=safe_open_positions,
            positions_after_entry=positions_after_entry,
            scanner_confidence=normalized_confidence,
        )
