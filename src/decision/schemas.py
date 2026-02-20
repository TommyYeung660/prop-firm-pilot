"""
Trade intent and decision record schemas for the Hybrid EA+LLM pipeline.

TradeIntent represents a trade opportunity discovered by the scanner,
flowing through the decision state machine: pending → claimed → ready_for_exec
→ executing → opened/rejected/failed → closed.

DecisionRecord is an immutable audit trail linking intent to execution outcome.

Usage:
    intent = TradeIntent(trade_date="2026-02-16", symbol="EURUSD", scanner_score=0.85)
    record = DecisionRecord(intent_id=intent.id)
"""

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

# ── Status Type ─────────────────────────────────────────────────────────────

IntentStatus = Literal[
    "pending",
    "claimed",
    "ready_for_exec",
    "executing",
    "opened",
    "rejected",
    "failed",
    "cancelled",
    "timed_out",
    "closed",
]


# ── Valid State Transitions ─────────────────────────────────────────────────

VALID_TRANSITIONS: dict[str, list[str]] = {
    "pending": ["claimed", "cancelled"],
    "claimed": ["ready_for_exec", "cancelled", "timed_out"],
    "ready_for_exec": ["executing"],
    "executing": ["opened", "rejected", "failed"],
    "opened": ["closed"],
    "rejected": [],
    "failed": [],
    "cancelled": [],
    "timed_out": ["pending"],  # Janitor can recycle timed_out → pending
    "closed": [],
}


# ── TradeIntent ─────────────────────────────────────────────────────────────


class TradeIntent(BaseModel):
    """A trade opportunity discovered by the scanner, awaiting LLM evaluation.

    Flows through the decision state machine:
        pending → claimed → ready_for_exec → executing → opened → closed

    Usage:
        intent = TradeIntent(
            trade_date="2026-02-16",
            symbol="EURUSD",
            scanner_score=0.85,
            scanner_confidence="high",
        )
    """

    id: str = Field(default_factory=lambda: uuid4().hex[:16])
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trade_date: str = Field(description="Trading date YYYY-MM-DD")
    symbol: str = Field(description="FX pair e.g. EURUSD")

    # Scanner outputs
    scanner_score: float = 0.0
    scanner_confidence: str = "medium"
    scanner_score_gap: float = 0.0
    scanner_drop_distance: float = 0.0
    scanner_topk_spread: float = 0.0

    # LLM decision (filled after claim)
    suggested_side: Literal["BUY", "SELL", "HOLD"] | None = None
    suggested_sl_pips: float | None = None
    suggested_tp_pips: float | None = None
    agent_risk_report: str = ""
    agent_state_json: str = ""  # JSON-serialized final_state from TradingAgents

    # Lifecycle
    source: Literal["scanner", "manual"] = "scanner"
    status: IntentStatus = "pending"
    claim_worker_id: str | None = None
    claim_ts: datetime | None = None
    claim_ttl_minutes: int = 30
    expires_at: datetime | None = None
    idempotency_key: str = Field(default_factory=lambda: uuid4().hex[:12])

    # Execution result (filled after execution)
    position_id: str | None = None
    executed_at: datetime | None = None
    execution_error: str | None = None
    compliance_snapshot: str = ""  # JSON of compliance check results

    def can_transition_to(self, new_status: IntentStatus) -> bool:
        """Check if transitioning from current status to new_status is valid."""
        return new_status in VALID_TRANSITIONS.get(self.status, [])


# ── DecisionRecord ──────────────────────────────────────────────────────────


class DecisionRecord(BaseModel):
    """Immutable audit record linking intent to execution outcome.

    Created when an intent is first inserted, updated at each lifecycle stage.
    Serves as a complete audit trail for post-trade analysis.

    Usage:
        record = DecisionRecord(intent_id="abc123", status="pending")
    """

    intent_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    claimed_at: datetime | None = None
    decided_at: datetime | None = None
    executed_at: datetime | None = None
    closed_at: datetime | None = None

    status: str = ""
    order_id: str | None = None
    position_id: str | None = None
    failure_reason: str = ""

    # Snapshots for post-trade analysis
    compliance_snapshot: str = ""
    execution_meta: str = ""  # JSON: entry_price, volume, sl, tp, etc.
