"""
Close-domain typed models.

Provides a small set of dataclasses shared by the close control plane,
reconciler, and close-related journal payload builders.

Usage:
    intent = CloseIntent(...)
    outcome = CloseOutcome(...)
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CloseIntent:
    """Canonical close-domain command."""

    trigger_source: str
    action_kind: str
    position_id: str
    intent_id: str
    symbol: str
    side: str
    reason_code: str
    requested_volume: float | None = None
    requested_sl: float | None = None
    requested_tp: float | None = None
    source_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class CloseOutcome:
    """Execution result emitted by the close control plane."""

    trigger_source: str
    action_kind: str
    execution_status: str
    readback_status: str
    final_close_reason: str = ""
    broker_result: dict[str, Any] = field(default_factory=dict)
    journal_payload: dict[str, Any] = field(default_factory=dict)
    meta_patch: dict[str, Any] = field(default_factory=dict)


@dataclass
class CloseReconciliation:
    """Canonical final close facts after broker/external reconciliation."""

    trigger_source: str
    action_kind: str
    final_close_reason: str
    resolution_path: str
    pnl: float
    close_price: float
    volume: float
    journal_payload: dict[str, Any] = field(default_factory=dict)
    meta_patch: dict[str, Any] = field(default_factory=dict)
