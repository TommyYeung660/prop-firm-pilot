"""
Optimization state models and file IO.

Provides a stable JSON schema for daily optimization outputs that
influence LLM filtering, A/B routing, and feedback loops.

Usage:
    state = load_state("data/optimization_state.json")
    save_state("data/optimization_state.json", state)
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field

# ── Exceptions ──────────────────────────────────────────────────────────────


class OptimizationStateError(Exception):
    """Base exception for optimization state operations."""


class OptimizationStateLoadError(OptimizationStateError):
    """Raised when loading optimization state fails."""


class OptimizationStateSaveError(OptimizationStateError):
    """Raised when saving optimization state fails."""


# ── Models ──────────────────────────────────────────────────────────────────


class Thresholds(BaseModel):
    """Dynamic confidence thresholds for filtering decisions."""

    min_confidence: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Minimum scanner confidence required.",
    )
    min_blended_confidence: float = Field(
        default=0.55,
        description="Minimum blended confidence score (0.0-1.0).",
    )


class ABTestState(BaseModel):
    """A/B testing configuration and aggregated results."""

    model_a: str = Field(default="volcengine/glm-4.7", description="Primary model ID")
    model_b: str = Field(default="gpt-5.2", description="Challenger model ID")
    ratio: float = Field(default=0.5, description="Traffic ratio for model_a")
    counts: dict[str, int] = Field(default_factory=dict, description="Decision counts")
    pnl_by_model: dict[str, float] = Field(
        default_factory=dict, description="Aggregated PnL by model"
    )


class OptimizationState(BaseModel):
    """Root optimization state persisted to JSON."""

    version: str = Field(default="1.0", description="Schema version")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="State generation timestamp (UTC)",
    )
    pnl_lookback_days: int = Field(default=7, description="PnL lookback window (days)")
    winrate_lookback_days: int = Field(
        default=14, description="Win-rate lookback window (days)"
    )

    global_thresholds: Thresholds = Field(default_factory=Thresholds)
    symbol_thresholds: dict[str, Thresholds] = Field(default_factory=dict)

    ab_test: ABTestState = Field(default_factory=ABTestState)
    feedback_pnl: dict[str, float] = Field(default_factory=dict)

    risk_per_trade_suggestion: float | None = Field(
        default=None, description="Suggested risk per trade (fraction)"
    )
    llm_cost_stats: dict[str, Any] = Field(default_factory=dict)
    factor_contributions: dict[str, Any] = Field(default_factory=dict)


# ── IO Helpers ──────────────────────────────────────────────────────────────


def load_state(path: str | Path) -> OptimizationState:
    """Load optimization state from JSON, falling back to defaults.

    Args:
        path: Path to the optimization state JSON file.

    Returns:
        OptimizationState object (defaulted on error).
    """
    file_path = Path(path)
    if not file_path.exists():
        return OptimizationState()
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return OptimizationState(**data)
    except Exception as e:
        logger.warning("OptimizationState: failed to load {}, using default ({})", file_path, e)
        return OptimizationState()


def save_state(path: str | Path, state: OptimizationState) -> None:
    """Persist optimization state to JSON.

    Args:
        path: Path to the optimization state JSON file.
        state: OptimizationState to save.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        file_path.write_text(
            json.dumps(state.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        logger.error("OptimizationState: failed to save {}: {}", file_path, e)
        raise OptimizationStateSaveError(str(e)) from e
