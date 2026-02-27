"""
A/B routing utilities for LLM model selection.

Deterministically assigns intents to model A/B by hashing
an identifier and comparing against a ratio bucket.

Usage:
    model_id = choose_model(intent_id, 0.5, "m1", "m2")
"""

import hashlib

from loguru import logger

from src.optimize.optimization_state import ABTestState

# ── Exceptions ──────────────────────────────────────────────────────────────


class ABTestingError(Exception):
    """Base exception for A/B testing utilities."""


# ── Public API ──────────────────────────────────────────────────────────────


def choose_model(intent_id: str, ratio: float, model_a: str, model_b: str) -> str:
    """Choose a model deterministically based on intent ID.

    Args:
        intent_id: Stable identifier for routing.
        ratio: Traffic ratio for model_a (0.0-1.0).
        model_a: Primary model ID.
        model_b: Challenger model ID.

    Returns:
        Selected model ID.
    """
    if ratio <= 0.0:
        return model_b
    if ratio >= 1.0:
        return model_a

    digest = hashlib.sha256(intent_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    chosen = model_a if bucket < ratio else model_b
    logger.debug(
        "ABTesting: intent={} bucket={:.4f} ratio={:.2f} -> {}",
        intent_id,
        bucket,
        ratio,
        chosen,
    )
    return chosen


def update_ab_stats(state: ABTestState, model_id: str, pnl: float | None) -> None:
    """Update A/B testing stats counters and optional PnL.

    Args:
        state: ABTestState to mutate.
        model_id: Model identifier used for a decision.
        pnl: Realized PnL to accumulate (optional).
    """
    state.counts[model_id] = state.counts.get(model_id, 0) + 1
    if pnl is not None:
        state.pnl_by_model[model_id] = state.pnl_by_model.get(model_id, 0.0) + pnl
