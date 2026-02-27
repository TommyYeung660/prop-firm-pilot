"""Tests for A/B routing utilities."""

from src.optimize.ab_testing import choose_model, update_ab_stats
from src.optimize.optimization_state import ABTestState

# ── Tests ───────────────────────────────────────────────────────────────────


def test_choose_model_deterministic() -> None:
    a = choose_model("abc123", 0.5, "m1", "m2")
    b = choose_model("abc123", 0.5, "m1", "m2")
    assert a == b


def test_choose_model_ratio_edges() -> None:
    assert choose_model("x", 1.0, "m1", "m2") == "m1"
    assert choose_model("x", 0.0, "m1", "m2") == "m2"


def test_update_ab_stats_counts_and_pnl() -> None:
    state = ABTestState(model_a="m1", model_b="m2", ratio=0.5)
    update_ab_stats(state, "m1", 2.5)
    update_ab_stats(state, "m1", None)
    assert state.counts["m1"] == 2
    assert state.pnl_by_model["m1"] == 2.5
