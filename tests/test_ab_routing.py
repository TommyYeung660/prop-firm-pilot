"""Tests for A/B test routing, stats, and wiring.

Covers:
- choose_model deterministic routing
- update_ab_stats accumulation
- OptimizationEngine preserves AB counts on refresh
- AgentDecision model_id field
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

from src.decision.agent_bridge import AgentDecision
from src.optimize.ab_testing import choose_model, update_ab_stats
from src.optimize.optimization_engine import OptimizationEngine
from src.optimize.optimization_state import ABTestState, OptimizationState, save_state

# ── choose_model ─────────────────────────────────────────────────────────────


def test_choose_model_deterministic() -> None:
    """Same intent_id always produces the same model choice."""
    intent_id = "abc123"
    model_a = "volcengine/glm-4.7"
    model_b = "gpt-5.2"
    ratio = 0.5

    results = [choose_model(intent_id, ratio, model_a, model_b) for _ in range(10)]
    assert len(set(results)) == 1, "choose_model must be deterministic"
    assert results[0] in (model_a, model_b)


def test_choose_model_ratio_boundaries() -> None:
    """ratio=0.0 → always model_b, ratio=1.0 → always model_a."""
    model_a = "a"
    model_b = "b"

    assert choose_model("any_id", 0.0, model_a, model_b) == model_b
    assert choose_model("any_id", 1.0, model_a, model_b) == model_a


def test_choose_model_different_intents_vary() -> None:
    """Different intent IDs should route to different models (probabilistically)."""
    model_a = "a"
    model_b = "b"
    choices = {choose_model(f"intent_{i}", 0.5, model_a, model_b) for i in range(100)}
    # With 100 different IDs and 50/50 ratio, we should see both models
    assert len(choices) == 2, "Expected both models to be chosen across 100 IDs"


# ── update_ab_stats ──────────────────────────────────────────────────────────


def test_update_ab_stats_accumulates() -> None:
    """Stats should accumulate counts and PnL."""
    state = ABTestState()

    update_ab_stats(state, "model_a", 10.0)
    update_ab_stats(state, "model_a", -5.0)
    update_ab_stats(state, "model_b", 20.0)

    assert state.counts == {"model_a": 2, "model_b": 1}
    assert state.pnl_by_model["model_a"] == 5.0
    assert state.pnl_by_model["model_b"] == 20.0


def test_update_ab_stats_none_pnl() -> None:
    """None PnL should still increment count but not PnL."""
    state = ABTestState()

    update_ab_stats(state, "model_a", None)

    assert state.counts == {"model_a": 1}
    assert state.pnl_by_model == {}


# ── OptimizationEngine AB count preservation ─────────────────────────────────


def test_optimization_engine_preserves_ab_counts(tmp_path: Path) -> None:
    """refresh_state should preserve existing AB counts from persisted state."""
    state_path = tmp_path / "optimization_state.json"

    # Seed with existing AB stats
    existing = OptimizationState()
    existing.ab_test = ABTestState(
        model_a="volcengine/glm-4.7",
        model_b="gpt-5.2",
        ratio=0.5,
        counts={"volcengine/glm-4.7": 5, "gpt-5.2": 3},
        pnl_by_model={"volcengine/glm-4.7": 100.0, "gpt-5.2": -20.0},
    )
    save_state(state_path, existing)

    # Create engine with mock dependencies
    mock_store = MagicMock()
    mock_store.get_closed_intents.return_value = []
    mock_store.get_intents_by_date.return_value = []

    engine = OptimizationEngine(
        store=mock_store,
        journal=None,
        state_path=str(state_path),
        ab_model_a="volcengine/glm-4.7",
        ab_model_b="gpt-5.2",
        ab_ratio=0.5,
    )

    result = engine.refresh_state()

    # Counts must survive refresh
    assert result.ab_test.counts == {"volcengine/glm-4.7": 5, "gpt-5.2": 3}
    assert result.ab_test.pnl_by_model == {"volcengine/glm-4.7": 100.0, "gpt-5.2": -20.0}
    assert result.ab_test.model_a == "volcengine/glm-4.7"
    assert result.ab_test.model_b == "gpt-5.2"
    assert result.ab_test.ratio == 0.5


def test_optimization_engine_empty_state_fresh(tmp_path: Path) -> None:
    """When no persisted state, AB counts should start empty."""
    state_path = tmp_path / "optimization_state.json"

    mock_store = MagicMock()
    mock_store.get_closed_intents.return_value = []
    mock_store.get_intents_by_date.return_value = []

    engine = OptimizationEngine(
        store=mock_store,
        journal=None,
        state_path=str(state_path),
    )

    result = engine.refresh_state()

    assert result.ab_test.counts == {}
    assert result.ab_test.pnl_by_model == {}


# ── AgentDecision model_id ───────────────────────────────────────────────────


def test_agent_decision_has_model_id() -> None:
    """AgentDecision should store model_id."""
    decision = AgentDecision(
        symbol="EURUSD",
        decision="BUY",
        final_state={},
        model_id="volcengine/glm-4.7",
    )
    assert decision.model_id == "volcengine/glm-4.7"


def test_agent_decision_default_model_id() -> None:
    """AgentDecision model_id defaults to empty string."""
    decision = AgentDecision(
        symbol="EURUSD",
        decision="HOLD",
        final_state={},
    )
    assert decision.model_id == ""


# ── _build_execution_meta model_id ───────────────────────────────────────────


def test_build_execution_meta_includes_model_id() -> None:
    """_build_execution_meta should include model_id when provided."""
    from src.execution.engine import ExecutionEngine

    meta_json = ExecutionEngine._build_execution_meta(
        fill_price=1.085,
        volume=0.05,
        side="BUY",
        sl_price=1.08,
        tp_price=1.095,
        sl_pips=50,
        tp_pips=100,
        pre_trade_bid=1.0849,
        pre_trade_ask=1.0851,
        slippage_pips=0.1,
        execution_latency_ms=150.0,
        random_delay_seconds=2.5,
        compliance_passed=True,
        order_raw_response={"orderId": "12345"},
        model_id="volcengine/glm-4.7",
    )
    data = json.loads(meta_json)
    assert data["model_id"] == "volcengine/glm-4.7"


def test_build_execution_meta_no_model_id() -> None:
    """_build_execution_meta should omit model_id when empty."""
    from src.execution.engine import ExecutionEngine

    meta_json = ExecutionEngine._build_execution_meta(
        fill_price=1.085,
        volume=0.05,
        side="BUY",
        sl_price=1.08,
        tp_price=1.095,
        sl_pips=50,
        tp_pips=100,
        pre_trade_bid=1.0849,
        pre_trade_ask=1.0851,
        slippage_pips=0.1,
        execution_latency_ms=150.0,
        random_delay_seconds=2.5,
        compliance_passed=True,
        order_raw_response={"orderId": "12345"},
    )
    data = json.loads(meta_json)
    assert "model_id" not in data
