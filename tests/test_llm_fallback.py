"""
Tests for LLM fallback logic — when primary model (right.codes) fails,
AgentBridge.decide() should retry with fallback model (kimi-k2.5).
"""

from unittest.mock import MagicMock

from src.decision.agent_bridge import AgentBridge, AgentDecision


def _make_bridge(
    fallback_model: str = "volcengine/kimi-k2.5",
    using_mock: bool = False,
) -> AgentBridge:
    """Create a minimal AgentBridge bypassing __init__ for unit testing."""
    bridge = object.__new__(AgentBridge)
    bridge._graph = MagicMock()
    bridge._graph_cls = MagicMock() if not using_mock else None
    bridge._using_mock = using_mock
    bridge._selected_analysts = ["market"]
    bridge._merged_config = {}
    bridge._current_model_id = "rightcodes/gpt-5.4"
    bridge._ab_state = None
    bridge._fallback_model = fallback_model
    bridge._default_config = {}
    bridge._config = {}
    bridge._agents_path = "."
    return bridge


# ── Tests ──────────────────────────────────────────────────────────────────


def test_fallback_on_primary_failure() -> None:
    """When primary model raises, fallback model should be tried and succeed."""
    bridge = _make_bridge()

    # Primary call raises, fallback call succeeds
    bridge._graph.propagate = MagicMock(
        side_effect=[
            Exception("500 right.codes timeout"),
            ({"trader_investment_plan": "BUY recommendation"}, "BUY"),
        ]
    )
    # After _apply_ab_model, graph_cls returns the same graph mock
    bridge._graph_cls.return_value = bridge._graph

    result = bridge.decide("EURUSD", "2026-03-10")

    assert isinstance(result, AgentDecision)
    assert result.decision == "BUY"
    assert result.model_id == "volcengine/kimi-k2.5"
    # propagate called twice: primary + fallback
    assert bridge._graph.propagate.call_count == 2


def test_both_models_fail_returns_hold() -> None:
    """When both primary and fallback fail, should return HOLD with error."""
    bridge = _make_bridge()

    bridge._graph.propagate = MagicMock(side_effect=Exception("LLM unavailable"))
    bridge._graph_cls.return_value = bridge._graph

    result = bridge.decide("EURUSD", "2026-03-10")

    assert isinstance(result, AgentDecision)
    assert result.decision == "HOLD"
    assert "error" in result.final_state or "fallback" in result.risk_report.lower()


def test_no_fallback_when_using_mock() -> None:
    """When using MockTradingGraph, fallback should NOT be attempted."""
    bridge = _make_bridge(using_mock=True)

    bridge._graph.propagate = MagicMock(side_effect=Exception("mock error"))

    result = bridge.decide("EURUSD", "2026-03-10")

    assert result.decision == "HOLD"
    # propagate called only once — no fallback retry
    assert bridge._graph.propagate.call_count == 1


def test_primary_success_no_fallback_needed() -> None:
    """When primary model succeeds, fallback should not be triggered at all."""
    bridge = _make_bridge()

    bridge._graph.propagate = MagicMock(
        return_value=({"trader_investment_plan": "SELL setup"}, "SELL")
    )

    result = bridge.decide("EURUSD", "2026-03-10")

    assert result.decision == "SELL"
    # propagate called exactly once
    assert bridge._graph.propagate.call_count == 1


def test_fallback_model_stored_in_init() -> None:
    """AgentBridge.__init__ should store _fallback_model attribute."""
    bridge = AgentBridge(agents_path="../../TradingAgents")
    assert hasattr(bridge, "_fallback_model")
    assert bridge._fallback_model == "volcengine/kimi-k2.5"
