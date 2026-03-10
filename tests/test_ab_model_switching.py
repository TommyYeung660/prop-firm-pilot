"""Tests for A/B test *actual* model switching in AgentBridge.

Covers:
- _apply_ab_model() no-op when model unchanged
- _apply_ab_model() rebuilds graph when model changes
- _apply_ab_model() skips rebuild when using mock
- _apply_ab_model() handles rebuild failure gracefully
- decide() applies AB model BEFORE propagate()
- decide() without AB state does not switch models
"""

from unittest.mock import MagicMock, patch

from src.decision.agent_bridge import AgentBridge, MockTradingGraph
from src.optimize.optimization_state import ABTestState

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_bridge(tmp_path, ab_state: ABTestState | None = None) -> AgentBridge:
    """Create an AgentBridge pre-loaded with mocks (no real TradingAgents import)."""
    bridge = AgentBridge(
        agents_path=tmp_path,
        selected_analysts=["market", "news"],
        config={"output_language": "繁體中文"},
    )
    # Simulate successful _ensure_loaded() without real imports
    mock_graph = MockTradingGraph()
    bridge._graph = mock_graph
    bridge._using_mock = False  # pretend real graph loaded
    bridge._graph_cls = MagicMock(return_value=MagicMock())
    bridge._default_config = {"deep_think_llm": "rightcodes/gpt-5.4"}
    bridge._merged_config = {
        "deep_think_llm": "rightcodes/gpt-5.4",
        "quick_think_llm": "rightcodes/gpt-5.4",
        "default_model": "rightcodes/gpt-5.4",
    }
    bridge._current_model_id = ""
    if ab_state is not None:
        bridge.set_ab_state(ab_state)
    return bridge


# ── _apply_ab_model tests ────────────────────────────────────────────────────


class TestApplyAbModel:
    """Tests for AgentBridge._apply_ab_model()."""

    def test_noop_when_same_model(self, tmp_path) -> None:
        """Should not rebuild graph when model_id matches current."""
        bridge = _make_bridge(tmp_path)
        bridge._current_model_id = "rightcodes/gpt-5.4"
        original_graph = bridge._graph

        bridge._apply_ab_model("rightcodes/gpt-5.4")

        assert bridge._graph is original_graph
        bridge._graph_cls.assert_not_called()

    def test_rebuilds_graph_on_model_change(self, tmp_path) -> None:
        """Should rebuild graph when model_id differs from current."""
        bridge = _make_bridge(tmp_path)
        bridge._current_model_id = "rightcodes/gpt-5.4"
        original_graph = bridge._graph

        bridge._apply_ab_model("volcengine/kimi-k2.5")

        # Graph class was called to create a new instance
        bridge._graph_cls.assert_called_once_with(
            selected_analysts=["market", "news"],
            config=bridge._merged_config,
        )
        # Graph was replaced
        assert bridge._graph is not original_graph
        assert bridge._graph is bridge._graph_cls.return_value
        # Current model updated
        assert bridge._current_model_id == "volcengine/kimi-k2.5"
        # Config was updated
        assert bridge._merged_config["deep_think_llm"] == "volcengine/kimi-k2.5"
        assert bridge._merged_config["quick_think_llm"] == "volcengine/kimi-k2.5"
        assert bridge._merged_config["default_model"] == "volcengine/kimi-k2.5"

    def test_skip_when_using_mock(self, tmp_path) -> None:
        """Should not rebuild when bridge is using mock fallback."""
        bridge = _make_bridge(tmp_path)
        bridge._using_mock = True
        bridge._current_model_id = "rightcodes/gpt-5.4"

        bridge._apply_ab_model("volcengine/kimi-k2.5")

        bridge._graph_cls.assert_not_called()
        # Model ID is still updated (for metadata tracking)
        assert bridge._current_model_id == "volcengine/kimi-k2.5"

    def test_skip_when_no_graph_cls(self, tmp_path) -> None:
        """Should not rebuild when graph_cls is None."""
        bridge = _make_bridge(tmp_path)
        bridge._graph_cls = None

        bridge._apply_ab_model("volcengine/kimi-k2.5")

        assert bridge._current_model_id == "volcengine/kimi-k2.5"

    def test_handles_rebuild_failure(self, tmp_path) -> None:
        """Should keep previous graph if rebuild fails."""
        bridge = _make_bridge(tmp_path)
        bridge._current_model_id = "rightcodes/gpt-5.4"
        original_graph = bridge._graph
        bridge._graph_cls.side_effect = RuntimeError("LLM init failed")

        bridge._apply_ab_model("volcengine/kimi-k2.5")

        # Graph should remain unchanged
        assert bridge._graph is original_graph
        # Model ID should NOT be updated on failure
        assert bridge._current_model_id == "rightcodes/gpt-5.4"

    def test_first_apply_from_empty(self, tmp_path) -> None:
        """First _apply_ab_model from empty _current_model_id triggers rebuild."""
        bridge = _make_bridge(tmp_path)
        assert bridge._current_model_id == ""

        bridge._apply_ab_model("rightcodes/gpt-5.4")

        bridge._graph_cls.assert_called_once()
        assert bridge._current_model_id == "rightcodes/gpt-5.4"


# ── decide() AB integration tests ───────────────────────────────────────────


class TestDecideAbIntegration:
    """Tests for AB model switching inside decide()."""

    def test_decide_applies_model_before_propagate(self, tmp_path) -> None:
        """choose_model → _apply_ab_model → propagate (correct order)."""
        ab_state = ABTestState(
            model_a="rightcodes/gpt-5.4",
            model_b="volcengine/kimi-k2.5",
            ratio=0.5,
        )
        bridge = _make_bridge(tmp_path, ab_state=ab_state)

        # Set up a real mock graph with propagate
        mock_graph = MagicMock()
        propagate_result = (
            {"final_trade_decision": "BUY", "trader_investment_plan": "test"},
            "BUY",
        )
        mock_graph.propagate.return_value = propagate_result
        bridge._graph = mock_graph

        # Track call order via patched _apply_ab_model (skip real rebuild)
        call_order = []

        def tracking_apply(model_id):
            call_order.append(("apply_ab_model", model_id))
            # Don't actually rebuild since mock_graph.propagate needs to work
            bridge._current_model_id = model_id

        bridge._apply_ab_model = tracking_apply

        # Track propagate calls without recursion: use side_effect that
        # records the call then returns the pre-set result directly
        mock_graph.propagate.side_effect = lambda **kw: (
            call_order.append(("propagate",)),
            propagate_result,
        )[-1]

        result = bridge.decide("EURUSD", "2026-03-09", intent_id="test-intent-001")

        # _apply_ab_model was called
        assert any(c[0] == "apply_ab_model" for c in call_order)
        # propagate was called
        assert any(c[0] == "propagate" for c in call_order)
        # _apply_ab_model was called BEFORE propagate
        apply_idx = next(i for i, c in enumerate(call_order) if c[0] == "apply_ab_model")
        propagate_idx = next(i for i, c in enumerate(call_order) if c[0] == "propagate")
        assert apply_idx < propagate_idx, "_apply_ab_model must be called before propagate"
        # model_id is set in result
        assert result.model_id != ""
        assert result.model_id in ("rightcodes/gpt-5.4", "volcengine/kimi-k2.5")

    def test_decide_without_ab_state_no_switch(self, tmp_path) -> None:
        """Without AB state, decide should not call _apply_ab_model."""
        bridge = _make_bridge(tmp_path, ab_state=None)

        mock_graph = MagicMock()
        mock_graph.propagate.return_value = (
            {"final_trade_decision": "HOLD", "trader_investment_plan": ""},
            "HOLD",
        )
        bridge._graph = mock_graph

        with patch.object(bridge, "_apply_ab_model") as mock_apply:
            result = bridge.decide("EURUSD", "2026-03-09", intent_id="test-001")

        mock_apply.assert_not_called()
        assert result.model_id == ""
        assert result.decision == "HOLD"

    def test_decide_without_intent_id_no_switch(self, tmp_path) -> None:
        """Without intent_id, decide should not call _apply_ab_model."""
        ab_state = ABTestState(
            model_a="rightcodes/gpt-5.4",
            model_b="volcengine/kimi-k2.5",
            ratio=0.5,
        )
        bridge = _make_bridge(tmp_path, ab_state=ab_state)

        mock_graph = MagicMock()
        mock_graph.propagate.return_value = (
            {"final_trade_decision": "SELL", "trader_investment_plan": "short"},
            "SELL",
        )
        bridge._graph = mock_graph

        with patch.object(bridge, "_apply_ab_model") as mock_apply:
            result = bridge.decide("EURUSD", "2026-03-09")  # no intent_id

        mock_apply.assert_not_called()
        assert result.model_id == ""

    def test_decide_ab_model_consistent_for_same_intent(self, tmp_path) -> None:
        """Same intent_id always selects the same model (deterministic)."""
        ab_state = ABTestState(
            model_a="rightcodes/gpt-5.4",
            model_b="volcengine/kimi-k2.5",
            ratio=0.5,
        )
        bridge = _make_bridge(tmp_path, ab_state=ab_state)

        # Pre-determine which model choose_model will pick for "stable-intent"
        from src.optimize.ab_testing import choose_model

        expected_model = choose_model(
            intent_id="stable-intent",
            ratio=0.5,
            model_a="rightcodes/gpt-5.4",
            model_b="volcengine/kimi-k2.5",
        )
        # Pre-set _current_model_id to avoid graph rebuild replacing our mock
        bridge._current_model_id = expected_model

        mock_graph = MagicMock()
        mock_graph.propagate.return_value = (
            {"final_trade_decision": "BUY", "trader_investment_plan": "long"},
            "BUY",
        )
        bridge._graph = mock_graph

        results = []
        for _ in range(5):
            result = bridge.decide("EURUSD", "2026-03-09", intent_id="stable-intent")
            results.append(result.model_id)

        assert len(set(results)) == 1, "Same intent must always pick the same model"
        assert results[0] == expected_model
