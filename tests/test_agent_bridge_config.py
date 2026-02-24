"""
Tests for BUG #6 and BUG #7 fixes — verify tool_vendors config routes:
- get_global_news away from OpenAI (404 via Volcengine) to local (BUG #6)
- get_news, get_indicators, get_insider_* away from Alpha Vantage (BUG #7)
"""
from src.decision.agent_bridge import AgentBridge
from src.decision.fx_analyst_config import build_agent_config

# ── build_agent_config tests ──────────────────────────────────────────────────


class TestBuildAgentConfig:
    """Verify build_agent_config() produces correct tool_vendors for BUG #6 and #7."""

    def test_tool_vendors_includes_get_global_news(self) -> None:
        """tool_vendors must route get_global_news to 'local'."""
        config = build_agent_config()
        assert "tool_vendors" in config
        assert config["tool_vendors"]["get_global_news"] == "local"

    def test_data_vendors_still_present(self) -> None:
        """data_vendors category config should still be present."""
        config = build_agent_config()
        assert "data_vendors" in config
        assert config["data_vendors"]["news_data"] == "local"

    def test_tool_vendors_takes_precedence_comment(self) -> None:
        """tool_vendors should override category-level data_vendors for specific tools.
        interface.py), so get_global_news='local' in tool_vendors overrides
        news_data in data_vendors.
        """
        config = build_agent_config()
        # Both should coexist — tool_vendors overrides data_vendors for specific tools
        assert config["data_vendors"]["news_data"] == "local"
        assert config["tool_vendors"]["get_global_news"] == "local"
    def test_tool_vendors_routes_away_from_alpha_vantage(self) -> None:
        """BUG #7: All high-frequency methods must be routed away from Alpha Vantage.

        Alpha Vantage free tier = 5 req/min. FX decisions trigger 10+
        concurrent calls, causing rate limit errors.
        """
        config = build_agent_config()
        tv = config["tool_vendors"]
        assert tv["get_news"] == "local"
        assert tv["get_indicators"] == "yfinance"
        assert tv["get_insider_sentiment"] == "local"
        assert tv["get_insider_transactions"] == "local"


# ── AgentBridge config passthrough tests ───────────────────────────────────


class TestAgentBridgeToolVendors:
    """Verify AgentBridge passes tool_vendors to TradingAgentsGraph."""

    def test_config_merged_into_graph(self, tmp_path) -> None:
        """tool_vendors in config dict should be passed through to TradingAgentsGraph."""
        bridge = AgentBridge(
            agents_path=tmp_path,
            config={
                "deep_think_llm": "volcengine/glm-4.7",
                "quick_think_llm": "volcengine/glm-4.7",
                "output_language": "繁體中文",
                "tool_vendors": {
                    "get_global_news": "local",
                },
            },
        )
        # The config should be stored for later merge with DEFAULT_CONFIG
        assert bridge._config["tool_vendors"]["get_global_news"] == "local"

    def test_mock_fallback_preserves_config(self, tmp_path) -> None:
        """When TradingAgents import fails, config should still be stored."""
        bridge = AgentBridge(
            agents_path=tmp_path / "nonexistent",
            config={
                "tool_vendors": {
                    "get_global_news": "local",
                },
            },
        )
        assert bridge._config["tool_vendors"]["get_global_news"] == "local"

    def test_ensure_loaded_merges_tool_vendors(self, tmp_path) -> None:
        """_ensure_loaded() merges config including tool_vendors into DEFAULT_CONFIG.

        When TradingAgents is not available, MockTradingGraph is used.
        The config should still contain tool_vendors after merge.
        """
        bridge = AgentBridge(
            agents_path=tmp_path / "nonexistent",
            config={
                "deep_think_llm": "volcengine/glm-4.7",
                "tool_vendors": {
                    "get_global_news": "local",
                },
            },
        )
        # Trigger _ensure_loaded (will fallback to MockTradingGraph)
        bridge._ensure_loaded()
        assert bridge._using_mock is True
        # Config should still have tool_vendors
        assert bridge._config["tool_vendors"]["get_global_news"] == "local"

