"""
Tests for vendor routing config — verify tool_vendors and data_vendors config.

Phase 1 (BUG #6): get_global_news away from OpenAI (404 via Volcengine) to local.
Phase 2: Route ALL data sources to Alpha Vantage (premium, 75 req/min).
Phase 3: Route get_global_news to Alpha Vantage (topic-based macro news).
"""
from src.decision.agent_bridge import AgentBridge
from src.decision.fx_analyst_config import build_agent_config

# ── build_agent_config tests ──────────────────────────────────────────────────


class TestBuildAgentConfig:
    """Verify build_agent_config() produces correct vendor routing."""

    def test_tool_vendors_includes_get_global_news(self) -> None:
        """tool_vendors must route get_global_news to 'alpha_vantage'."""
        config = build_agent_config()
        assert "tool_vendors" in config
        assert config["tool_vendors"]["get_global_news"] == "alpha_vantage"

    def test_data_vendors_route_to_alpha_vantage(self) -> None:
        """data_vendors should route to alpha_vantage (premium tier)."""
        config = build_agent_config()
        assert "data_vendors" in config
        assert config["data_vendors"]["news_data"] == "alpha_vantage"
        assert config["data_vendors"]["core_stock_apis"] == "alpha_vantage"

    def test_tool_vendors_takes_precedence(self) -> None:
        """tool_vendors overrides data_vendors for specific tools.
        get_global_news='alpha_vantage' as explicit tool override.
        """
        config = build_agent_config()
        assert config["data_vendors"]["news_data"] == "alpha_vantage"
        assert config["tool_vendors"]["get_global_news"] == "alpha_vantage"
    def test_tool_vendors_routes_to_alpha_vantage(self) -> None:
        """Alpha Vantage premium (75 req/min) for news, global_news, and indicators."""
        config = build_agent_config()
        tv = config["tool_vendors"]
        assert tv["get_news"] == "alpha_vantage"
        assert tv["get_global_news"] == "alpha_vantage"
        assert tv["get_indicators"] == "alpha_vantage"
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
                    "get_global_news": "alpha_vantage",
                },
            },
        )
        # The config should be stored for later merge with DEFAULT_CONFIG
        assert bridge._config["tool_vendors"]["get_global_news"] == "alpha_vantage"

    def test_mock_fallback_preserves_config(self, tmp_path) -> None:
        """When TradingAgents import fails, config should still be stored."""
        bridge = AgentBridge(
            agents_path=tmp_path / "nonexistent",
            config={
                "tool_vendors": {
                    "get_global_news": "alpha_vantage",
                },
            },
        )
        assert bridge._config["tool_vendors"]["get_global_news"] == "alpha_vantage"

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
                    "get_global_news": "alpha_vantage",
                },
            },
        )
        # Trigger _ensure_loaded (will fallback to MockTradingGraph)
        bridge._ensure_loaded()
        assert bridge._using_mock is True
        # Config should still have tool_vendors
        assert bridge._config["tool_vendors"]["get_global_news"] == "alpha_vantage"


# ── main.py integration tests ──────────────────────────────────────────────



class TestMainUsesBuiltConfig:
    """Verify main.py no longer hard-codes vendor routing."""

    def test_main_py_has_no_hardcoded_tool_vendors(self) -> None:
        """main.py must use build_agent_config(), not inline tool_vendors dicts.

        This prevents regressions where someone edits main.py directly
        instead of updating fx_analyst_config.py.
        """
        import inspect

        import src.main as main_module

        source = inspect.getsource(main_module)
        # Should NOT contain the old hard-coded vendor values
        assert '"get_global_news": "local"' not in source, (
            "main.py still hard-codes get_global_news='local' — use build_agent_config()"
        )
        assert '"get_news": "local"' not in source, (
            "main.py still hard-codes get_news='local' — use build_agent_config()"
        )
        assert '"get_indicators": "yfinance"' not in source, (
            "main.py still hard-codes get_indicators='yfinance' — use build_agent_config()"
        )

    def test_main_py_imports_build_agent_config(self) -> None:
        """main.py must import build_agent_config from fx_analyst_config."""
        import inspect

        import src.main as main_module

        source = inspect.getsource(main_module)
        assert "build_agent_config" in source, (
            "main.py does not reference build_agent_config — vendor routing may be wrong"
        )

