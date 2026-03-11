"""
Tests for vendor routing config — verify tool_vendors and data_vendors config.

Phase 1 (BUG #6): get_global_news away from OpenAI (404 via Volcengine) to local.
Phase 2: Route ALL data sources to Alpha Vantage (premium, 75 req/min).
Phase 3: Route get_global_news to Alpha Vantage (topic-based macro news).
"""

import os
from datetime import datetime, timezone
from types import SimpleNamespace

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

    def test_ensure_loaded_loads_llm_env_from_tradingagents(self, tmp_path, monkeypatch) -> None:
        """LLM env keys from TradingAgents .env should override os.environ."""
        env_path = tmp_path / ".env"
        env_path.write_text(
            "RIGHTCODE_API_KEY=rc_key\n"
            "VOLCENGINE_API_KEY=ve_key\n"
            "AIHUBMIX_API_KEY=ah_key\n"
            "LLM_QUICK_THINK_PRIMARY_MODEL=rightcodes/gpt-5.2\n"
            "NOT_LLM_KEY=should_not_apply\n",
            encoding="utf-8",
        )

        class DummyGraph:
            def __init__(self, selected_analysts, config):
                del selected_analysts, config

        fake_graph_module = SimpleNamespace(TradingAgentsGraph=DummyGraph)
        fake_default_module = SimpleNamespace(DEFAULT_CONFIG={})

        def fake_import(name: str):
            if name == "tradingagents.graph.trading_graph":
                return fake_graph_module
            if name == "tradingagents.default_config":
                return fake_default_module
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("src.decision.agent_bridge.importlib.import_module", fake_import)
        monkeypatch.delenv("RIGHTCODE_API_KEY", raising=False)
        monkeypatch.delenv("VOLCENGINE_API_KEY", raising=False)
        monkeypatch.delenv("AIHUBMIX_API_KEY", raising=False)
        monkeypatch.delenv("LLM_QUICK_THINK_PRIMARY_MODEL", raising=False)
        monkeypatch.delenv("NOT_LLM_KEY", raising=False)

        bridge = AgentBridge(agents_path=tmp_path, config={})
        bridge._ensure_loaded()

        assert os.getenv("RIGHTCODE_API_KEY") == "rc_key"
        assert os.getenv("VOLCENGINE_API_KEY") == "ve_key"
        assert os.getenv("AIHUBMIX_API_KEY") == "ah_key"
        assert os.getenv("LLM_QUICK_THINK_PRIMARY_MODEL") == "rightcodes/gpt-5.2"
        assert os.getenv("NOT_LLM_KEY") is None

    def test_ensure_loaded_overrides_portable_paths(self, tmp_path, monkeypatch) -> None:
        """AgentBridge should override bad default project/workspace/data paths."""
        captured_config: dict[str, object] = {}

        class DummyGraph:
            def __init__(self, selected_analysts, config):
                del selected_analysts
                captured_config.update(config)

        fake_graph_module = SimpleNamespace(TradingAgentsGraph=DummyGraph)
        fake_default_module = SimpleNamespace(
            DEFAULT_CONFIG={
                "project_dir": "/Users/admin/legacy/project",
                "workspace_dir": "/Users/admin/legacy/workspace",
                "data_dir": "/Users/admin/legacy/data",
            }
        )

        def fake_import(name: str):
            if name == "tradingagents.graph.trading_graph":
                return fake_graph_module
            if name == "tradingagents.default_config":
                return fake_default_module
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("src.decision.agent_bridge.importlib.import_module", fake_import)

        bridge = AgentBridge(agents_path=tmp_path, config={})
        bridge._ensure_loaded()

        expected_root = str(tmp_path.resolve())
        assert captured_config["project_dir"] == expected_root
        assert captured_config["workspace_dir"] == expected_root
        assert captured_config["data_dir"] == str(tmp_path.resolve() / "data")

    def test_ensure_loaded_respects_explicit_path_overrides(self, tmp_path, monkeypatch) -> None:
        """Explicit config paths should not be overridden by portability defaults."""
        captured_config: dict[str, object] = {}

        class DummyGraph:
            def __init__(self, selected_analysts, config):
                del selected_analysts
                captured_config.update(config)

        fake_graph_module = SimpleNamespace(TradingAgentsGraph=DummyGraph)
        fake_default_module = SimpleNamespace(DEFAULT_CONFIG={})

        def fake_import(name: str):
            if name == "tradingagents.graph.trading_graph":
                return fake_graph_module
            if name == "tradingagents.default_config":
                return fake_default_module
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("src.decision.agent_bridge.importlib.import_module", fake_import)

        bridge = AgentBridge(
            agents_path=tmp_path,
            config={
                "project_dir": "/custom/project",
                "workspace_dir": "/custom/workspace",
                "data_dir": "/custom/data",
            },
        )
        bridge._ensure_loaded()

        assert captured_config["project_dir"] == "/custom/project"
        assert captured_config["workspace_dir"] == "/custom/workspace"
        assert captured_config["data_dir"] == "/custom/data"

    def test_ensure_loaded_passes_stable_session_id_and_memory_path(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        """AgentBridge should pass session_id and memory_path to TradingAgentsGraph."""
        captured: dict[str, object] = {}

        class DummyGraph:
            def __init__(self, selected_analysts, config, session_id=None):
                del selected_analysts
                captured["config"] = config
                captured["session_id"] = session_id

        fake_graph_module = SimpleNamespace(TradingAgentsGraph=DummyGraph)
        fake_default_module = SimpleNamespace(DEFAULT_CONFIG={})

        def fake_import(name: str):
            if name == "tradingagents.graph.trading_graph":
                return fake_graph_module
            if name == "tradingagents.default_config":
                return fake_default_module
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("src.decision.agent_bridge.importlib.import_module", fake_import)

        memory_path = str(tmp_path / "memory-store")
        bridge = AgentBridge(
            agents_path=tmp_path,
            config={
                "session_id": "acct-123",
                "memory_path": memory_path,
            },
        )
        bridge._ensure_loaded()

        assert captured["session_id"] == "acct-123"
        assert captured["config"]["memory_path"] == memory_path


class TestAgentBridgeDateNormalize:
    """Verify trade_date normalization safeguards."""

    def test_normalize_trade_date_trims_invalid_suffix(self) -> None:
        """Invalid suffix should be trimmed to strict YYYY-MM-DD."""
        assert AgentBridge._normalize_trade_date("2026-02-2626") == "2026-02-26"

    def test_normalize_trade_date_invalid_input_falls_back_to_today(self) -> None:
        """Malformed date should fallback to today's UTC date."""
        expected_today = datetime.now(timezone.utc).date().isoformat()
        assert AgentBridge._normalize_trade_date("not-a-date") == expected_today


class TestAgentBridgeReflect:
    """Verify reflect() handles both legacy and structured payloads."""

    def test_reflect_passes_structured_payload_to_graph(self, tmp_path, monkeypatch) -> None:
        captured: dict[str, object] = {}

        class DummyGraph:
            def __init__(self, selected_analysts, config):
                del selected_analysts, config

            def reflect_and_remember(self, payload):
                captured["payload"] = payload

        fake_graph_module = SimpleNamespace(TradingAgentsGraph=DummyGraph)
        fake_default_module = SimpleNamespace(DEFAULT_CONFIG={})

        def fake_import(name: str):
            if name == "tradingagents.graph.trading_graph":
                return fake_graph_module
            if name == "tradingagents.default_config":
                return fake_default_module
            raise ModuleNotFoundError(name)

        monkeypatch.setattr("src.decision.agent_bridge.importlib.import_module", fake_import)

        bridge = AgentBridge(agents_path=tmp_path, config={})
        payload = {
            "symbol": "EURUSD",
            "realized_pnl": -12.5,
            "close_reason": "sl_hit",
            "market_event_context": "Volatility trigger",
        }

        bridge.reflect(payload)

        assert captured["payload"] == payload


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
