# TradingAgents LLM Delegation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ensure prop-firm-pilot stops overriding TradingAgents LLM models and instead loads LLM env vars from TradingAgents `.env`, while keeping FX-specific config intact.

**Architecture:** AgentBridge will load `agents_path/.env` and only apply LLM-related keys to `os.environ` before importing TradingAgents. Prop-firm config and YAML will no longer include `deep_think_llm` / `quick_think_llm`, and build_agent_config will only pass FX settings.

**Tech Stack:** Python 3.10, pydantic v2, python-dotenv, uv, pytest

---

### Task 1: Add Tests for TradingAgents `.env` LLM Injection

**Files:**
- Modify: `tests/test_agent_bridge_config.py`

**Step 1: Write the failing test**

```python
from dotenv import dotenv_values
import os

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
        monkeypatch.delenv("NOT_LLM_KEY", raising=False)

        bridge = AgentBridge(agents_path=tmp_path, config={})
        bridge._ensure_loaded()

        assert os.getenv("RIGHTCODE_API_KEY") == "rc_key"
        assert os.getenv("VOLCENGINE_API_KEY") == "ve_key"
        assert os.getenv("AIHUBMIX_API_KEY") == "ah_key"
        assert os.getenv("LLM_QUICK_THINK_PRIMARY_MODEL") == "rightcodes/gpt-5.2"
        assert os.getenv("NOT_LLM_KEY") is None
```

Also remove test config payloads that set `deep_think_llm` / `quick_think_llm` (they are no longer part of prop-firm config).

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent_bridge_config.py -q`
Expected: FAIL because AgentBridge does not yet load TradingAgents `.env` and/or tests still reference removed config fields.

**Step 3: Write minimal implementation**

Implementation is in Task 2 and Task 3.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agent_bridge_config.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_agent_bridge_config.py
git commit -m "Add tests for TradingAgents env injection"
```

---

### Task 2: Load TradingAgents LLM Env in AgentBridge

**Files:**
- Modify: `src/decision/agent_bridge.py`

**Step 1: Add env loading helper and call it**

```python
import os
from dotenv import dotenv_values

LLM_ENV_PREFIXES = ("RIGHTCODE_", "VOLCENGINE_", "AIHUBMIX_", "LLM_")

    def _load_tradingagents_env(self) -> None:
        env_path = self._agents_path / ".env"
        if not env_path.exists():
            logger.warning("AgentBridge: TradingAgents .env not found at {}", env_path)
            return
        values = dotenv_values(env_path)
        if not values:
            logger.warning("AgentBridge: TradingAgents .env empty at {}", env_path)
            return
        for key, value in values.items():
            if not key or value is None:
                continue
            if not key.startswith(LLM_ENV_PREFIXES):
                continue
            os.environ[key] = value

    def _ensure_loaded(self) -> None:
        if self._graph is not None:
            return
        self._load_tradingagents_env()
        ...
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_agent_bridge_config.py -q`
Expected: PASS

**Step 3: Commit**

```bash
git add src/decision/agent_bridge.py
git commit -m "Load TradingAgents LLM env in AgentBridge"
```

---

### Task 3: Remove LLM Model Fields from prop-firm Config

**Files:**
- Modify: `src/config.py`
- Modify: `src/decision/fx_analyst_config.py`
- Modify: `src/main.py`
- Modify: `config/default.yaml`
- Modify: `config/default.yaml.example`

**Step 1: Remove `deep_think_llm` / `quick_think_llm` from `AgentsConfig`**

```python
class AgentsConfig(BaseModel):
    """Bridge config for TradingAgents."""

    project_path: str = "../../TradingAgents"
    selected_analysts: list[str] = ["market", "news", "social"]
    output_language: str = "繁體中文"
```

**Step 2: Update `build_agent_config()` signature and payload**

```python
def build_agent_config(
    output_language: str = "繁體中文",
) -> dict[str, Any]:
    return {
        "output_language": output_language,
        "market_type": "fx",
        ...
    }
```

**Step 3: Update `main.py` to stop passing LLM models**

```python
config=build_agent_config(
    output_language=config.agents.output_language,
),
```

**Step 4: Remove YAML keys**

Delete `deep_think_llm` / `quick_think_llm` entries from both `config/default.yaml` and `config/default.yaml.example`.

**Step 5: Run tests**

Run: `uv run pytest tests/test_agent_bridge_config.py -q`
Expected: PASS

**Step 6: Commit**

```bash
git add src/config.py src/decision/fx_analyst_config.py src/main.py config/default.yaml config/default.yaml.example
git commit -m "Remove LLM model overrides from prop-firm config"
```

---

### Task 4: Update Env/Runbook Docs

**Files:**
- Modify: `.env.example`
- Modify: `AGENTS.md`
- Modify: `docs/ops_runbook.md`

**Step 1: Update `.env.example`**

Remove:
```
LLM_API_KEY=...
LLM_BASE_URL=...
```

Replace with a comment:
```
# LLM for TradingAgents is configured in ../../TradingAgents/.env
# (RIGHTCODE_*, VOLCENGINE_*, AIHUBMIX_*, LLM_*)
```

**Step 2: Update `AGENTS.md` environment variables section**

Replace the LLM entries with a note pointing to TradingAgents `.env`.

**Step 3: Update `docs/ops_runbook.md`**

Replace `LLM_API_KEY` / `LLM_BASE_URL` notes with the same TradingAgents `.env` guidance.

**Step 4: Run full tests**

Run: `uv run pytest`
Expected: PASS (warnings OK)

**Step 5: Commit**

```bash
git add .env.example AGENTS.md docs/ops_runbook.md
git commit -m "Update docs for TradingAgents LLM env delegation"
```
