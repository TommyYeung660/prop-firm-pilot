# E8 One 5K Config Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update `config/e8_one_5k_challenge.yaml` to isolate data paths, restrict symbols, and correct the risk parameter name/value.

**Architecture:** Configuration-only change. Extend existing E8 One config test to assert new overrides, then update YAML to satisfy the test. No runtime code changes.

**Tech Stack:** YAML config, Pydantic config loader, pytest.

---

### Task 1: Add config assertions (RED)

**Files:**
- Modify: `tests/test_prop_firm_guard_e8_one.py`

**Step 1: Write the failing test**

Append assertions in `test_config_loads_correctly`:

```python
        assert e8_one_config.symbols == ["EURUSD", "XAUUSD"]
        assert e8_one_config.execution.default_risk_pct == 0.005
        assert e8_one_config.decision_store.db_path == "data/decisions_e8_one_5k.db"
        assert e8_one_config.monitor.trade_journal_path == "data/trade_journal_e8_one_5k.jsonl"
        assert e8_one_config.monitor.memory_dir == "MEMORY_E8_ONE_5K"
        assert e8_one_config.optimization.state_path == "data/optimization_state_e8_one_5k.json"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prop_firm_guard_e8_one.py::TestPropFirmGuardE8One::test_config_loads_correctly -v`  
Expected: FAIL (symbols/risk/path assertions not matching).

---

### Task 2: Update E8 One YAML (GREEN)

**Files:**
- Modify: `config/e8_one_5k_challenge.yaml`

**Step 1: Update YAML**

Apply these overrides:

```yaml
symbols: [EURUSD, XAUUSD]

execution:
  default_risk_pct: 0.005

decision_store:
  db_path: "data/decisions_e8_one_5k.db"

monitor:
  trade_journal_path: "data/trade_journal_e8_one_5k.jsonl"
  memory_dir: "MEMORY_E8_ONE_5K"

optimization:
  state_path: "data/optimization_state_e8_one_5k.json"
```

Remove the incorrect `default_risk_per_trade` key if present.

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_prop_firm_guard_e8_one.py::TestPropFirmGuardE8One::test_config_loads_correctly -v`  
Expected: PASS.

**Step 3: Commit**

```bash
git add tests/test_prop_firm_guard_e8_one.py config/e8_one_5k_challenge.yaml
git commit -m "chore: isolate e8 one config paths and risk"
```

---

### Task 3: Quick sanity check (optional)

**Files:**
- None

**Step 1: Run full E8 One guard test**

Run: `uv run pytest tests/test_prop_firm_guard_e8_one.py -v`  
Expected: PASS.

