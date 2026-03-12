# E8 One 5K Universe Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand `config/e8_one_5k_challenge.yaml` from 4 to 7 major FX pairs and raise `scanner.topk` to 5 so E8 One 5K can generate more diversified candidates without changing runtime logic.

**Architecture:** Keep this as a configuration-only change. First update config regression assertions so the desired universe, websocket coverage, instrument map, and scanner top-k are explicit. Then update the YAML minimally to satisfy the tests and run targeted verification.

**Tech Stack:** YAML config, Pydantic config loading, pytest

---

### Task 1: Lock the desired E8 One 5K universe in tests

**Files:**
- Modify: `tests/test_prop_firm_guard_e8_one.py`

**Step 1: Write the failing test**

Extend `TestE8OneConfig.test_config_loads_correctly` with assertions for:

```python
        assert e8_one_config.symbols == [
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "AUDUSD",
            "NZDUSD",
            "USDCAD",
            "USDCHF",
        ]
        assert e8_one_config.websocket.symbols == [
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "AUDUSD",
            "NZDUSD",
            "USDCAD",
            "USDCHF",
        ]
        assert e8_one_config.scanner.topk == 5
        assert "NZDUSD" in e8_one_config.instruments
        assert "USDCAD" in e8_one_config.instruments
        assert "USDCHF" in e8_one_config.instruments
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prop_firm_guard_e8_one.py::TestE8OneConfig::test_config_loads_correctly -q`

Expected: FAIL because the current YAML still exposes 4 symbols, 4 websocket symbols, and default `scanner.topk=3`.

---

### Task 2: Update the E8 One 5K YAML minimally

**Files:**
- Modify: `config/e8_one_5k_challenge.yaml`

**Step 1: Update YAML**

Apply these changes:

```yaml
symbols:
  - EURUSD
  - GBPUSD
  - USDJPY
  - AUDUSD
  - NZDUSD
  - USDCAD
  - USDCHF

scanner:
  topk: 5

websocket:
  symbols:
    - EURUSD
    - GBPUSD
    - USDJPY
    - AUDUSD
    - NZDUSD
    - USDCAD
    - USDCHF

instruments:
  NZDUSD:
    pip_size: 0.0001
    pip_value: 10.0
    avg_spread_pips: 1.8
    min_lot: 0.01
    max_lot: 5.0
  USDCAD:
    pip_size: 0.0001
    pip_value: 10.0
    avg_spread_pips: 1.7
    min_lot: 0.01
    max_lot: 5.0
  USDCHF:
    pip_size: 0.0001
    pip_value: 10.0
    avg_spread_pips: 1.6
    min_lot: 0.01
    max_lot: 5.0
```

Do not change `execution.max_positions`, `default_risk_pct`, or tactical settings.

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_prop_firm_guard_e8_one.py::TestE8OneConfig::test_config_loads_correctly -q`

Expected: PASS

---

### Task 3: Run targeted config verification

**Files:**
- Test: `tests/test_prop_firm_guard_e8_one.py`
- Test: `tests/test_config.py`

**Step 1: Run targeted tests**

Run: `uv run pytest tests/test_prop_firm_guard_e8_one.py tests/test_config.py -q`

Expected: PASS

**Step 2: Summarize effective changes**

Confirm:

- E8 One 5K universe is now 7 major pairs
- websocket and scheduler symbol universe remain aligned
- scanner can emit up to 5 candidates per cycle
- no runtime code changed

