# Market Data WebSocket Bar Priority Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prefer fresh websocket closed bars over unnecessary REST refreshes once warm API cache is stale, and only log degraded REST fallback when REST is actually returned.

**Architecture:** Keep the change local to `MarketDataHub.get_bars()`. First add failing regression tests for the unwanted REST call and warning noise, then reorder the decision path to `warmup_cache -> websocket_cache -> rest_fallback`.

**Tech Stack:** Python 3.10, pytest, pandas, async market-data hub logic

---

### Task 1: Reproduce unnecessary REST refresh when websocket already has fresh bars

**Files:**
- Modify: `tests/data/test_market_data_hub.py`
- Test: `tests/data/test_market_data_hub.py`

**Step 1: Write the failing test**

Add a focused test that sets:

- stale `5m` warm cache,
- healthy websocket with a fresh closed `5m` bar,
- REST provider stub.

Assert:

- `get_bars(..., "5m", ...)` returns `websocket_cache`,
- provider call count stays `0`.

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_market_data_hub.py::test_market_data_hub_prefers_fresh_websocket_closed_5m_bars_before_rest_refresh -q`

Expected: `FAIL` because current code still refreshes REST first.

### Task 2: Reproduce unnecessary REST fallback warning when websocket already has fresh bars

**Files:**
- Modify: `tests/data/test_market_data_hub.py`
- Test: `tests/data/test_market_data_hub.py`

**Step 1: Write the failing test**

Patch `src.data.market_data_hub.logger.warning` and reuse the same stale-warm /
fresh-websocket setup.

Assert:

- `get_bars(..., "5m", ...)` returns `websocket_cache`,
- `logger.warning` is not called.

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_market_data_hub.py::test_market_data_hub_skips_rest_fallback_warning_when_websocket_closed_5m_bars_are_used -q`

Expected: `FAIL` because current code logs before checking websocket.

### Task 3: Implement minimal routing change

**Files:**
- Modify: `src/data/market_data_hub.py`
- Test: `tests/data/test_market_data_hub.py`

**Step 1: Reorder `get_bars()`**

After checking fresh warm cache:

- consult websocket closed bars first for non-forced-stale symbols,
- return `websocket_cache` immediately when fresh,
- otherwise perform REST refresh.

**Step 2: Tighten fallback logging**

Only call `_log_rest_fallback()` when the returned source is actually
`rest_fallback`.

**Step 3: Keep behavior narrow**

Do not change quote routing, refresh cooldown logic, or tactical stale-bar
filtering.

### Task 4: Verify focused coverage

**Files:**
- Verify: `src/data/market_data_hub.py`
- Verify: `tests/data/test_market_data_hub.py`

**Step 1: Run targeted regression tests**

Run:

- `uv run pytest tests/data/test_market_data_hub.py::test_market_data_hub_prefers_fresh_websocket_closed_5m_bars_before_rest_refresh -q`
- `uv run pytest tests/data/test_market_data_hub.py::test_market_data_hub_skips_rest_fallback_warning_when_websocket_closed_5m_bars_are_used -q`

Expected: both pass.

**Step 2: Run broader regression**

Run:

- `uv run pytest tests/data/test_market_data_hub.py -q`
- `uv run ruff check src/data/market_data_hub.py tests/data/test_market_data_hub.py`

Expected: all green.
