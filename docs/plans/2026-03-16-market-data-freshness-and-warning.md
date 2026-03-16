# Market Data Freshness And Warning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修正 closed-bar freshness 語義、節流 stale tactical warnings，並補 market-data fallback instrumentation。

**Architecture:** 保持 OHLC bar schema 不變，另外以 helper 計算 bar effective close time。`MarketDataHub` 與 `Scheduler` 共享同樣的 freshness 語義；warning 以 stateful summary key 做 heartbeat 節流。Instrumentation 只補 log 與測試，不擴大到新 metrics pipeline。

**Tech Stack:** Python 3.10, pandas, pytest, loguru

---

### Task 1: Add failing tests for market-data freshness semantics

**Files:**
- Modify: `tests/data/test_market_data_hub.py`
- Test: `tests/data/test_market_data_hub.py`

**Step 1: Write the failing tests**

- Add a test proving a websocket-aggregated closed `1h` bar is still fresh when judged by close time, even if its open time is older than the cache threshold.
- Add a test proving REST fallback warning logs both latest open time and implied close time.

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_market_data_hub.py -q`

**Step 3: Write minimal implementation**

- Add a helper in `src/data/market_data_hub.py` to compute latest bar effective close time from timeframe duration.
- Update freshness checks and REST fallback instrumentation to use that helper.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_market_data_hub.py -q`

**Step 5: Commit**

```bash
git add tests/data/test_market_data_hub.py src/data/market_data_hub.py
git commit -m "Fix closed-bar freshness semantics in market data hub"
```

### Task 2: Add failing tests for tactical stale-bar handling

**Files:**
- Modify: `tests/test_scheduler.py`
- Test: `tests/test_scheduler.py`

**Step 1: Write the failing tests**

- Add a test proving `_fetch_tactical_data()` keeps a closed `1h` hub bar when only open-time age is stale but close-time age is still valid.
- Add tests proving repeated identical stale tactical warnings are throttled and re-log after heartbeat/state change.

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scheduler.py -q`

**Step 3: Write minimal implementation**

- Update `src/scheduler/scheduler.py` tactical freshness helper to use effective close time.
- Add stateful stale-warning throttling with heartbeat.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_scheduler.py -q`

**Step 5: Commit**

```bash
git add tests/test_scheduler.py src/scheduler/scheduler.py
git commit -m "Throttle stale tactical warnings and fix 1h freshness"
```

### Task 3: Verify the integrated behavior

**Files:**
- Modify: `src/data/market_data_hub.py`
- Modify: `src/scheduler/scheduler.py`
- Test: `tests/data/test_market_data_hub.py`
- Test: `tests/test_scheduler.py`
- Test: `tests/test_tactical_exit_scheduler.py`

**Step 1: Run focused verification**

Run: `uv run pytest tests/data/test_market_data_hub.py tests/test_scheduler.py tests/test_tactical_exit_scheduler.py -q`

Expected: all pass

**Step 2: Run lint**

Run: `uv run ruff check src/data/market_data_hub.py src/scheduler/scheduler.py tests/data/test_market_data_hub.py tests/test_scheduler.py tests/test_tactical_exit_scheduler.py`

Expected: `All checks passed!`

**Step 3: Review diff**

Run: `git diff -- src/data/market_data_hub.py src/scheduler/scheduler.py tests/data/test_market_data_hub.py tests/test_scheduler.py`

**Step 4: Commit**

```bash
git add src/data/market_data_hub.py src/scheduler/scheduler.py tests/data/test_market_data_hub.py tests/test_scheduler.py
git commit -m "Improve tactical market-data freshness diagnostics"
```
