# MarketDataHub REST Fallback v1.4.6d Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修復 `MarketDataHub` 在 `5m/1h` stale same-tail 情境下重複 REST refresh 的迴圈，並把 runtime elapsed-bar close 正式接到 scheduler，使 websocket closed rollup bars 能自然 materialize，最後發版為 `v1.4.6d`。

**Architecture:** 這次修復分成兩條主線。第一條是在 `MarketDataHub` 內把既有 `1m` refresh suppression 擴展成可覆蓋 `5m` / `1h` 的共用機制，避免 stale warm cache 在 cooldown 內無進展時反覆打 REST。第二條是在 scheduler market-data sidecar 增加輕量 background loop，定期呼叫 `FXTickAggregator.close_elapsed_bars()`，讓 closed `5m` / `1h` bars 不再依賴下一個跨 bucket tick 才 finalize。

**Tech Stack:** Python 3.10, asyncio, pandas, pytest, ruff, loguru, Pydantic v2

---

### Task 1: MarketDataHub 5m/1h Refresh Suppression

**Files:**
- Modify: `src/data/market_data_hub.py`
- Test: `tests/data/test_market_data_hub.py`
- Reference: `docs/PropFirmPilot_MarketDataHub_REST_Fallback_Root_Cause_Report.md`

**Step 1: Write the failing tests**

Add regression tests proving:
- repeated stale `get_bars("5m")` calls only hit REST once within cooldown when the tail does not advance
- repeated stale `get_bars("1h")` calls only hit REST once within cooldown when the tail does not advance
- suppression must still allow a later refresh when cooldown expires or the cached tail advances

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_market_data_hub.py -q`
Expected: new `5m/1h` suppression tests fail because current implementation always refreshes non-`1m` timeframes

**Step 3: Write minimal implementation**

Change `MarketDataHub` so `_should_refresh_rest_cache()` becomes timeframe-agnostic and `get_bars()` only refreshes when the suppression gate allows it. Keep source semantics stable: stale fallback still returns `rest_fallback`, but repeated no-progress fetches must reuse the existing warm cache without another REST call or duplicate warning.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_market_data_hub.py -q`
Expected: all market-data hub tests pass, including the new stale `5m/1h` suppression coverage

### Task 2: Scheduler Runtime Elapsed-Bar Close Wiring

**Files:**
- Modify: `src/scheduler/scheduler.py`
- Test: `tests/test_scheduler.py`
- Optional test touch: `tests/data/test_fx_tick_aggregator.py`

**Step 1: Write the failing tests**

Add scheduler regression coverage proving:
- scheduler startup creates a background market-data sidecar task that closes elapsed bars in the aggregator on a fixed cadence
- scheduler shutdown cancels that sidecar task together with the websocket task
- the sidecar is only started when websocket market data is enabled and initialized

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scheduler.py -q`
Expected: new scheduler test fails because there is no elapsed-bar closer task yet

**Step 3: Write minimal implementation**

Add a scheduler-owned async loop that periodically calls `self._tick_aggregator.close_elapsed_bars(now=self._now_utc())`. Start it during `_initialize_market_data_hub()` after warmup and stop it in `stop()`. Keep the loop isolated from tactical logic; no scope expansion into `evaluation_interval_seconds` rewiring unless strictly required by tests.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_scheduler.py -q`
Expected: scheduler tests pass and the new runtime wiring regression stays green

### Task 3: Release Bump And Documentation

**Files:**
- Modify: `pyproject.toml`
- Modify: `docs/PropFirmPilot_changelog.md`

**Step 1: Add release metadata changes**

Update the shared project version from `1.4.6b` to `1.4.6d` and add a `v1.4.6d` changelog entry that documents:
- `5m/1h` REST fallback suppression
- runtime elapsed-bar close wiring for websocket rollups
- targeted regression coverage added in this fix

**Step 2: Verify version helpers resolve correctly**

Run: `uv run python -c "from src.version import get_app_version, get_release_tag; print(get_app_version()); print(get_release_tag())"`
Expected:
- `1.4.6d`
- `v1.4.6d`

### Task 4: Focused Verification

**Files:**
- N/A

**Step 1: Run targeted regression suites**

Run: `uv run pytest tests/data/test_market_data_hub.py tests/test_scheduler.py tests/data/test_fx_tick_aggregator.py -q`
Expected: all targeted tests pass

**Step 2: Run lint on changed files**

Run: `uv run ruff check src/data/market_data_hub.py src/scheduler/scheduler.py tests/data/test_market_data_hub.py tests/test_scheduler.py tests/data/test_fx_tick_aggregator.py`
Expected: `All checks passed!`

**Step 3: Review working tree**

Run: `git status --short`
Expected: only intended code/doc changes plus the pre-existing dirty files remain
