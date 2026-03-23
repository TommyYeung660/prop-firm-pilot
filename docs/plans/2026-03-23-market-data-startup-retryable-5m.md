# Market Data Startup Retryable 5m Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reclassify the cold-start window where websocket is healthy but has not yet produced the first closed `5m` bar, so scanner candidates enter the retryable tactical path instead of being hard-blocked as stale.

**Architecture:** Keep the fix local to `MarketDataHub.get_entry_readiness()`. Add one regression test that reproduces the stale-REST plus zero-websocket-5m-bar startup state, then add the smallest readiness branch needed to mark that state as retryable.

**Tech Stack:** Python 3.10, pytest, pandas, async market-data hub logic

---

### Task 1: Reproduce the cold-start stale-5m bug

**Files:**
- Modify: `tests/data/test_market_data_hub.py`
- Test: `tests/data/test_market_data_hub.py`

**Step 1: Write the failing test**

Add a test that sets up:

- healthy websocket status,
- fresh quote,
- fresh `1h` bar,
- same-day stale REST-backed `5m` bar,
- zero closed websocket `5m` bars.

Assert the expected fixed behavior:

- `entry_safe is True`
- `requires_tactical_retry is True`
- `pending_reason == "market_data.startup_5m_bar_pending"`

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_market_data_hub.py::test_entry_readiness_treats_stale_same_day_5m_rest_bars_as_startup_retryable_when_websocket_has_no_closed_5m_bar -q`

Expected: `FAIL` because current logic returns `market_data.bars_5m_stale`.

### Task 2: Implement the minimal readiness fix

**Files:**
- Modify: `src/data/market_data_hub.py`
- Test: `tests/data/test_market_data_hub.py`

**Step 1: Add narrow startup detection**

Inside `get_entry_readiness()`:

- inspect the in-memory websocket-derived closed `5m` bars for the symbol,
- when `5m` API bars are stale but websocket is healthy and has zero closed `5m`
  bars, convert the state to retryable startup pending,
- preserve existing hard blocks for missing quote, stale `1h`, prior-trade-date,
  and genuinely stale `5m` bars after startup.

**Step 2: Keep logic minimal**

Do not introduce scheduler changes, partial-bar usage, or broad grace periods.

**Step 3: Run test to verify it passes**

Run: `uv run pytest tests/data/test_market_data_hub.py::test_entry_readiness_treats_stale_same_day_5m_rest_bars_as_startup_retryable_when_websocket_has_no_closed_5m_bar -q`

Expected: `PASS`

### Task 3: Run focused regression coverage

**Files:**
- Verify: `tests/data/test_market_data_hub.py`

**Step 1: Run focused suite**

Run: `uv run pytest tests/data/test_market_data_hub.py -q`

Expected: all tests pass.

**Step 2: Run scheduler regression**

Run: `uv run pytest tests/test_scheduler.py::TestSchedulerScannerLoop::test_creates_intent_when_market_data_gap_is_retryable -q`

Expected: `PASS`

### Task 4: Verify workspace state

**Files:**
- Verify: `src/data/market_data_hub.py`
- Verify: `tests/data/test_market_data_hub.py`
- Verify: `docs/plans/2026-03-23-market-data-startup-retryable-5m-design.md`
- Verify: `docs/plans/2026-03-23-market-data-startup-retryable-5m.md`

**Step 1: Inspect git diff**

Run: `git diff -- src/data/market_data_hub.py tests/data/test_market_data_hub.py docs/plans/2026-03-23-market-data-startup-retryable-5m-design.md docs/plans/2026-03-23-market-data-startup-retryable-5m.md`

Expected: only the hotfix and related documentation changes appear.
