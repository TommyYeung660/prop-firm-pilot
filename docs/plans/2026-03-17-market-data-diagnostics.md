# Market Data Diagnostics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 為 `v1.5.0_beta_2` 增加 market-data hub lifecycle 與 websocket closed-bar diagnostics，並把這些欄位帶進 scheduler 的 metrics / scanner block payload。

**Architecture:** 保持現有 market-data pipeline 與 entry guard 判定不變，只在 `FXTickAggregator` 暴露只讀 closed-bar counts，並由 `MarketDataHub.feed_status()` 組成統一 diagnostics payload。`Scheduler` 重用同一份 feed status，避免再建一套平行診斷結構。

**Tech Stack:** Python 3.10, pandas, pytest, loguru

---

### Task 1: Add failing tests for market-data hub diagnostics

**Files:**
- Modify: `tests/data/test_market_data_hub.py`
- Test: `tests/data/test_market_data_hub.py`

**Step 1: Write the failing test**

```python
def test_feed_status_reports_lifecycle_and_closed_bar_counts():
    status = hub.feed_status()
    assert status["initialized_at"] == "2026-03-17T03:00:00+00:00"
    assert status["uptime_seconds"] == 390
    assert status["websocket_closed_bar_counts"]["EURUSD"] == {"1m": 6, "5m": 1, "1h": 0}
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/data/test_market_data_hub.py::test_feed_status_reports_lifecycle_and_closed_bar_counts -q`

Expected: `KeyError` / assertion failure because the new fields do not exist yet.

**Step 3: Write minimal implementation**

- Add a diagnostics method to `src/data/fx_tick_aggregator.py` that returns per-symbol closed bar counts.
- Add `initialized_at` and `uptime_seconds` handling to `src/data/market_data_hub.py`.
- Extend `feed_status()` to include the new lifecycle and closed-bar fields.

**Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/data/test_market_data_hub.py::test_feed_status_reports_lifecycle_and_closed_bar_counts -q`

Expected: `1 passed`

**Step 5: Commit**

```bash
git add tests/data/test_market_data_hub.py src/data/fx_tick_aggregator.py src/data/market_data_hub.py
git commit -m "Add market data hub diagnostics"
```

### Task 2: Add failing tests for scheduler diagnostics propagation

**Files:**
- Modify: `tests/test_scheduler.py`
- Test: `tests/test_scheduler.py`

**Step 1: Write the failing tests**

```python
def test_build_metrics_snapshot_includes_market_data_diagnostics():
    snapshot = sched._build_metrics_snapshot()
    assert snapshot["market_data"]["uptime_seconds"] == 42

async def test_scanner_loop_logs_market_data_guard_diagnostics(...):
    await _run_loop_once(scheduler, scheduler._scanner_loop())
    assert logged_payload["market_data_uptime_seconds"] == 42
```

**Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_scheduler.py::test_build_metrics_snapshot_includes_market_data_diagnostics tests/test_scheduler.py::TestScannerLoop::test_logs_market_data_entry_guard_with_feed_diagnostics -q`

Expected: assertion failure because the diagnostics fields are not propagated yet.

**Step 3: Write minimal implementation**

- Reuse `MarketDataHub.feed_status()` inside `src/scheduler/scheduler.py` when building the scanner block payload.
- Add the lifecycle and closed-bar diagnostics to the structured warning / incident payload without changing block conditions.

**Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_scheduler.py::test_build_metrics_snapshot_includes_market_data_diagnostics tests/test_scheduler.py::TestScannerLoop::test_logs_market_data_entry_guard_with_feed_diagnostics -q`

Expected: all pass

**Step 5: Commit**

```bash
git add tests/test_scheduler.py src/scheduler/scheduler.py
git commit -m "Propagate market data diagnostics to scheduler logs"
```

### Task 3: Verify integrated behavior

**Files:**
- Modify: `src/data/fx_tick_aggregator.py`
- Modify: `src/data/market_data_hub.py`
- Modify: `src/scheduler/scheduler.py`
- Test: `tests/data/test_market_data_hub.py`
- Test: `tests/test_scheduler.py`

**Step 1: Run focused tests**

Run: `uv run python -m pytest tests/data/test_market_data_hub.py tests/test_scheduler.py -q`

Expected: all pass

**Step 2: Run lint**

Run: `uv run ruff check src/data/fx_tick_aggregator.py src/data/market_data_hub.py src/scheduler/scheduler.py tests/data/test_market_data_hub.py tests/test_scheduler.py`

Expected: `All checks passed!`

**Step 3: Review diff**

Run: `git diff -- src/data/fx_tick_aggregator.py src/data/market_data_hub.py src/scheduler/scheduler.py tests/data/test_market_data_hub.py tests/test_scheduler.py docs/plans/2026-03-17-market-data-diagnostics-design.md docs/plans/2026-03-17-market-data-diagnostics.md`

**Step 4: Commit**

```bash
git add src/data/fx_tick_aggregator.py src/data/market_data_hub.py src/scheduler/scheduler.py tests/data/test_market_data_hub.py tests/test_scheduler.py docs/plans/2026-03-17-market-data-diagnostics-design.md docs/plans/2026-03-17-market-data-diagnostics.md
git commit -m "Add market data diagnostics for entry guard triage"
```
