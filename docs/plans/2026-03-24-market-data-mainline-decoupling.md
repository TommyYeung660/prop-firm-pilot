# Market Data Mainline Decoupling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 讓 market-data 主線不再把 websocket 當成 scanner / tactical 的隱性 hard dependency，並以 realtime quote polling 補足 live ingest。

**Architecture:** scheduler 永遠初始化 `MarketDataHub`，並新增 realtime polling loop 餵 `FXTickAggregator`。`MarketDataHub.get_entry_readiness()` 改成 aggregator-driven startup retry；`TacticalValidator` 對 fresh quote + missing `5m` bars 一律 `WAIT`。websocket 保留為 auxiliary ingest，不再決定主線語義。

**Tech Stack:** Python 3.10, asyncio, httpx, pandas, pytest

---

### Task 1: 寫 market-data readiness 的失敗測試

**Files:**
- Modify: `tests/data/test_market_data_hub.py`
- Modify: `src/data/market_data_hub.py`

**Step 1: Write the failing test**

新增測試，模擬：

- websocket client 未連線 / disabled semantics
- broker quote 可用
- `5m` warm cache 為同日 stale
- aggregator 尚未有 closed `5m` bar

預期：

```python
assert readiness.entry_safe is True
assert readiness.requires_tactical_retry is True
assert readiness.pending_reason == "market_data.startup_5m_bar_pending"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_market_data_hub.py::test_entry_readiness_marks_same_day_stale_5m_gap_as_retryable_without_websocket_health -q`

Expected: FAIL because current implementation still requires websocket `healthy`

**Step 3: Write minimal implementation**

在 `src/data/market_data_hub.py` 移除 `_is_startup_5m_bar_pending()` 對 websocket health 的依賴。

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/test_market_data_hub.py::test_entry_readiness_marks_same_day_stale_5m_gap_as_retryable_without_websocket_health -q`

Expected: PASS

### Task 2: 寫 tactical missing-5m-bars 的失敗測試

**Files:**
- Modify: `tests/test_tactical_validator.py`
- Modify: `src/decision/tactical_validator.py`

**Step 1: Write the failing test**

新增測試，模擬：

- `latest_bar_time` 來自 fresh quote
- `bars_5min` empty
- `bars_1h` 可空
- `quote_source = "broker_quote"`

預期：

```python
assert result.action == "WAIT"
assert result.summary_reason_code == "market_data.startup_5m_bar_pending"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tactical_validator.py::test_waits_for_first_5m_bar_when_fresh_quote_exists_but_5m_bars_are_missing_from_broker_quote -q`

Expected: FAIL because current implementation only special-cases `websocket_cache`

**Step 3: Write minimal implementation**

把 `src/decision/tactical_validator.py` 的 startup wait 條件改成「fresh quote exists + `bars_5min.empty`」。

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tactical_validator.py::test_waits_for_first_5m_bar_when_fresh_quote_exists_but_5m_bars_are_missing_from_broker_quote -q`

Expected: PASS

### Task 3: 寫 scheduler hub-init / polling 的失敗測試

**Files:**
- Modify: `tests/test_scheduler.py`
- Modify: `src/scheduler/scheduler.py`

**Step 1: Write the failing test**

新增測試，驗證：

- `config.websocket.enabled = False`
- `_initialize_market_data_hub()` 仍建立 `MarketDataHub`
- 會建立 realtime polling task
- 不會建立 websocket run task

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scheduler.py::test_initialize_market_data_hub_keeps_hub_when_websocket_disabled -q`

Expected: FAIL because current implementation直接 early return

**Step 3: Write minimal implementation**

在 `src/scheduler/scheduler.py`：

- 讓 hub initialization 不再受 `websocket.enabled` 阻擋
- 新增 `_market_data_poll_task`
- `websocket.enabled` 只決定是否啟 websocket ingest task

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_scheduler.py::test_initialize_market_data_hub_keeps_hub_when_websocket_disabled -q`

Expected: PASS

### Task 4: 寫 realtime polling ingest 的失敗測試

**Files:**
- Modify: `tests/test_scheduler.py`
- Modify: `src/scheduler/scheduler.py`
- Modify: `src/data/market_data_hub.py`

**Step 1: Write the failing test**

新增測試，驗證 `_poll_market_data_once()`：

- 會對每個 symbol 呼叫 `EodhdRealtimeProvider.fetch_quote`
- 對有效 quote 呼叫 hub 的 ingest 方法
- 對 `None` quote 不 ingest

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scheduler.py::test_poll_market_data_once_ingests_realtime_quotes_into_hub -q`

Expected: FAIL because helper / ingest method 尚不存在

**Step 3: Write minimal implementation**

- 在 `MarketDataHub` 新增 public ingest method
- 在 `Scheduler` 新增 `_poll_market_data_once()` 與 `_market_data_poll_loop()`

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_scheduler.py::test_poll_market_data_once_ingests_realtime_quotes_into_hub -q`

Expected: PASS

### Task 5: 跑 focused regression

**Files:**
- Test: `tests/data/test_market_data_hub.py`
- Test: `tests/test_tactical_validator.py`
- Test: `tests/test_scheduler.py`

**Step 1: Run focused tests**

Run: `uv run pytest tests/data/test_market_data_hub.py tests/test_tactical_validator.py tests/test_scheduler.py -q`

Expected: PASS

**Step 2: Run lint**

Run: `uv run ruff check src/data/market_data_hub.py src/decision/tactical_validator.py src/scheduler/scheduler.py tests/data/test_market_data_hub.py tests/test_tactical_validator.py tests/test_scheduler.py`

Expected: PASS

**Step 3: Summarize runtime impact**

記錄：

- websocket disabled/degraded 時，hub 是否仍在
- tactical 是否不再因 missing `5m` bars 誤入 pass-through
- realtime polling 是否成功餵 aggregator
