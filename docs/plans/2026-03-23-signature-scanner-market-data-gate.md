# Signature Scanner Market-Data Gate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 讓 scanner 不再因同日 `1h` stale bars 提前阻擋 TradeLocker / Signature runtime，改由 tactical layer 接手處理。

**Architecture:** 調整 `MarketDataHub.get_entry_readiness()`，把 scanner gate 縮到只處理 quote availability、trade-date readiness 與 `5m` 冷啟動 retry；不再由 scanner 對 `1h stale` 做硬阻擋。行為透過 `tests/data/test_market_data_hub.py` 先寫失敗測試後再最小化修改實作。

**Tech Stack:** Python 3.10, pandas, pytest, async market-data hub

---

### Task 1: 覆蓋 `1h stale` 不再硬阻擋

**Files:**
- Modify: `tests/data/test_market_data_hub.py`
- Modify: `src/data/market_data_hub.py`

**Step 1: Write the failing test**

更新現有 `test_entry_readiness_blocks_when_1h_bars_are_stale_even_with_fresh_quote`，改成驗證：

```python
assert readiness.entry_safe is True
assert readiness.block_reason == ""
assert readiness.requires_tactical_retry is False
assert readiness.bars_5m_fresh is True
assert readiness.bars_1h_fresh is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_market_data_hub.py::test_entry_readiness_blocks_when_1h_bars_are_stale_even_with_fresh_quote -q`

Expected: FAIL because current implementation still returns `market_data.bars_1h_stale`

**Step 3: Write minimal implementation**

在 `src/data/market_data_hub.py` 移除 `bars_1h_stale` 的 scanner block，保留 `trade_date_not_ready` 檢查。

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/test_market_data_hub.py::test_entry_readiness_blocks_when_1h_bars_are_stale_even_with_fresh_quote -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/data/test_market_data_hub.py src/data/market_data_hub.py docs/plans/2026-03-23-signature-scanner-market-data-gate-design.md docs/plans/2026-03-23-signature-scanner-market-data-gate.md
git commit -m "fix: align scanner market-data gate with tactical flow"
```

### Task 2: 覆蓋同日 `5m` stale + `1h` stale 的 cold-start retry

**Files:**
- Modify: `tests/data/test_market_data_hub.py`
- Modify: `src/data/market_data_hub.py`

**Step 1: Write the failing test**

新增測試，模擬：

- broker quote 可用
- websocket `healthy`
- `5m` REST bar 為同日但 stale
- `1h` REST bar 同日但也 stale
- aggregator 尚未產出 closed `5m` websocket bar

預期：

```python
assert readiness.entry_safe is True
assert readiness.requires_tactical_retry is True
assert readiness.pending_reason == "market_data.startup_5m_bar_pending"
assert readiness.bars_1h_fresh is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_market_data_hub.py::test_entry_readiness_marks_same_day_stale_5m_gap_as_retryable_even_when_1h_is_stale -q`

Expected: FAIL because current startup retry helper still requires `bars_1h_fresh`

**Step 3: Write minimal implementation**

調整 `_is_startup_5m_bar_pending()`，拿掉對 `bars_1h_fresh` 的依賴。

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/test_market_data_hub.py::test_entry_readiness_marks_same_day_stale_5m_gap_as_retryable_even_when_1h_is_stale -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/data/test_market_data_hub.py src/data/market_data_hub.py
git commit -m "fix: keep startup 5m retry path active when 1h bars lag"
```

### Task 3: Run the focused regression suite

**Files:**
- Test: `tests/data/test_market_data_hub.py`

**Step 1: Run targeted suite**

Run: `uv run pytest tests/data/test_market_data_hub.py -q`

Expected: PASS

**Step 2: Run lint for touched files**

Run: `uv run ruff check src/data/market_data_hub.py tests/data/test_market_data_hub.py`

Expected: PASS

**Step 3: Commit**

```bash
git add src/data/market_data_hub.py tests/data/test_market_data_hub.py
git commit -m "test: cover scanner market-data gate alignment"
```
