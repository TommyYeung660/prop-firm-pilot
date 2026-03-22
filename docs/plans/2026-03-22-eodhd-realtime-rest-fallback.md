# EODHD Real-Time REST Fallback Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Route market data through `websocket -> EODHD real-time REST -> intraday REST -> fail-closed` without introducing third-party data sources.

**Architecture:** Add an EODHD real-time REST quote fetcher, teach `MarketDataHub` to use it when websocket quotes are stale, and feed normalized snapshots into the existing `FXTickAggregator` so fresh `5m/1h` bars continue to close even when `/api/intraday` rollups lag.

**Tech Stack:** Python 3.10, `httpx`, pandas, loguru, pytest, respx.

---

### Task 1: Add failing provider tests for EODHD real-time REST

**Files:**
- Modify: `tests/test_fx_data_fetcher.py`
- Modify: `src/data/fx_data_fetcher.py`

**Step 1: Write the failing test**

Add tests covering:

- successful fetch from `/api/real-time/EURUSD.FOREX`
- payload normalization to `symbol/bid/ask/mid/timestamp_ms`
- empty or malformed payload returns `None`

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_fx_data_fetcher.py -k realtime
```

Expected: failure because the real-time provider/helper does not exist yet.

**Step 3: Write minimal implementation**

Implement the smallest possible EODHD real-time REST fetcher/helper in `src/data/fx_data_fetcher.py`.

**Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/test_fx_data_fetcher.py -k realtime
```

Expected: pass.

### Task 2: Add failing MarketDataHub quote fallback tests

**Files:**
- Modify: `tests/data/test_market_data_hub.py`
- Modify: `src/data/market_data_hub.py`

**Step 1: Write the failing test**

Add tests for:

- websocket stale + real-time REST available => `get_quote()` returns `rest_realtime`
- returned snapshot is fed into aggregator
- duplicate real-time timestamp is not re-fed

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/data/test_market_data_hub.py -k realtime
```

Expected: failure because `MarketDataHub` has no real-time REST path.

**Step 3: Write minimal implementation**

Extend `MarketDataHub` constructor and `get_quote()` to:

- accept a real-time REST quote provider
- normalize returned payload
- feed synthetic tick to aggregator
- mark quote source as `rest_realtime`

**Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/data/test_market_data_hub.py -k realtime
```

Expected: pass.

### Task 3: Add failing bar-routing tests for aggregator-first degraded mode

**Files:**
- Modify: `tests/data/test_market_data_hub.py`
- Modify: `src/data/market_data_hub.py`

**Step 1: Write the failing test**

Add tests for:

- stale intraday REST `5m` + fresh aggregator bars => `get_bars(..., "5m")` returns `websocket_cache`
- same pattern for `1h`
- still returns `rest_fallback` or empty when both aggregator and intraday are stale

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/data/test_market_data_hub.py -k \"realtime or aggregator\"
```

Expected: failure because current ordering prefers intraday REST before aggregator in degraded cases.

**Step 3: Write minimal implementation**

Adjust `get_bars()` ordering so:

- fresh warm cache remains first
- fresh aggregator bars are checked before stale intraday fallback result is returned

**Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/data/test_market_data_hub.py -k \"realtime or aggregator\"
```

Expected: pass.

### Task 4: Wire the new provider into scheduler construction

**Files:**
- Modify: `src/scheduler/scheduler.py`
- Modify: `tests/test_scheduler.py`

**Step 1: Write the failing test**

Add a scheduler initialization test asserting `MarketDataHub` receives an EODHD real-time REST provider callable in addition to the existing broker quote provider and intraday REST provider.

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_scheduler.py -k market_data_hub
```

Expected: failure because constructor arguments do not yet include the new provider.

**Step 3: Write minimal implementation**

Create a scheduler helper that fetches EODHD real-time REST snapshots and inject it into `MarketDataHub`.

**Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/test_scheduler.py -k market_data_hub
```

Expected: pass.

### Task 5: Run focused regression suite

**Files:**
- No code changes

**Step 1: Run focused verification**

Run:

```bash
uv run pytest tests/data/test_market_data_hub.py tests/data/test_fx_websocket_client.py tests/test_fx_data_fetcher.py tests/test_eodhd_websocket_live.py tests/test_scheduler.py -k \"market_data_hub or realtime or eodhd\"
```

Expected: all relevant tests pass.

**Step 2: Run lint for touched files**

Run:

```bash
uv run ruff check src/data/fx_data_fetcher.py src/data/market_data_hub.py src/scheduler/scheduler.py tests/test_fx_data_fetcher.py tests/data/test_market_data_hub.py tests/test_scheduler.py
```

Expected: no lint errors.

### Task 6: Record outcome

**Files:**
- Optionally modify: `docs/PropFirmPilot_changelog.md`

**Step 1: Add concise changelog note if implementation completes**

Mention that degraded forex market-data routing now prefers EODHD real-time REST before intraday historical fallback.

**Step 2: Final verification**

Re-run the focused pytest and ruff commands after any changelog/doc edits.
