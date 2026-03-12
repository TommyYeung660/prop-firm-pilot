# WebSocket Live Probe And REST Fallback Guard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a production-ready EODHD websocket live probe and stop repeated same-tail REST `1m` refresh loops when fallback data has made no progress.

**Architecture:** Put reusable probe logic in a new diagnostics module so tests can validate summary math directly, then expose it via a thin CLI script. In `MarketDataHub`, track the latest REST refresh state per `(symbol, timeframe)` and suppress repeated fallback refreshes for stale `1m` data when the latest bar timestamp has not advanced.

**Tech Stack:** Python 3.10, asyncio, websockets, httpx, pandas, pytest, ruff

---

### Task 1: Add failing tests for websocket live probe summaries

**Files:**
- Create: `tests/test_eodhd_websocket_live.py`
- Create: `src/diagnostics/__init__.py`
- Create: `src/diagnostics/eodhd_websocket_live.py`

**Step 1: Write the failing test**

Add tests proving that:

- websocket tick events are summarized into per-symbol `count`, `max_gap_sec`, and `latest_age_sec`
- REST `1min` bars are summarized into `rows`, `latest_bar_time`, and `latest_bar_age_sec`

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eodhd_websocket_live.py -q`
Expected: FAIL because the diagnostics module does not exist yet

**Step 3: Write minimal implementation**

- Add reusable summary helpers in `src/diagnostics/eodhd_websocket_live.py`
- Keep them pure where possible for easy test coverage

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eodhd_websocket_live.py -q`
Expected: PASS

### Task 2: Add failing test for repeated stale REST quote fallback

**Files:**
- Modify: `tests/data/test_market_data_hub.py`
- Modify: `src/data/market_data_hub.py`

**Step 1: Write the failing test**

Add a test proving that when:

- websocket quote is unavailable or forced stale
- warm-cache `1m` data is stale
- REST refresh returns the same latest `datetime` as the existing cache

then a second `get_quote()` call inside the retry cooldown does not invoke REST again.

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_market_data_hub.py -k no_progress -q`
Expected: FAIL because `MarketDataHub` currently refreshes every time

**Step 3: Write minimal implementation**

- Track per-key REST refresh attempt time and latest bar timestamp
- Suppress refresh within cooldown when the latest tail has not advanced
- Keep existing fallback return shape unchanged

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/test_market_data_hub.py -k no_progress -q`
Expected: PASS

### Task 3: Expose the live probe as a CLI script

**Files:**
- Create: `scripts/check_eodhd_websocket_live.py`
- Modify: `src/diagnostics/eodhd_websocket_live.py`

**Step 1: Add the CLI wrapper**

- Load `.env`
- Probe websocket for configured symbols
- Probe REST `1min` lag for the same symbols
- Print a concise human-readable summary

**Step 2: Run the script manually**

Run: `uv run python scripts/check_eodhd_websocket_live.py --symbols EURUSD GBPUSD USDJPY AUDUSD --duration 15`
Expected: authorized websocket session plus per-symbol websocket / REST summary output

### Task 4: Verify targeted suites

**Files:**
- Modify: `src/data/market_data_hub.py`
- Create/Modify: `src/diagnostics/eodhd_websocket_live.py`, `scripts/check_eodhd_websocket_live.py`
- Test: `tests/data/test_market_data_hub.py`, `tests/test_eodhd_websocket_live.py`

**Step 1: Run targeted tests**

Run: `uv run pytest tests/test_eodhd_websocket_live.py tests/data/test_market_data_hub.py -q`

**Step 2: Run targeted lint**

Run: `uv run ruff check src/data/market_data_hub.py src/diagnostics/eodhd_websocket_live.py scripts/check_eodhd_websocket_live.py tests/data/test_market_data_hub.py tests/test_eodhd_websocket_live.py`

**Step 3: Run script smoke test**

Run: `uv run python scripts/check_eodhd_websocket_live.py --symbols EURUSD GBPUSD USDJPY AUDUSD --duration 15`

**Step 4: Summarize evidence**

- Confirm websocket tick flow is observable from the script
- Confirm stale REST same-tail retries are suppressed in tests
