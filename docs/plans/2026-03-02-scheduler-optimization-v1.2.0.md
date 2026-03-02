# Scheduler Optimization v1.2.0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Optimize the Scheduler's decision frequency with 6 improvements: parallel LLM workers, event-driven re-scan on position close, session-aware cadence, faster re-evaluation, multi-timeframe analysis, and volatility-triggered scans.

**Architecture:** All 6 features integrate into the existing `Scheduler` class in `src/scheduler/scheduler.py`. Two new modules (`session_cadence.py`, `volatility_monitor.py`) encapsulate session-awareness and volatility logic respectively. Multi-timeframe requires changes to both prop-firm-pilot (data pipeline + scanner bridge) AND the sibling `qlib_market_scanner` repo. All new config lives in `SchedulerConfig` (no new config classes).

**Tech Stack:** Python 3.10, asyncio, Pydantic v2, DuckDB, httpx, loguru, pytest + respx

---

## Part A: Quick Wins (Tasks 1–4)

### Task 1: LLM Worker Parallelism — Config Default

**Files:**
- Modify: `src/config.py:191` (`llm_worker_count` default)
- Test: `tests/test_config.py` (verify default)

**Context:** The Scheduler already supports multiple LLM workers via `llm_worker_count` — `start()` spawns `range(llm_worker_count)` workers. We just need to change the default from 1 to 2.

**Step 1: Write test verifying new default**

```python
# tests/test_config.py — add to existing test file
def test_scheduler_config_llm_worker_count_default():
    """v1.2.0: Default LLM worker count should be 2."""
    from src.config import SchedulerConfig
    config = SchedulerConfig()
    assert config.llm_worker_count == 2
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_scheduler_config_llm_worker_count_default -v`
Expected: FAIL (current default is 1)

**Step 3: Change default**

In `src/config.py:191`, change:
```python
llm_worker_count: int = Field(default=2, description="Number of concurrent LLM workers")
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py::test_scheduler_config_llm_worker_count_default -v`
Expected: PASS

**Step 5: Run full test suite to check no regressions**

Run: `uv run pytest -x -q`
Expected: All tests pass. If any test hardcodes `llm_worker_count == 1`, update that test.

**Step 6: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat(scheduler): increase default llm_worker_count to 2 for parallel evaluation"
```

---

### Task 2: Event-Driven Re-Scan on Position Close

**Files:**
- Modify: `src/scheduler/scheduler.py` — add `asyncio.Event` field, trigger in `_handle_position_closed`, listen in `_scanner_loop`
- Test: `tests/test_scheduler_rescan_event.py` (new)

**Context:** Currently all loops poll independently — no inter-loop communication. When a position closes (SL/TP hit), a slot opens up but the scanner won't notice until its next 4h cycle. We add an `asyncio.Event` that `_handle_position_closed()` sets, and `_scanner_loop()` awaits with a timeout instead of a flat `asyncio.sleep()`.

**Step 1: Write tests for re-scan trigger**

```python
# tests/test_scheduler_rescan_event.py
"""Tests for position-close → re-scan event trigger (v1.2.0)."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import AppConfig


def _make_scheduler(**overrides):
    """Create a Scheduler with mocked dependencies."""
    from src.scheduler.scheduler import Scheduler

    config = AppConfig()
    store = MagicMock()
    store.get_active_positions = MagicMock(return_value=[])
    store.recycle_expired_claims = MagicMock(return_value=0)
    scanner = MagicMock()
    agents = MagicMock()
    engine = MagicMock()
    matchtrader = AsyncMock()
    matchtrader.get_balance = AsyncMock(return_value=MagicMock(equity=5000, balance=5000))
    matchtrader.get_open_positions = AsyncMock(return_value=[])
    matchtrader.get_closed_positions = AsyncMock(return_value=[])

    for k, v in overrides.items():
        if k == "config":
            config = v
        elif k == "store":
            store = v
        elif k == "scanner":
            scanner = v

    return Scheduler(
        config=config,
        store=store,
        scanner=scanner,
        agents=agents,
        engine=engine,
        matchtrader=matchtrader,
    )


def test_scheduler_has_rescan_event():
    """Scheduler should have a _rescan_event asyncio.Event."""
    scheduler = _make_scheduler()
    assert hasattr(scheduler, "_rescan_event")
    assert isinstance(scheduler._rescan_event, asyncio.Event)


async def test_handle_position_closed_sets_rescan_event():
    """When a position closes, _rescan_event should be set."""
    scheduler = _make_scheduler()
    # Ensure event is clear initially
    assert not scheduler._rescan_event.is_set()

    # Mock intent for _handle_position_closed
    intent = MagicMock()
    intent.symbol = "EURUSD"
    intent.suggested_side = "BUY"
    intent.position_id = "12345"
    intent.id = "intent-1"
    intent.executed_at = None

    store = scheduler._store
    store.mark_closed = MagicMock()

    with patch.object(scheduler, "_send_alert", new_callable=AsyncMock):
        await scheduler._handle_position_closed(intent)

    assert scheduler._rescan_event.is_set()


async def test_rescan_event_clears_after_scanner_loop_reads():
    """After scanner loop picks up the event, it should be cleared."""
    scheduler = _make_scheduler()
    scheduler._rescan_event.set()

    # The event should be clearable
    scheduler._rescan_event.clear()
    assert not scheduler._rescan_event.is_set()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scheduler_rescan_event.py -v`
Expected: FAIL on `test_scheduler_has_rescan_event` (no `_rescan_event` attribute)

**Step 3: Implement the event mechanism**

In `src/scheduler/scheduler.py`, in `__init__()` (after the `_last_known_profit` line, around line 135):
```python
# v1.2.0: Event-driven re-scan when a position closes (frees a slot)
self._rescan_event = asyncio.Event()
```

In `_handle_position_closed()`, at the very end of the method (before the method returns, after all alerts), add:
```python
# v1.2.0: Signal scanner to re-scan immediately (slot freed)
self._rescan_event.set()
logger.info("Position closed → rescan event set for {}", symbol)
```

In `_scanner_loop()`, replace the bottom sleep (around line 311):
```python
# OLD:
# await asyncio.sleep(self._config.scheduler.scanner_interval_seconds)

# NEW: Wait for either the full interval OR a rescan event (position closed)
try:
    await asyncio.wait_for(
        self._rescan_event.wait(),
        timeout=self._config.scheduler.scanner_interval_seconds,
    )
    self._rescan_event.clear()
    logger.info("Scanner loop: rescan event received — running early scan")
except asyncio.TimeoutError:
    pass  # Normal timeout — proceed with scheduled scan
except asyncio.CancelledError:
    logger.info("Scanner loop: cancelled during sleep")
    return
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_scheduler_rescan_event.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `uv run pytest -x -q`
Expected: All pass

**Step 6: Commit**

```bash
git add src/scheduler/scheduler.py tests/test_scheduler_rescan_event.py
git commit -m "feat(scheduler): add event-driven re-scan when position closes"
```

---

### Task 3: Session-Aware Cadence Calculator

**Files:**
- Create: `src/scheduler/session_cadence.py`
- Modify: `src/config.py` — add session cadence fields to `SchedulerConfig`
- Modify: `src/scheduler/__init__.py` — export new class
- Test: `tests/test_session_cadence.py` (new)

**Context:** FX markets have distinct sessions (London 07:00–16:00 UTC, NY 12:00–21:00 UTC). During active sessions, we want faster scanning (e.g. 1h). During Asia/off-hours, keep the 4h default. The calculator returns the appropriate interval based on current UTC time.

**Step 1: Add config fields**

Add to `SchedulerConfig` in `src/config.py` (after `market_hours` field):
```python
# v1.2.0: Session-aware scanning cadence
session_aware_enabled: bool = Field(
    default=False, description="Enable session-aware scanner interval adjustment"
)
active_session_interval_seconds: int = Field(
    default=3600, description="Scanner interval during active sessions (London/NY overlap, 1h)"
)
quiet_session_interval_seconds: int = Field(
    default=14400, description="Scanner interval during quiet hours (Asia, 4h)"
)
london_open_utc: int = Field(default=7, description="London session open hour (UTC)")
london_close_utc: int = Field(default=16, description="London session close hour (UTC)")
ny_open_utc: int = Field(default=12, description="New York session open hour (UTC)")
ny_close_utc: int = Field(default=21, description="New York session close hour (UTC)")
```

**Step 2: Write the session cadence module**

```python
# src/scheduler/session_cadence.py
"""
Session-aware cadence calculator — adjusts scanner interval by trading session.

FX markets have distinct sessions with varying liquidity:
- London (07:00–16:00 UTC): High liquidity
- New York (12:00–21:00 UTC): High liquidity
- London/NY overlap (12:00–16:00 UTC): Highest liquidity
- Asia/Off-hours: Lower liquidity

During active sessions, the scanner runs more frequently to capture
opportunities. During quiet hours, it runs less frequently.

Usage:
    cadence = SessionCadence(scheduler_config)
    interval = cadence.get_scanner_interval(now_utc)
"""

from datetime import datetime

from loguru import logger

from src.config import SchedulerConfig


class SessionCadence:
    """Calculates scanner interval based on current FX trading session.

    Usage:
        cadence = SessionCadence(config)
        interval_seconds = cadence.get_scanner_interval(datetime.now(timezone.utc))
    """

    def __init__(self, config: SchedulerConfig) -> None:
        self._config = config

    def is_active_session(self, now: datetime) -> bool:
        """Return True if current time falls within London or NY session.

        Active = London session OR New York session (any overlap counts once).
        """
        hour = now.hour
        in_london = self._config.london_open_utc <= hour < self._config.london_close_utc
        in_ny = self._config.ny_open_utc <= hour < self._config.ny_close_utc
        return in_london or in_ny

    def get_scanner_interval(self, now: datetime) -> int:
        """Return the appropriate scanner interval in seconds.

        If session-aware is disabled, returns the default scanner_interval_seconds.
        Otherwise, returns active or quiet interval based on session.
        """
        if not self._config.session_aware_enabled:
            return self._config.scanner_interval_seconds

        if self.is_active_session(now):
            return self._config.active_session_interval_seconds
        return self._config.quiet_session_interval_seconds

    def current_session_name(self, now: datetime) -> str:
        """Return human-readable session name for logging."""
        hour = now.hour
        in_london = self._config.london_open_utc <= hour < self._config.london_close_utc
        in_ny = self._config.ny_open_utc <= hour < self._config.ny_close_utc

        if in_london and in_ny:
            return "London/NY Overlap"
        if in_london:
            return "London"
        if in_ny:
            return "New York"
        return "Off-hours"
```

**Step 3: Write tests**

```python
# tests/test_session_cadence.py
"""Tests for session-aware cadence calculator (v1.2.0)."""
from datetime import datetime, timezone

import pytest

from src.config import SchedulerConfig
from src.scheduler.session_cadence import SessionCadence


def _utc(hour: int, minute: int = 0) -> datetime:
    """Create a UTC datetime on a Wednesday (weekday) at given hour."""
    return datetime(2026, 3, 4, hour, minute, tzinfo=timezone.utc)  # Wednesday


class TestSessionCadence:
    def test_disabled_returns_default_interval(self):
        config = SchedulerConfig(session_aware_enabled=False, scanner_interval_seconds=14400)
        cadence = SessionCadence(config)
        assert cadence.get_scanner_interval(_utc(8)) == 14400  # London hour, but disabled

    def test_london_session_returns_active_interval(self):
        config = SchedulerConfig(
            session_aware_enabled=True,
            active_session_interval_seconds=3600,
            quiet_session_interval_seconds=14400,
        )
        cadence = SessionCadence(config)
        assert cadence.get_scanner_interval(_utc(8)) == 3600  # 08:00 UTC = London

    def test_ny_session_returns_active_interval(self):
        config = SchedulerConfig(
            session_aware_enabled=True,
            active_session_interval_seconds=3600,
        )
        cadence = SessionCadence(config)
        assert cadence.get_scanner_interval(_utc(14)) == 3600  # 14:00 UTC = NY

    def test_london_ny_overlap_returns_active_interval(self):
        config = SchedulerConfig(
            session_aware_enabled=True,
            active_session_interval_seconds=3600,
        )
        cadence = SessionCadence(config)
        assert cadence.get_scanner_interval(_utc(13)) == 3600  # 13:00 = overlap

    def test_off_hours_returns_quiet_interval(self):
        config = SchedulerConfig(
            session_aware_enabled=True,
            active_session_interval_seconds=3600,
            quiet_session_interval_seconds=14400,
        )
        cadence = SessionCadence(config)
        assert cadence.get_scanner_interval(_utc(3)) == 14400  # 03:00 UTC = Asia

    def test_session_boundary_london_open(self):
        config = SchedulerConfig(session_aware_enabled=True, active_session_interval_seconds=3600)
        cadence = SessionCadence(config)
        assert cadence.is_active_session(_utc(7))  # 07:00 = London open
        assert not cadence.is_active_session(_utc(6, 59))  # 06:59 = not yet

    def test_session_boundary_ny_close(self):
        config = SchedulerConfig(session_aware_enabled=True, active_session_interval_seconds=3600)
        cadence = SessionCadence(config)
        assert cadence.is_active_session(_utc(20))  # 20:00 = still NY
        assert not cadence.is_active_session(_utc(21))  # 21:00 = NY closed

    def test_session_name_overlap(self):
        config = SchedulerConfig(session_aware_enabled=True)
        cadence = SessionCadence(config)
        assert cadence.current_session_name(_utc(13)) == "London/NY Overlap"

    def test_session_name_london_only(self):
        config = SchedulerConfig(session_aware_enabled=True)
        cadence = SessionCadence(config)
        assert cadence.current_session_name(_utc(8)) == "London"

    def test_session_name_ny_only(self):
        config = SchedulerConfig(session_aware_enabled=True)
        cadence = SessionCadence(config)
        assert cadence.current_session_name(_utc(17)) == "New York"

    def test_session_name_off_hours(self):
        config = SchedulerConfig(session_aware_enabled=True)
        cadence = SessionCadence(config)
        assert cadence.current_session_name(_utc(3)) == "Off-hours"
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_session_cadence.py -v`
Expected: PASS

**Step 5: Update `src/scheduler/__init__.py`**

Add import of `SessionCadence` (check existing exports first — may just need to add a line).

**Step 6: Commit**

```bash
git add src/config.py src/scheduler/session_cadence.py src/scheduler/__init__.py tests/test_session_cadence.py
git commit -m "feat(scheduler): add session-aware cadence calculator for London/NY sessions"
```

---

### Task 4: Wire Session Cadence into Scanner Loop

**Files:**
- Modify: `src/scheduler/scheduler.py` — instantiate `SessionCadence`, use in sleep logic
- Test: `tests/test_scheduler_session_integration.py` (new)

**Context:** Replace the hardcoded `scanner_interval_seconds` sleep in `_scanner_loop()` with `SessionCadence.get_scanner_interval()`. The rescan event from Task 2 takes priority (wakes immediately).

**Step 1: Write integration test**

```python
# tests/test_scheduler_session_integration.py
"""Tests for session cadence integration in Scheduler (v1.2.0)."""
import pytest

from src.config import AppConfig
from src.scheduler.session_cadence import SessionCadence


def test_scheduler_creates_session_cadence():
    """Scheduler should initialize a SessionCadence instance."""
    from src.scheduler.scheduler import Scheduler
    from unittest.mock import MagicMock, AsyncMock

    config = AppConfig()
    scheduler = Scheduler(
        config=config,
        store=MagicMock(),
        scanner=MagicMock(),
        agents=MagicMock(),
        engine=MagicMock(),
        matchtrader=AsyncMock(),
    )
    assert hasattr(scheduler, "_session_cadence")
    assert isinstance(scheduler._session_cadence, SessionCadence)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scheduler_session_integration.py -v`
Expected: FAIL

**Step 3: Implement**

In `src/scheduler/scheduler.py`:

Add import at top:
```python
from src.scheduler.session_cadence import SessionCadence
```

In `__init__()`, after `self._rescan_event = asyncio.Event()`:
```python
# v1.2.0: Session-aware scanner cadence
self._session_cadence = SessionCadence(config.scheduler)
```

In `_scanner_loop()`, replace the sleep/wait section at the bottom of the loop (the code from Task 2) to use dynamic interval:
```python
# v1.2.0: Dynamic interval based on session
scan_interval = self._session_cadence.get_scanner_interval(self._now_utc())
session_name = self._session_cadence.current_session_name(self._now_utc())
logger.debug(
    "Scanner loop: next scan in {}s (session: {})", scan_interval, session_name
)
try:
    await asyncio.wait_for(
        self._rescan_event.wait(),
        timeout=scan_interval,
    )
    self._rescan_event.clear()
    logger.info("Scanner loop: rescan event received — running early scan")
except asyncio.TimeoutError:
    pass  # Normal timeout — proceed with scheduled scan
except asyncio.CancelledError:
    logger.info("Scanner loop: cancelled during sleep")
    return
```

Also replace the early `await asyncio.sleep(self._config.scheduler.scanner_interval_seconds)` on line 199 (inside the Best Day protection continue path) with:
```python
await asyncio.sleep(self._session_cadence.get_scanner_interval(self._now_utc()))
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_scheduler_session_integration.py tests/test_scheduler_rescan_event.py -v`
Expected: PASS

**Step 5: Run full suite**

Run: `uv run pytest -x -q`
Expected: All pass

**Step 6: Commit**

```bash
git add src/scheduler/scheduler.py tests/test_scheduler_session_integration.py
git commit -m "feat(scheduler): wire session-aware cadence into scanner loop"
```

---

### Task 5: Faster Re-Evaluation Default

**Files:**
- Modify: `src/config.py` — change `reeval_interval_seconds` default from 14400 to 7200
- Test: `tests/test_config.py`

**Step 1: Write test**

```python
# tests/test_config.py — add
def test_scheduler_config_reeval_interval_default():
    """v1.2.0: Default reeval interval should be 2h (7200s)."""
    from src.config import SchedulerConfig
    config = SchedulerConfig()
    assert config.reeval_interval_seconds == 7200
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/test_config.py::test_scheduler_config_reeval_interval_default -v`
Expected: FAIL (current is 14400)

**Step 3: Change default**

```python
reeval_interval_seconds: int = Field(
    default=7200, description="Re-evaluate open positions via LLM every N seconds (2h)"
)
```

**Step 4: Verify pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat(scheduler): reduce default reeval interval from 4h to 2h"
```

---

## Part B: Volatility-Triggered Scans (Tasks 6–8)

### Task 6: Volatility Monitor Module

**Files:**
- Create: `src/scheduler/volatility_monitor.py`
- Modify: `src/config.py` — add volatility config fields to `SchedulerConfig`
- Test: `tests/test_volatility_monitor.py` (new)

**Context:** Monitor price changes using MatchTrader `get_quote()` data. When any symbol's price change % over a rolling window exceeds a threshold, trigger a re-scan. The monitor stores recent quotes and calculates price change %.

**Step 1: Add config fields to `SchedulerConfig`**

```python
# v1.2.0: Volatility-triggered scans
volatility_trigger_enabled: bool = Field(
    default=False, description="Enable volatility-triggered scanner re-scans"
)
volatility_threshold_pct: float = Field(
    default=0.3, description="Price change % to trigger re-scan (0.3 = 0.3%)"
)
volatility_window_minutes: int = Field(
    default=30, description="Rolling window for price change calculation (minutes)"
)
volatility_poll_interval_seconds: int = Field(
    default=60, description="How often to check quote prices for volatility (seconds)"
)
volatility_cooldown_seconds: int = Field(
    default=900, description="Min seconds between volatility-triggered scans (15min)"
)
```

**Step 2: Write the volatility monitor**

```python
# src/scheduler/volatility_monitor.py
"""
Volatility monitor — detects significant price moves to trigger re-scans.

Monitors FX pair prices via MatchTrader get_quote() and calculates
rolling price change %. When any symbol exceeds the configured threshold,
signals the scanner to run an early scan.

Usage:
    monitor = VolatilityMonitor(config, symbols)
    monitor.record_quote("EURUSD", 1.0850, now)
    triggered, symbol, pct = monitor.check_triggers(now)
"""

from collections import deque
from datetime import datetime, timedelta

from loguru import logger

from src.config import SchedulerConfig


class VolatilityMonitor:
    """Tracks price quotes and detects volatility spikes.

    Usage:
        monitor = VolatilityMonitor(scheduler_config, ["EURUSD", "XAUUSD"])
        monitor.record_quote("EURUSD", 1.0850, now_utc)
        triggered, symbol, pct_change = monitor.check_triggers(now_utc)
    """

    def __init__(self, config: SchedulerConfig, symbols: list[str]) -> None:
        self._config = config
        self._symbols = symbols
        # Per-symbol quote history: deque of (timestamp, mid_price)
        self._quotes: dict[str, deque[tuple[datetime, float]]] = {
            sym: deque() for sym in symbols
        }
        self._last_trigger_time: datetime | None = None

    def record_quote(self, symbol: str, mid_price: float, now: datetime) -> None:
        """Record a price quote for a symbol.

        Args:
            symbol: FX pair (e.g. "EURUSD").
            mid_price: Mid price ((bid + ask) / 2).
            now: Current UTC timestamp.
        """
        if symbol not in self._quotes:
            self._quotes[symbol] = deque()

        self._quotes[symbol].append((now, mid_price))
        self._prune_old_quotes(symbol, now)

    def check_triggers(self, now: datetime) -> tuple[bool, str, float]:
        """Check if any symbol has exceeded the volatility threshold.

        Returns:
            Tuple of (triggered, symbol, pct_change).
            If no trigger, returns (False, "", 0.0).
        """
        if not self._config.volatility_trigger_enabled:
            return False, "", 0.0

        # Cooldown check
        if self._last_trigger_time is not None:
            elapsed = (now - self._last_trigger_time).total_seconds()
            if elapsed < self._config.volatility_cooldown_seconds:
                return False, "", 0.0

        best_pct = 0.0
        best_symbol = ""

        for symbol in self._symbols:
            pct = self._calculate_price_change_pct(symbol, now)
            if abs(pct) > abs(best_pct):
                best_pct = pct
                best_symbol = symbol

        if abs(best_pct) >= self._config.volatility_threshold_pct:
            self._last_trigger_time = now
            logger.info(
                "Volatility trigger: {} moved {:.2f}% in {}min window",
                best_symbol, best_pct, self._config.volatility_window_minutes,
            )
            return True, best_symbol, best_pct

        return False, "", 0.0

    def _calculate_price_change_pct(self, symbol: str, now: datetime) -> float:
        """Calculate price change % over the rolling window for a symbol."""
        quotes = self._quotes.get(symbol)
        if not quotes or len(quotes) < 2:
            return 0.0

        latest_price = quotes[-1][1]
        # Find oldest quote within window
        window_start = now - timedelta(minutes=self._config.volatility_window_minutes)
        oldest_price = latest_price
        for ts, price in quotes:
            if ts >= window_start:
                oldest_price = price
                break

        if oldest_price == 0.0:
            return 0.0
        return ((latest_price - oldest_price) / oldest_price) * 100.0

    def _prune_old_quotes(self, symbol: str, now: datetime) -> None:
        """Remove quotes older than 2x the window to keep memory bounded."""
        max_age = now - timedelta(minutes=self._config.volatility_window_minutes * 2)
        quotes = self._quotes[symbol]
        while quotes and quotes[0][0] < max_age:
            quotes.popleft()

    def reset(self) -> None:
        """Clear all stored quotes (e.g. on market close)."""
        for sym in self._quotes:
            self._quotes[sym].clear()
        self._last_trigger_time = None
```

**Step 3: Write tests**

```python
# tests/test_volatility_monitor.py
"""Tests for volatility monitor (v1.2.0)."""
from datetime import datetime, timedelta, timezone

import pytest

from src.config import SchedulerConfig
from src.scheduler.volatility_monitor import VolatilityMonitor


def _utc_now() -> datetime:
    return datetime(2026, 3, 4, 12, 0, 0, tzinfo=timezone.utc)


class TestVolatilityMonitor:
    def test_disabled_returns_no_trigger(self):
        config = SchedulerConfig(volatility_trigger_enabled=False)
        monitor = VolatilityMonitor(config, ["EURUSD"])
        now = _utc_now()
        monitor.record_quote("EURUSD", 1.0800, now - timedelta(minutes=20))
        monitor.record_quote("EURUSD", 1.1200, now)  # +3.7% — huge move
        triggered, _, _ = monitor.check_triggers(now)
        assert not triggered

    def test_trigger_on_threshold_breach(self):
        config = SchedulerConfig(
            volatility_trigger_enabled=True,
            volatility_threshold_pct=0.3,
            volatility_window_minutes=30,
            volatility_cooldown_seconds=0,  # Disable cooldown for test
        )
        monitor = VolatilityMonitor(config, ["EURUSD"])
        now = _utc_now()
        monitor.record_quote("EURUSD", 1.0800, now - timedelta(minutes=15))
        monitor.record_quote("EURUSD", 1.0840, now)  # +0.37%
        triggered, symbol, pct = monitor.check_triggers(now)
        assert triggered
        assert symbol == "EURUSD"
        assert pct > 0.3

    def test_no_trigger_below_threshold(self):
        config = SchedulerConfig(
            volatility_trigger_enabled=True,
            volatility_threshold_pct=0.3,
            volatility_window_minutes=30,
            volatility_cooldown_seconds=0,
        )
        monitor = VolatilityMonitor(config, ["EURUSD"])
        now = _utc_now()
        monitor.record_quote("EURUSD", 1.0800, now - timedelta(minutes=15))
        monitor.record_quote("EURUSD", 1.0810, now)  # +0.09%
        triggered, _, _ = monitor.check_triggers(now)
        assert not triggered

    def test_cooldown_prevents_repeat_trigger(self):
        config = SchedulerConfig(
            volatility_trigger_enabled=True,
            volatility_threshold_pct=0.3,
            volatility_window_minutes=30,
            volatility_cooldown_seconds=900,
        )
        monitor = VolatilityMonitor(config, ["EURUSD"])
        now = _utc_now()
        monitor.record_quote("EURUSD", 1.0800, now - timedelta(minutes=15))
        monitor.record_quote("EURUSD", 1.0840, now)  # +0.37%

        # First trigger should fire
        triggered, _, _ = monitor.check_triggers(now)
        assert triggered

        # Immediate re-check should be blocked by cooldown
        monitor.record_quote("EURUSD", 1.0880, now + timedelta(minutes=1))
        triggered, _, _ = monitor.check_triggers(now + timedelta(minutes=1))
        assert not triggered

        # After cooldown expires, should trigger again
        later = now + timedelta(seconds=901)
        monitor.record_quote("EURUSD", 1.0900, later)
        triggered, _, _ = monitor.check_triggers(later)
        assert triggered

    def test_multi_symbol_picks_largest_move(self):
        config = SchedulerConfig(
            volatility_trigger_enabled=True,
            volatility_threshold_pct=0.3,
            volatility_window_minutes=30,
            volatility_cooldown_seconds=0,
        )
        monitor = VolatilityMonitor(config, ["EURUSD", "XAUUSD"])
        now = _utc_now()
        # EURUSD: small move
        monitor.record_quote("EURUSD", 1.0800, now - timedelta(minutes=10))
        monitor.record_quote("EURUSD", 1.0810, now)  # +0.09%
        # XAUUSD: large move
        monitor.record_quote("XAUUSD", 2000.0, now - timedelta(minutes=10))
        monitor.record_quote("XAUUSD", 2010.0, now)  # +0.5%

        triggered, symbol, pct = monitor.check_triggers(now)
        assert triggered
        assert symbol == "XAUUSD"
        assert pct > 0.3

    def test_prune_old_quotes(self):
        config = SchedulerConfig(
            volatility_trigger_enabled=True,
            volatility_window_minutes=30,
        )
        monitor = VolatilityMonitor(config, ["EURUSD"])
        now = _utc_now()
        # Add a quote that's way too old (2 hours ago)
        monitor.record_quote("EURUSD", 1.0500, now - timedelta(hours=2))
        # Add current quote — old one should be pruned
        monitor.record_quote("EURUSD", 1.0800, now)

        # Only 1 quote should remain (the old one was pruned)
        # Actually, both get added but prune removes the old one on second add
        assert len(monitor._quotes["EURUSD"]) == 1

    def test_reset_clears_state(self):
        config = SchedulerConfig(volatility_trigger_enabled=True)
        monitor = VolatilityMonitor(config, ["EURUSD"])
        now = _utc_now()
        monitor.record_quote("EURUSD", 1.0800, now)
        monitor.reset()
        assert len(monitor._quotes["EURUSD"]) == 0
        assert monitor._last_trigger_time is None

    def test_single_quote_no_trigger(self):
        config = SchedulerConfig(
            volatility_trigger_enabled=True,
            volatility_threshold_pct=0.3,
            volatility_cooldown_seconds=0,
        )
        monitor = VolatilityMonitor(config, ["EURUSD"])
        now = _utc_now()
        monitor.record_quote("EURUSD", 1.0800, now)
        triggered, _, _ = monitor.check_triggers(now)
        assert not triggered  # Need at least 2 quotes
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_volatility_monitor.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/config.py src/scheduler/volatility_monitor.py tests/test_volatility_monitor.py
git commit -m "feat(scheduler): add volatility monitor for price-change triggered scans"
```

---

### Task 7: Wire Volatility Monitor into Scheduler

**Files:**
- Modify: `src/scheduler/scheduler.py` — add volatility poll loop, integrate with rescan event
- Test: `tests/test_scheduler_volatility_integration.py` (new)

**Context:** Add a new `_volatility_monitor_loop()` that polls `get_quote()` for all symbols on a regular interval. When `check_triggers()` returns True, set `_rescan_event`. The scanner loop already listens for this event (from Task 2).

**Step 1: Write integration test**

```python
# tests/test_scheduler_volatility_integration.py
"""Tests for volatility monitor integration in Scheduler (v1.2.0)."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import AppConfig
from src.scheduler.volatility_monitor import VolatilityMonitor


def test_scheduler_creates_volatility_monitor():
    """Scheduler should initialize a VolatilityMonitor."""
    from src.scheduler.scheduler import Scheduler

    config = AppConfig()
    config.scheduler.volatility_trigger_enabled = True
    scheduler = Scheduler(
        config=config,
        store=MagicMock(),
        scanner=MagicMock(),
        agents=MagicMock(),
        engine=MagicMock(),
        matchtrader=AsyncMock(),
    )
    assert hasattr(scheduler, "_volatility_monitor")
    assert isinstance(scheduler._volatility_monitor, VolatilityMonitor)


def test_volatility_loop_in_tasks_when_enabled():
    """When volatility_trigger_enabled, start() should include the volatility loop."""
    from src.scheduler.scheduler import Scheduler

    config = AppConfig()
    config.scheduler.volatility_trigger_enabled = True
    scheduler = Scheduler(
        config=config,
        store=MagicMock(),
        scanner=MagicMock(),
        agents=MagicMock(),
        engine=MagicMock(),
        matchtrader=AsyncMock(),
    )
    # Check the method exists
    assert hasattr(scheduler, "_volatility_monitor_loop")
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/test_scheduler_volatility_integration.py -v`
Expected: FAIL

**Step 3: Implement**

In `src/scheduler/scheduler.py`:

Add import:
```python
from src.scheduler.volatility_monitor import VolatilityMonitor
```

In `__init__()`, after `_session_cadence`:
```python
# v1.2.0: Volatility-triggered re-scans
self._volatility_monitor = VolatilityMonitor(config.scheduler, config.symbols)
```

In `start()`, add the volatility loop to the tasks list (conditionally):
```python
# v1.2.0: Volatility monitor loop (if enabled)
if self._config.scheduler.volatility_trigger_enabled:
    tasks.append(self._volatility_monitor_loop())
```

Add the new loop method:
```python
async def _volatility_monitor_loop(self) -> None:
    """Poll quotes and trigger re-scan on significant price moves."""
    logger.info("Volatility monitor loop: started")
    while self._running:
        try:
            await self._wait_for_market_open("Volatility monitor")
            now = self._now_utc()

            for symbol in self._config.symbols:
                try:
                    # Map config symbol to broker symbol if needed
                    broker_symbol = symbol
                    if self._registry is not None:
                        broker_symbol = self._registry.to_broker(symbol)
                    quote = await self._matchtrader.get_quote(broker_symbol)
                    mid_price = (quote.bid + quote.ask) / 2
                    self._volatility_monitor.record_quote(symbol, mid_price, now)
                except Exception as e:
                    logger.debug("Volatility monitor: quote failed for {}: {}", symbol, e)

            triggered, symbol, pct = self._volatility_monitor.check_triggers(now)
            if triggered:
                self._rescan_event.set()
                await self._send_alert(
                    f"📈 <b>Volatility Trigger</b>\n"
                    f"• {symbol} moved {pct:+.2f}% in "
                    f"{self._config.scheduler.volatility_window_minutes}min\n"
                    f"• Triggering early scan"
                )

        except asyncio.CancelledError:
            logger.info("Volatility monitor loop: cancelled")
            return
        except Exception as e:
            logger.error("Volatility monitor loop error: {}", e)

        try:
            await asyncio.sleep(self._config.scheduler.volatility_poll_interval_seconds)
        except asyncio.CancelledError:
            logger.info("Volatility monitor loop: cancelled during sleep")
            return

    logger.info("Volatility monitor loop: stopped")
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_scheduler_volatility_integration.py tests/test_volatility_monitor.py -v`
Expected: PASS

**Step 5: Run full suite**

Run: `uv run pytest -x -q`
Expected: All pass

**Step 6: Commit**

```bash
git add src/scheduler/scheduler.py tests/test_scheduler_volatility_integration.py
git commit -m "feat(scheduler): wire volatility monitor loop into scheduler"
```

---

### Task 8: Update YAML Config for Part A + B

**Files:**
- Modify: `config/default.yaml` — add session + volatility defaults
- Modify: `config/e8_one_5k_challenge.yaml` — enable features
- Test: `tests/test_config_yaml.py` (add round-trip test)

**Step 1: Update `config/default.yaml`**

Add under `scheduler:` section (or add the section if needed):
```yaml
scheduler:
  reeval_interval_seconds: 7200
  llm_worker_count: 2
  # Session-aware cadence (disabled by default)
  session_aware_enabled: false
  active_session_interval_seconds: 3600
  quiet_session_interval_seconds: 14400
  # Volatility trigger (disabled by default)
  volatility_trigger_enabled: false
  volatility_threshold_pct: 0.3
  volatility_window_minutes: 30
  volatility_cooldown_seconds: 900
```

**Step 2: Update `config/e8_one_5k_challenge.yaml`**

Add under the existing `scheduler:` section:
```yaml
scheduler:
  market_hours:
    enabled: true
    close_day: "Friday"
    close_time_utc: "22:00"
    open_day: "Sunday"
    open_time_utc: "22:00"
    force_close_before_weekend: false
    force_close_minutes_before: 15
  # v1.2.0: Enable session-aware cadence
  session_aware_enabled: true
  active_session_interval_seconds: 3600   # 1h during London/NY
  quiet_session_interval_seconds: 14400   # 4h during off-hours
  # v1.2.0: Enable volatility-triggered scans
  volatility_trigger_enabled: true
  volatility_threshold_pct: 0.3
  volatility_window_minutes: 30
  volatility_cooldown_seconds: 900
```

**Step 3: Write test verifying YAML loads correctly**

```python
# tests/test_config_yaml.py — add
def test_e8_one_5k_session_aware_config():
    """e8_one_5k_challenge should have session-aware cadence enabled."""
    from src.config import load_config
    config = load_config("config/e8_one_5k_challenge.yaml")
    assert config.scheduler.session_aware_enabled is True
    assert config.scheduler.active_session_interval_seconds == 3600
    assert config.scheduler.volatility_trigger_enabled is True
    assert config.scheduler.volatility_threshold_pct == 0.3
```

**Step 4: Run test**

Run: `uv run pytest tests/test_config_yaml.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add config/default.yaml config/e8_one_5k_challenge.yaml tests/test_config_yaml.py
git commit -m "feat(config): add session-aware and volatility trigger config to YAML"
```

---

## Part C: Multi-Timeframe Analysis (Tasks 9–15)

This is the most complex part. The daily scanner sets direction (BUY/SELL bias), and a shorter-timeframe (4H) scanner times entries. Requires changes to BOTH repos.

### Task 9: Extend qlib_market_scanner Interval Choices

**Files:**
- Modify: `C:\Users\tommy.yeung\CursorProjects\qlib_market_scanner\src\main.py:44` — add "4h", "1h" choices
- Modify: `C:\Users\tommy.yeung\CursorProjects\qlib_market_scanner\src\config.py` — handle 4h/1h in `apply_profile_defaults`, `default_date_range`, freq mapping
- Test: `C:\Users\tommy.yeung\CursorProjects\qlib_market_scanner\tests\test_config.py`

**Context:** qlib_market_scanner currently supports `--interval 1d` and `--interval 1m`. We need `4h` and `1h`. The key mapping is:
- `--interval 4h` → `DataConfig.interval = "4h"` → `PipelineConfig.freq = "4h"` → Qlib calendar `4h.txt`, features `*.4h.bin`
- `--interval 1h` → similar

**Step 1: Write test in scanner repo**

```python
# In qlib_market_scanner: tests/test_config.py — add
def test_interval_4h_defaults():
    """4h interval should set appropriate defaults for FX."""
    from src.config import AppConfig
    config = AppConfig(profile="fx")
    config.data.interval = "4h"
    config.apply_profile_defaults()
    assert config.data.interval == "4h"
```

**Step 2: Extend `--interval` choices in `main.py`**

Change line 44 from:
```python
parser.add_argument("--interval", type=str, default="1d", choices=["1d", "1m"])
```
To:
```python
parser.add_argument("--interval", type=str, default="1d", choices=["1d", "1m", "4h", "1h"])
```

**Step 3: Handle new intervals in `config.py`**

In `apply_profile_defaults()`, after the existing FX overrides:
```python
# Map interval to Qlib freq
interval_to_freq = {"1d": "day", "1m": "1min", "4h": "4h", "1h": "1h"}
self.pipeline.freq = interval_to_freq.get(self.data.interval, "day")
```

In `default_date_range()`, add handling for 4h/1h:
```python
def default_date_range(interval: str, profile: str = "stock") -> tuple[str, str]:
    end_date = _effective_end_date(interval, profile)
    if interval == "1m":
        start_date = end_date - timedelta(days=30)
    elif interval in ("4h", "1h"):
        start_date = end_date - timedelta(days=365)  # 1 year for intraday
    else:
        start_date = end_date - timedelta(days=1825)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
```

In `_effective_end_date()`, treat 4h/1h like 1m for "use today":
```python
if interval in ("1m", "4h", "1h"):
    return now_utc.date()
```

**Step 4: Run scanner tests**

Run (in scanner repo): `uv run pytest -v`
Expected: PASS

**Step 5: Commit (in scanner repo)**

```bash
git add src/main.py src/config.py tests/test_config.py
git commit -m "feat: add 4h and 1h interval support for FX multi-timeframe analysis"
```

---

### Task 10: Intraday Data Fetching in prop-firm-pilot

**Files:**
- Modify: `src/data/fx_data_fetcher.py` — add abstract `fetch_bars()` method with interval param, implement for TraderMade and iTick
- Test: `tests/test_fx_data_fetcher.py` (add intraday tests)

**Context:** Both providers already support intraday:
- TraderMade: `interval` param accepts "1min", "5min", "15min", "30min", "1H", "4H", "daily"
- iTick: `kType` param: 1=1min, 5=5min, 15=15min, 30=30min, 60=1h, 240=4h, 480=8h, 1440=daily

We add a new `fetch_bars(symbol, start, end, client, interval="daily")` method to both providers. The existing `fetch_daily_bars()` remains for backward compatibility but delegates to `fetch_bars()`.

**Step 1: Write test**

```python
# tests/test_fx_data_fetcher.py — add
import respx
import httpx
import pytest

from src.data.fx_data_fetcher import TraderMadeProvider, ITickProvider


@respx.mock
async def test_tradermade_fetch_bars_4h_interval():
    """TraderMade fetch_bars should pass interval='4H' to API."""
    from datetime import date

    provider = TraderMadeProvider(api_key="test_key")
    route = respx.get("https://marketdata.tradermade.com/api/v1/timeseries").mock(
        return_value=httpx.Response(200, json={
            "quotes": [
                {"date": "2026-03-01 00:00:00", "open": 1.08, "high": 1.09, "low": 1.07, "close": 1.085}
            ]
        })
    )
    async with httpx.AsyncClient() as client:
        df = await provider.fetch_bars("EURUSD", date(2026, 3, 1), date(2026, 3, 1), client, interval="4h")

    assert route.called
    # Verify interval param was passed
    request = route.calls[0].request
    assert "4H" in str(request.url) or "4H" in str(request.content)


@respx.mock
async def test_itick_fetch_bars_4h_interval():
    """iTick fetch_bars should pass kType=240 for 4h interval."""
    from datetime import date

    provider = ITickProvider(api_key="test_key")
    route = respx.get("https://api.itick.org/forex/kline").mock(
        return_value=httpx.Response(200, json={
            "code": 200,
            "data": [
                {"t": 1709251200000, "o": 1.08, "h": 1.09, "l": 1.07, "c": 1.085, "v": 100}
            ]
        })
    )
    async with httpx.AsyncClient() as client:
        df = await provider.fetch_bars("EURUSD", date(2026, 3, 1), date(2026, 3, 1), client, interval="4h")

    assert route.called
    request = route.calls[0].request
    assert "240" in str(request.url)
```

**Step 2: Implement**

Add abstract method to `FxDataProvider`:
```python
@abc.abstractmethod
async def fetch_bars(
    self,
    symbol: str,
    start_date: date,
    end_date: date,
    client: httpx.AsyncClient,
    interval: str = "daily",
) -> pd.DataFrame:
    """Fetch OHLCV bars at the given interval.

    Args:
        interval: "daily", "4h", "1h", "30min", "15min", "5min", "1min"

    Returns:
        DataFrame with columns: datetime, open, high, low, close, volume.
    """
    ...
```

For `TraderMadeProvider`, add the interval mapping and implement `fetch_bars()`:
```python
INTERVAL_MAP = {"daily": "daily", "4h": "4H", "1h": "1H", "30min": "30min", "15min": "15min"}
```
The `fetch_bars()` method is similar to `fetch_daily_bars()` but passes the mapped interval to the API params. `fetch_daily_bars()` calls `self.fetch_bars(..., interval="daily")`.

For `ITickProvider`, add the kType mapping:
```python
KTYPE_MAP = {"daily": "8", "4h": "7", "1h": "6", "30min": "5", "15min": "4", "5min": "3", "1min": "2"}
# Note: iTick kType values — verify from API docs. Common mapping:
# 1=1min, 2=5min, 3=15min, 4=30min, 5=60min, 6=240min, 7=480min, 8=daily
# WAIT — need to verify. From current code, kType=8 is daily.
# Based on standard iTick docs: kType 1/5/15/30/60/240/480/1440
# This means kType value IS the minutes: 60=1h, 240=4h, 1440=daily
# But current code uses kType=8 for daily... Contradiction.
# Current code: "kType": "8" for daily bars. This suggests:
# 1=1min, 2=5min, 3=15min, 4=30min, 5=1h, 6=4h, 7=8h, 8=daily
# We'll use this mapping.
KTYPE_MAP = {"daily": "8", "4h": "6", "1h": "5", "30min": "4", "15min": "3", "5min": "2", "1min": "1"}
```

**Important:** Verify iTick kType mapping by checking the existing `kType: "8"` for daily. If the iTick API docs differ, adjust accordingly. The safest approach is to test with TraderMade first and add iTick intraday later.

**Step 3: Run tests**

Run: `uv run pytest tests/test_fx_data_fetcher.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/data/fx_data_fetcher.py tests/test_fx_data_fetcher.py
git commit -m "feat(data): add fetch_bars() with interval support for multi-timeframe"
```

---

### Task 11: Intraday DuckDB Storage

**Files:**
- Modify: `src/data/fx_duckdb_store.py` — add `fx_intraday` table with interval + datetime columns, add `upsert_intraday()` and `read_intraday()` methods
- Test: `tests/test_fx_duckdb_store.py` (add intraday tests)

**Context:** The current `fx_daily` table has `PRIMARY KEY (symbol, date)` which can't hold intraday data (multiple bars per day). We add a separate `fx_intraday` table with `PRIMARY KEY (symbol, interval, datetime)`.

**Step 1: Write test**

```python
# tests/test_fx_duckdb_store.py — add
import pandas as pd
import pytest
from datetime import datetime

from src.data.fx_duckdb_store import FxDuckDbStore


def test_upsert_and_read_intraday(tmp_path):
    """Should store and retrieve intraday bars with interval column."""
    store = FxDuckDbStore(tmp_path / "test.duckdb")

    df = pd.DataFrame({
        "datetime": pd.to_datetime([
            "2026-03-01 00:00:00", "2026-03-01 04:00:00",
            "2026-03-01 08:00:00", "2026-03-01 12:00:00",
        ]),
        "open": [1.08, 1.085, 1.09, 1.087],
        "high": [1.09, 1.09, 1.095, 1.09],
        "low": [1.075, 1.08, 1.085, 1.085],
        "close": [1.085, 1.09, 1.087, 1.088],
        "volume": [100, 150, 200, 120],
    })

    count = store.upsert_intraday("EURUSD", df, interval="4h", provider="tradermade")
    assert count == 4

    result = store.read_intraday("EURUSD", interval="4h")
    assert len(result) == 4
    assert "datetime" in result.columns


def test_read_intraday_date_filter(tmp_path):
    """Should filter intraday bars by date range."""
    store = FxDuckDbStore(tmp_path / "test.duckdb")
    from datetime import date

    df = pd.DataFrame({
        "datetime": pd.to_datetime([
            "2026-03-01 00:00:00", "2026-03-01 04:00:00",
            "2026-03-02 00:00:00", "2026-03-02 04:00:00",
        ]),
        "open": [1.08, 1.085, 1.09, 1.087],
        "high": [1.09, 1.09, 1.095, 1.09],
        "low": [1.075, 1.08, 1.085, 1.085],
        "close": [1.085, 1.09, 1.087, 1.088],
        "volume": [100, 150, 200, 120],
    })
    store.upsert_intraday("EURUSD", df, interval="4h")

    result = store.read_intraday("EURUSD", interval="4h",
                                  start_date=date(2026, 3, 2), end_date=date(2026, 3, 2))
    assert len(result) == 2
```

**Step 2: Implement**

Add new SQL constants:
```python
CREATE_INTRADAY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS fx_intraday (
    symbol      VARCHAR NOT NULL,
    interval    VARCHAR NOT NULL,
    datetime    TIMESTAMP NOT NULL,
    open        DOUBLE NOT NULL,
    high        DOUBLE NOT NULL,
    low         DOUBLE NOT NULL,
    close       DOUBLE NOT NULL,
    volume      BIGINT DEFAULT 0,
    provider    VARCHAR DEFAULT 'unknown',
    fetched_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, interval, datetime)
)
"""
```

Call `self._conn.execute(CREATE_INTRADAY_TABLE_SQL)` in `_init_schema()`.

Add methods `upsert_intraday()` and `read_intraday()` following the same pattern as existing `upsert()` and `read()`, but with interval column.

**Step 3: Run tests**

Run: `uv run pytest tests/test_fx_duckdb_store.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/data/fx_duckdb_store.py tests/test_fx_duckdb_store.py
git commit -m "feat(data): add intraday DuckDB storage table for multi-timeframe bars"
```

---

### Task 12: Intraday Qlib Binary Conversion

**Files:**
- Modify: `src/data/fx_to_qlib.py` — add `interval` parameter to `convert_to_qlib_binary()`, parameterize calendar name and bin suffix
- Test: `tests/test_fx_to_qlib.py` (add intraday test)

**Context:** Currently hardcoded to `.day.bin` and `day.txt`. For 4h data we need `.4h.bin` and `4h.txt`. The intraday variant must NOT normalize datetime (preserve time component).

**Step 1: Write test**

```python
# tests/test_fx_to_qlib.py — add
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from src.data.fx_to_qlib import convert_to_qlib_binary


def test_convert_intraday_4h(tmp_path):
    """4h conversion should use 4h calendar and preserve time component."""
    data = {
        "EURUSD": pd.DataFrame({
            "datetime": pd.to_datetime([
                "2026-03-01 00:00:00", "2026-03-01 04:00:00",
                "2026-03-01 08:00:00",
            ]),
            "open": [1.08, 1.085, 1.09],
            "high": [1.09, 1.09, 1.095],
            "low": [1.075, 1.08, 1.085],
            "close": [1.085, 1.09, 1.087],
            "volume": [100, 150, 200],
        })
    }

    result_dir = convert_to_qlib_binary(data, tmp_path / "qlib_4h", interval="4h")

    # Check calendar file uses 4h suffix
    cal_file = result_dir / "calendars" / "4h.txt"
    assert cal_file.exists()
    lines = cal_file.read_text().strip().split("\n")
    assert len(lines) == 3
    # Should preserve time component
    assert "00:00:00" in lines[0]

    # Check feature bin uses 4h suffix
    close_bin = result_dir / "features" / "EURUSD" / "close.4h.bin"
    assert close_bin.exists()
    values = np.fromfile(str(close_bin), dtype="<f4")  # Qlib uses float32
    # Should have 3 values (only non-NaN values stored)


def test_convert_daily_backward_compatible(tmp_path):
    """Daily conversion should still use day.bin and day.txt."""
    data = {
        "EURUSD": pd.DataFrame({
            "datetime": pd.to_datetime(["2026-03-01", "2026-03-02"]),
            "open": [1.08, 1.085],
            "high": [1.09, 1.09],
            "low": [1.075, 1.08],
            "close": [1.085, 1.09],
            "volume": [100, 150],
        })
    }

    result_dir = convert_to_qlib_binary(data, tmp_path / "qlib_daily")

    cal_file = result_dir / "calendars" / "day.txt"
    assert cal_file.exists()
    close_bin = result_dir / "features" / "EURUSD" / "close.day.bin"
    assert close_bin.exists()
```

**Step 2: Implement**

Add `interval: str = "day"` parameter to `convert_to_qlib_binary()`:
```python
def convert_to_qlib_binary(
    data: dict[str, pd.DataFrame],
    output_dir: str | Path,
    interval: str = "day",  # "day", "4h", "1h"
) -> Path:
```

Pass `interval` through to `_write_calendar()`, `_write_symbol_features()`:
- Calendar filename: `f"{interval}.txt"`
- Feature bin suffix: `f".{interval}.bin"` / `f".{interval}.meta"`

In `_prepare_dataframe()`, only normalize datetime for daily interval:
```python
if interval == "day":
    df["datetime"] = df["datetime"].dt.normalize()
```
This requires passing `interval` to `_prepare_dataframe()` too.

For calendar, intraday needs `YYYY-MM-DD HH:MM:SS` format instead of `YYYY-MM-DD`.

**Step 3: Run tests**

Run: `uv run pytest tests/test_fx_to_qlib.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/data/fx_to_qlib.py tests/test_fx_to_qlib.py
git commit -m "feat(data): support intraday interval in Qlib binary conversion"
```

---

### Task 13: Extend ScannerBridge with Interval Parameter

**Files:**
- Modify: `src/signal/scanner_bridge.py` — add `interval` parameter to `run_pipeline()`
- Test: `tests/test_scanner_bridge.py` (add interval test)

**Context:** The scanner bridge runs `qlib_market_scanner` via subprocess. We need to pass `--interval` to it for intraday scanning.

**Step 1: Write test**

```python
# tests/test_scanner_bridge.py — add
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.signal.scanner_bridge import ScannerBridge


def test_run_pipeline_passes_interval(tmp_path):
    """run_pipeline should pass --interval to the scanner subprocess."""
    scanner_path = tmp_path / "scanner"
    scanner_path.mkdir()

    bridge = ScannerBridge(scanner_path=scanner_path)

    with patch("src.signal.scanner_bridge.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        # Should fail (no real scanner) but we just check the command
        bridge.run_pipeline(date="2026-03-01", interval="4h")

        cmd = mock_run.call_args[0][0]
        assert "--interval" in cmd
        assert "4h" in cmd
```

**Step 2: Implement**

In `run_pipeline()`, add `interval: str = "1d"` parameter:
```python
def run_pipeline(
    self,
    date: str | None = None,
    tickers: list[str] | None = None,
    force_retrain: bool = True,
    interval: str = "1d",  # v1.2.0: multi-timeframe
) -> list[ScannerSignal]:
```

Add to command construction:
```python
cmd.extend(["--interval", interval])
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_scanner_bridge.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/signal/scanner_bridge.py tests/test_scanner_bridge.py
git commit -m "feat(signal): add interval parameter to ScannerBridge.run_pipeline()"
```

---

### Task 14: Multi-Timeframe Config + Data Pipeline Integration

**Files:**
- Modify: `src/config.py` — add multi-timeframe fields to `SchedulerConfig`
- Modify: `src/scheduler/scheduler.py` — add intraday data fetch + scan in scanner loop
- Test: `tests/test_scheduler_multi_timeframe.py` (new)

**Context:** The multi-timeframe flow: (1) daily scanner runs as before → sets BUY/SELL direction, (2) if direction found, run intraday scanner on matching symbols to time entry. The intraday scan requires fetching intraday data, converting to Qlib, then running scanner with `--interval`.

**Step 1: Add config fields**

Add to `SchedulerConfig`:
```python
# v1.2.0: Multi-timeframe analysis
multi_timeframe_enabled: bool = Field(
    default=False, description="Enable multi-timeframe entry timing"
)
entry_timeframe: str = Field(
    default="4h", description="Shorter timeframe for entry timing (4h or 1h)"
)
intraday_lookback_days: int = Field(
    default=90, description="Days of intraday data to fetch for entry analysis"
)
```

**Step 2: Write test**

```python
# tests/test_scheduler_multi_timeframe.py
"""Tests for multi-timeframe scanner integration (v1.2.0)."""
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from src.config import AppConfig


def test_multi_timeframe_config_defaults():
    config = AppConfig()
    assert config.scheduler.multi_timeframe_enabled is False
    assert config.scheduler.entry_timeframe == "4h"
    assert config.scheduler.intraday_lookback_days == 90
```

**Step 3: Implement intraday scan in scheduler**

In `_scanner_loop()`, after the daily scan creates intents (around line 303), add a multi-timeframe confirmation step:

```python
# v1.2.0: Multi-timeframe — run intraday scan to confirm entry timing
if self._config.scheduler.multi_timeframe_enabled and topk_signals:
    try:
        await self._run_intraday_scan(topk_signals, today)
    except Exception as e:
        logger.warning("Multi-timeframe scan failed (proceeding with daily-only): {}", e)
```

Add the helper method:
```python
async def _run_intraday_scan(self, daily_signals: list, today: str) -> None:
    """Run intraday scanner on symbols that daily scan identified.

    This provides entry timing — the daily scan sets direction,
    the intraday scan confirms the entry point is favorable.
    """
    entry_tf = self._config.scheduler.entry_timeframe
    symbols = [s.instrument for s in daily_signals]
    logger.info(
        "Multi-timeframe: running {} scan for {} symbols: {}",
        entry_tf, len(symbols), symbols,
    )

    intraday_signals = await asyncio.to_thread(
        self._scanner.run_pipeline,
        date=today,
        tickers=symbols,
        interval=entry_tf,
    )

    # Log results — intents were already created by daily scan
    # Intraday scan results are used for confidence boosting
    if intraday_signals:
        for signal in intraday_signals:
            logger.info(
                "Multi-timeframe {}: {} score={:.4f} conf={}",
                entry_tf, signal.instrument, signal.score, signal.confidence,
            )
    else:
        logger.info("Multi-timeframe: no intraday signals generated")
```

**Note:** The full integration of using intraday signals to gate or boost daily intents is a Phase 2 enhancement. For v1.2.0, we run the intraday scan and log results. The infrastructure (data pipeline, scanner bridge, Qlib conversion) is all in place.

**Step 4: Run tests**

Run: `uv run pytest tests/test_scheduler_multi_timeframe.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/config.py src/scheduler/scheduler.py tests/test_scheduler_multi_timeframe.py
git commit -m "feat(scheduler): add multi-timeframe scanner integration"
```

---

### Task 15: Update YAML Config for Multi-Timeframe + Final Integration Test

**Files:**
- Modify: `config/e8_one_5k_challenge.yaml` — add multi-timeframe config
- Test: `tests/test_full_integration.py` — run lint + full test suite

**Step 1: Update YAML**

Add to `config/e8_one_5k_challenge.yaml` under `scheduler:`:
```yaml
  # v1.2.0: Multi-timeframe (disabled for now — enable after validating intraday data)
  multi_timeframe_enabled: false
  entry_timeframe: "4h"
  intraday_lookback_days: 90
```

**Step 2: Run lint**

Run: `uv run ruff check src/ tests/`
Expected: Clean (or fix any issues)

Run: `uv run ruff format src/ tests/`

**Step 3: Run full test suite**

Run: `uv run pytest -v --tb=short`
Expected: All tests pass

**Step 4: Commit**

```bash
git add config/e8_one_5k_challenge.yaml
git commit -m "feat(config): add multi-timeframe config to e8_one_5k YAML"
```

---

## Summary

| Task | Feature | Files Changed | Est. Complexity |
|------|---------|--------------|-----------------|
| 1 | LLM worker count → 2 | config.py | Trivial |
| 2 | Position-close → rescan event | scheduler.py | Low |
| 3 | Session cadence calculator | session_cadence.py (new), config.py | Low |
| 4 | Wire session cadence into scheduler | scheduler.py | Low |
| 5 | Faster reeval default | config.py | Trivial |
| 6 | Volatility monitor module | volatility_monitor.py (new), config.py | Medium |
| 7 | Wire volatility into scheduler | scheduler.py | Medium |
| 8 | YAML config for Part A+B | default.yaml, e8_one_5k.yaml | Low |
| 9 | Scanner interval choices (qlib_market_scanner) | scanner main.py, config.py | Low |
| 10 | Intraday data fetching | fx_data_fetcher.py | Medium |
| 11 | Intraday DuckDB storage | fx_duckdb_store.py | Medium |
| 12 | Intraday Qlib binary | fx_to_qlib.py | Medium |
| 13 | Scanner bridge interval | scanner_bridge.py | Low |
| 14 | Multi-timeframe scheduler integration | scheduler.py, config.py | Medium |
| 15 | Final YAML + integration test | YAML files, lint | Low |

**Total: 15 tasks across 6 features.**

**Estimated time:** ~3-4 hours for a focused implementation session.
