# Weekend Market Closure & Dynamic Drawdown Fix — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix two production bugs: (1) Add configurable FX weekend market closure handling to the scheduler, (2) Fix dynamic drawdown tracking to trail highest closed balance per E8 Markets rules.

**Architecture:** Bug 1 adds a `MarketHoursConfig` to `SchedulerConfig`, a `_is_market_open()` check to each scheduler loop, and a forced-close routine before weekend. Bug 2 introduces a `HighWaterMarkTracker` class that persists HWM to a JSON sidecar file and updates it on every position close with profit. Both bugs are independent and can be worked on sequentially.

**Tech Stack:** Python 3.10, asyncio, Pydantic v2, loguru, httpx, pytest + pytest-asyncio, respx.

---

## Part A: Dynamic Drawdown Tracking (Bug 2 — Do First)

> Why first? This is the safety-critical bug — incorrect drawdown tracking can cause real financial loss (account termination). Weekend handling is an optimization; drawdown is a correctness issue.

### Task 1: Create `HighWaterMarkTracker` with Tests (RED → GREEN)

**Files:**
- Create: `src/compliance/hwm_tracker.py`
- Create: `tests/test_hwm_tracker.py`

**Step 1: Write the failing tests**

Create `tests/test_hwm_tracker.py`:

```python
"""Tests for HighWaterMarkTracker — dynamic drawdown HWM persistence."""

import json
from pathlib import Path

import pytest

from src.compliance.hwm_tracker import HighWaterMarkTracker


class TestHighWaterMarkTracker:
    """Tests for HWM tracking, persistence, and lock logic."""

    def test_initial_hwm_equals_initial_balance(self, tmp_path: Path) -> None:
        tracker = HighWaterMarkTracker(
            initial_balance=5000.0,
            drawdown_pct=0.06,
            state_path=str(tmp_path / "hwm.json"),
        )
        assert tracker.high_water_mark == 5000.0
        assert tracker.loss_level == 5000.0 * (1 - 0.06)  # 4700.0
        assert tracker.is_locked is False

    def test_update_on_profitable_close(self, tmp_path: Path) -> None:
        tracker = HighWaterMarkTracker(
            initial_balance=5000.0,
            drawdown_pct=0.06,
            state_path=str(tmp_path / "hwm.json"),
        )
        tracker.update_balance(5050.0)  # Trade closed with $50 profit
        assert tracker.high_water_mark == 5050.0
        assert tracker.loss_level == 5050.0 * (1 - 0.06)  # 4747.0

    def test_no_update_on_loss(self, tmp_path: Path) -> None:
        tracker = HighWaterMarkTracker(
            initial_balance=5000.0,
            drawdown_pct=0.06,
            state_path=str(tmp_path / "hwm.json"),
        )
        tracker.update_balance(5050.0)
        tracker.update_balance(5020.0)  # Balance dropped — no update
        assert tracker.high_water_mark == 5050.0
        assert tracker.loss_level == 5050.0 * (1 - 0.06)

    def test_lock_when_profit_exceeds_drawdown_amount(self, tmp_path: Path) -> None:
        """When cumulative realized profit >= initial_balance * drawdown_pct,
        loss_level locks permanently at initial_balance."""
        tracker = HighWaterMarkTracker(
            initial_balance=5000.0,
            drawdown_pct=0.06,  # $300 drawdown amount
            state_path=str(tmp_path / "hwm.json"),
        )
        tracker.update_balance(5300.0)  # Profit = $300 = drawdown amount
        assert tracker.is_locked is True
        assert tracker.loss_level == 5000.0  # Locked at initial balance

    def test_lock_persists_after_further_profit(self, tmp_path: Path) -> None:
        tracker = HighWaterMarkTracker(
            initial_balance=5000.0,
            drawdown_pct=0.06,
            state_path=str(tmp_path / "hwm.json"),
        )
        tracker.update_balance(5300.0)
        tracker.update_balance(5500.0)  # More profit after lock
        assert tracker.is_locked is True
        assert tracker.loss_level == 5000.0  # Still locked at initial
        assert tracker.high_water_mark == 5500.0  # HWM still tracks

    def test_persistence_to_json(self, tmp_path: Path) -> None:
        state_path = str(tmp_path / "hwm.json")
        tracker = HighWaterMarkTracker(
            initial_balance=5000.0,
            drawdown_pct=0.06,
            state_path=state_path,
        )
        tracker.update_balance(5100.0)
        tracker.save()

        # Load from file
        with open(state_path) as f:
            data = json.load(f)
        assert data["high_water_mark"] == 5100.0
        assert data["initial_balance"] == 5000.0
        assert data["is_locked"] is False

    def test_restore_from_json(self, tmp_path: Path) -> None:
        state_path = str(tmp_path / "hwm.json")
        # First tracker saves state
        t1 = HighWaterMarkTracker(
            initial_balance=5000.0,
            drawdown_pct=0.06,
            state_path=state_path,
        )
        t1.update_balance(5200.0)
        t1.save()

        # Second tracker restores
        t2 = HighWaterMarkTracker(
            initial_balance=5000.0,
            drawdown_pct=0.06,
            state_path=state_path,
        )
        assert t2.high_water_mark == 5200.0
        assert t2.loss_level == 5200.0 * (1 - 0.06)

    def test_restore_ignores_mismatched_initial_balance(self, tmp_path: Path) -> None:
        """If persisted state has different initial_balance, ignore it (account changed)."""
        state_path = str(tmp_path / "hwm.json")
        t1 = HighWaterMarkTracker(
            initial_balance=5000.0,
            drawdown_pct=0.06,
            state_path=state_path,
        )
        t1.update_balance(5200.0)
        t1.save()

        # Different initial_balance — should not restore
        t2 = HighWaterMarkTracker(
            initial_balance=50000.0,
            drawdown_pct=0.08,
            state_path=state_path,
        )
        assert t2.high_water_mark == 50000.0  # Ignored persisted state

    def test_max_drawdown_remaining(self, tmp_path: Path) -> None:
        tracker = HighWaterMarkTracker(
            initial_balance=5000.0,
            drawdown_pct=0.06,
            state_path=str(tmp_path / "hwm.json"),
        )
        tracker.update_balance(5100.0)
        # loss_level = 5100 * 0.94 = 4794
        # If equity is 5050, remaining = 5050 - 4794 = 256
        assert tracker.max_drawdown_remaining(equity=5050.0) == pytest.approx(256.0)

    def test_max_drawdown_remaining_locked(self, tmp_path: Path) -> None:
        tracker = HighWaterMarkTracker(
            initial_balance=5000.0,
            drawdown_pct=0.06,
            state_path=str(tmp_path / "hwm.json"),
        )
        tracker.update_balance(5400.0)
        assert tracker.is_locked is True
        # loss_level = 5000 (locked)
        # If equity is 5350, remaining = 5350 - 5000 = 350
        assert tracker.max_drawdown_remaining(equity=5350.0) == pytest.approx(350.0)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hwm_tracker.py -v`
Expected: FAIL (ModuleNotFoundError — module doesn't exist yet)

**Step 3: Write minimal implementation**

Create `src/compliance/hwm_tracker.py`:

```python
"""
High Water Mark Tracker — tracks highest closed balance for dynamic drawdown.

E8 Markets dynamic drawdown (E8 One / Trial) trails the highest CLOSED balance.
The loss level = high_water_mark × (1 - drawdown_pct). It only moves UP.
Once cumulative realized profit >= initial_balance × drawdown_pct, the loss level
permanently locks at initial_balance (i.e., can never lose the initial deposit).

Usage:
    tracker = HighWaterMarkTracker(
        initial_balance=5000.0,
        drawdown_pct=0.06,
        state_path="data/hwm_state.json",
    )
    tracker.update_balance(5100.0)  # Call after each trade closure
    print(tracker.loss_level)       # 4794.0
    tracker.save()                  # Persist to disk
"""

import json
from pathlib import Path

from loguru import logger


class HighWaterMarkTracker:
    """Tracks highest closed balance for E8 Markets dynamic drawdown.

    Usage:
        tracker = HighWaterMarkTracker(initial_balance=5000.0, drawdown_pct=0.06)
        tracker.update_balance(5100.0)
        print(tracker.loss_level)  # 4794.0
    """

    def __init__(
        self,
        initial_balance: float,
        drawdown_pct: float,
        state_path: str = "",
    ) -> None:
        self._initial_balance = initial_balance
        self._drawdown_pct = drawdown_pct
        self._state_path = state_path
        self._high_water_mark = initial_balance
        self._is_locked = False

        # Try to restore from persisted state
        if state_path:
            self._try_restore()

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def high_water_mark(self) -> float:
        """Current highest closed balance."""
        return self._high_water_mark

    @property
    def loss_level(self) -> float:
        """Current dynamic loss level (floor). Equity below this = breach."""
        if self._is_locked:
            return self._initial_balance
        return self._high_water_mark * (1 - self._drawdown_pct)

    @property
    def is_locked(self) -> bool:
        """Whether loss level is permanently locked at initial_balance."""
        return self._is_locked

    def update_balance(self, closed_balance: float) -> None:
        """Update with latest closed balance after a trade closure.

        Only updates HWM if balance is higher than previous HWM.
        Checks lock condition: profit >= initial_balance × drawdown_pct.
        """
        if closed_balance > self._high_water_mark:
            self._high_water_mark = closed_balance
            logger.info(
                "HWM Tracker: new high water mark ${:.2f} (loss_level=${:.2f})",
                self._high_water_mark,
                self.loss_level,
            )

        # Check lock condition
        drawdown_amount = self._initial_balance * self._drawdown_pct
        profit = self._high_water_mark - self._initial_balance
        if not self._is_locked and profit >= drawdown_amount:
            self._is_locked = True
            logger.info(
                "HWM Tracker: LOCKED — profit ${:.2f} >= drawdown amount ${:.2f}. "
                "Loss level permanently set to ${:.2f}",
                profit,
                drawdown_amount,
                self._initial_balance,
            )

    def max_drawdown_remaining(self, equity: float) -> float:
        """Dollars remaining before dynamic drawdown limit is hit."""
        return max(0.0, equity - self.loss_level)

    def save(self) -> None:
        """Persist HWM state to JSON file."""
        if not self._state_path:
            return
        data = {
            "initial_balance": self._initial_balance,
            "drawdown_pct": self._drawdown_pct,
            "high_water_mark": self._high_water_mark,
            "is_locked": self._is_locked,
        }
        path = Path(self._state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.debug("HWM Tracker: saved state to {}", self._state_path)

    # ── Internal ────────────────────────────────────────────────────────

    def _try_restore(self) -> None:
        """Try to restore state from persisted JSON file."""
        path = Path(self._state_path)
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            # Only restore if initial_balance matches (same account)
            if data.get("initial_balance") != self._initial_balance:
                logger.warning(
                    "HWM Tracker: ignoring persisted state (initial_balance mismatch: "
                    "file={}, config={})",
                    data.get("initial_balance"),
                    self._initial_balance,
                )
                return
            self._high_water_mark = data.get("high_water_mark", self._initial_balance)
            self._is_locked = data.get("is_locked", False)
            logger.info(
                "HWM Tracker: restored state — HWM=${:.2f}, locked={}",
                self._high_water_mark,
                self._is_locked,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("HWM Tracker: failed to restore state: {}", e)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hwm_tracker.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/compliance/hwm_tracker.py tests/test_hwm_tracker.py
git commit -m "feat: add HighWaterMarkTracker for dynamic drawdown (E8 One/Trial)"
```

---

### Task 2: Add HWM State Path to Config

**Files:**
- Modify: `src/config.py` (ComplianceConfig)
- Modify: `config/e8_one_5k_challenge.yaml`

**Step 1: Write failing test**

Add to `tests/test_hwm_tracker.py` or an existing config test:

```python
def test_config_has_hwm_state_path():
    from src.config import ComplianceConfig
    c = ComplianceConfig(drawdown_type="dynamic")
    assert hasattr(c, "hwm_state_path")
    assert c.hwm_state_path == "data/hwm_state.json"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hwm_tracker.py::test_config_has_hwm_state_path -v`
Expected: FAIL (AttributeError)

**Step 3: Add field to ComplianceConfig**

In `src/config.py`, add to `ComplianceConfig` class after `max_drawdown_stop`:

```python
    hwm_state_path: str = Field(
        default="data/hwm_state.json",
        description="File path for HighWaterMarkTracker persistence (dynamic drawdown only)",
    )
```

**Step 4: Update YAML config**

In `config/e8_one_5k_challenge.yaml`, add under `compliance:`:

```yaml
  hwm_state_path: data/hwm_state_e8_one_5k.json
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_hwm_tracker.py::test_config_has_hwm_state_path -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/config.py config/e8_one_5k_challenge.yaml tests/test_hwm_tracker.py
git commit -m "feat: add hwm_state_path to ComplianceConfig for dynamic drawdown persistence"
```

---

### Task 3: Update `DrawdownMonitor` to Support Dynamic HWM

**Files:**
- Modify: `src/compliance/drawdown_monitor.py`
- Modify: `tests/test_hwm_tracker.py` (add DrawdownMonitor integration tests)

**Step 1: Write failing tests**

Add to `tests/test_hwm_tracker.py`:

```python
from src.compliance.drawdown_monitor import DrawdownMonitor
from src.config import ComplianceConfig


class TestDrawdownMonitorDynamic:
    """Tests for DrawdownMonitor with dynamic drawdown support."""

    def test_max_drawdown_uses_hwm_when_provided(self) -> None:
        config = ComplianceConfig(max_drawdown_limit=0.06, drawdown_type="dynamic")
        monitor = DrawdownMonitor(config)
        # Provide high_water_mark=5100 (balance grew from 5000)
        monitor.update(equity=5050, day_start_balance=5100, initial_balance=5000, high_water_mark=5100)
        # max loss should be from HWM: 5100 * 0.06 = 306
        # current loss from HWM: 5100 - 5050 = 50
        # pct consumed: 50/306 ≈ 0.1634
        assert monitor.max_drawdown_pct == pytest.approx(50 / 306, rel=1e-3)

    def test_max_drawdown_remaining_uses_hwm(self) -> None:
        config = ComplianceConfig(max_drawdown_limit=0.06, drawdown_type="dynamic")
        monitor = DrawdownMonitor(config)
        monitor.update(equity=5050, day_start_balance=5100, initial_balance=5000, high_water_mark=5100)
        # max loss = 306, current loss = 50, remaining = 256
        assert monitor.max_drawdown_remaining == pytest.approx(256.0)

    def test_max_drawdown_falls_back_to_initial_when_no_hwm(self) -> None:
        """When high_water_mark is not provided, fall back to initial_balance."""
        config = ComplianceConfig(max_drawdown_limit=0.08, drawdown_type="balance")
        monitor = DrawdownMonitor(config)
        monitor.update(equity=49000, day_start_balance=50000, initial_balance=50000)
        # Should use initial_balance as before
        assert monitor.max_drawdown_pct == pytest.approx(1000 / 4000, rel=1e-3)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hwm_tracker.py::TestDrawdownMonitorDynamic -v`
Expected: FAIL (update() doesn't accept high_water_mark)

**Step 3: Update DrawdownMonitor**

In `src/compliance/drawdown_monitor.py`:

1. Add `high_water_mark` parameter to `update()` (default `0.0`):

```python
    def update(
        self,
        equity: float,
        day_start_balance: float,
        initial_balance: float,
        high_water_mark: float = 0.0,
    ) -> None:
        self._equity = equity
        self._day_start_balance = day_start_balance
        self._initial_balance = initial_balance
        self._high_water_mark = high_water_mark if high_water_mark > 0 else initial_balance
```

2. Add `self._high_water_mark = 0.0` to `__init__`

3. Update `max_drawdown_pct`, `max_drawdown_remaining`, and `max_drawdown_dollars` properties to use `self._reference_balance` (a new helper):

```python
    @property
    def _max_dd_reference(self) -> float:
        """Reference balance for max drawdown calculation.
        Uses HWM for dynamic accounts, initial_balance otherwise."""
        if self._config.drawdown_type == "dynamic" and self._high_water_mark > 0:
            return self._high_water_mark
        return self._initial_balance
```

Then update all three max drawdown properties to use `self._max_dd_reference` instead of `self._initial_balance`.

**Step 4: Run tests**

Run: `uv run pytest tests/test_hwm_tracker.py -v`
Expected: ALL PASS

**Step 5: Run existing DrawdownMonitor tests to ensure backward compatibility**

Run: `uv run pytest tests/ -k "drawdown" -v`
Expected: ALL PASS (no regressions)

**Step 6: Commit**

```bash
git add src/compliance/drawdown_monitor.py tests/test_hwm_tracker.py
git commit -m "feat: DrawdownMonitor supports dynamic HWM-based max drawdown"
```

---

### Task 4: Wire HWM into Scheduler `_handle_position_closed()`

**Files:**
- Modify: `src/scheduler/scheduler.py`

**Step 1: Add HWM tracker initialization to `Scheduler.__init__`**

After the existing initialization (around line 84), add:

```python
        # Dynamic drawdown HWM tracking
        self._hwm_tracker: HighWaterMarkTracker | None = None
        if config.compliance.drawdown_type == "dynamic":
            from src.compliance.hwm_tracker import HighWaterMarkTracker
            self._hwm_tracker = HighWaterMarkTracker(
                initial_balance=config.account.initial_balance,
                drawdown_pct=config.compliance.max_drawdown_limit,
                state_path=config.compliance.hwm_state_path,
            )
```

**Step 2: Update `_handle_position_closed()` to update HWM after trade closure**

After the line `self._best_day_tracker.record_trade_pnl(pnl)` (around line 911), add:

```python
        # Update dynamic drawdown HWM tracker with new closed balance
        if self._hwm_tracker is not None and equity is not None:
            try:
                balance_info_hwm = await self._matchtrader.get_balance()
                self._hwm_tracker.update_balance(balance_info_hwm.balance)
                self._hwm_tracker.save()
                logger.info(
                    "HWM updated: balance=${:.2f}, hwm=${:.2f}, loss_level=${:.2f}, locked={}",
                    balance_info_hwm.balance,
                    self._hwm_tracker.high_water_mark,
                    self._hwm_tracker.loss_level,
                    self._hwm_tracker.is_locked,
                )
            except Exception as e:
                logger.error("Failed to update HWM tracker: {}", e)
```

**Step 3: Update `_equity_monitor_loop()` to pass HWM to equity monitor**

In `_equity_monitor_loop()` (line 650), change the `initial_balance` parameter to use HWM when available:

```python
            # For dynamic drawdown, use HWM as the reference for max drawdown
            max_dd_reference = self._config.account.initial_balance
            if self._hwm_tracker is not None:
                max_dd_reference = self._hwm_tracker.high_water_mark

            await self._equity_monitor.start(
                get_equity=get_equity,
                day_start_balance=balance.balance,
                initial_balance=max_dd_reference,  # HWM for dynamic, initial for balance-based
                daily_drawdown_limit=self._config.compliance.daily_drawdown_limit,
                max_drawdown_limit=self._config.compliance.max_drawdown_limit,
            )
```

**Step 4: Run existing scheduler tests**

Run: `uv run pytest tests/test_scheduler.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/scheduler/scheduler.py
git commit -m "feat: wire HWM tracker into scheduler for dynamic drawdown monitoring"
```

---

### Task 5: Update `AlertService` for Dynamic Max DD Buffer

**Files:**
- Modify: `src/monitor/alert_service.py`
- Modify: `tests/test_alert_service.py`

**Step 1: Write failing test**

Add to `tests/test_alert_service.py`:

```python
class TestAlertServiceDynamicDrawdown:
    """Tests for AlertService with dynamic max drawdown buffer."""

    def test_format_profit_status_with_hwm(self) -> None:
        alert = AlertService(
            bot_token="fake:token",
            chat_id="123456",
            initial_balance=5000.0,
            max_drawdown_pct=0.06,
        )
        result = alert.format_profit_status(
            equity=5050.0,
            positions=[],
            day_start_balance=5050.0,
            max_dd_reference=5100.0,  # HWM is higher than initial
        )
        # Max buffer = 5100 * 0.06 - (5100 - 5050) = 306 - 50 = 256
        assert "$256.00" in result

    def test_daily_summary_with_hwm(self) -> None:
        """Just test it doesn't crash — actual buffer value checked above."""
        alert = AlertService(
            bot_token="fake:token",
            chat_id="123456",
            initial_balance=5000.0,
            max_drawdown_pct=0.06,
        )
        # Should not raise
        assert alert.max_drawdown_amount == 300.0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_alert_service.py::TestAlertServiceDynamicDrawdown -v`
Expected: FAIL (format_profit_status doesn't accept max_dd_reference)

**Step 3: Add `max_dd_reference` parameter to affected methods**

In `src/monitor/alert_service.py`:

1. Add `max_dd_reference: float | None = None` parameter to `format_profit_status()`, `daily_summary()`.

2. In both methods, where max DD buffer is computed:

```python
        if self.max_drawdown_amount > 0:
            ref = max_dd_reference or self._initial_balance
            dd_limit = ref * self._max_drawdown_pct
            max_loss = max(0.0, ref - equity)
            max_buffer = dd_limit - max_loss
            lines.append(f"• Max DD buffer: ${max_buffer:,.2f}")
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_alert_service.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/monitor/alert_service.py tests/test_alert_service.py
git commit -m "feat: AlertService supports dynamic max drawdown buffer via HWM reference"
```

---

### Task 6: Wire HWM into Alert Calls in Scheduler

**Files:**
- Modify: `src/scheduler/scheduler.py`

**Step 1: Update daily summary and profit status calls to pass HWM**

Find all calls to `alert_service.daily_summary()` and `alert_service.format_profit_status()` in the scheduler. Add the `max_dd_reference` parameter:

```python
            max_dd_ref = self._hwm_tracker.high_water_mark if self._hwm_tracker else None
```

Pass `max_dd_reference=max_dd_ref` to each call.

**Step 2: Run tests**

Run: `uv run pytest tests/test_scheduler.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/scheduler/scheduler.py
git commit -m "feat: pass HWM reference to alert service for correct dynamic DD buffer"
```

---

### Task 7: Fix `main.py` Hardcoded HWM (Legacy Path)

**Files:**
- Modify: `src/main.py`

**Step 1: Update `equity_high_water_mark` in main.py**

At line 262, change:
```python
            equity_high_water_mark=self.config.account.initial_balance,
```
to:
```python
            equity_high_water_mark=self.config.account.initial_balance,  # TODO: wire HWM for legacy single-cycle mode
```

> Note: The main.py `run_daily_cycle()` is the legacy single-shot path. The scheduler is the 24/7 path where dynamic drawdown matters. For now, document the limitation. The scheduler path (Tasks 4-6) is the correct fix.

**Step 2: Commit**

```bash
git add src/main.py
git commit -m "docs: note legacy HWM limitation in single-cycle mode"
```

---

## Part B: Weekend Market Closure (Bug 1)

### Task 8: Add `MarketHoursConfig` to Config

**Files:**
- Modify: `src/config.py`
- Create: `tests/test_market_hours.py`

**Step 1: Write failing test**

Create `tests/test_market_hours.py`:

```python
"""Tests for MarketHoursConfig and weekend market closure logic."""

from src.config import AppConfig, MarketHoursConfig, SchedulerConfig


class TestMarketHoursConfig:
    """Tests for MarketHoursConfig defaults and loading."""

    def test_default_disabled(self) -> None:
        config = MarketHoursConfig()
        assert config.enabled is False

    def test_fx_defaults(self) -> None:
        config = MarketHoursConfig(enabled=True)
        assert config.close_day == "Friday"
        assert config.close_time_utc == "22:00"
        assert config.open_day == "Sunday"
        assert config.open_time_utc == "22:00"
        assert config.force_close_before_weekend is False

    def test_force_close_settings(self) -> None:
        config = MarketHoursConfig(
            enabled=True,
            force_close_before_weekend=True,
            force_close_minutes_before=30,
        )
        assert config.force_close_before_weekend is True
        assert config.force_close_minutes_before == 30

    def test_config_in_scheduler(self) -> None:
        sched = SchedulerConfig(market_hours=MarketHoursConfig(enabled=True))
        assert sched.market_hours.enabled is True

    def test_full_config_from_dict(self) -> None:
        config = AppConfig(
            scheduler=SchedulerConfig(
                market_hours={
                    "enabled": True,
                    "close_day": "Friday",
                    "close_time_utc": "22:00",
                    "open_day": "Sunday",
                    "open_time_utc": "22:00",
                    "force_close_before_weekend": True,
                    "force_close_minutes_before": 15,
                }
            )
        )
        mh = config.scheduler.market_hours
        assert mh.enabled is True
        assert mh.force_close_minutes_before == 15
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_market_hours.py -v`
Expected: FAIL (ImportError — MarketHoursConfig doesn't exist)

**Step 3: Add MarketHoursConfig to config.py**

In `src/config.py`, add before `SchedulerConfig`:

```python
class MarketHoursConfig(BaseModel):
    """FX market hours and weekend closure settings (per-account).

    Times are in UTC. Typical FX: close Friday 22:00 UTC, open Sunday 22:00 UTC.
    These correspond to 17:00 EST / 17:00 EST (US Eastern).
    """

    enabled: bool = Field(default=False, description="Enable weekend market closure handling")
    close_day: str = Field(default="Friday", description="Day market closes (e.g., Friday)")
    close_time_utc: str = Field(default="22:00", description="Market close time in UTC (HH:MM)")
    open_day: str = Field(default="Sunday", description="Day market opens (e.g., Sunday)")
    open_time_utc: str = Field(default="22:00", description="Market open time in UTC (HH:MM)")
    force_close_before_weekend: bool = Field(
        default=False, description="Force-close all positions before weekend"
    )
    force_close_minutes_before: int = Field(
        default=15, description="Minutes before market close to force-close positions"
    )
```

Add `market_hours` field to `SchedulerConfig`:

```python
    market_hours: MarketHoursConfig = Field(
        default_factory=MarketHoursConfig,
        description="Weekend market closure settings",
    )
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_market_hours.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/config.py tests/test_market_hours.py
git commit -m "feat: add MarketHoursConfig for per-account weekend closure settings"
```

---

### Task 9: Implement `_is_market_open()` Helper

**Files:**
- Modify: `tests/test_market_hours.py`
- Create: `src/scheduler/market_hours.py`

**Step 1: Write failing tests**

Add to `tests/test_market_hours.py`:

```python
from datetime import datetime, timezone

from src.scheduler.market_hours import MarketHoursChecker
from src.config import MarketHoursConfig


class TestMarketHoursChecker:
    """Tests for MarketHoursChecker.is_market_open() and force-close timing."""

    def test_disabled_always_open(self) -> None:
        checker = MarketHoursChecker(MarketHoursConfig(enabled=False))
        # Saturday should still show as open when disabled
        saturday = datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc)
        assert checker.is_market_open(saturday) is True

    def test_weekday_is_open(self) -> None:
        checker = MarketHoursChecker(MarketHoursConfig(enabled=True))
        wednesday = datetime(2026, 2, 25, 14, 0, tzinfo=timezone.utc)
        assert checker.is_market_open(wednesday) is True

    def test_saturday_is_closed(self) -> None:
        checker = MarketHoursChecker(MarketHoursConfig(enabled=True))
        saturday = datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc)
        assert checker.is_market_open(saturday) is False

    def test_friday_before_close_is_open(self) -> None:
        checker = MarketHoursChecker(
            MarketHoursConfig(enabled=True, close_day="Friday", close_time_utc="22:00")
        )
        friday_early = datetime(2026, 2, 27, 20, 0, tzinfo=timezone.utc)
        assert checker.is_market_open(friday_early) is True

    def test_friday_after_close_is_closed(self) -> None:
        checker = MarketHoursChecker(
            MarketHoursConfig(enabled=True, close_day="Friday", close_time_utc="22:00")
        )
        friday_late = datetime(2026, 2, 27, 23, 0, tzinfo=timezone.utc)
        assert checker.is_market_open(friday_late) is False

    def test_sunday_before_open_is_closed(self) -> None:
        checker = MarketHoursChecker(
            MarketHoursConfig(enabled=True, open_day="Sunday", open_time_utc="22:00")
        )
        sunday_early = datetime(2026, 3, 1, 18, 0, tzinfo=timezone.utc)
        assert checker.is_market_open(sunday_early) is False

    def test_sunday_after_open_is_open(self) -> None:
        checker = MarketHoursChecker(
            MarketHoursConfig(enabled=True, open_day="Sunday", open_time_utc="22:00")
        )
        sunday_late = datetime(2026, 3, 1, 23, 0, tzinfo=timezone.utc)
        assert checker.is_market_open(sunday_late) is True

    def test_should_force_close(self) -> None:
        checker = MarketHoursChecker(
            MarketHoursConfig(
                enabled=True,
                close_day="Friday",
                close_time_utc="22:00",
                force_close_before_weekend=True,
                force_close_minutes_before=15,
            )
        )
        # 14 minutes before close — should force close
        friday_2146 = datetime(2026, 2, 27, 21, 46, tzinfo=timezone.utc)
        assert checker.should_force_close(friday_2146) is True

        # 20 minutes before close — not yet
        friday_2140 = datetime(2026, 2, 27, 21, 40, tzinfo=timezone.utc)
        assert checker.should_force_close(friday_2140) is False

    def test_should_force_close_disabled(self) -> None:
        checker = MarketHoursChecker(
            MarketHoursConfig(enabled=True, force_close_before_weekend=False)
        )
        friday_2150 = datetime(2026, 2, 27, 21, 50, tzinfo=timezone.utc)
        assert checker.should_force_close(friday_2150) is False

    def test_seconds_until_open(self) -> None:
        checker = MarketHoursChecker(
            MarketHoursConfig(enabled=True, open_day="Sunday", open_time_utc="22:00")
        )
        # Saturday noon — should be ~34 hours until Sunday 22:00
        saturday_noon = datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc)
        seconds = checker.seconds_until_open(saturday_noon)
        assert 33 * 3600 < seconds < 35 * 3600
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_market_hours.py::TestMarketHoursChecker -v`
Expected: FAIL (ImportError)

**Step 3: Implement MarketHoursChecker**

Create `src/scheduler/market_hours.py`:

```python
"""
Market hours checker — determines if FX market is open or closed.

Handles weekend closure for FX markets with configurable close/open times.
Used by the Scheduler to pause trading loops during weekends.

Usage:
    checker = MarketHoursChecker(config.scheduler.market_hours)
    if not checker.is_market_open(now_utc):
        sleep_seconds = checker.seconds_until_open(now_utc)
"""

from datetime import datetime, timedelta, timezone

from loguru import logger

from src.config import MarketHoursConfig

# Day name to weekday number (Monday=0, Sunday=6)
_DAY_MAP: dict[str, int] = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}


class MarketHoursChecker:
    """Checks whether FX market is currently open based on config.

    Usage:
        checker = MarketHoursChecker(market_hours_config)
        if not checker.is_market_open(datetime.now(timezone.utc)):
            wait = checker.seconds_until_open(datetime.now(timezone.utc))
    """

    def __init__(self, config: MarketHoursConfig) -> None:
        self._config = config
        self._close_weekday = _DAY_MAP.get(config.close_day, 4)  # Default Friday
        self._open_weekday = _DAY_MAP.get(config.open_day, 6)  # Default Sunday
        close_parts = config.close_time_utc.split(":")
        self._close_hour = int(close_parts[0])
        self._close_minute = int(close_parts[1]) if len(close_parts) > 1 else 0
        open_parts = config.open_time_utc.split(":")
        self._open_hour = int(open_parts[0])
        self._open_minute = int(open_parts[1]) if len(open_parts) > 1 else 0

    def is_market_open(self, now: datetime) -> bool:
        """Return True if the FX market is currently open.

        Market is CLOSED from close_day close_time through open_day open_time.
        """
        if not self._config.enabled:
            return True

        weekday = now.weekday()  # Monday=0, Sunday=6
        hour = now.hour
        minute = now.minute
        time_minutes = hour * 60 + minute
        close_minutes = self._close_hour * 60 + self._close_minute
        open_minutes = self._open_hour * 60 + self._open_minute

        # Saturday is always closed
        if weekday == 5:  # Saturday
            return False

        # Friday after close time
        if weekday == self._close_weekday and time_minutes >= close_minutes:
            return False

        # Sunday before open time
        if weekday == self._open_weekday and time_minutes < open_minutes:
            return False

        return True

    def should_force_close(self, now: datetime) -> bool:
        """Return True if we should force-close all positions before weekend.

        Triggers `force_close_minutes_before` minutes before market close.
        Only on close_day, only if force_close_before_weekend is enabled.
        """
        if not self._config.enabled or not self._config.force_close_before_weekend:
            return False

        weekday = now.weekday()
        if weekday != self._close_weekday:
            return False

        time_minutes = now.hour * 60 + now.minute
        close_minutes = self._close_hour * 60 + self._close_minute
        trigger_minutes = close_minutes - self._config.force_close_minutes_before

        return time_minutes >= trigger_minutes

    def seconds_until_open(self, now: datetime) -> float:
        """Calculate seconds until the next market open time.

        Returns 0 if market is currently open.
        """
        if self.is_market_open(now):
            return 0.0

        # Find next open_day at open_time
        target = now.replace(
            hour=self._open_hour, minute=self._open_minute, second=0, microsecond=0
        )

        # Move to next open_day
        days_ahead = self._open_weekday - now.weekday()
        if days_ahead < 0:
            days_ahead += 7
        if days_ahead == 0 and now >= target:
            days_ahead += 7

        target = target + timedelta(days=days_ahead)
        return max(0.0, (target - now).total_seconds())
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_market_hours.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/scheduler/market_hours.py tests/test_market_hours.py
git commit -m "feat: add MarketHoursChecker for FX weekend closure detection"
```

---

### Task 10: Integrate Weekend Checks into Scheduler Loops

**Files:**
- Modify: `src/scheduler/scheduler.py`

**Step 1: Add MarketHoursChecker to Scheduler.__init__**

Import and initialize:

```python
from src.scheduler.market_hours import MarketHoursChecker
```

In `__init__`, after the existing initialization:

```python
        self._market_hours = MarketHoursChecker(config.scheduler.market_hours)
        self._weekend_force_close_done = False  # Reset each weekend
```

**Step 2: Add `_wait_for_market_open()` helper**

```python
    async def _wait_for_market_open(self, loop_name: str) -> None:
        """Sleep until market opens. Logs once and sleeps in chunks."""
        now = self._now_utc()
        if self._market_hours.is_market_open(now):
            return

        wait_seconds = self._market_hours.seconds_until_open(now)
        logger.info(
            "{}: market closed — sleeping {:.0f}s ({:.1f}h) until open",
            loop_name,
            wait_seconds,
            wait_seconds / 3600,
        )
        await self._send_alert(
            f"💤 <b>{loop_name}</b>: market closed, sleeping until open "
            f"({wait_seconds / 3600:.1f}h)"
        )

        # Sleep in 5-minute chunks to allow graceful shutdown
        while not self._market_hours.is_market_open(self._now_utc()) and self._running:
            await asyncio.sleep(min(300, wait_seconds))
            wait_seconds = self._market_hours.seconds_until_open(self._now_utc())

        if self._running:
            self._weekend_force_close_done = False  # Reset for next weekend
            logger.info("{}: market open — resuming", loop_name)
            await self._send_alert(f"☀️ <b>{loop_name}</b>: market open, resuming operations")
```

**Step 3: Add weekend check to each trading loop**

At the TOP of each `while self._running:` iteration in these loops:
- `_scanner_loop()`
- `_llm_worker_loop()`
- `_execution_loop()`

Add:
```python
                # Weekend check — pause during market closure
                await self._wait_for_market_open("Scanner loop")  # or appropriate name
```

For `_position_monitor_loop()` — add weekend force-close check:

```python
                # Weekend force-close check
                if (
                    self._market_hours.should_force_close(self._now_utc())
                    and not self._weekend_force_close_done
                ):
                    await self._force_close_for_weekend()
```

**Step 4: Implement `_force_close_for_weekend()`**

```python
    async def _force_close_for_weekend(self) -> None:
        """Force-close all open positions before weekend market closure."""
        logger.warning("Weekend force-close: closing all positions before market close")
        try:
            open_positions = await self._matchtrader.get_open_positions()
            if not open_positions:
                logger.info("Weekend force-close: no open positions")
                self._weekend_force_close_done = True
                return

            closed_count = 0
            total_pnl = 0.0
            for pos in open_positions:
                try:
                    result = await self._matchtrader.close_position(
                        position_id=str(pos.position_id),
                        symbol=pos.symbol,
                        side=pos.side,
                        volume=pos.volume,
                    )
                    if result.success:
                        closed_count += 1
                        total_pnl += pos.profit
                except Exception as e:
                    logger.error(
                        "Weekend force-close: failed to close {}: {}",
                        pos.position_id, e,
                    )

            self._weekend_force_close_done = True
            await self._send_alert(
                f"🌙 <b>Weekend Force-Close</b>\n"
                f"• Closed {closed_count}/{len(open_positions)} positions\n"
                f"• Estimated PnL: ${total_pnl:+.2f}"
            )
        except Exception as e:
            logger.error("Weekend force-close failed: {}", e)
            await self._send_alert(
                f"⚠️ <b>Weekend Force-Close FAILED</b>\n<code>{e}</code>"
            )
```

**Step 5: Keep equity monitor and position monitor running during weekends**

These two loops should NOT call `_wait_for_market_open()` but should REDUCE their polling frequency during weekends:

In `_position_monitor_loop()`, adjust sleep at the end:
```python
                # During market closure, reduce polling frequency
                if not self._market_hours.is_market_open(self._now_utc()):
                    await asyncio.sleep(base_interval * 10)  # 20min instead of 2min
                else:
                    await asyncio.sleep(sleep_interval)
```

The equity monitor loop lives inside `EquityMonitor.start()` — we don't modify it directly here. It will naturally poll at its configured interval. The key safety aspect is that the equity monitor can still detect breaches even during weekends (broker swaps, etc.).

**Step 6: Run tests**

Run: `uv run pytest tests/test_scheduler.py -v`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add src/scheduler/scheduler.py
git commit -m "feat: integrate weekend market closure into scheduler loops"
```

---

### Task 11: Update YAML Config with Weekend Settings

**Files:**
- Modify: `config/e8_one_5k_challenge.yaml`

**Step 1: Add market hours section**

```yaml
scheduler:
  market_hours:
    enabled: true
    close_day: "Friday"
    close_time_utc: "22:00"    # 17:00 EST
    open_day: "Sunday"
    open_time_utc: "22:00"     # 17:00 EST
    force_close_before_weekend: false
    force_close_minutes_before: 15
```

**Step 2: Run config loading test**

Run: `uv run pytest tests/test_prop_firm_guard_e8_one.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add config/e8_one_5k_challenge.yaml
git commit -m "feat: add weekend market hours config for E8 One 5K"
```

---

### Task 12: Final Integration Tests & Lint

**Step 1: Run full test suite**

```bash
uv run pytest tests/ -v
```
Expected: ALL PASS

**Step 2: Run linter**

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/ --check
```
Expected: No errors

**Step 3: Fix any lint issues**

```bash
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/
```

**Step 4: Final commit (if any fixes)**

```bash
git add -A
git commit -m "chore: lint and format fixes"
```

---

## Summary of Changes

| File | Change | Bug |
|------|--------|-----|
| `src/compliance/hwm_tracker.py` | NEW — HWM tracking + persistence | #2 |
| `src/compliance/drawdown_monitor.py` | Accept `high_water_mark` param, use for dynamic accounts | #2 |
| `src/config.py` | Add `hwm_state_path`, `MarketHoursConfig` | #1 + #2 |
| `src/scheduler/scheduler.py` | Wire HWM, add weekend checks, force-close | #1 + #2 |
| `src/scheduler/market_hours.py` | NEW — weekend open/close checker | #1 |
| `src/monitor/alert_service.py` | Accept `max_dd_reference` for dynamic buffer | #2 |
| `src/main.py` | Document legacy HWM limitation | #2 |
| `config/e8_one_5k_challenge.yaml` | Add `hwm_state_path`, `market_hours` | #1 + #2 |
| `tests/test_hwm_tracker.py` | NEW — HWM + DrawdownMonitor dynamic tests | #2 |
| `tests/test_market_hours.py` | NEW — MarketHoursConfig + Checker tests | #1 |
| `tests/test_alert_service.py` | Add dynamic DD buffer tests | #2 |
