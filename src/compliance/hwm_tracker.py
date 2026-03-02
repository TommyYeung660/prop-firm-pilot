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
