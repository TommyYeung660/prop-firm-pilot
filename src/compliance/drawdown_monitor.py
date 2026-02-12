"""
Real-time drawdown monitor — tracks daily and max drawdown levels
and classifies alert severity for the equity monitoring loop.

This is a pure calculation module — no HTTP calls, no side effects.
The EquityMonitor calls update() periodically and reads the results.
"""

from typing import Literal

from loguru import logger

from src.config import ComplianceConfig


class DrawdownMonitor:
    """Tracks real-time drawdown levels against E8 Markets limits.

    Usage:
        monitor = DrawdownMonitor(compliance_config)
        monitor.update(equity=49200, day_start_balance=50000, initial_balance=50000)
        print(monitor.alert_level)          # "WARNING"
        print(monitor.daily_drawdown_pct)   # 0.32 (32% of daily limit consumed)
        print(monitor.daily_drawdown_remaining)  # 1700.0 dollars remaining
    """

    def __init__(self, config: ComplianceConfig) -> None:
        self._config = config
        self._equity = 0.0
        self._day_start_balance = 0.0
        self._initial_balance = 0.0

    def update(
        self,
        equity: float,
        day_start_balance: float,
        initial_balance: float,
    ) -> None:
        """Update drawdown monitor with latest account values.

        Should be called periodically (e.g. every 60 seconds).
        """
        self._equity = equity
        self._day_start_balance = day_start_balance
        self._initial_balance = initial_balance

    # ── Daily Drawdown ──────────────────────────────────────────────────

    @property
    def daily_drawdown_pct(self) -> float:
        """Current daily drawdown as fraction of the daily limit consumed.

        Returns 0.0 (no drawdown) to 1.0+ (limit breached).
        E.g. 0.5 means 50% of the 5% daily drawdown limit has been used.
        """
        if self._day_start_balance <= 0:
            return 0.0
        max_loss = self._day_start_balance * self._config.daily_drawdown_limit
        current_loss = max(0.0, self._day_start_balance - self._equity)
        return current_loss / max_loss if max_loss > 0 else 0.0

    @property
    def daily_drawdown_remaining(self) -> float:
        """Dollars remaining before daily drawdown limit is hit."""
        max_loss = self._day_start_balance * self._config.daily_drawdown_limit
        current_loss = max(0.0, self._day_start_balance - self._equity)
        return max(0.0, max_loss - current_loss)

    @property
    def daily_drawdown_dollars(self) -> float:
        """Current daily drawdown in dollars."""
        return max(0.0, self._day_start_balance - self._equity)

    # ── Max Drawdown ────────────────────────────────────────────────────

    @property
    def max_drawdown_pct(self) -> float:
        """Current max drawdown as fraction of the max limit consumed.

        Returns 0.0 (no drawdown) to 1.0+ (limit breached).
        """
        if self._initial_balance <= 0:
            return 0.0
        max_loss = self._initial_balance * self._config.max_drawdown_limit
        current_loss = max(0.0, self._initial_balance - self._equity)
        return current_loss / max_loss if max_loss > 0 else 0.0

    @property
    def max_drawdown_remaining(self) -> float:
        """Dollars remaining before max drawdown limit is hit."""
        max_loss = self._initial_balance * self._config.max_drawdown_limit
        current_loss = max(0.0, self._initial_balance - self._equity)
        return max(0.0, max_loss - current_loss)

    @property
    def max_drawdown_dollars(self) -> float:
        """Current total drawdown in dollars (from initial balance)."""
        return max(0.0, self._initial_balance - self._equity)

    # ── Alert Classification ────────────────────────────────────────────

    @property
    def alert_level(self) -> Literal["SAFE", "WARNING", "DANGER", "CRITICAL"]:
        """Classify current drawdown severity.

        SAFE:     < 50% of either limit consumed
        WARNING:  50-80% of either limit consumed
        DANGER:   80-90% (should_stop_trading threshold)
        CRITICAL: > 90% (auto-close all positions)
        """
        worst = max(self.daily_drawdown_pct, self.max_drawdown_pct)

        if worst >= 0.90:
            return "CRITICAL"
        if worst >= 0.80:
            return "DANGER"
        if worst >= 0.50:
            return "WARNING"
        return "SAFE"

    def summary(self) -> str:
        """Human-readable drawdown summary."""
        return (
            f"DD Monitor: daily={self.daily_drawdown_pct:.1%} "
            f"(${self.daily_drawdown_remaining:.0f} remaining), "
            f"max={self.max_drawdown_pct:.1%} "
            f"(${self.max_drawdown_remaining:.0f} remaining), "
            f"level={self.alert_level}"
        )
