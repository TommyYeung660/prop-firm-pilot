"""
Real-time equity monitor — periodically checks account equity
and triggers alerts / emergency actions when drawdown limits approach.

Designed to run as an async background task during trading hours.
"""

import asyncio
from typing import Any, Callable, Coroutine, Literal

from loguru import logger


class EquityMonitor:
    """Monitors account equity in real-time and triggers protective actions.

    Usage:
        monitor = EquityMonitor(
            check_interval=60,
            drawdown_alert_pct=0.80,
            auto_close_pct=0.90,
        )
        await monitor.start(
            get_balance=client.get_balance,
            on_alert=alert_service.drawdown_warning,
            on_emergency_close=client.close_all_positions,
            day_start_balance=50000,
            initial_balance=50000,
            daily_drawdown_limit=0.05,
            max_drawdown_limit=0.08,
        )
    """

    def __init__(
        self,
        check_interval: int = 60,
        drawdown_alert_pct: float = 0.80,
        auto_close_pct: float = 0.90,
    ) -> None:
        self._check_interval = check_interval
        self._alert_pct = drawdown_alert_pct
        self._auto_close_pct = auto_close_pct
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._last_alert_level: str = "SAFE"

    async def start(
        self,
        get_equity: Callable[[], Coroutine[Any, Any, float]],
        on_alert: Callable[[str, float, float, float], Coroutine[Any, Any, Any]] | None = None,
        on_emergency_close: Callable[[], Coroutine[Any, Any, Any]] | None = None,
        on_equity_snapshot: Callable[[float, float, float], Coroutine[Any, Any, Any]] | None = None,
        day_start_balance: float = 50000.0,
        initial_balance: float = 50000.0,
        daily_drawdown_limit: float = 0.05,
        max_drawdown_limit: float = 0.08,
    ) -> None:
        """Start the equity monitoring loop.

        Args:
            get_equity: Async callable that returns current equity.
            on_alert: Callback(level, daily_dd_pct, max_dd_pct, equity).
            on_emergency_close: Callback to close all positions.
            on_equity_snapshot: Callback(equity, daily_dd_pct, max_dd_pct) for logging.
            day_start_balance: Balance at the start of the trading day.
            initial_balance: Initial account balance ($50k for E8).
            daily_drawdown_limit: E8 daily drawdown limit (0.05 = 5%).
            max_drawdown_limit: E8 max drawdown limit (0.08 = 8%).
        """
        self._running = True
        logger.info(
            "EquityMonitor: starting (interval={}s, alert={}%, close={}%)",
            self._check_interval,
            self._alert_pct * 100,
            self._auto_close_pct * 100,
        )

        while self._running:
            try:
                equity = await get_equity()

                # Calculate drawdown percentages
                daily_dd_pct = self._calc_drawdown_pct(
                    equity, day_start_balance, daily_drawdown_limit
                )
                max_dd_pct = self._calc_drawdown_pct(equity, initial_balance, max_drawdown_limit)

                # Determine alert level
                worst_pct = max(daily_dd_pct, max_dd_pct)
                level = self._classify_level(worst_pct)

                # Log snapshot
                if on_equity_snapshot:
                    await on_equity_snapshot(equity, daily_dd_pct, max_dd_pct)

                # Trigger alert if level changed (upward)
                if self._is_escalation(level) and on_alert:
                    await on_alert(level, daily_dd_pct, max_dd_pct, equity)
                    self._last_alert_level = level

                # Emergency close
                if worst_pct >= self._auto_close_pct:
                    logger.critical(
                        "EquityMonitor: CRITICAL drawdown {:.1%} — triggering emergency close!",
                        worst_pct,
                    )
                    if on_emergency_close:
                        await on_emergency_close()
                    self._running = False
                    break

            except Exception as e:
                logger.error("EquityMonitor: check failed: {}", e)

            await asyncio.sleep(self._check_interval)

        logger.info("EquityMonitor: stopped")

    def stop(self) -> None:
        """Signal the monitor to stop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    @property
    def is_running(self) -> bool:
        return self._running

    def _calc_drawdown_pct(self, equity: float, reference_balance: float, limit: float) -> float:
        """Calculate what fraction of the drawdown limit has been consumed.

        Returns 0.0 (no drawdown) to 1.0+ (limit breached).
        """
        if reference_balance <= 0 or limit <= 0:
            return 0.0
        max_loss = reference_balance * limit
        current_loss = max(0.0, reference_balance - equity)
        return current_loss / max_loss if max_loss > 0 else 0.0

    def _classify_level(self, pct_of_limit: float) -> str:
        """Classify drawdown severity."""
        if pct_of_limit >= 0.90:
            return "CRITICAL"
        if pct_of_limit >= 0.80:
            return "DANGER"
        if pct_of_limit >= 0.50:
            return "WARNING"
        return "SAFE"

    def _is_escalation(self, new_level: str) -> bool:
        """Check if the new level is worse than the last alerted level."""
        levels = ["SAFE", "WARNING", "DANGER", "CRITICAL"]
        try:
            new_idx = levels.index(new_level)
            old_idx = levels.index(self._last_alert_level)
            return new_idx > old_idx
        except ValueError:
            return False
