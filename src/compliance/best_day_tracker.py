"""
40% Best Day Rule tracker — ensures single-day profit does not exceed
40% of the profit target, which for E8 Signature $50k = $1,600/day.

Tracks both realized and unrealized PnL to prevent approaching the limit.
"""

from loguru import logger


class BestDayTracker:
    """Tracks daily PnL against E8's 40% Best Day Rule.

    The Best Day Rule: No single trading day's profit may exceed 40% of
    the total profit target. For a $50k account with 8% target ($4,000),
    this means max $1,600 profit per day.

    Usage:
        tracker = BestDayTracker(best_day_limit=1600.0, stop_ratio=0.85)
        tracker.record_trade_pnl(500.0)   # Realized +$500
        tracker.update_unrealized(300.0)   # Floating +$300
        print(tracker.daily_pnl)           # 800.0
        print(tracker.remaining_capacity)  # 560.0 ($1360 safety - $800)
        print(tracker.can_take_more_profit(600))  # False
    """

    def __init__(
        self,
        best_day_limit: float = 1600.0,
        stop_ratio: float = 0.85,
    ) -> None:
        """
        Args:
            best_day_limit: Hard limit for single day profit ($1,600 for E8 $50k).
            stop_ratio: Safety margin ratio (0.85 = stop at 85% of limit).
        """
        self._best_day_limit = best_day_limit
        self._stop_ratio = stop_ratio
        self._realized_pnl = 0.0
        self._unrealized_pnl = 0.0

    @property
    def safe_limit(self) -> float:
        """Safety-adjusted daily profit limit."""
        return self._best_day_limit * self._stop_ratio

    @property
    def daily_pnl(self) -> float:
        """Total daily PnL (realized + unrealized)."""
        return self._realized_pnl + self._unrealized_pnl

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    @property
    def unrealized_pnl(self) -> float:
        return self._unrealized_pnl

    def record_trade_pnl(self, pnl: float) -> None:
        """Add realized PnL from a closed trade.

        Args:
            pnl: Profit/loss from the closed trade (positive = profit).
        """
        self._realized_pnl += pnl
        logger.debug(
            "BestDayTracker: recorded PnL ${:+.2f} (realized total: ${:+.2f})",
            pnl,
            self._realized_pnl,
        )

    def update_unrealized(self, unrealized_pnl: float) -> None:
        """Update floating (unrealized) PnL.

        Called periodically with the total unrealized PnL of all open positions.

        Args:
            unrealized_pnl: Current total floating PnL (replaces previous value).
        """
        self._unrealized_pnl = unrealized_pnl

    def can_take_more_profit(self, potential_profit: float) -> bool:
        """Check if taking additional profit would stay within the safe limit.

        Args:
            potential_profit: Estimated additional profit from a new trade.

        Returns:
            True if daily_pnl + potential_profit < safe_limit.
        """
        projected = self.daily_pnl + potential_profit
        return projected < self.safe_limit

    @property
    def remaining_capacity(self) -> float:
        """Dollars of profit remaining before hitting the safety limit.

        Returns 0 if already at or above the limit.
        """
        return max(0.0, self.safe_limit - self.daily_pnl)

    def should_close_winners(self) -> bool:
        """Check if we should proactively close winning positions.

        Returns True if daily PnL is approaching the limit and we should
        lock in profits by closing winners before the hard limit is hit.
        """
        # Close winners if we're at 90% of the safe limit
        aggressive_threshold = self.safe_limit * 0.90
        if self.daily_pnl >= aggressive_threshold:
            logger.warning(
                "BestDayTracker: daily PnL ${:.2f} approaching limit ${:.2f} — close winners!",
                self.daily_pnl,
                self.safe_limit,
            )
            return True
        return False

    def reset(self) -> None:
        """Reset for a new trading day. Called at day start."""
        prev_realized = self._realized_pnl
        self._realized_pnl = 0.0
        self._unrealized_pnl = 0.0
        logger.info(
            "BestDayTracker: reset (previous day realized: ${:+.2f})",
            prev_realized,
        )

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"BestDay: realized=${self._realized_pnl:+.2f}, "
            f"unrealized=${self._unrealized_pnl:+.2f}, "
            f"total=${self.daily_pnl:+.2f}, "
            f"remaining=${self.remaining_capacity:.2f} "
            f"(limit=${self._best_day_limit:.2f}, safety=${self.safe_limit:.2f})"
        )
