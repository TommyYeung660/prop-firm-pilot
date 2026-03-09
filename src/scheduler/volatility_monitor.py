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
        self._quotes: dict[str, deque[tuple[datetime, float]]] = {sym: deque() for sym in symbols}
        self._last_trigger_time: datetime | None = None
        self._last_trigger_per_symbol: dict[str, datetime] = {}
        self._global_min_interval_seconds: int = 900  # 15 min between ANY trigger

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
        # Global minimum interval check (prevents ANY trigger within 15 min)
        if self._last_trigger_time is not None:
            global_elapsed = (now - self._last_trigger_time).total_seconds()
            if global_elapsed < self._global_min_interval_seconds:
                return False, "", 0.0

        best_pct = 0.0
        best_symbol = ""

        for symbol in self._symbols:
            # Per-symbol cooldown check
            if symbol in self._last_trigger_per_symbol:
                sym_elapsed = (now - self._last_trigger_per_symbol[symbol]).total_seconds()
                if sym_elapsed < self._config.volatility_cooldown_seconds:
                    continue
            pct = self._calculate_price_change_pct(symbol, now)
            if abs(pct) > abs(best_pct):
                best_pct = pct
                best_symbol = symbol

        if abs(best_pct) >= self._config.volatility_threshold_pct:
            self._last_trigger_time = now
            self._last_trigger_per_symbol[best_symbol] = now
            logger.info(
                "Volatility trigger: {} moved {:.2f}% in {}min window",
                best_symbol,
                best_pct,
                self._config.volatility_window_minutes,
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
        self._last_trigger_per_symbol.clear()
