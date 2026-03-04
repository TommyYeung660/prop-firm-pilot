"""
Strategic decision cache — avoids redundant LLM calls when Qlib scores are static.

Production problem (2026-03-03): Volatility monitor triggers re-scan 15+ times/day,
each time re-running the full TradingAgents pipeline (~10 min) for the same
1D Qlib score that hasn't changed. Dache same symbol+direction decisions for 4H TTL.

Usage:
    cache = StrategicDecisionCache(ttl_seconds=14400)
    if cache.is_fresh("EURUSD", "SELL"):
        cached = cache.get_cached("EURUSD", "SELL")
        # Skip LLM, go directly to tactical validation
    else:
        decision = await agents.decide(...)
        cache.store("EURUSD", "SELL", decision_data)
"""

import time
from typing import Any

from loguru import logger


class StrategicDecisionCache:
    """In-memory cache for strategic LLM decisions with TTL.

    Keys are (symbol, direction) tuples. Each entry stores the full decision
    data dict and a timestamp. Entries expire after ttl_seconds.

    Thread-safe for single-writer patterns (scheduler runs in one async loop).

    Usage:
        cache = StrategicDecisionCache(ttl_seconds=14400)
        cache.store("EURUSD", "SELL", {"decision": "SELL", "risk_report": "..."})
        if cache.is_fresh("EURUSD", "SELL"):
            data = cache.get_cached("EURUSD", "SELL")
    """

    def __init__(self, ttl_seconds: int = 14400) -> None:
        self._ttl = ttl_seconds
        self._cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}

    def store(self, symbol: str, direction: str, data: dict[str, Any]) -> None:
        """Store a decision in the cache.

        Args:
            symbol: FX pair (e.g. "EURUSD").
            direction: "BUY" or "SELL".
            data: Full decision data dict to cache.
        """
        key = (symbol, direction)
        self._cache[key] = (time.monotonic(), data)
        logger.debug("Decision cache: stored {} {} (TTL={}s)", symbol, direction, self._ttl)

    def is_fresh(self, symbol: str, direction: str) -> bool:
        """Check if a cached decision exists and is within TTL.

        Args:
            symbol: FX pair.
            direction: "BUY" or "SELL".

        Returns:
            True if a non-expired entry exists for this symbol+direction.
        """
        key = (symbol, direction)
        entry = self._cache.get(key)
        if entry is None:
            return False
        stored_at, _ = entry
        return (time.monotonic() - stored_at) < self._ttl

    def get_cached(self, symbol: str, direction: str) -> dict[str, Any] | None:
        """Retrieve cached decision data if fresh.

        Args:
            symbol: FX pair.
            direction: "BUY" or "SELL".

        Returns:
            Cached data dict, or None if expired/missing.
        """
        if not self.is_fresh(symbol, direction):
            return None
        _, data = self._cache[(symbol, direction)]
        return data

    def invalidate(self, symbol: str) -> None:
        """Remove all cached entries for a symbol (both BUY and SELL).

        Args:
            symbol: FX pair to invalidate.
        """
        keys_to_remove = [k for k in self._cache if k[0] == symbol]
        for k in keys_to_remove:
            del self._cache[k]
        if keys_to_remove:
            logger.debug(
                "Decision cache: invalidated {} entries for {}",
                len(keys_to_remove),
                symbol,
            )

    def clear(self) -> None:
        """Remove all cached entries."""
        count = len(self._cache)
        self._cache.clear()
        logger.debug("Decision cache: cleared {} entries", count)
