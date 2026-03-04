"""
Tests for StrategicDecisionCache — LLM decision caching to avoid
redundant 10-minute TradingAgents runs when Qlib score is static intraday.
"""

import time

from src.scheduler.decision_cache import StrategicDecisionCache


class TestCacheStore:
    """Basic store and retrieve operations."""

    def test_store_and_retrieve(self) -> None:
        cache = StrategicDecisionCache(ttl_seconds=3600)
        cache.store("EURUSD", "SELL", {"decision": "SELL", "risk_report": "..."})
        assert cache.is_fresh("EURUSD", "SELL")

    def test_miss_on_empty_cache(self) -> None:
        cache = StrategicDecisionCache(ttl_seconds=3600)
        assert not cache.is_fresh("EURUSD", "SELL")

    def test_miss_on_different_direction(self) -> None:
        cache = StrategicDecisionCache(ttl_seconds=3600)
        cache.store("EURUSD", "SELL", {"decision": "SELL"})
        assert not cache.is_fresh("EURUSD", "BUY")

    def test_miss_on_different_symbol(self) -> None:
        cache = StrategicDecisionCache(ttl_seconds=3600)
        cache.store("EURUSD", "SELL", {"decision": "SELL"})
        assert not cache.is_fresh("GBPUSD", "SELL")

    def test_get_cached_returns_data(self) -> None:
        cache = StrategicDecisionCache(ttl_seconds=3600)
        data = {"decision": "SELL", "risk_report": "test"}
        cache.store("EURUSD", "SELL", data)
        result = cache.get_cached("EURUSD", "SELL")
        assert result is not None
        assert result["decision"] == "SELL"

    def test_get_cached_returns_none_on_miss(self) -> None:
        cache = StrategicDecisionCache(ttl_seconds=3600)
        assert cache.get_cached("EURUSD", "SELL") is None


class TestCacheTTL:
    """TTL expiration behavior."""

    def test_expired_entry_not_fresh(self) -> None:
        cache = StrategicDecisionCache(ttl_seconds=0)  # Instant expiry
        cache.store("EURUSD", "SELL", {"decision": "SELL"})
        time.sleep(0.01)
        assert not cache.is_fresh("EURUSD", "SELL")

    def test_fresh_entry_within_ttl(self) -> None:
        cache = StrategicDecisionCache(ttl_seconds=3600)
        cache.store("EURUSD", "SELL", {"decision": "SELL"})
        assert cache.is_fresh("EURUSD", "SELL")


class TestCacheInvalidation:
    """Manual cache invalidation."""

    def test_invalidate_symbol(self) -> None:
        cache = StrategicDecisionCache(ttl_seconds=3600)
        cache.store("EURUSD", "SELL", {"decision": "SELL"})
        cache.invalidate("EURUSD")
        assert not cache.is_fresh("EURUSD", "SELL")

    def test_invalidate_does_not_affect_other_symbols(self) -> None:
        cache = StrategicDecisionCache(ttl_seconds=3600)
        cache.store("EURUSD", "SELL", {"decision": "SELL"})
        cache.store("GBPUSD", "BUY", {"decision": "BUY"})
        cache.invalidate("EURUSD")
        assert cache.is_fresh("GBPUSD", "BUY")

    def test_clear_all(self) -> None:
        cache = StrategicDecisionCache(ttl_seconds=3600)
        cache.store("EURUSD", "SELL", {"decision": "SELL"})
        cache.store("GBPUSD", "BUY", {"decision": "BUY"})
        cache.clear()
        assert not cache.is_fresh("EURUSD", "SELL")
        assert not cache.is_fresh("GBPUSD", "BUY")
