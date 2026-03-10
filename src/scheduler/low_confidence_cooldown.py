"""
Low-confidence scanner cooldown tracker.

Tracks consecutive low-confidence LLM pre-filter cancellations per symbol.
When a symbol reaches the configured threshold, it enters a cooldown period
during which no new intents will be created for it.

Usage:
    tracker = LowConfidenceCooldown(cooldown_minutes=240, threshold=2)
    tracker.record_low_confidence("AUDUSD", now)
    if tracker.is_cooled_down("AUDUSD", now):
        # skip this symbol
"""

from datetime import datetime, timedelta


class LowConfidenceCooldown:
    """In-memory tracker for consecutive low-confidence cancellations per symbol.

    Usage:
        tracker = LowConfidenceCooldown(cooldown_minutes=240, threshold=2)
        tracker.record_low_confidence("AUDUSD", now)
        if tracker.is_cooled_down("AUDUSD", now):
            # skip creating intent for AUDUSD
    """

    def __init__(self, cooldown_minutes: int = 240, threshold: int = 2) -> None:
        self._cooldown_minutes = cooldown_minutes
        self._threshold = threshold
        self._counts: dict[str, int] = {}
        self._cooldown_until: dict[str, datetime] = {}

    def record_low_confidence(self, symbol: str, now: datetime) -> None:
        """Record a low-confidence cancellation for a symbol."""
        self._counts[symbol] = self._counts.get(symbol, 0) + 1
        if self._counts[symbol] >= self._threshold:
            self._cooldown_until[symbol] = now + timedelta(minutes=self._cooldown_minutes)

    def is_cooled_down(self, symbol: str, now: datetime) -> bool:
        """Check if a symbol is currently in cooldown."""
        if symbol not in self._cooldown_until:
            return False
        if now >= self._cooldown_until[symbol]:
            # Cooldown expired — clean up
            self._cooldown_until.pop(symbol, None)
            self._counts.pop(symbol, None)
            return False
        return True

    def reset_symbol(self, symbol: str) -> None:
        """Reset counter and cooldown for a symbol (e.g. after successful trade)."""
        self._counts.pop(symbol, None)
        self._cooldown_until.pop(symbol, None)

    def reset_all(self) -> None:
        """Reset all counters and cooldowns (e.g. at day boundary)."""
        self._counts.clear()
        self._cooldown_until.clear()

    def get_count(self, symbol: str) -> int:
        """Get current consecutive low-confidence count for a symbol."""
        return self._counts.get(symbol, 0)
