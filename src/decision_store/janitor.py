"""
Janitor — periodic cleanup of expired claims and old intents.

Runs as part of the Scheduler's async worker pool, recycling stale
claimed intents and purging terminal intents beyond retention period.

Usage:
    janitor = Janitor(store, retention_days=7)
    recycled, cleaned = janitor.run_cycle()
"""

from loguru import logger

from src.decision_store.sqlite_store import DecisionStore


class Janitor:
    """Cleans up expired claims and old terminal intents.

    Usage:
        janitor = Janitor(store, retention_days=7)
        recycled, cleaned = janitor.run_cycle()
    """

    def __init__(self, store: DecisionStore, retention_days: int = 7) -> None:
        self._store = store
        self._retention_days = retention_days

    def run_cycle(self) -> tuple[int, int]:
        """Execute one janitor cleanup cycle.

        1. Recycle expired claims (claimed → timed_out).
        2. Delete old terminal intents beyond retention period.

        Returns:
            Tuple of (recycled_count, cleaned_count).
        """
        recycled = self._store.recycle_expired_claims()
        if recycled > 0:
            logger.info("Janitor: recycled {} expired claims", recycled)

        cleaned = self._store.cleanup_old_intents(self._retention_days)
        if cleaned > 0:
            logger.info(
                "Janitor: cleaned {} old intents (retention={}d)", cleaned, self._retention_days
            )

        return recycled, cleaned
