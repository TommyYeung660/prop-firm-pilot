"""
Tests for DecisionStore.intent_exists() — intent deduplication fix.

Verifies that only *in-progress* intents (pending, claimed, ready_for_exec)
block new intent creation.  Completed (opened, closed) and terminal
(cancelled, timed_out, failed) intents must NOT block.

This regression suite guards the fix where the old query used
``status NOT IN ('cancelled', 'timed_out', 'failed')`` which incorrectly
treated opened/closed intents as blocking.
"""

import pytest

from src.decision.schemas import TradeIntent
from src.decision_store.sqlite_store import DecisionStore

TRADE_DATE = "2026-02-23"
SYMBOL = "EURUSD"
SOURCE = "scanner"


@pytest.fixture
def store(tmp_path) -> DecisionStore:
    db_path = f"{tmp_path}/test_intent.db"
    s = DecisionStore(db_path=db_path)
    yield s  # type: ignore[misc]
    s.close()


def _insert_intent_with_status(store: DecisionStore, status: str) -> TradeIntent:
    """Insert a TradeIntent then force its status via raw SQL."""
    intent = TradeIntent(
        trade_date=TRADE_DATE,
        symbol=SYMBOL,
        scanner_score=0.85,
        scanner_confidence="high",
        source=SOURCE,
    )
    store.insert_intent(intent)
    if status != "pending":  # pending is the default
        store._conn.execute(
            "UPDATE intents SET status = ? WHERE id = ?",
            (status, intent.id),
        )
        store._conn.commit()
    return intent


# ── In-progress statuses SHOULD block ──────────────────────────────────────


class TestIntentExistsBlocking:
    """In-progress intents must block new intent creation."""

    def test_pending_blocks(self, store: DecisionStore) -> None:
        _insert_intent_with_status(store, "pending")
        assert store.intent_exists(SYMBOL, TRADE_DATE, SOURCE) is True

    def test_claimed_blocks(self, store: DecisionStore) -> None:
        _insert_intent_with_status(store, "claimed")
        assert store.intent_exists(SYMBOL, TRADE_DATE, SOURCE) is True

    def test_ready_for_exec_blocks(self, store: DecisionStore) -> None:
        _insert_intent_with_status(store, "ready_for_exec")
        assert store.intent_exists(SYMBOL, TRADE_DATE, SOURCE) is True


# ── Non-blocking statuses SHOULD NOT block (the bug fix) ───────────────────


class TestIntentExistsNonBlocking:
    """Completed and terminal intents must NOT block new intent creation."""

    @pytest.mark.parametrize(
        "status",
        ["opened", "closed", "cancelled", "timed_out", "failed"],
        ids=["opened", "closed", "cancelled", "timed_out", "failed"],
    )
    def test_terminal_status_does_not_block(self, store: DecisionStore, status: str) -> None:
        """Intents in terminal/completed states should not prevent new intents."""
        _insert_intent_with_status(store, status)
        assert store.intent_exists(SYMBOL, TRADE_DATE, SOURCE) is False


# ── Edge cases ─────────────────────────────────────────────────────────────


class TestIntentExistsEdgeCases:
    """Edge cases for intent_exists."""

    def test_no_intent_returns_false(self, store: DecisionStore) -> None:
        """Empty store should return False."""
        assert store.intent_exists(SYMBOL, TRADE_DATE, SOURCE) is False

    def test_different_symbol_not_blocked(self, store: DecisionStore) -> None:
        """A pending intent for EURUSD should not block GBPUSD."""
        _insert_intent_with_status(store, "pending")
        assert store.intent_exists("GBPUSD", TRADE_DATE, SOURCE) is False

    def test_different_date_not_blocked(self, store: DecisionStore) -> None:
        """A pending intent for a different date should not block."""
        _insert_intent_with_status(store, "pending")
        assert store.intent_exists(SYMBOL, "2026-02-24", SOURCE) is False

    def test_different_source_not_blocked(self, store: DecisionStore) -> None:
        """A pending intent from 'manual' should not block 'scanner'."""
        intent = TradeIntent(
            trade_date=TRADE_DATE,
            symbol=SYMBOL,
            scanner_score=0.85,
            scanner_confidence="high",
            source="manual",
        )
        store.insert_intent(intent)
        assert store.intent_exists(SYMBOL, TRADE_DATE, "scanner") is False

    def test_closed_then_new_pending_blocks(self, store: DecisionStore) -> None:
        """After a closed intent, a fresh pending intent should block."""
        _insert_intent_with_status(store, "closed")
        _insert_intent_with_status(store, "pending")
        assert store.intent_exists(SYMBOL, TRADE_DATE, SOURCE) is True
