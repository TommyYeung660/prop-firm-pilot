"""
Tests for compliance rejection cooldown — prevents infinite retry loops.

Guards against C3 production bug: scanner creates new intent → LLM evaluates →
compliance rejects (Best Day) → scanner creates new intent → infinite loop
burning LLM tokens for hours.

The fix adds has_recent_rejection() to DecisionStore, which the scanner loop
checks before creating new intents.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.decision.schemas import TradeIntent
from src.decision_store.sqlite_store import DecisionStore

TRADE_DATE = "2026-03-03"
SYMBOL = "EURUSD"
SOURCE = "scanner"


@pytest.fixture
def store(tmp_path) -> DecisionStore:
    db_path = f"{tmp_path}/test_cooldown.db"
    s = DecisionStore(db_path=db_path)
    yield s  # type: ignore[misc]
    s.close()


def _create_rejected_intent(
    store: DecisionStore,
    symbol: str = SYMBOL,
    trade_date: str = TRADE_DATE,
    rejected_ago_minutes: int = 0,
) -> TradeIntent:
    """Insert an intent and force it to rejected status with a specific executed_at time."""
    intent = TradeIntent(
        trade_date=trade_date,
        symbol=symbol,
        scanner_score=0.85,
        scanner_confidence="high",
        source=SOURCE,
    )
    store.insert_intent(intent)
    rejected_at = datetime.now(timezone.utc) - timedelta(minutes=rejected_ago_minutes)
    store._conn.execute(
        "UPDATE intents SET status = 'rejected', executed_at = ? WHERE id = ?",
        (rejected_at.isoformat(), intent.id),
    )
    store._conn.commit()
    return intent


# ── has_recent_rejection basic behavior ────────────────────────────────────


class TestHasRecentRejection:
    """has_recent_rejection() returns True when a rejected intent exists
    within the cooldown window."""

    def test_recently_rejected_blocks(self, store: DecisionStore) -> None:
        """Intent rejected 5 minutes ago with 60-min cooldown → blocked."""
        _create_rejected_intent(store, rejected_ago_minutes=5)
        assert store.has_recent_rejection(SYMBOL, TRADE_DATE, cooldown_minutes=60) is True

    def test_old_rejection_does_not_block(self, store: DecisionStore) -> None:
        """Intent rejected 120 minutes ago with 60-min cooldown → not blocked."""
        _create_rejected_intent(store, rejected_ago_minutes=120)
        assert store.has_recent_rejection(SYMBOL, TRADE_DATE, cooldown_minutes=60) is False

    def test_no_rejection_does_not_block(self, store: DecisionStore) -> None:
        """No rejected intents → not blocked."""
        assert store.has_recent_rejection(SYMBOL, TRADE_DATE, cooldown_minutes=60) is False

    def test_different_symbol_not_blocked(self, store: DecisionStore) -> None:
        """Rejected EURUSD should not block GBPUSD."""
        _create_rejected_intent(store, symbol="EURUSD", rejected_ago_minutes=5)
        assert store.has_recent_rejection("GBPUSD", TRADE_DATE, cooldown_minutes=60) is False

    def test_different_date_not_blocked(self, store: DecisionStore) -> None:
        """Rejected intent for a different date should not block."""
        _create_rejected_intent(store, trade_date="2026-03-02", rejected_ago_minutes=5)
        assert store.has_recent_rejection(SYMBOL, TRADE_DATE, cooldown_minutes=60) is False

    def test_non_rejected_statuses_not_counted(self, store: DecisionStore) -> None:
        """Only 'rejected' status should trigger cooldown, not cancelled/failed."""
        intent = TradeIntent(
            trade_date=TRADE_DATE,
            symbol=SYMBOL,
            scanner_score=0.85,
            scanner_confidence="high",
            source=SOURCE,
        )
        store.insert_intent(intent)
        now_str = datetime.now(timezone.utc).isoformat()
        store._conn.execute(
            "UPDATE intents SET status = 'cancelled', executed_at = ? WHERE id = ?",
            (now_str, intent.id),
        )
        store._conn.commit()
        assert store.has_recent_rejection(SYMBOL, TRADE_DATE, cooldown_minutes=60) is False

    def test_zero_cooldown_always_allows(self, store: DecisionStore) -> None:
        """With cooldown_minutes=0, even recent rejections don't block."""
        _create_rejected_intent(store, rejected_ago_minutes=1)
        assert store.has_recent_rejection(SYMBOL, TRADE_DATE, cooldown_minutes=0) is False

    def test_edge_of_cooldown_window(self, store: DecisionStore) -> None:
        """Intent rejected exactly at cooldown boundary → not blocked (exclusive)."""
        _create_rejected_intent(store, rejected_ago_minutes=60)
        assert store.has_recent_rejection(SYMBOL, TRADE_DATE, cooldown_minutes=60) is False
