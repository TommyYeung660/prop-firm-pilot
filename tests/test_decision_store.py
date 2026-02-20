"""
Tests for src/decision_store/sqlite_store.py — DecisionStore CRUD operations.

Uses in-memory SQLite (`:memory:`) via tmp_path for isolation between tests.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.decision.schemas import TradeIntent
from src.decision_store.sqlite_store import (
    DecisionStore,
    DecisionStoreError,
    InvalidTransitionError,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: object) -> DecisionStore:
    """Create a fresh DecisionStore with a temporary database."""
    db_path = f"{tmp_path}/test_decisions.db"
    s = DecisionStore(db_path=db_path)
    yield s  # type: ignore[misc]
    s.close()


@pytest.fixture
def sample_intent() -> TradeIntent:
    """Create a sample pending TradeIntent."""
    return TradeIntent(
        trade_date="2026-02-16",
        symbol="EURUSD",
        scanner_score=0.85,
        scanner_confidence="high",
        scanner_score_gap=0.12,
        scanner_drop_distance=0.05,
        scanner_topk_spread=0.03,
    )


# ── Insert Tests ────────────────────────────────────────────────────────────


class TestInsertIntent:
    """Tests for DecisionStore.insert_intent()."""

    def test_insert_and_retrieve(self, store: DecisionStore, sample_intent: TradeIntent) -> None:
        """Inserted intent should be retrievable by ID."""
        store.insert_intent(sample_intent)
        retrieved = store.get_intent(sample_intent.id)
        assert retrieved is not None
        assert retrieved.id == sample_intent.id
        assert retrieved.symbol == "EURUSD"
        assert retrieved.trade_date == "2026-02-16"
        assert retrieved.scanner_score == 0.85
        assert retrieved.status == "pending"

    def test_insert_creates_decision_record(
        self, store: DecisionStore, sample_intent: TradeIntent
    ) -> None:
        """Inserting an intent should also create a companion DecisionRecord."""
        store.insert_intent(sample_intent)
        decision = store.get_decision(sample_intent.id)
        assert decision is not None
        assert decision.intent_id == sample_intent.id
        assert decision.status == "pending"

    def test_duplicate_idempotency_key_rejected(self, store: DecisionStore) -> None:
        """Inserting two intents with the same idempotency_key should raise."""
        intent1 = TradeIntent(
            trade_date="2026-02-16", symbol="EURUSD", idempotency_key="same-key-123"
        )
        intent2 = TradeIntent(
            trade_date="2026-02-16", symbol="GBPUSD", idempotency_key="same-key-123"
        )
        store.insert_intent(intent1)
        with pytest.raises(DecisionStoreError, match="Duplicate"):
            store.insert_intent(intent2)


# ── Claim Tests ─────────────────────────────────────────────────────────────


class TestClaimNextPending:
    """Tests for DecisionStore.claim_next_pending()."""

    def test_claim_returns_oldest_pending(self, store: DecisionStore) -> None:
        """claim_next_pending should return the oldest pending intent."""
        older = TradeIntent(trade_date="2026-02-15", symbol="EURUSD")
        newer = TradeIntent(trade_date="2026-02-16", symbol="GBPUSD")
        store.insert_intent(older)
        store.insert_intent(newer)

        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None
        assert claimed.id == older.id
        assert claimed.status == "claimed"
        assert claimed.claim_worker_id == "llm-0"
        assert claimed.claim_ts is not None
        assert claimed.expires_at is not None

    def test_claim_returns_none_when_empty(self, store: DecisionStore) -> None:
        """claim_next_pending should return None when no pending intents exist."""
        result = store.claim_next_pending("llm-0")
        assert result is None

    def test_claimed_intent_not_reclaimed(self, store: DecisionStore) -> None:
        """A claimed intent should not be returned by claim_next_pending again."""
        intent = TradeIntent(trade_date="2026-02-16", symbol="EURUSD")
        store.insert_intent(intent)

        first = store.claim_next_pending("llm-0")
        assert first is not None

        second = store.claim_next_pending("llm-1")
        assert second is None  # No more pending

    def test_claim_sets_expires_at(self, store: DecisionStore) -> None:
        """Claiming should set expires_at based on claim_ttl_minutes."""
        intent = TradeIntent(trade_date="2026-02-16", symbol="EURUSD", claim_ttl_minutes=15)
        store.insert_intent(intent)

        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None
        assert claimed.expires_at is not None
        assert claimed.claim_ts is not None
        # expires_at should be ~15 minutes after claim_ts
        delta = claimed.expires_at - claimed.claim_ts
        assert abs(delta.total_seconds() - 15 * 60) < 2  # within 2 seconds tolerance


# ── Update Decision Tests ───────────────────────────────────────────────────


class TestUpdateIntentDecision:
    """Tests for DecisionStore.update_intent_decision()."""

    def test_update_decision_fields(self, store: DecisionStore) -> None:
        """update_intent_decision should write LLM decision to the intent."""
        intent = TradeIntent(trade_date="2026-02-16", symbol="EURUSD")
        store.insert_intent(intent)
        store.claim_next_pending("llm-0")

        store.update_intent_decision(
            intent_id=intent.id,
            side="BUY",
            sl_pips=40.0,
            tp_pips=80.0,
            risk_report="Low risk environment",
            state_json='{"analysts": {"market": "bullish"}}',
        )

        updated = store.get_intent(intent.id)
        assert updated is not None
        assert updated.suggested_side == "BUY"
        assert updated.suggested_sl_pips == 40.0
        assert updated.suggested_tp_pips == 80.0
        assert updated.agent_risk_report == "Low risk environment"
        assert "bullish" in updated.agent_state_json


# ── State Transition Tests ──────────────────────────────────────────────────


class TestStateTransitions:
    """Tests for intent status transitions through the full lifecycle."""

    def _create_and_claim(self, store: DecisionStore) -> TradeIntent:
        """Helper: create an intent and claim it."""
        intent = TradeIntent(trade_date="2026-02-16", symbol="EURUSD")
        store.insert_intent(intent)
        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None
        return claimed

    def test_happy_path_full_lifecycle(self, store: DecisionStore) -> None:
        """Test the full happy path: pending → claimed → ready → executing → opened → closed."""
        claimed = self._create_and_claim(store)
        intent_id = claimed.id

        # claimed → ready_for_exec
        store.update_intent_decision(intent_id, "BUY", 40.0, 80.0, "report", "{}")
        store.mark_ready_for_exec(intent_id)
        assert store.get_intent(intent_id) is not None
        assert store.get_intent(intent_id).status == "ready_for_exec"  # type: ignore[union-attr]

        # ready_for_exec → executing
        store.mark_executing(intent_id)
        assert store.get_intent(intent_id).status == "executing"  # type: ignore[union-attr]

        # executing → opened
        store.mark_opened(intent_id, position_id="POS-12345")
        opened = store.get_intent(intent_id)
        assert opened is not None
        assert opened.status == "opened"
        assert opened.position_id == "POS-12345"
        assert opened.executed_at is not None

        # opened → closed
        store.mark_closed(intent_id)
        assert store.get_intent(intent_id).status == "closed"  # type: ignore[union-attr]

    def test_cancel_from_pending(self, store: DecisionStore) -> None:
        """Pending intents should be cancellable."""
        intent = TradeIntent(trade_date="2026-02-16", symbol="EURUSD")
        store.insert_intent(intent)

        store.mark_cancelled(intent.id, "Scanner invalidated signal")
        assert store.get_intent(intent.id).status == "cancelled"  # type: ignore[union-attr]

    def test_cancel_from_claimed_hold(self, store: DecisionStore) -> None:
        """Claimed intents should be cancellable (LLM returned HOLD)."""
        claimed = self._create_and_claim(store)
        store.mark_cancelled(claimed.id, "LLM returned HOLD")
        assert store.get_intent(claimed.id).status == "cancelled"  # type: ignore[union-attr]

    def test_reject_from_executing(self, store: DecisionStore) -> None:
        """executing → rejected should work with reason."""
        claimed = self._create_and_claim(store)
        store.update_intent_decision(claimed.id, "BUY", 40.0, 80.0, "report", "{}")
        store.mark_ready_for_exec(claimed.id)
        store.mark_executing(claimed.id)

        store.mark_rejected(claimed.id, "Daily drawdown limit reached")
        rejected = store.get_intent(claimed.id)
        assert rejected is not None
        assert rejected.status == "rejected"
        assert rejected.execution_error == "Daily drawdown limit reached"

    def test_fail_from_executing(self, store: DecisionStore) -> None:
        """executing → failed should work with error."""
        claimed = self._create_and_claim(store)
        store.update_intent_decision(claimed.id, "BUY", 40.0, 80.0, "report", "{}")
        store.mark_ready_for_exec(claimed.id)
        store.mark_executing(claimed.id)

        store.mark_failed(claimed.id, "MatchTrader API timeout")
        failed = store.get_intent(claimed.id)
        assert failed is not None
        assert failed.status == "failed"
        assert failed.execution_error == "MatchTrader API timeout"

    def test_invalid_transition_raises(self, store: DecisionStore) -> None:
        """Invalid transitions should raise InvalidTransitionError."""
        intent = TradeIntent(trade_date="2026-02-16", symbol="EURUSD")
        store.insert_intent(intent)

        # pending → opened is invalid
        with pytest.raises(InvalidTransitionError):
            store.mark_opened(intent.id, "POS-123")

        # pending → executing is invalid
        with pytest.raises(InvalidTransitionError):
            store.mark_executing(intent.id)

    def test_cancel_from_ready_raises(self, store: DecisionStore) -> None:
        """Cannot cancel from ready_for_exec state."""
        claimed = self._create_and_claim(store)
        store.update_intent_decision(claimed.id, "BUY", 40.0, 80.0, "report", "{}")
        store.mark_ready_for_exec(claimed.id)

        with pytest.raises(InvalidTransitionError):
            store.mark_cancelled(claimed.id, "Too late to cancel")


# ── Query Tests ─────────────────────────────────────────────────────────────


class TestQueries:
    """Tests for DecisionStore query methods."""

    def test_get_pending_intents(self, store: DecisionStore) -> None:
        """get_pending_intents should return only pending intents."""
        store.insert_intent(TradeIntent(trade_date="2026-02-16", symbol="EURUSD"))
        store.insert_intent(TradeIntent(trade_date="2026-02-16", symbol="GBPUSD"))
        store.claim_next_pending("llm-0")  # Claims EURUSD

        pending = store.get_pending_intents()
        assert len(pending) == 1
        assert pending[0].symbol == "GBPUSD"

    def test_get_ready_intents(self, store: DecisionStore) -> None:
        """get_ready_intents should return only ready_for_exec intents."""
        intent = TradeIntent(trade_date="2026-02-16", symbol="EURUSD")
        store.insert_intent(intent)
        store.claim_next_pending("llm-0")
        store.update_intent_decision(intent.id, "BUY", 40.0, 80.0, "report", "{}")
        store.mark_ready_for_exec(intent.id)

        ready = store.get_ready_intents()
        assert len(ready) == 1
        assert ready[0].id == intent.id

    def test_get_active_positions(self, store: DecisionStore) -> None:
        """get_active_positions should return only opened intents."""
        intent = TradeIntent(trade_date="2026-02-16", symbol="EURUSD")
        store.insert_intent(intent)
        store.claim_next_pending("llm-0")
        store.update_intent_decision(intent.id, "BUY", 40.0, 80.0, "report", "{}")
        store.mark_ready_for_exec(intent.id)
        store.mark_executing(intent.id)
        store.mark_opened(intent.id, "POS-123")

        active = store.get_active_positions()
        assert len(active) == 1
        assert active[0].position_id == "POS-123"

    def test_get_intents_by_date(self, store: DecisionStore) -> None:
        """get_intents_by_date should filter by trade_date."""
        store.insert_intent(TradeIntent(trade_date="2026-02-15", symbol="EURUSD"))
        store.insert_intent(TradeIntent(trade_date="2026-02-16", symbol="GBPUSD"))
        store.insert_intent(TradeIntent(trade_date="2026-02-16", symbol="USDJPY"))

        feb16 = store.get_intents_by_date("2026-02-16")
        assert len(feb16) == 2
        symbols = {i.symbol for i in feb16}
        assert symbols == {"GBPUSD", "USDJPY"}

    def test_get_intent_not_found(self, store: DecisionStore) -> None:
        """get_intent should return None for non-existent ID."""
        assert store.get_intent("nonexistent") is None

    def test_get_decision_not_found(self, store: DecisionStore) -> None:
        """get_decision should return None for non-existent intent_id."""
        assert store.get_decision("nonexistent") is None


# ── Idempotency Tests ───────────────────────────────────────────────────────


class TestIdempotency:
    """Tests for DecisionStore.intent_exists()."""

    def test_intent_exists_true(self, store: DecisionStore) -> None:
        """intent_exists should return True when a matching intent exists."""
        store.insert_intent(TradeIntent(trade_date="2026-02-16", symbol="EURUSD", source="scanner"))
        assert store.intent_exists("EURUSD", "2026-02-16", "scanner") is True

    def test_intent_exists_false_different_symbol(self, store: DecisionStore) -> None:
        """intent_exists should return False for different symbol."""
        store.insert_intent(TradeIntent(trade_date="2026-02-16", symbol="EURUSD", source="scanner"))
        assert store.intent_exists("GBPUSD", "2026-02-16", "scanner") is False

    def test_intent_exists_false_different_date(self, store: DecisionStore) -> None:
        """intent_exists should return False for different date."""
        store.insert_intent(TradeIntent(trade_date="2026-02-16", symbol="EURUSD", source="scanner"))
        assert store.intent_exists("EURUSD", "2026-02-17", "scanner") is False

    def test_intent_exists_ignores_cancelled(self, store: DecisionStore) -> None:
        """Cancelled intents should not block new ones (intent_exists returns False)."""
        intent = TradeIntent(trade_date="2026-02-16", symbol="EURUSD", source="scanner")
        store.insert_intent(intent)
        store.mark_cancelled(intent.id, "Stale signal")

        assert store.intent_exists("EURUSD", "2026-02-16", "scanner") is False


# ── Claim Management Tests ──────────────────────────────────────────────────


class TestClaimManagement:
    """Tests for expired claim recycling and old intent cleanup."""

    def test_recycle_expired_claims(self, store: DecisionStore) -> None:
        """recycle_expired_claims should mark expired claimed intents as timed_out."""
        intent = TradeIntent(trade_date="2026-02-16", symbol="EURUSD", claim_ttl_minutes=0)
        store.insert_intent(intent)
        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None

        # Force expires_at to the past
        past = datetime.now(timezone.utc) - timedelta(minutes=1)
        store._conn.execute(
            "UPDATE intents SET expires_at = ? WHERE id = ?",
            (past.isoformat(), intent.id),
        )
        store._conn.commit()

        count = store.recycle_expired_claims()
        assert count == 1

        recycled = store.get_intent(intent.id)
        assert recycled is not None
        assert recycled.status == "timed_out"

    def test_cleanup_old_intents(self, store: DecisionStore) -> None:
        """cleanup_old_intents should delete terminal intents older than retention."""
        intent = TradeIntent(trade_date="2026-01-01", symbol="EURUSD")
        store.insert_intent(intent)
        store.mark_cancelled(intent.id, "Old test")

        # Force created_at to 30 days ago
        old_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        store._conn.execute(
            "UPDATE intents SET created_at = ? WHERE id = ?",
            (old_date, intent.id),
        )
        store._conn.commit()

        count = store.cleanup_old_intents(retention_days=7)
        assert count == 1
        assert store.get_intent(intent.id) is None

    def test_cleanup_does_not_delete_active(self, store: DecisionStore) -> None:
        """cleanup_old_intents should NOT delete pending/claimed/opened intents."""
        intent = TradeIntent(trade_date="2026-01-01", symbol="EURUSD")
        store.insert_intent(intent)

        # Force created_at to 30 days ago (but status is still 'pending')
        old_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        store._conn.execute(
            "UPDATE intents SET created_at = ? WHERE id = ?",
            (old_date, intent.id),
        )
        store._conn.commit()

        count = store.cleanup_old_intents(retention_days=7)
        assert count == 0
        assert store.get_intent(intent.id) is not None


# ── Decision Record Sync Tests ──────────────────────────────────────────────


class TestDecisionRecordSync:
    """Tests that decision records stay in sync with intent transitions."""

    def test_decision_tracks_claim(self, store: DecisionStore) -> None:
        """Decision record should update claimed_at when intent is claimed."""
        intent = TradeIntent(trade_date="2026-02-16", symbol="EURUSD")
        store.insert_intent(intent)
        store.claim_next_pending("llm-0")

        decision = store.get_decision(intent.id)
        assert decision is not None
        assert decision.status == "claimed"
        assert decision.claimed_at is not None

    def test_decision_tracks_opened(self, store: DecisionStore) -> None:
        """Decision record should track position_id and executed_at on open."""
        intent = TradeIntent(trade_date="2026-02-16", symbol="EURUSD")
        store.insert_intent(intent)
        store.claim_next_pending("llm-0")
        store.update_intent_decision(intent.id, "BUY", 40.0, 80.0, "report", "{}")
        store.mark_ready_for_exec(intent.id)
        store.mark_executing(intent.id)
        store.mark_opened(intent.id, "POS-999")

        decision = store.get_decision(intent.id)
        assert decision is not None
        assert decision.status == "opened"
        assert decision.position_id == "POS-999"
        assert decision.executed_at is not None

    def test_decision_tracks_closed(self, store: DecisionStore) -> None:
        """Decision record should set closed_at when intent is closed."""
        intent = TradeIntent(trade_date="2026-02-16", symbol="EURUSD")
        store.insert_intent(intent)
        store.claim_next_pending("llm-0")
        store.update_intent_decision(intent.id, "BUY", 40.0, 80.0, "report", "{}")
        store.mark_ready_for_exec(intent.id)
        store.mark_executing(intent.id)
        store.mark_opened(intent.id, "POS-999")
        store.mark_closed(intent.id)

        decision = store.get_decision(intent.id)
        assert decision is not None
        assert decision.status == "closed"
        assert decision.closed_at is not None

    def test_decision_tracks_rejection(self, store: DecisionStore) -> None:
        """Decision record should track failure_reason on rejection."""
        intent = TradeIntent(trade_date="2026-02-16", symbol="EURUSD")
        store.insert_intent(intent)
        store.claim_next_pending("llm-0")
        store.update_intent_decision(intent.id, "BUY", 40.0, 80.0, "report", "{}")
        store.mark_ready_for_exec(intent.id)
        store.mark_executing(intent.id)
        store.mark_rejected(intent.id, "Drawdown limit")

        decision = store.get_decision(intent.id)
        assert decision is not None
        assert decision.status == "rejected"
        assert decision.failure_reason == "Drawdown limit"


# ── API Call Tracking Tests ─────────────────────────────────────────────────


class TestApiCallTracking:
    """Tests for DecisionStore rate limiter persistence."""

    def test_record_api_calls_returns_total(self, store: DecisionStore) -> None:
        """Should return cumulative call count for today."""
        total = store.record_api_calls(5)
        assert total == 5
        total = store.record_api_calls(3)
        assert total == 8

    def test_record_api_calls_default_one(self, store: DecisionStore) -> None:
        """Should default to recording 1 call."""
        total = store.record_api_calls()
        assert total == 1
        total = store.record_api_calls()
        assert total == 2

    def test_get_api_call_count_today(self, store: DecisionStore) -> None:
        """Should return count for today."""
        store.record_api_calls(10)
        assert store.get_api_call_count() == 10

    def test_get_api_call_count_specific_date(self, store: DecisionStore) -> None:
        """Should return 0 for dates with no recorded calls."""
        assert store.get_api_call_count("2020-01-01") == 0

    def test_get_api_call_count_no_data(self, store: DecisionStore) -> None:
        """Should return 0 when no calls have been recorded."""
        assert store.get_api_call_count() == 0


# ── Dashboard Query Helper Tests ────────────────────────────────────────────


class TestDashboardQueries:
    """Tests for monitoring convenience methods."""

    def test_get_daily_summary(self, store: DecisionStore) -> None:
        """Should return status counts for the given date."""
        # Insert 3 intents in various states
        for symbol in ("EURUSD", "GBPUSD", "USDJPY"):
            intent = TradeIntent(trade_date="2026-02-16", symbol=symbol)
            store.insert_intent(intent)

        # Claim one
        store.claim_next_pending("llm-0")

        summary = store.get_daily_summary("2026-02-16")
        assert summary["pending"] == 2
        assert summary["claimed"] == 1
        assert sum(summary.values()) == 3

    def test_get_daily_summary_defaults_to_today(self, store: DecisionStore) -> None:
        """Should default to today's date."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        intent = TradeIntent(trade_date=today, symbol="EURUSD")
        store.insert_intent(intent)

        summary = store.get_daily_summary()
        assert "pending" in summary

    def test_get_daily_summary_empty(self, store: DecisionStore) -> None:
        """Should return empty dict for dates with no intents."""
        summary = store.get_daily_summary("2020-01-01")
        assert summary == {}

    def test_get_success_rate(self, store: DecisionStore) -> None:
        """Should calculate success rate over terminal intents."""
        # Create 3 intents: 2 opened, 1 rejected
        for symbol in ("EURUSD", "GBPUSD", "USDJPY"):
            intent = TradeIntent(trade_date="2026-02-16", symbol=symbol)
            store.insert_intent(intent)
            store.claim_next_pending("llm-0")
            store.update_intent_decision(intent.id, "BUY", 40.0, 80.0, "report", "{}")
            store.mark_ready_for_exec(intent.id)
            store.mark_executing(intent.id)
            if symbol == "USDJPY":
                store.mark_rejected(intent.id, "Drawdown limit")
            else:
                store.mark_opened(intent.id, f"POS-{symbol}")

        stats = store.get_success_rate(days=30)
        assert stats["total"] == 3
        assert stats["success"] == 2
        assert stats["failure"] == 1
        assert abs(stats["success_rate"] - 0.6667) < 0.001

    def test_get_success_rate_no_data(self, store: DecisionStore) -> None:
        """Should return 0 rate when no terminal intents exist."""
        stats = store.get_success_rate()
        assert stats["total"] == 0
        assert stats["success_rate"] == 0.0

    def test_get_symbol_stats(self, store: DecisionStore) -> None:
        """Should return per-symbol statistics."""
        for symbol in ("EURUSD", "EURUSD", "GBPUSD"):
            intent = TradeIntent(trade_date="2026-02-16", symbol=symbol)
            store.insert_intent(intent)

        stats = store.get_symbol_stats(days=30)
        assert len(stats) == 2
        eur = next(s for s in stats if s["symbol"] == "EURUSD")
        assert eur["total"] == 2

    def test_get_pipeline_status(self, store: DecisionStore) -> None:
        """Should return count of active intents by status."""
        intent1 = TradeIntent(trade_date="2026-02-16", symbol="EURUSD")
        intent2 = TradeIntent(trade_date="2026-02-16", symbol="GBPUSD")
        store.insert_intent(intent1)
        store.insert_intent(intent2)
        store.claim_next_pending("llm-0")

        status = store.get_pipeline_status()
        assert status["pending"] == 1
        assert status["claimed"] == 1
        assert status["total_active"] == 2

    def test_get_pipeline_status_empty(self, store: DecisionStore) -> None:
        """Should return total_active=0 when no active intents."""
        status = store.get_pipeline_status()
        assert status["total_active"] == 0
