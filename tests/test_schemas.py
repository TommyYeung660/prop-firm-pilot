"""
Tests for src/decision/schemas.py — TradeIntent and DecisionRecord models.
"""

from datetime import datetime, timezone

from src.decision.schemas import (
    VALID_TRANSITIONS,
    DecisionRecord,
    IntentStatus,
    TradeIntent,
)

# ── TradeIntent Tests ───────────────────────────────────────────────────────


class TestTradeIntent:
    """Tests for TradeIntent Pydantic model."""

    def test_default_values(self) -> None:
        """TradeIntent should have sensible defaults for all optional fields."""
        intent = TradeIntent(trade_date="2026-02-16", symbol="EURUSD")
        assert intent.trade_date == "2026-02-16"
        assert intent.symbol == "EURUSD"
        assert intent.status == "pending"
        assert intent.source == "scanner"
        assert intent.scanner_score == 0.0
        assert intent.scanner_confidence == "medium"
        assert intent.suggested_side is None
        assert intent.claim_worker_id is None
        assert intent.claim_ts is None
        assert intent.position_id is None
        assert intent.executed_at is None
        assert intent.execution_error is None
        assert intent.claim_ttl_minutes == 30

    def test_auto_generated_fields(self) -> None:
        """id, created_at, and idempotency_key should be auto-generated."""
        intent = TradeIntent(trade_date="2026-02-16", symbol="EURUSD")
        assert len(intent.id) == 16
        assert len(intent.idempotency_key) == 12
        assert isinstance(intent.created_at, datetime)
        assert intent.created_at.tzinfo == timezone.utc

    def test_unique_ids(self) -> None:
        """Each TradeIntent should get unique id and idempotency_key."""
        a = TradeIntent(trade_date="2026-02-16", symbol="EURUSD")
        b = TradeIntent(trade_date="2026-02-16", symbol="EURUSD")
        assert a.id != b.id
        assert a.idempotency_key != b.idempotency_key

    def test_scanner_fields(self) -> None:
        """Scanner-related fields should accept provided values."""
        intent = TradeIntent(
            trade_date="2026-02-16",
            symbol="XAUUSD",
            scanner_score=0.95,
            scanner_confidence="high",
            scanner_score_gap=0.12,
            scanner_drop_distance=0.05,
            scanner_topk_spread=0.03,
        )
        assert intent.scanner_score == 0.95
        assert intent.scanner_confidence == "high"
        assert intent.scanner_score_gap == 0.12
        assert intent.scanner_drop_distance == 0.05
        assert intent.scanner_topk_spread == 0.03

    def test_llm_decision_fields(self) -> None:
        """LLM decision fields should be settable."""
        intent = TradeIntent(
            trade_date="2026-02-16",
            symbol="GBPUSD",
            suggested_side="BUY",
            suggested_sl_pips=50.0,
            suggested_tp_pips=100.0,
            agent_risk_report="Low risk",
            agent_state_json='{"final": "state"}',
        )
        assert intent.suggested_side == "BUY"
        assert intent.suggested_sl_pips == 50.0
        assert intent.suggested_tp_pips == 100.0
        assert intent.agent_risk_report == "Low risk"
        assert intent.agent_state_json == '{"final": "state"}'

    def test_can_transition_to_valid(self) -> None:
        """Valid transitions should return True."""
        intent = TradeIntent(trade_date="2026-02-16", symbol="EURUSD", status="pending")
        assert intent.can_transition_to("claimed") is True
        assert intent.can_transition_to("cancelled") is True

    def test_can_transition_to_invalid(self) -> None:
        """Invalid transitions should return False."""
        intent = TradeIntent(trade_date="2026-02-16", symbol="EURUSD", status="pending")
        assert intent.can_transition_to("opened") is False
        assert intent.can_transition_to("closed") is False
        assert intent.can_transition_to("executing") is False

    def test_terminal_states_have_no_transitions(self) -> None:
        """Terminal states (rejected, failed, cancelled, closed) should have no exits."""
        for terminal in ("rejected", "failed", "cancelled", "closed"):
            intent = TradeIntent(
                trade_date="2026-02-16",
                symbol="EURUSD",
                status=terminal,  # type: ignore[arg-type]
            )
            # Should not be able to transition anywhere
            for target in (
                "pending",
                "claimed",
                "ready_for_exec",
                "executing",
                "opened",
                "rejected",
                "failed",
                "cancelled",
                "closed",
            ):
                assert intent.can_transition_to(target) is False  # type: ignore[arg-type]


# ── VALID_TRANSITIONS Tests ─────────────────────────────────────────────────


class TestValidTransitions:
    """Tests for the state machine transition map."""

    def test_all_statuses_have_entries(self) -> None:
        """Every IntentStatus value should have an entry in VALID_TRANSITIONS."""
        all_statuses: list[IntentStatus] = [
            "pending",
            "claimed",
            "ready_for_exec",
            "executing",
            "opened",
            "rejected",
            "failed",
            "cancelled",
            "timed_out",
            "closed",
        ]
        for status in all_statuses:
            assert status in VALID_TRANSITIONS, f"Missing transition entry for {status}"

    def test_happy_path_chain(self) -> None:
        """Happy path: pending→claimed→ready→executing→opened→closed."""
        chain = [
            ("pending", "claimed"),
            ("claimed", "ready_for_exec"),
            ("ready_for_exec", "executing"),
            ("executing", "opened"),
            ("opened", "closed"),
        ]
        for from_s, to_s in chain:
            assert to_s in VALID_TRANSITIONS[from_s], f"{from_s} → {to_s} should be valid"

    def test_timed_out_can_recycle(self) -> None:
        """timed_out intents should be recyclable to pending."""
        assert "pending" in VALID_TRANSITIONS["timed_out"]


# ── DecisionRecord Tests ───────────────────────────────────────────────────


class TestDecisionRecord:
    """Tests for DecisionRecord Pydantic model."""

    def test_default_values(self) -> None:
        """DecisionRecord should have sensible defaults."""
        record = DecisionRecord(intent_id="abc123")
        assert record.intent_id == "abc123"
        assert isinstance(record.created_at, datetime)
        assert record.claimed_at is None
        assert record.decided_at is None
        assert record.executed_at is None
        assert record.closed_at is None
        assert record.status == ""
        assert record.order_id is None
        assert record.position_id is None
        assert record.failure_reason == ""
        assert record.compliance_snapshot == ""
        assert record.execution_meta == ""

    def test_all_fields_settable(self) -> None:
        """All DecisionRecord fields should accept explicit values."""
        now = datetime.now(timezone.utc)
        record = DecisionRecord(
            intent_id="xyz789",
            created_at=now,
            claimed_at=now,
            decided_at=now,
            executed_at=now,
            closed_at=now,
            status="closed",
            order_id="order-1",
            position_id="pos-1",
            failure_reason="",
            compliance_snapshot='{"checks": []}',
            execution_meta='{"entry_price": 1.0850}',
        )
        assert record.intent_id == "xyz789"
        assert record.status == "closed"
        assert record.order_id == "order-1"
        assert record.position_id == "pos-1"
