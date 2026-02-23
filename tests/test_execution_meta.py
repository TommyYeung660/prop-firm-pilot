"""Tests for execution_meta (Phase 2.7) — _build_execution_meta() and update_execution_meta()."""

import json

import pytest

from src.decision.schemas import TradeIntent
from src.decision_store.sqlite_store import DecisionStore
from src.execution.engine import ExecutionEngine

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    """Create a temporary DecisionStore for testing."""
    db_path = f"{tmp_path}/test.db"
    s = DecisionStore(db_path)
    yield s
    s.close()


def _advance_to_opened(store, intent):
    """Helper to advance an intent through the state machine to 'opened'."""
    store.insert_intent(intent)
    store.claim_next_pending("test-worker")
    store.update_intent_decision(intent.id, "BUY", 50.0, 100.0, "test", "{}")
    store.mark_ready_for_exec(intent.id)
    store.mark_executing(intent.id)
    store.mark_opened(intent.id, "POS-123")


# ── Test _build_execution_meta ──────────────────────────────────────────────


class TestBuildExecutionMeta:
    """Tests for ExecutionEngine._build_execution_meta() static method."""

    def test_all_fields_populated(self) -> None:
        """Test with all values populated (happy path)."""
        result = ExecutionEngine._build_execution_meta(
            fill_price=1.0856,
            volume=0.1,
            side="BUY",
            sl_price=1.0806,
            tp_price=1.0956,
            sl_pips=50.0,
            tp_pips=100.0,
            pre_trade_bid=1.08555,
            pre_trade_ask=1.08565,
            slippage_pips=0.5,
            execution_latency_ms=125.5,
            random_delay_seconds=0.75,
            compliance_passed=True,
            order_raw_response={"positionId": "POS-123", "openPrice": 1.0856},
        )

        # Verify it's a valid JSON string
        assert isinstance(result, str)

        # Parse and verify all expected fields are present
        data = json.loads(result)
        assert data["fill_price"] == 1.0856
        assert data["volume"] == 0.1
        assert data["side"] == "BUY"
        assert data["sl_price"] == 1.0806
        assert data["tp_price"] == 1.0956
        assert data["sl_pips"] == 50.0
        assert data["tp_pips"] == 100.0
        assert data["pre_trade_bid"] == 1.08555
        assert data["pre_trade_ask"] == 1.08565
        assert data["slippage_pips"] == 0.5
        assert data["execution_latency_ms"] == 125.5
        assert data["random_delay_seconds"] == 0.75
        assert data["compliance_passed"] is True
        assert data["order_raw_response"] == {
            "positionId": "POS-123",
            "openPrice": 1.0856,
        }

    def test_optional_fields_none(self) -> None:
        """Test with None values for optional fields."""
        result = ExecutionEngine._build_execution_meta(
            fill_price=None,
            volume=0.15,
            side="SELL",
            sl_price=1.0900,
            tp_price=1.0800,
            sl_pips=40.0,
            tp_pips=80.0,
            pre_trade_bid=None,
            pre_trade_ask=None,
            slippage_pips=None,
            execution_latency_ms=None,
            random_delay_seconds=1.25,
            compliance_passed=True,
            order_raw_response={"positionId": "POS-456"},
        )

        data = json.loads(result)
        assert data["fill_price"] is None
        assert data["pre_trade_bid"] is None
        assert data["pre_trade_ask"] is None
        assert data["slippage_pips"] is None
        assert data["execution_latency_ms"] is None

        # Ensure other fields are still present
        assert data["volume"] == 0.15
        assert data["side"] == "SELL"
        assert data["sl_pips"] == 40.0
        assert data["tp_pips"] == 80.0

    def test_json_roundtrip(self) -> None:
        """Test the output is valid JSON that can be parsed back to dict."""
        original_data = {
            "fill_price": 1.0950,
            "volume": 0.2,
            "side": "BUY",
            "sl_price": 1.0900,
            "tp_price": 1.1050,
            "sl_pips": 60.0,
            "tp_pips": 120.0,
            "pre_trade_bid": 1.09495,
            "pre_trade_ask": 1.09505,
            "slippage_pips": 0.2,
            "execution_latency_ms": 89.3,
            "random_delay_seconds": 0.5,
            "compliance_passed": True,
            "order_raw_response": {"test": "data"},
        }

        # Build meta using the method
        meta_json = ExecutionEngine._build_execution_meta(**original_data)

        # Parse back
        parsed = json.loads(meta_json)

        # Verify all fields match
        for key, value in original_data.items():
            assert parsed[key] == value

    def test_compliance_passed_false(self) -> None:
        """Test with compliance_passed=False."""
        result = ExecutionEngine._build_execution_meta(
            fill_price=1.0856,
            volume=0.1,
            side="BUY",
            sl_price=1.0806,
            tp_price=1.0956,
            sl_pips=50.0,
            tp_pips=100.0,
            pre_trade_bid=1.08555,
            pre_trade_ask=1.08565,
            slippage_pips=0.5,
            execution_latency_ms=125.5,
            random_delay_seconds=0.75,
            compliance_passed=False,
            order_raw_response={"error": "compliance failed"},
        )

        data = json.loads(result)
        assert data["compliance_passed"] is False

    def test_sell_side(self) -> None:
        """Test with SELL side."""
        result = ExecutionEngine._build_execution_meta(
            fill_price=1.0856,
            volume=0.1,
            side="SELL",
            sl_price=1.0906,
            tp_price=1.0756,
            sl_pips=50.0,
            tp_pips=100.0,
            pre_trade_bid=1.08555,
            pre_trade_ask=1.08565,
            slippage_pips=0.3,
            execution_latency_ms=98.2,
            random_delay_seconds=0.5,
            compliance_passed=True,
            order_raw_response={"positionId": "POS-789"},
        )

        data = json.loads(result)
        assert data["side"] == "SELL"
        assert data["fill_price"] == 1.0856


# ── Test update_execution_meta ──────────────────────────────────────────────


class TestUpdateExecutionMeta:
    """Tests for DecisionStore.update_execution_meta() method."""

    def test_update_and_persist(self, store) -> None:
        """Update execution meta and verify persistence by querying directly."""
        # Create an intent and advance it to opened state
        intent = TradeIntent(
            trade_date="2026-02-23",
            symbol="EURUSD",
            scanner_score=0.85,
        )
        _advance_to_opened(store, intent)

        # Build execution meta
        meta_json = ExecutionEngine._build_execution_meta(
            fill_price=1.0856,
            volume=0.1,
            side="BUY",
            sl_price=1.0806,
            tp_price=1.0956,
            sl_pips=50.0,
            tp_pips=100.0,
            pre_trade_bid=1.08555,
            pre_trade_ask=1.08565,
            slippage_pips=0.5,
            execution_latency_ms=125.5,
            random_delay_seconds=0.75,
            compliance_passed=True,
            order_raw_response={"positionId": "POS-123"},
        )

        # Update execution meta
        store.update_execution_meta(intent.id, meta_json)

        # Query directly from database to verify persistence
        row = store._conn.execute(
            "SELECT execution_meta FROM decisions WHERE intent_id = ?",
            (intent.id,),
        ).fetchone()

        assert row is not None
        assert row["execution_meta"] == meta_json

    def test_json_roundtrip(self, store) -> None:
        """Verify the JSON roundtrips correctly through the store."""
        intent = TradeIntent(
            trade_date="2026-02-23",
            symbol="GBPUSD",
            scanner_score=0.75,
        )
        _advance_to_opened(store, intent)

        # Create execution meta with all fields
        original_meta = {
            "fill_price": 1.2950,
            "volume": 0.15,
            "side": "SELL",
            "sl_price": 1.3000,
            "tp_price": 1.2850,
            "sl_pips": 40.0,
            "tp_pips": 80.0,
            "pre_trade_bid": 1.29495,
            "pre_trade_ask": 1.29505,
            "slippage_pips": 0.2,
            "execution_latency_ms": 156.7,
            "random_delay_seconds": 1.0,
            "compliance_passed": True,
            "order_raw_response": {"positionId": "POS-456", "openPrice": 1.2950},
        }

        meta_json = ExecutionEngine._build_execution_meta(**original_meta)
        store.update_execution_meta(intent.id, meta_json)

        # Retrieve and parse
        row = store._conn.execute(
            "SELECT execution_meta FROM decisions WHERE intent_id = ?",
            (intent.id,),
        ).fetchone()
        retrieved_meta = json.loads(row["execution_meta"])

        # Verify all fields match
        for key, value in original_meta.items():
            assert retrieved_meta[key] == value

    def test_update_with_none_optional_fields(self, store) -> None:
        """Test updating with None values for optional fields."""
        intent = TradeIntent(
            trade_date="2026-02-23",
            symbol="USDJPY",
            scanner_score=0.90,
        )
        _advance_to_opened(store, intent)

        # Build meta with None values
        meta_json = ExecutionEngine._build_execution_meta(
            fill_price=None,
            volume=0.05,
            side="BUY",
            sl_price=150.0,
            tp_price=152.0,
            sl_pips=50.0,
            tp_pips=100.0,
            pre_trade_bid=None,
            pre_trade_ask=None,
            slippage_pips=None,
            execution_latency_ms=None,
            random_delay_seconds=0.3,
            compliance_passed=True,
            order_raw_response={"positionId": "POS-789"},
        )

        store.update_execution_meta(intent.id, meta_json)

        # Retrieve and verify
        row = store._conn.execute(
            "SELECT execution_meta FROM decisions WHERE intent_id = ?",
            (intent.id,),
        ).fetchone()
        data = json.loads(row["execution_meta"])

        assert data["fill_price"] is None
        assert data["pre_trade_bid"] is None
        assert data["pre_trade_ask"] is None
        assert data["slippage_pips"] is None
        assert data["execution_latency_ms"] is None
        assert data["volume"] == 0.05
        assert data["side"] == "BUY"

    def test_multiple_updates(self, store) -> None:
        """Test that calling update multiple times works correctly."""
        intent = TradeIntent(
            trade_date="2026-02-23",
            symbol="EURUSD",
            scanner_score=0.80,
        )
        _advance_to_opened(store, intent)

        # First update
        meta1 = ExecutionEngine._build_execution_meta(
            fill_price=1.0850,
            volume=0.1,
            side="BUY",
            sl_price=1.0800,
            tp_price=1.0950,
            sl_pips=50.0,
            tp_pips=100.0,
            pre_trade_bid=1.08495,
            pre_trade_ask=1.08505,
            slippage_pips=0.2,
            execution_latency_ms=100.0,
            random_delay_seconds=0.5,
            compliance_passed=True,
            order_raw_response={"positionId": "POS-001"},
        )
        store.update_execution_meta(intent.id, meta1)

        # Second update (overwrites)
        meta2 = ExecutionEngine._build_execution_meta(
            fill_price=1.0900,
            volume=0.2,
            side="SELL",
            sl_price=1.0950,
            tp_price=1.0800,
            sl_pips=60.0,
            tp_pips=120.0,
            pre_trade_bid=1.08995,
            pre_trade_ask=1.09005,
            slippage_pips=0.3,
            execution_latency_ms=150.0,
            random_delay_seconds=0.75,
            compliance_passed=True,
            order_raw_response={"positionId": "POS-002"},
        )
        store.update_execution_meta(intent.id, meta2)

        # Verify final state is the second update
        row = store._conn.execute(
            "SELECT execution_meta FROM decisions WHERE intent_id = ?",
            (intent.id,),
        ).fetchone()
        data = json.loads(row["execution_meta"])

        assert data["fill_price"] == 1.0900
        assert data["volume"] == 0.2
        assert data["side"] == "SELL"
        assert data["order_raw_response"]["positionId"] == "POS-002"
