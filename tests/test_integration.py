"""
Integration tests for the Hybrid EA+LLM pipeline.

Tests the full flow: scanner signals → DecisionStore → LLM worker →
execution engine, using a real SQLite DecisionStore with mocked external I/O
(MatchTrader API, PropFirmGuard, PositionSizer).
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.compliance.prop_firm_guard import ComplianceResult
from src.config import (
    AccountConfig,
    AppConfig,
    ComplianceConfig,
    DecisionStoreConfig,
    ExecutionConfig,
    InstrumentConfig,
    SchedulerConfig,
)
from src.decision.agent_bridge import AgentDecision
from src.decision.decision_formatter import format_decision
from src.decision.schemas import TradeIntent
from src.decision_store.janitor import Janitor
from src.decision_store.sqlite_store import DecisionStore, DecisionStoreError
from src.execution.engine import ExecutionEngine
from src.signal.scanner_bridge import ScannerSignal

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: object) -> DecisionStore:
    """Create a fresh DecisionStore with a temporary database."""
    db_path = f"{tmp_path}/test_integration.db"
    s = DecisionStore(db_path=db_path)
    yield s  # type: ignore[misc]
    s.close()


@pytest.fixture
def config() -> AppConfig:
    """Minimal AppConfig for integration tests."""
    return AppConfig(
        account=AccountConfig(initial_balance=50000),
        compliance=ComplianceConfig(),
        execution=ExecutionConfig(
            max_positions=3,
            default_risk_pct=0.01,
            random_delay_min=0.0,
            random_delay_max=0.0,
        ),
        instruments={
            "EURUSD": InstrumentConfig(pip_value=10.0, pip_size=0.0001),
            "GBPUSD": InstrumentConfig(pip_value=10.0, pip_size=0.0001),
        },
        decision_store=DecisionStoreConfig(db_path=""),
        scheduler=SchedulerConfig(),
    )


@pytest.fixture
def mock_guard() -> MagicMock:
    """Mock PropFirmGuard that passes all checks by default."""
    guard = MagicMock()
    guard.check_all.return_value = ComplianceResult(
        passed=True, rule_name="ALL", reason="All checks passed"
    )
    guard.add_random_delay.return_value = 0.0
    return guard


@pytest.fixture
def mock_matchtrader() -> AsyncMock:
    """Mock MatchTraderClient with default balance and empty positions."""
    client = AsyncMock()
    client.get_balance.return_value = MagicMock(
        balance=50000.0, equity=50000.0, margin=0.0, free_margin=50000.0
    )
    client.get_open_positions.return_value = []
    client.open_position.return_value = MagicMock(
        success=True, position_id="pos-integration-001", message=""
    )
    return client


@pytest.fixture
def mock_sizer() -> MagicMock:
    """Mock PositionSizer with fixed volume."""
    sizer = MagicMock()
    sizer.calculate_volume.return_value = 0.1
    sizer.calculate_risk_amount.return_value = 50.0
    return sizer


@pytest.fixture
def engine(
    store: DecisionStore,
    mock_guard: MagicMock,
    mock_matchtrader: AsyncMock,
    mock_sizer: MagicMock,
    config: AppConfig,
) -> ExecutionEngine:
    """ExecutionEngine wired to real store + mocked externals."""
    return ExecutionEngine(
        store=store,
        guard=mock_guard,
        matchtrader=mock_matchtrader,
        sizer=mock_sizer,
        config=config,
    )


# ── Helper Functions ────────────────────────────────────────────────────────


def make_signal(instrument: str = "EURUSD", score: float = 0.85, rank: int = 1) -> ScannerSignal:
    """Create a ScannerSignal for testing."""
    return ScannerSignal(
        instrument=instrument,
        score=score,
        rank=rank,
        confidence="high",
        score_gap=0.12,
        drop_distance=0.05,
        topk_spread=0.08,
    )


def signal_to_intent(signal: ScannerSignal, trade_date: str = "2026-02-16") -> TradeIntent:
    """Convert a ScannerSignal into a TradeIntent (simulates scanner_loop)."""
    return TradeIntent(
        trade_date=trade_date,
        symbol=signal.instrument,
        scanner_score=signal.score,
        scanner_confidence=signal.confidence,
        scanner_score_gap=signal.score_gap,
        scanner_drop_distance=signal.drop_distance,
        scanner_topk_spread=signal.topk_spread,
        source="scanner",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=4),
    )


def simulate_llm_decision(
    store: DecisionStore,
    intent_id: str,
    symbol: str,
    decision: str = "BUY",
    worker_id: str = "llm-0",
) -> None:
    """Simulate LLM worker claiming and deciding on an intent.

    Mimics the _process_claimed_intent logic from scheduler.py:
    claim → decide → update fields → mark_ready_for_exec (or cancel).
    """
    agent_decision = AgentDecision(
        symbol=symbol,
        decision=decision,
        final_state={"summary": f"Test decision for {symbol}"},
        risk_report="Integration test risk report",
    )

    if agent_decision.is_actionable:
        formatted = format_decision(
            symbol=symbol,
            decision=agent_decision.decision,
            scanner_score=0.85,
            scanner_confidence="high",
            agent_state=agent_decision.final_state,
        )
        store.update_intent_decision(
            intent_id,
            agent_decision.decision,
            sl_pips=formatted.suggested_sl_pips,
            tp_pips=formatted.suggested_tp_pips,
            risk_report=agent_decision.risk_report,
            state_json=json.dumps(agent_decision.final_state, default=str),
        )
        store.mark_ready_for_exec(intent_id)
    else:
        store.mark_cancelled(intent_id, f"LLM decided {agent_decision.decision}")


# ── Integration Tests ───────────────────────────────────────────────────────


class TestHappyPathBuy:
    """Scanner → LLM (BUY) → Execute → Opened."""

    async def test_full_pipeline_buy(self, store: DecisionStore, engine: ExecutionEngine) -> None:
        # Phase 1: Scanner creates intent
        signal = make_signal("EURUSD", score=0.85)
        intent = signal_to_intent(signal)
        store.insert_intent(intent)

        # Phase 2: LLM claims and decides BUY
        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None
        assert claimed.id == intent.id
        simulate_llm_decision(store, intent.id, "EURUSD", "BUY")

        # Phase 3: Execution engine processes
        processed = await engine.execute_ready_intents()
        assert processed == 1

        # Verify final state
        final = store.get_intent(intent.id)
        assert final is not None
        assert final.status == "opened"
        assert final.position_id == "pos-integration-001"
        assert final.suggested_side == "BUY"


class TestHappyPathSell:
    """Scanner → LLM (SELL) → Execute → Opened."""

    async def test_full_pipeline_sell(self, store: DecisionStore, engine: ExecutionEngine) -> None:
        signal = make_signal("GBPUSD", score=0.90)
        intent = signal_to_intent(signal)
        store.insert_intent(intent)

        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None
        simulate_llm_decision(store, intent.id, "GBPUSD", "SELL")

        processed = await engine.execute_ready_intents()
        assert processed == 1

        final = store.get_intent(intent.id)
        assert final is not None
        assert final.status == "opened"
        assert final.suggested_side == "SELL"


class TestHoldCancelled:
    """Scanner → LLM (HOLD) → Cancelled — never reaches execution."""

    async def test_hold_cancels_intent(self, store: DecisionStore, engine: ExecutionEngine) -> None:
        intent = signal_to_intent(make_signal("EURUSD"))
        store.insert_intent(intent)

        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None
        simulate_llm_decision(store, intent.id, "EURUSD", "HOLD")

        # Execution should find nothing
        processed = await engine.execute_ready_intents()
        assert processed == 0

        final = store.get_intent(intent.id)
        assert final is not None
        assert final.status == "cancelled"


class TestComplianceRejected:
    """Scanner → LLM (BUY) → Compliance Rejected."""

    async def test_compliance_rejects_trade(
        self,
        store: DecisionStore,
        engine: ExecutionEngine,
        mock_guard: MagicMock,
    ) -> None:
        # Configure guard to reject
        mock_guard.check_all.return_value = ComplianceResult(
            passed=False,
            rule_name="daily_drawdown",
            reason="Daily drawdown limit reached (85%)",
        )

        intent = signal_to_intent(make_signal("EURUSD"))
        store.insert_intent(intent)

        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None
        simulate_llm_decision(store, intent.id, "EURUSD", "BUY")

        processed = await engine.execute_ready_intents()
        assert processed == 1

        final = store.get_intent(intent.id)
        assert final is not None
        assert final.status == "rejected"

        # Verify decision record
        decision = store.get_decision(intent.id)
        assert decision is not None
        assert decision.failure_reason == "Daily drawdown limit reached (85%)"


class TestApiError:
    """Scanner → LLM (BUY) → API Error → Failed."""

    async def test_api_error_marks_failed(
        self,
        store: DecisionStore,
        engine: ExecutionEngine,
        mock_matchtrader: AsyncMock,
    ) -> None:
        mock_matchtrader.open_position.side_effect = Exception("Connection timeout")

        intent = signal_to_intent(make_signal("EURUSD"))
        store.insert_intent(intent)

        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None
        simulate_llm_decision(store, intent.id, "EURUSD", "BUY")

        processed = await engine.execute_ready_intents()
        assert processed == 1

        final = store.get_intent(intent.id)
        assert final is not None
        assert final.status == "failed"
        assert "Connection timeout" in (final.execution_error or "")


class TestIdempotency:
    """Duplicate scanner signal is rejected by unique constraint."""

    def test_duplicate_idempotency_key_rejected(self, store: DecisionStore) -> None:
        intent1 = TradeIntent(
            trade_date="2026-02-16",
            symbol="EURUSD",
            scanner_score=0.85,
            idempotency_key="dedup-key-001",
        )
        store.insert_intent(intent1)

        intent2 = TradeIntent(
            trade_date="2026-02-16",
            symbol="EURUSD",
            scanner_score=0.90,
            idempotency_key="dedup-key-001",
        )
        with pytest.raises(DecisionStoreError, match="Duplicate intent"):
            store.insert_intent(intent2)

    def test_intent_exists_prevents_duplicate(self, store: DecisionStore) -> None:
        intent = signal_to_intent(make_signal("EURUSD"))
        store.insert_intent(intent)

        assert store.intent_exists("EURUSD", "2026-02-16", "scanner") is True
        assert store.intent_exists("GBPUSD", "2026-02-16", "scanner") is False
        assert store.intent_exists("EURUSD", "2026-02-17", "scanner") is False


class TestJanitorRecycle:
    """Claim timeout → Janitor recycles → Re-claimed → Executed."""

    async def test_expired_claim_recycled_and_reprocessed(
        self, store: DecisionStore, engine: ExecutionEngine
    ) -> None:
        intent = signal_to_intent(make_signal("EURUSD"))
        store.insert_intent(intent)

        # Claim the intent
        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None

        # Simulate timeout by backdating both claim_ts and expires_at
        past = datetime.now(timezone.utc) - timedelta(minutes=60)
        past_str = past.isoformat()
        expired_at_str = (past + timedelta(minutes=30)).isoformat()  # 30min TTL, still in past
        store._conn.execute(
            "UPDATE intents SET claim_ts = :ts, expires_at = :exp WHERE id = :id",
            {"ts": past_str, "exp": expired_at_str, "id": intent.id},
        )
        store._conn.commit()

        # Janitor recycles expired claims → timed_out
        janitor = Janitor(store, retention_days=7)
        recycled, cleaned = janitor.run_cycle()
        assert recycled >= 1

        # Verify intent is timed_out
        timed_out = store.get_intent(intent.id)
        assert timed_out is not None
        assert timed_out.status == "timed_out"

        # Transition timed_out → pending (valid per state machine)
        store._conn.execute(
            "UPDATE intents SET status = 'pending', claim_worker_id = NULL, "
            "claim_ts = NULL, expires_at = NULL WHERE id = :id",
            {"id": intent.id},
        )
        store._conn.commit()

        # Re-claim and process through full pipeline
        reclaimed = store.claim_next_pending("llm-1")
        assert reclaimed is not None
        assert reclaimed.id == intent.id
        simulate_llm_decision(store, intent.id, "EURUSD", "BUY")

        processed = await engine.execute_ready_intents()
        assert processed == 1

        final = store.get_intent(intent.id)
        assert final is not None
        assert final.status == "opened"


class TestMultipleIntents:
    """Multiple intents for different symbols process independently."""

    async def test_independent_pipeline_processing(
        self,
        store: DecisionStore,
        engine: ExecutionEngine,
        mock_matchtrader: AsyncMock,
    ) -> None:
        # Create intents for 2 symbols
        symbols = ["EURUSD", "GBPUSD"]
        intents = []
        for symbol in symbols:
            intent = signal_to_intent(make_signal(symbol))
            store.insert_intent(intent)
            intents.append(intent)

        # Claim and decide both
        for intent in intents:
            claimed = store.claim_next_pending("llm-0")
            assert claimed is not None
            simulate_llm_decision(store, claimed.id, claimed.symbol, "BUY")

        # Set up unique position IDs per call
        call_count = 0

        def position_id_factory(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return MagicMock(
                success=True,
                position_id=f"pos-multi-{call_count:03d}",
                message="",
            )

        mock_matchtrader.open_position.side_effect = position_id_factory

        # Execute all
        processed = await engine.execute_ready_intents()
        assert processed == 2

        # Verify each has unique position_id
        position_ids = set()
        for intent in intents:
            final = store.get_intent(intent.id)
            assert final is not None
            assert final.status == "opened"
            assert final.position_id is not None
            position_ids.add(final.position_id)
        assert len(position_ids) == 2


class TestExecutionTiming:
    """Execution only processes ready_for_exec intents, not pending."""

    async def test_pending_not_executed(
        self, store: DecisionStore, engine: ExecutionEngine
    ) -> None:
        # Insert pending intent (not yet claimed by LLM)
        intent = signal_to_intent(make_signal("EURUSD"))
        store.insert_intent(intent)

        # Execution engine should find nothing
        processed = await engine.execute_ready_intents()
        assert processed == 0

        # Intent remains pending
        current = store.get_intent(intent.id)
        assert current is not None
        assert current.status == "pending"

    async def test_claimed_not_executed(
        self, store: DecisionStore, engine: ExecutionEngine
    ) -> None:
        intent = signal_to_intent(make_signal("EURUSD"))
        store.insert_intent(intent)
        store.claim_next_pending("llm-0")

        # Claimed but not yet ready_for_exec
        processed = await engine.execute_ready_intents()
        assert processed == 0

        current = store.get_intent(intent.id)
        assert current is not None
        assert current.status == "claimed"

    async def test_only_ready_for_exec_processed(
        self, store: DecisionStore, engine: ExecutionEngine
    ) -> None:
        # Intent 1: pending (not touched)
        intent1 = signal_to_intent(make_signal("EURUSD"))
        store.insert_intent(intent1)

        # Intent 2: ready_for_exec (should be processed)
        intent2 = signal_to_intent(make_signal("GBPUSD"))
        store.insert_intent(intent2)
        claimed = store.claim_next_pending("llm-0")
        # claim_next_pending returns oldest pending — could be intent1 or intent2
        # We need intent2 to be ready, so claim intent1 first, then intent2
        if claimed and claimed.id == intent1.id:
            # Cancel intent1 and re-insert, or just claim intent2 next
            store.mark_cancelled(intent1.id, "test skip")
            claimed2 = store.claim_next_pending("llm-0")
            assert claimed2 is not None
            simulate_llm_decision(store, claimed2.id, "GBPUSD", "BUY")
            target_id = claimed2.id
        else:
            assert claimed is not None
            simulate_llm_decision(store, claimed.id, claimed.symbol, "BUY")
            target_id = claimed.id

        processed = await engine.execute_ready_intents()
        assert processed == 1

        final = store.get_intent(target_id)
        assert final is not None
        assert final.status == "opened"


class TestDecisionRecordAudit:
    """DecisionRecord tracks all lifecycle timestamps."""

    async def test_full_lifecycle_timestamps(
        self, store: DecisionStore, engine: ExecutionEngine
    ) -> None:
        intent = signal_to_intent(make_signal("EURUSD"))
        store.insert_intent(intent)

        # Check initial decision record
        decision = store.get_decision(intent.id)
        assert decision is not None
        assert decision.created_at is not None

        # Claim
        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None
        decision = store.get_decision(intent.id)
        assert decision is not None
        assert decision.claimed_at is not None

        # LLM decides
        simulate_llm_decision(store, intent.id, "EURUSD", "BUY")
        decision = store.get_decision(intent.id)
        assert decision is not None
        assert decision.decided_at is not None

        # Execute
        processed = await engine.execute_ready_intents()
        assert processed == 1

        # Final check — all timestamps populated
        decision = store.get_decision(intent.id)
        assert decision is not None
        assert decision.claimed_at is not None
        assert decision.decided_at is not None
        assert decision.executed_at is not None
        assert decision.position_id == "pos-integration-001"
        assert decision.status == "opened"
