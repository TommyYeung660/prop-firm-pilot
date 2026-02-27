"""Tests for OptimizationEngine."""

from pathlib import Path

import pytest

from src.decision.schemas import TradeIntent
from src.decision_store.sqlite_store import DecisionStore
from src.monitor.trade_journal import TradeJournal
from src.optimize.optimization_engine import OptimizationEngine

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: Path) -> DecisionStore:
    """Create a fresh DecisionStore with a temporary database."""
    db_path = tmp_path / "test_decisions.db"
    s = DecisionStore(db_path=str(db_path))
    yield s
    s.close()


@pytest.fixture
def journal(tmp_path: Path) -> TradeJournal:
    """Create a trade journal with a temporary file."""
    path = tmp_path / "trade_journal.jsonl"
    return TradeJournal(path)


# ── Helpers ────────────────────────────────────────────────────────────────


def _create_closed_intent(store: DecisionStore, symbol: str, pnl: float) -> None:
    intent = TradeIntent(trade_date="2026-02-20", symbol=symbol)
    store.insert_intent(intent)
    store.claim_next_pending("llm-0")
    store.mark_ready_for_exec(intent.id)
    store.mark_executing(intent.id)
    store.mark_opened(intent.id, position_id=f"pos-{symbol}")
    store.mark_closed(intent.id, realized_pnl=pnl, exit_reason="tp_hit")


# ── Tests ───────────────────────────────────────────────────────────────────


def test_engine_refresh_creates_state(
    tmp_path: Path, store: DecisionStore, journal: TradeJournal
) -> None:
    _create_closed_intent(store, "EURUSD", pnl=5.0)

    state_path = tmp_path / "state.json"
    engine = OptimizationEngine(
        store,
        journal,
        state_path=state_path,
        pnl_days=7,
        win_days=14,
    )
    state = engine.refresh_state()

    assert state_path.exists()
    assert state.generated_at
    assert state.feedback_pnl["EURUSD"] == 5.0
