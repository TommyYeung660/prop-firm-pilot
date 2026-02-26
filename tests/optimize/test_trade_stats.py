"""Tests for trade statistics aggregation."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.decision.schemas import TradeIntent
from src.decision_store.sqlite_store import DecisionStore
from src.monitor.trade_journal import TradeJournal
from src.optimize.trade_stats import build_pnl_feedback, compute_win_rates


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


def _create_closed_intent(store: DecisionStore, symbol: str, pnl: float) -> TradeIntent:
    intent = TradeIntent(trade_date="2026-02-20", symbol=symbol)
    store.insert_intent(intent)
    store.claim_next_pending("llm-0")
    store.mark_ready_for_exec(intent.id)
    store.mark_executing(intent.id)
    store.mark_opened(intent.id, position_id=f"pos-{symbol}")
    store.mark_closed(intent.id, realized_pnl=pnl, exit_reason="tp_hit")
    return intent


# ── Tests ───────────────────────────────────────────────────────────────────


def test_compute_win_rates_empty(store: DecisionStore) -> None:
    """Empty store should return 0.0 global win rate."""
    result = compute_win_rates(store, days=14)
    assert result["global"] == 0.0


def test_compute_win_rates_with_closed_trades(store: DecisionStore) -> None:
    """Win rates should be computed for global and per-symbol."""
    _create_closed_intent(store, "EURUSD", pnl=12.0)
    _create_closed_intent(store, "GBPUSD", pnl=-5.0)

    result = compute_win_rates(store, days=14)
    assert result["global"] == 0.5
    assert result["EURUSD"] == 1.0
    assert result["GBPUSD"] == 0.0


def test_build_pnl_feedback_from_store_and_journal(
    store: DecisionStore, journal: TradeJournal
) -> None:
    """PnL feedback should merge store and journal sources."""
    _create_closed_intent(store, "EURUSD", pnl=10.0)
    journal.log_trade(
        {
            "status": "CLOSED",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": "GBPUSD",
            "pnl": 5.0,
        }
    )

    feedback = build_pnl_feedback(store, journal, days=7)
    assert feedback["EURUSD"] == 10.0
    assert feedback["GBPUSD"] == 5.0
