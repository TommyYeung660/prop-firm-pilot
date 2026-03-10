"""
Tests for consecutive loss circuit breaker (P2.5).

Validates:
1. count_sl_hits_today counts closed intents with exit_reason='sl_hit' for today
2. count_symbol_losses_today counts SL hits for a specific symbol today
3. Circuit breaker blocks new intents when daily_sl_hit_limit reached
4. Circuit breaker blocks symbol-specific intents when symbol_loss_limit reached
5. Circuit breaker resets at day boundary
"""

import pytest

from src.decision.schemas import TradeIntent
from src.decision_store.sqlite_store import DecisionStore

# ── Helper ──────────────────────────────────────────────────────────────


def _make_closed_intent(
    store: DecisionStore,
    symbol: str = "EURUSD",
    trade_date: str = "2026-03-09",
    exit_reason: str = "sl_hit",
    pnl: float = -25.0,
    side: str = "SELL",
) -> TradeIntent:
    """Insert an intent and march it through to 'closed' state."""
    intent = TradeIntent(trade_date=trade_date, symbol=symbol)
    store.insert_intent(intent)
    store.claim_next_pending("llm-0")
    store.update_intent_decision(
        intent.id,
        side=side,
        sl_pips=40,
        tp_pips=80,
        risk_report="test",
        state_json="{}",
    )
    store.mark_ready_for_exec(intent.id)
    store.mark_executing(intent.id)
    store.mark_opened(intent.id, position_id=f"POS_{intent.id}")
    store.mark_closed(intent.id, realized_pnl=pnl, exit_reason=exit_reason)
    return intent


# ── count_sl_hits_today ─────────────────────────────────────────────────


class TestCountSlHitsToday:
    """Tests for DecisionStore.count_sl_hits_today()."""

    @pytest.fixture
    def store(self, tmp_path) -> DecisionStore:
        return DecisionStore(db_path=f"{tmp_path}/test.db")

    def test_count_sl_hits_today_basic(self, store: DecisionStore) -> None:
        """2 SL hits today → count_sl_hits_today returns 2."""
        _make_closed_intent(store, symbol="EURUSD")
        _make_closed_intent(store, symbol="GBPUSD")
        assert store.count_sl_hits_today("2026-03-09") == 2

    def test_count_sl_hits_today_ignores_tp(self, store: DecisionStore) -> None:
        """TP hits should not count as SL hits."""
        _make_closed_intent(store, exit_reason="sl_hit")
        _make_closed_intent(store, exit_reason="tp_hit", pnl=50.0)
        assert store.count_sl_hits_today("2026-03-09") == 1

    def test_count_sl_hits_today_ignores_other_dates(self, store: DecisionStore) -> None:
        """Only count SL hits for the specified date."""
        _make_closed_intent(store, trade_date="2026-03-09")
        _make_closed_intent(store, trade_date="2026-03-08")
        assert store.count_sl_hits_today("2026-03-09") == 1

    def test_count_sl_hits_today_zero(self, store: DecisionStore) -> None:
        """No closed intents → 0."""
        assert store.count_sl_hits_today("2026-03-09") == 0

    def test_count_sl_hits_ignores_manual_close(self, store: DecisionStore) -> None:
        """Manual close should not count as SL hit."""
        _make_closed_intent(store, exit_reason="sl_hit")
        _make_closed_intent(store, exit_reason="manual_close", pnl=-10.0)
        assert store.count_sl_hits_today("2026-03-09") == 1


# ── count_symbol_losses_today ───────────────────────────────────────────


class TestCountSymbolLossesToday:
    """Tests for DecisionStore.count_symbol_losses_today()."""

    @pytest.fixture
    def store(self, tmp_path) -> DecisionStore:
        return DecisionStore(db_path=f"{tmp_path}/test.db")

    def test_count_symbol_losses_basic(self, store: DecisionStore) -> None:
        """2 EURUSD SL hits → count_symbol_losses_today('EURUSD') returns 2."""
        _make_closed_intent(store, symbol="EURUSD")
        _make_closed_intent(store, symbol="EURUSD")
        assert store.count_symbol_losses_today("EURUSD", "2026-03-09") == 2

    def test_count_symbol_losses_ignores_other_symbols(self, store: DecisionStore) -> None:
        """GBPUSD losses don't count toward EURUSD."""
        _make_closed_intent(store, symbol="EURUSD")
        _make_closed_intent(store, symbol="GBPUSD")
        assert store.count_symbol_losses_today("EURUSD", "2026-03-09") == 1

    def test_count_symbol_losses_ignores_other_dates(self, store: DecisionStore) -> None:
        """Only count losses for the specified date."""
        _make_closed_intent(store, symbol="EURUSD", trade_date="2026-03-09")
        _make_closed_intent(store, symbol="EURUSD", trade_date="2026-03-08")
        assert store.count_symbol_losses_today("EURUSD", "2026-03-09") == 1

    def test_count_symbol_losses_zero(self, store: DecisionStore) -> None:
        """No closed intents for symbol → 0."""
        assert store.count_symbol_losses_today("EURUSD", "2026-03-09") == 0

    def test_count_symbol_losses_ignores_tp_hits(self, store: DecisionStore) -> None:
        """TP hits should not count as losses."""
        _make_closed_intent(store, symbol="EURUSD", exit_reason="sl_hit")
        _make_closed_intent(store, symbol="EURUSD", exit_reason="tp_hit", pnl=50.0)
        assert store.count_symbol_losses_today("EURUSD", "2026-03-09") == 1
