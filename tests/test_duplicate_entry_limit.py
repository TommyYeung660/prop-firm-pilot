"""
Tests for same-symbol same-direction duplicate entry limit (P2.6).

Validates:
1. has_active_position_for_symbol detects opened intents for a symbol
2. count_same_direction_today counts active same-direction intents only
3. Closed trades do not block same-day re-entry when flat
4. Different date or symbol does not count
"""

import pytest

from src.decision.schemas import TradeIntent
from src.decision_store.sqlite_store import DecisionStore

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_opened_intent(
    store: DecisionStore,
    symbol: str = "EURUSD",
    side: str = "SELL",
    trade_date: str = "2026-03-09",
) -> TradeIntent:
    """Insert an intent and march it through to 'opened' state."""
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
    return intent


def _make_closed_intent(
    store: DecisionStore,
    symbol: str = "EURUSD",
    side: str = "SELL",
    trade_date: str = "2026-03-09",
    exit_reason: str = "sl_hit",
    pnl: float = -25.0,
) -> TradeIntent:
    """Insert an intent and march it through to 'closed' state."""
    intent = _make_opened_intent(store, symbol=symbol, side=side, trade_date=trade_date)
    store.mark_closed(intent.id, realized_pnl=pnl, exit_reason=exit_reason)
    return intent


# ── has_active_position_for_symbol ───────────────────────────────────────


class TestHasActivePositionForSymbol:
    """Tests for DecisionStore.has_active_position_for_symbol()."""

    @pytest.fixture
    def store(self, tmp_path) -> DecisionStore:
        return DecisionStore(db_path=f"{tmp_path}/test.db")

    def test_has_active_position_true(self, store: DecisionStore) -> None:
        """Opened EURUSD intent → has_active_position_for_symbol('EURUSD') is True."""
        _make_opened_intent(store, symbol="EURUSD")
        assert store.has_active_position_for_symbol("EURUSD") is True

    def test_has_active_position_false(self, store: DecisionStore) -> None:
        """No opened intents → False."""
        assert store.has_active_position_for_symbol("EURUSD") is False

    def test_has_active_position_wrong_symbol(self, store: DecisionStore) -> None:
        """Opened GBPUSD → has_active_position_for_symbol('EURUSD') is False."""
        _make_opened_intent(store, symbol="GBPUSD")
        assert store.has_active_position_for_symbol("EURUSD") is False

    def test_closed_does_not_count(self, store: DecisionStore) -> None:
        """Closed intent should not count as active position."""
        _make_closed_intent(store, symbol="EURUSD")
        assert store.has_active_position_for_symbol("EURUSD") is False


# ── count_same_direction_today ───────────────────────────────────────────


class TestCountSameDirectionToday:
    """Tests for DecisionStore.count_same_direction_today()."""

    @pytest.fixture
    def store(self, tmp_path) -> DecisionStore:
        return DecisionStore(db_path=f"{tmp_path}/test.db")

    def test_count_same_direction_basic(self, store: DecisionStore) -> None:
        """2 opened EURUSD SELL today → count returns 2."""
        _make_opened_intent(store, side="SELL")
        _make_opened_intent(store, side="SELL")
        assert store.count_same_direction_today("EURUSD", "SELL", "2026-03-09") == 2

    def test_count_ignores_different_direction(self, store: DecisionStore) -> None:
        """EURUSD BUY doesn't count toward SELL."""
        _make_opened_intent(store, side="SELL")
        _make_opened_intent(store, side="BUY")
        assert store.count_same_direction_today("EURUSD", "SELL", "2026-03-09") == 1

    def test_count_ignores_different_date(self, store: DecisionStore) -> None:
        """Different trade_date doesn't count."""
        _make_opened_intent(store, trade_date="2026-03-09")
        _make_opened_intent(store, trade_date="2026-03-08")
        assert store.count_same_direction_today("EURUSD", "SELL", "2026-03-09") == 1

    def test_count_ignores_different_symbol(self, store: DecisionStore) -> None:
        """GBPUSD SELL doesn't count toward EURUSD SELL."""
        _make_opened_intent(store, symbol="EURUSD", side="SELL")
        _make_opened_intent(store, symbol="GBPUSD", side="SELL")
        assert store.count_same_direction_today("EURUSD", "SELL", "2026-03-09") == 1

    def test_count_ignores_closed_trades_when_flat(self, store: DecisionStore) -> None:
        """A closed EURUSD SELL should not block same-day re-entry when flat."""
        _make_closed_intent(store)
        assert store.count_same_direction_today("EURUSD", "SELL", "2026-03-09") == 0

    def test_count_includes_opened_but_not_closed(self, store: DecisionStore) -> None:
        """Only still-open same-direction intents count."""
        _make_closed_intent(store)
        _make_opened_intent(store, side="SELL")
        assert store.count_same_direction_today("EURUSD", "SELL", "2026-03-09") == 1

    def test_count_zero(self, store: DecisionStore) -> None:
        """No matching intents → 0."""
        assert store.count_same_direction_today("EURUSD", "SELL", "2026-03-09") == 0
