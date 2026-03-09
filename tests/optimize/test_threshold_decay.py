"""Tests for threshold decay logic."""

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.decision.schemas import TradeIntent
from src.decision_store.sqlite_store import DecisionStore
from src.optimize.optimization_engine import OptimizationEngine
from src.optimize.optimization_state import Thresholds
from src.optimize.thresholds import compute_thresholds
from src.optimize.trade_stats import compute_inactive_days

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: Path) -> Iterator[DecisionStore]:
    """Create a fresh DecisionStore with a temporary database."""
    db_path = tmp_path / "test_decisions.db"
    s = DecisionStore(db_path=str(db_path))
    yield s
    s.close()


# ── Helpers ────────────────────────────────────────────────────────────────


def _create_closed_intent(
    store: DecisionStore,
    symbol: str,
    pnl: float,
    created_at: datetime | None = None,
) -> None:
    intent = TradeIntent(trade_date="2026-02-20", symbol=symbol)
    store.insert_intent(intent)
    store.claim_next_pending("llm-0")
    store.mark_ready_for_exec(intent.id)
    store.mark_executing(intent.id)
    store.mark_opened(intent.id, position_id=f"pos-{symbol}")
    store.mark_closed(intent.id, realized_pnl=pnl, exit_reason="tp_hit")

    if created_at is not None:
        store._conn.execute(
            "UPDATE intents SET created_at = :ts WHERE id = :id",
            {"ts": created_at.isoformat(), "id": intent.id},
        )
        store._conn.commit()


# ── Tests: Threshold Decay ─────────────────────────────────────────────────


def test_no_decay_when_inactive_days_none() -> None:
    result = compute_thresholds(
        global_win_rate=0.40,
        symbol_win_rates={"EURUSD": 0.30},
        inactive_days=None,
    )
    assert result["EURUSD"].min_blended_confidence == 0.65


def test_no_decay_when_zero_inactive_days() -> None:
    result = compute_thresholds(
        global_win_rate=0.40,
        symbol_win_rates={"EURUSD": 0.30},
        inactive_days={"EURUSD": 0},
    )
    assert result["EURUSD"].min_blended_confidence == 0.65


def test_decay_after_1_day_inactive() -> None:
    result = compute_thresholds(
        global_win_rate=0.40,
        symbol_win_rates={"EURUSD": 0.30},
        inactive_days={"EURUSD": 1},
    )
    assert result["EURUSD"].min_blended_confidence == 0.63


def test_decay_after_2_days_inactive() -> None:
    result = compute_thresholds(
        global_win_rate=0.40,
        symbol_win_rates={"EURUSD": 0.30},
        inactive_days={"EURUSD": 2},
    )
    assert result["EURUSD"].min_blended_confidence == 0.62


def test_full_decay_after_3_days_inactive() -> None:
    result = compute_thresholds(
        global_win_rate=0.40,
        symbol_win_rates={"EURUSD": 0.30},
        inactive_days={"EURUSD": 3},
    )
    assert result["EURUSD"].min_blended_confidence == 0.60


def test_positive_adjustment_not_decayed() -> None:
    result = compute_thresholds(
        global_win_rate=0.40,
        symbol_win_rates={"EURUSD": 0.50},
        inactive_days={"EURUSD": 3},
    )
    assert result["EURUSD"].min_blended_confidence == 0.55


def test_no_adjustment_not_decayed() -> None:
    result = compute_thresholds(
        global_win_rate=0.40,
        symbol_win_rates={"EURUSD": 0.43},
        inactive_days={"EURUSD": 3},
    )
    assert result["EURUSD"].min_blended_confidence == 0.60


def test_mixed_symbols_with_different_inactivity() -> None:
    result = compute_thresholds(
        global_win_rate=0.40,
        symbol_win_rates={
            "EURUSD": 0.30,
            "GBPUSD": 0.30,
            "USDJPY": 0.50,
        },
        inactive_days={"EURUSD": 1, "GBPUSD": 3, "USDJPY": 2},
    )
    assert result["EURUSD"].min_blended_confidence == 0.63
    assert result["GBPUSD"].min_blended_confidence == 0.60
    assert result["USDJPY"].min_blended_confidence == 0.55


# ── Tests: compute_inactive_days ───────────────────────────────────────────


def test_compute_inactive_days_basic(store: DecisionStore) -> None:
    now = datetime.now(timezone.utc)
    _create_closed_intent(store, "EURUSD", pnl=1.0, created_at=now - timedelta(days=2, hours=1))

    inactive = compute_inactive_days(store, ["EURUSD"])

    assert inactive["EURUSD"] == 2


def test_compute_inactive_days_no_history(store: DecisionStore) -> None:
    inactive = compute_inactive_days(store, ["EURUSD"])

    assert inactive["EURUSD"] == 0


def test_compute_inactive_days_mixed(store: DecisionStore) -> None:
    now = datetime.now(timezone.utc)
    _create_closed_intent(store, "EURUSD", pnl=1.0, created_at=now - timedelta(days=1, minutes=5))
    _create_closed_intent(store, "GBPUSD", pnl=1.0, created_at=now - timedelta(days=4, minutes=1))

    inactive = compute_inactive_days(store, ["EURUSD", "GBPUSD", "USDJPY"])

    assert inactive["EURUSD"] == 1
    assert inactive["GBPUSD"] == 4
    assert inactive["USDJPY"] == 0


# ── Tests: OptimizationEngine wiring ───────────────────────────────────────


def test_engine_passes_inactive_days(
    tmp_path: Path, store: DecisionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _create_closed_intent(store, "EURUSD", pnl=5.0)
    captured: dict[str, dict[str, int] | None] = {"inactive": None}

    def _fake_compute_inactive_days(*_args: object, **_kwargs: object) -> dict[str, int]:
        return {"EURUSD": 2}

    def _fake_compute_thresholds(
        *,
        global_win_rate: float,
        symbol_win_rates: dict[str, float],
        inactive_days: dict[str, int] | None = None,
    ) -> dict[str, Thresholds]:
        _ = (global_win_rate, symbol_win_rates)
        captured["inactive"] = inactive_days
        base = Thresholds(min_confidence="medium", min_blended_confidence=0.55)
        return {"global": base, "EURUSD": base}

    monkeypatch.setattr(
        "src.optimize.optimization_engine.compute_inactive_days",
        _fake_compute_inactive_days,
    )
    monkeypatch.setattr(
        "src.optimize.optimization_engine.compute_thresholds",
        _fake_compute_thresholds,
    )

    state_path = tmp_path / "state.json"
    engine = OptimizationEngine(
        store,
        journal=None,
        state_path=state_path,
        pnl_days=7,
        win_days=14,
    )
    engine.refresh_state()

    assert captured["inactive"] == {"EURUSD": 2}
