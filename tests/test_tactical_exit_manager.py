"""Tests for tactical exit orchestration manager."""

from datetime import datetime, timezone

import pandas as pd

from src.config import TacticalExitConfig
from src.decision.tactical_exit_manager import TacticalExitManager, WriteBudgetSnapshot
from src.decision.tactical_exit_rules import TacticalExitSnapshot


def _make_healthy_buy_5min_bars(n: int = 50) -> pd.DataFrame:
    """Create a steady bullish 5m trend."""
    data: list[dict[str, float]] = []
    base = 1.1000
    for i in range(n):
        open_price = base + i * 0.00015
        close_price = open_price + 0.00010
        data.append(
            {
                "open": open_price,
                "high": close_price + 0.00008,
                "low": open_price - 0.00005,
                "close": close_price,
            }
        )
    return pd.DataFrame(data)


def _make_weakening_buy_5min_bars() -> pd.DataFrame:
    """Create a profitable long that ends with a strong bearish candle."""
    data = _make_healthy_buy_5min_bars(49).to_dict("records")
    last_open = float(data[-1]["close"]) + 0.00005
    data.append(
        {
            "open": last_open,
            "high": last_open + 0.00010,
            "low": last_open - 0.00060,
            "close": last_open - 0.00045,
        }
    )
    return pd.DataFrame(data)


def _make_downtrend_5min_bars(n: int = 50) -> pd.DataFrame:
    """Create a clean bearish 5m reversal."""
    data: list[dict[str, float]] = []
    base = 1.1080
    for i in range(n):
        open_price = base - i * 0.00018
        close_price = open_price - 0.00012
        data.append(
            {
                "open": open_price,
                "high": open_price + 0.00005,
                "low": close_price - 0.00008,
                "close": close_price,
            }
        )
    return pd.DataFrame(data)


def _make_trending_1h_bars(n: int = 40) -> pd.DataFrame:
    """Create 1h bars with stable ATR."""
    data: list[dict[str, float]] = []
    base = 1.0950
    for i in range(n):
        open_price = base + i * 0.00035
        close_price = open_price + 0.00025
        data.append(
            {
                "open": open_price,
                "high": close_price + 0.00080,
                "low": open_price - 0.00075,
                "close": close_price,
            }
        )
    return pd.DataFrame(data)


def _make_snapshot(
    *,
    current_price: float = 1.1035,
    sl_price: float | None = 1.0980,
    unrealized_r: float = 0.35,
    partial_close_done: bool = False,
    bars_5min: pd.DataFrame | None = None,
    bars_1h: pd.DataFrame | None = None,
    last_tactical_exit_action: str = "",
    last_tactical_exit_at: datetime | None = None,
) -> TacticalExitSnapshot:
    """Build a tactical exit snapshot for manager tests."""
    return TacticalExitSnapshot(
        position_id="POS-1",
        symbol="EURUSD",
        side="BUY",
        open_price=1.1000,
        current_price=current_price,
        volume=0.10,
        sl_price=sl_price,
        tp_price=1.1080,
        original_sl_price=1.0980,
        original_tp_price=1.1080,
        unrealized_r=unrealized_r,
        partial_close_done=partial_close_done,
        bars_5min=bars_5min if bars_5min is not None else pd.DataFrame(),
        bars_1h=bars_1h if bars_1h is not None else pd.DataFrame(),
        last_tactical_exit_action=last_tactical_exit_action,
        last_tactical_exit_at=last_tactical_exit_at,
    )


def test_evaluate_blocks_noncritical_write_when_budget_is_low() -> None:
    """Critical write-budget should suppress non-emergency tactical writes."""
    manager = TacticalExitManager(TacticalExitConfig())
    snapshot = _make_snapshot(
        current_price=1.1060,
        unrealized_r=1.1,
        bars_5min=_make_weakening_buy_5min_bars(),
        bars_1h=_make_trending_1h_bars(),
    )

    evaluation = manager.evaluate_position(
        snapshot=snapshot,
        budget=WriteBudgetSnapshot(write_remaining=20, daily_write_limit=2000),
        now=datetime.now(timezone.utc),
    )

    assert evaluation.decision.action == "HOLD"
    assert evaluation.skip_reason == "write_budget_blocked"


def test_evaluate_respects_modify_cooldown() -> None:
    """Repeated modify actions inside cooldown should be held back."""
    manager = TacticalExitManager(TacticalExitConfig())
    snapshot = _make_snapshot(
        last_tactical_exit_action="MOVE_TO_BREAKEVEN",
        last_tactical_exit_at=datetime(2026, 3, 12, 12, 4, tzinfo=timezone.utc),
    )

    evaluation = manager.evaluate_position(
        snapshot=snapshot,
        budget=WriteBudgetSnapshot(write_remaining=400, daily_write_limit=2000),
        now=datetime(2026, 3, 12, 12, 5, tzinfo=timezone.utc),
    )

    assert evaluation.decision.action == "HOLD"
    assert evaluation.skip_reason == "modify_cooldown"


def test_conflicting_exit_sets_llm_exception_flag() -> None:
    """Severe tactical conflict should mark the evaluation for LLM exception review."""
    manager = TacticalExitManager(TacticalExitConfig(use_llm_exception_path=True))
    snapshot = _make_snapshot(
        current_price=1.0985,
        unrealized_r=0.9,
        bars_5min=_make_downtrend_5min_bars(),
        bars_1h=_make_trending_1h_bars(),
    )

    evaluation = manager.evaluate_position(
        snapshot=snapshot,
        budget=WriteBudgetSnapshot(write_remaining=400, daily_write_limit=2000),
        now=datetime.now(timezone.utc),
    )

    assert evaluation.decision.action == "EXIT_NOW"
    assert evaluation.requires_llm_exception_review is True


def test_tp_reprice_uses_dedicated_cooldown_window() -> None:
    """Take-profit repricing should honor its own cooldown setting."""
    manager = TacticalExitManager(TacticalExitConfig())
    snapshot = _make_snapshot(
        current_price=1.1060,
        sl_price=1.1058,
        unrealized_r=1.4,
        bars_5min=_make_healthy_buy_5min_bars(),
        bars_1h=_make_trending_1h_bars(),
        last_tactical_exit_action="REPRICE_TP",
        last_tactical_exit_at=datetime(2026, 3, 12, 12, 0, tzinfo=timezone.utc),
    )

    evaluation = manager.evaluate_position(
        snapshot=snapshot,
        budget=WriteBudgetSnapshot(write_remaining=400, daily_write_limit=2000),
        now=datetime(2026, 3, 12, 12, 5, tzinfo=timezone.utc),
    )

    assert evaluation.decision.action == "HOLD"
    assert evaluation.skip_reason == "tp_reprice_cooldown"


def test_resolve_exit_config_returns_global_for_unknown_symbol() -> None:
    """Symbols without overrides should get the global config."""
    from src.config import SymbolTacticalOverride

    config = TacticalExitConfig(defensive_exit_loss_r=-0.35)
    overrides = {"EURCHF": SymbolTacticalOverride(defensive_exit_loss_r=-0.50)}
    manager = TacticalExitManager(config, symbol_overrides=overrides)

    resolved = manager.resolve_exit_config("AUDJPY")
    assert resolved.defensive_exit_loss_r == -0.35


def test_resolve_exit_config_applies_symbol_override() -> None:
    """EURCHF should get its overridden defensive_exit_loss_r."""
    from src.config import SymbolTacticalOverride

    config = TacticalExitConfig(
        defensive_exit_loss_r=-0.35,
        defensive_exit_min_hold_seconds=300,
    )
    overrides = {
        "EURCHF": SymbolTacticalOverride(
            defensive_exit_loss_r=-0.50,
            defensive_exit_min_hold_seconds=600,
        ),
    }
    manager = TacticalExitManager(config, symbol_overrides=overrides)

    resolved = manager.resolve_exit_config("EURCHF")
    assert resolved.defensive_exit_loss_r == -0.50
    assert resolved.defensive_exit_min_hold_seconds == 600
    # Other fields unchanged
    assert resolved.breakeven_activation_r == config.breakeven_activation_r


def test_resolve_exit_config_partial_override_keeps_defaults() -> None:
    """Override only one field; the rest should come from the global config."""
    from src.config import SymbolTacticalOverride

    config = TacticalExitConfig(
        defensive_exit_loss_r=-0.35,
        defensive_exit_min_hold_seconds=300,
    )
    overrides = {
        "EURCHF": SymbolTacticalOverride(defensive_exit_loss_r=-0.50),
    }
    manager = TacticalExitManager(config, symbol_overrides=overrides)

    resolved = manager.resolve_exit_config("EURCHF")
    assert resolved.defensive_exit_loss_r == -0.50
    assert resolved.defensive_exit_min_hold_seconds == 300  # From global


def test_evaluate_position_uses_symbol_override() -> None:
    """evaluate_position with symbol= should use resolved config."""
    from src.config import SymbolTacticalOverride

    config = TacticalExitConfig(
        defensive_exit_loss_r=-0.35,
        defensive_exit_min_hold_seconds=0,
        defensive_exit_require_strong_candle=True,
    )
    overrides = {
        "EURCHF": SymbolTacticalOverride(defensive_exit_loss_r=-0.50),
    }
    manager = TacticalExitManager(config, symbol_overrides=overrides)

    snapshot = TacticalExitSnapshot(
        position_id="POS-EURCHF",
        symbol="EURCHF",
        side="SELL",
        open_price=0.9400,
        current_price=0.9420,
        volume=0.10,
        sl_price=0.9450,
        tp_price=0.9350,
        original_sl_price=0.9450,
        original_tp_price=0.9350,
        unrealized_r=-0.40,  # Between -0.35 (global) and -0.50 (override)
        partial_close_done=False,
        hold_seconds=600,
        bars_5min=_make_healthy_buy_5min_bars(),  # Bullish bars = opposing for SELL
    )
    budget = WriteBudgetSnapshot(write_remaining=100, daily_write_limit=200)
    now = datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc)

    # With EURCHF override: -0.40 > -0.50, so IRSF should NOT trigger
    eval_eurchf = manager.evaluate_position(snapshot, budget, now, symbol="EURCHF")
    assert eval_eurchf.decision.reason != "initial_risk_structure_failure"

    # Without symbol (global config): -0.40 < -0.35, so IRSF should trigger
    eval_default = manager.evaluate_position(snapshot, budget, now)
    assert eval_default.decision.reason == "initial_risk_structure_failure"
