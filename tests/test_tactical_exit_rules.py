"""Tests for tactical exit pure-rule engine."""

from datetime import datetime, timezone

import pandas as pd

from src.config import TacticalExitConfig
from src.decision.tactical_exit_rules import (
    TacticalExitSnapshot,
    calculate_dynamic_take_profit,
    calculate_trailing_stop,
    choose_tactical_exit,
)


def _make_healthy_buy_5min_bars(n: int = 50) -> pd.DataFrame:
    """Create a bullish 5m structure with steady continuation."""
    data: list[dict[str, float]] = []
    base = 1.1000
    for i in range(n):
        open_price = base + i * 0.00015
        close_price = open_price + 0.00010
        high_price = close_price + 0.00008
        low_price = open_price - 0.00005
        data.append(
            {
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
            }
        )
    return pd.DataFrame(data)


def _make_weakening_buy_5min_bars() -> pd.DataFrame:
    """Create an uptrend that weakens with a sharp bearish closing candle."""
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


def _make_failed_buy_5min_bars(n: int = 50) -> pd.DataFrame:
    """Create a bearish 5m structure with a strong opposing close."""
    data: list[dict[str, float]] = []
    base = 1.1080
    for i in range(n - 1):
        open_price = base - i * 0.00018
        close_price = open_price - 0.00012
        high_price = open_price + 0.00006
        low_price = close_price - 0.00006
        data.append(
            {
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
            }
        )

    last_open = float(data[-1]["close"]) + 0.00008
    data.append(
        {
            "open": last_open,
            "high": last_open + 0.00005,
            "low": last_open - 0.00075,
            "close": last_open - 0.00060,
        }
    )
    return pd.DataFrame(data)


def _make_trending_1h_bars(n: int = 40) -> pd.DataFrame:
    """Create 1h bars with stable ATR and clear upward drift."""
    data: list[dict[str, float]] = []
    base = 1.0950
    for i in range(n):
        open_price = base + i * 0.00035
        close_price = open_price + 0.00025
        high_price = close_price + 0.00080
        low_price = open_price - 0.00075
        data.append(
            {
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
            }
        )
    return pd.DataFrame(data)


def _make_snapshot(
    *,
    current_price: float = 1.1035,
    sl_price: float | None = 1.0980,
    tp_price: float | None = 1.1080,
    unrealized_r: float = 0.35,
    partial_close_done: bool = False,
    hold_seconds: int | None = None,
    bars_5min: pd.DataFrame | None = None,
    bars_1h: pd.DataFrame | None = None,
) -> TacticalExitSnapshot:
    """Create a tactical exit snapshot with sane defaults for BUY tests."""
    return TacticalExitSnapshot(
        position_id="POS-1",
        symbol="EURUSD",
        side="BUY",
        open_price=1.1000,
        current_price=current_price,
        volume=0.10,
        sl_price=sl_price,
        tp_price=tp_price,
        original_sl_price=1.0980,
        original_tp_price=1.1080,
        unrealized_r=unrealized_r,
        hold_seconds=hold_seconds,
        partial_close_done=partial_close_done,
        bars_5min=bars_5min if bars_5min is not None else pd.DataFrame(),
        bars_1h=bars_1h if bars_1h is not None else pd.DataFrame(),
        last_tactical_exit_at=datetime(2026, 3, 12, 8, 0, tzinfo=timezone.utc),
    )


def test_choose_move_to_breakeven_at_protection_threshold() -> None:
    """Open profit above threshold should trigger the minimum protection layer."""
    decision = choose_tactical_exit(_make_snapshot(), TacticalExitConfig())

    assert decision.action == "MOVE_TO_BREAKEVEN"
    assert decision.state == "PROTECTION"
    assert decision.new_sl == 1.1000


def test_calculate_trailing_stop_never_widens_existing_sl() -> None:
    """Trailing-stop calculation should return no change if it would loosen risk."""
    snapshot = _make_snapshot(
        current_price=1.1065,
        sl_price=1.1045,
        unrealized_r=1.3,
        bars_5min=_make_healthy_buy_5min_bars(),
        bars_1h=_make_trending_1h_bars(),
    )

    candidate = calculate_trailing_stop(snapshot, TacticalExitConfig())

    assert candidate is None


def test_choose_partial_close_once_in_profit_protection() -> None:
    """Weakening profitable trades should partial-close before stop tightening."""
    snapshot = _make_snapshot(
        current_price=1.1060,
        unrealized_r=1.1,
        partial_close_done=False,
        bars_5min=_make_weakening_buy_5min_bars(),
        bars_1h=_make_trending_1h_bars(),
    )

    decision = choose_tactical_exit(snapshot, TacticalExitConfig())

    assert decision.action == "PARTIAL_CLOSE"
    assert decision.state == "PROFIT_PROTECTION"
    assert decision.partial_close_ratio == 0.5


def test_calculate_dynamic_take_profit_extends_target_in_trend_extension() -> None:
    """Healthy trend mode should propose a take-profit farther than the current TP."""
    snapshot = _make_snapshot(
        current_price=1.1060,
        sl_price=1.1058,
        unrealized_r=1.4,
        bars_5min=_make_healthy_buy_5min_bars(),
        bars_1h=_make_trending_1h_bars(),
    )

    candidate = calculate_dynamic_take_profit(
        snapshot,
        TacticalExitConfig(),
        state="TREND_EXTENSION",
    )

    assert candidate is not None
    assert candidate > 1.1080


def test_choose_defensive_exit_when_initial_risk_structure_fails() -> None:
    """Initial-risk positions should exit when short-term structure clearly breaks."""
    snapshot = _make_snapshot(
        current_price=1.0970,
        unrealized_r=-0.5,
        bars_5min=_make_failed_buy_5min_bars(),
        bars_1h=_make_trending_1h_bars(),
    )

    decision = choose_tactical_exit(snapshot, TacticalExitConfig(defensive_exit_loss_r=-0.35))

    assert decision.action == "EXIT_NOW"
    assert decision.state == "INITIAL_RISK"
    assert decision.reason == "initial_risk_structure_failure"


def test_no_defensive_exit_without_full_failure_confirmation() -> None:
    """Defensive exit should stay off when loss or structure confirmation is insufficient."""
    snapshot = _make_snapshot(
        current_price=1.0994,
        unrealized_r=-0.2,
        bars_5min=_make_weakening_buy_5min_bars(),
        bars_1h=_make_trending_1h_bars(),
    )

    decision = choose_tactical_exit(snapshot, TacticalExitConfig(defensive_exit_loss_r=-0.35))

    assert decision.action != "EXIT_NOW"


def test_severe_reversal_short_hold_falls_back_to_profit_protection() -> None:
    """Severe reversal should not force-exit before minimum hold time is reached."""
    snapshot = _make_snapshot(
        current_price=1.1060,
        unrealized_r=1.1,
        hold_seconds=120,
        partial_close_done=False,
        bars_5min=_make_failed_buy_5min_bars(),
        bars_1h=_make_trending_1h_bars(),
    )

    decision = choose_tactical_exit(snapshot, TacticalExitConfig())

    assert decision.action == "PARTIAL_CLOSE"
    assert decision.state == "PROFIT_PROTECTION"
    assert decision.reason == "profit_protection_partial_close"


def test_severe_reversal_insufficient_r_falls_back_to_protection() -> None:
    """Severe reversal should not force-exit if unrealized R is below threshold."""
    snapshot = _make_snapshot(
        current_price=1.1035,
        unrealized_r=0.35,
        hold_seconds=3600,
        bars_5min=_make_failed_buy_5min_bars(),
        bars_1h=_make_trending_1h_bars(),
    )

    decision = choose_tactical_exit(snapshot, TacticalExitConfig())

    assert decision.action == "MOVE_TO_BREAKEVEN"
    assert decision.state == "PROTECTION"
    assert decision.reason == "breakeven_threshold_reached"


def test_severe_reversal_with_sufficient_hold_and_r_exits_now() -> None:
    """Severe reversal should still force-exit when hold and R thresholds are met."""
    snapshot = _make_snapshot(
        current_price=1.1060,
        unrealized_r=1.1,
        hold_seconds=3600,
        bars_5min=_make_failed_buy_5min_bars(),
        bars_1h=_make_trending_1h_bars(),
    )

    decision = choose_tactical_exit(snapshot, TacticalExitConfig())

    assert decision.action == "EXIT_NOW"
    assert decision.state == "PROFIT_PROTECTION"
    assert decision.reason == "severe_tactical_reversal"
