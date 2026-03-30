"""
Tactical exit rule engine for open-position management.

Provides a deterministic state machine for tactical exits so scheduler and
execution code can remain thin orchestration layers.

Usage:
    snapshot = TacticalExitSnapshot(...)
    decision = choose_tactical_exit(snapshot, config)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, cast

import pandas as pd

from src.config import TacticalExitConfig
from src.decision.tactical_validator import compute_atr, compute_ema, compute_rsi

TacticalExitState = Literal[
    "INITIAL_RISK",
    "PROTECTION",
    "TREND_EXTENSION",
    "PROFIT_PROTECTION",
]
TacticalExitAction = Literal[
    "HOLD",
    "MOVE_TO_BREAKEVEN",
    "TRAIL_SL",
    "REPRICE_TP",
    "PARTIAL_CLOSE",
    "EXIT_NOW",
]


# ── Data Types ─────────────────────────────────────────────────────────────


@dataclass
class TacticalExitSnapshot:
    """Pure input bundle for tactical exit rule evaluation."""

    position_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    open_price: float
    current_price: float
    volume: float
    sl_price: float | None
    tp_price: float | None
    original_sl_price: float | None
    original_tp_price: float | None
    unrealized_r: float
    partial_close_done: bool
    hold_seconds: int | None = None
    bars_5min: pd.DataFrame = field(default_factory=pd.DataFrame)
    bars_1h: pd.DataFrame = field(default_factory=pd.DataFrame)
    prior_trailing_sl: float | None = None
    last_tactical_exit_action: str = ""
    last_tactical_exit_at: datetime | None = None


@dataclass
class TacticalExitDecision:
    """Deterministic output of tactical exit rule evaluation."""

    action: TacticalExitAction
    state: TacticalExitState
    reason: str
    new_sl: float | None = None
    new_tp: float | None = None
    partial_close_ratio: float | None = None
    requires_llm_exception: bool = False


@dataclass
class ExitSignalContext:
    """Derived indicator context used by the tactical exit state machine."""

    ema_aligned: bool | None = None
    adverse_rsi: bool = False
    weakening_rsi: bool = False
    opposing_candle: bool = False
    opposing_candle_strong: bool = False
    candle_body_ratio: float | None = None
    atr_1h: float | None = None
    atr_regime_ratio: float | None = None


# ── Indicator Helpers ──────────────────────────────────────────────────────


def _atr_regime_ratio(bars_1h: pd.DataFrame, period: int = 14) -> float | None:
    """Return current ATR divided by historical median ATR."""
    if bars_1h.empty or len(bars_1h) < period + 1:
        return None

    current_atr = compute_atr(bars_1h, period=period)
    if pd.isna(current_atr):
        return None

    all_atrs: list[float] = []
    for i in range(period + 1, len(bars_1h) + 1):
        atr_value = compute_atr(bars_1h.iloc[:i], period=period)
        if not pd.isna(atr_value):
            all_atrs.append(float(atr_value))

    if not all_atrs:
        return None

    median_atr = float(pd.Series(all_atrs).median())
    if median_atr <= 0:
        return None

    return float(current_atr) / median_atr


def _candle_context(snapshot: TacticalExitSnapshot) -> tuple[bool, bool, float | None]:
    """Return whether the latest candle opposes position direction and how strongly."""
    if snapshot.bars_5min.empty:
        return False, False, None

    last = snapshot.bars_5min.iloc[-1]
    bar_range = float(last["high"] - last["low"])
    body = abs(float(last["close"] - last["open"]))
    body_ratio = body / bar_range if bar_range > 0 else 0.0
    is_bearish = float(last["close"]) < float(last["open"])
    is_bullish = float(last["close"]) > float(last["open"])
    opposing = (snapshot.side == "BUY" and is_bearish) or (snapshot.side == "SELL" and is_bullish)
    return opposing, opposing and body_ratio > 0.3, body_ratio


def build_exit_signal_context(snapshot: TacticalExitSnapshot) -> ExitSignalContext:
    """Compute the signal bundle used by the exit state machine."""
    context = ExitSignalContext()

    if not snapshot.bars_5min.empty and len(snapshot.bars_5min) >= 26:
        closes = cast(pd.Series, snapshot.bars_5min["close"])
        ema_fast = float(compute_ema(closes, 8).iloc[-1])
        ema_slow = float(compute_ema(closes, 21).iloc[-1])
        context.ema_aligned = (
            ema_fast > ema_slow if snapshot.side == "BUY" else ema_fast < ema_slow
        )

        rsi = compute_rsi(closes, 14)
        if snapshot.side == "BUY":
            context.adverse_rsi = rsi < 45
            context.weakening_rsi = rsi < 50
        else:
            context.adverse_rsi = rsi > 55
            context.weakening_rsi = rsi > 50

    (
        context.opposing_candle,
        context.opposing_candle_strong,
        context.candle_body_ratio,
    ) = _candle_context(snapshot)

    if not snapshot.bars_1h.empty:
        atr_value = compute_atr(snapshot.bars_1h, period=14)
        context.atr_1h = None if pd.isna(atr_value) else float(atr_value)
        context.atr_regime_ratio = _atr_regime_ratio(snapshot.bars_1h, period=14)

    return context


# ── State Classification ───────────────────────────────────────────────────


def _is_severe_reversal(
    snapshot: TacticalExitSnapshot,
    context: ExitSignalContext,
    config: TacticalExitConfig,
) -> bool:
    """Return True when multiple tactical signals strongly contradict the position."""
    if context.ema_aligned is None:
        return False
    if (
        snapshot.hold_seconds is None
        or snapshot.hold_seconds < config.severe_reversal_min_hold_seconds
    ):
        return False
    return (
        snapshot.unrealized_r >= config.severe_reversal_min_r
        and context.ema_aligned is False
        and (context.adverse_rsi or context.opposing_candle_strong)
    )


def _should_defensive_exit_initial_risk(
    snapshot: TacticalExitSnapshot,
    context: ExitSignalContext,
    config: TacticalExitConfig,
) -> bool:
    """Return True when an initial-risk position shows clear structure failure."""
    if snapshot.unrealized_r > config.defensive_exit_loss_r:
        return False
    if context.ema_aligned is not False:
        return False
    if not context.adverse_rsi:
        return False
    if config.defensive_exit_require_strong_candle:
        return context.opposing_candle_strong
    return context.opposing_candle


def _is_trend_extension(context: ExitSignalContext) -> bool:
    """Return True when trend continuation still looks healthy."""
    has_positive_signal = (
        context.ema_aligned is not None
        or context.atr_regime_ratio is not None
        or context.candle_body_ratio is not None
    )
    if not has_positive_signal:
        return False

    atr_ok = context.atr_regime_ratio is None or 0.5 <= context.atr_regime_ratio <= 2.5
    ema_ok = context.ema_aligned is None or context.ema_aligned is True
    return atr_ok and ema_ok and not context.adverse_rsi and not context.opposing_candle_strong


def _is_profit_protection(
    snapshot: TacticalExitSnapshot,
    context: ExitSignalContext,
    config: TacticalExitConfig,
) -> bool:
    """Return True when the position is profitable but conditions have weakened."""
    if snapshot.unrealized_r < config.partial_close_min_r:
        return False

    atr_bad = context.atr_regime_ratio is not None and context.atr_regime_ratio > 1.35
    ema_bad = context.ema_aligned is False
    return bool(
        context.opposing_candle
        or context.weakening_rsi
        or atr_bad
        or ema_bad
    )


def classify_tactical_exit_state(
    snapshot: TacticalExitSnapshot,
    config: TacticalExitConfig,
    context: ExitSignalContext | None = None,
) -> TacticalExitState:
    """Classify the position into the tactical exit state machine."""
    context = context or build_exit_signal_context(snapshot)

    if snapshot.unrealized_r < config.breakeven_activation_r:
        return "INITIAL_RISK"
    if _is_profit_protection(snapshot, context, config):
        return "PROFIT_PROTECTION"
    if _is_trend_extension(context):
        return "TREND_EXTENSION"
    return "PROTECTION"


# ── Price Calculation Helpers ──────────────────────────────────────────────


def _reference_stop(snapshot: TacticalExitSnapshot) -> float | None:
    """Return the tightest stop reference already stored on the position."""
    if snapshot.prior_trailing_sl is not None:
        return snapshot.prior_trailing_sl
    if snapshot.sl_price is not None:
        return snapshot.sl_price
    return snapshot.original_sl_price


def _reference_tp(snapshot: TacticalExitSnapshot) -> float | None:
    """Return the current or original take-profit reference."""
    if snapshot.tp_price is not None:
        return snapshot.tp_price
    return snapshot.original_tp_price


def _stop_improves(snapshot: TacticalExitSnapshot, candidate: float | None) -> bool:
    """Return True when a stop candidate tightens risk without crossing the market."""
    if candidate is None:
        return False

    reference = _reference_stop(snapshot)
    if snapshot.side == "BUY":
        if reference is not None and candidate <= reference:
            return False
        return candidate < snapshot.current_price

    if reference is not None and candidate >= reference:
        return False
    return candidate > snapshot.current_price


def _tp_improves(
    snapshot: TacticalExitSnapshot,
    candidate: float | None,
    state: TacticalExitState,
) -> bool:
    """Return True when a TP candidate changes in the intended direction."""
    if candidate is None:
        return False

    reference = _reference_tp(snapshot)
    if reference is None:
        return False

    if snapshot.side == "BUY":
        if candidate <= snapshot.current_price:
            return False
        if state == "TREND_EXTENSION":
            return candidate > reference
        return candidate < reference

    if candidate >= snapshot.current_price:
        return False
    if state == "TREND_EXTENSION":
        return candidate < reference
    return candidate > reference


def calculate_trailing_stop(
    snapshot: TacticalExitSnapshot,
    config: TacticalExitConfig,
) -> float | None:
    """Calculate a tighter trailing stop, if available."""
    if snapshot.bars_5min.empty or snapshot.bars_1h.empty:
        return None

    atr_value = compute_atr(snapshot.bars_1h, period=14)
    if pd.isna(atr_value):
        return None

    recent_bars = snapshot.bars_5min.tail(12)
    atr_distance = float(atr_value) * config.atr_trailing_multiplier

    if snapshot.side == "BUY":
        recent_high = float(recent_bars["high"].max())
        candidate = recent_high - atr_distance
        if snapshot.unrealized_r >= config.breakeven_activation_r:
            candidate = max(candidate, snapshot.open_price)
    else:
        recent_low = float(recent_bars["low"].min())
        candidate = recent_low + atr_distance
        if snapshot.unrealized_r >= config.breakeven_activation_r:
            candidate = min(candidate, snapshot.open_price)

    if not _stop_improves(snapshot, candidate):
        return None
    return candidate


def calculate_dynamic_take_profit(
    snapshot: TacticalExitSnapshot,
    config: TacticalExitConfig,
    state: TacticalExitState,
) -> float | None:
    """Calculate a take-profit candidate for extension or contraction."""
    reference = _reference_tp(snapshot)
    if reference is None or snapshot.bars_1h.empty:
        return None

    atr_value = compute_atr(snapshot.bars_1h, period=14)
    if pd.isna(atr_value):
        return None

    atr_distance = float(atr_value)
    if state == "TREND_EXTENSION":
        multiplier = config.trend_extension_tp_atr_multiplier
        if snapshot.side == "BUY":
            candidate = max(reference, snapshot.current_price) + atr_distance * multiplier
        else:
            candidate = min(reference, snapshot.current_price) - atr_distance * multiplier
    else:
        multiplier = config.profit_protection_tp_atr_multiplier
        if snapshot.side == "BUY":
            candidate = min(reference, snapshot.current_price + atr_distance * multiplier)
        else:
            candidate = max(reference, snapshot.current_price - atr_distance * multiplier)

    if not _tp_improves(snapshot, candidate, state):
        return None
    return candidate


# ── State Machine ──────────────────────────────────────────────────────────


def choose_tactical_exit(
    snapshot: TacticalExitSnapshot,
    config: TacticalExitConfig,
) -> TacticalExitDecision:
    """Choose the highest-priority tactical exit action for a position."""
    context = build_exit_signal_context(snapshot)
    state = classify_tactical_exit_state(snapshot, config, context=context)

    if _is_severe_reversal(snapshot, context, config):
        return TacticalExitDecision(
            action="EXIT_NOW",
            state="PROFIT_PROTECTION",
            reason="severe_tactical_reversal",
            requires_llm_exception=True,
        )

    if state == "INITIAL_RISK" and _should_defensive_exit_initial_risk(snapshot, context, config):
        return TacticalExitDecision(
            action="EXIT_NOW",
            state=state,
            reason="initial_risk_structure_failure",
        )

    if state == "PROFIT_PROTECTION" and not snapshot.partial_close_done:
        return TacticalExitDecision(
            action="PARTIAL_CLOSE",
            state=state,
            reason="profit_protection_partial_close",
            partial_close_ratio=config.partial_close_ratio,
        )

    if state in {"TREND_EXTENSION", "PROFIT_PROTECTION"}:
        trailing_sl = calculate_trailing_stop(snapshot, config)
        if trailing_sl is not None:
            return TacticalExitDecision(
                action="TRAIL_SL",
                state=state,
                reason="atr_trailing_stop_improved",
                new_sl=trailing_sl,
            )

    if state == "PROTECTION" and _stop_improves(snapshot, snapshot.open_price):
        return TacticalExitDecision(
            action="MOVE_TO_BREAKEVEN",
            state=state,
            reason="breakeven_threshold_reached",
            new_sl=snapshot.open_price,
        )

    if state in {"TREND_EXTENSION", "PROFIT_PROTECTION"}:
        new_tp = calculate_dynamic_take_profit(snapshot, config, state=state)
        if new_tp is not None:
            return TacticalExitDecision(
                action="REPRICE_TP",
                state=state,
                reason="dynamic_take_profit_repriced",
                new_tp=new_tp,
            )

    return TacticalExitDecision(
        action="HOLD",
        state=state,
        reason="no_tactical_exit_action",
    )
