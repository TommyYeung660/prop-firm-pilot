"""
Tactical entry validator — low-timeframe confirmation for strategic decisions.

Sits between AgentBridge.decide() (strategic) and mark_ready_for_exec() (execution).
Uses 5min/1H data to validate entry timing without overriding strategic direction.
The tactical layer is the "soldier" — decides WHEN to execute, never WHAT direction.

Gate system:
- Hard Gates (ALL must pass): Spread, ATR Regime, Data Freshness
- Soft Gates (score ≥ min_score/3): EMA momentum, RSI state, Candle quality

Results: PASS (execute now), WAIT (retry in 5min), REJECT (expired after retries)

Usage:
    validator = TacticalValidator(config)
    result = await validator.validate(intent, side="SELL", tactical_data=data)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, cast

import pandas as pd
from loguru import logger

from src.config import TacticalConfig

# ── Data Types ─────────────────────────────────────────────────────────────


@dataclass
class GateResult:
    """Result of a single gate check."""

    gate_name: str
    passed: bool
    value: float | None = None
    threshold: str = ""
    detail: str = ""


@dataclass
class TacticalData:
    """Market data bundle for tactical validation.

    Populated by the caller (scheduler) from FxDataProvider and MatchTrader quotes.
    """

    bars_5min: pd.DataFrame = field(default_factory=pd.DataFrame)
    bars_1h: pd.DataFrame = field(default_factory=pd.DataFrame)
    current_spread: float = 0.0
    typical_spread: float = 0.0
    latest_bar_time: datetime | None = None


@dataclass
class TacticalResult:
    """Aggregate result of tactical validation."""

    action: Literal["PASS", "WAIT", "REJECT"]
    hard_gates: list[GateResult] = field(default_factory=list)
    soft_gates: list[GateResult] = field(default_factory=list)
    soft_score: int = 0
    soft_required: int = 2
    detail: str = ""

    def to_log_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSONL/DuckDB logging."""
        return {
            "action": self.action,
            "hard_gates": [
                {"gate": r.gate_name, "passed": r.passed, "value": r.value, "detail": r.detail}
                for r in self.hard_gates
            ],
            "soft_gates": [
                {"gate": r.gate_name, "passed": r.passed, "value": r.value, "detail": r.detail}
                for r in self.soft_gates
            ],
            "soft_score": self.soft_score,
            "soft_required": self.soft_required,
            "detail": self.detail,
        }


# ── Pure Indicator Functions ───────────────────────────────────────────────


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Compute Average True Range from OHLC DataFrame.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        period: Lookback period.

    Returns:
        Latest ATR value, or NaN if insufficient data.
    """
    if len(df) < period + 1:
        return float("nan")

    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    return float(atr.iloc[-1])


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Compute Exponential Moving Average.

    Args:
        series: Price series (typically close prices).
        period: EMA period.

    Returns:
        EMA series of same length as input.
    """
    ema = series.ewm(span=period, adjust=False).mean()
    return cast(pd.Series, ema)


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    """Compute Relative Strength Index.

    Args:
        series: Close price series.
        period: RSI lookback period.

    Returns:
        Latest RSI value (0-100).
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi_series = pd.Series(100 - (100 / (1 + rs)))
    return float(rsi_series.iloc[-1])


# ── TacticalValidator ──────────────────────────────────────────────────────


class TacticalValidator:
    """Tactical entry validator using Hard/Soft dual-layer gate system.

    Hard Gates (ALL must pass):
        1. Spread: current spread < multiplier × typical spread
        2. ATR Regime: min_ratio < current ATR / median ATR < max_ratio
        3. Data Freshness: latest bar age < max_age_seconds

    Soft Gates (score ≥ min_score out of 3):
        1. EMA Momentum: 5min EMA(8) vs EMA(21) alignment with trade direction
        2. RSI State: not in extreme zone opposing trade direction
        3. Candle Quality: latest bar body/range ratio > min threshold

    Usage:
        validator = TacticalValidator(config)
        result = validator.evaluate(side="SELL", tactical_data=data)
    """

    def __init__(self, config: TacticalConfig) -> None:
        self._config = config

    # ── Hard Gates ─────────────────────────────────────────────────────

    def _check_spread_gate(self, current_spread: float, typical_spread: float) -> GateResult:
        """Check if current spread is within acceptable range."""
        limit = self._config.hard_gates.spread_max_multiplier * typical_spread
        passed = current_spread < limit
        ratio = current_spread / typical_spread if typical_spread > 0 else float("inf")
        return GateResult(
            gate_name="spread",
            passed=passed,
            value=ratio,
            threshold=f"< {self._config.hard_gates.spread_max_multiplier}×",
            detail=(
                f"spread_ratio={ratio:.2f}, limit={self._config.hard_gates.spread_max_multiplier}×"
            ),
        )

    def _check_atr_regime_gate(self, current_atr: float, median_atr: float) -> GateResult:
        """Check if ATR is within normal regime (not dead market or extreme volatility)."""
        ratio = current_atr / median_atr if median_atr > 0 else float("inf")
        min_r = self._config.hard_gates.atr_min_ratio
        max_r = self._config.hard_gates.atr_max_ratio
        passed = min_r < ratio < max_r
        return GateResult(
            gate_name="atr_regime",
            passed=passed,
            value=ratio,
            threshold=f"{min_r}× < ATR ratio < {max_r}×",
            detail=f"atr_ratio={ratio:.2f}, range=[{min_r}, {max_r}]",
        )

    def _check_data_freshness_gate(self, latest_bar_time: datetime) -> GateResult:
        """Check if latest bar data is recent enough."""
        now = datetime.now(timezone.utc)
        age_seconds = (now - latest_bar_time).total_seconds()
        max_age = self._config.hard_gates.data_max_age_seconds
        passed = age_seconds < max_age
        return GateResult(
            gate_name="data_freshness",
            passed=passed,
            value=age_seconds,
            threshold=f"< {max_age}s",
            detail=f"age={age_seconds:.0f}s, max={max_age}s",
        )

    def check_hard_gates(self, data: TacticalData) -> list[GateResult]:
        """Run all hard gate checks. ALL must pass.

        Args:
            data: TacticalData bundle with spread, ATR bars, and bar timestamps.

        Returns:
            List of GateResult for each hard gate.
        """
        results: list[GateResult] = []

        # 1. Spread gate
        if data.typical_spread > 0 and data.current_spread > 0:
            results.append(self._check_spread_gate(data.current_spread, data.typical_spread))
        else:
            # v1.3.9: Pass-through when no spread data — don't block trades
            results.append(
                GateResult(
                    gate_name="spread",
                    passed=True,
                    detail="Spread gate skipped — no spread data available (pass-through)",
                )
            )

        # 2. ATR regime gate
        if not data.bars_1h.empty:
            current_atr = compute_atr(data.bars_1h, period=self._config.hard_gates.atr_period)
            # Median ATR over the full 1H dataset
            if not pd.isna(current_atr) and len(data.bars_1h) > self._config.hard_gates.atr_period:
                all_atrs: list[Any] = []
                period = self._config.hard_gates.atr_period
                for i in range(period + 1, len(data.bars_1h) + 1):
                    window = data.bars_1h.iloc[:i]
                    a = compute_atr(window, period=period)
                    if not pd.isna(a):
                        all_atrs.append(a)
                median_atr = float(pd.Series(all_atrs).median()) if all_atrs else current_atr
                results.append(self._check_atr_regime_gate(current_atr, median_atr))
            else:
                results.append(
                    GateResult(
                        gate_name="atr_regime",
                        passed=False,
                        detail="Insufficient 1H data for ATR calculation",
                    )
                )
        else:
            # v1.3.8: Pass-through when no 1H data — don't block trades due to missing data
            results.append(
                GateResult(
                    gate_name="atr_regime",
                    passed=True,
                    detail="ATR gate skipped — no 1H bar data available (pass-through)",
                )
            )

        # 3. Data freshness gate
        if data.latest_bar_time is not None:
            results.append(self._check_data_freshness_gate(data.latest_bar_time))
        else:
            results.append(
                GateResult(
                    gate_name="data_freshness",
                    passed=False,
                    detail="No bar timestamp available",
                )
            )

        return results

    # ── Soft Gates ─────────────────────────────────────────────────────

    def _check_ema_momentum_gate(
        self, side: Literal["BUY", "SELL"], bars_5min: pd.DataFrame
    ) -> GateResult:
        """Check if short-term momentum aligns with strategic direction."""
        if bars_5min.empty or len(bars_5min) < self._config.soft_gates.ema_slow + 5:
            # v1.3.8: Pass-through when insufficient data — don't penalize for missing bars
            return GateResult(
                gate_name="ema_momentum",
                passed=True,
                detail="EMA gate skipped — insufficient 5min data (pass-through)",
            )

        closes = cast(pd.Series, bars_5min["close"])
        fast = compute_ema(closes, self._config.soft_gates.ema_fast)
        slow = compute_ema(closes, self._config.soft_gates.ema_slow)

        fast_val = float(fast.iloc[-1])
        slow_val = float(slow.iloc[-1])

        if side == "BUY":
            passed = fast_val > slow_val
        else:
            passed = fast_val < slow_val

        return GateResult(
            gate_name="ema_momentum",
            passed=passed,
            value=fast_val - slow_val,
            threshold=(
                f"EMA({self._config.soft_gates.ema_fast}) "
                f"{'>' if side == 'BUY' else '<'} "
                f"EMA({self._config.soft_gates.ema_slow})"
            ),
            detail=(f"fast={fast_val:.5f}, slow={slow_val:.5f}, diff={fast_val - slow_val:.5f}"),
        )

    def _check_rsi_state_gate(
        self, side: Literal["BUY", "SELL"], bars_5min: pd.DataFrame
    ) -> GateResult:
        """Check if RSI is not in extreme zone opposing the trade direction."""
        if bars_5min.empty or len(bars_5min) < self._config.soft_gates.rsi_period + 5:
            # v1.3.8: Pass-through when insufficient data — don't penalize for missing bars
            return GateResult(
                gate_name="rsi_state",
                passed=True,
                detail="RSI gate skipped — insufficient 5min data (pass-through)",
            )

        closes = cast(pd.Series, bars_5min["close"])
        rsi = compute_rsi(closes, self._config.soft_gates.rsi_period)

        rsi_limit = (
            self._config.soft_gates.rsi_overbought
            if side == "BUY"
            else self._config.soft_gates.rsi_oversold
        )

        if side == "BUY":
            passed = rsi < rsi_limit
        else:
            passed = rsi > rsi_limit

        return GateResult(
            gate_name="rsi_state",
            passed=passed,
            value=rsi,
            threshold=f"RSI {'<' if side == 'BUY' else '>'} {rsi_limit}",
            detail=f"rsi={rsi:.1f}",
        )

    def _check_candle_quality_gate(self, bars_5min: pd.DataFrame) -> GateResult:
        """Check if latest candle has sufficient directional quality."""
        if bars_5min.empty:
            return GateResult(
                gate_name="candle_quality",
                passed=False,
                detail="No 5min bar data",
            )

        last = bars_5min.iloc[-1]
        bar_range = last["high"] - last["low"]
        if bar_range == 0:
            body_ratio = 0.0
        else:
            body_ratio = abs(last["close"] - last["open"]) / bar_range

        passed = bool(body_ratio > self._config.soft_gates.candle_min_body_ratio)
        return GateResult(
            gate_name="candle_quality",
            passed=passed,
            value=body_ratio,
            threshold=f"> {self._config.soft_gates.candle_min_body_ratio}",
            detail=f"body_ratio={body_ratio:.3f}",
        )

    def check_soft_gates(
        self,
        side: Literal["BUY", "SELL"],
        data: TacticalData,
    ) -> list[GateResult]:
        """Run all soft gate checks. Score-based: min_score out of 3 must pass."""
        bars = data.bars_5min
        return [
            self._check_ema_momentum_gate(side, bars),
            self._check_rsi_state_gate(side, bars),
            self._check_candle_quality_gate(bars),
        ]

    # ── Evaluate (Full pipeline) ──────────────────────────────────

    def evaluate(
        self,
        side: Literal["BUY", "SELL"],
        data: TacticalData,
    ) -> TacticalResult:
        """Run full tactical validation (Hard + Soft gates).

        Args:
            side: Strategic direction from LLM decision.
            data: TacticalData bundle.

        Returns:
            TacticalResult with action (PASS/WAIT/REJECT) and gate details.
        """
        # ── No-data guard: reject when ALL tactical signals are missing ──
        has_spread = data.current_spread > 0 or data.typical_spread > 0
        has_bars = not data.bars_5min.empty or not data.bars_1h.empty
        has_freshness = data.latest_bar_time is not None
        if not has_spread and not has_bars and not has_freshness:
            logger.warning("Tactical REJECT: no data — spread, bars, freshness all missing")
            return TacticalResult(
                action="REJECT",
                detail="No tactical data available — spread, bars, and freshness all missing",
            )

        hard_results = self.check_hard_gates(data)
        hard_passed = all(r.passed for r in hard_results)

        if not hard_passed:
            failed_gates = [r.gate_name for r in hard_results if not r.passed]
            logger.debug("Tactical WAIT: hard gates failed: {}", ", ".join(failed_gates))
            return TacticalResult(
                action="WAIT",
                hard_gates=hard_results,
                detail=f"Hard gates failed: {', '.join(failed_gates)}",
            )

        soft_results = self.check_soft_gates(side, data)
        soft_score = sum(1 for r in soft_results if r.passed)
        soft_required = self._config.soft_gates.min_score

        if soft_score >= soft_required:
            return TacticalResult(
                action="PASS",
                hard_gates=hard_results,
                soft_gates=soft_results,
                soft_score=soft_score,
                soft_required=soft_required,
                detail=(
                    f"All hard gates passed, soft score "
                    f"{soft_score}/{len(soft_results)} ≥ {soft_required}"
                ),
            )
        else:
            return TacticalResult(
                action="WAIT",
                hard_gates=hard_results,
                soft_gates=soft_results,
                soft_score=soft_score,
                soft_required=soft_required,
                detail=(f"Soft score {soft_score}/{len(soft_results)} < {soft_required}"),
            )
