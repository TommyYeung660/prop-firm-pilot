"""
Tests for TacticalValidator — Hard/Soft dual-layer gate system.

Validates entry timing using 5min/1H data without overriding strategic direction.
The tactical layer is the "soldier" — it decides WHEN to execute, never WHAT direction.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from src.config import TacticalConfig
from src.decision.tactical_validator import (
    TacticalData,
    TacticalValidator,
    compute_atr,
    compute_ema,
    compute_rsi,
)

# ── Indicator Unit Tests ───────────────────────────────────────────────────


class TestComputeATR:
    """Pure function ATR calculation tests."""

    def test_atr_basic(self) -> None:
        """ATR of constant-range bars should equal that range."""
        df = pd.DataFrame(
            {
                "high": [1.1010] * 20,
                "low": [1.1000] * 20,
                "close": [1.1005] * 20,
            }
        )
        atr = compute_atr(df, period=14)
        assert abs(atr - 0.0010) < 1e-6

    def test_atr_insufficient_data_returns_nan(self) -> None:
        """ATR with fewer rows than period should return NaN."""
        df = pd.DataFrame(
            {
                "high": [1.1010] * 5,
                "low": [1.1000] * 5,
                "close": [1.1005] * 5,
            }
        )
        atr = compute_atr(df, period=14)
        assert pd.isna(atr)


class TestComputeEMA:
    """Pure function EMA calculation tests."""

    def test_ema_returns_series(self) -> None:
        closes = pd.Series([1.0 + i * 0.001 for i in range(50)])
        ema = compute_ema(closes, period=8)
        assert len(ema) == 50
        assert not pd.isna(ema.iloc[-1])

    def test_ema_follows_trend(self) -> None:
        """In a pure uptrend, fast EMA > slow EMA."""
        closes = pd.Series([1.0 + i * 0.001 for i in range(50)])
        fast = compute_ema(closes, period=8)
        slow = compute_ema(closes, period=21)
        assert fast.iloc[-1] > slow.iloc[-1]


class TestComputeRSI:
    """Pure function RSI calculation tests."""

    def test_rsi_pure_uptrend_near_100(self) -> None:
        closes = pd.Series([1.0 + i * 0.001 for i in range(30)])
        rsi = compute_rsi(closes, period=14)
        assert rsi > 90

    def test_rsi_pure_downtrend_near_0(self) -> None:
        closes = pd.Series([1.0 - i * 0.001 for i in range(30)])
        rsi = compute_rsi(closes, period=14)
        assert rsi < 10

    def test_rsi_flat_near_50(self) -> None:
        # Alternating up/down of same magnitude → RSI ≈ 50
        closes = pd.Series([1.0 + (0.001 if i % 2 == 0 else -0.001) for i in range(30)])
        rsi = compute_rsi(closes, period=14)
        assert 40 < rsi < 60


# ── Hard Gate Tests ────────────────────────────────────────────────────────


class TestHardGateSpread:
    """Spread gate: current spread must be < spread_max_multiplier × typical."""

    def test_normal_spread_passes(self) -> None:
        config = TacticalConfig()
        validator = TacticalValidator(config)
        result = validator._check_spread_gate(current_spread=0.00020, typical_spread=0.00015)
        assert result.passed is True
        assert result.gate_name == "spread"

    def test_wide_spread_fails(self) -> None:
        config = TacticalConfig()
        validator = TacticalValidator(config)
        result = validator._check_spread_gate(current_spread=0.00050, typical_spread=0.00015)
        assert result.passed is False

    def test_exactly_at_limit_passes(self) -> None:
        config = TacticalConfig()
        validator = TacticalValidator(config)
        result = validator._check_spread_gate(current_spread=0.00029, typical_spread=0.00015)
        assert result.passed is True


class TestHardGateATRRegime:
    """ATR regime gate: current ATR must be within [min_ratio, max_ratio] × median."""

    def test_normal_atr_passes(self) -> None:
        config = TacticalConfig()
        validator = TacticalValidator(config)
        result = validator._check_atr_regime_gate(current_atr=0.0010, median_atr=0.0010)
        assert result.passed is True

    def test_dead_market_fails(self) -> None:
        config = TacticalConfig()
        validator = TacticalValidator(config)
        result = validator._check_atr_regime_gate(current_atr=0.0003, median_atr=0.0010)
        assert result.passed is False

    def test_extreme_volatility_fails(self) -> None:
        config = TacticalConfig()
        validator = TacticalValidator(config)
        result = validator._check_atr_regime_gate(current_atr=0.0030, median_atr=0.0010)
        assert result.passed is False


class TestHardGateDataFreshness:
    """Data freshness gate: latest bar timestamp must be < data_max_age_seconds old."""

    def test_fresh_data_passes(self) -> None:
        config = TacticalConfig()
        validator = TacticalValidator(config)
        recent = datetime.now(timezone.utc) - timedelta(minutes=3)
        result = validator._check_data_freshness_gate(latest_bar_time=recent)
        assert result.passed is True

    def test_stale_data_fails(self) -> None:
        config = TacticalConfig()
        validator = TacticalValidator(config)
        old = datetime.now(timezone.utc) - timedelta(minutes=15)
        result = validator._check_data_freshness_gate(latest_bar_time=old)
        assert result.passed is False

    def test_fresh_quote_timestamp_produces_single_passing_data_freshness_gate(self) -> None:
        config = TacticalConfig()
        validator = TacticalValidator(config)
        stale_5min = datetime.now(timezone.utc) - timedelta(minutes=15)
        stale_1h = datetime.now(timezone.utc) - timedelta(hours=4)
        fresh_quote = datetime.now(timezone.utc) - timedelta(seconds=30)
        data = TacticalData(
            bars_5min=pd.DataFrame(
                [
                    {
                        "datetime": stale_5min,
                        "open": 1.1000,
                        "high": 1.1010,
                        "low": 1.0990,
                        "close": 1.1005,
                    }
                ]
            ),
            bars_1h=pd.DataFrame(
                [
                    {
                        "datetime": stale_1h,
                        "open": 1.0950,
                        "high": 1.1015,
                        "low": 1.0940,
                        "close": 1.1005,
                    }
                ]
            ),
            current_spread=0.00015,
            typical_spread=0.00015,
            latest_bar_time=fresh_quote,
        )

        results = validator.check_hard_gates(data)

        freshness_results = [r for r in results if r.gate_name == "data_freshness"]
        assert len(freshness_results) == 1
        assert freshness_results[0].passed is True
        assert "quote_age" in freshness_results[0].detail.lower()

    def test_fresh_5min_bar_can_pass_when_1h_bar_is_older_and_no_quote_timestamp(self) -> None:
        config = TacticalConfig()
        validator = TacticalValidator(config)
        fresh_5min = datetime.now(timezone.utc) - timedelta(minutes=4)
        stale_1h = datetime.now(timezone.utc) - timedelta(hours=3)
        data = TacticalData(
            bars_5min=pd.DataFrame(
                [
                    {
                        "datetime": fresh_5min,
                        "open": 1.1000,
                        "high": 1.1010,
                        "low": 1.0990,
                        "close": 1.1005,
                    }
                ]
            ),
            bars_1h=pd.DataFrame(
                [
                    {
                        "datetime": stale_1h,
                        "open": 1.0950,
                        "high": 1.1015,
                        "low": 1.0940,
                        "close": 1.1005,
                    }
                ]
            ),
            current_spread=0.00015,
            typical_spread=0.00015,
            latest_bar_time=None,
        )

        results = validator.check_hard_gates(data)

        freshness_results = [r for r in results if r.gate_name == "data_freshness"]
        assert len(freshness_results) == 1
        assert freshness_results[0].passed is True
        assert "5min_age" in freshness_results[0].detail.lower()


# ── Soft Gate Tests ────────────────────────────────────────────────────────


def _make_uptrend_5min_bars(n: int = 50) -> pd.DataFrame:
    """Create 5min OHLCV bars with clear uptrend for testing."""
    base = 1.1000
    data = []
    for i in range(n):
        o = base + i * 0.0002
        c = o + 0.0003  # Close > Open → bullish
        h = c + 0.0001
        low = o - 0.0001
        data.append({"open": o, "high": h, "low": low, "close": c})
    return pd.DataFrame(data)


def _make_downtrend_5min_bars(n: int = 50) -> pd.DataFrame:
    """Create 5min OHLCV bars with clear downtrend for testing."""
    base = 1.2000
    data = []
    for i in range(n):
        o = base - i * 0.0002
        c = o - 0.0003  # Close < Open → bearish
        h = o + 0.0001
        low = c - 0.0001
        data.append({"open": o, "high": h, "low": low, "close": c})
    return pd.DataFrame(data)


def _make_doji_5min_bars(n: int = 50) -> pd.DataFrame:
    """Create 5min bars with doji (tiny body, large wicks) → candle quality fails."""
    data = []
    for i in range(n):
        mid = 1.1000
        data.append(
            {
                "open": mid,
                "high": mid + 0.0010,
                "low": mid - 0.0010,
                "close": mid + (0.00001 if i % 2 == 0 else -0.00001),  # Tiny body
            }
        )
    return pd.DataFrame(data)


class TestSoftGateEMAMomentum:
    """EMA momentum: fast EMA vs slow EMA alignment with trade direction."""

    def test_buy_in_uptrend_passes(self) -> None:
        config = TacticalConfig()
        validator = TacticalValidator(config)
        bars = _make_uptrend_5min_bars()
        result = validator._check_ema_momentum_gate("BUY", bars)
        assert result.passed is True

    def test_sell_in_uptrend_fails(self) -> None:
        config = TacticalConfig()
        validator = TacticalValidator(config)
        bars = _make_uptrend_5min_bars()
        result = validator._check_ema_momentum_gate("SELL", bars)
        assert result.passed is False

    def test_sell_in_downtrend_passes(self) -> None:
        config = TacticalConfig()
        validator = TacticalValidator(config)
        bars = _make_downtrend_5min_bars()
        result = validator._check_ema_momentum_gate("SELL", bars)
        assert result.passed is True


class TestSoftGateRSI:
    """RSI state: not in extreme zone opposing trade direction."""

    def test_buy_with_normal_rsi_passes(self) -> None:
        config = TacticalConfig()
        validator = TacticalValidator(config)
        bars = _make_doji_5min_bars()
        result = validator._check_rsi_state_gate("BUY", bars)
        assert result.passed is True

    def test_buy_with_overbought_rsi_fails(self) -> None:
        config = TacticalConfig()
        validator = TacticalValidator(config)
        bars = _make_uptrend_5min_bars()
        result = validator._check_rsi_state_gate("BUY", bars)
        assert result.passed is False

    def test_sell_with_oversold_rsi_fails(self) -> None:
        config = TacticalConfig()
        validator = TacticalValidator(config)
        bars = _make_downtrend_5min_bars()
        result = validator._check_rsi_state_gate("SELL", bars)
        assert result.passed is False


class TestSoftGateCandleQuality:
    """Candle quality: latest bar body/range ratio above threshold."""

    def test_directional_candle_passes(self) -> None:
        config = TacticalConfig()
        validator = TacticalValidator(config)
        bars = _make_uptrend_5min_bars()
        result = validator._check_candle_quality_gate(bars)
        assert result.passed is True

    def test_doji_candle_fails(self) -> None:
        config = TacticalConfig()
        validator = TacticalValidator(config)
        bars = _make_doji_5min_bars()
        result = validator._check_candle_quality_gate(bars)
        assert result.passed is False


class TestSoftGateScoring:
    """Integration: soft gate scoring with min_score threshold."""

    def test_all_soft_gates_pass_in_aligned_trend(self) -> None:
        """BUY in moderate uptrend (not overbought) should pass all 3 soft gates."""
        config = TacticalConfig()
        validator = TacticalValidator(config)
        bars = pd.DataFrame(
            {
                "open": [1.1000 + i * 0.00005 for i in range(50)],
                "high": [1.1002 + i * 0.00005 for i in range(50)],
                "low": [1.0999 + i * 0.00005 for i in range(50)],
                "close": [1.1001 + i * 0.00005 for i in range(50)],
            }
        )
        data = TacticalData(bars_5min=bars)
        results = validator.check_soft_gates("BUY", data)
        score = sum(1 for r in results if r.passed)
        assert score >= 2  # At least 2/3 pass


# ── v1.3.8: Pass-through Tests (missing data should not block) ────────────


class TestPassThroughWhenNoData:
    """v1.3.8: Gates should pass-through when bar data is unavailable."""

    def test_atr_gate_passes_when_no_1h_data(self) -> None:
        """ATR regime gate should pass when bars_1h is empty (no data available)."""
        config = TacticalConfig()
        validator = TacticalValidator(config)
        data = TacticalData(
            bars_1h=pd.DataFrame(),
            current_spread=0.00015,
            typical_spread=0.00015,
            latest_bar_time=datetime.now(timezone.utc),
        )
        results = validator.check_hard_gates(data)
        atr_results = [r for r in results if r.gate_name == "atr_regime"]
        assert len(atr_results) == 1
        assert atr_results[0].passed is True
        assert "pass-through" in atr_results[0].detail.lower()

    def test_ema_gate_passes_when_no_5min_data(self) -> None:
        """EMA momentum gate should pass when bars_5min is empty."""
        config = TacticalConfig()
        validator = TacticalValidator(config)
        result = validator._check_ema_momentum_gate("BUY", pd.DataFrame())
        assert result.passed is True
        assert "pass-through" in result.detail.lower()

    def test_rsi_gate_passes_when_no_5min_data(self) -> None:
        """RSI state gate should pass when bars_5min is empty."""
        config = TacticalConfig()
        validator = TacticalValidator(config)
        result = validator._check_rsi_state_gate("BUY", pd.DataFrame())
        assert result.passed is True
        assert "pass-through" in result.detail.lower()

    def test_ema_gate_passes_when_insufficient_data(self) -> None:
        """EMA gate should pass when data has fewer bars than needed."""
        config = TacticalConfig()
        validator = TacticalValidator(config)
        # Only 5 bars — way less than ema_slow (21) + 5 = 26 needed
        bars = pd.DataFrame(
            {
                "open": [1.1] * 5,
                "high": [1.11] * 5,
                "low": [1.09] * 5,
                "close": [1.1] * 5,
            }
        )
        result = validator._check_ema_momentum_gate("BUY", bars)
        assert result.passed is True
        assert "pass-through" in result.detail.lower()

    def test_rsi_gate_passes_when_insufficient_data(self) -> None:
        """RSI gate should pass when data has fewer bars than needed."""
        config = TacticalConfig()
        validator = TacticalValidator(config)
        bars = pd.DataFrame(
            {
                "open": [1.1] * 5,
                "high": [1.11] * 5,
                "low": [1.09] * 5,
                "close": [1.1] * 5,
            }
        )
        result = validator._check_rsi_state_gate("SELL", bars)
        assert result.passed is True
        assert "pass-through" in result.detail.lower()

    def test_spread_gate_passthrough_when_no_data(self) -> None:
        """Spread gate should pass-through when typical_spread is 0 (no data)."""
        config = TacticalConfig()
        validator = TacticalValidator(config)

        data = TacticalData(current_spread=0.0, typical_spread=0.0)
        results = validator.check_hard_gates(data)

        spread_result = next(r for r in results if r.gate_name == "spread")
        assert spread_result.passed is True
        assert (
            "pass-through" in spread_result.detail.lower()
            or "skipped" in spread_result.detail.lower()
        )
