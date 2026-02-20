"""
Tests for src/execution/position_sizer.py.

Tests cover:
- calculate_volume with normal cases, zero pips, unknown symbols, random offsets
- calculate_risk_amount with known and unknown symbols
- max_volume_for_risk with known and unknown symbols, zero pips
- estimate_pip_distance with known and unknown symbols
- Random offset application using unittest.mock.patch on random.uniform
"""

from unittest.mock import patch

import pytest

from src.config import ExecutionConfig, InstrumentConfig
from src.execution.position_sizer import PositionSizer

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def execution_config() -> ExecutionConfig:
    """Execution config with 1% default risk and 10% position offset."""
    return ExecutionConfig(
        default_risk_pct=0.01,
        max_risk_pct=0.02,
        position_offset_pct=0.10,
    )


@pytest.fixture
def instruments() -> dict[str, InstrumentConfig]:
    """Instrument configs for major FX pairs."""
    return {
        "EURUSD": InstrumentConfig(
            pip_value=10.0,  # $10 per pip for 1 lot
            pip_size=0.0001,
            min_lot=0.01,
            max_lot=50.0,
        ),
        "GBPUSD": InstrumentConfig(
            pip_value=10.0,
            pip_size=0.0001,
            min_lot=0.01,
            max_lot=30.0,
        ),
        "USDJPY": InstrumentConfig(
            pip_value=9.09,  # Approx $9.09 per pip for 1 lot
            pip_size=0.01,
            min_lot=0.01,
            max_lot=30.0,
        ),
        "XAUUSD": InstrumentConfig(
            pip_value=10.0,
            pip_size=0.01,
            min_lot=0.01,
            max_lot=20.0,
        ),
    }


@pytest.fixture
def sizer(
    execution_config: ExecutionConfig,
    instruments: dict[str, InstrumentConfig],
) -> PositionSizer:
    """PositionSizer instance with test config."""
    return PositionSizer(config=execution_config, instruments=instruments)


# ── calculate_volume ───────────────────────────────────────────────────────────


class TestCalculateVolume:
    """Test calculate_volume() method."""

    def test_normal_calculation(self, sizer: PositionSizer) -> None:
        """Basic volume calculation with positive stop loss."""
        # $50,000 equity * 1% risk = $500 risk
        # 50 pips * $10/pip = $500 per lot
        # $500 / $500 = 1.0 lot (before offset)
        equity = 50000.0
        sl_pips = 50.0
        volume = sizer.calculate_volume("EURUSD", equity, sl_pips)
        # With random offset [-10%, +10%], result should be in [0.9, 1.1]
        assert 0.9 <= volume <= 1.1

    def test_calculation_usdjpy(self, sizer: PositionSizer) -> None:
        """Volume calculation for USDJPY with different pip value."""
        # $50,000 * 1% = $500 risk
        # 50 pips * $9.09/pip = $454.5 per lot
        # $500 / $454.5 ≈ 1.1 lots (before offset)
        equity = 50000.0
        sl_pips = 50.0
        volume = sizer.calculate_volume("USDJPY", equity, sl_pips)
        # With random offset, result should be around 1.1
        assert 0.9 <= volume <= 1.3

    def test_zero_stop_loss_returns_min_lot(self, sizer: PositionSizer) -> None:
        """Zero or negative stop loss pips returns minimum lot."""
        assert sizer.calculate_volume("EURUSD", 50000.0, 0.0) == 0.01
        assert sizer.calculate_volume("EURUSD", 50000.0, -5.0) == 0.01

    def test_unknown_symbol_returns_min_lot(self, sizer: PositionSizer) -> None:
        """Unknown instrument returns minimum lot."""
        assert sizer.calculate_volume("UNKNOWN", 50000.0, 50.0) == 0.01

    def test_clamps_to_max_lot(self, sizer: PositionSizer) -> None:
        """Large volume is clamped to instrument's max lot."""
        # With tiny stop loss, calculation would produce huge volume
        # Should be clamped to XAUUSD's max_lot (20.0)
        volume = sizer.calculate_volume("XAUUSD", 50000.0, 1.0)
        # After random offset [-10%, +10%], should be around max_lot
        assert 18.0 <= volume <= 20.0

    def test_clamps_to_min_lot(self, sizer: PositionSizer) -> None:
        """Very small risk clamps to instrument's min lot."""
        # Tiny equity and huge stop loss = tiny volume
        # Should be clamped to min_lot (0.01)
        volume = sizer.calculate_volume("EURUSD", 100.0, 500.0)
        assert volume >= 0.01

    def test_random_offset_positive(self, sizer: PositionSizer) -> None:
        """Test that random.uniform's effect properly applies offset_pct.

        Mock random.uniform to return +10% offset to test deterministic behavior.
        """
        # Without offset: $50,000 * 1% = $500 risk
        # 50 pips * $10/pip = $500 per lot
        # $500 / $500 = 1.0 lot
        # With +10% offset: 1.0 * 1.10 = 1.1 lots
        with patch("src.execution.position_sizer.random.uniform", return_value=0.10):
            volume = sizer.calculate_volume("EURUSD", 50000.0, 50.0)
            assert volume == 1.10

    def test_random_offset_negative(self, sizer: PositionSizer) -> None:
        """Test negative random offset reduces volume."""
        # Without offset: 1.0 lot
        # With -10% offset: 1.0 * 0.90 = 0.9 lots
        with patch("src.execution.position_sizer.random.uniform", return_value=-0.10):
            volume = sizer.calculate_volume("EURUSD", 50000.0, 50.0)
            assert volume == 0.90

    def test_random_offset_zero(self, sizer: PositionSizer) -> None:
        """Test zero offset leaves volume unchanged."""
        with patch("src.execution.position_sizer.random.uniform", return_value=0.0):
            volume = sizer.calculate_volume("EURUSD", 50000.0, 50.0)
            assert volume == 1.0

    def test_rounding_to_0_01_precision(self, sizer: PositionSizer) -> None:
        """Volume is rounded to 0.01 lot precision."""
        # Mock to get a non-terminating decimal
        with patch("src.execution.position_sizer.random.uniform", return_value=0.05):
            volume = sizer.calculate_volume("EURUSD", 50000.0, 50.0)
            # 1.0 * 1.05 = 1.05, should be exactly 1.05
            assert volume == 1.05

    def test_volume_calculation_formula(self, sizer: PositionSizer) -> None:
        """Verify the exact formula: risk_amount / (stop_loss_pips × pip_value)."""
        # equity=10000, risk_pct=0.01 → $100 risk
        # sl_pips=20, pip_value=10.0 → $200 per lot
        # volume = 100 / 200 = 0.5 lots
        with patch("src.execution.position_sizer.random.uniform", return_value=0.0):
            volume = sizer.calculate_volume("EURUSD", 10000.0, 20.0)
            assert volume == 0.50


# ── calculate_risk_amount ──────────────────────────────────────────────────────


class TestCalculateRiskAmount:
    """Test calculate_risk_amount() method."""

    def test_known_instrument(self, sizer: PositionSizer) -> None:
        """Calculate risk for known instrument."""
        # volume=0.5, sl=50 pips, pip_value=10.0
        # risk = 0.5 * 50 * 10.0 = $250
        risk = sizer.calculate_risk_amount("EURUSD", 0.5, 50.0)
        assert risk == 250.0

    def test_unknown_instrument_defaults_to_10_per_pip(self, sizer: PositionSizer) -> None:
        """Unknown instrument assumes $10/pip as default."""
        # volume=0.5, sl=50 pips, default pip_value=10.0
        # risk = 0.5 * 50 * 10.0 = $250
        risk = sizer.calculate_risk_amount("UNKNOWN", 0.5, 50.0)
        assert risk == 250.0

    def test_rounding_to_2_decimals(self, sizer: PositionSizer) -> None:
        """Risk amount is rounded to 2 decimal places."""
        # volume=0.33, sl=30 pips, pip_value=10.0
        # risk = 0.33 * 30 * 10.0 = 99.0
        risk = sizer.calculate_risk_amount("EURUSD", 0.33, 30.0)
        assert risk == 99.0

    def test_usdjpy_different_pip_value(self, sizer: PositionSizer) -> None:
        """USDJPY has different pip value."""
        # volume=1.0, sl=50 pips, pip_value=9.09
        # risk = 1.0 * 50 * 9.09 = 454.5
        risk = sizer.calculate_risk_amount("USDJPY", 1.0, 50.0)
        assert risk == 454.5

    def test_zero_volume_zero_risk(self, sizer: PositionSizer) -> None:
        """Zero volume means zero risk."""
        risk = sizer.calculate_risk_amount("EURUSD", 0.0, 50.0)
        assert risk == 0.0

    def test_zero_stop_loss_zero_risk(self, sizer: PositionSizer) -> None:
        """Zero stop loss means zero risk."""
        risk = sizer.calculate_risk_amount("EURUSD", 1.0, 0.0)
        assert risk == 0.0


# ── max_volume_for_risk ───────────────────────────────────────────────────────


class TestMaxVolumeForRisk:
    """Test max_volume_for_risk() method."""

    def test_normal_calculation(self, sizer: PositionSizer) -> None:
        """Calculate max volume within risk budget."""
        # max_risk=$500, sl=50 pips, pip_value=10.0
        # volume = 500 / (50 * 10.0) = 1.0 lot
        volume = sizer.max_volume_for_risk("EURUSD", 500.0, 50.0)
        assert volume == 1.0

    def test_unknown_symbol_returns_min_lot(self, sizer: PositionSizer) -> None:
        """Unknown instrument returns minimum lot."""
        assert sizer.max_volume_for_risk("UNKNOWN", 500.0, 50.0) == 0.01

    def test_zero_stop_loss_returns_min_lot(self, sizer: PositionSizer) -> None:
        """Zero or negative stop loss returns minimum lot."""
        assert sizer.max_volume_for_risk("EURUSD", 500.0, 0.0) == 0.01
        assert sizer.max_volume_for_risk("EURUSD", 500.0, -5.0) == 0.01

    def test_clamps_to_max_lot(self, sizer: PositionSizer) -> None:
        """Large volume is clamped to instrument's max lot."""
        # Large risk budget with tiny stop loss → huge volume
        # Should clamp to GBPUSD's max_lot (30.0)
        volume = sizer.max_volume_for_risk("GBPUSD", 10000.0, 1.0)
        assert volume == 30.0

    def test_clamps_to_min_lot(self, sizer: PositionSizer) -> None:
        """Small volume is clamped to min lot."""
        # Tiny risk budget → tiny volume
        volume = sizer.max_volume_for_risk("EURUSD", 1.0, 50.0)
        assert volume == 0.01

    def test_rounding_to_2_decimals(self, sizer: PositionSizer) -> None:
        """Result is rounded to 0.01 precision."""
        # max_risk=333, sl=50, pip_value=10.0
        # volume = 333 / 500 = 0.666
        volume = sizer.max_volume_for_risk("EURUSD", 333.0, 50.0)
        assert volume == 0.67

    def test_xauusd_max_lot(self, sizer: PositionSizer) -> None:
        """XAUUSD has max_lot=20.0."""
        volume = sizer.max_volume_for_risk("XAUUSD", 50000.0, 1.0)
        assert volume == 20.0


# ── estimate_pip_distance ───────────────────────────────────────────────────────


class TestEstimatePipDistance:
    """Test estimate_pip_distance() method."""

    def test_eurusd_pip_distance(self, sizer: PositionSizer) -> None:
        """Calculate pip distance for EURUSD (pip_size=0.0001)."""
        # entry=1.0850, stop=1.0800
        # distance = 0.0050 / 0.0001 = 50 pips
        pips = sizer.estimate_pip_distance("EURUSD", 1.0850, 1.0800)
        assert pips == 50.0

    def test_buy_stop_below_entry(self, sizer: PositionSizer) -> None:
        """Buy trade with stop loss below entry."""
        pips = sizer.estimate_pip_distance("EURUSD", 1.0900, 1.0850)
        assert pips == 50.0

    def test_sell_stop_above_entry(self, sizer: PositionSizer) -> None:
        """Sell trade with stop loss above entry."""
        pips = sizer.estimate_pip_distance("EURUSD", 1.0850, 1.0900)
        assert pips == 50.0

    def test_usdjyp_pip_size(self, sizer: PositionSizer) -> None:
        """USDJPY has pip_size=0.01."""
        # entry=150.00, stop=149.50
        # distance = 0.50 / 0.01 = 50 pips
        pips = sizer.estimate_pip_distance("USDJPY", 150.00, 149.50)
        assert pips == 50.0

    def test_unknown_symbol_defaults_to_0_0001(self, sizer: PositionSizer) -> None:
        """Unknown symbol uses default pip_size=0.0001."""
        pips = sizer.estimate_pip_distance("UNKNOWN", 1.0850, 1.0800)
        assert pips == 50.0

    def test_rounding_to_1_decimal(self, sizer: PositionSizer) -> None:
        """Result is rounded to 1 decimal place."""
        # entry=1.0853, stop=1.0800
        # distance = 0.0053 / 0.0001 = 53.0 pips
        pips = sizer.estimate_pip_distance("EURUSD", 1.0853, 1.0800)
        assert pips == 53.0

    def test_xauusd_pip_size(self, sizer: PositionSizer) -> None:
        """XAUUSD has pip_size=0.01 (like JPY pairs)."""
        # entry=2900.00, stop=2895.00
        # distance = 5.00 / 0.01 = 500 pips
        pips = sizer.estimate_pip_distance("XAUUSD", 2900.00, 2895.00)
        assert pips == 500.0

    def test_same_prices_zero_pips(self, sizer: PositionSizer) -> None:
        """Same entry and stop prices = 0 pips."""
        pips = sizer.estimate_pip_distance("EURUSD", 1.0850, 1.0850)
        assert pips == 0.0
