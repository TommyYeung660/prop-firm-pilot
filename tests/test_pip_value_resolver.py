"""Tests for JPY-quoted pair pip-value resolution."""

import pytest

from src.execution.pip_value_resolver import resolve_usd_pip_value_for_symbol


def test_non_jpy_quote_pair_keeps_static_pip_value() -> None:
    """Non-JPY pairs should continue using the configured static pip value."""
    assert resolve_usd_pip_value_for_symbol("EURUSD", static_pip_value=10.0) == 10.0


def test_usdjpy_uses_live_symbol_price() -> None:
    """USDJPY pip value should be derived from its live quote."""
    pip_value = resolve_usd_pip_value_for_symbol(
        "USDJPY",
        static_pip_value=10.0,
        symbol_price=150.0,
    )
    assert pip_value == pytest.approx(round(1000.0 / 150.0, 4))


@pytest.mark.parametrize("symbol", ["EURJPY", "AUDJPY", "CADJPY"])
def test_jpy_crosses_use_usdjpy_conversion(symbol: str) -> None:
    """JPY crosses should convert 1000 JPY/pip back into USD via USDJPY."""
    pip_value = resolve_usd_pip_value_for_symbol(
        symbol,
        static_pip_value=10.0,
        usd_jpy_price=150.0,
    )
    assert pip_value == pytest.approx(round(1000.0 / 150.0, 4))


def test_invalid_or_missing_conversion_falls_back_to_static_value() -> None:
    """Missing or invalid USDJPY conversion should stay on the static safe fallback."""
    assert resolve_usd_pip_value_for_symbol("EURJPY", static_pip_value=6.67) == 6.67
    assert (
        resolve_usd_pip_value_for_symbol(
            "EURJPY",
            static_pip_value=6.67,
            usd_jpy_price=0.0,
        )
        == 6.67
    )
