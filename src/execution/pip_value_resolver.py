"""
Resolve live USD pip values for FX instruments.

JPY-quoted pairs require quote-aware conversion in a USD account because
their pip value changes with the USDJPY exchange rate. This module keeps
the conversion logic pure so execution paths can fetch quotes separately
and pass only the reference prices in.

Usage:
    pip_value = resolve_usd_pip_value_for_symbol("USDJPY", 6.67, symbol_price=150.0)
    cross_value = resolve_usd_pip_value_for_symbol("EURJPY", 6.67, usd_jpy_price=150.0)
"""

from __future__ import annotations

PIP_VALUE_JPY_PER_STANDARD_LOT = 1000.0


def quote_to_reference_price(bid: float | None, ask: float | None) -> float | None:
    """Convert a bid/ask quote into a usable reference price."""
    has_bid = bid is not None and bid > 0
    has_ask = ask is not None and ask > 0

    if has_bid and has_ask:
        return (bid + ask) / 2.0
    if has_ask:
        return ask
    if has_bid:
        return bid
    return None


def resolve_usd_pip_value_for_symbol(
    symbol: str,
    static_pip_value: float,
    symbol_price: float | None = None,
    usd_jpy_price: float | None = None,
) -> float:
    """Resolve the USD pip value for a symbol, falling back safely when needed."""
    normalized_symbol = symbol.upper()
    if not normalized_symbol.endswith("JPY"):
        return static_pip_value

    if normalized_symbol == "USDJPY":
        if symbol_price is None or symbol_price <= 0:
            return static_pip_value
        return round(PIP_VALUE_JPY_PER_STANDARD_LOT / symbol_price, 4)

    if usd_jpy_price is None or usd_jpy_price <= 0:
        return static_pip_value
    return round(PIP_VALUE_JPY_PER_STANDARD_LOT / usd_jpy_price, 4)
