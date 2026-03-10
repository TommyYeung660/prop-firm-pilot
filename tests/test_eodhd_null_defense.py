"""
Tests for EODHD intraday bar parser defensive null handling.

Ensures that None/null values in EODHD API responses (volume, OHLC fields)
are safely coerced to numeric defaults instead of crashing with float(None).
"""

from datetime import date

import httpx
import pytest
import respx

from src.data.fx_data_fetcher import EodhdProvider

# ── Unit Tests: _fetch_chunk null handling ──────────────────────────────────


@pytest.fixture
def provider() -> EodhdProvider:
    return EodhdProvider(api_key="test-key", max_retries=1)


async def test_null_volume_does_not_crash(provider: EodhdProvider) -> None:
    """Bars with None volume should default to 0, not crash."""
    bars_json = [
        {
            "datetime": "2026-03-10 10:00:00",
            "open": 1.0850,
            "high": 1.0860,
            "low": 1.0840,
            "close": 1.0855,
            "volume": None,
        },
    ]
    with respx.mock:
        respx.get("https://eodhd.com/api/intraday/EURUSD.FOREX").respond(200, json=bars_json)
        async with httpx.AsyncClient() as client:
            df = await provider._fetch_chunk(
                "EURUSD", date(2026, 3, 10), date(2026, 3, 10), client, "5m"
            )
    assert len(df) == 1
    assert df.iloc[0]["volume"] == 0


async def test_null_ohlc_fields_default_to_zero(provider: EodhdProvider) -> None:
    """Bars with None open/high/low/close should default to 0.0, not crash."""
    bars_json = [
        {
            "datetime": "2026-03-10 10:00:00",
            "open": None,
            "high": None,
            "low": None,
            "close": None,
            "volume": 100,
        },
    ]
    with respx.mock:
        respx.get("https://eodhd.com/api/intraday/EURUSD.FOREX").respond(200, json=bars_json)
        async with httpx.AsyncClient() as client:
            df = await provider._fetch_chunk(
                "EURUSD", date(2026, 3, 10), date(2026, 3, 10), client, "5m"
            )
    assert len(df) == 1
    assert df.iloc[0]["open"] == 0.0
    assert df.iloc[0]["high"] == 0.0
    assert df.iloc[0]["low"] == 0.0
    assert df.iloc[0]["close"] == 0.0


async def test_missing_ohlc_keys_default_to_zero(provider: EodhdProvider) -> None:
    """Bars missing OHLC keys entirely should default to 0.0, not KeyError."""
    bars_json = [
        {
            "datetime": "2026-03-10 10:00:00",
            # No open, high, low, close, volume keys at all
        },
    ]
    with respx.mock:
        respx.get("https://eodhd.com/api/intraday/EURUSD.FOREX").respond(200, json=bars_json)
        async with httpx.AsyncClient() as client:
            df = await provider._fetch_chunk(
                "EURUSD", date(2026, 3, 10), date(2026, 3, 10), client, "5m"
            )
    assert len(df) == 1
    assert df.iloc[0]["open"] == 0.0
    assert df.iloc[0]["volume"] == 0


async def test_empty_bars_response_returns_empty_df(provider: EodhdProvider) -> None:
    """Empty bars list from EODHD should return empty DataFrame, not crash."""
    with respx.mock:
        respx.get("https://eodhd.com/api/intraday/EURUSD.FOREX").respond(200, json=[])
        async with httpx.AsyncClient() as client:
            df = await provider._fetch_chunk(
                "EURUSD", date(2026, 3, 10), date(2026, 3, 10), client, "5m"
            )
    assert df.empty
    assert list(df.columns) == ["datetime", "open", "high", "low", "close", "volume"]


async def test_mixed_null_and_valid_bars(provider: EodhdProvider) -> None:
    """Mix of valid and null-bearing bars should all parse successfully."""
    bars_json = [
        {
            "datetime": "2026-03-10 10:00:00",
            "open": 1.0850,
            "high": 1.0860,
            "low": 1.0840,
            "close": 1.0855,
            "volume": 500,
        },
        {
            "datetime": "2026-03-10 10:05:00",
            "open": None,
            "high": 1.0870,
            "low": None,
            "close": 1.0865,
            "volume": None,
        },
    ]
    with respx.mock:
        respx.get("https://eodhd.com/api/intraday/EURUSD.FOREX").respond(200, json=bars_json)
        async with httpx.AsyncClient() as client:
            df = await provider._fetch_chunk(
                "EURUSD", date(2026, 3, 10), date(2026, 3, 10), client, "5m"
            )
    assert len(df) == 2
    # First bar: valid values
    assert df.iloc[0]["open"] == 1.0850
    assert df.iloc[0]["volume"] == 500
    # Second bar: nulls defaulted
    assert df.iloc[1]["open"] == 0.0
    assert df.iloc[1]["low"] == 0.0
    assert df.iloc[1]["volume"] == 0
