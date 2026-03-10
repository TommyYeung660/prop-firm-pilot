"""
Tests for EODHD zero-bar filtering.

EODHD API sometimes returns bars where all OHLC values are 0 (typically
during weekends, DST transitions, or data gaps).  These zero-bars
corrupt ATR/EMA/RSI calculations, causing tactical gate failures.

The fix: EodhdProvider.fetch_bars() must drop rows where open, high, low,
and close are ALL zero before returning the DataFrame.
"""

from datetime import date

import httpx
import pytest
import respx

from src.data.fx_data_fetcher import EodhdProvider

EODHD_URL = "https://eodhd.com/api/intraday/EURUSD.FOREX"


@pytest.fixture
def provider() -> EodhdProvider:
    return EodhdProvider(api_key="test-key", max_retries=1)


# ── Core: zero-bar filtering ────────────────────────────────────────────────


async def test_all_zero_ohlc_bars_are_dropped(provider: EodhdProvider) -> None:
    """Bars where open=high=low=close=0 should be removed from output."""
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
            "open": 0,
            "high": 0,
            "low": 0,
            "close": 0,
            "volume": 0,
        },
        {
            "datetime": "2026-03-10 10:10:00",
            "open": 1.0855,
            "high": 1.0870,
            "low": 1.0850,
            "close": 1.0865,
            "volume": 300,
        },
    ]
    with respx.mock:
        respx.get(EODHD_URL).respond(200, json=bars_json)
        async with httpx.AsyncClient() as client:
            df = await provider.fetch_bars(
                "EURUSD", date(2026, 3, 10), date(2026, 3, 10), client, interval="5min"
            )
    # Zero-bar at 10:05 should be dropped
    assert len(df) == 2
    assert df.iloc[0]["open"] == 1.0850
    assert df.iloc[1]["open"] == 1.0855


async def test_multiple_zero_bars_all_dropped(provider: EodhdProvider) -> None:
    """Multiple consecutive zero-bars should all be dropped."""
    bars_json = [
        {
            "datetime": "2026-03-10 10:00:00",
            "open": 0,
            "high": 0,
            "low": 0,
            "close": 0,
            "volume": 0,
        },
        {
            "datetime": "2026-03-10 10:05:00",
            "open": 0,
            "high": 0,
            "low": 0,
            "close": 0,
            "volume": 100,
        },
        {
            "datetime": "2026-03-10 10:10:00",
            "open": 1.0855,
            "high": 1.0870,
            "low": 1.0850,
            "close": 1.0865,
            "volume": 300,
        },
    ]
    with respx.mock:
        respx.get(EODHD_URL).respond(200, json=bars_json)
        async with httpx.AsyncClient() as client:
            df = await provider.fetch_bars(
                "EURUSD", date(2026, 3, 10), date(2026, 3, 10), client, interval="5min"
            )
    # Both zero-bars dropped, only the valid one remains
    assert len(df) == 1
    assert df.iloc[0]["open"] == 1.0855


async def test_all_bars_zero_returns_empty_df(provider: EodhdProvider) -> None:
    """If ALL bars are zero, result should be empty DataFrame."""
    bars_json = [
        {
            "datetime": "2026-03-10 10:00:00",
            "open": 0,
            "high": 0,
            "low": 0,
            "close": 0,
            "volume": 0,
        },
        {
            "datetime": "2026-03-10 10:05:00",
            "open": 0,
            "high": 0,
            "low": 0,
            "close": 0,
            "volume": 0,
        },
    ]
    with respx.mock:
        respx.get(EODHD_URL).respond(200, json=bars_json)
        async with httpx.AsyncClient() as client:
            df = await provider.fetch_bars(
                "EURUSD", date(2026, 3, 10), date(2026, 3, 10), client, interval="5min"
            )
    assert len(df) == 0


async def test_partial_zero_ohlc_bar_kept(provider: EodhdProvider) -> None:
    """Bars with SOME zero OHLC but not ALL should be kept (e.g. open=0 but close!=0)."""
    bars_json = [
        {
            "datetime": "2026-03-10 10:00:00",
            "open": 0,
            "high": 1.0860,
            "low": 0,
            "close": 1.0855,
            "volume": 100,
        },
    ]
    with respx.mock:
        respx.get(EODHD_URL).respond(200, json=bars_json)
        async with httpx.AsyncClient() as client:
            df = await provider.fetch_bars(
                "EURUSD", date(2026, 3, 10), date(2026, 3, 10), client, interval="5min"
            )
    # Partial zeros are NOT dropped — only ALL-zero OHLC bars
    assert len(df) == 1
    assert df.iloc[0]["high"] == 1.0860


async def test_zero_bar_filtering_works_for_1h_interval(provider: EodhdProvider) -> None:
    """Zero-bar filtering also works for 1h interval (used by ATR gate)."""
    bars_json = [
        {
            "datetime": "2026-03-10 00:00:00",
            "open": 0,
            "high": 0,
            "low": 0,
            "close": 0,
            "volume": 0,
        },
        {
            "datetime": "2026-03-10 01:00:00",
            "open": 1.0850,
            "high": 1.0860,
            "low": 1.0840,
            "close": 1.0855,
            "volume": 500,
        },
    ]
    with respx.mock:
        respx.get(EODHD_URL).respond(200, json=bars_json)
        async with httpx.AsyncClient() as client:
            df = await provider.fetch_bars(
                "EURUSD", date(2026, 3, 10), date(2026, 3, 10), client, interval="1h"
            )
    assert len(df) == 1
    assert df.iloc[0]["open"] == 1.0850


async def test_null_coerced_to_zero_then_filtered(provider: EodhdProvider) -> None:
    """Bars with all-null OHLC (coerced to 0 by _fetch_chunk) should also be filtered."""
    bars_json = [
        {
            "datetime": "2026-03-10 10:00:00",
            "open": None,
            "high": None,
            "low": None,
            "close": None,
            "volume": None,
        },
        {
            "datetime": "2026-03-10 10:05:00",
            "open": 1.0855,
            "high": 1.0870,
            "low": 1.0850,
            "close": 1.0865,
            "volume": 300,
        },
    ]
    with respx.mock:
        respx.get(EODHD_URL).respond(200, json=bars_json)
        async with httpx.AsyncClient() as client:
            df = await provider.fetch_bars(
                "EURUSD", date(2026, 3, 10), date(2026, 3, 10), client, interval="5min"
            )
    # null→0 coercion + zero-bar filtering = only valid bar remains
    assert len(df) == 1
    assert df.iloc[0]["open"] == 1.0855


async def test_zero_bar_count_logged(provider: EodhdProvider) -> None:
    """When zero-bars are dropped, a warning should be logged with the count."""
    from loguru import logger

    captured: list[str] = []
    sink_id = logger.add(lambda msg: captured.append(msg.record["message"]), level="WARNING")

    bars_json = [
        {
            "datetime": "2026-03-10 10:00:00",
            "open": 0,
            "high": 0,
            "low": 0,
            "close": 0,
            "volume": 0,
        },
        {
            "datetime": "2026-03-10 10:05:00",
            "open": 0,
            "high": 0,
            "low": 0,
            "close": 0,
            "volume": 0,
        },
        {
            "datetime": "2026-03-10 10:10:00",
            "open": 1.0855,
            "high": 1.0870,
            "low": 1.0850,
            "close": 1.0865,
            "volume": 300,
        },
    ]
    try:
        with respx.mock:
            respx.get(EODHD_URL).respond(200, json=bars_json)
            async with httpx.AsyncClient() as client:
                df = await provider.fetch_bars(
                    "EURUSD", date(2026, 3, 10), date(2026, 3, 10), client, interval="5min"
                )
        assert len(df) == 1
        # Check that a warning was logged about dropped zero-bars
        assert any("2" in msg and "zero" in msg.lower() for msg in captured)
    finally:
        logger.remove(sink_id)
