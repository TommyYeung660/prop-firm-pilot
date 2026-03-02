"""Tests for fx_data_fetcher — fetch_bars() multi-timeframe support (v1.2.0)."""

from datetime import date

import httpx
import pytest
import respx

from src.data.fx_data_fetcher import ITickProvider, TraderMadeProvider

# ── TraderMade fetch_bars ────────────────────────────────────────────────────


@respx.mock
async def test_tradermade_fetch_bars_daily_backward_compat():
    """fetch_daily_bars should still work via delegation to fetch_bars."""
    provider = TraderMadeProvider(api_key="test_key")
    route = respx.get("https://marketdata.tradermade.com/api/v1/timeseries").mock(
        return_value=httpx.Response(
            200,
            json={
                "quotes": [
                    {"date": "2026-03-01", "open": 1.08, "high": 1.09, "low": 1.07, "close": 1.085}
                ]
            },
        )
    )
    async with httpx.AsyncClient() as client:
        df = await provider.fetch_daily_bars("EURUSD", date(2026, 3, 1), date(2026, 3, 1), client)

    assert route.called
    assert len(df) == 1
    assert "datetime" in df.columns
    # Verify interval=daily was passed
    request = route.calls[0].request
    assert "interval=daily" in str(request.url)


@respx.mock
async def test_tradermade_fetch_bars_4h_interval():
    """TraderMade fetch_bars should pass interval='4H' to API."""
    provider = TraderMadeProvider(api_key="test_key")
    route = respx.get("https://marketdata.tradermade.com/api/v1/timeseries").mock(
        return_value=httpx.Response(
            200,
            json={
                "quotes": [
                    {
                        "date": "2026-03-01 00:00:00",
                        "open": 1.08,
                        "high": 1.09,
                        "low": 1.07,
                        "close": 1.085,
                    },
                    {
                        "date": "2026-03-01 04:00:00",
                        "open": 1.085,
                        "high": 1.095,
                        "low": 1.08,
                        "close": 1.09,
                    },
                ]
            },
        )
    )
    async with httpx.AsyncClient() as client:
        df = await provider.fetch_bars(
            "EURUSD", date(2026, 3, 1), date(2026, 3, 1), client, interval="4h"
        )

    assert route.called
    assert len(df) == 2
    # Verify interval=4H was passed in the URL
    request = route.calls[0].request
    assert "interval=4H" in str(request.url)


@respx.mock
async def test_tradermade_fetch_bars_1h_interval():
    """TraderMade fetch_bars should pass interval='1H' to API."""
    provider = TraderMadeProvider(api_key="test_key")
    route = respx.get("https://marketdata.tradermade.com/api/v1/timeseries").mock(
        return_value=httpx.Response(
            200,
            json={
                "quotes": [
                    {
                        "date": "2026-03-01 00:00:00",
                        "open": 1.08,
                        "high": 1.09,
                        "low": 1.07,
                        "close": 1.085,
                    },
                ]
            },
        )
    )
    async with httpx.AsyncClient() as client:
        await provider.fetch_bars(
            "EURUSD", date(2026, 3, 1), date(2026, 3, 1), client, interval="1h"
        )

    assert route.called
    request = route.calls[0].request
    assert "interval=1H" in str(request.url)


async def test_tradermade_fetch_bars_invalid_interval():
    """TraderMade fetch_bars should raise ValueError for unsupported interval."""
    provider = TraderMadeProvider(api_key="test_key")
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="Unsupported interval"):
            await provider.fetch_bars(
                "EURUSD", date(2026, 3, 1), date(2026, 3, 1), client, interval="2h"
            )


# ── iTick fetch_bars ─────────────────────────────────────────────────────────


@respx.mock
async def test_itick_fetch_bars_daily_backward_compat():
    """fetch_daily_bars should still work via delegation to fetch_bars."""
    provider = ITickProvider(api_key="test_key")
    route = respx.get("https://api.itick.org/forex/kline").mock(
        return_value=httpx.Response(
            200,
            json={
                "code": 200,
                "data": [
                    {"t": 1772323200000, "o": 1.08, "h": 1.09, "l": 1.07, "c": 1.085, "v": 100}
                ],
            },
        )
    )
    async with httpx.AsyncClient() as client:
        df = await provider.fetch_daily_bars("EURUSD", date(2026, 3, 1), date(2026, 3, 1), client)

    assert route.called
    assert len(df) == 1
    # Verify kType=8 (daily) was passed
    request = route.calls[0].request
    assert "kType=8" in str(request.url)


@respx.mock
async def test_itick_fetch_bars_4h_interval():
    """iTick fetch_bars should pass kType=6 for 4h interval."""
    provider = ITickProvider(api_key="test_key")
    route = respx.get("https://api.itick.org/forex/kline").mock(
        return_value=httpx.Response(
            200,
            json={
                "code": 200,
                "data": [
                    {"t": 1772323200000, "o": 1.08, "h": 1.09, "l": 1.07, "c": 1.085, "v": 100},
                    {"t": 1772337600000, "o": 1.085, "h": 1.095, "l": 1.08, "c": 1.09, "v": 150},
                ],
            },
        )
    )
    async with httpx.AsyncClient() as client:
        df = await provider.fetch_bars(
            "EURUSD", date(2026, 3, 1), date(2026, 3, 1), client, interval="4h"
        )

    assert route.called
    assert len(df) == 2
    # Verify kType=6 was passed
    request = route.calls[0].request
    assert "kType=6" in str(request.url)


@respx.mock
async def test_itick_fetch_bars_1h_interval():
    """iTick fetch_bars should pass kType=5 for 1h interval."""
    provider = ITickProvider(api_key="test_key")
    route = respx.get("https://api.itick.org/forex/kline").mock(
        return_value=httpx.Response(
            200,
            json={
                "code": 200,
                "data": [
                    {"t": 1772323200000, "o": 1.08, "h": 1.09, "l": 1.07, "c": 1.085, "v": 100},
                ],
            },
        )
    )
    async with httpx.AsyncClient() as client:
        await provider.fetch_bars(
            "EURUSD", date(2026, 3, 1), date(2026, 3, 1), client, interval="1h"
        )

    assert route.called
    request = route.calls[0].request
    assert "kType=5" in str(request.url)


async def test_itick_fetch_bars_invalid_interval():
    """iTick fetch_bars should raise ValueError for unsupported interval."""
    provider = ITickProvider(api_key="test_key")
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="Unsupported interval"):
            await provider.fetch_bars(
                "EURUSD", date(2026, 3, 1), date(2026, 3, 1), client, interval="2h"
            )
