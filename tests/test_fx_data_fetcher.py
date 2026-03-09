"""Tests for fx_data_fetcher — fetch_bars() multi-timeframe support (v1.2.0 / v1.4.0)."""

from datetime import date

import httpx
import pytest
import respx

from src.data.fx_data_fetcher import EodhdProvider, ITickProvider, TraderMadeProvider, _to_eodhd_symbol

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


# ── EODHD Provider ─────────────────────────────────────────────────────────


def test_to_eodhd_symbol_fx():
    """FX pairs should get .FOREX suffix."""
    assert _to_eodhd_symbol("EURUSD") == "EURUSD.FOREX"
    assert _to_eodhd_symbol("GBP/USD") == "GBPUSD.FOREX"
    assert _to_eodhd_symbol("USDJPY") == "USDJPY.FOREX"


def test_to_eodhd_symbol_passthrough():
    """Already-suffixed symbols pass through unchanged."""
    assert _to_eodhd_symbol("EURUSD.FOREX") == "EURUSD.FOREX"
    assert _to_eodhd_symbol("AAPL.US") == "AAPL.US"


@respx.mock
async def test_eodhd_fetch_bars_1h():
    """EODHD fetch_bars should return correct DataFrame for 1h interval."""
    provider = EodhdProvider(api_key="test_key")
    route = respx.get("https://eodhd.com/api/intraday/EURUSD.FOREX").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "datetime": "2026-03-01 00:00:00",
                    "gmtoffset": 0,
                    "open": 1.0800,
                    "high": 1.0900,
                    "low": 1.0700,
                    "close": 1.0850,
                    "volume": 1000,
                },
                {
                    "datetime": "2026-03-01 01:00:00",
                    "gmtoffset": 0,
                    "open": 1.0850,
                    "high": 1.0950,
                    "low": 1.0800,
                    "close": 1.0900,
                    "volume": 1200,
                },
            ],
        )
    )
    async with httpx.AsyncClient() as client:
        df = await provider.fetch_bars(
            "EURUSD", date(2026, 3, 1), date(2026, 3, 1), client, interval="1h"
        )

    assert route.called
    assert len(df) == 2
    assert list(df.columns) == ["datetime", "open", "high", "low", "close", "volume"]
    assert df["open"].iloc[0] == 1.08
    assert df["volume"].iloc[1] == 1200
    # Verify API params
    request = route.calls[0].request
    url_str = str(request.url)
    assert "interval=1h" in url_str
    assert "api_token=test_key" in url_str


@respx.mock
async def test_eodhd_fetch_bars_5min():
    """EODHD fetch_bars should map '5min' to '5m' API param."""
    provider = EodhdProvider(api_key="test_key")
    route = respx.get("https://eodhd.com/api/intraday/EURUSD.FOREX").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "datetime": "2026-03-01 10:00:00",
                    "open": 1.08,
                    "high": 1.085,
                    "low": 1.079,
                    "close": 1.082,
                    "volume": 500,
                },
            ],
        )
    )
    async with httpx.AsyncClient() as client:
        df = await provider.fetch_bars(
            "EURUSD", date(2026, 3, 1), date(2026, 3, 1), client, interval="5min"
        )

    assert route.called
    assert len(df) == 1
    request = route.calls[0].request
    assert "interval=5m" in str(request.url)


@respx.mock
async def test_eodhd_fetch_bars_empty_response():
    """EODHD fetch_bars should return empty DataFrame on empty JSON array."""
    provider = EodhdProvider(api_key="test_key")
    respx.get("https://eodhd.com/api/intraday/EURUSD.FOREX").mock(
        return_value=httpx.Response(200, json=[])
    )
    async with httpx.AsyncClient() as client:
        df = await provider.fetch_bars(
            "EURUSD", date(2026, 3, 1), date(2026, 3, 1), client, interval="1h"
        )

    assert df.empty
    assert list(df.columns) == ["datetime", "open", "high", "low", "close", "volume"]


@respx.mock
async def test_eodhd_fetch_bars_rate_limit_retry():
    """EODHD should retry on 429 rate limit."""
    provider = EodhdProvider(api_key="test_key", max_retries=2)
    route = respx.get("https://eodhd.com/api/intraday/GBPUSD.FOREX").mock(
        side_effect=[
            httpx.Response(429, text="Rate limit exceeded"),
            httpx.Response(
                200,
                json=[
                    {
                        "datetime": "2026-03-01 12:00:00",
                        "open": 1.26,
                        "high": 1.27,
                        "low": 1.25,
                        "close": 1.265,
                        "volume": 800,
                    },
                ],
            ),
        ]
    )
    async with httpx.AsyncClient() as client:
        df = await provider.fetch_bars(
            "GBPUSD", date(2026, 3, 1), date(2026, 3, 1), client, interval="1h"
        )

    assert route.call_count == 2
    assert len(df) == 1


async def test_eodhd_fetch_bars_invalid_interval():
    """EODHD fetch_bars should raise ValueError for unsupported intervals."""
    provider = EodhdProvider(api_key="test_key")
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="Unsupported interval"):
            await provider.fetch_bars(
                "EURUSD", date(2026, 3, 1), date(2026, 3, 1), client, interval="daily"
            )


async def test_eodhd_fetch_daily_bars_raises():
    """EODHD fetch_daily_bars should raise ValueError (intraday-only provider)."""
    provider = EodhdProvider(api_key="test_key")
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="intraday-only"):
            await provider.fetch_daily_bars(
                "EURUSD", date(2026, 3, 1), date(2026, 3, 1), client
            )
