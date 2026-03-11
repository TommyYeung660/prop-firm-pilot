"""Tests for the market-data hub and symbol-level fallback behavior."""

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from src.data.fx_tick_aggregator import FXTickAggregator
from src.data.fx_websocket_client import EODHDFXWebSocketClient, WebSocketTick
from src.data.market_data_hub import MarketDataHub


class DummyProvider:
    """Simple async provider stub for REST fallback and warmup tests."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, str, date, date]] = []

    async def fetch_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        client,
        interval: str = "daily",
    ) -> pd.DataFrame:
        self.calls.append((symbol, interval, start_date, end_date))
        return pd.DataFrame(self.rows)


def _tick(symbol: str, bid: float, ask: float, dt: datetime) -> WebSocketTick:
    return WebSocketTick(
        symbol=symbol,
        bid=bid,
        ask=ask,
        timestamp_ms=int(dt.timestamp() * 1000),
    )


@pytest.mark.asyncio
async def test_market_data_hub_uses_rest_fallback_for_stale_symbol() -> None:
    provider = DummyProvider(
        [
            {
                "datetime": pd.Timestamp("2026-03-11T12:00:00Z"),
                "open": 1.10,
                "high": 1.11,
                "low": 1.09,
                "close": 1.105,
                "volume": 0,
            }
        ]
    )
    hub = MarketDataHub(
        aggregator=FXTickAggregator(),
        websocket_client=EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD"]),
        rest_provider=provider,
        symbols=["EURUSD"],
    )
    hub.mark_symbol_stale("EURUSD")

    bars = await hub.get_bars("EURUSD", "5m", 50)

    assert bars.source == "rest_fallback"
    assert provider.calls[0][0] == "EURUSD"


@pytest.mark.asyncio
async def test_market_data_hub_prefers_websocket_cache_for_healthy_symbol() -> None:
    dt = datetime(2026, 3, 11, 12, 0, 5, tzinfo=timezone.utc)
    aggregator = FXTickAggregator()
    aggregator.add_tick(_tick("EURUSD", 1.10, 1.11, dt))
    aggregator.close_elapsed_bars(now=datetime(2026, 3, 11, 12, 1, 1, tzinfo=timezone.utc))

    client = EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD"], stale_after_seconds=45)
    client._record_tick(_tick("EURUSD", 1.10, 1.11, dt))
    hub = MarketDataHub(
        aggregator=aggregator,
        websocket_client=client,
        rest_provider=DummyProvider([]),
        symbols=["EURUSD"],
    )

    bars = await hub.get_bars("EURUSD", "1m", 10)

    assert bars.source == "websocket_cache"
    assert len(bars.bars) == 1


@pytest.mark.asyncio
async def test_market_data_hub_tracks_quote_freshness_separately_from_bar_freshness() -> None:
    aggregator = FXTickAggregator()
    aggregator.add_tick(
        _tick(
            "EURUSD",
            1.10,
            1.11,
            datetime(2026, 3, 11, 12, 0, 5, tzinfo=timezone.utc),
        )
    )
    aggregator.close_elapsed_bars(now=datetime(2026, 3, 11, 12, 1, 1, tzinfo=timezone.utc))

    provider = DummyProvider(
        [
            {
                "datetime": pd.Timestamp("2026-03-11T12:01:00Z"),
                "open": 1.11,
                "high": 1.12,
                "low": 1.10,
                "close": 1.115,
                "volume": 0,
            }
        ]
    )
    client = EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD"], stale_after_seconds=30)
    client._record_tick(
        _tick(
            "EURUSD",
            1.10,
            1.11,
            datetime(2026, 3, 11, 12, 0, 5, tzinfo=timezone.utc),
        )
    )
    hub = MarketDataHub(
        aggregator=aggregator,
        websocket_client=client,
        rest_provider=provider,
        symbols=["EURUSD"],
        now_provider=lambda: datetime(2026, 3, 11, 12, 1, 10, tzinfo=timezone.utc),
    )

    quote = await hub.get_quote("EURUSD")
    bars = await hub.get_bars("EURUSD", "1m", 10)

    assert quote.source == "rest_fallback"
    assert bars.source == "websocket_cache"


@pytest.mark.asyncio
async def test_market_data_hub_warmup_uses_rest_backfill() -> None:
    provider = DummyProvider(
        [
            {
                "datetime": pd.Timestamp("2026-03-11T12:00:00Z"),
                "open": 1.10,
                "high": 1.11,
                "low": 1.09,
                "close": 1.105,
                "volume": 0,
            },
            {
                "datetime": pd.Timestamp("2026-03-11T12:05:00Z"),
                "open": 1.105,
                "high": 1.115,
                "low": 1.10,
                "close": 1.11,
                "volume": 0,
            },
        ]
    )
    hub = MarketDataHub(
        aggregator=FXTickAggregator(),
        websocket_client=EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD"]),
        rest_provider=provider,
        symbols=["EURUSD"],
    )

    await hub.warmup()
    bars = await hub.get_bars("EURUSD", "5m", 10)

    assert bars.source == "warmup_cache"
    assert len(bars.bars) == 2
    assert provider.calls
