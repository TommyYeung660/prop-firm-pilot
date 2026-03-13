"""Tests for the market-data hub and symbol-level fallback behavior."""

import asyncio
from datetime import date, datetime, timezone
from unittest.mock import patch

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
        now_provider=lambda: datetime(2026, 3, 11, 12, 1, 10, tzinfo=timezone.utc),
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
        now_provider=lambda: datetime(2026, 3, 11, 12, 6, tzinfo=timezone.utc),
    )

    await hub.warmup()
    bars = await hub.get_bars("EURUSD", "5m", 10)

    assert bars.source == "warmup_cache"
    assert len(bars.bars) == 2
    assert provider.calls


@pytest.mark.asyncio
async def test_market_data_hub_quote_fallback_refreshes_from_cached_tail() -> None:
    now = datetime(2026, 3, 12, 12, 0, tzinfo=timezone.utc)
    provider = DummyProvider(
        [
            {
                "datetime": pd.Timestamp("2026-03-12T11:59:00Z"),
                "open": 1.11,
                "high": 1.12,
                "low": 1.10,
                "close": 1.115,
                "volume": 0,
            }
        ]
    )
    hub = MarketDataHub(
        aggregator=FXTickAggregator(),
        websocket_client=EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD"]),
        rest_provider=provider,
        symbols=["EURUSD"],
        bar_cache_max_age_seconds=60,
        now_provider=lambda: now,
    )
    hub._warm_cache[("EURUSD", "1m")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-12T11:55:00Z"),
                "open": 1.10,
                "high": 1.11,
                "low": 1.09,
                "close": 1.105,
                "volume": 0,
            }
        ]
    )
    hub.mark_symbol_stale("EURUSD")

    result = await hub.get_quote("EURUSD")

    assert result.source == "rest_fallback"
    assert provider.calls[0][2] == date(2026, 3, 12)


@pytest.mark.asyncio
async def test_market_data_hub_returns_no_quote_when_rest_fallback_tail_is_still_stale() -> None:
    now = datetime(2026, 3, 12, 12, 0, tzinfo=timezone.utc)
    provider = DummyProvider(
        [
            {
                "datetime": pd.Timestamp("2026-03-12T10:00:00Z"),
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
        bar_cache_max_age_seconds=60,
        now_provider=lambda: now,
    )
    hub._warm_cache[("EURUSD", "1m")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-12T10:00:00Z"),
                "open": 1.10,
                "high": 1.11,
                "low": 1.09,
                "close": 1.105,
                "volume": 0,
            }
        ]
    )
    hub.mark_symbol_stale("EURUSD")

    result = await hub.get_quote("EURUSD")

    assert result.source == "rest_fallback"
    assert result.quote is None
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_market_data_hub_refreshes_stale_warm_cache_incrementally() -> None:
    now = datetime(2026, 3, 12, 12, 0, tzinfo=timezone.utc)
    provider = DummyProvider(
        [
            {
                "datetime": pd.Timestamp("2026-03-12T11:55:00Z"),
                "open": 1.11,
                "high": 1.12,
                "low": 1.10,
                "close": 1.115,
                "volume": 0,
            }
        ]
    )
    hub = MarketDataHub(
        aggregator=FXTickAggregator(),
        websocket_client=EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD"]),
        rest_provider=provider,
        symbols=["EURUSD"],
        bar_cache_max_age_seconds=60,
        now_provider=lambda: now,
    )
    hub._warm_cache[("EURUSD", "5m")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-12T10:00:00Z"),
                "open": 1.10,
                "high": 1.11,
                "low": 1.09,
                "close": 1.105,
                "volume": 0,
            }
        ]
    )

    result = await hub.get_bars("EURUSD", "5m", 10)

    assert result.source == "rest_fallback"
    assert provider.calls[0][2] == date(2026, 3, 12)
    assert len(result.bars) == 2
    assert float(result.bars.iloc[-1]["close"]) == pytest.approx(1.115)


@pytest.mark.asyncio
async def test_market_data_hub_skips_repeated_quote_refresh_when_rest_tail_has_no_progress(
) -> None:
    current_now = datetime(2026, 3, 12, 12, 0, tzinfo=timezone.utc)
    provider = DummyProvider(
        [
            {
                "datetime": pd.Timestamp("2026-03-12T10:00:00Z"),
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
        bar_cache_max_age_seconds=60,
        rest_refresh_cooldown_seconds=300,
        now_provider=lambda: current_now,
    )
    hub._warm_cache[("EURUSD", "1m")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-12T10:00:00Z"),
                "open": 1.10,
                "high": 1.11,
                "low": 1.09,
                "close": 1.105,
                "volume": 0,
            }
        ]
    )
    hub.mark_symbol_stale("EURUSD")

    first = await hub.get_quote("EURUSD")
    current_now = datetime(2026, 3, 12, 12, 0, 30, tzinfo=timezone.utc)
    second = await hub.get_quote("EURUSD")

    assert first.source == "rest_fallback"
    assert second.source == "rest_fallback"
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_market_data_hub_suppresses_repeated_stale_quote_warnings_within_cooldown() -> None:
    current_now = datetime(2026, 3, 12, 12, 0, tzinfo=timezone.utc)
    provider = DummyProvider(
        [
            {
                "datetime": pd.Timestamp("2026-03-12T10:00:00Z"),
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
        bar_cache_max_age_seconds=60,
        rest_refresh_cooldown_seconds=300,
        now_provider=lambda: current_now,
    )
    hub._warm_cache[("EURUSD", "1m")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-12T10:00:00Z"),
                "open": 1.10,
                "high": 1.11,
                "low": 1.09,
                "close": 1.105,
                "volume": 0,
            }
        ]
    )
    hub.mark_symbol_stale("EURUSD")

    with patch("src.data.market_data_hub.logger.warning") as mock_warning:
        await hub.get_quote("EURUSD")
        current_now = datetime(2026, 3, 12, 12, 0, 30, tzinfo=timezone.utc)
        await hub.get_quote("EURUSD")

    assert len(provider.calls) == 1
    assert mock_warning.call_count == 1


@pytest.mark.asyncio
async def test_market_data_hub_skips_repeated_5m_bar_refresh_when_rest_tail_has_no_progress(
) -> None:
    current_now = datetime(2026, 3, 12, 12, 0, tzinfo=timezone.utc)
    provider = DummyProvider(
        [
            {
                "datetime": pd.Timestamp("2026-03-12T10:00:00Z"),
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
        bar_cache_max_age_seconds=60,
        rest_refresh_cooldown_seconds=300,
        now_provider=lambda: current_now,
    )
    hub._warm_cache[("EURUSD", "5m")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-12T10:00:00Z"),
                "open": 1.10,
                "high": 1.11,
                "low": 1.09,
                "close": 1.105,
                "volume": 0,
            }
        ]
    )

    first = await hub.get_bars("EURUSD", "5m", 10)
    current_now = datetime(2026, 3, 12, 12, 0, 30, tzinfo=timezone.utc)
    second = await hub.get_bars("EURUSD", "5m", 10)

    assert first.source == "rest_fallback"
    assert second.source == "rest_fallback"
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_market_data_hub_skips_repeated_1h_bar_refresh_when_rest_tail_has_no_progress(
) -> None:
    current_now = datetime(2026, 3, 12, 12, 0, tzinfo=timezone.utc)
    provider = DummyProvider(
        [
            {
                "datetime": pd.Timestamp("2026-03-12T08:00:00Z"),
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
        bar_cache_max_age_seconds=300,
        rest_refresh_cooldown_seconds=300,
        now_provider=lambda: current_now,
    )
    hub._warm_cache[("EURUSD", "1h")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-12T08:00:00Z"),
                "open": 1.10,
                "high": 1.11,
                "low": 1.09,
                "close": 1.105,
                "volume": 0,
            }
        ]
    )

    first = await hub.get_bars("EURUSD", "1h", 10)
    current_now = datetime(2026, 3, 12, 12, 0, 30, tzinfo=timezone.utc)
    second = await hub.get_bars("EURUSD", "1h", 10)

    assert first.source == "rest_fallback"
    assert second.source == "rest_fallback"
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_market_data_hub_deduplicates_concurrent_quote_refreshes() -> None:
    current_now = datetime(2026, 3, 12, 12, 0, tzinfo=timezone.utc)
    provider = DummyProvider(
        [
            {
                "datetime": pd.Timestamp("2026-03-12T10:00:00Z"),
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
        bar_cache_max_age_seconds=60,
        rest_refresh_cooldown_seconds=300,
        now_provider=lambda: current_now,
    )
    hub._warm_cache[("EURUSD", "1m")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-12T10:00:00Z"),
                "open": 1.10,
                "high": 1.11,
                "low": 1.09,
                "close": 1.105,
                "volume": 0,
            }
        ]
    )
    hub.mark_symbol_stale("EURUSD")

    first, second = await asyncio.gather(hub.get_quote("EURUSD"), hub.get_quote("EURUSD"))

    assert first.source == "rest_fallback"
    assert second.source == "rest_fallback"
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_market_data_hub_uses_timeframe_sized_cooldown_for_1h_refreshes() -> None:
    current_now = datetime(2026, 3, 12, 12, 0, tzinfo=timezone.utc)
    provider = DummyProvider(
        [
            {
                "datetime": pd.Timestamp("2026-03-12T08:00:00Z"),
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
        bar_cache_max_age_seconds=300,
        rest_refresh_cooldown_seconds=300,
        now_provider=lambda: current_now,
    )
    hub._warm_cache[("EURUSD", "1h")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-12T08:00:00Z"),
                "open": 1.10,
                "high": 1.11,
                "low": 1.09,
                "close": 1.105,
                "volume": 0,
            }
        ]
    )

    first = await hub.get_bars("EURUSD", "1h", 10)
    current_now = datetime(2026, 3, 12, 12, 6, tzinfo=timezone.utc)
    second = await hub.get_bars("EURUSD", "1h", 10)

    assert first.source == "rest_fallback"
    assert second.source == "rest_fallback"
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_market_data_hub_finalizes_elapsed_rollup_bars_before_lookup() -> None:
    aggregator = FXTickAggregator()
    for minute, price in enumerate([1.10, 1.11, 1.12, 1.13, 1.14]):
        aggregator.add_tick(
            _tick(
                "EURUSD",
                price,
                price + 0.01,
                datetime(2026, 3, 11, 12, minute, 10, tzinfo=timezone.utc),
            )
        )

    provider = DummyProvider([])
    hub = MarketDataHub(
        aggregator=aggregator,
        websocket_client=EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD"]),
        rest_provider=provider,
        symbols=["EURUSD"],
        now_provider=lambda: datetime(2026, 3, 11, 12, 5, 1, tzinfo=timezone.utc),
    )

    result = await hub.get_bars("EURUSD", "5m", 10)

    assert result.source == "websocket_cache"
    assert len(result.bars) == 1
    assert provider.calls == []
