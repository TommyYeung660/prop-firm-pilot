"""Tests for the market-data hub and symbol-level fallback behavior."""

import asyncio
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

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
async def test_market_data_hub_prefers_fresh_closed_1h_websocket_bar_before_rest_refresh(
) -> None:
    provider = DummyProvider([])
    aggregator = FXTickAggregator()
    aggregator.add_tick(
        _tick(
            "EURUSD",
            1.10,
            1.11,
            datetime(2026, 3, 11, 4, 52, 10, tzinfo=timezone.utc),
        )
    )
    aggregator.add_tick(
        _tick(
            "EURUSD",
            1.1001,
            1.1101,
            datetime(2026, 3, 11, 5, 0, 10, tzinfo=timezone.utc),
        )
    )
    aggregator.close_elapsed_bars(now=datetime(2026, 3, 11, 5, 1, tzinfo=timezone.utc))

    hub = MarketDataHub(
        aggregator=aggregator,
        websocket_client=EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD"]),
        rest_provider=provider,
        symbols=["EURUSD"],
        bar_cache_max_age_seconds=3600,
        now_provider=lambda: datetime(2026, 3, 11, 5, 52, tzinfo=timezone.utc),
    )

    bars = await hub.get_bars("EURUSD", "1h", 10)

    assert bars.source == "websocket_cache"
    assert len(bars.bars) == 1
    assert provider.calls == []


@pytest.mark.asyncio
async def test_market_data_hub_does_not_prefer_short_1h_websocket_history_over_stale_rest_cache(
) -> None:
    provider = DummyProvider([])
    aggregator = FXTickAggregator()
    tick_time = datetime(2026, 3, 27, 1, 5, tzinfo=timezone.utc)
    end_time = datetime(2026, 3, 27, 4, 4, tzinfo=timezone.utc)
    while tick_time <= end_time:
        aggregator.add_tick(_tick("EURUSD", 1.10, 1.11, tick_time))
        tick_time += timedelta(minutes=1)
    now = datetime(2026, 3, 27, 4, 5, tzinfo=timezone.utc)
    aggregator.close_elapsed_bars(now=now)

    hub = MarketDataHub(
        aggregator=aggregator,
        websocket_client=EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD"]),
        rest_provider=provider,
        symbols=["EURUSD"],
        bar_cache_max_age_seconds=3600,
        now_provider=lambda: now,
    )
    hub._warm_cache[("EURUSD", "1h")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-23T19:00:00Z") + pd.Timedelta(hours=idx),
                "open": 1.10,
                "high": 1.11,
                "low": 1.09,
                "close": 1.105,
                "volume": 0,
            }
            for idx in range(80)
        ]
    )

    bars = await hub.get_bars("EURUSD", "1h", 80)

    assert bars.source == "rest_fallback"
    assert len(bars.bars) == 80
    assert provider.calls == [("EURUSD", "1h", date(2026, 3, 27), date(2026, 3, 27))]


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
    assert bars.source == "warmup_cache"


@pytest.mark.asyncio
async def test_market_data_hub_prefers_broker_quote_over_websocket_cache() -> None:
    tick_dt = datetime(2026, 3, 17, 3, 0, 10, tzinfo=timezone.utc)
    now = datetime(2026, 3, 17, 3, 0, 20, tzinfo=timezone.utc)
    aggregator = FXTickAggregator()
    aggregator.add_tick(_tick("EURUSD", 1.1000, 1.1002, tick_dt))

    client = EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD"], stale_after_seconds=60)
    client._connected = True
    client._record_tick(_tick("EURUSD", 1.1000, 1.1002, tick_dt))
    broker_quote_provider = AsyncMock(
        return_value={
            "bid": 1.2001,
            "ask": 1.2003,
            "timestampMs": int((now.timestamp() - 2) * 1000),
        }
    )
    hub = MarketDataHub(
        aggregator=aggregator,
        websocket_client=client,
        rest_provider=DummyProvider([]),
        symbols=["EURUSD"],
        now_provider=lambda: now,
        broker_quote_provider=broker_quote_provider,
    )

    result = await hub.get_quote("EURUSD")

    assert result.source == "broker_quote"
    assert result.quote is not None
    assert result.quote["bid"] == pytest.approx(1.2001)
    assert result.quote["ask"] == pytest.approx(1.2003)
    broker_quote_provider.assert_awaited_once_with("EURUSD")


@pytest.mark.asyncio
async def test_market_data_hub_uses_realtime_rest_quote_when_websocket_is_missing() -> None:
    now = datetime(2026, 3, 17, 3, 0, 20, tzinfo=timezone.utc)
    aggregator = FXTickAggregator()
    quote_time = datetime(2026, 3, 17, 3, 0, 10, tzinfo=timezone.utc)
    realtime_quote_provider = AsyncMock(
        return_value={
            "symbol": "EURUSD",
            "bid": 1.2001,
            "ask": 1.2001,
            "mid": 1.2001,
            "timestamp_ms": int(quote_time.timestamp() * 1000),
        }
    )
    hub = MarketDataHub(
        aggregator=aggregator,
        websocket_client=EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD"]),
        rest_provider=DummyProvider([]),
        symbols=["EURUSD"],
        now_provider=lambda: now,
        realtime_quote_provider=realtime_quote_provider,
    )

    result = await hub.get_quote("EURUSD")

    assert result.source == "rest_realtime"
    assert result.quote is not None
    assert aggregator.latest_quote("EURUSD") is not None
    assert aggregator.latest_quote("EURUSD")["timestamp_ms"] == result.quote["timestamp_ms"]
    realtime_quote_provider.assert_awaited_once_with("EURUSD")


@pytest.mark.asyncio
async def test_market_data_hub_does_not_refeed_duplicate_realtime_quote_timestamp() -> None:
    now = datetime(2026, 3, 17, 3, 0, 20, tzinfo=timezone.utc)
    aggregator = FXTickAggregator()
    quote_time = datetime(2026, 3, 17, 3, 0, 10, tzinfo=timezone.utc)
    realtime_quote = {
        "symbol": "EURUSD",
        "bid": 1.2001,
        "ask": 1.2001,
        "mid": 1.2001,
        "timestamp_ms": int(quote_time.timestamp() * 1000),
    }
    realtime_quote_provider = AsyncMock(return_value=realtime_quote)
    hub = MarketDataHub(
        aggregator=aggregator,
        websocket_client=EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD"]),
        rest_provider=DummyProvider([]),
        symbols=["EURUSD"],
        now_provider=lambda: now,
        realtime_quote_provider=realtime_quote_provider,
    )

    with patch.object(aggregator, "add_tick", wraps=aggregator.add_tick) as mock_add_tick:
        first = await hub.get_quote("EURUSD")
        second = await hub.get_quote("EURUSD")

    assert first.source == "rest_realtime"
    assert second.source == "rest_realtime"
    assert mock_add_tick.call_count == 1


@pytest.mark.asyncio
async def test_market_data_hub_uses_realtime_rest_fed_ticks_to_build_fresh_5m_bars() -> None:
    current_now = datetime(2026, 3, 17, 4, 0, 30, tzinfo=timezone.utc)
    realtime_ticks = [
        datetime(2026, 3, 17, 4, 0, 10, tzinfo=timezone.utc),
        datetime(2026, 3, 17, 4, 1, 10, tzinfo=timezone.utc),
        datetime(2026, 3, 17, 4, 2, 10, tzinfo=timezone.utc),
        datetime(2026, 3, 17, 4, 3, 10, tzinfo=timezone.utc),
        datetime(2026, 3, 17, 4, 4, 10, tzinfo=timezone.utc),
        datetime(2026, 3, 17, 4, 5, 10, tzinfo=timezone.utc),
    ]
    realtime_quote_provider = AsyncMock(
        side_effect=[
            {
                "symbol": "EURUSD",
                "bid": 1.1000 + idx * 0.0001,
                "ask": 1.1000 + idx * 0.0001,
                "mid": 1.1000 + idx * 0.0001,
                "timestamp_ms": int(ts.timestamp() * 1000),
            }
            for idx, ts in enumerate(realtime_ticks)
        ]
    )
    provider = DummyProvider(
        [
            {
                "datetime": pd.Timestamp("2026-03-17T02:00:00Z"),
                "open": 1.08,
                "high": 1.09,
                "low": 1.07,
                "close": 1.085,
                "volume": 0,
            }
        ]
    )
    hub = MarketDataHub(
        aggregator=FXTickAggregator(),
        websocket_client=EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD"]),
        rest_provider=provider,
        symbols=["EURUSD"],
        now_provider=lambda: current_now,
        realtime_quote_provider=realtime_quote_provider,
        bar_cache_max_age_seconds=300,
    )

    for ts in realtime_ticks:
        current_now = ts + pd.Timedelta(seconds=20)
        result = await hub.get_quote("EURUSD")
        assert result.source == "rest_realtime"

    current_now = datetime(2026, 3, 17, 4, 6, 30, tzinfo=timezone.utc)
    bars = await hub.get_bars("EURUSD", "5m", 10)

    assert bars.source == "websocket_cache"
    assert not bars.bars.empty
    assert pd.Timestamp(bars.bars.iloc[-1]["datetime"]) == pd.Timestamp("2026-03-17T04:00:00Z")


@pytest.mark.asyncio
async def test_market_data_hub_prefers_api_cache_for_bars_when_websocket_also_has_data() -> None:
    now = datetime(2026, 3, 17, 4, 5, 1, tzinfo=timezone.utc)
    aggregator = FXTickAggregator()
    for minute, price in enumerate([1.10, 1.11, 1.12, 1.13, 1.14]):
        aggregator.add_tick(
            _tick(
                "EURUSD",
                price,
                price + 0.0002,
                datetime(2026, 3, 17, 4, minute, 10, tzinfo=timezone.utc),
            )
        )
    aggregator.close_elapsed_bars(now=now)

    hub = MarketDataHub(
        aggregator=aggregator,
        websocket_client=EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD"]),
        rest_provider=DummyProvider([]),
        symbols=["EURUSD"],
        now_provider=lambda: now,
    )
    hub._warm_cache[("EURUSD", "5m")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-17T04:00:00Z"),
                "open": 1.0900,
                "high": 1.1300,
                "low": 1.0800,
                "close": 1.1200,
                "volume": 0,
            }
        ]
    )

    result = await hub.get_bars("EURUSD", "5m", 10)

    assert result.source == "warmup_cache"
    assert float(result.bars.iloc[-1]["close"]) == pytest.approx(1.1200)


@pytest.mark.asyncio
async def test_market_data_hub_prefers_fresh_websocket_closed_5m_bars_before_rest_refresh() -> None:
    now = datetime(2026, 3, 17, 4, 5, 30, tzinfo=timezone.utc)
    provider = DummyProvider(
        [
            {
                "datetime": pd.Timestamp("2026-03-17T03:00:00Z"),
                "open": 1.08,
                "high": 1.09,
                "low": 1.07,
                "close": 1.085,
                "volume": 0,
            }
        ]
    )
    aggregator = FXTickAggregator()
    for minute, price in enumerate([1.10, 1.11, 1.12, 1.13, 1.14]):
        aggregator.add_tick(
            _tick(
                "EURUSD",
                price,
                price + 0.0002,
                datetime(2026, 3, 17, 4, minute, 10, tzinfo=timezone.utc),
            )
        )
    aggregator.close_elapsed_bars(now=datetime(2026, 3, 17, 4, 5, 1, tzinfo=timezone.utc))

    hub = MarketDataHub(
        aggregator=aggregator,
        websocket_client=EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD"]),
        rest_provider=provider,
        symbols=["EURUSD"],
        now_provider=lambda: now,
        bar_cache_max_age_seconds=300,
    )
    hub._warm_cache[("EURUSD", "5m")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-17T03:00:00Z"),
                "open": 1.0800,
                "high": 1.0900,
                "low": 1.0700,
                "close": 1.0850,
                "volume": 0,
            }
        ]
    )

    result = await hub.get_bars("EURUSD", "5m", 10)

    assert result.source == "websocket_cache"
    assert len(provider.calls) == 0
    assert pd.Timestamp(result.bars.iloc[-1]["datetime"]) == pd.Timestamp("2026-03-17T04:00:00Z")


@pytest.mark.asyncio
async def test_market_data_hub_skips_rest_fallback_warning_when_websocket_closed_5m_bars_are_used(
) -> None:
    now = datetime(2026, 3, 17, 4, 5, 30, tzinfo=timezone.utc)
    provider = DummyProvider(
        [
            {
                "datetime": pd.Timestamp("2026-03-17T03:00:00Z"),
                "open": 1.08,
                "high": 1.09,
                "low": 1.07,
                "close": 1.085,
                "volume": 0,
            }
        ]
    )
    aggregator = FXTickAggregator()
    for minute, price in enumerate([1.10, 1.11, 1.12, 1.13, 1.14]):
        aggregator.add_tick(
            _tick(
                "EURUSD",
                price,
                price + 0.0002,
                datetime(2026, 3, 17, 4, minute, 10, tzinfo=timezone.utc),
            )
        )
    aggregator.close_elapsed_bars(now=datetime(2026, 3, 17, 4, 5, 1, tzinfo=timezone.utc))

    hub = MarketDataHub(
        aggregator=aggregator,
        websocket_client=EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD"]),
        rest_provider=provider,
        symbols=["EURUSD"],
        now_provider=lambda: now,
        bar_cache_max_age_seconds=300,
    )
    hub._warm_cache[("EURUSD", "5m")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-17T03:00:00Z"),
                "open": 1.0800,
                "high": 1.0900,
                "low": 1.0700,
                "close": 1.0850,
                "volume": 0,
            }
        ]
    )

    with patch("src.data.market_data_hub.logger.warning") as mock_warning:
        result = await hub.get_bars("EURUSD", "5m", 10)

    assert result.source == "websocket_cache"
    assert mock_warning.call_count == 0
    assert len(provider.calls) == 0


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


def test_feed_status_reports_lifecycle_and_closed_bar_counts() -> None:
    initialized_at = datetime(2026, 3, 17, 3, 0, tzinfo=timezone.utc)
    now = datetime(2026, 3, 17, 3, 6, 30, tzinfo=timezone.utc)
    aggregator = FXTickAggregator()
    ticks = [
        _tick("EURUSD", 1.10, 1.11, datetime(2026, 3, 17, 3, 0, 5, tzinfo=timezone.utc)),
        _tick("EURUSD", 1.1001, 1.1101, datetime(2026, 3, 17, 3, 1, 5, tzinfo=timezone.utc)),
        _tick("EURUSD", 1.1002, 1.1102, datetime(2026, 3, 17, 3, 2, 5, tzinfo=timezone.utc)),
        _tick("EURUSD", 1.1003, 1.1103, datetime(2026, 3, 17, 3, 3, 5, tzinfo=timezone.utc)),
        _tick("EURUSD", 1.1004, 1.1104, datetime(2026, 3, 17, 3, 4, 5, tzinfo=timezone.utc)),
        _tick("EURUSD", 1.1005, 1.1105, datetime(2026, 3, 17, 3, 5, 5, tzinfo=timezone.utc)),
    ]
    for tick in ticks:
        aggregator.add_tick(tick)
    aggregator.close_elapsed_bars(now=now)

    client = EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD", "USDCHF"])
    hub = MarketDataHub(
        aggregator=aggregator,
        websocket_client=client,
        rest_provider=DummyProvider([]),
        symbols=["EURUSD", "USDCHF"],
        now_provider=lambda: now,
    )
    hub._initialized_at = initialized_at

    status = hub.feed_status()

    assert status["initialized_at"] == "2026-03-17T03:00:00+00:00"
    assert status["uptime_seconds"] == 390
    assert status["websocket_closed_bar_counts"]["EURUSD"] == {"1m": 6, "5m": 1, "1h": 0}
    assert status["websocket_closed_bar_counts"]["USDCHF"] == {"1m": 0, "5m": 0, "1h": 0}


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
        realtime_quote_provider=AsyncMock(return_value=None),
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
async def test_market_data_hub_blocks_entry_when_feed_is_degraded_and_rest_data_is_stale() -> None:
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
    client = EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD"])
    client._last_error = "keepalive ping timeout"

    hub = MarketDataHub(
        aggregator=FXTickAggregator(),
        websocket_client=client,
        rest_provider=provider,
        symbols=["EURUSD"],
        bar_cache_max_age_seconds=60,
        now_provider=lambda: now,
    )

    readiness = await hub.get_entry_readiness("EURUSD")

    assert readiness.entry_safe is False
    assert readiness.block_reason == "market_data.quote_unavailable"
    assert readiness.websocket_state == "degraded"
    assert readiness.ws_last_error == "keepalive ping timeout"
    assert readiness.quote_source == "rest_fallback"
    assert readiness.quote_available is False


@pytest.mark.asyncio
async def test_entry_readiness_marks_startup_5m_gap_as_retryable() -> None:
    tick_time = datetime(2026, 3, 17, 6, 42, 10, tzinfo=timezone.utc)
    aggregator = FXTickAggregator()
    aggregator.add_tick(_tick("EURUSD", 1.10, 1.1002, tick_time))

    client = EODHDFXWebSocketClient(
        api_token="token",
        symbols=["EURUSD"],
        stale_after_seconds=86400,
    )
    client._connected = True
    client._record_tick(_tick("EURUSD", 1.10, 1.1002, tick_time))

    hub = MarketDataHub(
        aggregator=aggregator,
        websocket_client=client,
        rest_provider=DummyProvider([]),
        symbols=["EURUSD"],
        now_provider=lambda: datetime(2026, 3, 17, 6, 42, 30, tzinfo=timezone.utc),
    )
    hub._warm_cache[("EURUSD", "1h")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-17T05:00:00Z"),
                "open": 1.0995,
                "high": 1.1005,
                "low": 1.0990,
                "close": 1.1000,
                "volume": 0,
            }
        ]
    )

    readiness = await hub.get_entry_readiness("EURUSD")

    assert readiness.entry_safe is True
    assert readiness.block_reason == ""
    assert readiness.requires_tactical_retry is True
    assert readiness.pending_reason == "market_data.startup_5m_bar_pending"
    assert readiness.websocket_state == "healthy"
    assert readiness.quote_source == "websocket_cache"
    assert readiness.quote_available is True
    assert readiness.bars_5m_fresh is False
    assert readiness.bars_1h_fresh is True


@pytest.mark.asyncio
async def test_entry_readiness_marks_stale_same_day_5m_rest_gap_as_startup_retryable(
) -> None:
    now = datetime(2026, 3, 23, 3, 31, 30, tzinfo=timezone.utc)
    tick_time = datetime(2026, 3, 23, 3, 31, 5, tzinfo=timezone.utc)
    aggregator = FXTickAggregator()
    aggregator.add_tick(_tick("AUDJPY", 100.0, 100.02, tick_time))

    client = EODHDFXWebSocketClient(
        api_token="token",
        symbols=["AUDJPY"],
        stale_after_seconds=86400,
    )
    client._connected = True
    client._record_tick(_tick("AUDJPY", 100.0, 100.02, tick_time))

    hub = MarketDataHub(
        aggregator=aggregator,
        websocket_client=client,
        rest_provider=DummyProvider([]),
        symbols=["AUDJPY"],
        now_provider=lambda: now,
        broker_quote_provider=AsyncMock(
            return_value={
                "bid": 100.0,
                "ask": 100.02,
                "timestampMs": int(tick_time.timestamp() * 1000),
            }
        ),
    )
    hub._warm_cache[("AUDJPY", "5m")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-23T02:00:00Z"),
                "open": 99.9,
                "high": 100.1,
                "low": 99.8,
                "close": 100.0,
                "volume": 0,
            }
        ]
    )
    hub._warm_cache[("AUDJPY", "1h")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-23T02:00:00Z"),
                "open": 99.8,
                "high": 100.2,
                "low": 99.7,
                "close": 100.0,
                "volume": 0,
            }
        ]
    )

    readiness = await hub.get_entry_readiness("AUDJPY")

    assert readiness.entry_safe is True
    assert readiness.block_reason == ""
    assert readiness.requires_tactical_retry is True
    assert readiness.pending_reason == "market_data.startup_5m_bar_pending"
    assert readiness.quote_source == "broker_quote"
    assert readiness.quote_available is True
    assert readiness.bars_5m_source == "rest_fallback"
    assert readiness.bars_5m_fresh is False
    assert readiness.bars_1h_fresh is True
    assert readiness.websocket_state == "healthy"


@pytest.mark.asyncio
async def test_entry_readiness_allows_scanner_progress_when_1h_bars_are_stale_but_5m_is_fresh(
) -> None:
    now = datetime(2026, 3, 19, 4, 15, tzinfo=timezone.utc)
    tick_time = datetime(2026, 3, 19, 4, 14, 50, tzinfo=timezone.utc)
    aggregator = FXTickAggregator()
    aggregator.add_tick(_tick("EURUSD", 1.10, 1.1002, tick_time))

    client = EODHDFXWebSocketClient(
        api_token="token",
        symbols=["EURUSD"],
        stale_after_seconds=86400,
    )
    client._connected = True
    client._record_tick(_tick("EURUSD", 1.10, 1.1002, tick_time))

    hub = MarketDataHub(
        aggregator=aggregator,
        websocket_client=client,
        rest_provider=DummyProvider([]),
        symbols=["EURUSD"],
        now_provider=lambda: now,
        broker_quote_provider=AsyncMock(
            return_value={
                "bid": 1.1001,
                "ask": 1.1003,
                "timestampMs": int(tick_time.timestamp() * 1000),
            }
        ),
    )
    hub._warm_cache[("EURUSD", "5m")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-19T04:10:00Z"),
                "open": 1.0998,
                "high": 1.1004,
                "low": 1.0994,
                "close": 1.1001,
                "volume": 0,
            }
        ]
    )
    hub._warm_cache[("EURUSD", "1h")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-19T02:00:00Z"),
                "open": 1.0980,
                "high": 1.1010,
                "low": 1.0970,
                "close": 1.0995,
                "volume": 0,
            }
        ]
    )

    readiness = await hub.get_entry_readiness("EURUSD")

    assert readiness.entry_safe is True
    assert readiness.block_reason == ""
    assert readiness.requires_tactical_retry is False
    assert readiness.pending_reason == ""
    assert readiness.quote_source == "broker_quote"
    assert readiness.quote_available is True
    assert readiness.bars_5m_fresh is True
    assert readiness.bars_1h_fresh is False
    assert readiness.websocket_state == "healthy"


@pytest.mark.asyncio
async def test_entry_readiness_marks_same_day_stale_5m_gap_as_retryable_even_when_1h_is_stale(
) -> None:
    now = datetime(2026, 3, 23, 6, 45, 30, tzinfo=timezone.utc)
    tick_time = datetime(2026, 3, 23, 6, 45, 5, tzinfo=timezone.utc)
    aggregator = FXTickAggregator()
    aggregator.add_tick(_tick("AUDJPY", 100.0, 100.02, tick_time))

    client = EODHDFXWebSocketClient(
        api_token="token",
        symbols=["AUDJPY"],
        stale_after_seconds=86400,
    )
    client._connected = True
    client._record_tick(_tick("AUDJPY", 100.0, 100.02, tick_time))

    hub = MarketDataHub(
        aggregator=aggregator,
        websocket_client=client,
        rest_provider=DummyProvider([]),
        symbols=["AUDJPY"],
        now_provider=lambda: now,
        broker_quote_provider=AsyncMock(
            return_value={
                "bid": 100.0,
                "ask": 100.02,
                "timestampMs": int(tick_time.timestamp() * 1000),
            }
        ),
    )
    hub._warm_cache[("AUDJPY", "5m")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-23T02:00:00Z"),
                "open": 99.9,
                "high": 100.1,
                "low": 99.8,
                "close": 100.0,
                "volume": 0,
            }
        ]
    )
    hub._warm_cache[("AUDJPY", "1h")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-23T02:00:00Z"),
                "open": 99.8,
                "high": 100.2,
                "low": 99.7,
                "close": 100.0,
                "volume": 0,
            }
        ]
    )

    readiness = await hub.get_entry_readiness("AUDJPY")

    assert readiness.entry_safe is True
    assert readiness.block_reason == ""
    assert readiness.requires_tactical_retry is True
    assert readiness.pending_reason == "market_data.startup_5m_bar_pending"
    assert readiness.quote_source == "broker_quote"
    assert readiness.quote_available is True
    assert readiness.bars_5m_source == "rest_fallback"
    assert readiness.bars_5m_fresh is False
    assert readiness.bars_1h_fresh is False
    assert readiness.websocket_state == "healthy"


@pytest.mark.asyncio
async def test_entry_readiness_marks_same_day_stale_5m_gap_as_retryable_without_websocket_health(
) -> None:
    now = datetime(2026, 3, 24, 6, 45, 30, tzinfo=timezone.utc)
    tick_time = datetime(2026, 3, 24, 6, 45, 5, tzinfo=timezone.utc)
    hub = MarketDataHub(
        aggregator=FXTickAggregator(),
        websocket_client=EODHDFXWebSocketClient(
            api_token="token",
            symbols=["USDCAD"],
            stale_after_seconds=45,
        ),
        rest_provider=DummyProvider([]),
        symbols=["USDCAD"],
        now_provider=lambda: now,
        broker_quote_provider=AsyncMock(
            return_value={
                "bid": 1.3735,
                "ask": 1.3737,
                "timestampMs": int(tick_time.timestamp() * 1000),
            }
        ),
    )
    hub._warm_cache[("USDCAD", "5m")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-24T02:00:00Z"),
                "open": 1.3720,
                "high": 1.3740,
                "low": 1.3710,
                "close": 1.3735,
                "volume": 0,
            }
        ]
    )
    hub._warm_cache[("USDCAD", "1h")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-24T02:00:00Z"),
                "open": 1.3715,
                "high": 1.3745,
                "low": 1.3705,
                "close": 1.3735,
                "volume": 0,
            }
        ]
    )

    readiness = await hub.get_entry_readiness("USDCAD")

    assert readiness.entry_safe is True
    assert readiness.block_reason == ""
    assert readiness.requires_tactical_retry is True
    assert readiness.pending_reason == "market_data.startup_5m_bar_pending"
    assert readiness.quote_source == "broker_quote"
    assert readiness.quote_available is True
    assert readiness.bars_5m_source == "rest_fallback"
    assert readiness.bars_5m_fresh is False
    assert readiness.bars_1h_fresh is False
    assert readiness.websocket_state == "disconnected"


@pytest.mark.asyncio
async def test_entry_readiness_blocks_when_non_websocket_quote_has_no_5m_bars() -> None:
    now = datetime(2026, 3, 19, 0, 15, tzinfo=timezone.utc)
    tick_time = datetime(2026, 3, 19, 0, 14, 50, tzinfo=timezone.utc)
    hub = MarketDataHub(
        aggregator=FXTickAggregator(),
        websocket_client=EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD"]),
        rest_provider=DummyProvider([]),
        symbols=["EURUSD"],
        now_provider=lambda: now,
        broker_quote_provider=AsyncMock(
            return_value={
                "bid": 1.1001,
                "ask": 1.1003,
                "timestampMs": int(tick_time.timestamp() * 1000),
            }
        ),
    )
    hub._warm_cache[("EURUSD", "1h")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-19T00:00:00Z"),
                "open": 1.0980,
                "high": 1.1010,
                "low": 1.0970,
                "close": 1.0995,
                "volume": 0,
            }
        ]
    )

    readiness = await hub.get_entry_readiness("EURUSD")

    assert readiness.entry_safe is False
    assert readiness.block_reason == "market_data.bars_5m_stale"
    assert readiness.requires_tactical_retry is False
    assert readiness.pending_reason == ""
    assert readiness.quote_source == "broker_quote"
    assert readiness.quote_available is True
    assert readiness.bars_5m_source == "rest_fallback"
    assert readiness.bars_5m_fresh is False
    assert readiness.bars_1h_source == "warmup_cache"
    assert readiness.bars_1h_fresh is True


@pytest.mark.asyncio
async def test_entry_readiness_blocks_when_latest_5m_bar_is_from_previous_utc_day() -> None:
    now = datetime(2026, 3, 19, 0, 10, tzinfo=timezone.utc)
    tick_time = datetime(2026, 3, 19, 0, 9, 50, tzinfo=timezone.utc)
    aggregator = FXTickAggregator()
    aggregator.add_tick(_tick("EURUSD", 1.10, 1.1002, tick_time))

    client = EODHDFXWebSocketClient(
        api_token="token",
        symbols=["EURUSD"],
        stale_after_seconds=86400,
    )
    client._connected = True
    client._record_tick(_tick("EURUSD", 1.10, 1.1002, tick_time))

    hub = MarketDataHub(
        aggregator=aggregator,
        websocket_client=client,
        rest_provider=DummyProvider([]),
        symbols=["EURUSD"],
        now_provider=lambda: now,
        broker_quote_provider=AsyncMock(
            return_value={
                "bid": 1.1001,
                "ask": 1.1003,
                "timestampMs": int(tick_time.timestamp() * 1000),
            }
        ),
    )
    hub._warm_cache[("EURUSD", "5m")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-18T23:50:00Z"),
                "open": 1.0998,
                "high": 1.1004,
                "low": 1.0994,
                "close": 1.1001,
                "volume": 0,
            }
        ]
    )
    hub._warm_cache[("EURUSD", "1h")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-18T23:00:00Z"),
                "open": 1.0985,
                "high": 1.1008,
                "low": 1.0980,
                "close": 1.0999,
                "volume": 0,
            }
        ]
    )

    readiness = await hub.get_entry_readiness("EURUSD")

    assert readiness.entry_safe is False
    assert readiness.block_reason == "market_data.trade_date_not_ready"
    assert readiness.requires_tactical_retry is False
    assert readiness.pending_reason == ""
    assert readiness.quote_source == "broker_quote"
    assert readiness.quote_available is True
    assert readiness.bars_5m_fresh is True
    assert readiness.bars_1h_fresh is True
    assert readiness.websocket_state == "healthy"


@pytest.mark.asyncio
async def test_entry_readiness_still_blocks_when_quote_is_missing() -> None:
    hub = MarketDataHub(
        aggregator=FXTickAggregator(),
        websocket_client=EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD"]),
        rest_provider=DummyProvider([]),
        symbols=["EURUSD"],
        now_provider=lambda: datetime(2026, 3, 17, 6, 42, 30, tzinfo=timezone.utc),
    )

    readiness = await hub.get_entry_readiness("EURUSD")

    assert readiness.entry_safe is False
    assert readiness.block_reason == "market_data.quote_unavailable"
    assert readiness.requires_tactical_retry is False
    assert readiness.pending_reason == ""


@pytest.mark.asyncio
async def test_entry_readiness_blocks_when_broker_quote_unavailable_even_with_websocket_quote(
) -> None:
    tick_time = datetime(2026, 3, 17, 6, 42, 10, tzinfo=timezone.utc)
    aggregator = FXTickAggregator()
    aggregator.add_tick(_tick("EURUSD", 1.10, 1.1002, tick_time))

    client = EODHDFXWebSocketClient(
        api_token="token",
        symbols=["EURUSD"],
        stale_after_seconds=86400,
    )
    client._connected = True
    client._record_tick(_tick("EURUSD", 1.10, 1.1002, tick_time))

    hub = MarketDataHub(
        aggregator=aggregator,
        websocket_client=client,
        rest_provider=DummyProvider([]),
        symbols=["EURUSD"],
        now_provider=lambda: datetime(2026, 3, 17, 6, 42, 30, tzinfo=timezone.utc),
        broker_quote_provider=AsyncMock(return_value=None),
    )
    hub._warm_cache[("EURUSD", "5m")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-17T06:35:00Z"),
                "open": 1.0995,
                "high": 1.1005,
                "low": 1.0990,
                "close": 1.1000,
                "volume": 0,
            }
        ]
    )
    hub._warm_cache[("EURUSD", "1h")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-17T05:00:00Z"),
                "open": 1.0995,
                "high": 1.1005,
                "low": 1.0990,
                "close": 1.1000,
                "volume": 0,
            }
        ]
    )

    readiness = await hub.get_entry_readiness("EURUSD")

    assert readiness.entry_safe is False
    assert readiness.block_reason == "market_data.broker_quote_unavailable"
    assert readiness.quote_available is True
    assert readiness.quote_source == "websocket_cache"


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
async def test_market_data_hub_rest_fallback_warning_includes_open_and_close_times() -> None:
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
        now_provider=lambda: current_now,
    )
    hub._warm_cache[("EURUSD", "1h")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-12T06:00:00Z"),
                "open": 1.10,
                "high": 1.11,
                "low": 1.09,
                "close": 1.105,
                "volume": 0,
            }
        ]
    )

    with patch("src.data.market_data_hub.logger.warning") as mock_warning:
        await hub.get_bars("EURUSD", "1h", 10)

    assert mock_warning.call_count == 1
    warning_args = mock_warning.call_args.args
    assert "latest_rest_bar_open_time" in warning_args[0]
    assert "latest_rest_bar_close_time" in warning_args[0]
    assert "latest_rest_bar_age_by_close_sec" in warning_args[0]
    assert "2026-03-12T10:00:00+00:00" in warning_args
    assert "2026-03-12T11:00:00+00:00" in warning_args
    assert 3600.0 in warning_args


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
async def test_market_data_hub_finalizes_elapsed_rollup_bars_before_websocket_first_lookup(
) -> None:
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


@pytest.mark.asyncio
async def test_feed_status_exposes_primary_routing_and_degraded_summary() -> None:
    now = datetime(2026, 3, 17, 8, 30, tzinfo=timezone.utc)
    tick_time = datetime(2026, 3, 17, 8, 29, 50, tzinfo=timezone.utc)
    aggregator = FXTickAggregator()
    aggregator.add_tick(_tick("EURUSD", 1.10, 1.1002, tick_time))
    client = EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD"], stale_after_seconds=45)
    client._last_error = "keepalive ping timeout"
    broker_quote_provider = AsyncMock(
        return_value={
            "bid": 1.1001,
            "ask": 1.1003,
            "timestampMs": int((now.timestamp() - 5) * 1000),
        }
    )
    hub = MarketDataHub(
        aggregator=aggregator,
        websocket_client=client,
        rest_provider=DummyProvider([]),
        symbols=["EURUSD"],
        now_provider=lambda: now,
        broker_quote_provider=broker_quote_provider,
    )
    hub._warm_cache[("EURUSD", "5m")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-17T08:25:00Z"),
                "open": 1.0995,
                "high": 1.1005,
                "low": 1.0990,
                "close": 1.1000,
                "volume": 0,
            }
        ]
    )
    hub._warm_cache[("EURUSD", "1h")] = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-17T07:00:00Z"),
                "open": 1.0990,
                "high": 1.1010,
                "low": 1.0980,
                "close": 1.1000,
                "volume": 0,
            }
        ]
    )

    await hub.get_entry_readiness("EURUSD")
    status = hub.feed_status()

    assert status["routing"]["quote_primary"] == "broker_quote"
    assert status["routing"]["bars_primary"] == "api_cache"
    assert status["degraded_summary"]["broker_quote"]["available"] is True
    assert status["degraded_summary"]["api_bars"]["5m_available"] is True
    assert status["degraded_summary"]["api_bars"]["1h_available"] is True
    assert status["degraded_summary"]["websocket_auxiliary"]["state"] == "degraded"
    assert status["degraded_summary"]["stale_age_thresholds"]["quote_ttl_seconds"] == 30
    assert status["degraded_summary"]["stale_age_thresholds"]["bar_cache_max_age_seconds"] == 3600
