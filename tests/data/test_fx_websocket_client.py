"""Tests for the EODHD FX WebSocket client."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.data.fx_websocket_client import EODHDFXWebSocketClient, WebSocketTick


def test_client_builds_subscribe_payload() -> None:
    client = EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD", "GBPUSD"])

    assert client._build_subscribe_message() == {
        "action": "subscribe",
        "symbols": "EURUSD,GBPUSD",
    }


def test_client_parses_valid_tick_message() -> None:
    client = EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD"])

    tick = client._parse_tick_message(
        {
            "s": "EURUSD",
            "b": 1.0825,
            "a": 1.0827,
            "t": 1_741_690_400_000,
        }
    )

    assert tick == WebSocketTick(
        symbol="EURUSD",
        bid=1.0825,
        ask=1.0827,
        timestamp_ms=1_741_690_400_000,
    )


def test_client_ignores_invalid_tick_message() -> None:
    client = EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD"])

    assert client._parse_tick_message({"s": "EURUSD", "b": 1.0825}) is None


@pytest.mark.asyncio
async def test_dispatch_tick_calls_registered_callbacks() -> None:
    client = EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD"])
    calls: list[WebSocketTick] = []

    async def handle_tick(tick: WebSocketTick) -> None:
        calls.append(tick)

    client.register_tick_callback(handle_tick)
    tick = WebSocketTick(symbol="EURUSD", bid=1.08, ask=1.09, timestamp_ms=123)

    await client._dispatch_tick(tick)

    assert calls == [tick]


def test_stale_symbols_uses_last_tick_timestamp() -> None:
    client = EODHDFXWebSocketClient(
        api_token="token",
        symbols=["EURUSD", "GBPUSD"],
        stale_after_seconds=30,
    )
    now = datetime(2026, 3, 11, 8, 0, tzinfo=timezone.utc)
    client._record_tick(
        WebSocketTick(
            symbol="EURUSD",
            bid=1.08,
            ask=1.09,
            timestamp_ms=int((now - timedelta(seconds=10)).timestamp() * 1000),
        )
    )
    client._record_tick(
        WebSocketTick(
            symbol="GBPUSD",
            bid=1.28,
            ask=1.29,
            timestamp_ms=int((now - timedelta(seconds=45)).timestamp() * 1000),
        )
    )

    assert client.stale_symbols(now=now) == {"GBPUSD"}


def test_compute_backoff_caps_at_max() -> None:
    client = EODHDFXWebSocketClient(
        api_token="token",
        symbols=["EURUSD"],
        reconnect_base_seconds=2,
        reconnect_max_seconds=300,
    )

    assert client._compute_backoff_seconds(0) == 2
    assert client._compute_backoff_seconds(1) == 4
    assert client._compute_backoff_seconds(10) == 300
