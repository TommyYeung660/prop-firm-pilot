"""
EODHD FX WebSocket client — real-time quote ingestion with reconnect support.

Maintains the latest tick per symbol, dispatches ticks to callbacks, and
tracks stale-symbol state for downstream market-data fallbacks.

Usage:
    client = EODHDFXWebSocketClient(api_token="...", symbols=["EURUSD"])
    client.register_tick_callback(handle_tick)
    await client.run()
"""

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import websockets
from loguru import logger
from pydantic import BaseModel, Field


class WebSocketTick(BaseModel):
    """Single FX tick from the WebSocket feed."""

    symbol: str = Field(description="FX symbol, e.g. EURUSD")
    bid: float = Field(description="Best bid price")
    ask: float = Field(description="Best ask price")
    timestamp_ms: int = Field(description="Provider timestamp in epoch milliseconds")

    @property
    def mid(self) -> float:
        """Return mid price for bar aggregation."""
        return (self.bid + self.ask) / 2.0

    @property
    def timestamp(self) -> datetime:
        """Return the timestamp as timezone-aware UTC datetime."""
        return datetime.fromtimestamp(self.timestamp_ms / 1000.0, tz=timezone.utc)


class EODHDFXWebSocketClient:
    """EODHD FOREX WebSocket client with reconnect and stale-symbol tracking.

    Usage:
        client = EODHDFXWebSocketClient(api_token="...", symbols=["EURUSD"])
        client.register_tick_callback(handle_tick)
        await client.run()
    """

    URL = "wss://ws.eodhistoricaldata.com/ws/forex"

    def __init__(
        self,
        api_token: str,
        symbols: list[str],
        reconnect_base_seconds: int = 2,
        reconnect_max_seconds: int = 300,
        stale_after_seconds: int = 45,
    ) -> None:
        self._api_token = api_token
        self._symbols = list(symbols)
        self._reconnect_base_seconds = reconnect_base_seconds
        self._reconnect_max_seconds = reconnect_max_seconds
        self._stale_after_seconds = stale_after_seconds
        self._callbacks: list[Callable[[WebSocketTick], Awaitable[None] | None]] = []
        self._last_ticks: dict[str, WebSocketTick] = {}
        self._last_message_at: datetime | None = None
        self._connected = False
        self._running = False
        self._last_error: str | None = None
        self._websocket: Any = None

    def _build_url(self) -> str:
        """Build the authenticated WebSocket URL."""
        return f"{self.URL}?api_token={self._api_token}"

    def _build_subscribe_message(self) -> dict[str, str]:
        """Build the provider subscription payload."""
        return {
            "action": "subscribe",
            "symbols": ",".join(self._symbols),
        }

    def register_tick_callback(
        self,
        callback: Callable[[WebSocketTick], Awaitable[None] | None],
    ) -> None:
        """Register a callback invoked for every parsed tick."""
        self._callbacks.append(callback)

    def get_last_tick(self, symbol: str) -> WebSocketTick | None:
        """Return the most recent tick for a symbol."""
        return self._last_ticks.get(symbol)

    def _parse_tick_message(self, payload: dict[str, Any]) -> WebSocketTick | None:
        """Parse a provider payload into a validated tick object."""
        symbol = payload.get("s")
        bid = payload.get("b")
        ask = payload.get("a")
        timestamp_ms = payload.get("t")
        if symbol is None or bid is None or ask is None or timestamp_ms is None:
            return None
        try:
            return WebSocketTick(
                symbol=str(symbol),
                bid=float(bid),
                ask=float(ask),
                timestamp_ms=int(timestamp_ms),
            )
        except (TypeError, ValueError):
            logger.debug("EODHDFXWebSocketClient: invalid tick payload {}", payload)
            return None

    def _record_tick(self, tick: WebSocketTick) -> None:
        """Store the latest tick and update heartbeat state."""
        self._last_ticks[tick.symbol] = tick
        self._last_message_at = tick.timestamp

    async def _dispatch_tick(self, tick: WebSocketTick) -> None:
        """Dispatch a tick to all registered callbacks."""
        for callback in self._callbacks:
            try:
                result = callback(tick)
                if inspect.isawaitable(result):
                    await result
            except Exception as e:
                logger.warning(
                    "EODHDFXWebSocketClient: tick callback failed for {} ({})",
                    tick.symbol,
                    e,
                )

    def _compute_backoff_seconds(self, attempt: int) -> int:
        """Compute bounded exponential reconnect backoff."""
        return min(self._reconnect_base_seconds * (2**attempt), self._reconnect_max_seconds)

    def stale_symbols(self, now: datetime | None = None) -> set[str]:
        """Return symbols whose latest tick is stale or missing."""
        if now is None:
            now = datetime.now(timezone.utc)
        stale_cutoff = timedelta(seconds=self._stale_after_seconds)
        stale: set[str] = set()
        for symbol in self._symbols:
            tick = self._last_ticks.get(symbol)
            if tick is None:
                stale.add(symbol)
                continue
            if now - tick.timestamp > stale_cutoff:
                stale.add(symbol)
        return stale

    def get_status(self) -> dict[str, Any]:
        """Expose client runtime state for monitoring and fallback decisions."""
        stale_symbols = sorted(self.stale_symbols())
        if not self._connected:
            state = "disconnected"
        elif stale_symbols:
            state = "degraded"
        else:
            state = "healthy"
        return {
            "state": state,
            "connected": self._connected,
            "running": self._running,
            "last_error": self._last_error,
            "last_message_at": self._last_message_at,
            "subscribed_symbols": list(self._symbols),
            "stale_symbols": stale_symbols,
        }

    async def run(self) -> None:
        """Connect, subscribe, and stream ticks until stopped."""
        self._running = True
        attempt = 0
        while self._running:
            try:
                logger.info(
                    "EODHDFXWebSocketClient: connecting to EODHD for {} symbols",
                    len(self._symbols),
                )
                async with websockets.connect(
                    self._build_url(),
                    ping_interval=20,
                    ping_timeout=20,
                ) as websocket:
                    self._websocket = websocket
                    self._connected = True
                    self._last_error = None
                    attempt = 0
                    await websocket.send(json.dumps(self._build_subscribe_message()))
                    async for raw_message in websocket:
                        if not self._running:
                            break
                        self._last_message_at = datetime.now(timezone.utc)
                        try:
                            payload = json.loads(raw_message)
                        except json.JSONDecodeError:
                            logger.debug(
                                "EODHDFXWebSocketClient: skipping non-JSON message {}",
                                raw_message,
                            )
                            continue
                        tick = self._parse_tick_message(payload)
                        if tick is None:
                            continue
                        self._record_tick(tick)
                        await self._dispatch_tick(tick)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._connected = False
                self._last_error = str(e)
                backoff = self._compute_backoff_seconds(attempt)
                attempt += 1
                logger.warning(
                    "EODHDFXWebSocketClient: connection failed ({}), reconnecting in {}s",
                    e,
                    backoff,
                )
                if self._running:
                    await asyncio.sleep(backoff)
            finally:
                self._websocket = None
                self._connected = False

    async def stop(self) -> None:
        """Stop streaming and close the active connection if present."""
        self._running = False
        websocket = self._websocket
        if websocket is not None:
            try:
                await websocket.close()
            except Exception as e:
                logger.debug("EODHDFXWebSocketClient: websocket close failed ({})", e)
