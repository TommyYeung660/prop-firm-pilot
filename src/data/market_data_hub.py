"""
Unified market-data hub for WebSocket-first FX ingestion.

Coordinates three market-data sources:
1. WebSocket-derived latest quotes and aggregated bars
2. REST warmup cache loaded at startup
3. REST fallback for forced-stale or cache-miss cases

Usage:
    hub = MarketDataHub(aggregator=agg, websocket_client=client, rest_provider=provider)
    await hub.warmup()
    bars = await hub.get_bars("EURUSD", "5m", 50)
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import httpx
import pandas as pd

from src.data.fx_tick_aggregator import FXTickAggregator
from src.data.fx_websocket_client import EODHDFXWebSocketClient


@dataclass
class QuoteResult:
    """Quote lookup result with explicit source metadata."""

    symbol: str
    source: Literal["websocket_cache", "rest_fallback"]
    quote: dict[str, Any] | None


@dataclass
class BarResult:
    """Bar lookup result with explicit source metadata."""

    symbol: str
    timeframe: Literal["1m", "5m", "1h"]
    source: Literal["websocket_cache", "warmup_cache", "rest_fallback"]
    bars: pd.DataFrame


class MarketDataHub:
    """Resolve quotes and intraday bars from cache first, REST as fallback."""

    _INTERVAL_MAP = {
        "1m": "1min",
        "5m": "5min",
        "1h": "1h",
    }
    _LOOKBACK_DAYS = {
        "1m": 2,
        "5m": 3,
        "1h": 7,
    }

    def __init__(
        self,
        aggregator: FXTickAggregator,
        websocket_client: EODHDFXWebSocketClient,
        rest_provider: Any,
        symbols: list[str],
        quote_ttl_seconds: int = 30,
        bar_cache_max_age_seconds: int = 3600,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._aggregator = aggregator
        self._websocket_client = websocket_client
        self._rest_provider = rest_provider
        self._symbols = list(symbols)
        self._quote_ttl_seconds = quote_ttl_seconds
        self._bar_cache_max_age_seconds = bar_cache_max_age_seconds
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))
        self._warm_cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._forced_stale_symbols: set[str] = set()

    async def warmup(self) -> None:
        """Backfill recent intraday bars into the warm cache for all symbols."""
        for symbol in self._symbols:
            for timeframe in ("1m", "5m", "1h"):
                self._warm_cache[(symbol, timeframe)] = await self._fetch_rest_bars(
                    symbol=symbol,
                    timeframe=timeframe,
                )

    def mark_symbol_stale(self, symbol: str) -> None:
        """Force a symbol to use REST fallback until explicitly cleared."""
        self._forced_stale_symbols.add(symbol)

    def clear_symbol_stale(self, symbol: str) -> None:
        """Clear manual forced-stale state."""
        self._forced_stale_symbols.discard(symbol)

    async def get_quote(self, symbol: str) -> QuoteResult:
        """Resolve latest quote from WebSocket cache first, REST fallback second."""
        if symbol not in self._forced_stale_symbols:
            quote = self._aggregator.latest_quote(symbol)
            tick = self._websocket_client.get_last_tick(symbol)
            if quote is not None and tick is not None:
                age = self._now_provider() - tick.timestamp
                if age <= timedelta(seconds=self._quote_ttl_seconds):
                    return QuoteResult(symbol=symbol, source="websocket_cache", quote=quote)
        bars = await self._fetch_rest_bars(symbol=symbol, timeframe="1m")
        quote = None
        if not bars.empty:
            last = bars.iloc[-1]
            close = float(last["close"])
            quote = {
                "symbol": symbol,
                "bid": close,
                "ask": close,
                "mid": close,
                "timestamp_ms": int(pd.Timestamp(last["datetime"]).timestamp() * 1000),
            }
        return QuoteResult(symbol=symbol, source="rest_fallback", quote=quote)

    async def get_bars(
        self,
        symbol: str,
        timeframe: Literal["1m", "5m", "1h"],
        limit: int,
    ) -> BarResult:
        """Resolve closed bars from websocket cache, warm cache, or REST fallback."""
        if symbol not in self._forced_stale_symbols:
            websocket_bars = self._bars_from_aggregator(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
            )
            if not websocket_bars.empty and self._bars_are_fresh(websocket_bars):
                return BarResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    source="websocket_cache",
                    bars=websocket_bars,
                )
        warm = self._warm_cache.get((symbol, timeframe))
        if warm is not None and not warm.empty:
            return BarResult(
                symbol=symbol,
                timeframe=timeframe,
                source="warmup_cache",
                bars=warm.tail(limit).reset_index(drop=True),
            )
        bars = await self._fetch_rest_bars(symbol=symbol, timeframe=timeframe)
        return BarResult(
            symbol=symbol,
            timeframe=timeframe,
            source="rest_fallback",
            bars=bars.tail(limit).reset_index(drop=True),
        )

    def feed_status(self) -> dict[str, Any]:
        """Expose current feed status and cache fallback state."""
        return {
            "websocket": self._websocket_client.get_status(),
            "forced_stale_symbols": sorted(self._forced_stale_symbols),
            "warm_cache_keys": sorted(f"{symbol}:{tf}" for symbol, tf in self._warm_cache.keys()),
        }

    def _bars_from_aggregator(
        self,
        symbol: str,
        timeframe: Literal["1m", "5m", "1h"],
        limit: int,
    ) -> pd.DataFrame:
        """Build a DataFrame from closed websocket-derived bars."""
        bars = self._aggregator.get_closed_bars(symbol, timeframe, limit)
        if not bars:
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        rows = [
            {
                "datetime": pd.Timestamp(bar.start_time),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": 0,
            }
            for bar in bars
        ]
        return pd.DataFrame(rows)

    def _bars_are_fresh(self, bars: pd.DataFrame) -> bool:
        """Check bar freshness independently from quote freshness."""
        latest_ts = pd.Timestamp(bars.iloc[-1]["datetime"]).to_pydatetime()
        if latest_ts.tzinfo is None:
            latest_ts = latest_ts.replace(tzinfo=timezone.utc)
        age = self._now_provider() - latest_ts
        return age <= timedelta(seconds=self._bar_cache_max_age_seconds)

    async def _fetch_rest_bars(
        self,
        symbol: str,
        timeframe: Literal["1m", "5m", "1h"],
    ) -> pd.DataFrame:
        """Fetch bars from the configured REST provider."""
        end_date = self._now_provider().date()
        start_date = end_date - timedelta(days=self._LOOKBACK_DAYS[timeframe])
        interval = self._INTERVAL_MAP[timeframe]
        async with httpx.AsyncClient() as client:
            bars = await self._rest_provider.fetch_bars(
                symbol,
                start_date,
                end_date,
                client,
                interval=interval,
            )
        if bars.empty:
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        bars = bars.sort_values("datetime").reset_index(drop=True)
        return bars
