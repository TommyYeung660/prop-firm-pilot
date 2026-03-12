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
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal

import httpx
import pandas as pd
from loguru import logger

from src.data.fx_tick_aggregator import FXTickAggregator
from src.data.fx_websocket_client import EODHDFXWebSocketClient
from src.monitor.operational_metrics import OperationalMetrics


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


@dataclass
class _RestRefreshState:
    """Tracks the latest observed REST tail for refresh suppression."""

    attempted_at: datetime
    latest_bar_at: datetime | None


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
        rest_refresh_cooldown_seconds: int = 300,
        now_provider: Callable[[], datetime] | None = None,
        operational_metrics: OperationalMetrics | None = None,
    ) -> None:
        self._aggregator = aggregator
        self._websocket_client = websocket_client
        self._rest_provider = rest_provider
        self._symbols = list(symbols)
        self._quote_ttl_seconds = quote_ttl_seconds
        self._bar_cache_max_age_seconds = bar_cache_max_age_seconds
        self._rest_refresh_cooldown_seconds = rest_refresh_cooldown_seconds
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))
        self._warm_cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._rest_refresh_state: dict[tuple[str, str], _RestRefreshState] = {}
        self._forced_stale_symbols: set[str] = set()
        self._metrics = operational_metrics

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
                    self._record_market_data_read("websocket_cache")
                    return QuoteResult(symbol=symbol, source="websocket_cache", quote=quote)
        bars = self._warm_cache.get((symbol, "1m"))
        rows_fetched = 0
        if bars is None or bars.empty or not self._bars_are_fresh(bars):
            if self._should_refresh_rest_cache(symbol=symbol, timeframe="1m"):
                bars, rows_fetched = await self._refresh_rest_cache(symbol=symbol, timeframe="1m")
            else:
                bars = self._warm_cache.get((symbol, "1m"))
            self._log_rest_fallback(
                symbol=symbol,
                timeframe="1m",
                rows_fetched=rows_fetched,
                bars=bars,
            )
        self._record_market_data_read("rest_fallback", rows_fetched)
        quote = self._build_quote_from_bars(symbol, bars)
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
                self._record_market_data_read("websocket_cache")
                return BarResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    source="websocket_cache",
                    bars=websocket_bars,
                )
        warm = self._warm_cache.get((symbol, timeframe))
        if warm is not None and not warm.empty and self._bars_are_fresh(warm):
            self._record_market_data_read("warmup_cache")
            return BarResult(
                symbol=symbol,
                timeframe=timeframe,
                source="warmup_cache",
                bars=warm.tail(limit).reset_index(drop=True),
            )
        bars, rows_fetched = await self._refresh_rest_cache(symbol=symbol, timeframe=timeframe)
        self._record_market_data_read("rest_fallback", rows_fetched)
        self._log_rest_fallback(
            symbol=symbol,
            timeframe=timeframe,
            rows_fetched=rows_fetched,
            bars=bars,
        )
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
        latest_ts = self._latest_bar_time(bars)
        if latest_ts is None:
            return False
        age = self._now_provider() - latest_ts
        return age <= timedelta(seconds=self._bar_cache_max_age_seconds)

    def _latest_bar_time(self, bars: pd.DataFrame | None) -> datetime | None:
        """Return the latest bar timestamp as aware UTC datetime."""
        if bars is None or bars.empty:
            return None
        latest_ts = pd.Timestamp(bars.iloc[-1]["datetime"]).to_pydatetime()
        if latest_ts.tzinfo is None:
            return latest_ts.replace(tzinfo=timezone.utc)
        return latest_ts.astimezone(timezone.utc)

    def _build_quote_from_bars(
        self,
        symbol: str,
        bars: pd.DataFrame | None,
    ) -> dict[str, Any] | None:
        """Build a synthetic quote from the latest available 1m bar."""
        if bars is None or bars.empty:
            return None
        last = bars.iloc[-1]
        close = float(last["close"])
        return {
            "symbol": symbol,
            "bid": close,
            "ask": close,
            "mid": close,
            "timestamp_ms": int(pd.Timestamp(last["datetime"]).timestamp() * 1000),
        }

    def _normalize_bars(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Sort and normalize provider bars to the expected schema."""
        if bars.empty:
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        return bars.sort_values("datetime").reset_index(drop=True)

    def _should_refresh_rest_cache(
        self,
        symbol: str,
        timeframe: Literal["1m", "5m", "1h"],
    ) -> bool:
        """Avoid repeated stale 1m refreshes when the REST tail has not advanced."""
        if timeframe != "1m":
            return True
        key = (symbol, timeframe)
        state = self._rest_refresh_state.get(key)
        if state is None:
            return True
        now = self._now_provider()
        if now - state.attempted_at >= timedelta(seconds=self._rest_refresh_cooldown_seconds):
            return True
        cached_latest = self._latest_bar_time(self._warm_cache.get(key))
        if cached_latest is None:
            return True
        if state.latest_bar_at is None:
            return False
        return cached_latest > state.latest_bar_at

    def _resolve_rest_window(
        self,
        symbol: str,
        timeframe: Literal["1m", "5m", "1h"],
    ) -> tuple[date, date]:
        """Resolve bounded REST backfill window using the latest cached tail when available."""
        end_date = self._now_provider().date()
        start_date = end_date - timedelta(days=self._LOOKBACK_DAYS[timeframe])
        warm = self._warm_cache.get((symbol, timeframe))
        if warm is None or warm.empty:
            return start_date, end_date
        latest_ts = pd.Timestamp(warm.iloc[-1]["datetime"]).to_pydatetime()
        if latest_ts.tzinfo is None:
            latest_ts = latest_ts.replace(tzinfo=timezone.utc)
        return max(latest_ts.date(), start_date), end_date

    async def _refresh_rest_cache(
        self,
        symbol: str,
        timeframe: Literal["1m", "5m", "1h"],
    ) -> tuple[pd.DataFrame, int]:
        """Refresh the REST-backed cache incrementally from the latest cached tail."""
        bars = await self._fetch_rest_bars(symbol=symbol, timeframe=timeframe)
        rows_fetched = len(bars)
        if bars.empty:
            cached = self._warm_cache.get((symbol, timeframe))
            if cached is None:
                cached = self._normalize_bars(pd.DataFrame())
            normalized_cached = self._normalize_bars(cached)
            self._warm_cache[(symbol, timeframe)] = normalized_cached
            self._rest_refresh_state[(symbol, timeframe)] = _RestRefreshState(
                attempted_at=self._now_provider(),
                latest_bar_at=self._latest_bar_time(normalized_cached),
            )
            return normalized_cached, rows_fetched
        cached = self._warm_cache.get((symbol, timeframe))
        if cached is not None and not cached.empty:
            bars = pd.concat([cached, bars], ignore_index=True)
            bars = bars.drop_duplicates(subset=["datetime"], keep="last")
        normalized = self._normalize_bars(bars)
        self._warm_cache[(symbol, timeframe)] = normalized
        self._rest_refresh_state[(symbol, timeframe)] = _RestRefreshState(
            attempted_at=self._now_provider(),
            latest_bar_at=self._latest_bar_time(normalized),
        )
        return normalized, rows_fetched

    def _record_market_data_read(self, source: str, row_count: int = 0) -> None:
        """Record market-data source usage in shared operational metrics."""
        if self._metrics is not None:
            self._metrics.record_market_data_read(source, row_count=row_count)

    def _log_rest_fallback(
        self,
        symbol: str,
        timeframe: Literal["1m", "5m", "1h"],
        rows_fetched: int,
        bars: pd.DataFrame | None,
    ) -> None:
        """Log degraded market-data fallback with current websocket health context."""
        status = self._websocket_client.get_status()
        latest_bar_at = self._latest_bar_time(bars)
        latest_bar_age_sec = None
        if latest_bar_at is not None:
            latest_bar_age_sec = round((self._now_provider() - latest_bar_at).total_seconds(), 1)
        logger.warning(
            "MarketDataHub: REST fallback for {} {} (rows_fetched={}, ws_state={}, last_error={}, "
            "latest_rest_bar_time={}, latest_rest_bar_age_sec={})",
            symbol,
            timeframe,
            rows_fetched,
            status.get("state"),
            status.get("last_error") or "none",
            latest_bar_at.isoformat() if latest_bar_at is not None else "none",
            latest_bar_age_sec if latest_bar_age_sec is not None else "none",
        )

    async def _fetch_rest_bars(
        self,
        symbol: str,
        timeframe: Literal["1m", "5m", "1h"],
    ) -> pd.DataFrame:
        """Fetch bars from the configured REST provider."""
        start_date, end_date = self._resolve_rest_window(symbol, timeframe)
        interval = self._INTERVAL_MAP[timeframe]
        async with httpx.AsyncClient() as client:
            bars = await self._rest_provider.fetch_bars(
                symbol,
                start_date,
                end_date,
                client,
                interval=interval,
            )
        return self._normalize_bars(bars)
