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

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal

import httpx
import pandas as pd
from loguru import logger

from src.data.fx_tick_aggregator import FXTickAggregator
from src.data.fx_websocket_client import EODHDFXWebSocketClient, WebSocketTick
from src.monitor.operational_metrics import OperationalMetrics


@dataclass
class QuoteResult:
    """Quote lookup result with explicit source metadata."""

    symbol: str
    source: Literal["broker_quote", "websocket_cache", "rest_realtime", "rest_fallback"]
    quote: dict[str, Any] | None


@dataclass
class BarResult:
    """Bar lookup result with explicit source metadata."""

    symbol: str
    timeframe: Literal["1m", "5m", "1h"]
    source: Literal["websocket_cache", "warmup_cache", "rest_fallback"]
    bars: pd.DataFrame


@dataclass
class EntryReadinessResult:
    """Entry-safety verdict derived from current quote and bar availability."""

    symbol: str
    entry_safe: bool
    block_reason: str
    websocket_state: str
    ws_last_error: str | None
    quote_source: str
    quote_available: bool
    bars_5m_source: str
    bars_5m_fresh: bool
    bars_1h_source: str
    bars_1h_fresh: bool
    broker_quote_available: bool = False
    api_bars_5m_fresh: bool = False
    api_bars_1h_fresh: bool = False
    requires_tactical_retry: bool = False
    pending_reason: str = ""


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
    _TIMEFRAME_SECONDS = {
        "1m": 60,
        "5m": 300,
        "1h": 3600,
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
        broker_quote_provider: Callable[[str], Awaitable[Any | None]] | None = None,
        realtime_quote_provider: Callable[[str], Awaitable[Any | None]] | None = None,
    ) -> None:
        self._aggregator = aggregator
        self._websocket_client = websocket_client
        self._rest_provider = rest_provider
        self._broker_quote_provider = broker_quote_provider
        self._realtime_quote_provider = realtime_quote_provider
        self._symbols = list(symbols)
        self._quote_ttl_seconds = quote_ttl_seconds
        self._bar_cache_max_age_seconds = bar_cache_max_age_seconds
        self._rest_refresh_cooldown_seconds = rest_refresh_cooldown_seconds
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))
        self._warm_cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._rest_refresh_state: dict[tuple[str, str], _RestRefreshState] = {}
        self._rest_refresh_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._forced_stale_symbols: set[str] = set()
        self._metrics = operational_metrics
        self._initialized_at = self._now_provider()
        self._broker_quote_available_by_symbol: dict[str, bool] = {}
        self._broker_quote_error_by_symbol: dict[str, str] = {}
        self._realtime_quote_timestamp_ms_by_symbol: dict[str, int] = {}

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
        """Resolve latest quote from broker primary, then websocket/rest fallback."""
        broker_quote = await self._fetch_broker_quote(symbol)
        if broker_quote is not None:
            self._broker_quote_available_by_symbol[symbol] = True
            self._broker_quote_error_by_symbol.pop(symbol, None)
            return QuoteResult(symbol=symbol, source="broker_quote", quote=broker_quote)
        if self._broker_quote_provider is not None:
            self._broker_quote_available_by_symbol[symbol] = False

        if symbol not in self._forced_stale_symbols:
            quote = self._aggregator.latest_quote(symbol)
            tick = self._websocket_client.get_last_tick(symbol)
            if quote is not None and tick is not None:
                age = self._now_provider() - tick.timestamp
                if age <= timedelta(seconds=self._quote_ttl_seconds):
                    self._record_market_data_read("websocket_cache")
                    return QuoteResult(symbol=symbol, source="websocket_cache", quote=quote)

        realtime_quote = await self._fetch_realtime_quote(symbol)
        if realtime_quote is not None:
            self._feed_realtime_quote(symbol=symbol, quote=realtime_quote)
            return QuoteResult(symbol=symbol, source="rest_realtime", quote=realtime_quote)
        bars = self._warm_cache.get((symbol, "1m"))
        rows_fetched = 0
        if bars is None or bars.empty or not self._bars_are_fresh(bars, "1m"):
            bars, rows_fetched, refreshed = await self._refresh_rest_cache_serialized(
                symbol=symbol,
                timeframe="1m",
            )
            if refreshed:
                self._log_rest_fallback(
                    symbol=symbol,
                    timeframe="1m",
                    rows_fetched=rows_fetched,
                    bars=bars,
                )
        self._record_market_data_read("rest_fallback", rows_fetched)
        if bars is None or bars.empty or not self._bars_are_fresh(bars, "1m"):
            return QuoteResult(symbol=symbol, source="rest_fallback", quote=None)
        quote = self._build_quote_from_bars(symbol, bars)
        return QuoteResult(symbol=symbol, source="rest_fallback", quote=quote)

    async def get_bars(
        self,
        symbol: str,
        timeframe: Literal["1m", "5m", "1h"],
        limit: int,
    ) -> BarResult:
        """Resolve closed bars from API cache first, websocket only as auxiliary fallback."""
        warm = self._warm_cache.get((symbol, timeframe))
        if warm is not None and not warm.empty and self._bars_are_fresh(warm, timeframe):
            self._record_market_data_read("warmup_cache")
            return BarResult(
                symbol=symbol,
                timeframe=timeframe,
                source="warmup_cache",
                bars=warm.tail(limit).reset_index(drop=True),
            )

        if symbol not in self._forced_stale_symbols:
            websocket_bars = self._fresh_websocket_bars(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
            )
            if websocket_bars is not None:
                self._record_market_data_read("websocket_cache")
                return BarResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    source="websocket_cache",
                    bars=websocket_bars,
                )

        rows_fetched = 0
        bars, rows_fetched, refreshed = await self._refresh_rest_cache_serialized(
            symbol=symbol,
            timeframe=timeframe,
        )
        if not bars.empty and self._bars_are_fresh(bars, timeframe):
            if refreshed:
                self._log_rest_fallback(
                    symbol=symbol,
                    timeframe=timeframe,
                    rows_fetched=rows_fetched,
                    bars=bars,
                )
            self._record_market_data_read("rest_fallback", rows_fetched)
            return BarResult(
                symbol=symbol,
                timeframe=timeframe,
                source="rest_fallback",
                bars=bars.tail(limit).reset_index(drop=True),
            )

        if refreshed:
            self._log_rest_fallback(
                symbol=symbol,
                timeframe=timeframe,
                rows_fetched=rows_fetched,
                bars=bars,
            )
        self._record_market_data_read("rest_fallback", rows_fetched)
        return BarResult(
            symbol=symbol,
            timeframe=timeframe,
            source="rest_fallback",
            bars=bars.tail(limit).reset_index(drop=True),
        )

    def feed_status(self) -> dict[str, Any]:
        """Expose current feed status and cache fallback state."""
        now = self._now_provider()
        websocket_status = self._websocket_client.get_status(now=now)
        api_5m_by_symbol = {symbol: self._api_bars_fresh(symbol, "5m") for symbol in self._symbols}
        api_1h_by_symbol = {symbol: self._api_bars_fresh(symbol, "1h") for symbol in self._symbols}
        broker_available_by_symbol = {
            symbol: self._broker_quote_available_by_symbol.get(symbol) for symbol in self._symbols
        }
        known_broker_states = [
            available
            for available in broker_available_by_symbol.values()
            if isinstance(available, bool)
        ]
        broker_available: bool | None = None
        if known_broker_states:
            broker_available = all(known_broker_states)

        return {
            "initialized_at": self._initialized_at.isoformat(),
            "uptime_seconds": max(0, int((now - self._initialized_at).total_seconds())),
            "routing": {
                "quote_primary": (
                    "broker_quote"
                    if self._broker_quote_provider is not None
                    else "websocket_cache"
                ),
                "bars_primary": "api_cache",
                "websocket_role": "auxiliary",
            },
            "degraded_summary": {
                "broker_quote": {
                    "enabled": self._broker_quote_provider is not None,
                    "available": broker_available,
                    "available_by_symbol": broker_available_by_symbol,
                    "last_errors": dict(self._broker_quote_error_by_symbol),
                },
                "api_bars": {
                    "5m_available": all(api_5m_by_symbol.values()) if api_5m_by_symbol else False,
                    "1h_available": all(api_1h_by_symbol.values()) if api_1h_by_symbol else False,
                    "5m_available_by_symbol": api_5m_by_symbol,
                    "1h_available_by_symbol": api_1h_by_symbol,
                },
                "websocket_auxiliary": {
                    "state": websocket_status.get("state", ""),
                    "last_error": websocket_status.get("last_error"),
                    "stale_symbols": websocket_status.get("stale_symbols", []),
                },
                "stale_age_thresholds": {
                    "quote_ttl_seconds": self._quote_ttl_seconds,
                    "bar_cache_max_age_seconds": self._bar_cache_max_age_seconds,
                    "rest_refresh_cooldown_seconds": self._rest_refresh_cooldown_seconds,
                },
            },
            "websocket": websocket_status,
            "websocket_closed_bar_counts": self._aggregator.get_closed_bar_counts(self._symbols),
            "forced_stale_symbols": sorted(self._forced_stale_symbols),
            "warm_cache_keys": sorted(f"{symbol}:{tf}" for symbol, tf in self._warm_cache.keys()),
        }

    async def get_entry_readiness(self, symbol: str) -> EntryReadinessResult:
        """Return whether current market data is sufficient for a new entry."""
        now = self._now_provider()
        websocket_status = self._websocket_client.get_status(now=now)
        websocket_state = str(websocket_status.get("state", ""))
        quote_result = await self.get_quote(symbol)
        bars_5m_result, bars_1h_result = await asyncio.gather(
            self.get_bars(symbol, "5m", 10),
            self.get_bars(symbol, "1h", 10),
        )
        quote_available = quote_result.quote is not None
        broker_quote_required = self._broker_quote_provider is not None
        broker_quote_available = quote_result.source == "broker_quote" and quote_available
        bars_5m_fresh = not bars_5m_result.bars.empty and self._bars_are_fresh(
            bars_5m_result.bars, "5m"
        )
        bars_1h_fresh = not bars_1h_result.bars.empty and self._bars_are_fresh(
            bars_1h_result.bars, "1h"
        )
        bars_5m_close_at = self._latest_bar_close_time(bars_5m_result.bars, "5m")
        bars_1h_close_at = self._latest_bar_close_time(bars_1h_result.bars, "1h")
        current_trade_date = now.date()

        block_reason = ""
        requires_tactical_retry = False
        pending_reason = ""
        if broker_quote_required and not broker_quote_available:
            block_reason = "market_data.broker_quote_unavailable"
        elif not quote_available:
            block_reason = "market_data.quote_unavailable"
        elif not bars_1h_fresh:
            block_reason = "market_data.bars_1h_stale"
        elif bars_5m_result.bars.empty:
            requires_tactical_retry = True
            pending_reason = "market_data.startup_5m_bar_pending"
        elif not bars_5m_fresh:
            if self._is_startup_5m_bar_pending(
                symbol=symbol,
                websocket_state=websocket_state,
                quote_available=quote_available,
                bars_1h_fresh=bars_1h_fresh,
                bars_5m_close_at=bars_5m_close_at,
                current_trade_date=current_trade_date,
            ):
                requires_tactical_retry = True
                pending_reason = "market_data.startup_5m_bar_pending"
            elif bars_5m_close_at is not None and bars_5m_close_at.date() < current_trade_date:
                block_reason = "market_data.trade_date_not_ready"
            else:
                block_reason = "market_data.bars_5m_stale"
        elif (
            bars_5m_close_at is not None and bars_5m_close_at.date() < current_trade_date
        ) or (
            bars_1h_close_at is not None and bars_1h_close_at.date() < current_trade_date
        ):
            block_reason = "market_data.trade_date_not_ready"

        return EntryReadinessResult(
            symbol=symbol,
            entry_safe=block_reason == "",
            block_reason=block_reason,
            requires_tactical_retry=requires_tactical_retry,
            pending_reason=pending_reason,
            websocket_state=websocket_state,
            ws_last_error=websocket_status.get("last_error"),
            quote_source=quote_result.source,
            quote_available=quote_available,
            broker_quote_available=broker_quote_available,
            bars_5m_source=bars_5m_result.source,
            bars_5m_fresh=bars_5m_fresh,
            api_bars_5m_fresh=self._api_bars_fresh(symbol, "5m"),
            bars_1h_source=bars_1h_result.source,
            bars_1h_fresh=bars_1h_fresh,
            api_bars_1h_fresh=self._api_bars_fresh(symbol, "1h"),
        )

    def _is_startup_5m_bar_pending(
        self,
        *,
        symbol: str,
        websocket_state: str,
        quote_available: bool,
        bars_1h_fresh: bool,
        bars_5m_close_at: datetime | None,
        current_trade_date: date,
    ) -> bool:
        """Detect cold-start windows before the first websocket 5m bar closes."""
        if symbol in self._forced_stale_symbols:
            return False
        if websocket_state != "healthy" or not quote_available or not bars_1h_fresh:
            return False
        if bars_5m_close_at is not None and bars_5m_close_at.date() < current_trade_date:
            return False

        self._aggregator.close_elapsed_bars(now=self._now_provider())
        return not self._aggregator.get_closed_bars(symbol, "5m", 1)

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

    def _fresh_websocket_bars(
        self,
        *,
        symbol: str,
        timeframe: Literal["1m", "5m", "1h"],
        limit: int,
    ) -> pd.DataFrame | None:
        """Return fresh closed websocket bars when available."""
        self._aggregator.close_elapsed_bars(now=self._now_provider())
        websocket_bars = self._bars_from_aggregator(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
        )
        if websocket_bars.empty or not self._bars_are_fresh(websocket_bars, timeframe):
            return None
        return websocket_bars

    def _bars_are_fresh(self, bars: pd.DataFrame, timeframe: Literal["1m", "5m", "1h"]) -> bool:
        """Check bar freshness independently from quote freshness."""
        latest_ts = self._latest_bar_close_time(bars, timeframe)
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

    def _latest_bar_close_time(
        self,
        bars: pd.DataFrame | None,
        timeframe: Literal["1m", "5m", "1h"],
    ) -> datetime | None:
        """Return the effective close time for the latest closed bar."""
        latest_open = self._latest_bar_time(bars)
        if latest_open is None:
            return None
        return latest_open + timedelta(seconds=self._TIMEFRAME_SECONDS[timeframe])

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

    async def _fetch_broker_quote(self, symbol: str) -> dict[str, Any] | None:
        """Fetch and normalize broker quote payload when a provider is configured."""
        if self._broker_quote_provider is None:
            return None
        try:
            raw_quote = await self._broker_quote_provider(symbol)
        except Exception as e:
            self._broker_quote_error_by_symbol[symbol] = str(e)
            logger.warning("MarketDataHub: broker quote fetch failed for {} ({})", symbol, e)
            return None

        normalized = self._normalize_quote_payload(symbol=symbol, payload=raw_quote)
        if normalized is None:
            self._broker_quote_error_by_symbol[symbol] = "empty_or_invalid_broker_quote"
            return None
        return normalized

    async def _fetch_realtime_quote(self, symbol: str) -> dict[str, Any] | None:
        """Fetch and validate an EODHD real-time REST quote when configured."""
        if self._realtime_quote_provider is None:
            return None
        try:
            raw_quote = await self._realtime_quote_provider(symbol)
        except Exception as e:
            logger.warning(
                "MarketDataHub: real-time REST quote fetch failed for {} ({})",
                symbol,
                e,
            )
            return None

        normalized = self._normalize_quote_payload(symbol=symbol, payload=raw_quote)
        if normalized is None:
            return None
        timestamp_ms = normalized.get("timestamp_ms")
        if not isinstance(timestamp_ms, int) or timestamp_ms <= 0:
            return None

        quote_time = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        age = self._now_provider() - quote_time
        if age > timedelta(seconds=self._quote_ttl_seconds):
            return None

        return normalized

    def _feed_realtime_quote(self, *, symbol: str, quote: dict[str, Any]) -> None:
        """Feed a validated real-time REST snapshot into the existing tick aggregator."""
        timestamp_ms = int(quote["timestamp_ms"])
        if self._realtime_quote_timestamp_ms_by_symbol.get(symbol) == timestamp_ms:
            return

        tick = WebSocketTick(
            symbol=symbol,
            bid=float(quote["bid"]),
            ask=float(quote["ask"]),
            timestamp_ms=timestamp_ms,
        )
        self._aggregator.add_tick(tick)
        self._realtime_quote_timestamp_ms_by_symbol[symbol] = timestamp_ms

    def _normalize_quote_payload(
        self,
        *,
        symbol: str,
        payload: Any,
    ) -> dict[str, Any] | None:
        """Normalize dict/object quote payload to the internal quote schema."""
        if payload is None:
            return None

        if isinstance(payload, dict):
            bid = payload.get("bid", 0)
            ask = payload.get("ask", 0)
            ts_ms = payload.get("timestamp_ms", 0) or payload.get("timestampMs", 0)
        else:
            bid = getattr(payload, "bid", 0)
            ask = getattr(payload, "ask", 0)
            ts_ms = getattr(payload, "timestamp_ms", 0) or getattr(payload, "timestampMs", 0)

        try:
            bid_f = float(bid)
            ask_f = float(ask)
        except (TypeError, ValueError):
            return None
        if bid_f <= 0 or ask_f <= 0:
            return None

        normalized: dict[str, Any] = {
            "symbol": symbol,
            "bid": bid_f,
            "ask": ask_f,
            "mid": (bid_f + ask_f) / 2.0,
        }
        try:
            ts_ms_int = int(ts_ms) if ts_ms else 0
        except (TypeError, ValueError):
            ts_ms_int = 0
        if ts_ms_int > 0:
            normalized["timestamp_ms"] = ts_ms_int
        return normalized

    def _normalize_bars(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Sort and normalize provider bars to the expected schema."""
        if bars.empty:
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        return bars.sort_values("datetime").reset_index(drop=True)

    def _api_bars_fresh(self, symbol: str, timeframe: Literal["5m", "1h"]) -> bool:
        """Whether the API-backed warm cache currently has fresh bars for a symbol/timeframe."""
        bars = self._warm_cache.get((symbol, timeframe))
        if bars is None or bars.empty:
            return False
        return self._bars_are_fresh(bars, timeframe)

    def _should_refresh_rest_cache(
        self,
        symbol: str,
        timeframe: Literal["1m", "5m", "1h"],
    ) -> bool:
        """Avoid repeated stale refreshes when the REST tail has not advanced."""
        key = (symbol, timeframe)
        state = self._rest_refresh_state.get(key)
        if state is None:
            return True
        now = self._now_provider()
        cooldown_seconds = max(
            self._rest_refresh_cooldown_seconds,
            self._TIMEFRAME_SECONDS[timeframe],
        )
        if now - state.attempted_at >= timedelta(seconds=cooldown_seconds):
            return True
        cached_latest = self._latest_bar_time(self._warm_cache.get(key))
        if state.latest_bar_at is None:
            return cached_latest is not None
        if cached_latest is None:
            return False
        return cached_latest > state.latest_bar_at

    async def _refresh_rest_cache_serialized(
        self,
        symbol: str,
        timeframe: Literal["1m", "5m", "1h"],
    ) -> tuple[pd.DataFrame, int, bool]:
        """Serialize same-key REST refreshes so concurrent callers share one attempt."""
        key = (symbol, timeframe)
        lock = self._rest_refresh_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._rest_refresh_locks[key] = lock
        async with lock:
            if not self._should_refresh_rest_cache(symbol=symbol, timeframe=timeframe):
                cached = self._warm_cache.get(key)
                if cached is None:
                    cached = self._normalize_bars(pd.DataFrame())
                return cached, 0, False
            bars, rows_fetched = await self._refresh_rest_cache(symbol=symbol, timeframe=timeframe)
            return bars, rows_fetched, True

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
        status = self._websocket_client.get_status(now=self._now_provider())
        latest_bar_open_at = self._latest_bar_time(bars)
        latest_bar_close_at = self._latest_bar_close_time(bars, timeframe)
        latest_bar_age_by_close_sec = None
        if latest_bar_close_at is not None:
            latest_bar_age_by_close_sec = round(
                (self._now_provider() - latest_bar_close_at).total_seconds(),
                1,
            )
        logger.warning(
            "MarketDataHub: REST fallback for {} {} (rows_fetched={}, ws_state={}, last_error={}, "
            "latest_rest_bar_open_time={}, latest_rest_bar_close_time={}, "
            "latest_rest_bar_age_by_close_sec={})",
            symbol,
            timeframe,
            rows_fetched,
            status.get("state"),
            status.get("last_error") or "none",
            latest_bar_open_at.isoformat() if latest_bar_open_at is not None else "none",
            latest_bar_close_at.isoformat() if latest_bar_close_at is not None else "none",
            latest_bar_age_by_close_sec if latest_bar_age_by_close_sec is not None else "none",
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
