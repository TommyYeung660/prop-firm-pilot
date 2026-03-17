"""
Tick aggregation for WebSocket-first FX market data.

Builds latest-quote state plus closed 1m/5m/1h bars from incoming ticks.
Only closed bars are exposed for downstream tactical and volatility logic.

Usage:
    agg = FXTickAggregator()
    agg.add_tick(tick)
    agg.close_elapsed_bars()
    bars = agg.get_closed_bars("EURUSD", "5m", limit=50)
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.data.fx_websocket_client import WebSocketTick


class AggregatedBar(BaseModel):
    """Closed aggregated OHLC bar."""

    symbol: str = Field(description="FX symbol")
    timeframe: Literal["1m", "5m", "1h"] = Field(description="Aggregated timeframe")
    start_time: datetime = Field(description="Bar open timestamp in UTC")
    end_time: datetime = Field(description="Bar close timestamp in UTC")
    open: float = Field(description="Open price")
    high: float = Field(description="High price")
    low: float = Field(description="Low price")
    close: float = Field(description="Close price")


class _OpenBar:
    """Mutable in-progress bar bucket."""

    def __init__(
        self,
        symbol: str,
        timeframe: Literal["1m", "5m", "1h"],
        start_time: datetime,
        end_time: datetime,
        open_price: float,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_time = start_time
        self.end_time = end_time
        self.open = open_price
        self.high = open_price
        self.low = open_price
        self.close = open_price

    def update(self, high: float, low: float, close: float) -> None:
        """Merge a new price range into the open bucket."""
        self.high = max(self.high, high)
        self.low = min(self.low, low)
        self.close = close

    def to_bar(self) -> AggregatedBar:
        """Materialize a closed bar snapshot."""
        return AggregatedBar(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_time=self.start_time,
            end_time=self.end_time,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
        )


class FXTickAggregator:
    """Aggregate WebSocket ticks into closed bars and latest quotes."""

    _DURATIONS: dict[str, timedelta] = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "1h": timedelta(hours=1),
    }

    def __init__(self) -> None:
        self._latest_ticks: dict[str, WebSocketTick] = {}
        self._open_1m_bars: dict[str, _OpenBar] = {}
        self._open_rollups: dict[tuple[str, str], _OpenBar] = {}
        self._closed_bars: dict[tuple[str, str], list[AggregatedBar]] = {}

    def add_tick(self, tick: WebSocketTick) -> None:
        """Ingest a tick and update latest quote + current 1m bucket."""
        self._latest_ticks[tick.symbol] = tick
        start_time = self._bucket_start(tick.timestamp, "1m")
        end_time = start_time + self._DURATIONS["1m"]
        current = self._open_1m_bars.get(tick.symbol)
        if current is not None and current.start_time < start_time:
            self._finalize_1m_bar(tick.symbol)
            current = None
        price = tick.mid
        if current is None:
            current = _OpenBar(
                symbol=tick.symbol,
                timeframe="1m",
                start_time=start_time,
                end_time=end_time,
                open_price=price,
            )
            self._open_1m_bars[tick.symbol] = current
        else:
            current.update(high=price, low=price, close=price)

    def latest_quote(self, symbol: str) -> dict[str, Any] | None:
        """Return the latest bid/ask/mid snapshot for a symbol."""
        tick = self._latest_ticks.get(symbol)
        if tick is None:
            return None
        return {
            "symbol": tick.symbol,
            "bid": tick.bid,
            "ask": tick.ask,
            "mid": tick.mid,
            "timestamp_ms": tick.timestamp_ms,
        }

    def close_elapsed_bars(self, now: datetime | None = None) -> dict[str, list[AggregatedBar]]:
        """Close open buckets that have fully elapsed by `now`."""
        if now is None:
            now = datetime.now(timezone.utc)
        closed: dict[str, list[AggregatedBar]] = {"1m": [], "5m": [], "1h": []}
        for symbol, bar in list(self._open_1m_bars.items()):
            if bar.end_time <= now:
                closed["1m"].append(self._finalize_1m_bar(symbol))
        for timeframe in ("5m", "1h"):
            for key, bar in list(self._open_rollups.items()):
                if key[1] != timeframe:
                    continue
                if bar.end_time <= now:
                    closed[timeframe].append(self._finalize_rollup_bar(key))
        return closed

    def get_closed_bars(
        self,
        symbol: str,
        timeframe: Literal["1m", "5m", "1h"],
        limit: int,
    ) -> list[AggregatedBar]:
        """Return the most recent closed bars for a symbol and timeframe."""
        bars = self._closed_bars.get((symbol, timeframe), [])
        if limit <= 0:
            return []
        return bars[-limit:]

    def get_closed_bar_counts(self, symbols: list[str] | None = None) -> dict[str, dict[str, int]]:
        """Return closed websocket-derived bar counts by symbol and timeframe."""
        target_symbols = sorted(symbols or self._latest_ticks.keys())
        counts: dict[str, dict[str, int]] = {}
        for symbol in target_symbols:
            counts[symbol] = {
                timeframe: len(self._closed_bars.get((symbol, timeframe), []))
                for timeframe in ("1m", "5m", "1h")
            }
        return counts

    def _finalize_1m_bar(self, symbol: str) -> AggregatedBar:
        """Close the current 1m bar and roll it into higher timeframes."""
        current = self._open_1m_bars.pop(symbol)
        bar = current.to_bar()
        self._closed_bars.setdefault((symbol, "1m"), []).append(bar)
        self._roll_up_bar(bar, "5m")
        self._roll_up_bar(bar, "1h")
        return bar

    def _roll_up_bar(self, bar: AggregatedBar, timeframe: Literal["5m", "1h"]) -> None:
        """Roll a closed 1m bar into the next aggregation timeframe."""
        key = (bar.symbol, timeframe)
        start_time = self._bucket_start(bar.start_time, timeframe)
        end_time = start_time + self._DURATIONS[timeframe]
        current = self._open_rollups.get(key)
        if current is not None and current.start_time < start_time:
            self._finalize_rollup_bar(key)
            current = None
        if current is None:
            current = _OpenBar(
                symbol=bar.symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                open_price=bar.open,
            )
            self._open_rollups[key] = current
        current.update(high=bar.high, low=bar.low, close=bar.close)

    def _finalize_rollup_bar(self, key: tuple[str, str]) -> AggregatedBar:
        """Close a 5m or 1h aggregation bucket."""
        current = self._open_rollups.pop(key)
        bar = current.to_bar()
        self._closed_bars.setdefault(key, []).append(bar)
        return bar

    @classmethod
    def _bucket_start(cls, ts: datetime, timeframe: Literal["1m", "5m", "1h"]) -> datetime:
        """Floor a timestamp to the relevant bucket start."""
        ts = ts.astimezone(timezone.utc)
        if timeframe == "1m":
            return ts.replace(second=0, microsecond=0)
        if timeframe == "5m":
            minute = ts.minute - (ts.minute % 5)
            return ts.replace(minute=minute, second=0, microsecond=0)
        return ts.replace(minute=0, second=0, microsecond=0)
