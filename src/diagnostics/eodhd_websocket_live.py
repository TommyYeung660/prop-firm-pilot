"""
EODHD websocket live probe helpers for production diagnostics.

Collects per-symbol websocket tick summaries and same-day REST 1m lag
summaries so operators can distinguish websocket feed issues from REST
provider lag.

Usage:
    summary = summarize_tick_events(["EURUSD"], events, now)
    rest = summarize_rest_bars("EURUSD", bars, now)
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import websockets
from dotenv import load_dotenv

from src.data.fx_data_fetcher import EodhdProvider
from src.data.fx_websocket_client import EODHDFXWebSocketClient


def summarize_tick_events(
    symbols: list[str],
    events: list[tuple[str, datetime]],
    now: datetime,
) -> dict[str, dict[str, Any]]:
    """Summarize per-symbol websocket tick cadence and latest age."""
    grouped: dict[str, list[datetime]] = {symbol: [] for symbol in symbols}
    for symbol, observed_at in events:
        if symbol not in grouped:
            grouped[symbol] = []
        grouped[symbol].append(observed_at)

    summary: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        timestamps = sorted(grouped.get(symbol, []))
        if not timestamps:
            summary[symbol] = {
                "count": 0,
                "max_gap_sec": 0.0,
                "latest_age_sec": None,
                "first_tick_time": None,
                "last_tick_time": None,
            }
            continue
        max_gap = 0.0
        for prev, current in zip(timestamps, timestamps[1:]):
            max_gap = max(max_gap, (current - prev).total_seconds())
        latest = timestamps[-1]
        summary[symbol] = {
            "count": len(timestamps),
            "max_gap_sec": round(max_gap, 3),
            "latest_age_sec": round((now - latest).total_seconds(), 3),
            "first_tick_time": timestamps[0].isoformat(),
            "last_tick_time": latest.isoformat(),
        }
    return summary


def summarize_rest_bars(symbol: str, bars: pd.DataFrame, now: datetime) -> dict[str, Any]:
    """Summarize same-day REST 1m bar freshness for a symbol."""
    rows = len(bars)
    latest_bar_time: datetime | None = None
    latest_bar_age_sec: float | None = None
    if rows > 0:
        latest_bar_time = _coerce_timestamp(bars.iloc[-1]["datetime"])
        latest_bar_age_sec = round((now - latest_bar_time).total_seconds(), 3)
    return {
        "symbol": symbol,
        "rows": rows,
        "latest_bar_time": latest_bar_time.isoformat() if latest_bar_time is not None else None,
        "latest_bar_age_sec": latest_bar_age_sec,
    }


async def probe_websocket(
    api_token: str,
    symbols: list[str],
    duration_seconds: int = 30,
    raw_sample_limit: int = 8,
) -> dict[str, Any]:
    """Collect raw websocket samples plus per-symbol tick summaries."""
    client = EODHDFXWebSocketClient(api_token=api_token, symbols=symbols)
    samples: list[Any] = []
    events: list[tuple[str, datetime]] = []

    async with websockets.connect(
        client._build_url(),
        ping_interval=20,
        ping_timeout=20,
    ) as websocket:
        await websocket.send(json.dumps(client._build_subscribe_message()))
        deadline = asyncio.get_running_loop().time() + duration_seconds
        while asyncio.get_running_loop().time() < deadline:
            timeout = max(0.1, deadline - asyncio.get_running_loop().time())
            try:
                raw_message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                break
            try:
                payload = json.loads(raw_message)
            except json.JSONDecodeError:
                payload = {"raw": raw_message[:200], "json": False}
            if len(samples) < raw_sample_limit:
                samples.append(payload)
            if not isinstance(payload, dict):
                continue
            tick = client._parse_tick_message(payload)
            if tick is None:
                continue
            events.append((tick.symbol, tick.timestamp))

    now = datetime.now(timezone.utc)
    return {
        "symbols": list(symbols),
        "duration_seconds": duration_seconds,
        "raw_samples": samples,
        "tick_summary": summarize_tick_events(symbols, events, now),
    }


async def probe_rest_bars(
    api_token: str,
    symbols: list[str],
) -> dict[str, dict[str, Any]]:
    """Collect same-day EODHD REST 1m lag summaries for each symbol."""
    provider = EodhdProvider(api_key=api_token)
    now = datetime.now(timezone.utc)
    summaries: dict[str, dict[str, Any]] = {}
    async with httpx.AsyncClient() as client:
        for symbol in symbols:
            bars = await provider.fetch_bars(
                symbol,
                now.date(),
                now.date(),
                client,
                interval="1min",
            )
            summaries[symbol] = summarize_rest_bars(symbol, bars, now)
    return summaries


def load_dotenv_api_key(dotenv_path: Path = Path(".env")) -> str:
    """Load .env and return the configured EODHD API key."""
    load_dotenv(dotenv_path)
    return os.getenv("EODHD_API_KEY", "").strip()


def _coerce_timestamp(value: Any) -> datetime:
    """Convert pandas / naive timestamps to aware UTC datetimes."""
    ts = pd.Timestamp(value).to_pydatetime()
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)
