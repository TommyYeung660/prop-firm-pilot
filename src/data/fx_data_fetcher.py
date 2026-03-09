"""
FX data fetcher — async multi-provider OHLCV data acquisition.

Supports TraderMade (primary, Tier-1 bank data) and iTick (backup, with volume).
Uses httpx.AsyncClient with retry logic and exponential backoff.

Usage:
    provider = create_provider("tradermade", api_key="YOUR_KEY")
    async with httpx.AsyncClient() as http:
        df = await provider.fetch_daily_bars("EURUSD", start, end, http)
"""

import abc
import asyncio
from datetime import date, datetime, timedelta, timezone

import httpx
import pandas as pd
from loguru import logger

# ── EODHD Constants ──────────────────────────────────────────────────────────

_EODHD_API_BASE = "https://eodhd.com"
_EODHD_FX_CURRENCIES = {
    "AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD", "USD",
    "XAU", "XAG", "HKD", "SGD",
}
# ── Abstract Base ───────────────────────────────────────────────────────────


class FxDataProvider(abc.ABC):
    """Abstract FX data provider interface."""

    @abc.abstractmethod
    async def fetch_daily_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        client: httpx.AsyncClient,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV bars for a symbol.

        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume.
            datetime is pd.Timestamp (date precision).
        """
        ...

    @abc.abstractmethod
    async def fetch_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        client: httpx.AsyncClient,
        interval: str = "daily",
    ) -> pd.DataFrame:
        """Fetch OHLCV bars at the given interval.

        Args:
            symbol: FX pair e.g. "EURUSD".
            start_date: Start date (inclusive).
            end_date: End date (inclusive).
            client: httpx.AsyncClient instance.
            interval: "daily", "4h", "1h", "30min", "15min", "5min", "1min".

        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume.
        """
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str: ...


# ── TraderMade Provider ─────────────────────────────────────────────────────


class TraderMadeProvider(FxDataProvider):
    """TraderMade REST API — Tier-1 bank-sourced FX data.

    Endpoint: GET https://marketdata.tradermade.com/api/v1/timeseries
    Free tier: 1000 req/month, max 1 year per request.
    Note: Does NOT return volume (set to 0).
    """

    BASE_URL = "https://marketdata.tradermade.com/api/v1/timeseries"
    MAX_DAYS_PER_REQUEST = 365  # Free tier limitation
    INTERVAL_MAP = {
        "daily": "daily",
        "4h": "4H",
        "1h": "1H",
        "30min": "30min",
        "15min": "15min",
        "5min": "5min",
        "1min": "1min",
    }

    def __init__(self, api_key: str, max_retries: int = 3) -> None:
        self._api_key = api_key
        self._max_retries = max_retries

    @property
    def name(self) -> str:
        return "tradermade"

    async def fetch_daily_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        client: httpx.AsyncClient,
    ) -> pd.DataFrame:
        """Fetch daily bars — delegates to fetch_bars()."""
        return await self.fetch_bars(symbol, start_date, end_date, client, interval="daily")

    async def fetch_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        client: httpx.AsyncClient,
        interval: str = "daily",
    ) -> pd.DataFrame:
        """Fetch bars at the given interval, paginating if range > 1 year."""
        api_interval = self.INTERVAL_MAP.get(interval)
        if api_interval is None:
            raise ValueError(
                f"Unsupported interval '{interval}'. Available: {list(self.INTERVAL_MAP.keys())}"
            )

        all_frames: list[pd.DataFrame] = []
        current_start = start_date

        while current_start <= end_date:
            chunk_end = min(current_start + timedelta(days=self.MAX_DAYS_PER_REQUEST), end_date)

            df = await self._fetch_chunk(symbol, current_start, chunk_end, client, api_interval)
            if not df.empty:
                all_frames.append(df)

            current_start = chunk_end + timedelta(days=1)

        if not all_frames:
            logger.warning(
                "TraderMade: no data returned for {} ({} to {}, interval={})",
                symbol,
                start_date,
                end_date,
                interval,
            )
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

        result = pd.concat(all_frames, ignore_index=True)
        result = (
            result.drop_duplicates(subset=["datetime"])
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        logger.info(
            "TraderMade: fetched {} rows for {} ({} to {}, interval={})",
            len(result),
            symbol,
            start_date,
            end_date,
            interval,
        )
        return result

    async def _fetch_chunk(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        client: httpx.AsyncClient,
        api_interval: str = "daily",
    ) -> pd.DataFrame:
        """Fetch a single chunk (max 1 year)."""
        params = {
            "currency": symbol,
            "api_key": self._api_key,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "format": "records",
            "interval": api_interval,
        }

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await client.get(self.BASE_URL, params=params, timeout=30.0)

                if response.status_code == 429:
                    wait = 2**attempt
                    logger.warning(
                        "TraderMade: rate limited, waiting {}s (attempt {})", wait, attempt
                    )
                    await asyncio.sleep(wait)
                    continue

                if response.status_code != 200:
                    logger.error(
                        "TraderMade: HTTP {} for {}: {}",
                        response.status_code,
                        symbol,
                        response.text[:300],
                    )
                    await asyncio.sleep(2**attempt)
                    continue

                data = response.json()
                quotes = data.get("quotes", [])

                if not quotes:
                    return pd.DataFrame(
                        columns=["datetime", "open", "high", "low", "close", "volume"]
                    )

                rows = []
                for q in quotes:
                    rows.append(
                        {
                            "datetime": pd.Timestamp(q["date"]),
                            "open": float(q["open"]),
                            "high": float(q["high"]),
                            "low": float(q["low"]),
                            "close": float(q["close"]),
                            "volume": 0,  # TraderMade does not provide volume
                        }
                    )

                return pd.DataFrame(rows)

            except httpx.HTTPError as e:
                wait = 2**attempt
                logger.warning(
                    "TraderMade: network error '{}', retry in {}s (attempt {})", e, wait, attempt
                )
                await asyncio.sleep(wait)

        logger.error(
            "TraderMade: failed after {} retries for {} ({} to {})",
            self._max_retries,
            symbol,
            start_date,
            end_date,
        )
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])


# ── iTick Provider ──────────────────────────────────────────────────────────


class ITickProvider(FxDataProvider):
    """iTick REST API — multi-market data with tick volume.

    Endpoint: GET https://api.itick.org/forex/kline
    Free tier: 5 req/min, ~7200/day. kType=8 for daily bars.
    """

    BASE_URL = "https://api.itick.org/forex/kline"
    MAX_BARS_PER_REQUEST = 1000
    RATE_LIMIT_DELAY = 12.0  # 5 req/min = 1 req per 12s
    KTYPE_MAP = {
        "daily": "8",
        "4h": "6",
        "1h": "5",
        "30min": "4",
        "15min": "3",
        "5min": "2",
        "1min": "1",
    }

    def __init__(self, api_key: str, max_retries: int = 3) -> None:
        self._api_key = api_key
        self._max_retries = max_retries

    @property
    def name(self) -> str:
        return "itick"

    async def fetch_daily_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        client: httpx.AsyncClient,
    ) -> pd.DataFrame:
        """Fetch daily bars — delegates to fetch_bars()."""
        return await self.fetch_bars(symbol, start_date, end_date, client, interval="daily")

    async def fetch_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        client: httpx.AsyncClient,
        interval: str = "daily",
    ) -> pd.DataFrame:
        """Fetch bars at the given interval using reverse pagination."""
        ktype = self.KTYPE_MAP.get(interval)
        if ktype is None:
            raise ValueError(
                f"Unsupported interval '{interval}'. Available: {list(self.KTYPE_MAP.keys())}"
            )

        all_frames: list[pd.DataFrame] = []
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)

        while end_ts >= start_ts:
            df = await self._fetch_chunk(symbol, end_ts, client, ktype)
            if df.empty:
                break

            all_frames.append(df)

            # Move end_ts to just before the earliest bar we got
            earliest = int(df["datetime"].min().timestamp() * 1000)
            if earliest >= end_ts:
                break  # No progress, avoid infinite loop
            end_ts = earliest - 1

            # Rate limit: 5 req/min
            await asyncio.sleep(self.RATE_LIMIT_DELAY)

        if not all_frames:
            logger.warning(
                "iTick: no data returned for {} ({} to {}, interval={})",
                symbol,
                start_date,
                end_date,
                interval,
            )
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

        result = pd.concat(all_frames, ignore_index=True)

        # Filter to requested date range
        start_ts_filter = pd.Timestamp(start_date)
        end_ts_filter = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        result = result[
            (result["datetime"] >= start_ts_filter) & (result["datetime"] <= end_ts_filter)
        ]
        result = (
            result.drop_duplicates(subset=["datetime"])
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        logger.info(
            "iTick: fetched {} rows for {} ({} to {}, interval={})",
            len(result),
            symbol,
            start_date,
            end_date,
            interval,
        )
        return result

    async def _fetch_chunk(
        self,
        symbol: str,
        end_ts: int,
        client: httpx.AsyncClient,
        ktype: str = "8",
    ) -> pd.DataFrame:
        """Fetch a single chunk of up to 1000 bars."""
        params = {
            "region": "GB",
            "code": symbol,
            "kType": ktype,
            "et": str(end_ts),
            "limit": str(self.MAX_BARS_PER_REQUEST),
            "token": self._api_key,
        }

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await client.get(self.BASE_URL, params=params, timeout=30.0)

                if response.status_code == 429:
                    wait = self.RATE_LIMIT_DELAY * attempt
                    logger.warning(
                        "iTick: rate limited, waiting {:.0f}s (attempt {})", wait, attempt
                    )
                    await asyncio.sleep(wait)
                    continue

                if response.status_code != 200:
                    logger.error(
                        "iTick: HTTP {} for {}: {}",
                        response.status_code,
                        symbol,
                        response.text[:300],
                    )
                    await asyncio.sleep(2**attempt)
                    continue

                data = response.json()

                if data.get("code") != 200:
                    logger.error("iTick: API error for {}: {}", symbol, data.get("msg", "unknown"))
                    return pd.DataFrame(
                        columns=["datetime", "open", "high", "low", "close", "volume"]
                    )

                bars = data.get("data", [])
                if not bars:
                    return pd.DataFrame(
                        columns=["datetime", "open", "high", "low", "close", "volume"]
                    )

                rows = []
                for bar in bars:
                    rows.append(
                        {
                            "datetime": pd.Timestamp(bar["t"], unit="ms"),
                            "open": float(bar["o"]),
                            "high": float(bar["h"]),
                            "low": float(bar["l"]),
                            "close": float(bar["c"]),
                            "volume": int(bar.get("v", 0)),
                        }
                    )

                return pd.DataFrame(rows)

            except httpx.HTTPError as e:
                wait = 2**attempt
                logger.warning(
                    "iTick: network error '{}', retry in {}s (attempt {})", e, wait, attempt
                )
                await asyncio.sleep(wait)

        logger.error("iTick: failed after {} retries for {}", self._max_retries, symbol)
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])


# ── EODHD Provider ──────────────────────────────────────────────────────────


def _to_eodhd_symbol(symbol: str) -> str:
    """Convert internal FX symbol to EODHD format (e.g. EURUSD -> EURUSD.FOREX)."""
    clean = symbol.replace("/", "").upper()
    if "." in clean:
        return clean
    if len(clean) == 6:
        from_ccy = clean[:3]
        to_ccy = clean[3:]
        if from_ccy in _EODHD_FX_CURRENCIES or to_ccy in _EODHD_FX_CURRENCIES:
            return f"{clean}.FOREX"
    return f"{clean}.US"


class EodhdProvider(FxDataProvider):
    """EODHD Intraday API — FX intraday bars (5min, 1h).

    Endpoint: GET https://eodhd.com/api/intraday/{SYMBOL}.FOREX
    Params: interval ('5m'/'1h'), from/to (unix timestamps), fmt ('json').
    Subscription: 'EOD Historical Data Extended + Intraday'.

    Usage:
        provider = EodhdProvider(api_key='YOUR_KEY')
        async with httpx.AsyncClient() as http:
            df = await provider.fetch_bars('EURUSD', start, end, http, interval='1h')
    """

    INTERVAL_MAP = {
        "5min": "5m",
        "1h": "1h",
        "15min": "15m",
        "30min": "30m",
        "1min": "1m",
    }
    MAX_DAYS_PER_CHUNK = 120  # EODHD intraday: ~120 days per request for 1h

    def __init__(self, api_key: str, max_retries: int = 3) -> None:
        self._api_key = api_key
        self._max_retries = max_retries

    @property
    def name(self) -> str:
        return "eodhd"

    async def fetch_daily_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        client: httpx.AsyncClient,
    ) -> pd.DataFrame:
        """Not supported for intraday provider — raises ValueError."""
        raise ValueError(
            "EodhdProvider is intraday-only. Use TraderMade/iTick for daily bars."
        )

    async def fetch_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        client: httpx.AsyncClient,
        interval: str = "1h",
    ) -> pd.DataFrame:
        """Fetch intraday bars from EODHD, paginating for long date ranges."""
        api_interval = self.INTERVAL_MAP.get(interval)
        if api_interval is None:
            raise ValueError(
                f"Unsupported interval '{interval}'. Available: {list(self.INTERVAL_MAP.keys())}"
            )

        all_frames: list[pd.DataFrame] = []
        current_start = start_date

        while current_start <= end_date:
            chunk_end = min(
                current_start + timedelta(days=self.MAX_DAYS_PER_CHUNK), end_date
            )
            df = await self._fetch_chunk(
                symbol, current_start, chunk_end, client, api_interval
            )
            if not df.empty:
                all_frames.append(df)
            current_start = chunk_end + timedelta(days=1)

        if not all_frames:
            logger.warning(
                "EODHD: no data returned for {} ({} to {}, interval={})",
                symbol, start_date, end_date, interval,
            )
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

        result = pd.concat(all_frames, ignore_index=True)
        result = (
            result.drop_duplicates(subset=["datetime"])
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        logger.info(
            "EODHD: fetched {} rows for {} ({} to {}, interval={})",
            len(result), symbol, start_date, end_date, interval,
        )
        return result

    async def _fetch_chunk(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        client: httpx.AsyncClient,
        api_interval: str = "1h",
    ) -> pd.DataFrame:
        """Fetch a single chunk of intraday bars from EODHD."""
        eodhd_sym = _to_eodhd_symbol(symbol)
        start_ts = int(
            datetime.combine(start_date, datetime.min.time())
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
        end_ts = int(
            (datetime.combine(end_date, datetime.min.time())
             .replace(tzinfo=timezone.utc) + timedelta(days=1)).timestamp()
        ) - 1

        url = f"{_EODHD_API_BASE}/api/intraday/{eodhd_sym}"
        params = {
            "interval": api_interval,
            "from": start_ts,
            "to": end_ts,
            "fmt": "json",
            "api_token": self._api_key,
        }

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await client.get(url, params=params, timeout=30.0)

                if response.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(
                        "EODHD: rate limited, waiting {}s (attempt {})", wait, attempt
                    )
                    await asyncio.sleep(wait)
                    continue

                if response.status_code != 200:
                    logger.error(
                        "EODHD: HTTP {} for {}: {}",
                        response.status_code, symbol, response.text[:300],
                    )
                    await asyncio.sleep(2 ** attempt)
                    continue

                data = response.json()

                if not data or not isinstance(data, list):
                    return pd.DataFrame(
                        columns=["datetime", "open", "high", "low", "close", "volume"]
                    )

                rows = []
                for bar in data:
                    rows.append({
                        "datetime": pd.Timestamp(bar["datetime"]),
                        "open": float(bar["open"]),
                        "high": float(bar["high"]),
                        "low": float(bar["low"]),
                        "close": float(bar["close"]),
                        "volume": int(bar.get("volume") or 0),
                    })

                return pd.DataFrame(rows)

            except httpx.HTTPError as e:
                wait = 2 ** attempt
                logger.warning(
                    "EODHD: network error '{}', retry in {}s (attempt {})",
                    e, wait, attempt,
                )
                await asyncio.sleep(wait)

        logger.error(
            "EODHD: failed after {} retries for {} ({} to {})",
            self._max_retries, symbol, start_date, end_date,
        )
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])


# ── Factory ─────────────────────────────────────────────────────────────────


def create_provider(provider: str, api_key: str) -> FxDataProvider:
    """Create an FX data provider by name.

    Args:
        provider: "tradermade", "itick", or "eodhd".
        api_key: API key for the provider.

    Returns:
        FxDataProvider instance.

    Raises:
        ValueError: If provider name is unknown.
    """
    providers = {
        "tradermade": TraderMadeProvider,
        "itick": ITickProvider,
        "eodhd": EodhdProvider,
    }

    cls = providers.get(provider)
    if cls is None:
        raise ValueError(
            f"Unknown FX data provider: '{provider}'. Available: {list(providers.keys())}"
        )

    return cls(api_key=api_key)


# ── Convenience ─────────────────────────────────────────────────────────────


async def fetch_all_symbols(
    provider: FxDataProvider,
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> dict[str, pd.DataFrame]:
    """Fetch daily bars for multiple symbols sequentially.

    Sequential to respect rate limits. Use with care.

    Args:
        provider: FxDataProvider instance.
        symbols: List of FX pairs.
        start_date: Start date.
        end_date: End date.

    Returns:
        Dict of symbol -> DataFrame.
    """
    results: dict[str, pd.DataFrame] = {}

    async with httpx.AsyncClient() as client:
        for symbol in symbols:
            logger.info("Fetching {} via {}...", symbol, provider.name)
            df = await provider.fetch_daily_bars(symbol, start_date, end_date, client)
            results[symbol] = df

    total_rows = sum(len(df) for df in results.values())
    logger.info(
        "fetch_all_symbols: {} symbols, {} total rows via {}",
        len(symbols),
        total_rows,
        provider.name,
    )
    return results
