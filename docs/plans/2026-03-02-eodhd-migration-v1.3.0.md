# EODHD Data Source Migration — v1.3.0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Alpha Vantage with EODHD as the primary financial data provider across all 3 repositories (TradingAgents, qlib_market_scanner, prop-firm-pilot), with a date-aware switchover mechanism (AV primary before March 21, EODHD primary after March 21).

**Architecture:** Add EODHD as a new vendor in TradingAgents' multi-vendor routing system (`VENDOR_METHODS`), add `EODHDFXFetcher` in qlib_market_scanner, and update prop-firm-pilot's config to route data requests through EODHD. The existing `route_to_vendor()` fallback mechanism ensures graceful degradation — if EODHD fails, requests automatically fall back to other vendors (AV, yfinance, local). A `SWITCHOVER_DATE` constant controls which vendor is primary.

**Tech Stack:** Python 3.10, httpx (async) for EODHD API calls in prop-firm-pilot/qlib_market_scanner, requests (sync) for TradingAgents (matching existing pattern), Pydantic for config, pytest + respx for testing.

---

## Repository Map

| Repo | Path | Role |
|------|------|------|
| **TradingAgents** | `C:\Users\tommy.yeung\CursorProjects\TradingAgents\` | LLM decision engine — needs EODHD vendor modules |
| **qlib_market_scanner** | `C:\Users\tommy.yeung\CursorProjects\qlib_market_scanner\` | Qlib signal scanner — needs `EODHDFXFetcher` + stock fetcher |
| **prop-firm-pilot** | `C:\Users\tommy.yeung\CursorProjects\prop-firm\prop-firm-pilot\` | Orchestrator — needs config update + switchover logic |

## EODHD API Reference (Quick Reference)

| Endpoint | URL Pattern | API Calls/req | Notes |
|----------|-------------|---------------|-------|
| EOD Historical | `/api/eod/{SYMBOL}` | 1 | Daily OHLCV |
| Intraday | `/api/intraday/{SYMBOL}?interval=1h` | 5 | 1H bars; no native 4H |
| Technical | `/api/technical/{SYMBOL}?function={func}` | 5 | SMA, EMA, RSI, MACD, BBANDS, ATR |
| News | `/api/news?s={SYMBOL}` | 5 + 5/ticker | Includes sentiment scores |
| Sentiment | `/api/sentiments?s={SYMBOLS}` | 1 | -1 to 1 scores |
| Fundamentals | `/api/fundamentals/{SYMBOL}` | 1 | Full company data |

**Symbol format:** FX = `EURUSD.FOREX`, Stocks = `AAPL.US`
**Free tier:** 20 API calls/day (demo tickers: `AAPL.US`, `EURUSD.FOREX`, `AMZN.US`)
**Paid tier (post 3/21):** 100,000 API calls/day

---

## Task 1: EODHD Common Utilities (TradingAgents)

**Files:**
- Create: `tradingagents/dataflows/eodhd_common.py`
- Test: `tests/test_eodhd_common.py`

### Step 1: Write the failing tests

```python
# tests/test_eodhd_common.py
import os
import pytest
import json
from unittest.mock import patch, MagicMock
from tradingagents.dataflows.eodhd_common import (
    get_api_key,
    to_eodhd_symbol,
    from_eodhd_symbol,
    EODHDRateLimitError,
    _make_api_request,
)


def test_get_api_key_returns_env_var():
    with patch.dict(os.environ, {"EODHD_API_KEY": "test_key_123"}):
        assert get_api_key() == "test_key_123"


def test_get_api_key_raises_when_missing():
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("EODHD_API_KEY", None)
        with pytest.raises(ValueError, match="EODHD_API_KEY"):
            get_api_key()


class TestToEodhdSymbol:
    def test_fx_pair_6_char(self):
        assert to_eodhd_symbol("EURUSD") == "EURUSD.FOREX"

    def test_fx_pair_with_slash(self):
        assert to_eodhd_symbol("EUR/USD") == "EURUSD.FOREX"

    def test_fx_pair_lowercase(self):
        assert to_eodhd_symbol("eurusd") == "EURUSD.FOREX"

    def test_us_stock(self):
        assert to_eodhd_symbol("AAPL") == "AAPL.US"

    def test_stock_with_exchange_suffix(self):
        # If already has suffix, pass through
        assert to_eodhd_symbol("AAPL.US") == "AAPL.US"

    def test_gold_xauusd(self):
        assert to_eodhd_symbol("XAUUSD") == "XAUUSD.FOREX"

    def test_jpy_pair(self):
        assert to_eodhd_symbol("USDJPY") == "USDJPY.FOREX"


class TestFromEodhdSymbol:
    def test_forex_to_plain(self):
        assert from_eodhd_symbol("EURUSD.FOREX") == "EURUSD"

    def test_stock_to_plain(self):
        assert from_eodhd_symbol("AAPL.US") == "AAPL"

    def test_no_suffix(self):
        assert from_eodhd_symbol("EURUSD") == "EURUSD"


class TestMakeApiRequest:
    @patch("tradingagents.dataflows.eodhd_common.requests.get")
    def test_successful_json_request(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"date": "2026-01-01", "open": 1.1}]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"EODHD_API_KEY": "test_key"}):
            result = _make_api_request("/api/eod/EURUSD.FOREX", {"fmt": "json"})
            assert isinstance(result, list)
            assert result[0]["date"] == "2026-01-01"

    @patch("tradingagents.dataflows.eodhd_common.requests.get")
    def test_rate_limit_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = Exception("429")
        mock_get.return_value = mock_response

        with patch.dict(os.environ, {"EODHD_API_KEY": "test_key"}):
            with pytest.raises(EODHDRateLimitError):
                _make_api_request("/api/eod/EURUSD.FOREX", {"fmt": "json"})
```

### Step 2: Run tests to verify they fail

Run: `cd C:\Users\tommy.yeung\CursorProjects\TradingAgents && uv run pytest tests/test_eodhd_common.py -v`
Expected: FAIL — module `eodhd_common` not found

### Step 3: Write minimal implementation

```python
# tradingagents/dataflows/eodhd_common.py
"""
EODHD (eodhistoricaldata.com) shared utilities.

Provides API key management, symbol translation, rate limit handling,
and a common HTTP request helper for all EODHD data modules.

Usage:
    from .eodhd_common import get_api_key, to_eodhd_symbol, _make_api_request
    data = _make_api_request("/api/eod/EURUSD.FOREX", {"fmt": "json"})
"""

import os
import json
import requests
from datetime import datetime


# ── Constants ───────────────────────────────────────────────────────
API_BASE_URL = "https://eodhd.com"

# Known FX symbols (6-char uppercase pairs)
_FX_CURRENCIES = {
    "AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD", "USD",
    "XAU", "XAG", "HKD", "SGD", "NOK", "SEK", "DKK", "ZAR",
    "TRY", "MXN", "PLN", "CZK", "HUF", "CNY", "INR", "THB",
}


# ── Exceptions ──────────────────────────────────────────────────────
class EODHDRateLimitError(Exception):
    """Raised when EODHD API rate limit (20/day free, 100K/day paid) is exceeded."""
    pass


class EODHDAPIError(Exception):
    """Raised for general EODHD API errors."""
    pass


# ── API Key ─────────────────────────────────────────────────────────
def get_api_key() -> str:
    """Retrieve EODHD API key from environment."""
    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        raise ValueError("EODHD_API_KEY environment variable is not set.")
    return api_key


# ── Symbol Translation ──────────────────────────────────────────────
def _is_fx_symbol(symbol: str) -> bool:
    """Check if symbol is an FX currency pair (e.g. EURUSD, GBP/USD, XAUUSD)."""
    clean = symbol.replace("/", "").upper()
    if len(clean) == 6:
        from_ccy = clean[:3]
        to_ccy = clean[3:]
        return from_ccy in _FX_CURRENCIES or to_ccy in _FX_CURRENCIES
    return False


def to_eodhd_symbol(symbol: str) -> str:
    """Convert internal symbol to EODHD format.

    Examples:
        EURUSD -> EURUSD.FOREX
        EUR/USD -> EURUSD.FOREX
        AAPL -> AAPL.US
        AAPL.US -> AAPL.US (pass-through)
    """
    # Already has exchange suffix
    if "." in symbol and not symbol.startswith("."):
        return symbol.upper()

    clean = symbol.replace("/", "").upper()

    if _is_fx_symbol(clean):
        return f"{clean}.FOREX"

    # Default: assume US stock
    return f"{clean}.US"


def from_eodhd_symbol(symbol: str) -> str:
    """Convert EODHD symbol back to internal format.

    Examples:
        EURUSD.FOREX -> EURUSD
        AAPL.US -> AAPL
    """
    if "." in symbol:
        return symbol.rsplit(".", 1)[0]
    return symbol


# ── HTTP Request Helper ─────────────────────────────────────────────
def _make_api_request(
    endpoint: str,
    params: dict | None = None,
    timeout: int = 30,
) -> dict | list | str:
    """Make an authenticated request to EODHD API.

    Args:
        endpoint: API path (e.g., "/api/eod/EURUSD.FOREX")
        params: Query parameters (api_token added automatically)
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response (dict or list)

    Raises:
        EODHDRateLimitError: When daily API call limit is exceeded
        EODHDAPIError: For other API errors
    """
    if params is None:
        params = {}

    params["api_token"] = get_api_key()

    if "fmt" not in params:
        params["fmt"] = "json"

    url = f"{API_BASE_URL}{endpoint}"

    try:
        response = requests.get(url, params=params, timeout=timeout)

        if response.status_code == 429:
            raise EODHDRateLimitError(
                "EODHD API rate limit exceeded (429). "
                f"Free tier: 20 calls/day, Paid: 100K calls/day."
            )

        response.raise_for_status()

        # Try JSON parse
        try:
            return response.json()
        except json.JSONDecodeError:
            # Some endpoints return CSV
            return response.text

    except EODHDRateLimitError:
        raise
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Failed to connect to EODHD API. Check internet connection.")
    except requests.exceptions.Timeout:
        raise TimeoutError("EODHD API request timed out.")
    except requests.exceptions.RequestException as e:
        raise EODHDAPIError(f"EODHD API request failed: {str(e)}")


def _filter_by_date_range(
    data: list[dict],
    start_date: str,
    end_date: str,
    date_key: str = "date",
) -> list[dict]:
    """Filter list of dicts by date range.

    Args:
        data: List of dicts with a date field
        start_date: Start date (yyyy-mm-dd)
        end_date: End date (yyyy-mm-dd)
        date_key: Key name for the date field

    Returns:
        Filtered list
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    filtered = []
    for item in data:
        date_str = item.get(date_key, "")
        if not date_str:
            continue
        try:
            # Handle both "YYYY-MM-DD" and "YYYY-MM-DD HH:MM:SS" formats
            item_dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
            if start_dt <= item_dt <= end_dt:
                filtered.append(item)
        except ValueError:
            continue

    return filtered
```

### Step 4: Run tests to verify they pass

Run: `cd C:\Users\tommy.yeung\CursorProjects\TradingAgents && uv run pytest tests/test_eodhd_common.py -v`
Expected: ALL PASS

### Step 5: Commit

```bash
cd C:\Users\tommy.yeung\CursorProjects\TradingAgents
git add tradingagents/dataflows/eodhd_common.py tests/test_eodhd_common.py
git commit -m "feat(eodhd): add common utilities — API key, symbol translation, HTTP helper"
```

---

## Task 2: EODHD Stock/FX Data Module (TradingAgents)

**Files:**
- Create: `tradingagents/dataflows/eodhd_stock.py`
- Test: `tests/test_eodhd_stock.py`

### Step 1: Write the failing tests

```python
# tests/test_eodhd_stock.py
import os
import pytest
from unittest.mock import patch, MagicMock
from tradingagents.dataflows.eodhd_stock import get_stock


class TestGetStock:
    """Test EODHD stock data retrieval — must return same CSV format as Alpha Vantage."""

    @patch("tradingagents.dataflows.eodhd_stock._make_api_request")
    def test_fx_pair_returns_csv(self, mock_request):
        """FX pair should call /api/eod/EURUSD.FOREX and return CSV string."""
        mock_request.return_value = [
            {"date": "2026-02-25", "open": 1.045, "high": 1.048, "low": 1.042, "close": 1.046, "adjusted_close": 1.046, "volume": 0},
            {"date": "2026-02-26", "open": 1.046, "high": 1.050, "low": 1.043, "close": 1.049, "adjusted_close": 1.049, "volume": 0},
        ]

        with patch.dict(os.environ, {"EODHD_API_KEY": "test_key"}):
            result = get_stock("EURUSD", "2026-02-25", "2026-02-26")

        # Must return CSV string (same format as alpha_vantage_stock.get_stock)
        assert isinstance(result, str)
        lines = result.strip().split("\n")
        assert len(lines) >= 2  # header + at least 1 data row
        header = lines[0]
        assert "timestamp" in header or "date" in header
        assert "open" in header
        assert "close" in header
        assert "volume" in header

    @patch("tradingagents.dataflows.eodhd_stock._make_api_request")
    def test_stock_symbol_translation(self, mock_request):
        """US stock should call with .US suffix."""
        mock_request.return_value = [
            {"date": "2026-02-25", "open": 180.0, "high": 182.0, "low": 179.0, "close": 181.5, "adjusted_close": 181.5, "volume": 50000000},
        ]

        with patch.dict(os.environ, {"EODHD_API_KEY": "test_key"}):
            result = get_stock("AAPL", "2026-02-25", "2026-02-25")

        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert "AAPL.US" in call_args[0][0]  # endpoint contains AAPL.US

    @patch("tradingagents.dataflows.eodhd_stock._make_api_request")
    def test_date_range_filtering(self, mock_request):
        """Only data within date range should be returned."""
        mock_request.return_value = [
            {"date": "2026-02-24", "open": 1.040, "high": 1.043, "low": 1.038, "close": 1.041, "adjusted_close": 1.041, "volume": 0},
            {"date": "2026-02-25", "open": 1.045, "high": 1.048, "low": 1.042, "close": 1.046, "adjusted_close": 1.046, "volume": 0},
            {"date": "2026-02-26", "open": 1.046, "high": 1.050, "low": 1.043, "close": 1.049, "adjusted_close": 1.049, "volume": 0},
        ]

        with patch.dict(os.environ, {"EODHD_API_KEY": "test_key"}):
            result = get_stock("EURUSD", "2026-02-25", "2026-02-25")

        # Should only contain 2026-02-25 data, not 02-24 or 02-26
        assert "2026-02-25" in result
        assert "2026-02-24" not in result
        assert "2026-02-26" not in result

    @patch("tradingagents.dataflows.eodhd_stock._make_api_request")
    def test_empty_response(self, mock_request):
        """Empty API response should return informative message."""
        mock_request.return_value = []

        with patch.dict(os.environ, {"EODHD_API_KEY": "test_key"}):
            result = get_stock("EURUSD", "2026-02-25", "2026-02-25")

        assert isinstance(result, str)
        # Should contain some indication of no data, not crash
```

### Step 2: Run tests to verify they fail

Run: `cd C:\Users\tommy.yeung\CursorProjects\TradingAgents && uv run pytest tests/test_eodhd_stock.py -v`
Expected: FAIL — module `eodhd_stock` not found

### Step 3: Write minimal implementation

```python
# tradingagents/dataflows/eodhd_stock.py
"""
EODHD stock and FX daily OHLCV data provider.

Drop-in replacement for alpha_vantage_stock.get_stock().
Returns CSV string in same format for LLM agent consumption.

Usage:
    from .eodhd_stock import get_stock
    csv_data = get_stock("EURUSD", "2026-02-01", "2026-02-28")
"""

from datetime import datetime
from io import StringIO
from .eodhd_common import _make_api_request, to_eodhd_symbol, _filter_by_date_range


def get_stock(symbol: str, start_date: str, end_date: str) -> str:
    """Returns daily OHLCV data as CSV string, matching Alpha Vantage output format.

    For FX pairs (e.g., EURUSD), uses EODHD EOD API with .FOREX suffix.
    For stocks (e.g., AAPL), uses EODHD EOD API with .US suffix.

    Args:
        symbol: Ticker symbol (e.g., "EURUSD", "AAPL", "IBM")
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        CSV string: timestamp,open,high,low,close,volume
    """
    eodhd_symbol = to_eodhd_symbol(symbol)

    endpoint = f"/api/eod/{eodhd_symbol}"
    params = {
        "from": start_date,
        "to": end_date,
        "fmt": "json",
    }

    try:
        data = _make_api_request(endpoint, params)
    except Exception as e:
        return f"Error fetching EODHD data for {symbol}: {str(e)}"

    if not data or not isinstance(data, list):
        return f"No data available for {symbol} from {start_date} to {end_date}"

    # Filter by date range (defensive — API should handle this, but be safe)
    filtered = _filter_by_date_range(data, start_date, end_date)

    if not filtered:
        return f"No data available for {symbol} from {start_date} to {end_date}"

    # Convert to CSV string matching Alpha Vantage format:
    # timestamp,open,high,low,close,volume
    lines = ["timestamp,open,high,low,close,volume"]
    for row in filtered:
        date = row.get("date", "")
        open_val = row.get("open", "")
        high_val = row.get("high", "")
        low_val = row.get("low", "")
        close_val = row.get("close", "")
        volume = row.get("volume", 0)
        lines.append(f"{date},{open_val},{high_val},{low_val},{close_val},{volume}")

    return "\n".join(lines)
```

### Step 4: Run tests to verify they pass

Run: `cd C:\Users\tommy.yeung\CursorProjects\TradingAgents && uv run pytest tests/test_eodhd_stock.py -v`
Expected: ALL PASS

### Step 5: Commit

```bash
cd C:\Users\tommy.yeung\CursorProjects\TradingAgents
git add tradingagents/dataflows/eodhd_stock.py tests/test_eodhd_stock.py
git commit -m "feat(eodhd): add stock/FX daily OHLCV data module"
```

---

## Task 3: EODHD Technical Indicators Module (TradingAgents)

**Files:**
- Create: `tradingagents/dataflows/eodhd_indicator.py`
- Test: `tests/test_eodhd_indicator.py`

### Step 1: Write the failing tests

```python
# tests/test_eodhd_indicator.py
import os
import pytest
from unittest.mock import patch
from tradingagents.dataflows.eodhd_indicator import get_indicator


class TestGetIndicator:
    """Test EODHD indicator — must return same string format as AV get_indicator."""

    @patch("tradingagents.dataflows.eodhd_indicator._make_api_request")
    def test_sma_returns_formatted_string(self, mock_request):
        mock_request.return_value = [
            {"date": "2026-02-25", "sma": 1.045},
            {"date": "2026-02-26", "sma": 1.046},
        ]

        with patch.dict(os.environ, {"EODHD_API_KEY": "test"}):
            result = get_indicator("EURUSD", "close_50_sma", "2026-02-26", 30)

        assert isinstance(result, str)
        assert "SMA" in result.upper() or "sma" in result.lower()
        assert "2026-02-25" in result or "2026-02-26" in result

    @patch("tradingagents.dataflows.eodhd_indicator._make_api_request")
    def test_rsi_indicator(self, mock_request):
        mock_request.return_value = [
            {"date": "2026-02-25", "rsi": 55.3},
            {"date": "2026-02-26", "rsi": 58.1},
        ]

        with patch.dict(os.environ, {"EODHD_API_KEY": "test"}):
            result = get_indicator("EURUSD", "rsi", "2026-02-26", 30)

        assert isinstance(result, str)
        assert "RSI" in result.upper()

    @patch("tradingagents.dataflows.eodhd_indicator._make_api_request")
    def test_macd_indicator(self, mock_request):
        mock_request.return_value = [
            {"date": "2026-02-25", "macd": 0.002, "macd_signal": 0.001, "macd_hist": 0.001},
            {"date": "2026-02-26", "macd": 0.003, "macd_signal": 0.002, "macd_hist": 0.001},
        ]

        with patch.dict(os.environ, {"EODHD_API_KEY": "test"}):
            result = get_indicator("EURUSD", "macd", "2026-02-26", 30)

        assert isinstance(result, str)
        assert "MACD" in result.upper()

    def test_unsupported_indicator_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            get_indicator("EURUSD", "fake_indicator", "2026-02-26", 30)

    @patch("tradingagents.dataflows.eodhd_indicator._make_api_request")
    def test_bbands_indicator(self, mock_request):
        mock_request.return_value = [
            {"date": "2026-02-25", "uband": 1.055, "mband": 1.045, "lband": 1.035},
        ]

        with patch.dict(os.environ, {"EODHD_API_KEY": "test"}):
            result = get_indicator("EURUSD", "boll_ub", "2026-02-25", 30)

        assert isinstance(result, str)
        assert "Bollinger" in result or "boll" in result.lower() or "Upper" in result
```

### Step 2: Run tests to verify they fail

Run: `cd C:\Users\tommy.yeung\CursorProjects\TradingAgents && uv run pytest tests/test_eodhd_indicator.py -v`
Expected: FAIL

### Step 3: Write minimal implementation

```python
# tradingagents/dataflows/eodhd_indicator.py
"""
EODHD technical indicator data provider.

Drop-in replacement for alpha_vantage_indicator.get_indicator().
Returns formatted string in same format for LLM agent consumption.

Usage:
    from .eodhd_indicator import get_indicator
    result = get_indicator("EURUSD", "rsi", "2026-02-26", 30)
"""

from datetime import datetime
from dateutil.relativedelta import relativedelta
from .eodhd_common import _make_api_request, to_eodhd_symbol


# ── Indicator Mapping ───────────────────────────────────────────────
# Maps internal indicator names to EODHD function names and response keys
INDICATOR_MAP = {
    "close_50_sma": {"function": "sma", "period": 50, "response_key": "sma"},
    "close_200_sma": {"function": "sma", "period": 200, "response_key": "sma"},
    "close_10_ema": {"function": "ema", "period": 10, "response_key": "ema"},
    "macd": {"function": "macd", "period": None, "response_key": "macd"},
    "macds": {"function": "macd", "period": None, "response_key": "macd_signal"},
    "macdh": {"function": "macd", "period": None, "response_key": "macd_hist"},
    "rsi": {"function": "rsi", "period": 14, "response_key": "rsi"},
    "boll": {"function": "bbands", "period": 20, "response_key": "mband"},
    "boll_ub": {"function": "bbands", "period": 20, "response_key": "uband"},
    "boll_lb": {"function": "bbands", "period": 20, "response_key": "lband"},
    "atr": {"function": "atr", "period": 14, "response_key": "atr"},
    "vwma": None,  # Not available via EODHD Technical API
}

INDICATOR_DESCRIPTIONS = {
    "close_50_sma": "50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance.",
    "close_200_sma": "200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups.",
    "close_10_ema": "10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points.",
    "macd": "MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes.",
    "macds": "MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades.",
    "macdh": "MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early.",
    "rsi": "RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence.",
    "boll": "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands.",
    "boll_ub": "Bollinger Upper Band: Typically 2 standard deviations above the middle line. Signals potential overbought conditions.",
    "boll_lb": "Bollinger Lower Band: Typically 2 standard deviations below the middle line. Indicates potential oversold conditions.",
    "atr": "ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes.",
    "vwma": "VWMA: Volume-weighted moving average. Not directly available from EODHD Technical API.",
}


def get_indicator(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int,
    interval: str = "daily",
    time_period: int = 14,
    series_type: str = "close",
) -> str:
    """Returns EODHD technical indicator values, matching AV get_indicator format.

    Args:
        symbol: Ticker symbol (e.g., "EURUSD", "AAPL")
        indicator: Indicator name (e.g., "rsi", "macd", "close_50_sma")
        curr_date: Current trading date, YYYY-mm-dd
        look_back_days: Number of days to look back
        interval: Time interval (daily, weekly, monthly) — only daily supported
        time_period: Number of data points for calculation
        series_type: Price type (close, open, high, low)

    Returns:
        Formatted string with indicator values and description
    """
    if indicator not in INDICATOR_MAP:
        raise ValueError(
            f"Indicator '{indicator}' is not supported. "
            f"Choose from: {list(INDICATOR_MAP.keys())}"
        )

    # VWMA not available via EODHD
    if INDICATOR_MAP[indicator] is None:
        return (
            f"## VWMA (Volume Weighted Moving Average) for {symbol}:\n\n"
            f"VWMA is not directly available from EODHD Technical API.\n\n"
            f"{INDICATOR_DESCRIPTIONS.get(indicator, '')}"
        )

    config = INDICATOR_MAP[indicator]
    eodhd_symbol = to_eodhd_symbol(symbol)

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_date_dt = curr_date_dt - relativedelta(days=look_back_days)
    start_date = start_date_dt.strftime("%Y-%m-%d")

    # Build API request
    endpoint = f"/api/technical/{eodhd_symbol}"
    params = {
        "function": config["function"],
        "from": start_date,
        "to": curr_date,
        "fmt": "json",
    }

    # Add period if applicable
    if config["period"] is not None:
        params["period"] = str(config["period"])

    # MACD-specific params
    if config["function"] == "macd":
        params["fast_period"] = "12"
        params["slow_period"] = "26"
        params["signal_period"] = "9"

    try:
        data = _make_api_request(endpoint, params)
    except Exception as e:
        return f"Error retrieving {indicator} data for {symbol}: {str(e)}"

    if not data or not isinstance(data, list):
        return f"No {indicator} data available for {symbol}"

    # Extract values
    response_key = config["response_key"]
    result_lines = []

    for item in data:
        date_str = item.get("date", "")
        if not date_str:
            continue

        try:
            item_dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
            if start_date_dt <= item_dt <= curr_date_dt:
                value = item.get(response_key, "N/A")
                result_lines.append((item_dt, f"{date_str[:10]}: {value}"))
        except ValueError:
            continue

    # Sort by date
    result_lines.sort(key=lambda x: x[0])
    values_str = "\n".join(line for _, line in result_lines)

    if not values_str:
        values_str = "No data available for the specified date range."

    description = INDICATOR_DESCRIPTIONS.get(indicator, "")

    return (
        f"## {indicator.upper()} values from {start_date} to {curr_date}:\n\n"
        f"{values_str}\n\n"
        f"{description}"
    )
```

### Step 4: Run tests to verify they pass

Run: `cd C:\Users\tommy.yeung\CursorProjects\TradingAgents && uv run pytest tests/test_eodhd_indicator.py -v`
Expected: ALL PASS

### Step 5: Commit

```bash
cd C:\Users\tommy.yeung\CursorProjects\TradingAgents
git add tradingagents/dataflows/eodhd_indicator.py tests/test_eodhd_indicator.py
git commit -m "feat(eodhd): add technical indicators module"
```

---

## Task 4: EODHD News Module (TradingAgents)

**Files:**
- Create: `tradingagents/dataflows/eodhd_news.py`
- Test: `tests/test_eodhd_news.py`

### Step 1: Write the failing tests

```python
# tests/test_eodhd_news.py
import os
import pytest
from unittest.mock import patch
from tradingagents.dataflows.eodhd_news import get_news, get_global_news


class TestGetNews:
    @patch("tradingagents.dataflows.eodhd_news._make_api_request")
    def test_returns_json_response(self, mock_request):
        mock_request.return_value = [
            {
                "date": "2026-02-26 10:30:00",
                "title": "EUR rises on ECB decision",
                "content": "The Euro gained...",
                "sentiment": {"polarity": 0.3, "neg": 0.1, "neu": 0.5, "pos": 0.4},
            }
        ]

        with patch.dict(os.environ, {"EODHD_API_KEY": "test"}):
            result = get_news("EURUSD", "2026-02-20", "2026-02-26")

        # Must return dict or str (same as AV get_news)
        assert result is not None

    @patch("tradingagents.dataflows.eodhd_news._make_api_request")
    def test_fx_symbol_handling(self, mock_request):
        mock_request.return_value = []

        with patch.dict(os.environ, {"EODHD_API_KEY": "test"}):
            get_news("EURUSD", "2026-02-20", "2026-02-26")

        call_args = mock_request.call_args
        # For FX, should use base currencies as search terms
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("params", {})


class TestGetGlobalNews:
    @patch("tradingagents.dataflows.eodhd_news._make_api_request")
    def test_returns_data(self, mock_request):
        mock_request.return_value = [
            {
                "date": "2026-02-26 08:00:00",
                "title": "Fed holds rates steady",
                "content": "The Federal Reserve...",
                "tags": ["monetary policy"],
            }
        ]

        with patch.dict(os.environ, {"EODHD_API_KEY": "test"}):
            result = get_global_news("2026-02-26", look_back_days=7, limit=5)

        assert result is not None
```

### Step 2: Run tests to verify they fail

Run: `cd C:\Users\tommy.yeung\CursorProjects\TradingAgents && uv run pytest tests/test_eodhd_news.py -v`
Expected: FAIL

### Step 3: Write minimal implementation

```python
# tradingagents/dataflows/eodhd_news.py
"""
EODHD news and sentiment data provider.

Drop-in replacement for alpha_vantage_news functions.
Returns JSON data matching the AV format for LLM agent consumption.

Usage:
    from .eodhd_news import get_news, get_global_news
    news = get_news("EURUSD", "2026-02-20", "2026-02-26")
"""

import json
from datetime import datetime, timedelta
from .eodhd_common import _make_api_request, to_eodhd_symbol, _is_fx_symbol


def get_news(ticker: str, start_date: str, end_date: str) -> dict | str:
    """Returns news articles with sentiment for a ticker.

    Args:
        ticker: Symbol (e.g., "EURUSD", "AAPL")
        start_date: Start date (yyyy-mm-dd)
        end_date: End date (yyyy-mm-dd)

    Returns:
        JSON string with news data
    """
    eodhd_symbol = to_eodhd_symbol(ticker)

    params = {
        "s": eodhd_symbol,
        "from": start_date,
        "to": end_date,
        "limit": "50",
        "fmt": "json",
    }

    try:
        data = _make_api_request("/api/news", params)
    except Exception as e:
        return f"Error fetching news for {ticker}: {str(e)}"

    if isinstance(data, list):
        return json.dumps({"articles": data, "source": "eodhd"}, indent=2)
    return str(data)


# Global macro news tags for broad market coverage
GLOBAL_NEWS_TAGS = "economy,monetary policy,federal reserve,central bank,inflation,gdp"


def get_global_news(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 5,
) -> dict | str:
    """Returns global macroeconomic news using EODHD tag-based search.

    Args:
        curr_date: Current date (yyyy-mm-dd)
        look_back_days: Days to look back
        limit: Max articles to return

    Returns:
        JSON string with global news data
    """
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_date = (curr_date_dt - timedelta(days=look_back_days)).strftime("%Y-%m-%d")

    params = {
        "t": GLOBAL_NEWS_TAGS,
        "from": start_date,
        "to": curr_date,
        "limit": str(limit),
        "fmt": "json",
    }

    try:
        data = _make_api_request("/api/news", params)
    except Exception as e:
        return f"Error fetching global news: {str(e)}"

    if isinstance(data, list):
        return json.dumps({"articles": data, "source": "eodhd", "type": "global"}, indent=2)
    return str(data)


def get_insider_transactions(symbol: str, curr_date: str = None) -> dict | str:
    """Returns insider transactions from EODHD.

    Args:
        symbol: Ticker symbol (e.g., "AAPL")
        curr_date: Current date (not used, for API compatibility)

    Returns:
        JSON string with insider transaction data
    """
    eodhd_symbol = to_eodhd_symbol(symbol)

    params = {
        "code": eodhd_symbol,
        "fmt": "json",
    }

    try:
        data = _make_api_request("/api/insider-transactions", params)
    except Exception as e:
        return f"Error fetching insider transactions for {symbol}: {str(e)}"

    if isinstance(data, list):
        return json.dumps({"transactions": data, "source": "eodhd"}, indent=2)
    return str(data)


def get_insider_sentiment(ticker: str, curr_date: str = None) -> str:
    """Returns insider sentiment derived from EODHD insider transactions.

    Calculates buy/sell ratio and sentiment metrics from transaction data,
    matching the format of alpha_vantage_news.get_insider_sentiment().

    Args:
        ticker: Ticker symbol
        curr_date: Current date (yyyy-mm-dd)

    Returns:
        Formatted string with insider sentiment analysis
    """
    from dateutil.relativedelta import relativedelta

    eodhd_symbol = to_eodhd_symbol(ticker)

    params = {
        "code": eodhd_symbol,
        "fmt": "json",
    }

    try:
        data = _make_api_request("/api/insider-transactions", params)
    except Exception as e:
        return f"Error fetching insider sentiment for {ticker}: {str(e)}"

    if not data or not isinstance(data, list):
        return ""

    # Calculate lookback period
    if curr_date:
        curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    else:
        curr_date_dt = datetime.now()

    lookback_date = curr_date_dt - relativedelta(days=90)
    lookback_str = lookback_date.strftime("%Y-%m-%d")

    # Filter and calculate sentiment (same logic as AV version)
    acquisitions = 0
    disposals = 0
    acquisition_shares = 0
    disposal_shares = 0

    for tx in data:
        tx_date = tx.get("date", tx.get("transaction_date", ""))
        if tx_date < lookback_str:
            continue

        shares = float(tx.get("shares", tx.get("transactionShares", 0)) or 0)
        action = tx.get("transactionType", tx.get("acquisition_or_disposal", ""))

        if "Buy" in action or "Purchase" in action or action == "A":
            acquisitions += 1
            acquisition_shares += abs(shares)
        elif "Sale" in action or "Sell" in action or action == "D":
            disposals += 1
            disposal_shares += abs(shares)

    total = acquisitions + disposals
    if total == 0:
        return (
            f"## {ticker} Insider Sentiment Data for {lookback_str} to "
            f"{curr_date_dt.strftime('%Y-%m-%d')}:\n\n"
            f"No insider transactions found in the past 90 days."
        )

    buy_ratio = acquisitions / total
    net_shares = acquisition_shares - disposal_shares
    mspr = acquisition_shares / (acquisition_shares + disposal_shares) if (acquisition_shares + disposal_shares) > 0 else 0

    if buy_ratio > 0.6:
        sentiment = "Bullish (insiders are net buyers)"
    elif buy_ratio < 0.4:
        sentiment = "Bearish (insiders are net sellers)"
    else:
        sentiment = "Neutral (balanced buying/selling)"

    return (
        f"## {ticker} Insider Sentiment Data for {lookback_str} to {curr_date_dt.strftime('%Y-%m-%d')}:\n\n"
        f"### Summary Metrics:\n"
        f"- **Total Transactions**: {total}\n"
        f"- **Acquisitions (Buys)**: {acquisitions}\n"
        f"- **Disposals (Sells)**: {disposals}\n"
        f"- **Buy Ratio**: {buy_ratio:.2%}\n"
        f"- **Net Shares Change**: {net_shares:+,.0f}\n"
        f"- **Monthly Share Purchase Ratio (MSPR)**: {mspr:.2%}\n\n"
        f"### Sentiment Interpretation: {sentiment}\n"
    )
```

### Step 4: Run tests to verify they pass

Run: `cd C:\Users\tommy.yeung\CursorProjects\TradingAgents && uv run pytest tests/test_eodhd_news.py -v`
Expected: ALL PASS

### Step 5: Commit

```bash
cd C:\Users\tommy.yeung\CursorProjects\TradingAgents
git add tradingagents/dataflows/eodhd_news.py tests/test_eodhd_news.py
git commit -m "feat(eodhd): add news and insider sentiment module"
```

---

## Task 5: EODHD Fundamentals Module (TradingAgents)

**Files:**
- Create: `tradingagents/dataflows/eodhd_fundamentals.py`
- Test: `tests/test_eodhd_fundamentals.py`

### Step 1: Write the failing tests

```python
# tests/test_eodhd_fundamentals.py
import os
import pytest
from unittest.mock import patch
from tradingagents.dataflows.eodhd_fundamentals import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
)


class TestEodhdFundamentals:
    @patch("tradingagents.dataflows.eodhd_fundamentals._make_api_request")
    def test_get_fundamentals_returns_string(self, mock_request):
        mock_request.return_value = {
            "General": {"Name": "Apple Inc"},
            "Highlights": {"MarketCapitalization": 3000000000000},
        }

        with patch.dict(os.environ, {"EODHD_API_KEY": "test"}):
            result = get_fundamentals("AAPL")

        assert isinstance(result, str)

    @patch("tradingagents.dataflows.eodhd_fundamentals._make_api_request")
    def test_get_balance_sheet_returns_string(self, mock_request):
        mock_request.return_value = {"Financials": {"Balance_Sheet": {"quarterly": {}}}}

        with patch.dict(os.environ, {"EODHD_API_KEY": "test"}):
            result = get_balance_sheet("AAPL")

        assert isinstance(result, str)

    @patch("tradingagents.dataflows.eodhd_fundamentals._make_api_request")
    def test_get_cashflow_returns_string(self, mock_request):
        mock_request.return_value = {"Financials": {"Cash_Flow": {"quarterly": {}}}}

        with patch.dict(os.environ, {"EODHD_API_KEY": "test"}):
            result = get_cashflow("AAPL")

        assert isinstance(result, str)

    @patch("tradingagents.dataflows.eodhd_fundamentals._make_api_request")
    def test_get_income_statement_returns_string(self, mock_request):
        mock_request.return_value = {"Financials": {"Income_Statement": {"quarterly": {}}}}

        with patch.dict(os.environ, {"EODHD_API_KEY": "test"}):
            result = get_income_statement("AAPL")

        assert isinstance(result, str)
```

### Step 2: Run tests to verify they fail

Run: `cd C:\Users\tommy.yeung\CursorProjects\TradingAgents && uv run pytest tests/test_eodhd_fundamentals.py -v`
Expected: FAIL

### Step 3: Write minimal implementation

```python
# tradingagents/dataflows/eodhd_fundamentals.py
"""
EODHD fundamentals data provider.

Drop-in replacement for alpha_vantage_fundamentals functions.
EODHD returns all fundamental data from a single endpoint.

Usage:
    from .eodhd_fundamentals import get_fundamentals, get_balance_sheet
    data = get_fundamentals("AAPL")
"""

import json
from .eodhd_common import _make_api_request, to_eodhd_symbol


def _get_full_fundamentals(ticker: str) -> dict:
    """Fetch full fundamentals data from EODHD (single API call)."""
    eodhd_symbol = to_eodhd_symbol(ticker)
    endpoint = f"/api/fundamentals/{eodhd_symbol}"
    params = {"fmt": "json"}

    try:
        data = _make_api_request(endpoint, params)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def get_fundamentals(ticker: str, curr_date: str = None) -> str:
    """Retrieve company fundamentals from EODHD.

    Args:
        ticker: Ticker symbol (e.g., "AAPL")
        curr_date: Current date (not used, for API compatibility)

    Returns:
        JSON string with company overview data
    """
    data = _get_full_fundamentals(ticker)
    if not data:
        return f"No fundamentals data available for {ticker}"

    # Extract overview-like data (matching AV OVERVIEW format)
    overview = {}
    general = data.get("General", {})
    highlights = data.get("Highlights", {})
    valuation = data.get("Valuation", {})

    overview.update(general)
    overview.update(highlights)
    overview.update(valuation)

    return json.dumps(overview, indent=2, default=str)


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    """Retrieve balance sheet from EODHD fundamentals.

    Args:
        ticker: Ticker symbol
        freq: "quarterly" or "annual"
        curr_date: Current date (not used)

    Returns:
        JSON string with balance sheet data
    """
    data = _get_full_fundamentals(ticker)
    if not data:
        return f"No balance sheet data available for {ticker}"

    financials = data.get("Financials", {})
    balance_sheet = financials.get("Balance_Sheet", {})
    freq_data = balance_sheet.get(freq, {})

    return json.dumps(freq_data, indent=2, default=str)


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    """Retrieve cash flow statement from EODHD fundamentals.

    Args:
        ticker: Ticker symbol
        freq: "quarterly" or "annual"
        curr_date: Current date (not used)

    Returns:
        JSON string with cash flow data
    """
    data = _get_full_fundamentals(ticker)
    if not data:
        return f"No cash flow data available for {ticker}"

    financials = data.get("Financials", {})
    cashflow = financials.get("Cash_Flow", {})
    freq_data = cashflow.get(freq, {})

    return json.dumps(freq_data, indent=2, default=str)


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    """Retrieve income statement from EODHD fundamentals.

    Args:
        ticker: Ticker symbol
        freq: "quarterly" or "annual"
        curr_date: Current date (not used)

    Returns:
        JSON string with income statement data
    """
    data = _get_full_fundamentals(ticker)
    if not data:
        return f"No income statement data available for {ticker}"

    financials = data.get("Financials", {})
    income = financials.get("Income_Statement", {})
    freq_data = income.get(freq, {})

    return json.dumps(freq_data, indent=2, default=str)
```

### Step 4: Run tests to verify they pass

Run: `cd C:\Users\tommy.yeung\CursorProjects\TradingAgents && uv run pytest tests/test_eodhd_fundamentals.py -v`
Expected: ALL PASS

### Step 5: Commit

```bash
cd C:\Users\tommy.yeung\CursorProjects\TradingAgents
git add tradingagents/dataflows/eodhd_fundamentals.py tests/test_eodhd_fundamentals.py
git commit -m "feat(eodhd): add fundamentals module (overview, balance sheet, cashflow, income)"
```

---

## Task 6: EODHD Re-export Module + Vendor Registration (TradingAgents)

**Files:**
- Create: `tradingagents/dataflows/eodhd.py`
- Modify: `tradingagents/dataflows/interface.py:1-186` (add EODHD imports + VENDOR_METHODS entries)
- Modify: `tradingagents/dataflows/interface.py:111` (add "eodhd" to VENDOR_LIST)
- Test: `tests/test_eodhd_registration.py`

### Step 1: Write the failing tests

```python
# tests/test_eodhd_registration.py
import pytest


def test_eodhd_in_vendor_list():
    from tradingagents.dataflows.interface import VENDOR_LIST
    assert "eodhd" in VENDOR_LIST


def test_eodhd_registered_for_stock_data():
    from tradingagents.dataflows.interface import VENDOR_METHODS
    assert "eodhd" in VENDOR_METHODS["get_stock_data"]


def test_eodhd_registered_for_indicators():
    from tradingagents.dataflows.interface import VENDOR_METHODS
    assert "eodhd" in VENDOR_METHODS["get_indicators"]


def test_eodhd_registered_for_news():
    from tradingagents.dataflows.interface import VENDOR_METHODS
    assert "eodhd" in VENDOR_METHODS["get_news"]


def test_eodhd_registered_for_global_news():
    from tradingagents.dataflows.interface import VENDOR_METHODS
    assert "eodhd" in VENDOR_METHODS["get_global_news"]


def test_eodhd_registered_for_fundamentals():
    from tradingagents.dataflows.interface import VENDOR_METHODS
    assert "eodhd" in VENDOR_METHODS["get_fundamentals"]


def test_eodhd_registered_for_balance_sheet():
    from tradingagents.dataflows.interface import VENDOR_METHODS
    assert "eodhd" in VENDOR_METHODS["get_balance_sheet"]


def test_eodhd_registered_for_cashflow():
    from tradingagents.dataflows.interface import VENDOR_METHODS
    assert "eodhd" in VENDOR_METHODS["get_cashflow"]


def test_eodhd_registered_for_income_statement():
    from tradingagents.dataflows.interface import VENDOR_METHODS
    assert "eodhd" in VENDOR_METHODS["get_income_statement"]


def test_eodhd_registered_for_insider_transactions():
    from tradingagents.dataflows.interface import VENDOR_METHODS
    assert "eodhd" in VENDOR_METHODS["get_insider_transactions"]


def test_eodhd_registered_for_insider_sentiment():
    from tradingagents.dataflows.interface import VENDOR_METHODS
    assert "eodhd" in VENDOR_METHODS["get_insider_sentiment"]


def test_eodhd_re_export_module():
    """Verify eodhd.py re-exports all functions."""
    from tradingagents.dataflows.eodhd import (
        get_stock,
        get_indicator,
        get_fundamentals,
        get_balance_sheet,
        get_cashflow,
        get_income_statement,
        get_news,
        get_global_news,
        get_insider_transactions,
        get_insider_sentiment,
    )
    # Just verify imports work
    assert callable(get_stock)
    assert callable(get_indicator)
```

### Step 2: Run tests to verify they fail

Run: `cd C:\Users\tommy.yeung\CursorProjects\TradingAgents && uv run pytest tests/test_eodhd_registration.py -v`
Expected: FAIL

### Step 3: Create re-export module

```python
# tradingagents/dataflows/eodhd.py
"""
EODHD data provider — re-export module.

Centralizes imports from all EODHD sub-modules for clean registration
in interface.py's VENDOR_METHODS.

Usage:
    from .eodhd import get_stock, get_indicator, get_news
"""

from .eodhd_stock import get_stock
from .eodhd_indicator import get_indicator
from .eodhd_fundamentals import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
)
from .eodhd_news import (
    get_news,
    get_global_news,
    get_insider_transactions,
    get_insider_sentiment,
)
```

### Step 4: Update interface.py — Add EODHD imports

Add after the alpha_vantage_options import block (after line 51):

```python
from .eodhd import (
    get_stock as get_eodhd_stock,
    get_indicator as get_eodhd_indicator,
    get_fundamentals as get_eodhd_fundamentals,
    get_balance_sheet as get_eodhd_balance_sheet,
    get_cashflow as get_eodhd_cashflow,
    get_income_statement as get_eodhd_income_statement,
    get_news as get_eodhd_news,
    get_global_news as get_eodhd_global_news,
    get_insider_transactions as get_eodhd_insider_transactions,
    get_insider_sentiment as get_eodhd_insider_sentiment,
)
from .eodhd_common import EODHDRateLimitError
```

### Step 5: Update VENDOR_LIST (line 111)

Change:
```python
VENDOR_LIST = ["local", "yfinance", "openai", "google"]
```
To:
```python
VENDOR_LIST = ["local", "yfinance", "openai", "google", "eodhd"]
```

### Step 6: Update VENDOR_METHODS — Add "eodhd" entries to each method

Add `"eodhd": get_eodhd_*` to each relevant method dict:

```python
VENDOR_METHODS = {
    "get_stock_data": {
        "eodhd": get_eodhd_stock,          # ADD THIS
        "alpha_vantage": get_alpha_vantage_stock,
        "yfinance": get_YFin_data_online,
        "local": get_YFin_data,
    },
    "get_indicators": {
        "eodhd": get_eodhd_indicator,       # ADD THIS
        "alpha_vantage": get_alpha_vantage_indicator,
        "yfinance": get_stock_stats_indicators_window,
        "local": partial(get_stock_stats_indicators_window, online=False),
    },
    "get_fundamentals": {
        "eodhd": get_eodhd_fundamentals,    # ADD THIS
        "alpha_vantage": get_alpha_vantage_fundamentals,
        "openai": get_fundamentals_openai,
    },
    "get_balance_sheet": {
        "eodhd": get_eodhd_balance_sheet,   # ADD THIS
        "alpha_vantage": get_alpha_vantage_balance_sheet,
        "yfinance": get_yfinance_balance_sheet,
        "local": get_simfin_balance_sheet,
    },
    "get_cashflow": {
        "eodhd": get_eodhd_cashflow,        # ADD THIS
        "alpha_vantage": get_alpha_vantage_cashflow,
        "yfinance": get_yfinance_cashflow,
        "local": get_simfin_cashflow,
    },
    "get_income_statement": {
        "eodhd": get_eodhd_income_statement, # ADD THIS
        "alpha_vantage": get_alpha_vantage_income_statement,
        "yfinance": get_yfinance_income_statement,
        "local": get_simfin_income_statements,
    },
    "get_news": {
        "eodhd": get_eodhd_news,            # ADD THIS
        "alpha_vantage": get_alpha_vantage_news,
        "openai": get_stock_news_openai,
        "google": get_google_news,
        "local": [get_finnhub_news, get_reddit_company_news, get_google_news],
    },
    "get_global_news": {
        "eodhd": get_eodhd_global_news,     # ADD THIS
        "alpha_vantage": get_alpha_vantage_global_news,
        "openai": get_global_news_openai,
        "local": get_reddit_global_news,
    },
    "get_insider_sentiment": {
        "eodhd": get_eodhd_insider_sentiment, # ADD THIS
        "alpha_vantage": get_alpha_vantage_insider_sentiment,
        "yfinance": get_yfinance_insider_sentiment,
        "local": get_finnhub_company_insider_sentiment,
    },
    "get_insider_transactions": {
        "eodhd": get_eodhd_insider_transactions, # ADD THIS
        "alpha_vantage": get_alpha_vantage_insider_transactions,
        "yfinance": get_yfinance_insider_transactions,
        "local": get_finnhub_company_insider_transactions,
    },
    # Options — NOT adding EODHD (using yfinance for options)
    "get_options_overview": { ... },  # unchanged
    "detect_unusual_options_activity": { ... },  # unchanged
    "analyze_options_sentiment": { ... },  # unchanged
}
```

### Step 7: Update route_to_vendor() — Handle EODHDRateLimitError

In `route_to_vendor()` (around line 285), add EODHD rate limit handling alongside AV:

```python
except AlphaVantageRateLimitError as e:
    if vendor == "alpha_vantage":
        print(f"RATE_LIMIT: Alpha Vantage rate limit exceeded, falling back")
    continue
except EODHDRateLimitError as e:
    print(f"RATE_LIMIT: EODHD rate limit exceeded, falling back to next vendor")
    print(f"DEBUG: Rate limit details: {e}")
    continue
```

### Step 8: Run tests to verify they pass

Run: `cd C:\Users\tommy.yeung\CursorProjects\TradingAgents && uv run pytest tests/test_eodhd_registration.py -v`
Expected: ALL PASS

### Step 9: Run full test suite to check nothing broken

Run: `cd C:\Users\tommy.yeung\CursorProjects\TradingAgents && uv run pytest -x -v`
Expected: All existing tests still pass

### Step 10: Commit

```bash
cd C:\Users\tommy.yeung\CursorProjects\TradingAgents
git add tradingagents/dataflows/eodhd.py tradingagents/dataflows/interface.py tests/test_eodhd_registration.py
git commit -m "feat(eodhd): register EODHD vendor in interface.py with full method coverage"
```

---

## Task 7: Switchover Date Mechanism (TradingAgents + prop-firm-pilot)

**Files:**
- Modify: `tradingagents/default_config.py:154-160` (change data_vendors defaults)
- Modify: `prop-firm-pilot/src/decision/fx_analyst_config.py:83-116` (add switchover logic)
- Test: `prop-firm-pilot/tests/test_switchover.py`

### Step 1: Write the failing tests

```python
# prop-firm-pilot/tests/test_switchover.py
import os
import pytest
from unittest.mock import patch
from datetime import date


class TestSwitchover:
    def test_before_switchover_uses_alpha_vantage(self):
        """Before March 21, config should use alpha_vantage as primary."""
        with patch("src.decision.fx_analyst_config.date") as mock_date:
            mock_date.today.return_value = date(2026, 3, 15)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            from src.decision.fx_analyst_config import build_agent_config
            config = build_agent_config()
            assert config["data_vendors"]["core_stock_apis"] == "alpha_vantage"
            assert config["data_vendors"]["news_data"] == "alpha_vantage"

    def test_after_switchover_uses_eodhd(self):
        """On or after March 21, config should use eodhd as primary."""
        with patch("src.decision.fx_analyst_config.date") as mock_date:
            mock_date.today.return_value = date(2026, 3, 21)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            from src.decision.fx_analyst_config import build_agent_config
            config = build_agent_config()
            assert config["data_vendors"]["core_stock_apis"] == "eodhd"
            assert config["data_vendors"]["news_data"] == "eodhd"

    def test_env_override_switchover_date(self):
        """EODHD_SWITCHOVER_DATE env var should override default."""
        with patch.dict(os.environ, {"EODHD_SWITCHOVER_DATE": "2026-04-01"}):
            with patch("src.decision.fx_analyst_config.date") as mock_date:
                mock_date.today.return_value = date(2026, 3, 25)
                mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
                from src.decision.fx_analyst_config import build_agent_config
                config = build_agent_config()
                # Still before 4/1, so should be AV
                assert config["data_vendors"]["core_stock_apis"] == "alpha_vantage"

    def test_env_force_eodhd(self):
        """EODHD_FORCE_PRIMARY=1 should always use EODHD regardless of date."""
        with patch.dict(os.environ, {"EODHD_FORCE_PRIMARY": "1"}):
            from src.decision.fx_analyst_config import build_agent_config
            config = build_agent_config()
            assert config["data_vendors"]["core_stock_apis"] == "eodhd"
```

### Step 2: Run tests to verify they fail

Run: `cd C:\Users\tommy.yeung\CursorProjects\prop-firm\prop-firm-pilot && uv run pytest tests/test_switchover.py -v`
Expected: FAIL

### Step 3: Update fx_analyst_config.py with switchover logic

```python
# Add to top of src/decision/fx_analyst_config.py (imports section)
import os
from datetime import date

# Switchover date: Before this date, use AV. On/after, use EODHD.
_SWITCHOVER_DATE_STR = os.getenv("EODHD_SWITCHOVER_DATE", "2026-03-21")
EODHD_SWITCHOVER_DATE = date.fromisoformat(_SWITCHOVER_DATE_STR)
EODHD_FORCE_PRIMARY = os.getenv("EODHD_FORCE_PRIMARY", "").strip() in ("1", "true", "yes")


def _get_primary_vendor() -> str:
    """Determine primary data vendor based on switchover date."""
    if EODHD_FORCE_PRIMARY:
        return "eodhd"
    if date.today() >= EODHD_SWITCHOVER_DATE:
        return "eodhd"
    return "alpha_vantage"
```

Update `build_agent_config()`:

```python
def build_agent_config(output_language: str = "繁體中文") -> dict[str, Any]:
    primary = _get_primary_vendor()

    return {
        "output_language": output_language,
        "market_type": "fx",
        "data_vendors": {
            "core_stock_apis": primary,
            "news_data": primary,
        },
        "tool_vendors": {
            "get_global_news": primary,
            "get_news": primary,
            "get_indicators": primary,
            "get_insider_sentiment": "local",
            "get_insider_transactions": "local",
        },
        "fx_mode": True,
        "fx_pairs": list(FX_PAIR_CONTEXT.keys()),
        "fx_key_events": FX_KEY_EVENTS,
    }
```

### Step 4: Update TradingAgents default_config.py data_vendors

No change needed to defaults — the config override from prop-firm-pilot's `build_agent_config()` takes precedence. The defaults in `default_config.py` remain `"alpha_vantage"` as a safe fallback.

### Step 5: Run tests to verify they pass

Run: `cd C:\Users\tommy.yeung\CursorProjects\prop-firm\prop-firm-pilot && uv run pytest tests/test_switchover.py -v`
Expected: ALL PASS

### Step 6: Commit

```bash
cd C:\Users\tommy.yeung\CursorProjects\prop-firm\prop-firm-pilot
git add src/decision/fx_analyst_config.py tests/test_switchover.py
git commit -m "feat: add EODHD switchover mechanism (date-based + env override)"
```

---

## Task 8: EODHD FX Fetcher (qlib_market_scanner)

**Files:**
- Modify: `src/data/fx_fetcher.py` (add `EODHDFXFetcher` class + update `download_universe`)
- Test: `tests/test_eodhd_fx_fetcher.py`

### Step 1: Write the failing tests

```python
# qlib_market_scanner/tests/test_eodhd_fx_fetcher.py
import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock


class TestEODHDFXFetcher:
    def test_fetch_daily_returns_dataframe(self):
        from src.data.fx_fetcher import EODHDFXFetcher

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"timestamp": 1740441600, "gmtoffset": 0, "open": 1.045, "high": 1.048, "low": 1.042, "close": 1.046, "volume": 0},
            {"timestamp": 1740528000, "gmtoffset": 0, "open": 1.046, "high": 1.050, "low": 1.043, "close": 1.049, "volume": 0},
        ]
        mock_response.raise_for_status.return_value = None

        with patch("src.data.fx_fetcher.requests.get", return_value=mock_response):
            fetcher = EODHDFXFetcher(api_key="test_key")
            df = fetcher.fetch_daily("EURUSD", "2025-02-25", "2025-02-26")

        assert isinstance(df, pd.DataFrame)
        assert "date" in df.columns
        assert "open" in df.columns
        assert "close" in df.columns
        assert "adj_close" in df.columns
        assert "volume" in df.columns

    def test_symbol_conversion(self):
        from src.data.fx_fetcher import EODHDFXFetcher
        fetcher = EODHDFXFetcher(api_key="test_key")
        assert fetcher._to_eodhd_symbol("EURUSD") == "EURUSD.FOREX"
        assert fetcher._to_eodhd_symbol("GBPUSD") == "GBPUSD.FOREX"

    def test_empty_response(self):
        from src.data.fx_fetcher import EODHDFXFetcher

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None

        with patch("src.data.fx_fetcher.requests.get", return_value=mock_response):
            fetcher = EODHDFXFetcher(api_key="test_key")
            df = fetcher.fetch_daily("EURUSD", "2025-02-25", "2025-02-26")

        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestDownloadUniverseEODHD:
    def test_eodhd_key_takes_priority(self):
        """When EODHD_API_KEY is set, should use EODHDFXFetcher."""
        with patch.dict(os.environ, {"EODHD_API_KEY": "test_key", "ALPHA_VANTAGE_API_KEY": "av_key"}):
            from src.data import fx_fetcher
            # Re-import to test fetcher selection logic
            # The download_universe function should check EODHD_API_KEY first
```

### Step 2: Run tests to verify they fail

Run: `cd C:\Users\tommy.yeung\CursorProjects\qlib_market_scanner && uv run pytest tests/test_eodhd_fx_fetcher.py -v`
Expected: FAIL — `EODHDFXFetcher` class not found

### Step 3: Add EODHDFXFetcher class to fx_fetcher.py

Add after the `AlphaVantageFXFetcher` class (after line 263):

```python
class EODHDFXFetcher:
    """EODHD API FX data fetcher (Synchronous).

    Docs: https://eodhd.com/financial-apis/intraday-historical-data-api
    Uses /api/eod/ for daily data and /api/intraday/ for intraday.

    Usage:
        fetcher = EODHDFXFetcher(api_key="your_key")
        df = fetcher.fetch_daily("EURUSD", "2025-01-01", "2025-12-31")
    """

    BASE_URL = "https://eodhd.com"

    def __init__(self, api_key: str, max_retries: int = 3):
        self._api_key = api_key
        self._max_retries = max_retries

    def _to_eodhd_symbol(self, symbol: str) -> str:
        """Convert internal symbol to EODHD format (e.g., EURUSD -> EURUSD.FOREX)."""
        clean = symbol.replace("/", "").upper()
        if "." not in clean:
            return f"{clean}.FOREX"
        return clean

    def fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch daily OHLCV data from EODHD EOD API.

        Args:
            symbol: FX pair (e.g., "EURUSD")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)

        Returns:
            DataFrame with columns: date, open, high, low, close, adj_close, volume
        """
        eodhd_symbol = self._to_eodhd_symbol(symbol)

        params = {
            "api_token": self._api_key,
            "fmt": "json",
            "from": start,
            "to": end,
        }

        for attempt in range(self._max_retries + 1):
            try:
                url = f"{self.BASE_URL}/api/eod/{eodhd_symbol}"
                response = requests.get(url, params=params, timeout=30)

                if response.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"EODHD rate limit hit, waiting {wait}s...")
                    time.sleep(wait)
                    continue

                if response.status_code != 200:
                    logger.warning(f"EODHD HTTP {response.status_code}: {response.text}")
                    time.sleep(2 ** attempt)
                    continue

                data = response.json()

                if not data or not isinstance(data, list):
                    return pd.DataFrame()

                rows = []
                for item in data:
                    rows.append({
                        "date": pd.Timestamp(item["date"]),
                        "open": float(item["open"]),
                        "high": float(item["high"]),
                        "low": float(item["low"]),
                        "close": float(item["close"]),
                        "adj_close": float(item.get("adjusted_close", item["close"])),
                        "volume": float(item.get("volume", 1.0)),  # FX may have 0 volume
                    })

                if not rows:
                    return pd.DataFrame()

                df = pd.DataFrame(rows)

                # Filter to exact range
                start_dt = pd.to_datetime(start)
                end_dt = pd.to_datetime(end)
                df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

                return df.sort_values("date").reset_index(drop=True)

            except Exception as e:
                logger.warning(f"EODHD fetch error ({symbol}): {e}")
                time.sleep(2 ** attempt)

        return pd.DataFrame()
```

### Step 4: Update download_universe() to prefer EODHD

In `download_universe()` (line 265-373), update the fetcher selection logic:

```python
def download_universe(
    tickers: Iterable[str],
    start: str,
    end: str,
    output_dir: str,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    cache_max_dates: Optional[dict[str, str]] = None,
) -> List[Path]:
    """Download FX data — EODHD preferred, Alpha Vantage fallback."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Fetcher priority: EODHD > Alpha Vantage > Mock
    eodhd_key = os.getenv("EODHD_API_KEY")
    av_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    fetcher = None

    if eodhd_key:
        logger.info("Using EODHD API for FX data")
        fetcher = EODHDFXFetcher(eodhd_key, max_retries)
    elif av_key:
        logger.info("Using Alpha Vantage API for FX data")
        fetcher = AlphaVantageFXFetcher(av_key, max_retries)
    else:
        logger.warning("No EODHD_API_KEY or ALPHA_VANTAGE_API_KEY found. Using MockFetcher.")
        fetcher = MockFetcher()

    # ... rest of function unchanged ...
```

### Step 5: Run tests to verify they pass

Run: `cd C:\Users\tommy.yeung\CursorProjects\qlib_market_scanner && uv run pytest tests/test_eodhd_fx_fetcher.py -v`
Expected: ALL PASS

### Step 6: Run full test suite

Run: `cd C:\Users\tommy.yeung\CursorProjects\qlib_market_scanner && uv run pytest -v`
Expected: Existing tests still pass

### Step 7: Commit

```bash
cd C:\Users\tommy.yeung\CursorProjects\qlib_market_scanner
git add src/data/fx_fetcher.py tests/test_eodhd_fx_fetcher.py
git commit -m "feat(eodhd): add EODHDFXFetcher with priority over AlphaVantage"
```

---

## Task 9: EODHD Stock Fetcher (qlib_market_scanner)

**Files:**
- Create: `src/data/eodhd_fetcher.py`
- Modify: `src/pipeline/runner.py:72-75` (add EODHD stock fetcher import)
- Test: `tests/test_eodhd_stock_fetcher.py`

### Step 1: Write the failing tests

```python
# qlib_market_scanner/tests/test_eodhd_stock_fetcher.py
import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock


class TestEODHDStockFetcher:
    @patch("src.data.eodhd_fetcher.requests.get")
    def test_download_ticker_returns_dataframe(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"date": "2026-02-25", "open": 180.0, "high": 182.0, "low": 179.0, "close": 181.5, "adjusted_close": 181.5, "volume": 50000000},
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        from src.data.eodhd_fetcher import download_ticker
        df = download_ticker("AAPL", "2026-02-25", "2026-02-25", "1d", api_key="test")

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "open" in df.columns
        assert "adj_close" in df.columns
```

### Step 2: Run tests to verify they fail

Run: `cd C:\Users\tommy.yeung\CursorProjects\qlib_market_scanner && uv run pytest tests/test_eodhd_stock_fetcher.py -v`
Expected: FAIL

### Step 3: Create eodhd_fetcher.py

```python
# src/data/eodhd_fetcher.py
"""
EODHD stock data fetcher for qlib_market_scanner.

Replaces alpha_vantage_fetcher.py for stock universe data downloads.
Same interface: download_ticker(), download_universe(), build_output_dir().

Usage:
    from src.data.eodhd_fetcher import download_universe
    paths = download_universe(tickers, start, end, "1d", output_dir)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests
from loguru import logger


BASE_URL = "https://eodhd.com"


class EODHDRateLimitError(RuntimeError):
    pass


def _to_eodhd_symbol(ticker: str) -> str:
    """Convert ticker to EODHD format (e.g., AAPL -> AAPL.US)."""
    if "." in ticker:
        return ticker
    return f"{ticker}.US"


def download_ticker(
    ticker: str,
    start: str,
    end: str,
    interval: str,
    api_key: str,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
) -> pd.DataFrame:
    """Download OHLCV data for a single ticker from EODHD.

    Args:
        ticker: Stock symbol (e.g., "AAPL")
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        interval: "1d" for daily (intraday not yet supported)
        api_key: EODHD API key
        max_retries: Max retry attempts
        backoff_factor: Backoff multiplier for retries

    Returns:
        DataFrame with columns: date, open, high, low, close, adj_close, volume
    """
    if not api_key:
        raise ValueError("EODHD_API_KEY is required.")

    eodhd_symbol = _to_eodhd_symbol(ticker)

    params = {
        "api_token": api_key,
        "fmt": "json",
        "from": start,
        "to": end,
    }

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            url = f"{BASE_URL}/api/eod/{eodhd_symbol}"
            response = requests.get(url, params=params, timeout=20)

            if response.status_code == 429:
                raise EODHDRateLimitError("EODHD rate limit exceeded")

            response.raise_for_status()
            data = response.json()

            if not data or not isinstance(data, list):
                logger.warning("EODHD returned no data for {}", ticker)
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df = df.rename(columns={"adjusted_close": "adj_close"})

            # Ensure required columns
            if "adj_close" not in df.columns:
                df["adj_close"] = df["close"]

            df["date"] = pd.to_datetime(df["date"])

            # Filter date range
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

            required = {"date", "open", "high", "low", "close", "volume"}
            if not required.issubset(df.columns):
                logger.warning("EODHD response missing columns for {}", ticker)
                return pd.DataFrame()

            # Select and order columns
            cols = ["date", "open", "high", "low", "close", "adj_close", "volume"]
            df = df[[c for c in cols if c in df.columns]]

            return df.sort_values("date").reset_index(drop=True)

        except EODHDRateLimitError:
            raise
        except Exception as exc:
            last_error = exc
            logger.warning(
                "EODHD download failed (attempt {}/{}): {} - {}",
                attempt + 1, max_retries + 1, ticker, exc,
            )

        if attempt < max_retries:
            sleep_seconds = backoff_factor * (2 ** attempt)
            time.sleep(sleep_seconds)

    if last_error is not None:
        raise last_error
    return pd.DataFrame()


def download_universe(
    tickers: Iterable[str],
    start: str,
    end: str,
    interval: str,
    output_dir: str,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    sleep_between_requests: float = 0.2,
    sleep_jitter: float = 0.1,
    rate_limit_per_sec: float = 5.0,
    cache_max_dates: Optional[dict[str, str]] = None,
) -> List[Path]:
    """Download stock data for universe from EODHD.

    Same interface as alpha_vantage_fetcher.download_universe().
    """
    import random

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    api_key = os.getenv("EODHD_API_KEY", "")
    ticker_list = [t for t in tickers if t]

    if cache_max_dates:
        original_end = end
        skip = [t for t in ticker_list if cache_max_dates.get(t, "") >= original_end]
        if skip:
            logger.info("Skipping {} tickers already up-to-date.", len(skip))
        ticker_list = [t for t in ticker_list if t not in skip]
        if not ticker_list:
            logger.info("All tickers up-to-date; skipping EODHD download.")
            return []

    min_interval = 1.0 / max(rate_limit_per_sec, 1.0)

    for ticker in ticker_list:
        try:
            request_start = time.time()
            df = download_ticker(
                ticker, start, end, interval,
                api_key=api_key,
                max_retries=max_retries,
                backoff_factor=backoff_factor,
            )
        except EODHDRateLimitError:
            raise
        except Exception as exc:
            logger.warning("EODHD download failed for {}: {}", ticker, exc)
            continue

        if df.empty:
            logger.warning("EODHD download empty after retries: {}", ticker)
        else:
            file_path = output_path / f"{ticker}.csv"
            df.to_csv(file_path, index=False)
            written.append(file_path)

        elapsed = time.time() - request_start
        sleep_for = max(0.0, min_interval - elapsed)
        if sleep_for > 0:
            time.sleep(sleep_for + random.random() * sleep_jitter)

    if not written:
        raise RuntimeError("All downloads failed (eodhd). Check network or API key.")

    return written


def build_output_dir(raw_dir: str, interval: str) -> str:
    """Build output directory path for the given interval."""
    suffix = "1min" if interval == "1m" else "day"
    return os.path.join(raw_dir, suffix)
```

### Step 4: Update runner.py to use EODHD for stock profile

In `runner.py` (around line 72-75), update the stock profile fetcher import:

```python
else:
    # Stock profile: prefer EODHD if API key available, fallback to AV
    eodhd_key = os.getenv("EODHD_API_KEY")
    if eodhd_key:
        from src.data.eodhd_fetcher import download_universe
        logger.info("Using Stock profile with EODHD fetcher")
    else:
        from src.data.alpha_vantage_fetcher import download_universe
        logger.info("Using Stock profile with Alpha Vantage fetcher")
```

### Step 5: Run tests to verify they pass

Run: `cd C:\Users\tommy.yeung\CursorProjects\qlib_market_scanner && uv run pytest tests/test_eodhd_stock_fetcher.py -v`
Expected: ALL PASS

### Step 6: Commit

```bash
cd C:\Users\tommy.yeung\CursorProjects\qlib_market_scanner
git add src/data/eodhd_fetcher.py src/pipeline/runner.py tests/test_eodhd_stock_fetcher.py
git commit -m "feat(eodhd): add EODHD stock fetcher with runner integration"
```

---

## Task 10: Environment Variable Setup + .env.example Updates

**Files:**
- Modify: `prop-firm-pilot/.env.example` (add EODHD keys)
- Modify: `qlib_market_scanner/.env.example` (add EODHD key if exists)
- Modify: `TradingAgents/.env.example` (add EODHD key if exists)

### Step 1: Update .env.example files

Add to each repo's `.env.example`:
```
# EODHD (eodhistoricaldata.com) — Primary data provider (v1.3.0+)
# Free tier: 20 API calls/day | Paid (All World Extended): 100K calls/day
EODHD_API_KEY=your_eodhd_api_key_here

# Switchover control (optional)
# EODHD_SWITCHOVER_DATE=2026-03-21    # Date to switch from AV to EODHD (default: 2026-03-21)
# EODHD_FORCE_PRIMARY=1               # Force EODHD regardless of date (for testing)
```

### Step 2: Commit

```bash
# In each repo, add and commit the .env.example changes
```

---

## Task 11: Version Bump to 1.3.0 (All 3 Repos)

**Files:**
- Modify: `prop-firm-pilot/pyproject.toml` (version = "1.3.0")
- Modify: `qlib_market_scanner/pyproject.toml` (version = "1.3.0")
- Modify: `TradingAgents/pyproject.toml` (version = "1.3.0")
- Modify: Any `__version__` strings if present

### Step 1: Find version strings

Run: `grep -r "version" pyproject.toml` in each repo root

### Step 2: Update versions to 1.3.0

### Step 3: Commit in each repo

```bash
git commit -m "chore: bump version to 1.3.0 — EODHD data migration"
```

---

## Task 12: Integration Testing with EODHD Free Tier

**Files:**
- Create: `TradingAgents/tests/test_eodhd_integration.py`
- Create: `qlib_market_scanner/tests/test_eodhd_integration.py`

### Step 1: Write integration tests (skipped without API key)

```python
# TradingAgents/tests/test_eodhd_integration.py
import os
import pytest

requires_eodhd = pytest.mark.skipif(
    not os.getenv("EODHD_API_KEY"),
    reason="EODHD_API_KEY not set"
)


@requires_eodhd
class TestEODHDIntegration:
    """Live API tests — only run when EODHD_API_KEY is set.

    Free tier demo tickers: AAPL.US, EURUSD.FOREX
    Rate limit: 20 calls/day — run sparingly!
    """

    def test_fx_daily_data(self):
        from tradingagents.dataflows.eodhd_stock import get_stock
        result = get_stock("EURUSD", "2026-02-01", "2026-02-28")
        assert isinstance(result, str)
        assert "timestamp" in result or "date" in result
        assert "open" in result

    def test_stock_daily_data(self):
        from tradingagents.dataflows.eodhd_stock import get_stock
        result = get_stock("AAPL", "2026-02-01", "2026-02-28")
        assert isinstance(result, str)
        assert "open" in result

    def test_indicator_rsi(self):
        from tradingagents.dataflows.eodhd_indicator import get_indicator
        result = get_indicator("EURUSD", "rsi", "2026-02-28", 30)
        assert isinstance(result, str)
        assert "RSI" in result.upper()

    def test_news(self):
        from tradingagents.dataflows.eodhd_news import get_news
        result = get_news("EURUSD", "2026-02-20", "2026-02-28")
        assert result is not None

    def test_vendor_routing(self):
        """Test that route_to_vendor works with eodhd config."""
        from tradingagents.dataflows.interface import route_to_vendor
        from tradingagents.dataflows.config import set_config

        set_config({
            "data_vendors": {"core_stock_apis": "eodhd"},
            "tool_vendors": {},
        })
        try:
            result = route_to_vendor("get_stock_data", "EURUSD", "2026-02-01", "2026-02-28")
            assert result is not None
        finally:
            # Reset config
            set_config({
                "data_vendors": {"core_stock_apis": "alpha_vantage"},
                "tool_vendors": {},
            })
```

### Step 2: Run integration tests manually

Run: `cd C:\Users\tommy.yeung\CursorProjects\TradingAgents && EODHD_API_KEY=your_key uv run pytest tests/test_eodhd_integration.py -v`

### Step 3: Commit

```bash
cd C:\Users\tommy.yeung\CursorProjects\TradingAgents
git add tests/test_eodhd_integration.py
git commit -m "test: add EODHD integration tests (skipped without API key)"
```

---

## Task 13: Final Verification + v1.3.0 Report

### Step 1: Run full test suites in all 3 repos

```bash
# TradingAgents
cd C:\Users\tommy.yeung\CursorProjects\TradingAgents && uv run pytest -v

# qlib_market_scanner
cd C:\Users\tommy.yeung\CursorProjects\qlib_market_scanner && uv run pytest -v

# prop-firm-pilot
cd C:\Users\tommy.yeung\CursorProjects\prop-firm\prop-firm-pilot && uv run pytest -v
```

### Step 2: Lint all repos

```bash
# Each repo
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

### Step 3: Write v1.3.0 report

Create: `prop-firm-pilot/docs/PropFirmPilot_v1.3.0_Report.md`

### Step 4: Final commit + tag

```bash
# Tag all repos with v1.3.0
git tag -a v1.3.0 -m "EODHD data source migration"
```

---

## Execution Checklist

| # | Task | Repo | Est. Time | Status |
|---|------|------|-----------|--------|
| 1 | EODHD Common Utilities | TradingAgents | 5 min | ⬜ |
| 2 | EODHD Stock/FX Module | TradingAgents | 5 min | ⬜ |
| 3 | EODHD Indicators Module | TradingAgents | 5 min | ⬜ |
| 4 | EODHD News Module | TradingAgents | 5 min | ⬜ |
| 5 | EODHD Fundamentals Module | TradingAgents | 5 min | ⬜ |
| 6 | Vendor Registration (interface.py) | TradingAgents | 10 min | ⬜ |
| 7 | Switchover Mechanism | prop-firm-pilot | 5 min | ⬜ |
| 8 | EODHD FX Fetcher | qlib_market_scanner | 10 min | ⬜ |
| 9 | EODHD Stock Fetcher | qlib_market_scanner | 10 min | ⬜ |
| 10 | .env.example Updates | All 3 repos | 2 min | ⬜ |
| 11 | Version Bump 1.3.0 | All 3 repos | 2 min | ⬜ |
| 12 | Integration Tests | TradingAgents + scanner | 5 min | ⬜ |
| 13 | Final Verification + Report | All 3 repos | 10 min | ⬜ |

**Total estimated: ~80 minutes**

---

## Key Design Decisions

1. **EODHD modules mirror AV module structure** — same function signatures, same return formats, enabling drop-in replacement via `VENDOR_METHODS` registration
2. **Switchover is config-driven** — `EODHD_SWITCHOVER_DATE` env var (default: 2026-03-21) controls which vendor is primary. `EODHD_FORCE_PRIMARY=1` for testing
3. **Fallback is automatic** — `route_to_vendor()` already handles multi-vendor fallback. EODHD failures cascade to AV → yfinance → local
4. **Options stay on yfinance** — No EODHD options module needed (user decision: "先用OpenBB/yfinance, 之後再考慮")
5. **Insider data stays on local/finnhub** — For FX mode, insider data comes from local vendors (finnhub)
6. **EODHD free tier for testing** — 20 calls/day with demo tickers (AAPL.US, EURUSD.FOREX). Integration tests auto-skip without API key
7. **No 4H aggregation in this version** — Scanner uses daily data. 4H would only be needed for prop-firm-pilot intraday analysis (future enhancement)
