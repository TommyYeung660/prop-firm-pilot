"""
MatchTrader REST API client for E8 Markets prop firm trading.

Handles JWT authentication, auto-refresh, rate limiting (2000 req/day),
and all trading operations (open/close/modify positions, balance queries).

Uses curl_cffi with Chrome TLS fingerprint impersonation to bypass
Cloudflare protection on mtr.e8markets.com.

API Reference:
    - https://app.theneo.io/match-trade/platform-api
    - https://docs.match-trade.com/docs/match-trader-api-documentation/
"""

import asyncio
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Literal

from curl_cffi.requests import AsyncSession
from loguru import logger
from pydantic import BaseModel, Field

from src.execution.broker_models import (
    BrokerBalanceInfo,
    BrokerClosedPosition,
    BrokerInstrumentInfo,
    BrokerOrderResult,
    BrokerPositionInfo,
    BrokerQuoteInfo,
)

# ── Response Models ─────────────────────────────────────────────────────────


class AuthTokens(BaseModel):
    """Tokens returned from login / refresh."""

    trading_api_token: str = Field(description="JWT for Auth-trading-api header")
    refresh_token: str = Field(description="Token used to refresh the JWT")
    system_uuid: str = Field(description="Account system UUID for API paths")


# Backward-compatible aliases for existing MatchTrader imports.
BalanceInfo = BrokerBalanceInfo
PositionInfo = BrokerPositionInfo
OrderResult = BrokerOrderResult
ClosedPosition = BrokerClosedPosition
QuoteInfo = BrokerQuoteInfo
InstrumentInfo = BrokerInstrumentInfo


# ── Rate Limiter ────────────────────────────────────────────────────────────


class RateLimiter:
    """Daily rate limiter for MatchTrader API (2000 req/day).

    Supports optional persistent storage via DecisionStore, so counts
    survive process restarts. Tracks read/write breakdown for analysis.

    Usage:
        limiter = RateLimiter(daily_limit=2000, store=decision_store)
        limiter.record(call_type="read")
        if limiter.can_proceed():
            # make request
    """

    def __init__(self, daily_limit: int = 2000, store: Any = None):
        self._daily_limit = daily_limit
        self._store = store
        self._count = 0
        self._read_count = 0
        self._write_count = 0
        self._reset_date = datetime.now(timezone.utc).date()
        # Load existing counts from store if available
        if store is not None:
            try:
                breakdown = store.get_api_call_breakdown()
                self._count = breakdown["total"]
                self._read_count = breakdown["read"]
                self._write_count = breakdown["write"]
                logger.info(
                    "RateLimiter: loaded {} existing API calls from store (read={}, write={})",
                    self._count,
                    self._read_count,
                    self._write_count,
                )
            except Exception as e:
                logger.warning("RateLimiter: failed to load counts from store: {}", e)

    def _maybe_reset(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self._reset_date:
            self._count = 0
            self._read_count = 0
            self._write_count = 0
            self._reset_date = today

    def record(self, call_type: str = "read") -> None:
        """Record an API call with optional type classification.

        Args:
            call_type: "read" for GET/query calls, "write" for POST/mutating calls.
        """
        self._maybe_reset()
        self._count += 1
        if call_type == "write":
            self._write_count += 1
        else:
            self._read_count += 1
        # Persist to store if available
        if self._store is not None:
            try:
                self._store.record_api_calls(count=1, call_type=call_type)
            except Exception as e:
                logger.warning("RateLimiter: failed to persist to store: {}", e)

    @property
    def remaining(self) -> int:
        self._maybe_reset()
        return max(0, self._daily_limit - self._count)

    @property
    def write_remaining(self) -> int:
        """Remaining write-budget for mutating endpoints (open/close/edit)."""
        self._maybe_reset()
        return max(0, self._daily_limit - self._write_count)

    @property
    def daily_write_limit(self) -> int:
        """Configured daily budget for write operations."""
        return self._daily_limit

    @property
    def count(self) -> int:
        self._maybe_reset()
        return self._count

    @property
    def read_count(self) -> int:
        """Number of read (GET) API calls today."""
        self._maybe_reset()
        return self._read_count

    @property
    def write_count(self) -> int:
        """Number of write (POST) API calls today."""
        self._maybe_reset()
        return self._write_count

    def can_proceed(self, reserve: int = 50) -> bool:
        """Check if we can make another mutating request, keeping emergency reserve."""
        return self.write_remaining > reserve


# ── MatchTrader Client ──────────────────────────────────────────────────────


class MatchTraderClient:
    """Async client for MatchTrader REST API.

    Usage:
        client = MatchTraderClient(
            base_url="https://mtr.e8markets.com",
            email="user@example.com",
            password="secret",
            broker_id="2",
            account_id="950552",
        )
        async with client:
            await client.login()
            balance = await client.get_balance()
            order = await client.open_position("EURUSD", "BUY", 0.1, sl=1.0500, tp=1.1000)
    """

    # Token lifetime: 15 min. Refresh at 12 min to be safe.
    TOKEN_REFRESH_SECONDS = 12 * 60

    def __init__(
        self,
        base_url: str,
        email: str,
        password: str,
        broker_id: str = "2",
        account_id: str | None = None,
        daily_request_limit: int = 2000,
        max_retries: int = 3,
        store: Any = None,
        on_retry: Callable[[], None] | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._email = email
        self._password = password
        self._broker_id = broker_id
        self._account_id = account_id
        self._max_retries = max_retries

        self._tokens: AuthTokens | None = None
        self._last_auth_time: float = 0.0
        self._rate_limiter = RateLimiter(daily_request_limit, store=store)
        self._session: AsyncSession[Any] | None = None
        self._on_retry = on_retry

    # ── Context Manager ─────────────────────────────────────────────────

    async def __aenter__(self) -> "MatchTraderClient":
        self._session = AsyncSession(
            impersonate="safari",
            timeout=30,
            allow_redirects=True,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def is_authenticated(self) -> bool:
        return self._tokens is not None

    @property
    def system_uuid(self) -> str:
        if not self._tokens:
            raise RuntimeError("Not authenticated. Call login() first.")
        return self._tokens.system_uuid

    @property
    def rate_limiter(self) -> RateLimiter:
        return self._rate_limiter

    # ── Auth ────────────────────────────────────────────────────────────

    async def login(self, *, _quiet: bool = False) -> AuthTokens:
        """Authenticate with MatchTrader and obtain JWT tokens.

        If account_id was provided at construction, selects that specific
        trading account. Otherwise falls back to the first account returned.
        """
        _log = logger.debug if _quiet else logger.info
        _log("MatchTrader: logging in as {}", self._email)

        response = await self._raw_request(
            "POST",
            "/manager/co-login",
            json={
                "email": self._email,
                "password": self._password,
                "brokerId": self._broker_id,
            },
            authenticated=False,
        )

        data = response.json()

        # Extract system UUID from accounts list
        accounts = data.get("accounts", [])
        if not accounts:
            raise RuntimeError(f"No trading accounts found for {self._email}")

        # Select account by ID if specified, otherwise use first
        if self._account_id:
            account = next(
                (a for a in accounts if a.get("tradingAccountId") == self._account_id),
                None,
            )
            if not account:
                available = [a.get("tradingAccountId", "?") for a in accounts]
                raise RuntimeError(f"Account {self._account_id} not found. Available: {available}")
        else:
            account = accounts[0]

        system_uuid = account.get("offer", {}).get("system", {}).get("uuid", "")
        if not system_uuid:
            # Fallback to legacy response format
            system_uuid = account.get("systemUUID", account.get("id", ""))
        if not system_uuid:
            raise RuntimeError("Could not extract systemUUID from login response")

        trading_api_token = account.get("tradingApiToken", data.get("tradingApiToken", ""))
        if not trading_api_token:
            raise RuntimeError("Could not extract tradingApiToken from login response")

        self._tokens = AuthTokens(
            trading_api_token=trading_api_token,
            refresh_token=data.get("token", data.get("refreshToken", "")),
            system_uuid=system_uuid,
        )
        self._last_auth_time = time.monotonic()

        selected_id = account.get("tradingAccountId", "?")
        _log(
            "MatchTrader: login successful. account={}, systemUUID={}, total_accounts={}",
            selected_id,
            system_uuid,
            len(accounts),
        )
        return self._tokens

    async def refresh_token(self) -> None:
        """Refresh JWT before it expires (15 min lifetime).

        Since MatchTrader changed their /refresh-token endpoint, we simply
        re-authenticate by calling login() which provides a fresh token
        and updates our internal timestamps.
        """
        if not self._tokens:
            raise RuntimeError("Cannot refresh: not authenticated.")

        logger.debug("MatchTrader: refreshing JWT token via re-login")
        await self.login(_quiet=True)
        logger.debug("MatchTrader: token refreshed successfully")

    async def _ensure_auth(self) -> None:
        """Auto-refresh token if it's about to expire."""
        if not self._tokens:
            raise RuntimeError("Not authenticated. Call login() first.")

        elapsed = time.monotonic() - self._last_auth_time
        if elapsed >= self.TOKEN_REFRESH_SECONDS:
            await self.refresh_token()

    # ── Account Info ────────────────────────────────────────────────────

    async def get_balance(self) -> BrokerBalanceInfo:
        """Get current account balance, equity, margin."""
        await self._ensure_auth()
        response = await self._api_request("GET", f"/mtr-api/{self.system_uuid}/balance")
        return BrokerBalanceInfo(**response.json())

    async def get_account_details(self) -> dict[str, Any]:
        """Get account details (leverage, offer name, etc.)."""
        await self._ensure_auth()
        response = await self._api_request("GET", f"/mtr-api/{self.system_uuid}/account-details")
        return response.json()

    async def get_effective_instruments(self) -> list[BrokerInstrumentInfo]:
        """Get tradeable instruments for this account.

        Returns only instruments available on the current account/offer.
        Uses /effective-instruments (NOT /instruments which includes
        non-tradeable symbols).

        Note: E8 account 950552 uses dot-suffix symbols (e.g. "EURUSD."
        instead of "EURUSD"). Always use the symbol from this list when
        opening positions.
        """
        await self._ensure_auth()
        response = await self._api_request(
            "GET", f"/mtr-api/{self.system_uuid}/effective-instruments"
        )
        data = response.json()
        instruments_raw = data if isinstance(data, list) else data.get("instruments", [])
        instruments = [BrokerInstrumentInfo(**item) for item in instruments_raw]
        logger.info("MatchTrader: loaded {} effective instruments", len(instruments))
        return instruments

    # ── Market Watch ────────────────────────────────────────────────────

    async def get_quote(self, symbol: str) -> BrokerQuoteInfo:
        """Get real-time bid/ask quote for an instrument.

        Calls the Market Watch /quotations endpoint to retrieve the
        broker's current pricing. Used for pre-trade slippage validation.

        Args:
            symbol: Broker symbol (e.g. "EURUSD." with dot suffix).

        Returns:
            BrokerQuoteInfo with bid, ask, high, low, and timestamp.

        Raises:
            MatchTraderError: If quote cannot be retrieved.
        """
        await self._ensure_auth()
        response = await self._api_request(
            "GET",
            f"/mtr-api/{self.system_uuid}/quotations?symbols={symbol}",
        )
        data = response.json()

        # API returns a list of quotes, find the matching symbol
        quotes = data if isinstance(data, list) else []
        if not quotes:
            raise MatchTraderError(f"No quote returned for {symbol}")

        # Match the requested symbol (first match)
        for q in quotes:
            if q.get("symbol", "") == symbol:
                return BrokerQuoteInfo(
                    symbol=q.get("symbol", ""),
                    bid=float(q.get("bid", 0)),
                    ask=float(q.get("ask", 0)),
                    high=float(q.get("high", 0)),
                    low=float(q.get("low", 0)),
                    timestampMs=int(q.get("timestampMs", 0)),
                )

        # Fallback: use first quote if symbol doesn't match exactly
        q = quotes[0]
        return BrokerQuoteInfo(
            symbol=q.get("symbol", ""),
            bid=float(q.get("bid", 0)),
            ask=float(q.get("ask", 0)),
            high=float(q.get("high", 0)),
            low=float(q.get("low", 0)),
            timestampMs=int(q.get("timestampMs", 0)),
        )

    # ── Position Queries ────────────────────────────────────────────────

    async def get_open_positions(self) -> list[BrokerPositionInfo]:
        """Get all currently open positions."""
        await self._ensure_auth()
        response = await self._api_request("GET", f"/mtr-api/{self.system_uuid}/open-positions")
        data = response.json()

        # API may return list directly or wrapped in a key
        positions_raw = data if isinstance(data, list) else data.get("positions", [])
        return [BrokerPositionInfo(**p) for p in positions_raw]

    async def get_closed_positions(
        self,
        from_ts: int,
        to_ts: int,
    ) -> list[BrokerClosedPosition]:
        """Get closed positions within a time range.

        Args:
            from_ts: Start timestamp (milliseconds).
            to_ts: End timestamp (milliseconds).
        """
        await self._ensure_auth()
        response = await self._api_request(
            "POST",
            f"/mtr-api/{self.system_uuid}/closed-positions",
            json={"from": from_ts, "to": to_ts},
        )
        data = response.json()
        positions_raw = data if isinstance(data, list) else data.get("operations", [])
        return [BrokerClosedPosition(**p) for p in positions_raw]

    # ── Trading Operations ──────────────────────────────────────────────

    async def open_position(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        volume: float,
        sl: float | None = None,
        tp: float | None = None,
    ) -> BrokerOrderResult:
        """Open a new trading position.

        Args:
            symbol: Instrument name (e.g. "EURUSD").
            side: "BUY" or "SELL".
            volume: Lot size (e.g. 0.1).
            sl: Stop loss price (optional).
            tp: Take profit price (optional).
        """
        await self._ensure_auth()

        try:
            # Determine actual symbol to trade (append '.' if needed for this account)
            trade_symbol = symbol
            instruments = await self.get_effective_instruments()
            for i in instruments:
                if i.symbol == symbol or i.symbol == f"{symbol}.":
                    trade_symbol = i.symbol
                    break

            body: dict[str, Any] = {
                "instrument": trade_symbol,
                "orderSide": side.upper(),
                "volume": volume,
            }
            if sl is not None:
                body["slPrice"] = sl
            if tp is not None:
                body["tpPrice"] = tp

            logger.info(
                "MatchTrader: opening {} {} {} lots (SL={}, TP={})",
                side,
                symbol,
                volume,
                sl,
                tp,
            )

            response = await self._api_request(
                "POST",
                f"/mtr-api/{self.system_uuid}/position/open",
                json=body,
            )
            data = response.json()
            return BrokerOrderResult(
                success=True,
                position_id=str(data.get("orderId", data.get("positionId", data.get("id", "")))),
                message="Position opened successfully",
                raw_response=data,
            )
        except MatchTraderError as e:
            logger.error("MatchTrader: failed to open position: {}", e)
            return BrokerOrderResult(
                success=False,
                message=str(e),
                raw_response={"error": str(e)},
            )

    async def close_position(
        self,
        position_id: str,
        symbol: str,
        side: str,
        volume: float,
    ) -> BrokerOrderResult:
        """Close an existing position.

        Args:
            position_id: The position ID to close.
            symbol: Instrument name.
            side: Original order side ("BUY" or "SELL").
            volume: Volume to close (MatchTrader requires this explicitly).
        """
        await self._ensure_auth()

        body: dict[str, Any] = {
            "positionId": position_id,
            "instrument": symbol,
            "orderSide": side.upper(),
            "volume": volume,
        }

        logger.info(
            "MatchTrader: closing position {} ({} {} vol={})",
            position_id,
            symbol,
            side,
            volume,
        )

        try:
            response = await self._api_request(
                "POST",
                f"/mtr-api/{self.system_uuid}/position/close",
                json=body,
            )
            data = response.json()
            return BrokerOrderResult(
                success=True,
                position_id=position_id,
                message="Position closed successfully",
                raw_response=data,
            )
        except MatchTraderError as e:
            logger.error("MatchTrader: failed to close position {}: {}", position_id, e)
            return BrokerOrderResult(
                success=False,
                position_id=position_id,
                message=str(e),
                raw_response={"error": str(e)},
            )

    async def close_all_positions(self) -> list[BrokerOrderResult]:
        """Emergency: close ALL open positions."""
        logger.warning("MatchTrader: CLOSING ALL POSITIONS (emergency)")
        positions = await self.get_open_positions()
        results = []

        for pos in positions:
            result = await self.close_position(
                position_id=pos.position_id,
                symbol=pos.symbol,
                side=pos.side,
                volume=pos.volume,
            )
            results.append(result)

        closed_count = sum(1 for r in results if r.success)
        logger.warning("MatchTrader: closed {}/{} positions", closed_count, len(positions))
        return results

    async def modify_position(
        self,
        position_id: str,
        symbol: str,
        side: str,
        volume: float,
        sl: float | None = None,
        tp: float | None = None,
    ) -> BrokerOrderResult:
        """Modify stop loss and/or take profit of an existing position.
        The MatchTrader editPosition endpoint requires the full position context
        (id, instrument, orderSide, volume) even when only updating SL/TP.

        Args:
            position_id: The position ID to modify.
            symbol: Instrument name (broker symbol, e.g. 'EURUSD.').
            side: Original order side ('BUY' or 'SELL').
            volume: Position volume.
            sl: New stop loss price (None = don't change).
            tp: New take profit price (None = don't change).
        """
        await self._ensure_auth()
        body: dict[str, Any] = {
            "id": position_id,
            "instrument": symbol,
            "orderSide": side.upper(),
            "volume": volume,
        }
        if sl is not None:
            body["slPrice"] = sl
        if tp is not None:
            body["tpPrice"] = tp
        logger.info(
            "MatchTrader: modifying position {} (SL={}, TP={})",
            position_id,
            sl,
            tp,
        )

        try:
            response = await self._api_request(
                "POST",
                f"/mtr-api/{self.system_uuid}/position/edit",
                json=body,
            )
            data = response.json()
            logger.debug("MatchTrader: modify_position raw response: {}", data)

            # Validate API response — status must be "OK"
            api_status = data.get("status", "")
            error_msg = data.get("errorMessage", "")
            if api_status != "OK":
                logger.error(
                    "MatchTrader: modify_position API returned non-OK status: "
                    "status={}, errorMessage={}, raw={}",
                    api_status,
                    error_msg,
                    data,
                )
                return BrokerOrderResult(
                    success=False,
                    position_id=position_id,
                    message=f"API returned status={api_status}: {error_msg}",
                    raw_response=data,
                )

            logger.info(
                "MatchTrader: position {} modified successfully (SL={}, TP={})",
                position_id,
                sl,
                tp,
            )
            return BrokerOrderResult(
                success=True,
                position_id=position_id,
                message="Position modified successfully",
                raw_response=data,
            )
        except MatchTraderError as e:
            logger.error("MatchTrader: failed to modify position {}: {}", position_id, e)
            return BrokerOrderResult(
                success=False,
                position_id=position_id,
                message=str(e),
                raw_response={"error": str(e)},
            )

    # ── SL/TP Read-Back Verification ────────────────────────────────────

    async def verify_sl_tp(
        self,
        position_id: str,
        expected_sl: float | None = None,
        expected_tp: float | None = None,
        tolerance: float = 1e-6,
        price_precision: int | None = None,
    ) -> bool:
        """Read back open position and verify SL/TP match expected values.

        Used after modify_position() to confirm the broker actually applied
        the SL/TP change. Returns False if position not found or values mismatch.

        Note: Calls get_open_positions (1 API call). Acceptable for breakeven
        verification since it triggers at most once per position lifetime.

        Args:
            position_id: The position to verify.
            expected_sl: Expected stop loss price (None = don't check).
            expected_tp: Expected take profit price (None = don't check).
            tolerance: Price comparison tolerance (broker may round).
            price_precision: Broker price precision used for normalization.
        """
        try:
            positions = await self.get_open_positions()
            for pos in positions:
                if str(pos.position_id) == position_id:
                    if expected_sl is not None:
                        actual_sl = pos.sl_price or 0.0
                        if price_precision is not None:
                            expected_sl = round(float(expected_sl), price_precision)
                            actual_sl = round(float(actual_sl), price_precision)
                        if abs(actual_sl - expected_sl) > tolerance:
                            logger.warning(
                                "MatchTrader: verify_sl_tp MISMATCH for {} — "
                                "expected SL={}, actual SL={}",
                                position_id,
                                expected_sl,
                                actual_sl,
                            )
                            return False
                    if expected_tp is not None:
                        actual_tp = pos.tp_price or 0.0
                        if price_precision is not None:
                            expected_tp = round(float(expected_tp), price_precision)
                            actual_tp = round(float(actual_tp), price_precision)
                        if abs(actual_tp - expected_tp) > tolerance:
                            logger.warning(
                                "MatchTrader: verify_sl_tp MISMATCH for {} — "
                                "expected TP={}, actual TP={}",
                                position_id,
                                expected_tp,
                                actual_tp,
                            )
                            return False
                    return True
            logger.warning(
                "MatchTrader: verify_sl_tp — position {} not found (may be closed)",
                position_id,
            )
            return False
        except Exception as e:
            logger.error(
                "MatchTrader: verify_sl_tp error for {}: {}",
                position_id,
                e,
            )
            return False

    # HTTP method type alias for curl_cffi compatibility
    HttpMethod = Literal["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "TRACE"]

    # ── HTTP Internals ──────────────────────────────────────────────────

    def _build_headers(self, authenticated: bool = True) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if authenticated and self._tokens:
            headers["Auth-trading-api"] = self._tokens.trading_api_token
        return headers

    @staticmethod
    def _classify_call_type(method: HttpMethod, path: str) -> Literal["read", "write"]:
        """Classify API request as read/write for budget controls.

        MatchTrader budget enforcement in production should focus on mutating
        order endpoints; query endpoints remain unrestricted.
        """
        write_paths = ("/position/open", "/position/close", "/position/edit")
        if method == "POST" and any(wp in path for wp in write_paths):
            return "write"
        return "read"

    async def _raw_request(
        self,
        method: HttpMethod,
        path: str,
        json: dict[str, Any] | None = None,
        authenticated: bool = True,
    ) -> Any:
        """Make a raw HTTP request without retry logic.

        Uses curl_cffi with Chrome TLS impersonation to bypass Cloudflare.
        Returns a response object with .status_code, .text, and .json() attributes.
        """
        if not self._session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        url = f"{self._base_url}{path}"
        headers = self._build_headers(authenticated)

        # curl_cffi uses 'data' for raw body and 'json' kwarg for JSON serialization
        response = await self._session.request(method, url, json=json, headers=headers)
        # Track call volume for observability (read/write split).
        call_type = self._classify_call_type(method, path)
        self._rate_limiter.record(call_type=call_type)

        if response.status_code == 401 and authenticated:
            raise MatchTraderAuthError("Authentication failed (401). Token may have expired.")

        if response.status_code == 429:
            raise MatchTraderRateLimitError("Rate limit exceeded (429).")

        if response.status_code >= 400:
            raise MatchTraderError(f"API error {response.status_code}: {response.text[:500]}")

        return response

    async def _api_request(
        self,
        method: HttpMethod,
        path: str,
        json: dict[str, Any] | None = None,
    ) -> Any:
        """Make an authenticated API request with retry + auto-refresh logic."""
        call_type = self._classify_call_type(method, path)
        if call_type == "write" and not self._rate_limiter.can_proceed():
            raise MatchTraderRateLimitError(
                "Daily WRITE request budget exhausted "
                f"({self._rate_limiter.write_count}/{self._rate_limiter.daily_write_limit} used). "
                "Remaining write requests are reserved for emergencies."
            )

        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                await self._ensure_auth()
                return await self._raw_request(method, path, json=json, authenticated=True)

            except MatchTraderAuthError:
                # Token expired mid-request — refresh and retry
                logger.warning("MatchTrader: auth failed, refreshing token (attempt {})", attempt)
                try:
                    await self.refresh_token()
                except Exception as refresh_err:
                    logger.error("MatchTrader: token refresh failed: {}", refresh_err)
                    # Re-login as last resort
                    await self.login()

            except MatchTraderRateLimitError as e:
                # Rate limited — exponential backoff
                wait = 2**attempt
                logger.warning("MatchTrader: rate limited, waiting {}s (attempt {})", wait, attempt)
                await asyncio.sleep(wait)
                if self._on_retry:
                    self._on_retry()
                last_error = e

            except Exception as e:
                # Network or other error — retry with backoff
                wait = 2**attempt
                logger.warning(
                    "MatchTrader: request error '{}', retrying in {}s (attempt {})",
                    e,
                    wait,
                    attempt,
                )
                await asyncio.sleep(wait)
                if self._on_retry:
                    self._on_retry()
                last_error = e

        raise MatchTraderError(f"Request failed after {self._max_retries} retries: {last_error}")


# ── Exceptions ──────────────────────────────────────────────────────────────


class MatchTraderError(Exception):
    """Base exception for MatchTrader API errors."""


class MatchTraderAuthError(MatchTraderError):
    """Authentication failure (expired/invalid token)."""


class MatchTraderRateLimitError(MatchTraderError):
    """Daily API request limit approached or exceeded."""
