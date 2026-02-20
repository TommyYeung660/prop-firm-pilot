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
from datetime import datetime, timezone
from typing import Any, Literal

from curl_cffi.requests import AsyncSession
from loguru import logger
from pydantic import AliasChoices, BaseModel, Field

# ── Response Models ─────────────────────────────────────────────────────────


class AuthTokens(BaseModel):
    """Tokens returned from login / refresh."""

    trading_api_token: str = Field(description="JWT for Auth-trading-api header")
    refresh_token: str = Field(description="Token used to refresh the JWT")
    system_uuid: str = Field(description="Account system UUID for API paths")


class BalanceInfo(BaseModel):
    """Account balance snapshot from MatchTrader."""

    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    free_margin: float = Field(default=0.0, alias="freeMargin")
    currency: str = "USD"

    model_config = {"populate_by_name": True}


class PositionInfo(BaseModel):
    """Open position details."""

    position_id: str = Field(
        validation_alias=AliasChoices("positionId", "id"),
        serialization_alias="positionId",
    )
    symbol: str = ""
    side: str = ""
    volume: float = 0.0
    open_price: float = Field(default=0.0, alias="openPrice")
    current_price: float = Field(default=0.0, alias="currentPrice")
    profit: float = 0.0
    sl_price: float | None = Field(default=None, alias="slPrice")
    tp_price: float | None = Field(default=None, alias="tpPrice")
    open_time: str = Field(default="", alias="openTime")

    model_config = {"populate_by_name": True}


class OrderResult(BaseModel):
    """Result from opening/closing/modifying a position."""

    success: bool = False
    position_id: str = ""
    message: str = ""
    raw_response: dict[str, Any] = {}


class ClosedPosition(BaseModel):
    """Historical closed position."""

    position_id: str = Field(default="", alias="positionId")
    symbol: str = ""
    side: str = ""
    volume: float = 0.0
    open_price: float = Field(default=0.0, alias="openPrice")
    close_price: float = Field(default=0.0, alias="closePrice")
    profit: float = 0.0
    open_time: str = Field(default="", alias="openTime")
    close_time: str = Field(default="", alias="closeTime")

    model_config = {"populate_by_name": True}


class TradingHours(BaseModel):
    """Single trading session window for an instrument."""

    day_number: int = Field(alias="dayNumber", description="Day of week (0=Sunday, 1=Monday...)")
    open_hours: int = Field(default=0, alias="openHours")
    open_minutes: int = Field(default=0, alias="openMinutes")
    open_seconds: int = Field(default=0, alias="openSeconds")
    close_hours: int = Field(default=0, alias="closeHours")
    close_minutes: int = Field(default=0, alias="closeMinutes")
    close_seconds: int = Field(default=0, alias="closeSeconds")

    model_config = {"populate_by_name": True}


class InstrumentInfo(BaseModel):
    """Effective instrument details from MatchTrader.

    Contains trading parameters, session hours, and contract specifications
    for instruments available on the account.

    Usage:
        instruments = await client.get_effective_instruments()
        eurusd = next(i for i in instruments if i.symbol == "EURUSD.")
        print(f"Min lot: {eurusd.volume_min}, Spread markup: {eurusd.ask_markup}")
    """

    symbol: str = ""
    alias: str = ""
    description: str = ""
    type: str = ""
    base_currency: str = Field(default="", alias="baseCurrency")
    quote_currency: str = Field(default="", alias="quoteCurrency")

    # Session & availability
    session_open: bool = Field(default=False, alias="sessionOpen")
    trading_hours: list[TradingHours] = Field(default_factory=list, alias="tradingHours")

    # Volume constraints
    volume_min: float = Field(default=0.01, alias="volumeMin")
    volume_max: float = Field(default=50.0, alias="volumeMax")
    volume_step: float = Field(default=0.01, alias="volumeStep")
    volume_precision: int = Field(default=2, alias="volumePrecision")

    # Pricing
    price_precision: int = Field(default=5, alias="pricePrecision")
    size_of_one_point: float = Field(default=0.0, alias="sizeOfOnePoint")
    contract_size: float = Field(default=100000, alias="contractSize")
    ask_markup: float = Field(default=0.0, alias="askMarkup")
    bid_markup: float = Field(default=0.0, alias="bidMarkup")

    # Leverage & margin
    leverage: float = 0.0
    fixed_leverage: bool = Field(default=False, alias="fixedLeverage")
    multiplier: float = 0.0
    multiplier_currency: str = Field(default="", alias="multiplierCurrency")
    divider: int = 1

    # Swaps
    swap_type: str = Field(default="PIPS", alias="swapType")
    swap_buy: float = Field(default=0.0, alias="swapBuy")
    swap_sell: float = Field(default=0.0, alias="swapSell")

    # Stops
    freeze_level: int = Field(default=0, alias="freezeLevel")
    stops_level: int = Field(default=0, alias="stopsLevel")

    # Termination
    termination_type: str = Field(default="UNDEFINED", alias="terminationType")
    termination_date: str | None = Field(default=None, alias="terminationDate")
    termination_date_iso: str | None = Field(default=None, alias="terminationDateIso")

    # Tags
    tags: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


# ── Rate Limiter ────────────────────────────────────────────────────────────


class RateLimiter:
    """Simple daily rate limiter for MatchTrader API (2000 req/day)."""

    def __init__(self, daily_limit: int = 2000):
        self._daily_limit = daily_limit
        self._count = 0
        self._reset_date = datetime.now(timezone.utc).date()

    def _maybe_reset(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self._reset_date:
            self._count = 0
            self._reset_date = today

    def record(self) -> None:
        self._maybe_reset()
        self._count += 1

    @property
    def remaining(self) -> int:
        self._maybe_reset()
        return max(0, self._daily_limit - self._count)

    @property
    def count(self) -> int:
        self._maybe_reset()
        return self._count

    def can_proceed(self, reserve: int = 50) -> bool:
        """Check if we can make a request, keeping a reserve for emergencies."""
        return self.remaining > reserve


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
    ):
        self._base_url = base_url.rstrip("/")
        self._email = email
        self._password = password
        self._broker_id = broker_id
        self._account_id = account_id
        self._max_retries = max_retries

        self._tokens: AuthTokens | None = None
        self._last_auth_time: float = 0.0
        self._rate_limiter = RateLimiter(daily_request_limit)
        self._session: AsyncSession[Any] | None = None

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

    async def login(self) -> AuthTokens:
        """Authenticate with MatchTrader and obtain JWT tokens.

        If account_id was provided at construction, selects that specific
        trading account. Otherwise falls back to the first account returned.
        """
        logger.info("MatchTrader: logging in as {}", self._email)

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
        logger.info(
            "MatchTrader: login successful. account={}, systemUUID={}, total_accounts={}",
            selected_id,
            system_uuid,
            len(accounts),
        )
        return self._tokens

    async def refresh_token(self) -> None:
        """Refresh JWT before it expires (15 min lifetime)."""
        if not self._tokens:
            raise RuntimeError("Cannot refresh: not authenticated.")

        logger.debug("MatchTrader: refreshing JWT token")

        response = await self._raw_request(
            "POST",
            "/refresh-token",
            json={"token": self._tokens.refresh_token},
            authenticated=False,
        )

        data = response.json()
        self._tokens.trading_api_token = data.get("tradingApiToken", self._tokens.trading_api_token)
        if "token" in data:
            self._tokens.refresh_token = data["token"]

        self._last_auth_time = time.monotonic()
        logger.debug("MatchTrader: token refreshed successfully")

    async def _ensure_auth(self) -> None:
        """Auto-refresh token if it's about to expire."""
        if not self._tokens:
            raise RuntimeError("Not authenticated. Call login() first.")

        elapsed = time.monotonic() - self._last_auth_time
        if elapsed >= self.TOKEN_REFRESH_SECONDS:
            await self.refresh_token()

    # ── Account Info ────────────────────────────────────────────────────

    async def get_balance(self) -> BalanceInfo:
        """Get current account balance, equity, margin."""
        await self._ensure_auth()
        response = await self._api_request("GET", f"/mtr-api/{self.system_uuid}/balance")
        return BalanceInfo(**response.json())

    async def get_account_details(self) -> dict[str, Any]:
        """Get account details (leverage, offer name, etc.)."""
        await self._ensure_auth()
        response = await self._api_request("GET", f"/mtr-api/{self.system_uuid}/account-details")
        return response.json()

    async def get_effective_instruments(self) -> list[InstrumentInfo]:
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
        instruments = [InstrumentInfo(**item) for item in instruments_raw]
        logger.info("MatchTrader: loaded {} effective instruments", len(instruments))
        return instruments

    # ── Position Queries ────────────────────────────────────────────────

    async def get_open_positions(self) -> list[PositionInfo]:
        """Get all currently open positions."""
        await self._ensure_auth()
        response = await self._api_request("GET", f"/mtr-api/{self.system_uuid}/open-positions")
        data = response.json()

        # API may return list directly or wrapped in a key
        positions_raw = data if isinstance(data, list) else data.get("positions", [])
        return [PositionInfo(**p) for p in positions_raw]

    async def get_closed_positions(
        self,
        from_ts: int,
        to_ts: int,
    ) -> list[ClosedPosition]:
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
        return [ClosedPosition(**p) for p in positions_raw]

    # ── Trading Operations ──────────────────────────────────────────────

    async def open_position(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        volume: float,
        sl: float | None = None,
        tp: float | None = None,
    ) -> OrderResult:
        """Open a new trading position.

        Args:
            symbol: Instrument name (e.g. "EURUSD").
            side: "BUY" or "SELL".
            volume: Lot size (e.g. 0.1).
            sl: Stop loss price (optional).
            tp: Take profit price (optional).
        """
        await self._ensure_auth()

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

        try:
            response = await self._api_request(
                "POST",
                f"/mtr-api/{self.system_uuid}/position/open",
                json=body,
            )
            data = response.json()
            return OrderResult(
                success=True,
                position_id=str(data.get("orderId", data.get("positionId", data.get("id", "")))),
                message="Position opened successfully",
                raw_response=data,
            )
        except MatchTraderError as e:
            logger.error("MatchTrader: failed to open position: {}", e)
            return OrderResult(
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
    ) -> OrderResult:
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
            return OrderResult(
                success=True,
                position_id=position_id,
                message="Position closed successfully",
                raw_response=data,
            )
        except MatchTraderError as e:
            logger.error("MatchTrader: failed to close position {}: {}", position_id, e)
            return OrderResult(
                success=False,
                position_id=position_id,
                message=str(e),
                raw_response={"error": str(e)},
            )

    async def close_all_positions(self) -> list[OrderResult]:
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
        sl: float | None = None,
        tp: float | None = None,
    ) -> OrderResult:
        """Modify stop loss and/or take profit of an existing position.

        Args:
            position_id: The position ID to modify.
            sl: New stop loss price (None = don't change).
            tp: New take profit price (None = don't change).
        """
        await self._ensure_auth()

        body: dict[str, Any] = {"positionId": position_id}
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
            return OrderResult(
                success=True,
                position_id=position_id,
                message="Position modified successfully",
                raw_response=data,
            )
        except MatchTraderError as e:
            logger.error("MatchTrader: failed to modify position {}: {}", position_id, e)
            return OrderResult(
                success=False,
                position_id=position_id,
                message=str(e),
                raw_response={"error": str(e)},
            )

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
        self._rate_limiter.record()

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
        if not self._rate_limiter.can_proceed():
            raise MatchTraderRateLimitError(
                f"Daily API request budget exhausted ({self._rate_limiter.count} used). "
                "Remaining requests reserved for emergencies."
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
                last_error = e

        raise MatchTraderError(f"Request failed after {self._max_retries} retries: {last_error}")


# ── Exceptions ──────────────────────────────────────────────────────────────


class MatchTraderError(Exception):
    """Base exception for MatchTrader API errors."""


class MatchTraderAuthError(MatchTraderError):
    """Authentication failure (expired/invalid token)."""


class MatchTraderRateLimitError(MatchTraderError):
    """Daily API request limit approached or exceeded."""
