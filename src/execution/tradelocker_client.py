"""
TradeLocker REST API client for broker-neutral execution flows.

This module keeps TradeLocker-specific routes and payload fields isolated
from execution/scheduler logic by normalizing responses into Broker* models.

Usage:
    client = TradeLockerClient(
        api_url="https://demo.tradelocker.com/backend-api",
        email="user@example.com",
        password="secret",
        server="demo",
    )
    async with client:
        await client.login()
        quote = await client.get_quote("EURUSD")
"""

from typing import Any, Literal

import httpx
from loguru import logger

from src.execution.broker_models import (
    BrokerBalanceInfo,
    BrokerClosedPosition,
    BrokerInstrumentInfo,
    BrokerOrderResult,
    BrokerPositionInfo,
    BrokerQuoteInfo,
)


class TradeLockerClient:
    """Async TradeLocker client that exposes the broker protocol surface.

    Usage:
        async with TradeLockerClient(...) as client:
            await client.login()
            balance = await client.get_balance()
    """

    def __init__(
        self,
        api_url: str,
        email: str,
        password: str,
        server: str,
        account_id: str | None = None,
        timeout_seconds: float = 30.0,
        store: Any = None,
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._email = email
        self._password = password
        self._server = server
        self._requested_account_id = account_id
        self._store = store
        self._timeout_seconds = timeout_seconds

        self._client: httpx.AsyncClient | None = None
        self._access_token = ""
        self._refresh_token = ""
        self._account_id = ""
        self._acc_num = ""
        self._symbol_meta: dict[str, dict[str, str]] = {}

    # ── Context Manager ─────────────────────────────────────────────────

    async def __aenter__(self) -> "TradeLockerClient":
        self._client = httpx.AsyncClient(base_url=self._api_url, timeout=self._timeout_seconds)
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def is_authenticated(self) -> bool:
        return bool(self._access_token)

    @property
    def account_id(self) -> str:
        return self._account_id

    @property
    def acc_num(self) -> str:
        return self._acc_num

    # ── Auth ────────────────────────────────────────────────────────────

    async def login(self, *, _quiet: bool = False) -> dict[str, Any]:
        """Authenticate and resolve the active TradeLocker account context."""
        _log = logger.debug if _quiet else logger.info
        _log("TradeLocker: logging in as {}", self._email)

        token_data = await self._request(
            "POST",
            "/auth/jwt/token",
            json={
                "email": self._email,
                "password": self._password,
                "server": self._server,
            },
            authenticated=False,
        )
        self._access_token = str(token_data.get("accessToken", token_data.get("access", "")))
        self._refresh_token = str(token_data.get("refreshToken", token_data.get("refresh", "")))
        if not self._access_token:
            raise RuntimeError("TradeLocker login failed: missing access token.")

        accounts_payload = await self._request("GET", "/auth/jwt/all-accounts")
        accounts = self._extract_list(accounts_payload, ["accounts"])
        if not accounts:
            raise RuntimeError("No TradeLocker account returned for the credentials.")

        selected = self._select_account(accounts)
        if selected is None:
            raise RuntimeError(f"Account {self._requested_account_id} not found.")

        self._account_id = str(selected.get("accountId", selected.get("id", "")))
        self._acc_num = str(
            selected.get(
                "accNum",
                selected.get("accountNum", selected.get("accountNumber", self._account_id)),
            )
        )
        if not self._account_id:
            raise RuntimeError("TradeLocker account selection failed: missing accountId.")
        if not self._acc_num:
            raise RuntimeError("TradeLocker account selection failed: missing accNum.")

        _log(
            "TradeLocker: login successful. account_id={}, acc_num={}",
            self._account_id,
            self._acc_num,
        )
        return {"accessToken": self._access_token, "refreshToken": self._refresh_token}

    async def _ensure_auth(self) -> None:
        if not self.is_authenticated or not self._account_id or not self._acc_num:
            await self.login(_quiet=True)

    def _select_account(self, accounts: list[dict[str, Any]]) -> dict[str, Any] | None:
        if self._requested_account_id:
            for account in accounts:
                account_id = str(account.get("accountId", account.get("id", "")))
                if account_id == self._requested_account_id:
                    return account
            return None
        return accounts[0]

    # ── Read APIs ───────────────────────────────────────────────────────

    async def get_balance(self) -> BrokerBalanceInfo:
        payload = await self._account_request("GET", f"/trade/accounts/{self.account_id}/state")
        state = payload if isinstance(payload, dict) else {}
        return BrokerBalanceInfo(
            balance=self._as_float(state.get("balance")),
            equity=self._as_float(state.get("equity")),
            margin=self._as_float(state.get("margin")),
            free_margin=self._as_float(state.get("freeMargin", state.get("freeMarginValue"))),
            currency=str(state.get("currency", "USD")),
        )

    async def get_effective_instruments(self) -> list[BrokerInstrumentInfo]:
        payload = await self._account_request(
            "GET",
            f"/trade/accounts/{self.account_id}/instruments",
        )
        rows = self._extract_list(payload, ["instruments", "data"])

        instruments: list[BrokerInstrumentInfo] = []
        for row in rows:
            symbol = str(row.get("symbol", row.get("name", "")))
            routes = row.get("routes", {})
            info_route_id = str(row.get("infoRouteId", routes.get("INFO", "")))
            trade_route_id = str(row.get("tradeRouteId", routes.get("TRADE", "")))
            tradable_instrument_id = str(
                row.get("tradableInstrumentId", row.get("id", row.get("instrumentId", "")))
            )

            if symbol:
                self._symbol_meta[symbol.upper()] = {
                    "tradableInstrumentId": tradable_instrument_id,
                    "infoRouteId": info_route_id,
                    "tradeRouteId": trade_route_id,
                }

            instruments.append(
                BrokerInstrumentInfo(
                    symbol=symbol,
                    alias=str(row.get("alias", symbol)),
                    description=str(row.get("description", "")),
                    type=str(row.get("type", "")),
                    base_currency=str(row.get("baseCurrency", "")),
                    quote_currency=str(row.get("quoteCurrency", "")),
                    session_open=bool(row.get("sessionOpen", False)),
                    volume_min=self._as_float(row.get("volumeMin", 0.01)),
                    volume_max=self._as_float(row.get("volumeMax", 50.0)),
                    volume_step=self._as_float(row.get("volumeStep", 0.01)),
                    volume_precision=int(row.get("volumePrecision", 2)),
                    price_precision=int(row.get("pricePrecision", 5)),
                    size_of_one_point=self._as_float(row.get("sizeOfOnePoint", 0.0)),
                    contract_size=self._as_float(row.get("contractSize", 100000)),
                    leverage=self._as_float(row.get("leverage", 0.0)),
                )
            )

        return instruments

    async def get_quote(self, symbol: str) -> BrokerQuoteInfo:
        await self._ensure_auth()
        meta = await self._resolve_symbol_meta(symbol)
        payload = await self._request(
            "GET",
            "/trade/quotes",
            params={
                "tradableInstrumentId": meta["tradableInstrumentId"],
                "routeId": meta["infoRouteId"] or meta["tradeRouteId"],
            },
            authenticated=True,
            account_scoped=True,
        )
        quote = self._extract_quote(payload)
        return BrokerQuoteInfo(
            symbol=str(quote.get("symbol", symbol)),
            bid=self._as_float(quote.get("bid")),
            ask=self._as_float(quote.get("ask")),
            high=self._as_float(quote.get("high")),
            low=self._as_float(quote.get("low")),
            timestamp_ms=self._as_int(quote.get("timestampMs", quote.get("timestamp", 0))),
        )

    async def get_open_positions(self) -> list[BrokerPositionInfo]:
        payload = await self._account_request("GET", f"/trade/accounts/{self.account_id}/positions")
        rows = self._extract_list(payload, ["positions", "data"])

        positions: list[BrokerPositionInfo] = []
        for row in rows:
            positions.append(
                BrokerPositionInfo(
                    position_id=str(row.get("positionId", row.get("id", ""))),
                    symbol=str(row.get("symbol", row.get("instrument", ""))),
                    side=self._normalize_side(str(row.get("side", row.get("direction", "")))),
                    volume=self._as_float(row.get("qty", row.get("volume", 0))),
                    open_price=self._as_float(row.get("openPrice", row.get("avgPrice", 0))),
                    current_price=self._as_float(row.get("currentPrice", row.get("markPrice", 0))),
                    profit=self._as_float(
                        row.get("profit", row.get("unrealizedPnl", row.get("unrealizedPl", 0)))
                    ),
                    sl_price=self._as_optional_float(row.get("stopLoss", row.get("slPrice"))),
                    tp_price=self._as_optional_float(row.get("takeProfit", row.get("tpPrice"))),
                    open_time=str(row.get("openTime", row.get("createdAt", ""))),
                )
            )

        return positions

    async def get_closed_positions(self, from_ts: int, to_ts: int) -> list[BrokerClosedPosition]:
        payload = await self._account_request(
            "GET",
            f"/trade/accounts/{self.account_id}/ordersHistory",
            params={"from": from_ts, "to": to_ts},
        )
        rows = self._extract_list(payload, ["orders", "data"])

        closed: list[BrokerClosedPosition] = []
        for row in rows:
            closed.append(
                BrokerClosedPosition(
                    position_id=str(row.get("positionId", row.get("id", ""))),
                    symbol=str(row.get("symbol", row.get("instrument", ""))),
                    side=self._normalize_side(str(row.get("side", row.get("direction", "")))),
                    volume=self._as_float(row.get("qty", row.get("volume", 0))),
                    open_price=self._as_float(row.get("openPrice", row.get("entryPrice", 0))),
                    close_price=self._as_float(row.get("closePrice", row.get("exitPrice", 0))),
                    profit=self._as_float(row.get("profit", row.get("realizedPnl", 0))),
                    open_time=str(row.get("openTime", row.get("openedAt", ""))),
                    close_time=str(row.get("closeTime", row.get("closedAt", ""))),
                    close_reason=str(row.get("closeReason", row.get("reason", ""))),
                )
            )

        return closed

    # ── Trading APIs ────────────────────────────────────────────────────

    async def open_position(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        volume: float,
        sl: float | None = None,
        tp: float | None = None,
    ) -> BrokerOrderResult:
        try:
            meta = await self._resolve_symbol_meta(symbol)
            body: dict[str, Any] = {
                "qty": volume,
                "routeId": meta["tradeRouteId"],
                "side": side.upper(),
                "validity": "IOC",
                "type": "MARKET",
                "price": 0,
                "tradableInstrumentId": meta["tradableInstrumentId"],
            }
            if sl is not None:
                body["stopLoss"] = sl
            if tp is not None:
                body["takeProfit"] = tp

            payload = await self._account_request(
                "POST",
                f"/trade/accounts/{self.account_id}/orders",
                json=body,
            )
            response_raw = payload if isinstance(payload, dict) else {"data": payload}
            return BrokerOrderResult(
                success=True,
                position_id=str(
                    response_raw.get(
                        "positionId",
                        response_raw.get("orderId", response_raw.get("id", "")),
                    )
                ),
                message="Position opened successfully",
                raw_response=response_raw,
            )
        except (TradeLockerError, RuntimeError, httpx.HTTPError) as exc:
            logger.error(
                "TradeLocker: failed to open position {} {} {}: {}",
                side,
                symbol,
                volume,
                exc,
            )
            return BrokerOrderResult(
                success=False,
                message=str(exc),
                raw_response={"error": str(exc)},
            )

    async def close_position(
        self,
        position_id: str,
        symbol: str,
        side: str,
        volume: float,
    ) -> BrokerOrderResult:
        _ = (symbol, side)
        qty = volume if volume > 0 else 0
        try:
            payload = await self._account_request(
                "DELETE",
                f"/trade/positions/{position_id}",
                params={"qty": qty},
            )
            response_raw = payload if isinstance(payload, dict) else {"data": payload}
            return BrokerOrderResult(
                success=True,
                position_id=position_id,
                message="Position closed successfully",
                raw_response=response_raw,
            )
        except (TradeLockerError, RuntimeError, httpx.HTTPError) as exc:
            logger.error("TradeLocker: failed to close position {}: {}", position_id, exc)
            return BrokerOrderResult(
                success=False,
                position_id=position_id,
                message=str(exc),
                raw_response={"error": str(exc)},
            )

    async def close_all_positions(self) -> list[BrokerOrderResult]:
        positions = await self.get_open_positions()
        results: list[BrokerOrderResult] = []
        for position in positions:
            results.append(
                await self.close_position(
                    position_id=position.position_id,
                    symbol=position.symbol,
                    side=position.side,
                    volume=0.0,
                )
            )
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
        _ = (symbol, side, volume)
        body: dict[str, Any] = {}
        if sl is not None:
            body["stopLoss"] = sl
        if tp is not None:
            body["takeProfit"] = tp

        try:
            payload = await self._account_request(
                "PATCH",
                f"/trade/positions/{position_id}",
                json=body,
            )
            response_raw = payload if isinstance(payload, dict) else {"data": payload}
            return BrokerOrderResult(
                success=True,
                position_id=position_id,
                message="Position modified successfully",
                raw_response=response_raw,
            )
        except (TradeLockerError, RuntimeError, httpx.HTTPError) as exc:
            logger.error("TradeLocker: failed to modify position {}: {}", position_id, exc)
            return BrokerOrderResult(
                success=False,
                position_id=position_id,
                message=str(exc),
                raw_response={"error": str(exc)},
            )

    async def verify_sl_tp(
        self,
        position_id: str,
        expected_sl: float | None = None,
        expected_tp: float | None = None,
        tolerance: float = 1e-6,
        price_precision: int | None = None,
    ) -> bool:
        """Read back position state and verify SL/TP values."""
        try:
            positions = await self.get_open_positions()
        except (TradeLockerError, RuntimeError, httpx.HTTPError):
            return False

        for position in positions:
            if str(position.position_id) != position_id:
                continue

            if expected_sl is not None:
                actual_sl = position.sl_price if position.sl_price is not None else 0.0
                check_sl = expected_sl
                if price_precision is not None:
                    actual_sl = round(actual_sl, price_precision)
                    check_sl = round(check_sl, price_precision)
                if abs(actual_sl - check_sl) > tolerance:
                    return False

            if expected_tp is not None:
                actual_tp = position.tp_price if position.tp_price is not None else 0.0
                check_tp = expected_tp
                if price_precision is not None:
                    actual_tp = round(actual_tp, price_precision)
                    check_tp = round(check_tp, price_precision)
                if abs(actual_tp - check_tp) > tolerance:
                    return False

            return True

        return False

    # ── HTTP Helpers ────────────────────────────────────────────────────

    async def _account_request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        await self._ensure_auth()
        return await self._request(
            method,
            path,
            params=params,
            json=json,
            authenticated=True,
            account_scoped=True,
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        authenticated: bool = True,
        account_scoped: bool = False,
    ) -> Any:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._api_url, timeout=self._timeout_seconds)

        headers = self._build_headers(authenticated=authenticated, account_scoped=account_scoped)
        response = await self._client.request(
            method=method,
            url=path,
            params=params,
            json=json,
            headers=headers,
        )
        if response.status_code == 401:
            raise TradeLockerAuthError("TradeLocker authentication failed (401).")
        if response.status_code >= 400:
            raise TradeLockerError(
                f"TradeLocker API error {response.status_code}: {response.text[:500]}"
            )

        try:
            return response.json()
        except ValueError:
            return {}

    def _build_headers(self, *, authenticated: bool, account_scoped: bool) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if authenticated and self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"
        if account_scoped and self._acc_num:
            headers["accNum"] = self._acc_num
        return headers

    # ── Parsing Helpers ─────────────────────────────────────────────────

    async def _resolve_symbol_meta(self, symbol: str) -> dict[str, str]:
        symbol_key = symbol.upper()
        if symbol_key not in self._symbol_meta:
            await self.get_effective_instruments()
        if symbol_key not in self._symbol_meta:
            raise TradeLockerError(f"TradeLocker instrument not found: {symbol}")
        return self._symbol_meta[symbol_key]

    @staticmethod
    def _extract_list(payload: Any, candidate_keys: list[str]) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for key in candidate_keys:
                value = payload.get(key)
                if isinstance(value, list):
                    return [row for row in value if isinstance(row, dict)]
            if isinstance(payload.get("d"), list):
                return [row for row in payload["d"] if isinstance(row, dict)]
        return []

    @staticmethod
    def _extract_quote(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            if "quotes" in payload and isinstance(payload["quotes"], list) and payload["quotes"]:
                first = payload["quotes"][0]
                return first if isinstance(first, dict) else {}
            return payload
        if isinstance(payload, list) and payload:
            first = payload[0]
            return first if isinstance(first, dict) else {}
        return {}

    @staticmethod
    def _normalize_side(value: str) -> str:
        side = value.upper()
        if side in {"BUY", "SELL"}:
            return side
        return value

    @staticmethod
    def _as_float(value: Any) -> float:
        if value is None or value == "":
            return 0.0
        return float(value)

    @staticmethod
    def _as_int(value: Any) -> int:
        if value is None or value == "":
            return 0
        return int(float(value))

    @staticmethod
    def _as_optional_float(value: Any) -> float | None:
        if value is None or value == "":
            return None
        return float(value)


class TradeLockerError(Exception):
    """Base exception for TradeLocker API failures."""


class TradeLockerAuthError(TradeLockerError):
    """Authentication failure from TradeLocker."""
