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

import asyncio
import time
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

POSITION_COLUMNS = (
    "id",
    "tradableInstrumentId",
    "routeId",
    "side",
    "qty",
    "avgPrice",
    "stopLossId",
    "takeProfitId",
    "openDate",
    "unrealizedPl",
    "strategyId",
)

ORDERS_HISTORY_COLUMNS = (
    "id",
    "tradableInstrumentId",
    "routeId",
    "qty",
    "side",
    "type",
    "status",
    "filledQty",
    "avgPrice",
    "price",
    "stopPrice",
    "validity",
    "expireDate",
    "createdDate",
    "lastModified",
    "isOpen",
    "positionId",
    "stopLoss",
    "stopLossType",
    "takeProfit",
    "takeProfitType",
    "strategyId",
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
        self._instrument_id_to_symbol: dict[str, str] = {}

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
            allow_reauth=False,
        )
        self._access_token = str(token_data.get("accessToken", token_data.get("access", "")))
        self._refresh_token = str(token_data.get("refreshToken", token_data.get("refresh", "")))
        if not self._access_token:
            raise RuntimeError("TradeLocker login failed: missing access token.")

        accounts_payload = await self._request(
            "GET",
            "/auth/jwt/all-accounts",
            allow_reauth=False,
        )
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
        payload = await self._account_request("GET", "/trade/accounts/{account_id}/state")
        state = self._extract_object(payload)
        details = state.get("accountDetailsData")
        if isinstance(details, list):
            return BrokerBalanceInfo(
                balance=self._value_at(details, 0),
                equity=self._value_at(details, 1),
                margin=self._value_at(details, 9),
                free_margin=self._value_at(details, 2),
                currency=str(state.get("currency", "USD")),
            )
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
            "/trade/accounts/{account_id}/instruments",
        )
        rows = self._extract_list(payload, ["instruments", "data"])

        instruments: list[BrokerInstrumentInfo] = []
        for row in rows:
            raw_symbol = str(row.get("symbol", row.get("name", "")))
            symbol = self._normalize_symbol(raw_symbol)
            routes = self._extract_routes(row.get("routes", {}))
            info_route_id = str(row.get("infoRouteId", routes.get("INFO", "")))
            trade_route_id = str(row.get("tradeRouteId", routes.get("TRADE", "")))
            tradable_instrument_id = str(
                row.get("tradableInstrumentId", row.get("id", row.get("instrumentId", "")))
            )

            if symbol:
                meta = {
                    "tradableInstrumentId": tradable_instrument_id,
                    "infoRouteId": info_route_id,
                    "tradeRouteId": trade_route_id,
                }
                self._symbol_meta[symbol.upper()] = meta
                if raw_symbol:
                    self._symbol_meta[raw_symbol.upper()] = meta
                if tradable_instrument_id:
                    self._instrument_id_to_symbol[tradable_instrument_id] = symbol

            instruments.append(
                BrokerInstrumentInfo(
                    symbol=symbol,
                    alias=str(row.get("alias", raw_symbol or symbol)),
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
            symbol=self._normalize_symbol(str(quote.get("symbol", symbol))),
            bid=self._as_float(quote.get("bid", quote.get("bp"))),
            ask=self._as_float(quote.get("ask", quote.get("ap"))),
            high=self._as_float(quote.get("high", quote.get("hp"))),
            low=self._as_float(quote.get("low", quote.get("lp"))),
            timestamp_ms=self._as_int(
                quote.get("timestampMs", quote.get("timestamp", quote.get("t", 0)))
            ),
        )

    async def get_open_positions(self) -> list[BrokerPositionInfo]:
        payload = await self._account_request("GET", "/trade/accounts/{account_id}/positions")
        rows = self._extract_rows(payload, ["positions", "data"])
        has_array_rows = any(isinstance(row, (list, tuple)) for row in rows)
        if rows and has_array_rows and not self._instrument_id_to_symbol:
            await self.get_effective_instruments()

        raw_positions = [self._row_as_dict(row, POSITION_COLUMNS) for row in rows]
        quote_by_symbol = await self._fetch_live_quotes_by_symbol(
            [
                self._resolve_position_symbol(values)
                for values in raw_positions
                if self._resolve_position_symbol(values)
            ]
        )
        open_orders = await self._fetch_open_orders_safe()
        order_by_id, orders_by_position = self._index_open_orders(open_orders)

        positions: list[BrokerPositionInfo] = []
        for values in raw_positions:
            symbol = self._resolve_position_symbol(values)
            side = self._normalize_side(str(values.get("side", values.get("direction", ""))))
            current_price = self._resolve_live_current_price(
                values=values,
                symbol=symbol,
                side=side,
                quote_by_symbol=quote_by_symbol,
            )
            profit = values.get(
                "profit",
                values.get("unrealizedPnl", values.get("unrealizedPl", 0)),
            )
            open_time = values.get(
                "openTime",
                values.get("createdAt", values.get("openDate", "")),
            )
            sl_price, tp_price = self._resolve_position_protective_prices(
                values=values,
                order_by_id=order_by_id,
                orders_by_position=orders_by_position,
            )
            positions.append(
                BrokerPositionInfo(
                    position_id=str(values.get("positionId", values.get("id", ""))),
                    symbol=symbol,
                    side=side,
                    volume=self._as_float(values.get("qty", values.get("volume", 0))),
                    open_price=self._as_float(values.get("openPrice", values.get("avgPrice", 0))),
                    current_price=self._as_float(current_price),
                    profit=self._as_float(profit),
                    sl_price=sl_price,
                    tp_price=tp_price,
                    open_time=str(open_time),
                )
            )

        return positions

    async def get_closed_positions(self, from_ts: int, to_ts: int) -> list[BrokerClosedPosition]:
        payload = await self._account_request(
            "GET",
            "/trade/accounts/{account_id}/ordersHistory",
            params={"from": from_ts, "to": to_ts},
        )
        rows = self._extract_rows(payload, ["ordersHistory", "orders", "data"])
        has_array_rows = any(isinstance(row, (list, tuple)) for row in rows)
        if rows and has_array_rows and not self._instrument_id_to_symbol:
            await self.get_effective_instruments()

        closed: list[BrokerClosedPosition] = []
        for row in rows:
            values = self._row_as_dict(row, ORDERS_HISTORY_COLUMNS)
            volume = values.get("filledQty", values.get("qty", values.get("volume", 0)))
            open_price = values.get(
                "openPrice",
                values.get("entryPrice", values.get("avgPrice", 0)),
            )
            close_price = values.get(
                "closePrice",
                values.get("exitPrice", values.get("price", values.get("avgPrice", 0))),
            )
            open_time = values.get(
                "openTime",
                values.get("openedAt", values.get("createdDate", "")),
            )
            close_time = values.get(
                "closeTime",
                values.get("closedAt", values.get("lastModified", "")),
            )
            close_reason = values.get(
                "closeReason",
                values.get("reason", values.get("status", "")),
            )
            closed.append(
                BrokerClosedPosition(
                    position_id=str(values.get("positionId", values.get("id", ""))),
                    symbol=self._resolve_position_symbol(values),
                    side=self._normalize_side(str(values.get("side", values.get("direction", "")))),
                    volume=self._as_float(volume),
                    open_price=self._as_float(open_price),
                    close_price=self._as_float(close_price),
                    profit=self._as_float(values.get("profit", values.get("realizedPnl", 0))),
                    open_time=str(open_time),
                    close_time=str(close_time),
                    close_reason=str(close_reason),
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
                "/trade/accounts/{account_id}/orders",
                json=body,
            )
            response_raw = self._extract_object(payload)
            if not response_raw:
                response_raw = payload if isinstance(payload, dict) else {"data": payload}
            response_raw = await self._enrich_order_response(
                response_raw=response_raw,
                symbol=symbol,
                side=side,
                volume=volume,
                tradable_instrument_id=meta["tradableInstrumentId"],
            )
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
        resolved_path = path.format(account_id=self.account_id)
        return await self._request(
            method,
            resolved_path,
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
        allow_reauth: bool = True,
    ) -> Any:
        response = await self._send_request(
            method,
            path,
            params=params,
            json=json,
            authenticated=authenticated,
            account_scoped=account_scoped,
        )
        if response.status_code == 401 and authenticated and allow_reauth:
            logger.warning(
                "TradeLocker: auth failed for {} {}, re-logging in and retrying once",
                method,
                path,
            )
            await self.login(_quiet=True)
            response = await self._send_request(
                method,
                path,
                params=params,
                json=json,
                authenticated=authenticated,
                account_scoped=account_scoped,
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

    async def _send_request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        authenticated: bool = True,
        account_scoped: bool = False,
    ) -> httpx.Response:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._api_url, timeout=self._timeout_seconds)

        headers = self._build_headers(authenticated=authenticated, account_scoped=account_scoped)
        return await self._client.request(
            method=method,
            url=path,
            params=params,
            json=json,
            headers=headers,
        )

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
            if isinstance(payload.get("d"), dict):
                for key in candidate_keys:
                    value = payload["d"].get(key)
                    if isinstance(value, list):
                        return [row for row in value if isinstance(row, dict)]
        return []

    @staticmethod
    def _extract_rows(payload: Any, candidate_keys: list[str]) -> list[Any]:
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, (dict, list, tuple))]
        if isinstance(payload, dict):
            for key in candidate_keys:
                value = payload.get(key)
                if isinstance(value, list):
                    return [row for row in value if isinstance(row, (dict, list, tuple))]
            if isinstance(payload.get("d"), dict):
                for key in candidate_keys:
                    value = payload["d"].get(key)
                    if isinstance(value, list):
                        return [row for row in value if isinstance(row, (dict, list, tuple))]
        return []

    @staticmethod
    def _extract_quote(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            if "quotes" in payload and isinstance(payload["quotes"], list) and payload["quotes"]:
                first = payload["quotes"][0]
                return first if isinstance(first, dict) else {}
            if isinstance(payload.get("d"), dict):
                return payload["d"]
            return payload
        if isinstance(payload, list) and payload:
            first = payload[0]
            return first if isinstance(first, dict) else {}
        return {}

    @staticmethod
    def _extract_object(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            if isinstance(payload.get("d"), dict):
                return payload["d"]
            return payload
        return {}

    @staticmethod
    def _row_as_dict(row: Any, columns: tuple[str, ...]) -> dict[str, Any]:
        if isinstance(row, dict):
            return row
        if isinstance(row, (list, tuple)):
            return {
                column: row[index]
                for index, column in enumerate(columns)
                if index < len(row)
            }
        return {}

    @staticmethod
    def _extract_routes(routes: Any) -> dict[str, Any]:
        if isinstance(routes, dict):
            return routes
        if isinstance(routes, list):
            resolved: dict[str, Any] = {}
            for route in routes:
                if not isinstance(route, dict):
                    continue
                route_type = str(route.get("type", "")).upper()
                if route_type:
                    resolved[route_type] = route.get("id", "")
            return resolved
        return {}

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return symbol.strip().rstrip(".+")

    @staticmethod
    def _normalize_side(value: str) -> str:
        side = value.upper()
        if side in {"BUY", "SELL"}:
            return side
        return value

    def _resolve_position_symbol(self, values: dict[str, Any]) -> str:
        symbol = str(values.get("symbol", values.get("instrument", "")))
        if symbol:
            return self._normalize_symbol(symbol)

        instrument_id = str(
            values.get("tradableInstrumentId", values.get("instrumentId", ""))
        ).strip()
        if instrument_id:
            return self._instrument_id_to_symbol.get(instrument_id, instrument_id)
        return ""

    async def _fetch_live_quotes_by_symbol(
        self,
        symbols: list[str],
    ) -> dict[str, BrokerQuoteInfo]:
        """Fetch one live quote per symbol and degrade safely on failures."""
        quotes: dict[str, BrokerQuoteInfo] = {}
        seen: set[str] = set()
        for symbol in symbols:
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            try:
                quotes[symbol] = await self.get_quote(symbol)
            except (TradeLockerError, RuntimeError, httpx.HTTPError) as exc:
                logger.warning("TradeLocker: quote enrichment failed for {}: {}", symbol, exc)
            except Exception as exc:
                logger.warning("TradeLocker: quote enrichment failed for {}: {}", symbol, exc)
        return quotes

    async def _fetch_open_orders_safe(self) -> list[dict[str, Any]]:
        """Fetch open orders for protective-price enrichment without failing positions read."""
        fetcher = getattr(self, "_fetch_open_orders", None)
        if not callable(fetcher):
            return []
        try:
            return await fetcher()
        except (TradeLockerError, RuntimeError, httpx.HTTPError) as exc:
            logger.warning("TradeLocker: open-order enrichment failed: {}", exc)
            return []
        except Exception as exc:
            logger.warning("TradeLocker: open-order enrichment failed: {}", exc)
            return []

    async def _fetch_open_orders(self) -> list[dict[str, Any]]:
        """Fetch active account orders for protective SL/TP enrichment."""
        payload = await self._account_request(
            "GET",
            "/trade/accounts/{account_id}/orders",
        )
        rows = self._extract_rows(payload, ["orders", "data"])
        return [self._row_as_dict(row, ORDERS_HISTORY_COLUMNS) for row in rows]

    @staticmethod
    def _index_open_orders(
        orders: list[dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        """Index open orders by order id and position id."""
        order_by_id: dict[str, dict[str, Any]] = {}
        orders_by_position: dict[str, list[dict[str, Any]]] = {}
        for order in orders:
            order_id = str(order.get("id", "")).strip()
            if order_id:
                order_by_id[order_id] = order
            position_id = str(order.get("positionId", "")).strip()
            if position_id:
                orders_by_position.setdefault(position_id, []).append(order)
        return order_by_id, orders_by_position

    def _resolve_live_current_price(
        self,
        *,
        values: dict[str, Any],
        symbol: str,
        side: str,
        quote_by_symbol: dict[str, BrokerQuoteInfo],
    ) -> float:
        """Resolve the live executable price for a position."""
        raw_current_price = self._as_float(
            values.get("currentPrice", values.get("markPrice", values.get("avgPrice", 0)))
        )
        quote = quote_by_symbol.get(symbol)
        if quote is None:
            return raw_current_price
        if side == "BUY" and quote.bid > 0:
            return quote.bid
        if side == "SELL" and quote.ask > 0:
            return quote.ask
        return raw_current_price

    def _resolve_position_protective_prices(
        self,
        *,
        values: dict[str, Any],
        order_by_id: dict[str, dict[str, Any]],
        orders_by_position: dict[str, list[dict[str, Any]]],
    ) -> tuple[float | None, float | None]:
        """Resolve SL/TP from inline payload first, then linked protective orders."""
        sl_price = self._as_optional_float(values.get("stopLoss", values.get("slPrice")))
        tp_price = self._as_optional_float(values.get("takeProfit", values.get("tpPrice")))

        position_id = str(values.get("positionId", values.get("id", ""))).strip()
        if sl_price is None:
            stop_loss_id = str(values.get("stopLossId", "")).strip()
            sl_price = self._extract_protective_price(
                order_id=stop_loss_id,
                order_kind="sl",
                order_by_id=order_by_id,
                orders_by_position=orders_by_position,
                position_id=position_id,
            )
        if tp_price is None:
            take_profit_id = str(values.get("takeProfitId", "")).strip()
            tp_price = self._extract_protective_price(
                order_id=take_profit_id,
                order_kind="tp",
                order_by_id=order_by_id,
                orders_by_position=orders_by_position,
                position_id=position_id,
            )
        return sl_price, tp_price

    def _extract_protective_price(
        self,
        *,
        order_id: str,
        order_kind: Literal["sl", "tp"],
        order_by_id: dict[str, dict[str, Any]],
        orders_by_position: dict[str, list[dict[str, Any]]],
        position_id: str,
    ) -> float | None:
        """Resolve a protective price from linked order ids or position-scoped orders."""
        if order_id and order_id in order_by_id:
            return self._extract_price_from_order(order_by_id[order_id], order_kind=order_kind)

        for order in orders_by_position.get(position_id, []):
            candidate = self._extract_price_from_order(order, order_kind=order_kind)
            if candidate is not None:
                return candidate
        return None

    def _extract_price_from_order(
        self,
        order: dict[str, Any],
        *,
        order_kind: Literal["sl", "tp"],
    ) -> float | None:
        """Extract a stop-loss or take-profit price from one open-order payload."""
        order_type = str(order.get("type", order.get("orderType", ""))).strip().lower()
        if order_kind == "sl":
            if order_type in {"stop", "stop_loss", "sl"}:
                return self._as_optional_float(order.get("stopPrice", order.get("price")))
            return self._as_optional_float(order.get("stopLoss"))

        if order_type in {"limit", "take_profit", "tp"}:
            return self._as_optional_float(order.get("price", order.get("stopPrice")))
        return self._as_optional_float(order.get("takeProfit"))

    async def _enrich_order_response(
        self,
        *,
        response_raw: dict[str, Any],
        symbol: str,
        side: Literal["BUY", "SELL"],
        volume: float,
        tradable_instrument_id: str,
    ) -> dict[str, Any]:
        position_id = str(
            response_raw.get(
                "positionId",
                response_raw.get("position_id", ""),
            )
        ).strip()
        if position_id:
            return response_raw

        recovered = await self._recover_recent_order(
            existing_order_id=str(
                response_raw.get(
                    "orderId",
                    response_raw.get("id", response_raw.get("order_id", "")),
                )
            ),
            symbol=symbol,
            side=side,
            volume=volume,
            tradable_instrument_id=tradable_instrument_id,
        )
        if not recovered:
            return response_raw

        merged = dict(response_raw)
        merged.update(recovered)
        return merged

    async def _recover_recent_order(
        self,
        *,
        existing_order_id: str,
        symbol: str,
        side: Literal["BUY", "SELL"],
        volume: float,
        tradable_instrument_id: str,
    ) -> dict[str, Any]:
        now_ms = int(time.time() * 1000)
        tolerance = 1e-9

        for attempt in range(3):
            payload = await self._account_request(
                "GET",
                "/trade/accounts/{account_id}/ordersHistory",
                params={
                    "from": now_ms - 300000,
                    "to": now_ms + 60000,
                },
            )
            rows = self._extract_rows(payload, ["ordersHistory", "orders", "data"])
            matches: list[tuple[int, dict[str, Any]]] = []
            for row in rows:
                values = self._row_as_dict(row, ORDERS_HISTORY_COLUMNS)
                if not values:
                    continue
                row_order_id = str(values.get("id", "")).strip()
                if existing_order_id and row_order_id == existing_order_id:
                    return {
                        "orderId": row_order_id,
                        "positionId": str(values.get("positionId", "")).strip(),
                        "openPrice": self._as_float(values.get("avgPrice", values.get("price", 0))),
                        "price": self._as_float(values.get("price", values.get("avgPrice", 0))),
                    }

                row_instrument_id = str(values.get("tradableInstrumentId", "")).strip()
                if tradable_instrument_id and row_instrument_id != tradable_instrument_id:
                    continue
                if self._normalize_side(str(values.get("side", ""))) != side:
                    continue
                row_volume = self._as_float(values.get("filledQty", values.get("qty", 0)))
                if abs(row_volume - volume) > tolerance:
                    continue
                status = str(values.get("status", "")).strip().lower()
                if status and status != "filled":
                    continue
                modified = self._as_int(values.get("lastModified", values.get("createdDate", 0)))
                matches.append((modified, values))

            if matches:
                _, latest = max(matches, key=lambda item: item[0])
                return {
                    "orderId": str(latest.get("id", "")).strip(),
                    "positionId": str(latest.get("positionId", "")).strip(),
                    "openPrice": self._as_float(latest.get("avgPrice", latest.get("price", 0))),
                    "price": self._as_float(latest.get("price", latest.get("avgPrice", 0))),
                }

            if attempt < 2:
                logger.warning(
                    "TradeLocker: order recovery pending for {} {} {} (attempt {}/3)",
                    side,
                    symbol,
                    volume,
                    attempt + 1,
                )
                await asyncio.sleep(0.5 * (attempt + 1))

        return {}

    @staticmethod
    def _value_at(values: list[Any], index: int) -> float:
        if index < 0 or index >= len(values):
            return 0.0
        return TradeLockerClient._as_float(values[index])

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
