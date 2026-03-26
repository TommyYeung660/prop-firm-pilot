"""
Tests for src/execution/tradelocker_client.py.

These tests lock the stable broker-neutral surface for TradeLocker.
"""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from src.execution.broker_models import (
    BrokerClosedPosition,
    BrokerInstrumentInfo,
    BrokerOrderResult,
    BrokerPositionInfo,
    BrokerQuoteInfo,
)
from src.execution.tradelocker_client import TradeLockerClient, TradeLockerError


@pytest.fixture
def client() -> TradeLockerClient:
    return TradeLockerClient(
        api_url="https://demo.tradelocker.com/backend-api",
        email="trader@example.com",
        password="secret",
        server="demo",
        account_id="ACC-2",
    )


async def test_login_auth_flow(client: TradeLockerClient) -> None:
    client._request = AsyncMock(
        side_effect=[
            {"accessToken": "access-token", "refreshToken": "refresh-token"},
            [
                {"accountId": "ACC-1", "accNum": "10001"},
                {"accountId": "ACC-2", "accNum": "10002"},
            ],
        ]
    )

    tokens = await client.login()

    assert tokens["accessToken"] == "access-token"
    assert tokens["refreshToken"] == "refresh-token"
    assert client.is_authenticated is True
    assert client.account_id == "ACC-2"
    assert client.acc_num == "10002"
    assert client._request.await_count == 2
    assert client._request.await_args_list[0].args == ("POST", "/auth/jwt/token")
    assert client._request.await_args_list[1].args == ("GET", "/auth/jwt/all-accounts")


async def test_account_selection_fails_when_account_not_found(client: TradeLockerClient) -> None:
    client._request = AsyncMock(
        side_effect=[
            {"accessToken": "access-token"},
            [{"accountId": "ACC-1", "accNum": "10001"}],
        ]
    )

    with pytest.raises(RuntimeError, match="Account ACC-2 not found"):
        await client.login()


async def test_account_request_reauths_on_401_and_retries_once(client: TradeLockerClient) -> None:
    async def fake_relogin(*, _quiet: bool = False) -> dict[str, str]:
        _ = _quiet
        client._access_token = "fresh-token"
        client._refresh_token = "fresh-refresh-token"
        client._account_id = "ACC-2"
        client._acc_num = "10002"
        return {
            "accessToken": client._access_token,
            "refreshToken": client._refresh_token,
        }

    client._access_token = "stale-token"
    client._refresh_token = "stale-refresh-token"
    client._account_id = "ACC-2"
    client._acc_num = "10002"
    client.login = AsyncMock(side_effect=fake_relogin)

    async with client:
        with respx.mock(assert_all_called=True) as router:
            state_route = router.get(
                "https://demo.tradelocker.com/backend-api/trade/accounts/ACC-2/state"
            ).mock(
                side_effect=[
                    httpx.Response(401, json={"message": "expired"}),
                    httpx.Response(200, json={"balance": "1000.0"}),
                ]
            )

            payload = await client._account_request("GET", "/trade/accounts/ACC-2/state")

    assert payload == {"balance": "1000.0"}
    assert client.login.await_count == 1
    assert state_route.call_count == 2
    assert state_route.calls[0].request.headers["Authorization"] == "Bearer stale-token"
    assert state_route.calls[1].request.headers["Authorization"] == "Bearer fresh-token"
    assert state_route.calls[0].request.headers["accNum"] == "10002"
    assert state_route.calls[1].request.headers["accNum"] == "10002"


async def test_get_balance_auto_login_uses_resolved_account_path(client: TradeLockerClient) -> None:
    async def fake_login(*, _quiet: bool = False) -> dict[str, str]:
        _ = _quiet
        client._access_token = "fresh-token"
        client._refresh_token = "fresh-refresh-token"
        client._account_id = "ACC-2"
        client._acc_num = "10002"
        return {
            "accessToken": client._access_token,
            "refreshToken": client._refresh_token,
        }

    client.login = AsyncMock(side_effect=fake_login)
    client._request = AsyncMock(
        return_value={
            "balance": "1000.0",
            "equity": "1005.0",
            "margin": "10.0",
            "freeMargin": "995.0",
            "currency": "USD",
        }
    )

    balance = await client.get_balance()

    assert balance.balance == 1000.0
    assert balance.equity == 1005.0
    assert client.login.await_count == 1
    assert client._request.await_args.args == ("GET", "/trade/accounts/ACC-2/state")


async def test_get_balance_parses_live_account_details_payload(client: TradeLockerClient) -> None:
    client._ensure_auth = AsyncMock()
    client._request = AsyncMock(
        return_value={
            "s": "ok",
            "d": {
                "accountDetailsData": [
                    1000.0,
                    1005.5,
                    995.5,
                    0.0,
                    1000.0,
                    0.0,
                    995.5,
                    0.0,
                    0.0,
                    10.0,
                    8.0,
                    90.0,
                    0.0,
                    0.0,
                    100.0,
                    0.0,
                    995.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    5.5,
                    5.5,
                    1,
                    0,
                ]
            },
        }
    )

    balance = await client.get_balance()

    assert balance.balance == 1000.0
    assert balance.equity == 1005.5
    assert balance.margin == 10.0
    assert balance.free_margin == 995.5
    assert balance.currency == "USD"


async def test_quote_parsing_returns_broker_model(client: TradeLockerClient) -> None:
    client._ensure_auth = AsyncMock()
    client._resolve_symbol_meta = AsyncMock(
        return_value={
            "tradableInstrumentId": "TI-EURUSD",
            "infoRouteId": "ROUTE-INFO",
            "tradeRouteId": "ROUTE-TRADE",
        }
    )
    client._request = AsyncMock(
        return_value={
            "symbol": "EURUSD",
            "bid": "1.08101",
            "ask": "1.08103",
            "high": "1.08200",
            "low": "1.07900",
            "timestampMs": 1726200032067,
        }
    )

    quote = await client.get_quote("EURUSD")

    assert isinstance(quote, BrokerQuoteInfo)
    assert quote.symbol == "EURUSD"
    assert quote.bid == 1.08101
    assert quote.ask == 1.08103
    assert quote.timestamp_ms == 1726200032067


async def test_quote_parsing_handles_live_bp_ap_payload(client: TradeLockerClient) -> None:
    client._ensure_auth = AsyncMock()
    client._resolve_symbol_meta = AsyncMock(
        return_value={
            "tradableInstrumentId": "TI-EURUSD",
            "infoRouteId": "948733",
            "tradeRouteId": "948735",
        }
    )
    client._request = AsyncMock(
        return_value={
            "s": "ok",
            "d": {
                "bp": 1.08101,
                "ap": 1.08103,
                "bs": 100000.0,
                "as": 100000.0,
            },
        }
    )

    quote = await client.get_quote("EURUSD")

    assert quote.symbol == "EURUSD"
    assert quote.bid == 1.08101
    assert quote.ask == 1.08103
    assert quote.timestamp_ms == 0


async def test_instrument_parsing_returns_broker_model(client: TradeLockerClient) -> None:
    client._account_request = AsyncMock(
        return_value=[
            {
                "name": "EURUSD",
                "description": "Euro vs US Dollar",
                "type": "FOREX",
                "baseCurrency": "EUR",
                "quoteCurrency": "USD",
                "sessionOpen": True,
                "volumeMin": "0.01",
                "volumeMax": "50",
                "volumeStep": "0.01",
                "volumePrecision": 2,
                "pricePrecision": 5,
                "sizeOfOnePoint": "0.00001",
                "contractSize": "100000",
                "leverage": "100",
                "tradableInstrumentId": "TI-EURUSD",
                "routes": {"INFO": "ROUTE-INFO", "TRADE": "ROUTE-TRADE"},
            }
        ]
    )

    instruments = await client.get_effective_instruments()

    assert len(instruments) == 1
    assert isinstance(instruments[0], BrokerInstrumentInfo)
    assert instruments[0].symbol == "EURUSD"
    assert instruments[0].session_open is True
    assert instruments[0].volume_min == 0.01
    assert instruments[0].price_precision == 5


async def test_instrument_parsing_handles_live_nested_payload_and_symbol_suffixes(
    client: TradeLockerClient,
) -> None:
    client._account_request = AsyncMock(
        return_value={
            "s": "ok",
            "d": {
                "instruments": [
                    {
                        "name": "EURUSD+",
                        "description": "Euro vs US Dollar",
                        "type": "FOREX",
                        "sessionOpen": True,
                        "volumeMin": "0.01",
                        "volumeMax": "50",
                        "volumeStep": "0.01",
                        "volumePrecision": 2,
                        "pricePrecision": 5,
                        "sizeOfOnePoint": "0.00001",
                        "contractSize": "100000",
                        "leverage": "100",
                        "tradableInstrumentId": 6119,
                        "routes": [
                            {"id": 948735, "type": "TRADE"},
                            {"id": 948733, "type": "INFO"},
                        ],
                    }
                ]
            },
        }
    )

    instruments = await client.get_effective_instruments()

    assert len(instruments) == 1
    assert instruments[0].symbol == "EURUSD"
    assert instruments[0].alias == "EURUSD+"
    assert client._symbol_meta["EURUSD"]["tradableInstrumentId"] == "6119"
    assert client._symbol_meta["EURUSD"]["infoRouteId"] == "948733"
    assert client._symbol_meta["EURUSD"]["tradeRouteId"] == "948735"
    assert client._symbol_meta["EURUSD+"]["tradableInstrumentId"] == "6119"


async def test_open_positions_parsing_returns_broker_model(client: TradeLockerClient) -> None:
    client._account_request = AsyncMock(
        return_value=[
            {
                "id": "POS-1",
                "symbol": "EURUSD",
                "side": "BUY",
                "qty": "0.10",
                "openPrice": "1.0800",
                "currentPrice": "1.0810",
                "profit": "10.5",
                "stopLoss": "1.0700",
                "takeProfit": "1.0900",
                "openTime": "2026-03-22T09:30:00Z",
            }
        ]
    )

    positions = await client.get_open_positions()

    assert len(positions) == 1
    assert isinstance(positions[0], BrokerPositionInfo)
    assert positions[0].position_id == "POS-1"
    assert positions[0].volume == 0.1
    assert positions[0].sl_price == 1.07
    assert positions[0].tp_price == 1.09


async def test_open_positions_parsing_handles_live_nested_payload(
    client: TradeLockerClient,
) -> None:
    client._account_request = AsyncMock(
        return_value={
            "s": "ok",
            "d": {
                "positions": [
                    {
                        "id": "POS-1",
                        "symbol": "EURUSD+",
                        "side": "BUY",
                        "qty": "0.10",
                        "avgPrice": "1.0800",
                        "unrealizedPl": "10.5",
                    }
                ]
            },
        }
    )

    positions = await client.get_open_positions()

    assert len(positions) == 1
    assert positions[0].position_id == "POS-1"
    assert positions[0].symbol == "EURUSD"
    assert positions[0].volume == 0.1
    assert positions[0].profit == 10.5


async def test_open_positions_parsing_handles_live_array_payload(
    client: TradeLockerClient,
) -> None:
    client._instrument_id_to_symbol = {"6125": "USDCAD"}
    client._account_request = AsyncMock(
        return_value={
            "s": "ok",
            "d": {
                "positions": [
                    [
                        "7421932185916132049",
                        "6125",
                        "948735",
                        "buy",
                        "0.62",
                        "1.37364",
                        None,
                        None,
                        "1774250624000",
                        "8.12",
                        "key-undefined",
                    ]
                ]
            },
        }
    )

    positions = await client.get_open_positions()

    assert len(positions) == 1
    assert positions[0].position_id == "7421932185916132049"
    assert positions[0].symbol == "USDCAD"
    assert positions[0].side == "BUY"
    assert positions[0].volume == 0.62
    assert positions[0].open_price == 1.37364
    assert positions[0].profit == 8.12
    assert positions[0].sl_price is None
    assert positions[0].tp_price is None


async def test_open_positions_live_array_uses_side_aware_quotes_for_current_price(
    client: TradeLockerClient,
) -> None:
    client._instrument_id_to_symbol = {
        "6124": "USDCHF",
        "6125": "AUDJPY",
    }
    client._account_request = AsyncMock(
        return_value={
            "s": "ok",
            "d": {
                "positions": [
                    [
                        "POS-BUY",
                        "6124",
                        "948735",
                        "buy",
                        "0.62",
                        "0.78947",
                        None,
                        None,
                        "1774250624000",
                        "209.75",
                        "key-1",
                    ],
                    [
                        "POS-SELL",
                        "6125",
                        "948735",
                        "sell",
                        "0.91",
                        "110.867",
                        None,
                        None,
                        "1774250624001",
                        "-74.19",
                        "key-2",
                    ],
                ]
            },
        }
    )
    client.get_quote = AsyncMock(
        side_effect=[
            BrokerQuoteInfo(symbol="USDCHF", bid=0.79190, ask=0.79198),
            BrokerQuoteInfo(symbol="AUDJPY", bid=110.950, ask=110.962),
        ]
    )

    positions = await client.get_open_positions()

    assert len(positions) == 2
    assert positions[0].current_price == 0.79190
    assert positions[1].current_price == 110.962


async def test_open_positions_live_array_recovers_sl_tp_from_protective_orders(
    client: TradeLockerClient,
) -> None:
    client._instrument_id_to_symbol = {"6124": "USDCHF"}
    client._account_request = AsyncMock(
        return_value={
            "s": "ok",
            "d": {
                "positions": [
                    [
                        "POS-BUY",
                        "6124",
                        "948735",
                        "buy",
                        "0.62",
                        "0.78947",
                        "ORDER-SL",
                        "ORDER-TP",
                        "1774250624000",
                        "209.75",
                        "key-1",
                    ]
                ]
            },
        }
    )
    client.get_quote = AsyncMock(
        return_value=BrokerQuoteInfo(
            symbol="USDCHF",
            bid=0.79190,
            ask=0.79198,
        )
    )
    client._fetch_open_orders = AsyncMock(
        return_value=[
            {
                "id": "ORDER-SL",
                "positionId": "POS-BUY",
                "type": "stop",
                "stopPrice": "0.78700",
            },
            {
                "id": "ORDER-TP",
                "positionId": "POS-BUY",
                "type": "limit",
                "price": "0.79350",
            },
        ]
    )

    positions = await client.get_open_positions()

    assert len(positions) == 1
    assert positions[0].sl_price == 0.78700
    assert positions[0].tp_price == 0.79350


async def test_open_positions_enrichment_failure_falls_back_to_raw_position(
    client: TradeLockerClient,
) -> None:
    client._instrument_id_to_symbol = {"6124": "USDCHF"}
    client._account_request = AsyncMock(
        return_value={
            "s": "ok",
            "d": {
                "positions": [
                    [
                        "POS-BUY",
                        "6124",
                        "948735",
                        "buy",
                        "0.62",
                        "0.78947",
                        None,
                        None,
                        "1774250624000",
                        "209.75",
                        "key-1",
                    ]
                ]
            },
        }
    )
    client.get_quote = AsyncMock(side_effect=RuntimeError("quote unavailable"))
    client._fetch_open_orders = AsyncMock(side_effect=RuntimeError("orders unavailable"))

    positions = await client.get_open_positions()

    assert len(positions) == 1
    assert positions[0].current_price == 0.78947
    assert positions[0].sl_price is None
    assert positions[0].tp_price is None


async def test_closed_positions_parsing_returns_broker_model(client: TradeLockerClient) -> None:
    client._account_request = AsyncMock(
        return_value=[
            {
                "positionId": "CLOSED-1",
                "symbol": "EURUSD",
                "side": "SELL",
                "qty": "0.20",
                "openPrice": "1.0900",
                "closePrice": "1.0850",
                "profit": "100.0",
                "openTime": "2026-03-20T09:30:00Z",
                "closeTime": "2026-03-20T14:30:00Z",
                "closeReason": "TAKE_PROFIT",
            }
        ]
    )

    closed = await client.get_closed_positions(from_ts=1700000000000, to_ts=1700003600000)

    assert len(closed) == 1
    assert isinstance(closed[0], BrokerClosedPosition)
    assert closed[0].position_id == "CLOSED-1"
    assert closed[0].volume == 0.2
    assert closed[0].profit == 100.0


async def test_closed_positions_parsing_handles_live_orders_history_array_payload(
    client: TradeLockerClient,
) -> None:
    client._instrument_id_to_symbol = {"6117": "AUDJPY"}
    client._account_request = AsyncMock(
        return_value={
            "s": "ok",
            "d": {
                "ordersHistory": [
                    [
                        "7421932185967501931",
                        "6117",
                        "948735",
                        "0.91",
                        "sell",
                        "market",
                        "Filled",
                        "0.91",
                        "110.867",
                        "110.867",
                        "0",
                        "IOC",
                        None,
                        "1774250619843",
                        "1774250620000",
                        "false",
                        "7421932185916132047",
                        None,
                        None,
                        None,
                        None,
                        "key-undefined",
                    ]
                ]
            },
        }
    )

    closed = await client.get_closed_positions(from_ts=1774250000000, to_ts=1774251000000)

    assert len(closed) == 1
    assert closed[0].position_id == "7421932185916132047"
    assert closed[0].symbol == "AUDJPY"
    assert closed[0].side == "SELL"
    assert closed[0].volume == 0.91
    assert closed[0].open_price == 110.867
    assert closed[0].close_price == 110.867


async def test_market_order_open_builds_provider_payload(client: TradeLockerClient) -> None:
    client._resolve_symbol_meta = AsyncMock(
        return_value={
            "tradableInstrumentId": "TI-EURUSD",
            "infoRouteId": "ROUTE-INFO",
            "tradeRouteId": "ROUTE-TRADE",
        }
    )
    client._account_request = AsyncMock(
        return_value={"orderId": "ORD-1", "positionId": "POS-NEW-1"}
    )

    result = await client.open_position(
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
        sl=1.08,
        tp=1.09,
    )

    assert isinstance(result, BrokerOrderResult)
    assert result.success is True
    assert result.position_id == "POS-NEW-1"
    body = client._account_request.await_args.kwargs["json"]
    assert body["qty"] == 0.1
    assert body["routeId"] == "ROUTE-TRADE"
    assert body["side"] == "BUY"
    assert body["validity"] == "IOC"
    assert body["type"] == "MARKET"
    assert body["price"] == 0
    assert body["tradableInstrumentId"] == "TI-EURUSD"
    assert body["stopLoss"] == 1.08
    assert body["takeProfit"] == 1.09


async def test_market_order_open_recovers_position_id_and_fill_price_from_orders_history(
    client: TradeLockerClient,
) -> None:
    client._instrument_id_to_symbol = {"6117": "AUDJPY"}
    client._resolve_symbol_meta = AsyncMock(
        return_value={
            "tradableInstrumentId": "6117",
            "infoRouteId": "948733",
            "tradeRouteId": "948735",
        }
    )
    client._account_request = AsyncMock(
        side_effect=[
            {"s": "ok", "d": {"orderId": "7421932185967501931"}},
            {
                "s": "ok",
                "d": {
                    "ordersHistory": [
                        [
                            "7421932185967501931",
                            "6117",
                            "948735",
                            "0.91",
                            "sell",
                            "market",
                            "Filled",
                            "0.91",
                            "110.867",
                            "110.867",
                            "0",
                            "IOC",
                            None,
                            "1774250619843",
                            "1774250620000",
                            "true",
                            "7421932185916132047",
                            None,
                            None,
                            None,
                            None,
                            "key-undefined",
                        ]
                    ],
                    "hasMore": False,
                },
            },
        ]
    )

    result = await client.open_position(symbol="AUDJPY", side="SELL", volume=0.91)

    assert result.success is True
    assert result.position_id == "7421932185916132047"
    assert result.raw_response["positionId"] == "7421932185916132047"
    assert result.raw_response["openPrice"] == 110.867
    assert client._account_request.await_args_list[1].args == (
        "GET",
        "/trade/accounts/{account_id}/ordersHistory",
    )


async def test_market_order_open_falls_back_to_positions_when_orders_history_is_rate_limited(
    client: TradeLockerClient,
) -> None:
    client._instrument_id_to_symbol = {"6118": "EURJPY"}
    client._resolve_symbol_meta = AsyncMock(
        return_value={
            "tradableInstrumentId": "6118",
            "infoRouteId": "948733",
            "tradeRouteId": "948735",
        }
    )
    client._account_request = AsyncMock(
        side_effect=[
            {"s": "ok", "d": {"orderId": "7421932185967885429"}},
            TradeLockerError(
                'TradeLocker API error 429: {"status":429,"error":"Too Many Requests",'
                '"path":"/clientapi/v1/accounts/ACC-2/ordersHistory"}'
            ),
            {
                "s": "ok",
                "d": {
                    "positions": [
                        [
                            "7421932185916212044",
                            "6118",
                            "948735",
                            "buy",
                            "0.92",
                            "184.329",
                            None,
                            None,
                            "",
                            "0.0",
                            "key-undefined",
                        ]
                    ]
                },
            },
        ]
    )

    result = await client.open_position(symbol="EURJPY", side="BUY", volume=0.92)

    assert result.success is True
    assert result.position_id == "7421932185916212044"
    assert result.raw_response["positionId"] == "7421932185916212044"
    assert result.raw_response["openPrice"] == 184.329


async def test_account_request_throttles_same_route_within_rate_limit_window(
    client: TradeLockerClient,
) -> None:
    client._access_token = "access-token"
    client._account_id = "ACC-2"
    client._acc_num = "10002"
    client._rate_limits_loaded = True
    client._route_rate_limits = {
        "GET_ORDERS": {
            "limit": 1,
            "interval_seconds": 1.0,
        }
    }
    client._send_request = AsyncMock(
        return_value=httpx.Response(200, json={"s": "ok", "d": {"orders": []}})
    )

    with patch(
        "src.execution.tradelocker_client.time.monotonic",
        side_effect=[100.0, 100.2, 100.2, 101.2, 101.2],
    ):
        with patch(
            "src.execution.tradelocker_client.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep:
            await client._account_request("GET", "/trade/accounts/{account_id}/orders")
            await client._account_request("GET", "/trade/accounts/{account_id}/orders")

    assert client._send_request.await_count == 2
    sleep.assert_awaited_once()
    assert sleep.await_args.args[0] == pytest.approx(0.8)


async def test_account_request_retries_after_429_using_route_interval(
    client: TradeLockerClient,
) -> None:
    client._access_token = "access-token"
    client._account_id = "ACC-2"
    client._acc_num = "10002"
    client._rate_limits_loaded = True
    client._route_rate_limits = {
        "GET_ORDERS": {
            "limit": 1,
            "interval_seconds": 1.0,
        }
    }
    client._send_request = AsyncMock(
        side_effect=[
            httpx.Response(
                429,
                json={
                    "timestamp": "2026-03-26T02:19:58.328Z",
                    "status": 429,
                    "error": "Too Many Requests",
                    "path": "/clientapi/v1/accounts/ACC-2/orders",
                },
            ),
            httpx.Response(200, json={"s": "ok", "d": {"orders": []}}),
        ]
    )

    with patch("src.execution.tradelocker_client.asyncio.sleep", new_callable=AsyncMock) as sleep:
        payload = await client._account_request("GET", "/trade/accounts/{account_id}/orders")

    assert payload == {"s": "ok", "d": {"orders": []}}
    assert client._send_request.await_count == 2
    sleep.assert_awaited_once()
    assert sleep.await_args.args[0] == pytest.approx(1.0)


async def test_prime_rate_limits_loads_trade_config_limits(client: TradeLockerClient) -> None:
    client._access_token = "access-token"
    client._account_id = "ACC-2"
    client._acc_num = "10002"
    client._send_request = AsyncMock(
        return_value=httpx.Response(
            200,
            json={
                "s": "ok",
                "d": {
                    "rateLimits": [
                        {
                            "rateLimitType": "GET_ORDERS",
                            "measure": "SECONDS",
                            "intervalNum": 2,
                            "limit": 3,
                        }
                    ]
                },
            },
        )
    )

    await client.prime_rate_limits()

    assert client._rate_limits_loaded is True
    assert client._route_rate_limits["GET_ORDERS"] == {
        "limit": 3,
        "interval_seconds": 2.0,
    }
    client._send_request.assert_awaited_once()


async def test_close_position_full_close_uses_qty_zero(client: TradeLockerClient) -> None:
    client._account_request = AsyncMock(return_value={"status": "ok"})

    result = await client.close_position(
        position_id="POS-1",
        symbol="EURUSD",
        side="BUY",
        volume=0.0,
    )

    assert isinstance(result, BrokerOrderResult)
    assert result.success is True
    assert result.position_id == "POS-1"
    assert client._account_request.await_args.args == ("DELETE", "/trade/positions/POS-1")
    assert client._account_request.await_args.kwargs["params"] == {"qty": 0}


async def test_close_position_partial_close_uses_volume(client: TradeLockerClient) -> None:
    client._account_request = AsyncMock(return_value={"status": "ok"})

    result = await client.close_position(
        position_id="POS-1",
        symbol="EURUSD",
        side="BUY",
        volume=0.03,
    )

    assert isinstance(result, BrokerOrderResult)
    assert result.success is True
    assert result.position_id == "POS-1"
    assert client._account_request.await_args.args == ("DELETE", "/trade/positions/POS-1")
    assert client._account_request.await_args.kwargs["params"] == {"qty": 0.03}


async def test_modify_position_uses_patch_route(client: TradeLockerClient) -> None:
    client._account_request = AsyncMock(return_value={"status": "ok"})

    result = await client.modify_position(
        position_id="POS-1",
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
        sl=1.081,
        tp=1.091,
    )

    assert isinstance(result, BrokerOrderResult)
    assert result.success is True
    assert result.position_id == "POS-1"
    assert client._account_request.await_args.args == ("PATCH", "/trade/positions/POS-1")
    assert client._account_request.await_args.kwargs["json"] == {
        "stopLoss": 1.081,
        "takeProfit": 1.091,
    }


async def test_verify_sl_tp_reads_back_position_values(client: TradeLockerClient) -> None:
    client.get_open_positions = AsyncMock(
        return_value=[
            BrokerPositionInfo(
                position_id="POS1",
                symbol="EURUSD",
                side="BUY",
                volume=0.1,
                sl_price=1.08,
                tp_price=1.09,
            )
        ]
    )

    verified = await client.verify_sl_tp(
        position_id="POS1",
        expected_sl=1.08,
        expected_tp=1.09,
    )

    assert verified is True
