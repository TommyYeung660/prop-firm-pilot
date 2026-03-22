"""
Tests for src/execution/tradelocker_client.py.

These tests lock the stable broker-neutral surface for TradeLocker.
"""

from unittest.mock import AsyncMock

import pytest

from src.execution.broker_models import (
    BrokerClosedPosition,
    BrokerInstrumentInfo,
    BrokerOrderResult,
    BrokerPositionInfo,
    BrokerQuoteInfo,
)
from src.execution.tradelocker_client import TradeLockerClient


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


async def test_close_position_uses_delete_route(client: TradeLockerClient) -> None:
    client._account_request = AsyncMock(return_value={"status": "ok"})

    result = await client.close_position(
        position_id="POS-1",
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
    )

    assert isinstance(result, BrokerOrderResult)
    assert result.success is True
    assert result.position_id == "POS-1"
    assert client._account_request.await_args.args == ("DELETE", "/trade/positions/POS-1")
    assert client._account_request.await_args.kwargs["params"] == {"qty": 0}


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
