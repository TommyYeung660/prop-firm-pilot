"""
Tests for broker-neutral execution models and protocol.

These tests define the minimum contract shared by execution/scheduler code
regardless of broker implementation.
"""

from typing import Any

from src.execution.broker_models import (
    BrokerBalanceInfo,
    BrokerClosedPosition,
    BrokerInstrumentInfo,
    BrokerOrderResult,
    BrokerPositionInfo,
    BrokerQuoteInfo,
)
from src.execution.broker_protocol import BrokerClientProtocol


def test_broker_position_info_exposes_execution_fields() -> None:
    position = BrokerPositionInfo(
        position_id="POS-1",
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
        open_price=1.1,
        current_price=1.101,
        profit=10.0,
    )
    assert position.position_id == "POS-1"
    assert position.symbol == "EURUSD"
    assert position.side == "BUY"
    assert position.volume == 0.1
    assert position.open_price == 1.1
    assert position.current_price == 1.101
    assert position.profit == 10.0


def test_broker_balance_info_exposes_execution_fields() -> None:
    balance = BrokerBalanceInfo(
        balance=5000.0,
        equity=5100.0,
        margin=100.0,
        free_margin=4900.0,
        currency="USD",
    )
    assert balance.balance == 5000.0
    assert balance.equity == 5100.0
    assert balance.margin == 100.0
    assert balance.free_margin == 4900.0
    assert balance.currency == "USD"


def test_broker_closed_position_quote_instrument_order_fields() -> None:
    closed = BrokerClosedPosition(
        position_id="POS-2",
        symbol="GBPUSD",
        side="SELL",
        volume=0.2,
        open_price=1.26,
        close_price=1.255,
        profit=100.0,
    )
    quote = BrokerQuoteInfo(
        symbol="EURUSD",
        bid=1.10001,
        ask=1.10003,
        timestamp_ms=1726200032067,
    )
    instrument = BrokerInstrumentInfo(
        symbol="EURUSD",
        price_precision=5,
        volume_min=0.01,
        volume_max=50.0,
    )
    result = BrokerOrderResult(
        success=True,
        position_id="POS-1",
        message="ok",
        raw_response={"orderId": "POS-1"},
    )

    assert closed.position_id == "POS-2"
    assert closed.close_price == 1.255
    assert closed.profit == 100.0
    assert quote.bid == 1.10001
    assert quote.ask == 1.10003
    assert quote.timestamp_ms == 1726200032067
    assert instrument.price_precision == 5
    assert instrument.volume_min == 0.01
    assert instrument.volume_max == 50.0
    assert result.success is True
    assert result.position_id == "POS-1"
    assert result.raw_response["orderId"] == "POS-1"


def test_broker_protocol_exposes_required_methods() -> None:
    required_methods = {
        "login",
        "get_balance",
        "get_open_positions",
        "get_closed_positions",
        "get_quote",
        "get_effective_instruments",
        "open_position",
        "close_position",
        "close_all_positions",
        "modify_position",
        "verify_sl_tp",
    }

    assert required_methods.issubset(set(BrokerClientProtocol.__dict__.keys()))


def test_broker_order_result_default_raw_response_is_dict() -> None:
    result = BrokerOrderResult()
    assert isinstance(result.raw_response, dict)
    assert result.raw_response == {}
    result.raw_response["note"] = "x"

    second = BrokerOrderResult()
    assert second.raw_response == {}
    assert "note" not in second.raw_response


def test_broker_models_are_pydantic_compatible_mapping_input() -> None:
    payload: dict[str, Any] = {
        "positionId": "POS-100",
        "symbol": "USDJPY",
        "side": "BUY",
        "volume": 0.1,
        "openPrice": 150.1,
        "currentPrice": 150.2,
        "profit": 15.0,
    }
    position = BrokerPositionInfo(**payload)
    assert position.position_id == "POS-100"
