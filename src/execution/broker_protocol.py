"""
Broker-neutral execution protocol.

Defines the minimum async broker surface required by execution and scheduler
components. Concrete clients (e.g., MatchTrader) should satisfy this protocol.

Usage:
    async def run(broker: BrokerClientProtocol) -> None:
        balance = await broker.get_balance()
        print(balance.equity)
"""

from typing import Any, Literal, Protocol

from src.execution.broker_models import (
    BrokerBalanceInfo,
    BrokerClosedPosition,
    BrokerInstrumentInfo,
    BrokerOrderResult,
    BrokerPositionInfo,
    BrokerQuoteInfo,
)


class BrokerClientProtocol(Protocol):
    """Async broker contract used by trading execution flows."""

    async def login(self, *, _quiet: bool = False) -> Any:
        ...

    async def get_balance(self) -> BrokerBalanceInfo:
        ...

    async def get_open_positions(self) -> list[BrokerPositionInfo]:
        ...

    async def get_closed_positions(self, from_ts: int, to_ts: int) -> list[BrokerClosedPosition]:
        ...

    async def get_quote(self, symbol: str) -> BrokerQuoteInfo:
        ...

    async def get_effective_instruments(self) -> list[BrokerInstrumentInfo]:
        ...

    async def open_position(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        volume: float,
        sl: float | None = None,
        tp: float | None = None,
    ) -> BrokerOrderResult:
        ...

    async def close_position(
        self,
        position_id: str,
        symbol: str,
        side: str,
        volume: float,
    ) -> BrokerOrderResult:
        ...

    async def close_all_positions(self) -> list[BrokerOrderResult]:
        ...

    async def modify_position(
        self,
        position_id: str,
        symbol: str,
        side: str,
        volume: float,
        sl: float | None = None,
        tp: float | None = None,
    ) -> BrokerOrderResult:
        ...

    async def verify_sl_tp(
        self,
        position_id: str,
        expected_sl: float | None = None,
        expected_tp: float | None = None,
        tolerance: float = 1e-6,
        price_precision: int | None = None,
    ) -> bool:
        ...
