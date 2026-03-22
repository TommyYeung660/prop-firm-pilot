"""
Broker-neutral execution models shared across broker implementations.

These models define the minimum data surface consumed by execution and
scheduler flows while preserving MatchTrader-compatible field aliases.

Usage:
    position = BrokerPositionInfo(positionId="POS-1", symbol="EURUSD", side="BUY")
    print(position.position_id)
"""

from typing import Any

from pydantic import AliasChoices, BaseModel, Field


class BrokerBalanceInfo(BaseModel):
    """Account balance snapshot in broker-neutral format."""

    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    free_margin: float = Field(default=0.0, alias="freeMargin")
    currency: str = "USD"

    model_config = {"populate_by_name": True}


class BrokerPositionInfo(BaseModel):
    """Open position details in broker-neutral format."""

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
    sl_price: float | None = Field(
        default=None,
        validation_alias=AliasChoices("stopLoss", "slPrice"),
        serialization_alias="slPrice",
    )
    tp_price: float | None = Field(
        default=None,
        validation_alias=AliasChoices("takeProfit", "tpPrice"),
        serialization_alias="tpPrice",
    )
    open_time: str = Field(default="", alias="openTime")

    model_config = {"populate_by_name": True}


class BrokerClosedPosition(BaseModel):
    """Historical closed position in broker-neutral format."""

    position_id: str = Field(default="", alias="positionId")
    symbol: str = ""
    side: str = ""
    volume: float = 0.0
    open_price: float = Field(default=0.0, alias="openPrice")
    close_price: float = Field(default=0.0, alias="closePrice")
    profit: float = 0.0
    open_time: str = Field(default="", alias="openTime")
    close_time: str = Field(default="", alias="closeTime")
    close_reason: str = Field(default="", alias="closeReason")

    model_config = {"populate_by_name": True}


class BrokerQuoteInfo(BaseModel):
    """Real-time quote (bid/ask) in broker-neutral format."""

    symbol: str = ""
    bid: float = 0.0
    ask: float = 0.0
    high: float = 0.0
    low: float = 0.0
    timestamp_ms: int = Field(default=0, alias="timestampMs")

    model_config = {"populate_by_name": True}


class BrokerInstrumentInfo(BaseModel):
    """Tradeable instrument details in broker-neutral format."""

    symbol: str = ""
    alias: str = ""
    description: str = ""
    type: str = ""
    base_currency: str = Field(default="", alias="baseCurrency")
    quote_currency: str = Field(default="", alias="quoteCurrency")

    # Session & availability
    session_open: bool = Field(default=False, alias="sessionOpen")
    trading_hours: list[dict[str, Any]] = Field(default_factory=list, alias="tradingHours")

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


class BrokerOrderResult(BaseModel):
    """Result for opening/closing/modifying a position."""

    success: bool = False
    position_id: str = ""
    message: str = ""
    raw_response: dict[str, Any] = Field(default_factory=dict)
