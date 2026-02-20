"""
Order lifecycle manager — coordinates position open/close with compliance checks.

Bridges PropFirmGuard, PositionSizer, and MatchTraderClient into a single
cohesive order flow. This is the "traffic controller" between decision and execution.
"""

from datetime import datetime, timezone
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field


class TradeSignal(BaseModel):
    """Incoming trade signal from TradingAgents decision engine."""

    symbol: str
    side: Literal["BUY", "SELL"]
    score: float = Field(description="Scanner model score")
    confidence: str = Field(description="high / medium / low")
    score_gap: float = 0.0
    suggested_sl_pips: float = Field(default=50.0, description="Suggested stop loss in pips")
    suggested_tp_pips: float = Field(default=100.0, description="Suggested take profit in pips")


class TradeRecord(BaseModel):
    """Complete record of an executed trade for journaling."""

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    symbol: str = ""
    side: str = ""
    volume: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_id: str = ""
    risk_amount: float = 0.0
    decision_score: float = 0.0
    confidence: str = ""
    compliance_passed: bool = True
    status: Literal["OPENED", "CLOSED", "REJECTED", "FAILED"] = "OPENED"
    close_price: float | None = None
    pnl: float | None = None
    close_reason: str = ""


class OrderManager:
    """Manages the lifecycle of trading orders.

    Responsibilities:
    - Convert trade signals into trade plans
    - Calculate entry/SL/TP prices from pip distances
    - Track active orders and their states
    - Provide order history for compliance tracking

    Note: This class does NOT make HTTP calls. It prepares data structures
    for MatchTraderClient and validates against PropFirmGuard.
    The main orchestrator (PropFirmPilot) coordinates the actual calls.
    """

    def __init__(self, instruments: dict[str, dict[str, Any]]) -> None:
        """
        Args:
            instruments: Map of symbol -> instrument config dict with pip_size, pip_value.
        """
        self._instruments = instruments
        self._active_orders: dict[str, TradeRecord] = {}  # position_id -> TradeRecord
        self._history: list[TradeRecord] = []

    def calculate_sl_tp_prices(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        entry_price: float,
        sl_pips: float,
        tp_pips: float,
    ) -> tuple[float, float]:
        """Calculate stop loss and take profit prices from pip distances.

        Args:
            symbol: Instrument name (e.g. "EURUSD").
            side: "BUY" or "SELL".
            entry_price: Current market price.
            sl_pips: Stop loss distance in pips.
            tp_pips: Take profit distance in pips.

        Returns:
            Tuple of (stop_loss_price, take_profit_price).
        """
        pip_size = self._get_pip_size(symbol)

        if side == "BUY":
            sl_price = entry_price - (sl_pips * pip_size)
            tp_price = entry_price + (tp_pips * pip_size)
        else:
            sl_price = entry_price + (sl_pips * pip_size)
            tp_price = entry_price - (tp_pips * pip_size)

        return round(sl_price, 5), round(tp_price, 5)

    def record_open(
        self,
        signal: TradeSignal,
        position_id: str,
        volume: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        risk_amount: float,
    ) -> TradeRecord:
        """Record a successfully opened trade."""
        record = TradeRecord(
            symbol=signal.symbol,
            side=signal.side,
            volume=volume,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_id=position_id,
            risk_amount=risk_amount,
            decision_score=signal.score,
            confidence=signal.confidence,
            status="OPENED",
        )
        self._active_orders[position_id] = record
        logger.info(
            "Order opened: {} {} {} lots @ {} (SL={}, TP={}, risk=${})",
            signal.side,
            signal.symbol,
            volume,
            entry_price,
            stop_loss,
            take_profit,
            risk_amount,
        )
        return record

    def record_close(
        self,
        position_id: str,
        close_price: float,
        pnl: float,
        reason: str = "signal",
    ) -> TradeRecord | None:
        """Record a closed trade and move to history."""
        record = self._active_orders.pop(position_id, None)
        if record is None:
            logger.warning("Order close: position {} not found in active orders", position_id)
            return None

        record.status = "CLOSED"
        record.close_price = close_price
        record.pnl = pnl
        record.close_reason = reason
        self._history.append(record)

        logger.info(
            "Order closed: {} {} pnl=${} (reason={})",
            record.symbol,
            position_id,
            pnl,
            reason,
        )
        return record

    def record_rejection(self, signal: TradeSignal, reason: str) -> TradeRecord:
        """Record a rejected trade (failed compliance)."""
        record = TradeRecord(
            symbol=signal.symbol,
            side=signal.side,
            decision_score=signal.score,
            confidence=signal.confidence,
            compliance_passed=False,
            status="REJECTED",
            close_reason=reason,
        )
        self._history.append(record)
        logger.warning("Order rejected: {} {} — {}", signal.side, signal.symbol, reason)
        return record

    @property
    def active_count(self) -> int:
        return len(self._active_orders)

    @property
    def active_orders(self) -> dict[str, TradeRecord]:
        return dict(self._active_orders)

    @property
    def history(self) -> list[TradeRecord]:
        return list(self._history)

    def total_active_risk(self) -> float:
        """Sum of risk amounts for all active orders."""
        return sum(r.risk_amount for r in self._active_orders.values())

    def _get_pip_size(self, symbol: str) -> float:
        """Get pip size for a symbol."""
        instrument = self._instruments.get(symbol)
        if instrument is None:
            logger.warning("Unknown instrument '{}', defaulting pip_size=0.0001", symbol)
            return 0.0001
        return instrument.get("pip_size", 0.0001)
