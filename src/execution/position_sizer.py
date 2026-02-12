"""
FX position sizer — calculates optimal lot size based on risk parameters.

Uses the standard FX risk formula:
    volume = risk_amount / (stop_loss_pips × pip_value)

Applies random offset (±10%) to prevent duplicate strategy detection by E8.
"""

import random
from typing import Dict

from loguru import logger

from src.config import ExecutionConfig, InstrumentConfig


class PositionSizer:
    """Calculates FX position sizes based on risk percentage and pip values.

    Usage:
        sizer = PositionSizer(exec_config, instruments)
        volume = sizer.calculate_volume("EURUSD", equity=50000, stop_loss_pips=50)
        risk = sizer.calculate_risk_amount("EURUSD", volume=0.1, stop_loss_pips=50)
    """

    def __init__(
        self,
        config: ExecutionConfig,
        instruments: Dict[str, InstrumentConfig],
    ) -> None:
        self._config = config
        self._instruments = instruments

    def calculate_volume(
        self,
        symbol: str,
        account_equity: float,
        stop_loss_pips: float,
    ) -> float:
        """Calculate optimal lot size for a trade.

        Formula:
            risk_amount = account_equity × default_risk_pct
            volume = risk_amount / (stop_loss_pips × pip_value)
            volume *= (1 ± position_offset_pct)   # random offset
            volume = clamp(min_lot, max_lot)
            volume = round to 0.01

        Args:
            symbol: FX pair (e.g. "EURUSD").
            account_equity: Current account equity in USD.
            stop_loss_pips: Stop loss distance in pips.

        Returns:
            Volume in lots (rounded to 0.01 precision).
        """
        inst = self._get_instrument(symbol)
        if inst is None:
            logger.error("PositionSizer: unknown instrument '{}', returning min lot", symbol)
            return 0.01

        if stop_loss_pips <= 0:
            logger.error("PositionSizer: stop_loss_pips must be > 0, got {}", stop_loss_pips)
            return 0.01

        # Core calculation
        risk_amount = account_equity * self._config.default_risk_pct
        pip_value = inst.pip_value
        volume = risk_amount / (stop_loss_pips * pip_value)

        # Apply random offset for anti-duplicate-strategy
        offset_pct = self._config.position_offset_pct
        offset = random.uniform(-offset_pct, offset_pct)
        volume *= 1 + offset

        # Clamp to instrument limits
        volume = max(inst.min_lot, min(inst.max_lot, volume))

        # Round to 0.01 lot precision
        volume = round(volume, 2)

        logger.debug(
            "PositionSizer: {} equity=${:.0f} SL={:.0f}pips → {:.2f} lots "
            "(risk=${:.2f}, pip_val=${:.2f}, offset={:+.1%})",
            symbol,
            account_equity,
            stop_loss_pips,
            volume,
            risk_amount,
            pip_value,
            offset,
        )
        return volume

    def calculate_risk_amount(
        self,
        symbol: str,
        volume: float,
        stop_loss_pips: float,
    ) -> float:
        """Calculate dollar risk for a given volume and stop loss.

        Formula: risk = volume × stop_loss_pips × pip_value

        Args:
            symbol: FX pair.
            volume: Position size in lots.
            stop_loss_pips: Stop loss distance in pips.

        Returns:
            Risk amount in USD.
        """
        inst = self._get_instrument(symbol)
        if inst is None:
            logger.warning("PositionSizer: unknown instrument '{}', estimating risk", symbol)
            return volume * stop_loss_pips * 10.0  # Assume $10/pip as default

        risk = volume * stop_loss_pips * inst.pip_value
        return round(risk, 2)

    def max_volume_for_risk(
        self,
        symbol: str,
        max_risk: float,
        stop_loss_pips: float,
    ) -> float:
        """Calculate maximum volume that stays within a dollar risk budget.

        Args:
            symbol: FX pair.
            max_risk: Maximum dollars at risk.
            stop_loss_pips: Stop loss distance in pips.

        Returns:
            Maximum volume in lots (rounded to 0.01).
        """
        inst = self._get_instrument(symbol)
        if inst is None or stop_loss_pips <= 0:
            return 0.01

        volume = max_risk / (stop_loss_pips * inst.pip_value)
        volume = max(inst.min_lot, min(inst.max_lot, volume))
        return round(volume, 2)

    def estimate_pip_distance(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
    ) -> float:
        """Calculate pip distance between entry and stop prices.

        Args:
            symbol: FX pair.
            entry_price: Entry price.
            stop_price: Stop loss price.

        Returns:
            Distance in pips.
        """
        inst = self._get_instrument(symbol)
        pip_size = inst.pip_size if inst else 0.0001

        pips = abs(entry_price - stop_price) / pip_size
        return round(pips, 1)

    def _get_instrument(self, symbol: str) -> InstrumentConfig | None:
        """Get instrument config, logging a warning if not found."""
        inst = self._instruments.get(symbol)
        if inst is None:
            logger.warning("PositionSizer: instrument '{}' not configured", symbol)
        return inst
