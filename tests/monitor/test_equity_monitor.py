"""
Tests for EquityMonitor one-shot checks and graded reactions.
"""

from unittest.mock import AsyncMock

import pytest

from src.monitor.equity_monitor import EquityMonitor


@pytest.mark.asyncio
async def test_check_once_triggers_reduce_exposure_before_emergency_close() -> None:
    """DANGER level should reduce exposure without triggering full emergency close."""
    monitor = EquityMonitor(check_interval=60, drawdown_alert_pct=0.80, auto_close_pct=0.90)
    on_alert = AsyncMock()
    on_reduce = AsyncMock()
    on_close = AsyncMock()

    async def get_equity() -> float:
        return 4780.0  # 88% of a 5% daily drawdown on 5k balance

    result = await monitor.check_once(
        get_equity=get_equity,
        on_alert=on_alert,
        on_reduce_exposure=on_reduce,
        on_emergency_close=on_close,
        day_start_balance=5000.0,
        initial_balance=5000.0,
        daily_drawdown_limit=0.05,
        max_drawdown_limit=0.08,
    )

    assert result["level"] == "DANGER"
    on_alert.assert_awaited_once()
    on_reduce.assert_awaited_once()
    on_close.assert_not_awaited()


@pytest.mark.asyncio
async def test_check_once_triggers_emergency_close_at_critical() -> None:
    """CRITICAL level should trigger full emergency close."""
    monitor = EquityMonitor(check_interval=60, drawdown_alert_pct=0.80, auto_close_pct=0.90)
    on_alert = AsyncMock()
    on_reduce = AsyncMock()
    on_close = AsyncMock()

    async def get_equity() -> float:
        return 4540.0  # 92% of a 5% daily drawdown on 5k balance

    result = await monitor.check_once(
        get_equity=get_equity,
        on_alert=on_alert,
        on_reduce_exposure=on_reduce,
        on_emergency_close=on_close,
        day_start_balance=5000.0,
        initial_balance=5000.0,
        daily_drawdown_limit=0.05,
        max_drawdown_limit=0.08,
    )

    assert result["level"] == "CRITICAL"
    on_alert.assert_awaited_once()
    on_close.assert_awaited_once()
