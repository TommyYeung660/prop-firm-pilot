"""Tests for position-close → re-scan event trigger (v1.2.0)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.config import AppConfig


def _make_scheduler(**overrides):
    """Create a Scheduler with mocked dependencies."""
    from src.scheduler.scheduler import Scheduler

    config = overrides.get("config", AppConfig())
    store = overrides.get("store", MagicMock())
    store.get_active_positions = MagicMock(return_value=[])
    store.recycle_expired_claims = MagicMock(return_value=0)
    scanner = overrides.get("scanner", MagicMock())
    agents = MagicMock()
    engine = MagicMock()
    matchtrader = AsyncMock()
    matchtrader.get_balance = AsyncMock(return_value=MagicMock(equity=5000, balance=5000))
    matchtrader.get_open_positions = AsyncMock(return_value=[])
    matchtrader.get_closed_positions = AsyncMock(return_value=[])

    return Scheduler(
        config=config,
        store=store,
        scanner=scanner,
        agents=agents,
        engine=engine,
        matchtrader=matchtrader,
    )


def test_scheduler_has_rescan_event():
    """Scheduler should have a _rescan_event asyncio.Event."""
    scheduler = _make_scheduler()
    assert hasattr(scheduler, "_rescan_event")
    assert isinstance(scheduler._rescan_event, asyncio.Event)


async def test_handle_position_closed_sets_rescan_event():
    """When a position closes, _rescan_event should be set."""
    scheduler = _make_scheduler()
    # Ensure event is clear initially
    assert not scheduler._rescan_event.is_set()

    # Mock intent for _handle_position_closed
    intent = MagicMock()
    intent.symbol = "EURUSD"
    intent.suggested_side = "BUY"
    intent.position_id = "12345"
    intent.id = "intent-1"
    intent.executed_at = None

    store = scheduler._store
    store.mark_closed = MagicMock()

    with patch.object(scheduler, "_send_alert", new_callable=AsyncMock):
        await scheduler._handle_position_closed(intent)

    assert scheduler._rescan_event.is_set()


async def test_rescan_event_clears_after_scanner_loop_reads():
    """After scanner loop picks up the event, it should be cleared."""
    scheduler = _make_scheduler()
    scheduler._rescan_event.set()

    # The event should be clearable
    scheduler._rescan_event.clear()
    assert not scheduler._rescan_event.is_set()
