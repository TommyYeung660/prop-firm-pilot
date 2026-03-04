"""
Tests for exit_reason classification in Scheduler._handle_position_closed().

Validates that positions closed by SL/TP are correctly classified even when
the broker's closed-positions API fails to return the position within the
initial query window. The fix adds:
1. Retry logic (3 attempts with exponential delay) for broker API lookup
2. Re-inference of exit_reason from fallback PnL when broker API fails

Production evidence: 3 profitable trades (+$40.92, +$46.41, +$35.42) were
misclassified as "manual_close" because broker API didn't return them, but
_last_known_profit had the correct PnL.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import (
    AccountConfig,
    AppConfig,
    ComplianceConfig,
    DecisionStoreConfig,
    MonitorConfig,
    SchedulerConfig,
)
from src.decision.schemas import TradeIntent
from src.decision_store.sqlite_store import DecisionStore
from src.scheduler.scheduler import Scheduler

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: object) -> DecisionStore:
    """Create a fresh DecisionStore with a temporary database."""
    db_path = f"{tmp_path}/test_exit_reason.db"
    s = DecisionStore(db_path=db_path)
    yield s  # type: ignore[misc]
    s.close()


@pytest.fixture
def config() -> AppConfig:
    """Minimal AppConfig for testing."""
    return AppConfig(
        account=AccountConfig(initial_balance=50000),
        compliance=ComplianceConfig(),
        scheduler=SchedulerConfig(
            scanner_interval_seconds=0,
            llm_poll_interval_seconds=0,
            execution_poll_interval_seconds=0,
            janitor_interval_seconds=0,
            llm_worker_count=1,
            equity_poll_interval_seconds=0,
            position_monitor_interval_seconds=0,
            daily_summary_hour_utc=22,
        ),
        decision_store=DecisionStoreConfig(),
        monitor=MonitorConfig(),
    )


@pytest.fixture
def mock_matchtrader() -> AsyncMock:
    """Mock MatchTraderClient."""
    client = AsyncMock()
    client.get_balance.return_value = MagicMock(
        balance=50000.0, equity=50000.0, margin=0.0, free_margin=50000.0
    )
    rate_limiter = MagicMock()
    rate_limiter.remaining = 1800
    rate_limiter._daily_limit = 2000
    client._rate_limiter = rate_limiter
    return client


@pytest.fixture
def scheduler(
    config: AppConfig,
    store: DecisionStore,
    mock_matchtrader: AsyncMock,
) -> Scheduler:
    """Create a Scheduler with minimal mocked dependencies."""
    return Scheduler(
        config=config,
        store=store,
        scanner=MagicMock(),
        agents=MagicMock(),
        engine=AsyncMock(),
        matchtrader=mock_matchtrader,
    )


def _make_opened_intent(
    position_id: str = "W123456",
    symbol: str = "AUDUSD",
    side: str = "SELL",
) -> TradeIntent:
    """Create a TradeIntent in 'opened' state with a position_id."""
    return TradeIntent(
        trade_date="2026-03-03",
        symbol=symbol,
        status="opened",
        position_id=position_id,
        suggested_side=side,
        executed_at=datetime(2026, 3, 3, 10, 0, 0, tzinfo=timezone.utc),
    )


def _make_closed_position(
    position_id: str = "W123456",
    profit: float = 40.92,
    close_price: float = 0.69200,
    open_price: float = 0.69600,
    volume: float = 0.08,
) -> MagicMock:
    """Create a mock ClosedPosition returned by broker API."""
    pos = MagicMock()
    pos.position_id = position_id
    pos.profit = profit
    pos.close_price = close_price
    pos.open_price = open_price
    pos.volume = volume
    return pos


# ── Tests: Retry Logic ──────────────────────────────────────────────────────


async def test_broker_api_found_on_first_attempt(
    scheduler: Scheduler,
    store: DecisionStore,
    mock_matchtrader: AsyncMock,
):
    """When broker returns the closed position on first try → correct classification."""
    intent = _make_opened_intent()
    store.insert_intent(intent)

    closed_pos = _make_closed_position(profit=40.92)
    mock_matchtrader.get_closed_positions.return_value = [closed_pos]

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await scheduler._handle_position_closed(intent)

    updated = store.get_intent(intent.id)
    assert updated is not None
    assert updated.exit_reason == "tp_hit"
    assert updated.realized_pnl == 40.92


async def test_broker_api_found_on_second_attempt(
    scheduler: Scheduler,
    store: DecisionStore,
    mock_matchtrader: AsyncMock,
):
    """Broker returns empty on first attempt, found on second → correct classification."""
    intent = _make_opened_intent()
    store.insert_intent(intent)

    closed_pos = _make_closed_position(profit=-22.50)
    # First attempt returns empty, second returns the position
    mock_matchtrader.get_closed_positions.side_effect = [[], [closed_pos]]

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await scheduler._handle_position_closed(intent)

    updated = store.get_intent(intent.id)
    assert updated is not None
    assert updated.exit_reason == "sl_hit"
    assert updated.realized_pnl == -22.50


async def test_broker_api_found_on_third_attempt(
    scheduler: Scheduler,
    store: DecisionStore,
    mock_matchtrader: AsyncMock,
):
    """Broker returns empty on first two attempts, found on third → correct classification."""
    intent = _make_opened_intent()
    store.insert_intent(intent)

    closed_pos = _make_closed_position(profit=35.00)
    mock_matchtrader.get_closed_positions.side_effect = [[], [], [closed_pos]]

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await scheduler._handle_position_closed(intent)

    updated = store.get_intent(intent.id)
    assert updated is not None
    assert updated.exit_reason == "tp_hit"
    assert updated.realized_pnl == 35.00


async def test_broker_api_never_returns_position(
    scheduler: Scheduler,
    store: DecisionStore,
    mock_matchtrader: AsyncMock,
):
    """Broker never returns position across all 3 retries → falls back to manual_close."""
    intent = _make_opened_intent()
    store.insert_intent(intent)

    mock_matchtrader.get_closed_positions.return_value = []

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await scheduler._handle_position_closed(intent)

    updated = store.get_intent(intent.id)
    assert updated is not None
    # No PnL data, no fallback → stays manual_close
    assert updated.exit_reason == "manual_close"
    assert updated.realized_pnl == 0.0


async def test_broker_api_exception_falls_through(
    scheduler: Scheduler,
    store: DecisionStore,
    mock_matchtrader: AsyncMock,
):
    """Broker API throws exception → falls through to fallback logic."""
    intent = _make_opened_intent()
    store.insert_intent(intent)

    mock_matchtrader.get_closed_positions.side_effect = Exception("API timeout")

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await scheduler._handle_position_closed(intent)

    updated = store.get_intent(intent.id)
    assert updated is not None
    assert updated.exit_reason == "manual_close"
    assert updated.realized_pnl == 0.0


# ── Tests: PnL Fallback Re-inference ────────────────────────────────────────


async def test_last_known_profit_positive_reinfers_tp_hit(
    scheduler: Scheduler,
    store: DecisionStore,
    mock_matchtrader: AsyncMock,
):
    """When broker API fails but _last_known_profit has positive PnL → tp_hit.

    This is the exact production bug: 3 trades with +$40.92, +$46.41, +$35.42
    were classified as manual_close because broker API didn't return them.
    """
    intent = _make_opened_intent()
    store.insert_intent(intent)

    # Broker API returns empty (simulating the production failure)
    mock_matchtrader.get_closed_positions.return_value = []

    # But we have last-known profit from position monitoring
    scheduler._last_known_profit["W123456"] = 40.92

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await scheduler._handle_position_closed(intent)

    updated = store.get_intent(intent.id)
    assert updated is not None
    assert updated.exit_reason == "tp_hit"
    assert updated.realized_pnl == 40.92


async def test_last_known_profit_negative_reinfers_sl_hit(
    scheduler: Scheduler,
    store: DecisionStore,
    mock_matchtrader: AsyncMock,
):
    """When broker API fails but _last_known_profit has negative PnL → sl_hit."""
    intent = _make_opened_intent()
    store.insert_intent(intent)

    mock_matchtrader.get_closed_positions.return_value = []
    scheduler._last_known_profit["W123456"] = -13.19

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await scheduler._handle_position_closed(intent)

    updated = store.get_intent(intent.id)
    assert updated is not None
    assert updated.exit_reason == "sl_hit"
    assert updated.realized_pnl == -13.19


async def test_last_known_profit_zero_stays_manual_close(
    scheduler: Scheduler,
    store: DecisionStore,
    mock_matchtrader: AsyncMock,
):
    """When broker API fails and _last_known_profit is 0.0 → manual_close (no data)."""
    intent = _make_opened_intent()
    store.insert_intent(intent)

    mock_matchtrader.get_closed_positions.return_value = []
    scheduler._last_known_profit["W123456"] = 0.0

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await scheduler._handle_position_closed(intent)

    updated = store.get_intent(intent.id)
    assert updated is not None
    assert updated.exit_reason == "manual_close"
    assert updated.realized_pnl == 0.0


# ── Tests: Best Day / Reeval Overrides Not Affected ─────────────────────────


async def test_best_day_close_not_reinferred(
    scheduler: Scheduler,
    store: DecisionStore,
    mock_matchtrader: AsyncMock,
):
    """best_day_close exit_reason is NOT overridden by PnL re-inference."""
    intent = _make_opened_intent()
    store.insert_intent(intent)

    mock_matchtrader.get_closed_positions.return_value = []
    scheduler._best_day_close_positions["W123456"] = 50.0

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await scheduler._handle_position_closed(intent)

    updated = store.get_intent(intent.id)
    assert updated is not None
    assert updated.exit_reason == "best_day_close"
    assert updated.realized_pnl == 50.0


async def test_reeval_close_not_reinferred(
    scheduler: Scheduler,
    store: DecisionStore,
    mock_matchtrader: AsyncMock,
):
    """reeval_close exit_reason is NOT overridden by PnL re-inference."""
    intent = _make_opened_intent()
    store.insert_intent(intent)

    mock_matchtrader.get_closed_positions.return_value = []
    scheduler._reevaluation_close_positions["W123456"] = -10.0

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await scheduler._handle_position_closed(intent)

    updated = store.get_intent(intent.id)
    assert updated is not None
    assert updated.exit_reason == "reeval_close"
    assert updated.realized_pnl == -10.0


# ── Tests: Retry Call Count ─────────────────────────────────────────────────


async def test_retry_calls_broker_api_up_to_3_times(
    scheduler: Scheduler,
    store: DecisionStore,
    mock_matchtrader: AsyncMock,
):
    """Verify broker API is called exactly 3 times when position is never found."""
    intent = _make_opened_intent()
    store.insert_intent(intent)

    mock_matchtrader.get_closed_positions.return_value = []

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await scheduler._handle_position_closed(intent)

    assert mock_matchtrader.get_closed_positions.call_count == 3


async def test_retry_stops_early_when_found(
    scheduler: Scheduler,
    store: DecisionStore,
    mock_matchtrader: AsyncMock,
):
    """Verify broker API stops retrying once position is found."""
    intent = _make_opened_intent()
    store.insert_intent(intent)

    closed_pos = _make_closed_position(profit=25.0)
    mock_matchtrader.get_closed_positions.side_effect = [[], [closed_pos]]

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await scheduler._handle_position_closed(intent)

    # Should stop after finding on 2nd attempt
    assert mock_matchtrader.get_closed_positions.call_count == 2
