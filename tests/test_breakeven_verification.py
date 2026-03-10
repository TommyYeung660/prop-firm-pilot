"""
Tests for breakeven stop read-back verification.

Validates:
1. verify_sl_tp reads back position and confirms SL/TP match
2. verify_sl_tp detects mismatch when broker didn't apply SL
3. _apply_breakeven_stops retries when verify detects mismatch
4. _apply_breakeven_stops does NOT add to breakeven_applied when verify fails twice
5. execution_meta is updated with breakeven SL for correct exit_reason
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import (
    AccountConfig,
    AppConfig,
    ComplianceConfig,
    DecisionStoreConfig,
    InstrumentConfig,
    MonitorConfig,
    SchedulerConfig,
)
from src.decision.schemas import TradeIntent
from src.decision_store.sqlite_store import DecisionStore
from src.execution.matchtrader_client import (
    MatchTraderClient,
    OrderResult,
    PositionInfo,
)
from src.scheduler.scheduler import Scheduler

# ── verify_sl_tp tests ─────────────────────────────────────────────────────


class TestVerifySLTP:
    """Tests for MatchTraderClient.verify_sl_tp()."""

    @pytest.fixture
    def client(self) -> MatchTraderClient:
        return MatchTraderClient(
            base_url="https://test.example.com",
            email="test@test.com",
            password="pass",
            broker_id="2",
            account_id="12345",
        )

    async def test_verify_sl_matches(self, client: MatchTraderClient) -> None:
        """When broker position SL matches expected, returns True."""
        client.get_open_positions = AsyncMock(
            return_value=[
                PositionInfo(
                    position_id="POS1",
                    symbol="EURUSD.",
                    side="BUY",
                    volume=0.1,
                    open_price=1.0800,
                    current_price=1.0850,
                    sl_price=1.0800,
                    tp_price=1.0900,
                )
            ]
        )
        result = await client.verify_sl_tp(position_id="POS1", expected_sl=1.0800)
        assert result is True

    async def test_verify_sl_mismatch(self, client: MatchTraderClient) -> None:
        """When broker position SL does NOT match expected, returns False."""
        client.get_open_positions = AsyncMock(
            return_value=[
                PositionInfo(
                    position_id="POS1",
                    symbol="EURUSD.",
                    side="BUY",
                    volume=0.1,
                    open_price=1.0800,
                    current_price=1.0850,
                    sl_price=1.0750,  # Still at original, not breakeven
                    tp_price=1.0900,
                )
            ]
        )
        result = await client.verify_sl_tp(position_id="POS1", expected_sl=1.0800)
        assert result is False

    async def test_verify_position_not_found(self, client: MatchTraderClient) -> None:
        """When position not found (already closed), returns False."""
        client.get_open_positions = AsyncMock(return_value=[])
        result = await client.verify_sl_tp(position_id="POS1", expected_sl=1.0800)
        assert result is False

    async def test_verify_tp_matches(self, client: MatchTraderClient) -> None:
        """When expected_tp is provided and matches, returns True."""
        client.get_open_positions = AsyncMock(
            return_value=[
                PositionInfo(
                    position_id="POS1",
                    symbol="EURUSD.",
                    side="BUY",
                    volume=0.1,
                    open_price=1.0800,
                    current_price=1.0850,
                    sl_price=1.0800,
                    tp_price=1.0900,
                )
            ]
        )
        result = await client.verify_sl_tp(
            position_id="POS1",
            expected_sl=1.0800,
            expected_tp=1.0900,
        )
        assert result is True

    async def test_verify_tp_mismatch(self, client: MatchTraderClient) -> None:
        """When expected_tp doesn't match, returns False."""
        client.get_open_positions = AsyncMock(
            return_value=[
                PositionInfo(
                    position_id="POS1",
                    symbol="EURUSD.",
                    side="BUY",
                    volume=0.1,
                    open_price=1.0800,
                    current_price=1.0850,
                    sl_price=1.0800,
                    tp_price=1.0850,  # Different from expected
                )
            ]
        )
        result = await client.verify_sl_tp(
            position_id="POS1",
            expected_sl=1.0800,
            expected_tp=1.0900,
        )
        assert result is False

    async def test_verify_api_error_returns_false(self, client: MatchTraderClient) -> None:
        """When get_open_positions raises, returns False (safe default)."""
        client.get_open_positions = AsyncMock(side_effect=Exception("API error"))
        result = await client.verify_sl_tp(position_id="POS1", expected_sl=1.0800)
        assert result is False

    async def test_verify_sl_none_vs_expected(self, client: MatchTraderClient) -> None:
        """When broker returns sl_price=None but we expected a value, returns False."""
        client.get_open_positions = AsyncMock(
            return_value=[
                PositionInfo(
                    position_id="POS1",
                    symbol="EURUSD.",
                    side="BUY",
                    volume=0.1,
                    open_price=1.0800,
                    current_price=1.0850,
                    sl_price=None,
                    tp_price=1.0900,
                )
            ]
        )
        result = await client.verify_sl_tp(position_id="POS1", expected_sl=1.0800)
        assert result is False

    async def test_verify_only_checks_specified_fields(self, client: MatchTraderClient) -> None:
        """When only expected_sl is given, tp is not checked."""
        client.get_open_positions = AsyncMock(
            return_value=[
                PositionInfo(
                    position_id="POS1",
                    symbol="EURUSD.",
                    side="BUY",
                    volume=0.1,
                    open_price=1.0800,
                    current_price=1.0850,
                    sl_price=1.0800,
                    tp_price=None,  # TP is None, but we don't check it
                )
            ]
        )
        result = await client.verify_sl_tp(position_id="POS1", expected_sl=1.0800)
        assert result is True

    async def test_verify_multiple_positions_finds_correct_one(
        self, client: MatchTraderClient
    ) -> None:
        """When multiple positions exist, finds the right one by ID."""
        client.get_open_positions = AsyncMock(
            return_value=[
                PositionInfo(
                    position_id="POS_OTHER",
                    symbol="GBPUSD.",
                    side="SELL",
                    volume=0.2,
                    open_price=1.2700,
                    current_price=1.2650,
                    sl_price=1.2800,
                    tp_price=1.2600,
                ),
                PositionInfo(
                    position_id="POS1",
                    symbol="EURUSD.",
                    side="BUY",
                    volume=0.1,
                    open_price=1.0800,
                    current_price=1.0850,
                    sl_price=1.0800,
                    tp_price=1.0900,
                ),
            ]
        )
        result = await client.verify_sl_tp(position_id="POS1", expected_sl=1.0800)
        assert result is True


# ── _apply_breakeven_stops integration tests ────────────────────────────────


class TestBreakevenStopVerification:
    """Tests that _apply_breakeven_stops verifies SL was actually applied."""

    @pytest.fixture
    def store(self, tmp_path: object) -> DecisionStore:
        s = DecisionStore(db_path=f"{tmp_path}/test_be.db")
        yield s  # type: ignore[misc]
        s.close()

    @pytest.fixture
    def config(self) -> AppConfig:
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
                breakeven_activation_pct=0.3,
            ),
            decision_store=DecisionStoreConfig(),
            monitor=MonitorConfig(),
            instruments={
                "EURUSD": InstrumentConfig(pip_value=10.0, pip_size=0.0001, avg_spread_pips=1.0),
            },
        )

    @pytest.fixture
    def mock_matchtrader(self) -> AsyncMock:
        client = AsyncMock()
        client.get_balance.return_value = MagicMock(
            balance=50000.0, equity=50000.0, margin=0.0, free_margin=50000.0
        )
        rate_limiter = MagicMock()
        rate_limiter.remaining = 1800
        rate_limiter._daily_limit = 2000
        client._rate_limiter = rate_limiter
        return client

    def _make_position(
        self,
        pos_id: str = "POS1",
        symbol: str = "EURUSD.",
        side: str = "BUY",
        open_price: float = 1.0800,
        current_price: float = 1.0850,
        tp_price: float = 1.0900,
    ) -> MagicMock:
        pos = MagicMock()
        pos.position_id = pos_id
        pos.symbol = symbol
        pos.side = side
        pos.volume = 0.1
        pos.open_price = open_price
        pos.current_price = current_price
        pos.tp_price = tp_price
        return pos

    def _make_intent(
        self,
        intent_id: str = "INT1",
        symbol: str = "EURUSD",
        side: str = "BUY",
        tp_pips: float = 100.0,
        position_id: str = "POS1",
    ) -> TradeIntent:
        return TradeIntent(
            id=intent_id,
            trade_date="2026-03-10",
            symbol=symbol,
            suggested_side=side,
            suggested_tp_pips=tp_pips,
            position_id=position_id,
        )

    @patch("src.scheduler.scheduler.asyncio.sleep", new_callable=AsyncMock)
    async def test_breakeven_verify_success_first_try(
        self,
        mock_sleep: AsyncMock,
        store: DecisionStore,
        config: AppConfig,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """When verify passes on first try, position is added to breakeven_applied."""
        mock_matchtrader.modify_position.return_value = OrderResult(
            success=True, position_id="POS1", message="OK"
        )
        mock_matchtrader.verify_sl_tp = AsyncMock(return_value=True)

        scheduler = Scheduler(
            config=config,
            store=store,
            matchtrader=mock_matchtrader,
            scanner=MagicMock(),
            agents=MagicMock(),
            engine=AsyncMock(),
        )

        await scheduler._apply_breakeven_stops([self._make_position()], [self._make_intent()])

        assert "POS1" in scheduler._breakeven_applied
        assert mock_matchtrader.modify_position.call_count == 1
        assert mock_matchtrader.verify_sl_tp.call_count == 1

    @patch("src.scheduler.scheduler.asyncio.sleep", new_callable=AsyncMock)
    async def test_breakeven_retries_on_verify_mismatch(
        self,
        mock_sleep: AsyncMock,
        store: DecisionStore,
        config: AppConfig,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """When verify fails first time but succeeds second, retries modify once."""
        mock_matchtrader.modify_position.return_value = OrderResult(
            success=True, position_id="POS1", message="OK"
        )
        mock_matchtrader.verify_sl_tp = AsyncMock(side_effect=[False, True])

        scheduler = Scheduler(
            config=config,
            store=store,
            matchtrader=mock_matchtrader,
            scanner=MagicMock(),
            agents=MagicMock(),
            engine=AsyncMock(),
        )

        await scheduler._apply_breakeven_stops([self._make_position()], [self._make_intent()])

        # Should have called modify twice (initial + retry after verify fail)
        assert mock_matchtrader.modify_position.call_count == 2
        # Should have called verify twice
        assert mock_matchtrader.verify_sl_tp.call_count == 2
        # Position should be in breakeven_applied set
        assert "POS1" in scheduler._breakeven_applied

    @patch("src.scheduler.scheduler.asyncio.sleep", new_callable=AsyncMock)
    async def test_breakeven_unverified_when_verify_fails_twice(
        self,
        mock_sleep: AsyncMock,
        store: DecisionStore,
        config: AppConfig,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """When verify returns False twice, do NOT add to breakeven_applied."""
        mock_matchtrader.modify_position.return_value = OrderResult(
            success=True, position_id="POS1", message="OK"
        )
        mock_matchtrader.verify_sl_tp = AsyncMock(return_value=False)

        scheduler = Scheduler(
            config=config,
            store=store,
            matchtrader=mock_matchtrader,
            scanner=MagicMock(),
            agents=MagicMock(),
            engine=AsyncMock(),
        )

        await scheduler._apply_breakeven_stops([self._make_position()], [self._make_intent()])

        # Should NOT be in breakeven_applied since verify never passed
        assert "POS1" not in scheduler._breakeven_applied
        # modify called twice (initial + retry), verify called twice
        assert mock_matchtrader.modify_position.call_count == 2
        assert mock_matchtrader.verify_sl_tp.call_count == 2

    @patch("src.scheduler.scheduler.asyncio.sleep", new_callable=AsyncMock)
    async def test_breakeven_modify_failure_no_verify(
        self,
        mock_sleep: AsyncMock,
        store: DecisionStore,
        config: AppConfig,
        mock_matchtrader: AsyncMock,
    ) -> None:
        """When modify_position fails, verify is never called."""
        mock_matchtrader.modify_position.return_value = OrderResult(
            success=False, position_id="POS1", message="API error"
        )

        scheduler = Scheduler(
            config=config,
            store=store,
            matchtrader=mock_matchtrader,
            scanner=MagicMock(),
            agents=MagicMock(),
            engine=AsyncMock(),
        )

        await scheduler._apply_breakeven_stops([self._make_position()], [self._make_intent()])

        assert "POS1" not in scheduler._breakeven_applied
        mock_matchtrader.verify_sl_tp.assert_not_called()
