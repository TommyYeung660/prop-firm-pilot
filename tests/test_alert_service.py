"""
Tests for src/monitor/alert_service.py and src/monitor/telegram_bot.py.

Tests cover:
- AlertService enhanced notifications (trade_opened, trade_closed, sl_tp_hit)
- AlertService daily_summary with profit target progress
- AlertService drawdown_warning with daily loss buffer
- AlertService format_profit_status and format_orders_list
- AlertService helper methods (_progress_bar, _profit_progress, _account_header)
- AlertService backward compatibility with original signatures
- TelegramBotHandler command parsing and dispatching
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.monitor.alert_service import AlertService
from src.monitor.telegram_bot import TelegramBotHandler

# ── AlertService Fixtures ───────────────────────────────────────────────────


@pytest.fixture
def alert_basic() -> AlertService:
    """AlertService with no account context (backward-compatible mode)."""
    return AlertService(bot_token="fake:token", chat_id="123456")


@pytest.fixture
def alert_with_context() -> AlertService:
    """AlertService with full account context for Trial account."""
    return AlertService(
        bot_token="fake:token",
        chat_id="123456",
        account_id="950552",
        initial_balance=5000.0,
        profit_target_pct=0.06,
        daily_loss_pct=0.02,
        max_drawdown_pct=0.04,
    )


@pytest.fixture
def alert_disabled() -> AlertService:
    """AlertService with no credentials (disabled)."""
    return AlertService(bot_token="", chat_id="")


# ── Computed Properties ─────────────────────────────────────────────────────


class TestAlertServiceProperties:
    """Test computed properties for account context."""

    def test_profit_target_amount(self, alert_with_context: AlertService) -> None:
        assert alert_with_context.profit_target_amount == 300.0  # 5000 * 0.06

    def test_daily_loss_amount(self, alert_with_context: AlertService) -> None:
        assert alert_with_context.daily_loss_amount == 100.0  # 5000 * 0.02

    def test_max_drawdown_amount(self, alert_with_context: AlertService) -> None:
        assert alert_with_context.max_drawdown_amount == 200.0  # 5000 * 0.04

    def test_zero_when_no_context(self, alert_basic: AlertService) -> None:
        assert alert_basic.profit_target_amount == 0.0
        assert alert_basic.daily_loss_amount == 0.0
        assert alert_basic.max_drawdown_amount == 0.0


# ── Helpers ─────────────────────────────────────────────────────────────────


class TestAlertServiceHelpers:
    """Test private helper methods."""

    def test_account_header_with_id(self, alert_with_context: AlertService) -> None:
        assert alert_with_context._account_header() == "[950552] "

    def test_account_header_without_id(self, alert_basic: AlertService) -> None:
        assert alert_basic._account_header() == ""

    def test_progress_bar_zero(self, alert_with_context: AlertService) -> None:
        bar = alert_with_context._progress_bar(0.0)
        assert bar == "[░░░░░░░░░░░░░░░░░░░░] 0.0%"

    def test_progress_bar_half(self, alert_with_context: AlertService) -> None:
        bar = alert_with_context._progress_bar(0.5)
        assert "██████████" in bar
        assert "50.0%" in bar

    def test_progress_bar_full(self, alert_with_context: AlertService) -> None:
        bar = alert_with_context._progress_bar(1.0)
        assert bar == "[████████████████████] 100.0%"

    def test_progress_bar_clamped_over(self, alert_with_context: AlertService) -> None:
        bar = alert_with_context._progress_bar(1.5)
        assert "100.0%" in bar

    def test_progress_bar_clamped_under(self, alert_with_context: AlertService) -> None:
        bar = alert_with_context._progress_bar(-0.5)
        assert "0.0%" in bar

    def test_profit_progress_with_target(self, alert_with_context: AlertService) -> None:
        result = alert_with_context._profit_progress(5150.0)
        assert "Profit Target" in result
        assert "$300.00" in result  # target amount
        assert "$+150.00" in result  # current PnL
        assert "$150.00" in result  # remaining

    def test_profit_progress_no_target(self, alert_basic: AlertService) -> None:
        result = alert_basic._profit_progress(5000.0)
        assert result == ""


# ── Send (disabled mode) ───────────────────────────────────────────────────


class TestAlertServiceSend:
    """Test send() in disabled mode (no HTTP calls)."""

    async def test_send_disabled_returns_false(self, alert_disabled: AlertService) -> None:
        result = await alert_disabled.send("test message")
        assert result is False


# ── Trade Opened ────────────────────────────────────────────────────────────


class TestTradeOpened:
    """Test trade_opened() notifications."""

    async def test_backward_compatible_call(self, alert_basic: AlertService) -> None:
        """Original 4-arg signature still works."""
        with patch.object(alert_basic, "send", new_callable=AsyncMock) as mock:
            mock.return_value = True
            result = await alert_basic.trade_opened("EURUSD", "BUY", 0.1, 1.08)
            assert result is True
            msg = mock.call_args[0][0]
            assert "Trade Opened" in msg
            assert "BUY" in msg
            assert "EURUSD" in msg

    async def test_with_sl_tp(self, alert_with_context: AlertService) -> None:
        with patch.object(alert_with_context, "send", new_callable=AsyncMock) as mock:
            mock.return_value = True
            await alert_with_context.trade_opened("EURUSD.", "BUY", 0.10, 1.085, sl=1.080, tp=1.090)
            msg = mock.call_args[0][0]
            assert "SL: 1.08" in msg
            assert "TP: 1.09" in msg

    async def test_with_equity_shows_progress(self, alert_with_context: AlertService) -> None:
        with patch.object(alert_with_context, "send", new_callable=AsyncMock) as mock:
            mock.return_value = True
            await alert_with_context.trade_opened("EURUSD.", "BUY", 0.10, 1.085, equity=5100.0)
            msg = mock.call_args[0][0]
            assert "Profit Target" in msg
            assert "$300.00" in msg

    async def test_with_position_id(self, alert_with_context: AlertService) -> None:
        with patch.object(alert_with_context, "send", new_callable=AsyncMock) as mock:
            mock.return_value = True
            await alert_with_context.trade_opened("EURUSD.", "BUY", 0.10, 1.085, position_id="W123")
            msg = mock.call_args[0][0]
            assert "W123" in msg

    async def test_account_header_shown(self, alert_with_context: AlertService) -> None:
        with patch.object(alert_with_context, "send", new_callable=AsyncMock) as mock:
            mock.return_value = True
            await alert_with_context.trade_opened("EURUSD.", "BUY", 0.1, 1.08)
            msg = mock.call_args[0][0]
            assert "[950552]" in msg


# ── Trade Closed ────────────────────────────────────────────────────────────


class TestTradeClosed:
    """Test trade_closed() notifications."""

    async def test_backward_compatible_call(self, alert_basic: AlertService) -> None:
        with patch.object(alert_basic, "send", new_callable=AsyncMock) as mock:
            mock.return_value = True
            result = await alert_basic.trade_closed("EURUSD", "BUY", 25.50, "manual")
            assert result is True
            msg = mock.call_args[0][0]
            assert "Trade Closed" in msg
            assert "✅" in msg
            assert "$+25.50" in msg

    async def test_loss_shows_red_emoji(self, alert_basic: AlertService) -> None:
        with patch.object(alert_basic, "send", new_callable=AsyncMock) as mock:
            mock.return_value = True
            await alert_basic.trade_closed("EURUSD", "BUY", -10.0, "sl")
            msg = mock.call_args[0][0]
            assert "❌" in msg
            assert "$-10.00" in msg

    async def test_with_prices(self, alert_with_context: AlertService) -> None:
        with patch.object(alert_with_context, "send", new_callable=AsyncMock) as mock:
            mock.return_value = True
            await alert_with_context.trade_closed(
                "EURUSD.",
                "BUY",
                25.0,
                "tp",
                volume=0.10,
                open_price=1.085,
                close_price=1.090,
            )
            msg = mock.call_args[0][0]
            assert "0.10 lots" in msg
            assert "1.085" in msg
            assert "1.09" in msg


# ── SL/TP Hit ───────────────────────────────────────────────────────────────


class TestSlTpHit:
    """Test sl_tp_hit() notifications."""

    async def test_stop_loss_hit(self, alert_with_context: AlertService) -> None:
        with patch.object(alert_with_context, "send", new_callable=AsyncMock) as mock:
            mock.return_value = True
            await alert_with_context.sl_tp_hit(
                "EURUSD.",
                "BUY",
                0.10,
                -15.0,
                "SL",
                1.080,
                equity=4985.0,
            )
            msg = mock.call_args[0][0]
            assert "🛑" in msg
            assert "Stop Loss Hit" in msg
            assert "$-15.00" in msg
            assert "1.08" in msg

    async def test_take_profit_hit(self, alert_with_context: AlertService) -> None:
        with patch.object(alert_with_context, "send", new_callable=AsyncMock) as mock:
            mock.return_value = True
            await alert_with_context.sl_tp_hit(
                "EURUSD.",
                "BUY",
                0.10,
                30.0,
                "TP",
                1.090,
                equity=5030.0,
            )
            msg = mock.call_args[0][0]
            assert "🎯" in msg
            assert "Take Profit Hit" in msg
            assert "$+30.00" in msg


# ── Drawdown Warning ───────────────────────────────────────────────────────


class TestDrawdownWarning:
    """Test drawdown_warning() notifications."""

    async def test_backward_compatible(self, alert_basic: AlertService) -> None:
        with patch.object(alert_basic, "send", new_callable=AsyncMock) as mock:
            mock.return_value = True
            result = await alert_basic.drawdown_warning("WARNING", 0.5, 0.3, 49000.0)
            assert result is True
            msg = mock.call_args[0][0]
            assert "Drawdown Alert: WARNING" in msg
            assert "🟡" in msg

    async def test_with_daily_buffer(self, alert_with_context: AlertService) -> None:
        with patch.object(alert_with_context, "send", new_callable=AsyncMock) as mock:
            mock.return_value = True
            await alert_with_context.drawdown_warning(
                "DANGER", 0.8, 0.5, 4920.0, day_start_balance=5000.0
            )
            msg = mock.call_args[0][0]
            assert "Daily loss buffer" in msg
            assert "$20.00 remaining" in msg  # 100 - 80 = 20


# ── Daily Summary ──────────────────────────────────────────────────────────


class TestDailySummary:
    """Test daily_summary() notifications."""

    async def test_backward_compatible(self, alert_basic: AlertService) -> None:
        with patch.object(alert_basic, "send", new_callable=AsyncMock) as mock:
            mock.return_value = True
            result = await alert_basic.daily_summary("2026-02-16", 3, 50.0, 5050.0, 0.2)
            assert result is True
            msg = mock.call_args[0][0]
            assert "Daily Summary" in msg
            assert "2026-02-16" in msg
            assert "$5,050.00" in msg

    async def test_with_profit_progress(self, alert_with_context: AlertService) -> None:
        with patch.object(alert_with_context, "send", new_callable=AsyncMock) as mock:
            mock.return_value = True
            await alert_with_context.daily_summary(
                "2026-02-16",
                5,
                100.0,
                5100.0,
                0.1,
                open_positions=2,
                day_start_balance=5000.0,
            )
            msg = mock.call_args[0][0]
            assert "Profit Target" in msg
            assert "Risk Status" in msg
            assert "Open positions: 2" in msg
            assert "Daily loss buffer" in msg
            assert "Max DD buffer" in msg


# ── Compliance & System Error (backward compat) ────────────────────────────


class TestBackwardCompat:
    """Test that compliance_rejection and system_error are unchanged."""

    async def test_compliance_rejection(self, alert_with_context: AlertService) -> None:
        with patch.object(alert_with_context, "send", new_callable=AsyncMock) as mock:
            mock.return_value = True
            result = await alert_with_context.compliance_rejection(
                "EURUSD.", "BUY", "daily drawdown exceeded"
            )
            assert result is True
            msg = mock.call_args[0][0]
            assert "Compliance Rejected" in msg
            assert "[950552]" in msg

    async def test_system_error(self, alert_with_context: AlertService) -> None:
        with patch.object(alert_with_context, "send", new_callable=AsyncMock) as mock:
            mock.return_value = True
            result = await alert_with_context.system_error("Connection timeout")
            assert result is True
            msg = mock.call_args[0][0]
            assert "System Error" in msg
            assert "Connection timeout" in msg


# ── Format Profit Status ───────────────────────────────────────────────────


class TestFormatProfitStatus:
    """Test format_profit_status() for /profit command."""

    def test_with_positions(self, alert_with_context: AlertService) -> None:
        positions = [
            {
                "symbol": "EURUSD.",
                "side": "BUY",
                "volume": 0.10,
                "profit": 25.0,
            },
            {
                "symbol": "GBPUSD.",
                "side": "SELL",
                "volume": 0.05,
                "profit": -5.0,
            },
        ]
        result = alert_with_context.format_profit_status(
            equity=5020.0,
            positions=positions,
            day_start_balance=5000.0,
        )
        assert "Profit Status" in result
        assert "$5,020.00" in result
        assert "Profit Target" in result
        assert "EURUSD." in result
        assert "GBPUSD." in result
        assert "🟢" in result  # EURUSD profit
        assert "🔴" in result  # GBPUSD loss
        assert "Risk Buffers" in result
        assert "Daily loss buffer" in result

    def test_no_positions(self, alert_with_context: AlertService) -> None:
        result = alert_with_context.format_profit_status(equity=5000.0, positions=[])
        assert "No open positions" in result

    def test_no_account_context(self, alert_basic: AlertService) -> None:
        result = alert_basic.format_profit_status(equity=5000.0, positions=[])
        assert "Profit Status" in result
        # No profit target section when no context
        assert "Profit Target" not in result


# ── Format Orders List ─────────────────────────────────────────────────────


class TestFormatOrdersList:
    """Test format_orders_list() for /orders command."""

    def test_with_closed_trades(self) -> None:
        closed = [
            {
                "symbol": "EURUSD.",
                "side": "BUY",
                "profit": 30.0,
                "close_time": "2026-02-16T10:30:00Z",
            },
            {
                "symbol": "GBPUSD.",
                "side": "SELL",
                "profit": -10.0,
                "close_time": "2026-02-15T14:00:00Z",
            },
        ]
        result = AlertService.format_orders_list(closed_trades=closed)
        assert "Orders" in result
        assert "Closed (last 2)" in result
        assert "EURUSD." in result
        assert "GBPUSD." in result

    def test_with_open_positions(self) -> None:
        open_pos = [
            {
                "symbol": "AUDUSD.",
                "side": "BUY",
                "volume": 0.05,
                "profit": 12.0,
            },
        ]
        result = AlertService.format_orders_list(closed_trades=[], open_positions=open_pos)
        assert "Open (1)" in result
        assert "AUDUSD." in result

    def test_empty(self) -> None:
        result = AlertService.format_orders_list(closed_trades=[])
        assert "No closed trades" in result

    def test_more_than_10_shows_last_10(self) -> None:
        closed = [
            {
                "symbol": f"PAIR{i}",
                "side": "BUY",
                "profit": float(i),
                "close_time": f"2026-02-{i + 1:02d}T10:00:00Z",
            }
            for i in range(15)
        ]
        result = AlertService.format_orders_list(closed_trades=closed)
        assert "last 10" in result


# ── TelegramBotHandler Tests ───────────────────────────────────────────────


@pytest.fixture
def mock_alert_service() -> AsyncMock:
    """Mock AlertService."""
    service = AsyncMock()
    service.send = AsyncMock(return_value=True)
    service.format_profit_status = MagicMock(return_value="profit status msg")
    return service


@pytest.fixture
def mock_trading_client() -> AsyncMock:
    """Mock MatchTraderClient."""
    client = AsyncMock()

    balance = MagicMock()
    balance.equity = 5100.0
    balance.balance = 5000.0
    client.get_balance = AsyncMock(return_value=balance)

    position = MagicMock()
    position.model_dump = MagicMock(
        return_value={
            "symbol": "EURUSD.",
            "side": "BUY",
            "volume": 0.1,
            "profit": 25.0,
        }
    )
    client.get_open_positions = AsyncMock(return_value=[position])
    client.get_closed_positions = AsyncMock(return_value=[])

    return client


@pytest.fixture
def mock_journal() -> MagicMock:
    """Mock TradeJournal."""
    return MagicMock()


@pytest.fixture
def bot_handler(
    mock_alert_service: AsyncMock,
    mock_trading_client: AsyncMock,
    mock_journal: MagicMock,
) -> TelegramBotHandler:
    """TelegramBotHandler with mocked dependencies."""
    return TelegramBotHandler(
        bot_token="fake:token",
        chat_id="123456",
        alert_service=mock_alert_service,
        trading_client=mock_trading_client,
        trade_journal=mock_journal,
    )


class TestTelegramBotHandler:
    """Test TelegramBotHandler command handlers."""

    def test_not_running_initially(self, bot_handler: TelegramBotHandler) -> None:
        assert bot_handler.is_running is False

    def test_stop_sets_flag(self, bot_handler: TelegramBotHandler) -> None:
        bot_handler._running = True
        bot_handler.stop()
        assert bot_handler.is_running is False

    async def test_cmd_profit(
        self,
        bot_handler: TelegramBotHandler,
        mock_trading_client: AsyncMock,
        mock_alert_service: AsyncMock,
    ) -> None:
        await bot_handler._cmd_profit()
        mock_trading_client.get_balance.assert_awaited_once()
        mock_trading_client.get_open_positions.assert_awaited_once()
        mock_alert_service.format_profit_status.assert_called_once()
        mock_alert_service.send.assert_awaited()

    async def test_cmd_orders(
        self,
        bot_handler: TelegramBotHandler,
        mock_trading_client: AsyncMock,
        mock_alert_service: AsyncMock,
    ) -> None:
        await bot_handler._cmd_orders()
        mock_trading_client.get_open_positions.assert_awaited_once()
        mock_trading_client.get_closed_positions.assert_awaited_once()
        mock_alert_service.send.assert_awaited()

    async def test_cmd_help(
        self,
        bot_handler: TelegramBotHandler,
        mock_alert_service: AsyncMock,
    ) -> None:
        await bot_handler._cmd_help()
        mock_alert_service.send.assert_awaited_once()
        msg = mock_alert_service.send.call_args[0][0]
        assert "PropFirmPilot Bot" in msg
        assert "/profit" in msg
        assert "/orders" in msg

    async def test_handle_unknown_command(
        self,
        bot_handler: TelegramBotHandler,
        mock_alert_service: AsyncMock,
    ) -> None:
        await bot_handler._handle_command("/unknown")
        mock_alert_service.send.assert_awaited_once()
        msg = mock_alert_service.send.call_args[0][0]
        assert "Unknown command" in msg

    async def test_handle_profit_command(
        self,
        bot_handler: TelegramBotHandler,
        mock_trading_client: AsyncMock,
    ) -> None:
        await bot_handler._handle_command("/profit")
        mock_trading_client.get_balance.assert_awaited_once()

    async def test_handle_orders_command(
        self,
        bot_handler: TelegramBotHandler,
        mock_trading_client: AsyncMock,
    ) -> None:
        await bot_handler._handle_command("/orders")
        mock_trading_client.get_open_positions.assert_awaited_once()

    async def test_handle_help_command(
        self,
        bot_handler: TelegramBotHandler,
        mock_alert_service: AsyncMock,
    ) -> None:
        await bot_handler._handle_command("/help")
        msg = mock_alert_service.send.call_args[0][0]
        assert "PropFirmPilot Bot" in msg

    async def test_handle_start_command(
        self,
        bot_handler: TelegramBotHandler,
        mock_alert_service: AsyncMock,
    ) -> None:
        await bot_handler._handle_command("/start")
        msg = mock_alert_service.send.call_args[0][0]
        assert "PropFirmPilot Bot" in msg

    async def test_cmd_profit_error_handling(
        self,
        bot_handler: TelegramBotHandler,
        mock_trading_client: AsyncMock,
        mock_alert_service: AsyncMock,
    ) -> None:
        """When trading client fails, error message is sent."""
        mock_trading_client.get_balance = AsyncMock(side_effect=Exception("connection refused"))
        await bot_handler._cmd_profit()
        mock_alert_service.send.assert_awaited()
        msg = mock_alert_service.send.call_args[0][0]
        assert "Failed" in msg

    async def test_disabled_bot_skips_start(self) -> None:
        """Bot with empty credentials should not start polling."""
        handler = TelegramBotHandler(
            bot_token="",
            chat_id="",
            alert_service=AsyncMock(),
            trading_client=AsyncMock(),
            trade_journal=MagicMock(),
        )
        await handler.start()
        assert handler.is_running is False
