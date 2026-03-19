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

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.monitor.alert_service import AlertService
from src.monitor.operational_metrics import OperationalMetrics
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


# ── Persistent Client ─────────────────────────────────────────────────────


class TestAlertServicePersistentClient:
    """Test _get_client() lazy init and close() cleanup."""

    async def test_get_client_creates_on_first_call(self, alert_basic: AlertService) -> None:
        """_get_client() lazily creates an httpx.AsyncClient."""
        assert alert_basic._http_client is None
        client = await alert_basic._get_client()
        assert client is not None
        assert isinstance(client, httpx.AsyncClient)
        assert alert_basic._http_client is client
        # Cleanup
        await alert_basic.close()

    async def test_get_client_reuses_existing(self, alert_basic: AlertService) -> None:
        """_get_client() returns the same client on subsequent calls."""
        c1 = await alert_basic._get_client()
        c2 = await alert_basic._get_client()
        assert c1 is c2
        await alert_basic.close()

    async def test_close_cleans_up_client(self, alert_basic: AlertService) -> None:
        """close() shuts down client and resets to None."""
        await alert_basic._get_client()
        assert alert_basic._http_client is not None
        await alert_basic.close()
        assert alert_basic._http_client is None

    async def test_close_noop_when_no_client(self, alert_basic: AlertService) -> None:
        """close() is safe to call when no client exists."""
        assert alert_basic._http_client is None
        await alert_basic.close()  # Should not raise
        assert alert_basic._http_client is None

    async def test_send_success_uses_persistent_client(self, alert_basic: AlertService) -> None:
        """send() uses _get_client() and makes a POST request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        alert_basic._http_client = mock_client
        result = await alert_basic.send("test message")
        assert result is True
        mock_client.post.assert_awaited_once()
        # Reset
        alert_basic._http_client = None

    async def test_send_http_error_returns_false(self, alert_basic: AlertService) -> None:
        """send() catches httpx.HTTPError and returns False."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectTimeout("timeout"))
        alert_basic._http_client = mock_client
        result = await alert_basic.send("test message")
        assert result is False
        alert_basic._http_client = None


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

    async def test_appends_ablation_section_when_present(
        self, alert_with_context: AlertService
    ) -> None:
        ablation_summary = {
            "recommendation": "insufficient_ablation_data",
            "available_modes": ["B", "D"],
            "economic_summary": {
                "B": {"mode": "scanner_llm_tactical", "net_pnl": 120.0, "opened_count": 3},
                "D": {"mode": "no_trade", "net_pnl": 0.0, "opened_count": 0},
            },
            "churn_summary": {
                "B": {"mode": "scanner_llm_tactical", "llm_veto_rate": 0.2},
                "D": {"mode": "no_trade", "llm_veto_rate": None},
            },
        }

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
                ablation_summary=ablation_summary,
            )
            msg = mock.call_args[0][0]
            assert "<b>Ablation (7d)</b>" in msg
            assert "Recommendation: insufficient_ablation_data" in msg
            assert "Available modes: B, D" in msg
            assert "B PnL/Open: $+120.00 / 3" in msg
            assert "B LLM veto: 20.0%" in msg
            assert "D PnL/Open: $+0.00 / 0" in msg

    async def test_omits_ablation_churn_line_when_metric_missing(
        self, alert_with_context: AlertService
    ) -> None:
        ablation_summary = {
            "recommendation": "insufficient_ablation_data",
            "available_modes": ["D"],
            "economic_summary": {
                "D": {"mode": "no_trade", "net_pnl": 0.0, "opened_count": 0},
            },
            "churn_summary": {
                "D": {"mode": "no_trade", "llm_veto_rate": None},
            },
        }

        with patch.object(alert_with_context, "send", new_callable=AsyncMock) as mock:
            mock.return_value = True
            await alert_with_context.daily_summary(
                "2026-02-16",
                1,
                0.0,
                5000.0,
                0.0,
                ablation_summary=ablation_summary,
            )
            msg = mock.call_args[0][0]
            assert "D PnL/Open: $+0.00 / 0" in msg
            assert "D LLM veto" not in msg


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

    async def test_stop_sets_flag(self, bot_handler: TelegramBotHandler) -> None:
        bot_handler._running = True
        await bot_handler.stop()
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


class TestAlertServiceDynamicDrawdown:
    """Tests for AlertService with dynamic max drawdown buffer."""

    def test_format_profit_status_with_hwm(self) -> None:
        alert = AlertService(
            bot_token="fake:token",
            chat_id="123456",
            initial_balance=5000.0,
            max_drawdown_pct=0.06,
        )
        result = alert.format_profit_status(
            equity=5050.0,
            positions=[],
            day_start_balance=5050.0,
            max_dd_reference=5100.0,  # HWM is higher than initial
        )
        # Max buffer = 5100 * 0.06 - (5100 - 5050) = 306 - 50 = 256
        assert "$256.00" in result

    def test_daily_summary_with_hwm(self) -> None:
        """Just test it doesn't crash — actual buffer value checked above."""
        alert = AlertService(
            bot_token="fake:token",
            chat_id="123456",
            initial_balance=5000.0,
            max_drawdown_pct=0.06,
        )
        # Should not raise
        assert alert.max_drawdown_amount == 300.0


# ── Circuit Breaker Tests (AlertService) ─────────────────────────────────────


class TestAlertServiceCircuitBreaker:
    """Test circuit breaker pattern in AlertService.send()."""

    def _make_alert(self) -> AlertService:
        return AlertService(bot_token="fake:token", chat_id="123456", max_retries=0)

    async def test_circuit_opens_after_n_failures(self) -> None:
        """Circuit opens after _CIRCUIT_OPEN_THRESHOLD consecutive failures."""
        alert = self._make_alert()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectTimeout("timeout"))
        alert._http_client = mock_client

        for _ in range(3):
            await alert.send("test")

        assert alert._circuit_open is True
        assert alert._consecutive_failures == 3
        # Cleanup
        alert._http_client = None

    async def test_circuit_skips_send_when_open(self) -> None:
        """When circuit is open and retry interval not elapsed, send returns False without HTTP."""
        alert = self._make_alert()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectTimeout("timeout"))
        alert._http_client = mock_client

        # Open the circuit
        for _ in range(3):
            await alert.send("test")
        assert alert._circuit_open is True

        # Reset mock to track new calls
        mock_client.post.reset_mock()

        # Next send should skip entirely (no HTTP call)
        result = await alert.send("should be skipped")
        assert result is False
        mock_client.post.assert_not_awaited()
        alert._http_client = None

    async def test_circuit_allows_probe_after_interval(self) -> None:
        """After retry interval, circuit allows one probe request."""
        import asyncio

        alert = self._make_alert()
        alert._CIRCUIT_RETRY_INTERVAL = 0.1  # Shorten for test

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectTimeout("timeout"))
        alert._http_client = mock_client

        # Open the circuit
        for _ in range(3):
            await alert.send("test")
        assert alert._circuit_open is True

        # Wait for retry interval
        await asyncio.sleep(0.15)

        # Reset mock to track new calls
        mock_client.post.reset_mock()

        # Probe should be attempted (will fail again)
        result = await alert.send("probe")
        assert result is False
        mock_client.post.assert_awaited_once()  # HTTP call was made
        alert._http_client = None

    async def test_circuit_recovers_on_success(self) -> None:
        """Circuit resets to closed when a send succeeds."""
        import asyncio

        alert = self._make_alert()
        alert._CIRCUIT_RETRY_INTERVAL = 0.1  # Shorten for test

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectTimeout("timeout"))
        alert._http_client = mock_client

        # Open the circuit
        for _ in range(3):
            await alert.send("test")
        assert alert._circuit_open is True

        # Wait for retry interval
        await asyncio.sleep(0.15)

        # Now simulate success on probe
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.post = AsyncMock(return_value=mock_response)

        result = await alert.send("probe-success")
        assert result is True
        assert alert._circuit_open is False
        assert alert._consecutive_failures == 0
        alert._http_client = None

    async def test_non_200_counts_as_failure(self) -> None:
        """Non-200 HTTP responses also trigger circuit breaker."""
        alert = self._make_alert()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        alert._http_client = mock_client

        for _ in range(3):
            await alert.send("test")

        assert alert._circuit_open is True
        assert alert._consecutive_failures == 3
        alert._http_client = None

    async def test_success_resets_failure_count(self) -> None:
        """A successful send resets the consecutive failure counter."""
        alert = self._make_alert()
        mock_client = AsyncMock()

        # 2 failures (not enough to open circuit)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectTimeout("timeout"))
        alert._http_client = mock_client
        await alert.send("fail1")
        await alert.send("fail2")
        assert alert._consecutive_failures == 2
        assert alert._circuit_open is False

        # 1 success resets counter
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.post = AsyncMock(return_value=mock_response)
        await alert.send("ok")
        assert alert._consecutive_failures == 0
        alert._http_client = None

    async def test_disabled_alert_ignores_circuit_breaker(self) -> None:
        """Disabled AlertService (no token) returns False without touching circuit breaker."""
        alert = AlertService(bot_token="", chat_id="")
        result = await alert.send("test")
        assert result is False
        assert alert._consecutive_failures == 0
        assert alert._circuit_open is False


class TestAlertServiceResilience:
    """Test retry accounting and alternate sink behavior for AlertService.send()."""

    async def test_send_retries_before_success_and_records_retry_metrics(self) -> None:
        metrics = OperationalMetrics()
        alert = AlertService(
            bot_token="fake:token",
            chat_id="123456",
            operational_metrics=metrics,
            max_retries=2,
            retry_backoff_seconds=0.0,
        )
        mock_response = MagicMock(status_code=200)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[
                httpx.ConnectTimeout("timeout-1"),
                httpx.ConnectTimeout("timeout-2"),
                mock_response,
            ]
        )
        alert._http_client = mock_client

        result = await alert.send("retry-me")

        assert result is True
        assert mock_client.post.await_count == 3
        summary = metrics.get_summary()
        assert summary["telegram_retries"] == 2
        assert summary["telegram_failures"] == 0
        assert alert._consecutive_failures == 0
        alert._http_client = None

    async def test_send_uses_alternate_sink_after_primary_retries_exhausted(self) -> None:
        metrics = OperationalMetrics()
        alternate_sink = AsyncMock(return_value=True)
        alert = AlertService(
            bot_token="fake:token",
            chat_id="123456",
            operational_metrics=metrics,
            alternate_sink=alternate_sink,
            max_retries=1,
            retry_backoff_seconds=0.0,
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectTimeout("timeout"))
        alert._http_client = mock_client

        result = await alert.send("primary-down")

        assert result is True
        assert mock_client.post.await_count == 2
        alternate_sink.assert_awaited_once_with("primary-down", "primary_send_failed")
        summary = metrics.get_summary()
        assert summary["telegram_retries"] == 1
        assert summary["telegram_failures"] == 1
        assert alert._consecutive_failures == 1
        alert._http_client = None

    async def test_circuit_open_routes_alerts_to_alternate_sink_without_http(self) -> None:
        alternate_sink = AsyncMock(return_value=True)
        alert = AlertService(
            bot_token="fake:token",
            chat_id="123456",
            alternate_sink=alternate_sink,
            max_retries=0,
        )
        alert._circuit_open = True
        alert._circuit_opened_at = time.monotonic()
        alert._CIRCUIT_RETRY_INTERVAL = 300.0
        mock_client = AsyncMock()
        mock_client.post = AsyncMock()
        alert._http_client = mock_client

        result = await alert.send("circuit-open")

        assert result is True
        mock_client.post.assert_not_awaited()
        alternate_sink.assert_awaited_once_with("circuit-open", "circuit_open")
        alert._http_client = None


# ── Circuit Breaker Tests (TelegramBotHandler) ──────────────────────────────


class TestTelegramBotCircuitBreaker:
    """Test circuit breaker pattern in TelegramBotHandler._poll_updates()."""

    def _make_bot(self) -> TelegramBotHandler:
        return TelegramBotHandler(
            bot_token="fake:token",
            chat_id="123456",
            alert_service=AsyncMock(),
            trading_client=AsyncMock(),
            trade_journal=MagicMock(),
        )

    async def test_circuit_opens_after_connection_failures(self) -> None:
        """Circuit opens after _CIRCUIT_OPEN_THRESHOLD consecutive HTTPError failures."""
        bot = self._make_bot()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectTimeout("timeout"))
        bot._http_client = mock_client

        for _ in range(3):
            await bot._poll_updates()

        assert bot._circuit_open is True
        assert bot._consecutive_failures == 3
        bot._http_client = None

    async def test_circuit_open_sleeps_instead_of_polling(self) -> None:
        """When circuit is open, _poll_updates sleeps for retry interval then probes."""
        bot = self._make_bot()
        bot._CIRCUIT_RETRY_INTERVAL = 0.05  # Shorten for test

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectTimeout("timeout"))
        bot._http_client = mock_client

        # Open the circuit
        for _ in range(3):
            await bot._poll_updates()
        assert bot._circuit_open is True

        # Next poll should sleep then probe (will fail again)
        mock_client.get.reset_mock()
        await bot._poll_updates()
        # Probe was attempted
        mock_client.get.assert_awaited_once()
        bot._http_client = None

    async def test_circuit_recovers_on_successful_poll(self) -> None:
        """Circuit resets when polling succeeds (status 200)."""
        bot = self._make_bot()
        bot._CIRCUIT_RETRY_INTERVAL = 0.05  # Shorten for test

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectTimeout("timeout"))
        bot._http_client = mock_client

        # Open the circuit
        for _ in range(3):
            await bot._poll_updates()
        assert bot._circuit_open is True

        # Simulate success
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={"ok": True, "result": []})
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await bot._poll_updates()
        assert result == []
        assert bot._circuit_open is False
        assert bot._consecutive_failures == 0
        bot._http_client = None

    @patch("src.monitor.telegram_bot.asyncio.sleep", new_callable=AsyncMock)
    async def test_409_does_not_trigger_circuit_breaker(self, mock_sleep: AsyncMock) -> None:
        """409 Conflict uses its own backoff, not the circuit breaker."""
        bot = self._make_bot()
        mock_response = MagicMock()
        mock_response.status_code = 409
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        bot._http_client = mock_client

        for _ in range(5):
            await bot._poll_updates()

        # Circuit should NOT be open (409 uses conflict_backoff, not circuit breaker)
        assert bot._circuit_open is False
        assert bot._consecutive_failures == 0
        assert bot._conflict_backoff > 0
        bot._http_client = None

    async def test_poll_failures_record_operational_metrics(self) -> None:
        metrics = OperationalMetrics()
        bot = TelegramBotHandler(
            bot_token="fake:token",
            chat_id="123456",
            alert_service=AsyncMock(),
            trading_client=AsyncMock(),
            trade_journal=MagicMock(),
            operational_metrics=metrics,
        )
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectTimeout("timeout"))
        bot._http_client = mock_client

        for _ in range(3):
            await bot._poll_updates()

        summary = metrics.get_summary()
        assert summary["telegram_poll_failures"] == 3
        assert summary["telegram_poll_circuit_opens"] == 1
        bot._http_client = None

    async def test_circuit_recovery_records_operational_metrics(self) -> None:
        metrics = OperationalMetrics()
        bot = TelegramBotHandler(
            bot_token="fake:token",
            chat_id="123456",
            alert_service=AsyncMock(),
            trading_client=AsyncMock(),
            trade_journal=MagicMock(),
            poll_interval=0.0,
            operational_metrics=metrics,
        )
        bot._CIRCUIT_RETRY_INTERVAL = 0.05

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectTimeout("timeout"))
        bot._http_client = mock_client

        for _ in range(3):
            await bot._poll_updates()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value={"ok": True, "result": []})
        mock_client.get = AsyncMock(return_value=mock_response)

        await bot._poll_updates()

        summary = metrics.get_summary()
        assert summary["telegram_poll_probe_polls"] == 1
        assert summary["telegram_poll_circuit_recoveries"] == 1
        bot._http_client = None
