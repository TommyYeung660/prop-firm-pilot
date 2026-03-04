"""
Telegram bot command handler — listens for /profit and /orders commands.

Polls Telegram getUpdates API and dispatches commands to provide
real-time account status and trade history via the bot.

Usage:
    bot = TelegramBotHandler(
        bot_token="123:ABC",
        chat_id="6385786935",
        alert_service=alert_service,
        trading_client=matchtrader_client,
        trade_journal=trade_journal,
    )
    await bot.start()  # Runs until stopped
    await bot.stop()
"""

import asyncio
import time
from typing import Any

import httpx
from loguru import logger


class TelegramBotHandler:
    """Async polling handler for Telegram bot commands (/profit, /orders).

    Uses getUpdates long-polling to listen for incoming commands and
    dispatches them to the appropriate handlers.

    Usage:
        handler = TelegramBotHandler(
            bot_token="123:ABC",
            chat_id="6385786935",
            alert_service=alert_service,
            trading_client=matchtrader_client,
            trade_journal=trade_journal,
        )
        await handler.start()
    """

    TELEGRAM_API = "https://api.telegram.org"

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        alert_service: Any,
        trading_client: Any,
        trade_journal: Any,
        poll_interval: float = 1.0,
    ) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._alert_service = alert_service
        self._trading_client = trading_client
        self._trade_journal = trade_journal
        self._poll_interval = poll_interval
        self._offset = 0
        self._conflict_backoff = 0.0
        self._running = False
        self._http_client: httpx.AsyncClient | None = None
        self._enabled = bool(bot_token and chat_id)

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the command polling loop.

        Runs indefinitely until stop() is called. Only processes messages
        from the configured chat_id for security.
        """
        if not self._enabled:
            logger.warning("TelegramBotHandler: not configured, skipping start")
            return

        self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))
        self._running = True
        logger.info("TelegramBotHandler: started polling for commands")

        while self._running:
            try:
                updates = await self._poll_updates()
                for update in updates:
                    update_id = update.get("update_id", 0)
                    message = update.get("message", {})
                    text = message.get("text", "")
                    chat = message.get("chat", {})
                    msg_chat_id = str(chat.get("id", ""))

                    # Security: only process messages from configured chat
                    if msg_chat_id != self._chat_id:
                        logger.debug(
                            "TelegramBotHandler: ignoring message from chat {}",
                            msg_chat_id,
                        )
                        self._offset = update_id + 1
                        continue

                    if text.startswith("/"):
                        command = text.split()[0].lower()
                        # Strip @botname suffix (e.g. /profit@mybot)
                        if "@" in command:
                            command = command.split("@")[0]
                        await self._handle_command(command)

                    self._offset = update_id + 1

            except asyncio.CancelledError:
                logger.info("TelegramBotHandler: cancelled, exiting")
                break
            except Exception as e:
                logger.error("TelegramBotHandler: poll error: {}", e)
            try:
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                logger.info("TelegramBotHandler: sleep cancelled, exiting")
                break

        logger.info("TelegramBotHandler: stopped")

    async def stop(self) -> None:
        """Signal the polling loop to stop and close the HTTP client."""
        self._running = False
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.info("TelegramBotHandler: stop requested")

    @property
    def is_running(self) -> bool:
        """Whether the polling loop is currently running."""
        return self._running

    # ── Polling ─────────────────────────────────────────────────────────

    async def _poll_updates(self) -> list[dict[str, Any]]:
        """Poll Telegram getUpdates API for new messages.

        Uses long-polling with a 10-second timeout to minimize API calls.
        On 409 Conflict (another instance polling the same bot token),
        applies exponential backoff to avoid thrashing.
        """
        # M1: Back off when a 409 conflict was recently detected
        if self._conflict_backoff > 0:
            logger.debug(
                "TelegramBotHandler: conflict backoff {:.0f}s",
                self._conflict_backoff,
            )
            await asyncio.sleep(self._conflict_backoff)

        url = (
            f"{self.TELEGRAM_API}/bot{self._bot_token}/getUpdates?offset={self._offset}&timeout=10"
        )
        try:
            client = self._http_client
            if client is None:
                logger.warning("TelegramBotHandler: HTTP client not initialized")
                return []
            response = await client.get(url)
            if response.status_code == 200:
                self._conflict_backoff = 0.0  # Reset on success
                data = response.json()
                return data.get("result", [])
            if response.status_code == 409:
                # M1: Another instance is polling this bot token
                self._conflict_backoff = min(
                    max(self._conflict_backoff * 2, 5.0), 120.0
                )
                logger.warning(
                    "TelegramBotHandler: 409 Conflict — another instance is polling"
                    " this bot token. Backing off {:.0f}s",
                    self._conflict_backoff,
                )
                return []
            logger.warning(
                "TelegramBotHandler: getUpdates returned {}",
                response.status_code,
            )
            return []
        except httpx.HTTPError as e:
            logger.warning(
                "TelegramBotHandler: getUpdates failed: {} ({})",
                type(e).__name__,
                e or "no details",
            )
            return []

    # ── Command Dispatch ────────────────────────────────────────────────

    async def _handle_command(self, command: str) -> None:
        """Dispatch a command to the appropriate handler."""
        logger.info("TelegramBotHandler: received command: {}", command)
        if command == "/profit":
            await self._cmd_profit()
        elif command == "/orders":
            await self._cmd_orders()
        elif command in ("/start", "/help"):
            await self._cmd_help()
        else:
            await self._send_reply("❓ Unknown command. Try /profit or /orders")

    # ── Command Handlers ────────────────────────────────────────────────

    async def _cmd_profit(self) -> None:
        """Handle /profit — show positions and profit target progress."""
        try:
            balance = await self._trading_client.get_balance()
            positions = await self._trading_client.get_open_positions()
            pos_dicts = [p.model_dump() for p in positions]

            message = self._alert_service.format_profit_status(
                equity=balance.equity,
                positions=pos_dicts,
                day_start_balance=balance.balance,
            )
            await self._send_reply(message)
        except Exception as e:
            logger.error("TelegramBotHandler: /profit failed: {}", e)
            await self._send_reply(f"❌ Failed to fetch profit status: {e}")

    async def _cmd_orders(self) -> None:
        """Handle /orders — show last 10 closed trades and open positions."""
        try:
            # Get open positions
            open_positions = await self._trading_client.get_open_positions()
            open_dicts = [p.model_dump() for p in open_positions]

            # Get closed positions for last 7 days
            now_ms = int(time.time() * 1000)
            week_ago_ms = now_ms - (7 * 24 * 60 * 60 * 1000)
            closed = await self._trading_client.get_closed_positions(
                from_ts=week_ago_ms, to_ts=now_ms
            )
            closed_dicts = [p.model_dump() for p in closed]

            # Import at usage to avoid circular import
            from src.monitor.alert_service import AlertService

            message = AlertService.format_orders_list(
                closed_trades=closed_dicts,
                open_positions=open_dicts,
            )
            await self._send_reply(message)
        except Exception as e:
            logger.error("TelegramBotHandler: /orders failed: {}", e)
            await self._send_reply(f"❌ Failed to fetch orders: {e}")

    async def _cmd_help(self) -> None:
        """Handle /start and /help — show available commands."""
        msg = (
            "🤖 <b>PropFirmPilot Bot</b>\n\n"
            "Available commands:\n"
            "/profit — Current positions & profit target progress\n"
            "/orders — Last 10 closed trades & open positions\n"
            "/help — Show this message"
        )
        await self._send_reply(msg)

    # ── Reply Helper ────────────────────────────────────────────────────

    async def _send_reply(self, text: str) -> None:
        """Send a reply message via the alert service."""
        await self._alert_service.send(text)
