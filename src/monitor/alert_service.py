"""
Telegram alert service for prop firm trading notifications.

Sends critical alerts for:
- Drawdown warnings (80%/90% of limits)
- Trade executions
- Compliance rejections
- System errors
"""

from typing import Any, Dict

import httpx
from loguru import logger


class AlertService:
    """Sends trading alerts via Telegram Bot API.

    Usage:
        alerts = AlertService(bot_token="123:ABC", chat_id="-100123")
        await alerts.send("🔴 Daily drawdown at 85%!")
        await alerts.trade_opened("EURUSD", "BUY", 0.1, 1.0800)
    """

    TELEGRAM_API = "https://api.telegram.org"

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._enabled = bool(bot_token and chat_id)

        if not self._enabled:
            logger.warning("AlertService: Telegram not configured (missing bot_token or chat_id)")

    async def send(self, message: str) -> bool:
        """Send a text message via Telegram.

        Returns True if sent successfully, False otherwise.
        """
        if not self._enabled:
            logger.debug("AlertService: skipping (not configured): {}", message[:80])
            return False

        url = f"{self.TELEGRAM_API}/bot{self._bot_token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": message,
            "parse_mode": "HTML",
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload)
                if response.status_code == 200:
                    return True
                logger.error(
                    "AlertService: Telegram API error {}: {}",
                    response.status_code,
                    response.text[:200],
                )
                return False
        except httpx.HTTPError as e:
            logger.error("AlertService: failed to send Telegram message: {}", e)
            return False

    # ── Convenience Methods ─────────────────────────────────────────────

    async def trade_opened(self, symbol: str, side: str, volume: float, price: float) -> bool:
        msg = (
            f"📈 <b>Trade Opened</b>\n• {side} {symbol}\n• Volume: {volume} lots\n• Price: {price}"
        )
        return await self.send(msg)

    async def trade_closed(self, symbol: str, side: str, pnl: float, reason: str) -> bool:
        emoji = "✅" if pnl >= 0 else "❌"
        msg = (
            f"{emoji} <b>Trade Closed</b>\n"
            f"• {side} {symbol}\n"
            f"• PnL: ${pnl:+.2f}\n"
            f"• Reason: {reason}"
        )
        return await self.send(msg)

    async def drawdown_warning(
        self,
        level: str,
        daily_dd_pct: float,
        max_dd_pct: float,
        equity: float,
    ) -> bool:
        emoji_map: Dict[str, str] = {
            "WARNING": "🟡",
            "DANGER": "🟠",
            "CRITICAL": "🔴",
        }
        emoji = emoji_map.get(level, "⚠️")
        msg = (
            f"{emoji} <b>Drawdown Alert: {level}</b>\n"
            f"• Daily DD: {daily_dd_pct:.1%}\n"
            f"• Max DD: {max_dd_pct:.1%}\n"
            f"• Equity: ${equity:,.2f}"
        )
        return await self.send(msg)

    async def compliance_rejection(self, symbol: str, side: str, reason: str) -> bool:
        msg = f"🚫 <b>Compliance Rejected</b>\n• {side} {symbol}\n• Reason: {reason}"
        return await self.send(msg)

    async def system_error(self, error_msg: str) -> bool:
        msg = f"💀 <b>System Error</b>\n<code>{error_msg[:500]}</code>"
        return await self.send(msg)

    async def daily_summary(
        self,
        date: str,
        trades: int,
        pnl: float,
        equity: float,
        daily_dd_pct: float,
    ) -> bool:
        emoji = "📊" if pnl >= 0 else "📉"
        msg = (
            f"{emoji} <b>Daily Summary — {date}</b>\n"
            f"• Trades: {trades}\n"
            f"• PnL: ${pnl:+.2f}\n"
            f"• Equity: ${equity:,.2f}\n"
            f"• Daily DD: {daily_dd_pct:.1%}"
        )
        return await self.send(msg)
