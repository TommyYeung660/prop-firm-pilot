"""
Telegram alert service for prop firm trading notifications.

Sends rich alerts for trade lifecycle events, drawdown warnings,
daily summaries, and provides formatters for /profit and /orders commands.

Usage:
    alerts = AlertService(
        bot_token="123:ABC",
        chat_id="-100123",
        account_id="950552",
        initial_balance=5000.0,
        profit_target_pct=0.06,
        daily_loss_pct=0.02,
        max_drawdown_pct=0.04,
    )
    await alerts.trade_opened("EURUSD.", "BUY", 0.10, 1.08500, equity=5050.0)
"""

import time
from typing import Any

import httpx
from loguru import logger


class AlertService:
    """Sends trading alerts via Telegram Bot API with account context.

    Supports per-account configuration for profit targets and drawdown limits.
    All notification methods are backward-compatible with the original signatures.

    Usage:
        alerts = AlertService(bot_token="123:ABC", chat_id="-100123")
        await alerts.send("🔴 Daily drawdown at 85%!")
        await alerts.trade_opened("EURUSD", "BUY", 0.1, 1.0800)
    """

    TELEGRAM_API = "https://api.telegram.org"

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        account_id: str = "",
        initial_balance: float = 0.0,
        profit_target_pct: float = 0.0,
        daily_loss_pct: float = 0.0,
        max_drawdown_pct: float = 0.0,
    ) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._account_id = account_id
        self._initial_balance = initial_balance
        self._profit_target_pct = profit_target_pct
        self._daily_loss_pct = daily_loss_pct
        self._max_drawdown_pct = max_drawdown_pct
        self._http_client: httpx.AsyncClient | None = None
        self._enabled = bool(bot_token and chat_id)

        # ── Circuit Breaker State ──────────────────────────────────────
        self._consecutive_failures: int = 0
        self._circuit_open: bool = False
        self._circuit_opened_at: float = 0.0
        self._CIRCUIT_OPEN_THRESHOLD: int = 3
        self._CIRCUIT_RETRY_INTERVAL: float = 300.0  # seconds

        if not self._enabled:
            logger.warning("AlertService: Telegram not configured (missing bot_token or chat_id)")

    async def _get_client(self) -> httpx.AsyncClient:
        """Return persistent HTTP client, creating lazily on first use."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=10.0))
        return self._http_client

    async def close(self) -> None:
        """Close the persistent HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    # ── Computed Properties ─────────────────────────────────────────────

    @property
    def profit_target_amount(self) -> float:
        return self._initial_balance * self._profit_target_pct

    @property
    def daily_loss_amount(self) -> float:
        return self._initial_balance * self._daily_loss_pct

    @property
    def max_drawdown_amount(self) -> float:
        return self._initial_balance * self._max_drawdown_pct

    # ── Core Send ───────────────────────────────────────────────────────

    async def send(self, message: str) -> bool:
        """Send a text message via Telegram.

        Uses a circuit breaker pattern: after ``_CIRCUIT_OPEN_THRESHOLD``
        consecutive failures the circuit opens and subsequent calls return
        ``False`` immediately (no HTTP request, no 10-second timeout wait).
        Every ``_CIRCUIT_RETRY_INTERVAL`` seconds one probe request is
        allowed through; on success the circuit resets to closed.

        Returns True if sent successfully, False otherwise.
        """
        if not self._enabled:
            logger.debug("AlertService: skipping (not configured): {}", message[:80])
            return False

        # ── Circuit Breaker: fast-fail when open ─────────────────────
        if self._circuit_open:
            elapsed = time.monotonic() - self._circuit_opened_at
            if elapsed < self._CIRCUIT_RETRY_INTERVAL:
                logger.debug(
                    "AlertService: circuit open, skipping send ({:.0f}s until probe)",
                    self._CIRCUIT_RETRY_INTERVAL - elapsed,
                )
                return False
            # Retry interval elapsed — allow one probe request
            logger.info(
                "AlertService: circuit open for {:.0f}s, attempting probe request",
                elapsed,
            )

        url = f"{self.TELEGRAM_API}/bot{self._bot_token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": message,
            "parse_mode": "HTML",
        }

        try:
            client = await self._get_client()
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                self._reset_circuit()
                return True
            logger.error(
                "AlertService: Telegram API error {}: {}",
                response.status_code,
                response.text[:200],
            )
            self._record_failure()
            return False
        except httpx.HTTPError as e:
            logger.error(
                "AlertService: failed to send Telegram message: {} ({})",
                type(e).__name__,
                e or "no details",
            )
            self._record_failure()
            return False

    # ── Circuit Breaker Helpers ─────────────────────────────────────────

    def _record_failure(self) -> None:
        """Increment failure counter and open circuit if threshold reached."""
        self._consecutive_failures += 1
        if not self._circuit_open and self._consecutive_failures >= self._CIRCUIT_OPEN_THRESHOLD:
            self._circuit_open = True
            self._circuit_opened_at = time.monotonic()
            logger.warning(
                "AlertService: circuit OPEN after {} consecutive failures "
                "— Telegram sends will be skipped for {:.0f}s",
                self._consecutive_failures,
                self._CIRCUIT_RETRY_INTERVAL,
            )

    def _reset_circuit(self) -> None:
        """Reset circuit breaker to closed state."""
        if self._circuit_open:
            logger.info(
                "AlertService: circuit CLOSED — Telegram recovered after {:.0f}s",
                time.monotonic() - self._circuit_opened_at,
            )
        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_opened_at = 0.0

    # ── Trade Notifications ─────────────────────────────────────────────

    async def trade_opened(
        self,
        symbol: str,
        side: str,
        volume: float,
        price: float,
        sl: float | None = None,
        tp: float | None = None,
        equity: float | None = None,
        position_id: str = "",
    ) -> bool:
        """Send trade opened notification with optional profit target progress."""
        lines = [
            f"📈 {self._account_header()}<b>Trade Opened</b>",
            f"• {side} {symbol} {volume:.2f} lots",
            f"• Price: {price}",
        ]
        if sl is not None or tp is not None:
            sl_str = f"{sl}" if sl is not None else "—"
            tp_str = f"{tp}" if tp is not None else "—"
            lines.append(f"• SL: {sl_str} / TP: {tp_str}")
        if position_id:
            lines.append(f"• Position: {position_id}")

        if equity is not None and self.profit_target_amount > 0:
            lines.append("")
            lines.append(self._profit_progress(equity))

        return await self.send("\n".join(lines))

    async def trade_closed(
        self,
        symbol: str,
        side: str,
        pnl: float,
        reason: str,
        volume: float = 0.0,
        open_price: float = 0.0,
        close_price: float = 0.0,
        equity: float | None = None,
        position_id: str = "",
    ) -> bool:
        """Send trade closed notification with PnL and optional profit progress."""
        emoji = "✅" if pnl >= 0 else "❌"
        lines = [
            f"{emoji} {self._account_header()}<b>Trade Closed</b>",
            f"• {side} {symbol}",
        ]
        if volume > 0:
            lines.append(f"• Volume: {volume:.2f} lots")
        if open_price > 0 and close_price > 0:
            lines.append(f"• {open_price} → {close_price}")
        lines.append(f"• PnL: ${pnl:+.2f}")
        lines.append(f"• Reason: {reason}")
        if position_id:
            lines.append(f"• Position: {position_id}")

        if equity is not None and self.profit_target_amount > 0:
            lines.append("")
            lines.append(self._profit_progress(equity))

        return await self.send("\n".join(lines))

    async def sl_tp_hit(
        self,
        symbol: str,
        side: str,
        volume: float,
        pnl: float,
        hit_type: str,
        trigger_price: float,
        equity: float | None = None,
        position_id: str = "",
    ) -> bool:
        """Send SL/TP hit notification.

        Args:
            hit_type: "SL" for stop loss, "TP" for take profit.
        """
        emoji = "🛑" if hit_type == "SL" else "🎯"
        label = "Stop Loss Hit" if hit_type == "SL" else "Take Profit Hit"
        lines = [
            f"{emoji} {self._account_header()}<b>{label}</b>",
            f"• {side} {symbol} {volume:.2f} lots",
            f"• Trigger: {trigger_price}",
            f"• PnL: ${pnl:+.2f}",
        ]
        if position_id:
            lines.append(f"• Position: {position_id}")

        if equity is not None and self.profit_target_amount > 0:
            lines.append("")
            lines.append(self._profit_progress(equity))

        return await self.send("\n".join(lines))

    # ── Drawdown & Compliance ───────────────────────────────────────────

    async def drawdown_warning(
        self,
        level: str,
        daily_dd_pct: float,
        max_dd_pct: float,
        equity: float,
        day_start_balance: float | None = None,
    ) -> bool:
        """Send drawdown alert with optional daily loss buffer info."""
        emoji_map: dict[str, str] = {
            "WARNING": "🟡",
            "DANGER": "🟠",
            "CRITICAL": "🔴",
        }
        emoji = emoji_map.get(level, "⚠️")
        lines = [
            f"{emoji} {self._account_header()}<b>Drawdown Alert: {level}</b>",
            f"• Daily DD: {daily_dd_pct:.1%}",
            f"• Max DD: {max_dd_pct:.1%}",
            f"• Equity: ${equity:,.2f}",
        ]
        if day_start_balance is not None and self.daily_loss_amount > 0:
            daily_loss_used = max(0.0, day_start_balance - equity)
            buffer = self.daily_loss_amount - daily_loss_used
            lines.append(f"• Daily loss buffer: ${buffer:,.2f} remaining")

        return await self.send("\n".join(lines))

    async def compliance_rejection(self, symbol: str, side: str, reason: str) -> bool:
        """Send compliance rejection notification."""
        msg = (
            f"🚫 {self._account_header()}<b>Compliance Rejected</b>\n"
            f"• {side} {symbol}\n"
            f"• Reason: {reason}"
        )
        return await self.send(msg)

    async def system_error(self, error_msg: str) -> bool:
        """Send system error notification."""
        msg = f"💀 {self._account_header()}<b>System Error</b>\n<code>{error_msg[:500]}</code>"
        return await self.send(msg)

    # ── Daily Summary ───────────────────────────────────────────────────

    async def daily_summary(
        self,
        date: str,
        trades: int,
        pnl: float,
        equity: float,
        daily_dd_pct: float,
        open_positions: int = 0,
        day_start_balance: float | None = None,
        max_dd_reference: float | None = None,
    ) -> bool:
        """Send end-of-day summary with profit target progress and risk status."""
        emoji = "📊" if pnl >= 0 else "📉"
        lines = [
            f"{emoji} {self._account_header()}<b>Daily Summary — {date}</b>",
            f"• Trades: {trades}",
            f"• PnL: ${pnl:+.2f}",
            f"• Equity: ${equity:,.2f}",
            f"• Open positions: {open_positions}",
        ]

        # Profit target progress
        if self.profit_target_amount > 0:
            lines.append("")
            lines.append(self._profit_progress(equity))

        # Risk status
        lines.append("")
        lines.append("<b>Risk Status</b>")
        lines.append(f"• Daily DD used: {daily_dd_pct:.1%}")
        if day_start_balance is not None and self.daily_loss_amount > 0:
            daily_loss_used = max(0.0, day_start_balance - equity)
            buffer = self.daily_loss_amount - daily_loss_used
            lines.append(f"• Daily loss buffer: ${buffer:,.2f}")
        if self.max_drawdown_amount > 0:
            ref = max_dd_reference or self._initial_balance
            dd_limit = ref * self._max_drawdown_pct
            max_loss = max(0.0, ref - equity)
            max_buffer = dd_limit - max_loss
            lines.append(f"• Max DD buffer: ${max_buffer:,.2f}")

        return await self.send("\n".join(lines))

    # ── Command Formatters ──────────────────────────────────────────────

    def format_profit_status(
        self,
        equity: float,
        positions: list[dict[str, Any]],
        day_start_balance: float | None = None,
        max_dd_reference: float | None = None,
    ) -> str:
        """Format profit status for /profit command response.

        Args:
            equity: Current account equity.
            positions: List of open position dicts (from PositionInfo.model_dump()).
            day_start_balance: Balance at start of trading day.

        Returns:
            Formatted HTML string ready to send.
        """
        lines = [
            f"💰 {self._account_header()}<b>Profit Status</b>",
            f"• Equity: ${equity:,.2f}",
        ]

        # Profit target progress
        if self.profit_target_amount > 0:
            lines.append("")
            lines.append(self._profit_progress(equity))

        # Open positions
        if positions:
            lines.append("")
            lines.append(f"<b>Open Positions ({len(positions)})</b>")
            for pos in positions:
                symbol = pos.get("symbol", "?")
                side = pos.get("side", "?")
                vol = pos.get("volume", 0.0)
                pnl = pos.get("profit", 0.0)
                pos_emoji = "🟢" if pnl >= 0 else "🔴"
                lines.append(f"{pos_emoji} {side} {symbol} {vol} lots — ${pnl:+.2f}")
        else:
            lines.append("")
            lines.append("📭 No open positions")

        # Drawdown buffers
        lines.append("")
        lines.append("<b>Risk Buffers</b>")
        if day_start_balance is not None and self.daily_loss_amount > 0:
            daily_loss_used = max(0.0, day_start_balance - equity)
            daily_buffer = self.daily_loss_amount - daily_loss_used
            lines.append(f"• Daily loss buffer: ${daily_buffer:,.2f}")
        if self.max_drawdown_amount > 0:
            ref = max_dd_reference or self._initial_balance
            dd_limit = ref * self._max_drawdown_pct
            max_loss = max(0.0, ref - equity)
            max_buffer = dd_limit - max_loss
            lines.append(f"• Max DD buffer: ${max_buffer:,.2f}")

        return "\n".join(lines)

    @staticmethod
    def format_orders_list(
        closed_trades: list[dict[str, Any]],
        open_positions: list[dict[str, Any]] | None = None,
    ) -> str:
        """Format orders list for /orders command response.

        Args:
            closed_trades: List of closed trade dicts
                (from ClosedPosition.model_dump()).
            open_positions: List of open position dicts (optional).

        Returns:
            Formatted HTML string ready to send.
        """
        lines: list[str] = ["📋 <b>Orders</b>"]

        # Open positions section
        if open_positions:
            lines.append("")
            lines.append(f"<b>Open ({len(open_positions)})</b>")
            for pos in open_positions:
                symbol = pos.get("symbol", "?")
                side = pos.get("side", "?")
                vol = pos.get("volume", 0.0)
                pnl = pos.get("profit", 0.0)
                pos_emoji = "🟢" if pnl >= 0 else "🔴"
                lines.append(f"{pos_emoji} {side} {symbol} {vol} lots — ${pnl:+.2f}")

        # Last 10 closed trades
        last_10 = closed_trades[-10:] if len(closed_trades) > 10 else closed_trades
        if last_10:
            lines.append("")
            lines.append(f"<b>Closed (last {len(last_10)})</b>")
            for trade in reversed(last_10):
                symbol = trade.get("symbol", "?")
                side = trade.get("side", "?")
                pnl = trade.get("profit", 0.0)
                close_time = trade.get("close_time", "")
                # Truncate close_time to date+time
                if len(close_time) > 16:
                    close_time = close_time[:16]
                trade_emoji = "🟢" if pnl >= 0 else "🔴"
                lines.append(f"{trade_emoji} {side} {symbol} ${pnl:+.2f} | {close_time}")
        else:
            lines.append("")
            lines.append("📭 No closed trades")

        return "\n".join(lines)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _account_header(self) -> str:
        """Return '[account_id] ' prefix if account_id is set, else ''."""
        if self._account_id:
            return f"[{self._account_id}] "
        return ""

    def _progress_bar(self, pct: float, width: int = 20) -> str:
        """Return a text progress bar like [██████░░░░░░░░░░░░░░] 30.0%."""
        pct = max(0.0, min(pct, 1.0))
        filled = int(width * pct)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {pct:.1%}"

    def _profit_progress(self, equity: float) -> str:
        """Return multi-line profit target progress block."""
        target = self.profit_target_amount
        if target <= 0:
            return ""
        current_pnl = equity - self._initial_balance
        pct = current_pnl / target if target > 0 else 0.0
        remaining = target - current_pnl
        lines = [
            "📊 <b>Profit Target</b>",
            f"• Target: ${target:,.2f} ({self._profit_target_pct:.1%})",
            f"• Current PnL: ${current_pnl:+,.2f}",
            f"• Remaining: ${remaining:,.2f}",
            f"• {self._progress_bar(pct)}",
        ]
        return "\n".join(lines)
