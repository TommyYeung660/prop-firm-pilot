"""
Live Telegram bot command test — starts polling loop for /profit and /orders.

Connects to real MatchTrader API (account 950552) and listens for commands.
Send /profit, /orders, or /help in Telegram to test.

Press Ctrl+C to stop.

Usage:
    uv run python scripts/test_telegram_bot_live.py
"""

import asyncio
import os
import signal
import sys

# Fix Windows console encoding for emoji output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.execution.matchtrader_client import MatchTraderClient
from src.monitor.alert_service import AlertService
from src.monitor.telegram_bot import TelegramBotHandler


async def main() -> None:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    api_url = os.getenv("MATCHTRADER_API_URL", "")
    username = os.getenv("MATCHTRADER_USERNAME", "")
    password = os.getenv("MATCHTRADER_PASSWORD", "")
    broker_id = os.getenv("MATCHTRADER_BROKER_ID", "2")
    account_id = os.getenv("MATCHTRADER_ACCOUNT_ID", "950552")

    if not all([bot_token, chat_id, api_url, username, password]):
        print("ERROR: Missing required env vars. Check .env file.")
        sys.exit(1)

    alerts = AlertService(
        bot_token=bot_token,
        chat_id=chat_id,
        account_id=account_id,
        initial_balance=5000.0,
        profit_target_pct=0.06,
        daily_loss_pct=0.02,
        max_drawdown_pct=0.04,
    )

    client = MatchTraderClient(
        base_url=api_url,
        email=username,
        password=password,
        broker_id=broker_id,
        account_id=account_id,
    )

    async with client:
        await client.login()
        print(f"✅ MatchTrader login OK (account {account_id})")

        # Quick balance check
        balance = await client.get_balance()
        print(f"   Balance: ${balance.balance:.2f} | Equity: ${balance.equity:.2f}")

        bot = TelegramBotHandler(
            bot_token=bot_token,
            chat_id=chat_id,
            alert_service=alerts,
            trading_client=client,
            trade_journal=None,  # Not needed for commands
        )

        print("\n🤖 Bot is running! Send commands in Telegram:")
        print("   /profit  — Show positions & profit target progress")
        print("   /orders  — Show last 10 trades & open positions")
        print("   /help    — Show available commands")
        print("\n   Press Ctrl+C to stop.\n")

        # Send startup notification
        await alerts.send("🤖 <b>Bot Started</b>\nListening for commands...")

        try:
            await bot.start()
        except asyncio.CancelledError:
            pass
        finally:
            bot.stop()
            await alerts.send("🤖 <b>Bot Stopped</b>")
            print("\n🛑 Bot stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
