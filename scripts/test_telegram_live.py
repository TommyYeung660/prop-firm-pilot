"""
Live Telegram integration test — sends sample notifications to verify formatting.

Usage:
    uv run python scripts/test_telegram_live.py
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.monitor.alert_service import AlertService


async def main() -> None:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not bot_token or not chat_id:
        print("ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env")
        sys.exit(1)

    alerts = AlertService(
        bot_token=bot_token,
        chat_id=chat_id,
        account_id="950552",
        initial_balance=5000.0,
        profit_target_pct=0.06,
        daily_loss_pct=0.02,
        max_drawdown_pct=0.04,
    )

    print("=== Live Telegram Notification Test ===\n")

    # 1. Trade Opened
    print("1. Sending trade_opened...")
    ok = await alerts.trade_opened(
        symbol="EURUSD.",
        side="BUY",
        volume=0.05,
        price=1.08520,
        sl=1.08200,
        tp=1.09000,
        equity=5012.50,
        position_id="12345678",
    )
    print(f"   Result: {'OK' if ok else 'FAILED'}")
    await asyncio.sleep(1)

    # 2. Trade Closed (profit)
    print("2. Sending trade_closed (profit)...")
    ok = await alerts.trade_closed(
        symbol="EURUSD.",
        side="BUY",
        pnl=48.00,
        reason="TP hit",
        volume=0.05,
        open_price=1.08520,
        close_price=1.09000,
        equity=5060.50,
        position_id="12345678",
    )
    print(f"   Result: {'OK' if ok else 'FAILED'}")
    await asyncio.sleep(1)

    # 3. Trade Closed (loss)
    print("3. Sending trade_closed (loss)...")
    ok = await alerts.trade_closed(
        symbol="GBPUSD.",
        side="SELL",
        pnl=-32.00,
        reason="SL hit",
        volume=0.03,
        open_price=1.26800,
        close_price=1.27120,
        equity=5028.50,
        position_id="87654321",
    )
    print(f"   Result: {'OK' if ok else 'FAILED'}")
    await asyncio.sleep(1)

    # 4. SL Hit
    print("4. Sending sl_tp_hit (SL)...")
    ok = await alerts.sl_tp_hit(
        symbol="USDJPY.",
        side="BUY",
        volume=0.10,
        pnl=-45.00,
        hit_type="SL",
        trigger_price=149.500,
        equity=4983.50,
        position_id="11223344",
    )
    print(f"   Result: {'OK' if ok else 'FAILED'}")
    await asyncio.sleep(1)

    # 5. TP Hit
    print("5. Sending sl_tp_hit (TP)...")
    ok = await alerts.sl_tp_hit(
        symbol="XAUUSD.",
        side="BUY",
        volume=0.02,
        pnl=86.00,
        hit_type="TP",
        trigger_price=2950.00,
        equity=5114.50,
        position_id="55667788",
    )
    print(f"   Result: {'OK' if ok else 'FAILED'}")
    await asyncio.sleep(1)

    # 6. Drawdown Warning
    print("6. Sending drawdown_warning (WARNING)...")
    ok = await alerts.drawdown_warning(
        level="WARNING",
        daily_dd_pct=0.012,
        max_dd_pct=0.018,
        equity=4940.00,
        day_start_balance=5000.00,
    )
    print(f"   Result: {'OK' if ok else 'FAILED'}")
    await asyncio.sleep(1)

    # 7. Drawdown Critical
    print("7. Sending drawdown_warning (CRITICAL)...")
    ok = await alerts.drawdown_warning(
        level="CRITICAL",
        daily_dd_pct=0.019,
        max_dd_pct=0.035,
        equity=4825.00,
        day_start_balance=5000.00,
    )
    print(f"   Result: {'OK' if ok else 'FAILED'}")
    await asyncio.sleep(1)

    # 8. Compliance Rejection
    print("8. Sending compliance_rejection...")
    ok = await alerts.compliance_rejection(
        symbol="EURUSD.",
        side="BUY",
        reason="Daily drawdown limit reached (85% threshold)",
    )
    print(f"   Result: {'OK' if ok else 'FAILED'}")
    await asyncio.sleep(1)

    # 9. System Error
    print("9. Sending system_error...")
    ok = await alerts.system_error(
        error_msg="MatchTrader API timeout after 3 retries: ConnectionError('mtr.e8markets.com')",
    )
    print(f"   Result: {'OK' if ok else 'FAILED'}")
    await asyncio.sleep(1)

    # 10. Daily Summary (profitable day)
    print("10. Sending daily_summary (profit)...")
    ok = await alerts.daily_summary(
        date="2026-02-16",
        trades=5,
        pnl=72.50,
        equity=5072.50,
        daily_dd_pct=0.005,
        open_positions=1,
        day_start_balance=5000.00,
    )
    print(f"   Result: {'OK' if ok else 'FAILED'}")
    await asyncio.sleep(1)

    # 11. Daily Summary (loss day)
    print("11. Sending daily_summary (loss)...")
    ok = await alerts.daily_summary(
        date="2026-02-15",
        trades=3,
        pnl=-28.00,
        equity=4972.00,
        daily_dd_pct=0.012,
        open_positions=0,
        day_start_balance=5000.00,
    )
    print(f"   Result: {'OK' if ok else 'FAILED'}")
    await asyncio.sleep(1)

    # 12. /profit command format
    print("12. Sending format_profit_status...")
    profit_msg = alerts.format_profit_status(
        equity=5045.00,
        positions=[
            {"symbol": "EURUSD.", "side": "BUY", "volume": 0.05, "profit": 12.50},
            {"symbol": "XAUUSD.", "side": "SELL", "volume": 0.01, "profit": -8.30},
        ],
        day_start_balance=5000.00,
    )
    ok = await alerts.send(profit_msg)
    print(f"   Result: {'OK' if ok else 'FAILED'}")
    await asyncio.sleep(1)

    # 13. /profit command (no positions)
    print("13. Sending format_profit_status (no positions)...")
    profit_msg = alerts.format_profit_status(
        equity=5100.00,
        positions=[],
        day_start_balance=5000.00,
    )
    ok = await alerts.send(profit_msg)
    print(f"   Result: {'OK' if ok else 'FAILED'}")
    await asyncio.sleep(1)

    # 14. /orders command format
    print("14. Sending format_orders_list...")
    orders_msg = AlertService.format_orders_list(
        closed_trades=[
            {
                "symbol": "EURUSD.",
                "side": "BUY",
                "profit": 48.00,
                "close_time": "2026-02-16T09:30:00",
            },
            {
                "symbol": "GBPUSD.",
                "side": "SELL",
                "profit": -32.00,
                "close_time": "2026-02-16T11:15:00",
            },
            {
                "symbol": "USDJPY.",
                "side": "BUY",
                "profit": 15.50,
                "close_time": "2026-02-16T14:00:00",
            },
            {
                "symbol": "XAUUSD.",
                "side": "BUY",
                "profit": 86.00,
                "close_time": "2026-02-16T16:45:00",
            },
        ],
        open_positions=[
            {"symbol": "AUDUSD.", "side": "BUY", "volume": 0.03, "profit": 5.20},
        ],
    )
    ok = await alerts.send(orders_msg)
    print(f"   Result: {'OK' if ok else 'FAILED'}")
    await asyncio.sleep(1)

    # 15. /orders command (empty)
    print("15. Sending format_orders_list (empty)...")
    orders_msg = AlertService.format_orders_list(closed_trades=[], open_positions=[])
    ok = await alerts.send(orders_msg)
    print(f"   Result: {'OK' if ok else 'FAILED'}")

    print("\n=== All notifications sent! Check Telegram. ===")


if __name__ == "__main__":
    asyncio.run(main())
