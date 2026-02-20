"""
MatchTrader test trade — validates the full open → query → close lifecycle.

Opens a 0.01 lot EURUSD BUY on the E8 Trial account (demo=true), verifies the
position appears in open positions, waits briefly, then closes it and confirms
the closure. No real money is at risk.

Run with:
    uv run python scripts/test_trade.py

Optional flags:
    --skip-close     Open position but DON'T close (leave it open for manual inspection)
    --dry-run        Run login + balance only, skip actual trading

Requires .env with:
    MATCHTRADER_API_URL, MATCHTRADER_USERNAME, MATCHTRADER_PASSWORD,
    MATCHTRADER_BROKER_ID, MATCHTRADER_ACCOUNT_ID
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.execution.matchtrader_client import MatchTraderClient

# ── Constants ───────────────────────────────────────────────────────────────

SYMBOL = "EURUSD."
SIDE = "BUY"
VOLUME = 0.01  # Minimum lot size


# ── Helpers ─────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MatchTrader test trade script")
    parser.add_argument(
        "--skip-close",
        action="store_true",
        help="Open position but don't close it (leave open for inspection)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Login and check balance only, skip actual trading",
    )
    return parser.parse_args()


def _load_env() -> tuple[str, str, str, str, str | None]:
    """Load and validate required environment variables."""
    load_dotenv()

    base_url = os.getenv("MATCHTRADER_API_URL", "").rstrip("/")
    email = os.getenv("MATCHTRADER_USERNAME", "")
    password = os.getenv("MATCHTRADER_PASSWORD", "")
    broker_id = os.getenv("MATCHTRADER_BROKER_ID", "2")
    account_id = os.getenv("MATCHTRADER_ACCOUNT_ID")

    if not all([base_url, email, password]):
        logger.error("Missing required env vars (MATCHTRADER_API_URL, USERNAME, PASSWORD)")
        sys.exit(1)

    return base_url, email, password, broker_id, account_id


# ── Main test flow ──────────────────────────────────────────────────────────


async def main() -> None:
    """Execute test trade lifecycle: login → balance → open → verify → close → verify."""
    args = _parse_args()
    base_url, email, password, broker_id, account_id = _load_env()

    logger.info("=" * 70)
    logger.info("MatchTrader Test Trade — {} {} {} lots", SIDE, SYMBOL, VOLUME)
    logger.info("=" * 70)
    logger.info("Base URL:   {}", base_url)
    logger.info("Account:    {}", account_id or "(auto-select)")
    logger.info("Mode:       {}", "DRY RUN" if args.dry_run else "LIVE TRADE (demo)")
    logger.info("=" * 70)

    async with MatchTraderClient(
        base_url=base_url,
        email=email,
        password=password,
        broker_id=broker_id,
        account_id=account_id,
    ) as client:
        # ── Step 1: Login ───────────────────────────────────────────────
        logger.info("")
        logger.info("── Step 1/6: Login ──")
        tokens = await client.login()
        logger.info("  Login OK — UUID: {}", tokens.system_uuid)

        # ── Step 2: Check balance ───────────────────────────────────────
        logger.info("")
        logger.info("── Step 2/6: Check balance (before trade) ──")
        balance_before = await client.get_balance()
        logger.info(
            "  Balance: ${:.2f} | Equity: ${:.2f} | Free Margin: ${:.2f}",
            balance_before.balance,
            balance_before.equity,
            balance_before.free_margin,
        )

        # ── Step 3: Check existing positions ────────────────────────────
        logger.info("")
        logger.info("── Step 3/6: Check existing positions ──")
        existing_positions = await client.get_open_positions()
        logger.info("  {} existing position(s)", len(existing_positions))
        for pos in existing_positions:
            logger.info(
                "    {} {} {} @ {:.5f} | PnL: ${:.2f}",
                pos.position_id,
                pos.side,
                pos.symbol,
                pos.open_price,
                pos.profit,
            )

        if args.dry_run:
            logger.info("")
            logger.info("DRY RUN complete — skipping trade execution.")
            logger.info("=" * 70)
            return

        # ── Step 4: Open position ───────────────────────────────────────
        logger.info("")
        logger.info("── Step 4/6: Open {} {} {} lots ──", SIDE, SYMBOL, VOLUME)

        # Open without SL/TP for simplest possible test — avoids issues with
        # hardcoded price offsets being stale vs current market price.
        open_result = await client.open_position(
            symbol=SYMBOL,
            side=SIDE,
            volume=VOLUME,
        )

        if not open_result.success:
            logger.error("  FAILED to open position: {}", open_result.message)
            logger.error("  Raw response: {}", open_result.raw_response)
            sys.exit(1)

        position_id = open_result.position_id
        logger.info("  Position OPENED — ID: {}", position_id)
        logger.info("  Raw response: {}", open_result.raw_response)

        # ── Step 5: Verify position appears ─────────────────────────────
        logger.info("")
        logger.info("── Step 5/6: Verify position in open positions ──")

        # Brief wait for position to settle
        logger.info("  Waiting 3s for position to settle...")
        await asyncio.sleep(3)

        open_positions = await client.get_open_positions()
        logger.info("  {} open position(s) now", len(open_positions))

        our_position = None
        for pos in open_positions:
            marker = " <<<" if pos.position_id == position_id else ""
            logger.info(
                "    {} {} {} {:.2f} lots @ {:.5f} | PnL: ${:.2f}{}",
                pos.position_id,
                pos.side,
                pos.symbol,
                pos.volume,
                pos.open_price,
                pos.profit,
                marker,
            )
            if pos.position_id == position_id:
                our_position = pos

        if our_position:
            logger.info("  Position verified in open positions list")
        else:
            logger.warning(
                "  Position {} NOT found in open positions — may use a different ID format",
                position_id,
            )
            # Try to find our position by symbol — prefer matching our symbol over random fallback
            symbol_matches = [p for p in open_positions if p.symbol == SYMBOL]
            if symbol_matches:
                logger.info(
                    "  Found {} position(s) matching {}, using most recent",
                    len(symbol_matches),
                    SYMBOL,
                )
                our_position = symbol_matches[-1]
                position_id = our_position.position_id
            elif open_positions:
                logger.warning(
                    "  No {} positions found, aborting close to avoid closing wrong position",
                    SYMBOL,
                )
                logger.warning(
                    "  Open positions: {}", [(p.position_id, p.symbol) for p in open_positions]
                )
                sys.exit(1)

        if args.skip_close:
            logger.info("")
            logger.info("--skip-close flag set — position left open for manual inspection.")
            logger.info("Position ID: {}", position_id)
            logger.info("=" * 70)
            return

        # ── Step 6: Close position ──────────────────────────────────────
        logger.info("")
        logger.info("── Step 6/6: Close position {} ──", position_id)

        # Brief wait before closing
        logger.info("  Waiting 2s before closing...")
        await asyncio.sleep(2)

        # Use position's own symbol/side for close — not the constants, in case fallback found a different position
        close_symbol = our_position.symbol if our_position else SYMBOL
        close_side = our_position.side if our_position else SIDE
        close_volume = our_position.volume if our_position else VOLUME

        close_result = await client.close_position(
            position_id=position_id,
            symbol=close_symbol,
            side=close_side,
            volume=close_volume,
        )

        if not close_result.success:
            logger.error("  FAILED to close position: {}", close_result.message)
            logger.error("  Raw response: {}", close_result.raw_response)

            # Try with opposite side (some APIs expect the closing side)
            opposite_side = "SELL" if close_side == "BUY" else "BUY"
            logger.info("  Retrying with opposite side ({})...", opposite_side)
            close_result = await client.close_position(
                position_id=position_id,
                symbol=close_symbol,
                side=opposite_side,
                volume=close_volume,
            )

            if not close_result.success:
                logger.error("  FAILED again: {}", close_result.message)
                logger.error("  Raw: {}", close_result.raw_response)
                logger.warning("  Position {} may still be OPEN — close manually!", position_id)
                sys.exit(1)

        logger.info("  Position CLOSED successfully")
        logger.info("  Raw response: {}", close_result.raw_response)

        # ── Final verification ──────────────────────────────────────────
        logger.info("")
        logger.info("── Final verification ──")
        await asyncio.sleep(2)

        final_positions = await client.get_open_positions()
        position_still_open = any(p.position_id == position_id for p in final_positions)

        balance_after = await client.get_balance()
        logger.info(
            "  Balance after: ${:.2f} (was ${:.2f}, delta: ${:+.2f})",
            balance_after.balance,
            balance_before.balance,
            balance_after.balance - balance_before.balance,
        )
        logger.info("  Open positions remaining: {}", len(final_positions))

        if position_still_open:
            logger.warning("  Position {} still appears in open positions!", position_id)
        else:
            logger.info("  Position {} confirmed closed", position_id)

    # ── Summary ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST TRADE COMPLETE")
    logger.info("  Open:  {} (ID: {})", "OK" if open_result.success else "FAIL", position_id)
    logger.info("  Close: {}", "OK" if close_result.success else "FAIL")
    logger.info(
        "  P&L:   ${:+.2f}",
        balance_after.balance - balance_before.balance,
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
