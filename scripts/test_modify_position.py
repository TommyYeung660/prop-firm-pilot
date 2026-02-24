"""
Test modify_position() — sets SL/TP on an existing position and verifies.

This script:
1. Logs in and fetches open positions
2. Picks the first position
3. Calculates safe SL/TP based on current price
4. Calls modify_position() and logs the full raw API response
5. Re-fetches positions to confirm SL/TP are actually set

Run with:
    uv run python scripts/test_modify_position.py
"""

import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.execution.matchtrader_client import MatchTraderClient


async def main() -> None:
    load_dotenv()

    base_url = os.getenv("MATCHTRADER_API_URL", "").rstrip("/")
    email = os.getenv("MATCHTRADER_USERNAME", "")
    password = os.getenv("MATCHTRADER_PASSWORD", "")
    broker_id = os.getenv("MATCHTRADER_BROKER_ID", "2")
    account_id = os.getenv("MATCHTRADER_ACCOUNT_ID")

    logger.info("=" * 70)
    logger.info("Test modify_position() — set SL/TP and verify")
    logger.info("=" * 70)

    async with MatchTraderClient(
        base_url=base_url,
        email=email,
        password=password,
        broker_id=broker_id,
        account_id=account_id,
    ) as client:
        await client.login()

        # 1. Get current positions
        positions = await client.get_open_positions()
        if not positions:
            logger.warning("No open positions — cannot test modify_position()")
            return

        pos = positions[0]
        logger.info("")
        logger.info("── Target Position (BEFORE modify) ──")
        logger.info("  ID:     {}", pos.position_id)
        logger.info("  Symbol: {}", pos.symbol)
        logger.info("  Side:   {}", pos.side)
        logger.info("  Volume: {}", pos.volume)
        logger.info("  Open:   {:.5f}", pos.open_price)
        logger.info("  SL:     {}", pos.sl_price)
        logger.info("  TP:     {}", pos.tp_price)

        # 2. Calculate safe SL/TP (wide enough to not trigger)
        if pos.side.upper() == "BUY":
            # BUY: SL below open, TP above open
            sl_price = round(pos.open_price * 0.99, 5)  # -1%
            tp_price = round(pos.open_price * 1.02, 5)  # +2%
        else:
            # SELL: SL above open, TP below open
            sl_price = round(pos.open_price * 1.01, 5)  # +1%
            tp_price = round(pos.open_price * 0.98, 5)  # -2%

        logger.info("")
        logger.info("── Calling modify_position() ──")
        logger.info("  Setting SL={}, TP={}", sl_price, tp_price)

        result = await client.modify_position(
            position_id=pos.position_id,
            symbol=pos.symbol,
            side=pos.side,
            volume=pos.volume,
            sl=sl_price,
            tp=tp_price,
        )

        logger.info("")
        logger.info("── modify_position() Result ──")
        logger.info("  Success: {}", result.success)
        logger.info("  Message: {}", result.message)
        logger.info(
            "  Raw response:\n{}",
            json.dumps(result.raw_response, indent=2, default=str),
        )

        if not result.success:
            logger.error("modify_position() FAILED — aborting verification")
            return

        # 3. Re-fetch and verify SL/TP are set
        logger.info("")
        logger.info("── Verifying: re-fetching positions ──")
        positions_after = await client.get_open_positions()

        target = next(
            (p for p in positions_after if p.position_id == pos.position_id),
            None,
        )
        if not target:
            logger.error("Position {} not found after modify!", pos.position_id)
            return

        logger.info("  SL (before): {} → SL (after): {}", pos.sl_price, target.sl_price)
        logger.info("  TP (before): {} → TP (after): {}", pos.tp_price, target.tp_price)

        # 4. Verdict
        logger.info("")
        if target.sl_price is not None and target.tp_price is not None:
            logger.info("✅ SUCCESS: SL/TP are set on the broker!")
        else:
            logger.error("❌ FAIL: SL/TP still None after modify_position()!")
            logger.error("   This means the API accepted but didn't apply SL/TP.")

        # Also dump raw API response for position fields
        logger.info("")
        logger.info("── Raw open-positions response (for field verification) ──")
        try:
            await client._ensure_auth()
            response = await client._api_request(
                "GET",
                f"/mtr-api/{client.system_uuid}/open-positions",
            )
            raw_data = response.json()
            logger.info("Raw response:\n{}", json.dumps(raw_data, indent=2, default=str))
        except Exception as e:
            logger.error("Failed to get raw positions: {}", e)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Test complete")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())