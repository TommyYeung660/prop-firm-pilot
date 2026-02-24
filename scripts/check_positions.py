"""
Quick position inspection — logs in and prints full position details including SL/TP.

Run with:
    uv run python scripts/check_positions.py
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
    logger.info("Position Inspector — checking SL/TP on all open positions")
    logger.info("=" * 70)

    async with MatchTraderClient(
        base_url=base_url,
        email=email,
        password=password,
        broker_id=broker_id,
        account_id=account_id,
    ) as client:
        # Login
        tokens = await client.login()
        logger.info("Login OK — account={}, UUID={}", account_id, tokens.system_uuid)

        # Balance
        bal = await client.get_balance()
        logger.info(
            "Balance: ${:.2f} | Equity: ${:.2f} | Free Margin: ${:.2f}",
            bal.balance,
            bal.equity,
            bal.free_margin,
        )

        # Open positions — get raw data too
        logger.info("")
        logger.info("── Open Positions ──")
        positions = await client.get_open_positions()
        logger.info("{} open position(s)", len(positions))

        for pos in positions:
            logger.info("")
            logger.info("  Position ID:   {}", pos.position_id)
            logger.info("  Symbol:        {}", pos.symbol)
            logger.info("  Side:          {}", pos.side)
            logger.info("  Volume:        {}", pos.volume)
            logger.info("  Open Price:    {:.5f}", pos.open_price)
            logger.info("  Current Price: {:.5f}", pos.current_price)
            logger.info("  Profit:        ${:.2f}", pos.profit)
            logger.info("  SL Price:      {}", pos.sl_price)
            logger.info("  TP Price:      {}", pos.tp_price)
            logger.info("  Open Time:     {}", pos.open_time)

        # Also get RAW positions data to see all fields
        logger.info("")
        logger.info("── Raw Position API Response ──")
        try:
            await client._ensure_auth()
            response = await client._api_request(
                "GET",
                f"/mtr-api/{client.system_uuid}/position/open",
            )
            raw_data = response.json()
            logger.info("Raw response:\n{}", json.dumps(raw_data, indent=2, default=str))
        except Exception as e:
            logger.error("Failed to get raw positions: {}", e)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Inspection complete")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
