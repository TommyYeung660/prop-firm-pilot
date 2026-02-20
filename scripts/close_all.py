"""
Script to manually close all open positions on the MatchTrader account.
Usage: uv run python scripts/close_all.py
"""
import asyncio
import os
import sys

# Add project root to path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from loguru import logger

from src.execution.matchtrader_client import MatchTraderClient

async def main() -> None:
    load_dotenv()
    
    async with MatchTraderClient(
        base_url=os.getenv("MATCHTRADER_API_URL", ""),
        email=os.getenv("MATCHTRADER_USERNAME", ""),
        password=os.getenv("MATCHTRADER_PASSWORD", ""),
        broker_id=os.getenv("MATCHTRADER_BROKER_ID", "2"),
        account_id=os.getenv("MATCHTRADER_ACCOUNT_ID"),
    ) as client:
        await client.login()
        
        positions = await client.get_open_positions()
        logger.info("Found {} open positions", len(positions))
        
        for pos in positions:
            logger.info("Closing position {} ({} {})", pos.position_id, pos.side, pos.symbol)
            await client.close_position(
                position_id=pos.position_id,
                symbol=pos.symbol,
                side=pos.side,
                volume=pos.volume,
            )
            logger.info("Successfully closed {}", pos.position_id)

if __name__ == "__main__":
    asyncio.run(main())
