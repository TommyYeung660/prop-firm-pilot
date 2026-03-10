"""
Quote Freshness Inspector — verifies MatchTrader quote timestamps for data_freshness gate.

Logs into MatchTrader, fetches live quotes for all configured FX pairs,
and shows whether each quote would pass the data_freshness tactical gate
(age < 600 seconds by default).

Run with:
    uv run python scripts/check_quote_freshness.py
"""

import asyncio
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.execution.matchtrader_client import MatchTraderClient

# FX pairs to check (broker symbols use dot suffix)
PAIRS = ["EURUSD.", "GBPUSD.", "USDJPY.", "AUDUSD."]
DATA_FRESHNESS_THRESHOLD_S = 600  # from config data_max_age_seconds


async def main() -> None:
    load_dotenv()

    base_url = os.getenv("MATCHTRADER_API_URL", "").rstrip("/")
    email = os.getenv("MATCHTRADER_USERNAME", "")
    password = os.getenv("MATCHTRADER_PASSWORD", "")
    broker_id = os.getenv("MATCHTRADER_BROKER_ID", "2")
    account_id = os.getenv("MATCHTRADER_ACCOUNT_ID")

    now_utc = datetime.now(timezone.utc)
    logger.info("=" * 70)
    logger.info("Quote Freshness Inspector")
    logger.info("Current UTC time: {}", now_utc.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Freshness threshold: {}s", DATA_FRESHNESS_THRESHOLD_S)
    logger.info("=" * 70)

    async with MatchTraderClient(
        base_url=base_url,
        email=email,
        password=password,
        broker_id=broker_id,
        account_id=account_id,
    ) as client:
        tokens = await client.login()
        logger.info("Login OK — account={}, UUID={}", account_id, tokens.system_uuid)
        logger.info("")

        all_fresh = True

        for symbol in PAIRS:
            try:
                quote = await client.get_quote(symbol)
                quote_time = datetime.fromtimestamp(quote.timestamp_ms / 1000.0, tz=timezone.utc)
                now = datetime.now(timezone.utc)
                age_s = (now - quote_time).total_seconds()
                is_fresh = age_s < DATA_FRESHNESS_THRESHOLD_S
                status = "✅ FRESH" if is_fresh else "❌ STALE"

                if not is_fresh:
                    all_fresh = False

                logger.info("── {} ──", symbol)
                logger.info("  Bid:        {:.5f}", quote.bid)
                logger.info("  Ask:        {:.5f}", quote.ask)
                logger.info(
                    "  Timestamp:  {} ({}ms)",
                    quote_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    quote.timestamp_ms,
                )
                logger.info("  Age:        {:.1f}s", age_s)
                logger.info("  Gate:       {} (threshold={}s)", status, DATA_FRESHNESS_THRESHOLD_S)
                logger.info("")

            except Exception as e:
                all_fresh = False
                logger.error("── {} ── FAILED: {}", symbol, e)
                logger.info("")

        logger.info("=" * 70)
        if all_fresh:
            logger.info("RESULT: All quotes FRESH — data_freshness gate would PASS")
        else:
            logger.info("RESULT: Some quotes STALE — data_freshness gate would FAIL")
        logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
