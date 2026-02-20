import asyncio
import os
from dotenv import load_dotenv
from src.execution.matchtrader_client import MatchTraderClient

load_dotenv()


async def close_all():
    client = MatchTraderClient(
        base_url="https://mtr.e8markets.com",
        email=os.getenv("MATCHTRADER_USERNAME"),
        password=os.getenv("MATCHTRADER_PASSWORD"),
        broker_id=os.getenv("MATCHTRADER_BROKER_ID", "2"),
        account_id="950383",
    )

    async with client:
        await client.login()
        positions = await client.get_open_positions()
        print(f"Found {len(positions)} open positions")
        for p in positions:
            print(f"Closing {p.position_id} {p.symbol} {p.volume}...")
            res = await client.close_position(
                position_id=p.position_id, symbol=p.symbol, side=p.side, volume=p.volume
            )
            print(res.message)


if __name__ == "__main__":
    asyncio.run(close_all())
