import asyncio
import os
from dotenv import load_dotenv
from src.execution.matchtrader_client import MatchTraderClient

load_dotenv()

async def test_balance():
    async with MatchTraderClient(
        base_url=os.getenv("MATCHTRADER_API_URL"),
        email=os.getenv("MATCHTRADER_USERNAME"),
        password=os.getenv("MATCHTRADER_PASSWORD"),
    ) as client:
        await client.login()
        balance = await client.get_balance()
        print(f"Balance: {balance.balance}")
        print(f"Equity: {balance.equity}")

asyncio.run(test_balance())
