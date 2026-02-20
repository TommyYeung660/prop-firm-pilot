import asyncio
import os
from dotenv import load_dotenv
from src.execution.matchtrader_client import MatchTraderClient

load_dotenv()


async def test_5k_account_login():
    email = os.getenv("MATCHTRADER_USERNAME")
    password = os.getenv("MATCHTRADER_PASSWORD")
    broker_id = os.getenv("MATCHTRADER_BROKER_ID", "2")
    account_id = "950383"  # E8 One 5K Challenge Account

    print(f"Connecting to E8 MatchTrader (Account: {account_id})...")
    client = MatchTraderClient(
        base_url="https://mtr.e8markets.com",
        email=email,
        password=password,
        broker_id=broker_id,
        account_id=account_id,
    )

    async with client:
        try:
            tokens = await client.login()
            print("Login Successful!")
            print(f"System UUID: {tokens.system_uuid}")

            balance = await client.get_balance()
            print(f"\n[Balance Info]")
            print(f"Balance: {balance.balance}")
            print(f"Equity: {balance.equity}")
            print(f"Margin: {balance.margin}")
            print(f"Free Margin: {balance.free_margin}")

            positions = await client.get_open_positions()
            print(f"\n[Open Positions]: {len(positions)}")
            for p in positions:
                print(f"- {p.symbol} {p.side} {p.volume} lots (Profit: {p.profit})")

            instruments = await client.get_effective_instruments()
            print(f"\n[Effective Instruments]: {len(instruments)}")
            eurusd = next((i for i in instruments if "EURUSD" in i.symbol), None)
            if eurusd:
                print(f"Found EURUSD: {eurusd.symbol} (Min Lot: {eurusd.volume_min})")

            print("\nMatchTraderClient 5K Account Connection Test Passed!")

        except Exception as e:
            print(f"Error during connection: {e}")


if __name__ == "__main__":
    asyncio.run(test_5k_account_login())
