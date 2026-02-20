import asyncio
import os
import time
from dotenv import load_dotenv

from src.config import load_config
from src.decision_store.sqlite_store import DecisionStore
from src.decision.schemas import TradeIntent
from src.execution.matchtrader_client import MatchTraderClient
from src.execution.instrument_registry import InstrumentRegistry
from src.compliance.prop_firm_guard import PropFirmGuard
from src.execution.position_sizer import PositionSizer
from src.execution.engine import ExecutionEngine

load_dotenv()


class MockPositionSizer:
    def calculate_volume(self, symbol: str, equity: float, sl_pips: float) -> float:
        return 0.01  # Force 0.01 lot for the test trade

    def calculate_risk_amount(self, symbol: str, volume: float, sl_pips: float) -> float:
        return 1.0  # Dummy $1.00 risk


async def test_live_execution():
    print("Loading config: config/e8_one_5k_challenge.yaml")
    config = load_config("config/e8_one_5k_challenge.yaml")

    # 1. Initialize MatchTrader
    client = MatchTraderClient(
        base_url="https://mtr.e8markets.com",
        email=os.getenv("MATCHTRADER_USERNAME"),
        password=os.getenv("MATCHTRADER_PASSWORD"),
        broker_id=os.getenv("MATCHTRADER_BROKER_ID", "2"),
        account_id="950383",  # E8 One 5K Challenge Account
    )

    async with client:
        await client.login()
        print("\n[MatchTrader Logged In]")

        # 2. Build Instrument Registry
        registry = await InstrumentRegistry.from_matchtrader(client, ["EURUSD"])

        # 3. Create In-Memory DB & Insert Mock Intent
        store = DecisionStore(":memory:")
        intent = TradeIntent(
            trade_date="2026-02-20",
            symbol="EURUSD",
            scanner_score=0.9,
            status="ready_for_exec",
            suggested_side="BUY",
            suggested_sl_pips=50,
            suggested_tp_pips=100,
        )
        store.insert_intent(intent)

        # 4. Init Guard & Sizer
        guard = PropFirmGuard(config.compliance, config.execution, config.instruments)
        sizer = MockPositionSizer()

        # 5. Run Execution Engine
        engine = ExecutionEngine(
            store=store,
            guard=guard,
            matchtrader=client,
            sizer=sizer,
            config=config,
            instrument_registry=registry,
        )

        print("\n[Running Execution Engine on Mock Intent]")
        await engine.execute_ready_intents()

        # 6. Check Result in Store
        updated_intent = store.get_intent(intent.id)
        print(f"Intent Status: {updated_intent.status}")

        if updated_intent.status == "opened" and updated_intent.position_id:
            pos_id = updated_intent.position_id
            print(f"Trade successfully opened! Position ID: {pos_id}")

            # Immediately close the trade to minimize risk
            print("\n[Closing Trade Immediately to Minimize Risk]")
            time.sleep(1)  # small delay
            res = await client.close_position(
                position_id=pos_id, symbol=registry.to_broker("EURUSD"), side="BUY", volume=0.01
            )
            if res.success:
                print(f"Trade {pos_id} successfully closed!")
            else:
                print(f"Failed to close trade: {res.message}")
        else:
            print("Trade execution failed or rejected.")
            print(f"Reason: {updated_intent.execution_error or updated_intent.compliance_snapshot}")


if __name__ == "__main__":
    asyncio.run(test_live_execution())
