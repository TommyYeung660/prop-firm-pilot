import asyncio
import os
from pathlib import Path
from src.config import load_config
from src.decision_store.sqlite_store import DecisionStore
from src.decision.agent_bridge import AgentBridge
from src.decision.schemas import TradeIntent


async def test_e2e_scanner_llm():
    # 1. Setup Config & Store
    config = load_config("config/e8_signature_50k.yaml")
    store = DecisionStore(":memory:")

    date = "2026-02-12"

    # 2. Mock a Scanner Signal
    # We will directly insert a TradeIntent as if ScannerBridge just ran
    intent = TradeIntent(
        trade_date=date,
        symbol="EURUSD",
        source="scanner",
        scanner_score=0.85,
        scanner_confidence="high",
        scanner_rank=1,
    )
    store.insert_intent(intent)
    intent_id = intent.id
    print(f"Created Intent ID: {intent_id}")

    # 3. Claim the Intent
    claimed = store.claim_next_pending("worker_1")
    if not claimed:
        print("Failed to claim intent!")
        return

    print(f"Claimed Intent: {claimed.symbol} for date {claimed.trade_date}")

    # 4. AgentBridge (LLM Decision)
    bridge = AgentBridge(
        agents_path="../../TradingAgents",
        selected_analysts=["macro"],  # Use only macro to make it fast
        config=config.agents.model_dump(),
    )

    print("Running LLM Decision Engine...")
    decision = await bridge.decide_async(
        symbol=claimed.symbol,
        trade_date=claimed.trade_date,
        qlib_data={
            "score": claimed.scanner_score,
            "confidence": claimed.scanner_confidence,
        },
    )

    print(f"\nDecision Result for {decision.symbol}:")
    print(f"Action: {decision.decision}")

    # 5. Save Decision back to store
    store.update_intent_decision(
        intent_id=claimed.id,
        side=decision.decision,
        sl_pips=20.0,
        tp_pips=40.0,
        risk_report=decision.risk_report[:500] if decision.risk_report else "No specific reasoning",
        state_json="{}",
    )

    # 6. Verify Store
    updated = store.get_intent(claimed.id)
    print("\nFinal Intent Status in DB:")
    print(f"Status: {updated.status}")
    print(f"Decision: {updated.suggested_side}")
    print(f"Reasoning: {updated.agent_risk_report}")


if __name__ == "__main__":
    asyncio.run(test_e2e_scanner_llm())
