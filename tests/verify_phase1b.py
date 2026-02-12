import sys
import os
import shutil
from pathlib import Path
import asyncio

# Add project root
sys.path.append(os.getcwd())

from src.config import load_config
from src.signal.scanner_bridge import ScannerBridge
from src.decision.agent_bridge import AgentBridge
# from src.data.fx_data_fetcher import FxDataFetcher # Not used directly here


async def test_phase1b():
    print("[START] Phase 1b Verification...")

    # 1. Load Config
    config = load_config("config/e8_signature_50k.yaml")
    print(f"[OK] Config loaded. Symbols: {config.symbols}")

    # 2. Setup Data Fetcher (Mock)

    # 3. Run Scanner via Bridge
    print("\n[STEP] Running Scanner Bridge...")
    scanner = ScannerBridge(
        scanner_path=config.scanner.project_path, topk=config.scanner.topk, profile="fx"
    )

    # We pass a date to force pipeline run
    signals = scanner.run_pipeline(
        date="2024-01-05",  # Match our scanner mock data range
        tickers=config.symbols,
    )

    if not signals:
        print("[FAIL] Scanner returned no signals. Check logs.")
        return

    print(f"[OK] Scanner returned {len(signals)} signals:")
    for s in signals:
        print(f"   - {s.instrument}: Rank {s.rank}, Score {s.score:.4f}, Conf {s.confidence}")

    # 4. Run Agent Bridge
    print("\n[STEP] Running Agent Bridge (Mock Mode likely)...")
    agents = AgentBridge(
        agents_path=config.agents.project_path, selected_analysts=config.agents.selected_analysts
    )

    decisions = agents.decide_batch(
        signals=[s.to_qlib_data() | {"instrument": s.instrument} for s in signals],
        trade_date="2024-01-05",
    )

    print(f"[OK] Agents made {len(decisions)} decisions:")
    for d in decisions:
        print(f"   - {d.symbol}: {d.decision}")

    print("\n[DONE] Phase 1b Complete! System is integrated.")


if __name__ == "__main__":
    # Ensure logs dir exists
    Path("logs").mkdir(exist_ok=True)
    asyncio.run(test_phase1b())
