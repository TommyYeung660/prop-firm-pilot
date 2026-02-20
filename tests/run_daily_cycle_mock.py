import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root
sys.path.append(os.getcwd())

from loguru import logger

from src.config import load_config
from src.main import PropFirmPilot

# Configure logging to stdout
logger.remove()
logger.add(
    sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | {message}"
)


async def run_mock_cycle():
    print("\n[TEST] Starting Phase 1 End-to-End Simulation")
    print("==================================================")

    # 1. Load Config
    config = load_config("config/e8_signature_50k.yaml")

    # 2. Mock MatchTrader Client
    # We want to intercept login, get_balance, and open_position
    mock_client = AsyncMock()

    # Mock Balance
    mock_balance = MagicMock()
    mock_balance.balance = 50000.0
    mock_balance.equity = 50000.0
    mock_balance.margin = 0.0
    mock_client.get_balance.return_value = mock_balance

    # Mock Open Position
    mock_order = MagicMock()
    mock_order.success = True
    mock_order.position_id = "POS_123456"
    mock_order.message = "Order opened successfully"
    mock_client.open_position.return_value = mock_order

    # Mock Open Positions List (Empty initially)
    mock_client.get_open_positions.return_value = []

    # Context Manager support
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    # 3. Patch MatchTraderClient class
    with patch("src.main.MatchTraderClient", return_value=mock_client):
        # 4. Initialize Pilot
        pilot = PropFirmPilot(config)

        # Force ScannerBridge to look at our mock data path if needed,
        # or rely on the fallback logic we just added.
        # Ensure we have signals to act on.

        # 5. Run Cycle
        print("\n[RUN] Running Daily Cycle (Simulated)...")
        # Use the date matching our mock signals file to ensure hits
        await pilot.run_daily_cycle(date_override="2024-01-05")

    print("\n[TEST] Simulation Complete.")
    print("Check the logs above to verify:")
    print("1. Scanner loaded signals")
    print("2. Agents made decisions")
    print("3. Trade was 'executed' (Mock)")


if __name__ == "__main__":
    asyncio.run(run_mock_cycle())
