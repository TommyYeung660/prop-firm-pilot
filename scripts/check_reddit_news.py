"""
Verify whether reddit global news data is actually available.

Checks if TradingAgents' get_reddit_global_news() returns real data
or silently returns empty string (which would mean 'local' vendor
is effectively a no-op for global news).

Usage:
    python scripts/check_reddit_news.py
"""

import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    # Add TradingAgents to sys.path
    agents_path = Path(__file__).resolve().parent.parent.parent.parent / "TradingAgents"
    if not agents_path.exists():
        print(f"ERROR: TradingAgents not found at {agents_path}")
        sys.exit(1)

    sys.path.insert(0, str(agents_path))

    from tradingagents.dataflows.local import get_reddit_global_news

    today = datetime.now().strftime("%Y-%m-%d")
    print(f"Checking reddit global news for date: {today}")
    print(f"TradingAgents path: {agents_path}")

    # Check if reddit_data directory exists
    reddit_data_dir = agents_path / "tradingagents" / "dataflows" / "data" / "reddit_data"
    if not reddit_data_dir.exists():
        # Try alternate location from config
        from tradingagents.dataflows.config import DATA_DIR
        reddit_data_dir = Path(DATA_DIR) / "reddit_data"

    print(f"Reddit data directory: {reddit_data_dir}")
    print(f"  Exists: {reddit_data_dir.exists()}")

    if reddit_data_dir.exists():
        contents = list(reddit_data_dir.iterdir())
        print(f"  Contents: {len(contents)} items")
        for item in contents[:10]:
            print(f"    - {item.name}")
    else:
        print("  WARNING: Directory does not exist — get_reddit_global_news() will return empty!")

    # Actually call the function
    print("\nCalling get_reddit_global_news()...")
    try:
        result = get_reddit_global_news(today, look_back_days=7, limit=5)
        if result:
            print(f"SUCCESS: Got {len(result)} chars of news data")
            print(f"Preview (first 500 chars):\n{result[:500]}")
        else:
            print("EMPTY: get_reddit_global_news() returned empty string")
            print("  This means 'local' vendor for get_global_news is a NO-OP.")
            print("  Consider switching to 'openai' or implementing alpha_vantage global news.")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
