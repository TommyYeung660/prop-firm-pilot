"""
Live EODHD websocket probe for production troubleshooting.

Verifies whether the websocket feed is delivering ticks for configured FX
symbols and compares that against same-day REST 1m lag so degraded-state
analysis can distinguish websocket outages from REST provider delay.

Usage:
    uv run python scripts/check_eodhd_websocket_live.py
    uv run python scripts/check_eodhd_websocket_live.py --symbols EURUSD GBPUSD --duration 20
"""

import argparse
from pathlib import Path

from src.config import load_config
from src.diagnostics.eodhd_websocket_live import (
    load_dotenv_api_key,
    probe_rest_bars,
    probe_websocket,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe EODHD websocket and REST 1m lag")
    parser.add_argument(
        "--config",
        default="config/e8_one_5k_challenge.yaml",
        help="Config file used to resolve default symbols",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Optional symbol override (default: websocket symbols from config)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Websocket probe duration in seconds",
    )
    parser.add_argument(
        "--raw-samples",
        type=int,
        default=8,
        help="Maximum number of raw websocket samples to print",
    )
    return parser.parse_args()


async def _run() -> None:
    args = _parse_args()
    config = load_config(args.config)
    symbols = args.symbols or config.websocket.symbols or list(config.symbols)
    api_key = load_dotenv_api_key(Path(".env"))
    if not api_key:
        raise SystemExit("EODHD_API_KEY missing in environment or .env")

    websocket_probe = await probe_websocket(
        api_token=api_key,
        symbols=symbols,
        duration_seconds=args.duration,
        raw_sample_limit=args.raw_samples,
    )
    rest_probe = await probe_rest_bars(api_token=api_key, symbols=symbols)

    print("=== WebSocket Raw Samples ===")
    for sample in websocket_probe["raw_samples"]:
        print(sample)

    print("\n=== WebSocket Tick Summary ===")
    for symbol in symbols:
        print({**{"symbol": symbol}, **websocket_probe["tick_summary"][symbol]})

    print("\n=== REST 1m Lag Summary ===")
    for symbol in symbols:
        print(rest_probe[symbol])


def main() -> None:
    """CLI entry point."""
    import asyncio

    asyncio.run(_run())


if __name__ == "__main__":
    main()
