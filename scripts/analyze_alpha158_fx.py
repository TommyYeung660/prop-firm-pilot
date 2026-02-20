"""
Analyze Alpha158 factor predictive power on FX data.

Reads Qlib binary FX data, computes Alpha158 factors, evaluates each factor's
time-series IC/IR, and classifies them as effective/weak/dead.

Run with:
    uv run python scripts/analyze_alpha158_fx.py

Options:
    --data-dir PATH     Path to qlib_fx data directory (default: auto-detect)
    --output PATH       CSV output path (default: data/alpha158_fx_ic_ir_report.csv)
    --forward-period N  Forward return period in days (default: 1)
    --ic-window N       Rolling IC window size (default: 20)
"""

import argparse
import os
import sys
from pathlib import Path

from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.research.alpha158_evaluator import Alpha158Evaluator


def find_data_dir() -> Path:
    """Auto-detect the qlib_fx data directory.

    Searches relative paths from the project root.

    Returns:
        Path to qlib_fx directory.

    Raises:
        FileNotFoundError: If data directory cannot be found.
    """
    project_root = Path(__file__).parent.parent.resolve()
    candidates = [
        project_root / ".." / "qlib_market_scanner" / "data" / "qlib_fx",
        project_root / ".." / ".." / "qlib_market_scanner" / "data" / "qlib_fx",
        Path.home() / "CursorProjects" / "qlib_market_scanner" / "data" / "qlib_fx",
    ]

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and (resolved / "features").exists():
            logger.info("Found qlib_fx data at: {}", resolved)
            return resolved

    raise FileNotFoundError(
        "Cannot find qlib_fx data directory. Use --data-dir to specify the path manually."
    )


def main() -> None:
    """Run Alpha158 FX factor IC/IR analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze Alpha158 factor predictive power on FX data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to qlib_fx data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/alpha158_fx_ic_ir_report.csv",
        help="CSV output path (default: data/alpha158_fx_ic_ir_report.csv)",
    )
    parser.add_argument(
        "--forward-period",
        type=int,
        default=1,
        help="Forward return period in days (default: 1)",
    )
    parser.add_argument(
        "--ic-window",
        type=int,
        default=20,
        help="Rolling IC window size (default: 20)",
    )

    args = parser.parse_args()

    # Find data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            logger.error("Data directory not found: {}", data_dir)
            sys.exit(1)
    else:
        data_dir = find_data_dir()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    logger.info("Starting Alpha158 FX factor analysis")
    logger.info("Data dir: {}", data_dir)
    logger.info("Output: {}", output_path)
    logger.info(
        "Forward period: {} day(s), IC window: {}",
        args.forward_period,
        args.ic_window,
    )

    evaluator = Alpha158Evaluator(
        data_dir=data_dir,
        forward_period=args.forward_period,
        ic_window=args.ic_window,
    )

    results = evaluator.run()

    if not results:
        logger.error("No results produced. Check data directory.")
        sys.exit(1)

    # Print report
    report = evaluator.generate_report(results)
    logger.info("Analysis complete:\n{}", report)

    # Save CSV
    evaluator.save_csv(results, output_path)
    logger.info("Report saved to {}", output_path)

    # Summary stats
    effective = [r for r in results if r.category == "effective"]

    if len(effective) < 10:
        logger.warning(
            "Only {} effective factors found. "
            "Consider developing FX-specific factors (Carry, Momentum, MR).",
            len(effective),
        )


if __name__ == "__main__":
    main()
