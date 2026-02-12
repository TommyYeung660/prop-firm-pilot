"""
Bridge to qlib_market_scanner — runs the scanner pipeline and
converts its output into the format expected by TradingAgents.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class ScannerSignal:
    """Parsed signal from qlib_market_scanner output."""

    def __init__(
        self,
        instrument: str,
        score: float,
        rank: int,
        confidence: str,
        score_gap: float = 0.0,
        drop_distance: float = 0.0,
        topk_spread: float = 0.0,
        weight: float = 0.0,
    ) -> None:
        self.instrument = instrument
        self.score = score
        self.rank = rank
        self.confidence = confidence
        self.score_gap = score_gap
        self.drop_distance = drop_distance
        self.topk_spread = topk_spread
        self.weight = weight

    def to_qlib_data(self) -> Dict[str, Any]:
        """Convert to qlib_data dict for TradingAgents.propagate().

        This matches the interface in run_qlib_integration.py:
            qlib_data = {
                "score": ..., "signal_strength": ..., "confidence": ...,
                "score_gap": ..., "drop_distance": ..., "topk_spread": ...,
            }
        """
        # Map confidence to signal_strength
        strength_map = {"high": "STRONG", "medium": "MODERATE", "low": "WEAK"}
        return {
            "score": self.score,
            "signal_strength": strength_map.get(self.confidence, "MODERATE"),
            "confidence": self.confidence,
            "score_gap": self.score_gap,
            "drop_distance": self.drop_distance,
            "topk_spread": self.topk_spread,
        }

    def __repr__(self) -> str:
        return (
            f"ScannerSignal({self.instrument}, score={self.score:.4f}, "
            f"rank={self.rank}, confidence={self.confidence})"
        )


class ScannerBridge:
    """Bridge to run qlib_market_scanner and parse its output.

    The scanner is run as a subprocess to avoid Python environment conflicts.
    """

    def __init__(
        self,
        scanner_path: str | Path,
        topk: int = 3,
        profile: str = "fx",
    ) -> None:
        self._scanner_path = Path(scanner_path).resolve()
        self._topk = topk
        self._profile = profile

        if not self._scanner_path.exists():
            logger.warning("ScannerBridge: scanner path does not exist: {}", self._scanner_path)

    def run_pipeline(
        self,
        date: str | None = None,
        tickers: List[str] | None = None,
    ) -> List[ScannerSignal]:
        """Run the scanner pipeline and return parsed signals.

        Args:
            date: Override date for the pipeline (YYYY-MM-DD). None = today.
            tickers: Optional list of tickers to scan (comma separated string passed to CLI).

        Returns:
            List of ScannerSignal sorted by rank (best first).
        """
        logger.info(
            "ScannerBridge: running pipeline (profile={}, date={})",
            self._profile,
            date or "default",
        )

        # Build command: python -m src.main --profile fx ...
        cmd = [
            sys.executable,
            "-m",
            "src.main",
            "--profile",
            self._profile,
        ]

        if date:
            # Check if scanner supports --date (main.py typically infers from date range)
            # Our modified main.py has --start/--end. Let's use --end as the target date.
            # And set start to something reasonable if needed, or rely on config default.
            # For backtesting/verification, date is usually the end date.
            cmd.extend(["--end", date])

        if tickers:
            cmd.extend(["--tickers", ",".join(tickers)])

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self._scanner_path),
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout for data download + training
            )

            if result.returncode != 0:
                logger.error(
                    "ScannerBridge: pipeline failed (exit={}):\nstdout: {}\nstderr: {}",
                    result.returncode,
                    result.stdout[-1000:],  # Last 1000 chars
                    result.stderr[-1000:],
                )
                # Fallback: Try to read the output file anyway if it exists
                # This is useful if the pipeline failed at a later stage but signals were generated,
                # or if we are in a dev environment where we manually placed a mock signal file.
                signals_path = self._scanner_path / "outputs" / "signals" / "signals.csv"
                if signals_path.exists():
                    logger.warning(
                        "ScannerBridge: attempting fallback to existing signals file: {}",
                        signals_path,
                    )
                    return self.load_signals_from_file(signals_path)

                return []

            logger.info("ScannerBridge: pipeline finished successfully.")

            # Read output file directly
            # Path: outputs/signals/signals.csv
            signals_path = self._scanner_path / "outputs" / "signals" / "signals.csv"
            return self.load_signals_from_file(signals_path)

        except subprocess.TimeoutExpired:
            logger.error("ScannerBridge: pipeline timed out after 600s")
            return []
        except FileNotFoundError as e:
            logger.error("ScannerBridge: failed to run pipeline: {}", e)
            return []

    def load_signals_from_file(self, path: str | Path) -> List[ScannerSignal]:
        """Load signals from a pre-existing signals.csv file."""
        path = Path(path)
        if not path.exists():
            logger.error("ScannerBridge: signals file not found: {}", path)
            return []

        signals = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    try:
                        # CSV columns: datetime,instrument,score,rank,confidence,score_gap,drop_distance,topk_spread,weight
                        # But scanner output might vary slightly. Let's be robust.

                        inst = row.get("instrument", row.get("ticker", ""))
                        if not inst:
                            continue

                        signal = ScannerSignal(
                            instrument=inst,
                            score=float(row.get("score", 0)),
                            rank=int(float(row.get("rank", i + 1))),  # int(float()) handles "1.0"
                            confidence=row.get("confidence", "medium"),
                            score_gap=float(row.get("score_gap", 0)),
                            drop_distance=float(row.get("drop_distance", 0)),
                            topk_spread=float(row.get("topk_spread", 0)),
                            weight=float(row.get("weight", 0)),
                        )
                        signals.append(signal)
                    except (ValueError, TypeError) as e:
                        logger.warning("ScannerBridge: skipping malformed row {}: {}", i, e)

        except Exception as e:
            logger.error("ScannerBridge: failed to read CSV: {}", e)
            return []

        # Filter by topk if needed, or just return all
        signals.sort(key=lambda s: s.rank)
        logger.info("ScannerBridge: loaded {} signals (top: {})", len(signals), signals[:3])
        return signals
