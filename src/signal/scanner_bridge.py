"""
Bridge to qlib_market_scanner — runs the scanner pipeline and
converts its output into the format expected by TradingAgents.

v1.4.0: Added pipeline caching to avoid redundant retrain runs when
the daily candle hasn't closed yet (signals won't change intraday).
"""

import csv
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class _PipelineCache:
    """Cached result from a scanner pipeline run.

    When the scanner's daily model hasn't received today's candle yet,
    re-running the pipeline produces identical signals.  Caching avoids
    ~10 min of wasted retrain compute per redundant call.
    """

    request_date: str  # date param passed to run_pipeline
    interval: str
    signal_date: str  # actual date of the returned signals
    signals: list = field(default_factory=list)
    is_stale: bool = False  # True when signal_date < request_date


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
        entry_timeframe: str = "4h",
    ) -> None:
        self.instrument = instrument
        self.score = score
        self.rank = rank
        self.confidence = confidence
        self.score_gap = score_gap
        self.drop_distance = drop_distance
        self.topk_spread = topk_spread
        self.weight = weight
        self.entry_timeframe = entry_timeframe

    def to_qlib_data(self) -> dict[str, Any]:
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
            "entry_timeframe": self.entry_timeframe,
        }

    def __repr__(self) -> str:
        return (
            f"ScannerSignal({self.instrument}, score={self.score:.4f}, "
            f"rank={self.rank}, confidence={self.confidence})"
        )


class ScannerBridge:
    """Bridge to run qlib_market_scanner and parse its output.

    The scanner is run as a subprocess to avoid Python environment conflicts.

    v1.4.0: Caches pipeline results within the same (date, interval) to skip
    redundant retrain runs when the daily candle hasn't closed yet.
    """

    def __init__(
        self,
        scanner_path: str | Path,
        topk: int = 3,
        profile: str = "fx",
        entry_timeframe: str = "4h",
    ) -> None:
        self._scanner_path = Path(scanner_path).resolve()
        self._topk = topk
        self._profile = profile
        self._entry_timeframe = entry_timeframe
        self._cache: _PipelineCache | None = None

        if not self._scanner_path.exists():
            logger.warning("ScannerBridge: scanner path does not exist: {}", self._scanner_path)

    # ── Pipeline cache helpers ──────────────────────────────────────────

    def _get_cached(self, date: str | None, interval: str) -> list[ScannerSignal] | None:
        """Return cached signals if a valid cache entry exists, else None."""
        if self._cache is None:
            return None
        if self._cache.request_date != date or self._cache.interval != interval:
            return None

        logger.info(
            "ScannerBridge: returning cached signals (signal_date={}, stale={}) "
            "— skipping pipeline re-run for date={}",
            self._cache.signal_date,
            self._cache.is_stale,
            date,
        )
        return list(self._cache.signals)  # defensive copy

    def _update_cache(
        self,
        request_date: str | None,
        interval: str,
        signals: list[ScannerSignal],
        signal_date: str,
    ) -> None:
        """Store pipeline result in cache."""
        is_stale = request_date is not None and signal_date != request_date
        self._cache = _PipelineCache(
            request_date=request_date or "",
            interval=interval,
            signal_date=signal_date,
            signals=list(signals),  # defensive copy
            is_stale=is_stale,
        )
        if is_stale:
            logger.info(
                "ScannerBridge: cached stale signals (signal_date={} != request_date={}) "
                "— subsequent calls for same date will skip pipeline",
                signal_date,
                request_date,
            )

    def invalidate_cache(self) -> None:
        """Clear the pipeline cache.  Called externally when a cache bust is needed."""
        self._cache = None

    def _is_recoverable_pipeline_failure(self, stdout: str, stderr: str) -> bool:
        """Return True when the subprocess failed after already generating usable signals."""
        failure_text = f"{stdout}\n{stderr}".lower()
        return "the benchmark [" in failure_text and "does not exist" in failure_text

    # ── Main pipeline ───────────────────────────────────────────────────

    def run_pipeline(
        self,
        date: str | None = None,
        tickers: list[str] | None = None,
        force_retrain: bool = True,
        interval: str = "1d",
        max_signal_age_days: int | None = None,
    ) -> list[ScannerSignal]:
        """Run the scanner pipeline and return parsed signals.

        Args:
            date: Override date for the pipeline (YYYY-MM-DD). None = today.
            tickers: Optional list of tickers to scan (comma separated string passed to CLI).
            force_retrain: Force model retrain (avoids stale cached models across days).
            interval: Data interval for multi-timeframe scanning (e.g. '1d', '4h', '1h').
            max_signal_age_days: If provided, reject signals older than this many days.

        Returns:
            List of ScannerSignal sorted by rank (best first).
        """
        # ── Cache check: skip expensive pipeline if signals won't change ──
        # Daily models only produce new scores after the day's candle closes.
        # Re-running the pipeline intraday yields identical results, wasting
        # ~10 min of retrain compute per call.
        cached = self._get_cached(date, interval)
        if cached is not None:
            return cached

        logger.info(
            "ScannerBridge: running pipeline (profile={}, date={})",
            self._profile,
            date or "default",
        )

        # Use 'uv run' to ensure we use the scanner's own environment/dependencies
        # instead of inheriting the current pilot environment.
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "src.main",
            "--profile",
            self._profile,
        ]

        # Force retrain to avoid stale cached models producing identical scores
        # across days. Without this, the model cache hash only changes when
        # train/valid segment boundaries shift (rarely with daily additions).
        if force_retrain:
            cmd.append("--retrain")

        if date:
            # Check if scanner supports --date (main.py typically infers from date range)
            # Our modified main.py has --start/--end. Let's use --end as the target date.
            # And set start to something reasonable if needed, or rely on config default.
            # For backtesting/verification, date is usually the end date.
            cmd.extend(["--end", date])

        if tickers:
            cmd.extend(["--tickers", ",".join(tickers)])

        cmd.extend(["--interval", interval])
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self._scanner_path),
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout for data download + training
            )

            if result.returncode != 0:
                signals_path = self._scanner_path / "outputs" / "signals" / "signals.csv"
                if signals_path.exists():
                    signals, signal_date = self.load_signals_from_file(
                        signals_path,
                        target_date=date,
                        max_signal_age_days=max_signal_age_days,
                    )
                    if signals:
                        if self._is_recoverable_pipeline_failure(result.stdout, result.stderr):
                            logger.warning(
                                "ScannerBridge: recoverable pipeline failure after signals "
                                "generated (exit={}) — using signals file {}",
                                result.returncode,
                                signals_path,
                            )
                        else:
                            logger.error(
                                "ScannerBridge: pipeline failed (exit={}):\nstdout: {}\nstderr: {}",
                                result.returncode,
                                result.stdout[-1000:],  # Last 1000 chars
                                result.stderr[-1000:],
                            )
                            logger.warning(
                                "ScannerBridge: attempting fallback to existing signals file: {}",
                                signals_path,
                            )
                        self._update_cache(date, interval, signals, signal_date)
                        return signals

                logger.error(
                    "ScannerBridge: pipeline failed (exit={}):\nstdout: {}\nstderr: {}",
                    result.returncode,
                    result.stdout[-1000:],  # Last 1000 chars
                    result.stderr[-1000:],
                )

                return []

            logger.info("ScannerBridge: pipeline finished successfully.")

            # Read output file directly
            # Path: outputs/signals/signals.csv
            signals_path = self._scanner_path / "outputs" / "signals" / "signals.csv"
            signals, signal_date = self.load_signals_from_file(
                signals_path,
                target_date=date,
                max_signal_age_days=max_signal_age_days,
            )
            if signals:
                self._update_cache(date, interval, signals, signal_date)
            return signals

        except subprocess.TimeoutExpired:
            logger.error("ScannerBridge: pipeline timed out after 600s")
            return []
        except FileNotFoundError as e:
            logger.error("ScannerBridge: failed to run pipeline: {}", e)
            return []

    def load_signals_from_file(
        self,
        path: str | Path,
        target_date: str | None = None,
        max_signal_age_days: int | None = None,
    ) -> tuple[list[ScannerSignal], str]:
        """Load signals from a pre-existing signals.csv file.

        Args:
            path: Path to the signals.csv file.
            target_date: If provided (YYYY-MM-DD), only return signals for this date.
                         If no signals match the exact date, falls back to the latest date.
                         If None, returns only the latest date's signals.
            max_signal_age_days: If provided, reject signals older than this many days
                                 relative to target_date. None = no freshness check.

        Returns:
            Tuple of (signals, chosen_date).  chosen_date is the actual date of
            the returned signals (may differ from target_date if data is stale).
            Returns ([], "") if no signals are available.
        """
        path = Path(path)
        if not path.exists():
            logger.error("ScannerBridge: signals file not found: {}", path)
            return [], ""

        all_signals: dict[str, list[ScannerSignal]] = {}
        try:
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    try:
                        inst = row.get("instrument", row.get("ticker", ""))
                        if not inst:
                            continue

                        row_date = row.get("datetime", "").split(" ")[0]  # YYYY-MM-DD

                        signal = ScannerSignal(
                            instrument=inst,
                            score=float(row.get("score", 0)),
                            rank=int(float(row.get("rank", i + 1))),
                            confidence=row.get("confidence", "medium"),
                            score_gap=float(row.get("score_gap", 0)),
                            drop_distance=float(row.get("drop_distance", 0)),
                            topk_spread=float(row.get("topk_spread", 0)),
                            weight=float(row.get("weight", 0)),
                            entry_timeframe=self._entry_timeframe,
                        )

                        if row_date not in all_signals:
                            all_signals[row_date] = []
                        all_signals[row_date].append(signal)

                    except (ValueError, TypeError) as e:
                        logger.warning("ScannerBridge: skipping malformed row {}: {}", i, e)

        except Exception as e:
            logger.error("ScannerBridge: failed to read CSV: {}", e)
            return [], ""

        if not all_signals:
            logger.warning("ScannerBridge: no signals found in {}", path)
            return [], ""

        # Pick the target date's signals, or fall back to latest date
        available_dates = sorted(all_signals.keys())
        if target_date and target_date in all_signals:
            chosen_date = target_date
        else:
            chosen_date = available_dates[-1]  # latest date
            if target_date and target_date not in all_signals:
                logger.warning(
                    "ScannerBridge: target_date {} not found in signals (available: {} to {}). "
                    "Falling back to latest date {}.",
                    target_date,
                    available_dates[0],
                    available_dates[-1],
                    chosen_date,
                )

        # v1.3.0: Signal freshness guard — reject stale signals
        if max_signal_age_days is not None and target_date:
            from datetime import date as date_type

            try:
                target_dt = date_type.fromisoformat(target_date)
                chosen_dt = date_type.fromisoformat(chosen_date)
                age_days = (target_dt - chosen_dt).days
                if age_days > max_signal_age_days:
                    logger.warning(
                        "ScannerBridge: signal date {} is {} days old (max={}), "
                        "rejecting stale signals for target_date {}",
                        chosen_date,
                        age_days,
                        max_signal_age_days,
                        target_date,
                    )
                    return [], ""
            except ValueError:
                logger.warning(
                    "ScannerBridge: could not parse dates for freshness check "
                    "(target={}, chosen={}), proceeding without check",
                    target_date,
                    chosen_date,
                )

        signals = all_signals[chosen_date]
        signals.sort(key=lambda s: s.rank)
        is_stale = target_date is not None and chosen_date != target_date
        logger.info(
            "ScannerBridge: loaded {} signals for date {} (target={}, stale={}, "
            "total dates in file: {})",
            len(signals),
            chosen_date,
            target_date,
            is_stale,
            len(available_dates),
        )
        return signals, chosen_date
