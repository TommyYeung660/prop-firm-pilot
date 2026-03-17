"""
Bridge to qlib_market_scanner — runs the scanner pipeline and
converts its output into the format expected by TradingAgents.

v1.4.0: Added pipeline caching to avoid redundant retrain runs when
the daily candle hasn't closed yet (signals won't change intraday).
"""

import csv
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

SUPPORTED_SIGNAL_SCHEMA_VERSIONS = {"fx_signal_v1"}
SUPPORTED_SCANNER_VERSIONS = {"v1.5.0", "v1.5.0_beta"}
PASSING_VALIDATION_STATUSES = {"", "ok", "pass", "passed", "ready", "success", "valid", "validated"}
REQUIRED_SIGNAL_COLUMNS = {
    "datetime",
    "instrument",
    "score",
    "rank",
    "score_gap",
    "drop_distance",
    "topk_spread",
    "confidence",
    "weight",
    "profile",
    "scanner_version",
    "schema_version",
    "cadence",
    "label_version",
    "regime_label",
    "market_date",
}


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
        profile: str = "fx",
        scanner_version: str = "",
        schema_version: str = "",
        cadence: str = "",
        label_version: str = "",
        regime_label: str = "",
        market_date: str = "",
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
        self.profile = profile
        self.scanner_version = scanner_version
        self.schema_version = schema_version
        self.cadence = cadence
        self.label_version = label_version
        self.regime_label = regime_label
        self.market_date = market_date

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
        benchmark: str = "FX",
    ) -> None:
        self._scanner_path = Path(scanner_path).resolve()
        self._topk = topk
        self._profile = profile
        self._entry_timeframe = entry_timeframe
        self._benchmark = benchmark
        self._cache: _PipelineCache | None = None
        self._last_rejection_reason_code: str = ""
        self._last_rejection_message: str = ""

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
        if is_stale:
            logger.info(
                "ScannerBridge: refusing to cache stale signals "
                "(signal_date={} != request_date={})",
                signal_date,
                request_date,
            )
            return
        self._cache = _PipelineCache(
            request_date=request_date or "",
            interval=interval,
            signal_date=signal_date,
            signals=list(signals),  # defensive copy
            is_stale=is_stale,
        )

    def invalidate_cache(self) -> None:
        """Clear the pipeline cache.  Called externally when a cache bust is needed."""
        self._cache = None

    def get_last_rejection_reason_code(self) -> str:
        """Return the most recent contract/freshness rejection reason, if any."""
        return self._last_rejection_reason_code

    def get_last_rejection_message(self) -> str:
        """Return the most recent contract/freshness rejection message, if any."""
        return self._last_rejection_message

    def _clear_rejection_reason(self) -> None:
        self._last_rejection_reason_code = ""
        self._last_rejection_message = ""

    def _set_rejection_reason(self, reason_code: str, message: str) -> None:
        self._last_rejection_reason_code = reason_code
        self._last_rejection_message = message

    def _first_existing_path(self, candidates: list[Path]) -> Path | None:
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if candidate.exists():
                return candidate
        return None

    def _bundle_family(self, signals_path: Path) -> str:
        if signals_path.parent.name == "signals" and len(signals_path.parents) > 1:
            parent_name = signals_path.parents[1].name
            if parent_name == "outputs":
                return "runtime"
            if parent_name == "scanner_outputs":
                return "shared_export"
        return "standalone"

    def _resolve_manifest_path(self, signals_path: Path) -> Path | None:
        family = self._bundle_family(signals_path)
        if family == "runtime":
            candidates = [signals_path.parents[1] / "manifest.json"]
        elif family == "shared_export":
            candidates = [signals_path.parents[2] / "manifest.json"]
        else:
            candidates = [signals_path.parent / "manifest.json"]
            if len(signals_path.parents) > 1:
                candidates.append(signals_path.parents[1] / "manifest.json")
            if len(signals_path.parents) > 2:
                candidates.append(signals_path.parents[2] / "manifest.json")
        return self._first_existing_path(candidates)

    def _resolve_metrics_path(self, signals_path: Path) -> Path | None:
        family = self._bundle_family(signals_path)
        if family == "runtime":
            candidates = [signals_path.parents[1] / "metrics" / "metrics.json"]
        elif family == "shared_export":
            candidates = [signals_path.parents[1] / "metrics" / "metrics.json"]
        else:
            candidates = [signals_path.parent / "metrics.json"]
            if len(signals_path.parents) > 1:
                candidates.append(signals_path.parents[1] / "metrics" / "metrics.json")
            if len(signals_path.parents) > 2:
                candidates.append(
                    signals_path.parents[2]
                    / "scanner_outputs"
                    / "metrics"
                    / "metrics.json"
                )
        return self._first_existing_path(candidates)

    def _read_json_file(self, path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path.name} must contain a JSON object")
        return payload

    def _load_contract_context(self, signals_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        family = self._bundle_family(signals_path)
        family_prefix = "" if family == "standalone" else f"{family} "
        manifest_path = self._resolve_manifest_path(signals_path)
        if manifest_path is None:
            raise ValueError(f"{family_prefix}manifest.json not found")
        manifest = self._read_json_file(manifest_path)
        metrics: dict[str, Any] = {}
        metrics_path = self._resolve_metrics_path(signals_path)
        if metrics_path is not None:
            metrics = self._read_json_file(metrics_path)
        return manifest, metrics

    def _validation_status(
        self, manifest: dict[str, Any], metrics: dict[str, Any]
    ) -> str:
        for payload in (manifest, metrics):
            validation = payload.get("validation", {})
            if isinstance(validation, dict):
                status = validation.get("status", "")
                if isinstance(status, str) and status.strip():
                    return status.strip().lower()
        return ""

    def _validate_contract(
        self,
        *,
        signals_path: Path,
        fieldnames: set[str],
    ) -> tuple[str, str, str, str]:
        try:
            manifest, metrics = self._load_contract_context(signals_path)
        except Exception as e:
            raise ValueError(f"missing or invalid manifest bundle: {e}") from e

        schema_versions = manifest.get("schema_versions")
        if not isinstance(schema_versions, dict):
            raise ValueError("manifest missing schema_versions")
        if not manifest.get("research_run_id"):
            raise ValueError("manifest missing research_run_id")

        manifest_schema = str(schema_versions.get("signals_csv", "")).strip()
        manifest_scanner_version = str(manifest.get("scanner_version", "")).strip()
        manifest_cadence = str(manifest.get("cadence", "")).strip()
        manifest_label_version = str(manifest.get("label_version", "")).strip()

        if manifest_schema not in SUPPORTED_SIGNAL_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported schema version: {manifest_schema}")
        if manifest_scanner_version not in SUPPORTED_SCANNER_VERSIONS:
            raise ValueError(f"unsupported scanner version: {manifest_scanner_version}")

        missing_columns = REQUIRED_SIGNAL_COLUMNS - fieldnames
        if missing_columns:
            raise ValueError(f"signals csv missing required columns: {sorted(missing_columns)}")

        validation_status = self._validation_status(manifest, metrics)
        if validation_status == "degraded":
            raise RuntimeError("scanner.bundle.degraded")
        if validation_status == "stale":
            raise RuntimeError("scanner.bundle.stale")
        if validation_status and validation_status not in PASSING_VALIDATION_STATUSES:
            raise ValueError(f"unsupported validation status: {validation_status}")

        return (
            manifest_schema,
            manifest_scanner_version,
            manifest_cadence,
            manifest_label_version,
        )

    def _is_recoverable_pipeline_failure(self, stdout: str, stderr: str) -> bool:
        """Return True when the subprocess failed after already generating usable signals."""
        failure_text = f"{stdout}\n{stderr}".lower()
        return "the benchmark [" in failure_text and "does not exist" in failure_text

    def _should_retry_without_benchmark(self, stdout: str, stderr: str) -> bool:
        """Return True when an older scanner CLI rejects the --benchmark argument."""
        failure_text = f"{stdout}\n{stderr}".lower()
        return "unrecognized arguments:" in failure_text and "--benchmark" in failure_text

    def _run_scanner_subprocess(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        """Run scanner CLI with consistent subprocess settings."""
        return subprocess.run(
            cmd,
            cwd=str(self._scanner_path),
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout for data download + training
        )

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
        self._clear_rejection_reason()
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
            "--benchmark",
            self._benchmark,
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

        cmd.extend(["--topk", str(self._topk)])
        cmd.extend(["--interval", interval])
        cmd_without_benchmark = list(cmd)
        if "--benchmark" in cmd_without_benchmark:
            benchmark_index = cmd_without_benchmark.index("--benchmark")
            del cmd_without_benchmark[benchmark_index : benchmark_index + 2]
        try:
            result = self._run_scanner_subprocess(cmd)
            if result.returncode != 0 and self._should_retry_without_benchmark(
                result.stdout, result.stderr
            ):
                logger.warning(
                    "ScannerBridge: scanner CLI rejected --benchmark; retrying without it "
                    "for backward compatibility"
                )
                result = self._run_scanner_subprocess(cmd_without_benchmark)

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
        self._clear_rejection_reason()
        if not path.exists():
            logger.error("ScannerBridge: signals file not found: {}", path)
            return [], ""

        all_signals: dict[str, list[ScannerSignal]] = {}
        try:
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = set(reader.fieldnames or [])
                try:
                    (
                        manifest_schema,
                        manifest_scanner_version,
                        manifest_cadence,
                        manifest_label_version,
                    ) = self._validate_contract(
                        signals_path=path,
                        fieldnames=fieldnames,
                    )
                except RuntimeError as e:
                    reason_code = str(e)
                    message = f"bundle rejected: {reason_code}"
                    self._set_rejection_reason(reason_code, message)
                    logger.warning("ScannerBridge: {}", message)
                    return [], ""
                except ValueError as e:
                    message = f"contract validation failed: {e}"
                    self._set_rejection_reason("scanner.contract.invalid", message)
                    logger.warning("ScannerBridge: {}", message)
                    return [], ""

                for i, row in enumerate(reader):
                    try:
                        inst = row.get("instrument", row.get("ticker", ""))
                        if not inst:
                            continue

                        if row.get("profile", "").strip() != self._profile:
                            raise ValueError(
                                "profile mismatch for "
                                f"{inst}: {row.get('profile', '')} != {self._profile}"
                            )
                        if row.get("schema_version", "").strip() != manifest_schema:
                            raise ValueError(
                                "schema mismatch for "
                                f"{inst}: {row.get('schema_version', '')} != {manifest_schema}"
                            )
                        if row.get("scanner_version", "").strip() != manifest_scanner_version:
                            raise ValueError(
                                "scanner_version mismatch for "
                                f"{inst}: {row.get('scanner_version', '')} "
                                f"!= {manifest_scanner_version}"
                            )
                        row_label_version = row.get("label_version", "").strip()
                        if manifest_label_version and row_label_version != manifest_label_version:
                            raise ValueError(
                                "label_version mismatch for "
                                f"{inst}: {row_label_version} != {manifest_label_version}"
                            )
                        row_cadence = row.get("cadence", "").strip()
                        if manifest_cadence and row_cadence and row_cadence != manifest_cadence:
                            raise ValueError(
                                f"cadence mismatch for {inst}: {row_cadence} != {manifest_cadence}"
                            )

                        row_date = row.get("market_date", "").strip() or row.get(
                            "datetime", ""
                        ).split(" ")[0]
                        if not row_date:
                            raise ValueError(f"missing market_date for {inst}")

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
                            profile=row.get("profile", self._profile),
                            scanner_version=row.get("scanner_version", manifest_scanner_version),
                            schema_version=row.get("schema_version", manifest_schema),
                            cadence=row.get("cadence", manifest_cadence),
                            label_version=row.get("label_version", manifest_label_version),
                            regime_label=row.get("regime_label", ""),
                            market_date=row_date,
                        )

                        if row_date not in all_signals:
                            all_signals[row_date] = []
                        all_signals[row_date].append(signal)

                    except (ValueError, TypeError) as e:
                        if "mismatch" in str(e) or "missing market_date" in str(e):
                            message = f"contract validation failed: {e}"
                            self._set_rejection_reason("scanner.contract.invalid", message)
                            logger.warning("ScannerBridge: {}", message)
                            return [], ""
                        logger.warning("ScannerBridge: skipping malformed row {}: {}", i, e)

        except Exception as e:
            logger.error("ScannerBridge: failed to read CSV: {}", e)
            return [], ""

        if not all_signals:
            logger.warning("ScannerBridge: no signals found in {}", path)
            return [], ""

        # Pick the target date's signals, or fail closed when the live target date is missing.
        available_dates = sorted(all_signals.keys())
        if target_date and target_date in all_signals:
            chosen_date = target_date
        else:
            if target_date and target_date not in all_signals:
                self._set_rejection_reason(
                    "scanner.bundle.target_date_missing",
                    (
                        f"target_date {target_date} not found in signals bundle "
                        f"(available: {available_dates[0]} to {available_dates[-1]})"
                    ),
                )
                logger.warning(
                    "ScannerBridge: target_date {} not found in signals (available: {} to {}), "
                    "rejecting bundle for live ingestion.",
                    target_date,
                    available_dates[0],
                    available_dates[-1],
                )
                return [], ""
            chosen_date = available_dates[-1]  # latest date

        # v1.3.0: Signal freshness guard — reject stale signals
        if max_signal_age_days is not None and target_date:
            from datetime import date as date_type

            try:
                target_dt = date_type.fromisoformat(target_date)
                chosen_dt = date_type.fromisoformat(chosen_date)
                age_days = (target_dt - chosen_dt).days
                if age_days > max_signal_age_days:
                    self._set_rejection_reason(
                        "scanner.bundle.stale",
                        (
                            f"signal date {chosen_date} is {age_days} days old for target_date "
                            f"{target_date} (max={max_signal_age_days})"
                        ),
                    )
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
        self._clear_rejection_reason()
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
