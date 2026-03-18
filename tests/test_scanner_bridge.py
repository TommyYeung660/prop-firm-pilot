"""
Tests for ScannerBridge — CSV signal parsing and Scanner→DecisionStore pipeline.

Covers:
- load_signals_from_file(): real CSV fixtures → ScannerSignal objects
- ScannerBridge constructor and subprocess fallback behavior
- End-to-end: Scanner signals → TradeIntent → DecisionStore lifecycle

Uses real CSV fixtures in tests/fixtures/scanner/ and real DecisionStore
(in-memory SQLite). Only subprocess calls are mocked.
"""

import json
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.decision.agent_bridge import AgentDecision
from src.decision.decision_formatter import format_decision
from src.decision.schemas import TradeIntent
from src.decision_store.sqlite_store import DecisionStore
from src.signal.scanner_bridge import ScannerBridge, ScannerSignal

# ── Constants ───────────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "scanner"
DEFAULT_SCANNER_VERSION = "v1.5.0_beta"
DEFAULT_SIGNAL_SCHEMA_VERSION = "fx_signal_v1"
DEFAULT_SIGNAL_SCHEMA_VERSION_V2 = "fx_signal_v2"
DEFAULT_LABEL_VERSION = "cost_aware_directional_return_v1"


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def bridge(tmp_path: Path) -> ScannerBridge:
    """ScannerBridge with a temporary scanner path (exists)."""
    return ScannerBridge(scanner_path=tmp_path, topk=3, profile="fx")


@pytest.fixture
def store(tmp_path: Path) -> DecisionStore:
    """Fresh DecisionStore with a temporary database."""
    db_path = str(tmp_path / "test_scanner_bridge.db")
    s = DecisionStore(db_path=db_path)
    yield s  # type: ignore[misc]
    s.close()


def _build_contract_manifest(
    *,
    scanner_version: str = DEFAULT_SCANNER_VERSION,
    schema_version: str = DEFAULT_SIGNAL_SCHEMA_VERSION,
    label_version: str = DEFAULT_LABEL_VERSION,
    cadence: str = "1d",
    validation_status: str = "passed",
) -> dict[str, object]:
    return {
        "source": "qlib_market_scanner",
        "bundle_version": "fx_bundle_v1",
        "scanner_version": scanner_version,
        "schema_versions": {
            "signals_csv": schema_version,
            "signals_json": schema_version,
            "metrics_json": "fx_metrics_v1",
            "manifest_json": "fx_bundle_v1",
        },
        "research_run_id": "test_run_20260316",
        "config_fingerprint": "testfingerprint",
        "generated_at": "2026-03-16T00:00:00Z",
        "data_date_range": {"start": "2026-02-15", "end": "2026-02-16"},
        "universe": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD"],
        "cadence": cadence,
        "label_version": label_version,
        "validation": {"status": validation_status},
    }


def _build_metrics_payload(validation_status: str = "passed") -> dict[str, object]:
    return {
        "signal": {},
        "confidence": {},
        "backtest": {},
        "research": {},
        "regime": {},
        "validation": {"status": validation_status},
    }


def _write_contract_sidecars(
    manifest_dir: Path,
    metrics_dir: Path,
    *,
    scanner_version: str = DEFAULT_SCANNER_VERSION,
    schema_version: str = DEFAULT_SIGNAL_SCHEMA_VERSION,
    label_version: str = DEFAULT_LABEL_VERSION,
    cadence: str = "1d",
    validation_status: str = "passed",
) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps(
            _build_contract_manifest(
                scanner_version=scanner_version,
                schema_version=schema_version,
                label_version=label_version,
                cadence=cadence,
                validation_status=validation_status,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    (metrics_dir / "metrics.json").write_text(
        json.dumps(_build_metrics_payload(validation_status), indent=2),
        encoding="utf-8",
    )


def _seed_scanner_output_bundle(
    scanner_root: Path,
    fixture_name: str = "signals_single.csv",
    *,
    scanner_version: str = DEFAULT_SCANNER_VERSION,
    schema_version: str = DEFAULT_SIGNAL_SCHEMA_VERSION,
    label_version: str = DEFAULT_LABEL_VERSION,
    cadence: str = "1d",
    validation_status: str = "passed",
) -> Path:
    signals_dir = scanner_root / "outputs" / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    dest = signals_dir / "signals.csv"
    shutil.copy(FIXTURES_DIR / fixture_name, dest)
    _write_contract_sidecars(
        scanner_root / "outputs",
        scanner_root / "outputs" / "metrics",
        scanner_version=scanner_version,
        schema_version=schema_version,
        label_version=label_version,
        cadence=cadence,
        validation_status=validation_status,
    )
    return dest


def _seed_runtime_output_only(
    scanner_root: Path,
    fixture_name: str = "signals_single.csv",
    *,
    validation_status: str = "passed",
) -> Path:
    signals_dir = scanner_root / "outputs" / "signals"
    metrics_dir = scanner_root / "outputs" / "metrics"
    signals_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    dest = signals_dir / "signals.csv"
    shutil.copy(FIXTURES_DIR / fixture_name, dest)
    (metrics_dir / "metrics.json").write_text(
        json.dumps(_build_metrics_payload(validation_status), indent=2),
        encoding="utf-8",
    )
    return dest


def _write_signal_csv(
    path: Path,
    *,
    schema_version: str = DEFAULT_SIGNAL_SCHEMA_VERSION,
    side: str | None = None,
    market_date: str = "2026-02-16",
    instrument: str = "EURUSD",
    score: float = 0.61,
    rank: int = 1,
    confidence: str = "high",
) -> None:
    header = (
        "datetime,instrument,score,rank,score_gap,drop_distance,topk_spread,"
        "confidence,weight,profile,scanner_version,schema_version,cadence,"
        "label_version,regime_label,market_date"
    )
    row = (
        f"{market_date},{instrument},{score},{rank},0.05,0.18,0.12,{confidence},0.333,"
        f"fx,{DEFAULT_SCANNER_VERSION},{schema_version},1d,"
        f"{DEFAULT_LABEL_VERSION},trend,{market_date}"
    )
    if schema_version == DEFAULT_SIGNAL_SCHEMA_VERSION_V2:
        header = f"{header},side"
        row = f"{row},{'' if side is None else side}"
    path.write_text(f"{header}\n{row}\n", encoding="utf-8")


# ── Section 1: CSV Parsing Tests ────────────────────────────────────────────


class TestCSVParsing:
    """Tests for ScannerBridge.load_signals_from_file() with real CSV fixtures."""

    def test_load_sample_signals_count(self, bridge: ScannerBridge) -> None:
        """signals_sample.csv has 5 FX pairs → 5 ScannerSignals."""
        signals, _ = bridge.load_signals_from_file(FIXTURES_DIR / "signals_sample.csv")
        assert len(signals) == 5

    def test_load_sample_signals_sorted_by_rank(self, bridge: ScannerBridge) -> None:
        """Signals should be sorted by rank ascending."""
        signals, _ = bridge.load_signals_from_file(FIXTURES_DIR / "signals_sample.csv")
        ranks = [s.rank for s in signals]
        assert ranks == sorted(ranks)

    def test_load_sample_first_signal_fields(self, bridge: ScannerBridge) -> None:
        """First signal (rank=1) should be XAUUSD with correct fields."""
        signals, _ = bridge.load_signals_from_file(FIXTURES_DIR / "signals_sample.csv")
        first = signals[0]
        assert first.instrument == "XAUUSD"
        assert first.rank == 1
        assert first.confidence == "high"
        assert 0.53 < first.score < 0.54  # 0.5389...
        assert first.score_gap > 0
        assert first.drop_distance > 0
        assert first.topk_spread > 0
        assert first.weight > 0

    def test_load_sample_signal_contract_metadata(self, bridge: ScannerBridge) -> None:
        """v1.5.0 fixtures should expose scanner contract metadata on each signal."""
        signals, _ = bridge.load_signals_from_file(FIXTURES_DIR / "signals_sample.csv")
        first = signals[0]

        assert first.scanner_version == DEFAULT_SCANNER_VERSION
        assert first.schema_version == DEFAULT_SIGNAL_SCHEMA_VERSION
        assert first.market_date == "2026-02-16"
        assert first.label_version == DEFAULT_LABEL_VERSION

    def test_load_sample_signal_qlib_data(self, bridge: ScannerBridge) -> None:
        """.to_qlib_data() should return correct dict structure."""
        signals, _ = bridge.load_signals_from_file(FIXTURES_DIR / "signals_sample.csv")
        first = signals[0]  # XAUUSD, confidence=high
        qlib = first.to_qlib_data()

        assert "score" in qlib
        assert qlib["score"] == first.score
        assert qlib["signal_strength"] == "STRONG"  # high → STRONG
        assert qlib["confidence"] == "high"
        assert qlib["score_gap"] == first.score_gap
        assert qlib["drop_distance"] == first.drop_distance
        assert qlib["topk_spread"] == first.topk_spread

    def test_load_sample_medium_confidence_qlib(self, bridge: ScannerBridge) -> None:
        """Medium confidence → MODERATE signal_strength."""
        signals, _ = bridge.load_signals_from_file(FIXTURES_DIR / "signals_sample.csv")
        # EURUSD is rank 3, confidence=medium
        medium_signals = [s for s in signals if s.confidence == "medium"]
        assert len(medium_signals) >= 1
        qlib = medium_signals[0].to_qlib_data()
        assert qlib["signal_strength"] == "MODERATE"

    def test_load_signals_accepts_fx_signal_v2_with_side(
        self, bridge: ScannerBridge, tmp_path: Path
    ) -> None:
        csv_path = tmp_path / "signals_v2.csv"
        _write_signal_csv(
            csv_path,
            schema_version=DEFAULT_SIGNAL_SCHEMA_VERSION_V2,
            side="long",
        )
        _write_contract_sidecars(
            tmp_path,
            tmp_path,
            schema_version=DEFAULT_SIGNAL_SCHEMA_VERSION_V2,
        )

        signals, chosen_date = bridge.load_signals_from_file(csv_path, target_date="2026-02-16")

        assert chosen_date == "2026-02-16"
        assert len(signals) == 1
        assert signals[0].side == "long"

    def test_load_v2_signals_rejects_missing_side(
        self, bridge: ScannerBridge, tmp_path: Path
    ) -> None:
        csv_path = tmp_path / "signals_v2_missing_side.csv"
        _write_signal_csv(
            csv_path,
            schema_version=DEFAULT_SIGNAL_SCHEMA_VERSION_V2,
            side=None,
        )
        _write_contract_sidecars(
            tmp_path,
            tmp_path,
            schema_version=DEFAULT_SIGNAL_SCHEMA_VERSION_V2,
        )

        signals, chosen_date = bridge.load_signals_from_file(csv_path, target_date="2026-02-16")

        assert signals == []
        assert chosen_date == ""
        assert bridge.get_last_rejection_reason_code() == "scanner.contract.invalid"

    def test_load_single_signal(self, bridge: ScannerBridge) -> None:
        """signals_single.csv has 1 signal: EURUSD, score≈0.92, high confidence."""
        signals, _ = bridge.load_signals_from_file(FIXTURES_DIR / "signals_single.csv")
        assert len(signals) == 1
        s = signals[0]
        assert s.instrument == "EURUSD"
        assert abs(s.score - 0.92) < 0.01
        assert s.confidence == "high"
        assert s.rank == 1

    def test_load_malformed_skips_bad_rows(self, bridge: ScannerBridge) -> None:
        """signals_malformed.csv: valid EURUSD + USDJPY, skips empty instrument + bad score."""
        signals, _ = bridge.load_signals_from_file(FIXTURES_DIR / "signals_malformed.csv")
        instruments = [s.instrument for s in signals]
        # Row 1: EURUSD valid
        # Row 2: empty instrument → skipped
        # Row 3: GBPUSD with "not_a_number" score → skipped (ValueError)
        # Row 4: USDJPY valid
        assert len(signals) == 2
        assert "EURUSD" in instruments
        assert "USDJPY" in instruments

    def test_load_multiday_returns_latest_date_only(self, bridge: ScannerBridge) -> None:
        """signals_multiday.csv has 2 dates → returns only latest date (3 signals)."""
        signals, _ = bridge.load_signals_from_file(FIXTURES_DIR / "signals_multiday.csv")
        assert len(signals) == 3  # Only 2026-02-16 (latest date)
        instruments = {s.instrument for s in signals}
        assert instruments == {"XAUUSD", "EURUSD", "AUDUSD"}

    def test_load_multiday_with_target_date(self, bridge: ScannerBridge) -> None:
        """target_date selects signals for that specific date."""
        signals, _ = bridge.load_signals_from_file(
            FIXTURES_DIR / "signals_multiday.csv", target_date="2026-02-15"
        )
        assert len(signals) == 3  # EURUSD, GBPUSD, USDJPY on 02-15
        instruments = {s.instrument for s in signals}
        assert instruments == {"EURUSD", "GBPUSD", "USDJPY"}

    def test_load_multiday_missing_target_date_rejects(self, bridge: ScannerBridge) -> None:
        """target_date not in CSV → reject instead of falling back."""
        signals, chosen_date = bridge.load_signals_from_file(
            FIXTURES_DIR / "signals_multiday.csv", target_date="2026-01-01"
        )
        assert signals == []
        assert chosen_date == ""
        assert bridge.get_last_rejection_reason_code() == "scanner.bundle.target_date_missing"

    def test_load_empty_file(self, bridge: ScannerBridge, tmp_path: Path) -> None:
        """CSV with header only → empty list."""
        empty_csv = tmp_path / "empty_signals.csv"
        empty_csv.write_text(
            "datetime,instrument,score,rank,score_gap,drop_distance,topk_spread,confidence,weight\n"
        )
        signals, _ = bridge.load_signals_from_file(empty_csv)
        assert signals == []

    def test_load_signals_rejects_unknown_schema_version(
        self, bridge: ScannerBridge, tmp_path: Path
    ) -> None:
        """Unknown signal schema should trip the pilot ingestion gate."""
        csv_path = tmp_path / "signals.csv"
        csv_path.write_text(
            (FIXTURES_DIR / "signals_single.csv")
            .read_text(encoding="utf-8")
            .replace(DEFAULT_SIGNAL_SCHEMA_VERSION, "fx_signal_v999"),
            encoding="utf-8",
        )
        _write_contract_sidecars(
            tmp_path,
            tmp_path,
            schema_version="fx_signal_v999",
        )

        signals, chosen_date = bridge.load_signals_from_file(csv_path, target_date="2026-02-16")

        assert signals == []
        assert chosen_date == ""
        assert bridge.get_last_rejection_reason_code() == "scanner.contract.invalid"

    def test_load_signals_accepts_beta_2_scanner_bundle(
        self, bridge: ScannerBridge, tmp_path: Path
    ) -> None:
        """beta_2 scanner bundles should pass the ingestion version gate."""
        csv_path = tmp_path / "signals.csv"
        csv_path.write_text(
            (FIXTURES_DIR / "signals_single.csv")
            .read_text(encoding="utf-8")
            .replace(DEFAULT_SCANNER_VERSION, "v1.5.0_beta_2"),
            encoding="utf-8",
        )
        _write_contract_sidecars(
            tmp_path,
            tmp_path,
            scanner_version="v1.5.0_beta_2",
        )

        signals, chosen_date = bridge.load_signals_from_file(csv_path, target_date="2026-02-16")

        assert len(signals) == 1
        assert chosen_date == "2026-02-16"
        assert signals[0].scanner_version == "v1.5.0_beta_2"

    def test_load_signals_rejects_missing_manifest(
        self, bridge: ScannerBridge, tmp_path: Path
    ) -> None:
        """Missing manifest must reject the bundle before any parsing proceeds."""
        csv_path = tmp_path / "signals.csv"
        shutil.copy(FIXTURES_DIR / "signals_single.csv", csv_path)

        signals, chosen_date = bridge.load_signals_from_file(csv_path, target_date="2026-02-16")

        assert signals == []
        assert chosen_date == ""
        assert bridge.get_last_rejection_reason_code() == "scanner.contract.invalid"

    def test_load_signals_rejects_degraded_bundle(
        self, bridge: ScannerBridge, tmp_path: Path
    ) -> None:
        """Degraded validation status should be rejected for live ingestion."""
        csv_path = tmp_path / "signals.csv"
        shutil.copy(FIXTURES_DIR / "signals_single.csv", csv_path)
        _write_contract_sidecars(tmp_path, tmp_path, validation_status="degraded")

        signals, chosen_date = bridge.load_signals_from_file(csv_path, target_date="2026-02-16")

        assert signals == []
        assert chosen_date == ""
        assert bridge.get_last_rejection_reason_code() == "scanner.bundle.degraded"

    def test_runtime_outputs_do_not_fallback_to_shared_export_manifest(
        self, bridge: ScannerBridge, tmp_path: Path
    ) -> None:
        csv_path = _seed_runtime_output_only(tmp_path)
        _write_contract_sidecars(
            tmp_path / "data" / "shared_export",
            tmp_path / "data" / "shared_export" / "scanner_outputs" / "metrics",
        )

        signals, chosen_date = bridge.load_signals_from_file(csv_path, target_date="2026-02-16")

        assert signals == []
        assert chosen_date == ""
        assert bridge.get_last_rejection_reason_code() == "scanner.contract.invalid"
        assert "runtime manifest.json not found" in bridge.get_last_rejection_message()

    def test_runtime_outputs_report_missing_runtime_manifest_even_if_shared_export_is_invalid(
        self, bridge: ScannerBridge, tmp_path: Path
    ) -> None:
        csv_path = _seed_runtime_output_only(tmp_path)
        shared_manifest_dir = tmp_path / "data" / "shared_export"
        shared_manifest_dir.mkdir(parents=True, exist_ok=True)
        (shared_manifest_dir / "manifest.json").write_text(
            json.dumps({"source": "legacy_bundle"}, indent=2),
            encoding="utf-8",
        )

        signals, chosen_date = bridge.load_signals_from_file(csv_path, target_date="2026-02-16")

        assert signals == []
        assert chosen_date == ""
        assert bridge.get_last_rejection_reason_code() == "scanner.contract.invalid"
        assert "runtime manifest.json not found" in bridge.get_last_rejection_message()

    def test_scanner_signal_repr(self) -> None:
        """ScannerSignal repr should include key fields."""
        signal = ScannerSignal(
            instrument="EURUSD",
            score=0.85,
            rank=1,
            confidence="high",
        )
        r = repr(signal)
        assert "EURUSD" in r
        assert "0.85" in r
        assert "rank=1" in r
        assert "high" in r

    def test_directional_quality_uses_one_minus_score_for_short(self) -> None:
        signal = ScannerSignal(
            instrument="USDCHF",
            score=0.12,
            rank=1,
            confidence="high",
            side="short",
        )

        qlib = signal.to_qlib_data()

        assert qlib["scanner_direction_quality"] == 0.88


# ── Section 2: Constructor and Path Tests ───────────────────────────────────


class TestScannerBridgeInit:
    """Tests for ScannerBridge constructor and subprocess fallback behavior."""

    def test_init_resolves_path(self, tmp_path: Path) -> None:
        """Scanner path should be resolved to absolute."""
        bridge = ScannerBridge(scanner_path=tmp_path / "relative" / ".." / "actual")
        assert bridge._scanner_path.is_absolute()

    def test_warns_missing_path(self, tmp_path: Path) -> None:
        """Should log warning when scanner path doesn't exist."""
        nonexistent = tmp_path / "nonexistent_scanner"
        # No exception, just warning logged
        bridge = ScannerBridge(scanner_path=nonexistent)
        assert bridge._scanner_path == nonexistent.resolve()

    def test_run_pipeline_fallback_to_file(self, tmp_path: Path) -> None:
        """When subprocess fails but signals.csv exists, should fall back to file."""
        _seed_scanner_output_bundle(tmp_path)

        bridge = ScannerBridge(scanner_path=tmp_path)

        # Mock subprocess.run to fail
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "error output"
        mock_result.stderr = "some error"

        with patch("subprocess.run", return_value=mock_result):
            signals = bridge.run_pipeline(date="2026-02-16")

        # Should have fallen back to the signals.csv file
        assert len(signals) == 1
        assert signals[0].instrument == "EURUSD"

    def test_run_pipeline_includes_configured_benchmark(self, tmp_path: Path) -> None:
        """Configured benchmark should be passed to qlib scanner CLI."""
        _seed_scanner_output_bundle(tmp_path)

        bridge = ScannerBridge(scanner_path=tmp_path, benchmark="FX")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ok"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            signals = bridge.run_pipeline(date="2026-02-16")

        assert len(signals) == 1
        cmd = mock_run.call_args.args[0]
        assert "--benchmark" in cmd
        assert cmd[cmd.index("--benchmark") + 1] == "FX"

    def test_run_pipeline_includes_configured_topk(self, tmp_path: Path) -> None:
        """Configured topk should be passed to qlib scanner CLI."""
        _seed_scanner_output_bundle(tmp_path)

        bridge = ScannerBridge(scanner_path=tmp_path, topk=5)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ok"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            signals = bridge.run_pipeline(date="2026-02-16")

        assert len(signals) == 1
        cmd = mock_run.call_args.args[0]
        assert "--topk" in cmd
        assert cmd[cmd.index("--topk") + 1] == "5"

    def test_run_pipeline_includes_configured_topk_short(self, tmp_path: Path) -> None:
        """Configured topk_short should be passed to qlib scanner CLI."""
        _seed_scanner_output_bundle(tmp_path)

        bridge = ScannerBridge(scanner_path=tmp_path, topk_short=1)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ok"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            signals = bridge.run_pipeline(date="2026-02-16")

        assert len(signals) == 1
        cmd = mock_run.call_args.args[0]
        assert "--topk-short" in cmd
        assert cmd[cmd.index("--topk-short") + 1] == "1"

    def test_run_pipeline_retries_without_benchmark_when_cli_rejects_argument(
        self, tmp_path: Path
    ) -> None:
        """Older qlib scanner CLIs should be retried without --benchmark."""
        _seed_scanner_output_bundle(tmp_path)

        bridge = ScannerBridge(scanner_path=tmp_path, benchmark="FX")

        first_result = MagicMock()
        first_result.returncode = 2
        first_result.stdout = ""
        first_result.stderr = "main.py: error: unrecognized arguments: --benchmark FX"

        second_result = MagicMock()
        second_result.returncode = 0
        second_result.stdout = "ok"
        second_result.stderr = ""

        with patch("subprocess.run", side_effect=[first_result, second_result]) as mock_run:
            signals = bridge.run_pipeline(date="2026-02-16")

        assert len(signals) == 1
        assert signals[0].instrument == "EURUSD"
        assert mock_run.call_count == 2

        first_cmd = mock_run.call_args_list[0].args[0]
        second_cmd = mock_run.call_args_list[1].args[0]
        assert "--benchmark" in first_cmd
        assert "--benchmark" not in second_cmd

    def test_run_pipeline_timeout(self, tmp_path: Path) -> None:
        """subprocess.TimeoutExpired → empty list."""
        bridge = ScannerBridge(scanner_path=tmp_path)

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="uv", timeout=600),
        ):
            signals = bridge.run_pipeline()

        assert signals == []

    def test_run_pipeline_file_not_found(self, tmp_path: Path) -> None:
        """FileNotFoundError (uv not found) → empty list."""
        bridge = ScannerBridge(scanner_path=tmp_path)

        with patch(
            "subprocess.run",
            side_effect=FileNotFoundError("uv not found"),
        ):
            signals = bridge.run_pipeline()

        assert signals == []

    def test_run_pipeline_logs_incomplete_when_target_date_did_not_land(
        self, tmp_path: Path
    ) -> None:
        """Process success alone must not count as pipeline success."""
        _seed_scanner_output_bundle(tmp_path, fixture_name="signals_multiday.csv")

        bridge = ScannerBridge(scanner_path=tmp_path)
        mock_result = MagicMock(returncode=0, stdout="ok", stderr="")

        with (
            patch("subprocess.run", return_value=mock_result),
            patch("src.signal.scanner_bridge.logger") as mock_logger,
        ):
            signals = bridge.run_pipeline(date="2026-02-17")

        assert signals == []
        assert bridge.get_last_rejection_reason_code() == "scanner.bundle.target_date_missing"
        success_calls = [
            call
            for call in mock_logger.info.call_args_list
            if call.args and call.args[0].startswith("ScannerBridge: pipeline succeeded")
        ]
        assert success_calls == []
        mock_logger.warning.assert_any_call(
            "ScannerBridge: pipeline incomplete "
            "(process_success={}, artifact_available={}, ingestion_success={}, "
            "target_date_matched={}, signal_date={}, signal_count={}, rejection_reason={})",
            True,
            True,
            False,
            False,
            "",
            0,
            "scanner.bundle.target_date_missing",
        )

    def test_run_pipeline_logs_success_only_after_target_date_ingestion(
        self, tmp_path: Path
    ) -> None:
        """Success requires process, artifact, ingestion, and target-date match."""
        _seed_scanner_output_bundle(tmp_path)

        bridge = ScannerBridge(scanner_path=tmp_path)
        mock_result = MagicMock(returncode=0, stdout="ok", stderr="")

        with (
            patch("subprocess.run", return_value=mock_result),
            patch("src.signal.scanner_bridge.logger") as mock_logger,
        ):
            signals = bridge.run_pipeline(date="2026-02-16")

        assert len(signals) == 1
        assert signals[0].instrument == "EURUSD"
        mock_logger.info.assert_any_call(
            "ScannerBridge: pipeline succeeded "
            "(process_success={}, artifact_available={}, ingestion_success={}, "
            "target_date_matched={}, signal_date={}, signal_count={})",
            True,
            True,
            True,
            True,
            "2026-02-16",
            1,
        )


# ── Section 3: Scanner → DecisionStore E2E Pipeline ─────────────────────────


def _signal_to_intent(signal: ScannerSignal, trade_date: str) -> TradeIntent:
    """Convert a ScannerSignal into a TradeIntent (mirrors Scheduler._scanner_loop logic)."""
    return TradeIntent(
        trade_date=trade_date,
        symbol=signal.instrument,
        scanner_score=signal.score,
        scanner_confidence=signal.confidence,
        scanner_score_gap=signal.score_gap,
        scanner_drop_distance=signal.drop_distance,
        scanner_topk_spread=signal.topk_spread,
        scanner_version=signal.scanner_version,
        scanner_schema_version=signal.schema_version,
        scanner_market_date=signal.market_date,
        scanner_label_version=signal.label_version,
        scanner_side=signal.side,
        source="scanner",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=4),
    )


class TestE2EPipeline:
    """End-to-end tests: Scanner CSV → ScannerSignal → TradeIntent → DecisionStore."""

    def test_scanner_signals_to_intents(self, bridge: ScannerBridge, store: DecisionStore) -> None:
        """Load signals_sample.csv → create intents → verify in store."""
        trade_date = "2026-02-16"
        signals, _ = bridge.load_signals_from_file(FIXTURES_DIR / "signals_sample.csv")
        assert len(signals) == 5

        for signal in signals:
            intent = _signal_to_intent(signal, trade_date)
            store.insert_intent(intent)

        intents = store.get_intents_by_date(trade_date)
        assert len(intents) == 5

        # Verify fields transferred correctly
        symbols = {i.symbol for i in intents}
        assert symbols == {"XAUUSD", "USDJPY", "EURUSD", "GBPUSD", "AUDUSD"}

        for intent in intents:
            assert intent.status == "pending"
            assert intent.source == "scanner"
            assert intent.scanner_score > 0
            assert intent.scanner_confidence in ("high", "medium", "low")
            assert intent.scanner_version == DEFAULT_SCANNER_VERSION
            assert intent.scanner_schema_version == DEFAULT_SIGNAL_SCHEMA_VERSION
            assert intent.scanner_market_date == trade_date
            assert intent.scanner_label_version == DEFAULT_LABEL_VERSION

    def test_scanner_dedup_prevents_duplicate_intents(
        self, bridge: ScannerBridge, store: DecisionStore
    ) -> None:
        """Loading same CSV twice → intent_exists() prevents second insert."""
        trade_date = "2026-02-16"
        signals, _ = bridge.load_signals_from_file(FIXTURES_DIR / "signals_sample.csv")

        # First pass: insert all
        for signal in signals:
            intent = _signal_to_intent(signal, trade_date)
            store.insert_intent(intent)

        # Second pass: check intent_exists before inserting (mirrors scheduler logic)
        duplicates_skipped = 0
        for signal in signals:
            if store.intent_exists(signal.instrument, trade_date, "scanner"):
                duplicates_skipped += 1
                continue
            # Would insert here, but should never reach this
            store.insert_intent(_signal_to_intent(signal, trade_date))

        assert duplicates_skipped == 5  # All 5 should be skipped

        intents = store.get_intents_by_date(trade_date)
        assert len(intents) == 5  # Still only 5, not 10

    def test_v2_signal_to_intent_copies_scanner_side(
        self, bridge: ScannerBridge, store: DecisionStore, tmp_path: Path
    ) -> None:
        trade_date = "2026-02-16"
        csv_path = tmp_path / "signals_v2_short.csv"
        _write_signal_csv(
            csv_path,
            schema_version=DEFAULT_SIGNAL_SCHEMA_VERSION_V2,
            side="short",
            instrument="USDCHF",
            score=0.12,
        )
        _write_contract_sidecars(
            tmp_path,
            tmp_path,
            schema_version=DEFAULT_SIGNAL_SCHEMA_VERSION_V2,
        )

        signals, _ = bridge.load_signals_from_file(csv_path, target_date=trade_date)

        assert len(signals) == 1
        intent = _signal_to_intent(signals[0], trade_date)
        store.insert_intent(intent)
        persisted = store.get_intent(intent.id)

        assert persisted is not None
        assert persisted.scanner_side == "short"

    def test_scanner_to_llm_to_execution(self, bridge: ScannerBridge, store: DecisionStore) -> None:
        """Full pipeline: signal → intent → claim → LLM BUY → ready_for_exec."""
        trade_date = "2026-02-16"
        signals, _ = bridge.load_signals_from_file(FIXTURES_DIR / "signals_single.csv")
        assert len(signals) == 1

        signal = signals[0]
        intent = _signal_to_intent(signal, trade_date)
        store.insert_intent(intent)

        # Claim
        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None
        assert claimed.symbol == "EURUSD"

        # Simulate LLM decision (BUY)
        agent_decision = AgentDecision(
            symbol="EURUSD",
            decision="BUY",
            final_state={"summary": "Scanner test BUY"},
            risk_report="Test risk report",
        )
        formatted = format_decision(
            symbol="EURUSD",
            decision="BUY",
            scanner_score=signal.score,
            scanner_confidence=signal.confidence,
            agent_state=agent_decision.final_state,
        )
        store.update_intent_decision(
            intent.id,
            side=agent_decision.decision,
            sl_pips=formatted.suggested_sl_pips,
            tp_pips=formatted.suggested_tp_pips,
            risk_report=agent_decision.risk_report,
            state_json=json.dumps(agent_decision.final_state, default=str),
        )
        store.mark_ready_for_exec(intent.id)

        # Verify final state
        final = store.get_intent(intent.id)
        assert final is not None
        assert final.status == "ready_for_exec"
        assert final.suggested_side == "BUY"
        assert final.suggested_sl_pips is not None
        assert final.suggested_tp_pips is not None

    def test_multiday_scanner_independent(
        self, bridge: ScannerBridge, store: DecisionStore
    ) -> None:
        """Multiday CSV with target_date: only create intents for target date."""
        target_date = "2026-02-16"
        signals, _ = bridge.load_signals_from_file(
            FIXTURES_DIR / "signals_multiday.csv", target_date=target_date
        )
        # load_signals_from_file now filters by date automatically
        assert len(signals) == 3  # XAUUSD, EURUSD, AUDUSD on 2026-02-16

        for signal in signals:
            intent = _signal_to_intent(signal, target_date)
            store.insert_intent(intent)

        intents = store.get_intents_by_date(target_date)
        assert len(intents) == 3
        intent_symbols = {i.symbol for i in intents}
        assert intent_symbols == {"XAUUSD", "EURUSD", "AUDUSD"}

    def test_scanner_topk_limits_intents(self, bridge: ScannerBridge, store: DecisionStore) -> None:
        """Scheduler respects topk=3: only top 3 signals become intents."""
        trade_date = "2026-02-16"
        signals, _ = bridge.load_signals_from_file(FIXTURES_DIR / "signals_sample.csv")
        assert len(signals) == 5

        # Only take top-K (bridge._topk = 3)
        for signal in signals[: bridge._topk]:
            intent = _signal_to_intent(signal, trade_date)
            store.insert_intent(intent)

        intents = store.get_intents_by_date(trade_date)
        assert len(intents) == 3  # topk=3

    def test_hold_decision_cancels_intent(
        self, bridge: ScannerBridge, store: DecisionStore
    ) -> None:
        """HOLD decision from LLM → intent cancelled, never executed."""
        trade_date = "2026-02-16"
        signals, _ = bridge.load_signals_from_file(FIXTURES_DIR / "signals_single.csv")
        intent = _signal_to_intent(signals[0], trade_date)
        store.insert_intent(intent)

        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None

        # LLM decides HOLD → cancel
        store.mark_cancelled(intent.id, "LLM decided HOLD")

        final = store.get_intent(intent.id)
        assert final is not None
        assert final.status == "cancelled"


# ── Section 4: Interval Parameter (v1.2.0) ────────────────────────────────


class TestIntervalParameter:
    """Tests for ScannerBridge.run_pipeline() interval parameter."""

    def test_run_pipeline_passes_interval(self, tmp_path: Path) -> None:
        """run_pipeline should pass --interval to the scanner subprocess."""
        bridge = ScannerBridge(scanner_path=tmp_path)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
            bridge.run_pipeline(date="2026-03-01", interval="4h")

            cmd = mock_run.call_args[0][0]
            assert "--interval" in cmd
            assert "4h" in cmd

    def test_run_pipeline_default_interval_1d(self, tmp_path: Path) -> None:
        """Default interval should be '1d' (not passed to cmd if default)."""
        bridge = ScannerBridge(scanner_path=tmp_path)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
            bridge.run_pipeline(date="2026-03-01")

            cmd = mock_run.call_args[0][0]
            assert "--interval" in cmd
            assert "1d" in cmd


# ── Section 5: Signal Freshness Guard (v1.3.0) ────────────────────────────────────


class TestSignalFreshness:
    """Tests for signal freshness guard in load_signals_from_file (v1.3.0)."""

    def test_fresh_signals_returned_normally(self, bridge: ScannerBridge, tmp_path: Path) -> None:
        """Signals within max_signal_age_days should be returned as-is."""
        from datetime import date

        today = date.today().isoformat()
        fresh_csv = tmp_path / "signals_fresh.csv"
        fresh_csv.write_text(
            (
                "datetime,instrument,score,rank,score_gap,drop_distance,topk_spread,"
                "confidence,weight,profile,scanner_version,schema_version,cadence,"
                "label_version,regime_label,market_date\n"
                f"{today},EURUSD,0.61,1,0.05,0.18,0.12,high,0.333,fx,"
                f"{DEFAULT_SCANNER_VERSION},{DEFAULT_SIGNAL_SCHEMA_VERSION},1d,"
                f"{DEFAULT_LABEL_VERSION},trend,{today}\n"
            ),
            encoding="utf-8",
        )
        _write_contract_sidecars(tmp_path, tmp_path)

        signals, _ = bridge.load_signals_from_file(
            fresh_csv, target_date=today, max_signal_age_days=2
        )
        assert len(signals) == 1
        assert signals[0].instrument == "EURUSD"

    def test_missing_target_date_signals_rejected(self, bridge: ScannerBridge) -> None:
        """Missing target_date should hard block live ingestion."""
        signals, chosen_date = bridge.load_signals_from_file(
            FIXTURES_DIR / "signals_stale.csv",
            target_date="2026-03-03",
            max_signal_age_days=2,
        )
        assert signals == []
        assert chosen_date == ""
        assert bridge.get_last_rejection_reason_code() == "scanner.bundle.target_date_missing"

    def test_missing_target_date_rejects_even_with_weekend_tolerance(
        self, bridge: ScannerBridge
    ) -> None:
        """Weekend tolerance should not revive missing target-date bundles."""
        signals, chosen_date = bridge.load_signals_from_file(
            FIXTURES_DIR / "signals_stale.csv",
            target_date="2026-02-22",  # Sunday — 2 days after Feb 20 (Friday)
            max_signal_age_days=2,
        )
        assert signals == []
        assert chosen_date == ""
        assert bridge.get_last_rejection_reason_code() == "scanner.bundle.target_date_missing"

    def test_missing_target_date_rejects_when_no_freshness_check(
        self, bridge: ScannerBridge
    ) -> None:
        """Strict target-date matching should apply even when freshness check is disabled."""
        signals, chosen_date = bridge.load_signals_from_file(
            FIXTURES_DIR / "signals_stale.csv",
            target_date="2026-12-31",  # Way in the future
        )
        assert signals == []
        assert chosen_date == ""
        assert bridge.get_last_rejection_reason_code() == "scanner.bundle.target_date_missing"


# ── Section 6: Pipeline Cache (v1.4.0) ─────────────────────────────────────


class TestPipelineCache:
    """Tests for _PipelineCache smart-skip when daily candle hasn't closed."""

    def test_cache_miss_runs_subprocess(self, tmp_path: Path) -> None:
        """First run_pipeline call should invoke subprocess."""
        _seed_scanner_output_bundle(tmp_path)

        bridge = ScannerBridge(scanner_path=tmp_path)

        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            signals = bridge.run_pipeline(date="2026-02-16", interval="1d")

        mock_run.assert_called_once()
        assert len(signals) == 1
        assert signals[0].instrument == "EURUSD"

    def test_cache_hit_skips_subprocess(self, tmp_path: Path) -> None:
        """Second run_pipeline with same (date, interval) should skip subprocess."""
        _seed_scanner_output_bundle(tmp_path)

        bridge = ScannerBridge(scanner_path=tmp_path)

        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            # First call — subprocess runs
            signals1 = bridge.run_pipeline(date="2026-02-16", interval="1d")
            assert mock_run.call_count == 1

            # Second call — cache hit, subprocess NOT called again
            signals2 = bridge.run_pipeline(date="2026-02-16", interval="1d")
            assert mock_run.call_count == 1  # still 1, not 2

        assert len(signals1) == 1
        assert len(signals2) == 1
        assert signals1[0].instrument == signals2[0].instrument

    def test_cache_miss_on_different_date(self, tmp_path: Path) -> None:
        """Different date should miss cache and run subprocess again."""
        _seed_scanner_output_bundle(tmp_path)

        bridge = ScannerBridge(scanner_path=tmp_path)

        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            bridge.run_pipeline(date="2026-02-16", interval="1d")
            assert mock_run.call_count == 1

            bridge.run_pipeline(date="2026-02-17", interval="1d")
            assert mock_run.call_count == 2  # different date → cache miss

    def test_cache_miss_on_different_interval(self, tmp_path: Path) -> None:
        """Different interval should miss cache and run subprocess again."""
        _seed_scanner_output_bundle(tmp_path)

        bridge = ScannerBridge(scanner_path=tmp_path)

        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            bridge.run_pipeline(date="2026-02-16", interval="1d")
            assert mock_run.call_count == 1

            bridge.run_pipeline(date="2026-02-16", interval="4h")
            assert mock_run.call_count == 2  # different interval → cache miss

    def test_invalidate_cache_forces_rerun(self, tmp_path: Path) -> None:
        """invalidate_cache() should clear cache, next call runs subprocess."""
        _seed_scanner_output_bundle(tmp_path)

        bridge = ScannerBridge(scanner_path=tmp_path)

        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            bridge.run_pipeline(date="2026-02-16", interval="1d")
            assert mock_run.call_count == 1

            bridge.invalidate_cache()

            bridge.run_pipeline(date="2026-02-16", interval="1d")
            assert mock_run.call_count == 2  # cache busted → re-run

    def test_cache_returns_defensive_copy(self, tmp_path: Path) -> None:
        """Cached signals should be a copy, not the same list object."""
        _seed_scanner_output_bundle(tmp_path)

        bridge = ScannerBridge(scanner_path=tmp_path)

        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result):
            signals1 = bridge.run_pipeline(date="2026-02-16", interval="1d")
            signals2 = bridge.run_pipeline(date="2026-02-16", interval="1d")

        # Same content, different list objects (defensive copy)
        assert signals1[0].instrument == signals2[0].instrument
        assert signals1 is not signals2

    def test_missing_target_date_pipeline_result_not_cached(self, tmp_path: Path) -> None:
        """Rejected missing-target-date results should not be cached."""
        _seed_scanner_output_bundle(tmp_path, fixture_name="signals_multiday.csv")

        bridge = ScannerBridge(scanner_path=tmp_path)

        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            signals1 = bridge.run_pipeline(date="2026-02-17", interval="1d")
            assert mock_run.call_count == 1
            assert signals1 == []

            signals2 = bridge.run_pipeline(date="2026-02-17", interval="1d")
            assert mock_run.call_count == 2
            assert signals2 == []

    def test_empty_pipeline_result_not_cached(self, tmp_path: Path) -> None:
        """Empty pipeline result should NOT be cached (no signals to cache)."""
        bridge = ScannerBridge(scanner_path=tmp_path)

        mock_result = MagicMock(returncode=1, stdout="", stderr="error")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            # No signals file fallback → empty result
            signals1 = bridge.run_pipeline(date="2026-02-16", interval="1d")
            assert signals1 == []
            assert mock_run.call_count == 1

            # Second call should still run subprocess (nothing was cached)
            signals2 = bridge.run_pipeline(date="2026-02-16", interval="1d")
            assert signals2 == []
            assert mock_run.call_count == 2  # ran again because nothing cached
