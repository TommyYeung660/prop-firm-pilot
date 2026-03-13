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

    def test_load_multiday_missing_target_date_falls_back(self, bridge: ScannerBridge) -> None:
        """target_date not in CSV → falls back to latest date."""
        signals, _ = bridge.load_signals_from_file(
            FIXTURES_DIR / "signals_multiday.csv", target_date="2026-01-01"
        )
        assert len(signals) == 3  # Falls back to 2026-02-16 (latest)

    def test_load_empty_file(self, bridge: ScannerBridge, tmp_path: Path) -> None:
        """CSV with header only → empty list."""
        empty_csv = tmp_path / "empty_signals.csv"
        empty_csv.write_text(
            "datetime,instrument,score,rank,score_gap,drop_distance,topk_spread,confidence,weight\n"
        )
        signals, _ = bridge.load_signals_from_file(empty_csv)
        assert signals == []

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
        # Set up scanner path with output file
        signals_dir = tmp_path / "outputs" / "signals"
        signals_dir.mkdir(parents=True)
        src_csv = FIXTURES_DIR / "signals_single.csv"
        shutil.copy(src_csv, signals_dir / "signals.csv")

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

    def test_run_pipeline_recovers_benchmark_failure_without_error_log(
        self, tmp_path: Path
    ) -> None:
        """Benchmark/report failure after signal generation should be treated as recoverable."""
        signals_dir = tmp_path / "outputs" / "signals"
        signals_dir.mkdir(parents=True)
        src_csv = FIXTURES_DIR / "signals_single.csv"
        shutil.copy(src_csv, signals_dir / "signals.csv")

        bridge = ScannerBridge(scanner_path=tmp_path)

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "scanner output"
        mock_result.stderr = (
            "ValueError: The benchmark ['EURUSD'] does not exist. "
            "Please provide the right benchmark"
        )

        with (
            patch("subprocess.run", return_value=mock_result),
            patch("src.signal.scanner_bridge.logger.error") as mock_error,
        ):
            signals = bridge.run_pipeline(date="2026-02-16")

        assert len(signals) == 1
        mock_error.assert_not_called()

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
        import csv
        from datetime import date

        today = date.today().isoformat()
        fresh_csv = tmp_path / "signals_fresh.csv"
        with open(fresh_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "datetime",
                    "instrument",
                    "score",
                    "rank",
                    "score_gap",
                    "drop_distance",
                    "topk_spread",
                    "confidence",
                    "weight",
                ]
            )
            writer.writerow([today, "EURUSD", "0.61", "1", "0.05", "0.18", "0.12", "high", "0.333"])

        signals, _ = bridge.load_signals_from_file(
            fresh_csv, target_date=today, max_signal_age_days=2
        )
        assert len(signals) == 1
        assert signals[0].instrument == "EURUSD"

    def test_stale_signals_rejected(self, bridge: ScannerBridge) -> None:
        """Signals older than max_signal_age_days should return empty list."""
        signals, chosen_date = bridge.load_signals_from_file(
            FIXTURES_DIR / "signals_stale.csv",
            target_date="2026-03-03",
            max_signal_age_days=2,
        )
        # 2026-02-20 is 11 days before 2026-03-03 → stale → rejected
        assert signals == []
        assert chosen_date == ""

    def test_stale_signals_weekend_tolerance(self, bridge: ScannerBridge) -> None:
        """Friday signals should be valid on Sunday (2-day tolerance)."""
        signals, _ = bridge.load_signals_from_file(
            FIXTURES_DIR / "signals_stale.csv",
            target_date="2026-02-22",  # Sunday — 2 days after Feb 20 (Friday)
            max_signal_age_days=2,
        )
        # 2026-02-20 → 2026-02-22 = 2 days ≤ max_signal_age_days=2 → accepted
        assert len(signals) == 3

    def test_stale_signals_default_no_check(self, bridge: ScannerBridge) -> None:
        """When max_signal_age_days=None (default), no freshness check."""
        signals, _ = bridge.load_signals_from_file(
            FIXTURES_DIR / "signals_stale.csv",
            target_date="2026-12-31",  # Way in the future
        )
        # Default behavior: fallback to latest date, no staleness rejection
        assert len(signals) == 3


# ── Section 6: Pipeline Cache (v1.4.0) ─────────────────────────────────────


class TestPipelineCache:
    """Tests for _PipelineCache smart-skip when daily candle hasn't closed."""

    def test_cache_miss_runs_subprocess(self, tmp_path: Path) -> None:
        """First run_pipeline call should invoke subprocess."""
        signals_dir = tmp_path / "outputs" / "signals"
        signals_dir.mkdir(parents=True)
        shutil.copy(FIXTURES_DIR / "signals_single.csv", signals_dir / "signals.csv")

        bridge = ScannerBridge(scanner_path=tmp_path)

        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            signals = bridge.run_pipeline(date="2026-02-16", interval="1d")

        mock_run.assert_called_once()
        assert len(signals) == 1
        assert signals[0].instrument == "EURUSD"

    def test_cache_hit_skips_subprocess(self, tmp_path: Path) -> None:
        """Second run_pipeline with same (date, interval) should skip subprocess."""
        signals_dir = tmp_path / "outputs" / "signals"
        signals_dir.mkdir(parents=True)
        shutil.copy(FIXTURES_DIR / "signals_single.csv", signals_dir / "signals.csv")

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
        signals_dir = tmp_path / "outputs" / "signals"
        signals_dir.mkdir(parents=True)
        shutil.copy(FIXTURES_DIR / "signals_single.csv", signals_dir / "signals.csv")

        bridge = ScannerBridge(scanner_path=tmp_path)

        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            bridge.run_pipeline(date="2026-02-16", interval="1d")
            assert mock_run.call_count == 1

            bridge.run_pipeline(date="2026-02-17", interval="1d")
            assert mock_run.call_count == 2  # different date → cache miss

    def test_cache_miss_on_different_interval(self, tmp_path: Path) -> None:
        """Different interval should miss cache and run subprocess again."""
        signals_dir = tmp_path / "outputs" / "signals"
        signals_dir.mkdir(parents=True)
        shutil.copy(FIXTURES_DIR / "signals_single.csv", signals_dir / "signals.csv")

        bridge = ScannerBridge(scanner_path=tmp_path)

        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            bridge.run_pipeline(date="2026-02-16", interval="1d")
            assert mock_run.call_count == 1

            bridge.run_pipeline(date="2026-02-16", interval="4h")
            assert mock_run.call_count == 2  # different interval → cache miss

    def test_invalidate_cache_forces_rerun(self, tmp_path: Path) -> None:
        """invalidate_cache() should clear cache, next call runs subprocess."""
        signals_dir = tmp_path / "outputs" / "signals"
        signals_dir.mkdir(parents=True)
        shutil.copy(FIXTURES_DIR / "signals_single.csv", signals_dir / "signals.csv")

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
        signals_dir = tmp_path / "outputs" / "signals"
        signals_dir.mkdir(parents=True)
        shutil.copy(FIXTURES_DIR / "signals_single.csv", signals_dir / "signals.csv")

        bridge = ScannerBridge(scanner_path=tmp_path)

        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result):
            signals1 = bridge.run_pipeline(date="2026-02-16", interval="1d")
            signals2 = bridge.run_pipeline(date="2026-02-16", interval="1d")

        # Same content, different list objects (defensive copy)
        assert signals1[0].instrument == signals2[0].instrument
        assert signals1 is not signals2

    def test_stale_cache_still_returned(self, tmp_path: Path) -> None:
        """When signal_date < request_date (stale), cache still returns signals."""
        # signals_multiday.csv has dates 2026-02-15 and 2026-02-16
        # Requesting 2026-02-17 will fall back to latest (2026-02-16) — stale
        signals_dir = tmp_path / "outputs" / "signals"
        signals_dir.mkdir(parents=True)
        shutil.copy(FIXTURES_DIR / "signals_multiday.csv", signals_dir / "signals.csv")

        bridge = ScannerBridge(scanner_path=tmp_path)

        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            # First call: stale (request 02-17, signals are from 02-16)
            signals1 = bridge.run_pipeline(date="2026-02-17", interval="1d")
            assert mock_run.call_count == 1
            assert len(signals1) == 3  # 02-16 signals

            # Second call: cache hit (stale signals still cached)
            signals2 = bridge.run_pipeline(date="2026-02-17", interval="1d")
            assert mock_run.call_count == 1  # no re-run
            assert len(signals2) == 3

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
