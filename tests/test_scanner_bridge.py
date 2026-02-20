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
        signals = bridge.load_signals_from_file(FIXTURES_DIR / "signals_sample.csv")
        assert len(signals) == 5

    def test_load_sample_signals_sorted_by_rank(self, bridge: ScannerBridge) -> None:
        """Signals should be sorted by rank ascending."""
        signals = bridge.load_signals_from_file(FIXTURES_DIR / "signals_sample.csv")
        ranks = [s.rank for s in signals]
        assert ranks == sorted(ranks)

    def test_load_sample_first_signal_fields(self, bridge: ScannerBridge) -> None:
        """First signal (rank=1) should be XAUUSD with correct fields."""
        signals = bridge.load_signals_from_file(FIXTURES_DIR / "signals_sample.csv")
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
        signals = bridge.load_signals_from_file(FIXTURES_DIR / "signals_sample.csv")
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
        signals = bridge.load_signals_from_file(FIXTURES_DIR / "signals_sample.csv")
        # EURUSD is rank 3, confidence=medium
        medium_signals = [s for s in signals if s.confidence == "medium"]
        assert len(medium_signals) >= 1
        qlib = medium_signals[0].to_qlib_data()
        assert qlib["signal_strength"] == "MODERATE"

    def test_load_single_signal(self, bridge: ScannerBridge) -> None:
        """signals_single.csv has 1 signal: EURUSD, score≈0.92, high confidence."""
        signals = bridge.load_signals_from_file(FIXTURES_DIR / "signals_single.csv")
        assert len(signals) == 1
        s = signals[0]
        assert s.instrument == "EURUSD"
        assert abs(s.score - 0.92) < 0.01
        assert s.confidence == "high"
        assert s.rank == 1

    def test_load_malformed_skips_bad_rows(self, bridge: ScannerBridge) -> None:
        """signals_malformed.csv: valid EURUSD + USDJPY, skips empty instrument + bad score."""
        signals = bridge.load_signals_from_file(FIXTURES_DIR / "signals_malformed.csv")
        instruments = [s.instrument for s in signals]
        # Row 1: EURUSD valid
        # Row 2: empty instrument → skipped
        # Row 3: GBPUSD with "not_a_number" score → skipped (ValueError)
        # Row 4: USDJPY valid
        assert len(signals) == 2
        assert "EURUSD" in instruments
        assert "USDJPY" in instruments

    def test_load_multiday_returns_all(self, bridge: ScannerBridge) -> None:
        """signals_multiday.csv has 6 rows across 2 dates → all 6 loaded."""
        signals = bridge.load_signals_from_file(FIXTURES_DIR / "signals_multiday.csv")
        assert len(signals) == 6

    def test_load_nonexistent_file(self, bridge: ScannerBridge) -> None:
        """Nonexistent file → empty list, no exception."""
        signals = bridge.load_signals_from_file("/does/not/exist/signals.csv")
        assert signals == []

    def test_load_empty_file(self, bridge: ScannerBridge, tmp_path: Path) -> None:
        """CSV with header only → empty list."""
        empty_csv = tmp_path / "empty_signals.csv"
        empty_csv.write_text(
            "datetime,instrument,score,rank,score_gap,drop_distance,topk_spread,confidence,weight\n"
        )
        signals = bridge.load_signals_from_file(empty_csv)
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
        signals = bridge.load_signals_from_file(FIXTURES_DIR / "signals_sample.csv")
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
        signals = bridge.load_signals_from_file(FIXTURES_DIR / "signals_sample.csv")

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
        signals = bridge.load_signals_from_file(FIXTURES_DIR / "signals_single.csv")
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
        """Multiday CSV: only create intents for target date, not all dates."""
        target_date = "2026-02-16"
        signals = bridge.load_signals_from_file(FIXTURES_DIR / "signals_multiday.csv")
        assert len(signals) == 6

        # Filter by target date — this is not done by load_signals_from_file
        # (it loads all rows). The caller (scheduler) would filter by date.
        # Here we simulate the scheduler's date filtering by reading the CSV
        # manually and matching dates. Since ScannerSignal doesn't store datetime,
        # we use a different approach: load via CSV reader to get dates.
        import csv

        target_instruments = set()
        with open(FIXTURES_DIR / "signals_multiday.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["datetime"] == target_date:
                    target_instruments.add(row["instrument"])

        assert len(target_instruments) == 3  # XAUUSD, EURUSD, AUDUSD on 2026-02-16

        # Insert only target date signals
        for signal in signals:
            if signal.instrument in target_instruments:
                # Check for duplicates (e.g., EURUSD appears on both dates)
                if not store.intent_exists(signal.instrument, target_date, "scanner"):
                    intent = _signal_to_intent(signal, target_date)
                    store.insert_intent(intent)

        intents = store.get_intents_by_date(target_date)
        assert len(intents) == 3
        intent_symbols = {i.symbol for i in intents}
        assert intent_symbols == {"XAUUSD", "EURUSD", "AUDUSD"}

    def test_scanner_topk_limits_intents(self, bridge: ScannerBridge, store: DecisionStore) -> None:
        """Scheduler respects topk=3: only top 3 signals become intents."""
        trade_date = "2026-02-16"
        signals = bridge.load_signals_from_file(FIXTURES_DIR / "signals_sample.csv")
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
        signals = bridge.load_signals_from_file(FIXTURES_DIR / "signals_single.csv")
        intent = _signal_to_intent(signals[0], trade_date)
        store.insert_intent(intent)

        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None

        # LLM decides HOLD → cancel
        store.mark_cancelled(intent.id, "LLM decided HOLD")

        final = store.get_intent(intent.id)
        assert final is not None
        assert final.status == "cancelled"
