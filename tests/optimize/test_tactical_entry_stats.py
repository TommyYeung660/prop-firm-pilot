"""Tests for tactical entry calibration aggregation."""

from pathlib import Path

from src.monitor.trade_journal import TradeJournal
from src.optimize.tactical_entry_stats import build_daily_entry_calibration_snapshot


def test_build_daily_entry_calibration_snapshot_groups_by_symbol_session_regime(
    tmp_path: Path,
) -> None:
    journal = TradeJournal(tmp_path / "trade_journal.jsonl")
    journal.log_event(
        "TACTICAL_RESULT",
        {
            "timestamp": "2026-03-14T08:00:00+00:00",
            "symbol": "EURUSD",
            "resolution": "RETRY_PENDING",
            "summary_reason_code": "spread.fail.ratio_too_wide",
            "context": {"session_label": "london", "regime_label": "normal"},
            "provenance": {"data_source": "rest_fallback"},
        },
    )
    journal.log_event(
        "TACTICAL_RESULT",
        {
            "timestamp": "2026-03-14T08:05:00+00:00",
            "symbol": "EURUSD",
            "resolution": "EXECUTE_NOW",
            "summary_reason_code": "tactical.pass.all_gates_aligned",
            "context": {"session_label": "london", "regime_label": "normal"},
            "provenance": {"data_source": "websocket_cache"},
        },
    )

    snapshot = build_daily_entry_calibration_snapshot(journal, "2026-03-14")

    assert snapshot["date"] == "2026-03-14"
    assert len(snapshot["groups"]) == 1
    group = snapshot["groups"][0]
    assert group["symbol"] == "EURUSD"
    assert group["session_label"] == "london"
    assert group["regime_label"] == "normal"
    assert group["wait_rate"] == 0.5
    assert group["pass_rate"] == 0.5
    assert group["rest_fallback_ratio"] == 0.5
    assert group["top_reason_codes"][0]["reason_code"] == "spread.fail.ratio_too_wide"


def test_build_daily_entry_calibration_snapshot_aggregates_failed_hard_gate_codes(
    tmp_path: Path,
) -> None:
    journal = TradeJournal(tmp_path / "trade_journal.jsonl")
    journal.log_event(
        "TACTICAL_RESULT",
        {
            "timestamp": "2026-03-14T08:00:00+00:00",
            "symbol": "USDCAD",
            "resolution": "RETRY_PENDING",
            "summary_reason_code": "spread.fail.ratio_too_wide",
            "failed_hard_gate_reason_codes": [
                "spread.fail.ratio_too_wide",
                "atr.fail.insufficient_1h_data",
            ],
            "context": {"session_label": "new_york", "regime_label": "normal"},
            "provenance": {"data_source": "rest_fallback"},
        },
    )
    journal.log_event(
        "TACTICAL_RESULT",
        {
            "timestamp": "2026-03-14T08:05:00+00:00",
            "symbol": "USDCAD",
            "resolution": "RETRY_PENDING",
            "summary_reason_code": "spread.fail.ratio_too_wide",
            "failed_hard_gate_reason_codes": [
                "spread.fail.ratio_too_wide",
            ],
            "context": {"session_label": "new_york", "regime_label": "normal"},
            "provenance": {"data_source": "rest_fallback"},
        },
    )

    snapshot = build_daily_entry_calibration_snapshot(journal, "2026-03-14")

    group = snapshot["groups"][0]
    assert group["failed_hard_gate_counts"]["spread.fail.ratio_too_wide"] == 2
    assert group["failed_hard_gate_counts"]["atr.fail.insufficient_1h_data"] == 1
    assert group["top_failed_hard_gates"][0]["reason_code"] == "spread.fail.ratio_too_wide"
