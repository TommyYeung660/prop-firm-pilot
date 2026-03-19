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


def test_build_daily_entry_calibration_snapshot_includes_entry_funnel_fields(
    tmp_path: Path,
) -> None:
    journal = TradeJournal(tmp_path / "trade_journal.jsonl")
    journal.log_event(
        "METRICS_SNAPSHOT",
        {
            "timestamp": "2026-03-14T22:00:00+00:00",
            "entry_funnel_mode": "scanner_llm_tactical",
            "entry_funnel": {
                "scanner_candidates": 5,
                "intents_created": 4,
                "llm_vetoes": 1,
                "llm_cancels": 1,
                "tactical_waits": 2,
                "tactical_expires": 1,
                "no_trade_count": 2,
                "no_trade_reasons": {"llm_veto": 1, "tactical_expire": 1},
            },
        },
    )
    journal.log_event(
        "TRADE_OPENED",
        {
            "timestamp": "2026-03-14T09:15:00+00:00",
            "symbol": "EURUSD",
            "intent_id": "i-1",
        },
    )
    journal.log_event(
        "TRADE_OPENED",
        {
            "timestamp": "2026-03-14T12:30:00+00:00",
            "symbol": "USDJPY",
            "intent_id": "i-2",
        },
    )

    snapshot = build_daily_entry_calibration_snapshot(journal, "2026-03-14")

    assert snapshot["entry_funnel_mode"] == "scanner_llm_tactical"
    assert snapshot["scanner_candidates"] == 5
    assert snapshot["intents_created"] == 4
    assert snapshot["opened_count"] == 2
    assert snapshot["llm_vetoes"] == 1
    assert snapshot["llm_cancels"] == 1
    assert snapshot["tactical_waits"] == 2
    assert snapshot["tactical_expires"] == 1
    assert snapshot["no_trade_count"] == 2
    assert snapshot["no_trade_reasons"]["llm_veto"] == 1
    assert snapshot["llm_veto_rate"] == 0.2


def test_build_daily_entry_calibration_snapshot_derives_economic_fields_from_trade_closes(
    tmp_path: Path,
) -> None:
    journal = TradeJournal(tmp_path / "trade_journal.jsonl")
    for timestamp, pnl in [
        ("2026-03-14T09:15:00+00:00", 50.0),
        ("2026-03-14T10:30:00+00:00", -20.0),
        ("2026-03-14T14:05:00+00:00", 10.0),
    ]:
        journal.log_event(
            "TRADE_CLOSED",
            {
                "timestamp": timestamp,
                "symbol": "EURUSD",
                "pnl": pnl,
            },
        )

    snapshot = build_daily_entry_calibration_snapshot(journal, "2026-03-14")

    assert snapshot["net_pnl"] == 40.0
    assert snapshot["profit_factor"] == 3.0
    assert snapshot["max_drawdown"] == 20.0
