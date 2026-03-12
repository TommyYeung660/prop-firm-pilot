"""Tests for EODHD websocket live probe summary helpers."""

from datetime import datetime, timezone

import pandas as pd

from src.diagnostics.eodhd_websocket_live import summarize_rest_bars, summarize_tick_events


def test_summarize_tick_events_tracks_count_max_gap_and_latest_age() -> None:
    now = datetime(2026, 3, 12, 12, 0, 10, tzinfo=timezone.utc)
    events = [
        ("EURUSD", datetime(2026, 3, 12, 12, 0, 0, tzinfo=timezone.utc)),
        ("EURUSD", datetime(2026, 3, 12, 12, 0, 6, tzinfo=timezone.utc)),
        ("GBPUSD", datetime(2026, 3, 12, 12, 0, 3, tzinfo=timezone.utc)),
    ]

    summary = summarize_tick_events(["EURUSD", "GBPUSD", "AUDUSD"], events, now)

    assert summary["EURUSD"]["count"] == 2
    assert summary["EURUSD"]["max_gap_sec"] == 6.0
    assert summary["EURUSD"]["latest_age_sec"] == 4.0
    assert summary["GBPUSD"]["count"] == 1
    assert summary["GBPUSD"]["max_gap_sec"] == 0.0
    assert summary["GBPUSD"]["latest_age_sec"] == 7.0
    assert summary["AUDUSD"]["count"] == 0
    assert summary["AUDUSD"]["latest_age_sec"] is None


def test_summarize_rest_bars_reports_latest_bar_age() -> None:
    now = datetime(2026, 3, 12, 12, 0, 0, tzinfo=timezone.utc)
    bars = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-03-12T11:50:00Z"),
                "open": 1.10,
                "high": 1.11,
                "low": 1.09,
                "close": 1.105,
                "volume": 0,
            },
            {
                "datetime": pd.Timestamp("2026-03-12T11:58:00Z"),
                "open": 1.105,
                "high": 1.12,
                "low": 1.10,
                "close": 1.115,
                "volume": 0,
            },
        ]
    )

    summary = summarize_rest_bars("EURUSD", bars, now)

    assert summary["symbol"] == "EURUSD"
    assert summary["rows"] == 2
    assert summary["latest_bar_time"] == "2026-03-12T11:58:00+00:00"
    assert summary["latest_bar_age_sec"] == 120.0
