"""Tests for fx_duckdb_store — intraday storage (v1.2.0)."""

from datetime import date

import pandas as pd

from src.data.fx_duckdb_store import FxDuckDbStore


def test_upsert_and_read_intraday(tmp_path):
    """Should store and retrieve intraday bars with interval column."""
    store = FxDuckDbStore(tmp_path / "test.duckdb")

    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2026-03-01 00:00:00",
                    "2026-03-01 04:00:00",
                    "2026-03-01 08:00:00",
                    "2026-03-01 12:00:00",
                ]
            ),
            "open": [1.08, 1.085, 1.09, 1.087],
            "high": [1.09, 1.09, 1.095, 1.09],
            "low": [1.075, 1.08, 1.085, 1.085],
            "close": [1.085, 1.09, 1.087, 1.088],
            "volume": [100, 150, 200, 120],
        }
    )

    count = store.upsert_intraday("EURUSD", df, interval="4h", provider="tradermade")
    assert count == 4

    result = store.read_intraday("EURUSD", interval="4h")
    assert len(result) == 4
    assert "datetime" in result.columns
    store.close()


def test_read_intraday_date_filter(tmp_path):
    """Should filter intraday bars by date range."""
    store = FxDuckDbStore(tmp_path / "test.duckdb")

    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2026-03-01 00:00:00",
                    "2026-03-01 04:00:00",
                    "2026-03-02 00:00:00",
                    "2026-03-02 04:00:00",
                ]
            ),
            "open": [1.08, 1.085, 1.09, 1.087],
            "high": [1.09, 1.09, 1.095, 1.09],
            "low": [1.075, 1.08, 1.085, 1.085],
            "close": [1.085, 1.09, 1.087, 1.088],
            "volume": [100, 150, 200, 120],
        }
    )
    store.upsert_intraday("EURUSD", df, interval="4h")

    result = store.read_intraday(
        "EURUSD", interval="4h", start_date=date(2026, 3, 2), end_date=date(2026, 3, 2)
    )
    assert len(result) == 2
    store.close()


def test_upsert_intraday_deduplicates(tmp_path):
    """Upserting same data twice should not create duplicates."""
    store = FxDuckDbStore(tmp_path / "test.duckdb")

    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-03-01 00:00:00", "2026-03-01 04:00:00"]),
            "open": [1.08, 1.085],
            "high": [1.09, 1.09],
            "low": [1.075, 1.08],
            "close": [1.085, 1.09],
            "volume": [100, 150],
        }
    )

    store.upsert_intraday("EURUSD", df, interval="4h")
    store.upsert_intraday("EURUSD", df, interval="4h")  # Second upsert

    result = store.read_intraday("EURUSD", interval="4h")
    assert len(result) == 2  # Not 4
    store.close()


def test_different_intervals_are_separate(tmp_path):
    """4h and 1h data for same symbol should be stored separately."""
    store = FxDuckDbStore(tmp_path / "test.duckdb")

    df_4h = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-03-01 00:00:00", "2026-03-01 04:00:00"]),
            "open": [1.08, 1.085],
            "high": [1.09, 1.09],
            "low": [1.075, 1.08],
            "close": [1.085, 1.09],
            "volume": [100, 150],
        }
    )
    df_1h = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2026-03-01 00:00:00", "2026-03-01 01:00:00", "2026-03-01 02:00:00"]
            ),
            "open": [1.08, 1.082, 1.084],
            "high": [1.09, 1.088, 1.09],
            "low": [1.075, 1.08, 1.082],
            "close": [1.085, 1.084, 1.088],
            "volume": [100, 80, 90],
        }
    )

    store.upsert_intraday("EURUSD", df_4h, interval="4h")
    store.upsert_intraday("EURUSD", df_1h, interval="1h")

    result_4h = store.read_intraday("EURUSD", interval="4h")
    result_1h = store.read_intraday("EURUSD", interval="1h")
    assert len(result_4h) == 2
    assert len(result_1h) == 3
    store.close()


def test_upsert_intraday_empty_df(tmp_path):
    """Upserting empty DataFrame should return 0."""
    store = FxDuckDbStore(tmp_path / "test.duckdb")
    count = store.upsert_intraday("EURUSD", pd.DataFrame(), interval="4h")
    assert count == 0
    store.close()



def test_upsert_no_transaction_nesting_error(tmp_path):
    """upsert should not fail with cannot start a transaction error."""
    store = FxDuckDbStore(str(tmp_path / "test.duckdb"))
    df = pd.DataFrame({
        "date": [pd.Timestamp("2026-03-01")],
        "open": [1.08],
        "high": [1.09],
        "low": [1.07],
        "close": [1.085],
        "volume": [0],
    })

    # First upsert should work
    count = store.upsert("EURUSD", df, provider="test")
    assert count == 1

    # Second upsert (same data) should also work without transaction error
    count = store.upsert("EURUSD", df, provider="test")
    assert count == 1
    store.close()
