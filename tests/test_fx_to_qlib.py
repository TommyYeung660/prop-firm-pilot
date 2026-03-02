"""Tests for fx_to_qlib — intraday Qlib binary conversion (v1.2.0)."""

import numpy as np
import pandas as pd

from src.data.fx_to_qlib import convert_to_qlib_binary


def test_convert_intraday_4h(tmp_path):
    """4h conversion should use 4h calendar and preserve time component."""
    data = {
        "EURUSD": pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    [
                        "2026-03-01 00:00:00",
                        "2026-03-01 04:00:00",
                        "2026-03-01 08:00:00",
                    ]
                ),
                "open": [1.08, 1.085, 1.09],
                "high": [1.09, 1.09, 1.095],
                "low": [1.075, 1.08, 1.085],
                "close": [1.085, 1.09, 1.087],
                "volume": [100, 150, 200],
            }
        )
    }

    result_dir = convert_to_qlib_binary(data, tmp_path / "qlib_4h", interval="4h")

    # Check calendar file uses 4h suffix
    cal_file = result_dir / "calendars" / "4h.txt"
    assert cal_file.exists()
    lines = cal_file.read_text().strip().split("\n")
    assert len(lines) == 3
    # Should preserve time component
    assert "00:00:00" in lines[0]
    assert "04:00:00" in lines[1]

    # Check feature bin uses 4h suffix
    close_bin = result_dir / "features" / "EURUSD" / "close.4h.bin"
    assert close_bin.exists()
    values = np.fromfile(str(close_bin), dtype="<f4")
    assert len(values) == 3


def test_convert_daily_backward_compatible(tmp_path):
    """Daily conversion should still use day.bin and day.txt."""
    data = {
        "EURUSD": pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2026-03-01", "2026-03-02"]),
                "open": [1.08, 1.085],
                "high": [1.09, 1.09],
                "low": [1.075, 1.08],
                "close": [1.085, 1.09],
                "volume": [100, 150],
            }
        )
    }

    result_dir = convert_to_qlib_binary(data, tmp_path / "qlib_daily")

    cal_file = result_dir / "calendars" / "day.txt"
    assert cal_file.exists()
    close_bin = result_dir / "features" / "EURUSD" / "close.day.bin"
    assert close_bin.exists()

    # Calendar entries should NOT have time component for daily
    lines = cal_file.read_text().strip().split("\n")
    assert "00:00:00" not in lines[0]


def test_convert_intraday_1h(tmp_path):
    """1h conversion should use 1h calendar and bin suffix."""
    data = {
        "EURUSD": pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2026-03-01 00:00:00", "2026-03-01 01:00:00"]),
                "open": [1.08, 1.085],
                "high": [1.09, 1.09],
                "low": [1.075, 1.08],
                "close": [1.085, 1.09],
                "volume": [100, 150],
            }
        )
    }

    result_dir = convert_to_qlib_binary(data, tmp_path / "qlib_1h", interval="1h")

    cal_file = result_dir / "calendars" / "1h.txt"
    assert cal_file.exists()
    close_bin = result_dir / "features" / "EURUSD" / "close.1h.bin"
    assert close_bin.exists()
