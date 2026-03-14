"""
Tests for 'No Data = No Trade' safety guard.

When ALL tactical data (spread, bars, freshness) is missing, trades should
be REJECTED instead of silently passing through.
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.config import TacticalConfig
from src.decision.tactical_validator import TacticalData, TacticalValidator


@pytest.fixture
def validator() -> TacticalValidator:
    """Default TacticalValidator with stock config."""
    return TacticalValidator(config=TacticalConfig())


# ── Tests ──────────────────────────────────────────────────────────────────


def test_empty_tactical_data_rejects(validator: TacticalValidator) -> None:
    """TacticalData with all defaults (empty DFs, zero spread, no bar time) → REJECT."""
    data = TacticalData()  # all defaults: empty bars, 0 spread, None bar time
    result = validator.evaluate(side="BUY", data=data)
    assert result.action == "REJECT"
    assert result.resolution == "SKIP_CANCEL"
    assert result.summary_reason_code == "data.reject.no_tactical_inputs"
    assert "no" in result.detail.lower() or "missing" in result.detail.lower()


def test_no_bars_no_spread_no_freshness_rejects(validator: TacticalValidator) -> None:
    """Explicitly zero spread + empty bars + no timestamp → REJECT."""
    data = TacticalData(
        current_spread=0.0,
        typical_spread=0.0,
        bars_5min=pd.DataFrame(),
        bars_1h=pd.DataFrame(),
        latest_bar_time=None,
    )
    result = validator.evaluate(side="SELL", data=data)
    assert result.action == "REJECT"
    assert result.resolution == "SKIP_CANCEL"
    assert result.summary_reason_code == "data.reject.no_tactical_inputs"


def test_partial_data_with_spread_does_not_reject(validator: TacticalValidator) -> None:
    """When spread data exists but bars are empty, should NOT be REJECT (partial data)."""
    data = TacticalData(
        current_spread=0.00030,
        typical_spread=0.00020,
        latest_bar_time=datetime.now(timezone.utc),
    )
    result = validator.evaluate(side="BUY", data=data)
    # Partial data should be evaluated by normal gate logic, not blanket-rejected
    assert result.action != "REJECT"


def test_partial_data_with_bars_does_not_reject(validator: TacticalValidator) -> None:
    """When bar data exists but spread is 0, should NOT be REJECT (partial data)."""
    bars = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-03-10 08:00", periods=30, freq="5min"),
            "open": [1.085] * 30,
            "high": [1.086] * 30,
            "low": [1.084] * 30,
            "close": [1.0855] * 30,
            "volume": [100] * 30,
        }
    )
    data = TacticalData(
        bars_5min=bars,
        current_spread=0.0,
        typical_spread=0.0,
        latest_bar_time=datetime.now(timezone.utc),
    )
    result = validator.evaluate(side="BUY", data=data)
    assert result.action != "REJECT"


def test_partial_data_with_freshness_does_not_reject(validator: TacticalValidator) -> None:
    """When only freshness exists (no spread, no bars), still counts as partial."""
    data = TacticalData(
        latest_bar_time=datetime.now(timezone.utc),
    )
    result = validator.evaluate(side="SELL", data=data)
    # Has at least one signal (freshness) → not blanket-rejected
    assert result.action != "REJECT"
