"""Tests for dynamic confidence thresholds."""

from src.optimize.thresholds import compute_thresholds


# ── Tests ───────────────────────────────────────────────────────────────────


def test_stepwise_thresholds_low_winrate() -> None:
    result = compute_thresholds(global_win_rate=0.40, symbol_win_rates={})
    assert result["global"].min_confidence == "high"
    assert result["global"].min_blended_confidence == 0.65


def test_stepwise_thresholds_high_winrate() -> None:
    result = compute_thresholds(global_win_rate=0.60, symbol_win_rates={})
    assert result["global"].min_confidence == "low"
    assert result["global"].min_blended_confidence == 0.45


def test_symbol_threshold_adjustment() -> None:
    result = compute_thresholds(
        global_win_rate=0.50,
        symbol_win_rates={"EURUSD": 0.60, "GBPUSD": 0.40},
    )
    assert result["global"].min_confidence == "medium"
    assert result["global"].min_blended_confidence == 0.55
    assert result["EURUSD"].min_blended_confidence == 0.50
    assert result["GBPUSD"].min_blended_confidence == 0.60
