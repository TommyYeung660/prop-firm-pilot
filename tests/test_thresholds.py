"""
Tests for threshold computation against the current production policy.

Ensures:
1. Cold-start tier (win_rate < 0.20) uses relaxed thresholds
2. Losing tier (0.20-0.45) stays at medium confidence with a slightly tighter blend
3. Medium tier (0.45-0.55) uses medium confidence
4. High win_rate tier (> 0.55) uses low confidence (permissive)
5. Per-symbol adjustments work correctly
"""

from src.optimize.thresholds import _stepwise_threshold, compute_thresholds


class TestStepwiseThresholds:
    """Tests for the _stepwise_threshold function."""

    def test_cold_start_very_low_win_rate(self):
        """win_rate < 0.20 should use medium/0.48 (cold-start, relaxed)."""
        t = _stepwise_threshold(0.0)
        assert t.min_confidence == "medium"
        assert t.min_blended_confidence == 0.48

    def test_cold_start_at_boundary(self):
        """win_rate = 0.19 should still be in cold-start tier."""
        t = _stepwise_threshold(0.19)
        assert t.min_confidence == "medium"
        assert t.min_blended_confidence == 0.48

    def test_low_win_rate_tier(self):
        """0.20 <= win_rate < 0.45 should use medium/0.52."""
        t = _stepwise_threshold(0.20)
        assert t.min_confidence == "medium"
        assert t.min_blended_confidence == 0.52

    def test_low_win_rate_upper_boundary(self):
        """win_rate = 0.44 should still be in low tier."""
        t = _stepwise_threshold(0.44)
        assert t.min_confidence == "medium"
        assert t.min_blended_confidence == 0.52

    def test_medium_win_rate_tier(self):
        """0.45 <= win_rate <= 0.55 should use medium/0.50."""
        t = _stepwise_threshold(0.50)
        assert t.min_confidence == "medium"
        assert t.min_blended_confidence == 0.50

    def test_high_win_rate_tier(self):
        """win_rate > 0.55 should use low/0.45 (permissive)."""
        t = _stepwise_threshold(0.60)
        assert t.min_confidence == "low"
        assert t.min_blended_confidence == 0.45


class TestComputeThresholds:
    """Tests for the compute_thresholds function."""

    def test_cold_start_symbol_gets_relaxed_global(self):
        """A symbol with 0% win rate (cold start) should be gated less aggressively."""
        result = compute_thresholds(
            global_win_rate=0.0,
            symbol_win_rates={"EURUSD": 0.0},
        )
        assert result["global"].min_confidence == "medium"
        assert result["global"].min_blended_confidence == 0.48
        # Per-symbol adjustment is minimal (delta = 0)
        assert result["EURUSD"].min_blended_confidence == 0.48

    def test_mixed_symbols(self):
        """Different symbols should get appropriate thresholds based on global rate."""
        result = compute_thresholds(
            global_win_rate=0.50,
            symbol_win_rates={"EURUSD": 0.50, "XAUUSD": 0.60},
        )
        # Global is medium tier
        assert result["global"].min_confidence == "medium"
        # EURUSD: delta = 0 → no adjustment
        assert result["EURUSD"].min_blended_confidence == 0.50
        # XAUUSD: delta = +0.10 > 0.05 → adj = 0.03, blended = 0.50 - 0.03 = 0.47
        assert result["XAUUSD"].min_blended_confidence == 0.47

    def test_returns_global_key(self):
        """Result should always contain a 'global' key."""
        result = compute_thresholds(
            global_win_rate=0.50,
            symbol_win_rates={},
        )
        assert "global" in result
