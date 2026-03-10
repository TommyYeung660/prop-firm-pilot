"""
Tests for low-confidence scanner cooldown (P3.10).

Validates:
1. LowConfidenceCooldown tracker counts consecutive cancellations
2. Cooldown is applied after threshold cancellations
3. Cooldown expires after configured duration
4. Counter resets on successful trade
5. Different symbols have independent counters
"""

from datetime import datetime, timedelta, timezone

from src.scheduler.low_confidence_cooldown import LowConfidenceCooldown


class TestLowConfidenceCooldown:
    """Tests for LowConfidenceCooldown tracker."""

    def test_initial_state_no_cooldown(self):
        """New tracker has no cooldowns."""
        tracker = LowConfidenceCooldown(cooldown_minutes=240, threshold=2)
        now = datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)
        assert tracker.is_cooled_down("AUDUSD", now) is False

    def test_single_cancel_no_cooldown(self):
        """One cancellation below threshold doesn't trigger cooldown."""
        tracker = LowConfidenceCooldown(cooldown_minutes=240, threshold=2)
        now = datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)
        tracker.record_low_confidence("AUDUSD", now)
        assert tracker.is_cooled_down("AUDUSD", now) is False

    def test_threshold_triggers_cooldown(self):
        """Reaching threshold triggers cooldown."""
        tracker = LowConfidenceCooldown(cooldown_minutes=240, threshold=2)
        now = datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)
        tracker.record_low_confidence("AUDUSD", now)
        tracker.record_low_confidence("AUDUSD", now)
        assert tracker.is_cooled_down("AUDUSD", now) is True

    def test_cooldown_expires(self):
        """Cooldown expires after configured duration."""
        tracker = LowConfidenceCooldown(cooldown_minutes=240, threshold=2)
        now = datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)
        tracker.record_low_confidence("AUDUSD", now)
        tracker.record_low_confidence("AUDUSD", now)
        later = now + timedelta(minutes=241)
        assert tracker.is_cooled_down("AUDUSD", later) is False

    def test_cooldown_not_expired(self):
        """Cooldown still active before expiry."""
        tracker = LowConfidenceCooldown(cooldown_minutes=240, threshold=2)
        now = datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)
        tracker.record_low_confidence("AUDUSD", now)
        tracker.record_low_confidence("AUDUSD", now)
        later = now + timedelta(minutes=120)
        assert tracker.is_cooled_down("AUDUSD", later) is True

    def test_reset_clears_cooldown(self):
        """reset_symbol() clears counter and cooldown."""
        tracker = LowConfidenceCooldown(cooldown_minutes=240, threshold=2)
        now = datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)
        tracker.record_low_confidence("AUDUSD", now)
        tracker.record_low_confidence("AUDUSD", now)
        tracker.reset_symbol("AUDUSD")
        assert tracker.is_cooled_down("AUDUSD", now) is False

    def test_independent_symbols(self):
        """Different symbols have independent counters."""
        tracker = LowConfidenceCooldown(cooldown_minutes=240, threshold=2)
        now = datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)
        tracker.record_low_confidence("AUDUSD", now)
        tracker.record_low_confidence("AUDUSD", now)
        assert tracker.is_cooled_down("AUDUSD", now) is True
        assert tracker.is_cooled_down("EURUSD", now) is False

    def test_get_count(self):
        """get_count() returns current counter for symbol."""
        tracker = LowConfidenceCooldown(cooldown_minutes=240, threshold=2)
        now = datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)
        assert tracker.get_count("AUDUSD") == 0
        tracker.record_low_confidence("AUDUSD", now)
        assert tracker.get_count("AUDUSD") == 1

    def test_daily_reset(self):
        """reset_all() clears everything (called at day boundary)."""
        tracker = LowConfidenceCooldown(cooldown_minutes=240, threshold=2)
        now = datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc)
        tracker.record_low_confidence("AUDUSD", now)
        tracker.record_low_confidence("AUDUSD", now)
        tracker.reset_all()
        assert tracker.is_cooled_down("AUDUSD", now) is False
        assert tracker.get_count("AUDUSD") == 0
