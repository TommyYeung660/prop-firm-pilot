"""Tests for volatility monitor (v1.2.0)."""

from datetime import datetime, timedelta, timezone

from src.config import SchedulerConfig
from src.scheduler.volatility_monitor import VolatilityMonitor


def _utc_now() -> datetime:
    return datetime(2026, 3, 4, 12, 0, 0, tzinfo=timezone.utc)


class TestVolatilityMonitor:
    def test_disabled_returns_no_trigger(self):
        config = SchedulerConfig(volatility_trigger_enabled=False)
        monitor = VolatilityMonitor(config, ["EURUSD"])
        now = _utc_now()
        monitor.record_quote("EURUSD", 1.0800, now - timedelta(minutes=20))
        monitor.record_quote("EURUSD", 1.1200, now)  # +3.7% — huge move
        triggered, _, _ = monitor.check_triggers(now)
        assert not triggered

    def test_trigger_on_threshold_breach(self):
        config = SchedulerConfig(
            volatility_trigger_enabled=True,
            volatility_threshold_pct=0.3,
            volatility_window_minutes=30,
            volatility_cooldown_seconds=0,  # Disable cooldown for test
        )
        monitor = VolatilityMonitor(config, ["EURUSD"])
        now = _utc_now()
        monitor.record_quote("EURUSD", 1.0800, now - timedelta(minutes=15))
        monitor.record_quote("EURUSD", 1.0840, now)  # +0.37%
        triggered, symbol, pct = monitor.check_triggers(now)
        assert triggered
        assert symbol == "EURUSD"
        assert pct > 0.3

    def test_no_trigger_below_threshold(self):
        config = SchedulerConfig(
            volatility_trigger_enabled=True,
            volatility_threshold_pct=0.3,
            volatility_window_minutes=30,
            volatility_cooldown_seconds=0,
        )
        monitor = VolatilityMonitor(config, ["EURUSD"])
        now = _utc_now()
        monitor.record_quote("EURUSD", 1.0800, now - timedelta(minutes=15))
        monitor.record_quote("EURUSD", 1.0810, now)  # +0.09%
        triggered, _, _ = monitor.check_triggers(now)
        assert not triggered

    def test_cooldown_prevents_repeat_trigger(self):
        config = SchedulerConfig(
            volatility_trigger_enabled=True,
            volatility_threshold_pct=0.3,
            volatility_window_minutes=30,
            volatility_cooldown_seconds=900,
        )
        monitor = VolatilityMonitor(config, ["EURUSD"])
        now = _utc_now()
        monitor.record_quote("EURUSD", 1.0800, now - timedelta(minutes=15))
        monitor.record_quote("EURUSD", 1.0840, now)  # +0.37%

        # First trigger should fire
        triggered, _, _ = monitor.check_triggers(now)
        assert triggered

        # Immediate re-check should be blocked by cooldown
        monitor.record_quote("EURUSD", 1.0880, now + timedelta(minutes=1))
        triggered, _, _ = monitor.check_triggers(now + timedelta(minutes=1))
        assert not triggered

        # After cooldown expires, should trigger again
        later = now + timedelta(seconds=901)
        monitor.record_quote("EURUSD", 1.0900, later)
        triggered, _, _ = monitor.check_triggers(later)
        assert triggered

    def test_multi_symbol_picks_largest_move(self):
        config = SchedulerConfig(
            volatility_trigger_enabled=True,
            volatility_threshold_pct=0.3,
            volatility_window_minutes=30,
            volatility_cooldown_seconds=0,
        )
        monitor = VolatilityMonitor(config, ["EURUSD", "XAUUSD"])
        now = _utc_now()
        # EURUSD: small move
        monitor.record_quote("EURUSD", 1.0800, now - timedelta(minutes=10))
        monitor.record_quote("EURUSD", 1.0810, now)  # +0.09%
        # XAUUSD: large move
        monitor.record_quote("XAUUSD", 2000.0, now - timedelta(minutes=10))
        monitor.record_quote("XAUUSD", 2010.0, now)  # +0.5%

        triggered, symbol, pct = monitor.check_triggers(now)
        assert triggered
        assert symbol == "XAUUSD"
        assert pct > 0.3

    def test_prune_old_quotes(self):
        config = SchedulerConfig(
            volatility_trigger_enabled=True,
            volatility_window_minutes=30,
        )
        monitor = VolatilityMonitor(config, ["EURUSD"])
        now = _utc_now()
        # Add a quote that's way too old (2 hours ago)
        monitor.record_quote("EURUSD", 1.0500, now - timedelta(hours=2))
        # Add current quote — old one should be pruned
        monitor.record_quote("EURUSD", 1.0800, now)

        # Only 1 quote should remain (the old one was pruned)
        assert len(monitor._quotes["EURUSD"]) == 1

    def test_reset_clears_state(self):
        config = SchedulerConfig(volatility_trigger_enabled=True)
        monitor = VolatilityMonitor(config, ["EURUSD"])
        now = _utc_now()
        monitor.record_quote("EURUSD", 1.0800, now)
        monitor.reset()
        assert len(monitor._quotes["EURUSD"]) == 0
        assert monitor._last_trigger_time is None
        assert monitor._last_trigger_per_symbol == {}

    def test_single_quote_no_trigger(self):
        config = SchedulerConfig(
            volatility_trigger_enabled=True,
            volatility_threshold_pct=0.3,
            volatility_cooldown_seconds=0,
        )
        monitor = VolatilityMonitor(config, ["EURUSD"])
        now = _utc_now()
        monitor.record_quote("EURUSD", 1.0800, now)
        triggered, _, _ = monitor.check_triggers(now)
        assert not triggered  # Need at least 2 quotes


    def test_per_symbol_cooldown_independent(self):
        """Different symbols should have independent cooldowns (after global interval)."""
        config = SchedulerConfig(
            volatility_trigger_enabled=True,
            volatility_threshold_pct=0.3,
            volatility_window_minutes=30,
            volatility_cooldown_seconds=1800,  # 30 min per-symbol
        )
        monitor = VolatilityMonitor(config, ["EURUSD", "XAUUSD"])
        now = _utc_now()

        # Trigger EURUSD first
        monitor.record_quote("EURUSD", 1.0800, now - timedelta(minutes=10))
        monitor.record_quote("EURUSD", 1.0850, now)  # +0.46%
        triggered, sym, _ = monitor.check_triggers(now)
        assert triggered
        assert sym == "EURUSD"

        # 16 min later — past global 15min but within EURUSD's 30min cooldown
        t1 = now + timedelta(minutes=16)
        # EURUSD still has a big move, but per-symbol cooldown blocks it
        monitor.record_quote("EURUSD", 1.0800, now + timedelta(minutes=10))
        monitor.record_quote("EURUSD", 1.0860, t1)
        # XAUUSD also moves — it should trigger since no per-symbol cooldown
        monitor.record_quote("XAUUSD", 2000.0, now)
        monitor.record_quote("XAUUSD", 2010.0, t1)  # +0.5%
        triggered2, sym2, _ = monitor.check_triggers(t1)
        assert triggered2
        assert sym2 == "XAUUSD"

    def test_global_min_interval_blocks_all_symbols(self):
        """Within global 15min interval, no symbol should trigger even if cooldown allows."""
        config = SchedulerConfig(
            volatility_trigger_enabled=True,
            volatility_threshold_pct=0.3,
            volatility_window_minutes=30,
            volatility_cooldown_seconds=60,  # Very short per-symbol cooldown
        )
        monitor = VolatilityMonitor(config, ["EURUSD", "XAUUSD"])
        now = _utc_now()

        # Trigger EURUSD
        monitor.record_quote("EURUSD", 1.0800, now - timedelta(minutes=10))
        monitor.record_quote("EURUSD", 1.0850, now)
        triggered, _, _ = monitor.check_triggers(now)
        assert triggered

        # 5 min later — past per-symbol cooldown (60s) but within global (900s)
        t1 = now + timedelta(minutes=5)
        monitor.record_quote("XAUUSD", 2000.0, now)
        monitor.record_quote("XAUUSD", 2020.0, t1)  # +1.0% big move
        triggered2, _, _ = monitor.check_triggers(t1)
        assert not triggered2  # Global interval blocks it
