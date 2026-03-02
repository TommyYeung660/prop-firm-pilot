"""Tests for HighWaterMarkTracker — dynamic drawdown HWM persistence."""

import json
from pathlib import Path

import pytest

from src.compliance.hwm_tracker import HighWaterMarkTracker


class TestHighWaterMarkTracker:
    """Tests for HWM tracking, persistence, and lock logic."""

    def test_initial_hwm_equals_initial_balance(self, tmp_path: Path) -> None:
        tracker = HighWaterMarkTracker(
            initial_balance=5000.0,
            drawdown_pct=0.06,
            state_path=str(tmp_path / "hwm.json"),
        )
        assert tracker.high_water_mark == 5000.0
        assert tracker.loss_level == 5000.0 * (1 - 0.06)  # 4700.0
        assert tracker.is_locked is False

    def test_update_on_profitable_close(self, tmp_path: Path) -> None:
        tracker = HighWaterMarkTracker(
            initial_balance=5000.0,
            drawdown_pct=0.06,
            state_path=str(tmp_path / "hwm.json"),
        )
        tracker.update_balance(5050.0)  # Trade closed with $50 profit
        assert tracker.high_water_mark == 5050.0
        assert tracker.loss_level == 5050.0 * (1 - 0.06)  # 4747.0

    def test_no_update_on_loss(self, tmp_path: Path) -> None:
        tracker = HighWaterMarkTracker(
            initial_balance=5000.0,
            drawdown_pct=0.06,
            state_path=str(tmp_path / "hwm.json"),
        )
        tracker.update_balance(5050.0)
        tracker.update_balance(5020.0)  # Balance dropped — no update
        assert tracker.high_water_mark == 5050.0
        assert tracker.loss_level == 5050.0 * (1 - 0.06)

    def test_lock_when_profit_exceeds_drawdown_amount(self, tmp_path: Path) -> None:
        """When cumulative realized profit >= initial_balance * drawdown_pct,
        loss_level locks permanently at initial_balance."""
        tracker = HighWaterMarkTracker(
            initial_balance=5000.0,
            drawdown_pct=0.06,  # $300 drawdown amount
            state_path=str(tmp_path / "hwm.json"),
        )
        tracker.update_balance(5300.0)  # Profit = $300 = drawdown amount
        assert tracker.is_locked is True
        assert tracker.loss_level == 5000.0  # Locked at initial balance

    def test_lock_persists_after_further_profit(self, tmp_path: Path) -> None:
        tracker = HighWaterMarkTracker(
            initial_balance=5000.0,
            drawdown_pct=0.06,
            state_path=str(tmp_path / "hwm.json"),
        )
        tracker.update_balance(5300.0)
        tracker.update_balance(5500.0)  # More profit after lock
        assert tracker.is_locked is True
        assert tracker.loss_level == 5000.0  # Still locked at initial
        assert tracker.high_water_mark == 5500.0  # HWM still tracks

    def test_persistence_to_json(self, tmp_path: Path) -> None:
        state_path = str(tmp_path / "hwm.json")
        tracker = HighWaterMarkTracker(
            initial_balance=5000.0,
            drawdown_pct=0.06,
            state_path=state_path,
        )
        tracker.update_balance(5100.0)
        tracker.save()

        # Load from file
        with open(state_path) as f:
            data = json.load(f)
        assert data["high_water_mark"] == 5100.0
        assert data["initial_balance"] == 5000.0
        assert data["is_locked"] is False

    def test_restore_from_json(self, tmp_path: Path) -> None:
        state_path = str(tmp_path / "hwm.json")
        # First tracker saves state
        t1 = HighWaterMarkTracker(
            initial_balance=5000.0,
            drawdown_pct=0.06,
            state_path=state_path,
        )
        t1.update_balance(5200.0)
        t1.save()

        # Second tracker restores
        t2 = HighWaterMarkTracker(
            initial_balance=5000.0,
            drawdown_pct=0.06,
            state_path=state_path,
        )
        assert t2.high_water_mark == 5200.0
        assert t2.loss_level == 5200.0 * (1 - 0.06)

    def test_restore_ignores_mismatched_initial_balance(self, tmp_path: Path) -> None:
        """If persisted state has different initial_balance, ignore it (account changed)."""
        state_path = str(tmp_path / "hwm.json")
        t1 = HighWaterMarkTracker(
            initial_balance=5000.0,
            drawdown_pct=0.06,
            state_path=state_path,
        )
        t1.update_balance(5200.0)
        t1.save()

        # Different initial_balance — should not restore
        t2 = HighWaterMarkTracker(
            initial_balance=50000.0,
            drawdown_pct=0.08,
            state_path=state_path,
        )
        assert t2.high_water_mark == 50000.0  # Ignored persisted state

    def test_max_drawdown_remaining(self, tmp_path: Path) -> None:
        tracker = HighWaterMarkTracker(
            initial_balance=5000.0,
            drawdown_pct=0.06,
            state_path=str(tmp_path / "hwm.json"),
        )
        tracker.update_balance(5100.0)
        # loss_level = 5100 * 0.94 = 4794
        # If equity is 5050, remaining = 5050 - 4794 = 256
        assert tracker.max_drawdown_remaining(equity=5050.0) == pytest.approx(256.0)

    def test_max_drawdown_remaining_locked(self, tmp_path: Path) -> None:
        tracker = HighWaterMarkTracker(
            initial_balance=5000.0,
            drawdown_pct=0.06,
            state_path=str(tmp_path / "hwm.json"),
        )
        tracker.update_balance(5400.0)
        assert tracker.is_locked is True
        # loss_level = 5000 (locked)
        # If equity is 5350, remaining = 5350 - 5000 = 350
        assert tracker.max_drawdown_remaining(equity=5350.0) == pytest.approx(350.0)


def test_config_has_hwm_state_path():
    from src.config import ComplianceConfig
    c = ComplianceConfig(drawdown_type="dynamic")
    assert hasattr(c, "hwm_state_path")
    assert c.hwm_state_path == "data/hwm_state.json"
