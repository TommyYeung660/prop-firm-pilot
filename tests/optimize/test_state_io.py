"""Tests for optimization state IO."""

from pathlib import Path

from src.optimize.optimization_state import OptimizationState, load_state, save_state


# ── Tests ─────────────────────────────────────────────────────────────────--


def test_load_state_missing_returns_default(tmp_path: Path) -> None:
    """Missing file should return default state."""
    state = load_state(tmp_path / "missing.json")
    assert isinstance(state, OptimizationState)
    assert state.version == "1.0"


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """Saved state should load back with same version."""
    path = tmp_path / "state.json"
    state = OptimizationState(version="1.0")
    save_state(path, state)
    loaded = load_state(path)
    assert loaded.version == "1.0"
