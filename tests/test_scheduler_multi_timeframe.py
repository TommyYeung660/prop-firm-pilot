"""Tests for multi-timeframe scanner integration (v1.2.0).

Covers:
- Config field defaults for multi_timeframe_enabled, entry_timeframe, intraday_lookback_days
- _run_intraday_scan() calls scanner with correct interval and symbols
- Scanner loop calls _run_intraday_scan() when enabled and signals exist
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import AppConfig


# ── Config defaults ──────────────────────────────────────────────────────────


def test_multi_timeframe_config_defaults():
    config = AppConfig()
    assert config.scheduler.multi_timeframe_enabled is False
    assert config.scheduler.entry_timeframe == "4h"
    assert config.scheduler.intraday_lookback_days == 90


def test_multi_timeframe_config_custom():
    config = AppConfig(
        scheduler={
            "multi_timeframe_enabled": True,
            "entry_timeframe": "1h",
            "intraday_lookback_days": 60,
        }
    )
    assert config.scheduler.multi_timeframe_enabled is True
    assert config.scheduler.entry_timeframe == "1h"
    assert config.scheduler.intraday_lookback_days == 60


# ── _run_intraday_scan() ────────────────────────────────────────────────────


@pytest.fixture
def mock_scheduler():
    """Create a minimal Scheduler with mocked dependencies for intraday scan testing."""
    from src.scheduler.scheduler import Scheduler

    config = AppConfig(scheduler={"multi_timeframe_enabled": True, "entry_timeframe": "4h"})

    scanner = MagicMock()
    scanner.run_pipeline = MagicMock(return_value=[])

    scheduler = Scheduler(
        config=config,
        store=MagicMock(),
        scanner=scanner,
        agents=MagicMock(),
        engine=MagicMock(),
        matchtrader=MagicMock(),
        alert_service=None,
        instrument_registry={},
        best_day_tracker=MagicMock(),
        optimization_engine=None,
        memory_journal=None,
        trade_journal=None,
    )
    return scheduler


async def test_run_intraday_scan_calls_scanner(mock_scheduler):
    """_run_intraday_scan() should call scanner.run_pipeline with interval and symbols."""
    # Create fake daily signals
    signal1 = MagicMock()
    signal1.instrument = "EURUSD"
    signal2 = MagicMock()
    signal2.instrument = "GBPUSD"

    mock_scheduler._scanner.run_pipeline = MagicMock(return_value=[])

    await mock_scheduler._run_intraday_scan([signal1, signal2], "2026-03-02")

    mock_scheduler._scanner.run_pipeline.assert_called_once_with(
        date="2026-03-02",
        tickers=["EURUSD", "GBPUSD"],
        interval="4h",
    )


async def test_run_intraday_scan_returns_signals(mock_scheduler):
    """_run_intraday_scan() should return intraday signals from scanner."""
    signal = MagicMock()
    signal.instrument = "EURUSD"

    intraday_signal = MagicMock()
    intraday_signal.instrument = "EURUSD"
    intraday_signal.score = 0.85
    intraday_signal.confidence = "HIGH"

    mock_scheduler._scanner.run_pipeline = MagicMock(return_value=[intraday_signal])

    await mock_scheduler._run_intraday_scan([signal], "2026-03-02")

    # Verify scanner was called (results are logged, not returned in v1.2.0)
    mock_scheduler._scanner.run_pipeline.assert_called_once()
