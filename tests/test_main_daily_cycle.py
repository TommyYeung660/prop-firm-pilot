"""
Regression tests for PropFirmPilot.run_daily_cycle().

Focuses on the legacy single-cycle path, which must now respect side-aware
scanner signals the same way the async scheduler path does.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import AppConfig, ExecutionConfig, InstrumentConfig, ScannerConfig
from src.decision.agent_bridge import AgentDecision
from src.main import PropFirmPilot
from src.signal.scanner_bridge import ScannerSignal


@pytest.fixture
def config(tmp_path) -> AppConfig:
    """Minimal config for exercising the legacy daily-cycle path."""
    return AppConfig(
        scanner=ScannerConfig(
            project_path=str(tmp_path / "qlib_market_scanner"),
            topk=2,
            topk_short=1,
        ),
        execution=ExecutionConfig(
            random_delay_min=0.0,
            random_delay_max=0.0,
        ),
        instruments={
            "EURUSD": InstrumentConfig(pip_value=10.0, pip_size=0.0001),
            "USDCHF": InstrumentConfig(pip_value=10.0, pip_size=0.0001),
        },
    )


@pytest.fixture
def matchtrader_client() -> AsyncMock:
    """Async MatchTrader client stub used by run_daily_cycle()."""
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    client.get_balance.return_value = MagicMock(
        balance=50000.0,
        equity=50000.0,
        margin=0.0,
        free_margin=50000.0,
    )
    client.get_open_positions.return_value = []
    return client


def _make_signal(
    *,
    instrument: str,
    score: float,
    rank: int,
    side: str,
) -> ScannerSignal:
    return ScannerSignal(
        instrument=instrument,
        score=score,
        rank=rank,
        confidence="high",
        score_gap=0.02,
        drop_distance=0.02,
        topk_spread=0.02,
        scanner_version="v1.5.0_beta",
        schema_version="fx_signal_v2",
        label_version="cost_aware_directional_return_v1",
        market_date="2026-03-17",
        side=side,
    )


def _build_pilot(config: AppConfig) -> PropFirmPilot:
    pilot = PropFirmPilot(config)
    pilot.scanner = MagicMock()
    pilot.agents = MagicMock()
    pilot.journal = MagicMock()
    pilot.memory_journal = MagicMock()
    pilot.alert_service = MagicMock()
    pilot.alert_service.system_error = AsyncMock(return_value=True)
    pilot.alert_service.trade_opened = AsyncMock(return_value=True)
    pilot._execute_trade = AsyncMock()
    return pilot


async def test_run_daily_cycle_sorts_side_aware_candidates_by_directional_quality(
    config: AppConfig,
    matchtrader_client: AsyncMock,
) -> None:
    """Short quality should use 1-score before topk slicing in daily-cycle mode."""
    config.scanner.topk = 1
    pilot = _build_pilot(config)
    pilot.scanner.run_pipeline.return_value = [
        _make_signal(instrument="EURUSD", score=0.81, rank=1, side="long"),
        _make_signal(instrument="USDCHF", score=0.12, rank=2, side="short"),
    ]
    pilot.agents.decide.return_value = AgentDecision(
        symbol="USDCHF",
        decision="SELL",
        final_state={"summary": "side-aware short"},
    )

    with patch("src.main.MatchTraderClient", return_value=matchtrader_client):
        await pilot.run_daily_cycle(date_override="2026-03-17")

    pilot.agents.decide.assert_called_once()
    assert pilot.agents.decide.call_args.kwargs["symbol"] == "USDCHF"
    pilot._execute_trade.assert_awaited_once()


async def test_run_daily_cycle_skips_short_signal_when_agent_reverses_to_buy(
    config: AppConfig,
    matchtrader_client: AsyncMock,
) -> None:
    """A short scanner candidate must not open a BUY trade in legacy mode."""
    pilot = _build_pilot(config)
    pilot.scanner.run_pipeline.return_value = [
        _make_signal(instrument="USDCHF", score=0.12, rank=1, side="short"),
    ]
    pilot.agents.decide.return_value = AgentDecision(
        symbol="USDCHF",
        decision="BUY",
        final_state={"summary": "reverse long"},
    )

    with patch("src.main.MatchTraderClient", return_value=matchtrader_client):
        await pilot.run_daily_cycle(date_override="2026-03-17")

    pilot._execute_trade.assert_not_awaited()


async def test_run_daily_cycle_keeps_only_best_side_per_symbol(
    config: AppConfig,
    matchtrader_client: AsyncMock,
) -> None:
    """Legacy mode should not trade both directions for the same symbol."""
    config.scanner.topk = 2
    pilot = _build_pilot(config)
    pilot.scanner.run_pipeline.return_value = [
        _make_signal(instrument="USDCAD", score=0.62, rank=1, side="long"),
        _make_signal(instrument="USDCAD", score=0.62, rank=2, side="short"),
    ]
    pilot.agents.decide.return_value = AgentDecision(
        symbol="USDCAD",
        decision="BUY",
        final_state={"summary": "best side only"},
    )

    with patch("src.main.MatchTraderClient", return_value=matchtrader_client):
        await pilot.run_daily_cycle(date_override="2026-03-17")

    pilot.agents.decide.assert_called_once()
    assert pilot.agents.decide.call_args.kwargs["symbol"] == "USDCAD"
    pilot._execute_trade.assert_awaited_once()


async def test_run_daily_cycle_skips_long_signal_when_agent_reverses_to_sell(
    config: AppConfig,
    matchtrader_client: AsyncMock,
) -> None:
    """A long scanner candidate must not open a SELL trade in legacy mode."""
    pilot = _build_pilot(config)
    pilot.scanner.run_pipeline.return_value = [
        _make_signal(instrument="EURUSD", score=0.81, rank=1, side="long"),
    ]
    pilot.agents.decide.return_value = AgentDecision(
        symbol="EURUSD",
        decision="SELL",
        final_state={"summary": "reverse short"},
    )

    with patch("src.main.MatchTraderClient", return_value=matchtrader_client):
        await pilot.run_daily_cycle(date_override="2026-03-17")

    pilot._execute_trade.assert_not_awaited()
