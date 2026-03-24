"""
Regression tests for PropFirmPilot.run_daily_cycle().

Focuses on the legacy single-cycle path, which must now respect side-aware
scanner signals the same way the async scheduler path does.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import AppConfig, ExecutionConfig, InstrumentConfig, ScannerConfig
from src.decision.agent_bridge import AgentDecision
from src.main import PropFirmPilot, _run_scheduler
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
        label_version="binary_forward_return_sign_v1",
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

    with patch("src.main.build_broker_client", return_value=matchtrader_client):
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

    with patch("src.main.build_broker_client", return_value=matchtrader_client):
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

    with patch("src.main.build_broker_client", return_value=matchtrader_client):
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

    with patch("src.main.build_broker_client", return_value=matchtrader_client):
        await pilot.run_daily_cycle(date_override="2026-03-17")

    pilot._execute_trade.assert_not_awaited()


async def test_run_daily_cycle_uses_broker_factory(
    config: AppConfig,
    matchtrader_client: AsyncMock,
) -> None:
    """Daily cycle should build broker via factory, not direct MatchTrader constructor."""
    pilot = _build_pilot(config)
    pilot.scanner.run_pipeline.return_value = []

    with patch("src.main.build_broker_client", return_value=matchtrader_client) as mock_factory:
        await pilot.run_daily_cycle(date_override="2026-03-17")

    mock_factory.assert_called_once_with(config, store=None)
    matchtrader_client.login.assert_awaited_once()


async def test_run_monitor_only_uses_broker_factory(
    config: AppConfig,
    matchtrader_client: AsyncMock,
) -> None:
    """Monitor-only path should build broker via factory."""
    pilot = _build_pilot(config)
    pilot.equity_monitor = MagicMock()
    pilot.equity_monitor.start = AsyncMock(return_value=None)

    with patch("src.main.build_broker_client", return_value=matchtrader_client) as mock_factory:
        await pilot.run_monitor_only()

    mock_factory.assert_called_once_with(config, store=None)
    matchtrader_client.login.assert_awaited_once()
    pilot.equity_monitor.start.assert_awaited_once()


async def test_run_scheduler_uses_broker_factory_with_store(config: AppConfig) -> None:
    """Scheduler startup should build broker via factory and pass DecisionStore through."""
    config.execution.broker_backend = "tradelocker"
    config.tradelocker.account_id = "TL-ACC-01"
    store = MagicMock()
    broker_client = AsyncMock()
    broker_client.__aenter__.return_value = broker_client
    broker_client.__aexit__.return_value = False
    broker_client.login = AsyncMock(return_value={"accessToken": "ok"})
    registry = MagicMock()
    registry.tradeable_symbols = ["EURUSD"]
    registry.untradeable_symbols = []
    scheduler = MagicMock()
    scheduler.recover_stale_claims = AsyncMock(return_value=0)
    scheduler.start = AsyncMock(return_value=None)
    scheduler.stop = AsyncMock(return_value=None)
    bot_handler = MagicMock()
    bot_handler.start = AsyncMock(return_value=None)
    bot_handler.stop = AsyncMock(return_value=None)
    alert_service = MagicMock()
    alert_service.send = AsyncMock(return_value=True)
    alert_service.close = AsyncMock(return_value=None)
    operational_metrics = MagicMock()

    class _ImmediateEvent:
        def set(self) -> None:
            return None

        async def wait(self) -> None:
            return None

    with (
        patch("src.main.build_broker_client", return_value=broker_client) as mock_factory,
        patch("src.main.ScannerBridge", return_value=MagicMock()),
        patch("src.main.AgentBridge", return_value=MagicMock()),
        patch("src.main.AlertService", return_value=alert_service),
        patch("src.main.OperationalMetrics", return_value=operational_metrics),
        patch("src.main.TradeJournal", return_value=MagicMock()),
        patch("src.main.MemoryJournal", return_value=MagicMock()),
        patch("src.main.TelegramBotHandler", return_value=bot_handler),
        patch("src.main.asyncio.Event", return_value=_ImmediateEvent()),
        patch("src.decision_store.sqlite_store.DecisionStore", return_value=store),
        patch(
            "src.execution.instrument_registry.InstrumentRegistry.from_broker",
            new=AsyncMock(return_value=registry),
        ),
        patch("src.optimize.optimization_engine.OptimizationEngine", return_value=MagicMock()),
        patch("src.compliance.prop_firm_guard.PropFirmGuard", return_value=MagicMock()),
        patch("src.execution.position_sizer.PositionSizer", return_value=MagicMock()),
        patch("src.execution.engine.ExecutionEngine", return_value=MagicMock()),
        patch("src.scheduler.scheduler.Scheduler", return_value=scheduler),
    ):
        await _run_scheduler(config)

    mock_factory.assert_called_once_with(config, store=store)
    broker_client.login.assert_awaited_once()


async def test_run_daily_cycle_bypasses_agent_when_tradingagents_disabled(
    config: AppConfig,
    matchtrader_client: AsyncMock,
) -> None:
    """With TradingAgents disabled, legacy mode should route directly from scanner side."""
    config.agents.enabled = False
    pilot = _build_pilot(config)
    pilot.scanner.run_pipeline.return_value = [
        _make_signal(instrument="USDCHF", score=0.12, rank=1, side="short"),
    ]

    with patch("src.main.build_broker_client", return_value=matchtrader_client):
        await pilot.run_daily_cycle(date_override="2026-03-17")

    pilot.agents.decide.assert_not_called()
    pilot._execute_trade.assert_awaited_once()
    assert pilot._execute_trade.await_args.kwargs["side"] == "SELL"
