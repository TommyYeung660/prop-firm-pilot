"""Tests for execution-side portfolio risk guard primitives."""

from src.config import ExecutionConfig
from src.execution.portfolio_risk_guard import OpenPositionRiskSnapshot, PortfolioRiskGuard


def test_portfolio_risk_guard_blocks_when_total_open_risk_exceeds_budget() -> None:
    config = ExecutionConfig(max_total_open_risk_pct=0.02, reserve_risk_for_open_positions=True)
    guard = PortfolioRiskGuard(config)

    decision = guard.evaluate_next_entry(
        next_symbol="EURUSD",
        next_side="BUY",
        next_risk_pct=0.01,
        open_positions=[
            OpenPositionRiskSnapshot(symbol="GBPUSD", side="BUY", open_risk_pct=0.012),
        ],
    )

    assert decision.allowed is False
    assert decision.reason_code == "portfolio_risk.total_open_risk_exceeded"


def test_portfolio_risk_guard_blocks_when_same_direction_concentration_exceeds_limit() -> None:
    config = ExecutionConfig(max_same_direction_positions=2)
    guard = PortfolioRiskGuard(config)

    decision = guard.evaluate_next_entry(
        next_symbol="EURUSD",
        next_side="BUY",
        next_risk_pct=0.005,
        open_positions=[
            OpenPositionRiskSnapshot(symbol="GBPUSD", side="BUY", open_risk_pct=0.005),
            OpenPositionRiskSnapshot(symbol="AUDUSD", side="BUY", open_risk_pct=0.005),
        ],
    )

    assert decision.allowed is False
    assert decision.reason_code == "portfolio_risk.same_direction_limit_exceeded"


def test_portfolio_risk_guard_blocks_when_currency_exposure_exceeds_limit() -> None:
    config = ExecutionConfig(max_currency_exposure_per_ccy=2)
    guard = PortfolioRiskGuard(config)

    decision = guard.evaluate_next_entry(
        next_symbol="USDCHF",
        next_side="SELL",
        next_risk_pct=0.005,
        open_positions=[
            OpenPositionRiskSnapshot(symbol="EURUSD", side="BUY", open_risk_pct=0.005),
            OpenPositionRiskSnapshot(symbol="GBPUSD", side="SELL", open_risk_pct=0.005),
        ],
    )

    assert decision.allowed is False
    assert decision.reason_code == "portfolio_risk.currency_exposure_exceeded"


def test_portfolio_risk_guard_reject_reason_codes_are_deterministic() -> None:
    config = ExecutionConfig(max_total_open_risk_pct=0.02, reserve_risk_for_open_positions=True)
    guard = PortfolioRiskGuard(config)

    open_positions = [
        OpenPositionRiskSnapshot(symbol="GBPUSD", side="BUY", open_risk_pct=0.012),
    ]
    first = guard.evaluate_next_entry(
        next_symbol="EURUSD",
        next_side="BUY",
        next_risk_pct=0.01,
        open_positions=open_positions,
    )
    second = guard.evaluate_next_entry(
        next_symbol="EURUSD",
        next_side="BUY",
        next_risk_pct=0.01,
        open_positions=open_positions,
    )

    assert first.allowed is False
    assert second.allowed is False
    assert first.reason_code == "portfolio_risk.total_open_risk_exceeded"
    assert second.reason_code == "portfolio_risk.total_open_risk_exceeded"


def test_portfolio_risk_guard_treats_zero_same_direction_limit_as_fail_closed() -> None:
    config = ExecutionConfig(max_same_direction_positions=0)
    guard = PortfolioRiskGuard(config)

    decision = guard.evaluate_next_entry(
        next_symbol="EURUSD",
        next_side="BUY",
        next_risk_pct=0.005,
        open_positions=[],
    )

    assert decision.allowed is False
    assert decision.reason_code == "portfolio_risk.same_direction_limit_exceeded"


def test_portfolio_risk_guard_treats_zero_currency_exposure_limit_as_fail_closed() -> None:
    config = ExecutionConfig(max_currency_exposure_per_ccy=0)
    guard = PortfolioRiskGuard(config)

    decision = guard.evaluate_next_entry(
        next_symbol="EURUSD",
        next_side="BUY",
        next_risk_pct=0.005,
        open_positions=[],
    )

    assert decision.allowed is False
    assert decision.reason_code == "portfolio_risk.currency_exposure_exceeded"
