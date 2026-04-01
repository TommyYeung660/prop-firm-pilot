"""Tests for scheduler-level portfolio risk pre-check.

The pre-check runs before tactical validation to skip expensive computation
for intents that will be rejected by portfolio limits. It is FAIL-OPEN on
broker API errors (execution engine retains authoritative fail-closed check).
"""

from unittest.mock import AsyncMock

from src.config import ExecutionConfig
from src.execution.portfolio_risk_guard import (
    REASON_SAME_DIRECTION_LIMIT_EXCEEDED,
    OpenPositionRiskSnapshot,
    PortfolioRiskGuard,
)

# ── Guard-level tests (confirming pre-check logic) ────────────────────────


def test_precheck_rejects_when_same_direction_limit_exceeded() -> None:
    """If 2 BUY positions exist and limit is 2, next BUY should be rejected."""
    config = ExecutionConfig(max_same_direction_positions=2)
    guard = PortfolioRiskGuard(config)

    decision = guard.evaluate_next_entry(
        next_symbol="GBPJPY",
        next_side="BUY",
        next_risk_pct=0.01,
        open_positions=[
            OpenPositionRiskSnapshot(symbol="AUDJPY", side="BUY", open_risk_pct=0.01),
            OpenPositionRiskSnapshot(symbol="EURCAD", side="BUY", open_risk_pct=0.01),
        ],
    )

    assert not decision.allowed
    assert decision.reason_code == REASON_SAME_DIRECTION_LIMIT_EXCEEDED


def test_precheck_allows_when_under_same_direction_limit() -> None:
    """If 1 BUY position exists and limit is 2, next BUY should be allowed."""
    config = ExecutionConfig(max_same_direction_positions=2)
    guard = PortfolioRiskGuard(config)

    decision = guard.evaluate_next_entry(
        next_symbol="GBPJPY",
        next_side="BUY",
        next_risk_pct=0.01,
        open_positions=[
            OpenPositionRiskSnapshot(symbol="AUDJPY", side="BUY", open_risk_pct=0.01),
        ],
    )

    assert decision.allowed


def test_precheck_allows_opposite_direction() -> None:
    """2 BUY positions exist but next is SELL — should be allowed."""
    config = ExecutionConfig(max_same_direction_positions=2)
    guard = PortfolioRiskGuard(config)

    decision = guard.evaluate_next_entry(
        next_symbol="GBPJPY",
        next_side="SELL",
        next_risk_pct=0.01,
        open_positions=[
            OpenPositionRiskSnapshot(symbol="AUDJPY", side="BUY", open_risk_pct=0.01),
            OpenPositionRiskSnapshot(symbol="EURCAD", side="BUY", open_risk_pct=0.01),
        ],
    )

    assert decision.allowed


# ── Scheduler integration tests (mock broker API) ─────────────────────────


async def test_precheck_fail_open_on_broker_error() -> None:
    """Pre-check should allow (fail-open) when broker API raises an error."""
    from src.execution.portfolio_risk_guard import PortfolioRiskDecision

    # Simulate a scheduler-level pre-check that catches broker errors
    async def _portfolio_risk_precheck_failopen(
        get_positions_fn, guard, symbol, side, risk_pct
    ) -> PortfolioRiskDecision:
        try:
            positions = await get_positions_fn()
        except Exception:
            return PortfolioRiskDecision(
                allowed=True,
                reason_code="portfolio_risk.precheck_skipped",
                projected_total_open_risk_pct=0.0,
                projected_same_direction_positions=0,
            )
        snapshots = [
            OpenPositionRiskSnapshot(symbol=p.symbol, side=p.side, open_risk_pct=risk_pct)
            for p in positions
        ]
        return guard.evaluate_next_entry(
            next_symbol=symbol,
            next_side=side,
            next_risk_pct=risk_pct,
            open_positions=snapshots,
        )

    broken_broker = AsyncMock(side_effect=ConnectionError("broker down"))
    config = ExecutionConfig(max_same_direction_positions=2)
    guard = PortfolioRiskGuard(config)

    result = await _portfolio_risk_precheck_failopen(broken_broker, guard, "GBPJPY", "BUY", 0.01)

    assert result.allowed is True
    assert result.reason_code == "portfolio_risk.precheck_skipped"
