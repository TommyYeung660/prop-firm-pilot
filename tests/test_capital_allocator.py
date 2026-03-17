"""Tests for execution-side bounded capital allocation."""

import pytest

from src.config import ExecutionConfig
from src.execution.capital_allocator import BoundedCapitalAllocator


@pytest.fixture
def config() -> ExecutionConfig:
    """Execution config used for bounded capital allocation tests."""
    return ExecutionConfig(
        max_positions=5,
        default_risk_pct=0.009,
        max_risk_pct=0.02,
    )


def test_high_confidence_sparse_portfolio_uses_max_risk_pct(config: ExecutionConfig) -> None:
    """High-confidence sparse portfolios should use the bounded max risk."""
    allocator = BoundedCapitalAllocator(config)

    decision = allocator.allocate_entry_risk(open_positions=0, scanner_confidence="high")

    assert decision.effective_risk_pct == pytest.approx(0.02)
    assert decision.bounded_cap_pct == pytest.approx(0.02)
    assert decision.uplift_applied is True


def test_medium_confidence_only_uses_half_uplift_headroom(config: ExecutionConfig) -> None:
    """Medium confidence should only consume half of the available uplift headroom."""
    allocator = BoundedCapitalAllocator(config)

    decision = allocator.allocate_entry_risk(open_positions=0, scanner_confidence="medium")

    assert decision.effective_risk_pct == pytest.approx(0.0145)
    assert decision.uplift_applied is True


def test_low_confidence_keeps_default_risk(config: ExecutionConfig) -> None:
    """Low confidence should not receive capital utilization uplift."""
    allocator = BoundedCapitalAllocator(config)

    decision = allocator.allocate_entry_risk(open_positions=0, scanner_confidence="low")

    assert decision.effective_risk_pct == pytest.approx(config.default_risk_pct)
    assert decision.uplift_applied is False


def test_near_full_portfolio_returns_to_default_risk(config: ExecutionConfig) -> None:
    """When the portfolio is nearly full, bounded allocation should collapse to default risk."""
    allocator = BoundedCapitalAllocator(config)

    decision = allocator.allocate_entry_risk(open_positions=4, scanner_confidence="high")

    assert decision.effective_risk_pct == pytest.approx(config.default_risk_pct)
    assert decision.bounded_cap_pct == pytest.approx(config.default_risk_pct)
    assert decision.uplift_applied is False
