"""Tests for close control plane execution behavior."""

from unittest.mock import AsyncMock

import pytest

from src.decision.close_models import CloseIntent
from src.execution.matchtrader_client import OrderResult


def _normalize_price(symbol: str, price: float | None) -> float | None:
    _ = symbol
    if price is None:
        return None
    return round(price, 5)


def _resolve_precision(symbol: str) -> int | None:
    _ = symbol
    return 5


@pytest.mark.asyncio
async def test_execute_modify_only_returns_verify_failed_when_readback_mismatch() -> None:
    from src.decision.close_control_plane import CloseControlPlane

    broker = AsyncMock()
    broker.modify_position.return_value = OrderResult(
        success=True,
        position_id="POS-1",
        message="OK",
    )
    broker.verify_sl_tp.return_value = False

    control = CloseControlPlane(
        matchtrader=broker,
        normalize_price=_normalize_price,
        price_precision_resolver=_resolve_precision,
    )

    outcome = await control.execute(
        CloseIntent(
            trigger_source="tactical_exit",
            action_kind="modify_only",
            position_id="POS-1",
            intent_id="INT-1",
            symbol="EURUSD",
            side="BUY",
            requested_sl=1.101234,
            requested_tp=1.109876,
            reason_code="breakeven_threshold_reached",
        )
    )

    assert outcome.execution_status == "verify_failed"
    assert outcome.readback_status == "mismatch"
    assert outcome.broker_result["success"] is True


@pytest.mark.asyncio
async def test_execute_partial_close_returns_submitted_pending_reconcile() -> None:
    from src.decision.close_control_plane import CloseControlPlane

    broker = AsyncMock()
    broker.close_position.return_value = OrderResult(
        success=True,
        position_id="POS-1",
        message="OK",
    )

    control = CloseControlPlane(
        matchtrader=broker,
        normalize_price=_normalize_price,
        price_precision_resolver=_resolve_precision,
    )

    outcome = await control.execute(
        CloseIntent(
            trigger_source="reduce_exposure",
            action_kind="partial_close",
            position_id="POS-1",
            intent_id="INT-1",
            symbol="EURUSD",
            side="BUY",
            requested_volume=0.05,
            reason_code="drawdown_reduce_exposure",
        )
    )

    assert outcome.execution_status == "submitted"
    assert outcome.readback_status == "pending_reconcile"
    assert outcome.final_close_reason == ""


@pytest.mark.asyncio
async def test_duplicate_full_close_is_suppressed_before_second_broker_write() -> None:
    from src.decision.close_control_plane import CloseControlPlane

    broker = AsyncMock()
    broker.close_position.return_value = OrderResult(
        success=True,
        position_id="POS-1",
        message="OK",
    )

    control = CloseControlPlane(
        matchtrader=broker,
        normalize_price=_normalize_price,
        price_precision_resolver=_resolve_precision,
    )

    intent = CloseIntent(
        trigger_source="reeval_close",
        action_kind="full_close",
        position_id="POS-1",
        intent_id="INT-1",
        symbol="EURUSD",
        side="BUY",
        requested_volume=0.10,
        reason_code="reverse_signal_close",
    )

    first = await control.execute(intent)
    second = await control.execute(intent)

    assert first.execution_status == "submitted"
    assert second.execution_status == "skipped"
    assert second.readback_status == "not_needed"
    assert broker.close_position.await_count == 1
