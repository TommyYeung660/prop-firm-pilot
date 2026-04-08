"""Tests for close control plane execution behavior."""

from unittest.mock import AsyncMock

import pytest

from src.decision.close_models import CloseIntent
from src.execution.broker_models import BrokerOrderResult
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
        verify_retry_delay_seconds=0.01,
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


@pytest.mark.asyncio
async def test_close_control_plane_broker_protocol_paths_are_preserved() -> None:
    from src.decision.close_control_plane import CloseControlPlane

    broker = AsyncMock()
    broker.modify_position.return_value = BrokerOrderResult(
        success=True,
        position_id="POS-MOD",
        message="OK",
        raw_response={},
    )
    broker.verify_sl_tp.return_value = True
    broker.close_position.return_value = BrokerOrderResult(
        success=True,
        position_id="POS-CLOSE",
        message="OK",
        raw_response={},
    )

    control = CloseControlPlane(
        broker=broker,
        normalize_price=_normalize_price,
        price_precision_resolver=_resolve_precision,
        verify_retry_delay_seconds=0.01,
    )

    modify_outcome = await control.execute(
        CloseIntent(
            trigger_source="tactical_exit",
            action_kind="modify_only",
            position_id="POS-MOD",
            intent_id="INT-MOD",
            symbol="EURUSD",
            side="BUY",
            requested_sl=1.10111,
            requested_tp=1.10999,
            reason_code="breakeven_threshold_reached",
        )
    )
    close_outcome = await control.execute(
        CloseIntent(
            trigger_source="reeval_close",
            action_kind="full_close",
            position_id="POS-CLOSE",
            intent_id="INT-CLOSE",
            symbol="EURUSD",
            side="BUY",
            requested_volume=0.10,
            reason_code="reverse_signal_close",
        )
    )

    assert modify_outcome.execution_status == "accepted"
    assert close_outcome.execution_status == "submitted"
    broker.modify_position.assert_awaited_once()
    broker.verify_sl_tp.assert_awaited_once()
    broker.close_position.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_control_plane_positional_matchtrader_call_remains_supported() -> None:
    from src.decision.close_control_plane import CloseControlPlane

    broker = AsyncMock()
    broker.modify_position.return_value = OrderResult(
        success=True,
        position_id="POS-LEGACY",
        message="OK",
    )
    broker.verify_sl_tp.return_value = True

    control = CloseControlPlane(
        broker, _normalize_price, _resolve_precision, verify_retry_delay_seconds=0.01
    )

    outcome = await control.execute(
        CloseIntent(
            trigger_source="tactical_exit",
            action_kind="modify_only",
            position_id="POS-LEGACY",
            intent_id="INT-LEGACY",
            symbol="EURUSD",
            side="BUY",
            requested_sl=1.10123,
            requested_tp=1.10987,
            reason_code="breakeven_threshold_reached",
        )
    )

    assert outcome.execution_status == "accepted"
    broker.modify_position.assert_awaited_once()
    broker.verify_sl_tp.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_modify_retries_verify_on_first_mismatch() -> None:
    """If first verify_sl_tp fails but second succeeds, outcome should be accepted."""
    from src.decision.close_control_plane import CloseControlPlane

    broker = AsyncMock()
    broker.modify_position.return_value = OrderResult(
        success=True, position_id="POS-R1", message="OK"
    )
    broker.verify_sl_tp = AsyncMock(side_effect=[False, True])

    control = CloseControlPlane(
        broker=broker,
        normalize_price=_normalize_price,
        price_precision_resolver=_resolve_precision,
        verify_retry_delay_seconds=0.01,
        verify_max_retries=1,
    )

    outcome = await control.execute(
        CloseIntent(
            trigger_source="tactical_exit",
            action_kind="modify_only",
            position_id="POS-R1",
            intent_id="INT-R1",
            symbol="CADJPY",
            side="BUY",
            requested_sl=95.500,
            requested_tp=97.000,
            reason_code="breakeven_threshold_reached",
        )
    )

    assert outcome.execution_status == "accepted"
    assert outcome.readback_status == "verified"
    assert broker.verify_sl_tp.call_count == 2


@pytest.mark.asyncio
async def test_execute_modify_returns_mismatch_after_all_retries_exhausted() -> None:
    """If all verify attempts fail, outcome should be verify_failed."""
    from src.decision.close_control_plane import CloseControlPlane

    broker = AsyncMock()
    broker.modify_position.return_value = OrderResult(
        success=True, position_id="POS-R2", message="OK"
    )
    broker.verify_sl_tp = AsyncMock(return_value=False)

    control = CloseControlPlane(
        broker=broker,
        normalize_price=_normalize_price,
        price_precision_resolver=_resolve_precision,
        verify_retry_delay_seconds=0.01,
        verify_max_retries=1,
    )

    outcome = await control.execute(
        CloseIntent(
            trigger_source="tactical_exit",
            action_kind="modify_only",
            position_id="POS-R2",
            intent_id="INT-R2",
            symbol="CADJPY",
            side="BUY",
            requested_sl=95.500,
            requested_tp=97.000,
            reason_code="breakeven_threshold_reached",
        )
    )

    assert outcome.execution_status == "verify_failed"
    assert outcome.readback_status == "mismatch"
    assert broker.verify_sl_tp.call_count == 2  # initial + 1 retry


@pytest.mark.asyncio
async def test_execute_modify_no_retry_when_max_retries_zero() -> None:
    """With verify_max_retries=0, only one verify attempt is made."""
    from src.decision.close_control_plane import CloseControlPlane

    broker = AsyncMock()
    broker.modify_position.return_value = OrderResult(
        success=True, position_id="POS-R3", message="OK"
    )
    broker.verify_sl_tp = AsyncMock(return_value=False)

    control = CloseControlPlane(
        broker=broker,
        normalize_price=_normalize_price,
        price_precision_resolver=_resolve_precision,
        verify_retry_delay_seconds=0.01,
        verify_max_retries=0,
    )

    outcome = await control.execute(
        CloseIntent(
            trigger_source="tactical_exit",
            action_kind="modify_only",
            position_id="POS-R3",
            intent_id="INT-R3",
            symbol="EURUSD",
            side="SELL",
            requested_sl=1.10500,
            requested_tp=1.09500,
            reason_code="trailing_stop_update",
        )
    )

    assert outcome.execution_status == "verify_failed"
    assert broker.verify_sl_tp.call_count == 1
