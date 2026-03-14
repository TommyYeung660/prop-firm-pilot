"""Tests for close reconciler canonical close facts."""

from unittest.mock import MagicMock

from src.decision.close_models import CloseOutcome


def _resolve_pip_size(symbol: str) -> float:
    _ = symbol
    return 0.0001


def test_reconcile_modify_only_keeps_trigger_but_classifies_final_sl_hit() -> None:
    from src.decision.close_reconciler import CloseReconciler

    reconciler = CloseReconciler(pip_size_resolver=_resolve_pip_size)
    pending_outcome = CloseOutcome(
        trigger_source="tactical_exit",
        action_kind="modify_only",
        execution_status="accepted",
        readback_status="verified",
    )
    broker_closed = MagicMock(
        profit=-20.0,
        close_price=1.0980,
        open_price=1.1000,
        volume=0.1,
        close_reason="",
    )

    result = reconciler.reconcile(
        symbol="EURUSD",
        pending_outcome=pending_outcome,
        broker_closed=broker_closed,
        fallback_pnl=0.0,
        execution_meta={"sl_price": 1.0980, "tp_price": 1.1080},
        matched=True,
    )

    assert result.trigger_source == "tactical_exit"
    assert result.action_kind == "modify_only"
    assert result.final_close_reason == "sl_hit"
    assert result.resolution_path == "broker_api"


def test_reconcile_pending_reeval_close_outranks_fallback_pnl_sign() -> None:
    from src.decision.close_reconciler import CloseReconciler

    reconciler = CloseReconciler(pip_size_resolver=_resolve_pip_size)
    pending_outcome = CloseOutcome(
        trigger_source="reeval_close",
        action_kind="full_close",
        execution_status="submitted",
        readback_status="pending_reconcile",
    )

    result = reconciler.reconcile(
        symbol="EURUSD",
        pending_outcome=pending_outcome,
        broker_closed=None,
        fallback_pnl=-15.3,
        execution_meta={"sl_price": 1.0980, "tp_price": 1.1080},
        used_reeval=True,
    )

    assert result.trigger_source == "reeval_close"
    assert result.action_kind == "full_close"
    assert result.final_close_reason == "reeval_close"
    assert result.resolution_path == "reeval_close"


def test_reconcile_external_close_without_pending_outcome_defaults_to_manual_or_broker() -> None:
    from src.decision.close_reconciler import CloseReconciler

    reconciler = CloseReconciler(pip_size_resolver=_resolve_pip_size)

    result = reconciler.reconcile(
        symbol="EURUSD",
        pending_outcome=None,
        broker_closed=None,
        fallback_pnl=12.5,
        execution_meta={"sl_price": 1.0980, "tp_price": 1.1080},
        used_last_known=True,
    )

    assert result.trigger_source == "manual_or_broker"
    assert result.action_kind == "external_detected_close"
    assert result.final_close_reason == "tp_hit"
    assert result.resolution_path == "last_known_profit"


def test_reconcile_best_day_close_outranks_negative_pnl_reinference() -> None:
    from src.decision.close_reconciler import CloseReconciler

    reconciler = CloseReconciler(pip_size_resolver=_resolve_pip_size)
    pending_outcome = CloseOutcome(
        trigger_source="best_day_close",
        action_kind="full_close",
        execution_status="submitted",
        readback_status="pending_reconcile",
    )

    result = reconciler.reconcile(
        symbol="EURUSD",
        pending_outcome=pending_outcome,
        broker_closed=None,
        fallback_pnl=-7.0,
        execution_meta={"sl_price": 1.0980, "tp_price": 1.1080},
        used_best_day=True,
    )

    assert result.trigger_source == "best_day_close"
    assert result.action_kind == "full_close"
    assert result.final_close_reason == "best_day_close"
    assert result.resolution_path == "best_day_close"
