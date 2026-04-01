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


def test_reconcile_tactical_full_close_preserves_reason_code_over_positive_fallback_pnl() -> None:
    from src.decision.close_reconciler import CloseReconciler

    reconciler = CloseReconciler(pip_size_resolver=_resolve_pip_size)
    pending_outcome = CloseOutcome(
        trigger_source="tactical_exit",
        action_kind="full_close",
        execution_status="submitted",
        readback_status="pending_reconcile",
    )

    result = reconciler.reconcile(
        symbol="CADJPY",
        pending_outcome=pending_outcome,
        broker_closed=None,
        fallback_pnl=2.64,
        execution_meta={
            "tactical_exit_reason": "severe_tactical_reversal",
            "close_control": {
                "trigger_source": "tactical_exit",
                "action_kind": "full_close",
                "reason_code": "severe_tactical_reversal",
            },
        },
        used_last_known=True,
    )

    assert result.trigger_source == "tactical_exit"
    assert result.action_kind == "full_close"
    assert result.final_close_reason == "severe_tactical_reversal"
    assert result.resolution_path == "last_known_profit"


# ── PnL sign vs close reason consistency tests ─────────────────────────────


def test_broker_take_profit_but_negative_pnl_resolves_to_sl_hit() -> None:
    """GBPJPY production bug: broker says TAKE_PROFIT but PnL=-248.4 → sl_hit."""
    from src.decision.close_reconciler import CloseReconciler

    reconciler = CloseReconciler(pip_size_resolver=_resolve_pip_size)
    broker_closed = MagicMock(
        profit=-248.4,
        close_price=210.502,
        open_price=210.907,
        volume=0.92,
        close_reason="TAKE_PROFIT",
    )

    result = reconciler.reconcile(
        symbol="GBPJPY",
        pending_outcome=None,
        broker_closed=broker_closed,
        fallback_pnl=0.0,
        execution_meta={"sl_price": 210.407, "tp_price": 211.907},
        matched=True,
    )

    assert result.final_close_reason == "sl_hit"
    assert result.resolution_path == "broker_api"


def test_broker_stop_loss_but_positive_pnl_resolves_to_tp_hit() -> None:
    """Mirror case: broker says STOP_LOSS but PnL=+150 → tp_hit."""
    from src.decision.close_reconciler import CloseReconciler

    reconciler = CloseReconciler(pip_size_resolver=_resolve_pip_size)
    broker_closed = MagicMock(
        profit=150.0,
        close_price=1.1080,
        open_price=1.1000,
        volume=0.5,
        close_reason="STOP_LOSS",
    )

    result = reconciler.reconcile(
        symbol="EURUSD",
        pending_outcome=None,
        broker_closed=broker_closed,
        fallback_pnl=0.0,
        execution_meta={"sl_price": 1.0920, "tp_price": 1.1080},
        matched=True,
    )

    assert result.final_close_reason == "tp_hit"


def test_close_price_near_tp_but_negative_pnl_resolves_to_sl_hit() -> None:
    """Close price within tolerance of TP but PnL is negative → sl_hit."""
    from src.decision.close_reconciler import CloseReconciler

    reconciler = CloseReconciler(pip_size_resolver=_resolve_pip_size)
    broker_closed = MagicMock(
        profit=-50.0,
        close_price=1.10805,  # within 3-pip tolerance of TP 1.1081
        open_price=1.1000,
        volume=0.5,
        close_reason="",
    )

    result = reconciler.reconcile(
        symbol="EURUSD",
        pending_outcome=None,
        broker_closed=broker_closed,
        fallback_pnl=0.0,
        execution_meta={"sl_price": 1.0920, "tp_price": 1.1081},
        matched=True,
    )

    assert result.final_close_reason == "sl_hit"


def test_small_pnl_near_zero_does_not_override_broker_reason() -> None:
    """PnL very close to zero (e.g., -$0.50 from commission) trusts broker reason."""
    from src.decision.close_reconciler import CloseReconciler

    reconciler = CloseReconciler(pip_size_resolver=_resolve_pip_size)
    broker_closed = MagicMock(
        profit=-0.50,
        close_price=1.1080,
        open_price=1.1000,
        volume=0.1,
        close_reason="TAKE_PROFIT",
    )

    result = reconciler.reconcile(
        symbol="EURUSD",
        pending_outcome=None,
        broker_closed=broker_closed,
        fallback_pnl=0.0,
        execution_meta={"sl_price": 1.0920, "tp_price": 1.1080},
        matched=True,
    )

    assert result.final_close_reason == "tp_hit"


def test_pnl_sign_override_does_not_affect_tactical_exit_reasons() -> None:
    """Tactical exit reasons (e.g., initial_risk_structure_failure) are preserved."""
    from src.decision.close_reconciler import CloseReconciler

    reconciler = CloseReconciler(pip_size_resolver=_resolve_pip_size)
    pending_outcome = CloseOutcome(
        trigger_source="tactical_exit",
        action_kind="full_close",
        execution_status="submitted",
        readback_status="pending_reconcile",
    )
    broker_closed = MagicMock(
        profit=-130.55,
        close_price=109.086,
        open_price=109.586,
        volume=1.06,
        close_reason="STOP_LOSS",
    )

    result = reconciler.reconcile(
        symbol="AUDJPY",
        pending_outcome=pending_outcome,
        broker_closed=broker_closed,
        fallback_pnl=0.0,
        execution_meta={
            "sl_price": 109.086,
            "tp_price": 110.586,
            "tactical_exit_reason": "initial_risk_structure_failure",
            "close_control": {"reason_code": "initial_risk_structure_failure"},
        },
        matched=True,
    )

    assert result.final_close_reason == "initial_risk_structure_failure"
