"""Tests for close-domain typed models."""

from src.decision.close_models import CloseIntent, CloseOutcome, CloseReconciliation


def test_close_intent_defaults_for_modify_action() -> None:
    intent = CloseIntent(
        trigger_source="tactical_exit",
        action_kind="modify_only",
        position_id="POS-1",
        intent_id="INT-1",
        symbol="EURUSD",
        side="BUY",
        reason_code="atr_trailing_stop_improved",
    )

    assert intent.requested_volume is None
    assert intent.requested_sl is None
    assert intent.requested_tp is None
    assert intent.source_context == {}


def test_close_outcome_carries_execution_and_readback_status() -> None:
    outcome = CloseOutcome(
        trigger_source="tactical_exit",
        action_kind="modify_only",
        execution_status="verify_failed",
        readback_status="mismatch",
    )

    assert outcome.final_close_reason == ""
    assert outcome.meta_patch == {}


def test_close_reconciliation_defaults_to_empty_patches() -> None:
    result = CloseReconciliation(
        trigger_source="manual_or_broker",
        action_kind="external_detected_close",
        final_close_reason="manual_close",
        resolution_path="last_known_profit_fallback",
        pnl=12.5,
        close_price=1.108,
        volume=0.1,
    )

    assert result.journal_payload == {}
    assert result.meta_patch == {}
