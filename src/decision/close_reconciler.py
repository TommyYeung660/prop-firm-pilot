"""
Close reconciler.

Combines pending close outcomes, broker closed-position facts, and fallback
execution metadata into canonical close facts.

Usage:
    reconciler = CloseReconciler(pip_size_resolver=...)
    result = reconciler.reconcile(...)
"""

from collections.abc import Callable
from typing import Any

from loguru import logger

from src.decision.close_models import CloseOutcome, CloseReconciliation

# PnL threshold below which we trust broker/price-match over PnL sign.
# Absorbs small commissions, swaps, and rounding near breakeven.
_PNL_SIGN_OVERRIDE_THRESHOLD = 1.0


class CloseReconciler:
    """Produce canonical close facts from broker and fallback data."""

    def __init__(self, pip_size_resolver: Callable[[str], float]) -> None:
        self._pip_size_resolver = pip_size_resolver

    def reconcile(
        self,
        *,
        symbol: str,
        pending_outcome: CloseOutcome | None,
        broker_closed: Any | None,
        fallback_pnl: float,
        execution_meta: dict[str, Any],
        matched: bool = False,
        used_execution_meta: bool = False,
        used_best_day: bool = False,
        used_reeval: bool = False,
        used_last_known: bool = False,
    ) -> CloseReconciliation:
        """Merge available close facts into one canonical reconciliation result."""
        trigger_source = "manual_or_broker"
        action_kind = "external_detected_close"
        if pending_outcome is not None:
            trigger_source = pending_outcome.trigger_source
            action_kind = pending_outcome.action_kind

        pnl = float(getattr(broker_closed, "profit", 0.0) or fallback_pnl or 0.0)
        close_price = float(getattr(broker_closed, "close_price", 0.0) or 0.0)
        volume = float(getattr(broker_closed, "volume", 0.0) or 0.0)

        final_close_reason = self._resolve_final_close_reason(
            trigger_source=trigger_source,
            action_kind=action_kind,
            close_price=close_price,
            pnl=pnl,
            execution_meta=execution_meta,
            broker_close_reason=str(getattr(broker_closed, "close_reason", "") or ""),
            symbol=symbol,
        )
        resolution_path = self._determine_resolution_path(
            matched=matched,
            used_execution_meta=used_execution_meta,
            used_best_day=used_best_day,
            used_reeval=used_reeval,
            used_last_known=used_last_known,
        )
        payload = {
            "trigger_source": trigger_source,
            "action_kind": action_kind,
            "final_close_reason": final_close_reason,
            "resolution_path": resolution_path,
        }
        return CloseReconciliation(
            trigger_source=trigger_source,
            action_kind=action_kind,
            final_close_reason=final_close_reason,
            resolution_path=resolution_path,
            pnl=pnl,
            close_price=close_price,
            volume=volume,
            journal_payload=payload,
            meta_patch={"close_control": payload},
        )

    def _resolve_final_close_reason(
        self,
        *,
        trigger_source: str,
        action_kind: str,
        close_price: float,
        pnl: float,
        execution_meta: dict[str, Any],
        broker_close_reason: str,
        symbol: str,
    ) -> str:
        """Resolve the canonical final close reason with fixed priority rules."""
        # Priority 1: Explicit triggers (unambiguous, no PnL check needed)
        if trigger_source in {"emergency_close", "best_day_close", "reeval_close"}:
            return trigger_source
        # Priority 2: Tactical exit with reason code (unambiguous)
        if trigger_source == "tactical_exit" and action_kind == "full_close":
            close_control = execution_meta.get("close_control", {})
            if isinstance(close_control, dict):
                reason_code = str(close_control.get("reason_code", "") or "")
                if reason_code:
                    return reason_code
            tactical_reason = str(execution_meta.get("tactical_exit_reason", "") or "")
            if tactical_reason:
                return tactical_reason
            return "tactical_exit"

        # Priority 3: Broker-reported reason (validate against PnL sign)
        broker_reason = broker_close_reason.upper()
        if broker_reason == "TAKE_PROFIT":
            return self._pnl_sign_override("tp_hit", pnl, symbol)
        if broker_reason == "STOP_LOSS":
            return self._pnl_sign_override("sl_hit", pnl, symbol)
        if broker_reason == "STOP_OUT":
            return "broker_stopout"

        # Priority 4: Close price vs SL/TP prices (validate against PnL sign)
        sl_price = float(
            execution_meta.get("breakeven_sl")
            or execution_meta.get("trailing_sl")
            or execution_meta.get("sl_price")
            or 0.0
        )
        tp_price = float(execution_meta.get("dynamic_tp") or execution_meta.get("tp_price") or 0.0)
        tolerance = self._pip_size_resolver(symbol) * 3

        if close_price != 0.0:
            if tp_price and abs(close_price - tp_price) <= tolerance:
                return self._pnl_sign_override("tp_hit", pnl, symbol)
            if sl_price and abs(close_price - sl_price) <= tolerance:
                return self._pnl_sign_override("sl_hit", pnl, symbol)

        # Priority 5: Fallback — PnL sign (consistent by definition)
        if pnl > 0:
            return "tp_hit"
        if pnl < 0:
            return "sl_hit"
        return "manual_close"

    @staticmethod
    def _pnl_sign_override(candidate_reason: str, pnl: float, symbol: str) -> str:
        """Override candidate reason when PnL sign clearly contradicts it.

        Only triggers when |PnL| exceeds the threshold, so tiny commissions
        or swap charges near zero don't flip the reason.
        """
        if abs(pnl) <= _PNL_SIGN_OVERRIDE_THRESHOLD:
            return candidate_reason
        if candidate_reason == "tp_hit" and pnl < 0:
            logger.warning(
                "Close reason override: {} PnL={:.2f} contradicts tp_hit → sl_hit",
                symbol,
                pnl,
            )
            return "sl_hit"
        if candidate_reason == "sl_hit" and pnl > 0:
            logger.warning(
                "Close reason override: {} PnL={:.2f} contradicts sl_hit → tp_hit",
                symbol,
                pnl,
            )
            return "tp_hit"
        return candidate_reason

    @staticmethod
    def _determine_resolution_path(
        *,
        matched: bool,
        used_execution_meta: bool,
        used_best_day: bool,
        used_reeval: bool,
        used_last_known: bool,
    ) -> str:
        """Return the data source path that supplied close facts."""
        if matched:
            return "broker_api"
        if used_best_day:
            return "best_day_close"
        if used_reeval:
            return "reeval_close"
        if used_execution_meta:
            return "execution_meta"
        if used_last_known:
            return "last_known_profit"
        return "unknown"
