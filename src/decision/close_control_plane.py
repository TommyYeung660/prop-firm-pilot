"""
Close control plane.

Routes close-domain intents through a single broker execution path and returns
typed close outcomes for later reconciliation.

Usage:
    control = CloseControlPlane(matchtrader, normalize_price, resolve_precision)
    control = CloseControlPlane(
        broker=broker,
        normalize_price=normalize_price,
        price_precision_resolver=resolve_precision,
    )
    outcome = await control.execute(intent)
"""

import asyncio
from collections.abc import Callable

from loguru import logger

from src.decision.close_models import CloseIntent, CloseOutcome
from src.execution.broker_protocol import BrokerClientProtocol


class CloseControlPlane:
    """Execute close intents through a single broker-facing interface."""

    def __init__(
        self,
        matchtrader: BrokerClientProtocol | None = None,
        normalize_price: Callable[[str, float | None], float | None] | None = None,
        price_precision_resolver: Callable[[str], int | None] | None = None,
        *,
        broker: BrokerClientProtocol | None = None,
        verify_retry_delay_seconds: float = 0.5,
        verify_max_retries: int = 1,
    ) -> None:
        selected_broker = broker if broker is not None else matchtrader
        if selected_broker is None:
            raise ValueError("CloseControlPlane requires a broker client")
        if normalize_price is None:
            raise ValueError("CloseControlPlane requires normalize_price")
        if price_precision_resolver is None:
            raise ValueError("CloseControlPlane requires price_precision_resolver")
        self._matchtrader = selected_broker
        self._normalize_price = normalize_price
        self._price_precision_resolver = price_precision_resolver
        self._verify_retry_delay_seconds = verify_retry_delay_seconds
        self._verify_max_retries = verify_max_retries
        self._pending_close_actions: dict[str, CloseOutcome] = {}

    async def execute(self, intent: CloseIntent) -> CloseOutcome:
        """Execute one close intent and return its typed outcome."""
        if intent.action_kind == "modify_only":
            return await self._execute_modify_only(intent)
        if intent.action_kind in {"partial_close", "full_close"}:
            return await self._execute_close(intent)
        raise ValueError(f"Unsupported action_kind: {intent.action_kind}")

    async def _execute_modify_only(self, intent: CloseIntent) -> CloseOutcome:
        expected_sl = self._normalize_price(intent.symbol, intent.requested_sl)
        expected_tp = self._normalize_price(intent.symbol, intent.requested_tp)
        result = await self._matchtrader.modify_position(
            position_id=intent.position_id,
            symbol=intent.symbol,
            side=intent.side,
            volume=intent.requested_volume,
            sl=expected_sl,
            tp=expected_tp,
        )
        broker_result = result.model_dump() if hasattr(result, "model_dump") else {}
        if not result.success:
            return CloseOutcome(
                trigger_source=intent.trigger_source,
                action_kind=intent.action_kind,
                execution_status="skipped",
                readback_status="not_needed",
                broker_result=broker_result,
            )

        precision = self._price_precision_resolver(intent.symbol)
        max_attempts = 1 + self._verify_max_retries
        for attempt in range(max_attempts):
            delay = self._verify_retry_delay_seconds * (attempt + 1)
            await asyncio.sleep(delay)
            verified = await self._matchtrader.verify_sl_tp(
                position_id=intent.position_id,
                expected_sl=expected_sl,
                expected_tp=expected_tp,
                price_precision=precision,
            )
            if verified:
                return CloseOutcome(
                    trigger_source=intent.trigger_source,
                    action_kind=intent.action_kind,
                    execution_status="accepted",
                    readback_status="verified",
                    broker_result=broker_result,
                )
            if attempt < max_attempts - 1:
                logger.info(
                    "Modify verify attempt {}/{} failed for {}, retrying in {:.1f}s",
                    attempt + 1,
                    max_attempts,
                    intent.position_id,
                    self._verify_retry_delay_seconds * (attempt + 2),
                )

        return CloseOutcome(
            trigger_source=intent.trigger_source,
            action_kind=intent.action_kind,
            execution_status="verify_failed",
            readback_status="mismatch",
            broker_result=broker_result,
        )

    async def _execute_close(self, intent: CloseIntent) -> CloseOutcome:
        pending = self._pending_close_actions.get(intent.position_id)
        if pending is not None:
            return CloseOutcome(
                trigger_source=intent.trigger_source,
                action_kind=intent.action_kind,
                execution_status="skipped",
                readback_status="not_needed",
                broker_result={},
            )

        result = await self._matchtrader.close_position(
            position_id=intent.position_id,
            symbol=intent.symbol,
            side=intent.side,
            volume=intent.requested_volume,
        )
        broker_result = result.model_dump() if hasattr(result, "model_dump") else {}
        if not result.success:
            return CloseOutcome(
                trigger_source=intent.trigger_source,
                action_kind=intent.action_kind,
                execution_status="skipped",
                readback_status="not_needed",
                broker_result=broker_result,
            )

        outcome = CloseOutcome(
            trigger_source=intent.trigger_source,
            action_kind=intent.action_kind,
            execution_status="submitted",
            readback_status="pending_reconcile",
            broker_result=broker_result,
        )
        self._pending_close_actions[intent.position_id] = outcome
        return outcome
