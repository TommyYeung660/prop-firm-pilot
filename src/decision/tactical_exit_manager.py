"""
Tactical exit orchestration manager.

Adds operational guardrails on top of the pure tactical exit rule engine:
cooldowns, write-budget suppression, and LLM exception escalation markers.

Usage:
    manager = TacticalExitManager(config.tactical.exit)
    evaluation = manager.evaluate_position(snapshot, budget, now)
"""

from dataclasses import dataclass
from datetime import datetime

from src.config import SymbolTacticalOverride, TacticalExitConfig
from src.decision.tactical_exit_rules import (
    TacticalExitDecision,
    TacticalExitSnapshot,
    choose_tactical_exit,
)


@dataclass
class WriteBudgetSnapshot:
    """Small immutable view of broker write-budget state."""

    write_remaining: int
    daily_write_limit: int

    @property
    def critical(self) -> bool:
        """Return True when only a small fraction of write budget remains."""
        if self.daily_write_limit <= 0:
            return False
        return self.write_remaining < int(self.daily_write_limit * 0.15)


@dataclass
class TacticalExitEvaluation:
    """Exit decision after orchestration-level suppressions are applied."""

    decision: TacticalExitDecision
    skip_reason: str = ""
    requires_llm_exception_review: bool = False


class TacticalExitManager:
    """Apply cooldown and write-budget policy on top of pure exit rules."""

    def __init__(
        self,
        config: TacticalExitConfig,
        symbol_overrides: dict[str, SymbolTacticalOverride] | None = None,
    ) -> None:
        self._config = config
        self._symbol_overrides = symbol_overrides or {}

    def _hold_result(
        self,
        *,
        state: str,
        reason: str,
        skip_reason: str,
        requires_llm_exception_review: bool = False,
    ) -> TacticalExitEvaluation:
        """Return a HOLD evaluation with preserved state and skip context."""
        return TacticalExitEvaluation(
            decision=TacticalExitDecision(
                action="HOLD",
                state=state,
                reason=reason,
            ),
            skip_reason=skip_reason,
            requires_llm_exception_review=requires_llm_exception_review,
        )

    def _cooldown_reason(
        self,
        snapshot: TacticalExitSnapshot,
        decision: TacticalExitDecision,
        now: datetime,
    ) -> str:
        """Return the active cooldown reason, if any."""
        if snapshot.last_tactical_exit_at is None:
            return ""

        elapsed_seconds = (now - snapshot.last_tactical_exit_at).total_seconds()

        if decision.action == "REPRICE_TP":
            if elapsed_seconds < self._config.tp_reprice_cooldown_seconds:
                return "tp_reprice_cooldown"
            return ""

        if decision.action in {"MOVE_TO_BREAKEVEN", "TRAIL_SL", "PARTIAL_CLOSE"}:
            if elapsed_seconds < self._config.modify_cooldown_seconds:
                return "modify_cooldown"

        return ""

    @staticmethod
    def _is_budget_critical_action(action: str) -> bool:
        """Return True when an action should bypass low-budget suppression."""
        return action in {"EXIT_NOW", "MOVE_TO_BREAKEVEN"}

    def resolve_exit_config(self, symbol: str) -> TacticalExitConfig:
        """Return exit config with per-symbol overrides applied."""
        override = self._symbol_overrides.get(symbol)
        if override is None:
            return self._config
        data = self._config.model_dump()
        if override.defensive_exit_loss_r is not None:
            data["defensive_exit_loss_r"] = override.defensive_exit_loss_r
        if override.defensive_exit_min_hold_seconds is not None:
            data["defensive_exit_min_hold_seconds"] = override.defensive_exit_min_hold_seconds
        return TacticalExitConfig(**data)

    def evaluate_position(
        self,
        snapshot: TacticalExitSnapshot,
        budget: WriteBudgetSnapshot,
        now: datetime,
        *,
        symbol: str = "",
    ) -> TacticalExitEvaluation:
        """Evaluate one position under tactical exit rules plus operating guards."""
        config = self.resolve_exit_config(symbol) if symbol else self._config
        decision = choose_tactical_exit(snapshot, config)
        requires_llm_exception_review = (
            decision.requires_llm_exception and self._config.use_llm_exception_path
        )

        if decision.action == "HOLD":
            return TacticalExitEvaluation(
                decision=decision,
                requires_llm_exception_review=requires_llm_exception_review,
            )

        cooldown_reason = self._cooldown_reason(snapshot, decision, now)
        if cooldown_reason:
            return self._hold_result(
                state=decision.state,
                reason=cooldown_reason,
                skip_reason=cooldown_reason,
                requires_llm_exception_review=requires_llm_exception_review,
            )

        if budget.critical and not self._is_budget_critical_action(decision.action):
            return self._hold_result(
                state=decision.state,
                reason="write_budget_blocked",
                skip_reason="write_budget_blocked",
                requires_llm_exception_review=requires_llm_exception_review,
            )

        return TacticalExitEvaluation(
            decision=decision,
            requires_llm_exception_review=requires_llm_exception_review,
        )
