"""
Optimization engine — aggregates stats and writes optimization state.

Collects recent trade performance metrics, computes thresholds,
and persists a daily optimization_state.json for runtime gating.

Usage:
    engine = OptimizationEngine(store, journal, "data/optimization_state.json")
    state = engine.refresh_state()
"""

from pathlib import Path

from loguru import logger

from src.decision_store.sqlite_store import DecisionStore
from src.monitor.trade_journal import TradeJournal
from src.optimize.optimization_state import (
    ABTestState,
    OptimizationState,
    load_state,
    save_state,
)
from src.optimize.thresholds import compute_thresholds
from src.optimize.trade_stats import build_pnl_feedback, compute_inactive_days, compute_win_rates

# ── Exceptions ──────────────────────────────────────────────────────────────


class OptimizationEngineError(Exception):
    """Base exception for OptimizationEngine operations."""


# ── Engine ──────────────────────────────────────────────────────────────────


class OptimizationEngine:
    """Aggregate trade stats and write optimization state.

    Usage:
        engine = OptimizationEngine(store, journal, "data/optimization_state.json")
        state = engine.refresh_state()
    """

    def __init__(
        self,
        store: DecisionStore,
        journal: TradeJournal | None,
        state_path: str | Path,
        pnl_days: int = 7,
        win_days: int = 14,
        ab_model_a: str = "rightcodes/gpt-5.4",
        ab_model_b: str = "volcengine/kimi-k2.5",
        ab_ratio: float = 0.5,
    ) -> None:
        self._store = store
        self._journal = journal
        self._state_path = Path(state_path)
        self._pnl_days = pnl_days
        self._win_days = win_days
        self._ab_model_a = ab_model_a
        self._ab_model_b = ab_model_b
        self._ab_ratio = ab_ratio

    def refresh_state(self) -> OptimizationState:
        """Recompute and persist optimization state.

        Returns:
            OptimizationState containing latest metrics.
        """
        win_rates = compute_win_rates(self._store, days=self._win_days)
        symbol_rates = {k: v for k, v in win_rates.items() if k != "global"}

        # H3: Compute inactivity for threshold decay
        inactive = compute_inactive_days(self._store, list(symbol_rates.keys()))

        thresholds = compute_thresholds(
            global_win_rate=win_rates.get("global", 0.0),
            symbol_win_rates=symbol_rates,
            inactive_days=inactive,
        )

        state = OptimizationState(
            pnl_lookback_days=self._pnl_days,
            winrate_lookback_days=self._win_days,
            global_thresholds=thresholds["global"],
            symbol_thresholds={k: v for k, v in thresholds.items() if k != "global"},
            feedback_pnl=build_pnl_feedback(self._store, self._journal, days=self._pnl_days),
        )

        # v1.3.9: Preserve existing AB counts from persisted state
        existing = load_state(self._state_path)
        existing_ab = existing.ab_test if existing else ABTestState()

        state.ab_test = ABTestState(
            model_a=self._ab_model_a,
            model_b=self._ab_model_b,
            ratio=self._ab_ratio,
            counts=existing_ab.counts,
            pnl_by_model=existing_ab.pnl_by_model,
        )

        save_state(self._state_path, state)
        logger.info("OptimizationEngine: state updated at {}", state.generated_at)
        return state
