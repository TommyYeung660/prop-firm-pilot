"""
Trade statistics aggregation for optimization feedback loops.

Computes win rates and PnL feedback from DecisionStore and TradeJournal.

Usage:
    rates = compute_win_rates(store, days=14)
    pnl = build_pnl_feedback(store, journal, days=7)
"""

from collections import defaultdict
from loguru import logger

from src.decision_store.sqlite_store import DecisionStore
from src.monitor.trade_journal import TradeJournal

# ── Exceptions ──────────────────────────────────────────────────────────────


class TradeStatsError(Exception):
    """Base exception for trade stats operations."""


# ── Public API ──────────────────────────────────────────────────────────────


def compute_win_rates(store: DecisionStore, days: int = 14) -> dict[str, float]:
    """Compute global and per-symbol win rates over a lookback window.

    Args:
        store: DecisionStore instance.
        days: Lookback window in days.

    Returns:
        Dict with "global" and per-symbol win rates.
    """
    intents = store.get_closed_intents(days=days)
    wins: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)

    for intent in intents:
        pnl = intent.realized_pnl or 0.0
        totals[intent.symbol] += 1
        if pnl > 0:
            wins[intent.symbol] += 1

    result: dict[str, float] = {}
    total_global = sum(totals.values())
    win_global = sum(wins.values())
    result["global"] = win_global / total_global if total_global > 0 else 0.0

    for symbol, total in totals.items():
        result[symbol] = wins[symbol] / total if total > 0 else 0.0

    logger.debug(
        "TradeStats: computed win rates (days={}, global={:.2f})",
        days,
        result["global"],
    )
    return result


def build_pnl_feedback(
    store: DecisionStore,
    journal: TradeJournal | None,
    days: int = 7,
) -> dict[str, float]:
    """Aggregate realized PnL by symbol from store and optional journal.

    Args:
        store: DecisionStore instance.
        journal: TradeJournal instance (optional).
        days: Lookback window in days.

    Returns:
        Dict mapping symbol to total PnL.
    """
    pnl_by_symbol: dict[str, float] = defaultdict(float)

    intents = store.get_closed_intents(days=days)
    for intent in intents:
        pnl_by_symbol[intent.symbol] += float(intent.realized_pnl or 0.0)

    if journal is not None:
        for entry in journal.get_closed_trades(days=days):
            symbol = entry.get("symbol", "")
            if not symbol:
                continue
            pnl_by_symbol[symbol] += float(entry.get("pnl", 0.0))

    logger.debug("TradeStats: aggregated pnl for {} symbols", len(pnl_by_symbol))
    return dict(pnl_by_symbol)
