"""
Trade statistics aggregation for optimization feedback loops.

Computes win rates and PnL feedback from DecisionStore and TradeJournal.

Usage:
    rates = compute_win_rates(store, days=14)
    pnl = build_pnl_feedback(store, journal, days=7)
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone

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


def compute_inactive_days(store: DecisionStore, symbols: list[str]) -> dict[str, int]:
    """Compute days since last closed trade for each symbol.

    Args:
        store: DecisionStore instance.
        symbols: List of symbols to check.

    Returns:
        Dict mapping symbol to days since last closed trade.
        Symbols with no trade history get 0 (no penalty for new symbols).
    """
    intents = store.get_closed_intents(days=90)
    last_trade: dict[str, datetime] = {}

    for intent in intents:
        if intent.symbol not in symbols:
            continue
        if intent.created_at is None:
            continue
        if isinstance(intent.created_at, datetime):
            created_at = intent.created_at
        elif isinstance(intent.created_at, str):
            created_at = datetime.fromisoformat(intent.created_at)
        else:
            continue
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if intent.symbol not in last_trade or created_at > last_trade[intent.symbol]:
            last_trade[intent.symbol] = created_at

    now = datetime.now(timezone.utc)
    day_seconds = int(timedelta(days=1).total_seconds())
    inactive_days: dict[str, int] = {}

    for symbol in symbols:
        if symbol not in last_trade:
            inactive_days[symbol] = 0
            continue
        delta = now - last_trade[symbol]
        days = int(delta.total_seconds() // day_seconds)
        inactive_days[symbol] = max(0, days)

    logger.debug("TradeStats: computed inactive days for {} symbols", len(inactive_days))
    return inactive_days
