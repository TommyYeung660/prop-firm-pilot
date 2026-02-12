"""
Trade journal — persistent JSONL log of all trading activity.

Appends trade records to a JSONL file for post-analysis,
compliance audit trails, and TradingAgents' reflect_and_remember() feedback.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger


class TradeJournal:
    """Append-only JSONL trade journal.

    Usage:
        journal = TradeJournal("data/trade_journal.jsonl")
        journal.log_trade(trade_record.model_dump())
        journal.log_event("COMPLIANCE_REJECT", {"reason": "daily drawdown"})

        # For TradingAgents feedback
        returns = journal.get_daily_returns("2026-02-12")
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log_trade(self, trade_data: Dict[str, Any]) -> None:
        """Append a trade record to the journal."""
        entry = {
            "type": "TRADE",
            **trade_data,
        }
        self._append(entry)
        logger.debug("Journal: logged trade for {}", trade_data.get("symbol", "?"))

    def log_event(self, event_type: str, details: Dict[str, Any] | None = None) -> None:
        """Log a non-trade event (compliance check, equity alert, etc.)."""
        entry = {
            "type": event_type,
            **(details or {}),
        }
        self._append(entry)

    def log_equity_snapshot(
        self,
        balance: float,
        equity: float,
        daily_pnl: float,
        open_positions: int,
    ) -> None:
        """Log periodic equity snapshot for drawdown analysis."""
        self._append(
            {
                "type": "EQUITY_SNAPSHOT",
                "balance": balance,
                "equity": equity,
                "daily_pnl": daily_pnl,
                "open_positions": open_positions,
            }
        )

    def get_daily_returns(self, date_str: str) -> List[Dict[str, Any]]:
        """Read all closed trades for a specific date.

        Used by TradingAgents' reflect_and_remember() to learn from results.

        Args:
            date_str: Date string prefix to match (e.g. "2026-02-12").

        Returns:
            List of trade records that were closed on the given date.
        """
        results = []
        if not self._path.exists():
            return results

        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Journal: skipping malformed line")
                    continue

                if entry.get("type") != "TRADE":
                    continue
                if entry.get("status") != "CLOSED":
                    continue

                timestamp = entry.get("timestamp", "")
                if timestamp.startswith(date_str):
                    results.append(entry)

        return results

    def get_all_trades(self) -> List[Dict[str, Any]]:
        """Read all trade records from journal."""
        results = []
        if not self._path.exists():
            return results

        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") == "TRADE":
                    results.append(entry)

        return results

    def _append(self, entry: Dict[str, Any]) -> None:
        """Append a JSON line to the journal file."""
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except OSError as e:
            logger.error("Journal: failed to write entry: {}", e)
