"""
Tactical entry calibration snapshot aggregation.

Builds daily symbol/session/regime statistics from journaled tactical
entry verdicts so tactical thresholds can be calibrated offline.

Usage:
    snapshot = build_daily_entry_calibration_snapshot(journal, "2026-03-14")
"""

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from src.monitor.trade_journal import TradeJournal


def _safe_rate(numerator: int, denominator: int) -> float:
    """Return a stable rounded rate for small daily groups."""
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _safe_count(value: Any, default: int = 0) -> int:
    """Normalize snapshot counters to non-negative integers."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(value, 0)
    return default


def _safe_pnl(value: Any) -> float | None:
    """Normalize journal PnL values while ignoring invalid payloads."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _closed_trade_pnls(journal: TradeJournal, date_str: str) -> list[float]:
    """Return realized PnL values from TRADE_CLOSED events for one day."""
    events = sorted(
        journal.get_events("TRADE_CLOSED", date_str=date_str),
        key=lambda event: str(event.get("timestamp", "")),
    )
    pnls: list[float] = []
    for event in events:
        pnl = _safe_pnl(event.get("pnl"))
        if pnl is not None:
            pnls.append(pnl)
    return pnls


def _closed_trade_attribution(
    journal: TradeJournal,
    date_str: str,
) -> tuple[dict[str, int], dict[str, int], int]:
    """Return close-path attribution counts from TRADE_CLOSED events for one day."""
    events = sorted(
        journal.get_events("TRADE_CLOSED", date_str=date_str),
        key=lambda event: str(event.get("timestamp", "")),
    )
    trigger_source_counts: defaultdict[str, int] = defaultdict(int)
    close_reason_counts: defaultdict[str, int] = defaultdict(int)
    tactical_exit_close_count = 0

    for event in events:
        trigger_source = str(event.get("trigger_source", "") or "")
        final_close_reason = str(
            event.get("final_close_reason", event.get("reason", event.get("exit_reason", ""))) or ""
        )
        if trigger_source:
            trigger_source_counts[trigger_source] += 1
            if trigger_source == "tactical_exit":
                tactical_exit_close_count += 1
        if final_close_reason:
            close_reason_counts[final_close_reason] += 1

    return (
        dict(trigger_source_counts),
        dict(close_reason_counts),
        tactical_exit_close_count,
    )


def _profit_factor_from_pnls(pnls: list[float]) -> float:
    """Approximate profit factor from daily realized close events."""
    gross_profit = sum(pnl for pnl in pnls if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
    if gross_loss > 0:
        return round(gross_profit / gross_loss, 4)
    if gross_profit > 0:
        return round(gross_profit, 4)
    return 1.0


def _max_drawdown_from_pnls(pnls: list[float]) -> float:
    """Approximate realized max drawdown from cumulative close-event PnL."""
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in pnls:
        cumulative += pnl
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    return round(max_drawdown, 4)


def _latest_metrics_snapshot(journal: TradeJournal, date_str: str) -> dict[str, Any]:
    """Return the latest METRICS_SNAPSHOT event for the date if present."""
    events = journal.get_events("METRICS_SNAPSHOT", date_str=date_str)
    if not events:
        return {}
    ordered_events = sorted(events, key=lambda event: str(event.get("timestamp", "")))
    latest = ordered_events[-1]
    return latest if isinstance(latest, dict) else {}


def build_daily_entry_calibration_snapshot(
    journal: TradeJournal,
    date_str: str,
    metrics_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate daily tactical entry verdicts by symbol, session, and regime."""
    events = journal.get_events("TACTICAL_RESULT", date_str=date_str)
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}

    for event in events:
        context = event.get("context", {}) or {}
        provenance = event.get("provenance", {}) or {}
        symbol = str(event.get("symbol", ""))
        session_label = str(context.get("session_label", "unknown"))
        regime_label = str(context.get("regime_label", "unknown"))
        key = (symbol, session_label, regime_label)
        group = grouped.setdefault(
            key,
            {
                "symbol": symbol,
                "session_label": session_label,
                "regime_label": regime_label,
                "total_verdicts": 0,
                "pass_count": 0,
                "wait_count": 0,
                "degrade_count": 0,
                "timeout_count": 0,
                "cancel_count": 0,
                "rest_fallback_count": 0,
                "mixed_source_count": 0,
                "reason_counts": defaultdict(int),
                "failed_hard_gate_counts": defaultdict(int),
            },
        )

        group["total_verdicts"] += 1
        resolution = str(event.get("resolution", ""))
        if resolution == "EXECUTE_NOW":
            group["pass_count"] += 1
        elif resolution == "RETRY_PENDING":
            group["wait_count"] += 1
        elif resolution == "EXECUTE_DEGRADED":
            group["degrade_count"] += 1
        elif resolution == "EXPIRE_TIMEOUT":
            group["timeout_count"] += 1
        elif resolution == "SKIP_CANCEL":
            group["cancel_count"] += 1

        data_source = str(provenance.get("data_source", ""))
        if data_source == "rest_fallback":
            group["rest_fallback_count"] += 1
        elif data_source == "mixed":
            group["mixed_source_count"] += 1

        reason_code = str(event.get("summary_reason_code", ""))
        if reason_code:
            group["reason_counts"][reason_code] += 1

        failed_hard_gate_codes = event.get("failed_hard_gate_reason_codes", [])
        if isinstance(failed_hard_gate_codes, list):
            for code in failed_hard_gate_codes:
                if isinstance(code, str) and code:
                    group["failed_hard_gate_counts"][code] += 1

    groups: list[dict[str, Any]] = []
    for _, group in sorted(grouped.items()):
        total = int(group["total_verdicts"])
        reason_counts = dict(group.pop("reason_counts"))
        failed_hard_gate_counts = dict(group.pop("failed_hard_gate_counts"))
        top_reason_codes = [
            {"reason_code": reason_code, "count": count}
            for reason_code, count in sorted(
                reason_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:5]
        ]
        top_failed_hard_gates = [
            {"reason_code": reason_code, "count": count}
            for reason_code, count in sorted(
                failed_hard_gate_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:5]
        ]
        groups.append(
            {
                **group,
                "pass_rate": _safe_rate(group["pass_count"], total),
                "wait_rate": _safe_rate(group["wait_count"], total),
                "degrade_rate": _safe_rate(group["degrade_count"], total),
                "timeout_rate": _safe_rate(group["timeout_count"], total),
                "cancel_rate": _safe_rate(group["cancel_count"], total),
                "rest_fallback_ratio": _safe_rate(group["rest_fallback_count"], total),
                "mixed_source_ratio": _safe_rate(group["mixed_source_count"], total),
                "top_reason_codes": top_reason_codes,
                "failed_hard_gate_counts": failed_hard_gate_counts,
                "top_failed_hard_gates": top_failed_hard_gates,
            }
        )

    metrics_snapshot = metrics_snapshot or _latest_metrics_snapshot(journal, date_str)
    entry_funnel = metrics_snapshot.get("entry_funnel", {})
    if not isinstance(entry_funnel, dict):
        entry_funnel = {}

    scanner_candidates = _safe_count(entry_funnel.get("scanner_candidates"))
    intents_created = _safe_count(
        entry_funnel.get("intents_created"),
        default=len(journal.get_events("INTENT_CREATED", date_str=date_str)),
    )
    llm_vetoes = _safe_count(entry_funnel.get("llm_vetoes"))
    llm_cancels = _safe_count(entry_funnel.get("llm_cancels"))
    tactical_waits = _safe_count(entry_funnel.get("tactical_waits"))
    tactical_expires = _safe_count(entry_funnel.get("tactical_expires"))
    no_trade_count = _safe_count(entry_funnel.get("no_trade_count"))
    no_trade_reasons = entry_funnel.get("no_trade_reasons", {})
    if not isinstance(no_trade_reasons, dict):
        no_trade_reasons = {}
    opened_count = len(journal.get_events("TRADE_OPENED", date_str=date_str))
    closed_trade_pnls = _closed_trade_pnls(journal, date_str)
    close_trigger_source_counts, close_reason_counts, tactical_exit_close_count = (
        _closed_trade_attribution(journal, date_str)
    )
    degrade_to_exec_count = sum(
        1
        for event in events
        if str(event.get("resolution", "")) == "EXECUTE_DEGRADED"
    )

    return {
        "date": date_str,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entry_funnel_mode": str(metrics_snapshot.get("entry_funnel_mode", "unknown")),
        "scanner_candidates": scanner_candidates,
        "intents_created": intents_created,
        "opened_count": opened_count,
        "llm_vetoes": llm_vetoes,
        "llm_cancels": llm_cancels,
        "tactical_waits": tactical_waits,
        "tactical_expires": tactical_expires,
        "no_trade_count": no_trade_count,
        "no_trade_reasons": no_trade_reasons,
        "llm_veto_rate": _safe_rate(llm_vetoes, scanner_candidates),
        "net_pnl": round(sum(closed_trade_pnls), 4),
        "profit_factor": _profit_factor_from_pnls(closed_trade_pnls),
        "max_drawdown": _max_drawdown_from_pnls(closed_trade_pnls),
        "degrade_to_exec_count": degrade_to_exec_count,
        "tactical_exit_close_count": tactical_exit_close_count,
        "close_trigger_source_counts": close_trigger_source_counts,
        "close_reason_counts": close_reason_counts,
        "group_count": len(groups),
        "total_verdicts": len(events),
        "groups": groups,
    }
