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
        "group_count": len(groups),
        "total_verdicts": len(events),
        "groups": groups,
    }
