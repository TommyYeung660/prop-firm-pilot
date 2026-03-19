"""
Rule-based diagnostics for entry-funnel ablation snapshots.

Aggregates mode-level snapshots into a comparable A/B/C/D report with
economic, funnel, churn, and recommendation sections.

Usage:
    result = analyze_ablation(snapshots)
"""

from typing import Any

MODE_ORDER: list[tuple[str, str, str]] = [
    ("A", "scanner_tactical", "scanner -> tactical -> execution"),
    ("B", "scanner_llm_tactical", "scanner -> LLM(confirm/veto) -> tactical -> execution"),
    ("C", "tactical_only", "tactical-only -> execution"),
    ("D", "no_trade", "no-trade / admission-only baseline"),
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Return float-like values while ignoring invalid inputs."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Return int-like values while ignoring invalid inputs."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    """Return a rounded ratio when the denominator is positive."""
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 4)


def _average(values: list[float]) -> float | None:
    """Return a rounded arithmetic mean when values are present."""
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _group_rows_by_mode(snapshots: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Bucket raw snapshots by entry funnel mode."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for snapshot in snapshots:
        mode = str(snapshot.get("entry_funnel_mode", "")).strip()
        if not mode:
            continue
        grouped.setdefault(mode, []).append(snapshot)
    return grouped


def _aggregate_mode(mode: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate raw snapshots for a single entry-funnel mode."""
    scanner_candidates = sum(_safe_int(row.get("scanner_candidates")) for row in rows)
    intents_created = sum(_safe_int(row.get("intents_created")) for row in rows)
    opened_count = sum(_safe_int(row.get("opened_count")) for row in rows)
    llm_vetoes = sum(_safe_int(row.get("llm_vetoes")) for row in rows)
    llm_cancels = sum(_safe_int(row.get("llm_cancels")) for row in rows)
    tactical_waits = sum(_safe_int(row.get("tactical_waits")) for row in rows)
    tactical_expires = sum(_safe_int(row.get("tactical_expires")) for row in rows)
    no_trade_count = sum(_safe_int(row.get("no_trade_count")) for row in rows)
    net_pnl = round(sum(_safe_float(row.get("net_pnl")) for row in rows), 4)

    profit_factor_values = [
        _safe_float(row.get("profit_factor"))
        for row in rows
        if isinstance(row.get("profit_factor"), (int, float))
        and not isinstance(row.get("profit_factor"), bool)
    ]
    max_drawdown_values = [
        _safe_float(row.get("max_drawdown"))
        for row in rows
        if isinstance(row.get("max_drawdown"), (int, float))
        and not isinstance(row.get("max_drawdown"), bool)
    ]

    return {
        "mode": mode,
        "days": len(rows),
        "economic": {
            "net_pnl": net_pnl,
            "opened_count": opened_count,
            "expectancy_per_opened_trade": _safe_ratio(net_pnl, opened_count) or 0.0,
            "profit_factor": _average(profit_factor_values),
            "max_drawdown": max(max_drawdown_values) if max_drawdown_values else None,
        },
        "funnel": {
            "scanner_candidates": scanner_candidates,
            "intents_created": intents_created,
            "opened_count": opened_count,
            "intent_creation_rate": _safe_ratio(intents_created, scanner_candidates),
            "opened_trade_rate": _safe_ratio(opened_count, scanner_candidates),
            "intent_to_open_rate": _safe_ratio(opened_count, intents_created),
        },
        "churn": {
            "llm_vetoes": llm_vetoes,
            "llm_cancels": llm_cancels,
            "tactical_waits": tactical_waits,
            "tactical_expires": tactical_expires,
            "no_trade_count": no_trade_count,
            "llm_veto_rate": _safe_ratio(llm_vetoes, scanner_candidates),
            "tactical_wait_then_expire_rate": _safe_ratio(tactical_expires, tactical_waits),
        },
    }


def _profit_factor(summary: dict[str, Any], fallback: float = 0.0) -> float:
    """Extract a numeric profit factor from an economic summary."""
    value = summary.get("profit_factor")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return fallback


def _beats_no_trade(mode_summary: dict[str, Any], baseline_summary: dict[str, Any]) -> bool:
    """Return whether a traded mode shows positive edge versus no-trade baseline."""
    return (
        _safe_float(mode_summary.get("net_pnl")) > _safe_float(baseline_summary.get("net_pnl"))
        and _safe_float(mode_summary.get("expectancy_per_opened_trade")) > _safe_float(
            baseline_summary.get("expectancy_per_opened_trade")
        )
        and _profit_factor(mode_summary) > max(1.0, _profit_factor(baseline_summary, fallback=1.0))
    )


def _llm_stably_beats_scanner(
    llm_economic: dict[str, Any],
    llm_churn: dict[str, Any],
    scanner_economic: dict[str, Any],
    scanner_churn: dict[str, Any],
) -> bool:
    """Return whether pipeline B shows stable incremental value over pipeline A."""
    llm_wait_expire = _safe_float(llm_churn.get("tactical_wait_then_expire_rate"), default=1.0)
    scanner_wait_expire = _safe_float(
        scanner_churn.get("tactical_wait_then_expire_rate"),
        default=1.0,
    )
    return (
        _safe_float(llm_economic.get("net_pnl")) > _safe_float(scanner_economic.get("net_pnl"))
        and _safe_float(llm_economic.get("expectancy_per_opened_trade"))
        >= _safe_float(scanner_economic.get("expectancy_per_opened_trade"))
        and _profit_factor(llm_economic) >= _profit_factor(scanner_economic)
        and llm_wait_expire <= scanner_wait_expire
    )


def _tactical_only_beats_scanner(
    tactical_economic: dict[str, Any],
    scanner_economic: dict[str, Any],
) -> bool:
    """Return whether tactical-only has evidence to replace the scanner baseline."""
    return (
        _safe_float(tactical_economic.get("net_pnl")) > _safe_float(scanner_economic.get("net_pnl"))
        and _safe_float(tactical_economic.get("expectancy_per_opened_trade"))
        >= _safe_float(scanner_economic.get("expectancy_per_opened_trade"))
        and _profit_factor(tactical_economic) >= _profit_factor(scanner_economic)
    )


def _build_report_sections(
    grouped_rows: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build economic, funnel, and churn sections keyed by A/B/C/D labels."""
    economic_summary: dict[str, Any] = {}
    funnel_summary: dict[str, Any] = {}
    churn_summary: dict[str, Any] = {}

    for code, mode, _ in MODE_ORDER:
        aggregated = _aggregate_mode(mode, grouped_rows.get(mode, []))
        economic_summary[code] = {"mode": mode, **aggregated["economic"]}
        funnel_summary[code] = {"mode": mode, **aggregated["funnel"]}
        churn_summary[code] = {"mode": mode, **aggregated["churn"]}

    return economic_summary, funnel_summary, churn_summary


def _available_mode_labels(grouped_rows: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Return available A/B/C/D labels in canonical order."""
    labels: list[str] = []
    for code, mode, _ in MODE_ORDER:
        if grouped_rows.get(mode):
            labels.append(code)
    return labels


def _recommendation(
    economic_summary: dict[str, Any],
    churn_summary: dict[str, Any],
    available_modes: list[str],
) -> tuple[str, str]:
    """Return the deterministic recommendation and short reason."""
    if len(available_modes) < len(MODE_ORDER):
        return (
            "insufficient_ablation_data",
            "A/B/C/D mode coverage is incomplete in the current snapshot window.",
        )

    a_economic = economic_summary["A"]
    b_economic = economic_summary["B"]
    c_economic = economic_summary["C"]
    d_economic = economic_summary["D"]

    a_beats_d = _beats_no_trade(a_economic, d_economic)
    b_beats_d = _beats_no_trade(b_economic, d_economic)
    c_beats_d = _beats_no_trade(c_economic, d_economic)

    if not a_beats_d and not b_beats_d and not c_beats_d:
        return (
            "return_to_no_trade_shadow_mode",
            "A/B/C do not show positive edge over the no-trade baseline.",
        )
    if not a_beats_d:
        return (
            "downgrade_scanner_to_shadow_only",
            "Pipeline A does not beat the no-trade baseline.",
        )
    if not _llm_stably_beats_scanner(
        b_economic,
        churn_summary["B"],
        a_economic,
        churn_summary["A"],
    ):
        return (
            "downgrade_llm_to_confirm_veto",
            "Pipeline B does not show stable incremental value over pipeline A.",
        )
    if not _tactical_only_beats_scanner(c_economic, a_economic):
        return (
            "reject_tactical_only_as_default",
            "Pipeline C does not show evidence that it can replace scanner admission.",
        )
    return (
        "keep_collecting_validation_data",
        "Current ablation ordering does not justify a stronger architecture change yet.",
    )


def analyze_ablation(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate mode snapshots into a deterministic ablation report."""
    grouped_rows = _group_rows_by_mode(snapshots)
    available_modes = _available_mode_labels(grouped_rows)
    economic_summary, funnel_summary, churn_summary = _build_report_sections(grouped_rows)
    recommendation, recommendation_reason = _recommendation(
        economic_summary,
        churn_summary,
        available_modes,
    )

    return {
        "mode_labels": {
            code: {"mode": mode, "description": description}
            for code, mode, description in MODE_ORDER
        },
        "available_modes": available_modes,
        "economic_summary": economic_summary,
        "funnel_summary": funnel_summary,
        "churn_summary": churn_summary,
        "recommendation": recommendation,
        "recommendation_reason": recommendation_reason,
    }
