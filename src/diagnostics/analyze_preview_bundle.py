"""
Preview bundle diagnostics for tactical gate incident triage.

Parses a production log bundle and emits structured diagnostics so the same
incident analysis can be rerun deterministically.

Usage:
    python -m src.diagnostics.analyze_preview_bundle --bundle <bundle_path> --format json
    python -m src.diagnostics.analyze_preview_bundle --bundle <bundle_path> --format markdown
"""

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.version import get_release_tag

WEBSOCKET_FAILURE_RE = re.compile(
    r"EODHDFXWebSocketClient: connection failed \((?P<reason>.+?)\), reconnecting"
)
REST_FALLBACK_RE = re.compile(
    r"MarketDataHub: REST fallback for (?P<symbol>[A-Z]+)\s+"
    r"(?P<timeframe>[0-9a-zA-Z]+).*?"
    r"latest_rest_bar_age_by_close_sec=(?P<age>[0-9.]+)"
)


def _safe_ratio(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return round(num / den, 4)


def _percentile(sorted_values: list[float], p: float) -> float | None:
    if not sorted_values:
        return None
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]
    idx = (len(sorted_values) - 1) * (p / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    w = idx - lo
    return sorted_values[lo] * (1.0 - w) + sorted_values[hi] * w


def _bucket_stale_age(age_sec: float) -> str:
    if age_sec < 600:
        return "<10m"
    if age_sec < 3600:
        return "10m-1h"
    if age_sec < 21600:
        return "1h-6h"
    if age_sec < 43200:
        return "6h-12h"
    return ">=12h"


def _empty_log_analysis(log_path: Path | None) -> dict[str, Any]:
    return {
        "available": False,
        "path": str(log_path) if log_path is not None else None,
        "lines_scanned": 0,
        "websocket_failure_counts": {"total": 0, "by_reason": {}},
        "rest_fallback_counts": {
            "total": 0,
            "by_symbol": {},
            "by_timeframe": {},
        },
        "stale_age_distribution": {
            "count": 0,
            "min_sec": None,
            "p50_sec": None,
            "p90_sec": None,
            "p99_sec": None,
            "max_sec": None,
            "bucket_counts": {},
        },
        "market_data_quote_unavailable_log_count": 0,
    }


def _extract_source(
    event: dict[str, Any], provenance: dict[str, Any], keys: list[str]
) -> str:
    for key in keys:
        value = event.get(key) or provenance.get(key)
        if value is None:
            continue
        source = str(value).strip()
        if source:
            return source
    return ""


def _summarize_source_usage(raw_counts: Counter[str], total_slots: int) -> dict[str, Any]:
    known_total = int(sum(raw_counts.values()))
    websocket_cache = int(raw_counts.get("websocket_cache", 0))
    rest_fallback = int(raw_counts.get("rest_fallback", 0))
    other_known = known_total - websocket_cache - rest_fallback
    missing = max(total_slots - known_total, 0)
    return {
        "websocket_cache": websocket_cache,
        "rest_fallback": rest_fallback,
        "other_known": other_known,
        "missing": missing,
        "known_total": known_total,
        "total_slots": total_slots,
        "websocket_to_rest_ratio": _safe_ratio(websocket_cache, rest_fallback),
    }


def _choose_main_log(log_dir: Path) -> Path | None:
    if not log_dir.exists():
        return None
    release_logs = sorted(log_dir.glob(f"*{get_release_tag()}.log"))
    if release_logs:
        return release_logs[-1]
    all_logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    if all_logs:
        return all_logs[-1]
    return None


def _analyze_log(log_path: Path | None) -> dict[str, Any]:
    websocket_reasons = Counter()
    rest_fallback_by_symbol = Counter()
    rest_fallback_by_timeframe = Counter()
    stale_age_values: list[float] = []
    quote_unavailable_log_lines = 0
    lines_scanned = 0

    if log_path is None or not log_path.exists() or not log_path.is_file():
        return _empty_log_analysis(log_path)

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            lines_scanned += 1
            line = raw.strip()
            if not line:
                continue

            ws_match = WEBSOCKET_FAILURE_RE.search(line)
            if ws_match:
                websocket_reasons[ws_match.group("reason")] += 1

            fallback_match = REST_FALLBACK_RE.search(line)
            if fallback_match:
                symbol = fallback_match.group("symbol")
                timeframe = fallback_match.group("timeframe")
                age = float(fallback_match.group("age"))
                rest_fallback_by_symbol[symbol] += 1
                rest_fallback_by_timeframe[timeframe] += 1
                stale_age_values.append(age)

            if "market_data.quote_unavailable" in line:
                quote_unavailable_log_lines += 1

    stale_sorted = sorted(stale_age_values)
    bucket_counts = Counter(_bucket_stale_age(age) for age in stale_sorted)
    stale_stats = {
        "count": len(stale_sorted),
        "min_sec": stale_sorted[0] if stale_sorted else None,
        "p50_sec": _percentile(stale_sorted, 50),
        "p90_sec": _percentile(stale_sorted, 90),
        "p99_sec": _percentile(stale_sorted, 99),
        "max_sec": stale_sorted[-1] if stale_sorted else None,
        "bucket_counts": dict(bucket_counts),
    }

    return {
        "available": True,
        "path": str(log_path),
        "lines_scanned": lines_scanned,
        "websocket_failure_counts": {
            "total": int(sum(websocket_reasons.values())),
            "by_reason": dict(websocket_reasons),
        },
        "rest_fallback_counts": {
            "total": int(sum(rest_fallback_by_symbol.values())),
            "by_symbol": dict(rest_fallback_by_symbol),
            "by_timeframe": dict(rest_fallback_by_timeframe),
        },
        "stale_age_distribution": stale_stats,
        "market_data_quote_unavailable_log_count": quote_unavailable_log_lines,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
    return events


def _analyze_trade_journal(journal_path: Path) -> dict[str, Any]:
    events = _load_jsonl(journal_path)
    tactical_events = [e for e in events if e.get("type") == "TACTICAL_RESULT"]
    summary_reason_counts = Counter()
    failed_hard_gate_reason_counts = Counter()
    data_source_counts = Counter()
    quote_source_counts = Counter()
    bars_5m_source_counts = Counter()
    bars_1h_source_counts = Counter()
    quote_unavailable_event_count = 0

    for event in tactical_events:
        reason_code = str(event.get("summary_reason_code", "")).strip()
        if reason_code:
            summary_reason_counts[reason_code] += 1

        provenance = event.get("provenance", {})
        if not isinstance(provenance, dict):
            provenance = {}

        data_source = str(provenance.get("data_source", "")).strip()
        quote_source = _extract_source(event, provenance, ["quote_source"])
        bars_5m_source = _extract_source(
            event,
            provenance,
            ["bars_5m_source", "bars_5min_source"],
        )
        bars_1h_source = _extract_source(event, provenance, ["bars_1h_source"])
        if data_source:
            data_source_counts[data_source] += 1
        if quote_source:
            quote_source_counts[quote_source] += 1
        if bars_5m_source:
            bars_5m_source_counts[bars_5m_source] += 1
        if bars_1h_source:
            bars_1h_source_counts[bars_1h_source] += 1

        failed_codes = event.get("failed_hard_gate_reason_codes")
        if isinstance(failed_codes, list):
            for code in failed_codes:
                failed_code = str(code).strip()
                if failed_code:
                    failed_hard_gate_reason_counts[failed_code] += 1
        else:
            hard_gates = event.get("hard_gates", [])
            if isinstance(hard_gates, list):
                for gate in hard_gates:
                    if not isinstance(gate, dict):
                        continue
                    if gate.get("passed") is True:
                        continue
                    failed_code = str(gate.get("reason_code", "")).strip()
                    if failed_code:
                        failed_hard_gate_reason_counts[failed_code] += 1

    for event in events:
        if "market_data.quote_unavailable" in json.dumps(event, ensure_ascii=False):
            quote_unavailable_event_count += 1

    component_source_counts = quote_source_counts + bars_5m_source_counts + bars_1h_source_counts

    return {
        "available": journal_path.exists(),
        "path": str(journal_path),
        "total_events": len(events),
        "tactical_result_events": len(tactical_events),
        "tactical_reason_code_distribution": dict(summary_reason_counts),
        "failed_hard_gate_reason_codes": dict(failed_hard_gate_reason_counts),
        "provenance_counts": {
            "data_source": dict(data_source_counts),
            "quote_source": dict(quote_source_counts),
            "bars_5m_source": dict(bars_5m_source_counts),
            "bars_1h_source": dict(bars_1h_source_counts),
            "websocket_cache_vs_rest_fallback": {
                "data_source_view": _summarize_source_usage(
                    data_source_counts,
                    len(tactical_events),
                ),
                "component_source_view": {
                    "overall": _summarize_source_usage(
                        component_source_counts,
                        len(tactical_events) * 3,
                    ),
                    "quote_source": _summarize_source_usage(
                        quote_source_counts,
                        len(tactical_events),
                    ),
                    "bars_5m_source": _summarize_source_usage(
                        bars_5m_source_counts,
                        len(tactical_events),
                    ),
                    "bars_1h_source": _summarize_source_usage(
                        bars_1h_source_counts,
                        len(tactical_events),
                    ),
                },
            },
        },
        "market_data_quote_unavailable_journal_count": quote_unavailable_event_count,
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    log = summary["log_analysis"]
    journal = summary["journal_analysis"]
    websocket_vs_rest = journal["provenance_counts"]["websocket_cache_vs_rest_fallback"]
    component_source_view = websocket_vs_rest["component_source_view"]
    lines: list[str] = []
    lines.append("# Preview Bundle Diagnostics")
    lines.append("")
    lines.append(f"- Bundle: `{summary['bundle_path']}`")
    lines.append(f"- Generated at (UTC): `{summary['generated_at_utc']}`")
    lines.append("")
    lines.append("## Websocket Failure Counts")
    ws_counts = log["websocket_failure_counts"]
    lines.append(f"- Total failures: **{ws_counts['total']}**")
    for reason, count in ws_counts["by_reason"].items():
        lines.append(f"- {reason}: {count}")
    lines.append("")
    lines.append("## REST Fallback Counts")
    fallback = log["rest_fallback_counts"]
    lines.append(f"- Total fallback logs: **{fallback['total']}**")
    lines.append(
        f"- By timeframe: `{json.dumps(fallback['by_timeframe'], ensure_ascii=False)}`"
    )
    lines.append(f"- By symbol: `{json.dumps(fallback['by_symbol'], ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Stale-Age Distribution")
    stale = log["stale_age_distribution"]
    lines.append(
        "- Count / min / p50 / p90 / p99 / max (sec): "
        f"`{stale['count']}` / `{stale['min_sec']}` / `{stale['p50_sec']}` / "
        f"`{stale['p90_sec']}` / `{stale['p99_sec']}` / `{stale['max_sec']}`"
    )
    lines.append(f"- Buckets: `{json.dumps(stale['bucket_counts'], ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Quote Unavailable")
    lines.append(
        f"- Log count: `{log['market_data_quote_unavailable_log_count']}`, "
        f"Journal count: `{journal['market_data_quote_unavailable_journal_count']}`"
    )
    lines.append("")
    lines.append("## Tactical Reason-Code Distribution")
    lines.append(
        f"- `{json.dumps(journal['tactical_reason_code_distribution'], ensure_ascii=False)}`"
    )
    lines.append("")
    lines.append("## Failed Hard-Gate Reason Codes")
    lines.append(
        f"- `{json.dumps(journal['failed_hard_gate_reason_codes'], ensure_ascii=False)}`"
    )
    lines.append("")
    lines.append("## Provenance (websocket_cache vs rest_fallback)")
    lines.append(
        f"- Data source counts: "
        f"`{json.dumps(journal['provenance_counts']['data_source'], ensure_ascii=False)}`"
    )
    lines.append(
        f"- Data-source view: "
        f"`{json.dumps(websocket_vs_rest['data_source_view'], ensure_ascii=False)}`"
    )
    lines.append(
        f"- Component-source overall: "
        f"`{json.dumps(component_source_view['overall'], ensure_ascii=False)}`"
    )
    lines.append("")
    return "\n".join(lines)


def build_summary(bundle_path: Path) -> dict[str, Any]:
    log_dir = bundle_path / "raw" / "logs"
    journal_path = bundle_path / "raw" / "data" / "trade_journal_e8_one_5k.jsonl"
    chosen_log = _choose_main_log(log_dir)
    log_analysis = _analyze_log(chosen_log)
    journal_analysis = _analyze_trade_journal(journal_path)

    return {
        "bundle_path": str(bundle_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "log_analysis": log_analysis,
        "journal_analysis": journal_analysis,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze preview production log bundle.")
    parser.add_argument(
        "--bundle",
        required=True,
        help="Path to bundle root directory (contains raw/ and summary/).",
    )
    parser.add_argument(
        "--format",
        default="json",
        choices=["json", "markdown"],
        help="Output format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_path = Path(args.bundle).expanduser().resolve()
    summary = build_summary(bundle_path)
    if args.format == "json":
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    print(_render_markdown(summary))


if __name__ == "__main__":
    main()
