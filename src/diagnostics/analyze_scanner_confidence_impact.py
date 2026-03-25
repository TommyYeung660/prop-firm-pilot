"""
Scanner confidence retune impact diagnostics for prop-firm-pilot.

Compares baseline and retuned scanner candidate sets and quantifies how the
changed confidence labels alter downstream filtering and capital-uplift behavior.

Usage:
    summary = analyze_confidence_impact(...)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import AppConfig, load_config


REQUIRED_CANDIDATE_COLUMNS = {
    "datetime",
    "instrument",
    "side",
    "alpha_score",
    "alpha_confidence",
    "publish_status",
    "alpha_rank",
}

CONFIDENCE_SCORE_MAP: dict[str, float] = {"high": 0.9, "medium": 0.6, "low": 0.3}
CAPITAL_UPLIFT_FACTORS: dict[str, float] = {"high": 1.0, "medium": 0.5, "low": 0.0}


def _validate_candidate_columns(df: pd.DataFrame, label: str) -> None:
    missing = sorted(REQUIRED_CANDIDATE_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{label} rows missing required columns: {missing}")


def _normalize_confidence(value: Any) -> str:
    return str(value or "").strip().lower()


def _confidence_score(value: Any) -> float:
    return CONFIDENCE_SCORE_MAP.get(_normalize_confidence(value), 0.5)


def _uplift_factor(value: Any) -> float:
    return CAPITAL_UPLIFT_FACTORS.get(_normalize_confidence(value), 0.0)


def _directional_quality(row: pd.Series, score_column: str, side_column: str) -> float:
    score = float(row[score_column])
    side = str(row[side_column]).strip().lower()
    return 1.0 - score if side == "short" else score


def _live_mask(df: pd.DataFrame, *, topk_long: int, topk_short: int) -> pd.Series:
    publish_status = df["publish_status"].astype(str).str.strip().str.lower()
    side = df["side"].astype(str).str.strip().str.lower()
    alpha_rank = pd.to_numeric(df["alpha_rank"], errors="coerce")

    long_mask = side.eq("long") & alpha_rank.le(max(int(topk_long), 0))
    short_mask = side.eq("short") & alpha_rank.le(max(int(topk_short), 0))
    return publish_status.eq("published") & (long_mask | short_mask)


def _distribution(series: pd.Series) -> dict[str, int]:
    counts = Counter(str(value) for value in series.tolist())
    return dict(counts)


def _uplift_distribution(series: pd.Series) -> dict[str, int]:
    counts = Counter(f"{float(value):.1f}" for value in series.tolist())
    return dict(sorted(counts.items(), key=lambda item: float(item[0])))


def _summarize_source(
    df: pd.DataFrame,
    *,
    confidence_column: str,
    score_column: str,
    side_column: str,
    min_confidence: str,
    min_blended_confidence: float,
) -> dict[str, Any]:
    if df.empty:
        return {
            "confidence_distribution": {},
            "side_confidence_distribution": {},
            "prefilter": {"pass_count": 0, "pass_rate": 0.0},
            "capital_uplift_distribution": {},
        }

    confidence = df[confidence_column].map(_normalize_confidence)
    confidence_scores = df[confidence_column].map(_confidence_score)
    directional_quality = df.apply(
        lambda row: _directional_quality(row, score_column=score_column, side_column=side_column),
        axis=1,
    )
    blended = 0.6 * confidence_scores + 0.4 * directional_quality
    passes = (confidence_scores >= _confidence_score(min_confidence)) & (
        blended >= float(min_blended_confidence)
    )
    uplift = df[confidence_column].map(_uplift_factor)

    side_confidence_distribution: dict[str, dict[str, int]] = {}
    for side, side_df in df.assign(_confidence=confidence).groupby(side_column, sort=True):
        side_confidence_distribution[str(side)] = _distribution(side_df["_confidence"])

    return {
        "confidence_distribution": _distribution(confidence),
        "side_confidence_distribution": side_confidence_distribution,
        "prefilter": {
            "pass_count": int(passes.sum()),
            "pass_rate": float(passes.mean()) if len(df) else 0.0,
        },
        "capital_uplift_distribution": _uplift_distribution(uplift),
    }


def analyze_confidence_impact(
    *,
    baseline_df: pd.DataFrame,
    retuned_df: pd.DataFrame,
    topk_long: int,
    topk_short: int,
    min_confidence: str,
    min_blended_confidence: float,
    default_risk_pct: float,
    max_risk_pct: float,
    max_positions: int,
) -> dict[str, Any]:
    _validate_candidate_columns(baseline_df, "baseline")
    _validate_candidate_columns(retuned_df, "retuned")

    join_keys = ["datetime", "instrument", "side"]
    baseline = baseline_df.copy()
    retuned = retuned_df.copy()
    baseline["_baseline_live"] = _live_mask(baseline, topk_long=topk_long, topk_short=topk_short)
    retuned["_retuned_live"] = _live_mask(retuned, topk_long=topk_long, topk_short=topk_short)

    matched = baseline.merge(
        retuned,
        on=join_keys,
        how="inner",
        suffixes=("_baseline", "_retuned"),
    )
    baseline_keys = set(tuple(row) for row in baseline[join_keys].itertuples(index=False, name=None))
    retuned_keys = set(tuple(row) for row in retuned[join_keys].itertuples(index=False, name=None))

    if matched.empty:
        comparison_live = matched.copy()
    else:
        comparison_live = matched.loc[matched["_baseline_live"] | matched["_retuned_live"]].copy()

    baseline_live = comparison_live.loc[comparison_live["_baseline_live"]].copy()
    retuned_live = comparison_live.loc[comparison_live["_retuned_live"]].copy()

    changed_live = comparison_live.loc[
        comparison_live["alpha_confidence_baseline"].map(_normalize_confidence)
        != comparison_live["alpha_confidence_retuned"].map(_normalize_confidence)
    ].copy()

    baseline_prefilter = _summarize_source(
        baseline_live,
        confidence_column="alpha_confidence_baseline",
        score_column="alpha_score_baseline",
        side_column="side",
        min_confidence=min_confidence,
        min_blended_confidence=min_blended_confidence,
    )
    retuned_prefilter = _summarize_source(
        retuned_live,
        confidence_column="alpha_confidence_retuned",
        score_column="alpha_score_retuned",
        side_column="side",
        min_confidence=min_confidence,
        min_blended_confidence=min_blended_confidence,
    )

    newly_passed_count = max(
        retuned_prefilter["prefilter"]["pass_count"] - baseline_prefilter["prefilter"]["pass_count"],
        0,
    )

    sample_changed_rows = [
        {
            "datetime": str(row["datetime"]),
            "instrument": str(row["instrument"]),
            "side": str(row["side"]),
            "alpha_score_baseline": float(row["alpha_score_baseline"]),
            "alpha_score_retuned": float(row["alpha_score_retuned"]),
            "alpha_confidence_baseline": _normalize_confidence(row["alpha_confidence_baseline"]),
            "alpha_confidence_retuned": _normalize_confidence(row["alpha_confidence_retuned"]),
        }
        for _, row in changed_live.head(10).iterrows()
    ]

    return {
        "analysis_context": {
            "topk_long": int(topk_long),
            "topk_short": int(topk_short),
            "min_confidence": _normalize_confidence(min_confidence),
            "min_blended_confidence": float(min_blended_confidence),
            "default_risk_pct": float(default_risk_pct),
            "max_risk_pct": float(max_risk_pct),
            "max_positions": int(max_positions),
        },
        "row_alignment": {
            "baseline_row_count": int(len(baseline)),
            "retuned_row_count": int(len(retuned)),
            "matched_row_count": int(len(matched)),
            "baseline_unmatched_row_count": int(len(baseline_keys - retuned_keys)),
            "retuned_unmatched_row_count": int(len(retuned_keys - baseline_keys)),
            "baseline_live_row_count": int(len(baseline_live)),
            "retuned_live_row_count": int(len(retuned_live)),
            "live_row_count": int(len(comparison_live)),
        },
        "baseline_summary": baseline_prefilter,
        "retuned_summary": retuned_prefilter,
        "delta_summary": {
            "confidence_changed_row_count": int(len(changed_live)),
            "prefilter_newly_passed_count": int(newly_passed_count),
            "prefilter_pass_count_delta": int(
                retuned_prefilter["prefilter"]["pass_count"] - baseline_prefilter["prefilter"]["pass_count"]
            ),
            "prefilter_pass_rate_delta": float(
                retuned_prefilter["prefilter"]["pass_rate"] - baseline_prefilter["prefilter"]["pass_rate"]
            ),
        },
        "sample_changed_rows": sample_changed_rows,
    }


def _resolved_runtime_inputs(config_path: Path | None) -> dict[str, Any]:
    config = load_config(config_path) if config_path is not None else AppConfig()
    override = config.scheduler.llm_threshold_override
    return {
        "topk_long": int(config.scanner.topk),
        "topk_short": int(config.scanner.topk_short),
        "min_confidence": str(override.min_confidence),
        "min_blended_confidence": float(override.min_blended_confidence),
        "default_risk_pct": float(config.execution.default_risk_pct),
        "max_risk_pct": float(config.execution.max_risk_pct),
        "max_positions": int(config.execution.max_positions),
        "threshold_source": "override" if bool(override.enabled) else "config_default",
    }


def build_summary(
    *,
    baseline_candidates_path: str | Path,
    retuned_candidates_path: str | Path,
    config_path: str | Path | None = None,
    sample_limit: int = 10,
) -> dict[str, Any]:
    baseline_path = Path(baseline_candidates_path).expanduser().resolve()
    retuned_path = Path(retuned_candidates_path).expanduser().resolve()
    resolved_config_path = Path(config_path).expanduser().resolve() if config_path is not None else None

    runtime_inputs = _resolved_runtime_inputs(resolved_config_path)
    summary = analyze_confidence_impact(
        baseline_df=pd.read_csv(baseline_path),
        retuned_df=pd.read_csv(retuned_path),
        topk_long=runtime_inputs["topk_long"],
        topk_short=runtime_inputs["topk_short"],
        min_confidence=runtime_inputs["min_confidence"],
        min_blended_confidence=runtime_inputs["min_blended_confidence"],
        default_risk_pct=runtime_inputs["default_risk_pct"],
        max_risk_pct=runtime_inputs["max_risk_pct"],
        max_positions=runtime_inputs["max_positions"],
    )
    summary["analysis_context"]["threshold_source"] = runtime_inputs["threshold_source"]
    summary["baseline_candidates_path"] = str(baseline_path)
    summary["retuned_candidates_path"] = str(retuned_path)
    summary["config_path"] = str(resolved_config_path) if resolved_config_path is not None else None
    summary["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["sample_changed_rows"] = summary["sample_changed_rows"][: max(int(sample_limit), 0)]
    return summary


def _render_markdown(summary: dict[str, Any]) -> str:
    ctx = summary["analysis_context"]
    row_alignment = summary["row_alignment"]
    baseline = summary["baseline_summary"]
    retuned = summary["retuned_summary"]
    delta = summary["delta_summary"]

    lines: list[str] = []
    lines.append("# Scanner Confidence Impact")
    lines.append("")
    lines.append(f"- Baseline candidates: `{summary['baseline_candidates_path']}`")
    lines.append(f"- Retuned candidates: `{summary['retuned_candidates_path']}`")
    lines.append(f"- Config: `{summary['config_path']}`")
    lines.append(f"- Generated at (UTC): `{summary['generated_at_utc']}`")
    lines.append("")
    lines.append("## Runtime Context")
    lines.append(
        f"- topk_long / topk_short: `{ctx['topk_long']}` / `{ctx['topk_short']}`"
    )
    lines.append(
        f"- Prefilter threshold: `{ctx['min_confidence']}` + blended `>= {ctx['min_blended_confidence']}`"
    )
    lines.append(f"- Threshold source: `{ctx['threshold_source']}`")
    lines.append("")
    lines.append("## Row Alignment")
    lines.append(
        f"- baseline / retuned / matched / live rows: `{row_alignment['baseline_row_count']}` / "
        f"`{row_alignment['retuned_row_count']}` / `{row_alignment['matched_row_count']}` / "
        f"`{row_alignment['live_row_count']}`"
    )
    lines.append(
        f"- baseline unmatched / retuned unmatched: `{row_alignment['baseline_unmatched_row_count']}` / "
        f"`{row_alignment['retuned_unmatched_row_count']}`"
    )
    lines.append("")
    lines.append("## Confidence Distribution")
    lines.append(
        f"- Baseline: `{json.dumps(baseline['confidence_distribution'], ensure_ascii=False)}`"
    )
    lines.append(
        f"- Retuned: `{json.dumps(retuned['confidence_distribution'], ensure_ascii=False)}`"
    )
    lines.append("")
    lines.append("## Prefilter Pass Rate")
    lines.append(
        f"- Baseline: `{baseline['prefilter']['pass_count']}` rows / `{baseline['prefilter']['pass_rate']:.4f}`"
    )
    lines.append(
        f"- Retuned: `{retuned['prefilter']['pass_count']}` rows / `{retuned['prefilter']['pass_rate']:.4f}`"
    )
    lines.append(
        f"- Delta count / rate: `{delta['prefilter_pass_count_delta']}` / `{delta['prefilter_pass_rate_delta']:.4f}`"
    )
    lines.append("")
    lines.append("## Capital Uplift Distribution")
    lines.append(
        f"- Baseline: `{json.dumps(baseline['capital_uplift_distribution'], ensure_ascii=False)}`"
    )
    lines.append(
        f"- Retuned: `{json.dumps(retuned['capital_uplift_distribution'], ensure_ascii=False)}`"
    )
    lines.append("")
    if summary["sample_changed_rows"]:
        lines.append("## Sample Changed Rows")
        for row in summary["sample_changed_rows"]:
            lines.append(
                f"- {row['datetime']} {row['instrument']} {row['side']}: "
                f"{row['alpha_confidence_baseline']} -> {row['alpha_confidence_retuned']}"
            )
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze scanner confidence retune impact.")
    parser.add_argument("--baseline-candidates", required=True, help="Path to baseline alpha_candidates.csv")
    parser.add_argument("--retuned-candidates", required=True, help="Path to retuned alpha_candidates.csv")
    parser.add_argument("--config", default="", help="Optional account config YAML path.")
    parser.add_argument("--format", choices=["json", "markdown"], default="json", help="Output format.")
    parser.add_argument("--sample-limit", type=int, default=10, help="Max changed rows to emit in summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(
        baseline_candidates_path=args.baseline_candidates,
        retuned_candidates_path=args.retuned_candidates,
        config_path=args.config or None,
        sample_limit=args.sample_limit,
    )
    if args.format == "json":
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    print(_render_markdown(summary))


if __name__ == "__main__":
    main()
