"""Tests for scanner confidence impact diagnostics."""

from pathlib import Path

import pandas as pd
import yaml

from src.diagnostics.analyze_scanner_confidence_impact import (
    _render_markdown,
    analyze_confidence_impact,
    build_summary,
)


def _make_candidates(confidence_by_key: dict[tuple[str, str, str], str]) -> pd.DataFrame:
    rows = [
        {
            "datetime": "2026-03-24",
            "instrument": "EURUSD",
            "side": "long",
            "alpha_score": 0.49,
            "alpha_confidence": confidence_by_key[("2026-03-24", "EURUSD", "long")],
            "publish_status": "published",
            "alpha_rank": 1.0,
        },
        {
            "datetime": "2026-03-24",
            "instrument": "GBPUSD",
            "side": "long",
            "alpha_score": 0.44,
            "alpha_confidence": confidence_by_key[("2026-03-24", "GBPUSD", "long")],
            "publish_status": "published",
            "alpha_rank": 2.0,
        },
        {
            "datetime": "2026-03-24",
            "instrument": "USDJPY",
            "side": "short",
            "alpha_score": 0.12,
            "alpha_confidence": confidence_by_key[("2026-03-24", "USDJPY", "short")],
            "publish_status": "published",
            "alpha_rank": 1.0,
        },
        {
            "datetime": "2026-03-24",
            "instrument": "AUDUSD",
            "side": "short",
            "alpha_score": 0.18,
            "alpha_confidence": confidence_by_key[("2026-03-24", "AUDUSD", "short")],
            "publish_status": "published",
            "alpha_rank": 2.0,
        },
    ]
    return pd.DataFrame(rows)


def test_analyze_confidence_impact_reports_prefilter_and_uplift_deltas() -> None:
    baseline_df = _make_candidates(
        {
            ("2026-03-24", "EURUSD", "long"): "low",
            ("2026-03-24", "GBPUSD", "long"): "low",
            ("2026-03-24", "USDJPY", "short"): "low",
            ("2026-03-24", "AUDUSD", "short"): "low",
        }
    )
    retuned_df = _make_candidates(
        {
            ("2026-03-24", "EURUSD", "long"): "medium",
            ("2026-03-24", "GBPUSD", "long"): "medium",
            ("2026-03-24", "USDJPY", "short"): "high",
            ("2026-03-24", "AUDUSD", "short"): "medium",
        }
    )

    result = analyze_confidence_impact(
        baseline_df=baseline_df,
        retuned_df=retuned_df,
        topk_long=1,
        topk_short=1,
        min_confidence="medium",
        min_blended_confidence=0.55,
        default_risk_pct=0.009,
        max_risk_pct=0.02,
        max_positions=5,
    )

    assert result["row_alignment"]["matched_row_count"] == 4
    assert result["row_alignment"]["live_row_count"] == 2
    assert result["baseline_summary"]["confidence_distribution"] == {"low": 2}
    assert result["retuned_summary"]["confidence_distribution"] == {"medium": 1, "high": 1}
    assert result["baseline_summary"]["prefilter"]["pass_count"] == 0
    assert result["retuned_summary"]["prefilter"]["pass_count"] == 2
    assert result["baseline_summary"]["capital_uplift_distribution"] == {"0.0": 2}
    assert result["retuned_summary"]["capital_uplift_distribution"] == {"0.5": 1, "1.0": 1}
    assert result["delta_summary"]["confidence_changed_row_count"] == 2
    assert result["delta_summary"]["prefilter_newly_passed_count"] == 2


def test_analyze_confidence_impact_returns_empty_live_summary_when_no_rows_align() -> None:
    baseline_df = pd.DataFrame(
        [
            {
                "datetime": "2026-03-24",
                "instrument": "EURUSD",
                "side": "long",
                "alpha_score": 0.49,
                "alpha_confidence": "low",
                "publish_status": "published",
                "alpha_rank": 1.0,
            }
        ]
    )
    retuned_df = pd.DataFrame(
        [
            {
                "datetime": "2026-03-25",
                "instrument": "EURUSD",
                "side": "long",
                "alpha_score": 0.49,
                "alpha_confidence": "medium",
                "publish_status": "published",
                "alpha_rank": 1.0,
            }
        ]
    )

    result = analyze_confidence_impact(
        baseline_df=baseline_df,
        retuned_df=retuned_df,
        topk_long=1,
        topk_short=0,
        min_confidence="medium",
        min_blended_confidence=0.55,
        default_risk_pct=0.009,
        max_risk_pct=0.02,
        max_positions=5,
    )

    assert result["row_alignment"]["matched_row_count"] == 0
    assert result["row_alignment"]["baseline_unmatched_row_count"] == 1
    assert result["row_alignment"]["retuned_unmatched_row_count"] == 1
    assert result["row_alignment"]["live_row_count"] == 0
    assert result["baseline_summary"]["confidence_distribution"] == {}
    assert result["retuned_summary"]["confidence_distribution"] == {}
    assert result["baseline_summary"]["prefilter"]["pass_count"] == 0
    assert result["retuned_summary"]["prefilter"]["pass_count"] == 0
    assert result["sample_changed_rows"] == []


def test_build_summary_reads_account_config_and_renders_markdown(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.csv"
    retuned_path = tmp_path / "retuned.csv"
    config_path = tmp_path / "account.yaml"

    _make_candidates(
        {
            ("2026-03-24", "EURUSD", "long"): "low",
            ("2026-03-24", "GBPUSD", "long"): "low",
            ("2026-03-24", "USDJPY", "short"): "low",
            ("2026-03-24", "AUDUSD", "short"): "low",
        }
    ).to_csv(baseline_path, index=False)
    _make_candidates(
        {
            ("2026-03-24", "EURUSD", "long"): "medium",
            ("2026-03-24", "GBPUSD", "long"): "medium",
            ("2026-03-24", "USDJPY", "short"): "high",
            ("2026-03-24", "AUDUSD", "short"): "medium",
        }
    ).to_csv(retuned_path, index=False)
    config_path.write_text(
        yaml.safe_dump(
            {
                "scanner": {"topk": 1, "topk_short": 1},
                "scheduler": {
                    "llm_threshold_override": {
                        "enabled": True,
                        "min_confidence": "medium",
                        "min_blended_confidence": 0.55,
                    }
                },
                "execution": {
                    "default_risk_pct": 0.009,
                    "max_risk_pct": 0.02,
                    "max_positions": 5,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    summary = build_summary(
        baseline_candidates_path=baseline_path,
        retuned_candidates_path=retuned_path,
        config_path=config_path,
        sample_limit=5,
    )

    assert summary["analysis_context"]["topk_long"] == 1
    assert summary["analysis_context"]["topk_short"] == 1
    assert summary["analysis_context"]["min_confidence"] == "medium"
    assert summary["retuned_summary"]["prefilter"]["pass_count"] == 2

    markdown = _render_markdown(summary)

    assert "# Scanner Confidence Impact" in markdown
    assert "Prefilter Pass Rate" in markdown
    assert "Capital Uplift Distribution" in markdown
