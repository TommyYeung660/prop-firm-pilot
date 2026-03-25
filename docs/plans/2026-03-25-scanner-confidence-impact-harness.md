# Scanner Confidence Impact Harness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reusable diagnostics CLI that compares baseline vs retuned scanner `alpha_candidates.csv` files and quantifies how the confidence retune changes downstream `prop-firm-pilot` gating and risk-uplift behavior.

**Architecture:** Keep the analysis pure and deterministic. Put the comparison logic in a new diagnostics module that operates on dataframes and resolved config values, then add a thin CLI wrapper for CSV/config loading and `json`/`markdown` rendering. Fix the pre-existing preview-bundle baseline test first so the worktree starts from a usable verification baseline.

**Tech Stack:** Python 3.10, pandas, argparse, existing `src.config` loader, pytest

---

### Task 1: Repair the pre-existing diagnostics baseline test

**Files:**
- Modify: `tests/diagnostics/test_analyze_preview_bundle.py`

**Step 1: Keep the failing test focused on current release-tag behavior**

Make the test use `get_release_tag()` instead of a stale hardcoded preview filename.

**Step 2: Run the single test to verify it passes**

Run: `uv run python -m pytest tests\diagnostics\test_analyze_preview_bundle.py::test_choose_main_log_prefers_current_release_tag_over_latest_generic_log -q`

Expected: PASS

**Step 3: Run the small diagnostics baseline suite**

Run: `uv run python -m pytest tests\test_entry_funnel_ablation.py tests\diagnostics\test_analyze_preview_bundle.py -q`

Expected: PASS

### Task 2: Write failing unit tests for the impact analyzer core

**Files:**
- Create: `tests/diagnostics/test_analyze_scanner_confidence_impact.py`

**Step 1: Write a failing test for downstream pass-rate and confidence-delta analysis**

Add a minimal fixture with matched baseline/retuned rows showing:

- baseline all `low`
- retuned rows split into `medium` / `high`
- long/short directional quality handled correctly
- current config selection (`topk=1`, `topk_short=1`) limits the analyzed live rows

Example shape:

```python
def test_analyze_confidence_impact_reports_prefilter_and_uplift_deltas() -> None:
    baseline = pd.DataFrame([...])
    retuned = pd.DataFrame([...])

    result = analyze_confidence_impact(
        baseline_df=baseline,
        retuned_df=retuned,
        topk_long=1,
        topk_short=1,
        min_confidence="medium",
        min_blended_confidence=0.55,
        default_risk_pct=0.009,
        max_risk_pct=0.02,
        max_positions=5,
    )

    assert result["row_alignment"]["matched_row_count"] == 2
    assert result["baseline_summary"]["confidence_distribution"] == {"low": 2}
    assert result["retuned_summary"]["confidence_distribution"] == {"medium": 1, "high": 1}
    assert result["baseline_summary"]["prefilter"]["pass_count"] == 0
    assert result["retuned_summary"]["prefilter"]["pass_count"] == 2
```

**Step 2: Write a failing test for zero-match behavior**

Add a case where baseline and retuned do not align. Assert the result still returns:

- zero matched rows
- non-zero unmatched counts
- empty but present summary sections

**Step 3: Run tests to verify they fail**

Run: `uv run python -m pytest tests\diagnostics\test_analyze_scanner_confidence_impact.py -q`

Expected: FAIL because the new analyzer module does not exist yet.

### Task 3: Implement the analyzer core

**Files:**
- Create: `src/diagnostics/analyze_scanner_confidence_impact.py`

**Step 1: Add minimal analysis helpers**

Implement:

- required-column validation
- `(datetime, instrument, side)` inner join alignment
- live-row filtering by `publish_status`, `alpha_rank`, `topk_long`, `topk_short`
- confidence label scoring using the scheduler mapping (`high=0.9`, `medium=0.6`, `low=0.3`)
- directional quality using `1 - alpha_score` for `short`
- blended prefilter score
- capital uplift factor mapping (`high=1.0`, `medium=0.5`, `low=0.0`)

Add a public pure function:

```python
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
    ...
```

**Step 2: Make the new core tests pass**

Run: `uv run python -m pytest tests\diagnostics\test_analyze_scanner_confidence_impact.py -q`

Expected: PASS

### Task 4: Add rendering and CLI coverage

**Files:**
- Modify: `src/diagnostics/analyze_scanner_confidence_impact.py`
- Modify: `tests/diagnostics/test_analyze_scanner_confidence_impact.py`

**Step 1: Add config-resolved CLI entry points**

Implement:

- `build_summary(...)`
- `_render_markdown(summary)`
- `parse_args()`
- `main()`

CLI flags:

- `--baseline-candidates`
- `--retuned-candidates`
- `--config` (optional)
- `--format {json,markdown}`
- `--sample-limit` (optional, default small)

If `--config` is provided, use `load_config()` to resolve:

- `scanner.topk`
- `scanner.topk_short`
- `scheduler.llm_threshold_override`
- `execution.default_risk_pct`
- `execution.max_risk_pct`
- `execution.max_positions`

If no config is provided, use `AppConfig()` defaults.

**Step 2: Add CLI/output tests**

Add tests asserting:

- `build_summary()` resolves config-backed thresholds
- markdown output includes pass-rate and uplift sections
- CLI help path parses

**Step 3: Run diagnostics tests**

Run: `uv run python -m pytest tests\diagnostics\test_analyze_scanner_confidence_impact.py tests\diagnostics\test_analyze_preview_bundle.py tests\test_entry_funnel_ablation.py -q`

Expected: PASS

### Task 5: Run an end-to-end dry-run against the scanner study artifacts

**Files:**
- No code changes expected

**Step 1: Run the new CLI against existing upstream study artifacts**

Use:

```bash
uv run python -m src.diagnostics.analyze_scanner_confidence_impact ^
  --baseline-candidates ..\..\..\qlib_market_scanner\.worktrees\fx-confidence-calibration-study\tmp\formal_fx_calibration_run\runtime_from_existing_pred\signals\alpha_candidates.csv ^
  --retuned-candidates ..\..\..\qlib_market_scanner\.worktrees\fx-confidence-calibration-study\tmp\formal_fx_calibration_run\runtime_from_existing_pred_retuned\signals\alpha_candidates.csv ^
  --config config\e8_one_5k_challenge.yaml ^
  --format json
```

**Step 2: Verify the output captures the expected directional impact**

Check that:

- retuned confidence distribution differs materially from baseline
- prefilter pass count/rate increases from baseline
- capital uplift distribution changes from all-low to mixed `low/medium/high`

### Task 6: Run final focused verification

**Files:**
- No code changes expected

**Step 1: Run the full targeted suite**

Run: `uv run python -m pytest tests\diagnostics\test_analyze_scanner_confidence_impact.py tests\diagnostics\test_analyze_preview_bundle.py tests\test_entry_funnel_ablation.py tests\test_scanner_bridge.py tests\scheduler\test_llm_thresholds.py tests\test_capital_allocator.py -q`

Expected: PASS

**Step 2: Run CLI help verification**

Run: `uv run python -m src.diagnostics.analyze_scanner_confidence_impact --help`

Expected: usage text prints without import errors.
