# Daily Summary Ablation Telegram Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Append a 7-day rolling ablation summary to the existing Telegram
daily summary without changing live trading behavior.

**Architecture:** Scheduler reads recent
`TACTICAL_ENTRY_CALIBRATION_SNAPSHOT` events from the trade journal, converts
them into a structured ablation report through
`analyze_entry_funnel_ablation.py`, and passes that report into
`AlertService.daily_summary()` for final Telegram rendering. Missing-mode cases
must surface as `insufficient_ablation_data` rather than a fake full ablation.

**Tech Stack:** Python 3.10, async scheduler, TradeJournal JSONL events,
Telegram AlertService formatting, pytest, ruff

---

### Task 1: Add insufficient-data handling to ablation diagnostics

**Files:**
- Modify: `C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/diagnostics/analyze_entry_funnel_ablation.py`
- Modify: `C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/tests/test_entry_funnel_ablation.py`

**Step 1: Write the failing test**

Add a test where only two modes are present in the 7-day snapshots and assert:

- `recommendation == "insufficient_ablation_data"`
- `available_modes == ["B", "D"]`

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_entry_funnel_ablation.py -q -k insufficient`
Expected: FAIL because diagnostics currently has no insufficient-data path.

**Step 3: Write minimal implementation**

Update diagnostics so it:

- detects when fewer than four A/B/C/D modes have data
- returns `insufficient_ablation_data`
- includes a stable `available_modes` list in the output

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_entry_funnel_ablation.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/diagnostics/analyze_entry_funnel_ablation.py tests/test_entry_funnel_ablation.py
git commit -m "feat: add insufficient-data ablation recommendation"
```

### Task 2: Extend AlertService daily summary rendering

**Files:**
- Modify: `C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/monitor/alert_service.py`
- Modify: `C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/tests/test_alert_service.py`

**Step 1: Write the failing tests**

Add tests asserting:

- `daily_summary(..., ablation_summary=...)` appends an `<b>Ablation (7d)</b>`
  section to the same message
- `daily_summary()` without `ablation_summary` remains backward-compatible

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_alert_service.py -q -k daily_summary`
Expected: FAIL because `daily_summary()` does not accept ablation input yet.

**Step 3: Write minimal implementation**

Add an optional `ablation_summary: dict[str, Any] | None = None` argument to
`daily_summary()` and render:

- recommendation
- available modes
- one short evidence line per available mode

Keep the message bounded and omit irrelevant churn fields per mode.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_alert_service.py -q -k daily_summary`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/monitor/alert_service.py tests/test_alert_service.py
git commit -m "feat: add ablation section to daily summary alert"
```

### Task 3: Wire 7-day ablation into scheduler daily summary

**Files:**
- Modify: `C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/scheduler/scheduler.py`
- Modify: `C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/tests/test_scheduler.py`

**Step 1: Write the failing scheduler tests**

Add tests asserting:

- scheduler reads the last 7 days of
  `TACTICAL_ENTRY_CALIBRATION_SNAPSHOT`
- scheduler passes the computed `ablation_summary` into
  `AlertService.daily_summary()`
- same-day summary still works when no trade journal is configured

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scheduler.py -q -k ablation`
Expected: FAIL because `_send_daily_summary()` does not build or pass the
ablation summary yet.

**Step 3: Write minimal implementation**

Add a scheduler helper such as `_build_rolling_ablation_summary(date_str)` that:

- computes the trailing 7 calendar dates
- loads `TACTICAL_ENTRY_CALIBRATION_SNAPSHOT` events from the journal
- calls `analyze_ablation()`
- returns `None` when no journal exists

Then pass that result into `AlertService.daily_summary()`.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_scheduler.py -q -k ablation`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/scheduler/scheduler.py tests/test_scheduler.py
git commit -m "feat: wire rolling ablation into daily summary"
```

### Task 4: Run bounded verification on changed scope

**Files:**
- Verify: `C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/diagnostics/analyze_entry_funnel_ablation.py`
- Verify: `C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/monitor/alert_service.py`
- Verify: `C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/scheduler/scheduler.py`
- Verify: `C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/tests/test_entry_funnel_ablation.py`
- Verify: `C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/tests/test_alert_service.py`
- Verify: `C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/tests/test_scheduler.py`

**Step 1: Run targeted pytest**

Run:
`uv run pytest tests/test_entry_funnel_ablation.py tests/test_alert_service.py tests/test_scheduler.py -q`
Expected: PASS.

**Step 2: Run changed-scope lint**

Run:
`uv run ruff check src/diagnostics/analyze_entry_funnel_ablation.py src/monitor/alert_service.py src/scheduler/scheduler.py tests/test_entry_funnel_ablation.py tests/test_alert_service.py tests/test_scheduler.py`
Expected: `All checks passed!`

**Step 3: Commit**

```bash
git add src tests docs
git commit -m "feat: append rolling ablation summary to daily alert"
```
