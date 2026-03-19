# Daily Summary Ablation Telegram Design

> **Date:** 2026-03-19
>
> **Status:** Approved for planning
>
> **Scope:** Append a 7-day rolling entry-funnel ablation summary to the
> existing Telegram daily summary message.

---

## 1. Goal

Make the current daily Telegram trade summary carry a short, operator-readable
ablation section that answers:

- what recommendation the last 7 days support
- which entry-funnel modes have data
- a few high-signal economic and churn counters per available mode

This must not change live trading decisions. It only changes daily summary
generation and Telegram presentation.

---

## 2. Constraints

- The current live path must remain unchanged.
- The ablation summary must be derived from existing journal evidence.
- The recommendation must remain deterministic and rule-based.
- The daily summary must stay readable in one Telegram message.
- Missing modes must not be hidden or guessed.

---

## 3. Approaches Considered

### Option 1: Scheduler computes a structured ablation summary and passes it to AlertService

Pros:

- keeps journal/data access inside scheduler
- keeps Telegram formatting inside AlertService
- reuses the existing `analyze_ablation()` diagnostics logic

Cons:

- touches both scheduler and alert formatting

### Option 2: AlertService reads the journal and computes ablation itself

Pros:

- smaller scheduler diff

Cons:

- mixes data access into the alert layer
- harder to test and maintain

### Option 3: Scheduler formats a raw Telegram text block and AlertService just appends it

Pros:

- fastest path

Cons:

- report logic leaks into scheduler
- formatting becomes harder to test cleanly

### Chosen Option

Use Option 1.

This keeps responsibilities clean:

- scheduler gathers the last 7 days of snapshots
- diagnostics computes the report
- AlertService renders the Telegram section

---

## 4. Data Window

The ablation section uses a fixed rolling 7-day calendar window ending on the
current summary date.

Source events:

- `TACTICAL_ENTRY_CALIBRATION_SNAPSHOT`

The scheduler will read those snapshots from the trade journal, collect the
rows inside the 7-day window, and pass them to `analyze_ablation()`.

---

## 5. Insufficient Data Rule

If the last 7 days do not contain all four modes `A/B/C/D`, the system must not
pretend that a full ablation exists.

Instead it should emit:

- recommendation: `insufficient_ablation_data`
- a list of available modes
- partial per-mode evidence for those available modes only

This preserves honesty while still giving the operator daily visibility.

---

## 6. Telegram Output Shape

The ablation section is appended to the same Telegram daily summary message.

Minimal section shape:

```text
<b>Ablation (7d)</b>
• Recommendation: insufficient_ablation_data
• Available modes: B, D
• B PnL/Open: $+120.00 / 3
• B LLM veto: 20.0%
• D PnL/Open: $+0.00 / 0
```

Rules:

- always show recommendation
- always show available mode labels
- show only high-signal counters
- if a churn metric is not meaningful for a mode, omit it
- do not dump raw JSON into Telegram

---

## 7. Component Changes

### 7.1 Diagnostics

`src/diagnostics/analyze_entry_funnel_ablation.py`

Add an insufficient-data recommendation path and preserve the existing
structured output shape.

### 7.2 Scheduler

`src/scheduler/scheduler.py`

Add a helper that:

- reads the last 7 days of `TACTICAL_ENTRY_CALIBRATION_SNAPSHOT`
- calls `analyze_ablation()`
- passes the result into `AlertService.daily_summary()`

### 7.3 AlertService

`src/monitor/alert_service.py`

Extend `daily_summary()` with an optional `ablation_summary` argument and
render the `<b>Ablation (7d)</b>` section when provided.

---

## 8. Testing

Tests should cover:

- diagnostics returns `insufficient_ablation_data` when modes are incomplete
- AlertService appends the ablation block to the same daily summary message
- scheduler collects the correct rolling window and forwards the summary
- backward compatibility when no ablation summary is provided

---

## 9. Success Criteria

This design is successful when:

- the daily Telegram summary includes a readable ablation section
- missing modes are shown honestly instead of inferred
- no live trading behavior changes
- the feature remains deterministic and bounded
