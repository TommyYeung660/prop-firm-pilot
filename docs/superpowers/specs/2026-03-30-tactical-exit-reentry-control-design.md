# Tactical Exit Re-entry Control Design

**Date:** 2026-03-30

**Goal:** Stop the current production pattern where tactical exits close positions for tiny profits or tactical losses, then the scanner recreates the same-symbol trade within minutes.

## Problem Summary

Production evidence from `prod_logs_20260330_v1.5.0_stable` shows three linked behaviors:

1. `severe_tactical_reversal` can full-close a position very early, sometimes within 1-3 minutes of entry and at very small positive PnL.
2. Tactical losses such as `initial_risk_structure_failure` are operationally similar to stop losses, but they do not count toward the current daily or per-symbol SL circuit breakers.
3. The account config already defines `tactical.intent_dedup.cooldown_after_close_seconds: 1800`, but the runtime does not enforce a same-symbol post-close cooldown during scanner admission.

This creates a churn loop:

- scanner reuses the same cached daily signal
- entry opens with a conservative low-confidence SL/TP profile
- tactical exit closes early on tactical structure rules
- the symbol becomes flat again
- the next scanner cycle recreates the same symbol trade

## Goals

- Enforce a real same-symbol cooldown after any close so the system does not immediately re-enter.
- Count tactical loss exits as `sl-like` losses for admission circuit breakers.
- Prevent `severe_tactical_reversal` from full-closing trades that have barely been open or have only reached trivial profit.

## Non-Goals

- Do not redesign the full tactical exit state machine.
- Do not retune the general `DEFAULT_SL_TP` table or LLM/scanner confidence mapping.
- Do not change close-reason taxonomy for analytics; tactical exits should keep their tactical reason codes.

## Approved Design

### 1. Same-Symbol Recent-Close Cooldown

Use `config.tactical.intent_dedup.cooldown_after_close_seconds` as an actual scanner admission gate.

Behavior:

- Before creating a new scanner intent for `symbol`, query the store for a recently closed intent for that same `symbol`.
- The cooldown applies to all close reasons, not just losses.
- The authoritative timestamp is `closed_at`.
- For backward compatibility with legacy rows, the query may fall back to `updated_at` if `closed_at` is null.

Reasoning:

- The operational requirement is "do not reopen the same symbol immediately after any close."
- This should block fast re-entry after both tiny tactical wins and tactical losses.

### 2. `sl-like` Tactical Loss Classification

Extend scanner breaker accounting so tactical losses with negative realized PnL are treated as stop-loss-like losses.

`sl-like` losses are defined as closed intents where:

- `status = 'closed'`
- `realized_pnl < 0`
- `exit_reason` is one of:
  - `sl_hit`
  - `initial_risk_structure_failure`
  - `severe_tactical_reversal`

Behavior:

- Daily breaker uses `count_sl_like_losses_today(trade_date)`.
- Per-symbol breaker uses `count_symbol_sl_like_losses_today(symbol, trade_date)`.
- Positive tactical exits must not count as losses, even if the reason is `severe_tactical_reversal`.

Reasoning:

- Current production losses are often recorded with tactical reason codes instead of `sl_hit`.
- Breakers should respond to loss semantics, not only one exact taxonomy label.

### 3. `severe_tactical_reversal` Full-Close Gate

Restrict `severe_tactical_reversal` so it can only full-close once the position has both:

- `hold_seconds >= severe_reversal_min_hold_seconds`
- `unrealized_r >= severe_reversal_min_r`

New tactical exit config fields:

- `severe_reversal_min_hold_seconds: int = 900`
- `severe_reversal_min_r: float = 0.5`

Behavior:

- If the tactical reversal pattern is detected but either threshold is not met, do not emit `EXIT_NOW` from that rule.
- The position should continue through the normal state machine and may still resolve to `MOVE_TO_BREAKEVEN`, `TRAIL_SL`, `PARTIAL_CLOSE`, or `HOLD`.
- The rule remains eligible for `EXIT_NOW` only after both thresholds are satisfied.

Reasoning:

- Current logic only requires `unrealized_r > 0`, which is too sensitive.
- A full close should require meaningful hold time and meaningful profit, not a tiny transient favorable move.

## Data Flow Changes

### Scanner Admission Path

Current:

- scanner signal
- active-position guard
- compliance headroom and rejection cooldown
- loss breakers
- create new intent

New:

- scanner signal
- active-position guard
- recent-close cooldown guard
- compliance headroom and rejection cooldown
- `sl-like` loss breakers
- create new intent

### Tactical Exit Path

Current:

- build snapshot
- detect severe reversal with only `unrealized_r > 0`
- possibly emit `EXIT_NOW`

New:

- build snapshot including `hold_seconds`
- detect severe reversal pattern
- require both minimum hold time and minimum `R`
- only then allow `EXIT_NOW`

## File-Level Plan

- Modify [src/config.py](/Users/admin/Documents/projects/prop-firm-pilot/src/config.py)
  - add the two new tactical exit config fields
- Modify the active account YAML in use
  - set explicit values for the new severe-reversal thresholds
- Modify [src/decision/tactical_exit_rules.py](/Users/admin/Documents/projects/prop-firm-pilot/src/decision/tactical_exit_rules.py)
  - extend `TacticalExitSnapshot` with `hold_seconds`
  - gate `severe_tactical_reversal` with minimum hold/R thresholds
- Modify [src/scheduler/scheduler.py](/Users/admin/Documents/projects/prop-firm-pilot/src/scheduler/scheduler.py)
  - populate `hold_seconds`
  - enforce recent-close cooldown in scanner admission
  - switch breaker queries to `sl-like` counters
- Modify [src/decision_store/sqlite_store.py](/Users/admin/Documents/projects/prop-firm-pilot/src/decision_store/sqlite_store.py)
  - add recent-close cooldown query
  - add `sl-like` daily and per-symbol loss counters

## Test Strategy

### DecisionStore

- add a test proving a recently closed symbol is reported as cooling down
- add a test proving an old close outside the cooldown does not block
- add tests proving:
  - `sl_hit` counts as `sl-like`
  - `initial_risk_structure_failure` with negative PnL counts as `sl-like`
  - `severe_tactical_reversal` with positive PnL does not count as `sl-like`

### Scheduler

- add a scanner regression test showing a symbol closed moments ago is skipped during cooldown
- add breaker regression tests showing tactical losses now trigger:
  - per-symbol loss lock
  - daily loss circuit breaker

### Tactical Exit Rules

- add a failing test showing `severe_tactical_reversal` does not `EXIT_NOW` when hold time is too short
- add a failing test showing `severe_tactical_reversal` does not `EXIT_NOW` when `unrealized_r` is below threshold
- add a test preserving `EXIT_NOW` once both thresholds are met

## Risks

- Using too long a cooldown may suppress valid same-day re-entry opportunities. This is acceptable for the immediate production problem because the observed failure is over-trading on the same cached signal.
- Counting tactical losses as `sl-like` losses may trip breakers earlier than before. This is intentional because those losses are already economically equivalent to failed entries.
- Tightening `severe_tactical_reversal` may leave some weak trades open longer. This is acceptable because the current behavior is demonstrably too eager.

## Rollout Expectation

After this change:

- a symbol closed moments ago will not be recreated immediately by the scanner
- tactical losses will contribute to the same operational breakers as stop losses
- tiny favorable moves will no longer be enough to trigger `severe_tactical_reversal` full-close
