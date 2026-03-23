# TradeLocker Tactical Exit Snapshot Design

**Date:** 2026-03-23

**Goal:** Make the `TradeLocker` open-position snapshot reliable enough for the full tactical exit state machine so `MOVE_TO_BREAKEVEN`, `TRAIL_SL`, `REPRICE_TP`, `PARTIAL_CLOSE`, and `EXIT_NOW` all operate from real live broker state instead of partial payload fallbacks.

## Context

The current `TradeLocker` runtime can open, modify, and close positions, but the live `/positions` payload is not rich enough for tactical exit decisions:

- `current_price` can fall back to `avgPrice`, which is effectively the entry price.
- `sl_price` and `tp_price` can be `null` even when protective orders exist.
- tactical exit state classification in [scheduler.py](/C:/Users/tommy.yeung\CursorProjects\prop-firm\prop-firm-pilot\src\scheduler\scheduler.py#L3087) depends on `pos.current_price`, `pos.sl_price`, and `pos.tp_price`.
- the pure rule engine in [tactical_exit_rules.py](/C:/Users/tommy.yeung\CursorProjects\prop-firm\prop-firm-pilot\src\decision\tactical_exit_rules.py#L384) assumes that snapshot fields already reflect live market reality.

That mismatch allows `/profit` to show a live floating PnL while tactical exit still sees `0R` because it is reading a stale or incomplete position snapshot.

## Root Cause

The tactical exit pipeline is broker-neutral only at the model layer. It still relies on `BrokerPositionInfo` being semantically complete.

For `TradeLocker`, [tradelocker_client.py](/C:/Users/tommy.yeung\CursorProjects\prop-firm\prop-firm-pilot\src\execution\tradelocker_client.py#L302) currently builds `BrokerPositionInfo` directly from the `/positions` payload:

- `current_price` falls back to `currentPrice -> markPrice -> avgPrice`
- `sl_price` and `tp_price` only read inline `stopLoss` / `takeProfit`

For live array payloads, `avgPrice` is often present while `currentPrice`, `stopLoss`, and `takeProfit` are absent. That makes tactical exit read a structurally valid but operationally wrong snapshot.

## Recommended Approach

### Selected: broker-layer enrichment plus scheduler-level guardrails

Use a two-layer fix:

1. enrich `TradeLockerClient.get_open_positions()` so it returns a normalized live snapshot
2. keep tactical-exit-specific guardrails in `Scheduler` so the exit engine can safely degrade when broker readback is incomplete

This keeps the rule engine pure and makes the data correction reusable across all tactical exit actions.

## Architecture

### Layer 1: TradeLocker live snapshot enrichment

Enhance [tradelocker_client.py](/C:/Users/tommy.yeung\CursorProjects\prop-firm\prop-firm-pilot\src\execution\tradelocker_client.py) so `get_open_positions()` does more than parse raw payload fields.

For each polling cycle:

- parse the broker `/positions` payload as today
- fetch one quote per symbol used by the open positions
- fetch the account open-order surface needed to recover linked `stopLoss` / `takeProfit`
- map quote and order data back onto each `BrokerPositionInfo`

Expected normalization rules:

- `BUY` positions use live `bid` as `current_price`
- `SELL` positions use live `ask` as `current_price`
- `sl_price` and `tp_price` are recovered from linked protective orders when the position payload omits them

### Layer 2: tactical exit consumption

Keep [scheduler.py](/C:/Users/tommy.yeung\CursorProjects\prop-firm\prop-firm-pilot\src\scheduler\scheduler.py) as the only place that turns a broker position into a `TacticalExitSnapshot`, but make it explicitly depend on the enriched `BrokerPositionInfo` contract.

This means:

- `unrealized_r` will reflect the true market move
- state transitions into `PROTECTION`, `TREND_EXTENSION`, and `PROFIT_PROTECTION` become meaningful
- modify-action verification has a usable `sl_price` / `tp_price` surface

## Tactical Exit Coverage

The fix must cover all tactical exit action types:

- `MOVE_TO_BREAKEVEN`: requires correct `current_price` for `unrealized_r` and correct `sl_price` readback after modify
- `TRAIL_SL`: requires correct `current_price` and prior stop reference
- `REPRICE_TP`: requires correct `current_price` and usable current/original TP reference
- `PARTIAL_CLOSE`: requires correct tactical state classification so partial-close only triggers after genuine profit-state transitions
- `EXIT_NOW`: requires correct tactical state and reversal context so urgent exits are not delayed by a false `INITIAL_RISK` classification

## Data Sources

### Current price

Use [TradeLockerClient.get_quote()](/C:/Users/tommy.yeung\CursorProjects\prop-firm\prop-firm-pilot\src\execution\tradelocker_client.py#L277) as the authoritative source when the position payload lacks a real mark/current price.

For side-aware valuation:

- `BUY`: `current_price = bid`
- `SELL`: `current_price = ask`

This matches executable liquidation logic more closely than midpoint pricing.

### Stop-loss / take-profit

When `/positions` omits protective levels, enrich them from the broker order surface.

Implementation should prefer:

1. direct inline `stopLoss` / `takeProfit` if present
2. linked order identifiers from the position payload if resolvable
3. active open-order records for the same `positionId`

If no protective order can be resolved, leave the field `None` rather than inventing a price.

## Failure Policy

This feature should improve fidelity without creating a new hard dependency that blocks monitoring.

- If quote enrichment fails for one symbol, keep the raw parsed position and log a warning.
- If protective-order enrichment fails, leave `sl_price` / `tp_price` as-is and log a warning.
- If the whole enrichment pass fails, `get_open_positions()` should still return a best-effort parsed list unless the underlying broker call itself failed.

At the scheduler layer:

- modify-style tactical actions should continue to depend on normal close-control verification
- if readback remains incomplete, the action should be treated as unverified rather than silently accepted

## Performance Constraints

The fix must not multiply API traffic per position.

Required behavior:

- one `/positions` call per polling cycle
- one quote call per unique symbol, not per position
- one open-order retrieval per cycle, not per position

This keeps the enrichment cost bounded and avoids recreating the same 429 pressure pattern that already appeared elsewhere in the runtime.

## Testing Strategy

### TradeLocker client tests

Add tests proving:

- live array payload positions get side-aware `current_price` from quotes
- missing `sl_price` / `tp_price` are recovered from active protective orders
- enrichment degrades safely when quotes or order enrichment fail

### Tactical exit regression tests

Add tests proving enriched snapshots enable:

- `MOVE_TO_BREAKEVEN` after a real `0.3R` move
- `TRAIL_SL` and `REPRICE_TP` with correct modify/readback behavior
- `PARTIAL_CLOSE` and `EXIT_NOW` after proper state classification

## Out Of Scope

- changing tactical exit rules themselves
- changing scanner or entry tactical gate logic
- changing account configuration thresholds
- replacing the current `TradeLocker` quote API with websocket quotes
- broad refactors of the broker protocol unrelated to tactical exit reliability
