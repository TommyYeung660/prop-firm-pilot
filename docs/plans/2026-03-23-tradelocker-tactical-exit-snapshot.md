# TradeLocker Tactical Exit Snapshot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enrich `TradeLocker` open-position snapshots so the full tactical exit state machine uses live current price and protective-order data for `MOVE_TO_BREAKEVEN`, `TRAIL_SL`, `REPRICE_TP`, `PARTIAL_CLOSE`, and `EXIT_NOW`.

**Architecture:** Normalize `TradeLocker` positions inside the broker client by overlaying side-aware quotes and protective-order enrichment onto the raw `/positions` payload. Then add tactical-exit regression tests at the scheduler layer to prove all five action types behave correctly against the enriched snapshot surface.

**Tech Stack:** Python 3.10, async `httpx`, Pydantic v2 broker models, scheduler tactical exit pipeline, pytest, ruff.

---

## Execution Preconditions

1. Do not modify existing local `config/*.yaml` changes.
2. Do not edit or delete existing `docs/plans/*` files; only add the new plan/design files for this work.
3. Follow strict TDD: write the failing test first for each behavior change.
4. Keep production edits bounded to `TradeLockerClient`, tactical-exit scheduler wiring, and targeted tests.

---

### Task 1: Add failing TradeLocker enrichment tests

**Files:**
- Modify: `tests/test_tradelocker_client.py`

**Step 1: Write the failing tests**

Add tests for:

- live array payload positions using quote enrichment to set `current_price`
- `BUY` positions taking `bid` and `SELL` positions taking `ask`
- missing `sl_price` / `tp_price` being recovered from active protective orders
- quote/order enrichment failing safely without crashing `get_open_positions()`

Use cases like:

```python
async def test_open_positions_live_payload_uses_quote_for_current_price():
    positions = await client.get_open_positions()
    assert positions[0].current_price == 0.7919
```

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_tradelocker_client.py -k "current_price or protective orders or enrichment" -q
```

Expected: FAIL because `get_open_positions()` still falls back to `avgPrice` and does not recover SL/TP.

**Step 3: Write minimal implementation**

In `src/execution/tradelocker_client.py`:

- add helper methods to fetch active open orders needed for protective-order recovery
- add side-aware quote enrichment by unique symbol
- update `get_open_positions()` to merge raw position rows with quote/order overlays

**Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/test_tradelocker_client.py -k "current_price or protective orders or enrichment" -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/execution/tradelocker_client.py tests/test_tradelocker_client.py
git commit -m "fix: enrich tradelocker live positions for tactical exits"
```

---

### Task 2: Add failing tactical-exit regression tests for live enriched snapshots

**Files:**
- Modify: `tests/test_tactical_exit_scheduler.py`
- Modify: `tests/test_tactical_exit_execution.py`
- Modify: `tests/test_scheduler.py`

**Step 1: Write the failing tests**

Add tests proving:

- an enriched live position crosses `0.3R` and triggers `MOVE_TO_BREAKEVEN`
- enriched `sl_price` / `tp_price` support `TRAIL_SL` and `REPRICE_TP`
- state classification can advance to profit states that enable `PARTIAL_CLOSE` and `EXIT_NOW`
- tactical exit degrades safely when broker enrichment leaves critical fields unavailable

Use tests like:

```python
async def test_tactical_exit_cycle_uses_enriched_current_price_for_breakeven():
    await scheduler._run_tactical_exit_cycle([position], [intent])
    scheduler._handle_tactical_exit_evaluation.assert_awaited_once()
```

**Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_tactical_exit_scheduler.py tests/test_tactical_exit_execution.py tests/test_scheduler.py -k "tactical and enriched" -q
```

Expected: FAIL because scheduler tactical-exit tests do not yet cover the live `TradeLocker` snapshot defect.

**Step 3: Write minimal implementation**

In `src/scheduler/scheduler.py`:

- add minimal tactical guardrails needed to consume enriched position data safely
- keep rule selection in the pure tactical exit engine unchanged
- ensure modify-style actions continue to require verified readback

**Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/test_tactical_exit_scheduler.py tests/test_tactical_exit_execution.py tests/test_scheduler.py -k "tactical and enriched" -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/scheduler/scheduler.py tests/test_tactical_exit_scheduler.py tests/test_tactical_exit_execution.py tests/test_scheduler.py
git commit -m "fix: cover tactical exits with enriched tradelocker state"
```

---

### Task 3: Run focused integration verification

**Files:**
- Modify: `src/execution/tradelocker_client.py`
- Modify: `src/scheduler/scheduler.py`
- Modify: `tests/test_tradelocker_client.py`
- Modify: `tests/test_tactical_exit_scheduler.py`
- Modify: `tests/test_tactical_exit_execution.py`
- Modify: `tests/test_scheduler.py`

**Step 1: Run focused tests**

Run:

```bash
uv run pytest tests/test_tradelocker_client.py tests/test_tactical_exit_scheduler.py tests/test_tactical_exit_execution.py tests/test_scheduler.py
```

Expected: PASS.

**Step 2: Run lint**

Run:

```bash
uv run ruff check src/execution/tradelocker_client.py src/scheduler/scheduler.py tests/test_tradelocker_client.py tests/test_tactical_exit_scheduler.py tests/test_tactical_exit_execution.py tests/test_scheduler.py
```

Expected: `All checks passed!`

**Step 3: Inspect diff**

Run:

```bash
git diff -- src/execution/tradelocker_client.py src/scheduler/scheduler.py tests/test_tradelocker_client.py tests/test_tactical_exit_scheduler.py tests/test_tactical_exit_execution.py tests/test_scheduler.py
```

Expected: only tactical-exit snapshot enrichment and its tests are included.

**Step 4: Commit**

```bash
git add src/execution/tradelocker_client.py src/scheduler/scheduler.py tests/test_tradelocker_client.py tests/test_tactical_exit_scheduler.py tests/test_tactical_exit_execution.py tests/test_scheduler.py
git commit -m "fix: normalize tradelocker snapshots for tactical exits"
```

---

### Task 4: Verify against live broker data

**Files:**
- No file changes required unless the live validation exposes a reproducible defect

**Step 1: Run a read-only verification script**

Run a script that:

- logs into `TradeLocker`
- fetches live open positions
- fetches quotes for symbols with open positions
- prints `open_price`, enriched `current_price`, `sl_price`, `tp_price`, and computed `unrealized_r`

**Step 2: Compare with expected tactical behavior**

Confirm:

- profitable positions no longer report `current_price == open_price` unless the market is actually flat
- positions that should be in `PROTECTION` are no longer stuck in `INITIAL_RISK`
- protective prices are present when broker-side protection exists

**Step 3: If validation is clean, push**

```bash
git push origin main
```

Expected: push succeeds with only this tactical-exit snapshot patch.
