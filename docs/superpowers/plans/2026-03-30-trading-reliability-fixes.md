# Trading Reliability Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the three proven production failure modes from the 2026-03-26 to 2026-03-30 log bundle: blank `position_id` opens, stale opened-intent blockage, and spread-hard-fail degradation into live execution.

**Architecture:** Fail closed at the broker boundary, not after state has already been persisted. Keep scanner capacity logic based on blocking positions rather than every `opened` row, while preserving `opened` rows for reconciliation in the position monitor. Tighten tactical policy so a spread hard-gate failure can retry, but can never auto-degrade into execution.

**Tech Stack:** Python 3.10, asyncio, sqlite3, pytest, AsyncMock, loguru

---

## Root Cause Summary

1. `TradeLockerClient.open_position()` currently returns `success=True` even when recovery cannot resolve a broker `positionId`, which lets `ExecutionEngine` persist `status='opened'` with an empty id and then fail to set SL/TP.
2. `DecisionStore.has_active_position_for_symbol()` treats every `opened` row as blocking, including stale rows with blank `position_id`; `Scheduler` therefore keeps skipping symbols even after the broker reports no live position.
3. `TacticalValidator._hard_gate_wait_allows_degrade()` currently allows degrade for `spread.fail.ratio_too_wide`, so retry exhaustion can still execute a trade that has continuously failed a hard spread gate.

## File Map

- Modify: `src/execution/tradelocker_client.py`
  Responsibility: broker response normalization and order/position recovery.
- Modify: `src/execution/engine.py`
  Responsibility: intent state transitions around live execution and SL/TP setup.
- Modify: `src/decision_store/sqlite_store.py`
  Responsibility: opened-intent reconciliation helpers and scanner-facing blocking queries.
- Modify: `src/scheduler/scheduler.py`
  Responsibility: scanner admission, position-monitor reconciliation, and orphan cleanup.
- Modify: `src/decision/tactical_validator.py`
  Responsibility: tactical retry/degrade policy classification.
- Test: `tests/test_tradelocker_client.py`
- Test: `tests/test_engine.py`
- Test: `tests/test_decision_store.py`
- Test: `tests/test_scheduler.py`
- Test: `tests/test_tactical_validator.py`

## Out Of Scope For This Plan

- Scanner bundle generation lag on session open. The evidence shows a temporary target-date miss that self-healed within one hour, but the root cause may live in the sibling scanner repo rather than this repo.
- Strategy threshold tuning for low-confidence AUDUSD repetition. That is a profitability/control problem, but this plan is limited to reliability and safety regressions already proven by production logs.

### Task 1: Fail Closed When TradeLocker Cannot Resolve A Position ID

**Files:**
- Modify: `src/execution/tradelocker_client.py`
- Modify: `src/execution/engine.py`
- Test: `tests/test_tradelocker_client.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1: Write the failing TradeLocker client test**

```python
async def test_market_order_open_returns_failure_when_position_recovery_stays_unresolved(
    client: TradeLockerClient,
) -> None:
    client._resolve_symbol_meta = AsyncMock(
        return_value={
            "tradableInstrumentId": "TI-EURUSD",
            "infoRouteId": "ROUTE-INFO",
            "tradeRouteId": "ROUTE-TRADE",
        }
    )
    client._account_request = AsyncMock(return_value={"orderId": "ORD-1"})
    client._enrich_order_response = AsyncMock(
        return_value={"orderId": "ORD-1", "positionId": "", "openPrice": 0}
    )

    result = await client.open_position(symbol="EURUSD", side="BUY", volume=0.10)

    assert result.success is False
    assert result.position_id == ""
    assert "position recovery unresolved" in result.message
    assert result.raw_response["orderId"] == "ORD-1"
```

- [ ] **Step 2: Write the failing execution-engine guard test**

```python
async def test_execute_ready_intent_marks_failed_when_broker_returns_blank_position_id(
    store: DecisionStore,
    config: AppConfig,
    mock_guard: MagicMock,
    mock_matchtrader: AsyncMock,
    mock_sizer: MagicMock,
) -> None:
    mock_matchtrader.open_position.return_value = MagicMock(
        success=True,
        position_id="",
        message="Position opened successfully",
        raw_response={"orderId": "ORD-1", "positionId": ""},
    )
    mock_matchtrader.get_open_positions.return_value = []
    mock_matchtrader.get_balance.return_value = MagicMock(balance=50000.0, equity=50000.0)
    engine = ExecutionEngine(
        store=store,
        guard=mock_guard,
        matchtrader=mock_matchtrader,
        sizer=mock_sizer,
        config=config,
    )

    _make_ready_intent(store, symbol="EURUSD", side="BUY")
    await engine.execute_ready_intents()

    failed = store.get_pending_intents()
    assert failed == []
    intents = store.get_intents_by_date(Scheduler._today_str())
    assert intents[0].status == "failed"
    assert "missing broker position_id" in (intents[0].execution_error or "")
```

- [ ] **Step 3: Run the targeted tests to verify both fail first**

Run:

```bash
uv run pytest tests/test_tradelocker_client.py -k unresolved
uv run pytest tests/test_engine.py -k blank_position_id
```

Expected:

```text
FAILED tests/test_tradelocker_client.py::test_market_order_open_returns_failure_when_position_recovery_stays_unresolved
FAILED tests/test_engine.py::test_execute_ready_intent_marks_failed_when_broker_returns_blank_position_id
```

- [ ] **Step 4: Implement fail-closed broker normalization and defensive engine guard**

```python
# src/execution/tradelocker_client.py
response_raw = await self._enrich_order_response(
    response_raw=response_raw,
    symbol=symbol,
    side=side,
    volume=volume,
    tradable_instrument_id=meta["tradableInstrumentId"],
)
resolved_position_id = str(
    response_raw.get("positionId", response_raw.get("orderId", response_raw.get("id", "")))
).strip()
if not resolved_position_id:
    logger.error(
        "TradeLocker: unresolved position recovery after market order {} {} {}",
        side,
        symbol,
        volume,
    )
    return BrokerOrderResult(
        success=False,
        position_id="",
        message="TradeLocker position recovery unresolved",
        raw_response=response_raw,
    )
return BrokerOrderResult(
    success=True,
    position_id=resolved_position_id,
    message="Position opened successfully",
    raw_response=response_raw,
)
```

```python
# src/execution/engine.py
if order.success:
    resolved_position_id = str(order.position_id or "").strip()
    if not resolved_position_id:
        await asyncio.to_thread(
            self._store.mark_failed,
            intent_id,
            "Broker execution returned success but missing broker position_id",
        )
        self._log_trade_event(
            "TRADE_FAILED",
            {
                "intent_id": intent_id,
                "symbol": symbol,
                "side": side,
                "reason": "missing broker position_id",
            },
        )
        logger.error(
            "ExecutionEngine: broker returned success without position_id for intent {}",
            intent_id,
        )
        return
    await asyncio.to_thread(self._store.mark_opened, intent_id, resolved_position_id)
```

- [ ] **Step 5: Re-run the targeted tests and the surrounding broker/execution suites**

Run:

```bash
uv run pytest tests/test_tradelocker_client.py
uv run pytest tests/test_engine.py -k "position_id or sl_tp"
```

Expected:

```text
... passed
```

- [ ] **Step 6: Commit**

```bash
git add tests/test_tradelocker_client.py tests/test_engine.py src/execution/tradelocker_client.py src/execution/engine.py
git commit -m "fix: fail closed when tradelocker position recovery is unresolved"
```

### Task 2: Reconcile Orphaned Opened Intents And Stop Scanner Blockage

**Files:**
- Modify: `src/decision_store/sqlite_store.py`
- Modify: `src/scheduler/scheduler.py`
- Test: `tests/test_decision_store.py`
- Test: `tests/test_scheduler.py`

- [ ] **Step 1: Write the failing DecisionStore query test**

```python
def test_has_active_position_for_symbol_ignores_stale_opened_row_without_position_id(
    store: DecisionStore,
) -> None:
    intent = TradeIntent(trade_date="2026-03-30", symbol="AUDUSD")
    store.insert_intent(intent)
    store.claim_next_pending("llm-0")
    store.update_intent_decision(intent.id, "SELL", 20.0, 40.0, "report", "{}")
    store.mark_ready_for_exec(intent.id)
    store.mark_executing(intent.id)
    store.mark_opened(intent.id, position_id="")
    store._conn.execute(
        "UPDATE intents SET executed_at = ? WHERE id = ?",
        ("2026-03-28T04:09:29+00:00", intent.id),
    )
    store._conn.commit()

    assert store.has_active_position_for_symbol("AUDUSD", unresolved_grace_seconds=90) is False
```

- [ ] **Step 2: Write the failing scheduler reconciliation test**

```python
async def test_position_monitor_marks_blank_opened_intent_failed_after_repair_grace(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
) -> None:
    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )
    intent = TradeIntent(trade_date=Scheduler._today_str(), symbol="AUDUSD")
    store.insert_intent(intent)
    store.claim_next_pending("llm-0")
    store.update_intent_decision(intent.id, "SELL", 20.0, 40.0, "report", "{}")
    store.mark_ready_for_exec(intent.id)
    store.mark_executing(intent.id)
    store.mark_opened(intent.id, position_id="")
    store._conn.execute(
        "UPDATE intents SET executed_at = ? WHERE id = ?",
        ("2026-03-28T04:09:29+00:00", intent.id),
    )
    store._conn.commit()

    await sched._expire_unresolved_opened_intents(open_positions=[], opened_intents=store.get_active_positions())

    failed = store.get_intent(intent.id)
    assert failed is not None
    assert failed.status == "failed"
    assert "opened without recoverable broker position_id" in (failed.execution_error or "")
```

- [ ] **Step 3: Run the targeted tests to confirm the stale-row behavior fails today**

Run:

```bash
uv run pytest tests/test_decision_store.py -k stale_opened_row_without_position_id
uv run pytest tests/test_scheduler.py -k blank_opened_intent_failed_after_repair_grace
```

Expected:

```text
FAILED ...
```

- [ ] **Step 4: Add explicit opened-intent reconciliation helpers to the store**

```python
# src/decision_store/sqlite_store.py
def has_active_position_for_symbol(
    self,
    symbol: str,
    *,
    unresolved_grace_seconds: int = 90,
) -> bool:
    cutoff = _dt_to_str(datetime.now(timezone.utc) - timedelta(seconds=unresolved_grace_seconds))
    row = self._conn.execute(
        """SELECT 1 FROM intents
           WHERE symbol = :symbol
             AND status = 'opened'
             AND (
                 TRIM(COALESCE(position_id, '')) != ''
                 OR executed_at IS NULL
                 OR executed_at >= :cutoff
             )
           LIMIT 1""",
        {"symbol": symbol, "cutoff": cutoff},
    ).fetchone()
    return row is not None

def count_blocking_open_positions(self, *, unresolved_grace_seconds: int = 90) -> int:
    cutoff = _dt_to_str(datetime.now(timezone.utc) - timedelta(seconds=unresolved_grace_seconds))
    row = self._conn.execute(
        """SELECT COUNT(*) AS cnt FROM intents
           WHERE status = 'opened'
             AND (
                 TRIM(COALESCE(position_id, '')) != ''
                 OR executed_at IS NULL
                 OR executed_at >= :cutoff
             )""",
        {"cutoff": cutoff},
    ).fetchone()
    return row["cnt"] if row else 0

def mark_opened_reconciliation_failed(self, intent_id: str, reason: str) -> None:
    now = datetime.now(timezone.utc)
    updated = self._conn.execute(
        """UPDATE intents
           SET status = 'failed',
               execution_error = :reason,
               executed_at = COALESCE(executed_at, :executed_at)
           WHERE id = :id AND status = 'opened'""",
        {"reason": reason, "executed_at": _dt_to_str(now), "id": intent_id},
    ).rowcount
    if not updated:
        self._conn.rollback()
        raise InvalidTransitionError(
            f"Cannot mark {intent_id} reconciliation-failed: not in 'opened' state"
        )
    self._conn.execute(
        """UPDATE decisions
           SET status = 'failed',
               failure_reason = :reason
           WHERE intent_id = :intent_id""",
        {"reason": reason, "intent_id": intent_id},
    )
    self._conn.commit()
```

- [ ] **Step 5: Reconcile stale blank-position intents inside the scheduler**

```python
# src/scheduler/scheduler.py
OPENED_POSITION_ID_GRACE_SECONDS = 90

open_count = await asyncio.to_thread(
    self._store.count_blocking_open_positions,
    unresolved_grace_seconds=OPENED_POSITION_ID_GRACE_SECONDS,
)

has_active = await asyncio.to_thread(
    self._store.has_active_position_for_symbol,
    signal.instrument,
    unresolved_grace_seconds=OPENED_POSITION_ID_GRACE_SECONDS,
)
```

```python
# src/scheduler/scheduler.py
async def _expire_unresolved_opened_intents(
    self,
    open_positions: list[Any],
    opened_intents: list[TradeIntent],
) -> None:
    now = self._now_utc()
    claimed_position_ids = {
        str(getattr(pos, "position_id", "")).strip() for pos in open_positions
    }
    for intent in opened_intents:
        current_position_id = str(intent.position_id or "").strip()
        if current_position_id:
            continue
        executed_at = intent.executed_at
        if executed_at is None:
            continue
        if (now - executed_at).total_seconds() < OPENED_POSITION_ID_GRACE_SECONDS:
            continue
        matched = self._match_open_position_for_intent(
            intent,
            open_positions,
            claimed_position_ids=claimed_position_ids,
        )
        if matched is not None:
            continue
        reason = "Intent opened without recoverable broker position_id"
        await asyncio.to_thread(self._store.mark_opened_reconciliation_failed, intent.id, reason)
        logger.error(
            "Position monitor: expiring unresolved opened intent {} ({}) after grace window",
            intent.id,
            intent.symbol,
        )
```

```python
# src/scheduler/scheduler.py
await self._repair_missing_opened_position_ids(open_positions, opened_intents)
await self._expire_unresolved_opened_intents(open_positions, opened_intents)
```

- [ ] **Step 6: Re-run the targeted tests and the broader scheduler/store suites**

Run:

```bash
uv run pytest tests/test_decision_store.py -k "repair_opened_position_id or stale_opened_row_without_position_id"
uv run pytest tests/test_scheduler.py -k "blank_opened_intent_failed_after_repair_grace or tactical_wait"
```

Expected:

```text
... passed
```

- [ ] **Step 7: Commit**

```bash
git add tests/test_decision_store.py tests/test_scheduler.py src/decision_store/sqlite_store.py src/scheduler/scheduler.py
git commit -m "fix: reconcile orphan opened intents and unblock scanner symbols"
```

### Task 3: Prevent Spread Hard-Gate WAIT From Degrading Into Execution

**Files:**
- Modify: `src/decision/tactical_validator.py`
- Test: `tests/test_tactical_validator.py`
- Test: `tests/test_scheduler.py`

- [ ] **Step 1: Add the failing tactical-validator unit test**

```python
def test_spread_hard_gate_wait_disallows_degrade(self) -> None:
    config = TacticalConfig()
    validator = TacticalValidator(config)
    data = TacticalData(
        bars_5min=pd.DataFrame(
            [
                {
                    "datetime": datetime.now(timezone.utc) - timedelta(minutes=5 - idx),
                    "open": 1.1000,
                    "high": 1.1008,
                    "low": 1.0995,
                    "close": 1.1002,
                }
                for idx in range(20)
            ]
        ),
        current_spread=0.00050,
        typical_spread=0.00015,
        latest_bar_time=datetime.now(timezone.utc),
        quote_source="broker_quote",
        data_source="mixed",
    )

    result = validator.evaluate(side="BUY", data=data)

    assert result.action == "WAIT"
    assert result.summary_reason_code == "spread.fail.ratio_too_wide"
    assert result.policy_hints["degrade_allowed"] is False
```

- [ ] **Step 2: Add the failing scheduler regression test**

```python
async def test_spread_hard_gate_retry_expiry_times_out_instead_of_execute_degraded(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
) -> None:
    from src.decision.tactical_validator import GateResult, TacticalResult

    config.tactical.enabled = True
    config.tactical.shadow_mode = False
    config.tactical.retry.max_retries = 1
    config.tactical.retry.interval_seconds = 0
    config.tactical.retry.jitter_seconds = 0
    config.tactical.retry.expire_action = "degrade"

    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )
    intent = TradeIntent(trade_date=Scheduler._today_str(), symbol="USDJPY", scanner_score=0.7)
    store.insert_intent(intent)
    claimed = store.claim_next_pending("llm-0")
    assert claimed is not None

    spread_wait = TacticalResult(
        action="WAIT",
        resolution="RETRY_PENDING",
        detail="Hard gates failed: spread",
        summary_reason_code="spread.fail.ratio_too_wide",
        hard_gates=[
            GateResult(
                gate_name="spread",
                passed=False,
                status="FAIL",
                reason_code="spread.fail.ratio_too_wide",
                detail="spread_ratio=8.69, limit=2.0×",
            )
        ],
        policy_hints={"retryable": True, "degrade_allowed": False},
    )

    with (
        patch.object(sched, "_run_tactical_validation", new_callable=AsyncMock) as mock_tac,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_tac.side_effect = [spread_wait, spread_wait]
        await sched._process_claimed_intent("llm-0", claimed)

    final = store.get_intent(claimed.id)
    assert final is not None
    assert final.status == "timed_out"
```

- [ ] **Step 3: Run the failing tests**

Run:

```bash
uv run pytest tests/test_tactical_validator.py -k spread_hard_gate_wait_disallows_degrade
uv run pytest tests/test_scheduler.py -k spread_hard_gate_retry_expiry_times_out_instead_of_execute_degraded
```

Expected:

```text
FAILED ...
```

- [ ] **Step 4: Tighten the degrade policy classifier**

```python
# src/decision/tactical_validator.py
def _hard_gate_wait_allows_degrade(self, gate_results: list[GateResult]) -> bool:
    """Return whether a hard-gate WAIT may safely degrade into execution."""
    non_degradable_prefixes = ("market_data.", "data.reject.", "freshness.", "spread.")
    non_degradable_reason_codes = {
        "atr.fail.insufficient_1h_data",
    }
    for gate in gate_results:
        if gate.passed:
            continue
        reason_code = gate.reason_code or ""
        if reason_code.startswith(non_degradable_prefixes):
            return False
        if reason_code in non_degradable_reason_codes:
            return False
    return True
```

- [ ] **Step 5: Re-run validator and scheduler tactical suites**

Run:

```bash
uv run pytest tests/test_tactical_validator.py
uv run pytest tests/test_scheduler.py -k "tactical_wait or execute_degraded"
```

Expected:

```text
... passed
```

- [ ] **Step 6: Commit**

```bash
git add tests/test_tactical_validator.py tests/test_scheduler.py src/decision/tactical_validator.py
git commit -m "fix: disallow degrade on spread hard-gate failures"
```

## Verification Checklist

- `uv run pytest tests/test_tradelocker_client.py`
- `uv run pytest tests/test_engine.py`
- `uv run pytest tests/test_decision_store.py`
- `uv run pytest tests/test_scheduler.py -k "tactical or blank_opened or active_position"`
- `uv run pytest tests/test_tactical_validator.py`
- `uv run ruff check src/ tests/`

## Expected Production Outcomes

- No new `Intent ... opened with position ` blank-id log lines.
- No new `SL/TP NOT SET` lines caused by missing `position_id`.
- Symbols with orphaned blank-id rows stop blocking scanner admission after the grace window.
- `spread.fail.ratio_too_wide` intents time out after retries instead of reaching `EXECUTE_DEGRADED`.

## Deferred Follow-Up After This Plan Lands

- Add a dedicated operational metric for `opened_without_position_id`.
- Decide whether scanner session-open target-date lag belongs in this repo or the sibling scanner repo.
- Revisit low-confidence AUDUSD thresholding only after reliability fixes are verified in production.
