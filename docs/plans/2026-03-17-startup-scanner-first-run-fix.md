# Startup Scanner First-Run Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 讓 production 啟動後第一輪 scanner 在 `quote fresh` 但 `5m` tactical bars 尚未形成時，不再直接失效，而是建立 intent 並進入 `tactical_pending`，等待第一根 websocket `5m` closed bar 後自動重試。

**Architecture:** 保留現有 `Scanner -> Intent -> LLM -> Tactical -> Execution` 主幹，不另建新 queue。核心改動是把 `MarketDataHub.get_entry_readiness()` 對 `bars_5m_unavailable` 的語義從 scanner-stage hard block，縮窄成「startup warmup retryable」；同時在 `TacticalValidator` 顯式把這個狀態轉成 `WAIT / RETRY_PENDING`，避免因 soft-gate skipped-pass 而誤放行。

**Tech Stack:** Python 3.10、asyncio、pytest / pytest-asyncio、loguru、pandas、pydantic、httpx、websockets

---

### Task 1: Reclassify startup `5m` gap in `MarketDataHub`

**Files:**
- Modify: `src/data/market_data_hub.py`
- Test: `tests/data/test_market_data_hub.py`

**Step 1: Write the failing tests**

Add two focused tests in `tests/data/test_market_data_hub.py`:

```python
@pytest.mark.asyncio
async def test_entry_readiness_marks_startup_5m_gap_as_retryable() -> None:
    aggregator = FXTickAggregator()
    tick_time = datetime(2026, 3, 17, 6, 42, 10, tzinfo=timezone.utc)
    aggregator.add_tick(_tick("EURUSD", 1.10, 1.1002, tick_time))

    client = EODHDFXWebSocketClient(api_token="token", symbols=["EURUSD"])
    client._connected = True
    client._record_tick(_tick("EURUSD", 1.10, 1.1002, tick_time))

    hub = MarketDataHub(
        aggregator=aggregator,
        websocket_client=client,
        rest_provider=DummyProvider([]),
        symbols=["EURUSD"],
        now_provider=lambda: datetime(2026, 3, 17, 6, 42, 30, tzinfo=timezone.utc),
    )

    readiness = await hub.get_entry_readiness("EURUSD")

    assert readiness.entry_safe is True
    assert readiness.requires_tactical_retry is True
    assert readiness.pending_reason == "market_data.startup_5m_bar_pending"
    assert readiness.quote_available is True
    assert readiness.bars_5m_fresh is False


@pytest.mark.asyncio
async def test_entry_readiness_still_blocks_when_quote_is_missing() -> None:
    hub = ...
    readiness = await hub.get_entry_readiness("EURUSD")

    assert readiness.entry_safe is False
    assert readiness.requires_tactical_retry is False
    assert readiness.block_reason == "market_data.quote_unavailable"
```

**Step 2: Run the focused tests to verify they fail**

Run:

```bash
uv run pytest tests/data/test_market_data_hub.py -k "entry_readiness" -v
```

Expected:
- 新增的 startup retryable test 失敗，因為目前 `bars_5m_unavailable` 直接映射成 `entry_safe=False`

**Step 3: Write the minimal implementation**

In `src/data/market_data_hub.py`:

```python
@dataclass
class EntryReadinessResult:
    symbol: str
    entry_safe: bool
    block_reason: str
    requires_tactical_retry: bool = False
    pending_reason: str = ""
    ...
```

Update `get_entry_readiness()` so this state:

- `quote_available == True`
- websocket state is `healthy`
- `bars_5m_fresh == False`

returns:

```python
EntryReadinessResult(
    symbol=symbol,
    entry_safe=True,
    block_reason="",
    requires_tactical_retry=True,
    pending_reason="market_data.startup_5m_bar_pending",
    ...
)
```

Keep these as hard blocks:

- `market_data.quote_unavailable`
- `market_data.feed_degraded`
- `market_data.readiness_error`

Do not treat missing `5m` bars as retryable when websocket is unhealthy or quote is absent.

**Step 4: Run the market-data tests**

Run:

```bash
uv run pytest tests/data/test_market_data_hub.py tests/data/test_fx_websocket_client.py -q
```

Expected:
- All tests pass
- Existing websocket / hub behavior remains intact

**Step 5: Commit**

```bash
git add tests/data/test_market_data_hub.py src/data/market_data_hub.py
git commit -m "fix: classify startup 5m gaps as tactical retryable"
```

### Task 2: Let scanner loop admit retryable first-run candidates

**Files:**
- Modify: `src/scheduler/scheduler.py`
- Test: `tests/test_scheduler.py`

**Step 1: Write the failing test**

Add a scheduler regression in `tests/test_scheduler.py`:

```python
async def test_scanner_creates_intent_when_market_data_gap_is_retryable(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
    tmp_path: Path,
) -> None:
    sched = Scheduler(...)
    sched._market_data_ready = True
    sched._market_data_hub = MagicMock()
    sched._market_data_hub.get_entry_readiness = AsyncMock(
        return_value=MagicMock(
            entry_safe=True,
            requires_tactical_retry=True,
            pending_reason="market_data.startup_5m_bar_pending",
            quote_available=True,
            quote_source="websocket_cache",
            bars_5m_source="websocket_cache",
            bars_1h_source="warmup_cache",
        )
    )
    mock_scanner.run_pipeline.return_value = [_make_mock_signal("EURUSD")]

    await _run_loop_once(sched, sched._scanner_loop())

    intents = store.get_intents_by_date(Scheduler._today_str())
    assert len(intents) == 1
```

Also assert the journal contains a non-skip event capturing the pending reason, for example:

```python
assert admitted_event["pending_reason"] == "market_data.startup_5m_bar_pending"
```

**Step 2: Run the focused scheduler test to verify it fails**

Run:

```bash
uv run pytest tests/test_scheduler.py -k "market_data_gap_is_retryable" -v
```

Expected:
- FAIL because current scanner loop always `continue`s on any `entry_safe=False`-style market-data guard outcome and never creates the intent

**Step 3: Write the minimal implementation**

In `src/scheduler/scheduler.py`, update the scanner-loop branch around `_get_entry_readiness()`:

```python
entry_readiness = await self._get_entry_readiness(signal.instrument)
if entry_readiness is not None:
    if not entry_readiness.entry_safe:
        ...  # existing SCANNER_SKIP hard block
        continue
    if getattr(entry_readiness, "requires_tactical_retry", False):
        self._log_trade_event(
            "SCANNER_ADMITTED",
            {
                "symbol": signal.instrument,
                "reason": "market_data_startup_retryable",
                "pending_reason": entry_readiness.pending_reason,
                "quote_source": entry_readiness.quote_source,
                "bars_5m_source": entry_readiness.bars_5m_source,
                "bars_1h_source": entry_readiness.bars_1h_source,
            },
        )
```

Do not skip intent creation when readiness is retryable.

Keep existing idempotency checks untouched:

- `intent_exists()`
- `has_active_position_for_symbol()`
- capacity and cooldown checks

**Step 4: Run scheduler tests**

Run:

```bash
uv run pytest tests/test_scheduler.py -k "market_data_entry_not_safe or market_data_gap_is_retryable" -v
```

Expected:
- Hard-block test still passes for true quote/feed failures
- New retryable-gap test passes and produces one intent

**Step 5: Commit**

```bash
git add tests/test_scheduler.py src/scheduler/scheduler.py
git commit -m "fix: admit retryable first-run scanner candidates"
```

### Task 3: Force tactical `WAIT` until the first usable `5m` bar exists

**Files:**
- Modify: `src/decision/tactical_validator.py`
- Test: `tests/test_tactical_validator.py`
- Test: `tests/test_scheduler.py`

**Step 1: Write the failing tests**

Add a new tactical validator regression:

```python
def test_waits_for_first_5m_bar_when_quote_is_fresh_but_bars_are_missing() -> None:
    config = TacticalConfig()
    validator = TacticalValidator(config)
    data = TacticalData(
        bars_5min=pd.DataFrame(),
        bars_1h=pd.DataFrame(),
        current_spread=0.0,
        typical_spread=0.00015,
        latest_bar_time=datetime.now(timezone.utc) - timedelta(seconds=20),
        quote_source="websocket_cache",
        bars_5min_source="",
        bars_1h_source="",
        data_source="websocket_cache",
    )

    result = validator.evaluate(side="BUY", data=data)

    assert result.action == "WAIT"
    assert result.resolution == "RETRY_PENDING"
    assert result.summary_reason_code == "market_data.startup_5m_bar_pending"
```

Add one scheduler lifecycle regression:

```python
async def test_tactical_retry_promotes_intent_after_first_5m_bar_arrives(...) -> None:
    ...
    first_result = TacticalResult(
        action="WAIT",
        resolution="RETRY_PENDING",
        summary_reason_code="market_data.startup_5m_bar_pending",
    )
    second_result = TacticalResult(
        action="PASS",
        resolution="EXECUTE_NOW",
        summary_reason_code="tactical.pass.all_gates_aligned",
    )
    ...
    assert store.get_intent(intent.id).status == "ready_for_exec"
```

**Step 2: Run the focused tests to verify they fail**

Run:

```bash
uv run pytest tests/test_tactical_validator.py -k "first_5m_bar" -v
uv run pytest tests/test_scheduler.py -k "tactical_retry_promotes_intent_after_first_5m_bar_arrives" -v
```

Expected:
- Tactical validator test fails because current soft-gate skipped-pass behavior can resolve to `PASS`

**Step 3: Write the minimal implementation**

In `src/decision/tactical_validator.py`, add an explicit pre-hard-gate branch near the top of `evaluate()`:

```python
if (
    data.latest_bar_time is not None
    and data.bars_5min.empty
    and data.quote_source == "websocket_cache"
):
    return TacticalResult(
        action="WAIT",
        resolution="RETRY_PENDING",
        detail="Awaiting first websocket 5m closed bar after startup",
        summary_reason_code="market_data.startup_5m_bar_pending",
        policy_hints=self._default_policy_hints(retryable=True, degrade_allowed=True),
        provenance=self._build_provenance(data),
    )
```

This branch must happen before the soft-gate skipped-pass logic.

Do not change:

- hard fail when all tactical inputs are missing
- normal PASS path when `bars_5min` actually exists
- true degraded / quote-missing behavior

**Step 4: Run the tactical tests**

Run:

```bash
uv run pytest tests/test_tactical_validator.py tests/test_scheduler.py -k "first_5m_bar or tactical_retry_promotes_intent_after_first_5m_bar_arrives" -v
```

Expected:
- Startup warmup now yields deterministic `RETRY_PENDING`
- Retry path promotes the existing intent instead of waiting for the next scanner loop

**Step 5: Commit**

```bash
git add tests/test_tactical_validator.py tests/test_scheduler.py src/decision/tactical_validator.py
git commit -m "fix: wait for first websocket 5m bar before execution"
```

### Task 4: Tighten diagnostics and run targeted verification

**Files:**
- Modify: `src/scheduler/scheduler.py`
- Modify: `tests/test_scheduler.py`

**Step 1: Write the failing test**

Add a diagnostics regression to `tests/test_scheduler.py`:

```python
async def test_scanner_admitted_event_records_startup_retryable_context(...) -> None:
    ...
    admitted_event = next(e for e in events if e["type"] == "SCANNER_ADMITTED")
    assert admitted_event["reason"] == "market_data_startup_retryable"
    assert admitted_event["pending_reason"] == "market_data.startup_5m_bar_pending"
    assert admitted_event["quote_source"] == "websocket_cache"
```

**Step 2: Run the focused diagnostics test to verify it fails**

Run:

```bash
uv run pytest tests/test_scheduler.py -k "SCANNER_ADMITTED" -v
```

Expected:
- FAIL because this admission event does not exist yet

**Step 3: Write the minimal implementation**

In `src/scheduler/scheduler.py`:

- Keep the existing `SCANNER_SKIP` path for true hard blocks
- Add the new `SCANNER_ADMITTED` event only for startup retryable cases
- Include enough fields for operator triage:

```python
{
    "symbol": signal.instrument,
    "reason": "market_data_startup_retryable",
    "pending_reason": entry_readiness.pending_reason,
    "quote_source": entry_readiness.quote_source,
    "bars_5m_source": entry_readiness.bars_5m_source,
    "bars_1h_source": entry_readiness.bars_1h_source,
}
```

**Step 4: Run the targeted verification suite**

Run:

```bash
uv run pytest tests/data/test_market_data_hub.py tests/data/test_fx_websocket_client.py tests/test_tactical_validator.py tests/test_scheduler.py -q
```

Run:

```bash
uv run ruff check src/data/market_data_hub.py src/scheduler/scheduler.py src/decision/tactical_validator.py tests/data/test_market_data_hub.py tests/test_tactical_validator.py tests/test_scheduler.py
```

Expected:
- All targeted tests pass
- Ruff returns `All checks passed!`

**Step 5: Commit**

```bash
git add tests/test_scheduler.py src/scheduler/scheduler.py
git commit -m "feat: add startup retry diagnostics for first-run scanner recovery"
```
