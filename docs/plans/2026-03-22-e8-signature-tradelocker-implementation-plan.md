# E8 Signature TradeLocker Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `TradeLocker` as a broker backend for `v1.5.0_stable` while preserving the current macOS server deployment model and maintaining `MatchTrader` compatibility.

**Architecture:** Introduce a broker-neutral execution contract, adapt the existing `MatchTrader` path to that contract, then implement a new `TradeLockerClient` behind the same interface. Keep the current scheduler, execution, tactical-exit, and legacy daily-cycle behavior intact by minimizing changes above the broker boundary.

**Tech Stack:** Python 3.10, async broker clients, Pydantic v2 models, httpx or provider-specific HTTP client, pytest, ruff, SQLite-backed scheduler/execution pipeline.

---

## Execution Preconditions

Before implementation starts:

1. Work from a clean branch or isolated worktree.
2. Follow strict TDD for every behavior change.
3. Keep all production changes bounded to the files listed below.
4. Do not change scanner logic, TradingAgents logic, or EODHD tactical-bar sourcing as part of this plan.

---

### Task 1: Introduce Broker-Neutral Models And Protocol

**Files:**
- Create: `src/execution/broker_models.py`
- Create: `src/execution/broker_protocol.py`
- Modify: `src/execution/matchtrader_client.py`
- Modify: `tests/test_matchtrader_client.py`
- Create: `tests/test_broker_models.py`

**Step 1: Write the failing model/protocol tests**

Add tests for:

- broker-neutral models carrying the fields currently consumed by execution and scheduler code
- `MatchTraderClient` responses still satisfying those model shapes

Use tests like:

```python
def test_broker_position_info_exposes_execution_fields():
    position = BrokerPositionInfo(
        position_id="POS-1",
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
        open_price=1.1,
        current_price=1.101,
        profit=10.0,
    )
    assert position.position_id == "POS-1"
```

**Step 2: Run tests to verify RED**

Run:

```bash
uv run pytest tests/test_broker_models.py tests/test_matchtrader_client.py -k "broker or model" -q
```

Expected: FAIL because the broker-neutral models/protocol do not exist yet.

**Step 3: Write minimal implementation**

Create:

- `BrokerBalanceInfo`
- `BrokerPositionInfo`
- `BrokerClosedPosition`
- `BrokerQuoteInfo`
- `BrokerInstrumentInfo`
- `BrokerOrderResult`

In `src/execution/broker_protocol.py`, define a `Protocol` exposing:

- `login`
- `get_balance`
- `get_open_positions`
- `get_closed_positions`
- `get_quote`
- `get_effective_instruments`
- `open_position`
- `close_position`
- `close_all_positions`
- `modify_position`
- `verify_sl_tp`

Update `src/execution/matchtrader_client.py` imports and return annotations to use the broker-neutral models where possible without changing runtime behavior.

**Step 4: Run tests to verify GREEN**

Run:

```bash
uv run pytest tests/test_broker_models.py tests/test_matchtrader_client.py -k "broker or model" -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/execution/broker_models.py src/execution/broker_protocol.py src/execution/matchtrader_client.py tests/test_broker_models.py tests/test_matchtrader_client.py
git commit -m "refactor: add broker-neutral execution contract"
```

---

### Task 2: Generalize Instrument Registry And Execution Consumers

**Files:**
- Modify: `src/execution/instrument_registry.py`
- Modify: `src/execution/engine.py`
- Modify: `src/decision/close_control_plane.py`
- Modify: `tests/test_engine.py`
- Modify: `tests/test_close_control_plane.py`
- Modify: `tests/test_scheduler.py`

**Step 1: Write the failing regression tests**

Add tests for:

- `InstrumentRegistry` building from a broker-neutral client
- `ExecutionEngine` accepting a broker-protocol client instead of `MatchTraderClient`
- `CloseControlPlane` continuing to use `modify_position`, `close_position`, and `verify_sl_tp` through the protocol

Use tests like:

```python
async def test_instrument_registry_builds_from_broker_protocol(fake_broker):
    registry = await InstrumentRegistry.from_broker(fake_broker, ["EURUSD"])
    assert registry.to_broker("EURUSD") == "EURUSD"
```

**Step 2: Run tests to verify RED**

Run:

```bash
uv run pytest tests/test_engine.py tests/test_close_control_plane.py tests/test_scheduler.py -k "broker_protocol or instrument_registry" -q
```

Expected: FAIL because the registry and execution consumers are still MatchTrader-specific.

**Step 3: Write minimal implementation**

Refactor:

- `InstrumentRegistry.from_matchtrader()` into a broker-neutral constructor such as `from_broker()`
- type references in `ExecutionEngine` and `CloseControlPlane` to the new protocol
- imports and helper assumptions that mention MatchTrader only as a concrete backend, not a required type

Do not change business logic. Only change the dependency boundary.

**Step 4: Run tests to verify GREEN**

Run:

```bash
uv run pytest tests/test_engine.py tests/test_close_control_plane.py tests/test_scheduler.py -k "broker_protocol or instrument_registry" -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/execution/instrument_registry.py src/execution/engine.py src/decision/close_control_plane.py tests/test_engine.py tests/test_close_control_plane.py tests/test_scheduler.py
git commit -m "refactor: make execution consumers broker-neutral"
```

---

### Task 3: Add TradeLocker Config And Broker Factory

**Files:**
- Modify: `src/config.py`
- Create: `src/execution/broker_factory.py`
- Modify: `.env.example`
- Modify: `tests/test_config.py`
- Create: `tests/test_broker_factory.py`

**Step 1: Write the failing config/factory tests**

Add tests for:

- `execution.broker_backend` supporting `matchtrader` and `tradelocker`
- TradeLocker config/env fields loading correctly
- broker factory returning the correct backend

Use tests like:

```python
def test_broker_factory_returns_tradelocker_for_tradelocker_backend(config):
    config.execution.broker_backend = "tradelocker"
    broker = build_broker_client(config, store=None)
    assert broker.__class__.__name__ == "TradeLockerClient"
```

**Step 2: Run tests to verify RED**

Run:

```bash
uv run pytest tests/test_config.py tests/test_broker_factory.py -k "broker_backend or tradelocker" -q
```

Expected: FAIL because the config fields and factory do not exist yet.

**Step 3: Write minimal implementation**

In `src/config.py`, add:

- `ExecutionConfig.broker_backend`
- a dedicated `TradeLockerConfig` or equivalent config block

In `.env.example`, add:

- `TRADELOCKER_API_URL`
- `TRADELOCKER_EMAIL`
- `TRADELOCKER_PASSWORD`
- `TRADELOCKER_SERVER`
- `TRADELOCKER_ACCOUNT_ID`

In `src/execution/broker_factory.py`, add a single builder that constructs:

- `MatchTraderClient` when backend is `matchtrader`
- `TradeLockerClient` when backend is `tradelocker`

**Step 4: Run tests to verify GREEN**

Run:

```bash
uv run pytest tests/test_config.py tests/test_broker_factory.py -k "broker_backend or tradelocker" -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/config.py src/execution/broker_factory.py .env.example tests/test_config.py tests/test_broker_factory.py
git commit -m "feat: add broker backend config and factory"
```

---

### Task 4: Implement TradeLocker Client For Stable Execution Surface

**Files:**
- Create: `src/execution/tradelocker_client.py`
- Create: `tests/test_tradelocker_client.py`

**Step 1: Write the failing client tests**

Add tests for:

- login/auth flow
- account selection
- quote parsing
- instrument parsing
- open positions parsing
- closed positions parsing
- market order open
- close position
- modify position
- `verify_sl_tp` via read-back from positions

Use response fixtures that normalize into the broker-neutral models.

Example:

```python
async def test_verify_sl_tp_reads_back_position_values(client):
    client.get_open_positions = AsyncMock(
        return_value=[
            BrokerPositionInfo(position_id="POS1", symbol="EURUSD", side="BUY", volume=0.1, sl_price=1.08)
        ]
    )
    verified = await client.verify_sl_tp(position_id="POS1", expected_sl=1.08)
    assert verified is True
```

**Step 2: Run tests to verify RED**

Run:

```bash
uv run pytest tests/test_tradelocker_client.py -q
```

Expected: FAIL because the client does not exist yet.

**Step 3: Write minimal implementation**

Implement:

- async context manager
- auth/session handling
- account-scoped request helpers
- response normalization into broker-neutral models
- read-back verification logic using `get_open_positions()`

Keep all provider-specific field names and routing details inside this file.

**Step 4: Run tests to verify GREEN**

Run:

```bash
uv run pytest tests/test_tradelocker_client.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/execution/tradelocker_client.py tests/test_tradelocker_client.py
git commit -m "feat: add tradelocker broker client"
```

---

### Task 5: Wire Main And Scheduler To Broker Factory

**Files:**
- Modify: `src/main.py`
- Modify: `src/scheduler/scheduler.py`
- Modify: `tests/test_main_daily_cycle.py`
- Modify: `tests/test_scheduler.py`

**Step 1: Write the failing wiring tests**

Add tests for:

- `main` creating the configured broker backend
- scheduler startup using the broker-neutral client
- legacy daily cycle using the selected backend
- `TradeLocker` backend preserving current quote/position-driven scheduler flows

Use tests like:

```python
def test_run_scheduler_builds_tradelocker_backend(monkeypatch):
    ...
    assert build_broker_client_called_with_backend == "tradelocker"
```

**Step 2: Run tests to verify RED**

Run:

```bash
uv run pytest tests/test_main_daily_cycle.py tests/test_scheduler.py -k "tradelocker or broker_factory" -q
```

Expected: FAIL because startup still constructs `MatchTraderClient` directly.

**Step 3: Write minimal implementation**

Refactor startup paths in `src/main.py` to:

- build the broker client through `broker_factory`
- avoid hard-coded `MATCHTRADER_*` assumptions at the wiring level

Refactor `src/scheduler/scheduler.py` only as needed to accept the generic client instance while preserving behavior.

Do not redesign scheduler loops.

**Step 4: Run tests to verify GREEN**

Run:

```bash
uv run pytest tests/test_main_daily_cycle.py tests/test_scheduler.py -k "tradelocker or broker_factory" -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/main.py src/scheduler/scheduler.py tests/test_main_daily_cycle.py tests/test_scheduler.py
git commit -m "feat: wire tradelocker backend into runtime startup"
```

---

### Task 6: End-To-End Stable Regression Pass And Ops Documentation

**Files:**
- Modify: `docs/ops_runbook.md`
- Modify: `docs/PropFirmPilot_v1.5.0_road_map.md`
- Modify: `README.md`
- Modify: `tests/test_engine.py`
- Modify: `tests/test_close_control_plane.py`
- Modify: `tests/test_scheduler.py`
- Modify: `tests/test_config.py`

**Step 1: Write any remaining failing regression tests**

Add targeted tests for:

- TradeLocker backend execution happy path
- tactical modify/verify path
- scheduler account snapshot and close reconciliation compatibility
- config examples for E8 Signature backend selection

**Step 2: Run tests to verify RED**

Run:

```bash
uv run pytest tests/test_engine.py tests/test_close_control_plane.py tests/test_scheduler.py tests/test_config.py -k "tradelocker or broker_backend" -q
```

Expected: FAIL if any integration gap remains.

**Step 3: Write minimal implementation/docs**

Complete remaining glue, then update operator docs with:

- required TradeLocker env vars
- selected backend config example
- expected startup flow for E8 Signature
- known limitations for stable release

**Step 4: Run final verification**

Run:

```bash
uv run pytest tests/test_tradelocker_client.py tests/test_broker_factory.py tests/test_engine.py tests/test_close_control_plane.py tests/test_scheduler.py tests/test_main_daily_cycle.py tests/test_config.py -k "tradelocker or broker_backend or instrument_registry" -q
uv run ruff check src/execution/broker_models.py src/execution/broker_protocol.py src/execution/broker_factory.py src/execution/tradelocker_client.py src/execution/matchtrader_client.py src/execution/instrument_registry.py src/execution/engine.py src/decision/close_control_plane.py src/main.py src/scheduler/scheduler.py tests/test_broker_models.py tests/test_broker_factory.py tests/test_tradelocker_client.py tests/test_engine.py tests/test_close_control_plane.py tests/test_scheduler.py tests/test_main_daily_cycle.py tests/test_config.py
```

Expected: PASS and clean lint output.

**Step 5: Commit**

```bash
git add docs/ops_runbook.md docs/PropFirmPilot_v1.5.0_road_map.md README.md tests/test_engine.py tests/test_close_control_plane.py tests/test_scheduler.py tests/test_config.py
git commit -m "docs: finalize tradelocker stable integration"
```
