# E8 Signature TradeLocker Integration Design

**Date:** 2026-03-22

**Goal:** Enable `v1.5.0_stable` to support E8 Signature accounts on the existing macOS server by introducing a broker abstraction and adding `TradeLocker` as a new execution backend.

## Context

The current production execution path is tightly coupled to `MatchTrader`:

- [src/main.py](/C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/main.py)
- [src/scheduler/scheduler.py](/C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/scheduler/scheduler.py)
- [src/execution/engine.py](/C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/execution/engine.py)
- [src/decision/close_control_plane.py](/C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/decision/close_control_plane.py)
- [src/execution/instrument_registry.py](/C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/execution/instrument_registry.py)
- [src/execution/matchtrader_client.py](/C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/execution/matchtrader_client.py)

E8 Signature does not support MatchTrader. The user also requires the deployment target to remain the current macOS server. That rules out MT5 as the primary stable path because the official Python integration is terminal-coupled and Windows-first. `cTrader` remains viable, but its official Open API integration is event-driven and structurally farther from the current REST-centric broker client shape. `TradeLocker` exposes a public API that is much closer to the existing architecture, making it the best first stable target.

## Platform Decision

### Rejected: MT5-first

Reasons:

- Official Python integration is built around a local MT5 terminal session, not a headless broker REST service.
- That conflicts with the current macOS server deployment model.
- Supporting MT5 would likely require a Windows terminal host, VM, or a remote bridge, which expands the `v1.5.0_stable` scope beyond a safe broker integration.

### Deferred: cTrader-first

Reasons:

- Official support is strong and it remains a good future backend candidate.
- The official Python SDK is callback/event-driven and materially different from the project's current async request-response execution model.
- Starting with `cTrader` would force a larger architecture change than necessary for the stable milestone.

### Selected: TradeLocker-first

Reasons:

- Works with the existing macOS server model.
- Public API capabilities align with current broker responsibilities: auth, account selection, quotes, positions, order entry, order close, and SL/TP modification.
- Most compatible with the current `MatchTraderClient -> ExecutionEngine -> Scheduler` execution surface.

## Architecture

### Recommended approach: minimal compatible broker abstraction

Introduce a bounded broker-neutral contract and keep the rest of the runtime as unchanged as possible.

This design intentionally does not attempt a full broker-domain rewrite. The stable goal is to add `TradeLocker` without breaking the current `MatchTrader` route and without widening scope into scanner, market-data, or strategy changes.

### New layers

1. Broker-neutral models
2. Broker client protocol
3. Broker factory
4. `TradeLockerClient` implementation

### Existing layers that should consume the abstraction

- `PropFirmPilot` legacy daily cycle
- `Scheduler`
- `ExecutionEngine`
- `CloseControlPlane`
- `InstrumentRegistry`
- operator-facing monitoring flows that read balance, positions, or quotes

## Broker Contract

The contract should stay limited to methods the codebase already depends on:

- `login()`
- `get_balance()`
- `get_open_positions()`
- `get_closed_positions(from_ts, to_ts)`
- `get_quote(symbol)`
- `get_effective_instruments()`
- `open_position(symbol, side, volume, sl=None, tp=None)`
- `close_position(position_id, symbol, side, volume)`
- `close_all_positions()`
- `modify_position(position_id, symbol, side, volume, sl=None, tp=None)`
- `verify_sl_tp(position_id, expected_sl=None, expected_tp=None, tolerance=..., price_precision=None)`

The broker-neutral data models should match the current domain usage closely:

- `BrokerBalanceInfo`
- `BrokerPositionInfo`
- `BrokerClosedPosition`
- `BrokerQuoteInfo`
- `BrokerInstrumentInfo`
- `BrokerOrderResult`

The first version should be intentionally conservative and look similar to the current MatchTrader models so the rest of the runtime can migrate with minimal churn.

## TradeLocker Integration Boundaries

### In scope for `v1.5.0_stable`

- Authenticate against TradeLocker
- Select one configured trading account
- Fetch balance/equity
- Fetch open positions
- Fetch recent closed positions
- Fetch quotes for live spread/freshness checks
- Fetch effective instruments for symbol mapping
- Open market positions
- Close full or partial positions
- Modify SL/TP
- Read back position state to verify SL/TP changes
- Run the existing scheduler and tactical exit flows against the new backend

### Explicitly out of scope for this stable milestone

- TradeLocker websocket market data
- replacing EODHD tactical bars with broker bar history
- multi-account orchestration
- pending orders, stop orders, limit orders
- cTrader integration in the same milestone
- a comprehensive broker capability matrix
- redesigning strategy logic, tactical rules, or TradingAgents behavior

## File-Level Design

### New files

- `src/execution/broker_models.py`
- `src/execution/broker_protocol.py`
- `src/execution/broker_factory.py`
- `src/execution/tradelocker_client.py`
- `tests/test_broker_factory.py`
- `tests/test_tradelocker_client.py`

### Existing files to refactor

- `src/execution/matchtrader_client.py`
- `src/execution/engine.py`
- `src/execution/instrument_registry.py`
- `src/decision/close_control_plane.py`
- `src/main.py`
- `src/scheduler/scheduler.py`
- `src/config.py`
- `.env.example`
- `tests/test_matchtrader_client.py`
- `tests/test_engine.py`
- `tests/test_close_control_plane.py`
- `tests/test_scheduler.py`
- `tests/test_main_daily_cycle.py`
- `tests/test_config.py`

## Data Flow

### Startup

1. Config selects `execution.broker_backend`.
2. Broker factory builds the correct broker client.
3. Runtime logs in.
4. Instrument registry builds config-symbol to broker-symbol mappings from broker instruments.

### Entry execution

1. Scheduler or legacy daily cycle prepares a `ready_for_exec` intent.
2. `ExecutionEngine` fetches broker balance, positions, and quotes through the protocol.
3. `ExecutionEngine` opens the position using the broker client.
4. `ExecutionEngine` modifies SL/TP and verifies them via broker read-back.
5. Store, journal, alerting, and metrics continue unchanged.

### Exit execution

1. `CloseControlPlane` issues `modify_only`, `partial_close`, or `full_close` through the protocol.
2. If it is a modify path, the broker client must support deterministic read-back verification.
3. `CloseReconciler`, scheduler close handling, and journal paths continue to operate on broker-neutral result payloads.

## Error Handling

The integration must remain fail-closed.

- If broker auth fails, startup must stop rather than downgrade to mock execution.
- If account selection is ambiguous, startup must stop.
- If instrument mapping is incomplete, untradeable symbols must be reported explicitly and skipped.
- If quote fetch fails, the existing execution-side slippage guard may proceed without the check only where the current code already allows it.
- If SL/TP modification succeeds but verify fails, the result must remain `verify_failed`, matching the current safety posture.
- If closed-position reconciliation lags, runtime should preserve the existing delayed reconciliation behavior rather than guessing final state.

## Testing Strategy

The implementation should remain test-first and staged.

### Unit tests

- broker-neutral model compatibility
- broker factory selection
- TradeLocker auth and response parsing
- instrument normalization and symbol mapping
- open/close/modify/verify flows

### Integration-level regression tests

- `ExecutionEngine` with broker protocol
- `CloseControlPlane` with broker protocol
- scheduler entry path using broker-neutral quotes/positions
- legacy daily cycle using the selected backend

### Stability checks

- existing MatchTrader tests must continue to pass after protocol extraction
- targeted scheduler and execution regressions must stay green

## Risks

### Instrument normalization risk

`InstrumentRegistry` is currently MatchTrader-centric and assumes dot-suffix style symbol handling. The design should generalize it into a broker symbol mapper without overengineering. The stable target is still just FX majors and crosses already present in config.

### API semantics mismatch risk

TradeLocker order and position models may not line up exactly with MatchTrader's field semantics. The broker client must absorb that mismatch and expose normalized models upstream.

### Verification risk

`verify_sl_tp()` is not optional in this project. If TradeLocker lacks a dedicated verification endpoint, the client must implement read-back verification using `get_open_positions()` and normalized comparisons.

### Scope risk

This work can easily expand into a generic broker platform rewrite. The design explicitly rejects that. Only the surfaces required to run `v1.5.0_stable` on E8 Signature through TradeLocker belong in this milestone.

## Recommendation

Proceed with `TradeLocker-first` using a minimal broker abstraction. Keep the current runtime topology, preserve MatchTrader compatibility, and constrain the milestone to execution-path parity on macOS server deployment.

## References

- MT5 Python integration overview: https://www.mql5.com/en/docs/python_metatrader5
- MT5 initialize docs: https://www.mql5.com/en/docs/python_metatrader5/mt5initialize_py
- cTrader Open API overview: https://help.ctrader.com/open-api/
- cTrader Python SDK docs: https://help.ctrader.com/open-api/python-SDK/python-sdk-index/
- TradeLocker public API getting started: https://public-api.tradelocker.com/docs/getting-started
