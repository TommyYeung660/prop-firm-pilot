# EODHD Real-Time REST Fallback Design

**Date:** 2026-03-22
**Status:** Approved for implementation

## Problem

`MarketDataHub` currently routes:

1. quotes: `broker_quote -> websocket_cache -> intraday REST 1m -> synthetic quote`
2. bars: `warmup_cache / intraday REST -> websocket_cache`

This fails badly when:

- EODHD forex websocket stops delivering ticks
- EODHD `/api/intraday/*.FOREX` keeps lagging on `5m/1h` rollups

We verified on 2026-03-19 that:

- websocket authorization could succeed while tick delivery stalled
- `/api/intraday/*.FOREX` `1m` data could be newer than its own `5m/1h`
- `/api/real-time/*.FOREX` is available under the current subscription and is the correct low-latency REST product line

## Goal

Add an official EODHD real-time REST fallback so runtime market-data routing becomes:

`websocket -> EODHD real-time REST -> EODHD intraday REST -> fail-closed`

## Non-Goals

- No `yfinance` integration
- No relaxation of entry/tactical freshness thresholds
- No change to broker quote being the execution-side truth source
- No attempt to backfill large historical gaps from real-time REST snapshots

## Approach

### 1. Add an EODHD real-time REST quote fetcher

Introduce a small provider that calls:

- `GET https://eodhd.com/api/real-time/{SYMBOL}.FOREX?api_token=...&fmt=json`

It should normalize payload into the internal quote schema:

- `symbol`
- `bid`
- `ask`
- `mid`
- `timestamp_ms`

Because the endpoint returns OHLC snapshot data instead of bid/ask, the fallback quote will use:

- `bid = close`
- `ask = close`

This is acceptable for market-data continuity and aggregator feeding, but not treated as a broker execution quote.

### 2. Feed real-time REST snapshots into the existing aggregator

When websocket is stale or missing, `MarketDataHub.get_quote()` should try the EODHD real-time REST provider before falling back to intraday `1m`.

If a real-time REST quote is returned:

- store it as a `QuoteResult(source="rest_realtime")`
- convert it into a synthetic `WebSocketTick`
- feed it into `FXTickAggregator`

This keeps the current bar-building pipeline intact and lets the system derive fresh `1m/5m/1h` closed bars without inventing a second bar engine.

### 3. Use aggregator-derived bars before stale intraday REST bars

`MarketDataHub.get_bars()` should remain conservative:

- prefer fresh warmup/API cache when available
- otherwise prefer fresh aggregator-derived bars
- only then use intraday REST fallback

This makes runtime bar freshness depend on:

- websocket ticks when healthy
- real-time REST-fed synthetic ticks when websocket is degraded
- intraday REST only as historical/warmup fallback

### 4. Preserve fail-closed behavior

If websocket is stale, real-time REST fails, and intraday REST remains stale or unavailable:

- `get_quote()` returns no usable quote
- `get_bars()` returns stale/empty bars as today
- entry guard and tactical sanitizer continue blocking

No trading permissiveness is added.

## Expected Behavior

### Runtime during websocket outage

- quote requests continue to resolve from EODHD real-time REST
- aggregator continues receiving time-advancing snapshots
- fresh `5m` bars can keep closing
- fresh `1h` bars can continue closing on the next hour boundary

### Startup during weekend / market close

- today is 2026-03-22 and FX is closed
- tests must not assume live bars beyond the last market session
- implementation should rely on mocked timestamps and synthetic responses, not live market-open assumptions

## Risks

### Snapshot cadence

Real-time REST is not a tick stream. If polled too slowly, bars will be sparse. The hub should only use it opportunistically on demand, which is enough for current scanner/tactical polling cadence.

### Duplicate synthetic ticks

Repeated snapshots with the same timestamp may re-feed the aggregator. The implementation should suppress duplicates per symbol/timestamp.

### Quote provenance

Diagnostics must keep quote source explicit so production analysis can distinguish:

- `broker_quote`
- `websocket_cache`
- `rest_realtime`
- `rest_fallback`

## Files Expected To Change

- `src/data/market_data_hub.py`
- `src/data/fx_data_fetcher.py`
- `tests/data/test_market_data_hub.py`
- `tests/test_fx_data_fetcher.py`
- optional docs/changelog if behavior is user-visible enough to record
