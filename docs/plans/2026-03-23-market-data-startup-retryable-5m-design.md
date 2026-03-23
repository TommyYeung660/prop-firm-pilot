# Market Data Startup Retryable 5m Design

**Date:** 2026-03-23

**Problem**

`v1.5.0_stable` can block all new entries shortly after FX market open even when:

- scanner produced valid candidates,
- broker quote is fresh,
- websocket feed is healthy,
- `1h` bars are fresh enough for tactical context.

The block happens because `MarketDataHub.get_entry_readiness()` only treats the
"startup pending" state as retryable when `5m` bars are completely missing.
When EODHD intraday REST returns same-day but stale `5m` bars and the websocket
aggregator has not yet closed its first `5m` bar, the current logic returns
`market_data.bars_5m_stale` instead of a retryable startup state.

**Observed Runtime Pattern**

- Websocket starts successfully and receives fresh ticks.
- Broker quote remains available.
- EODHD intraday REST `5m` tail lags by more than the configured freshness window.
- Websocket aggregator still has zero closed `5m` bars because the first bucket
  has not elapsed yet.
- Scanner candidates are skipped as hard blocks instead of being admitted into
  the retryable tactical path.

**Design Goal**

Reclassify this narrow cold-start window as:

- `entry_safe=True`
- `requires_tactical_retry=True`
- `pending_reason="market_data.startup_5m_bar_pending"`

This keeps the existing tactical retry flow intact without weakening normal
stale-bar protections.

**Recommended Approach**

Add a narrow detection branch inside `MarketDataHub.get_entry_readiness()`:

- only when `5m` bars are stale,
- only when websocket feed is currently healthy,
- only when the in-memory websocket aggregator still has zero closed `5m` bars
  for that symbol,
- only when `1h` bars are still fresh,
- only when quote is available.

If all conditions hold, treat the symbol as "startup pending" instead of a hard
`bars_5m_stale` block.

**Why This Approach**

- It fixes the exact production failure mode.
- It preserves the current scheduler behavior because scheduler already knows
  how to admit retryable startup candidates.
- It avoids broad grace periods and avoids using partial, unclosed `5m` bars for
  tactical decisions.

**Non-Goals**

- No change to execution, broker login, or intent lifecycle.
- No change to `trade_date_not_ready`.
- No change to the underlying EODHD source ordering.
