# Market Data WebSocket Bar Priority Design

**Date:** 2026-03-23

**Problem**

After the startup retryable hotfix, production can trade successfully, but
`MarketDataHub` still emits repeated `REST fallback` warnings every few minutes
even when the websocket feed is healthy and the system already has enough
closed websocket-derived bars for ongoing tactical work.

The current `get_bars()` order is:

1. fresh warm cache
2. REST refresh attempt
3. websocket closed bars
4. stale REST tail return

This means the hub still performs unnecessary REST refresh attempts and logs a
degraded warning before consulting websocket bars.

**Goal**

Reduce unnecessary REST calls and warning noise without weakening safeguards.

**Approved Direction**

For closed bars only:

1. keep `warmup_cache` first when it is fresh,
2. if warm cache is stale, check websocket closed bars next,
3. only if websocket also lacks fresh closed bars, attempt REST refresh,
4. only log `REST fallback` when the final returned bar source is actually REST.

**Why This Is Safe**

- Quotes are unchanged.
- Entry readiness logic remains intact.
- Tactical validation still rejects stale bars after sanitization.
- Forced-stale symbols still skip websocket and go straight to REST.

**Expected Runtime Effect**

- After websocket has produced fresh closed `5m` / `1h` bars, periodic tactical
  checks stop generating repeated `REST fallback` warnings.
- REST request volume drops because healthy websocket bar reads no longer force
  a refresh-first path.
- The system still falls back to REST when websocket bars are absent or stale.

**Non-Goals**

- No change to broker quote routing.
- No change to execution/compliance logic.
- No attempt to synthesize bars from partial, still-open websocket buckets.
