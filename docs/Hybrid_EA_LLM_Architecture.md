# Hybrid EA+LLM Architecture Blueprint

> **Version**: 1.0  
> **Date**: 2026-02-16  
> **Status**: DRAFT — Awaiting approval before implementation  
> **Scope**: Transform `prop-firm-pilot` from a sequential daily-cycle system into an asynchronous, multi-layer trading pipeline that decouples signal generation, LLM analysis, and trade execution.

---

## 1. Problem Statement

The current `PropFirmPilot.run_daily_cycle()` is **fully synchronous and sequential**:

```
fetch_data → run_scanner (up to 10 min) → LLM decide per symbol (10+ min each) → execute
```

This creates three critical problems:

1. **Total latency**: Scanner (10 min) + LLM per symbol (10+ min × 3 symbols) = **40+ minutes** before the first trade executes. For daily (D1) signals this is acceptable, but it blocks the system from ever supporting intraday (4H/1H) signals.

2. **Tight coupling**: If the scanner fails, no decisions are made. If LLM fails, no trades execute. There is no retry, no fallback, no partial execution.

3. **No persistence**: Decisions exist only in memory during `run_daily_cycle()`. If the process crashes mid-cycle, all work is lost.

### Design Goals

| Goal | Constraint |
|------|-----------|
| Decouple scanner, LLM, and execution into independent async workers | Single machine, single process (asyncio) |
| Persist all decisions to survive crashes | SQLite (no Redis/Kafka) |
| Support future intraday signals (4H/1H) without rewriting | Modular scheduler |
| Never weaken compliance checks | PropFirmGuard remains the single gate |
| Incremental migration — existing code stays working | New modules added alongside old |

---

## 2. Architecture Overview

### Three-Layer Design

```
                    ┌─────────────────────────────────┐
                    │        STRATEGY LAYER            │
                    │                                  │
                    │  ┌──────────┐    ┌────────────┐  │
                    │  │ Scanner  │    │ LLM Worker │  │
                    │  │ (4h cycle)│   │ (async pool)│  │
                    │  └────┬─────┘    └──────┬─────┘  │
                    │       │                 │        │
                    └───────┼─────────────────┼────────┘
                            │    WRITE        │ READ+WRITE
                            ▼                 ▼
                    ┌─────────────────────────────────┐
                    │     DECISION STORE (SQLite)      │
                    │                                  │
                    │  intents table ←→ decisions table │
                    │  WAL mode, single-writer safe     │
                    └───────────────┬──────────────────┘
                                    │ READ
                                    ▼
                    ┌─────────────────────────────────┐
                    │       EXECUTION LAYER            │
                    │                                  │
                    │  ┌───────────┐  ┌─────────────┐  │
                    │  │ Execution │  │ PropFirm    │  │
                    │  │ Engine    │──│ Guard       │  │
                    │  └─────┬─────┘  └─────────────┘  │
                    │        │                         │
                    │        ▼                         │
                    │  ┌─────────────┐                  │
                    │  │ MatchTrader │                  │
                    │  │ Client      │                  │
                    │  └─────────────┘                  │
                    └─────────────────────────────────┘
                                    │
                    ┌───────────────┴──────────────────┐
                    │       MONITORING LAYER            │
                    │  EquityMonitor + Janitor + Alerts │
                    └─────────────────────────────────┘
```

### Data Flow (Happy Path)

```
1. Scanner runs → produces ScannerSignal[] for top-K symbols
2. For each signal, a TradeIntent is INSERTed into SQLite (status=pending)
3. LLM Worker claims a pending intent (status=claimed, claim_ts=now)
4. LLM calls TradingAgents.propagate() → gets BUY/SELL/HOLD
5. If BUY/SELL: intent updated to ready_for_exec with decision details
   If HOLD: intent updated to cancelled
6. Execution Engine picks up ready_for_exec intents
7. PropFirmGuard.check_all() runs → PASS or REJECT
8. If PASS: MatchTraderClient.open_position() → intent status=opened
   If REJECT: intent status=rejected with compliance reason
9. Janitor cleans up expired/old intents periodically
```

---

## 3. Decision State Machine

```
                    ┌─────────┐
                    │ pending  │ ← Scanner creates intent
                    └────┬────┘
                         │ LLM Worker claims
                         ▼
                    ┌─────────┐     claim_ttl expired
                    │ claimed  │ ──────────────────────► timed_out
                    └────┬────┘
                         │ LLM returns BUY/SELL
                         ▼
                ┌────────────────┐
                │ ready_for_exec │
                └───────┬────────┘
                        │ Execution Engine picks up
                        ▼
                   ┌───────────┐
                   │ executing │
                   └─────┬─────┘
                    ╱    │    ╲
                   ╱     │     ╲
                  ▼      ▼      ▼
             ┌────────┐ ┌────────┐ ┌────────┐
             │ opened │ │rejected│ │ failed │
             └───┬────┘ └────────┘ └────────┘
                 │ Position closed (TP/SL/manual)
                 ▼
             ┌────────┐
             │ closed │
             └────────┘

Special transitions:
  pending  → cancelled  (user cancel, scanner invalidation)
  claimed  → cancelled  (LLM returns HOLD)
  claimed  → timed_out  (claim_ttl exceeded, Janitor recycles)
```

### Status Values

| Status | Owner | Description |
|--------|-------|-------------|
| `pending` | Scanner | Intent created, awaiting LLM analysis |
| `claimed` | LLM Worker | Worker is actively analyzing this intent |
| `ready_for_exec` | LLM Worker | LLM decided BUY/SELL, awaiting execution |
| `executing` | Execution Engine | Engine has picked up, calling MatchTrader |
| `opened` | Execution Engine | Position successfully opened |
| `rejected` | Execution Engine | PropFirmGuard rejected the trade |
| `failed` | Execution Engine | MatchTrader API error |
| `cancelled` | Various | Explicitly cancelled (HOLD, user, stale) |
| `timed_out` | Janitor | Claim TTL expired, recycled to pending or cancelled |
| `closed` | Monitor/Manual | Position closed (TP/SL hit or manual) |

---

## 4. Data Models

### 4.1 TradeIntent (New: `src/decision/schemas.py`)

```python
class TradeIntent(BaseModel):
    """A trade opportunity discovered by the scanner, awaiting LLM evaluation."""

    id: str = Field(default_factory=lambda: uuid4().hex[:16])
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trade_date: str = Field(description="Trading date YYYY-MM-DD")
    symbol: str = Field(description="FX pair e.g. EURUSD")

    # Scanner outputs
    scanner_score: float = 0.0
    scanner_confidence: str = "medium"
    scanner_score_gap: float = 0.0
    scanner_drop_distance: float = 0.0
    scanner_topk_spread: float = 0.0

    # LLM decision (filled after claim)
    suggested_side: Literal["BUY", "SELL", "HOLD"] | None = None
    suggested_sl_pips: float | None = None
    suggested_tp_pips: float | None = None
    agent_risk_report: str = ""
    agent_state_json: str = ""  # JSON-serialized final_state from TradingAgents

    # Lifecycle
    source: Literal["scanner", "manual"] = "scanner"
    status: Literal[
        "pending", "claimed", "ready_for_exec", "executing",
        "opened", "rejected", "failed", "cancelled", "timed_out", "closed"
    ] = "pending"
    claim_worker_id: str | None = None
    claim_ts: datetime | None = None
    claim_ttl_minutes: int = 30
    expires_at: datetime | None = None
    idempotency_key: str = Field(default_factory=lambda: uuid4().hex[:12])

    # Execution result (filled after execution)
    position_id: str | None = None
    executed_at: datetime | None = None
    execution_error: str | None = None
    compliance_snapshot: str = ""  # JSON of compliance check results
```

### 4.2 DecisionRecord (New: `src/decision/schemas.py`)

```python
class DecisionRecord(BaseModel):
    """Immutable audit record linking intent to execution outcome."""

    intent_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    claimed_at: datetime | None = None
    decided_at: datetime | None = None
    executed_at: datetime | None = None
    closed_at: datetime | None = None

    status: str = ""
    order_id: str | None = None
    position_id: str | None = None
    failure_reason: str = ""

    # Snapshots for post-trade analysis
    compliance_snapshot: str = ""
    execution_meta: str = ""  # JSON: entry_price, volume, sl, tp, etc.
```

---

## 5. SQLite Decision Store

### 5.1 Schema (New: `src/decision_store/sqlite_store.py`)

```sql
-- Enable WAL for concurrent reads during writes
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS intents (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,     -- ISO 8601
    trade_date      TEXT NOT NULL,
    symbol          TEXT NOT NULL,

    -- Scanner data
    scanner_score       REAL DEFAULT 0,
    scanner_confidence  TEXT DEFAULT 'medium',
    scanner_score_gap   REAL DEFAULT 0,
    scanner_drop_distance REAL DEFAULT 0,
    scanner_topk_spread REAL DEFAULT 0,

    -- LLM decision
    suggested_side      TEXT,          -- BUY/SELL/HOLD or NULL
    suggested_sl_pips   REAL,
    suggested_tp_pips   REAL,
    agent_risk_report   TEXT DEFAULT '',
    agent_state_json    TEXT DEFAULT '',

    -- Lifecycle
    source              TEXT DEFAULT 'scanner',
    status              TEXT DEFAULT 'pending',
    claim_worker_id     TEXT,
    claim_ts            TEXT,
    claim_ttl_minutes   INTEGER DEFAULT 30,
    expires_at          TEXT,
    idempotency_key     TEXT UNIQUE,

    -- Execution
    position_id         TEXT,
    executed_at         TEXT,
    execution_error     TEXT,
    compliance_snapshot TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_intents_status ON intents(status);
CREATE INDEX IF NOT EXISTS idx_intents_trade_date ON intents(trade_date);
CREATE INDEX IF NOT EXISTS idx_intents_symbol_date ON intents(symbol, trade_date);

CREATE TABLE IF NOT EXISTS decisions (
    intent_id       TEXT PRIMARY KEY REFERENCES intents(id),
    created_at      TEXT NOT NULL,
    claimed_at      TEXT,
    decided_at      TEXT,
    executed_at     TEXT,
    closed_at       TEXT,

    status          TEXT DEFAULT '',
    order_id        TEXT,
    position_id     TEXT,
    failure_reason  TEXT DEFAULT '',

    compliance_snapshot TEXT DEFAULT '',
    execution_meta     TEXT DEFAULT ''
);
```

### 5.2 DecisionStore API

```python
class DecisionStore:
    """SQLite-backed store for trade intents and decisions.

    Thread-safe for single-writer, multiple-reader pattern (WAL mode).
    All methods are synchronous — called from async code via asyncio.to_thread().
    """

    def __init__(self, db_path: str = "data/decisions.db") -> None: ...

    # ── Intent Lifecycle ────────────────────────────────────────────
    def insert_intent(self, intent: TradeIntent) -> None: ...
    def claim_next_pending(self, worker_id: str) -> TradeIntent | None: ...
    def update_intent_decision(
        self, intent_id: str, side: str, sl_pips: float, tp_pips: float,
        risk_report: str, state_json: str,
    ) -> None: ...
    def mark_ready_for_exec(self, intent_id: str) -> None: ...
    def mark_executing(self, intent_id: str) -> None: ...
    def mark_opened(self, intent_id: str, position_id: str) -> None: ...
    def mark_rejected(self, intent_id: str, reason: str) -> None: ...
    def mark_failed(self, intent_id: str, error: str) -> None: ...
    def mark_cancelled(self, intent_id: str, reason: str) -> None: ...
    def mark_closed(self, intent_id: str) -> None: ...

    # ── Queries ─────────────────────────────────────────────────────
    def get_pending_intents(self) -> list[TradeIntent]: ...
    def get_ready_intents(self) -> list[TradeIntent]: ...
    def get_active_positions(self) -> list[TradeIntent]: ...
    def get_intent(self, intent_id: str) -> TradeIntent | None: ...
    def get_intents_by_date(self, trade_date: str) -> list[TradeIntent]: ...

    # ── Claim Management ────────────────────────────────────────────
    def recycle_expired_claims(self) -> int: ...
    def cleanup_old_intents(self, retention_days: int = 7) -> int: ...

    # ── Idempotency ─────────────────────────────────────────────────
    def intent_exists(self, symbol: str, trade_date: str, source: str) -> bool: ...
```

---

## 6. Scheduler Design

### 6.1 Multi-Cycle Orchestrator (New: `src/scheduler/scheduler.py`)

The Scheduler replaces `PropFirmPilot.run_daily_cycle()` as the top-level entry point. It manages multiple async workers on different cadences.

```python
class Scheduler:
    """Async orchestrator managing scanner, LLM workers, and execution engine.

    Usage:
        scheduler = Scheduler(config, decision_store)
        await scheduler.start()  # Runs until interrupted
    """

    async def start(self) -> None:
        """Launch all workers as concurrent asyncio tasks."""
        await asyncio.gather(
            self._scanner_loop(),
            self._llm_worker_loop(worker_id="llm-0"),
            self._execution_loop(),
            self._janitor_loop(),
            self._equity_monitor_loop(),
        )
```

### 6.2 Worker Cadences

| Worker | Cycle | Description |
|--------|-------|-------------|
| **Scanner Loop** | Every 4 hours (configurable) | Runs `ScannerBridge.run_pipeline()`, inserts `TradeIntent` per top-K signal |
| **LLM Worker(s)** | Continuous (poll every 30s) | Claims pending intents, calls `AgentBridge.decide()`, updates intent with result |
| **Execution Engine** | Continuous (poll every 10s) | Picks up `ready_for_exec` intents, runs compliance, executes via MatchTrader |
| **Janitor** | Every 10 minutes | Recycles expired claims, cleans up old intents (>7 days) |
| **Equity Monitor** | Every 60 seconds | Polls account equity, triggers drawdown alerts |

### 6.3 Scanner Loop

```python
async def _scanner_loop(self) -> None:
    while True:
        try:
            signals = await asyncio.to_thread(
                self._scanner.run_pipeline,
                date=today_str(),
                tickers=self._config.symbols,
            )
            for signal in signals[:self._config.scanner.topk]:
                # Idempotency: skip if intent already exists for this symbol+date
                if self._store.intent_exists(signal.instrument, today_str(), "scanner"):
                    continue

                intent = TradeIntent(
                    trade_date=today_str(),
                    symbol=signal.instrument,
                    scanner_score=signal.score,
                    scanner_confidence=signal.confidence,
                    scanner_score_gap=signal.score_gap,
                    scanner_drop_distance=signal.drop_distance,
                    scanner_topk_spread=signal.topk_spread,
                    source="scanner",
                    expires_at=now_utc() + timedelta(hours=4),
                )
                self._store.insert_intent(intent)
                logger.info("Scanner: created intent for {}", signal.instrument)
        except Exception as e:
            logger.error("Scanner loop error: {}", e)

        await asyncio.sleep(self._config.scheduler.scanner_interval_seconds)
```

### 6.4 LLM Worker Loop

```python
async def _llm_worker_loop(self, worker_id: str) -> None:
    while True:
        intent = await asyncio.to_thread(
            self._store.claim_next_pending, worker_id
        )
        if intent is None:
            await asyncio.sleep(30)  # No work, wait
            continue

        try:
            qlib_data = {
                "score": intent.scanner_score,
                "signal_strength": intent.scanner_confidence,
                "confidence": intent.scanner_confidence,
                "score_gap": intent.scanner_score_gap,
                "drop_distance": intent.scanner_drop_distance,
                "topk_spread": intent.scanner_topk_spread,
            }

            decision = await asyncio.to_thread(
                self._agents.decide,
                symbol=intent.symbol,
                trade_date=intent.trade_date,
                qlib_data=qlib_data,
            )

            if decision.is_actionable:
                await asyncio.to_thread(
                    self._store.update_intent_decision,
                    intent.id, decision.decision,
                    sl_pips=50.0, tp_pips=100.0,  # From DecisionFormatter in future
                    risk_report=decision.risk_report,
                    state_json=json.dumps(decision.final_state, default=str),
                )
                await asyncio.to_thread(self._store.mark_ready_for_exec, intent.id)
            else:
                await asyncio.to_thread(
                    self._store.mark_cancelled, intent.id, "LLM decided HOLD"
                )
        except Exception as e:
            logger.error("LLM Worker {}: error on intent {}: {}", worker_id, intent.id, e)
            await asyncio.to_thread(
                self._store.mark_failed, intent.id, str(e)
            )
```

### 6.5 Execution Engine Loop

```python
async def _execution_loop(self) -> None:
    while True:
        intents = await asyncio.to_thread(self._store.get_ready_intents)
        for intent in intents:
            await asyncio.to_thread(self._store.mark_executing, intent.id)

            # Build TradePlan for compliance
            trade_plan = self._build_trade_plan(intent)
            account_snapshot = await self._get_account_snapshot()

            # Compliance gate
            result = self._guard.check_all(trade_plan, account_snapshot)
            if not result.passed:
                await asyncio.to_thread(
                    self._store.mark_rejected, intent.id, result.reason
                )
                continue

            # Random delay (anti-duplicate-strategy)
            delay = self._guard.add_random_delay()
            await asyncio.sleep(delay)

            # Execute
            try:
                order = await self._matchtrader.open_position(
                    symbol=intent.symbol,
                    side=intent.suggested_side,
                    volume=trade_plan.volume,
                )
                if order.success:
                    await asyncio.to_thread(
                        self._store.mark_opened, intent.id, order.position_id
                    )
                else:
                    await asyncio.to_thread(
                        self._store.mark_failed, intent.id, order.message
                    )
            except Exception as e:
                await asyncio.to_thread(
                    self._store.mark_failed, intent.id, str(e)
                )

        await asyncio.sleep(10)
```

---

## 7. Timing Budget

### 7.1 Per-Cycle Timing (4-hour scanner cycle)

| Phase | Duration | Notes |
|-------|----------|-------|
| Scanner subprocess | 2-10 min | Data download + LightGBM training |
| Intent insertion | <1s | SQLite writes, negligible |
| LLM per symbol | 5-15 min | TradingAgents.propagate() + LLM API calls |
| Compliance check | <100ms | Pure calculation, no I/O |
| Trade execution | 1-3s | MatchTrader REST API call |
| **Total for 3 symbols** | **17-48 min** | But scanner and LLM run concurrently |

### 7.2 Slippage Analysis

| Signal Timeframe | LLM Delay Impact | Recommendation |
|------------------|-------------------|----------------|
| D1 (daily) | **LOW** — Major FX pairs move ~50-100 pips/day. 10-min delay = ~2-5 pip move. | Safe for current system |
| 4H | **LOW-MEDIUM** — Depends on session (London open has higher volatility). | Acceptable with limit orders |
| 1H | **MEDIUM** — Consider pre-computing LLM decisions for known patterns. | Future: add urgency classification |
| ≤15M | **HIGH** — Strategy-breaking. Do not use LLM for sub-hourly signals. | Not supported in this architecture |

### 7.3 E8 Markets API Budget

| Operation | Requests/trade | Daily budget (2000) |
|-----------|---------------|---------------------|
| Login + balance check | 2 | Fixed |
| Open position | 1 | Per trade |
| Set SL/TP (modify) | 1 | Per trade |
| Close position | 1 | Per trade |
| Equity polling (60s) | ~1440 | 24h × 60 |
| **Available for trades** | ~550 | 2000 - 1440 - 10 reserve |
| **Max trades/day** | ~137 | 550 / 4 calls per trade |

**Conclusion**: With 60s equity polling, we have budget for ~137 trades/day. Current strategy targets 1-3 trades/day — well within limits. If equity polling is reduced to every 5 minutes, budget increases to ~1700 available for trades.

---

## 8. Configuration Extensions

### New config sections (add to `src/config.py`)

```python
class DecisionStoreConfig(BaseModel):
    """SQLite decision store settings."""
    db_path: str = "data/decisions.db"
    claim_ttl_minutes: int = 30
    intent_retention_days: int = 7

class SchedulerConfig(BaseModel):
    """Multi-cycle scheduler settings."""
    scanner_interval_seconds: int = 14400   # 4 hours
    llm_poll_interval_seconds: int = 30
    execution_poll_interval_seconds: int = 10
    janitor_interval_seconds: int = 600     # 10 minutes
    llm_worker_count: int = 1               # Start with 1, scale to 3
    equity_poll_interval_seconds: int = 60
```

### Updated AppConfig

```python
class AppConfig(BaseModel):
    # ... existing fields ...
    decision_store: DecisionStoreConfig = DecisionStoreConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
```

---

## 9. New Module Map

### Files to CREATE

| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `src/decision/schemas.py` | TradeIntent, DecisionRecord Pydantic models | ~80 |
| `src/decision_store/__init__.py` | Package init | ~1 |
| `src/decision_store/sqlite_store.py` | DecisionStore class with all SQL operations | ~250 |
| `src/decision_store/janitor.py` | TTL cleanup for expired claims and old intents | ~60 |
| `src/scheduler/__init__.py` | Package init | ~1 |
| `src/scheduler/scheduler.py` | Multi-cycle async orchestrator | ~200 |
| `src/execution/engine.py` | Execution Engine worker (compliance + execute) | ~120 |

### Files to MODIFY

| File | Change | Risk |
|------|--------|------|
| `src/config.py` | Add `DecisionStoreConfig`, `SchedulerConfig` | LOW — additive only |
| `src/main.py` | Add `--scheduler` mode alongside existing `--daily` and `--monitor-only` | LOW — new CLI flag |
| `src/decision/agent_bridge.py` | Add `async decide_async()` wrapper around `decide()` | LOW — wrapper only |
| `src/signal/scanner_bridge.py` | No changes needed — Scheduler calls it directly | NONE |
| `src/compliance/prop_firm_guard.py` | No changes needed — Execution Engine calls it directly | NONE |

### Files NOT modified (safety-critical, no changes needed)

| File | Reason |
|------|--------|
| `src/compliance/drawdown_monitor.py` | Pure calculation, used as-is by PropFirmGuard |
| `src/compliance/best_day_tracker.py` | Pure calculation, used as-is |
| `src/execution/matchtrader_client.py` | Existing API is sufficient; idempotency handled at intent level |
| `src/execution/position_sizer.py` | Used by Execution Engine as-is |

---

## 10. Implementation Phases

### Phase 2A: Foundation (Est. 1-2 days)

**Goal**: Persistence layer + data models. System still uses old daily cycle, but decisions are recorded.

1. Create `src/decision/schemas.py` — TradeIntent, DecisionRecord models
2. Create `src/decision_store/sqlite_store.py` — Full DecisionStore with all CRUD operations
3. Add `DecisionStoreConfig` and `SchedulerConfig` to `src/config.py`
4. Unit tests for DecisionStore (insert, claim, state transitions, idempotency)
5. Unit tests for TradeIntent serialization

**Verification**: `pytest tests/test_decision_store.py` passes, SQLite DB created with correct schema.

### Phase 2B: Async Pipeline (Est. 2-3 days)

**Goal**: Scheduler replaces daily cycle. All three workers running.

1. Create `src/scheduler/scheduler.py` — Scanner loop, LLM worker loop, Execution loop
2. Create `src/execution/engine.py` — Compliance check + execute flow
3. Create `src/decision_store/janitor.py` — Expired claim recycling, old intent cleanup
4. Add `async decide_async()` to `src/decision/agent_bridge.py`
5. Add `--scheduler` mode to `src/main.py` (keep `--daily` for backward compatibility)
6. Integration tests: Scanner → Intent → LLM → Execute → Status transitions

**Verification**: `python -m src.main --config config/e8_signature_50k.yaml --scheduler` launches all workers, processes mock signals end-to-end.

### Phase 2C: Hardening (Est. 2-3 days)

**Goal**: Production-ready with monitoring, alerting, and graceful shutdown.

1. Add Telegram alerts for: intent created, trade executed, compliance rejected, errors
2. Graceful shutdown (SIGINT/SIGTERM → cancel workers → flush pending writes)
3. Startup recovery (check for `claimed` intents from crashed session → recycle)
4. Rate limiter refinement (track MatchTrader API calls per day in SQLite)
5. Dashboard query helpers for monitoring (today's intents, success rate, etc.)
6. Ops runbook documentation

**Verification**: Kill process mid-cycle, restart, verify no duplicate trades (idempotency), verify stale claims are recycled.

---

## 11. Migration Strategy

The new Scheduler mode runs **alongside** the existing daily cycle mode. No breaking changes.

```
# Old way (still works, unchanged)
python -m src.main --config config/e8_signature_50k.yaml

# New way (Scheduler mode)
python -m src.main --config config/e8_signature_50k.yaml --scheduler

# Monitor-only (still works, unchanged)
python -m src.main --config config/e8_signature_50k.yaml --monitor-only
```

Once the Scheduler mode is verified in production, the old `run_daily_cycle()` can be deprecated (but not removed — it serves as a simpler fallback).

---

## 12. Risk Analysis

| Risk | Severity | Mitigation |
|------|----------|------------|
| SQLite write contention under load | LOW | WAL mode + `busy_timeout=5000`. Single writer at a time. Our write rate is <1/sec. |
| LLM Worker crash leaves intent as "claimed" forever | MEDIUM | Janitor recycles claims older than `claim_ttl_minutes` (30 min). Startup recovery also checks. |
| Duplicate trades on restart | LOW | `idempotency_key` per intent. `intent_exists()` check before insert. `mark_executing` is atomic. |
| MatchTrader rate limit exceeded | MEDIUM | Execution Engine tracks daily API calls. PropFirmGuard.check_api_request_budget() is a pre-exec gate. Reduce equity poll frequency if needed. |
| Scanner subprocess blocks event loop | LOW | Already wrapped in `asyncio.to_thread()`. 600s timeout in subprocess. |

---

## 13. Key Design Decisions

### Why SQLite, not Redis/Kafka?

- Single machine deployment — no need for distributed messaging
- SQLite WAL mode handles our concurrency needs (1 writer, N readers)
- Write rate is extremely low (<1 write/second) — SQLite handles millions/day
- Crash recovery is built into SQLite (ACID transactions)
- Zero operational overhead — no additional service to manage
- Data persists automatically for post-trade analysis

### Why not modify TradingAgents internals?

- TradingAgents is a separate research project with its own release cycle
- `propagate()` already accepts `qlib_data` — the interface is sufficient
- We control timeouts externally (asyncio timeout wrapper)
- Modifications would create a fork that's hard to maintain

### Why PropFirmGuard stays in the Execution Layer?

- Guard must check the **latest** account state at execution time, not at decision time
- Account balance changes between LLM decision and execution (other trades, equity moves)
- Guard is the final safety gate — it must be as close to execution as possible
- Moving it earlier would create a TOCTOU (time-of-check-time-of-use) vulnerability

### Why 4-hour scanner interval?

- D1 signals change once per day — 4h catches the new daily bar plus gives 5 more checks
- Scanner takes 2-10 minutes — running every 4h is not wasteful
- Aligns with major FX sessions (Sydney, Tokyo, London, New York)
- Configurable via `SchedulerConfig.scanner_interval_seconds`

---

## 14. Appendix: E8 Markets Constraints Summary

| Rule | Limit | Safety Margin (85%) | Impact |
|------|-------|---------------------|--------|
| Daily Drawdown | 5% of day-start balance ($2,500) | $2,125 | Soft Breach → pause trading for the day |
| Max Drawdown | 8% of initial balance ($4,000) | $3,400 | Hard Breach → account terminated |
| Best Day Rule | 40% of profit target ($1,600) | $1,360 | No single day profit above this |
| API Requests | 2,000/day | Reserve 50 for emergencies | Includes all requests (TP/SL modifications) |
| Anti-HFT | No more than 50% of trades held <1 min | N/A | Random delay + position sizing offset |
| Duplicate Strategy | Same EA forbidden across multiple accounts | N/A | Random offset on position sizes |

---

## 15. Open Questions

1. **Equity polling frequency**: 60s = 1440 API calls/day. Should we reduce to 5 minutes (288 calls/day) to free up budget? Or use WebSocket if available?

2. **LLM worker count**: Start with 1, but if 3 symbols need analysis and each takes 10 min, that's 30 min sequential. Should we start with 3 workers (one per symbol)?

3. **Scanner output caching**: If scanner runs every 4h but D1 bar only changes once/day, should the scanner detect no-change and skip re-training?

4. **Position monitoring**: Currently the system opens positions but doesn't actively monitor for close conditions (beyond equity monitor). Should the Execution Engine also poll open positions for TP/SL status and update intent status to `closed`?

---

*End of Blueprint*
