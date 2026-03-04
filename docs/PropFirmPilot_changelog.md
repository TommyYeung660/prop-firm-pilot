# PropFirmPilot Changelog

All notable changes to the PropFirmPilot trading system.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) style.
Versioning: [Semantic Versioning](https://semver.org/).

> **Scope**: This changelog covers the core `prop-firm-pilot` repository.
> Cross-repo impacts on `TradingAgents` and `qlib_market_scanner` are noted where relevant.

---

## [1.3.6] — 2026-03-04

**Quick Config Tuning — Faster Volatility Response & LLM Pickup**

### Changed
- `volatility_poll_interval_seconds`: 60 → 15s (4× faster spike detection)
- `volatility_cooldown_seconds`: 900 → 300s (3× faster re-scan after spike)
- `volatility_threshold_pct`: 0.3 → 0.2% (lower trigger sensitivity)
- `llm_poll_interval_seconds`: 30 → 10s (3× faster LLM worker pickup)

### Verified
- Reflection loop confirmed active in production — `scheduler.py` calls `agents.reflect()` on every position close; TradingAgents reflects across 5 agent memories via ChromaDB
- Expected end-to-end latency improvement: ~81s → ~40s

### Files
- `src/config.py`, `config/e8_one_5k_challenge.yaml`, `tests/test_config.py`
- 89 core tests passed, 785 full suite passed

---

## [1.3.5] — 2026-03-03

**EODHD Intraday Dual-Timeframe + Production Hardening**

### Added
- **Dual-timeframe strategy** (1D trend confirmation + 4H entry timing)
  - EODHD intraday 1H fetch with local 4H OHLCV aggregation
  - TradingAgents intraday OHLCV/indicators tools for 4H data
  - Local indicator computation: SMA, EMA, RSI, MACD, Bollinger, ATR on 4H bars
  - `market_analyst` dual-timeframe FX prompt (trend on daily, entry on 4H)
  - Config separation: `scanner_timeframe` (1D) vs `agent_timeframe` (4H)
  - Qlib 4H frequency compatibility workaround
- **Pipeline cache** for redundant scanner runs (H2 hotfix)
- **Threshold decay** for inactive symbols (H3 hotfix)
- **Telegram circuit breaker** — auto-degradation after 3 failures, 300s probe interval
- Persistent `httpx` client for Telegram connectivity stability

### Fixed
- **C1**: HOLD→BUY mapping — added `risk_report` cross-validation before overriding HOLD
- **C2**: LLM refusal detection — catch model refusals that bypass normal parsing
- **C3**: Compliance rejection cooldown — 120min cooldown to avoid re-evaluating rejected symbols
- **H1**: `exit_reason` broker API retry + PnL inference when API data incomplete
- **M1**: Telegram 409 Conflict — exponential backoff on concurrent update errors

### Tested
- Production evaluation: 18-hour run, 11 issues found → 9 fixed
- Dual-timeframe backtest: 1D IR=1.179, 4H IR=0.299 → kept scanner on 1D for signal quality
- 5 production hotfixes (tool binding, ATR numpy, EODHD vendor, None OHLC, Qlib freq)
- 785 tests passed

---

## [1.3.0] — 2026-03-02

**EODHD Data Migration + v1.2.0 Production Performance Fixes**

### Added — Part A: EODHD Migration
- Full data source migration: Alpha Vantage → EODHD ($29.99/month, 100K calls/day)
- 7 new EODHD modules in TradingAgents (stock, indicator, news, fundamentals, common, config, utils)
- Date-aware switchover mechanism (`EODHD_SWITCHOVER_DATE=2026-03-21`) for gradual migration
- `qlib_market_scanner` EODHD fetcher with priority routing
- Three-repo version unification (all repos → 1.3.0)

### Fixed — Part B: v1.2.0 Production Fixes
- **Fix 1**: Enable multi-timeframe analysis (MTF: daily + 4H data pipeline now active)
- **Fix 2**: Signal freshness guard (`max_signal_age_days=2`) — reject stale scanner signals
- **Fix 3**: Enable macro analyst (央行利率、NFP、CPI data sources wired in)
- **Fix 4**: Scheduler staleness integration — stale signal detection in scheduler loop
- **Fix 5**: Version string update across all entry points

### Stats
- 8 new modules, 12 new test files, 17 modified files, 74 new tests

---

## [1.2.0] — 2026-03-02

**Scheduler Optimization — Parallelism, Session Awareness, Volatility Triggers**

### Added
- **LLM Worker parallelism**: 1 → 2 concurrent workers (configurable `max_llm_workers`)
- **Event-driven re-scan**: `asyncio.Event` triggers immediate scanner run on position close
- **Session-aware cadence**: London/NY active hours = 1h scan interval, off-peak = 4h
- **Volatility-triggered scans**: `VolatilityMonitor` (threshold 0.3%, cooldown 15min)
- **Multi-timeframe data infrastructure**: daily + 4H/1H data pipeline (fetch + store, analysis not yet wired)
- **DST auto-adaptation**: automatic timezone handling for Europe/London, America/New_York, Europe/Athens

### Changed
- Re-evaluation interval shortened: 4h → 2h for open positions
- Scanner signal date filtering to reject stale data
- XAUUSD removed from tradeable pairs (spread too wide for prop firm rules)

### New Modules
- `session_cadence.py` — Session-aware scan scheduling
- `volatility_monitor.py` — Real-time volatility spike detection
- `dst_utils.py` — DST-aware timezone utilities

### Stats
- 51 files changed, +2,266 net lines, 697 tests passed

---

## [1.1.0] — 2026-02-27

**Memory & Feedback Loop — Trade Learning, Dynamic Thresholds**

### Added
- **MemoryJournal**: every LLM decision recorded to `MEMORY/{date}.md`, PnL appended on position close
- **OptimizationEngine**: daily auto-refresh of `optimization_state.json` — tracks 14-day win rate, 7-day PnL trend
- **Dual-layer dynamic confidence thresholds**: pre-LLM filtering (scanner confidence) + post-LLM filtering (agent conviction)
- **TradeJournal pipeline events**: Intent → LLM → CANCEL/OPEN → CLOSE/REJECT/FAIL lifecycle tracking
- **A/B testing infrastructure**: base structure for model comparison (glm-4.7 vs gpt-5.2), not yet wired to scheduler

### Stats
- 584 tests passed

---

## [1.0.0] — 2026-02-25

**Full System Launch — E8 Markets $5,000 Trial Account**

### Architecture
- **Three-layer async pipeline**: Strategy (scanner + LLM) → Execution (MatchTrader) → Monitoring (equity + alerts)
- **7 concurrent async loops** in Scheduler: scanner, LLM workers, re-evaluation, equity monitor, compliance, alert, health check
- **Pydantic v2 config** with YAML deep merge (`default.yaml` + account-specific override)

### Core Modules
- **TradingAgents integration**: multi-agent LLM debate engine (market, news, social analysts → risk manager → portfolio manager)
- **Qlib Alpha158 scanner**: 158-factor model for FX signal generation via `qlib_market_scanner`
- **MatchTrader REST client**: JWT auth, rate limiting (2000 calls/day), exponential backoff retries
- **PropFirmGuard compliance engine** — 5 pre-trade checks, all must pass:
  - Daily drawdown: 4% of day-start balance (with 85% safety margin)
  - Max drawdown: 6% of initial balance (with 85% safety margin)
  - Best Day Rule: no single day profit > 40% of target ($1,600 on $50k account)
  - Position count: max 3 concurrent positions
  - API quota guard: respect MatchTrader daily call limit
- **Re-evaluation mechanism**: LLM re-assesses open positions every 4h
- **Telegram integration**: 15 notification types + bot commands for remote monitoring

### Configuration
- E8 Trial $5,000 account: 4% daily drawdown, 6% max drawdown, 0.5% per-trade risk
- 14 FX pairs configured (majors + crosses, excluding XAUUSD)
- Safety margins at 85% of all limits (not 100%)

### Stats
- 42 Python files, 8,335 lines of code, 548 tests passed
