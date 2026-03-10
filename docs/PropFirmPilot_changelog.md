# PropFirmPilot Changelog

All notable changes to the PropFirmPilot trading system.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) style.
Versioning: [Semantic Versioning](https://semver.org/).

> **Scope**: This changelog covers the core `prop-firm-pilot` repository.
> Cross-repo impacts on `TradingAgents` and `qlib_market_scanner` are noted where relevant.

---

## [1.3.9a] — 2026-03-10
**v1.3.9 Production Hardening — Breakeven Verify, LLM Fallback, Circuit Breaker, Operational Metrics**

### Fixed
- **P1.1**: Breakeven SL modification unverified — added `verify_sl_tp()` with retry logic to confirm broker actually applied SL changes
- **P1.2**: EODHD `volume: null` crashes pandas — added `_sanitize_bar()` null-to-0 defense in `fx_data_fetcher.py`
- **P1.3**: Primary LLM failure with no fallback — added `_fallback_model` field + retry-with-fallback in `decide()`
- **P1.4**: No Data = No Trade guard missing — added `_has_minimum_data()` check before LLM calls when EODHD returns empty bars
- **P2.5**: No consecutive loss circuit breaker — 3+ SL hits on same symbol pauses trading for that symbol for the day
- **P2.6**: No duplicate entry limit — max 2 same-direction trades per symbol per day
- **P2.7**: Risk meta not parsed — extract structured fields (entry_style, avoid_zone, trigger_zone, invalid_if, max_same_day_attempts) from LLM risk reports
- **P3.9**: Trade retrospection insufficient — broker API retry 3→5 attempts + PnL-based close reason inference in new `close_resolution.py`
- **P3.12**: 4 pre-existing config test assertions mismatched YAML values (`default_risk_pct`, `shadow_mode`, `max_drawdown_stop`)

### Added
- `src/monitor/operational_metrics.py` (NEW): API retry stats, latency tracking (p50/p95/p99), system uptime metrics
- `src/scheduler/close_resolution.py` (NEW): PnL-based close reason inference (tp_hit/sl_hit/breakeven/manual_or_unknown)
- `src/scheduler/low_confidence_cooldown.py` (NEW): per-symbol cancellation cooldown (3 cancels → 30min cooldown)
- `verify_sl_tp()` method in `src/execution/matchtrader_client.py`: post-modification SL/TP verification with retry
- `_sanitize_bar()` in `src/data/fx_data_fetcher.py`: null OHLCV field defense
- `_has_minimum_data()` in `src/decision/fx_analyst_config.py`: minimum bar count guard
- `_fallback_model` + retry-with-fallback in `src/decision/agent_bridge.py`
- Consecutive loss circuit breaker + duplicate entry limit in `src/scheduler/scheduler.py`
- Scanner low-confidence cooldown integration in `src/scheduler/scheduler.py`
- Operational metrics summary via Telegram in `src/monitor/alert_service.py`
- 10 new test files, 99 new tests total

### Changed
- `matchtrader_client.py`: broker API retry increased from 3 to 5 attempts
- Test assertions aligned with current `config/e8_one_5k_challenge.yaml` values across 4 test files

### Tested
- **996 tests passed** (was 897; +99 new tests, 4 pre-existing failures fixed)
- 39 files changed, +4,705/-161 lines
- Branch: `fix/v1.3.9-p1-fixes`

### Known Issue
- **kimi-k2.5 `max_completion_tokens` exceeds limit**: When LLM falls back to `volcengine/kimi-k2.5`, TradingAgents sends `max_completion_tokens=128000` which exceeds the model's 32768 limit. **Workaround**: prod reverted to gpt-5.2 + glm-4.7. **Fix pending**: per-model token limit mapping in `_apply_ab_model()`

### Files
- New: `src/monitor/operational_metrics.py`, `src/scheduler/close_resolution.py`, `src/scheduler/low_confidence_cooldown.py`
- Modified: `src/config.py`, `src/data/fx_data_fetcher.py`, `src/decision/agent_bridge.py`, `src/decision/fx_analyst_config.py`, `src/decision/tactical_validator.py`, `src/decision_store/sqlite_store.py`, `src/execution/matchtrader_client.py`, `src/monitor/alert_service.py`, `src/monitor/telegram_bot.py`, `src/scheduler/scheduler.py`
- Tests (NEW): `test_breakeven_verification.py`, `test_circuit_breaker.py`, `test_close_retrospection.py`, `test_duplicate_entry_limit.py`, `test_eodhd_null_defense.py`, `test_llm_fallback.py`, `test_low_confidence_cooldown.py`, `test_no_data_no_trade.py`, `test_operational_metrics.py`, `test_risk_meta_extraction.py`
- Tests (Modified): `test_ab_model_switching.py`, `test_alert_service.py`, `test_decision_store.py`, `test_exit_reason_classification.py`, `test_fx_data_fetcher.py`, `test_fx_duckdb_store.py`, `test_prop_firm_guard_e8_one.py`, `test_scanner_bridge.py`, `test_scheduler.py`, `test_scheduler_multi_timeframe.py`, `test_switchover.py`, `test_tactical_integration.py`, `test_volatility_monitor.py`

## [1.3.9] — 2026-03-09
**v1.3.7 Production Bugfixes (Part 2) — Notification Data, AB Testing, Race Conditions, DuckDB**

### Fixed
- **P2 #4/#14**: TP/SL notifications showed "0.00 lots" — fall back to `execution_meta` JSON for volume and prices
- **P2 #10**: Scanner Score was identical all day — skip intraday rescans when `scanner_timeframe == "1d"` (daily model by design)
- **P2 #9**: HOLD decision but position opened — cancel stale `ready_for_exec` intents when HOLD is decided
- **P2 #11**: AB Test counts/pnl empty `{}` — wired `choose_model()` into `agent_bridge`, record stats on position close, fixed counts reset bug in optimization engine
- **P2 #2 (remaining)**: Spread gate always failed — added per-instrument `avg_spread_pips` config and allow spread gate pass-through when data is missing

### Added
- `tests/test_ab_routing.py` to cover AB routing behavior
- `EodhdProvider` class in `src/data/fx_data_fetcher.py`, async intraday bar data provider supporting 5min/1h/15min/30min intervals via EODHD API
- EODHD intraday wiring in `src/scheduler/scheduler.py`, `_fetch_tactical_data()` now fetches real 5min + 1h bars via `asyncio.gather()`, making ATR regime, EMA momentum, RSI state, candle quality, and data freshness tactical gates functional
- `_apply_ab_model()` method in `src/decision/agent_bridge.py`, rebuilds TradingAgentsGraph with selected model, enabling real AB test model switching (not just metadata logging)
- `tests/test_ab_model_switching.py` (NEW), 10 tests covering `_apply_ab_model` behavior and `decide()` AB integration
- 8 new EODHD provider tests in `tests/test_fx_data_fetcher.py`

### Changed
- DuckDB transaction handling now guards against nested `BEGIN TRANSACTION` calls
- Breakeven threshold lowered from 0.5 to 0.3 in config
- AB test model defaults updated: `ab_model_a: "rightcodes/gpt-5.4"`, `ab_model_b: "volcengine/kimi-k2.5"` in `src/config.py`, `src/optimize/optimization_engine.py`, `src/optimize/optimization_state.py`, `config/e8_one_5k_challenge.yaml`
- `choose_model()` now called BEFORE `propagate()` in `agent_bridge.decide()`, AB test actually switches the LLM model used for decisions
- LLM models upgraded in TradingAgents: `gpt-5.2` → `gpt-5.4`, `glm-4.7` → `kimi-k2.5` (5 files: `.env`, `.env.example`, `default_config.py`, 2 test files)

### Tested
- 897 tests passed; +749/-24 lines changed across 9 files (Batch 1-3)

### Files
- `src/data/fx_duckdb_store.py`, `src/decision/agent_bridge.py`, `src/decision/tactical_validator.py`
- `src/decision_store/sqlite_store.py`, `src/execution/engine.py`, `src/main.py`
- `src/optimize/optimization_engine.py`, `src/scheduler/scheduler.py`
- `src/data/fx_data_fetcher.py`, `src/config.py`, `src/optimize/optimization_state.py`
- `config/e8_one_5k_challenge.yaml`
- `tests/test_ab_routing.py` (NEW), `tests/test_config.py`, `tests/test_decision_store.py`
- `tests/test_fx_duckdb_store.py`, `tests/test_scheduler.py`, `tests/test_scheduler_multi_timeframe.py`
- `tests/test_tactical_validator.py`
- `tests/test_ab_model_switching.py` (NEW), `tests/test_fx_data_fetcher.py`
- TradingAgents: `.env.example`, `tradingagents/default_config.py`, `tests/test_recursion_limit.py`, `tests/test_telegram_model_switch.py`

## [1.3.8] — 2026-03-09
**v1.3.7 Production Bugfixes — Cross-Contamination, LLM Bias, Over-Filtering, Infinite Loops**

### Fixed
- **P1 #1/#6**: EURUSD 98.7% cancellation rate — added a cold-start threshold tier (0.55 blended) in `thresholds.py`
- **P1 #3**: 160 rescans in 5 days — raised volatility threshold to 0.5%, added a 30-minute cooldown, removed auto-rescan on position close
- **P1 #7**: Best Day infinite retry loop — added `best_day_paused_today` daily stop flag in scheduler
- **P1 #2 (partial)**: Tactical Gate always produced identical output — gate now pass-through when bar data is unavailable
- **P0 #15**: EURUSD evaluation used AUDUSD data — fixed `self.ticker` race condition in `trading_graph.py` by passing ticker as a parameter
- **P1 #8**: 95% SELL bias — randomized BUY/HOLD/SELL option order in the signal extraction prompt
- **P1 #12**: LLM refused trading instructions — added explicit authorization and simulation context to trader agent prompt

### Added
- None.

### Changed
- None.

### Tested
- 879 tests passed (prop-firm-pilot)
- TradingAgents tests passed

### Files
- prop-firm-pilot: `src/config.py`, `src/optimize/thresholds.py`, `src/scheduler/scheduler.py`, `src/decision/tactical_validator.py`, `tests/test_config.py`, `tests/test_scheduler.py`, `tests/test_thresholds.py`
- TradingAgents: `tradingagents/graph/trading_graph.py`, `tradingagents/agents/traders/trader.py`

## [1.3.7] — 2026-03-04

**Tactical Execution Module — Shadow Mode Entry Validation & Decision Caching**

### Added
- **TacticalValidator** module (`src/decision/tactical_validator.py`): low-timeframe entry validation with hard gates (spread, ATR regime) and soft gates (momentum, volatility rank, session quality)
  - 5 technical indicator functions: SMA, EMA, RSI, ATR, Bollinger Bands
  - Configurable gate weights and thresholds via `TacticalConfig`
  - Shadow mode: logs gate results without blocking trades (preparation for future enforcement)
- **StrategicDecisionCache** (`src/scheduler/decision_cache.py`): TTL-based LLM decision deduplication to prevent redundant LLM calls for the same symbol within a configurable window
- **`tactical_pending` intent status**: new state in DecisionStore for intents awaiting tactical validation (between `claimed` and `ready_for_exec`)
- **Consolidated changelog** (`docs/PropFirmPilot_changelog.md`): added v1.0.0-v1.3.6 history
- **v1.3.7 roadmap** (`docs/PropFirmPilot_v1.3.5_road_map.md`): tactical execution module design and implementation plan

### Changed
- Scheduler pipeline extended: `claimed` -> `tactical_pending` -> `ready_for_exec` (new intermediate state)
- `config/e8_one_5k_challenge.yaml`: added tactical execution configuration block (gate weights, thresholds, shadow mode flag)
- `src/config.py`: added `TacticalConfig` Pydantic model with validation

### Tested
- 16 files changed, +2,473/-10 lines
- New test files: `test_tactical_validator.py` (286 lines), `test_decision_cache.py` (83 lines), `test_tactical_integration.py`
- Updated: `test_config.py`, `test_decision_store.py`, `test_scheduler.py`, `test_schemas.py`

### Files
- `src/decision/tactical_validator.py` (NEW), `src/scheduler/decision_cache.py` (NEW)
- `src/config.py`, `src/decision/schemas.py`, `src/decision_store/sqlite_store.py`, `src/scheduler/scheduler.py`
- `config/e8_one_5k_challenge.yaml`
- `docs/PropFirmPilot_changelog.md`, `docs/PropFirmPilot_v1.3.5_road_map.md`
- Tests: `test_tactical_validator.py`, `test_decision_cache.py`, `test_tactical_integration.py`, `test_config.py`, `test_decision_store.py`, `test_scheduler.py`, `test_schemas.py`


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
