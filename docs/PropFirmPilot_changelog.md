# PropFirmPilot Changelog

All notable changes to the PropFirmPilot trading system.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) style.
Versioning: [Semantic Versioning](https://semver.org/).

> **Scope**: This changelog covers the core `prop-firm-pilot` repository.
> Cross-repo impacts on `TradingAgents` and `qlib_market_scanner` are noted where relevant.

---

## [Unreleased]
**Post-v1.5.0_preview Stable Acceptance Window**

> **Status**: `v1.5.0_preview` 已於 `2026-03-17` 成為當前主線 preview release，建立在 `v1.5.0_beta_2` operational repair baseline 之上
> **Reason**: core pilot 已先完成 bounded capital utilization uplift preview implementation，並同步核對 `qlib_market_scanner` / `qlib_rd_agent` 的 upstream contract 已落地；下一步不是再做同一輪 feature 實作，而是把 preview uplift 與 exposure / memory / validation closure 一起驗收成 `v1.5.0 stable`

### Implemented: Side-Aware Scanner Live Activation
- side-aware FX scanner live path 現已在 preview acceptance window 落地；`config/e8_one_5k_challenge.yaml` 啟用 `scanner.topk_short = 1`，讓 live ingestion 可以同時接受 long 與 bounded short 候選
- `ScannerBridge` 現在支援 `fx_signal_v2`，會把 `scanner_side` 與 `scanner_direction_quality` 帶進 TradingAgents context；`TradeIntent` / `DecisionStore` 也會持久化 `scanner_side`
- scheduler 現在會以 `(symbol, side)` 去重，對 mixed long/short bundle 以 direction-aware quality 排序，並把 side-aware quality 套進 blended confidence、decision cache key 與 decision formatting
- TradingAgents 對 scanner side 只剩 confirm / veto 權限；reverse actionable decision 會被取消為 `direction_mismatch`
- legacy `PropFirmPilot.run_daily_cycle()` 現在也補齊同樣的 direction-aware ranking 與 reverse-side hard veto，不再只在 scheduler mode 正確

### Cross-Repo Note
- `qlib_market_scanner` 已補上 `--topk-short` runtime activation，signals export 會保留 `configured_topk_short`，作為 preview live activation 的上游契約
- 本 repo 這輪工作不是重做 scanner alpha，而是把 upstream side-aware bundle 變成 live-safe ingestion contract

### Validated
- `uv run pytest tests/test_config.py tests/test_scanner_bridge.py tests/scheduler/test_llm_thresholds.py tests/test_scheduler.py -q` → `218 passed`
- `uv run pytest tests/test_main_daily_cycle.py tests/test_main_logging.py -q` → `5 passed`
- `uv run ruff check src/main.py tests/test_main_daily_cycle.py` → `All checks passed!`
- `uv run ruff format --check src/main.py tests/test_main_daily_cycle.py` → `2 files already formatted`

### Planned: v1.5.0 (stable) — Stable Acceptance On Top Of Preview Uplift
- 定義第一個「穩定版」基準：系統必須能穩定可靠地進出場，而不是只靠 hotfix 維持可用
- 將 `v1.5.0_preview` 已落地的 bounded capital utilization uplift 納入 multi-day stable acceptance，而不是把 uplift 本身重新實作一次
- tactical entry 與 tactical exit 需達到 production-grade reliability，包含 data provenance、execution integrity、close verification 與 postmortem replayability
- 把 `v1.5.0_beta` 已吸收的 scanner contract、cadence decision 與 entry / exit control plane 升級為 stable release gate，而不是再做一次大重寫
- 不重新開啟 `qlib_market_scanner` 的 FX release cadence 選型；`v1.5.0 stable` 直接沿用已凍結的 `1d` canonical cadence
- 重新設計 `TradingAgents` 與 intraday FX 場景的耦合方式，使其能消化更高頻率的 scanner / market context 與交易記憶
- 建立穩定可靠的交易記憶體系，將 trade journal、reflection、lesson memory、execution outcome 串成可持續改善的 learning loop
- 將多元化倉位與資金效率正式納入風控與配置層，而不是只做單筆交易最小風險化

### Cross-Repo Note
- `v1.5.0_preview` 已建立在已核對的 upstream contract 之上：`qlib_market_scanner` 已具備 `1d` canonical cadence、`dsr_net_oos_daily_v1`、scorecard / decision artifacts 與 RD-Agent factor promotion gate；`qlib_rd_agent` 已具備 `discovered/candidate/manifest` artifact 與 `runs/<run_id>/...` archive contract
- `v1.5.0 stable` 的工作重點是 integrated acceptance，不是再替 upstream scanner / rd-agent 重新補 contract plumbing

### Tracking
- Roadmap: `docs/PropFirmPilot_v1.5.0_road_map.md`
- Cross-repo note: `docs/PropFirmPilot_v1.5.0_Cross_Repo_Change_Note.md`
- Implementation plan: `docs/plans/2026-03-17-v1.5.0-side-aware-live-activation.md`

---

## [1.5.0_preview] — 2026-03-17
**Bounded Capital Utilization Preview**

> **Status**: 已成為當前主線 preview release，作為 `v1.5.0 stable` 前的 bounded uplift implementation lane
> **Reason**: `v1.5.0_beta_2` 已先完成 operational repair gate，但 execution sizing 仍固定鎖在單筆 `default_risk_pct`；preview 的任務是先用 bounded、audit-friendly 的方式改善 sparse portfolio 下的資金使用率，而不提前宣稱 stable acceptance 已完成

### Added
- `src/execution/capital_allocator.py`：新增 `BoundedCapitalAllocator` 與 `CapitalAllocationDecision`，用 `default_risk_pct * max_positions` 的名目預算、`open_positions` 與 `scanner_confidence` 計算單筆 `effective_risk_pct`
- `PositionSizer.calculate_volume()` 現在支援 `risk_pct_override`，execution path 可以把 bounded uplift 後的風險百分比直接帶入 sizing
- `ExecutionEngine` 現在會在 `_build_trade_plan()` 階段套用 bounded allocation，並把 `risk_pct` / `capital_allocation` 寫入 execution metadata，保留 audit trail
- 新增 preview 專屬測試覆蓋：capital allocator、risk override、engine integration、execution meta risk audit fields、preview version identity

### Changed
- 專案版本 identity 現在切到 `display_version = 1.5.0_preview`，包裝版本切到 `1.5.0rc0`，讓 runtime / release tag / docs 都明確表達這是 preview lane
- `docs/PropFirmPilot_v1.5.0_road_map.md` 現在改以 `beta_2 -> preview uplift -> stable acceptance` 作為 `1.5.0` 主線敘事

### Cross-Repo
- `qlib_market_scanner` 已具備 FX 7-pair universe、`dsr_net_oos_daily_v1` selection metric、`label_version` / `research_cadences` metadata、cadence scorecard / decision artifacts，以及 `candidate -> promoted -> report` RD-Agent factor gate
- `qlib_rd_agent` 已具備 `discovered_factors.yaml`、`candidate_factors.yaml`、`factor_manifest.json` 與 `runs/<run_id>/...` archive upload contract；preview uplift 直接建立在這些已落地的 upstream 邊界上

### Validated
- `uv run pytest tests/test_capital_allocator.py -q` → `4 passed`
- `uv run pytest tests/test_position_sizer.py -q` → `34 passed`
- `uv run pytest tests/test_engine.py -q` → `58 passed`
- `uv run pytest tests/test_execution_meta.py -q` → `10 passed`
- `uv run pytest tests/test_version.py -q` → `4 passed`
- `uv run ruff check src/execution/capital_allocator.py src/execution/position_sizer.py src/execution/engine.py tests/test_capital_allocator.py tests/test_position_sizer.py tests/test_engine.py tests/test_execution_meta.py tests/test_version.py` → `All checks passed!`

### Files
- New: `src/execution/capital_allocator.py`, `tests/test_capital_allocator.py`
- Modified: `src/execution/engine.py`, `src/execution/position_sizer.py`, `tests/test_engine.py`, `tests/test_execution_meta.py`, `tests/test_position_sizer.py`, `tests/test_version.py`, `pyproject.toml`, `docs/PropFirmPilot_v1.5.0_road_map.md`

---

## [1.5.0_beta_2] — 2026-03-17
**Operational Hardening Repair Release**

> **Status**: 已合入 `main`，作為 `v1.5.0 stable` 前的 operational repair baseline
> **Reason**: `2026-03-16` 晚間到 `2026-03-17` 上午的 production run 暴露出 stale signal fallback、market-data degraded entry、startup first-run `5m` bar gap、tactical pending lifecycle、sqlite snapshot write safety 與 tactical alert noise 等 live correctness 缺口，需先收斂後才能進 stable gate

### Fixed
- `ScannerBridge` 現在在 live target date 缺失時 fail-closed，不再 fallback 到最新可用 signals，且 stale rejection 不會污染 pipeline cache
- `EODHDFXWebSocketClient` 與 `MarketDataHub` 現在會把 WebSocket `keepalive ping timeout` / reconnect failure 映射成 entry-visible degraded state，scheduler 在 feed 不安全時直接 block 新 intents / 新開倉
- 啟動後第一輪 scanner 若遇到 `quote fresh + websocket healthy + first 5m closed bar pending`，現在不再因 `bars_5m_unavailable` 直接白跑；scanner 會保留 candidate、建立 intent，並交由 tactical 層以 `market_data.startup_5m_bar_pending` 走 `WAIT / RETRY_PENDING`
- tactical pending lifecycle 現在會把 `expires_at` 對齊完整 retry budget，避免 janitor 在 tactical retry 尚未跑完前提早回收 intent
- `SQLiteDecisionStore.insert_equity_snapshot()` 現在與其他寫入共用 `_write_lock`，避免長跑中出現 nested transaction 錯誤
- tactical Telegram alert 現在有 keyed throttling / dedupe，且 trade event payload 會帶出 scanner / feed / deadline diagnostics
- `MarketDataHub.feed_status()` 與 scheduler `SCANNER_SKIP` payload 現在會帶出 `initialized_at`、`uptime_seconds` 與 per-symbol websocket closed bar counts，讓 `market_data.bars_5m_unavailable` 能直接區分是 websocket closed bars 尚未 ready，還是 REST fallback 本身 stale

### Changed
- runtime shared version source 現在顯示 `1.5.0_beta_2`，release tag / packer 預設值同步輸出 `v1.5.0_beta_2`
- pilot ingestion gate 現在同時接受 `v1.5.0`、`v1.5.0_beta` 與 `v1.5.0_beta_2` scanner bundle，保留與既有 beta artifacts 的向後相容
- `config/e8_one_5k_challenge.yaml` 現在把 off-hours scanner cadence 從 `7200s` 降到 `3600s`，降低 market-data block 後在淡時段等待下一輪 scanner 的時間
- `config/e8_one_5k_challenge.yaml` 現在依「高頻常調參數 / 低頻基礎參數」重排，並為主要欄位補齊中文作用註解，讓日常調參與 incident 時的 config triage 更直接
- scheduler 現在會對 startup `5m` warmup gap 記錄 `SCANNER_ADMITTED` 診斷事件，讓 operator 能區分「正常等待第一根 websocket `5m` close」與真正的 market-data hard block

### Validated
- `uv run python -m pytest tests/data/test_market_data_hub.py tests/test_scheduler.py tests/test_config.py tests/test_tactical_integration.py -q` → `161 passed`
- `uv run ruff check src/data/fx_tick_aggregator.py src/data/market_data_hub.py src/scheduler/scheduler.py tests/data/test_market_data_hub.py tests/test_scheduler.py tests/test_config.py tests/test_tactical_integration.py` → `All checks passed!`
- `uv run python -c "from src.config import load_config; cfg = load_config('config/e8_one_5k_challenge.yaml'); print(cfg.account.initial_balance); print(cfg.scheduler.quiet_session_interval_seconds); print(cfg.tactical.soft_gates.min_score)"` → `5000.0` / `3600` / `2`

---

## [1.5.0_beta] — 2026-03-16
**Scanner Contract Gate And Cross-Repo Beta Baseline**

> **Status**: 已合入 `main`，作為 `v1.5.0 stable` 的 beta integration baseline
> **Reason**: 上游 `qlib_market_scanner` 已凍結 FX cadence decision 與 versioned artifact contract；pilot 必須在主線正式吸收 bundle validation、metadata persistence 與 beta version identity，才能開始 stable gate 驗證

### Added
- `ScannerBridge` 現在會在 ingest 前驗證 `manifest.json` / `metrics.json` 的 schema version、scanner version、validation status 與 required signal columns
- `TradeIntent` / `DecisionStore` 現在會持久化 `scanner_version`、`scanner_schema_version`、`scanner_market_date` 與 `scanner_label_version`
- scanner bundle 被拒收時，scheduler 現在會記錄 `SCANNER_BUNDLE_REJECTED` trade event 並發出對應 alert
- scanner fixtures 現在帶有 versioned manifest / metrics sidecars，測試不再只依賴裸 `signals.csv`

### Changed
- runtime shared version source 現在顯示 `1.5.0_beta`，讓 runtime / release tag / packer 版本一致
- pilot ingestion 現在同時接受 `v1.5.0` 與 `v1.5.0_beta` scanner bundle，兼容既有研究 artifacts 與 beta upstream release
- 上游 `qlib_market_scanner` 已正式凍結 FX canonical release cadence 為 `1d`；pilot runtime 不因這次 beta 升級而切換 `scanner_timeframe`
- runtime `outputs/signals/signals.csv` 現在只會解析同一 bundle family 下的 `outputs/manifest.json` 與 `outputs/metrics/metrics.json`，不再回退讀取 legacy `data/shared_export` sidecars

### Cross-Repo
- 上游 `qlib_market_scanner` 已在 `v1.5.0_beta` 完成 FX cadence matrix、versioned export contract、DuckDB intraday timestamp preservation 與 cadence-leg isolation
- 本 repo 的責任是把這些 artifacts 轉成可拒收 degraded / stale / invalid bundle 的 ingestion gate，而不是自行重做 cadence 選型

### Validated
- `uv run pytest tests/test_scanner_bridge.py tests/test_scheduler.py tests/test_version.py -q` → `169 passed`
- `uv run ruff check src/signal/scanner_bridge.py tests/test_scanner_bridge.py tests/test_scheduler.py tests/test_version.py` → `All checks passed!`
- `uv run python -c "from src.version import get_app_version, get_release_tag; print(get_app_version()); print(get_release_tag())"` → `1.5.0_beta` / `v1.5.0_beta`

---

## [1.4.9] — 2026-03-16
**Market Data Freshness Semantics And Tactical Warning Throttling**

> **Status**: 已合入 `main`，等待下一輪 market-open production validation
> **Reason**: `v1.4.8a` production follow-up 顯示 tactical exit 雖然有持續執行，但 closed `1h` bar freshness 仍以 open-time 判斷，且 stale tactical-bar warning 會對相同狀態每分鐘重複刷屏，導致 operator 難以分辨是 provider 資料過舊還是本地 freshness 語義偏差

### Fixed
- `MarketDataHub` 的 bar freshness 現在以 effective close time 判斷，不再用 bucket open time 錯殺 websocket 聚合出的 closed `1h` bar
- scheduler tactical stale-bar sanitize 也改為用 effective close time 判斷，避免 closed `1h` bar 在 live exit path 被過早視為 stale
- repeated identical stale tactical-bar warnings 現在會做 stateful throttling；內容不變時只保留 15 分鐘 heartbeat，不再每分鐘刷同一條 warning

### Added
- `MarketDataHub` REST fallback warning 現在同時輸出 `latest_rest_bar_open_time`、`latest_rest_bar_close_time` 與 `latest_rest_bar_age_by_close_sec`
- tactical stale-bar warning 現在同時輸出 `latest_open` 與 `latest_close`，可直接區分 provider 資料本身過舊與本地 freshness semantics 問題
- regression tests 覆蓋 closed `1h` websocket bar 以 close time 判定仍屬 fresh、stale tactical warning throttling，以及 fallback instrumentation 欄位

### Validated
- `uv run pytest tests/data/test_market_data_hub.py tests/test_scheduler.py tests/test_tactical_exit_scheduler.py -q` → `142 passed`
- `uv run ruff check src/data/market_data_hub.py src/scheduler/scheduler.py tests/data/test_market_data_hub.py tests/test_scheduler.py tests/test_tactical_exit_scheduler.py docs/plans/2026-03-16-market-data-freshness-and-warning-design.md docs/plans/2026-03-16-market-data-freshness-and-warning.md` → `All checks passed!`
- `uv run python -c "from src.version import get_app_version, get_release_tag; print(get_app_version()); print(get_release_tag())"` → `1.4.9` / `v1.4.9`

---

## [1.4.8a] — 2026-03-16
**Tactical Exit Observability And Runtime Contract Follow-Up**

> **Status**: 已合入 `main`，等待下一輪 market-open production validation
> **Reason**: production run 暴露出 runtime version 仍顯示為 `v1.4.6d`、scanner 未將 configured `topk` 傳遞到 CLI、tactical exit 在純 `HOLD` 情境下缺少可觀測性，且 stale `5m/1h` tactical bars 仍可能進入 live exit 判斷

### Fixed
- `pyproject.toml` 的 `display_version` 已更新到 `1.4.8a`，runtime / packer / release tag 現在會一致顯示 `v1.4.8a`
- `ScannerBridge.run_pipeline()` 現在會把 configured `--topk` 傳給 `qlib_market_scanner`，使 pilot runtime 與 scanner subprocess 的 universe / ranking 契約一致
- `Scheduler` 的 position monitor 現在會採用較快的 tactical exit cadence，不再讓 `tactical.exit.evaluation_interval_seconds` 實際上被較慢的 monitor interval 蓋掉
- tactical exit cycle 現在即使只有 `HOLD` 結果，也會輸出 operator-visible summary log，方便確認 open position 仍被持續監測
- stale `5m/1h` tactical bars 現在會在 scheduler 入口先被過濾，避免像 prod log 中那種過舊的 REST fallback bars 直接進入 live exit decision path

### Changed
- `docs/PropFirmPilot_v1.5.0_Profitability_Outlook_Report.md` 的 scanner universe 描述已改成更精確口徑，明確區分 `qlib_market_scanner` 研究 baseline 與 `prop-firm-pilot` runtime 可覆寫 universe

### Added
- 回歸測試覆蓋 scanner CLI `--topk` 傳遞
- 回歸測試覆蓋 stale `5m/1h` tactical bars 會被 scheduler 丟棄
- 回歸測試覆蓋 tactical exit hold-only cycle 仍會產生摘要 log，且 position monitor cadence 會 honour tactical exit interval

### Validated
- `uv run pytest tests/test_scanner_bridge.py tests/test_scheduler.py tests/test_tactical_exit_scheduler.py tests/test_version.py -q` → `163 passed`
- `uv run ruff check src/scheduler/scheduler.py src/signal/scanner_bridge.py tests/test_scanner_bridge.py tests/test_scheduler.py tests/test_tactical_exit_scheduler.py tests/test_version.py` → `All checks passed!`
- `uv run python -c "from src.version import get_app_version, get_release_tag; print(get_app_version()); print(get_release_tag())"` → `1.4.8a` / `v1.4.8a`

---

## [1.4.8] — 2026-03-14
**Close Control Plane And Canonical Close Reconciliation**

> **Status**: 已合入 `main`，等待 market-open production validation
> **Reason**: tactical exit、reduce exposure、emergency close、best-day close、LLM re-eval close 與 external-detected close 之間仍缺少單一 close-domain contract，導致 journal、metadata、close reason 與 broker action 邊界漂移

### Added
- `src/decision/close_models.py`：新增 `CloseIntent`、`CloseOutcome`、`CloseReconciliation` typed close-domain schema
- `src/decision/close_control_plane.py`：新增單一 close execution service，統一路由 `modify_only`、`partial_close`、`full_close`
- `src/decision/close_reconciler.py`：新增 canonical close reconciler，統一 `trigger_source`、`action_kind`、`final_close_reason`、`resolution_path`
- 新增 close-domain 測試：`tests/test_close_models.py`、`tests/test_close_control_plane.py`、`tests/test_close_reconciler.py`

### Changed
- `Scheduler` 現在把 tactical exit、drawdown reduce exposure、emergency close、best-day close、LLM re-eval close 全部改走 `CloseControlPlane`
- `_handle_position_closed()` 現在以 `CloseReconciler` 產生 canonical close facts，並維持 `DecisionStore.exit_reason = final_close_reason` 的向後相容
- `TRADE_CLOSED` 與 `CLOSE_CONTROL_EVENT` 現在攜帶統一的 close-control 欄位，降低 journal / alert / postmortem 漂移
- tactical modify / trailing / breakeven / partial close 的 broker read-back 與 pending close tracking 現在共用同一條控制面

### Validated
- `uv run pytest tests/test_close_models.py tests/test_close_control_plane.py tests/test_close_reconciler.py tests/test_tactical_exit_execution.py tests/test_tactical_exit_scheduler.py tests/test_reevaluation.py tests/test_exit_reason_classification.py tests/test_scheduler.py tests/monitor/test_equity_monitor.py -q` → `168 passed`
- `uv run ruff check src/decision/close_models.py src/decision/close_control_plane.py src/decision/close_reconciler.py src/scheduler/scheduler.py tests/test_close_models.py tests/test_close_control_plane.py tests/test_close_reconciler.py tests/test_tactical_exit_execution.py tests/test_tactical_exit_scheduler.py tests/test_reevaluation.py tests/test_exit_reason_classification.py tests/test_scheduler.py tests/monitor/test_equity_monitor.py` → `All checks passed!`
- `uv run pytest tests/test_circuit_breaker.py tests/test_operational_metrics.py tests/test_memory_journal.py -q` → `37 passed`

### Files
- `src/decision/close_models.py`
- `src/decision/close_control_plane.py`
- `src/decision/close_reconciler.py`
- `src/scheduler/scheduler.py`
- `tests/test_close_models.py`
- `tests/test_close_control_plane.py`
- `tests/test_close_reconciler.py`
- `tests/test_tactical_exit_execution.py`
- `tests/test_reevaluation.py`
- `tests/test_scheduler.py`

---

## [1.4.6d] — 2026-03-13
**MarketDataHub REST Fallback Loop Fix**

> **Status**: Released market-data hotfix
> **Reason**: `5m/1h` stale same-tail bar fetches were still re-hitting REST on every tactical read, and websocket rollup bars were not being elapsed-closed before hub lookup

### Fixed
- `MarketDataHub.get_bars()` 現在在讀取 websocket cache 前會先執行 `FXTickAggregator.close_elapsed_bars()`，讓已到期的 `1m/5m/1h` buckets 可以在 hub lookup 時即時 finalize，而不是只能等下一個跨 bucket tick
- `MarketDataHub` 的 REST refresh suppression 不再只限於 `1m` quote path；`5m` / `1h` stale same-tail bar path 現在也會套用同一套 cooldown + no-progress guard
- `get_bars()` 在 cooldown 內遇到同一個 stale REST tail 時，不再重複 refresh 或重複刷 fallback warning，而是重用既有 warm cache 內容走 `rest_fallback` 讀路徑
- shared version helper 現在會優先讀取 `pyproject.toml` 內的 `display_version`，讓 runtime / packer / release tag 可維持 `v1.4.6d`，同時保留 `[project].version` 的 PEP 440 相容性供 `uv` 使用

### Added
- regression tests 覆蓋 stale `5m` / `1h` REST same-tail suppression
- regression test 覆蓋 `MarketDataHub.get_bars()` 會在 lookup 前 finalize elapsed `5m` rollup bars
- version helper regression test 改為鎖定 `display_version` 對外顯示行為

### Validated
- `uv run python -c "from src.version import get_app_version, get_release_tag; print(get_app_version()); print(get_release_tag())"` → `1.4.6d` / `v1.4.6d`
- `uv run pytest tests/data/test_market_data_hub.py tests/data/test_fx_tick_aggregator.py tests/test_version.py -q` → `20 passed`
- `uv run ruff check src/data/market_data_hub.py src/version.py tests/data/test_market_data_hub.py tests/test_version.py` → `All checks passed!`

---

## [1.4.6c] — 2026-03-13
**Scanner CLI Backward-Compatibility Hotfix**

> **Status**: Released production compatibility hotfix
> **Reason**: Production target still had an older `qlib_market_scanner` CLI that rejected `--benchmark FX`, causing scanner subprocess failure even though the new pilot release expected benchmark-aware scanner versions

### Fixed
- `ScannerBridge.run_pipeline()` 現在遇到 scanner CLI 回報 `unrecognized arguments: --benchmark ...` 時，會自動移除 `--benchmark <value>` 後重試一次，保持對舊版 `qlib_market_scanner` 的向下相容
- benchmark-aware 新版 scanner 仍維持原本 `--benchmark FX` 呼叫路徑，不影響已升級環境
- 若舊版 CLI 只是不支援 benchmark 參數，prop-firm-pilot 掃描流程不再因參數不相容而中斷

### Added
- 回歸測試覆蓋「第一次 subprocess 因 `--benchmark` 不被支援而失敗，第二次自動改成不帶 benchmark 重試」的 prod 相容性案例

### Validated
- `uv run pytest tests/test_scanner_bridge.py::TestScannerBridgeInit::test_run_pipeline_retries_without_benchmark_when_cli_rejects_argument` → `1 passed`
- `uv run pytest tests/test_scanner_bridge.py` → `39 passed`
- `uv run ruff check src/signal/scanner_bridge.py tests/test_scanner_bridge.py` → `All checks passed!`

---

## [1.4.6b] — 2026-03-13
**Tactical Freshness Recovery + Prod Bundle Dropbox Sync**

> **Status**: Released tactical-entry hotfix and diagnostics workflow upgrade
> **Reason**: Tactical WAITs were still getting stuck on `data_freshness` during degraded market-data periods, and prod diagnostics packaging/export was still too manual for fast incident analysis

### Fixed
- `Scheduler._fetch_tactical_data()` 不再因為 hub 只拿到 bars 就提早返回；現在若 hub 沒有提供可用 quote timestamp，仍會回退到 MatchTrader quote 取得 freshness timestamp
- tactical hard gate 的 `data_freshness` 在 websocket degraded / mixed-source 情境下，不會再因為略過 broker quote fallback 而錯誤卡住進場
- 針對這次 `NZDUSD` / `USDCHF` 症狀做了 live probe；EODHD websocket 已驗證可對 7 個配置貨幣對持續出 tick，新增 pair 不是根因
- `DropboxArtifactsClient` 對 Dropbox `not_found` 的判斷改為只接受真正的 `LookupError.not_found`，避免把其他 path 類錯誤誤當成缺資料夾
- `pack_prod_logs.py` 生成 log summary 時，現在會正確讀取 run-specific log files，不會因為新的 logging 策略而漏讀主要執行日誌

### Added
- `scripts/pack_prod_logs.py` 現在會把 diagnostics zip 自動上傳到 Dropbox 路徑 `/prop-firm-pilot/prod_logs/<account_name>/`
- 新增 `scripts/unpack_prod_logs.py`，可從 Dropbox 下載最新 prod bundle 並解壓到 repo 根目錄
- prod bundle 內新增 `bundle_manifest.json`、`raw/config/` config snapshots，以及 manifest 內的 included config file listing，方便 LLM 對齊版本、account、config 與 included files
- 新增回歸測試，覆蓋「hub 只有 bars、沒有 freshness quote 時仍須回退到 MatchTrader」以及 Dropbox `not_found` 錯誤辨識

### Changed
- runtime logging 改為每次程序啟動建立新的 run-specific log file，而不是持續增長單一 `prop_firm_pilot.log`
- `pack_prod_logs.py` 的 log 收集與摘要來源改為優先匹配 `prop_firm_pilot_*.log*` 與舊格式 `prop_firm_pilot.log*`，減少無關 log 噪音並對齊新的 per-run logging

### Operational Notes
- `2026-03-13` live websocket probe 已證實：`EURUSD / GBPUSD / USDJPY / AUDUSD / NZDUSD / USDCAD / USDCHF` 全部都有 websocket ticks
- 同一次 probe 也再次證實：EODHD REST `1min` bar 仍明顯落後當前時間，因此 degraded 狀態下仍可能看到 `REST fallback` warning；本版修的是 tactical freshness 不再因此被誤卡死，不是根治 EODHD REST lag

---

## [1.4.5a] — 2026-03-13
**Tactical Re-entry And Stale Quote Guard**

> **Status**: Released hotfix for the first `v1.4.5` production run
> **Reason**: Post-`2026-03-12 23:00` production artifacts showed same-direction entry blocking after flat closes, and `MarketDataHub` was still emitting synthetic quotes from clearly stale REST `1m` tails during websocket degradation

### Fixed
- `same-direction` duplicate entry guard no longer counts `closed` intents as active same-day attempts
- same-day re-entry is now allowed again once the symbol is flat, while still counting `ready_for_exec` / `executing` / `opened` intents as active
- `MarketDataHub.get_quote()` no longer returns a synthetic quote when REST fallback `1m` bars are still stale after refresh / no-progress suppression
- stale REST `1m` fallback data can no longer feed downstream quote consumers such as volatility monitoring as if it were current market data

### Root Cause
- `DecisionStore.count_same_direction_today()` counted both `opened` and `closed` intents
- with `max_same_direction_per_day = 1`, one completed AUDUSD SELL on `2026-03-12` caused later flat-state AUDUSD SELL intents to be cancelled with `Same-direction limit ... already attempted 1x today`
- `MarketDataHub.get_quote()` used bar freshness only to decide whether to refresh REST cache, but still built and returned a quote from the stale cached tail when that refresh produced no fresh progress

### Added
- regression coverage proving closed same-direction trades do not block re-entry when flat
- regression coverage proving stale REST fallback tails do not produce a usable quote

### Files
- `src/decision_store/sqlite_store.py`
- `src/data/market_data_hub.py`
- `tests/test_duplicate_entry_limit.py`
- `tests/data/test_market_data_hub.py`
- `pyproject.toml`

---

## [1.4.1] — 2026-03-12
**Production Reliability / Observability Hardening**

> **Status**: Released reliability / observability hardening release
> **Reason**: Defined from the production incident review of `2026-03-11 16:33` to `2026-03-12 13:58` (UTC+8)

### Target Scope
- **P0**: 修復 reflection / memory result identity mapping，防止 closed-trade learning loop 被 cross-symbol outcome 污染
- **P1**: 建立單一版本來源，消除 runtime / pack / docs version drift
- **P1**: 收斂 WebSocket degraded 時的 REST fallback 成本，補齊 market-data source health metrics
- **P1**: 強化 Telegram polling circuit degrade / recover 的 metrics 與告警
- **P2**: 補齊 cooldown / cancel loop 的診斷資料與 log bundle manifest 完整性

### Implemented In This Worktree
- `src/version.py` 成為 shared version helper；runtime startup log 與 `scripts/pack_prod_logs.py` 改為共用同一版本來源，explicit `--version` mismatch 會 fail-fast
- `TelegramBotHandler` 現在會把 polling failure、circuit open、probe、recovery transition 寫入 shared `OperationalMetrics`
- `MarketDataHub` degraded path 改為優先重用 REST-backed warm cache，cache stale 時再做 incremental REST refresh，不再每次都回退到固定 lookback 全段重抓
- `MarketDataHub` 新增 same-tail REST refresh suppression：當 stale `1m` fallback 沒有拿到更晚的最新 bar 時，短時間內不再重複整段重抓，並在 warning 補上 `latest_rest_bar_time` / `latest_rest_bar_age_sec`
- `Scheduler` 的 `METRICS_SNAPSHOT` 現在會附帶 market-data feed status，讓 postmortem 可直接看到 websocket `healthy / degraded / disconnected` 狀態
- `Scheduler` 現在會把 `recent_rejection_cooldown` 與 `low_confidence_cooldown` 寫成 structured `SCANNER_SKIP` event，讓 cooldown / cancel loop 可以直接回放分析
- `scripts/pack_prod_logs.py` 現在會產生 deterministic `decisions_summary.md` / `telegram_summary.md` fallback，且 `INDEX.md` 只列出實際存在的 summary files，不再引用不存在的條目
- 新增 `scripts/check_eodhd_websocket_live.py` 與 `src/diagnostics/eodhd_websocket_live.py`，可在 target environment 直接驗證 websocket tick flow、per-symbol tick gap，以及 EODHD REST `1min` bar lag

### Tracking
- Incident report: `docs/PropFirmPilot_v1.4.1_Incident_Report.md`
- Release checklist: `docs/PropFirmPilot_v1.4.1_Release_Checklist.md`
- Websocket / fallback design: `docs/plans/2026-03-12-websocket-rest-fallback-design.md`
- Websocket / fallback implementation plan: `docs/plans/2026-03-12-websocket-rest-fallback-fix.md`
- Next planned continuation: `docs/PropFirmPilot_v1.4.0_road_map.md` (`v1.4.2`)

### Operational Notes
- 2026-03-12 的 live probe 已證實：EODHD websocket 在 target environment 可持續收到 `EURUSD / GBPUSD / USDJPY / AUDUSD` ticks，問題不在 parser 或 subscribe path
- 同一次 probe 也證實：EODHD REST `1min` 當日資料最新 bar 仍明顯落後當前時間；因此 repeated fallback 的主要成本來自 stale REST tail，而非 websocket 無法出數
- `v1.4.1` 對這個現象的處理目標是先壓制 repeated same-tail refresh 並強化診斷，不是把 EODHD REST `1min` provider lag 視為已被根治

---

## [1.4.0] — 2026-03-11
**WebSocket-First Market Data + Closed-Trade Learning Loop**

### Added
- `src/data/fx_websocket_client.py`：EODHD FX WebSocket client，負責 subscribe、tick parse、bounded reconnect、feed stale detection
- `src/data/fx_tick_aggregator.py`：tick -> latest quote / closed `1m` bars / roll-up `5m` + `1h` bars
- `src/data/market_data_hub.py`：統一 market-data read path，支援 `websocket_cache` / `warmup_cache` / `rest_fallback`
- `Scheduler._build_reflection_payload()`：平倉後輸出 structured reflection payload，攜帶 outcome、market context、risk context
- TradingAgents 新測試：`tests/test_memory_reflection.py`、`tests/test_prompt_memory_injection.py`

### Changed
- `src/config.py` + `config/e8_one_5k_challenge.yaml` 新增 `websocket` config block，並預設 `enabled: true`、`use_as_primary_market_data: true`
- `src/scheduler/scheduler.py` 啟動時先 warm up market-data hub，再啟動 WebSocket sidecar；平倉後反射改走 structured payload
- `src/scheduler/volatility_monitor.py` 改為 WebSocket-first quote path；`_fetch_tactical_data()` 改為 local aggregated bars first，REST 僅保留 cold-start / stale fallback
- `src/decision/tactical_validator.py` 新增 `quote_source` / `bars_5min_source` / `bars_1h_source` / `data_source` metadata，明確暴露資料來源
- TradingAgents memory layer 現在會持久化 structured trade lesson metadata，並在 decision-time retrieval 後注入 `retrieved_trade_lessons` 至 trader prompt

### Operational Notes
- 本版採 **aggressive rollout**：WebSocket 成為預設 primary market-data path
- REST 未被移除，只保留 broker state、news、cold-start historical backfill 與 feed degraded fallback
- reflection / memory retrieval 皆維持 best-effort，不可阻塞 broker close flow 或 decision flow

### Tested
- prop-firm-pilot targeted suites：`188 passed`
- TradingAgents targeted suites：`6 passed`
- v1.3.9c regression set：`214 passed`
- prop-firm-pilot lint：`uv run ruff check src tests` → `All checks passed!`
- TradingAgents changed-file lint：`uv run ruff check tradingagents/agents/utils/memory.py tradingagents/graph/reflection.py tradingagents/agents/utils/agent_states.py tradingagents/graph/propagation.py tradingagents/agents/trader/trader.py tradingagents/graph/trading_graph.py tests/test_memory_reflection.py tests/test_prompt_memory_injection.py` → `All checks passed!`
- 完整驗證命令與結果彙總記錄於 `docs/PropFirmPilot_v1.4.0_Report.md`

### Files
- prop-firm-pilot：`src/config.py`, `src/data/fx_websocket_client.py`, `src/data/fx_tick_aggregator.py`, `src/data/market_data_hub.py`, `src/scheduler/scheduler.py`, `src/scheduler/volatility_monitor.py`, `src/decision/tactical_validator.py`
- prop-firm-pilot tests：`tests/data/test_fx_websocket_client.py`, `tests/data/test_fx_tick_aggregator.py`, `tests/data/test_market_data_hub.py`, `tests/test_scheduler.py`, `tests/test_volatility_monitor.py`, `tests/test_agent_bridge_config.py`
- TradingAgents：`tradingagents/agents/utils/memory.py`, `tradingagents/graph/reflection.py`, `tradingagents/graph/trading_graph.py`, `tradingagents/graph/propagation.py`, `tradingagents/agents/utils/agent_states.py`, `tradingagents/agents/trader/trader.py`

---

## [1.3.9c] — 2026-03-11
**OODA Loop Closure — Scheduler Risk Wiring, Tactical Retry, Strategic Cache, Persistent Memory, News Trigger**

### Fixed
- **P0**: Scheduler mode 的 `EquityMonitor` 現在完整接上 `drawdown_warning`、分級減倉、`close_all_positions()` 與 equity snapshot callback，補上 24/7 主流程的風控執行斷點
- **P0**: Tactical `WAIT` 不再直接取消 intent，改為 `claimed -> tactical_pending -> retry -> PASS/degrade/cancel`，真正落地 roadmap 的等待重判閉環
- **P0**: `StrategicDecisionCache` 正式接入 `_process_claimed_intent()`，相同 strategic signal 不再重複呼叫 LLM
- **P1**: Scheduler 啟動時即 refresh optimization state 並同步 AB routing，避免系統長時間跑在 default thresholds / 空 AB 狀態
- **P1**: TradingAgents 記憶鏈路補上 deterministic `session_id`、graph rebuild session continuity，以及持久化 Chroma storage path
- **P2**: Volatility / news trigger 現在都會立即強制一次 equity check，避免事件驅動 rescan 與風控觀察脫節

### Added
- `src/scheduler/news_event_trigger.py`：以 Alpha Vantage NEWS_SENTIMENT 輪詢新 macro headline，觸發早掃描
- `EquityMonitor.check_once()`：支援單次權益檢查與分級反應（alert / reduce exposure / emergency close）
- Historical PnL context 與 market event context 注入 TradingAgents prompt
- 2 個新測試檔：`tests/monitor/test_equity_monitor.py`、`tests/scheduler/test_news_event_trigger.py`

### Changed
- `src/config.py`：新增 `reduce_exposure_pct` 與 news trigger config
- `src/decision/fx_analyst_config.py` / `src/main.py`：Agent config 現在可傳 `memory_path` 與 `session_id`
- TradingAgents `AgentState` / `Propagator` / `trader` prompt 已支援 `historical_pnl_context` 與 `market_event_context`
- TradingAgents memory backend 由 in-memory Chroma 升級為可選 persistent Chroma

### Tested
- **208 related tests passed** across scheduler / decision cache / decision store / agent bridge / optimization integration / trade journal / equity monitor / news trigger
- Ruff lint: `uv run ruff check src tests` → 0 errors

### Files
- Modified: `src/config.py`, `src/decision/agent_bridge.py`, `src/decision/fx_analyst_config.py`, `src/main.py`, `src/monitor/equity_monitor.py`, `src/scheduler/scheduler.py`
- New: `src/scheduler/news_event_trigger.py`
- Tests (Modified): `tests/scheduler/test_optimization_integration.py`, `tests/test_ab_model_switching.py`, `tests/test_agent_bridge_config.py`, `tests/test_scheduler.py`
- Tests (New): `tests/monitor/test_equity_monitor.py`, `tests/scheduler/test_news_event_trigger.py`
- Cross-repo impacts on TradingAgents: `tradingagents/agents/utils/agent_states.py`, `tradingagents/agents/utils/memory.py`, `tradingagents/agents/trader/trader.py`, `tradingagents/graph/propagation.py`, `tradingagents/graph/trading_graph.py`

---

## [1.3.9b] — 2026-03-11
**v1.3.9a Production Tuning — Scanner Position-Aware, Threshold Review, Memory Journal Diff, Config Aggressiveness**

### Fixed
- **P0**: Scanner generated intents for symbols with active positions — added `has_active_position_for_symbol()` check in `_scanner_loop` before intent creation, preventing wasted LLM evaluations and Duplicate Entry Guard noise
- **P1**: Blended confidence threshold too strict for cold-start/losing tiers — losing tier `min_confidence` downgraded from "high" to "medium", `min_blended_confidence` reduced across tiers (cold-start 0.50→0.48, losing 0.60→0.52, default 0.55→0.50), per-symbol adjustment narrowed ±0.05→±0.03

### Added
- `_compute_diff()` method in `src/monitor/memory_journal.py`: compares current decision with previous for same symbol, renders "Δ Changes vs Previous Decision" section in memory journal entries
- `_last_decisions` dict in `MemoryJournal.__init__`: tracks last decision per symbol for diff computation

### Changed
- `config/e8_one_5k_challenge.yaml`: `default_risk_pct` 0.007→0.01 ($35→$50/trade), `active_session_interval_seconds` 3600→1800, `quiet_session_interval_seconds` 14400→7200, `tactical.soft_gates.min_score` 2→1
- Compliance parameters (drawdown limits, best day rule) unchanged — safety margins preserved

### Tested
- **48 related tests passed** across 4 test files (thresholds, threshold_decay, config, prop_firm_guard_e8_one)
- Ruff lint + format: 0 warnings

### Files
- Modified: `src/scheduler/scheduler.py`, `src/optimize/thresholds.py`, `src/monitor/memory_journal.py`, `config/e8_one_5k_challenge.yaml`
- Tests (Modified): `tests/optimize/test_thresholds.py`, `tests/optimize/test_threshold_decay.py`, `tests/test_config.py`, `tests/test_prop_firm_guard_e8_one.py`

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
