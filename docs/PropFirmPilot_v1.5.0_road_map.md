# PropFirmPilot v1.5.0 之後 — `v1.5.0` 到 `v1.5.9` 單一入口路線圖

> **更新日期**: `2026-03-18`
>
> **文件角色**: `v1.5.0` 到 `v1.5.9` 的唯一入口 roadmap
>
> **適用範圍**: `prop-firm-pilot` 主線規劃，並直接納入 `qlib_market_scanner` / `qlib_rd_agent` / `TradingAgents` 的必要依賴
>
> **當前主線狀態**: `v1.5.0_preview` 已於 `2026-03-17` 在 `v1.5.0_beta_2` baseline 上落地 bounded capital utilization uplift preview；`2026-03-18` 已再吸收一輪 preview incident remediation（Best Day semantics、compliance admission、market-data routing、scanner success contract、Telegram fallback），並在 stable acceptance window 內完成 side-aware scanner live activation（`fx_signal_v2`、`topk_short = 1`、direction-aware gating）；`v1.5.0_stable` 仍待 open-book worst-case natural-SL drawdown guard、exposure / memory / validation acceptance closure
>
> **閱讀原則**: 若你只想知道 `1.5.0` 到 `1.5.9` 應做什麼，先看這份；不需要先回頭讀複數文檔

---

## 1. 這份文件要回答什麼

這份 roadmap 專門回答四件事：

1. `v1.5.0 stable` 之前，哪些工作其實已經做完，不應再重複規劃？
2. `v1.5.0 stable` 還缺哪些必要 closure，才能被保守地稱為第一個 stable milestone？
3. `1.5.1` 到 `1.5.9` 應如何分波段推進，而不是變成任意功能桶？
4. 哪些項目屬於 cross-repo 依賴，不能只在 `prop-firm-pilot` 單 repo 內宣稱完成？

這份文件取代的是 `1.5.x` 版本規劃入口，不取代：

- changelog 的發版記錄功能
- 盈利報告的審查與結論功能
- deployment manual 的操作手冊功能

---

## 2. 現況總結: `v1.5.0 stable` 已做了什麼、還沒做什麼

### 2.1 已完成，無需再規劃

以下能力已經是進入 `v1.5.0 stable` 前的既成基線：

| 類別 | 已完成內容 | 來源版本 / 狀態 |
|---|---|---|
| **Market data baseline** | `broker quote-first + API bars-first + websocket auxiliary` hybrid routing、warm-cache、degraded summary 已落地 | `v1.5.0_preview` remediation |
| **Freshness semantics** | effective close time freshness、stale tactical warning throttling 已落地 | `v1.4.9` |
| **Close control plane** | canonical close schema、`CloseControlPlane`、`CloseReconciler` 已落地 | `v1.4.8` |
| **Scanner contract gate** | `manifest/schema/version/validation` 檢查、required signal columns 驗證已落地 | `v1.5.0_beta` |
| **Scanner metadata persistence** | `scanner_version`、`scanner_schema_version`、`scanner_market_date`、`scanner_label_version` 已能落盤 | `v1.5.0_beta` |
| **Best Day semantics** | 新單 gate 已改為 actual `daily_pnl` only，不再把 hypothetical TP profit 當成當日 PnL | `v1.5.0_preview` remediation |
| **Compliance admission guard** | deterministic compliance headroom 已前移到 candidate / intent creation | `v1.5.0_preview` remediation |
| **Scanner success contract** | success 已收緊為 process + artifact + ingestion + `target_date` matched | `v1.5.0_preview` remediation |
| **Alert resilience baseline** | Telegram retry accounting、failure metrics、`ALERT_FALLBACK` secondary sink 已落地 | `v1.5.0_preview` remediation |
| **Side-aware scanner live ingestion** | `fx_signal_v2`、`scanner_side` persistence、direction-aware ranking / threshold / veto、legacy daily-cycle parity guard 已落地 | `v1.5.0_preview` acceptance window |
| **Version identity baseline** | `qlib_market_scanner` 的 scanner contract baseline 已凍結在 `1.5.0_beta` / `v1.5.0_beta`；`prop-firm-pilot` 主線則已 bump 到 `1.5.0_preview` / `v1.5.0_preview`，明確標記 bounded uplift preview lane | `v1.5.0_beta` → `v1.5.0_preview` |
| **FX scanner release cadence decision** | upstream 已完成第一輪 FX cadence research，canonical cadence 凍結為 `1d` | `qlib_market_scanner v1.5.0_beta` |
| **Runtime bundle family isolation** | runtime `outputs/*` 與 legacy `data/shared_export/*` 已分流，不再混讀 sidecars | `v1.5.0_beta` |

此外，以下 cross-repo implementation 已核對為落地，不應再在本 repo roadmap 中被當成未實作假設：

| Repo | 已核對落地的實作 | 目前意義 |
|---|---|---|
| `qlib_market_scanner` | `FX_TICKERS` 已是 7 pairs，`get_profile_selection_metric("fx") == "dsr_net_oos_daily_v1"` | FX scanner baseline 與 selection metric 已定型，不需重開 universe / metric 討論 |
| `qlib_market_scanner` | `PipelineConfig.label_version` 與 `research_cadences` 已落地 | downstream 可依賴穩定的 label / cadence metadata |
| `qlib_market_scanner` | `experiment_fx_alpha_matrix.py` 已有 cadence matrix runner 與 scorecard / decision artifacts 輸出 | preview / stable 不需再重做 upstream cadence research plumbing |
| `qlib_market_scanner` | `rdagent_factors.py` 已有 `candidate -> promoted -> report` factor gate | RD-Agent factor ingestion 已有明確 promotion boundary |
| `qlib_rd_agent` | `qlib_runner.py` 已會寫出 `discovered_factors.yaml`、`candidate_factors.yaml`、`factor_manifest.json` | factor artifact contract 已存在，不需在 pilot 端假設 upstream 尚未產生 |
| `qlib_rd_agent` | `dropbox_sync.py` 已支援 factors 三件套上傳與 `runs/<run_id>/...` run archive upload | research provenance / archive contract 已可被 roadmap 視為 upstream baseline |

### 2.2 `v1.5.0_beta_2` 已作為 stable 前修復閘合入主線

根據 `2026-03-16` 晚間到 `2026-03-17` 上午這輪 production run，`v1.5.0_beta_2` 已先完成 stable 前必須經過的 repair gate，並作為後續 `v1.5.0 stable` 驗證的 operational baseline。

`v1.5.0_beta_2` 的角色不是新增 feature，而是把這輪長時間運行暴露出的 P0 / P1 incident 收斂掉，避免系統在 market data、signal freshness、tactical state transition 與 operator observability 上繼續帶病進 stable；同時吸收必要的 scanner cadence / config ergonomics 調整，並修正啟動後首輪 scanner 因第一根 `5m` bar 尚未形成而容易白跑的問題，降低下一輪 live 驗證的操作摩擦。

| 嚴重度 | `v1.5.0_beta_2` 已納入的 repair / tuning |
|---|---|
| **P0** | market data freshness / websocket degradation guard：禁止把明顯 stale 的 REST bars 繼續送進 tactical / scanner live path |
| **P0** | stale signal hard block：新 UTC 日若找不到 target date signals，不得 fallback 後繼續當成可交易日訊號 |
| **P0** | tactical pending timeout / expired claim recycle closure：intent 必須 deterministic 地完成、取消或失敗，不可長時間 recycle 漂移 |
| **P1** | EquityMonitor transaction nesting 修復：避免長跑中出現 `cannot start a transaction within a transaction` |
| **P1** | Telegram tactical gate 限流 / 去重：降低 operator noise，不再出現高頻重複提示 |
| **P1** | startup first-run `5m` bar recovery：當 `quote fresh + websocket healthy` 但第一根 websocket `5m` closed bar 尚未形成時，不再直接 block scanner，而是建立 intent 並轉入 tactical retry |
| **P1** | incident diagnostics closure：`bars_5m_unavailable` 類 incident 現在會帶出 market-data hub 初始化時間、uptime 與 per-symbol websocket closed bar counts |
| **P2** | account config tuning / ergonomics：`e8_one_5k_challenge` 已把 off-hours scanner cadence 下調到 `3600s`，並按「高頻常調 / 低頻基礎」重排 YAML 結構與中文註解 |

### 2.3 `v1.5.0_preview` incident remediation 已成為 stable 前 correctness hardening baseline

根據 `prod_logs_20260318_v1.5.0_preview` 的 incident review，preview lane 雖已完成 bounded capital utilization uplift，但仍暴露出一組不能留到 stable 才處理的 correctness defects。這輪 remediation 已在本 repo 實作完成，stable gate 應視其為 baseline，而不是待選 enhancement。

| 嚴重度 | `v1.5.0_preview` remediation 已納入的 corrective actions |
|---|---|
| **P0** | `Best Day Rule` 新單 gate 改為 actual `daily_pnl` only，禁止 hypothetical TP profit 造成 no-trade false reject |
| **P0** | deterministic compliance headroom 前移到 candidate / intent creation，避免必定被拒的單子仍完整跑過 tactical / execution |
| **P0** | tactical hard-gate observability 已拆開 `spread`、`atr_regime`、`data_freshness`，並可對 USDCAD spread / ATR warmup 做後續 calibration review |
| **P1** | market-data routing 改為 `broker quote-first + API bars-first + websocket auxiliary`，websocket degraded 不再單獨 hard-block 系統 |
| **P1** | scanner success contract 改為 process + artifact + ingestion + `target_date` matched 才算 success |
| **P1** | Telegram send 補上 retry accounting、failure metrics 與 secondary sink；primary channel 失效時至少寫入 `ALERT_FALLBACK` journal event |

這輪 remediation 的關鍵 evidence 如下：

- websocket failures `15`
- REST fallback warnings `129`
- `market_data.quote_unavailable` log/journal 各 `2`
- `spread.fail.ratio_too_wide = 72`
- `atr.fail.insufficient_1h_data = 133`

### 2.4 屬於 `v1.5.0 stable` 必須完成

以下項目若未完成，`v1.5.0` 只能停留在 beta integration baseline，不能保守地稱為 stable：

| 主題 | `v1.5.0 stable` 必須完成的 closure |
|---|---|
| **Operational repair closure** | `v1.5.0_beta_2` 與 `2026-03-18` preview remediation 所定義的 P0 / P1 repairs 必須先收斂，stable 不接受帶病升版 |
| **Tactical entry integrity** | deterministic reason code、source provenance、score breakdown、state transition closure |
| **Tactical exit audit closure** | trigger source、broker read-back、journal consistency、postmortem replayability |
| **Bounded capital utilization uplift** | preview implementation 已在本 repo 落地；`v1.5.0 stable` 尚需 multi-day validation、integrated acceptance，並證明 uplift 不會讓 open-book worst-case natural-SL portfolio loss 穿透 `daily_drawdown_stop`；同時需與 preview remediation 的 regression-free validation、exposure / memory / validation closure 一起驗收 |
| **Portfolio / exposure guard v1** | base/quote currency exposure budget、同向集中暴露上限、低相關 setup budget，以及 portfolio-level risk reservation / aggregate open-risk guard |
| **Trade-memory v1** | raw event / reflection / retrieval 三層分工、schema freeze、quality gate |
| **Validation closure** | multi-day market-open validation、live-vs-research consistency review、P0/P1 tactical incident 收斂，以及 side-aware mixed long/short live path acceptance |
| **Stable acceptance gate** | 把 scanner contract、entry / exit control plane、memory、exposure guard 放到同一個 release gate 內驗收 |

### 2.5 明確延後到 `1.5.x`

以下項目重要，但不應把它們混進 `v1.5.0 stable` 的最小 closure：

| 主題 | 延後原因 |
|---|---|
| **`1h` / hybrid scanner promotion** | `1d` canonical cadence 才剛凍結，stable 版本先收斂 contract 與 validation，不應立刻翻轉 runtime 預設 |
| **TradingAgents intraday schema 大改** | 需要建立在 memory v1、entry / exit taxonomy、scanner metadata 穩定之後 |
| **LLM layer ablation** | 屬於 `1.5.x` 的 validation accumulation，不是 `v1.5.0` 的 minimum gate |
| **Memory ablation** | 需等 trade-memory contract 先凍結，再做可信比較 |
| **更完整的 portfolio construction / capital allocation** | `v1.5.0 stable` 只做 bounded capital utilization uplift；完整 session / setup / currency / correlation-aware capital engine 仍延後 |

### 2.6 明確不屬於 `1.5.x`

以下內容不應在 `1.5.x` 被當成默認承諾：

- 宣稱系統已被證明具備長期盈利能力
- 將 FX runtime scanner 預設直接切到 `1h` 或更高頻
- 多帳號 / 多 broker / dashboard 大型產品化擴張
- 在沒有長樣本 live evidence 前擴大資金規模主張

---

## 3. `v1.5.x` 的總原則

### 3.1 Validation-first，不是 feature-first

`v1.5.x` 的主軸不是堆更多功能，而是把：

- tactical correctness
- cross-repo contract
- memory quality
- exposure discipline
- research-to-live consistency

逐步收斂成可以被保守驗收的 stable system。

### 3.2 `1d` canonical cadence 在 `1.5.x` 內視為已凍結 baseline

`1.5.x` 可以做 follow-up research、metadata 擴充、bounded productization，但不應在沒有新 acceptance gate 的情況下把 runtime 預設從 `1d` 改為 `1h` 或 hybrid。

### 3.3 Cross-repo contract 先於新 alpha 敘事

只要 `prop-firm-pilot`、`qlib_market_scanner`、`TradingAgents` 的契約還不穩，就不應把 `1.5.x` 主敘事寫成「更強 alpha engine」。

### 3.4 版本節奏採波段分工，不做假精度承諾

這份 roadmap 仍然列出 `1.5.0` 到 `1.5.9`，但 `1.5.4+` 採建議波段分工，而不是假裝今天就能準確鎖死每一個小版本的所有細節。

---

## 4. 波段總覽

| 版本 | 角色 | 主軸 |
|---|---|---|
| `1.5.0_beta_2` | beta hardening repair gate | 已收斂 freshness、stale signals、tactical timeout、monitor / alert hardening，並補齊 market-data diagnostics 與 account config tuning |
| `1.5.0_preview` | bounded uplift + incident remediation preview lane | 先落地 bounded capital utilization uplift，再吸收 preview incident remediation，形成 stable 前的最新 correctness baseline |
| `1.5.0` | 第一個 stable acceptance gate | 把 preview uplift 與 preview remediation 一起納入 entry / exit / memory / exposure / validation 的整體驗收，形成第一個可保守稱為 stable 的 release |
| `1.5.1` | post-stable correctness sweep | 修正 first stable run 暴露的 correctness 與 taxonomy drift |
| `1.5.2` | validation accumulation | 強化 live-vs-research consistency 與 multi-day acceptance evidence |
| `1.5.3` | operator + risk hardening | 收斂 operator diagnostics、scheduler consistency、exposure guard v1 hardening |
| `1.5.4` | memory foundation | trade-memory schema freeze、raw / reflection / retrieval 分層 |
| `1.5.5` | memory quality gate | lesson quality gate、retrieval policy、memory observability |
| `1.5.6` | capital efficiency v1 | 在 stable 的 bounded uplift 之上，完成 session / setup / currency / correlation-aware allocation |
| `1.5.7` | bounded scanner follow-up | 不改 runtime default 的前提下，補 scanner horizon / timeframe metadata productization |
| `1.5.8` | agent alignment | TradingAgents intraday-aware contract alignment，仍以 bounded change 為主 |
| `1.5.9` | next-major preflight | 統整 `1.5.x` 證據，決定 `1.6.0` 的真正主題 |

---

## 5. `v1.5.0_beta_2` — Operational Hardening Repair Release

### 5.1 版本定位

`v1.5.0_beta_2` 是 `v1.5.0 stable` 前的修復閘，不是新 feature release。

它的任務是根據這輪 production run 暴露出的真實 incident，優先收斂 operational correctness：市場資料 freshness、signal freshness、tactical state transition、monitor reliability 與 operator noise。這一版也同步吸收了必要的 scanner cadence 調整、帳號 config 可維護性整理，以及 startup first-run `5m` bar recovery，讓下一輪 stable 驗證能在更低操作摩擦下進行。

### 5.2 按嚴重度排序的修復清單

| 嚴重度 | 修復主題 | 修復要求 | 驗收標準 |
|---|---|---|---|
| **P0** | **Market data freshness / websocket degradation guard** | 當 websocket 斷線或 handshake timeout 時，live path 不得繼續使用顯著 stale 的 REST 1m/5m/1h bars 參與 tactical 判斷 | 長時間斷線時系統會 deterministic 地 block / degrade，而不是靜默使用數小時前資料 |
| **P0** | **Stale signal hard block on new UTC day** | 若 target date signals 不存在，scanner 可記錄 incident，但不得 fallback 後直接建立可交易 intents | 新 UTC 日不存在 fresh signal 時，系統明確停在 non-tradable state |
| **P0** | **Tactical pending timeout / expired claim recycle closure** | 釐清 intent 在 BUY / SELL 後為何卡在 pending，讓 timeout 會導向 deterministic cancel / fail，而不是反覆 recycle | 不再出現長時間 `tactical retry aborted` 與 claim recycle 漂移 |
| **P1** | **EquityMonitor transaction nesting** | 修正 monitor 與持久化之間的 transaction 邊界 | 長跑期間不再出現 `cannot start a transaction within a transaction` |
| **P1** | **Telegram tactical gate throttling / dedupe** | tactical gate 類提示需有節流與去重 | 不再出現固定週期洗版式重複通知 |
| **P1** | **Startup first-run `5m` bar recovery** | 啟動後若第一輪 scanner 只缺第一根 websocket `5m` closed bar，不得直接因 `bars_5m_unavailable` 失效；需保留 candidate、建立 intent，並以 `market_data.startup_5m_bar_pending` 進入 tactical retry | 第一輪 scanner 不再白跑；既有 intent 會在下一根 `5m` close 後自動續跑，而不是硬等下一輪 scanner cadence |
| **P1** | **Operator diagnostics / incident bundle closure** | 將 freshness、signal date、intent state、close facts 聚合到可直接讀的 incident artifact，並補上 hub `initialized_at` / `uptime_seconds` / websocket closed bar counts | operator 不需手工拼多份 log 才知道一次 incident 的 root cause，也能直接判斷 `bars_5m_unavailable` 是本地聚合未 ready 還是 provider stale |
| **P2** | **Account config tuning / ergonomics** | 降低 off-hours scanner wait time，並重整 account config 可讀性 | 淡時段 scanner cadence 改為 `3600s`，`config/e8_one_5k_challenge.yaml` 可直接區分高頻與低頻調參區 |
| **P2** | **Startup scanner contract self-check hardening** | 補強 runtime bundle compatibility 提示，避免再出現 beta 啟動後才發現 manifest 不合 | contract 不合時能即時 fail fast 並留下明確 diagnostics |

### 5.3 明確不納入 `v1.5.0_beta_2`

- 不處理「資金使用率偏低」本身，這是優化不是 repair
- 不做完整 capital efficiency / portfolio optimizer
- 不做 scanner cadence promotion
- 不做大規模 TradingAgents schema / prompt 重構

### 5.4 完成條件

- 新 UTC 日不會再使用 stale signals 建立 live intents
- 顯著 stale 的 REST fallback bars 不會再進入 tactical live path
- 啟動後第一輪 scanner 不會再因第一根 `5m` bar 尚未形成而直接白跑；會建立 intent 並交由 tactical retry 等待第一根 websocket `5m` close
- tactical pending path 可 deterministic 收斂，不再靠 recycle 漂移
- EquityMonitor 可穩定跑過多小時 market-open 視窗
- Telegram tactical gate 通知頻率被節流到 operator 可接受水位
- `bars_5m_unavailable` 類 incident 可直接看出 market-data hub uptime 與 websocket closed bar accumulation 狀態
- `config/e8_one_5k_challenge.yaml` 已可按高頻調參 / 低頻基礎兩種操作場景直接維護，且淡時段 scanner cadence 降為 `3600s`

### 5.5 `v1.5.0_preview` — Bounded Capital Utilization Preview

`v1.5.0_preview` 建立在 `v1.5.0_beta_2` 的 operational repair baseline 上，任務是把 bounded capital utilization uplift 先做成可驗證的 preview implementation，而不是直接把整個 stable gate 宣稱完成。

這一版在本 repo 已落地：

- `BoundedCapitalAllocator` 依 `default_risk_pct * max_positions` 的名目預算、當前 `open_positions` 與 `scanner_confidence`，計算 `effective_risk_pct`
- `PositionSizer` 支援 risk override，execution path 會把 uplift 後的 `risk_pct` 傳進 sizing
- `ExecutionEngine` 會把 `risk_pct` 與 capital allocation metadata 寫入 execution meta，保留 audit trail
- live scanner path 已啟用 side-aware ingestion：`ScannerBridge` 支援 `fx_signal_v2`、`TradeIntent` / SQLite 持久化 `scanner_side`、scheduler 以 direction-aware quality 排序 mixed long/short bundles
- `TradingAgents` 在 side-aware path 中只允許 confirm / veto scanner direction；反方向 actionable decision 會被取消為 `direction_mismatch`
- legacy `run_daily_cycle()` 已與 scheduler 對齊 side-aware ranking 與 reverse-side hard veto，避免 non-scheduler path 留下舊邏輯缺口
- 專案 version identity 已切到 `1.5.0_preview` / `v1.5.0_preview`，包裝版本為 `1.5.0rc0`

這一版同時建立在已核對的 upstream 事實上：

- `qlib_market_scanner` 已穩定提供 FX `1d` canonical cadence、`dsr_net_oos_daily_v1` selection metric、`label_version` / `research_cadences` metadata，以及 cadence scorecard / decision artifacts
- `qlib_market_scanner` 已補上 `--topk-short` runtime activation，讓 preview live lane 可以顯式輸出 bounded short candidates
- `qlib_market_scanner` 已完成 RD-Agent `candidate -> promoted -> report` factor gate
- `qlib_rd_agent` 已完成 `discovered_factors.yaml`、`candidate_factors.yaml`、`factor_manifest.json` 與 `runs/<run_id>/...` archive upload contract

這一版仍不能被稱為 `v1.5.0 stable`，因為 stable 尚缺：

- tactical exit 不介入時，open-book worst-case natural-SL portfolio loss 仍可能穿透 `daily_drawdown_stop`；stable 必須補上 portfolio-level risk reservation / aggregate open-risk guard
- exposure guard v1 integrated acceptance
- trade-memory v1 與 quality gate
- multi-day market-open validation、side-aware mixed long/short acceptance 與 integrated stable acceptance gate

但 `2026-03-18` 的 preview incident 也已明確證明，stable gate 不能只驗收 uplift，還必須驗收以下 remediation 不回退：

- `Best Day Rule` actual-`daily_pnl` semantics 不回退到 projected-profit gating
- compliance headroom admission guard 仍在 candidate / intent creation 前生效
- market-data 仍維持 `broker quote-first + API bars-first + websocket auxiliary`
- scanner success 仍要求 `target_date` 真正落盤
- Telegram alerting 在 primary channel 失效時仍保有 retry accounting 與 journal fallback

---

## 6. `v1.5.0` — Stable Gate Closure

### 6.1 版本定位

`v1.5.0` 是第一個 stable milestone，不是新 feature 大包。

它的任務是把 `v1.5.0_beta` 已經吸收進主線的 scanner contract、version identity 與 `1d` cadence baseline，加上 `v1.5.0_preview` 已落地的 bounded capital utilization uplift 與 `2026-03-18` preview incident remediation，經過 `v1.5.0_beta_2` repair gate 後，升級成可以被保守驗收的 stable trading system gate。

### 6.2 本 repo 主責

- 完成 tactical entry production-grade closure
- 完成 tactical exit audit / read-back / postmortem closure
- 將 preview 已落地的 bounded capital utilization uplift 納入 stable acceptance，並證明 Best Day / compliance admission / market-data / scanner / alert remediation 沒有回退
- 上線 portfolio-level open-risk reservation / daily drawdown budget guard，保證 tactical exit 不介入時的 worst-case natural-SL 組合損失不穿透 `daily_drawdown_stop`
- 上線 exposure guard v1
- 上線 trade-memory v1 與 quality gate v1
- 完成 multi-day stable acceptance gate

### 6.3 Cross-repo 依賴

| Repo | `v1.5.0` 需要它完成什麼 |
|---|---|
| `qlib_market_scanner` | `1d` canonical cadence、`dsr_net_oos_daily_v1`、`label_version` / `research_cadences`、scorecard / decision artifacts、RD-Agent promoted factor gate 都已落地；stable 只要求這些 contract 繼續穩定且 live-compatible |
| `qlib_rd_agent` | `candidate/discovered/manifest` artifact 與 `runs/<run_id>/...` archive contract 已落地；stable 只要求 artifact freshness、lineage 與 scanner ingestion consistency 不漂移 |
| `TradingAgents` | 暫不要求大改，但其 risk output / lesson consumption 至少不能破壞 stable contract |

### 6.4 這版明確不做什麼

- 不把 runtime scanner 預設升到 `1h`
- 不做大規模 TradingAgents prompt / tool 架構重寫
- 不做完整 portfolio optimizer
- 不把資金使用率優化擴張成完整 capital engine 重寫
- 不宣稱已證明長期盈利

### 6.5 完成條件

- `v1.5.0_beta_2` 所定義的 P0 / P1 repair 已先收斂
- `2026-03-18` preview remediation 所定義的 Best Day、compliance admission、market-data routing、scanner success contract、Telegram fallback 修正已經過 multi-day regression-free validation
- entry verdict 有完整 deterministic reason taxonomy
- exit action 有完整 broker read-back 與 canonical close facts
- bounded capital utilization uplift 已完成 stable acceptance，且未削弱 safety-critical guard
- tactical exit 不介入的 worst-case natural-SL 情境下，open-book portfolio risk 仍不會突破 `daily_drawdown_stop`
- exposure guard v1 在 live path 可用
- trade-memory schema 與 quality gate 已落地
- 多日 market-open validation 無重複 P0/P1 tactical correctness incident

---

## 7. `v1.5.1` 到 `v1.5.3` — Stability / Validation Accumulation Wave

### 7.1 `v1.5.1` — Post-Stable Correctness Sweep

**定位**
第一個 stable 後修正版，只修 first stable run 暴露出的 correctness 與 schema drift。

**本 repo 主責**
- 修正 entry / exit reason taxonomy 漂移
- 修正 journal / alert / diagnostics 不一致
- 修正 acceptance gate 中發現的 false positive / false negative

**Cross-repo 依賴**
- `qlib_market_scanner`: 不改 contract，只接受 bugfix 級修正
- `TradingAgents`: 不改大 schema，只修與 stable contract 衝突的欄位或解析

**明確不做**
- 不做 scanner cadence promotion
- 不做 memory 架構升級

**完成條件**
- `v1.5.0` 首輪實盤暴露的 correctness incidents 全部被歸零或有明確緩解

### 7.2 `v1.5.2` — Validation Accumulation Release

**定位**
把 stable 後的 multi-day evidence 累積成正式 acceptance 資產。

**本 repo 主責**
- 強化 live-vs-research consistency instrumentation
- 強化 acceptance artifact 保存與摘要
- 對 tactical metrics、slippage、hold profile、rejection reasons 做更乾淨的對照輸出

**Cross-repo 依賴**
- `qlib_market_scanner`: 穩定輸出 research-side metadata，方便與 live trail 對比
- `TradingAgents`: 保持 decision schema 穩定，方便做 attribution review

**明確不做**
- 不引入新 alpha 敘事
- 不做 prompt 大改

**完成條件**
- 可以對同一段 sample 回答 research 假設與 live 事實是否一致

### 7.3 `v1.5.3` — Operator And Risk Hardening

**定位**
把 stable 之後仍偏手工排查的操作面與 exposure guard v1 做硬化。

**本 repo 主責**
- 強化 operator diagnostics、incident bundle、postmortem convenience
- 補 exposure guard v1 的邊界條件
- 強化 scheduler consistency 與 risk guard 可觀測性

**Cross-repo 依賴**
- `qlib_market_scanner`: 無新增主契約要求
- `TradingAgents`: 無新增主契約要求

**明確不做**
- 不做 memory layer 重構

**完成條件**
- operator 不再依賴跨多份 log 手工拼接才能判斷一次交易事故

---

## 8. `v1.5.4` 到 `v1.5.6` — Memory / Capital Efficiency Wave

### 8.1 `v1.5.4` — Trade-Memory Foundation

**定位**
建立 stable trade-memory contract，而不是只累積更多文字或 debug payload。

**本 repo 主責**
- 凍結 raw event / reflection / retrieval 三層 schema
- 明確切分 `TradeJournal`、memory journal、lesson memory 的角色
- 建立 close-to-memory pipeline 基礎時序

**Cross-repo 依賴**
- `TradingAgents`: 必須配合新的 lesson / retrieval contract
- `qlib_market_scanner`: 僅需維持 scanner metadata 穩定，供 memory indexing 使用

**明確不做**
- 不做大量 retrieval 策略實驗

**完成條件**
- 記憶相關欄位第一次形成 stable schema，而不是各模塊各自擴張

### 8.2 `v1.5.5` — Memory Quality Gate

**定位**
在 memory foundation 之上，阻止低品質事件直接進長期記憶。

**本 repo 主責**
- 為不完整 close、reason 不明、execution 不一致事件加上 quality gate
- 建立 lesson generation gate
- 建立 retrieval observability 與 quality metrics

**Cross-repo 依賴**
- `TradingAgents`: 根據新的 retrieval policy 消費記憶，而不是直接吃所有 lessons

**明確不做**
- 不宣稱 memory 已證明能提升 PnL

**完成條件**
- 記憶系統先證明自己不會持續累積垃圾，再談效果提升

### 8.3 `v1.5.6` — Capital Efficiency v1

**定位**
建立在 `v1.5.0 stable` 已完成的 bounded capital utilization uplift 之上，進一步把系統從單筆保守風控，提升到完整的 capital efficiency v1。

**本 repo 主責**
- 建立 session / setup / currency / correlation-aware budget v1
- 強化 sizing 與 exposure guard 之間的協調
- 讓多筆低相關 setup 可以被更有紀律地配置

**Cross-repo 依賴**
- `TradingAgents`: risk output 要能承載更清楚的 portfolio context
- `qlib_market_scanner`: 不需改 cadence，但 metadata 需穩定支援 setup grouping

**明確不做**
- 不做完整 portfolio optimizer
- 不做 broader capital expansion claim

**完成條件**
- 系統不只會避免過度曝險，也能開始有紀律地使用風險預算

---

## 9. `v1.5.7` 到 `v1.5.9` — Bounded Research Productization / Next-Major Prep Wave

### 9.1 `v1.5.7` — Bounded Scanner Follow-Up

**定位**
只在不改變 `1d` runtime default 的前提下，補 scanner follow-up research 的 product boundary。

**本 repo 主責**
- 擴充 scanner metadata 的下游承接能力，例如 horizon / timeframe / effective window
- 為未來 bounded `1h` / hybrid follow-up 做接口準備

**Cross-repo 依賴**
- `qlib_market_scanner`: 若要輸出新 metadata，必須以 bounded、backward-compatible 方式提供

**明確不做**
- 不把 `1h` 或 hybrid 直接切成 production default

**完成條件**
- downstream 已能理解更細的 scanner metadata，但 stable default 不變

### 9.2 `v1.5.8` — TradingAgents Alignment

**定位**
在 memory v1、capital efficiency v1、scanner metadata 更清楚之後，再做 bounded 的 agent alignment。

**本 repo 主責**
- 重新定義 agent state 與 pilot execution contract 的對接點
- 限縮 prompt / tool / memory 的上下文膨脹
- 強化 decision-to-action consistency

**Cross-repo 依賴**
- `TradingAgents`: 需配合 intraday-aware state schema 與 risk output contract
- `qlib_market_scanner`: 維持 metadata 穩定

**明確不做**
- 不在這版宣稱 LLM 已成為主要 alpha engine

**完成條件**
- agent alignment 變成 bounded、可驗證的 contract work，而不是任意 prompt 演化

### 9.3 `v1.5.9` — `1.6.0` Preflight

**定位**
整理整個 `1.5.x` 的證據與缺口，決定 `1.6.0` 的真正主題。

**本 repo 主責**
- 整理 `1.5.x` acceptance evidence
- 回顧哪些假設成立、哪些沒有成立
- 明確界定 `1.6.0` 是要走：
  - deeper alpha research
  - broader portfolio engine
  - stronger agent system
  - 或更強的 production operations

**Cross-repo 依賴**
- 三個 repo 都要提供可核對的 acceptance evidence，而不是口頭結論

**明確不做**
- 不在 evidence 不足時強行定義 `1.6.0` 大敘事

**完成條件**
- `1.6.0` 的方向是被證據推動，而不是被慣性推動

---

## 10. Cross-Repo 責任邊界

| 主題 | `prop-firm-pilot` 主責 | `qlib_market_scanner` 依賴 | `qlib_rd_agent` 依賴 | `TradingAgents` 依賴 |
|---|---|---|---|---|
| **Beta_2 repair gate** | freshness / stale signal / tactical timeout / monitor hardening | runtime bundle contract 與 signal-date metadata 需可被可靠驗證 | 無新增 mandatory change | 不可讓 decision state transition 破壞 deterministic failure handling |
| **Preview uplift** | bounded allocator / sizing override / execution audit metadata | 穩定輸出 scanner confidence / score metadata，讓 uplift 可被 bounded 使用 | candidate / discovered / manifest / archive contract 已存在即可，無新增 preview blocker | 不破壞既有 stable fields 與 decision schema |
| **Stable gate** | entry / exit / exposure / memory / validation closure，補上 open-book worst-case natural-SL drawdown guard，並完成 preview uplift acceptance | 穩定輸出 `1d` scanner contract、score metadata、promotion boundary | 維持 factor artifact / archive lineage 一致，避免 upstream provenance 漂移 | 不破壞既有 stable contract |
| **Memory** | schema、quality gate、pipeline | 提供穩定 scanner metadata | 提供可追溯的 factor artifact / archive lineage | lesson consumption / retrieval policy 配合 |
| **Capital efficiency** | exposure budget、allocation discipline | 提供可分群的穩定 metadata | 無新增 mandatory change，但 artifact lineage 要穩定可追溯 | risk output 要能承載 portfolio context |
| **Follow-up research productization** | bounded ingestion / metadata support | bounded metadata extension，不改 default cadence | 維持 candidate / archive contract backward-compatible | bounded context alignment |
| **Next-major planning** | 以 acceptance evidence 統整方向 | 提供 research evidence | 提供 research provenance evidence | 提供 agent effectiveness evidence |

---

## 11. `v1.5.0 stable` 的最終判定

### 11.1 可以說「已做了」的

- scanner contract freeze 已進主線
- `1d` canonical FX cadence 已凍結
- runtime bundle contract 已可被 pilot 可靠 ingest
- close control plane 已建立
- market data freshness semantics 已收斂
- `v1.5.0_beta_2` 的 repair scope 已被明確定義
- bounded capital utilization uplift preview 已在本 repo 落地
- side-aware scanner live activation 已在 preview acceptance window 落地
- `qlib_market_scanner` / `qlib_rd_agent` 的 upstream implementation 已核對為實作存在

### 11.2 還不能說「已完成」的

- tactical entry 已達 production-grade reliability
- tactical exit 已達 fully auditable stable closure
- bounded capital utilization uplift 已完成 stable acceptance 與 multi-day validation，且 open-book worst-case natural-SL drawdown guard 已閉環
- side-aware mixed long/short live path 已完成 multi-day stable acceptance 與 operator evidence 累積
- trade-memory contract 已穩定
- portfolio / exposure control 已達 stable v1
- integrated validation gate 已成熟

### 11.3 對 `1.5.x` 的正確期待

`1.5.x` 的角色不是大改世界觀，而是：

- 讓 `v1.5.0 stable` 真正站穩
- 先經過 `v1.5.0_beta_2` repair，再把 `v1.5.0_preview` 的 bounded uplift 與 side-aware scanner live path 一起納入 stable acceptance，之後補 memory 與 bounded research productization
- 累積足夠證據，決定 `1.6.0` 要往哪裡走

---

## 12. 一句話版結論

如果只用一句話描述這份 roadmap：

> `v1.5.0_beta_2` 先做 operational repair；`v1.5.0_preview` 先落地 bounded capital utilization uplift 與 side-aware scanner live activation；`v1.5.0` 再完成第一個 stable acceptance gate；之後的 `1.5.1-1.5.9` 才沿著 correctness、validation、memory、capital efficiency 與 bounded follow-up 繼續推進。`
