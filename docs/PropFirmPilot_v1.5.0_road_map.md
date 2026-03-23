# PropFirmPilot `v1.5.0_stable` 路線圖

> **更新日期**: `2026-03-23`
>
> **文件角色**: `v1.5.0_stable` 主線入口文件
>
> **適用範圍**: `prop-firm-pilot` 主線實作，以及它與 `qlib_market_scanner`、`TradingAgents`、`qlib_rd_agent` 的必要上下游邊界
>
> **當前主線狀態**: `main` 現已對齊 `v1.5.0_stable` release identity，並具備 stable implementation baseline，包括 broker-neutral runtime startup、`TradeLocker-first` backend path、`TradingAgents` 預設關閉的 deterministic entry path、side-aware scanner contract、market-data continuity fallback、tactical entry/exit、以及 canonical close reconciliation；後續工作重點已從版本 bump 轉為 stable acceptance 累積、portfolio / memory closure 與 operator evidence 收斂
>
> **閱讀原則**: 若你只想知道目前 stable 主線已做了什麼、還差什麼，先看這份；若你要看一筆交易怎樣開始與結束，再看 `docs/PropFirmPilot_v1.5.0_stable_Report.md`

---

## 1. 這份文件要回答什麼

這份 roadmap 專門回答四件事：

1. 目前 `main` 上，哪些 `v1.5.0_stable` 能力已經是既成基線？
2. 哪些能力現在可以保守地說「stable implementation 已落地」？
3. 在正式切到 `v1.5.0_stable` release 之前，還差哪些 closure？
4. stable 之後的優先開發方向應如何排序？

這份文件不再把 `preview` / `preview_2` 當成主敘事，只在必要地方保留它們作為 stable baseline 的來源背景。

---

## 2. 當前 stable implementation baseline

### 2.1 主線已落地的基線

| 類別 | 當前 stable implementation baseline | 目前意義 |
|---|---|---|
| **Runtime startup** | `main.py` 的 daily cycle、monitor-only、scheduler 三個入口都已經統一走 broker factory | runtime 不再在入口層硬綁 `MatchTrader` |
| **Broker backend** | `execution.broker_backend` 可選 `matchtrader` 或 `tradelocker` | `TradeLocker-first` 已是 `E8 Signature` 目標路徑，`MatchTrader` 仍保留兼容 |
| **TradingAgents runtime policy** | `TRADINGAGENTS_ENABLED` 未設時，`agents.enabled = false`，`scanner_llm_tactical` 會自動降級為 `scanner_tactical`，tactical exit 的 LLM exception path 也會關閉 | stable 預設不依賴 LLM 才能開倉與管倉 |
| **Scanner contract** | `ScannerBridge` 已驗證 `manifest.json` / `metrics.json` / `signals.csv` 的 version、schema、validation status、required columns 與 `target_date` | downstream ingestion 已是 contract-first，而不是讀到 `signals.csv` 就直接交易 |
| **Scanner output semantics** | side-aware `fx_signal_v2`、`scanner_side` persistence、direction-aware candidate ranking 已落地 | scanner 已能穩定輸出 long / short 候選，不再靠隱含正負值推導 |
| **Universe baseline** | first-batch 10-pair baseline 已落地：`EURUSD`、`GBPUSD`、`USDJPY`、`AUDUSD`、`NZDUSD`、`USDCAD`、`USDCHF`、`EURJPY`、`AUDJPY`、`CADJPY` | stable 驗收基線不再是最早的 7-pair universe |
| **Market data routing** | live path 已採 `broker quote-first + API bars-first + websocket auxiliary` | websocket 退化不再單獨 hard-block 系統 |
| **Market data continuity fallback** | websocket 退化時，quote continuity 會先走 `EODHD real-time REST -> intraday REST -> fail-closed`，並回餵 aggregator 持續關閉 bars | stable 已具備 `1m/5m/1h` continuity repair，而不是只靠 websocket |
| **Tactical entry** | hard gate / soft gate / retry / timeout / degraded semantics 已落地，stale intraday bars 會 fail-closed | 系統不會靜默使用明顯過時的 `5m/1h` bars 進場 |
| **Execution** | execution path 已具備 Best Day hard gate、compliance gate、bounded capital allocation、dynamic JPY pip-value sizing、pre-trade slippage check、execution meta persistence | 下單路徑已從單純「送單」收斂為風控 + audit path |
| **Close lifecycle** | `CloseControlPlane`、`CloseReconciler`、canonical close facts 與 resolution path 已落地 | modify / partial / full close 與 post-close attribution 已統一 |
| **Monitoring / ops** | equity monitor、position monitor、janitor、daily summary、volatility monitor、Telegram fallback、trade journal 均已在 scheduler mode 接線 | long-running runtime 不再只靠單一 loop + ad hoc log |

### 2.2 `TradingAgents` 在 stable 主線的正確定位

`TradingAgents` 仍被保留在主線能力範圍內，但它在目前 stable implementation 的定位已經改變：

1. **預設是關閉的**
   - `load_config()` 在 env override 階段會把 `TRADINGAGENTS_ENABLED` 視為最終開關。
   - 若未設或明確關閉，`agents.enabled = false`。

2. **stable 預設 entry funnel 是 deterministic path**
   - `entry_funnel_mode` 若原本配置為 `scanner_llm_tactical`，會自動降級為 `scanner_tactical`。
   - 實際預設開倉路徑變成 `scanner -> tactical -> execution`。

3. **LLM path 改成顯式 opt-in**
   - 只有在 `TRADINGAGENTS_ENABLED=true` 時，才會恢復 `scanner -> TradingAgents -> tactical -> execution`。
   - 即使開啟 LLM，agent 在 side-aware path 也只允許 confirm / veto scanner direction，不允許反手改方向。

4. **這是安全預設，不是功能缺失**
   - stable implementation 已保留 agent-enabled path。
   - 但 stable 主線不再假設沒有 LLM 就不能交易。

### 2.3 Cross-repo 邊界已經收斂

| Repo | 在 stable baseline 中扮演的角色 | pilot 端可依賴的事實 |
|---|---|---|
| `qlib_market_scanner` | 上游 research / selection / signal export 系統 | `1d` canonical cadence、versioned bundle、side-aware export、10-pair first batch、metadata contract |
| `TradingAgents` | 可選 strategic confirm / veto 層 | 僅在顯式開啟時介入 entry 與 re-evaluation；不是 stable 預設依賴 |
| `qlib_rd_agent` | factor research provenance 與 archive contract | `candidate/discovered/manifest` artifact 與 run archive contract 已存在，但不直接負責 runtime execution |

換句話說，`prop-firm-pilot` 在 stable baseline 的主責，已經明確收斂成：

- 吃 contract-validated scanner outputs
- 做 live admission、tactical gating、execution、position management、close reconciliation
- 在 broker / market-data / compliance / monitoring 之間維持 deterministic trading runtime

---

## 3. 目前可以保守宣稱「已完成」的 stable 能力

### 3.1 Entry 與 admission

以下能力已可視為 stable implementation baseline，而不是仍待設計：

- `ScannerBridge` 已將 scanner ingestion 收斂為 contract-first path：沒有正確 manifest、schema、validation status 或 target-date 的 bundle，不會被 live ingestion 接受。
- scheduler scanner loop 已具備 candidate dedup、top-k admission、capacity control、active-position skip、recent-rejection cooldown、low-confidence cooldown 與 circuit breaker。
- deterministic compliance headroom 已前移到 candidate / intent admission；明知會被拒的 setup 不再白跑後續 pipeline。
- side-aware scanner ingestion 已與 runtime direction guard 對齊；default path 不會把 scanner long 變成 live SELL，或把 scanner short 變成 live BUY。

### 3.2 Tactical correctness

- `TacticalValidator` 的 hard / soft gates、summary reason code 與 `RETRY_PENDING / SKIP_CANCEL / EXPIRE_TIMEOUT / EXECUTE_DEGRADED` resolution 已落地。
- stale / previous-day intraday bars 會在 admission 或 tactical path fail-closed。
- startup 首輪 `5m` bar 尚未形成時，系統會走明確的 startup retryable path，而不是直接白跑或靜默退化。

### 3.3 Execution 與 broker-neutral runtime

- 這條主線對應先前 stable rollout 中的 `Task 5`: broker-neutral runtime startup 與 `TradeLocker-first` backend 接線。
- broker-neutral client factory 已接上 runtime startup，不再只有單一 broker path。
- `TradeLocker` backend 已進入可實際啟動的 runtime path，是 `E8 Signature` 的主目標路徑。
- `MatchTrader` backend 仍保留，作為現有 account 與兼容回退路徑。
- `ExecutionEngine` 已將 account snapshot、Best Day hard gate、compliance gate、sizing、slippage protection、SL/TP write-back、execution meta persistence 串成完整 pipeline。
- JPY quote pairs 的 live pip-value sizing 已不再只依賴靜態 YAML 值。

### 3.4 Position management 與 close lifecycle

- `Position monitor` 已可檢測 position disappear、拉 broker closed positions、套 execution-meta fallback、再做 canonical close reconciliation。
- `CloseControlPlane` 已統一路由 modify-only、partial close、full close。
- `CloseReconciler` 已可把 broker facts、execution meta、best-day/reeval fallback、last-known profit 合併成 canonical close reason 與 resolution path。
- tactical exit manager 已能做 breakeven、trail、TP reprice、partial close、defensive exit。

### 3.5 Monitoring / operator path

- `AlertService` 已有 retry accounting、failure metrics、circuit breaker 與 `ALERT_FALLBACK` secondary sink。
- `TradeJournal` 已提供 append-only lifecycle log，能承接 `TRADE_*`、`SCANNER_*`、`TACTICAL_*`、`ALERT_FALLBACK` 等事件。
- scheduler mode 的 scanner / execution / janitor / equity monitor / position monitor / daily summary / volatility monitor loops 已形成長跑 runtime 骨架。

---

## 4. 仍未完成的 `v1.5.0_stable` release closure

目前的狀態是 **stable implementation baseline 已經存在**，但 **stable release closure 還沒有全部完成**。正式切到 `v1.5.0_stable` 前，至少還差以下幾項：

| 工作流 | 尚未 closure 的原因 | 需要的交付 |
|---|---|---|
| **Open-book worst-case risk guard** | 系統已有單筆風控與 tactical exit，但尚未完全證明 tactical 不介入時，open-book natural-SL 組合風險不會穿透 `daily_drawdown_stop` | portfolio-level open-risk reservation / aggregate worst-case guard |
| **Exposure / portfolio budget v1** | 同向、同貨幣、相關性暴露仍未形成完整 stable contract | base/quote exposure budget、setup grouping、portfolio-level admission guard |
| **Trade-memory v1** | 目前已有 journal / reflection / execution meta，但尚未凍結成 stable memory schema 與 quality gate | raw event / reflection / retrieval 分層與 quality gate |
| **Multi-day acceptance evidence** | implementation 已落地，但還缺夠乾淨的 multi-day market-open validation 與 operator evidence | acceptance bundle、run summary、incident-free evidence、runbook closure |
| **Agent-enabled path acceptance** | LLM path 已保留，但 stable 預設已切 default-off；若未來要把 agent path 納入 stable release 敘事，還需單獨驗收 | confirm / veto path、re-evaluation path、memory interaction 的 bounded acceptance |

### 4.1 這裡最重要的判斷

`v1.5.0_stable` 目前最大的剩餘工作，不是「再加新 feature」，而是：

- 把已經落地的 implementation 做完整 acceptance
- 把尚未 formalized 的 portfolio / memory / evidence closure 補齊
- 在 release identity 已切到 stable 之後，繼續補齊 acceptance 與 portfolio / memory closure

---

## 5. Stable 之後的開發優先方向

stable release closure 做完之後，`1.5.x` 的後續優先級應維持精簡，不再回到 preview 時代那種過長 patch 敘事。

### 5.1 Priority 1: Stable acceptance 後清理

- 收斂 stable 首輪 live run 暴露的 correctness drift
- 清理 operator diagnostics、alert taxonomy、postmortem convenience
- 完成正式 stable release notes / runbook closure

### 5.2 Priority 2: Portfolio / exposure discipline

- 建立 exposure budget v1
- 補 portfolio-level admission / sizing coordination
- 將 bounded capital allocation 從單筆 uplift 擴展到組合層紀律

### 5.3 Priority 3: Trade-memory v1

- 凍結 stable memory schema
- 建立 lesson quality gate 與 retrieval policy
- 區分 trade journal、reflection、長期記憶的責任邊界

### 5.4 Priority 4: Agent-enabled bounded path

- 在 default-off 基線上，重新定義 agent-enabled path 的 acceptance boundary
- 保持 confirm / veto / exception path 的 bounded scope
- 避免再次讓 stable 主線依賴 unbounded LLM behavior

### 5.5 Priority 5: Bounded scanner follow-up

- 維持 `1d` canonical cadence 為 release baseline
- 只在 evidence 足夠時擴 scanner metadata / follow-up productization
- 不在 `1.5.x` 內重新打開大規模 cadence default promotion

---

## 6. `v1.5.0_stable` 的正確結論

### 6.1 可以說「已經有了」的

- stable 需要的主要 runtime 架構已經在 `main` 上存在
- broker-neutral startup、TradeLocker-first path、TradingAgents default-off policy 已落地
- scanner ingestion、market-data routing、tactical entry、execution、close reconciliation、operator monitoring 已形成閉環 implementation baseline

### 6.2 還不能說「正式完成」的

- 版本 identity 已切到 `1.5.0_stable`，但 stable acceptance 證據仍需持續累積
- portfolio-level worst-case open-book risk closure 尚未完成
- exposure / trade-memory / acceptance evidence 尚未全部 formalized

### 6.3 一句話版結論

> `prop-firm-pilot` 的 `main` 現已切到 `v1.5.0_stable` release identity，並具備 stable implementation baseline；接下來的重點不再是版本 bump，而是把 portfolio / memory / acceptance closure 補成真正完整的 stable release evidence。 
