# PropFirmPilot `v1.5.0_stable` 系統報告

> **更新日期**: `2026-03-23`
>
> **文件角色**: `v1.5.0_stable` implementation baseline 的系統解說文件
>
> **適用範圍**: 目前 `main` 上的 stable runtime 實作，包括 scanner ingestion、scheduler、tactical entry/exit、execution、monitoring、close reconciliation 與 broker-neutral backend
>
> **狀態說明**: 本文件描述的是當前 `main` 已採用的 `v1.5.0_stable` release identity 與 stable runtime implementation baseline；目前重點已從版本 bump 轉為 stable acceptance 證據、portfolio closure 與 operator evidence 累積

---

## 1. 這份文件要回答什麼

這份 stable report 主要回答兩件事：

1. `prop-firm-pilot` 目前有哪些核心功能，它們各自在做什麼？
2. 一筆交易如何從 scanner 開始，經過 scheduler、tactical、execution、position monitor，最後被平倉、記錄與總結？

這份文件不是 changelog，也不是 operator runbook。它的角色是把目前 stable implementation 變成一份可連續閱讀的系統說明。

---

## 2. 一頁式系統總覽

### 2.1 預設 stable 路徑

目前 stable 預設的完整交易鏈如下：

```text
qlib market scanner
-> ScannerBridge contract validation
-> Scheduler scanner loop
-> TradeIntent admission
-> Tactical validation
-> ExecutionEngine
-> Broker position
-> Position monitor / tactical exit
-> Close control + close reconciliation
-> Trade journal / alerts / daily summary
```

### 2.2 可選的 agent 路徑

若顯式開啟 `TRADINGAGENTS_ENABLED=true`，entry path 會變成：

```text
qlib market scanner
-> ScannerBridge contract validation
-> Scheduler scanner loop
-> TradeIntent admission
-> TradingAgents confirm / veto
-> Tactical validation
-> ExecutionEngine
-> Broker position
-> Position monitor / close reconciliation
```

### 2.3 為什麼 stable 預設不走 LLM

目前 stable 主線刻意把 `TradingAgents` 設為 default-off：

- `TRADINGAGENTS_ENABLED` 未設時，`agents.enabled = false`
- `entry_funnel_mode` 會自動從 `scanner_llm_tactical` 降級成 `scanner_tactical`
- position re-evaluation 不啟動
- tactical exit 的 LLM exception path 關閉

因此，stable 預設實際上是一條 **deterministic 的 scanner-driven pipeline**：

```text
scanner -> tactical -> execution
```

這不是把 LLM 功能刪掉，而是把它改成顯式 opt-in。

---

## 3. 啟動與系統基線

### 3.1 系統啟動時先做什麼

runtime 啟動時，系統會先建立以下幾層基線：

1. **載入 YAML + env override**
   - `load_config()` 會先做 YAML deep merge。
   - 然後在 env override 階段套用 `TRADINGAGENTS_ENABLED`。
   - 這一步會決定 stable 預設是否啟用 LLM path。

2. **選定 broker backend**
   - `execution.broker_backend = "matchtrader"` 或 `"tradelocker"`。
   - broker client 一律透過 broker factory 建立。
   - 這讓 daily cycle、monitor-only、scheduler 三條入口都共用同一個 backend 邊界。

3. **登入 broker、建立 instrument registry**
   - runtime 啟動後先登入 broker。
   - 再用 live broker effective instruments 建立 symbol mapping。
   - 這讓 config symbols 與 broker symbols 可以雙向對應。

4. **接上 market data、journal、alerts、close control**
   - market-data path 已固定為 quote-first + bars-first + websocket auxiliary。
   - `TradeJournal` 接收 lifecycle events。
   - `AlertService` 負責 Telegram 及 fallback sink。
   - `CloseControlPlane` / `CloseReconciler` 負責 close-domain execution 與 attribution。

### 3.2 Scheduler mode 的主要 loop

在 scheduler mode，系統不是單次跑完就結束，而是多個 loop 長時間並行運作。

| Loop | 作用 |
|---|---|
| **Scanner loop** | 週期性跑 scanner，建立新 intents |
| **LLM worker / decision worker** | 處理 `claimed` intents，做 scanner-side routing 或 LLM confirm/veto |
| **Execution loop** | 把 `ready_for_exec` intents 送進 broker |
| **Janitor loop** | 回收 stale claims、清理舊 intents |
| **Equity monitor loop** | 監控 drawdown / equity 風險 |
| **Position monitor loop** | 對 open positions 做 tactical exit、best-day close、close detection |
| **Volatility monitor loop** | 提供波動度觀測與掃描節奏支援 |
| **Daily summary loop** | 產出日級 summary / ablation / operator digest |

這個多 loop 結構，是 stable runtime 能長跑的基礎。

---

## 4. 一筆新交易怎樣開始

### 4.1 上游 scanner 先產生 signals

交易的起點不是 broker，也不是 LLM，而是 `qlib_market_scanner`。

scanner 在這條鏈中的角色很明確：

- 它是上游 research / selection / signal export 系統
- 它負責輸出 versioned bundle，而不是直接負責風控或執行
- pilot 端只會消費 contract-valid 的 scanner outputs

目前 stable path 依賴的 scanner contract 包括：

- `signals.csv`
- `manifest.json`
- `metrics.json`
- `schema_version`
- `scanner_version`
- `label_version`
- `market_date`
- side-aware `fx_signal_v2` fields

### 4.2 `ScannerBridge` 如何驗證 scanner bundle

`ScannerBridge.run_pipeline()` 不只是在 shell 裡跑一次 scanner CLI，而是會檢查整個 ingest contract：

1. scanner subprocess 是否成功
2. signals artifact 是否存在
3. bundle ingestion 是否成功
4. `target_date` 是否真的對上請求日期

因此，對 live runtime 來說，成功的定義不是「CLI exit code = 0」，而是：

```text
process_success
+ artifact_available
+ ingestion_success
+ target_date_matched
```

若 bundle 缺 target date、validation status 是 `stale/degraded`、或 schema 不正確，live path 會 fail-closed。

### 4.3 Scheduler scanner loop 怎樣把 signals 變成 candidates

scanner loop 讀到 signals 之後，不會直接一股腦建立 intents，而是先做幾層收斂：

1. **同一 symbol / side 去重**
   - side-aware bundle 可能包含多個 rows。
   - scheduler 會先根據 signal quality 選出最佳 candidate。

2. **排序**
   - 以 `score`、`confidence`、side-aware quality 做排序。
   - 再套 `topk` / `topk_short` 配置。

3. **capacity 控制**
   - 會檢查目前 open positions + pipeline intents 是否已達上限。
   - 若已滿，scanner loop 不會再建立新 intents。

### 4.4 哪些情況會在 intent 建立前被擋掉

在 stable runtime，一筆單通常會先死在 admission，而不是等到 execution 才發現不該下。

常見的 early block 包括：

- 已有 active position
- 已有 in-progress intent
- compliance admission headroom 不足
- recent rejection cooldown
- low-confidence cooldown
- daily SL hit circuit breaker
- same-symbol loss lock
- market-data entry guard

這些 block 的目標不是「更嚴格」，而是 **避免白跑整條 pipeline**。

### 4.5 通過 admission 之後會發生什麼

若一個 candidate 通過前述 admission checks，scheduler 會建立 `TradeIntent`。

`TradeIntent` 至少會持久化以下資訊：

- `symbol`
- `trade_date`
- `scanner_score`
- `scanner_confidence`
- `scanner_side`
- `scanner_version`
- `scanner_schema_version`
- `scanner_market_date`
- `scanner_label_version`
- `expires_at`

從這一刻起，這筆交易就正式進入 scheduler 的 state machine。

---

## 5. 決策層怎樣工作

### 5.1 stable 預設模式是 `scanner_tactical`

這是目前最重要的 stable 行為變更之一。

當 `TradingAgents` 關閉時，decision layer 不再呼叫 LLM，而是直接用 scanner side 做決策路由：

- scanner `long` -> `BUY`
- scanner `short` -> `SELL`

也就是說，stable 預設不是：

```text
scanner -> LLM -> tactical
```

而是：

```text
scanner -> direct side routing -> tactical
```

### 5.2 如果開啟 `TradingAgents`，它做的是什麼

若顯式設 `TRADINGAGENTS_ENABLED=true`，系統會恢復 `scanner_llm_tactical` path。

但即使如此，LLM 在 stable path 中也不是無限制自由決策，而是有明確邊界：

- agent 只能 **confirm** scanner direction
- 或 **veto** 變成 `HOLD`
- 若 agent 給出反方向 actionable decision，會被當成 `direction_mismatch` 取消

因此，LLM 的角色是 **bounded confirm / veto layer**，不是主導整個 execution side。

### 5.3 哪些情況會讓 intent 在決策層被取消

常見取消情況包括：

- `no_trade` mode
- `tactical_only` mode 但缺戰術訊號來源
- `scanner_tactical` 下沒有有效 `scanner_side`
- LLM pre-filter 低信心
- LLM post-filter 低信心
- direction mismatch
- Best Day protection active
- duplicate active position
- same-direction daily attempt limit
- LLM 決定 `HOLD`

換句話說，decision layer 不是只會給 `BUY/SELL`，它也在負責 **消滅不值得往下跑的 intents**。

### 5.4 這層的主要目的

這一層的目的不是追求複雜，而是把 strategic side 與 live execution side 接起來，並在 tactical 之前先做一次 bounded routing / veto。

---

## 6. Tactical gate 怎樣決定可不可以進場

### 6.1 tactical 看的不是 scanner alpha，而是當下能不能安全進場

tactical layer 的角色，是把 strategic intent 轉成 **entry-timing decision**。

它處理的問題是：

- 現在 spread 是否過寬？
- `5m/1h` bars 是否足夠新鮮？
- 當下是不是死市場或極端波動？
- 當前短線 momentum / candle quality 是否支持這筆 entry？

### 6.2 stable market-data 路由怎樣支撐 tactical

目前 stable market-data path 的關鍵語義是：

- **quote**: broker quote-first
- **bars**: API bars-first
- **websocket**: auxiliary，不是唯一真相來源

若 websocket 退化，系統會做：

```text
EODHD real-time REST
-> intraday REST
-> fail-closed
```

這樣做的目的是：

- 儘量維持 quote / bar continuity
- 但不在資料明顯 stale 時硬做 entry

### 6.3 hard gates 在看什麼

hard gates 必須全部過關，常見項目包括：

- spread
- ATR regime
- data freshness
- `1h` data sufficiency
- `5m` tactical inputs availability

若 hard gate 失敗，通常會直接 `SKIP_CANCEL`、`RETRY_PENDING` 或 `EXPIRE_TIMEOUT`。

### 6.4 soft gates 在看什麼

soft gates 是加權式 entry quality check，常見項目包括：

- EMA alignment
- RSI state
- candle body quality

soft gates 不一定要求全過，但至少要達到最低分數。

### 6.5 tactical 的幾種輸出語義

| Resolution | 意義 |
|---|---|
| **PASS / ready** | 可以往 execution 走 |
| **RETRY_PENDING** | 不是直接否決，而是等待下一輪 tactical retry |
| **SKIP_CANCEL** | 條件明確不合，不再嘗試 |
| **EXPIRE_TIMEOUT** | tactical retry 預算已耗盡 |
| **EXECUTE_DEGRADED** | 在嚴格條件下退化允許執行，但會留下明確 attribution |

stable runtime 的關鍵點是：**過時資料會 fail-closed，而不是靜默當成可交易。**

---

## 7. 交易怎樣真正送到 broker

### 7.1 `ExecutionEngine` 的工作不是單純送單

一筆 intent 進到 `ready_for_exec` 後，會由 `ExecutionEngine` 接手。

它的主要步驟是：

1. mark intent 為 `executing`
2. 建立 account snapshot
3. 做 execution-side Best Day hard gate
4. 建立 trade plan
5. 跑 compliance checks
6. 拉 pre-trade quote 做 slippage protection
7. 透過 broker client 開倉
8. 寫入 SL/TP
9. 持久化 execution meta
10. 發送 alert / trade event

### 7.2 compliance 在這裡做什麼

`PropFirmGuard` 是 safety-critical 核心。

目前它至少負責：

- daily drawdown 檢查
- max drawdown 檢查
- Best Day Rule 檢查
- API quota / position-related protections
- admission headroom 判斷

而 execution path 上還有一個額外重點：

- 就算 scheduler admission 已過，execution side 仍會再做一次 Best Day hard gate

這是 race-condition guard，不讓「進場前可下、到送單瞬間不該下」的情況漏掉。

### 7.3 sizing 現在怎樣做

stable execution path 不再只用單一固定 lot sizing，而是把幾個因素接起來：

- account equity
- open positions
- bounded capital allocation
- live pip value
- instrument constraints

特別是 JPY pairs：

- `USDJPY`
- `EURJPY`
- `AUDJPY`
- `CADJPY`

現在會以 live `USDJPY` quote 解 USD pip value，不再只依賴靜態 YAML。

### 7.4 開倉成功後系統會留下什麼

若 broker 開倉成功，系統會持久化：

- `position_id`
- `volume`
- `fill_price`
- `sl_price`
- `tp_price`
- `risk_pct`
- capital allocation metadata
- compliance snapshot
- execution meta

這些資料之後會被 position monitor、tactical exit、close reconciliation 重複使用。

---

## 8. 持倉期間系統怎樣管理風險

### 8.1 `Equity monitor` 在做什麼

`Equity monitor` 會持續輪詢 equity / balance 狀態，監控 drawdown 與風險水位。

它的角色不是做單筆 entry timing，而是看整體帳戶是否接近 prop-firm 規則的風險邊界。

### 8.2 `Position monitor` 是持倉期間的主控制器

一旦 position 開出來，接下來最重要的 loop 是 `position monitor`。

它負責：

- 對 open positions 做 tactical exit evaluation
- 更新 BestDayTracker 的 unrealized pnl
- 追蹤 last-known profit
- 偵測某個 `position_id` 是否已從 broker open positions 消失
- 觸發 close handling

### 8.3 tactical exit 現在能做哪些事

持倉期間，tactical exit manager 不只是看停利停損是否被 hit，它還可以主動執行：

- `MOVE_TO_BREAKEVEN`
- `TRAIL_SL`
- `REPRICE_TP`
- `PARTIAL_CLOSE`
- `EXIT_NOW`

這些動作都會透過 `CloseControlPlane` 走同一條 broker-facing path，而不是各模組私自寫 broker API。

### 8.4 Best Day 接近上限時會怎樣

若系統偵測到 Best Day Rule 接近安全上限，`position monitor` 會主動關閉獲利中的 positions，避免日內已實現 / 未實現利潤把帳戶推向違規。

這個 path 最終也會被 close reconciliation 吃進 canonical close facts。

### 8.5 position re-evaluation 什麼時候會啟動

position re-evaluation 是 agent-enabled 的例外路徑，不是 stable 預設。

它只有在以下條件下才會啟動：

- `TradingAgents` 已啟用
- 不是 mock agents
- 持倉時間已超過最低 reevaluation hold time
- 到達 reevaluation interval

而且它的 bounded 目標也很明確：

- 只在 agent 給出 **反向 actionable signal** 時才會觸發 close
- `HOLD` 表示保持不動
- 同向訊號表示確認持倉

---

## 9. 一筆交易怎樣結束

### 9.1 最常見的 close 起點

對系統來說，一筆交易的結束通常不是因為它主動知道「TP hit 了」，而是因為：

- `position monitor` 發現某個 `position_id` 已不在 broker open positions 裡

這表示這筆單可能已經：

- hit TP
- hit SL
- 被 tactical exit 關閉
- 被 best-day close 關閉
- 被 reevaluation 關閉
- 被 broker / manual 操作關閉

### 9.2 `CloseControlPlane` 負責什麼

所有主動的 close-domain 動作都會經過 `CloseControlPlane`。

它統一處理三種 action：

- `modify_only`
- `partial_close`
- `full_close`

這樣做的作用是：

- tactical exit
- best-day close
- reevaluation close

都能共用同一條 broker-facing control plane，而不是各自實作自己的關倉邏輯。

### 9.3 `CloseReconciler` 如何決定最後的 close facts

position 真正消失之後，系統會嘗試湊齊 close facts：

1. broker closed positions
2. execution meta
3. best-day / reevaluation recorded pnl
4. last-known profit fallback

接著由 `CloseReconciler` 決定：

- `trigger_source`
- `action_kind`
- `final_close_reason`
- `resolution_path`

常見的 canonical close reasons 包括：

- `tp_hit`
- `sl_hit`
- `broker_stopout`
- `best_day_close`
- `reeval_close`
- `manual_close`

### 9.4 close 完成後系統會做哪些收尾

一筆交易結束後，系統至少會做以下幾件事：

- `mark_closed`，把 pnl、exit price、exit reason、hold duration 寫回 store
- 寫入 `TRADE_CLOSED` trade event
- invalidates decision cache
- 更新 BestDayTracker
- 更新 HWM tracker
- 若 pnl 非 0，嘗試呼叫 `agents.reflect()` 做 post-trade reflection
- 寫 alert / summary / journal

這代表 close path 不是一個 log-only path，而是會反向影響：

- 後續 admission
- Best Day tracking
- optimization / AB stats
- memory / reflection material

---

## 10. 各功能模組在整條鏈中的角色

| 模組 | 在 stable 實作中的角色 | 主要責任 |
|---|---|---|
| **`ScannerBridge`** | scanner downstream contract adapter | 驗證 bundle、選定 target date、把 scanner output 轉成 pilot 可用 signals |
| **`Scheduler`** | 交易 runtime orchestration core | 生成 intents、跑 decision/tactical、管理 loops、協調 monitor 與 close paths |
| **`TacticalValidator`** | entry-timing gate | 檢查當下 `5m/1h` 市況是否支持進場 |
| **`ExecutionEngine`** | execution pipeline owner | compliance、sizing、slippage、broker open、SL/TP、execution meta |
| **`PropFirmGuard`** | safety-critical compliance core | drawdown、Best Day、admission headroom 與 execution-side gating |
| **`BrokerFactory`** | broker-neutral runtime selector | 在 `matchtrader` / `tradelocker` 間切換 backend |
| **`TradeLockerClient` / `MatchTraderClient`** | concrete broker adapters | 登入、報價、開倉、關倉、修改單、持倉查詢 |
| **`CloseControlPlane`** | close-domain execution bus | 統一路由 modify / partial / full close |
| **`CloseReconciler`** | canonical close attribution layer | 合併 broker facts 與 fallback data，產出最終 close reason |
| **`AlertService`** | operator notification layer | Telegram send、retry、circuit breaker、fallback sink |
| **`TradeJournal`** | append-only lifecycle log | 保存 trade / scanner / tactical / alert / equity events |

---

## 11. 當前 stable 邊界與未完成項

### 11.1 已落地的 stable implementation 邊界

目前可以很明確地說：

- 系統已具備一條 **不依賴 LLM 的完整 trading path**
- broker-neutral runtime 已完成接線
- scanner -> tactical -> execution -> close reconciliation 已形成完整閉環
- market-data continuity fallback 與 canonical close attribution 都已落地

### 11.2 還未完成的 stable release closure

目前還不能說「正式 stable release 已完成」，主要是因為以下幾項仍未 formalize：

- open-book worst-case natural-SL portfolio risk guard
- exposure / portfolio budget v1
- trade-memory v1 與 quality gate
- multi-day acceptance evidence
- version bump 與 release identity closure

### 11.3 為什麼 `TradingAgents` default-off 是合理的

在當前 stable 邊界下，`TradingAgents` default-off 的意義是：

- 先確保 deterministic trading runtime 本身可獨立成立
- 再把 LLM path 當成顯式 opt-in 的 bounded enhancement
- 避免 stable release 仍被 unbounded LLM behavior 綁住

因此，這個預設不代表 agent 路徑被放棄，而是代表 stable 主線先把 execution correctness 放在前面。

---

## 12. 一句話版結論

> `PropFirmPilot` 目前的 `v1.5.0_stable` implementation 已經能用 `scanner -> tactical -> execution -> position monitor -> close reconciliation` 這條 deterministic 主路徑完整跑完一筆交易；而 `TradingAgents` 則被保留為顯式 opt-in 的 bounded confirm / veto 層，而不是 stable 預設依賴。 
