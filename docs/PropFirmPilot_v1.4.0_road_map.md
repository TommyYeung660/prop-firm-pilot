# PropFirmPilot v1.4.0 之後 — 發展路線圖

> **更新日期**: 2026-03-14
>
> **主線狀態**: `main` 已合入 `v1.4.8` Close Control Plane 實作（2026-03-14）
>
> **最新 prod hotfix 基線**: `v1.4.6d`（MarketDataHub REST Fallback Loop Fix）
>
> **涵蓋範圍**: `prop-firm-pilot` · `qlib_market_scanner` · `TradingAgents` 三倉庫協作
>
> **帳戶階段**: E8 Markets One-Phase $5,000 Challenge

---

## 目錄

1. [這份文件現在要回答什麼](#1-這份文件現在要回答什麼)
2. [截至 v1.4.8 mainline 已落地的基線](#2-截至-v148-mainline-已落地的基線)
3. [當前真正的瓶頸是什麼](#3-當前真正的瓶頸是什麼)
4. [修訂後版本路線：v1.4.7 → v1.5.0](#4-修訂後版本路線v147--v150)
5. [v1.5.0（stable）詳細設計](#5-v150stable詳細設計)
6. [統一路線圖與里程碑](#6-統一路線圖與里程碑)

---

## 1. 這份文件現在要回答什麼

`v1.4.0` 到 `v1.4.6d` 之間，系統的實際演進已經和早期 roadmap 不同。

今天真正需要回答的，不再是「要不要做 WebSocket-first」、「要不要加 learning loop」，而是以下四個問題：

1. 截至 `v1.4.8` mainline，哪些能力已經成為既成基線，不能再寫成未來式？
2. 為什麼在 `v1.4.8` 已進入 `main` 之後，仍然要保留 `v1.4.9` 作為 market-open bugfix / 微調版本，而不是直接跳到更大的 feature scope？
3. `v1.5.0` 為什麼要被定義成第一個 `stable` 版本，而且 stable 到底代表什麼？
4. `prop-firm-pilot`、`qlib_market_scanner`、`TradingAgents` 三個 repo 在 `v1.5.0` 前各自要負責什麼？

### 1.1 重寫原則

- **先承認既成事實，再排未來版本。** 已經進入主線的能力，不再重複規劃。
- **先把 tactical entry / exit 做穩，再追求更高頻的 alpha 與更多交易宗數。**
- **`v1.5.0` 不是功能堆疊版，而是穩定版。** 其目標是把進出場、記憶、風控、資金效率與跨 repo 契約一起穩住。
- **分鐘級交易不等於分鐘級亂掃描。** 是否採用小時級或分鐘級量化分析，必須由 `qlib_market_scanner` 的研究結果決定，而不是憑直覺升頻。
- **TradingAgents 不能再假設 daily-stock cadence。** 若 scanner 與 runtime 走向 intraday，agent 也必須同步調整節奏、記憶與 prompt 邊界。

---

## 2. 截至 v1.4.8 mainline 已落地的基線

### 2.1 已成為主線能力的內容

| 能力 | 截至 `v1.4.8` mainline 的狀態 | 意義 |
|---|---|---|
| **Observe** | `fx_websocket_client.py` + `fx_tick_aggregator.py` + `market_data_hub.py` 已形成 WebSocket-first、REST fallback、warm-cache、degraded handling 的市場資料基線 | 不再是 PoC，而是 production 依賴 |
| **Learn** | structured reflection payload、persistent lesson retrieval、`historical_pnl_context`、`retrieved_trade_lessons` 已存在 | 學習迴圈已閉合，但還未完全穩定化 |
| **Act** | 已有 tactical entry gate、tactical pending / retry lifecycle、close control plane、close reconciler、execution metadata 回寫 | 已具備完整控制面雛形 |
| **Operate** | run-specific logs、bundle manifest、Dropbox diagnostics sync、live probe、shared version helper 已落地 | 已具備事故排查基線 |
| **Close consistency** | `v1.4.8` 已把 tactical exit、reduce exposure、emergency close、best-day close、reeval close 收斂到單一 close-domain schema | 平倉不再只是分散路徑，而是可審計控制面 |
| **Hotfix discipline** | `v1.4.5a`、`v1.4.6b`、`v1.4.6c`、`v1.4.6d` 連續修正 stale fallback、scanner compatibility、freshness、REST loop | 顯示目前瓶頸在可靠性與控制面一致性，而不是缺功能 |

### 2.2 已知事實

- 系統已經不是「每日一次掃描 + 單次決策」那麼簡單，而是持續運作的 24/7 async pipeline。
- tactical entry / exit 已存在，但仍有 correctness、provenance、read-back verification、state consistency 的尾端問題。
- `qlib_market_scanner` 現在仍偏向日線節奏，這限制了每日可交易宗數與 intraday alpha 的利用率。
- `TradingAgents` 已經能用 lessons 與市場上下文，但仍未真正為 intraday FX 節奏而設計。
- 市場資料問題已證明：只修單一 bug 不夠，必須把 entry/exit、memory、risk、scanner cadence 一起重整。

---

## 3. 當前真正的瓶頸是什麼

### 3.1 Tactical entry 還未達到 production-grade reliability

`v1.4.6b` 和 `v1.4.6d` 已修掉最危險的 freshness / fallback loop 問題，但 entry 端仍欠缺：

- `WAIT` / `retry` / `degrade` / `timed_out` 的可解釋性
- per-symbol / session / regime 的實證校準
- source provenance 與 stale attribution 的統一輸出
- candidate 被跳過、延後、降級時的 deterministic reason code

這表示 entry 現在是「能跑」，但還不是「穩定可靠」。

### 3.2 Tactical exit 已經有骨架，但還不夠可審計

現況不是「沒有 exit」，而是 exit 還不夠可靠：

- duplicate close / partial close / trailing / reprice 的 read-back 驗證仍需統一
- tactical exit 與 emergency close、reduce exposure、manual close 的邊界還需收斂
- trade journal、execution metadata、alert、close reason 仍可能出現不一致
- postmortem 時還不能保證完整 replay 每一步出場理由與執行結果

這也是為什麼 `v1.4.8` 必須專注在 exit，而不是把 exit 當作附屬小修。

### 3.3 `qlib_market_scanner` 的日線設計限制了交易頻率

目前 scanner 的根本問題不是分數高低，而是 cadence mismatch：

- 原設計偏向日線 / 美股量化研究
- 本專案是 FX、持續監察、持續管理、需要更高頻率候選更新
- 即使 tactical monitor 是分鐘級，若 strategic score 每日才有效變一次，候選池還是嚴重受限

因此真正要研究的是：

- `qlib` 是否能有效支援 FX 的 `1h`、`15m`、甚至更細粒度研究？
- 若可以，哪些 horizon / label / feature engineering 才有實際訊號價值？
- 若不適合直接走分鐘級，是否應先走 `1d + 1h` 或 `4h + 1h` 的 hybrid 模式？

### 3.4 TradingAgents 也受制於同樣的 cadence mismatch

即使 scanner 升頻，若 agent 還是用 daily-style prompt 與工具節奏，仍然會出現：

- decision context 過重、過慢
- intraday setup 無法被正確表述
- tactical signal 與 agent risk narrative 彼此脫節
- lessons 雖存在，卻不一定能反映當前 session / regime / setup 類型

所以 `TradingAgents` 的問題不只是模型好不好，而是：

- state schema 是否能承載 intraday FX context
- prompt / tool / memory retrieval 是否能匹配更高頻的候選更新
- risk output 是否能穩定轉成可執行的 entry / exit 約束

### 3.5 交易記憶還沒有成為穩定、可持續改善的基礎設施

目前記憶能力已存在，但仍分散在多個層次：

- `TradeJournal`
- `MemoryJournal`
- structured reflection payload
- TradingAgents lesson memory
- execution outcome / close reason metadata

真正缺的是單一記憶契約：

- 什麼事件必須被記錄？
- 何時產生 lesson？
- 何時可被 retrieval？
- 哪些欄位是 stable schema，哪些只是 debug metadata？

若這一層不穩，TradingAgents 的自我改善就會一直漂移。

### 3.6 資金利用效率仍偏向單筆交易視角

目前系統更像「盡量別犯大錯」，而不是「在合規下高效率運用資金」：

- 缺少 portfolio-level exposure budget
- 缺少同幣別 / 高相關倉位的資金分配規則
- 缺少在低相關 setup 間做多元化配置的正式控制面

`v1.5.0` 若要稱為 stable，就不能只會穩定開一筆倉，而要能穩定管理多筆、不同相關性的倉位。

### 3.7 三個 repo 之間仍缺少穩定的產品契約

目前三倉庫的問題不是不能協作，而是協作邊界仍偏工程暫定：

- scanner 輸出欄位與含義仍可能變動
- TradingAgents 的 state / prompt / risk schema 仍容易隨 feature 演進漂移
- prop-firm-pilot 需要額外 defensive logic 來承受外部 repo 的變化

`v1.5.0` 前必須把這個問題升級成產品層級的 interface contract。

---

## 4. 修訂後版本路線：v1.4.7 → v1.5.0

### 4.1 基線：`v1.4.6d`

`v1.4.6d` 的角色是結束 market-data fallback loop 這條 hotfix 線，讓後續版本能把焦點從救火轉到 tactical stabilization。

這版之後的判斷基準是：

- 市場資料 path 已可用，但必須補足 provenance 與 operational consistency
- tactical entry / exit 已存在，但還未達 stable release 的可靠性要求
- `qlib_market_scanner` 與 `TradingAgents` 的 cadence mismatch 已成為新主瓶頸

### 4.2 v1.4.7 — Tactical Entry Fixes & Optimization

這一版只做一件事：**把 entry 做穩。**

#### 目標

- 將 tactical entry 從「可運行」提升為「可回放、可校準、可運維」
- 把 entry path 的 reason code、fallback reason、rescan source、score breakdown 統一
- 降低 false WAIT、假性 stale、intent churn、無效候選重複評估

#### 核心工作

| 類別 | 工作項 |
|---|---|
| **Entry correctness** | 收斂 mixed-source freshness、degraded path、scanner candidate gating、session-aware skip / retry 邏輯 |
| **Lifecycle clarity** | 釐清 `claimed -> tactical_pending -> ready_for_exec / degrade / timed_out / cancelled` 狀態轉移 |
| **Calibration** | 建立 per-symbol / session / regime 的 tactical threshold calibration 方法與輸出 |
| **Provenance** | 為 candidate、scanner score、bars / quote source、rescan trigger、entry verdict 補齊 metadata |
| **Cross-repo contract** | 明確定義 pilot 需要 scanner 輸出的欄位、時間戳、score semantics 與 freshness 含義 |

#### 完成條件

- 每次 entry verdict 都能回答「為什麼進、為什麼等、為什麼跳過」
- tactical entry 的 postmortem 不再依賴肉眼拼 logs
- 下一版 exit hardening 不需要再反向修 entry metadata

### 4.3 v1.4.8 — Tactical Exit Fixes & Optimization

這一版只做一件事：**把 exit 做穩。**

#### 目前狀態

- 已於 `2026-03-14` 合入 `main`
- close-domain schema、`CloseControlPlane`、`CloseReconciler` 已落地
- tactical exit、reduce exposure、emergency close、best-day close、LLM re-eval close 已統一走 close control plane
- 下一步不再是擴大 scope，而是等待 market-open validation

#### 目標

- 將 tactical exit 做到 production-grade execution integrity
- 把 exit action、broker read-back、journal、alert、close reason 收斂成單一事實來源
- 讓出場結果可以可靠回寫到 learning loop

#### 核心工作

| 類別 | 工作項 |
|---|---|
| **Exit correctness** | duplicate close 防護、partial close 一致性、reprice / trailing / breakeven read-back verification |
| **Action provenance** | 明確區分 tactical exit、reduce exposure、emergency close、manual close、broker-side close |
| **Journal consistency** | trade journal、execution metadata、alert text、close reason 的欄位定義與回寫順序統一 |
| **Replayability** | 為每次 exit 產出完整 action trail，支援 postmortem 與 memory ingestion |
| **Cross-repo alignment** | 對齊 TradingAgents risk output schema，避免 agent 建議和實際 exit control plane 分裂 |

#### 完成條件

- 每次平倉都能回答「誰觸發、依據什麼、broker 最終回報什麼」
- exit 結果可直接餵進 memory / lesson pipeline，而不需人工清洗
- tactical exit 與 emergency path 的邊界可以穩定運營

### 4.4 v1.4.9 — Bugfix And Micro-Tuning Pass

這一版的任務是：**用 market-open 實盤結果驗證 `v1.4.7` 與 `v1.4.8`，只修 bug，不擴 scope。**

#### 目標

- 驗證 entry hardening 與 close-control 主線在真實 market-open 下的 correctness
- 只修正實盤暴露出的行為缺陷、欄位漂移與 tolerance 細節
- 為 `v1.5.0 stable` 準備乾淨、已驗證的 tactical 基線

#### 核心工作

| 類別 | 工作項 |
|---|---|
| **Correctness bugfix** | 修正 market-open run 暴露出的 close reason 誤分類、trigger source 漂移、pending outcome 邊界問題 |
| **Micro-tuning** | 調整 threshold、read-back tolerance、journal payload、alert wording 等小幅參數與欄位 |
| **Validation gate** | 用週一實盤結果驗證 `v1.4.7` / `v1.4.8` 是否達到 stable 前置條件 |
| **No new scope** | 不新增 exposure budget、portfolio optimizer、跨 repo 新契約，避免 validation 版變成 feature 版 |

#### 完成條件

- `v1.4.7` / `v1.4.8` 的主線 bug 已由 market-open run 驗證並收斂
- tactical close 與 tactical entry 的 postmortem 不再出現 schema / reason taxonomy 漂移
- `v1.5.0` 可以直接接手 stable gate、memory、capital efficiency 與跨 repo 對齊工作

### 4.5 v1.5.0（stable）— Broader Decision / Risk Upgrade

`v1.5.0` 的定位不是一般 feature release，而是第一個 **stable release milestone**。

它的要求不是「功能很多」，而是：

- tactical entry 穩定可靠
- tactical exit 穩定可靠
- scanner / agent / memory / risk / execution 之間的產品契約穩定
- 系統能用多元化倉位更有效率地運用資金

`v1.5.0` 的詳細設計見下一節。

---

## 5. v1.5.0（stable）詳細設計

### 5.1 版本定義

`v1.5.0` 被定義為 stable，代表它必須滿足以下產品層級條件，而不只是「當天看起來沒壞」：

1. **進場穩定可靠**：不因 mixed-source、stale fallback、candidate churn、狀態漂移而頻繁誤判。
2. **出場穩定可靠**：不因 duplicate close、read-back 不一致、journal 漂移而讓真實倉位狀態失真。
3. **可持續改善**：每次交易的上下文、執行結果、lesson、失敗模式都能穩定進入 memory layer。
4. **資金運用效率提升**：系統不只懂得避錯，也懂得在合規下配置多筆低相關倉位。

#### 5.1.1 Stable 不代表什麼

- 不代表一定已做完分鐘級 alpha engine。
- 不代表已支援多帳號或 dashboard。
- 不代表不再需要 hotfix。

Stable 在這裡的意思是：**核心交易閉環可以被信任，可以被運營，可以被持續優化。**

#### 5.1.2 建議 release gate

以下條件應作為 `v1.5.0` 的最低驗收標準：

| 類別 | 建議 gate |
|---|---|
| **Entry integrity** | 所有 entry verdict 都有完整 reason code、score breakdown、source provenance |
| **Exit integrity** | 所有 exit action 都有 trigger source、broker read-back、close reason、journal write-back |
| **Memory integrity** | close 後 lesson / reflection / outcome metadata 有穩定落盤與可檢索性 |
| **Cross-repo contract** | scanner output schema、agent risk schema、pilot ingestion schema 形成固定契約 |
| **Operational stability** | 連續多日運行不再出現相同類型的 P0/P1 tactical correctness incident |

### 5.2 `v1.5.0` 的目標運作模型

#### 5.2.1 目標資料流

1. `qlib_market_scanner` 產出候選訊號、分數、horizon、feature provenance、信號時間戳。
2. `prop-firm-pilot` 以 tactical entry control plane 驗證 candidate 是否在當前 session / regime / freshness 條件下可進。
3. `TradingAgents` 讀取 scanner context、即時市場上下文、歷史 lessons、portfolio 狀態，輸出可執行的風險結論。
4. `prop-firm-pilot` 執行開倉、監控、動態管理、出場，所有 action 都有 reason taxonomy 與 broker read-back。
5. close 後由 pilot 產生 structured outcome，送入 trade journal 與 lesson memory。
6. `TradingAgents` 在下一輪決策中以 symbol / setup / session / regime / outcome 類型檢索 relevant memory。

#### 5.2.2 目標控制原則

- **scanner 負責產生可研究、可排序的候選，不負責最終執行。**
- **pilot 負責 tactical correctness、execution integrity、risk enforcement。**
- **agents 負責更高階的情境推理與風險敘事，但輸出必須能穩定映射到 pilot 的執行 schema。**
- **memory 不只是紀錄，而是下一次 decision 的可檢索基礎設施。**

### 5.3 工作流 A：Tactical Entry / Exit 穩定可靠

這一條對應你的最終目標第 1 點。

#### 5.3.1 問題定義

- entry / exit 已有功能，但仍常靠 hotfix 修 correctness 邊界
- 相同交易在 postmortem 時，仍可能需要從多份 log 拼出真相
- 進出場的資料來源、判斷理由、風控理由、broker 結果尚未完全統一

#### 5.3.2 設計方向

| 子系統 | 設計要求 |
|---|---|
| **Entry validator** | 每個 verdict 都要有 deterministic reason code、score breakdown、source provenance、freshness summary |
| **Intent lifecycle** | `claimed / tactical_pending / ready_for_exec / opened / timed_out / cancelled / closed` 轉移規則完全固定 |
| **Exit manager** | 所有 close 類動作都要可區分 trigger source、read-back 結果與最終 close reason |
| **Journal / alerts** | 只接受來自單一事實來源的欄位，不再各自組字串猜狀態 |
| **Diagnostics** | 任一交易可重建「何時被選中、何時等待、何時進、何時出、為何如此」 |

#### 5.3.3 預期交付

- entry / exit reason taxonomy
- 統一的 action provenance schema
- 完整的 broker read-back verification path
- tactical replay / incident bundle 可直接支援 postmortem 與 memory ingestion

### 5.4 工作流 B：`qlib_market_scanner` 的 intraday FX 研究能力

這一條對應你的最終目標第 2 點。

#### 5.4.1 問題定義

目前 scanner 分數主要依賴日線變化，這在 FX intraday 專案上有兩個直接後果：

- 每日可交易候選宗數被嚴重限制
- 即使分鐘級監察很勤快，也只是反覆檢查同一批 daily-style candidate

#### 5.4.2 核心研究問題

`v1.5.0` 不應直接假設「分鐘級一定更好」，而應回答：

1. `qlib` 對 FX 做 `1h` / `15m` / `5m` 研究是否技術可行？
2. 這些頻率下的 label、特徵與回測結果是否有實際訊號價值？
3. 在 execution cost、spread、噪音下，哪個頻率最適合當 scanner cadence？

#### 5.4.3 建議分階段設計

| 階段 | 目標 | 說明 |
|---|---|---|
| **Phase 1: feasibility** | 驗證 `1h` 與 `15m` 資料管線、label、feature 工程是否穩定可生成 | 先不直接上 production |
| **Phase 2: effectiveness** | 比較 `1d`、`1d+1h`、`4h+1h`、`1h` 等模式的 turnover、IC、命中率與 drawdown 特性 | 找出 FX 真的可用的 cadence |
| **Phase 3: productization** | 將研究結果轉成 production scanner output schema | 明確定義 score、horizon、confidence、timestamp |

#### 5.4.4 需要補的能力

- FX-specific feature engineering，而不是沿用美股因子假設
- session-aware label 設計，例如 London / New York session move
- horizon-aware score，例如 next `1h`、next `4h`、next session，而不是 next day
- scanner output metadata，包括 `score_timeframe`、`prediction_horizon`、`feature_snapshot_time`

#### 5.4.5 成功標準

- 不再只有「每天同一批信號」可交易
- intraday 候選增加時，訊號品質沒有因噪音而全面崩潰
- pilot 能明確知道 scanner score 的時間框架與有效期限

### 5.5 工作流 C：TradingAgents 與 intraday FX 的重新對齊

這一條對應你的最終目標第 3 點。

#### 5.5.1 問題定義

若 scanner 進入較高頻 cadence，TradingAgents 也必須同步調整，否則會出現：

- candidate 變快，但 agent 還是用過慢的 daily 敘事節奏
- prompt 過長，decision latency 過高
- agent 無法理解 tactical setup 與 scanner horizon 的差別
- risk output 仍偏抽象，難以穩定映射到 pilot

#### 5.5.2 設計方向

| 面向 | 設計要求 |
|---|---|
| **State schema** | 必須能明確承載 `scanner_horizon`、`entry_window`、`session_context`、`portfolio_context` |
| **Prompting** | 區分 strategic thesis 與 tactical execution context，避免把 intraday 問題寫成 daily 宏觀作文 |
| **Tools** | 依頻率重新定義 agent 需要的市場資料工具，不必所有 agent 都拿完整長窗資料 |
| **Risk output** | 輸出需標準化，能被 pilot 直接用於 entry / exit / position sizing |
| **Latency** | 決策路徑需控制上下文體積與工具數量，避免升頻後延遲爆炸 |

#### 5.5.3 預期交付

- intraday-aware agent state schema
- 更清楚的 scanner context → agent decision → pilot execution 契約
- 能根據 symbol / session / setup / outcome 檢索記憶的 retrieval policy
- 更少 prompt 漂移與更高的 decision-to-action 一致性

### 5.6 工作流 D：穩定可靠的交易記憶，供 TradingAgents 持續改善

這一條對應你的最終目標第 4 點。

#### 5.6.1 問題定義

現在系統已經「有記憶」，但還沒有「穩定記憶基礎設施」。

缺口包括：

- 不同來源的記憶欄位尚未形成單一 schema
- 不是所有重要事件都會穩定產生 lesson
- retrieval 仍可能缺乏 setup / session / regime 對齊
- debug 資訊與長期可用的 learning 資訊還沒有乾淨分層

#### 5.6.2 設計方向

| 層級 | 設計要求 |
|---|---|
| **Raw event layer** | entry verdict、execution result、exit action、broker read-back、PnL outcome 全部結構化存檔 |
| **Reflection layer** | 從 raw event 生成標準化 lesson，而不是任意文字摘要 |
| **Retrieval layer** | 以 symbol、setup、session、regime、outcome type 為主鍵檢索最相關 lessons |
| **Quality gate** | schema 不完整、close reason 不明、execution 結果不一致的事件不得直接進長期記憶 |

#### 5.6.3 預期交付

- 統一的 trade-memory schema
- `TradeJournal` / `MemoryJournal` / TradingAgents lesson memory 的角色切分
- close-to-memory pipeline 的明確時序
- memory quality metrics，避免垃圾 lesson 累積

### 5.7 工作流 E：多元化倉位與資金效率

這是你在 stable 目標中額外強調的部分，也是 `v1.5.0` 必須新增的設計主軸。

#### 5.7.1 問題定義

現在系統較接近單筆風險最小化，而不是整體資金效率最大化：

- 有可進的 trade，不代表值得佔用當前風險預算
- 多筆看似不同的交易，可能其實高度集中在 USD 暴露
- 若只會守單筆風控，系統就很難穩定提高 capital utilization

#### 5.7.2 設計方向

| 類別 | 設計要求 |
|---|---|
| **Exposure budget** | 對 base / quote currency、方向、session 建立風險預算 |
| **Diversification policy** | 優先分配給低相關、不同事件驅動、不同 session 條件的 setup |
| **Position sizing** | lot sizing 不只看單筆 stop distance，也要看組合內既有暴露 |
| **Portfolio guard** | 明確限制高度相關幣對同向持倉與集中暴露 |

#### 5.7.3 原則

- 不是靠更大槓桿或更高 aggressiveness 來提升效率
- 是靠更好的倉位分配與更乾淨的風險預算來提升效率

### 5.8 三個 repo 的責任切分

| Repo | `v1.5.0` 主要責任 | 不應承擔的責任 |
|---|---|---|
| **`prop-firm-pilot`** | tactical correctness、execution integrity、portfolio guard、memory ingestion、ops diagnostics | 直接承擔 alpha research 或 agent prompt 實驗 |
| **`qlib_market_scanner`** | intraday / hybrid cadence research、FX-specific factor / label 設計、穩定的 scanner output schema | 直接替代 pilot 的 tactical execution 判斷 |
| **`TradingAgents`** | intraday-aware decision schema、memory-aware reasoning、標準化風險輸出 | 自行決定 broker execution 細節或覆寫 pilot risk guard |

### 5.9 `v1.5.0` 明確不納入的範圍

以下項目仍重要，但不應與 stable 版混在一起：

- Dashboard / 視覺化控制台
- 多帳號治理
- RD-Agent 週末自動化
- 全面 async 重寫所有外部工具

這些都應排在 stable 之後，因為它們不該分散目前最重要的閉環穩定化工作。

---

## 6. 統一路線圖與里程碑

### 6.1 修訂後版本時間線

```text
2026-03-11
  │
  ├─ v1.4.0 ✅ ─── WebSocket-first OODA Learning Loop
  │                · WebSocket-first market data hub
  │                · structured reflection payload
  │                · persistent lesson retrieval
  │
  ├─ v1.4.1 ✅ ─── Reliability / Observability Hardening
  │                · live probe
  │                · diagnostics bundle
  │                · degraded fallback suppression
  │
  ├─ v1.4.5a ✅ ─── Tactical Re-entry And Stale Quote Guard
  ├─ v1.4.6b ✅ ─── Tactical Freshness Recovery + Prod Diagnostics Workflow
  ├─ v1.4.6c ✅ ─── Scanner CLI Backward-Compatibility Hotfix
  ├─ v1.4.6d ✅ ─── MarketDataHub REST Fallback Loop Fix
  │
  ├─ v1.4.7 ────── Tactical Entry Fixes & Optimization
  │                · entry correctness
  │                · calibration
  │                · provenance
  │
  ├─ v1.4.8 ✅ ─── Tactical Exit Fixes & Optimization
  │                · close control plane
  │                · canonical reconciliation
  │                · unified trade-closed payload
  │
  ├─ v1.4.9 ────── Bugfix And Micro-Tuning Pass
  │                · market-open validation
  │                · correctness bugfix
  │                · narrow tuning only
  │
  └─ v1.5.0 ────── Stable: Broader Decision / Risk Upgrade
                   · stable entry / exit
                   · intraday-capable scanner research outcome
                   · intraday-aware TradingAgents
                   · stable trade memory
                   · diversified capital deployment
```

### 6.2 優先級矩陣

| 改進項目 | 交易品質影響 | 實施難度 | ROI | 建議版本 |
|---|:---:|:---:|:---:|---|
| Entry reason taxonomy / provenance | 🔴 高 | 🟡 中 | ⭐⭐⭐ | `v1.4.7` |
| Entry threshold calibration | 🔴 高 | 🟡 中 | ⭐⭐⭐ | `v1.4.7` |
| Exit read-back verification | 🔴 高 | 🟡 中 | ⭐⭐⭐ | `v1.4.8` |
| Exit replayability / journal consistency | 🔴 高 | 🟡 中 | ⭐⭐⭐ | `v1.4.8` |
| Market-open bugfix / micro-tuning pass | 🔴 高 | 🟡 中 | ⭐⭐⭐ | `v1.4.9` |
| Exposure budget / diversified sizing | 🔴 高 | 🟡 中 | ⭐⭐⭐ | `v1.5.0` |
| Intraday scanner feasibility / effectiveness research | 🔴 高 | 🔴 高 | ⭐⭐⭐ | `v1.5.0` |
| TradingAgents intraday schema alignment | 🔴 高 | 🔴 高 | ⭐⭐⭐ | `v1.5.0` |
| Stable trade memory infrastructure | 🔴 高 | 🟡 中 | ⭐⭐⭐ | `v1.5.0` |

### 6.3 里程碑定義

| 里程碑 | 達成條件 | 目標版本 |
|---|---|---|
| **M0: `v1.4.6d` hotfix 基線** | market-data fallback loop、scanner CLI compatibility、tactical freshness hotfix 已收斂 | 已完成 |
| **M1: Entry 可回放** | 每次 entry verdict 都有 deterministic reason code、source provenance、score breakdown | `v1.4.7` |
| **M2: Exit 可審計** | 每次 exit 都有 trigger source、broker read-back、journal consistency、close reason；close-control 已進入 `main` | `v1.4.8` |
| **M3: Tactical 驗證收斂** | `v1.4.7` / `v1.4.8` 經 market-open run 驗證後，主線只剩可接受的小幅微調 | `v1.4.9` |
| **M4: Stable 閉環成立** | scanner cadence、agent schema、trade memory、capital efficiency 與 tactical control plane 一起穩定運作 | `v1.5.0` |

---

## 結語

從 `v1.4.0` 到 `v1.4.6d`，系統最大的進展不是又多了幾個 feature，而是已經把 WebSocket-first market data、tactical control plane、lesson loop、diagnostics workflow 都推進到了真正可營運的程度。

下一階段的正確做法，不是再堆新功能，而是按順序完成：

1. `v1.4.7` 把 entry 做穩
2. `v1.4.8` 把 exit close-control 做穩，並已經先合入 `main`
3. `v1.4.9` 只做 `v1.4.7 / v1.4.8` 的 bugfix、微調與 market-open validation
4. `v1.5.0` 把 scanner、agents、memory、risk、execution 一起提升到 stable 等級

這樣的版本順序，才能讓 `v1.5.0` 真正代表一個可以被信任的 FX 自動交易系統基線，而不是另一個需要連續熱修補才能勉強運行的版本。
