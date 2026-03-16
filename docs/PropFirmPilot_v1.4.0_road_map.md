# PropFirmPilot v1.4.0 之後 — 發展路線圖

> **更新日期**: 2026-03-16
>
> **主線狀態**: `main` 已合入 `v1.5.0_beta` cross-repo scanner contract 與 beta version baseline（2026-03-16）
>
> **最新 tactical bugfix 基線**: `v1.4.9`（Market Data Freshness Semantics And Tactical Warning Throttling）
>
> **最新 cross-repo beta 基線**: `v1.5.0_beta`（Scanner Contract Gate And Cross-Repo Beta Baseline）
>
> **涵蓋範圍**: `prop-firm-pilot` · `qlib_market_scanner` · `TradingAgents` 三倉庫協作
>
> **帳戶階段**: E8 Markets One-Phase $5,000 Challenge

---

## 目錄

1. [這份文件現在要回答什麼](#1-這份文件現在要回答什麼)
2. [截至 v1.5.0_beta mainline 已落地的基線](#2-截至-v150_beta-mainline-已落地的基線)
3. [當前真正的瓶頸是什麼](#3-當前真正的瓶頸是什麼)
4. [修訂後版本路線：v1.4.7 → v1.5.0](#4-修訂後版本路線v147--v150)
5. [v1.5.0（stable）詳細設計](#5-v150stable詳細設計)
6. [統一路線圖與里程碑](#6-統一路線圖與里程碑)

---

## 1. 這份文件現在要回答什麼

`v1.4.0` 到 `v1.4.6d` 之間，系統的實際演進已經和早期 roadmap 不同。

今天真正需要回答的，不再是「要不要做 WebSocket-first」、「要不要加 learning loop」，而是以下四個問題：

1. 截至 `v1.5.0_beta` mainline，哪些能力已經成為既成基線，不能再寫成未來式？
2. `v1.4.9` 與 `v1.5.0_beta` 各自解決了什麼，為什麼現在仍不能直接把主線稱為 `stable`？
3. `v1.5.0` 為什麼仍要被定義成第一個 `stable` 版本，而且 stable 到底代表什麼？
4. `prop-firm-pilot`、`qlib_market_scanner`、`TradingAgents` 三個 repo 在 `v1.5.0 stable` 前各自還剩下什麼責任？

### 1.1 重寫原則

- **先承認既成事實，再排未來版本。** 已經進入主線的能力，不再重複規劃。
- **先把 tactical entry / exit 做穩，再追求更高頻的 alpha 與更多交易宗數。**
- **`v1.5.0` 不是功能堆疊版，而是穩定版。** 其目標是把進出場、記憶、風控、資金效率與跨 repo 契約一起穩住。
- **分鐘級交易不等於分鐘級亂掃描。** `qlib_market_scanner` 已在 `v1.5.0_beta` 完成第一輪 FX cadence 研究，release cadence 先凍結為 `1d`；後續升頻必須拿新證據，而不是憑直覺翻轉。
- **TradingAgents 不能再假設 daily-stock cadence。** 若 scanner 與 runtime 走向 intraday，agent 也必須同步調整節奏、記憶與 prompt 邊界。

---

## 2. 截至 v1.5.0_beta mainline 已落地的基線

### 2.1 已成為主線能力的內容

| 能力 | 截至 `v1.5.0_beta` mainline 的狀態 | 意義 |
|---|---|---|
| **Observe** | `fx_websocket_client.py` + `fx_tick_aggregator.py` + `market_data_hub.py` 已形成 WebSocket-first、REST fallback、warm-cache、degraded handling 基線，且 `v1.4.9` 已把 bar freshness semantics 收斂到 effective close time | 不再是 PoC，而是 production 依賴 |
| **Learn** | structured reflection payload、persistent lesson retrieval、`historical_pnl_context`、`retrieved_trade_lessons` 已存在 | 學習迴圈已閉合，但還未完全穩定化 |
| **Act** | 已有 tactical entry gate、tactical pending / retry lifecycle、close control plane、close reconciler、execution metadata 回寫，且 `v1.5.0_beta` 已把 scanner bundle validation gate 合入主線 | 已具備完整控制面雛形 |
| **Operate** | run-specific logs、bundle manifest、Dropbox diagnostics sync、live probe、shared version helper 已落地 | 已具備事故排查基線 |
| **Close consistency** | `v1.4.8` 已把 tactical exit、reduce exposure、emergency close、best-day close、reeval close 收斂到單一 close-domain schema | 平倉不再只是分散路徑，而是可審計控制面 |
| **Cross-repo contract** | `v1.5.0_beta` 已把 scanner `manifest/schema/version/validation status` ingestion gate、metadata persistence 與 upstream `1d` canonical cadence 決議吸收到 pilot 主線 | 三倉庫契約已從 ad-hoc integration 升級為 beta 基線 |
| **Hotfix discipline** | `v1.4.5a`、`v1.4.6b`、`v1.4.6c`、`v1.4.6d`、`v1.4.9` 連續修正 stale fallback、scanner compatibility、freshness semantics、REST loop 與 warning noise | 顯示目前瓶頸在可靠性與控制面一致性，而不是缺功能 |

### 2.2 已知事實

- 系統已經不是「每日一次掃描 + 單次決策」那麼簡單，而是持續運作的 24/7 async pipeline。
- tactical entry / exit 已存在，但仍有 correctness、provenance、read-back verification、state consistency 的尾端問題。
- `qlib_market_scanner` 已完成第一輪 FX cadence research，`v1.5.0_beta` 正式凍結 release cadence 為 `1d`；眼前重點已不是「要不要立刻升到 `1h`」，而是如何在穩定契約下吸收研究結果。
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

### 3.3 `qlib_market_scanner` 的 cadence 問題已從「是否可研究」轉為「如何穩定落地」

`v1.5.0_beta` 之前，scanner 的核心問題是 cadence mismatch；`v1.5.0_beta` 之後，這個問題已經有了第一輪正式答案。

目前已知事實是：

- 上游 `qlib_market_scanner` 已完成 FX cadence matrix，比較 `1d`、`1h`、`4h+1h`、`1d+1h`
- 第一輪正式 release decision 維持 `1d` 為 canonical FX cadence
- `1h` 仍是後續最值得深挖的 follow-up 候選，但不屬於 `v1.5.0_beta` 的 release default
- pilot 現在已把這個結果吸收為 versioned bundle contract，而不是繼續把 cadence choice 當作未定問題

因此眼前真正的問題變成：

- 如何把 `1d` release cadence 的 contract、metadata、validation gate 穩定落地到多日 market-open 運行
- 如何保留 `1h` / hybrid cadence 作為 `v1.5.x` follow-up 研究，而不污染當前 stable gate
- 如何讓 downstream 明確理解 scanner score 的時間框架、有效期限與 label version

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

### 3.7 三個 repo 之間已有 beta 契約，但還沒有 stable 契約

`v1.5.0_beta` 已把三倉庫協作邊界從 ad-hoc integration 推進到 beta contract，但還沒有完成 stable-level closure。

已經落地的部分：

- scanner 輸出已有 `manifest/schema/version/validation` bundle 契約
- pilot 已能拒收 degraded / stale / invalid scanner bundle
- 上游 FX canonical cadence 與 label family 已有第一輪凍結口徑

仍未完成的部分：

- TradingAgents 的 state / prompt / risk schema 仍容易隨 feature 演進漂移
- multi-day validation 尚未證明這套 beta contract 已足以作為 stable release gate
- 記憶、portfolio guard 與 agent schema 仍未被納入同一個 release-acceptance 框架

`v1.5.0 stable` 前必須把這套 beta contract 升級成產品層級的 stable interface contract。

---

## 4. 修訂後版本路線：v1.4.7 → v1.5.0

### 4.1 基線：`v1.4.9`

`v1.4.9` 的角色是收斂 `v1.4.8a` 延續下來的 freshness semantics 與 warning noise 問題，讓後續版本能把焦點從單點 hotfix 轉到 cross-repo beta integration。

這版之後的判斷基準是：

- 市場資料 path 已可用，且 closed-bar freshness semantics 已和 tactical stale-bar sanitize 對齊
- tactical entry / exit 已存在，但還未達 stable release 的可靠性要求
- 下一個主瓶頸不再是 REST loop / stale warning，而是 scanner / agent / pilot 三者的 stable contract closure

### 4.2 v1.4.7 — Tactical Entry Fixes & Optimization

這一版作為設計工作包仍然成立，但其核心收斂內容已被後續 `v1.4.8a`、`v1.4.9` 與 `v1.5.0_beta` 部分吸收。

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

這一版的任務是：**用 market-open 實盤結果驗證 `v1.4.7` / `v1.4.8` 的 tactical 主線，只修 bug，不擴 scope。**

#### 目前狀態

- 已於 `2026-03-16` 合入 `main`
- `MarketDataHub` 與 scheduler stale-bar sanitize 已統一採用 effective close time freshness semantics
- stale tactical-bar warnings 已做 stateful throttling，operator 不再每分鐘看到同一條重複 warning

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

### 4.5 v1.5.0_beta — Cross-Repo Contract Freeze And Beta Validation Gate

這一版的任務是：**把 upstream scanner research 結果與 versioned bundle contract 正式吸收到 pilot `main`，但仍保留 beta 身分。**

#### 目標

- 將 `prop-firm-pilot` 與 `qlib_market_scanner` 版本統一到 `1.5.0_beta`
- 在 pilot 主線正式落地 scanner manifest/schema/version/validation gate
- 持久化 scanner metadata，讓 intents / journal / postmortem 能看見 scanner contract facts
- 明確宣告 FX canonical release cadence 維持 `1d`，不在 beta 階段翻轉 runtime scanner cadence

#### 核心工作

| 類別 | 工作項 |
|---|---|
| **Bundle validation** | 驗證 scanner manifest、schema version、scanner version、validation status 與 required signal columns |
| **Metadata persistence** | 將 `scanner_version`、`scanner_schema_version`、`scanner_market_date`、`scanner_label_version` 寫入 `TradeIntent` / `DecisionStore` |
| **Rejection handling** | 對 degraded / stale / invalid bundle 產生 deterministic rejection reason code、journal event 與 operator alert |
| **Cross-repo alignment** | 對齊 upstream `v1.5.0_beta` release identity，並保留對既有 `v1.5.0` artifacts 的向後相容 |
| **Docs / release identity** | 將 roadmap、changelog、runtime version source 及 cross-repo note 改寫成 beta baseline 口徑 |

#### 完成條件

- `main` 已能 ingest versioned scanner bundle，而不是只吃 ad-hoc CSV
- `prop-firm-pilot` 與 `qlib_market_scanner` 對外版本字串一致顯示 `1.5.0_beta`
- 下一步 stable 驗證可以針對真實跨 repo 契約進行，而不是針對暫時性 fixture 假設進行

### 4.6 v1.5.0（stable）— Broader Decision / Risk Upgrade

`v1.5.0` 的定位不是一般 feature release，而是第一個 **stable release milestone**。

它的要求不是「功能很多」，而是：

- tactical entry 穩定可靠
- tactical exit 穩定可靠
- scanner / agent / memory / risk / execution 之間的產品契約穩定
- 系統能用多元化倉位更有效率地運用資金

`v1.5.0` 的詳細設計見下一節。

---

## 5. v1.5.0（stable）詳細設計

> **2026-03-16 更新**:
> `v1.5.0_beta` 已先把 scanner contract freeze、beta version identity 與 FX `1d` cadence decision 合入 `main`。
> 以下內容描述的是 beta 之後仍需完成的 stable gate closure，而不是說這些工作都還沒開始。

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

`v1.5.0_beta` 前，這個工作流的核心是判斷 FX release cadence 應不應升到更高頻；`v1.5.0_beta` 後，第一輪 release decision 已經做完。

目前 scanner 相關的現況是：

- `qlib_market_scanner` 已完成 `1d`、`1h`、`4h+1h`、`1d+1h` matrix
- 第一輪正式 release decision 維持 `1d` 為 canonical FX cadence
- `1h` 仍是最值得在 `v1.5.x` 深挖的 follow-up 候選，但不屬於當前 stable release default
- pilot 已將這個決議吸收為 versioned ingestion contract，而不是把 cadence choice 留在 runtime 配置層即興決定

#### 5.4.2 核心研究問題

因此 `v1.5.0 stable` 在這條工作流真正要回答的是：

1. 如何把已凍結的 `1d` release cadence 轉成 multi-day、可驗收的 stable contract？
2. 如何保留 `1h` / hybrid cadence 作為 `v1.5.x` follow-up research，而不污染當前 stable gate？
3. scanner metadata、prediction horizon 與 label version 要如何持續被 downstream 正確理解與驗證？

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
  ├─ v1.4.7 ~~~~~~ Tactical Entry Fixes & Optimization
  │                · work-package scope later absorbed by v1.4.8a / v1.4.9 / v1.5.0_beta
  │                · entry correctness
  │                · provenance
  │
  ├─ v1.4.8 ✅ ─── Tactical Exit Fixes & Optimization
  │                · close control plane
  │                · canonical reconciliation
  │                · unified trade-closed payload
  │
  ├─ v1.4.9 ✅ ─── Bugfix And Micro-Tuning Pass
  │                · market-open validation follow-up
  │                · close-time freshness semantics
  │                · stale warning throttling
  │
  ├─ v1.5.0_beta ✅ ─── Scanner Contract Gate And Cross-Repo Beta Baseline
  │                     · versioned scanner bundle ingestion
  │                     · scanner metadata persistence
  │                     · FX canonical cadence frozen to 1d
  │
  └─ v1.5.0 ────── Stable Gate Closure
                    · stable entry / exit
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
| Scanner contract freeze / bundle validation gate | 🔴 高 | 🟡 中 | ⭐⭐⭐ | `v1.5.0_beta` |
| Exposure budget / diversified sizing | 🔴 高 | 🟡 中 | ⭐⭐⭐ | `v1.5.0` |
| Intraday scanner feasibility / effectiveness research | 🔴 高 | 🔴 高 | ⭐⭐⭐ | `v1.5.0_beta` |
| TradingAgents intraday schema alignment | 🔴 高 | 🔴 高 | ⭐⭐⭐ | `v1.5.0` |
| Stable trade memory infrastructure | 🔴 高 | 🟡 中 | ⭐⭐⭐ | `v1.5.0` |

### 6.3 里程碑定義

| 里程碑 | 達成條件 | 目標版本 |
|---|---|---|
| **M0: `v1.4.6d` hotfix 基線** | market-data fallback loop、scanner CLI compatibility、tactical freshness hotfix 已收斂 | 已完成 |
| **M1: Entry 可回放** | 每次 entry verdict 都有 deterministic reason code、source provenance、score breakdown | `v1.4.7` |
| **M2: Exit 可審計** | 每次 exit 都有 trigger source、broker read-back、journal consistency、close reason；close-control 已進入 `main` | `v1.4.8` |
| **M3: Tactical 驗證收斂** | `v1.4.7` / `v1.4.8` 經 market-open run 驗證後，主線只剩可接受的小幅微調 | `v1.4.9` |
| **M4: Beta 契約凍結** | scanner cadence decision、bundle schema、version identity 與 pilot ingestion gate 已一起進入 `main` | `v1.5.0_beta` |
| **M5: Stable 閉環成立** | agent schema、trade memory、capital efficiency 與 tactical control plane 一起穩定運作 | `v1.5.0` |

---

## 結語

從 `v1.4.0` 到 `v1.4.6d`，系統最大的進展不是又多了幾個 feature，而是已經把 WebSocket-first market data、tactical control plane、lesson loop、diagnostics workflow 都推進到了真正可營運的程度。

下一階段的正確做法，不是再堆新功能，而是按順序完成：

1. `v1.4.7` 的 entry-hardening scope 已由後續版本逐步吸收
2. `v1.4.8` 把 exit close-control 做穩，並已先合入 `main`
3. `v1.4.9` 收斂 freshness semantics 與 warning noise，完成 bugfix / micro-tuning pass
4. `v1.5.0_beta` 已把 scanner contract、cadence decision 與 beta version identity 吸收入 `main`
5. `v1.5.0` 再把 scanner、agents、memory、risk、execution 一起提升到 stable 等級

這樣的版本順序，才能讓 `v1.5.0` 真正代表一個可以被信任的 FX 自動交易系統基線，而不是另一個需要連續熱修補才能勉強運行的版本。
