# PropFirmPilot v1.4.0 之後 — 發展路線圖

> **更新日期**: 2026-03-12
>
> **當前版本**: v1.4.1（Production Reliability / Observability Hardening）
>
> **涵蓋範圍**: `prop-firm-pilot` · `qlib_market_scanner` · `TradingAgents` 三倉庫協作
>
> **帳戶階段**: E8 Markets One-Phase $5,000 Challenge

---

## 目錄

1. [這份 v1.4.0 路線圖要回答什麼](#1-這份-v140-路線圖要回答什麼)
2. [v1.3.5 路線圖 vs v1.4.0 實際落地](#2-v135-路線圖-vs-v140-實際落地)
3. [v1.4.0 之後的核心缺口](#3-v140-之後的核心缺口)
4. [修訂後版本路線](#4-修訂後版本路線)
5. [統一路線圖](#5-統一路線圖)

---

## 1. 這份 v1.4.0 路線圖要回答什麼

`docs/PropFirmPilot_v1.3.5_road_map.md` 當時把 `v1.4.0` 規劃成三條主線：

- 學習迴圈閉合
- 新聞事件觸發
- WebSocket PoC

但 `docs/PropFirmPilot_v1.4.0_Report.md` 顯示，實際完成的不是三個彼此獨立的小功能，而是一次更大的**控制面升級**：

- `Observe` 從輪詢式資料取得，升級為 **WebSocket-first + REST fallback**
- `Orient/Decide` 從「只看當下 signal」升級為「帶歷史績效、事件上下文、歷史 lessons 的決策」
- `Act` 從「平倉即結束」升級為「平倉結果結構化回寫，成為下一輪決策輸入」

因此，這份文件的目的不是重複版本報告，而是回答三件事：

1. `v1.3.5` 時代規劃的哪些東西，到了 `v1.4.0` 已經真正落地？
2. `v1.4.0` 雖然完成閉環，但還剩哪些架構缺口沒有補完？
3. 接下來的版本優先序，應該如何從「功能堆疊」改成「控制面深化」？

### 1.1 路線圖的重寫原則

從 `v1.4.0` 開始，版本優先序應遵守以下原則：

- **先強化既有 OODA 控制面，再增加外圍功能**。先把即時觀測、學習回流、風險隔離做穩，再談 UI、多帳號。
- **先提升交易品質與風險品質，再提升功能數量**。優先做 lessons 擴散、動態出場、組合風險，而不是先做展示層。
- **WebSocket-first 是平台基線，不再只是 PoC**。後續版本要做的是整合與觀測，不是重新論證它值不值得。
- **合規與 safety-critical 規則不作為換速度的代價**。drawdown、best day rule、market-hours、equity monitor 仍是不可退讓的底線。

---

## 2. v1.3.5 路線圖 vs v1.4.0 實際落地

### 2.1 已完成或超額完成的項目

| v1.3.5 路線圖規劃 | 當時預期 | v1.4.0 實際落地 | 判定 |
|---|---|---|---|
| **學習迴圈閉合** | 啟用 `reflect_and_remember()`、注入歷史盈虧 | `Scheduler` 會建立 structured reflection payload；`TradingAgents` 會持久化 lesson memory，並在決策前檢索 `retrieved_trade_lessons` | ✅ 完成，而且比原規劃更完整 |
| **歷史盈虧注入 LLM 提示詞** | 把近 7 日交易表現注入 prompt | `historical_pnl_context` 成為 graph state 的正式欄位，並進入 trader prompt | ✅ 完成 |
| **新聞事件觸發** | 新增 `NewsEventTrigger` 觸發重掃 | `Scheduler.start()` 已包含 `news event loop`，並與 `market_event_context`、`rescan_event` 串接 | ✅ 完成 |
| **WebSocket PoC** | 建立 EODHD WebSocket client 原型 | 實際落地為 `fx_websocket_client.py` + `fx_tick_aggregator.py` + `market_data_hub.py`，形成 WebSocket-first Observe 層 | ✅ 超額完成 |
| **戰術層與即時資料互補** | 讓 WebSocket 未來可支援 tactical layer | tactical validation 已改為 hub-first data fetch，直接吃 WebSocket-derived quote 與 closed bars | ✅ 完成整合 |
| **24x7 控制面方向** | 從全輪詢走向事件驅動 | `Scheduler` 在 market data、volatility、news、scanner、execution、reflection 間形成 steady-state runtime flow | ✅ 完成骨架 |

### 2.2 部分完成、改道完成、或仍未完成的項目

| 項目 | v1.3.5 路線圖預期 | v1.4.0 現況 | 判定 |
|---|---|---|---|
| **緊急平倉增強** | 波動觸發後立即刷新權益，並有 80%/90% 分級處置 | `v1.4.0` 報告確認 equity monitor 與 drawdown guards 仍在，但未把分級減倉/全平寫成明確新基線 | ⚠️ 部分完成 |
| **MemoryJournal 語意升級** | `prop-firm-pilot` 內部記憶系統向量化 | `v1.4.0` 實際走的是 `TradingAgents` persistent lesson memory 路線，形成替代方案，但 `MemoryJournal` 本身尚未被統一進同一記憶體系 | ⚠️ 改道完成一部分 |
| **動態 SL/TP** | ATR-based 動態出場 | `v1.4.0` 仍以 Observe 與 learning loop 為主，未把 exit management 升級為主要變更 | ❌ 未完成 |
| **相關性檢測 / 組合風險** | 避免多倉位共振風險 | `v1.4.0` 仍以單交易意圖與單標的 decision flow 為主，缺少 portfolio-level correlation layer | ❌ 未完成 |
| **TradingAgents async 化** | `requests` 遷移至 async `httpx` | `v1.4.0` 未把這件事當 release 主題，LLM 周邊資料工具延遲仍有壓縮空間 | ❌ 未完成 |
| **RD-Agent 因子自動化** | 週末因子進化 | `RdAgentBridge` 仍屬後續方向，不是 `v1.4.0` 完成項 | ❌ 未完成 |
| **Dashboard / 多帳號管理** | 原早期 roadmap 中的功能性擴張 | 在 `v1.4.0` 之後仍應維持低優先序，因為當前瓶頸仍是交易品質與風控品質，不是展示層 | ⏸️ 持續延後 |

### 2.3 對照後的結論

如果只看 `v1.3.5` 路線圖，會以為 `v1.4.0` 是：

- 一個學習功能版本
- 一個新聞觸發版本
- 一個 WebSocket 試驗版本

但從 `v1.4.0` 報告反推，實際上它已經是：

- 一個 **Observe 層重構版本**
- 一個 **Act -> Orient 回流閉環版本**
- 一個 **三倉庫控制面拼裝完成版本**

所以 `v1.4.0` 之後的 roadmap，不應再把焦點放在「要不要做 WebSocket」或「要不要做 learning loop」，而是要回答：

- 如何讓這個新控制面更穩、更可觀測、更可解釋？
- 如何把 lessons、事件、風險邊界繼續往下游與橫向擴散？
- 如何把進場品質、出場品質、組合風險一起拉上來？

---

## 3. v1.4.0 之後的核心缺口

`v1.4.0` 已經把系統推進到可持續運行的 OODA 閉環，但從 `v1.4.0` 報告的已知限制與後續方向來看，下一階段至少還有六個明確缺口。

### 3.1 Lessons 仍主要集中在 trader prompt

目前 `retrieved_trade_lessons` 的主注入點是 trader prompt。這已經比 `v1.3.5` 時期強很多，但仍代表：

- bull / bear / judge 等其他 agent node 仍可能沒有共享同等程度的歷史教訓
- 學習效果仍偏單點增益，而不是整張 decision graph 的共同記憶

**結論**：下一步不是再做「有沒有 lessons」，而是做「lessons 分佈到哪些節點」。

### 3.2 WebSocket-first 目前主要覆蓋 market data path

`v1.4.0` 已把 FX 市場資料升級為 WebSocket-first，但：

- broker execution 仍是 REST
- equity / account state 仍不是真正串流
- REST fallback 的即時性，仍不等於 live tick feed

**結論**：下一步要做的是 runtime hardening、status telemetry、fallback quality，而不是宣稱整個系統都已 fully streaming。

### 3.3 MarketDataHub 缺少更清楚的 ops observability

`v1.4.0` 已有 `websocket_cache`、`warmup_cache`、`rest_fallback` 三層資料來源，但後續仍需要更清楚地知道：

- 每次 decision / tactical / volatility 讀到的是哪一層資料
- feed stale 是偶發還是系統性問題
- warmup 視窗、hub lookback 與實際 downstream 需求是否一致

**結論**：若沒有把 source telemetry 與狀態輸出做好，WebSocket-first 只會變成「理論上更快」，而不是「營運上更透明」。

### 3.4 Exit quality 仍落後於 entry quality

`v1.4.0` 大幅強化了 Observe 與 Learn，但出場管理仍不是本版主角：

- 動態 SL/TP 未落地
- 波動 regime 變化對 stop/target 的調整仍有限
- 緊急事件下的分級反應仍不夠明確

**結論**：若下一版不補 exit management，系統會出現「進場比以前聰明，但出場仍偏保守靜態」的結構失衡。

### 3.5 缺少 portfolio-level risk view

當前控制面主要是單 symbol 的 Observe -> Decide -> Act 閉環，但真實 production 風險往往不是單筆造成，而是多筆倉位共振：

- USD 暴露集中
- 高相關幣對同向持倉
- 同一事件下多筆倉位同時承受跳空風險

**結論**：相關性檢測與組合曝險限制，應該在 learning loop 之後成為下一個高優先交易品質項目。

### 3.6 Scanner 與 LLM 工具鏈仍有延遲壓縮空間

`v1.4.0` 已用 WebSocket-first 把 Observe 層提速，但報告也清楚指出：

- `qlib_market_scanner` 仍是 strategic scanner，不是 streaming alpha engine
- `TradingAgents` 周邊工具鏈尚未全面 async 化
- `observe -> rescan -> decide` 的事件傳播延遲還可再縮短

**結論**：`v2.0.0` 的重點不應只是「更多功能」，而應是把現有控制面再往事件驅動與低延遲推進一步。

---

## 4. 修訂後版本路線

### 4.1 v1.4.2 — Runtime Hardening 與可觀測性補強

`v1.4.1` 已經先完成第一輪 reliability / observability hardening，因此原本 roadmap 裡規劃給「下一版 `v1.4.1`」的運行面深化工作，現在整體順延為 `v1.4.2`。

`v1.4.2` 的定位，不是再做大功能，而是把 `v1.4.1` 已建立的控制面補到 production 更可控。

#### 目標

- 把 WebSocket-first 變成可營運、可診斷、可降級的正式基線
- 把 event-driven rescan 的來源與品質觀測做清楚
- 把緊急風險反應補成更明確的 operational playbook

#### 核心工作

| 類別 | 工作項 |
|---|---|
| **Hub Telemetry** | 為 quote / bars / tactical / volatility 增加 source telemetry、freshness metrics、stale reason logging |
| **Rescan Provenance** | 區分 volatility trigger、news trigger、slot freed、schedule tick 等不同重掃來源 |
| **Warmup/Fallback 校準** | 對齊 `warmup_*_bars`、hub lookback、tactical/volatility 的實際消費視窗 |
| **News Trigger 調優** | 關鍵詞、冷卻、去重與告警節流調整，避免過度重掃 |
| **Emergency Response 補強** | 補上「異常事件 -> 立即 equity refresh -> 分級反應」的明確策略與日誌 |

#### 預期效果

- 能精確回答每次交易決策是吃到哪一層資料
- 能快速定位 feed stale、fallback 過多、event storm 等問題
- 能把 `v1.4.1` 從「第一輪 hardening 完成」推進到「運維成熟」

### 4.2 v1.5.0 — 決策品質與出場品質升級

當 `v1.4.2` 把運行面穩住後，下一個高 ROI 版本應聚焦在 **讓交易決策與出場策略一起變得更好**。

#### 目標

- 把 lessons 從單點 prompt 擴展到多節點決策圖
- 把 entry-side 強化延伸到 exit-side
- 加入最基本的組合風險感知

#### 核心工作

| 類別 | 工作項 |
|---|---|
| **Lesson Expansion** | 將 `retrieved_trade_lessons` 擴展到 bull / bear / judge 等更多 agent node |
| **Memory Unification** | 整理 `TradeJournal`、`MemoryJournal`、lesson memory 的角色，避免記憶系統分裂 |
| **Dynamic Exit** | ATR/regime-aware 動態 SL/TP、trailing stop 收緊、事件驅動 reprice |
| **Correlation Control** | 新增同幣別曝險上限、相關幣對同向持倉限制、portfolio guard |
| **Tactical Calibration** | 依幣對、session、波動 regime 校準 tactical gate 門檻 |

#### 預期效果

- 決策不只知道「過去犯過什麼錯」，還能在更多 agent 節點一致使用這些教訓
- 平均 MAE、平均 R、極端行情下的出場品質更有機會改善
- 從單筆風險控制，進一步升級到組合層風險控制

### 4.3 v2.0.0 — 事件驅動深化與 async 化

`v2.0.0` 的主題，不是另起爐灶，而是把 `v1.4.x` 已建立的控制面再往低延遲與高吞吐推進。

#### 目標

- 縮短 `observe -> rescan -> decide` 傳播時間
- 減少 `TradingAgents` 周邊資料工具的阻塞成本
- 讓 scanner 與研究流程更自然地接上事件驅動 runtime

#### 核心工作

| 類別 | 工作項 |
|---|---|
| **TradingAgents Async 化** | 將同步 HTTP 工具遷移到 `httpx.AsyncClient`，改善多來源資料抓取延遲 |
| **Event Propagation 優化** | 縮短 WebSocket 異常、market event、rescan、LLM 工作佇列間的傳播延遲 |
| **More WebSocket-aware Consumers** | 讓更多 downstream consumer 直接感知 hub state，而不是各自補抓資料 |
| **Scanner Runtime 優化** | 讓 strategic scanner 更自然支援 event-driven repeated invocation |
| **RD-Agent 週末自動化** | 將 weekend factor discovery 正式接入例行研究流程 |

#### 預期效果

- 端到端反應速度繼續下降
- LLM 決策等待時間與外部資料阻塞時間進一步縮短
- 研究、掃描、決策、執行之間的切換成本更低

### 4.4 v2.5.0 — Ops Dashboard 與可視化控制台

Dashboard 仍然值得做，但它的正確時機是在控制面成熟之後，而不是之前。

#### 目標

- 把現有控制面狀態對操作者與維運者可視化
- 提升事故排查與日常運維效率

#### 核心工作

| 類別 | 工作項 |
|---|---|
| **Runtime Overview** | Portfolio overview、equity curve、active positions、intent lifecycle |
| **Market Data Health** | WebSocket 連線狀態、symbol freshness、fallback 比例、hub source distribution |
| **Memory Review** | trade lessons、recent reflection、historical pnl context 的可視化瀏覽 |
| **Ops View** | scanner signals、alerts、error trends、daily summary 集中查看 |

#### 判斷原則

Dashboard 的價值在於讓成熟系統更可控，而不是替不穩定系統做漂亮外殼。

### 4.5 v3.0.0 — 擴張到 $50k 與多帳號治理

`v3.0.0` 仍應維持在更後面，因為多帳號管理只有在單帳號控制面穩定盈利時才真正有意義。

#### 目標

- 完成 `$5k` challenge 的穩定通過與規則驗證
- 把風控、監控、告警、資料隔離提升到可支援更高資金量的等級

#### 核心工作

| 類別 | 工作項 |
|---|---|
| **Capital Scale-up** | 驗證 `$50k` 帳戶下的風控參數、曝險密度與 execution 行為 |
| **Multi-account Isolation** | 帳戶級配置隔離、記憶隔離、日誌隔離、風控隔離 |
| **Centralized Alerting** | 統一 Telegram / ops alerts，支援多帳戶狀態總覽 |
| **Production Governance** | 更完整的事故回顧、日報、週報、回測對照、版本 gate |

#### 判斷原則

多帳號與大資金不是功能伸展，而是治理能力伸展；在此之前，必須先把單帳號控制面做到穩、快、可學習、可解釋。

---

## 5. 統一路線圖

### 5.1 修訂後版本時間線

```text
2026-03-11
  │
  ├─ v1.4.0 ✅ ─── WebSocket-first OODA Learning Loop
  │                · WebSocket-first market data hub
  │                · structured reflection payload
  │                · persistent lesson retrieval
  │                · event-driven scanner/news/volatility integration
  │
  ├─ v1.4.1 ✅ ─── Production Reliability / Observability Hardening
  │                · memory identity guard
  │                · shared version source
  │                · Telegram polling metrics
  │                · diagnostics bundle hardening
  │                · websocket live probe + same-tail REST suppression
  │
  ├─ v1.4.2 ────── Runtime Hardening + Observability            [1–2 週]
  │                · hub telemetry / source attribution
  │                · rescan provenance / event quality
  │                · warmup/fallback 校準
  │                · emergency response 補強
  │
  ├─ v1.5.0 ────── Decision Quality + Exit Quality             [2–4 週]
  │                · lessons 擴散到更多 agent nodes
  │                · dynamic SL/TP / trailing logic
  │                · correlation / portfolio guard
  │                · tactical gate 校準
  │
  ├─ v2.0.0 ────── Event-Driven Scale + Async Tools            [4–6 週]
  │                · TradingAgents async 化
  │                · observe -> decide 延遲再壓縮
  │                · scanner/runtime 事件驅動深化
  │                · RD-Agent 週末自動化
  │
  ├─ v2.5.0 ────── Ops Dashboard                               [2–3 週]
  │                · runtime / memory / market data / alerts 可視化
  │
  └─ v3.0.0 ────── $50k Production Expansion                   [持續]
                   · $5k challenge 穩定通過
                   · 多帳號治理
                   · 更高資金量風控與監控
```

### 5.2 優先級矩陣

| 改進項目 | 交易品質影響 | 實施難度 | ROI | 建議版本 |
|---|:---:|:---:|:---:|---|
| Hub telemetry / source attribution | 🔴 高 | 🟡 中 | ⭐⭐⭐ | `v1.4.2` |
| Emergency response 補強 | 🔴 高 | 🟡 中 | ⭐⭐⭐ | `v1.4.2` |
| Lessons 擴散到更多 agent nodes | 🔴 高 | 🟡 中 | ⭐⭐⭐ | `v1.5.0` |
| Dynamic SL/TP / trailing logic | 🔴 高 | 🔴 高 | ⭐⭐⭐ | `v1.5.0` |
| Correlation / portfolio guard | 🔴 高 | 🟡 中 | ⭐⭐⭐ | `v1.5.0` |
| Tactical gate 校準 | 🟡 中 | 🟡 中 | ⭐⭐ | `v1.5.0` |
| TradingAgents async 化 | 🟡 中 | 🟡 中 | ⭐⭐ | `v2.0.0` |
| RD-Agent 週末自動化 | 🟡 中 | 🔴 高 | ⭐ | `v2.0.0` |
| Ops Dashboard | 🟢 低 | 🟡 中 | ⭐ | `v2.5.0` |
| 多帳號管理 | 🟢 低 | 🔴 高 | ⭐ | `v3.0.0` |

### 5.3 里程碑定義

| 里程碑 | 達成條件 | 目標版本 |
|---|---|---|
| **M1: 控制面可觀測** | 能清楚回答每次 quote/bar/decision 的資料來源、freshness 與 rescan 原因 | `v1.4.2` |
| **M2: 決策與出場一起升級** | lessons 擴展到多 agent node，dynamic exit 與 correlation guard 成為正式基線 | `v1.5.0` |
| **M3: 事件驅動深化** | `observe -> rescan -> decide` 延遲再下降，TradingAgents 工具鏈 async 化 | `v2.0.0` |
| **M4: 系統運維可視化** | runtime、memory、market data health、alerts 可集中查看 | `v2.5.0` |
| **M5: 更高資金量治理** | 單帳號穩定通過後，支援 `$50k` 與多帳號的配置、監控、風控隔離 | `v3.0.0` |

---

## 結語

`v1.3.5` 時期的 roadmap，核心是在補功能缺口；`v1.4.0` 之後的 roadmap，核心應改成**深化控制面**。

現在系統已經不再只是：

- 有 scanner
- 有 LLM
- 有 execution

而是已經具備：

- WebSocket-first 的 Observe 層
- 可回流 lessons 的 Learn 層
- 事件驅動的 steady-state runtime
- 明確的 degraded / failure isolation 哲學

所以下一階段最重要的不是再多堆幾個模組，而是把這套 OODA trading machine 變得：

- 更可觀測
- 更一致
- 更會出場
- 更懂組合風險
- 更適合往更高資金量擴張

這才是 `v1.4.0` 之後，真正合理的發展方向。
