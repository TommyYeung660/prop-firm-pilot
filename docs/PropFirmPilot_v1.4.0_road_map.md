# PropFirmPilot v1.4.0 之後 — 發展路線圖

> **更新日期**: 2026-03-13
>
> **當前版本**: v1.4.6b（Tactical Freshness Recovery + Prod Bundle Dropbox Sync）
>
> **涵蓋範圍**: `prop-firm-pilot` · `qlib_market_scanner` · `TradingAgents` 三倉庫協作
>
> **帳戶階段**: E8 Markets One-Phase $5,000 Challenge

---

## 目錄

1. [這份 v1.4.0 路線圖現在要回答什麼](#1-這份-v140-路線圖現在要回答什麼)
2. [v1.3.5 路線圖 vs v1.4.6b 實際落地](#2-v135-路線圖-vs-v146b-實際落地)
3. [v1.4.6b 之後的核心缺口](#3-v146b-之後的核心缺口)
4. [修訂後版本路線](#4-修訂後版本路線)
5. [統一路線圖](#5-統一路線圖)

---

## 1. 這份 v1.4.0 路線圖現在要回答什麼

`docs/PropFirmPilot_v1.3.5_road_map.md` 當時把 `v1.4.0` 規劃成三條主線：

- 學習迴圈閉合
- 新聞事件觸發
- WebSocket PoC

但到了現在，實際情況已經不是單看 `v1.4.0` 報告就足夠，而是要把 `v1.4.0` 到 `v1.4.6b` 的既成事實一起納入：

- `Observe` 已經從輪詢式資料取得，升級成 **WebSocket-first + REST fallback + live degradation handling**
- `Orient/Decide` 不只吃當下 signal，還能帶入 `historical_pnl_context`、`market_event_context`、`retrieved_trade_lessons`
- `Act` 不只會平倉，現在程式碼基線內也已有 tactical exit state machine、scheduler wiring、execution metadata 回寫
- production diagnostics 不再只靠手工收集 log，`v1.4.6b` 已把 run-specific log、bundle manifest、Dropbox bundle sync 帶進運維流程

因此，這份文件的目的不是重複 release note，而是回答三件事：

1. `v1.3.5` 時代規劃的哪些東西，到了 `v1.4.6b` 已經真正落地？
2. 原本 roadmap 裡打算放在 `v1.4.2` / `v1.4.6` 的哪些工作，其實已被拆散吸收進 `v1.4.1`、現行 tactical baseline、`v1.4.5a`、`v1.4.6b`？
3. 以 `v1.4.6b` 為新基線之後，真正還值得排進後續版本的缺口是什麼？

### 1.1 路線圖的重寫原則

從現在開始，版本優先序應遵守以下原則：

- **先承認既成事實，再排後續版本**。已經進主線或已經在 hotfix 裡落地的能力，不再寫成未來式。
- **先把 tactical control plane 做到可回放、可診斷、可營運，再擴外圍功能**。entry/exit 與 runtime provenance 的可信度，優先於 UI、多帳號、展示層。
- **原本屬於 `v1.4.2` 的 hardening scope，若已被拆入多個版本，就視為 tail work，而不是重新宣告一個不存在的獨立版本。**
- **WebSocket-first 是平台基線，不再只是 PoC**。後續工作重點是 source attribution、fallback quality、事件來源追蹤，不是重新論證 WebSocket 值不值得做。
- **合規與 safety-critical 規則不作為換速度的代價**。drawdown、best day rule、market-hours、equity monitor 仍是不可退讓底線。

---

## 2. v1.3.5 路線圖 vs v1.4.6b 實際落地

### 2.1 已完成或超額完成的項目

| v1.3.5 路線圖規劃 | 當時預期 | 截至 v1.4.6b 的實際落地 | 判定 |
|---|---|---|---|
| **學習迴圈閉合** | 啟用 `reflect_and_remember()`、注入歷史盈虧 | `v1.3.9c` + `v1.4.0` 已完成 structured reflection payload、persistent lesson retrieval，並把 `historical_pnl_context` / `retrieved_trade_lessons` 接進決策流 | ✅ 完成，而且比原規劃更完整 |
| **歷史盈虧注入 LLM 提示詞** | 把近 7 日交易表現注入 prompt | `historical_pnl_context` 已成 graph state 正式欄位，並與 reflection / lesson retrieval 一起進入 decision-time prompt | ✅ 完成 |
| **新聞事件觸發** | 新增 `NewsEventTrigger` 觸發重掃 | `news_event_trigger.py`、`market_event_context`、`rescan_event` 已接入 scheduler steady-state runtime | ✅ 完成 |
| **WebSocket PoC** | 建立 EODHD WebSocket client 原型 | 實際落地為 `fx_websocket_client.py` + `fx_tick_aggregator.py` + `market_data_hub.py`，`v1.4.1` 再補 live probe 與 degraded fallback suppression | ✅ 超額完成 |
| **戰術層與即時資料互補** | 讓 WebSocket 未來可支援 tactical layer | tactical validator 已改為 hub-first data path；`WAIT -> tactical_pending -> retry`、stale quote guard、freshness fallback 也一路補到 `v1.4.6b` | ✅ 完成整合，且已進入 live hardening 階段 |
| **24x7 控制面方向** | 從全輪詢走向事件驅動 | `Scheduler` 現已在 market data、volatility、news、scanner、execution、reflection 間形成 steady-state runtime，並搭配 run-specific log 與 prod bundle workflow | ✅ 完成骨架且進入運維化 |

### 2.2 部分完成、改道完成、或仍未完成的項目

| 項目 | v1.3.5 路線圖預期 | 截至 v1.4.6b 現況 | 判定 |
|---|---|---|---|
| **緊急平倉增強** | 波動觸發後立即刷新權益，並有 80%/90% 分級處置 | `EquityMonitor.check_once()` 已支援 alert / reduce exposure / emergency close，但異常事件、權益刷新、tactical exit、postmortem provenance 仍未完全統一成單一 playbook | ⚠️ 部分完成 |
| **MemoryJournal 語意升級** | `prop-firm-pilot` 內部記憶系統向量化 | 實際走的是 `TradingAgents` persistent lesson memory 路線；`MemoryJournal` 本身尚未與 `TradeJournal` / lesson memory 完整統一 | ⚠️ 改道完成一部分 |
| **動態 SL/TP** | ATR-based 動態出場 | 現行程式碼已具備 `tactical_exit_rules.py`、`tactical_exit_manager.py`、scheduler wiring，以及 breakeven / trailing / reprice / partial close 基線；但尚未提升為完整 dynamic exit baseline，也未與組合風險聯動 | ⚠️ 部分完成 |
| **相關性檢測 / 組合風險** | 避免多倉位共振風險 | 仍以單 symbol 決策與單筆倉位控制為主，缺少 portfolio-level correlation / exposure guard | ❌ 未完成 |
| **TradingAgents async 化** | `requests` 遷移至 async `httpx` | 尚未成為正式 release 主題，LLM 周邊資料抓取與事件傳播延遲仍有壓縮空間 | ❌ 未完成 |
| **RD-Agent 因子自動化** | 週末因子進化 | `RdAgentBridge` 仍屬後續方向，不是現行基線能力 | ❌ 未完成 |
| **Dashboard / 多帳號管理** | 原早期 roadmap 中的功能性擴張 | 在 `v1.4.6b` 後仍應維持低優先序，因為當前瓶頸依然是 tactical control quality、風控品質與運維透明度 | ⏸️ 持續延後 |

### 2.3 對照後的結論

如果只看 `v1.3.5` 路線圖，會以為 `v1.4.0` 之後要先補：

- learning loop
- news trigger
- WebSocket 試驗
- tactical entry / exit

但從當前程式碼與 `v1.4.1`、`v1.4.5a`、`v1.4.6b` 的實際落地來看，真實進展其實是：

- `v1.4.0` 完成 **Observe + Learn 閉環**
- `v1.4.1` 開始 **runtime hardening / degraded fallback / metrics snapshot**
- 現行主線已具備 **tactical exit baseline**
- `v1.4.5a` / `v1.4.6b` 持續修 **same-direction re-entry、stale quote、freshness fallback、prod diagnostics workflow**

所以，接下來的 roadmap 不應再把：

- `v1.4.2` 寫成「當前版本」
- `v1.4.6` 寫成「尚未開始的 tactical entry 修復」
- `v1.4.7` 寫成「從零開始的 tactical exit 功能」

真正合理的重寫方式，是把 `v1.4.6b` 當成新基線，並把原定後續路線順延成：

- 先完成 tactical control plane 的可回放與 provenance
- 再做 broader decision / risk governance
- 最後才往 async scale、dashboard、多帳號延伸

---

## 3. v1.4.6b 之後的核心缺口

`v1.4.6b` 已把系統推進到「可運行、可 hotfix、可打包 postmortem」的階段，但以現行基線來看，下一階段仍有六個明確缺口。

### 3.1 Lessons 仍主要集中在 trader prompt

目前 `retrieved_trade_lessons` 的主要注入點仍是 trader prompt。這代表：

- bull / bear / judge 等其他 agent node 未必共享相同程度的歷史教訓
- 學習效果仍偏向單點增益，而不是整張 decision graph 的共同記憶

**結論**：下一步不是再做「有沒有 lessons」，而是做「lessons 分佈到哪些節點，以及如何與 journal/memory 邊界統一」。  

### 3.2 Market-data provenance 與 fallback quality 仍不夠完整

`v1.4.1` 到 `v1.4.6b` 已經補上：

- same-tail REST refresh suppression
- live websocket probe
- broker quote freshness fallback
- run-specific log + bundle manifest + Dropbox sync

但實際上仍缺少：

- 每次 decision / tactical / volatility 使用的是哪一層資料的統一輸出
- rescan 是由 volatility、news、slot freed、schedule tick 還是 manual recovery 觸發的明確 provenance
- stale feed 是偶發、系統性、還是 provider lag 的可量化判讀

**結論**：WebSocket-first 現在是可用基線，但還不是可完整解釋的營運基線。  

### 3.3 Tactical entry 已脫離最危險區，但還沒完成校準

`v1.4.6b` 已經處理：

- `data_freshness` 在 hub 只有 bars 時回退到 MatchTrader quote
- degraded / mixed-source 情境下不再因 freshness timestamp 缺失而卡死
- `WAIT` 生命周期與 `timed_out` 狀態較之前更一致

仍待補齊的部分是：

- per-symbol / session / regime threshold calibration
- tactical reason code 與 score breakdown 的統一
- `WAIT` / `DEGRADE` / `PASS` 在 postmortem 中的可追溯性

**結論**：entry-side 工作從現在開始應以 calibration 與 diagnostics 為主，而不是再做救火式 correctness fix。  

### 3.4 Tactical exit 已有基線，但 operational consistency 仍不足

現行主線內已經有：

- `tactical_exit_rules.py`
- `tactical_exit_manager.py`
- scheduler tactical exit cycle wiring
- execution metadata persistence

但仍然需要更強的：

- action replay 與 structured diagnostics
- duplicate close / partial close / read-back verification 的一致性保證
- tactical exit 與 emergency close、reduce exposure、LLM exception path 的邊界釐清

**結論**：後續版本的重點不該是「新增 tactical exit」，而是把既有 tactical exit 變成真正可營運、可審計的控制面。  

### 3.5 仍缺少 portfolio-level risk view

當前控制面依然以單 symbol、單筆 intent 為中心，但 production 風險常來自多筆倉位共振：

- USD 暴露集中
- 高相關幣對同向持倉
- 同一宏觀事件下多筆倉位一起承受跳空風險

**結論**：相關性檢測與組合曝險上限，應在 tactical control plane 穩定之後，成為下一個高優先交易品質項目。  

### 3.6 Scanner 與 LLM 工具鏈仍有延遲壓縮空間

現在的 Observe 層已經比 `v1.3.5` 快得多，但整體端到端延遲仍受限於：

- `qlib_market_scanner` 本質上仍是 strategic scanner，不是 streaming alpha engine
- `TradingAgents` 周邊資料工具尚未全面 async 化
- `observe -> rescan -> decide` 之間還有事件佇列與工具等待成本

**結論**：`v2.0.0` 的主題仍應是事件驅動深化與 async 化，而不是另起爐灶。  

---

## 4. 修訂後版本路線

### 4.1 截至 v1.4.6b 已落地的控制面補強

原 roadmap 裡規劃成 `v1.4.2` 與 `v1.4.6` 的一部分工作，實際上已被拆散吸收進多個版本與現行程式碼基線，因此不應再作為未來待辦重複列出。

| 版本 / 基線 | 已落地內容 | 對後續 roadmap 的意義 |
|---|---|---|
| **`v1.4.1`** | shared version source、Telegram polling metrics、degraded REST fallback suppression、`METRICS_SNAPSHOT` feed status、live websocket probe、diagnostics bundle hardening | 原本屬於 `v1.4.2` 的第一輪 hardening，已不再是未來工作 |
| **現行 tactical exit 基線（源自 `v1.4.5` workstream）** | `tactical_exit_rules.py`、`tactical_exit_manager.py`、scheduler wiring、execution metadata persistence、breakeven / trailing / reprice / partial close baseline | `v1.4.7` 的重點不再是「從零做 exit」，而是 hardening 既有 exit control plane |
| **`v1.4.5a`** | same-direction re-entry fix、stale REST synthetic quote guard | tactical entry correctness 已進入 live issue 修補階段，而非概念驗證階段 |
| **`v1.4.6b`** | broker quote freshness fallback、degraded tactical freshness recovery、run-specific log、bundle manifest、Dropbox diagnostics sync、7-pair websocket live probe | 後續版本應把重點放在 provenance、calibration、action integrity，而不是繼續修補最基礎的 freshness 死鎖 |

**結論**：從現在開始，正式的未來路線應從 `v1.4.7` 往後排，而不是把 `v1.4.2` 或 `v1.4.6` 當成尚未開始的空白版本。  

### 4.2 v1.4.7 — Tactical Control Hardening 與 Provenance 補完

這一版的定位是：**把現有 tactical entry / exit 與 runtime 觀測鏈補到可回放、可診斷、可營運。**

#### 目標

- 完成 source telemetry、rescan provenance、stale reason 的統一輸出
- 把 tactical entry 的剩餘 calibration / diagnostics tail work 收尾
- 把 tactical exit 的 action integrity、journal 一致性與 structured logging 補齊
- 把異常事件、equity refresh、reduce exposure、emergency close 的運維 playbook 串起來

#### 核心工作

| 類別 | 工作項 |
|---|---|
| **Source Provenance** | 為 quote / bars / tactical / decision / volatility 補齊資料來源、freshness、fallback reason 與 stale attribution |
| **Rescan Provenance** | 區分 volatility trigger、news trigger、slot freed、schedule tick、manual recovery 等不同重掃來源 |
| **Entry Tail Calibration** | 校準 per-symbol / session / regime tactical thresholds，整理 `WAIT` / `DEGRADE` / `PASS` 的 diagnostics 與 score breakdown |
| **Exit Action Integrity** | 強化 duplicate close 防護、partial close / read-back verification、一致的 trade journal / close reason / execution metadata 回寫 |
| **Emergency Playbook** | 對齊異常事件 -> equity refresh -> reduce exposure / tactical exit / emergency close 的邊界與日誌 |

#### 預期效果

- 能清楚回答每次交易「為何可進、為何等待、為何出場、用了哪一層資料」
- tactical entry / exit 從「能工作」進一步提升到「能可靠回放與 postmortem」
- 原本殘留的 `v1.4.2` observability tail work，會在這一版完成，而不再獨立成版本

### 4.3 v1.4.8 — 延後的決策品質與風險治理升級

原本較廣泛的 `v1.4.x` decision / risk scope，不應再插在 tactical hotfix 之前。以 `v1.4.6b` 為基線後，這一批工作整體順延到 `v1.4.8`。

#### 目標

- 把 lessons 從 trader prompt 擴展到多節點決策圖
- 整理 `TradeJournal`、`MemoryJournal`、lesson memory 的系統邊界
- 把現有 tactical exit baseline 升級成更完整的 dynamic exit baseline
- 把單筆交易控制進一步升級到組合層風險控制

#### 核心工作

| 類別 | 工作項 |
|---|---|
| **Lesson Expansion** | 將 `retrieved_trade_lessons` 擴展到 bull / bear / judge 等更多 agent node |
| **Memory Unification** | 整理 `TradeJournal`、`MemoryJournal`、lesson memory 的角色，避免記憶系統分裂 |
| **Dynamic Exit Baseline** | 將 ATR/regime-aware 動態 SL/TP、trailing、事件驅動 reprice 提升為正式策略基線 |
| **Correlation Control** | 新增同幣別曝險上限、相關幣對同向持倉限制、portfolio guard |

#### 預期效果

- 決策不只知道「過去犯過什麼錯」，也能在更多 agent 節點一致使用這些教訓
- 出場邏輯從 tactical baseline 進一步升級為更完整的 dynamic exit baseline
- 從單筆風險控制，進一步升級到組合層風險控制

### 4.4 v2.0.0 — 事件驅動深化與 async 化

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

### 4.5 v2.5.0 — Ops Dashboard 與可視化控制台

Dashboard 仍然值得做，但它的正確時機是在 tactical control plane 與 decision provenance 成熟之後，而不是之前。

#### 目標

- 把現有控制面狀態對操作者與維運者可視化
- 提升事故排查與日常運維效率

#### 核心工作

| 類別 | 工作項 |
|---|---|
| **Runtime Overview** | portfolio overview、equity curve、active positions、intent lifecycle |
| **Market Data Health** | WebSocket 連線狀態、symbol freshness、fallback 比例、hub source distribution |
| **Memory Review** | trade lessons、recent reflection、historical pnl context 的可視化瀏覽 |
| **Ops View** | scanner signals、alerts、error trends、daily summary 集中查看 |

#### 判斷原則

Dashboard 的價值在於讓成熟系統更可控，而不是替不夠透明的系統做漂亮外殼。

### 4.6 v3.0.0 — 擴張到 $50k 與多帳號治理

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
  │                · shared version source
  │                · Telegram polling metrics
  │                · degraded REST suppression
  │                · websocket live probe
  │                · diagnostics bundle hardening
  │
  ├─ 現行 tactical exit 基線 ✅ ─── (`v1.4.5` workstream)
  │                · tactical exit rules + manager + scheduler wiring
  │                · breakeven / trailing / reprice / partial close baseline
  │                · execution metadata persistence
  │
  ├─ v1.4.5a ✅ ─── Tactical Re-entry And Stale Quote Guard
  │                · closed intents no longer block same-day re-entry
  │                · stale REST synthetic quote no longer leaks downstream
  │
  ├─ v1.4.6b ✅ ─── Tactical Freshness Recovery + Prod Diagnostics Workflow
  │                · MatchTrader quote fallback for tactical freshness
  │                · run-specific logs + bundle manifest
  │                · Dropbox bundle sync / unpack workflow
  │                · 7-pair websocket live probe confirmation
  │
  ├─ 原規劃 `v1.4.2` ↷ 已拆入以上版本
  │                · 不再作為獨立未來版本存在
  │
  ├─ v1.4.7 ────── Tactical Control Hardening + Provenance     [1–2 週]
  │                · source / rescan provenance
  │                · entry tail calibration
  │                · exit action integrity
  │                · emergency playbook 對齊
  │
  ├─ v1.4.8 ────── Deferred Decision / Risk Upgrade            [2–4 週]
  │                · lessons 擴散到更多 agent nodes
  │                · memory unification
  │                · dynamic exit baseline
  │                · correlation / portfolio guard
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
| Source telemetry / rescan provenance | 🔴 高 | 🟡 中 | ⭐⭐⭐ | `v1.4.7` |
| Tactical entry threshold calibration / diagnostics | 🔴 高 | 🟡 中 | ⭐⭐⭐ | `v1.4.7` |
| Tactical exit action integrity / replayability | 🔴 高 | 🟡 中 | ⭐⭐⭐ | `v1.4.7` |
| Emergency response playbook 對齊 | 🔴 高 | 🟡 中 | ⭐⭐⭐ | `v1.4.7` |
| Lessons 擴散到更多 agent nodes | 🔴 高 | 🟡 中 | ⭐⭐⭐ | `v1.4.8` |
| Memory unification | 🟡 中 | 🟡 中 | ⭐⭐ | `v1.4.8` |
| Dynamic SL/TP / trailing baseline 升級 | 🔴 高 | 🔴 高 | ⭐⭐⭐ | `v1.4.8` |
| Correlation / portfolio guard | 🔴 高 | 🟡 中 | ⭐⭐⭐ | `v1.4.8` |
| TradingAgents async 化 | 🟡 中 | 🟡 中 | ⭐⭐ | `v2.0.0` |
| RD-Agent 週末自動化 | 🟡 中 | 🔴 高 | ⭐ | `v2.0.0` |
| Ops Dashboard | 🟢 低 | 🟡 中 | ⭐ | `v2.5.0` |
| 多帳號管理 | 🟢 低 | 🔴 高 | ⭐ | `v3.0.0` |

### 5.3 里程碑定義

| 里程碑 | 達成條件 | 目標版本 |
|---|---|---|
| **M0: `v1.4.6b` 現行基線** | tactical freshness recovery、run-specific logs、bundle manifest、Dropbox diagnostics sync 已落地 | 已完成 |
| **M1: 戰術控制可回放** | 能清楚回答每次進場 / 等待 / 出場的資料來源、freshness、rescan 原因與 action 結果 | `v1.4.7` |
| **M2: 決策與風險正式升級** | lessons 擴展到多 agent node，dynamic exit baseline 與 correlation guard 成為正式基線 | `v1.4.8` |
| **M3: 事件驅動深化** | `observe -> rescan -> decide` 延遲再下降，TradingAgents 工具鏈 async 化 | `v2.0.0` |
| **M4: 系統運維可視化** | runtime、memory、market data health、alerts 可集中查看 | `v2.5.0` |
| **M5: 更高資金量治理** | 單帳號穩定通過後，支援 `$50k` 與多帳號的配置、監控、風控隔離 | `v3.0.0` |

---

## 結語

`v1.3.5` 時期的 roadmap，核心是在補功能缺口；以 `v1.4.6b` 為基線後，核心應改成**承認既有 tactical / runtime 基線，然後深化控制面**。

現在系統已經不再只是：

- 有 scanner
- 有 LLM
- 有 execution

而是已經具備：

- WebSocket-first 的 Observe 層
- 可回流 lessons 的 Learn 層
- 現行 tactical exit 基線
- degraded / fallback hotfix 經驗累積
- 可打包、可上傳、可回放的 production diagnostics workflow

所以下一階段最重要的，不是再把 tactical entry / exit 寫成未來藍圖，而是先把這套 OODA trading machine 變得：

- 更可回放
- 更可觀測
- 更一致
- 更懂組合風險
- 更適合往更高資金量擴張

這才是 `v1.4.6b` 之後，真正合理的發展方向。
