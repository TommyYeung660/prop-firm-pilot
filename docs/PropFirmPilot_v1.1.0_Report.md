# PropFirmPilot v1.1.0 — 變更總結報告（增量版）

> **報告日期**: 2026-02-27  
> **版本**: v1.1.0（增量更新）  
> **基準版本**: v1.0.0（`docs/PropFirmPilot_v1.0_Report.md`）  
> **聚焦範圍**: 3.2 記憶與反饋迴路、3.3 日誌與記憶驅動優化、3.4 LLM 優化功能落地

---

## 目錄

1. 變更摘要
2. 對應 v1.0 3.2/3.3/3.4 的落地結果
3. 核心改動詳解
4. 對整體系統的影響分析
5. 兼容性、風險與非目標
6. 驗證與測試結果
7. 後續建議

---

## 1. 變更摘要

- **MemoryJournal 已補齊**：Scheduler 每次 LLM 決策（含 HOLD）都寫入 `MEMORY/{YYYY-MM-DD}.md`，平倉後補記結果。
- **OptimizationState 落地**：每日 summary 觸發 `optimization_state.json` 更新，成為動態門檻與回饋的唯一運行態來源。
- **雙層動態門檻**：LLM 前（pre-filter）與 LLM 後（post-filter）同時生效，降低低信心訊號進入執行流程。
- **交易事件可觀測性提升**：TradeJournal 追加 pipeline 事件（Intent/LLM/CANCEL/OPEN/CLOSE/REJECT/FAIL）。
- **A/B 測試支援完成基礎設施**：模型分流與統計結構已就位（`glm-4.7` vs `gpt-5.2`），可由 AgentBridge 進一步接線使用。

---

## 2. 對應 v1.0 3.2/3.3/3.4 的落地結果

### 3.2 記憶與反饋迴路（已落地）

- **現狀問題**：v1.0 報告中規劃 MemoryJournal，但生產環境僅存在 `data/trade_journal.jsonl`，`MEMORY/` 目錄未產生任何文件。
- **v1.1.0 修復**：
  - Scheduler 中新增 MemoryJournal 入口，**每次 LLM 決策都記錄**（含 HOLD）。
  - 平倉後追加 `Trade Result`，將 PnL 與 exit reason 補入同日記錄。
- **產物**：
  - `MEMORY/{YYYY-MM-DD}.md`（新增、每日滾動）
  - 內容包含決策上下文、信心度、風險報告、結果補記

### 3.3 基於 Log 與記憶的優化（已落地核心管線）

- **新增 OptimizationEngine**：聚合 DecisionStore + TradeJournal，產出 `optimization_state.json`。
- **統計視窗**：
  - 勝率視窗 14 天
  - PnL 回饋 7 天
- **輸出內容**：
  - `global_thresholds` + `symbol_thresholds`
  - `feedback_pnl`（symbol → 累積 PnL）
  - A/B 測試統計欄位
- **觸發方式**：Daily Summary 時段自動刷新（非手動）。

### 3.4 LLM 優化功能（已落地）

- **動態信心閾值**：
  - 分段規則：勝率 <45% → 提高門檻；>55% → 降低門檻；其餘維持。
  - 同時作用於 **LLM 前**與**LLM 後**。
- **歷史盈虧回饋**：
  - 以 `DecisionStore` 為主，TradeJournal 為輔（7 天）。
  - 結果寫入 optimization_state，供 LLM context 或策略調整使用。
- **A/B 測試**：
  - 支援 `volcengine/glm-4.7` vs `gpt-5.2`。
  - 分流計算與統計結構已完成，尚需在 AgentBridge 端接線落地。

---

## 3. 核心改動詳解

### 3.1 新增 / 修改模組

- `src/optimize/optimization_state.py`：狀態模型與 IO
- `src/optimize/optimization_engine.py`：勝率/PnL 聚合與寫檔
- `src/optimize/trade_stats.py`：統計聚合
- `src/optimize/thresholds.py`：門檻規則計算
- `src/optimize/ab_testing.py`：A/B 分流與統計
- `src/scheduler/scheduler.py`：
  - Daily summary 觸發 optimization refresh
  - LLM pre/post 門檻過濾
  - MemoryJournal / TradeJournal 事件紀錄
- `src/monitor/memory_journal.py`：新增 `log_decision` 與 `append_trade_result`
- `src/monitor/trade_journal.py`：事件時間戳補齊
- `src/execution/engine.py`：TradeJournal 開/拒/失敗記錄
- `src/config.py`：新增 `OptimizationConfig`

### 3.2 重要流程變更（決策與資料流）

- **LLM 前門檻**：低信心 scanner 訊號會直接被取消，減少不必要 LLM 成本。
- **LLM 後門檻**：即便 LLM BUY/SELL，若 blended 信心不足仍會被取消。
- **平倉後回饋**：Trade result 自動補記至 MemoryJournal，形成閉環。
- **每日自動優化**：optimization_state.json 變成「唯一運行態優化來源」。

---

## 4. 對整體系統的影響分析

### 4.1 決策品質

- **低信心訊號被雙層過濾**，避免低質量交易進入執行層。
- **門檻可動態回應近期勝率變化**，提升適應性。

### 4.2 資料可觀測性

- MemoryJournal 由「規劃功能」變為「實際產物」，可直接追溯每筆決策與結果。
- TradeJournal 新增 pipeline 事件，形成完整生命週期視圖。

### 4.3 系統可靠性與風控

- compliance 模組未改動，安全邊界不變。
- optimization_state 更新失敗也不阻斷交易流程（以 warning 記錄）。

### 4.4 成本與運營

- pre-filter 會減少 LLM 呼叫次數，降低 API 成本。
- daily state refresh 的 IO 開銷可控（單一 JSON 輸出）。

---

## 5. 兼容性、風險與非目標

- **兼容性**：
  - 舊版流程不受破壞，新增能力為增量行為。
  - optimization_state.json 可不存在，系統會用預設門檻運行。
- **風險**：
  - 目前 A/B 分流尚未接到 AgentBridge（僅基礎設施），需後續整合。
- **非目標**：
  - 未修改 `src/compliance/` 安全規則。
  - 未回寫 YAML（狀態僅寫 JSON）。

---

## 6. 驗證與測試結果

- `uv run ruff check src/ tests/` ✅
- `uv run pytest` ✅（584 passed, 70 warnings）
- 警告皆為 `tests/test_alpha158_evaluator.py` 的 pandas fragmentation warning（性能提示，非功能錯誤）。

---

## 7. 後續建議

- **A/B 模型分流落地**：將 `choose_model()` 接入 AgentBridge，寫入 model_id 與延遲統計。
- **把 optimization_state 回饋注入 LLM prompt**：讓 TradingAgents 能真正讀取歷史 PnL 與門檻。
- **MemoryJournal 與 TradeJournal 的統一查詢工具**：為 Dashboard 或分析工具提供統一接口。

---

> **v1.1.0 結論**：
> 本次更新完成 v1.0 報告中 3.2/3.3/3.4 的核心落地，建立完整的「記憶 → 反饋 → 動態門檻」閉環，並顯著提升系統可觀測性與決策質量控制。


---

## 附錄 A — 檔案/模組對照表

### A.1 新增模組

| 模組 | 檔案 | 說明 |
|------|------|------|
| OptimizationState | `src/optimize/optimization_state.py` | 優化狀態 schema + IO |
| OptimizationEngine | `src/optimize/optimization_engine.py` | 勝率/PNL 聚合 → state 輸出 |
| Trade Stats | `src/optimize/trade_stats.py` | 統計勝率與 PnL 回饋 |
| Thresholds | `src/optimize/thresholds.py` | 動態信心閾值計算 |
| A/B Testing | `src/optimize/ab_testing.py` | 模型分流與統計 |

### A.2 修改模組

| 模組 | 檔案 | 改動 |
|------|------|------|
| Scheduler | `src/scheduler/scheduler.py` | pre/post 門檻、daily refresh、Memory/TradeJournal event |
| Main | `src/main.py` | scheduler 模式注入 Optimization/Mem/TradeJournal |
| MemoryJournal | `src/monitor/memory_journal.py` | 新增 log_decision / append_trade_result |
| TradeJournal | `src/monitor/trade_journal.py` | event 自動補 timestamp |
| ExecutionEngine | `src/execution/engine.py` | 開/拒/失敗事件紀錄 |
| Config | `src/config.py` | 新增 OptimizationConfig |

---

## 附錄 B — 變更清單（依功能區塊）

### B.1 記憶與反饋迴路
- Scheduler LLM 決策記錄 → `MEMORY/{date}.md`
- 平倉後補記結果（PnL + exit reason）

### B.2 優化狀態與門檻
- 14 天勝率、7 天 PnL 回饋
- `optimization_state.json` 每日自動更新
- 動態門檻同時作用於 LLM 前/後

### B.3 事件可觀測性
- TradeJournal event：Intent Created / LLM Decision / Cancel / Close / Reject / Fail

### B.4 A/B 測試基礎
- 模型分流與統計結構就位（glm-4.7 vs gpt-5.2）
