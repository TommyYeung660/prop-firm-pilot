# 記憶與反饋優化設計（PropFirmPilot）

日期：2026-02-26

## 背景
根據 `docs/PropFirmPilot_v1.0_Report.md` 的 3.2/3.3/3.4，本次補齊 MemoryJournal 生產寫入缺口，並實作自動化優化流程。

## 目標
- 每次 LLM 決策（含 HOLD、合規擋單）都寫入 `MEMORY/{YYYY-MM-DD}.md`，且平倉後補記結果。
- 每日自動產生 `data/optimization_state.json`，供動態門檻與策略優化使用。
- 動態信心閾值同時作用於「LLM 前」與「LLM 後」。
- 支援 7 天歷史盈虧回饋注入與 14 天勝率視窗。
- 支援 A/B 測試（`volcengine/glm-4.7` vs `gpt-5.2`）。

## 非目標
- 不修改任何 `src/compliance/` 安全規則。
- 不回寫 YAML 設定檔（所有優化狀態僅寫入 JSON state）。

## 架構與元件
- 新增 `OptimizationEngine`：彙總 `decisions.db` + `trade_journal.jsonl`，輸出 `optimization_state.json`。
- Scheduler：
  - LLM 前讀 state 進行信心門檻過濾（避免送入 LLM）。
  - LLM 後讀 state 進行決策降級（改為 HOLD）。
  - Daily summary 時段觸發優化更新。
- MemoryJournal：由 Scheduler 寫入決策記憶；平倉時補記結果。
- TradeJournal：補記 LLM 決策、開倉、平倉、拒單等事件。

## 資料流
1. Scanner → 建立 Intent。
2. LLM 前門檻 → 低信心訊號不送入 LLM。
3. LLM 決策 → MemoryJournal 寫入（含 HOLD）。
4. LLM 後門檻 → 低信心決策降級為 HOLD。
5. 開/平倉 → TradeJournal + DecisionStore 更新。
6. Daily summary → OptimizationEngine 更新 state。

## 動態信心閾值
- 14 天勝率視窗，分段階梯規則：
  - 勝率 < 45%：提高門檻
  - 45%–55%：維持
  - > 55%：降低
- 混合全局 + 品種微調：全局作基準、品種勝率做小幅修正。
- 同時套用：
  - `min_confidence`（low/medium/high）
  - `blended_confidence`（0–1）

## 歷史盈虧回饋注入
- 回看 7 天內 closed trades（以 `decisions.db` 為主，jsonl 補充）。
- 形成 `symbol -> pnl` 回饋摘要，注入 LLM context。

## A/B 測試
- 模型：`volcengine/glm-4.7` vs `gpt-5.2`。
- 依比例或輪替分流（在 LLM worker 決策時決定）。
- 結果寫入 state 的 A/B 統計區。

## 自適應風控（僅建議值）
- 依近 14 天績效產出 `risk_per_trade` 建議值寫入 state，不改 compliance。

## 狀態檔
- `data/optimization_state.json` 為唯一運行時優化狀態來源。
- 每日更新，需包含版本、時間戳、全局/品種門檻、A/B 統計、回饋摘要。

## 測試
- OptimizationEngine 統計與門檻計算。
- LLM 前/後門檻判定。
- MemoryJournal 寫入（含 HOLD 與補記結果）。

## 驗收標準
- `MEMORY/{YYYY-MM-DD}.md` 每日有產出（含 HOLD）。
- `data/optimization_state.json` 每日更新。
- 動態門檻同時作用於 LLM 前後。
- A/B 測試結果有記錄。
- 不改動 compliance 規則。
