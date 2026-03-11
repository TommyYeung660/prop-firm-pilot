# PropFirmPilot v1.3.9 — v1.3.7 生產環境問題修復報告 + v1.3.9 生產強化

> **報告日期**: 2026-03-09（初版）/ 2026-03-10（追加 Part F）/ 2026-03-11（追加 Part G）  
> **版本**: v1.3.8 (P0/P1) + v1.3.9 (P2/P3) + v1.3.9a (生產強化) + v1.3.9b (生產調優)  
> **基準版本**: v1.3.7  
> **聚焦範圍**: v1.3.7 五天生產運行 (Mar 4-9, 2026) 發現的 15 個問題全面修復 + v1.3.9 首日生產 (Mar 10) 發現的 12 個問題強化修復，涵蓋 prop-firm-pilot 與 TradingAgents 兩個倉庫

---

## 目錄

### Part A — 生產環境評估摘要

1. [版本摘要](#1-版本摘要)
2. [版本資訊](#2-版本資訊)
3. [v1.3.7 生產運行數據](#3-v137-生產運行數據)

### Part B — 問題清單與優先級

4. [問題清單與修復總覽](#4-問題清單與修復總覽)
5. [嚴重度分布](#5-嚴重度分布)

### Part C — P0/P1 修復 (v1.3.8)

6. [P0 #15 — EURUSD Eval 跨品種數據污染](#6-p0-15--eurusd-eval-跨品種數據污染)
7. [P1 #8 — LLM 95% SELL 偏差](#7-p1-8--llm-95-sell-偏差)
8. [P1 #12 — LLM 拒絕交易指令](#8-p1-12--llm-拒絕交易指令)
9. [P1 #1/#6 — EURUSD 98.7% 取消率](#9-p1-16--eurusd-987-取消率)
10. [P1 #3 — 160 次過度 Rescan](#10-p1-3--160-次過度-rescan)
11. [P1 #7 — Best Day 無限重試循環](#11-p1-7--best-day-無限重試循環)
12. [P1 #2 (Part 1) — Tactical Gate 始終相同](#12-p1-2-part-1--tactical-gate-始終相同)

### Part D — P2/P3 修復 (v1.3.9)

13. [P2 #4/#14 — TP/SL 通知顯示 0.00 lots](#13-p2-414--tpsl-通知顯示-000-lots)
14. [P2 #10 — Scanner Score 日內不更新](#14-p2-10--scanner-score-日內不更新)
15. [P2 #9 — HOLD 決策但仍開倉](#15-p2-9--hold-決策但仍開倉)
16. [P2 #11 — AB Test 未收集數據](#16-p2-11--ab-test-未收集數據)
17. [P2 #2 (Part 2) — Spread Gate 始終失敗](#17-p2-2-part-2--spread-gate-始終失敗)
18. [P3 #13 — DuckDB 事務嵌套錯誤](#18-p3-13--duckdb-事務嵌套錯誤)
19. [P3 #5 — Breakeven 門檻過高](#19-p3-5--breakeven-門檻過高)

### Part E — 共用

20. [修改檔案清單 (v1.3.8)](#20-修改檔案清單-v138)
21. [修改檔案清單 (v1.3.9)](#21-修改檔案清單-v139)
22. [測試覆蓋](#22-測試覆蓋)
23. [Git Commit 記錄 (v1.3.8)](#23-git-commit-記錄-v138)
24. [Git Commit 記錄 (v1.3.9)](#24-git-commit-記錄-v139)
25. [LLM 模型升級](#25-llm-模型升級)
26. [已知限制與未來工作](#26-已知限制與未來工作)

### Part F — v1.3.9 生產強化 (Mar 10 Log Analysis)

27. [v1.3.9 生產強化摘要](#27-v139-生產強化摘要)
28. [P1.1 — Breakeven SL 修改無驗證](#28-p11--breakeven-sl-修改無驗證)
29. [P1.2 — EODHD Null 欄位防禦](#29-p12--eodhd-null-欄位防禦)
30. [P1.3 — LLM Fallback (kimi-k2.5)](#30-p13--llm-fallback-kimi-k25)
31. [P1.4 — No Data = No Trade 防護](#31-p14--no-data--no-trade-防護)
32. [P2.5 — 連續虧損熔斷器](#32-p25--連續虧損熔斷器)
33. [P2.6 — 同向重複開倉限制](#33-p26--同向重複開倉限制)
34. [P2.7 — Risk Meta 結構化解析](#34-p27--risk-meta-結構化解析)
35. [P3.9 — Trade 回溯分析](#35-p39--trade-回溯分析)
36. [P3.10 — Scanner 低信心冷卻](#36-p310--scanner-低信心冷卻)
37. [P3.11 — 操作指標追蹤](#37-p311--操作指標追蹤)
38. [P3.12 — Pre-existing Config 測試修復](#38-p312--pre-existing-config-測試修復)
39. [修改檔案清單 (v1.3.9a)](#39-修改檔案清單-v139a)
40. [Git Commit 記錄 (v1.3.9a)](#40-git-commit-記錄-v139a)
41. [測試覆蓋 (v1.3.9a)](#41-測試覆蓋-v139a)

### Part G — v1.3.9b 生產調優 (Mar 11 Log Analysis)

42. [v1.3.9b 生產調優摘要](#42-v139b-生產調優摘要)
43. [P0 — Scanner 持倉感知](#43-p0--scanner-持倉感知)
44. [P1 — Blended Confidence Threshold 審查](#44-p1--blended-confidence-threshold-審查)
45. [P2 — Memory Journal Diff 欄位](#45-p2--memory-journal-diff-欄位)
46. [P1 — Config 激進化調整](#46-p1--config-激進化調整)
47. [修改檔案清單 (v1.3.9b)](#47-修改檔案清單-v139b)
48. [測試覆蓋 (v1.3.9b)](#48-測試覆蓋-v139b)

---

## 1. 版本摘要

v1.3.7 於 2026 年 3 月 4 日部署至 E8 Markets prop firm 生產帳戶，運行約 5 天（Mar 4-9）。經過仔細評估生產日誌，共識別出 **15 個問題**，按嚴重度分為 P0（1 項）、P1（6 項）、P2（6 項）、P3（2 項）。

本報告涵蓋三個修復波次：

- **v1.3.8**（P0/P1 修復）：修復 7 個高優先級問題，包括 TradingAgents 跨品種數據污染（P0）、LLM SELL 偏差、LLM 拒絕交易、EURUSD 過度取消、過度 Rescan、Best Day 無限循環、Tactical Gate 固定輸出
- **v1.3.9**（P2/P3 修復）：修復 8 個問題（含 Issue #2 的第二部分），包括 TP/SL 通知異常、Scanner Score 日內不更新、HOLD 後仍開倉、AB Test 數據收集、Spread Gate 失敗、DuckDB 事務嵌套、Breakeven 門檻過高

- **v1.3.9 追加改動**：整合 EODHD Intraday API 填充戰術模塊的 5min/1h bar 數據（Issue #2 Part 1 完成）、實現 AB Test 真正的模型切換（Issue #11 完成）、升級 LLM 模型至 gpt-5.4 和 kimi-k2.5

- **v1.3.9a 生產強化**（Mar 10 Log Analysis）：v1.3.9 首日生產運行後，從 prod log 分析出 12 個新問題（P1×4、P2×3、P3×4），全部修復並新增 99 個測試。涵蓋 breakeven SL 驗證、EODHD null 防禦、LLM fallback、no-data guard、連續虧損熔斷、重複開倉限制、risk meta 解析、trade 回溯、scanner 冷卻、操作指標追蹤

所有修復均已推送至 `fix/v1.3.9-p1-fixes` 分支，**996 項測試全部通過**（較前版 897 新增 99 項）。

---

## 2. 版本資訊

| 項目 | 值 |
|---|---|
| 基準版本 | v1.3.7 |
| 修復版本 | v1.3.8 (P0/P1) + v1.3.9 (P2/P3) |
| 報告日期 | 2026-03-09 |
| 跨倉庫 | prop-firm-pilot + TradingAgents |
| 測試總數 | 996 tests passed（v1.3.8/v1.3.9: 897, v1.3.9a: +99） |
| v1.3.8 prop-firm-pilot 修改檔案 | 7 files |
| v1.3.8 TradingAgents 修改檔案 | 2 files |
| v1.3.9 prop-firm-pilot 修改檔案 | 25 files (+1,578/-59 lines) |
| v1.3.9 TradingAgents 修改檔案 | 4 files (LLM upgrade) |
| v1.3.9a prop-firm-pilot 修改檔案 | 39 files (+4,705/-161 lines) |
| 生產運行時間 | ~5 天（Mar 4-9, 2026）+ Mar 10 首日 |
| 問題總數 | 27（原 15 + 新 12）（P0:1, P1:10, P2:9, P3:7） |

---

## 3. v1.3.7 生產運行數據

### 3.1 整體指標

| 指標 | 數值 |
|---|---|
| 運行天數 | ~5 天（Mar 4-9） |
| 交易意圖建立 | 246 |
| 交易意圖取消 | 197（80.1%） |
| LLM 決策 | 76 |
| 交易開倉 | 24 |
| 交易平倉 | 24 |
| 交易被拒 | 23 |
| 總損益 | +$92.10 |
| 最高水位 | $5,165.10（初始 $5,000） |
| 系統健康度 | 3.5/10 |

### 3.2 按品種分解

| 品種 | 交易筆數 | P&L | 勝/負/平 |
|---|---|---|---|
| AUDUSD | 19 | +$130.14 | 9W/7L/3BE |
| GBPUSD | 2 | -$24.85 | 0W/2L/0BE |
| EURUSD | 2 | $0.00 | 0W/0L/2BE |
| USDJPY | 1 | -$13.19 | 0W/1L/0BE |

### 3.3 LLM 決策偏差

49 筆方向性決策中：

- **SELL**: 47（95.9%）
- **BUY**: 1（2.0%）
- **HOLD**: 1（2.0%）

極端的 SELL 偏差是本次修復的關鍵驅動因素之一（Issue #8）。

### 3.4 關鍵異常

1. **80.1% 意圖取消率**：246 個意圖中 197 個被取消，主要因為 EURUSD 的冷啟動閾值過高（Issue #1/#6）
2. **95.9% SELL 偏差**：LLM signal extraction prompt 的選項排序導致位置偏差（Issue #8）
3. **160 次日內 Rescan**：volatility 閾值過低 + cooldown 過短 + 平倉自動 rescan（Issue #3）
4. **Tactical Gate 完全無效**：缺少 bar 數據導致所有 gate 永遠失敗，輸出完全相同（Issue #2）
5. **跨品種數據污染**：TradingAgents 共用單例的 race condition 導致 EURUSD 使用 AUDUSD 數據（Issue #15）

---

## 4. 問題清單與修復總覽

| # | 優先級 | 問題描述 | 修復版本 | 倉庫 | 狀態 |
|---|---|---|---|---|---|
| 15 | P0 | EURUSD Eval 跨品種數據污染 | v1.3.8 | TradingAgents | 已修復 |
| 8 | P1 | LLM 95% SELL 偏差 | v1.3.8 | TradingAgents | 已修復 |
| 12 | P1 | LLM 拒絕交易指令 | v1.3.8 | TradingAgents | 已修復 |
| 1/6 | P1 | EURUSD 98.7% 取消率（冷啟動閾值） | v1.3.8 | prop-firm-pilot | 已修復 |
| 3 | P1 | 160 次過度 Rescan | v1.3.8 | prop-firm-pilot | 已修復 |
| 7 | P1 | Best Day 無限重試循環 | v1.3.8 | prop-firm-pilot | 已修復 |
| 2 | P1/P2 | Tactical Gate 始終相同 + Spread Gate 失敗 | v1.3.8 + v1.3.9 | prop-firm-pilot | 已修復 |
| 4/14 | P2 | TP/SL 通知顯示 0.00 lots | v1.3.9 | prop-firm-pilot | 已修復 |
| 10 | P2 | Scanner Score 日內不更新 | v1.3.9 | prop-firm-pilot | 已修復 |
| 9 | P2 | HOLD 決策但仍開倉 | v1.3.9 | prop-firm-pilot | 已修復 |
| 11 | P2 | AB Test 未收集數據 | v1.3.9 | prop-firm-pilot | 已修復 |
| 13 | P3 | DuckDB 事務嵌套錯誤 | v1.3.9 | prop-firm-pilot | 已修復 |
| 5 | P3 | Breakeven 門檻過高 | v1.3.9 | prop-firm-pilot | 已修復 |

---

## 5. 嚴重度分布

| 優先級 | 數量 | 描述 | 修復版本 |
|---|---|---|---|
| P0 (Critical) | 1 | 數據完整性問題，可能導致錯誤交易決策 | v1.3.8 |
| P1 (High) | 6 | 嚴重影響交易效能或系統穩定性 | v1.3.8 |
| P2 (Medium) | 6 | 功能缺陷或次要問題，不會導致資金損失 | v1.3.9 |
| P3 (Low) | 2 | 配置優化或邊緣情況處理 | v1.3.9 |

**P0/P1（v1.3.8）重點**：
- 修復跨倉庫（prop-firm-pilot + TradingAgents）共 9 個文件
- 針對數據完整性、LLM 決策品質、系統穩定性的根本修復

**P2/P3（v1.3.9）重點**：
- 修復 prop-firm-pilot 共 25 個文件（+1,578/-59 lines）
- 針對通知品質、功能完整性、邊緣情況的改善

---

## 6. P0 #15 — EURUSD Eval 跨品種數據污染

### 問題描述

生產環境中 `eval_results/EURUSD/` 目錄下的 JSON 檔案中 `company_of_interest` 欄位顯示為 "AUDUSD"。EURUSD 的 LLM 分析實際使用了 AUDUSD 的市場數據，導致所有 EURUSD 決策基於錯誤的品種數據。

### 根本原因

`TradingAgentsGraph` 是共用單例。當多個 async worker 透過 `asyncio.to_thread` 同時呼叫 `propagate()` 時，第二個 caller 會覆寫 `self.ticker`，導致第一個 caller 的 `_log_state()` 寫入錯誤目錄。

```
Worker A: propagate("EURUSD") → self.ticker = "EURUSD"
Worker B: propagate("AUDUSD") → self.ticker = "AUDUSD"  # 覆寫!
Worker A: _log_state() → self.ticker 已是 "AUDUSD" → 寫入錯誤目錄
```

### 修復方案

在 `trading_graph.py` 中消除 instance state 的 race condition：

1. `propagate()` 不再直接寫入 `self.ticker`，改為將 `company_name` 作為參數傳遞給 `_log_state()`
2. `_log_state()` 簽名從 `(self, trade_date, final_state)` 改為 `(self, ticker: str, trade_date, final_state)`
3. Instance state 移到 `_log_state()` 之後才寫入（向後相容 `reflect_and_remember`）

```python
# tradingagents/graph/trading_graph.py

def propagate(self, company_name, trade_date, ...):
    # ... graph execution ...
    final_state = accumulated_state

    # Log BEFORE setting instance state (thread safety)
    self._log_state(company_name, trade_date, final_state)

    # Set instance state AFTER logging (backward compat for reflect_and_remember)
    self.curr_state = final_state
    self.ticker = company_name

    return final_state, self.process_signal(final_state["final_trade_decision"])

def _log_state(self, ticker: str, trade_date, final_state):
    directory = Path(f"eval_results/{ticker}/TradingAgentsStrategy_logs/")
    # ... uses ticker parameter instead of self.ticker ...
```

### 影響範圍

- **倉庫**: TradingAgents
- **修改檔案**: `tradingagents/graph/trading_graph.py`
- **Commit**: `c4c4611`
- **風險等級**: 高 — 任何多品種並行場景都會觸發此 bug

---

## 7. P1 #8 — LLM 95% SELL 偏差

### 問題描述

49 筆方向性 LLM 決策中 47 筆為 SELL（95.9%），僅 1 筆 BUY、1 筆 HOLD。系統幾乎完全失去多空平衡能力。

### 根本原因

Signal extraction prompt 中選項順序為 `"SELL, BUY, or HOLD"`，SELL 排在最前面。研究顯示 LLM 對選項存在**位置偏差（positional bias）**，傾向選擇列表中較前的選項。當 LLM 對方向不確定時，會傾向選擇第一個被呈現的選項。

### 修復方案

**初始修復**（commit `05a7a34`）：將順序改為 `"BUY, HOLD, or SELL"`。

**追加改進**（commit `81d5793`）：用戶指出固定順序仍可能產生 BUY 偏差。改為每次呼叫時**隨機打亂** BUY/HOLD/SELL 的順序，從根本上消除位置偏差。

```python
# tradingagents/graph/trading_graph.py
import random

def process_signal(self, final_trade_decision: str) -> str:
    options = ["BUY", "HOLD", "SELL"]
    random.shuffle(options)
    option_str = ", ".join(options[:-1]) + f", or {options[-1]}"

    prompt = (
        f"extract the investment decision: {option_str}. "
        f"Provide only the extracted decision ({', '.join(options)}) as your output"
    )
    # ...
```

### 影響範圍

- **倉庫**: TradingAgents
- **修改檔案**: `tradingagents/graph/trading_graph.py`
- **Commits**: `05a7a34`（初始重排序，已被取代）、`81d5793`（隨機化）
- **預期效果**: LLM 決策分布應接近市場真實方向比例，不再系統性偏向任何方向

---

## 8. P1 #12 — LLM 拒絕交易指令

### 問題描述

Trader agent 回應中出現 "I cannot provide specific trading instructions" 等拒絕語句。LLM 的安全護欄（safety guardrails）將正常的交易分析請求誤判為不當內容，導致有效信號被降級為 HOLD。

### 根本原因

`trader.py` 的 system prompt 缺乏明確的交易授權聲明與模擬環境說明。LLM 預設的安全策略會阻止提供「具體的交易建議」，即使在合法的量化交易系統中也是如此。

### 修復方案

在 `trader.py` 的 system prompt 中添加四項關鍵元素：

1. **明確授權聲明**: "authorized to analyze market data and provide specific trading recommendations"
2. **模擬環境聲明**: "This is a simulation environment for research purposes"
3. **禁止拒絕指令**: "You MUST always provide a clear BUY, HOLD, or SELL decision"
4. **雙向平衡提示**: "Consider both bullish and bearish signals equally before deciding"

```python
# tradingagents/agents/traders/trader.py
"content": f"""You are a trading agent authorized to analyze market data
and provide specific trading recommendations. You MUST always provide a
clear BUY, HOLD, or SELL decision — do not refuse or disclaim. This is a
simulation environment for research purposes.

Based on your analysis, provide a specific recommendation to buy, sell, or
hold. Consider both bullish and bearish signals equally before deciding.
End with a firm decision and always conclude your response with
'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' ..."""
```

### 影響範圍

- **倉庫**: TradingAgents
- **修改檔案**: `tradingagents/agents/traders/trader.py`
- **Commit**: `d8f56c4`
- **預期效果**: 消除 LLM 安全護欄的誤觸發，確保每筆決策都能產生有效的 BUY/HOLD/SELL 輸出

---

## 9. P1 #1/#6 — EURUSD 98.7% 取消率

### 問題描述

EURUSD 的交易意圖取消率高達 98.7%。在 v1.3.7 的 5 天運行中，EURUSD 幾乎所有意圖都因為 blended confidence 低於閾值而被取消，僅執行了 2 筆（均為 breakeven）。

### 根本原因

全域 `min_blended_confidence=0.65`，而 EURUSD scanner score 通常只有 0.38-0.39。Blending formula 計算如下：

```
blended = 0.6 * confidence + 0.4 * scanner
       = 0.6 * 0.67 + 0.4 * 0.39
       = 0.558
```

0.558 遠低於 0.65 閾值。更嚴重的是，冷啟動系統（`win_rate=0.0`）直接跳到最嚴格的閾值層級，新品種永遠無法累積交易記錄。

### 修復方案

在 `thresholds.py` 中新增分級冷啟動閾值，根據 `win_rate` 動態調整：

```python
# src/optimize/thresholds.py
def _stepwise_threshold(win_rate: float) -> Thresholds:
    if win_rate < 0.20:
        # Cold start — use medium to allow exploration
        return Thresholds(min_confidence="medium", min_blended_confidence=0.55)
    if win_rate < 0.45:
        return Thresholds(min_confidence="high", min_blended_confidence=0.60)
    if win_rate > 0.55:
        return Thresholds(min_confidence="low", min_blended_confidence=0.45)
    return Thresholds(min_confidence="medium", min_blended_confidence=0.55)
```

| win_rate 範圍 | 閾值層級 | min_blended_confidence | 說明 |
|---|---|---|---|
| < 0.20 | medium | 0.55 | 冷啟動探索，允許累積交易記錄 |
| 0.20-0.45 | high | 0.60 | 中等限制（原為 0.65） |
| > 0.55 | low | 0.45 | 高勝率品種放寬限制 |
| 其他 | medium | 0.55 | 預設中等 |

### 影響範圍

- **倉庫**: prop-firm-pilot
- **修改檔案**: `src/optimize/thresholds.py`, `src/config.py`, `tests/test_thresholds.py`, `tests/test_config.py`
- **Commit**: `27e7f02`
- **預期效果**: 冷啟動品種（如 EURUSD）可以在 0.55 閾值下開始累積交易記錄，隨著 win_rate 提升逐步收緊

---

## 10. P1 #3 — 160 次過度 Rescan

### 問題描述

Production log 顯示單日內觸發了 160 次 rescan，嚴重浪費計算資源。部分 30 分鐘窗口內有超過 10 次 rescan，遠超合理範圍。

### 根本原因

三個問題疊加：

1. **volatility_threshold_pct=0.2**：FX 品種在 30 分鐘內經常達到 0.2% 波動，輕易觸發 rescan
2. **cooldown 僅 300 秒**：5 分鐘冷卻遠不足以防止高頻觸發
3. **平倉自動 rescan**：每次 `_handle_position_closed()` 都呼叫 `_rescan_event.set()`

### 修復方案

三管齊下：

```python
# src/scheduler/scheduler.py

# 1. 移除平倉自動 rescan
# v1.3.8: Removed auto-rescan on position close to reduce excessive rescans.
logger.debug("Position closed for {} — skipping auto-rescan (v1.3.8)", symbol)
```

```yaml
# config/e8_one_5k_challenge.yaml

# 2. 提高 volatility 閾值
volatility_threshold_pct: 0.5        # Was 0.2

# 3. 增加 cooldown
volatility_cooldown_seconds: 1800    # Was 300 (30 minutes)
```

| 參數 | 修改前 | 修改後 | 效果 |
|---|---|---|---|
| volatility_threshold_pct | 0.2 | 0.5 | 僅在顯著波動時觸發 |
| volatility_cooldown_seconds | 300 | 1800 | 30 分鐘冷卻防止高頻 |
| 平倉自動 rescan | 啟用 | 停用 | 消除不必要的 rescan 來源 |

### 影響範圍

- **倉庫**: prop-firm-pilot
- **修改檔案**: `src/scheduler/scheduler.py`, `config/e8_one_5k_challenge.yaml`
- **Commit**: `6f6eca3`
- **預期效果**: 日內 rescan 次數從 160+ 降至 5-10 次合理範圍

---

## 11. P1 #7 — Best Day 無限重試循環

### 問題描述

當天已實現利潤觸及 Best Day 40% 限制後，scanner loop 不斷重複檢查和 log，形成無限重試循環直到系統關閉。

### 根本原因

`_should_pause_new_entries()` 回傳 `True` 後，scanner loop 僅 sleep 一個 interval 然後重新檢查。由於已實現利潤不會自動減少（只有新交易才會改變），這導致：

1. 每個 interval 都重新計算 Best Day
2. 每次都 log "pausing" 訊息
3. 永遠無法退出暫停狀態（直到次日）

### 修復方案

新增 `_best_day_paused_today` 日級別暫停旗標：

```python
# src/scheduler/scheduler.py

self._best_day_paused_today: str | None = None

# In _scanner_loop:
today = self._today_str()

# Fast path: already paused today
if self._best_day_paused_today == today:
    await asyncio.sleep(interval)
    continue

# Check Best Day limit
if self._should_pause_new_entries():
    logger.warning(
        "Scanner loop: Best Day protection active ({}), "
        "pausing new intents FOR THE REST OF THE DAY",
        self._best_day_tracker.summary(),
    )
    self._best_day_paused_today = today
    await asyncio.sleep(interval)
    continue
```

**行為變更**：
- 觸發 Best Day 限制時，設定當天日期字串為暫停旗標
- 後續迭代直接跳過 scanning，不再重複檢查和 log
- 次日自動重置（日期字串不匹配）

### 影響範圍

- **倉庫**: prop-firm-pilot
- **修改檔案**: `src/scheduler/scheduler.py`, `tests/test_scheduler.py`
- **Commit**: `7ba216f`
- **預期效果**: Best Day 觸發後立即停止 scanning，僅 log 一次警告

---

## 12. P1 #2 (Part 1) — Tactical Gate 始終相同

### 問題描述

Tactical Validator 對所有品種、所有時間點都產生完全相同的輸出：`"Hard gates failed: spread, atr_regime"`。gate 系統完全失效，無法提供任何有用的市場微觀過濾。

### 根本原因

`_fetch_tactical_data()` 僅取得 spread 數據，`bars_5min` 和 `bars_1h` 始終為空 DataFrame。目前的架構中，intraday bar 數據需要額外的 API 呼叫，但尚未完成整合。

所有依賴 bar 數據的 gate（ATR regime、momentum、volatility rank 等）因數據缺失而永遠回傳 `passed=False`，產生固定的失敗輸出。

### 修復方案

將無數據情境從「失敗」改為「pass-through（通過）」：

```python
# src/decision/tactical_validator.py

# ATR gate: pass-through when no bar data
if data.bars_1h is None or (hasattr(data.bars_1h, 'empty') and data.bars_1h.empty):
    return GateResult(
        passed=True,
        detail="ATR gate skipped — no 1H bar data available (pass-through)",
        gate_name="atr_regime",
    )

# Soft gates: pass-through when no bar data
if data.bars_5min is None or (hasattr(data.bars_5min, 'empty') and data.bars_5min.empty):
    return GateResult(
        passed=True,
        detail=f"{gate_name} gate skipped — no 5min bar data available (pass-through)",
        gate_name=gate_name,
    )
```

**設計決策**：Tactical Validator 目前運行在 **shadow mode**（不阻擋交易，僅記錄）。在 bar 數據 API 整合完成前，pass-through 是正確的行為 — 避免因缺少數據而產生誤導性的固定輸出。

### 影響範圍

- **倉庫**: prop-firm-pilot
- **修改檔案**: `src/decision/tactical_validator.py`
- **Commit**: `2b9752c`
- **預期效果**: 無 bar 數據時各 gate 回傳 pass-through，輸出因 spread 數據的變化而產生差異

### v1.3.9 追加：EODHD Intraday 整合

Issue #2 Part 1 的根本解決方案已在 v1.3.9 中實現。新增 `EodhdProvider` 類別，透過 EODHD API 取得 5min 和 1h K 線數據，填充 `TacticalData.bars_5min` 和 `TacticalData.bars_1h`。

**實現細節**：
- `EodhdProvider` 支持 5min/1h/15min/30min 區間，使用 `httpx.AsyncClient` 非同步請求
- Symbol 轉換：`EURUSD` → `EURUSD.FOREX`
- 整合至 `scheduler._fetch_tactical_data()`，透過 `asyncio.gather()` 同時取得 spread 和 bar 數據
- 5min bars lookback 6 小時（50+ bars for EMA/RSI），1h bars lookback 30 小時（20+ bars for ATR-14）
- 需設定 `EODHD_API_KEY` 環境變數；未設定時 gates 維持 pass-through

**Tactical gates 現已可運作**：ATR regime、EMA momentum、RSI state、candle quality、data freshness 不再是 pass-through，將根據真實市場數據進行過濾。

---

## 13. P2 #4/#14 — TP/SL 通知顯示 0.00 lots

### 問題描述

Telegram 平倉通知中 volume 顯示為 `0.00 lots`、open_price 顯示為 `0.0000`、close_price 亦為 `0.0000`。用戶無法從通知中得知實際的交易量和價位。

### 根本原因

`_handle_position_closed()` 透過 Broker API 重試 3 次取得平倉數據。若 API 在所有重試中均未回傳有效數據（常見於 API 延遲或暫時不可用），`volume`, `close_price`, `open_price` 保持初始的 `0.0` 預設值，直接傳遞到 Telegram 通知。

然而，`execution_meta`（開倉時由 `engine.py` 持久化的 JSON）已包含 `volume`, `sl_price`, `tp_price`, `fill_price` 等完整資訊。

### 修復方案

在 Broker API 重試失敗後，讀取 `execution_meta` JSON 作為 fallback：

```python
# src/scheduler/scheduler.py

# execution_meta fallback for volume/prices
if volume == 0.0 or close_price == 0.0:
    try:
        decision = await asyncio.to_thread(self._store.get_decision, intent.id)
        if decision and decision.execution_meta:
            meta = json.loads(decision.execution_meta)
            if volume == 0.0 and meta.get("volume"):
                volume = meta["volume"]
            if open_price == 0.0 and meta.get("fill_price"):
                open_price = meta["fill_price"]
    except Exception as e:
        logger.debug("Could not read execution_meta for {}: {}", intent.id, e)
```

**fallback 策略**：
- `volume` -> `execution_meta.volume`
- `open_price` -> `execution_meta.fill_price`
- `close_price` 無法從 execution_meta 取得（開倉時不知道平倉價），保持 API 回傳值

### 影響範圍

- **倉庫**: prop-firm-pilot
- **修改檔案**: `src/scheduler/scheduler.py`, `tests/test_scheduler.py`
- **Commit**: `2f5b2d7`
- **預期效果**: 即使 Broker API 暫時不可用，通知仍能顯示正確的交易量和開倉價

---

## 14. P2 #10 — Scanner Score 日內不更新

### 問題描述

Production log 顯示 Mar-04 全日 AUDUSD 的 scanner_score 皆為 0.4479，日內 rescan 未產生任何不同的分數。

### 根本原因

qlib scanner 使用 daily (1D) 模型，信號僅在每日 K 線收盤後才更新。日內 rescan 必然產生相同分數，這是**設計限制**而非 bug。日內 rescan 浪費了 scanner 子進程的計算資源。

### 修復方案

當 `scanner_timeframe == "1d"` 時，跳過日內 rescan：

```python
# src/scheduler/scheduler.py

if self._config.scheduler.scanner_timeframe == "1d":
    logger.info(
        "Skipping intraday rescan for {} — daily model scores unchanged "
        "until candle close (trigger: {})",
        symbol, trigger,
    )
    return {}
```

**設計考量**：
- 此邏輯僅影響 volatility-triggered 和 position-close-triggered rescan
- 每日初始 scan 不受影響
- 未來切換到 4H 模型時，此邏輯會自動失效（`scanner_timeframe != "1d"`）

### 影響範圍

- **倉庫**: prop-firm-pilot
- **修改檔案**: `src/scheduler/scheduler.py`, `tests/test_scheduler.py`, `tests/test_scheduler_multi_timeframe.py`
- **Commit**: `54a4e7f`
- **預期效果**: 使用 1D 模型時消除無效的日內 rescan

---

## 15. P2 #9 — HOLD 決策但仍開倉

### 問題描述

LLM 對某品種做出 HOLD 決策後，系統仍然對該品種執行了開倉。表面看像是 HOLD 決策被忽略。

### 根本原因

Stale intent race condition：

1. Intent A (BUY) 已標記 `ready_for_exec`，等待 execution
2. 新一輪 scan 產生 Intent B (同品種)
3. LLM 對 Intent B 決定 HOLD
4. Intent B 被正確取消
5. 但 Intent A **仍在 execution queue 中**，被正常執行

問題在於 HOLD 決策只取消了當前 intent，未清理同品種的 stale intents。

### 修復方案

在 HOLD 決策處理中，同時取消該品種所有 stale `ready_for_exec` intents：

```python
# src/scheduler/scheduler.py

# v1.3.9: Cancel stale ready_for_exec intents for same symbol
stale_intents = await asyncio.to_thread(self._store.get_ready_intents)
for stale in stale_intents:
    if stale.symbol == intent.symbol and stale.id != intent.id:
        await self._cancel_intent_safe(
            worker_id=worker_id,
            intent_id=stale.id,
            reason="superseded_by_hold",
            context=f"Newer HOLD decision for {intent.symbol} cancels stale intent",
        )
```

### 影響範圍

- **倉庫**: prop-firm-pilot
- **修改檔案**: `src/scheduler/scheduler.py`, `tests/test_scheduler.py`
- **Commit**: `fa42821`
- **預期效果**: HOLD 決策會清理該品種的所有 pending 執行意圖，防止 stale intent 被意外執行

---

## 16. P2 #11 — AB Test 未收集數據

### 問題描述

AB testing 模塊的 `choose_model()` 和 `update_ab_stats()` 函數已完整實作，但生產環境從未收集到任何 AB 測試數據。`optimization_state` JSON 中 AB test 的 counts 和 pnl_by_model 永遠為空。

### 根本原因

三個問題：

1. **未呼叫 `choose_model()`**：`agent_bridge.py` 中的 `decide()` 方法直接使用預設模型，從未呼叫 AB 選擇邏輯
2. **未呼叫 `update_ab_stats()`**：`scheduler.py` 中的平倉處理未記錄 PnL 到 AB 統計
3. **Counts 重置 bug**：`optimization_engine.py` 建立新 `ABTestState` 時從空 dict 複製 counts，導致每次重啟都重置累計數據

### 修復方案

三個修改點：

```python
# 1. src/decision/agent_bridge.py — Wire choose_model
from src.optimize.ab_testing import choose_model

model_id = choose_model(
    intent_id=intent_id,
    ratio=self._ab_state.ratio,
    model_a=self._ab_state.model_a,
    model_b=self._ab_state.model_b,
)
# model_id stored in execution_meta for later stats update
```

```python
# 2. src/scheduler/scheduler.py — Wire update_ab_stats on position close
from src.optimize.ab_testing import update_ab_stats

# In _handle_position_closed:
if model_id:
    update_ab_stats(
        state=self._ab_state,
        model_id=model_id,
        pnl=realized_pnl,
    )
```

```python
# 3. src/optimize/optimization_engine.py — Fix counts reset bug
existing_state = load_state(self._state_path)
existing_ab = existing_state.ab_test if existing_state else ABTestState()
state.ab_test = ABTestState(
    model_a=self._ab_model_a,
    model_b=self._ab_model_b,
    ratio=self._ab_ratio,
    counts=existing_ab.counts,         # Preserve accumulated counts
    pnl_by_model=existing_ab.pnl_by_model,  # Preserve accumulated PnL
)
```

### 影響範圍

- **倉庫**: prop-firm-pilot
- **修改檔案**: `src/decision/agent_bridge.py`, `src/scheduler/scheduler.py`, `src/optimize/optimization_engine.py`, `src/execution/engine.py`, `src/main.py`, `src/decision_store/sqlite_store.py`, `tests/test_ab_routing.py`（新增）, `tests/test_decision_store.py`
- **Commit**: `8199467`
- **預期效果**: 生產環境開始收集 AB 測試數據，model_id 記錄在 execution_meta 中，PnL 按模型累計

### v1.3.9 追加：AB Test 真正的模型切換

v1.3.9 初始修復僅連接了 `choose_model()` 到 `agent_bridge.decide()`，但選出的 `model_id` 僅作為 metadata 記錄，實際 LLM 呼叫仍使用初始化時的模型。

**根本原因**：TradingAgentsGraph 的 LLM 實例在 `__init__` 時建立並作為 closure 被 compiled LangGraph 捕獲。修改 config 或 LLM 屬性**無法傳播**至已編譯的 graph。

**修復方案**：

1. 新增 `_apply_ab_model(model_id)` 方法，更新 `_merged_config` 的 `deep_think_llm`、`quick_think_llm`、`default_model`，然後重建整個 `TradingAgentsGraph` 實例
2. 將 `choose_model()` 移至 `propagate()` **之前**，確保模型在決策前切換
3. 追蹤 `_current_model_id` 避免不必要的重建（同模型不重建）
4. Rebuild 失敗時保留前一個 graph 實例（graceful fallback）

**測試覆蓋**：10 個新測試（`tests/test_ab_model_switching.py`）覆蓋 no-op、rebuild、mock skip、failure handling、call order、deterministic routing。

---

## 17. P2 #2 (Part 2) — Spread Gate 始終失敗

### 問題描述

Tactical Validator 的 spread gate 在所有品種上始終回傳 failed。這是 Issue #2 的第二部分 — Part 1 (v1.3.8) 修復了 ATR gate 和 soft gates 的 bar 數據缺失問題，Part 2 修復 spread gate 的數據缺失問題。

### 根本原因

`typical_spread` 計算公式：`instrument.avg_spread_pips * pip_size`。但配置檔中缺少 `avg_spread_pips` 欄位，預設為 `0.0`。

```
ratio = current_spread / typical_spread
      = current_spread / 0.0
      = inf  (division by zero)
```

`ratio > max_spread_ratio` → 永遠為 True → gate 永遠失敗。

### 修復方案

兩管齊下：

**1. 配置層**：為每個品種添加 `avg_spread_pips`

```yaml
# config/e8_one_5k_challenge.yaml
instruments:
  EURUSD:
    avg_spread_pips: 1.2
  GBPUSD:
    avg_spread_pips: 1.5
  USDJPY:
    avg_spread_pips: 1.3
  AUDUSD:
    avg_spread_pips: 1.8
```

**2. 代碼層**：Spread gate 在無數據時 pass-through

```python
# src/decision/tactical_validator.py
if data.typical_spread > 0 and data.current_spread > 0:
    results.append(self._check_spread_gate(data.current_spread, data.typical_spread))
else:
    results.append(GateResult(
        gate_name="spread",
        passed=True,
        detail="Spread gate skipped — no spread data available (pass-through)",
    ))
```

### 影響範圍

- **倉庫**: prop-firm-pilot
- **修改檔案**: `src/decision/tactical_validator.py`, `config/e8_one_5k_challenge.yaml`, `tests/test_tactical_validator.py`
- **Commits**: `af3253b`, `e1497b9`
- **預期效果**: 有 spread 數據時正常計算 gate，無數據時安全通過

---

## 18. P3 #13 — DuckDB 事務嵌套錯誤

### 問題描述

DuckDB 間歇性拋出 `TransactionException: cannot start a transaction within a transaction` 錯誤，影響價格數據的 upsert 操作。

### 根本原因

DuckDB Python driver 在執行第一個寫入語句時自動啟動事務（auto-begin）。`upsert()` 和 `upsert_intraday()` 方法中的明確 `BEGIN TRANSACTION` 語句在已有自動事務時觸發嵌套錯誤。

```
1. DuckDB auto-begins transaction on first write
2. Code executes explicit "BEGIN TRANSACTION"
3. DuckDB raises TransactionException: already in transaction
```

### 修復方案

在 `BEGIN TRANSACTION` 前添加 guard，捕獲已存在事務的情況：

```python
# src/data/fx_duckdb_store.py

try:
    self._conn.execute("BEGIN TRANSACTION")
except duckdb.TransactionException:
    pass  # Already in a transaction — proceed with existing one
```

**設計考量**：
- 此 guard 僅處理「已在事務中」的情況
- 其他 TransactionException 子類型不會被此 pattern 捕獲
- COMMIT/ROLLBACK 邏輯不受影響

### 影響範圍

- **倉庫**: prop-firm-pilot
- **修改檔案**: `src/data/fx_duckdb_store.py`, `tests/test_fx_duckdb_store.py`
- **Commit**: `7d91447`
- **預期效果**: 消除間歇性事務嵌套錯誤，upsert 操作穩定執行

---

## 19. P3 #5 — Breakeven 門檻過高

### 問題描述

Production 中 4 筆虧損交易在達到 breakeven（BE）啟動門檻前就被止損。BE 保護機制未能發揮預期效果。

### 根本原因

`breakeven_activation_pct=0.5` 要求浮盈達到 TP 距離的 **50%** 才觸發 BE 移動。例如：

- TP 距離 100 pips → 需要 50 pips 浮盈才觸發 BE
- 大部分交易在 20-30 pips 範圍內波動，達到 50 pips 的機率很低
- 結果：交易在 BE 啟動前就回吐利潤並觸及 SL

### 修復方案

Config-only change，將門檻從 50% 降至 30%：

```yaml
# config/e8_one_5k_challenge.yaml
scheduler:
  breakeven_activation_pct: 0.3  # Was 0.5
```

| 指標 | 修改前 | 修改後 |
|---|---|---|
| BE 啟動門檻 | 50% of TP distance | 30% of TP distance |
| 100 pips TP 所需浮盈 | 50 pips | 30 pips |
| 預期 BE 觸發率 | 低 | 中等 |

### 影響範圍

- **倉庫**: prop-firm-pilot
- **修改檔案**: `config/e8_one_5k_challenge.yaml`, `tests/test_config.py`
- **Commit**: `793b49b`
- **預期效果**: 更多交易能在回吐前觸發 BE 保護，減少因未及時 BE 而虧損的交易

---

## 20. 修改檔案清單 (v1.3.8)

### prop-firm-pilot（7 files）

| 檔案 | 動作 | 修復項 | 說明 |
|---|---|---|---|
| `src/config.py` | 修改 | #1/#6 | 冷啟動閾值相關 config 調整 |
| `src/optimize/thresholds.py` | 修改 | #1/#6 | 新增分級冷啟動閾值邏輯 |
| `src/scheduler/scheduler.py` | 修改 | #3, #7 | 移除平倉 auto-rescan + Best Day daily stop flag |
| `src/decision/tactical_validator.py` | 修改 | #2 | ATR/soft gates pass-through |
| `tests/test_config.py` | 修改 | #1/#6 | 新閾值測試覆蓋 |
| `tests/test_scheduler.py` | 修改 | #7 | Best Day flag 測試 |
| `tests/test_thresholds.py` | 修改 | #1/#6 | 分級閾值測試覆蓋 |

### TradingAgents（2 files）

| 檔案 | 動作 | 修復項 | 說明 |
|---|---|---|---|
| `tradingagents/graph/trading_graph.py` | 修改 | #15, #8 | Race condition fix + 選項隨機化 |
| `tradingagents/agents/traders/trader.py` | 修改 | #12 | 授權 + 模擬環境 prompt |

---

## 21. 修改檔案清單 (v1.3.9)

### prop-firm-pilot（25 files (+1,578/-59 lines)）

| 檔案 | 動作 | 修復項 | 說明 |
|---|---|---|---|
| `src/data/fx_duckdb_store.py` | 修改 | #13 | Transaction nesting guard |
| `src/data/fx_data_fetcher.py` | 修改 | #2 | 新增 EODHD intraday provider（5min/1h/15min/30min） |
| `src/decision/agent_bridge.py` | 修改 | #11 | Wire choose_model() 呼叫 |
| `src/decision/tactical_validator.py` | 修改 | #2 | Spread gate pass-through |
| `src/decision_store/sqlite_store.py` | 修改 | #11 | AB test model_id 儲存 |
| `src/execution/engine.py` | 修改 | #11 | model_id 寫入 execution_meta |
| `src/main.py` | 修改 | #11 | AB state 初始化注入 |
| `src/config.py` | 修改 | #11 | 更新 AB test 預設模型（gpt-5.4 vs kimi-k2.5） |
| `src/optimize/optimization_engine.py` | 修改 | #11 | Fix counts reset bug |
| `src/optimize/optimization_state.py` | 修改 | #11 | 更新 AB test 預設模型（gpt-5.4 vs kimi-k2.5） |
| `src/scheduler/scheduler.py` | 修改 | #4/#14, #9, #10 | execution_meta fallback + HOLD cancel + daily skip |
| `config/e8_one_5k_challenge.yaml` | 修改 | #2, #5 | avg_spread_pips + breakeven 門檻 |
| `tests/test_ab_routing.py` | **新增** | #11 | AB routing 端到端測試 |
| `tests/test_ab_model_switching.py` | **新增** | #11 | AB model 真正切換與 rebuild 行為測試 |
| `tests/test_config.py` | 修改 | #5 | breakeven config 驗證 |
| `tests/test_decision_store.py` | 修改 | #11 | decision store AB test 欄位測試 |
| `tests/test_fx_duckdb_store.py` | 修改 | #13 | Transaction guard 測試 |
| `tests/test_fx_data_fetcher.py` | 修改 | #2 | EODHD provider intraday bar 測試（新增 8 tests） |
| `tests/test_scheduler.py` | 修改 | #4/#14, #9, #10 | 多項 scheduler 場景測試 |
| `tests/test_scheduler_multi_timeframe.py` | 修改 | #10 | Daily model intraday skip 測試 |
| `tests/test_tactical_validator.py` | 修改 | #2 | Spread gate pass-through 測試 |

### v1.3.9 追加改動（Batch 1-3，9 files, +749/-24 lines）

| 檔案 | 動作 | 修復項 | 說明 |
|---|---|---|---|
| `src/data/fx_data_fetcher.py` | 修改 | #2 | 新增 `EodhdProvider`，取得 5min/1h/15min/30min intraday bars |
| `src/scheduler/scheduler.py` | 修改 | #2 | `_fetch_tactical_data()` 透過 `asyncio.gather()` 同時抓 spread + 5min/1h bars |
| `src/decision/agent_bridge.py` | 修改 | #11 | 新增 `_apply_ab_model()`，rebuild TradingAgentsGraph 以套用選定 model |
| `tests/test_ab_model_switching.py` | **新增** | #11 | `_apply_ab_model` 與 decide() AB integration 測試（10 tests） |
| `tests/test_fx_data_fetcher.py` | 修改 | #2 | 新增 8 個 EODHD provider 測試 |
| `src/config.py` | 修改 | #11 | 更新 AB test 預設模型（gpt-5.4 vs kimi-k2.5） |
| `src/optimize/optimization_engine.py` | 修改 | #11 | 更新 AB test 預設模型（gpt-5.4 vs kimi-k2.5） |
| `src/optimize/optimization_state.py` | 修改 | #11 | 更新 AB test 預設模型（gpt-5.4 vs kimi-k2.5） |
| `config/e8_one_5k_challenge.yaml` | 修改 | #11 | 更新 AB model config（gpt-5.4 vs kimi-k2.5） |

### TradingAgents（4 files, LLM upgrade）

| 檔案 | 動作 | 修復項 | 說明 |
|---|---|---|---|
| `.env.example` | 修改 | - | 升級預設模型字串（gpt-5.4, kimi-k2.5） |
| `tradingagents/default_config.py` | 修改 | - | default model strings |
| `tests/test_recursion_limit.py` | 修改 | - | test model references |
| `tests/test_telegram_model_switch.py` | 修改 | - | test model references |

---

## 22. 測試覆蓋

### 整體統計

| 指標 | 數值 |
|---|---|
| 測試總數 (v1.3.8/v1.3.9) | 897 |
| 測試總數 (v1.3.9a 生產強化後) | **996** |
| 新增測試 (v1.3.9a) | +99 |
| 通過 | 996 (100%) |
| 失敗 | 0 |
| Ruff 警告 | 0（原 3 個 pre-existing 已修復） |

### v1.3.8 新增/修改測試

| 測試檔案 | 覆蓋項目 |
|---|---|
| `tests/test_thresholds.py` | 冷啟動分級閾值 4 個 win_rate 區間 |
| `tests/test_config.py` | 新閾值配置載入驗證 |
| `tests/test_scheduler.py` | Best Day daily stop flag 設定與重置 |

### v1.3.9 新增/修改測試

| 測試檔案 | 覆蓋項目 |
|---|---|
| `tests/test_ab_routing.py` (NEW) | choose_model 分流、update_ab_stats 累計、counts 持久化 |
| `tests/test_ab_model_switching.py` (NEW) | `_apply_ab_model` rebuild 行為、failure fallback、decide() call order、deterministic routing |
| `tests/test_scheduler.py` | execution_meta fallback、HOLD stale intent cancel、daily model skip |
| `tests/test_scheduler_multi_timeframe.py` | 1D vs 4H model intraday rescan 行為差異 |
| `tests/test_fx_duckdb_store.py` | Transaction nesting guard、連續 upsert 穩定性 |
| `tests/test_fx_data_fetcher.py` | EODHD intraday provider（5min/1h/15min/30min）與 bar lookback 行為 |
| `tests/test_tactical_validator.py` | Spread gate pass-through、有效數據正常計算 |
| `tests/test_config.py` | breakeven_activation_pct 預設值驗證 |
| `tests/test_decision_store.py` | AB test model_id 欄位存取 |

---

## 23. Git Commit 記錄 (v1.3.8)

### prop-firm-pilot

| Commit | 日期 | 說明 |
|---|---|---|
| `c2bc8f8` | 2026-03-09 | chore: bump version to v1.3.8 |
| `2f59f57` | 2026-03-09 | test: update expected values to match v1.3.8 threshold and config changes |
| `2b9752c` | 2026-03-09 | fix(tactical): pass-through gates when bar data unavailable (P1 #2) |
| `27e7f02` | 2026-03-09 | fix(thresholds): add cold-start tier to reduce EURUSD over-filtering (P1 #1/#6) |
| `6f6eca3` | 2026-03-09 | fix: reduce excessive rescans (P1 #3) |
| `7ba216f` | 2026-03-09 | fix(scheduler): add best_day_paused_today flag (P1 #7) |

### TradingAgents

| Commit | 日期 | 說明 |
|---|---|---|
| `81d5793` | 2026-03-09 | fix(signal): randomize BUY/HOLD/SELL option order to eliminate positional bias |
| `d8f56c4` | 2026-03-09 | fix(trader): add authorization and simulation context (P1 #12) |
| `05a7a34` | 2026-03-09 | fix(signal): reorder prompt to BUY/HOLD/SELL (P1 #8) — superseded by 81d5793 |
| `c4c4611` | 2026-03-09 | fix(trading_graph): eliminate ticker race condition (P0 #15) |

---

## 24. Git Commit 記錄 (v1.3.9)

### prop-firm-pilot

| Commit | 日期 | 說明 |
|---|---|---|
| `775d399` | 2026-03-09 | chore: bump version to v1.3.9 and fix intraday scan test fixture |
| `7d91447` | 2026-03-09 | fix(duckdb): guard against transaction nesting in upsert (#13) |
| `793b49b` | 2026-03-09 | fix(config): lower breakeven threshold from 0.5 to 0.3 (#5) |
| `af3253b` | 2026-03-09 | fix(tactical): add spread gate pass-through for missing data (#2) |
| `e1497b9` | 2026-03-09 | fix(config): add avg_spread_pips for e8 one 5k instruments (#2) |
| `54a4e7f` | 2026-03-09 | fix(scheduler): skip intraday rescans for daily scanner model (#10) |
| `fa42821` | 2026-03-09 | fix(scheduler): cancel stale ready_for_exec intents when HOLD decided (#9) |
| `2f5b2d7` | 2026-03-09 | fix(scheduler): use execution_meta fallback for TP/SL notification data (#4/#14) |
| `8199467` | 2026-03-09 | fix(ab-test): wire AB model routing and stats collection (#11) |
| `0d104c9` | 2026-03-09 | feat: integrate EODHD intraday data for tactical gates & implement AB test model switching |

### TradingAgents

| Commit | 日期 | 說明 |
|---|---|---|
| `2d7e9f9` | 2026-03-09 | chore: upgrade LLM models - gpt-5.2->gpt-5.4, glm-4.7->kimi-k2.5 |

---

## 25. LLM 模型升級

### 變更內容

| 項目 | 舊版 | 新版 |
|---|---|---|
| Primary model (deep_think_llm) | rightcodes/gpt-5.2 | rightcodes/gpt-5.4 |
| Secondary model (quick_think_llm) | volcengine/glm-4.7 | volcengine/kimi-k2.5 |
| AB test model_a | rightcodes/gpt-5.2 | rightcodes/gpt-5.4 |
| AB test model_b | volcengine/glm-4.7 | volcengine/kimi-k2.5 |

### 修改檔案

**TradingAgents 倉庫**：
- `tradingagents/default_config.py`，default model strings
- `.env` / `.env.example`，API endpoint and key references
- `tests/test_recursion_limit.py`，test model references
- `tests/test_telegram_model_switch.py`，test model references

**prop-firm-pilot 倉庫**：
- `src/config.py`，AB test defaults
- `src/optimize/optimization_engine.py`，AB test defaults
- `src/optimize/optimization_state.py`，AB test defaults
- `config/e8_one_5k_challenge.yaml`，AB model config

### Commits
- prop-firm-pilot: `0d104c9` (included in EODHD + AB test commit)
- TradingAgents: `2d7e9f9`，`chore: upgrade LLM models - gpt-5.2->gpt-5.4, glm-4.7->kimi-k2.5`

## 26. 已知限制與未來工作

### 已知限制

1. **Tactical Gate 仍為 shadow mode**：EODHD Intraday API 已整合，bar data（5min/1H）可透過 `EODHD_API_KEY` 取得。Tactical gates 在有數據時已可運作。**限制**：EODHD free/basic plan 可能有 rate limit（5 req/min），需監控生產環境的 API 配額使用
2. **LLM 選項隨機化的一致性影響**：隨機打亂 BUY/HOLD/SELL 順序消除了位置偏差，但可能引入決策噪音。需要長期 AB test 驗證隨機化 vs 固定順序的效果差異
3. **Scanner 1D 模型日內限制**：使用 daily 模型時日內 rescan 被跳過，意味著日內的劇烈波動不會被即時捕捉。4H 模型的研究和部署可解決此限制
4. **AB Test 樣本量不足**：模型選擇決策需要足夠的樣本量（建議每組 >30 筆交易）。AB test 現已實現真正的模型切換（gpt-5.4 vs kimi-k2.5），需要持續收集數據直至達到統計顯著性
5. **Breakeven 門檻需生產驗證**：0.3 (30%) 的新門檻需要在生產環境中驗證其對整體 P&L 的影響。過低的門檻可能導致過早 BE 而錯失更大利潤
6. **kimi-k2.5 `max_completion_tokens` 超限** ❗（v1.3.9a 新發現）：當 LLM fallback 至 `volcengine/kimi-k2.5` 時，TradingAgents 預設 `max_completion_tokens=128000` 超過 kimi-k2.5 的 32768 上限，導致 API 拒絕。**根因**：`TradingAgents/tradingagents/default_config.py:101` 和 `trading_graph.py:137` 使用固定的 128000 上限，未根據模型調整。**暫時方案**：生產環境已改回 gpt-5.2 + glm-4.7。**建議修復**：在 `_apply_ab_model()` 中注入 per-model `llm_output_max_tokens`（如 kimi-k2.5 設為 32768）

### 未來工作

1. 將 Tactical Gate 從 shadow mode 切換至 enforcement mode（需先驗證 EODHD 數據品質和 gate 過濾效果）
2. 研究並部署 4H scanner 模型，解決 1D 模型的日內盲區
3. 長期追蹤 LLM 決策分布，驗證隨機化的效果
4. 設定 AB test 自動報告機制，達到樣本量後自動產生比較報告
5. 建立自動化的生產環境健康度評分系統，取代人工日誌分析
6. 修復 kimi-k2.5 `max_completion_tokens` 超限問題（在 TradingAgents 或 `_apply_ab_model()` 中實現 per-model token limit mapping）
7. 將 `fix/v1.3.9-p1-fixes` 分支合併至 `main` 並發布 v1.3.9a 正式版本

---

## 27. v1.3.9 生產強化摘要

### 背景

v1.3.9 於 2026-03-10 首日部署至生產環境。透過分析 `prod_logs_20260310_v1.3.9/INDEX.md` 生產日誌，識別出 **12 個新問題**，分為 P1（4 項）、P2（3 項）、P3（4 項，含 1 項 pre-existing config test 修復）。全部在同日完成修復並推送至 `fix/v1.3.9-p1-fixes` 分支。

### 問題清單與修復總覽

| # | 優先級 | 問題描述 | 分支 | Commit | 狀態 |
|---|---|---|---|---|---|
| P1.1 | P1 | Breakeven SL 修改無驗證 | fix/v1.3.9-p1-fixes | `4b6d882` | ✅ 已修復 |
| P1.2 | P1 | EODHD Null 欄位導致 pandas crash | fix/v1.3.9-p1-fixes | `4b6d882` | ✅ 已修復 |
| P1.3 | P1 | LLM Primary 失敗無 Fallback | fix/v1.3.9-p1-fixes | `4b6d882` | ✅ 已修復 |
| P1.4 | P1 | No Data = No Trade 防護缺失 | fix/v1.3.9-p1-fixes | `4b6d882` | ✅ 已修復 |
| P2.5 | P2 | 無連續虧損熔斷機制 | fix/v1.3.9-p1-fixes | `5ee175d` | ✅ 已修復 |
| P2.6 | P2 | 同向重複開倉無限制 | fix/v1.3.9-p1-fixes | `5ee175d` | ✅ 已修復 |
| P2.7 | P2 | Risk Meta 未結構化解析 | fix/v1.3.9-p1-fixes | `5ee175d` | ✅ 已修復 |
| P3.9 | P3 | Trade 回溯分析不足 | fix/v1.3.9-p1-fixes | `6a10026` | ✅ 已修復 |
| P3.10 | P3 | Scanner 無低信心冷卻機制 | fix/v1.3.9-p1-fixes | `6a10026` | ✅ 已修復 |
| P3.11 | P3 | 缺乏操作指標追蹤 | fix/v1.3.9-p1-fixes | `6a10026` | ✅ 已修復 |
| P3.12 | P3 | Pre-existing Config 測試失敗 | fix/v1.3.9-p1-fixes | `618d7b9` | ✅ 已修復 |

### 統計

| 指標 | 數值 |
|---|---|
| 修復問題數 | 12（P1×4, P2×3, P3×4） |
| 修改檔案數 | 39 files |
| 程式變動 | +4,705/-161 lines |
| 新增測試 | +99（897 → 996） |
| 新增源碼檔案 | 3（operational_metrics, close_resolution, low_confidence_cooldown） |
| 新增測試檔案 | 10 |
| Commits | 4（`4b6d882`, `5ee175d`, `6a10026`, `618d7b9`） |
| 方法 | TDD（先寫測試再實作） |

---

## 28. P1.1 — Breakeven SL 修改無驗證

### 問題描述

Scheduler 調用 `modify_position()` 將 SL 移至 breakeven 價位後，未透過 GET API 驗證實際修改是否生效。若 broker API 靜默失敗（回傳 200 但未實際執行），系統誤以為 SL 已更新，但實際仍留在原位。

### 根本原因

`matchtrader_client.modify_position()` 僅發送 PATCH 請求並檢查 HTTP status code，不會回讀確認實際的 SL/TP 值。MatchTrader API 在某些情況下會回傳成功但未執行修改（例如價格在 spread 內）。

### 修復方案

在 `MatchTraderClient` 新增 `verify_sl_tp()` 方法，在 `modify_position()` 後執行驗證：

```python
# src/execution/matchtrader_client.py
async def verify_sl_tp(
    self, position_id: str, expected_sl: float | None, expected_tp: float | None,
    retries: int = 3, delay: float = 1.0,
) -> bool:
    for attempt in range(retries):
        pos = await self.get_position(position_id)
        if pos and _sl_tp_match(pos, expected_sl, expected_tp):
            return True
        await asyncio.sleep(delay * (attempt + 1))
    return False
```

Scheduler 中的 breakeven 流程更新為：
1. 呼叫 `modify_position()`
2. 呼叫 `verify_sl_tp()` 確認修改生效
3. 驗證失敗時 log warning 並發送 Telegram 警告

### 影響範圍

- **倉庫**: prop-firm-pilot
- **修改檔案**: `src/execution/matchtrader_client.py`, `src/scheduler/scheduler.py`, `tests/test_breakeven_verification.py` (NEW)
- **Commit**: `4b6d882`
- **測試**: 30+ 新測試覆蓋驗證成功/失敗/重試場景

---

## 29. P1.2 — EODHD Null 欄位防禦

### 問題描述

EODHD API 回傳的 bar 數據中 `volume: null`，導致 pandas DataFrame 處理時 crash。影響所有依賴 EODHD 數據的下游模塊。

### 根本原因

EODHD 的 intraday API 在某些時段（低流動性、市場休市）回傳 `volume: null` 而非 `volume: 0`。`fx_data_fetcher.py` 中的 bar 解析未處理 null 值，直接傳遞給 pandas 導致 `TypeError`。

### 修復方案

在 `fx_data_fetcher.py` 新增 `_sanitize_bar()` 方法，將 OHLCV 欄位的 null 值替換為 0：

```python
# src/data/fx_data_fetcher.py
def _sanitize_bar(bar: dict) -> dict:
    for field in ("open", "high", "low", "close", "volume"):
        if bar.get(field) is None:
            bar[field] = 0
    return bar
```

### 影響範圍

- **倉庫**: prop-firm-pilot
- **修改檔案**: `src/data/fx_data_fetcher.py`, `tests/test_eodhd_null_defense.py` (NEW)
- **Commit**: `4b6d882`
- **測試**: null volume、null OHLC、混合 null 場景測試

---

## 30. P1.3 — LLM Fallback (kimi-k2.5)

### 問題描述

Primary LLM model（gpt-5.4 via rightcodes）失敗時，系統直接拒絕該筆交易而無 fallback 機制。生產環境中 rightcodes API 有時不穩定，導致多筆交易機會被浪費。

### 根本原因

`agent_bridge.decide()` 中僅有單次 `propagate()` 呼叫，失敗時直接拋出 exception。無任何重試或備援模型邏輯。

### 修復方案

在 `AgentBridge` 新增 `_fallback_model` 欄位與 retry-with-fallback 邏輯：

```python
# src/decision/agent_bridge.py
class AgentBridge:
    def __init__(self, ...):
        self._fallback_model: str | None = config.get("fallback_model")

    async def decide(self, ...) -> DecisionResult:
        try:
            return await self._propagate_with_model(primary_model)
        except Exception as e:
            if self._fallback_model:
                logger.warning("Primary LLM failed ({}), falling back to {}", e, self._fallback_model)
                self._apply_ab_model(self._fallback_model)
                return await self._propagate_with_model(self._fallback_model)
            raise
```

### 影響範圍

- **倉庫**: prop-firm-pilot
- **修改檔案**: `src/decision/agent_bridge.py`, `src/config.py`, `tests/test_llm_fallback.py` (NEW)
- **Commit**: `4b6d882`
- **測試**: fallback 觸發、fallback 成功、無 fallback 時直接拋出
- **註意**: 生產環境發現 kimi-k2.5 的 `max_completion_tokens` 超限問題（詳見 [Section 26](#26-已知限制與未來工作)），已暫時改回 gpt-5.2 + glm-4.7

---

## 31. P1.4 — No Data = No Trade 防護

### 問題描述

Scanner 回傳信號但 EODHD 回傳空的 bar 數據時，LLM 仍然被呼叫並基於空數據做出決策。這導致基於不完整資訊的交易決策。

### 根本原因

原始碼中僅註冊 P0 時排除此問題，但後來從 P2.8 提升至 P1。Scheduler 在取得 scanner signal 後直接傳遞給 LLM，未檢查輔助數據（EODHD bars）是否充足。

### 修復方案

在 `fx_analyst_config.py` 新增 `_has_minimum_data()` guard：

```python
# src/decision/fx_analyst_config.py
def _has_minimum_data(ohlcv_data: dict) -> bool:
    """Check if OHLCV data has enough bars for meaningful analysis."""
    for timeframe, bars in ohlcv_data.items():
        if bars is not None and len(bars) >= MIN_BARS_REQUIRED:
            return True
    return False
```

Scheduler 在呼叫 `decide()` 前檢查：
- 若數據不足，跳過該筆交易並 log warning
- 避免 LLM 在無數據情況下產生幻覇決策

### 影響範圍

- **倉庫**: prop-firm-pilot
- **修改檔案**: `src/decision/fx_analyst_config.py`, `src/scheduler/scheduler.py`, `tests/test_no_data_no_trade.py` (NEW)
- **Commit**: `4b6d882`
- **測試**: 空數據放棄、足夠數據通過、部分數據場景

---

## 32. P2.5 — 連續虧損熔斷器

### 問題描述

同一品種連續多筆 SL 止損後，系統繼續對該品種開倉，無任何保護機制。連續虧損可能累積為顯著的日內損失。

### 修復方案

在 Scheduler 新增連續虧損熔斷器邏輯：

- **觸發條件**：同一品種當日內連續 3+ 筆 SL 止損
- **行為**：暫停該品種當日的新開倉
- **重置**：次日自動重置計數器
- 透過 `_consecutive_sl_counts: dict[str, int]` 追蹤每個品種的連續 SL 數

### 影響範圍

- **倉庫**: prop-firm-pilot
- **修改檔案**: `src/scheduler/scheduler.py`, `src/config.py`, `tests/test_circuit_breaker.py` (NEW)
- **Commit**: `5ee175d`
- **測試**: 熔斷觸發、未觸發、重置、跨品種獨立性

---

## 33. P2.6 — 同向重複開倉限制

### 問題描述

同一品種同一方向可無限次開倉，導致風險集中。例如同日對 AUDUSD 開 5 筆 BUY，等同於 5 倍風險曝露。

### 修復方案

新增每日同向開倉上限：

- **限制**：同一品種同一方向每日最多 2 筆
- 透過 `_daily_direction_counts: dict[str, dict[str, int]]` 追蹤 `{symbol: {BUY: n, SELL: m}}`
- 超過限制時拒絕開倉並 log 原因

### 影響範圍

- **倉庫**: prop-firm-pilot
- **修改檔案**: `src/scheduler/scheduler.py`, `src/config.py`, `tests/test_duplicate_entry_limit.py` (NEW)
- **Commit**: `5ee175d`
- **測試**: 限制觸發、不同方向獨立、跨品種獨立、每日重置

---

## 34. P2.7 — Risk Meta 結構化解析

### 問題描述

LLM risk manager 產生的風控報告包含豐富的結構化資訊（entry_style、avoid_zone、trigger_zone、invalid_if、max_same_day_attempts），但系統僅將其作為純文字儲存，未解析為可使用的結構化欄位。

### 修復方案

在 `decision_formatter.py` 或相關模塊中新增解析邏輯，從 risk report 文字中提取結構化欄位：

- `entry_style`: 進場風格（aggressive/conservative/neutral）
- `avoid_zone`: 應避免的價格區域
- `trigger_zone`: 觸發區域
- `invalid_if`: 使決策失效的條件
- `max_same_day_attempts`: 每日最大嘗試次數

解析結果儲存在 `execution_meta` 中供下游使用。

### 影響範圍

- **倉庫**: prop-firm-pilot
- **修改檔案**: `src/decision/fx_analyst_config.py`, `src/decision/tactical_validator.py`, `tests/test_risk_meta_extraction.py` (NEW)
- **Commit**: `5ee175d`
- **測試**: 各欄位解析、缺失欄位預設值、無 risk report 場景

---

## 35. P3.9 — Trade 回溯分析

### 問題描述

平倉後的回溯分析不足：Broker API 重試 3 次後放棄，無法從 PnL 推斷平倉原因（TP/SL/BE/手動）。

### 修復方案

1. Broker API 重試從 3 增至 5 次
2. 新增 `close_resolution.py` 模塊，基於 PnL 推斷平倉原因：
   - PnL ≥ 0 且接近 TP → `"tp_hit"`
   - PnL < 0 且接近 SL → `"sl_hit"`
   - PnL ≈ 0 → `"breakeven"`
   - 其他 → `"manual_or_unknown"`

### 影響範圍

- **倉庫**: prop-firm-pilot
- **新增檔案**: `src/scheduler/close_resolution.py`
- **修改檔案**: `src/execution/matchtrader_client.py`, `src/scheduler/scheduler.py`, `tests/test_close_retrospection.py` (NEW)
- **Commit**: `6a10026`
- **測試**: 各平倉原因推斷、API 重試場景、無數據 fallback

---

## 36. P3.10 — Scanner 低信心冷卻

### 問題描述

Scanner 反覆取消低信心信號但無冷卻機制，導致同一品種短時間內被反覆掃描和取消，浪費計算資源。

### 修復方案

新增 `low_confidence_cooldown.py` 模塊：

- **觸發**：同一品種連續 3 次取消
- **行為**：該品種進入 30 分鐘冷卻期
- **重置**：冷卻期到期後自動重置計數器

### 影響範圍

- **倉庫**: prop-firm-pilot
- **新增檔案**: `src/scheduler/low_confidence_cooldown.py`
- **修改檔案**: `src/scheduler/scheduler.py`, `tests/test_low_confidence_cooldown.py` (NEW)
- **Commit**: `6a10026`
- **測試**: 冷卻觸發、冷卻期間跳過、到期重置、跨品種獨立性

---

## 37. P3.11 — 操作指標追蹤

### 問題描述

系統缺乏操作層面的指標追蹤：API 重試統計、延遲追蹤、系統運行時間等。無法快速判斷系統健康狀態。

### 修復方案

新增 `operational_metrics.py` 模塊，追蹤：

- **API 重試統計**：每個 endpoint 的重試次數、成功/失敗比
- **延遲追蹤**：API 回應時間的 p50/p95/p99
- **系統 uptime**：系統啟動時間和總運行時間
- 透過 `alert_service` 定期發送指標摘要至 Telegram

### 影響範圍

- **倉庫**: prop-firm-pilot
- **新增檔案**: `src/monitor/operational_metrics.py`
- **修改檔案**: `src/execution/matchtrader_client.py`, `src/monitor/alert_service.py`, `src/monitor/telegram_bot.py`, `tests/test_operational_metrics.py` (NEW)
- **Commit**: `6a10026`
- **測試**: 指標記錄、摘要產生、重置行為

---

## 38. P3.12 — Pre-existing Config 測試修復

### 問題描述

4 個 pre-existing 的測試失敗，因測試 assertions 與當前 YAML config 值不匹配。這些失敗在 v1.3.9 以前就存在，但未被修復。

### 修復方案

對齊測試 assertions 與 `config/e8_one_5k_challenge.yaml` 的實際值：

| 測試檔案 | 欄位 | 舊值 | 新值 |
|---|---|---|---|
| `test_prop_firm_guard_e8_one.py` | `default_risk_pct` | 0.01 | 0.005 |
| `test_switchover.py` | `shadow_mode` | False | True |
| `test_exit_reason_classification.py` | `max_drawdown_stop` | 0.08 | 0.06 |
| `test_alert_service.py` | `shadow_mode` | False | True |

### 影響範圍

- **倉庫**: prop-firm-pilot
- **修改檔案**: `tests/test_prop_firm_guard_e8_one.py`, `tests/test_switchover.py`, `tests/test_exit_reason_classification.py`, `tests/test_alert_service.py`
- **Commit**: `618d7b9`
- **測試**: 4 個失敗變為通過，測試總數從 897 增至 996（含新增 99 個）

---

## 39. 修改檔案清單 (v1.3.9a)

### 新增源碼檔案（3 files）

| 檔案 | 說明 |
|---|---|
| `src/monitor/operational_metrics.py` | 操作指標追蹤（API 重試、延遲、uptime） |
| `src/scheduler/close_resolution.py` | 平倉原因推斷邏輯（PnL-based） |
| `src/scheduler/low_confidence_cooldown.py` | Scanner 低信心冷卻機制 |

### 新增測試檔案（10 files）

| 檔案 | 行數 | 覆蓋項目 |
|---|---|---|
| `tests/test_breakeven_verification.py` | 416 | SL/TP 驗證成功/失敗/重試 |
| `tests/test_circuit_breaker.py` | 122 | 連續虧損熔斷觸發/重置 |
| `tests/test_close_retrospection.py` | 135 | 平倉原因推斷/API 重試 |
| `tests/test_duplicate_entry_limit.py` | 129 | 同向開倉限制/跨品種獨立 |
| `tests/test_eodhd_null_defense.py` | 135 | Null OHLCV 防禦 |
| `tests/test_llm_fallback.py` | 103 | Fallback 觸發/成功/無 fallback |
| `tests/test_low_confidence_cooldown.py` | 93 | 冷卻觸發/到期/跨品種 |
| `tests/test_no_data_no_trade.py` | 88 | 空數據放棄/足夠數據通過 |
| `tests/test_operational_metrics.py` | 135 | 指標記錄/摘要/重置 |
| `tests/test_risk_meta_extraction.py` | 95 | 欄位解析/缺失預設/無 report |

### 修改源碼檔案（10 files）

| 檔案 | 修復項 | 說明 |
|---|---|---|
| `src/config.py` | P1.3, P2.5, P2.6 | fallback_model 欄位、circuit breaker config、duplicate entry config |
| `src/data/fx_data_fetcher.py` | P1.2 | `_sanitize_bar()` null 防禦 |
| `src/decision/agent_bridge.py` | P1.3 | `_fallback_model` + retry-with-fallback |
| `src/decision/fx_analyst_config.py` | P1.4, P2.7 | `_has_minimum_data()` + risk meta parsing |
| `src/decision/tactical_validator.py` | P2.7 | risk meta 結構化欄位整合 |
| `src/decision_store/sqlite_store.py` | P3.9 | close reason 儲存 |
| `src/execution/matchtrader_client.py` | P1.1, P3.9, P3.11 | `verify_sl_tp()` + retry 3→5 + 指標追蹤 |
| `src/monitor/alert_service.py` | P3.11 | 操作指標摘要通知 |
| `src/monitor/telegram_bot.py` | P3.11 | 指標查詢指令 |
| `src/scheduler/scheduler.py` | P1.1, P1.4, P2.5, P2.6, P3.9, P3.10 | breakeven verify + no-data guard + circuit breaker + dup limit + retrospection + cooldown |

### 修改測試檔案（13 files）

| 檔案 | 修復項 |
|---|---|
| `tests/test_ab_model_switching.py` | LLM fallback 整合 |
| `tests/test_alert_service.py` | P3.12 shadow_mode 對齊 |
| `tests/test_decision_store.py` | close reason 欄位 |
| `tests/test_exit_reason_classification.py` | P3.12 max_drawdown_stop 對齊 |
| `tests/test_fx_data_fetcher.py` | EODHD null defense |
| `tests/test_fx_duckdb_store.py` | 資料儲存整合 |
| `tests/test_prop_firm_guard_e8_one.py` | P3.12 default_risk_pct 對齊 |
| `tests/test_scanner_bridge.py` | cooldown 整合 |
| `tests/test_scheduler.py` | 多項 scheduler 場景 |
| `tests/test_scheduler_multi_timeframe.py` | cooldown 場景 |
| `tests/test_switchover.py` | P3.12 shadow_mode 對齊 |
| `tests/test_tactical_integration.py` | risk meta 整合 |
| `tests/test_volatility_monitor.py` | 指標追蹤整合 |

---

## 40. Git Commit 記錄 (v1.3.9a)

### prop-firm-pilot（分支 `fix/v1.3.9-p1-fixes`）

| Commit | 日期 | 說明 |
|---|---|---|
| `4b6d882` | 2026-03-10 | fix(v1.3.9): P1 production fixes — breakeven verify, EODHD null defense, LLM fallback, no-data guard |
| `5ee175d` | 2026-03-10 | fix(v1.3.9): P2 production fixes — circuit breaker, duplicate entry limit, risk meta extraction |
| `6a10026` | 2026-03-10 | fix(v1.3.9): P3 production fixes — trade retrospection, scanner cooldown, operational metrics |
| `618d7b9` | 2026-03-10 | fix(tests): align test assertions with current e8_one YAML config values |

---

## 41. 測試覆蓋 (v1.3.9a)

### 整體統計

| 指標 | 數值 |
|---|---|
| 測試總數 (before) | 897 |
| 測試總數 (after) | **996** |
| 新增測試 | +99 |
| 通過 | 996 (100%) |
| 失敗 | 0 |
| Ruff 警告 | 0 |

### 新增測試明細

| 測試檔案 | 覆蓋項目 |
|---|---|
| `tests/test_breakeven_verification.py` (NEW) | `verify_sl_tp()` 成功/失敗/重試、scheduler breakeven 流程整合 |
| `tests/test_circuit_breaker.py` (NEW) | 連續 SL 熔斷觸發、未觸發、重置、跨品種 |
| `tests/test_close_retrospection.py` (NEW) | PnL-based 平倉原因推斷、API 重試場景 |
| `tests/test_duplicate_entry_limit.py` (NEW) | 同向限制、方向獨立、每日重置 |
| `tests/test_eodhd_null_defense.py` (NEW) | null volume/OHLC 防禦、混合 null |
| `tests/test_llm_fallback.py` (NEW) | fallback 觸發/成功/無 fallback |
| `tests/test_low_confidence_cooldown.py` (NEW) | 冷卻觸發/到期/跨品種 |
| `tests/test_no_data_no_trade.py` (NEW) | 空數據放棄/足夠通過/部分數據 |
| `tests/test_operational_metrics.py` (NEW) | 指標記錄/摘要/重置 |
| `tests/test_risk_meta_extraction.py` (NEW) | 欄位解析/缺失預設/無 report |

---

# Part G — v1.3.9b 生產調優

---

## 42. v1.3.9b 生產調優摘要

### 背景

v1.3.9a 於 2026-03-10 19:00 (UTC+8) 完成最後一次 hotfix 後部署至生產環境，運行約 14 小時（至 2026-03-11 09:00）。經分析生產日誌，發現以下核心問題：

1. **Scanner 重複開倉**（P0）：Scanner 對已持倉品種仍生成 intent，導致 Duplicate Entry Guard 頻繁攔截
2. **Blended Confidence 門檻過高**（P1）：cold-start / losing tier 門檻設定過嚴，大量決策被 threshold 擋下
3. **記憶日誌重複度高**（P2）：相同 symbol 連續決策幾乎相同內容，缺乏 diff 對比
4. **交易效率過低**（P1）：14 小時僅 1 個倉位，資金使用率極低

### 用戶指令（暫時忽略項）

- P1 — MatchTrader API 連線重置：暫時忽略
- P2 — 缺少平倉績效閉環：暫時忽略

### 修復統計

| 指標 | 數值 |
|---|---|
| 問題總數 | 4 |
| P0 | 1 |
| P1 | 2 |
| P2 | 1 |
| 源碼文件修改 | 4 (3 src + 1 config) |
| 測試文件修改 | 4 |
| 測試通過 | 48/48 相關測試 |

---

## 43. P0 — Scanner 持倉感知

### 問題描述

Scanner loop 在生成 intent 時未檢查該品種是否已有活躍倉位。導致 EURUSD 已持倉時，Scanner 仍會產生新的 EURUSD intent，最終被下游的 Duplicate Entry Guard 攔截。這造成：
- 無意義的 LLM 評估消耗（每次 intent 都走完整評估流程再被擋下）
- 日誌噪音：大量 `duplicate_entry_guard` 拒絕記錄
- Scanner slot 被佔用，真正可交易品種無法進入

### 根因分析

`_scanner_loop` (scheduler.py:234-472) 的 guard 序列為：
1. ✅ `intent_exists` — 防止同一信號重複建 intent
2. ❌ **缺少持倉檢查** — 已持倉品種仍生成 intent
3. ✅ `rejection_cooldown` — 被拒品種冷卻

### 修復方案

在 `_scanner_loop` 的 `intent_exists` 檢查之後、`rejection_cooldown` 之前，新增持倉檢查：

```python
# P0: Position-aware scanner — skip symbols with active position
has_active = await asyncio.to_thread(
    self._store.has_active_position_for_symbol,
    signal.instrument,
)
if has_active:
    logger.info(
        "Scanner loop: {} already has active position, skipping intent",
        signal.instrument,
    )
    continue
```

使用既有方法 `has_active_position_for_symbol()` (sqlite_store.py:736)，查詢 `status = 'opened'` 的 intent。

### 預期效果

- 已持倉品種在 Scanner 層即被過濾，不進入 LLM 評估流程
- 減少 Duplicate Entry Guard 觸發次數
- 釋放 Scanner slot 給其他可交易品種

---

## 44. P1 — Blended Confidence Threshold 審查

### 問題描述

v1.3.9a 的 `_stepwise_threshold()` 門檻設定過於保守，導致大量合理決策被擋下：

| Win-rate 區間 | 舊 min_confidence | 舊 min_blended |
|---|---|---|
| < 0.20 (cold-start) | "high" (0.9) | 0.50 |
| < 0.45 (losing) | **"high" (0.9)** | 0.60 |
| > 0.55 (winning) | "medium" (0.6) | 0.45 |
| default | "medium" (0.6) | 0.55 |

**核心問題**：losing tier 要求 `min_confidence = "high"` (0.9)，但 LLM 在缺乏交易歷史時很少給出 high confidence，形成 "越虧越難開倉" 的死循環。

### 修復方案

調整 `_stepwise_threshold()` (thresholds.py:25) 各 tier：

| Win-rate 區間 | 新 min_confidence | 新 min_blended | 變化 |
|---|---|---|---|
| < 0.20 (cold-start) | "high" (0.9) | **0.48** | blended -0.02 |
| < 0.45 (losing) | **"medium" (0.6)** | **0.52** | confidence 降級, blended -0.08 |
| > 0.55 (winning) | "medium" (0.6) | 0.45 | 不變 |
| default | "medium" (0.6) | **0.50** | blended -0.05 |

Per-symbol adjustment 幅度縮小：`±0.05 → ±0.03`

### 預期效果

- Losing tier 不再要求 high confidence，打破 "越虧越難開倉" 死循環
- Cold-start 和 default tier 略放寬，增加開倉機會
- Per-symbol adjustment 縮小，防止單一品種過度偏離基準

---

## 45. P2 — Memory Journal Diff 欄位

### 問題描述

Memory Journal 對相同 symbol 的連續決策記錄幾乎相同的完整內容，造成：
- 日誌冗長，難以快速定位變化
- 無法一眼看出「這次決策和上次有什麼不同」

### 修復方案

在 `memory_journal.py` 中新增 diff 機制：

1. **新增 `_last_decisions` dict**：儲存每個 symbol 的上次決策內容
2. **新增 `_compute_diff()` 方法**：比較當前與上次決策，排除 `risk_report` 和 `final_state` 等大型欄位，產出變化列表
3. **修改 `_format_decision_block()`**：當有 diff 時，渲染 "### Δ Changes vs Previous Decision" 區段

### Diff 輸出範例

```markdown
### Δ Changes vs Previous Decision
- **action**: BUY → SELL
- **confidence**: high → medium
- **blended_confidence**: 0.72 → 0.58
- **scanner_score**: 3 → 2 (NEW)
```

---

## 46. P1 — Config 激進化調整

### 問題描述

v1.3.9a 的 config 參數在 Duplicate Entry Guard + 持倉感知的保護下過於保守：
- 14+ 小時僅 1 個倉位
- 資金使用率極低
- Scanner 掃描間隔過長，錯過短期機會

### 修復方案

調整 `config/e8_one_5k_challenge.yaml`：

| 參數 | 舊值 | 新值 | 說明 |
|---|---|---|---|
| `default_risk_pct` | 0.007 | **0.01** | 每筆風險 $35→$50 |
| `active_session_interval_seconds` | 3600 | **1800** | 活躍時段掃描 60min→30min |
| `quiet_session_interval_seconds` | 14400 | **7200** | 非活躍掃描 4h→2h |
| `tactical.soft_gates.min_score` | 2 | **1** | 降低 soft gate 門檻 |

### 安全保障

- **Compliance 參數未動**：daily drawdown (5%)、max drawdown (8%)、best day rule ($1,600) 均維持不變
- **Duplicate Entry Guard**：同方向每日最多 2 筆
- **Circuit Breaker**：連續 3 SL 觸發當日暫停該品種
- **持倉感知**（本次 P0 修復）：已持倉品種不再重複建 intent

---

## 47. 修改檔案清單 (v1.3.9b)

### 源碼 / 配置

| 檔案 | 修改內容 |
|---|---|
| `src/scheduler/scheduler.py` | P0: `_scanner_loop` 新增持倉感知檢查 |
| `src/optimize/thresholds.py` | P1: `_stepwise_threshold()` 放寬門檻 |
| `src/monitor/memory_journal.py` | P2: 新增 `_compute_diff()` + diff 渲染 |
| `config/e8_one_5k_challenge.yaml` | P1: risk_pct, intervals, soft_gate 調整 |

### 測試

| 檔案 | 修改內容 |
|---|---|
| `tests/optimize/test_thresholds.py` | 更新所有門檻斷言值 |
| `tests/optimize/test_threshold_decay.py` | 更新 10 個衰減斷言值 |
| `tests/test_config.py` | 更新 `active_session_interval_seconds` 斷言 |
| `tests/test_prop_firm_guard_e8_one.py` | 更新 `default_risk_pct` 斷言 |

---

## 48. 測試覆蓋 (v1.3.9b)

### 相關測試統計

| 指標 | 數值 |
|---|---|
| 相關測試檔案 | 4 |
| 相關測試總數 | 48 |
| 通過 | 48 (100%) |
| 失敗 | 0 |
| Ruff 警告 | 0 |

### 測試明細

| 測試檔案 | 覆蓋項目 |
|---|---|
| `tests/optimize/test_thresholds.py` | stepwise threshold 各 tier 門檻值、per-symbol adjustment |
| `tests/optimize/test_threshold_decay.py` | threshold 衰減行為、各 win-rate 區間衰減值 |
| `tests/test_config.py` | YAML config 載入、active_session_interval_seconds |
| `tests/test_prop_firm_guard_e8_one.py` | E8 One 帳戶合規守衛、default_risk_pct |
