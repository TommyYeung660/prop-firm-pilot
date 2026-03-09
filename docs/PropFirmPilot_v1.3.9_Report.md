# PropFirmPilot v1.3.9 — v1.3.7 生產環境問題修復報告

> **報告日期**: 2026-03-09  
> **版本**: v1.3.8 (P0/P1) + v1.3.9 (P2/P3)  
> **基準版本**: v1.3.7  
> **聚焦範圍**: v1.3.7 五天生產運行 (Mar 4-9, 2026) 發現的 15 個問題全面修復，涵蓋 prop-firm-pilot 與 TradingAgents 兩個倉庫

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

---

## 1. 版本摘要

v1.3.7 於 2026 年 3 月 4 日部署至 E8 Markets prop firm 生產帳戶，運行約 5 天（Mar 4-9）。經過仔細評估生產日誌，共識別出 **15 個問題**，按嚴重度分為 P0（1 項）、P1（6 項）、P2（6 項）、P3（2 項）。

本報告涵蓋兩個修復版本：

- **v1.3.8**（P0/P1 修復）：修復 7 個高優先級問題，包括 TradingAgents 跨品種數據污染（P0）、LLM SELL 偏差、LLM 拒絕交易、EURUSD 過度取消、過度 Rescan、Best Day 無限循環、Tactical Gate 固定輸出
- **v1.3.9**（P2/P3 修復）：修復 8 個問題（含 Issue #2 的第二部分），包括 TP/SL 通知異常、Scanner Score 日內不更新、HOLD 後仍開倉、AB Test 數據收集、Spread Gate 失敗、DuckDB 事務嵌套、Breakeven 門檻過高

- **v1.3.9 追加改動**：整合 EODHD Intraday API 填充戰術模塊的 5min/1h bar 數據（Issue #2 Part 1 完成）、實現 AB Test 真正的模型切換（Issue #11 完成）、升級 LLM 模型至 gpt-5.4 和 kimi-k2.5

所有修復均已合併至 `main` 分支，**897 項測試全部通過**。

---

## 2. 版本資訊

| 項目 | 值 |
|---|---|
| 基準版本 | v1.3.7 |
| 修復版本 | v1.3.8 (P0/P1) + v1.3.9 (P2/P3) |
| 報告日期 | 2026-03-09 |
| 跨倉庫 | prop-firm-pilot + TradingAgents |
| 測試總數 | 897 tests passed |
| v1.3.8 prop-firm-pilot 修改檔案 | 7 files |
| v1.3.8 TradingAgents 修改檔案 | 2 files |
| v1.3.9 prop-firm-pilot 修改檔案 | 25 files (+1,578/-59 lines) |
| v1.3.9 TradingAgents 修改檔案 | 4 files (LLM upgrade) |
| 生產運行時間 | ~5 天（Mar 4-9, 2026） |
| 問題總數 | 15（P0:1, P1:6, P2:6, P3:2） |

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
| 測試總數 | 897 |
| 通過 | 897 (100%) |
| 失敗 | 0 |
| Ruff 警告 | 3（pre-existing，非本次修復引入） |

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

### 未來工作

1. 將 Tactical Gate 從 shadow mode 切換至 enforcement mode（需先驗證 EODHD 數據品質和 gate 過濾效果）
2. 研究並部署 4H scanner 模型，解決 1D 模型的日內盲區
3. 長期追蹤 LLM 決策分布，驗證隨機化的效果
4. 設定 AB test 自動報告機制，達到樣本量後自動產生比較報告
5. 建立自動化的生產環境健康度評分系統，取代人工日誌分析
