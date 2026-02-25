# PropFirmPilot v1.0.0 — 項目總結報告

> **報告日期**: 2026-02-25  
> **版本**: v1.0.0 — 全自動 FX 交易系統正式版  
> **作者**: Tommy Yeung  
> **目標**: 將 `qlib_market_scanner`、`TradingAgents`、`qlib_rd_agent` 三個量化項目整合為 E8 Markets Prop Firm 帳號上的全自動 FX 交易系統  
> **交易市場**: FX (EURUSD, GBPUSD, USDJPY, AUDUSD, XAUUSD)  
> **執行平台**: MatchTrader REST API  
> **帳號**: 950552 (E8 Trial $5,000)

---

## 目錄

1. [項目概述與投資效益](#1-項目概述與投資效益)
2. [系統架構與核心模組](#2-系統架構與核心模組)
3. [LLM 投資效益優化機制](#3-llm-投資效益優化機制)
4. [生產環境運行現狀](#4-生產環境運行現狀)
5. [進一步功能規劃](#5-進一步功能規劃)
6. [代碼庫統計](#6-代碼庫統計)
7. [風險與限制](#7-風險與限制)

---

## 1. 項目概述與投資效益

### 1.1 項目定位

PropFirmPilot 是一套**完全自動化**的外匯交易系統，設計用於 E8 Markets Prop Firm 帳號。系統將三個獨立的量化研究項目——Qlib 市場掃描器、LLM 多智能體決策引擎、因子進化研究——整合為一條 24/7 不間斷運行的交易管線。

**核心理念**: 以量化信號（Qlib Scanner）作為交易候選篩選器，以 LLM 多智能體辯論（TradingAgents）作為最終決策引擎，以嚴格的合規引擎（PropFirmGuard）作為風控底線，實現「信號生成 → 智能決策 → 合規檢查 → 自動執行 → 實時監控」的全鏈路自動化。

### 1.2 投資效益分析

#### 開發投入

| 項目 | 數值 |
|------|------|
| 開發週期 | 約 3 週（2026-02-05 ~ 2026-02-25） |
| 源代碼 | 42 個 Python 文件，8,335 行 |
| 測試代碼 | 22 個測試文件，9,698 行，548 個測試用例 |
| 外部依賴修改 | TradingAgents 3 個文件（Alpha Vantage FX 適配） |

#### 持續運營成本

| 項目 | 月費 | 說明 |
|------|------|------|
| Alpha Vantage Premium | ~$50/月 | 75 req/min，涵蓋新聞、技術指標、價格數據 |
| LLM API（火山引擎 GLM-4） | ~$20-50/月 | 每筆決策約 3-5 輪 LLM 辯論 |
| Mac Studio 電力 | ~$10/月 | 24/7 低功耗運行 |
| **合計** | **~$80-110/月** | |

#### 收益潛力（E8 Trial $5,000 帳號）

| 階段 | 目標 | 說明 |
|------|------|------|
| Phase 1 通過 | 9% 利潤（$450） | 無時間限制，最大回撤 6% |
| Phase 2 通過 | 5% 利潤（$250） | 同上 |
| 實盤帳號 | 持續盈利 → 分成 | E8 提供 80% 利潤分成 |
| 擴展至 $50k | 同比例放大 | 月均目標 $2,000-5,000 利潤 |

#### 投資回報比

以月均運營成本 $100 計算：
- **Phase 1 通過即回本**：$450 利潤 > $100 月費
- **實盤帳號穩定盈利**：ROI 預估 5-20 倍（取決於策略勝率與持倉時間）
- **邊際成本趨近於零**：系統建成後，新增帳號的增量成本僅為 LLM API 調用費用

### 1.3 相比人工交易的優勢

| 維度 | 人工交易 | PropFirmPilot |
|------|----------|---------------|
| 執行紀律 | 受情緒影響 | 100% 機械化執行 |
| 監控時間 | 需盯盤 | 24/7 自動監控 |
| 合規遵守 | 可能遺忘規則 | 硬編碼 5 項合規檢查 |
| 決策品質 | 單一觀點 | 多智能體辯論（市場、新聞、社會情緒） |
| 反應速度 | 分鐘級 | 秒級（60s 輪詢） |
| 可擴展性 | 同時管 1-2 帳號 | 輕鬆擴展至多帳號 |

---

## 2. 系統架構與核心模組

### 2.1 三層異步架構

```
┌─────────────────────────────────────────────────────────────────────┐
│                        策略層 (Strategy Layer)                       │
│                                                                     │
│  ┌──────────────────────┐         ┌─────────────────────────────┐   │
│  │  Scanner Loop (4h)    │         │  LLM Worker(s) (poll 30s)   │   │
│  │                       │ INSERT  │                              │   │
│  │  qlib_market_scanner  │ ──────→ │  TradingAgents 多智能體      │   │
│  │  Alpha158 因子掃描    │         │  market + news + social      │   │
│  │  → TradeIntent        │         │  → BUY / SELL / HOLD         │   │
│  └──────────────────────┘         └──────────────┬──────────────┘   │
│                                                  │ UPDATE            │
└──────────────────────────────────────────────────┼──────────────────┘
                                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  決策存儲 (Decision Store — SQLite WAL)               │
│                                                                     │
│   intents 表:  pending → claimed → ready_for_exec → opened/rejected │
│   decisions 表: 完整決策記錄歸檔                                      │
│   支援 Dashboard 查詢 · TTL 清理 · 崩潰恢復                           │
└────────────────────────────────────┬────────────────────────────────┘
                                     │ READ
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       執行層 (Execution Layer)                       │
│                                                                     │
│  ┌───────────────────┐  ┌────────────────┐  ┌────────────────────┐  │
│  │ ExecutionEngine    │→│ PropFirmGuard   │→│ MatchTraderClient   │  │
│  │ 每 10s 輪詢        │  │ 5 項合規檢查    │  │ JWT + 限速 + 重試   │  │
│  └───────────────────┘  └────────────────┘  └────────────────────┘  │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │InstrumentRegistry│  │PositionSizer │  │ OrderManager           │ │
│  │ 動態符號映射      │  │ 風險計算+隨機 │  │ 訂單生命週期管理       │ │
│  └──────────────────┘  └──────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                     │
┌────────────────────────────────────┴────────────────────────────────┐
│                       監控層 (Monitoring Layer)                       │
│                                                                     │
│  ┌──────────────┐  ┌────────────┐  ┌────────────┐  ┌─────────────┐ │
│  │EquityMonitor │  │AlertService│  │TradeJournal│  │TelegramBot  │ │
│  │ 60s 淨值監控  │  │ 15 類通知   │  │ JSONL 日誌 │  │ 命令處理     │ │
│  └──────────────┘  └────────────┘  └────────────┘  └─────────────┘ │
│                                                                     │
│  ┌────────────────┐  ┌──────────────┐  ┌──────────────────────┐    │
│  │PositionMonitor │  │ DailySummary │  │ MemoryJournal        │    │
│  │ SL/TP 觸發檢測 │  │ 每日報告      │  │ 交易決策記憶 (.md)   │    │
│  └────────────────┘  └──────────────┘  └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心模組清單

| 模組 | 文件 | 行數 | 職責 |
|------|------|:---:|------|
| **Scheduler** | `scheduler/scheduler.py` | 1,087 | 7 個並發 async 循環的總指揮 |
| **MatchTraderClient** | `execution/matchtrader_client.py` | 743 | REST API 客戶端（JWT 認證、限速、重試） |
| **DecisionStore** | `decision_store/sqlite_store.py` | 841 | SQLite WAL 持久化決策存儲 |
| **ExecutionEngine** | `execution/engine.py` | 692 | 合規檢查 → 下單執行 |
| **PropFirmPilot** | `main.py` | 575 | 主入口與模式切換（daily/scheduler/monitor） |
| **AlertService** | `monitor/alert_service.py` | 430 | Telegram 15 種交易通知 |
| **PropFirmGuard** | `compliance/prop_firm_guard.py` | 370 | E8 合規檢查（安全關鍵模組） |
| **AgentBridge** | `decision/agent_bridge.py` | 302 | TradingAgents 多智能體橋接 |
| **InstrumentRegistry** | `execution/instrument_registry.py` | 226 | 動態品種映射與驗證 |
| **TelegramBotHandler** | `monitor/telegram_bot.py` | 225 | Bot 命令處理 (`/profit`, `/orders`) |
| **OrderManager** | `execution/order_manager.py` | 204 | 訂單生命週期管理 |
| **PositionSizer** | `execution/position_sizer.py` | 172 | 風險百分比倉位計算 + 隨機偏移 |
| **MemoryJournal** | `monitor/memory_journal.py` | 159 | Markdown 交易決策日記 |
| **EquityMonitor** | `monitor/equity_monitor.py` | 154 | 實時淨值監控 + 緊急平倉 |
| **FX Analyst Config** | `decision/fx_analyst_config.py` | 132 | 集中式 vendor 路由配置 |
| **TradeJournal** | `monitor/trade_journal.py` | 129 | 追加式 JSONL 交易日誌 |

### 2.3 數據流：從信號到下單

```
1. Scanner Loop (每 4 小時)
   │  qlib_market_scanner --profile fx --date 2026-02-25
   │  → 產出 ~5000 條信號，取 Top-K 候選品種
   ▼
2. TradeIntent 寫入 DecisionStore
   │  symbol=GBPUSD, score=0.72, confidence=high
   ▼
3. LLM Worker 認領 Intent
   │  TradingAgents.propagate("GBPUSD", "2026-02-25")
   │  → market_analyst: 技術面分析 (Alpha Vantage RSI/SMA/EMA)
   │  → news_analyst: 新聞情緒分析 (Alpha Vantage NEWS_SENTIMENT)
   │  → social_analyst: 社會情緒分析
   │  → 多智能體辯論 → 最終決策: BUY / SELL / HOLD
   ▼
4. Re-evaluation（已持倉品種）
   │  如果 LLM 決策與現有持倉方向相反 → 自動平倉
   │  如果 HOLD → 保持現有持倉不動
   ▼
5. Execution Engine (每 10 秒)
   │  PositionSizer: 根據 0.5% 風險計算手數 + 隨機偏移
   │  PropFirmGuard: 5 項合規檢查（全部通過才下單）
   │  MatchTraderClient: 發送 REST API 開倉指令
   ▼
6. Monitoring (持續)
   │  EquityMonitor: 每 60s 檢查淨值 → 接近回撤限制時警報
   │  PositionMonitor: 檢測 SL/TP 觸發 → 更新 DecisionStore
   │  AlertService: 所有事件推送到 Telegram
   │  MemoryJournal: 記錄決策理由到 MEMORY/{date}.md
   │  TradeJournal: 追加交易記錄到 JSONL
```

### 2.4 Re-evaluation 決策機制

Re-evaluation 是系統對**已開倉持倉**進行週期性 LLM 重新審視的機制，確保持倉方向持續符合最新市場狀況。此機制運行在 Position Monitor Loop 內（`_reevaluate_open_positions()`），是系統主動風控的核心組件。

#### 觸發時機與節流控制

| 參數 | 預設值 | 說明 |
|------|:---:|------|
| `reeval_min_hold_seconds` | 3,600s（1 小時） | 開倉後的最短持有時間，在此之前不會進行首次 re-evaluation |
| `reeval_interval_seconds` | 14,400s（4 小時） | 兩次 re-evaluation 之間的最短間隔 |
| `position_monitor_interval_seconds` | 120s | Position Monitor Loop 的輪詢頻率 |

**節流邏輯**：
1. **首次評估**：檢查持倉時長是否 ≥ `reeval_min_hold_seconds`。若持倉不足 1 小時，跳過評估（防止剛開倉就被 LLM 翻盤）。
2. **後續評估**：檢查距離上次評估是否 ≥ `reeval_interval_seconds`。若未滿 4 小時，跳過評估（避免 LLM API 成本過高）。
3. **已平倉跳過**：若持倉已被 re-evaluation 平倉（存在於 `_reevaluation_close_positions` 集合中），跳過。
4. **Mock 代理跳過**：若 `AgentBridge` 使用的是 Mock 代理（測試環境），跳過所有 re-evaluation。

#### LLM 決策輸入

Re-evaluation 調用與新開倉決策使用**同一套 TradingAgents 多智能體辯論引擎**，但額外注入持倉上下文：

```python
qlib_data = {
    # 原始 Scanner 信號數據
    "score": intent.scanner_score,
    "confidence": intent.scanner_confidence,
    "score_gap": intent.scanner_score_gap,
    # Re-evaluation 專用持倉上下文
    "position_side": pos.side,           # 當前持倉方向 (BUY/SELL)
    "unrealized_pnl": pos.profit,        # 當前未實現盈虧
    "entry_price": pos.open_price,       # 開倉價格
    "current_price": pos.current_price,  # 當前市場價格
    "hold_duration_seconds": hold_duration,  # 已持有秒數
}
```

LLM 基於這些數據經過 Market Analyst → News Analyst → Social Analyst → Investment Debate → Risk Debate 的完整辯論流程，產出最終決策。

#### 決策矩陣

```
┌──────────────────┬──────────────────┬──────────────────────────────┐
│  當前持倉方向      │  LLM 決策        │  系統動作                     │
├──────────────────┼──────────────────┼──────────────────────────────┤
│  BUY             │  BUY             │  ✅ 確認 — 繼續持有           │
│  BUY             │  HOLD            │  ✅ 觀望 — 繼續持有           │
│  BUY             │  SELL            │  🔄 反轉信號 — 自動平倉       │
│  SELL            │  SELL            │  ✅ 確認 — 繼續持有           │
│  SELL            │  HOLD            │  ✅ 觀望 — 繼續持有           │
│  SELL            │  BUY             │  🔄 反轉信號 — 自動平倉       │
└──────────────────┴──────────────────┴──────────────────────────────┘
```

**核心原則**：只有當 LLM 決策與持倉方向**完全相反**時才會觸發平倉。同方向信號視為「確認」，HOLD 視為「觀望」，兩者都保持持倉不動。

#### 平倉流程

```
1. 檢測到反轉信號 (is_reversal = True)
   ▼
2. 調用 MatchTraderClient.close_position(position_id, symbol, side, volume)
   ▼
3. 若平倉成功：
   │  • 記錄 position_id 到 _reevaluation_close_positions（含未實現 PnL）
   │  • 發送 Telegram 通知：🔄 Re-evaluation Close
   │  • Position Monitor 下一輪偵測到倉位消失 → _handle_position_closed()
   │  • exit_reason 設為 "reeval_close"
   ▼
4. 若平倉失敗：
   │  • 不記錄到 _reevaluation_close_positions
   │  • 下一個 reeval 週期重試
```

#### 實際運行範例（2026-02-25 生產日誌）

```
GBPUSD 持倉 (BUY 0.09 lots)
  ↓ 4 小時後觸發 Re-evaluation
  ↓ TradingAgents 多智能體辯論：技術面轉弱、新聞偏空
  ↓ LLM 決策：SELL
  ↓ is_reversal = (BUY position + SELL signal) = True
  ↓ 自動平倉 → exit_reason = "reeval_close"
  ↓ Telegram 通知：🔄 Re-evaluation Close GBPUSD
```

#### 測試覆蓋（16 個測試用例）

| 場景 | 測試 |
|------|------|
| HOLD 保持持倉 | `test_hold_decision_keeps_position_open` |
| 同方向確認 | `test_buy_decision_keeps_position_open`, `test_sell_signal_on_sell_position_keeps_open` |
| 反轉平倉 (BUY→SELL) | `test_sell_signal_on_buy_position_closes` |
| 反轉平倉 (SELL→BUY) | `test_buy_signal_on_sell_position_closes` |
| 節流：最近評估過 | `test_throttle_skips_recently_evaluated` |
| 節流：間隔到期允許 | `test_throttle_allows_after_interval` |
| 最短持有時間 | `test_min_hold_time_skips_early_reeval`, `test_min_hold_time_allows_after_threshold` |
| Mock 代理跳過 | `test_mock_llm_skips_evaluation` |
| 平倉失敗不記錄 | `test_close_failure_does_not_add_to_set` |
| PnL Fallback | `test_reeval_close_uses_unrealized_pnl_fallback`, `test_reeval_close_prefers_broker_pnl_over_fallback` |
| exit_reason 覆蓋 | `test_exit_reason_set_to_reevaluation_hold`, `test_normal_close_not_overridden` |


### 2.5 合規引擎（安全關鍵）

PropFirmGuard 是系統的**最後防線**，所有交易必須通過以下 5 項檢查：

| 檢查項目 | E8 Trial $5k 限制 | 安全邊際 | 說明 |
|----------|:---:|:---:|------|
| 每日回撤 | 4% ($200) | 85% → $170 觸停 | 當日虧損達 $170 即停止交易 |
| 最大回撤 | 6% ($300) | 85% → $255 觸停 | 動態追蹤高水位線 |
| 最佳日規則 | 40% × $300 = $120 | 85% → $102 | 單日盈利上限 |
| 持倉數量 | 3 倉位 | — | 最多同時持有 3 個方向 |
| API 額度 | 2,000/日 | — | MatchTrader 日請求上限 |

### 2.6 外部數據源配置

所有數據源統一由 `fx_analyst_config.py` 集中管理：

| 數據類型 | Primary | Fallback | 說明 |
|----------|---------|----------|------|
| `get_stock_data` | Alpha Vantage | yfinance → local | FX_DAILY OHLCV |
| `get_indicators` | Alpha Vantage | yfinance → local | SMA/EMA/RSI/MACD 等技術指標 |
| `get_news` | Alpha Vantage | openai → google → local | 帶 ticker 的新聞情緒 |
| `get_global_news` | Alpha Vantage | openai → local | 基於主題的宏觀新聞 |
| `get_insider_*` | local | — | FX 無內部人交易 |

---

## 3. LLM 投資效益優化機制

### 3.1 當前 LLM 決策架構

系統採用 **TradingAgents** 多智能體框架，每筆交易決策經過以下流程：

```
                    ┌─────────────────────┐
                    │   Qlib Scanner 信號  │
                    │  score=0.72, BUY     │
                    └─────────┬───────────┘
                              ▼
              ┌───────────────────────────────┐
              │      TradingAgents 決策引擎    │
              │                               │
              │  ┌─────────┐  ┌─────────┐    │
              │  │ Market   │  │  News   │    │
              │  │ Analyst  │  │ Analyst │    │
              │  │ (技術面) │  │ (新聞)  │    │
              │  └────┬────┘  └────┬────┘    │
              │       │            │          │
              │  ┌────┴────────────┴────┐    │
              │  │   Social Analyst      │    │
              │  │   (社會情緒)          │    │
              │  └──────────┬───────────┘    │
              │             ▼                 │
              │  ┌──────────────────────┐    │
              │  │  Investment Debate   │    │
              │  │  (多角度投資辯論)     │    │
              │  └──────────┬──────────┘    │
              │             ▼                │
              │  ┌──────────────────────┐    │
              │  │    Risk Debate       │    │
              │  │  (風險評估辯論)       │    │
              │  └──────────┬──────────┘    │
              │             ▼                │
              │  ┌──────────────────────┐    │
              │  │  Final Decision      │    │
              │  │  BUY / SELL / HOLD   │    │
              │  └──────────────────────┘    │
              └───────────────────────────────┘
```

**三位分析師各自從不同維度提供觀點**：
- **Market Analyst**: 基於 Alpha Vantage 技術指標（RSI、SMA、EMA、MACD 等）分析價格走勢
- **News Analyst**: 基於 Alpha Vantage NEWS_SENTIMENT API 分析新聞情緒（帶 ticker 過濾）
- **Social Analyst**: 基於網路搜索分析散戶情緒、COT 持倉數據、市場共識

分析結果進入**兩輪辯論**（投資辯論 + 風險辯論），最終產出帶有完整理由的交易決策。

### 3.2 記憶與反饋迴路

#### MemoryJournal（交易記憶）

每筆交易的決策過程被完整記錄在 `MEMORY/{YYYY-MM-DD}.md` 中：

```markdown
## 12:26:41 UTC - GBPUSD SELL

### Trade Details
- **Symbol**: GBPUSD
- **Side**: SELL
- **Volume**: 0.09
- **Stop Loss**: 1.27500
- **Take Profit**: 1.25000
- **Risk Amount**: $25.00

### Scanner Signal (Qlib)
- **Score**: 0.7208
- **Confidence**: high
- **Score Gap**: 0.0138

### TradingAgents Reasoning
**Decision**: SELL
**Risk Report**:
  Based on bearish RSI divergence and negative news sentiment...
```

#### TradeJournal（交易日誌）

所有交易活動以 JSONL 格式追加記錄，包含：
- 開倉/平倉時間、價格、滑點
- 執行延遲（毫秒級）
- 盈虧結果
- 合規檢查結果

### 3.3 基於 Log 和記憶的投資效益優化路徑

以下是系統可利用現有 Log 和記憶數據進行優化的具體方式：

#### 優化路徑 A：交易決策品質分析

**數據來源**: `MemoryJournal` (.md) + `TradeJournal` (.jsonl)

| 分析維度 | 數據 | 優化動作 |
|----------|------|----------|
| 勝率統計 | 每筆交易結果 vs LLM 預判 | 調整 LLM 信心閾值 |
| 品種表現 | 各品種盈虧分布 | 動態調整品種權重 |
| 時段表現 | 不同交易時段的勝率 | 優化 Scanner 掃描時間窗口 |
| 辯論品質 | Risk Report 與實際結果對比 | 微調 Analyst 提示詞 |

**實現方式**: 在 `TradingAgents` 的 `reflect_and_remember()` 中注入歷史交易結果，讓 LLM 在下一次決策時參考過去的成敗經驗。

#### 優化路徑 B：Scanner 信號驗證

**數據來源**: `ScannerBridge` 輸出 + 實際交易結果

| 分析維度 | 數據 | 優化動作 |
|----------|------|----------|
| Score 與實際收益相關性 | Qlib score vs 交易 PnL | 調整 score 閾值 |
| Confidence 準確度 | high/medium/low vs 勝率 | 過濾低信心信號 |
| 因子有效性 | Alpha158 各因子貢獻度 | 週末 `qlib_rd_agent` 因子進化 |

#### 優化路徑 C：風控參數自適應

**數據來源**: `EquityMonitor` 快照 + `PropFirmGuard` 拒絕記錄

| 分析維度 | 數據 | 優化動作 |
|----------|------|----------|
| 回撤曲線 | 每日淨值快照 | 動態調整 risk_per_trade |
| 合規拒絕頻率 | 被攔截的交易比例 | 優化持倉數量和倉位大小 |
| 最佳日利用率 | 距離 Best Day 限制的距離 | 在限制內最大化當日收益 |

#### 優化路徑 D：LLM 成本效益

**數據來源**: 運行 Log（LLM API 調用次數和耗時）

| 分析維度 | 數據 | 優化動作 |
|----------|------|----------|
| 每筆決策 API 成本 | Token 消耗 × 單價 | 在 HOLD 高概率場景跳過 LLM |
| 決策耗時 | LLM 辯論端到端時間 | 調整 quick_think_llm 使用場景 |
| 重複調用 | 同品種重複分析 | 增加決策緩存 TTL |

### 3.4 未來 LLM 優化功能規劃

| 功能 | 說明 | 優先級 |
|------|------|:---:|
| **歷史盈虧回饋注入** | 將過去 N 天交易結果注入 LLM 提示詞，作為 context | 🔴 High |
| **動態信心閾值** | 根據近 30 天勝率自動調整 LLM 決策信心要求 | 🟡 Medium |
| **因子貢獻度分析** | 識別有效/無效因子，優化 Scanner 輸入 | 🟡 Medium |
| **多模型 A/B 測試** | 對比 GLM-4 vs Claude vs GPT-4o 的決策品質 | 🟢 Low |
| **自適應風險參數** | 根據帳號狀態動態調整 risk_per_trade | 🟡 Medium |

---

## 4. 生產環境運行現狀

### 4.1 部署環境

| 項目 | 配置 |
|------|------|
| 主機 | Mac Studio (Apple Silicon) |
| Python | 3.10.18 |
| 包管理 | uv |
| 進程管理 | tmux (計劃遷移至 launchd) |
| 運行模式 | `--scheduler` (24/7 async pipeline) |

### 4.2 運行狀態（2026-02-25 最新）

| 指標 | 狀態 |
|------|------|
| 系統啟動 | ✅ PropFirmPilot v1.0.0 正常啟動 |
| MatchTrader 登入 | ✅ 帳號 950552 連線成功 |
| 品種映射 | ✅ 5/5 品種可交易 |
| Scanner 掃描 | ✅ 4,965 信號 → 3 候選品種 |
| 數據源路由 | ✅ 全部 Alpha Vantage (Primary) |
| LLM 決策 | ✅ GBPUSD → SELL（反向信號自動平倉） |
| 合規檢查 | ✅ 3 倉位滿倉 |
| 監控循環 | ✅ 7 個 async 循環全部運行 |
| Telegram 通知 | ✅ 正常推送 |
| 錯誤 | ✅ 零錯誤 |

### 4.3 當前持倉

| 品種 | 方向 | 手數 |
|------|:---:|:---:|
| GBPUSD | BUY | 0.09 |
| GBPUSD | (re-eval平倉中) | — |
| AUDUSD | BUY | — |

---

## 5. 進一步功能規劃

### 5.1 Dashboard — Portfolio 管理介面

**目標**: 提供 Web UI 以可視化方式管理交易系統，無需查閱終端日誌。

#### 核心功能

```
┌─────────────────────────────────────────────────────────────────┐
│                    PropFirmPilot Dashboard                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Portfolio Overview                                       │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌──────────────┐   │   │
│  │  │ 淨值    │  │ 日盈虧  │  │ 總盈虧  │  │ 回撤百分比    │   │   │
│  │  │ $5,042  │  │ +$12   │  │ +$42   │  │ 0.8% / 6%   │   │   │
│  │  └────────┘  └────────┘  └────────┘  └──────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────┐  ┌──────────────────────────────┐      │
│  │  Open Positions      │  │  淨值曲線圖                   │      │
│  │  ┌──────────────┐   │  │                               │      │
│  │  │ GBPUSD BUY   │   │  │  ───────/\──/\──────          │      │
│  │  │ 0.09 lot     │   │  │         $5,042                │      │
│  │  │ PnL: +$8.50  │   │  │                               │      │
│  │  ├──────────────┤   │  └──────────────────────────────┘      │
│  │  │ AUDUSD BUY   │   │                                        │
│  │  │ 0.07 lot     │   │  ┌──────────────────────────────┐      │
│  │  │ PnL: +$3.20  │   │  │  Trading Log                  │      │
│  │  └──────────────┘   │  │  12:26 GBPUSD SELL → 平倉      │      │
│  └─────────────────────┘  │  12:10 Scanner: 4965 signals   │      │
│                            │  12:10 System started          │      │
│                            └──────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

#### 頁面規劃

| 頁面 | 功能 | 數據來源 |
|------|------|----------|
| **Portfolio** | 淨值、持倉、盈虧、回撤曲線 | MatchTraderClient + DecisionStore |
| **Trade History** | 歷史交易列表、篩選、排序 | TradeJournal (.jsonl) |
| **Memory** | 瀏覽每日交易記憶、LLM 決策理由 | MemoryJournal (.md) |
| **Scanner** | Qlib 信號分數、品種排名、信心度 | ScannerBridge 輸出 |
| **Agent Reports** | TradingAgents 完整辯論過程 | DecisionStore (final_state) |
| **Compliance** | 合規狀態、回撤圖表、安全邊際 | PropFirmGuard + EquityMonitor |
| **Logs** | 實時日誌查看、錯誤過濾 | loguru 日誌文件 |
| **Settings** | 帳號切換、參數調整 | config YAML |

#### 技術方案建議

| 方案 | 優點 | 缺點 | 推薦度 |
|------|------|------|:---:|
| **Streamlit** | 快速開發、Python 原生、數據展示強 | 效能一般、自訂 UI 受限 | ⭐⭐⭐⭐ |
| **Next.js + FastAPI** | 完整 Web 體驗、響應式 | 開發週期長、需維護前後端 | ⭐⭐⭐ |
| **Gradio** | 極速原型、LLM 互動友好 | UI 限制大 | ⭐⭐ |

**推薦方案**: **Streamlit** — 與 Python 生態完美整合，可直接讀取 DecisionStore (SQLite)、TradeJournal (JSONL)、MemoryJournal (.md)，開發週期 3-5 天。

### 5.2 帳號管理系統

#### 多帳號切換

```yaml
# config/accounts.yaml (未來)
accounts:
  e8_trial_5k:
    config: config/e8_trial_5k.yaml
    credentials_env_prefix: E8_TRIAL
    status: active
    
  e8_signature_50k:
    config: config/e8_signature_50k.yaml
    credentials_env_prefix: E8_SIG
    status: standby
```

**功能規劃**:
- Dashboard 下拉選單切換帳號
- 每個帳號獨立的 DecisionStore、TradeJournal、MemoryJournal
- 統一的 Telegram 通知（標註帳號名稱）
- 帳號間持倉對沖檢測

### 5.3 智能報告系統

#### Qlib Scanner 報告

| 功能 | 說明 |
|------|------|
| 信號分布圖 | 各品種 score 的時間序列 |
| 因子貢獻度 | Alpha158 各因子的權重和有效性 |
| 模型健康度 | 預測分數是否出現退化 |
| 重訓練日誌 | 記錄何時觸發 `force_retrain` 及結果 |

#### TradingAgents 報告

| 功能 | 說明 |
|------|------|
| 辯論摘要 | 投資辯論和風險辯論的關鍵觀點 |
| 分析師共識度 | 3 位分析師的一致性評分 |
| 決策信心追蹤 | LLM 信心度 vs 實際結果的歷史對比 |
| 成本追蹤 | 每次決策的 LLM token 消耗 |

### 5.4 進階交易功能

| 功能 | 說明 | 優先級 | 預估工時 |
|------|------|:---:|:---:|
| **日內信號支持** | Scanner 4H/1H 頻率掃描 | 🟡 Medium | 1 週 |
| **動態 SL/TP** | 基於 ATR 自動計算止損止盈 | 🟡 Medium | 3 天 |
| **移動止損** | 盈利超過 N pips 後啟動移動止損 | 🟡 Medium | 2 天 |
| **部分平倉** | 分批止盈策略 | 🟢 Low | 3 天 |
| **相關性檢測** | 防止高相關品種同方向持倉 | 🟡 Medium | 2 天 |
| **週末因子進化** | `qlib_rd_agent` FX 因子挖掘 | 🟢 Low | 1 週 |
| **Breakeven Stop** | 盈利達標後將 SL 移至開倉價 | ✅ 已有基礎 | 1 天 |

### 5.5 開發路線圖

```
v1.0.0 (當前) ─── 全自動 FX 交易系統正式版
  │                 ✅ 24/7 Scheduler
  │                 ✅ Alpha Vantage 全數據源
  │                 ✅ 3 倉位並行
  │                 ✅ 548 測試用例
  │
  ├── v1.1.0 ────── LLM 回饋優化
  │                 □ 歷史盈虧注入 LLM 提示詞
  │                 □ 動態信心閾值
  │                 □ 交易品質統計報告
  │
  ├── v1.2.0 ────── Streamlit Dashboard
  │                 □ Portfolio Overview
  │                 □ Trade History + Memory 瀏覽
  │                 □ 實時 Log 查看
  │
  ├── v1.3.0 ────── 多帳號管理
  │                 □ 帳號切換
  │                 □ 獨立數據隔離
  │                 □ 統一 Telegram 通知
  │
  ├── v2.0.0 ────── 進階交易策略
  │                 □ 日內信號 (4H/1H)
  │                 □ 動態 SL/TP (ATR-based)
  │                 □ 相關性檢測
  │                 □ 週末因子進化
  │
  └── v3.0.0 ────── 實盤 $50k 帳號
                    □ Phase 1 & 2 通過
                    □ 風控參數優化
                    □ 生產監控告警升級
```

---

## 6. 代碼庫統計

### 6.1 源代碼概覽

| 指標 | 數值 |
|------|:---:|
| Python 源文件 | 42 |
| 源代碼行數 | 8,335 |
| 測試文件 | 22 |
| 測試代碼行數 | 9,698 |
| 測試用例數 | 548（全部通過） |
| Lint 狀態 | Clean（ruff E/F/I/N/W/UP） |
| Type 覆蓋 | 所有函數簽名完整標註 |

### 6.2 配置文件

| 文件 | 行數 | 用途 |
|------|:---:|------|
| `config/default.yaml` | 70 | 系統預設值 |
| `config/e8_trial_5k.yaml` | 65 | Trial $5,000 帳戶配置 |
| `config/e8_signature_50k.yaml` | 57 | Signature $50,000 帳戶配置 |
| `pyproject.toml` | 78 | 依賴管理與工具配置 |

### 6.3 Git 提交歷史（最近 10 個關鍵提交）

| Commit | 說明 |
|--------|------|
| `9473081` | fix: 移除 main.py 硬編碼 vendor，改用集中式 `build_agent_config()` |
| `82adacd` | fix: manual_close PnL fallback 修復 |
| `8b75397` | feat: `get_global_news` 路由至 Alpha Vantage topic-based 宏觀新聞 |
| `3437758` | feat: 全面路由至 Alpha Vantage、force_retrain、max_positions=3 |
| `f037114` | fix: 抑制 token 刷新時的重複登入日誌 |
| `737d573` | fix: 提取真實成交價格用於開倉通知 |
| `8b16d6a` | fix: re-eval HOLD 語義、PnL fallback、Scanner 容量檢查 |
| `112b0d5` | feat: API 限速保護（持久化 RateLimiter） |
| `3fcf84f` | fix: SL/TP broker 端未設定問題 |
| `a782319` | feat: 盈虧平衡止損 + LLM re-evaluation |

---

## 7. 風險與限制

### 7.1 已知風險

| 風險 | 嚴重度 | 緩解措施 |
|------|:---:|------|
| Alpha Vantage CSV 解析偶發異常 | 🟢 Low | 數據仍成功取得，不影響決策 |
| LLM 幻覺導致錯誤決策 | 🟡 Medium | 多智能體辯論 + PropFirmGuard 合規閘門 |
| MatchTrader API 不可用 | 🟡 Medium | 指數退避重試 + Telegram 告警 |
| Mac Studio 斷電/崩潰 | 🟡 Medium | DecisionStore 持久化 + 啟動恢復 |
| Scanner 模型退化 | 🟡 Medium | `force_retrain` 機制 + 週末因子進化 |

### 7.2 已知限制

| 限制 | 說明 | 計劃解決版本 |
|------|------|:---:|
| 無 Web Dashboard | 僅可通過 Telegram 和終端查看 | v1.2.0 |
| 單帳號運行 | 不支持同時管理多帳號 | v1.3.0 |
| 日線信號 | 不支持日內 4H/1H 信號 | v2.0.0 |
| 固定 SL/TP | 未根據市場波動動態調整 | v2.0.0 |
| Python 3.10 | Google API 將在 2026-10 停止支持 | v1.1.0 |

---

> **PropFirmPilot v1.0.0** — 全自動 FX 交易系統，從信號生成到下單執行，全程無人值守。
