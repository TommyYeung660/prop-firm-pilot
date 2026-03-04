
# PropFirmPilot v1.3.5 — EODHD Intraday Dual-Timeframe (1D Trend + 4H Entry) 版本報告

> **報告日期**: 2026-03-03  
> **版本**: v1.3.5（EODHD Intraday Dual-Timeframe: 1D Trend + 4H Entry）  
> **基準版本**: v1.3.0（EODHD 數據源遷移 + 交易業績修復）  
> **聚焦範圍**: (A) qlib_market_scanner 新增 EODHD intraday FX 1H→4H 本地聚合；(B) TradingAgents 新增 intraday OHLCV/Indicators tools 與 dual-timeframe FX prompt；(C) prop-firm-pilot 增加 entry_timeframe hint 並支援 scanner_timeframe / agent_timeframe 分離；(D) 彙整 prod 測試結果與生產 Hotfix

---

## 目錄

### Part A — EODHD Intraday 雙時間框架實施

1. [版本摘要](#1-版本摘要)
2. [版本資訊](#2-版本資訊)
3. [雙時間框架策略背景](#3-雙時間框架策略背景)
4. [功能總覽](#4-功能總覽)
5. [功能詳述](#5-功能詳述)
6. [架構圖](#6-架構圖)
7. [EODHD Intraday API 規格](#7-eodhd-intraday-api-規格)
8. [修改檔案清單](#8-修改檔案清單)
9. [測試覆蓋](#9-測試覆蓋)

### Part B — Prod 測試結果 + 效益分析

10. [Prod 測試環境](#10-prod-測試環境)
11. [Phase 1: qlib_scanner 1D 基準測試](#11-phase-1-qlib_scanner-1d-基準測試)
12. [Phase 2: TradingAgents + Telegram 測試](#12-phase-2-tradingagents--telegram-測試)
13. [Phase 3a: qlib_scanner 1D Backtest](#13-phase-3a-qlib_scanner-1d-backtest)
14. [Phase 3b: qlib_scanner 4H Backtest](#14-phase-3b-qlib_scanner-4h-backtest)
15. [效益分析與時間框架建議](#15-效益分析與時間框架建議)

### Part C — 生產環境 Hotfix

16. [生產環境 Hotfixes](#16-生產環境-hotfixes)

### Part E — 18 小時生產運行評估與修復

20. [18 小時 Prod 運行評估](#20-18-小時-prod-運行評估)
21. [問題清單與修復總覽](#21-問題清單與修復總覽)
22. [Critical 修復 (C1–C3)](#22-critical-修復-c1c3)
23. [High 修復 (H1–H3)](#23-high-修復-h1h3)
24. [Medium 修復 (M1–M3)](#24-medium-修復-m1m3)
25. [Low 項目 (L1–L2)](#25-low-項目-l1l2)
26. [修復檔案清單](#26-修復檔案清單)
27. [測試覆蓋（Bug Fix）](#27-測試覆蓋bug-fix)
28. [Git Commit 記錄（Bug Fix）](#28-git-commit-記錄bug-fix)

### Part F — Telegram 連線穩定性修復

29. [Telegram 連線問題背景](#29-telegram-連線問題背景)
30. [持久 HTTP 客戶端修復](#30-持久-http-客戶端修復)
31. [Circuit Breaker 自動降級機制](#31-circuit-breaker-自動降級機制)
32. [修復檔案清單（Telegram）](#32-修復檔案清單telegram)
33. [測試覆蓋（Telegram）](#33-測試覆蓋telegram)
34. [Git Commit 記錄（Telegram）](#34-git-commit-記錄telegram)

### Part D — 共用

17. [已知限制與未來工作](#17-已知限制與未來工作)
18. [相依性變更](#18-相依性變更)
19. [Git Commit 記錄](#19-git-commit-記錄)

---

## 1. 版本摘要

v1.3.5 的核心目標是把 **FX 的 intraday 數據 (EODHD /api/intraday)** 安全地接入既有系統，並落實 Oracle 建議的 **Dual-Timeframe** 策略：

- **1D（Daily）**：維持 qlib_market_scanner + Qlib Alpha158 pipeline 做 **Trend Direction**（方向與結構由長週期因子決定）
- **4H（4-Hour）**：由 EODHD intraday 取得 **1H candles**，在本地聚合成 **4H candles**，提供 TradingAgents 做 **Entry Timing**（進場時機由日內動量與結構確認）
- **Filter**：當 1D Trend 與 4H Entry 不一致時，策略偏向 **HOLD**，避免在逆勢波動中誤入場

此版本同時整理生產環境中遇到的實作落差，透過 Hotfix 修正 tool binding、ATR 計算、EODHD 週末 None OHLC bar、以及 Qlib 對 4H freq 的相容性問題。

---

## 2. 版本資訊

| 項目 | 值 |
|---|---|
| 版本號 | v1.3.5 |
| 發佈日期 | 2026-03-03 |
| 基準版本 | v1.3.0 |
| 核心策略 | Dual-Timeframe（1D Trend + 1H→4H Entry） |
| Intraday Vendor | EODHD `/api/intraday/`（1H），本地聚合 4H |
| Qlib freq 相容性 | 4H bars 以 `day` 形式供 Qlib API 使用（見 Hotfix `3c0e0f4`） |

### 跨倉庫版本統一

| 倉庫 | v1.3.0 | v1.3.5 |
|---|---|---|
| prop-firm-pilot | v1.3.0 | v1.3.5 |
| TradingAgents | v1.3.0 | v1.3.5 |
| qlib_market_scanner | v1.3.0 | v1.3.5 |

---

## 3. 雙時間框架策略背景

### 3.1 問題定義

在 v1.3.0 已完成 EODHD 數據源遷移後，系統具備穩定的 daily (1D) OHLCV 與技術指標。但 FX 策略在 **進場時機** 仍需要更高頻的結構資訊：

- 1D 因子模型（Qlib Alpha158）適合判斷趨勢方向與中長週期結構
- 交易執行（Entry/Exit）若只依賴 1D，容易在日內反轉或回撤時誤入場
- EODHD 提供 intraday API，可補足 1H/4H K 線，讓 LLM decision layer 做更精細的 timing

### 3.2 Oracle 建議：Hybrid Dual-Timeframe

本版本採取 Hybrid：

- **Scanner（Qlib）固定使用 1D** 產生方向性信號（Trend Signal）
- **Agent（TradingAgents）使用 4H** 進行進場確認（Entry Signal）
- 僅當 Trend 與 Entry 一致時才進一步建議交易

### 3.3 量化結果導向（只使用已提供數據）

- **1D**：Information Ratio (IR)=**1.179**，Win Rate (WR)=**54.9%**，annualized return **~82%**
- **4H（Qlib Backtest）**：IR=**0.299**，WR=**54.25%**，annualized return **4.86%**，max drawdown **-22.76%**，low confidence % **30.2%**

結論：**Qlib 的 Alpha158 pipeline 應維持在 1D**；4H 更適合作為 TradingAgents 的 decision context，用於 divergence detection 與 timing。

---

## 4. 功能總覽

v1.3.5 的主要工作跨三倉庫拆分為多個 implementation tasks。

| # | Task | Repo | 目的 |
|---:|---|---|---|
| 1 | Add `fetch_intraday()` / `aggregate_to_4h()` / `download_universe_intraday()` | qlib_market_scanner | 取得 EODHD 1H FX intraday，並本地聚合為 4H bars |
| 2 | runner interval routing (`1h`/`4h`) | qlib_market_scanner | `--interval` 支援 1H/4H；4H 走 1H download 後聚合 |
| 3 | Qlib workflow intraday freq compatibility | qlib_market_scanner | 對 Qlib API 對 freq 的限制做兼容映射（4H→day） |
| 4 | Add `get_intraday_stock_data` tool (1H + 4H) | TradingAgents | LLM 可直接取得 1H 與聚合後的 4H OHLCV |
| 5 | Local intraday indicator compute (4H) | TradingAgents | 因 EODHD technical API 僅支援 daily，需本地計算 SMA/EMA/RSI/MACD/Boll/ATR |
| 6 | FX prompt dual-timeframe update | TradingAgents | 將分析流程升級為 1D Trend + 4H Entry 的框架 |
| 7 | Pass `entry_timeframe` hint into qlib_data | prop-firm-pilot | Scanner signals 帶入 entry_timeframe，讓 TradingAgents 選用 intraday tools |
| 8 | Version bump all repos to 1.3.5 | All | 三倉庫版本一致，便於 prod traceability |

---

## 5. 功能詳述

本節以 task 為單位描述實作內容，並盡量維持與 v1.3.0 報告一致的敘述結構：先說責任範圍，再列出關鍵介面與設計決策。

### 5.1 Task 1 — qlib_market_scanner: EODHD intraday fetch + 4H aggregation

**責任範圍**：在 `src/data/fx_fetcher.py` 擴充 EODHD FX fetcher，使其能從 `/api/intraday/` 取得 1H candles，並提供 1H→4H 聚合能力，最後提供批量下載 `download_universe_intraday()`。

**新增能力**（重點介面）：

```python
# src/data/fx_fetcher.py

class EODHDFXFetcher:
    def fetch_intraday(self, symbol: str, start: str, end: str, interval: str = "1h") -> pd.DataFrame:
        """Return DataFrame: date, open, high, low, close, adj_close, volume."""

def aggregate_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1H OHLCV to 4H bars, origin=start_day (UTC 00:00 anchor)."""

def download_universe_intraday(..., interval: str = "1h", ...) -> List[Path]:
    """Download intraday FX from EODHD only (no fallback)."""
```

**關鍵設計**：

1. **日期參數轉 Unix timestamps**：EODHD intraday endpoint 使用 `from/to` Unix seconds。
2. **UTC timestamps**：回傳 `datetime` 字串以 UTC 表示，直接 `pd.to_datetime()`。
3. **FX volume**：可為 0，視為合法值。
4. **週末/假日防禦**：EODHD intraday 在週末可能回 `open/high/low/close=None`，需跳過 bar（見 Hotfix `f6d6caf`）。
5. **4H anchor**：使用 `resample("4h", origin="start_day")` 使 4H blocks 對齊 00:00/04:00/...。

**4H 聚合規則（OHLCV）**：

| 欄位 | 聚合規則 |
|---|---|
| open | first |
| high | max |
| low | min |
| close | last |
| adj_close | last |
| volume | sum |

---

### 5.2 Task 2 — qlib_market_scanner: runner interval routing

**責任範圍**：更新 `src/pipeline/runner.py` 讓 FX profile 能根據 `--interval` 走不同下載路徑：

- `interval in ("1h", "4h")`：固定先抓 1H（EODHD intraday），若目標是 4H 再本地聚合覆寫 CSV
- `interval == "1d"`：走既有 daily 下載流程（EODHD daily / AV fallback）

**關鍵分支（簡化示意）**：

```python
if config.profile == "fx":
    if config.data.interval in ("1h", "4h"):
        download_universe_intraday(..., interval="1h")
    else:
        download_universe(...)

if config.profile == "fx" and config.data.interval == "4h":
    for csv_file in Path(raw_dir).glob("*.csv"):
        df = pd.read_csv(csv_file, parse_dates=["date"])
        df_4h = aggregate_to_4h(df)
        df_4h.to_csv(csv_file, index=False)
```

**設計原因**：EODHD 不提供原生 4H endpoint；以 1H→4H 聚合確保 4H bars 與 00:00 anchor 一致。

---

### 5.3 Task 3 — qlib_market_scanner: Qlib workflow intraday freq 相容性

**責任範圍**：更新 `src/pipeline/qlib_workflow.py`，在 Qlib 只支援 `day/min/week/month` 的限制下，讓 4H/1H backtest 仍能跑通。

**核心機制**：

- 引入 `_to_qlib_freq()` 將內部 freq 轉為 Qlib 可接受值
- 4H bars 在 Qlib API 內視作 `day`，1H bars 視作 `1min`（僅作 API 兼容；資料本質仍是 4H/1H）

```python
_QLIB_FREQ_MAP = {"4h": "day", "1h": "1min"}

def _to_qlib_freq(freq: str) -> str:
    return _QLIB_FREQ_MAP.get(freq, freq)
```

**同時調整**：

- intraday 的 segment window 以 **bar 數** 近似 train/valid window（FX 可用歷史較短）
- intraday 的 label 定義使用較短 horizon 的 forward return（對應較短週期）

> ⚠️ **注意**：此兼容策略屬於「讓 pipeline 可運行」的工程妥協；Qlib calendar 並不真正理解 4H granularity（見第 15 節與第 17 節）。

---

### 5.4 Task 4 — TradingAgents: intraday OHLCV dataflow + tool routing

**責任範圍**：在 TradingAgents 新增 EODHD intraday dataflow，並在 vendor routing (`interface.py`) 中註冊，使 agent 能透過 tool 取得 1H + 4H OHLCV。

**資料流**：

- `tradingagents/dataflows/eodhd_stock.py`：新增 intraday fetch 與 4H aggregation helper
- `tradingagents/dataflows/interface.py`：新增 tool category `intraday_stock_apis` 與 method `get_intraday_stock_data`

**對 LLM 的輸出格式**：

```text
## 1H OHLCV
timestamp,open,high,low,close,volume
...

## 4H OHLCV
timestamp,open,high,low,close,volume
...
```

**原因**：LLM 需要同時看到 1H（細節）與 4H（結構）來判斷 entry timing，避免只靠單一時間框架導致誤判。

---

### 5.5 Task 5 — TradingAgents: 本地計算 intraday technical indicators（4H）

**背景**：EODHD `/api/technical/` 僅支援 daily interval；若要在 4H 上看 RSI/MACD/ATR 等，需要本地計算。

**責任範圍**：新增 `tradingagents/dataflows/eodhd_intraday_indicator.py`，並在 `interface.py` 註冊 `get_intraday_indicators`。

**支援指標（與 daily 對齊）**：

| 內部名稱 | 說明 |
|---|---|
| `close_50_sma` | SMA(50) |
| `close_200_sma` | SMA(200) |
| `close_10_ema` | EMA(10) |
| `rsi` | RSI(14) |
| `macd` / `macds` / `macdh` | MACD(12,26,9) |
| `boll` / `boll_ub` / `boll_lb` | Bollinger(20,2) |
| `atr` | ATR(14) |

**ATR 生產問題與修正（摘要）**：

- 在 prod 中觀察到 pandas `pd.concat(..., axis=1)` 在部分版本/索引狀態會觸發 duplicate-key/assemble 類錯誤
- Hotfix `22ff4e8` 將 ATR 的 TR 計算改為 numpy `column_stack + nanmax`，避免 concat 對 index/column 的敏感性

---

### 5.6 Task 6 — TradingAgents: market_analyst dual-timeframe prompt

**責任範圍**：更新 `tradingagents/agents/analysts/market_analyst.py` 的 FX prompt 與 analysis process，使其以「1D Trend + 4H Entry」作為決策骨架。

**核心 prompt 變更（概念摘要）**：

- 新增「Multi-Timeframe Analysis (CRITICAL for v1.3.5)」段落
- analysis process 從 daily-only 擴展為同時納入 intraday context：
  - daily OHLCV → daily indicators → intraday OHLCV（1H+4H）→ 4H indicators → timeframe agreement filter → synthesize

**策略規則（文本層）**：

- 若 1D trend 與 4H momentum/structure **同向**：允許給出 BUY/SELL entry 建議
- 若 **分歧**：偏向 HOLD/NEUTRAL

---

### 5.7 Task 7 — prop-firm-pilot: entry_timeframe hint passthrough

**責任範圍**：在 `src/signal/scanner_bridge.py` 讓 scanner signal 的 `to_qlib_data()` 帶上 `entry_timeframe` 欄位，作為 TradingAgents 的 hint。

```python
class ScannerSignal:
    def __init__(..., entry_timeframe: str = "4h") -> None:
        self.entry_timeframe = entry_timeframe

    def to_qlib_data(self) -> dict[str, Any]:
        return {
            ...,
            "entry_timeframe": self.entry_timeframe,
        }
```

**原因**：Scanner 仍以 1D 產生 trend signal；agent 端透過 `entry_timeframe` 決定是否拉取 intraday context（避免每次都多打 intraday API）。

---

### 5.8 Task 8 — 版本號一致性（All repos）

**責任範圍**：三倉庫版本號更新至 1.3.5，確保 prod 日誌與回溯一致。

**prop-firm-pilot 版本字串**：`src/main.py` 顯示 `PropFirmPilot v1.3.5 starting`（便於定位 prod log）。

---

## 6. 架構圖

### 6.1 Dual-Timeframe 數據流

```text
                           ┌──────────────────────┐
                           │       EODHD API       │
                           │  /api/eod/ (1D)       │
                           │  /api/intraday/ (1H)  │
                           └───────────┬──────────┘
                                       │
                     ┌─────────────────┼──────────────────┐
                     │                 │                  │
              ┌──────▼──────┐   ┌──────▼──────┐    ┌──────▼──────┐
              │qlib_market   │   │ TradingAgents│    │ prop-firm   │
              │ _scanner     │   │ (LLM decision)│   │  -pilot     │
              │ (Qlib Alpha) │   │              │    │ (scheduler) │
              └──────┬──────┘   └──────┬───────┘    └──────┬──────┘
                     │                 │                  │
     1D Trend Signal │         4H Entry Timing            │ Orchestration
     (IR=1.179)      │         (intraday tools)           │ + compliance
                     │                 │                  │
              ┌──────▼─────────────────▼──────────────────▼──────┐
              │  Dual-Timeframe Filter: 1D trend must agree 4H   │
              │  If divergence → HOLD (avoid counter-trend entry)│
              └──────────────────────────────────────────────────┘
```

### 6.2 TradingAgents intraday 工具結構（v1.3.5）

```text
TradingAgents/
└── tradingagents/
    ├── agents/utils/
    │   └── intraday_tools.py           # get_intraday_stock_data / get_intraday_indicators [新增]
    ├── dataflows/
    │   ├── eodhd_stock.py              # intraday fetch + 4H aggregation [修改]
    │   ├── eodhd_intraday_indicator.py # local indicators (4H) [新增]
    │   ├── interface.py                # VENDOR_METHODS + tool categories [修改]
    │   └── eodhd.py                    # re-export [修改]
    ├── agents/analysts/
    │   └── market_analyst.py           # dual-timeframe FX prompt [修改]
    ├── graph/
    │   └── trading_graph.py            # ToolNode includes intraday tools [修改]
    └── default_config.py               # enable intraday vendor categories [修改]
```

### 6.3 qlib_market_scanner intraday 管線結構（v1.3.5）

```text
qlib_market_scanner/
└── src/
    ├── data/
    │   └── fx_fetcher.py       # EODHD intraday fetch + aggregate_to_4h [修改]
    └── pipeline/
        ├── runner.py           # interval routing (1d/1h/4h) [修改]
        └── qlib_workflow.py    # _to_qlib_freq + intraday label/segments [修改]
```

---

## 7. EODHD Intraday API 規格

### 7.1 Endpoint

- **OHLCV (intraday)**: `GET /api/intraday/{SYMBOL}.FOREX?interval=1h`
- **無原生 4H**：需用 1H 在本地聚合成 4H

### 7.2 Request 參數

| 參數 | 型別 | 說明 |
|---|---|---|
| `api_token` | string | EODHD API key |
| `interval` | string | `1h`（本版本固定抓 1H，再聚合 4H） |
| `from` | int | Unix timestamp（UTC seconds） |
| `to` | int | Unix timestamp（UTC seconds） |
| `fmt` | string | `json` |

### 7.3 Response 格式（概念示意）

```json
[
  {
    "datetime": "YYYY-MM-DD HH:MM:SS",
    "gmtoffset": 0,
    "open": "<float>",
    "high": "<float>",
    "low": "<float>",
    "close": "<float>",
    "volume": "<number>"
  }
]
```

### 7.4 邊界條件（FX 特性）

| 問題 | 行為 | v1.3.5 對策 |
|---|---|---|
| 週末 / 假日 | 可能回 `open/high/low/close=None` bar | 下載端跳過 None OHLC bars（Hotfix `f6d6caf`） |
| FX volume | 可能為 0 | 保留欄位、視為合法 |
| 4H | 無原生 endpoint | 固定抓 1H，本地聚合 4H（UTC 00:00 anchor） |

### 7.5 成本估算（已提供資訊）

- **每次 request 成本**：~**5 API calls**
- **1H 最長歷史**：可達 **7200 days**

---

## 8. 修改檔案清單

本節列出 v1.3.5 相對於 v1.3.0 的關鍵變更檔案（含新增檔案行數）。表格格式沿用 v1.3.0 報告風格：以倉庫分組，列出檔案、動作、說明。

### TradingAgents

| 檔案 | 動作 | 說明 |
|---|---|---|
| `tradingagents/agents/utils/intraday_tools.py` | **新增** | LangChain tools：`get_intraday_stock_data` / `get_intraday_indicators`（73 行） |
| `tradingagents/dataflows/eodhd_intraday_indicator.py` | **新增** | intraday 技術指標本地計算（4H），含 SMA/EMA/RSI/MACD/Boll/ATR（250 行） |
| `tradingagents/dataflows/eodhd_stock.py` | **修改** | 新增 intraday OHLCV fetch + 1H→4H aggregation + dual output formatting（223 行） |
| `tradingagents/dataflows/interface.py` | **修改** | 新增 tool categories：`intraday_stock_apis` / `intraday_indicator_apis`；新增 `VENDOR_METHODS` 路由：`get_intraday_stock_data`、`get_intraday_indicators`（370 行） |
| `tradingagents/agents/analysts/market_analyst.py` | **修改** | FX prompt 新增 Multi-Timeframe Analysis + 更新 analysis process（342 行） |
| `tradingagents/agents/utils/agent_utils.py` | **修改** | FX 情境綁定 intraday tools（Hotfix `fa2834b` 覆蓋範圍）（369 行） |
| `tradingagents/graph/trading_graph.py` | **修改** | ToolNode 加入 intraday tools（Hotfix `fa2834b` 覆蓋範圍）（450 行） |
| `tradingagents/default_config.py` | **修改** | data_vendors 新增 intraday 類別並預設 `eodhd`；包含 Hotfix `1050cf8`（178 行） |
| `tradingagents/dataflows/eodhd.py` | **修改** | re-export 新增 intraday 函數（25 行） |
| `pyproject.toml` | **修改** | 版本號 → 1.3.5（37 行） |
| `uv.lock` | **修改** | lockfile 更新（22ff4e8 觸及）（5519 行） |

### qlib_market_scanner

| 檔案 | 動作 | 說明 |
|---|---|---|
| `src/data/fx_fetcher.py` | **修改** | `EODHDFXFetcher.fetch_intraday()`、`aggregate_to_4h()`、`download_universe_intraday()`；並跳過 None OHLC bars（Hotfix `f6d6caf`）（650 行） |
| `src/pipeline/runner.py` | **修改** | interval routing：`1h/4h` 走 intraday download；4H 下載後本地聚合覆寫；兼容 4h freq（Hotfix `3c0e0f4` 觸及）（421 行） |
| `src/pipeline/qlib_workflow.py` | **修改** | `_to_qlib_freq()` + `_QLIB_FREQ_MAP`（4h→day, 1h→1min）；intraday label 與 segment sizing（Hotfix `3c0e0f4`）（853 行） |
| `tests/test_fx_fetcher_intraday.py` | **新增** | intraday fetch / 4H 聚合 / download_universe_intraday 單元測試（212 行） |
| `pyproject.toml` | **修改** | 版本號 → 1.3.5（37 行） |

### prop-firm-pilot

| 檔案 | 動作 | 說明 |
|---|---|---|
| `src/signal/scanner_bridge.py` | **修改** | `ScannerSignal` 增加 `entry_timeframe` 並輸出至 `to_qlib_data()`，作為 TradingAgents intraday hint（303 行） |
| `src/main.py` | **修改** | 版本字串更新為 v1.3.5；並在 daily cycle 依 `scheduler.scanner_timeframe` 呼叫 scanner（591 行） |
| `config/e8_one_5k_challenge.yaml` | **修改** | 增加 `scanner_timeframe: "1d"` 與 `agent_timeframe: "4h"` 分離配置（101 行） |
| `src/config.py` | **修改** | 新增/調整 scheduler 相關 config 欄位以支援 timeframe 分離（379 行） |
| `src/scheduler/scheduler.py` | **修改** | scheduler pipeline 參數傳遞與 timeframe 分離 wiring（1660 行） |
| `docs/plans/2026-03-03-v1.3.5-eodhd-intraday-dual-timeframe.md` | **新增** | v1.3.5 實作計畫（447 行） |
| `pyproject.toml` | **修改** | 版本號 → 1.3.5（78 行） |

---

## 9. 測試覆蓋

v1.3.5 的驗證以 prod 測試為主，並在 qlib_market_scanner 補上 intraday 單元測試。

### 9.1 qlib_market_scanner（intraday 單元測試）

| 測試檔案 | 說明 |
|---|---|
| `tests/test_fx_fetcher_intraday.py` | `fetch_intraday()` 回傳資料格式、`aggregate_to_4h()` 聚合正確性、`download_universe_intraday()` CSV 輸出與錯誤處理 |

### 9.2 TradingAgents（prod 驗證為主）

| 類別 | 驗證重點 |
|---|---|
| intraday tools | `market_analyst` 可呼叫 `get_intraday_stock_data` / `get_intraday_indicators` 並拿到 4H 指標 |
| indicators | 本地計算 SMA/EMA/RSI/MACD/Boll/ATR 正常、ATR 不再出現 concat/duplicate-key 類錯誤（Hotfix `22ff4e8`） |

### 9.3 prop-firm-pilot（integration wiring）

| 類別 | 驗證重點 |
|---|---|
| scanner_bridge | `entry_timeframe` 能完整從 ScannerSignal 傳遞到 `qlib_data` |
| scheduler | scanner_timeframe / agent_timeframe 分離後，不影響既有 daily cycle 的穩定性 |

---

## 10. Prod 測試環境

| 項目 | 值 |
|---|---|
| 目標帳戶 | E8 One Challenge（FX） |
| 核心驗證目標 | intraday data + indicators 可在 prod 可靠取得；dual-timeframe divergence 可被識別並導向 HOLD |

---

## 11. Phase 1: qlib_scanner 1D 基準測試

**目標**：確認在引入 intraday 之前，1D Alpha158 pipeline 在 EODHD daily data 上仍維持良好表現。

**結果（已提供數據）**：

| Metric | 1D |
|---|---|
| Information Ratio | **1.179** |
| Win Rate | **54.9%** |
| Annualized Return | **~82%** |

**結論**：保留 scanner 固定 1D 作為 Trend Signal 的來源。

---

## 12. Phase 2: TradingAgents + Telegram 測試

**目標**：確認 TradingAgents 在 FX 情境下能取得 intraday OHLCV 與 4H 指標，並把 dual-timeframe 框架輸出為可執行的 decision。

**驗證重點（質性結果）**：

1. intraday indicators 可成功在本地計算：SMA/EMA/RSI/MACD/Bollinger/ATR
2. dual-timeframe 分析可偵測 1D bullish vs 4H bearish divergence → 推導為 HOLD
3. Telegram bot integration 可正常送出分析摘要（作為 prod observable signal）

---

## 13. Phase 3a: qlib_scanner 1D Backtest

**目標**：確認 1D backtest 全流程可運行（data download → qlib dump → training → signals）。

**結果**：1D backtest 可正常運行（作為 4H backtest 的對照組）。

---

## 14. Phase 3b: qlib_scanner 4H Backtest

**目標**：量化檢驗 Qlib Alpha158 在 4H freq 上的效果，並與 1D 做對照。

**關鍵結果（已提供數據）**：

| Metric | 1D | 4H |
|---|---:|---:|
| Information Ratio | **1.179** | **0.299** |
| Win Rate | **54.9%** | **54.25%** |
| Annualized Return | **~82%** | **4.86%** |
| Max Drawdown | — | **-22.76%** |
| Low Confidence % | — | **30.2%** |

**結論**：4H 不適合作為 Qlib 主 pipeline；應作為 TradingAgents 的 entry timing context。

---

## 15. 效益分析與時間框架建議

本節以 Q0–Q4 的形式整理可行建議與原因，並補充工程約束（Qlib freq 限制、FX universe 規模等）。

### 15.1 關鍵發現（Q0–Q4）

| 編號 | 結論 | 建議 |
|---|---|---|
| Q0 | Execution layer 以 EODHD WebSocket 具潛在優勢 | 未納入本版；作為未來工作（見第 17 節） |
| Q1 | Qlib 4H 顯著劣於 1D | Scanner 固定 1D；不建議以 Qlib 4H 做主模型 |
| Q2 | TradingAgents 4H 有助於決策品質 | 以 4H indicators 檢測 divergence 與 timing |
| Q3 | Hybrid 1D+4H 最合理 | 1D（方向）+ 4H（進場）雙框架一致才入場 |
| Q4 | YAML 應分離 scanner_timeframe 與 agent_timeframe | 在 account config 中提供 `scanner_timeframe` / `agent_timeframe`（見第 8 節） |

### 15.2 為何 4H 在 Qlib 表現差（工程與統計層原因）

1. **Alpha158 因子模型設計初衷偏向 daily equity**：4H 對 FX 的 microstructure 變化更敏感，且因子穩定性下降。
2. **FX universe 規模過小**：僅少量 FX instruments 時，cross-sectional ranking 在 4H 更難形成穩定排序。
3. **低信心比例偏高**：4H signals low confidence % 為 **30.2%**，表示模型對 4H 分布分離度不足。
4. **Qlib freq 限制**：Qlib API 不支援 4H，因此以「4H bars treated as day」方式兼容；這是可運行但非嚴格的 4H 日曆語義。

### 15.3 建議的配置範式（scanner vs agent 分離）

```yaml
# config/e8_one_5k_challenge.yaml (excerpt)
scheduler:
  # v1.3.5: 雙時間框架分離 — Scanner 固定 1D (Alpha158 最佳), Agent 使用 4H 進場
  scanner_timeframe: "1d"
  agent_timeframe: "4h"
```

---

## 16. 生產環境 Hotfixes

本節僅列出使用者指定的 5 個 Hotfix commit hash（不得新增其他 hash），並說明其 root cause 與修復範圍。

| # | Commit | 類型 | Root Cause | 修復摘要 | 主要影響檔案 |
|---:|---|---|---|---|---|
| 1 | `fa2834b` | TradingAgents | intraday tools 已存在但未綁定到 market_analyst / ToolNode | 綁定 intraday tools 到 FX 分析流程，確保 agent 可調用 | `agent_utils.py`, `trading_graph.py`, `market_analyst.py`, `default_config.py`, `intraday_tools.py` |
| 2 | `22ff4e8` | TradingAgents | `get_intraday_indicators` 未傳 `interval=4h`，且 ATR 計算在 prod 觸發 pandas concat 類錯誤 | 補上 `"4h"` 參數；ATR 改 numpy 寫法避免 duplicate-key/concat 問題 | `intraday_tools.py`, `eodhd_intraday_indicator.py` |
| 3 | `1050cf8` | TradingAgents | 預設 vendor 仍可能偏向 Alpha Vantage | 將 primary data vendor 切換為 `eodhd`（配置層） | `default_config.py` |
| 4 | `f6d6caf` | qlib_market_scanner | EODHD intraday 週末 bar 可能回 None OHLC 導致解析/聚合問題 | intraday fetch 跳過 None OHLC bars | `src/data/fx_fetcher.py` |
| 5 | `3c0e0f4` | qlib_market_scanner | Qlib API 不支援 `4h` freq | 將 4H freq map 到 `day` 供 Qlib API 呼叫（工程兼容） | `src/pipeline/qlib_workflow.py`, `src/pipeline/runner.py` |

---

## 17. 已知限制與未來工作

### 17.1 已知限制

1. **Qlib 不真正支援 4H calendar**：目前以「4H bars treated as day」方式兼容，適合做工程驗證，但不保證統計語義嚴格。
2. **intraday features 的 unit tests 覆蓋不足**：TradingAgents 的 intraday indicators 以 prod 測試為主，缺少完整 deterministic 單元測試（可補強）。
3. **EODHD intraday 週末資料不完整**：需持續依賴 defensive parsing（None OHLC skip）。
4. **Config wiring 風險**：在 prod wiring 上需確保 ScannerBridge init 與 run_pipeline 的參數（scanner_timeframe / agent_timeframe / entry_timeframe）在所有入口一致傳遞；此類 wiring bug 需在後續 commit 持續收斂。

### 17.2 未來工作（不在本版範圍）

| 項目 | 說明 | 目的 |
|---|---|---|
| EODHD WebSocket | 以 WebSocket 取代部分 REST polling | 降低延遲、降低 API 成本、提升 execution timing |
| TradingAgents intraday tests | 增加固定輸入 OHLCV 的指標單元測試 | 降低 prod 才暴露的指標計算 bug 風險 |
| 更合理的 4H backtest framework | 若需 4H 量化回測，評估替代框架或自建 calendar | 避免 Qlib freq mapping 的語義偏差 |

---

## 18. 相依性變更

| 類別 | 變更 |
|---|---|
| Python 依賴 | 無新增相依性（pandas/numpy/loguru 已存在） |
| 外部服務 | 仍使用 EODHD（REST）；未新增 WebSocket 依賴 |

---

## 19. Git Commit 記錄

本節列出使用者指定的 8 個 commits（不得增加/修改 hash），作為 v1.3.5 的可追溯記錄。

| Repo | Commit | Description |
|---|---|---|
| qlib_market_scanner | `13d0311` | v1.3.5 intraday fetch + 4H aggregation + runner routing + Qlib workflow freq |
| TradingAgents | `d78ae57` | v1.3.5 intraday stock API + EODHD indicator local compute + market_analyst dual-TF |
| prop-firm-pilot | `5031bc3` | v1.3.5 scanner_bridge entry_timeframe param |
| TradingAgents | `fa2834b` | Hotfix: intraday tool binding |
| TradingAgents | `22ff4e8` | Hotfix: ATR numpy rewrite + 4h arg |
| TradingAgents | `1050cf8` | Hotfix: data vendor switch to EODHD |
| qlib_market_scanner | `f6d6caf` | Hotfix: skip None OHLC bars |
| qlib_market_scanner | `3c0e0f4` | Hotfix: Qlib 4h freq compatibility |

---

> **Part A–D 報告結束** — PropFirmPilot v1.3.5（EODHD Intraday Dual-Timeframe: 1D Trend + 4H Entry）

---

## 20. 18 小時 Prod 運行評估

> **評估日期**: 2026-03-04
> **評估範圍**: v1.3.5 於 E8 One Challenge 帳戶的首次 18 小時生產運行（2026-03-03 00:52 UTC → 2026-03-04 約 07:00 UTC）
> **日誌來源**: `prod_log_results/03032026_to_04032026_v1_3_5/`

v1.3.5 部署至 E8 One Challenge 帳戶後，首次連續運行 ~18 小時。透過全面分析生產日誌、交易記錄、與即時帳戶狀態，共識別出 **11 項問題**（3 Critical、3 High、3 Medium、2 Low）。本 Part E 記錄評估結果與所有修復。

### 20.1 運行環境

| 項目 | 值 |
|---|---|
| 目標帳戶 | E8 One Challenge (Account #950383) |
| API | MatchTrader REST (`mtr.e8markets.com`) |
| 運行模式 | Scheduler（24/7 async pipeline） |
| 配置檔 | `config/e8_one_5k_challenge.yaml` |
| Scanner 時間框架 | 1D（Qlib Alpha158） |
| Agent 時間框架 | 4H（TradingAgents） |
| 監控幣對 | EURUSD, GBPUSD, USDJPY |

### 20.2 運行結果概要

| 指標 | 值 |
|---|---|
| 總運行時間 | ~18 小時 |
| Scanner 循環次數 | ~12 次（其中 11 次為冗餘） |
| LLM 決策次數 | 多次 BUY/SELL 決策 |
| 實際開倉 | 2 筆（USDJPY SELL、EURUSD SELL） |
| 合規拒絕次數 | 多次（觸發 C3 無限重試） |
| Telegram 409 錯誤 | 持續發生（M1） |

---

## 21. 問題清單與修復總覽

| # | 優先級 | 問題 | 修復狀態 | 修復方式 |
|---:|---|---|---|---|
| C1 | **Critical** | HOLD 決策被錯誤映射為 BUY | ✅ 已修復 | risk_report 交叉驗證 |
| C2 | **Critical** | LLM 拒絕回應被映射為 SELL | ✅ 已修復 | 拒絕模式檢測 |
| C3 | **Critical** | 合規拒絕後無限重試循環 | ✅ 已修復 | 冷卻期機制 |
| H1 | **High** | exit_reason 分類錯誤 | ✅ 已修復 | Broker API 重試 + PnL 推斷 |
| H2 | **High** | Scanner 每次重跑浪費 ~10 分鐘 | ✅ 已修復 | Pipeline cache 智能跳過 |
| H3 | **High** | 閾值正反饋死循環 | ✅ 已修復 | 不活躍符號閾值衰減 |
| M1 | **Medium** | Telegram 409 Conflict 持續報錯 | ✅ 已修復 | 指數退避 |
| M2 | **Medium** | 平倉記錄 PnL=0.0 | ✅ 由 H1 覆蓋 | 同 H1 |
| M3 | **Medium** | LLM SELL 偏見 | 📋 已記錄 | 模型行為，非程式碼 bug |
| L1 | **Low** | Log 噪音累積 | ✅ 已存在 | rotation 10 MB + retention 30 天 |
| L2 | **Low** | Alpha158 因子頻率不適配 | 📋 已記錄 | 研究/配置問題 |

---

## 22. Critical 修復 (C1–C3)

### 22.1 C1 — HOLD 決策被錯誤映射為 BUY

**Root Cause**: TradingAgents 的 `propagate()` 回傳值可能與 `risk_report` 內容矛盾。Production 日誌中觀察到 risk_report 明確建議 HOLD，但 `propagate()` 回傳了 BUY。三個來源出現三種不同值：trader=SELL, risk_report=HOLD, propagate()=BUY。

**修復**: 在 `AgentBridge` 新增 `validate_decision()` 方法，實施 risk_report 交叉驗證：

1. 以正則表達式 `_PROPOSAL_RE` 從 risk_report 提取最終建議（HOLD/BUY/SELL）
2. 當 risk_report 建議與 `propagate()` 回傳值矛盾時，**以 risk_report 為準**
3. 若 risk_report 明確建議 HOLD，強制覆寫決策為 HOLD

**核心程式碼**:

```python
# src/decision/agent_bridge.py
_PROPOSAL_RE = re.compile(
    r'(?:final\s+(?:trading\s+)?(?:proposal|recommendation|decision)|recommendation)\s*:?\s*(BUY|SELL|HOLD)',
    re.IGNORECASE,
)
```

### 22.2 C2 — LLM 拒絕回應被映射為 SELL

**Root Cause**: GPT-5.2 偶爾回傳拒絕回應（如「我無法依照你的要求提供」），但 `propagate()` 仍回傳 SELL，系統將其視為有效決策。

**修復**: 在 `validate_decision()` 中新增 LLM 拒絕模式檢測：

```python
# src/decision/agent_bridge.py
_REFUSAL_PATTERNS = [
    re.compile(r'(?:I\s+cannot|I\'m\s+unable|I\s+can\'t).*(?:provide|recommend|suggest)', re.I),
    re.compile(r'(?:無法|不能).*(?:提供|建議|推薦)', re.I),
    re.compile(r'(?:I\s+do\s+not|I\s+don\'t).*(?:have\s+enough|recommend)', re.I),
]
```

當任一拒絕模式匹配 risk_report 時，強制決策為 HOLD 並記錄原因。

### 22.3 C3 — 合規拒絕後無限重試循環

**Root Cause**: Scanner 產生信號 → LLM 決策 BUY → 合規檢查拒絕 → 下一次循環重複同樣流程 → 永無止境地消耗 LLM API credits。

**修復**:

1. 在 `DecisionStore`（SQLite）新增 `has_recent_rejection(symbol, minutes)` 方法
2. 在 `SchedulerConfig` 新增 `rejection_cooldown_minutes: int = 120`
3. Scheduler 在處理信號前檢查該 symbol 是否在冷卻期內
4. 冷卻期內的信號直接跳過，避免重複呼叫 LLM

```python
# src/config.py
rejection_cooldown_minutes: int = Field(
    default=120,
    description="Minutes to wait after a compliance rejection before retrying the same symbol."
)
```

---

## 23. High 修復 (H1–H3)

### 23.1 H1 — exit_reason 分類錯誤

**Root Cause**: 當倉位被 TP/SL 觸發自動平倉後，Broker API 需要短暫時間處理。系統在 2 秒內查詢，若 API 尚未回傳已平倉的 position，則 `exit_reason` 預設為 `manual_close`。這不是真正的手動平倉，而是 API 延遲的假陽性。

**修復**:

1. **3 次重試機制**：以指數退避（2s → 4s → 8s）重試 Broker API 查詢
2. **PnL 推斷**：若重試後仍未拿到 position 資料，根據 PnL 推斷：
   - PnL > 0 → `tp_hit`（止盈觸發）
   - PnL < 0 → `sl_hit`（止損觸發）
   - PnL = 0 → `manual_close`（保留預設）
3. **`_last_known_profit` 備援**：若 PnL 為 0，嘗試從最後已知利潤推斷
4. **best_day / reevaluation 平倉路徑**也加入相同的 PnL 備援邏輯

> **驗證**：透過 MatchTrader API 即時查詢 E8 One 帳戶，確認所有在倉倉位均有 SL 和 TP。H1 的「無 SL/TP」部分為**假陽性**。

### 23.2 H2 — Scanner 冗餘重跑

**Root Cause**: FX 使用 Qlib 1D（daily）模型，信號只在每日 K 線收盤（~17:00 UTC）後才更新。但 Scheduler 每 4 小時觸發一次 scanner pipeline（含 ~10 分鐘的 retrain），在日內重跑完全是浪費。

**Production 證據**: 03-03 00:52–10:02 期間使用 02-27 信號（11 次冗餘運行）。新信號直到 17:39 才出現。

**修復**: 新增 `_PipelineCache` dataclass：

```python
# src/signal/scanner_bridge.py
@dataclass
class _PipelineCache:
    """Cache for pipeline results keyed by (request_date, interval)."""
    request_date: str
    interval: str
    signals: list
    timestamp: float
```

- Cache key 為 `(request_date, interval)`
- 同一 key 命中時直接回傳快取結果，跳過 retrain
- `load_signals_from_file()` 回傳型別改為 `tuple[list, bool]`（signals + is_cached flag）
- ~18 個測試呼叫點同步更新

### 23.3 H3 — 閾值正反饋死循環

**Root Cause**: `OptimizationEngine.refresh_state()` → `compute_thresholds()` 中的 per-symbol 閾值調整形成正反饋死循環：

1. 全局 win_rate < 0.45 → `min_blended_confidence = 0.65`
2. Per-symbol win rate 更差 → `_adjust_blended()` 加 0.05 → 閾值升至 **0.70**
3. Blended confidence ~0.516 < 0.70 → 信號永遠被過濾
4. 無交易 → 無新資料 → 閾值維持高位 → **死循環**

**修復**: 對不活躍符號的閾值調整引入時間衰減：

```python
# src/optimize/thresholds.py
# 衰減公式：adj 在 3 天內線性歸零
decay_factor = max(0.0, 1.0 - inactive_days / 3.0)
adj_decayed = adj * decay_factor
```

- `trade_stats.py`: 新增 `compute_inactive_days()` 計算每個 symbol 距上次交易的天數
- `optimization_engine.py`: 在 `refresh_state()` 中計算 `inactive_days` 並傳入 `compute_thresholds()`
- 僅對 `adj < 0`（表現較差 → 閾值被抬高）的情況進行衰減
- 3 天無交易後閾值調整完全歸零，恢復為全局基準值

---

## 24. Medium 修復 (M1–M3)

### 24.1 M1 — Telegram 409 Conflict 持續報錯

**Root Cause**: `TelegramBotHandler` 使用 `getUpdates` 長輪詢。當兩個 process 同時 poll 同一個 bot token 時，Telegram API 回傳 409 Conflict。這是部署層面問題（如同時運行兩個 scheduler 實例）。

**修復**: 在 `_poll_updates()` 新增 409 專屬檢測與指數退避：

```python
# src/monitor/telegram_bot.py
if response.status_code == 409:
    self._conflict_backoff = min(max(self._conflict_backoff * 2, 5.0), 120.0)
    logger.warning(
        "TelegramBotHandler: 409 Conflict — another instance is polling"
        " this bot token. Backing off {:.0f}s", self._conflict_backoff,
    )
    return []
```

- 初始退避 5 秒，每次翻倍，上限 120 秒
- 成功後立即重置退避為 0

### 24.2 M2 — 平倉記錄 PnL=0.0

**已由 H1 修復覆蓋**。H1 的 3 次重試 + PnL 推斷 + `_last_known_profit` 備援邏輯解決了 PnL 為 0 的問題。不需要額外的程式碼修改。

### 24.3 M3 — LLM SELL 偏見

**已記錄，非程式碼 bug**。Production 觀察到 LLM（尤其 GPT-5.2）在 FX 分析時傾向於給出 SELL 建議。這屬於模型行為偏差，需要透過 prompt engineering 或模型選擇來處理，不在本次修復範圍。

---

## 25. Low 項目 (L1–L2)

### 25.1 L1 — Log 噪音累積

**已存在完整實現**。檢查發現 `setup_logging()` 函數（`src/main.py` 第 523–540 行）已配置完善的 loguru file handler：

```python
# src/main.py
logger.add(
    config.logging.file,         # logs/prop_firm_pilot.log
    level=config.logging.level,   # INFO
    rotation=config.logging.rotation,  # 10 MB
    retention=config.logging.retention, # 30 days
    encoding="utf-8",
)
```

對應 YAML 配置：

```yaml
# config/default.yaml
logging:
  level: "INFO"
  file: "logs/prop_firm_pilot.log"
  rotation: "10 MB"
  retention: "30 days"
```

不需要額外修改。

### 25.2 L2 — Alpha158 因子頻率不適配

**已記錄，研究/配置問題**。Qlib Alpha158 因子模型設計初衷偏向 daily equity，在 4H FX 上表現顯著下降（IR 從 1.179 降至 0.299）。這與第 15 節的分析一致：Scanner 應維持 1D，4H 僅作為 TradingAgents 的 entry timing context。

---

## 26. 修復檔案清單

### prop-firm-pilot（Bug Fix commits）

| 檔案 | 動作 | 修復項 | 說明 |
|---|---|---|---|
| `src/decision/agent_bridge.py` | **修改** | C1+C2 | `validate_decision()` + `_REFUSAL_PATTERNS` + `_PROPOSAL_RE` |
| `src/decision_store/sqlite_store.py` | **修改** | C3 | `has_recent_rejection(symbol, minutes)` 方法 |
| `src/scheduler/scheduler.py` | **修改** | C3, H1 | 拒絕冷卻期檢查 + exit_reason 重試與 PnL 推斷 |
| `src/config.py` | **修改** | C3 | `rejection_cooldown_minutes` 欄位 |
| `src/signal/scanner_bridge.py` | **修改** | H2 | `_PipelineCache` + `load_signals_from_file()` tuple 回傳 |
| `src/optimize/thresholds.py` | **修改** | H3 | `compute_thresholds()` 不活躍符號衰減邏輯 |
| `src/optimize/trade_stats.py` | **修改** | H3 | `compute_inactive_days()` 函數 |
| `src/optimize/optimization_engine.py` | **修改** | H3 | `refresh_state()` 串接 inactive_days |
| `src/monitor/telegram_bot.py` | **修改** | M1 | 409 Conflict 指數退避 |

### TradingAgents（Encoding Fix）

| 檔案 | 動作 | 修復項 | 說明 |
|---|---|---|---|
| `tradingagents/graph/trading_graph.py` | **修改** | Encoding | `_log_state()` 加入 `encoding="utf-8"` + `ensure_ascii=False` |

---

## 27. 測試覆蓋（Bug Fix）

| 測試檔案 | 修復項 | 測試數 | 說明 |
|---|---|---|---|
| `tests/test_agent_bridge_decision_validation.py` | C1+C2 | 18 | risk_report 交叉驗證 + LLM 拒絕檢測 |
| `tests/test_rejection_cooldown.py` | C3 | 8 | 冷卻期機制完整覆蓋 |
| `tests/test_exit_reason_classification.py` | H1 | 12 | exit_reason 重試 + PnL 推斷 |
| `tests/test_scheduler.py` | H1 | 2（更新） | 既有測試適配 H1 改動 |
| `tests/test_scanner_bridge.py` | H2 | 8（新增）+ 18（更新） | Pipeline cache + load_signals_from_file 呼叫點 |
| `tests/optimize/test_threshold_decay.py` | H3 | 12 | 不活躍符號衰減完整覆蓋 |
| `tests/optimize/test_thresholds.py` | H3 | 24（通過） | 既有閾值測試回歸驗證 |
| `tests/optimize/test_scheduler_thresholds.py` | H3 | 3（通過） | Scheduler 閾值整合測試 |

---

## 28. Git Commit 記錄（Bug Fix）

本節列出 v1.3.5 生產評估後的 Bug Fix commits，按修復優先級排序。

### prop-firm-pilot（8 commits）

| Commit | Description |
|---|---|
| `be0263d` | fix(C1+C2): add risk_report cross-validation and LLM refusal detection in AgentBridge |
| `2f3b8d2` | fix(C3): add compliance rejection cooldown to prevent infinite retry loops |
| `64d9e9c` | chore: clean up unused imports in C1+C2 test file |
| `959c91d` | chore: remove obsolete e8_trial and e8_signature config files |
| `91e6e3e` | fix(H1): add broker API retry with exponential backoff and PnL-based exit_reason re-inference |
| `6e08d8e` | fix(H2): add pipeline cache to skip redundant scanner reruns when daily candle unchanged |
| `e634e2e` | fix(H3): add threshold decay for inactive symbols to break positive feedback dead loop |
| `b76afb8` | fix(M1): add exponential backoff for Telegram 409 Conflict errors |

### TradingAgents（1 commit）

| Commit | Description |
|---|---|
| `74b394b` | fix: write JSON state logs with UTF-8 encoding and ensure_ascii=False |

---

## Part F — Telegram 連線穩定性修復

## 29. Telegram 連線問題背景

部署 Part E 修復後，生產環境出現持續的 Telegram 連線失敗：

```
ERROR    | ConnectTimeout
ERROR    | AlertService: failed to send Telegram message:
ERROR    | getUpdates failed:
```

### 根因分析

| 項目 | 說明 |
|---|---|
| **現象** | `AlertService.send()` 和 `TelegramBotHandler._poll_updates()` 每次呼叫都等待 10 秒後逾時 |
| **錯誤訊息** | `ConnectTimeout` 無內容（httpcore 的 `ConnectTimeout` class 本身不帶 message body） |
| **根因** | ISP/企業網路封鎖 Telegram IP 範圍 (`149.154.x.x`) |
| **驗證** | Mac 伺服器和 Windows 開發機均無法 curl 到 `api.telegram.org`（IPv4/IPv6 皆逾時）；VPN 和 5G 行動網路正常 |
| **影響** | 不影響交易流程（`AlertService.send()` 內部 catch 所有異常），但造成每次呼叫 10 秒延遲 + 日誌洪水 |

### 三個層面的修復

| # | 修復 | 目的 |
|---|---|---|
| 1 | 持久 HTTP 客戶端 | 消除每次請求重建連線的開銷 |
| 2 | 錯誤診斷改進 | 讓 ConnectTimeout 顯示有意義的資訊而非空字串 |
| 3 | Circuit Breaker 自動降級 | 網路不可達時立即跳過，避免 10 秒延遲和日誌洪水 |

---

## 30. 持久 HTTP 客戶端修復

### 問題

原始實作中，`AlertService` 和 `TelegramBotHandler` 每次發送請求時都建立新的 `httpx.AsyncClient`，並在請求完成後關閉。這導致：
- 每次 HTTP 請求都需要重新建立 TCP 連線（熱路復用不可能）
- 連線失敗時無法從連線池快速重試
- `bot_handler.stop()` 未被 `await`，導致 `RuntimeWarning: coroutine 'stop' was never awaited`

### 修復內容

**TelegramBotHandler** (`src/monitor/telegram_bot.py`):
- `start()` 時建立持久 `httpx.AsyncClient(timeout=30.0)`
- `stop()` 改為 `async def`，關閉時清理客戶端
- `_poll_updates()` 和 `_send_message()` 共用同一客戶端實例

**AlertService** (`src/monitor/alert_service.py`):
- 新增 `_get_client()` lazy init 方法，首次使用時建立 `httpx.AsyncClient(timeout=10.0)`
- 新增 `async close()` 清理方法
- `send()` 內的 `async with httpx.AsyncClient()` 替換為 `self._get_client()`

**main.py** (`src/main.py`):
- `await bot_handler.stop()`（原為 bare call）
- `await alert_service.close()` 加入 shutdown 流程

### 錯誤診斷改進

```python
# 修復前：ConnectTimeout 顯示空字串
logger.error("failed: {}", e)  # 輸出: "failed: "

# 修復後：顯示異常類型名稱
logger.error("failed: {} - {}", type(e).__name__, e)  # 輸出: "failed: ConnectTimeout - "
```

---

## 31. Circuit Breaker 自動降級機制

### 設計理念

當 Telegram API 不可達時，每次嘗試發送都會等待 10 秒才逾時。Circuit Breaker 模式在連續失敗後「斷開」電路，立即跳過發送，避免無謂的延遲和日誌雜訊。

### 狀態機過渡

```
CLOSED (正常)
  │
  │ 連續 3 次失敗
  ▼
OPEN (降級)
  │
  │ 300秒後允許 1 次探針請求
  ▼
HALF-OPEN (探針)
  │
  ├─ 成功 → CLOSED
  └─ 失敗 → OPEN
```

### AlertService Circuit Breaker

```python
# 新增欄位
_consecutive_failures: int = 0
_circuit_open: bool = False
_circuit_opened_at: float = 0.0
_CB_FAILURE_THRESHOLD: int = 3      # 連續失敗次數閾值
_CB_RETRY_INTERVAL: float = 300.0   # 探針間隔（秒）
```

**`send()` 流程**:
1. 檢查 circuit 狀態
2. 若 OPEN 且未過探針間隔 → `return False`（無 HTTP、無延遲）
3. 若已過探針間隔 → 允許一次請求（HALF-OPEN）
4. 成功 → `_consecutive_failures = 0`, `_circuit_open = False`
5. 失敗 → `_consecutive_failures += 1`，若遞增至閾值 → OPEN

### TelegramBotHandler Circuit Breaker

獨立於 AlertService，擁有自己的 circuit breaker 狀態：

```python
_cb_consecutive_failures: int = 0
_cb_circuit_open: bool = False
_cb_circuit_opened_at: float = 0.0
_CB_FAILURE_THRESHOLD: int = 3
_CB_RETRY_INTERVAL: float = 300.0
```

**行為差異**:
- 當 circuit OPEN 時，sleep `_CB_RETRY_INTERVAL` 秒而非每秒輪詢
- 409 Conflict 使用自己的指數退避機制，**不觸發** circuit breaker
- 進入 OPEN 時 log `warning`，恢復時 log `info`

### 409 Conflict 獨立性

```python
# 409 有專屬退避機制（Part E M1 修復），不計入 CB 失敗次數
if response.status_code == 409:
    self._conflict_backoff = min(self._conflict_backoff * 2, 120.0)
    # 不增加 _cb_consecutive_failures
    continue
```

這確保當另一個程序佔用 bot token 時（409），不會誤判為網路不可達而開啟 CB。

---

## 32. 修復檔案清單（Telegram）

| 檔案 | 變更 |
|---|---|
| `src/monitor/alert_service.py` | 持久 httpx 客戶端 + circuit breaker |
| `src/monitor/telegram_bot.py` | 持久 httpx 客戶端 + circuit breaker + 409 獨立性 |
| `src/main.py` | `await bot_handler.stop()` + `await alert_service.close()` |
| `tests/test_alert_service.py` | 6 持久客戶端測試 + 11 CB 測試 + 1 更新 |
| `scripts/test_telegram_bot_live.py` | `await bot.stop()` |

---

## 33. 測試覆蓋（Telegram）

### 新增測試統計

| 測試類別 | 數量 | 涵蓋範圍 |
|---|---|---|
| `TestAlertServicePersistentClient` | 6 | lazy init、重用、清理、noop close、發送成功/失敗 |
| `TestAlertServiceCircuitBreaker` | 7 | N 次失敗後開啟、開啟時跳過、探針間隔、成功恢復、non-200 失敗、重置計數、停用時忽略 |
| `TestTelegramBotCircuitBreaker` | 4 | 失敗後開啟、sleep 取代 polling、成功恢復、409 不觸發 CB |
| 更新測試 | 1 | `test_stop_sets_flag` 改為 `await` |
| **總計** | **18** | |

### 執行結果

```
tests/test_alert_service.py: 68 passed ✅
（原 51 + 新增 17）
```

---

## 34. Git Commit 記錄（Telegram）

### prop-firm-pilot（3 commits）

| Commit | Description |
|---|---|
| `7c8712c` | fix: use persistent httpx client for Telegram polling and improve error diagnostics |
| `26eb06a` | fix: use persistent httpx client in AlertService and await bot_handler.stop() |
| `7229b13` | feat: add circuit breaker auto-degradation for Telegram connectivity failures |

---

> **報告結束** — PropFirmPilot v1.3.5（含 Part E：18 小時生產運行評估與修復 9 commits + Part F：Telegram 連線穩定性修復 3 commits，共 12 commits across 2 repos）
