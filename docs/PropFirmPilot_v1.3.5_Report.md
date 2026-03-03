
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

> **報告結束** — PropFirmPilot v1.3.5（EODHD Intraday Dual-Timeframe: 1D Trend + 4H Entry）
