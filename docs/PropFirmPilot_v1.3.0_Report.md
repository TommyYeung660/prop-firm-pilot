# PropFirmPilot v1.3.0 — EODHD 數據源遷移版本報告

> **報告日期**: 2026-03-02  
> **版本**: v1.3.0（EODHD Data Source Migration）  
> **基準版本**: v1.2.0（`ee2afd1`，排程優化 + DST 自動適配 + Scanner 修復）  
> **聚焦範圍**: 將三個倉庫的主要金融數據源從 Alpha Vantage 遷移至 EODHD，支援日期感知自動切換

---

## 目錄

1. [版本摘要](#1-版本摘要)
2. [版本資訊](#2-版本資訊)
3. [遷移背景](#3-遷移背景)
4. [功能總覽](#4-功能總覽)
5. [功能詳述](#5-功能詳述)
6. [架構圖](#6-架構圖)
7. [數據源對照表](#7-數據源對照表)
8. [Switchover 機制](#8-switchover-機制)
9. [組態參考](#9-組態參考)
10. [修改檔案清單](#10-修改檔案清單)
11. [測試覆蓋](#11-測試覆蓋)
12. [已知限制與未來工作](#12-已知限制與未來工作)
13. [相依性變更](#13-相依性變更)

---

## 1. 版本摘要

v1.3.0 完成了從 **Alpha Vantage** 到 **EODHD (eodhistoricaldata.com)** 的數據源遷移。Alpha Vantage 訂閱於 2026-03-21 到期，同日切換至 EODHD 付費方案（EOD + Intraday All World Extended, $29.99/月）。

此版本的核心設計原則是 **零停機遷移**：

- **日期感知切換**：`EODHD_SWITCHOVER_DATE`（預設 `2026-03-21`）控制主數據源；切換前使用 Alpha Vantage，切換後使用 EODHD
- **環境變數強制覆寫**：`EODHD_FORCE_PRIMARY=1` 可在任何日期強制使用 EODHD（用於測試）
- **Fallback 保護**：TradingAgents 的 `route_to_vendor()` 機制確保 EODHD 失敗時自動降級至其他 vendor
- **三倉庫統一版本號**：TradingAgents、qlib_market_scanner、prop-firm-pilot 同步升級至 1.3.0

### 改動規模

| 倉庫 | 新增檔案 | 修改檔案 | 新增測試 |
|---|---|---|---|
| TradingAgents | 7 modules + 7 test files | 3 files | 52 tests (47 unit + 5 integration) |
| qlib_market_scanner | 1 module + 3 test files | 4 files | 12 tests (8 unit + 4 integration) |
| prop-firm-pilot | 1 test file | 3 files | 4 tests |

**總計**: 8 新模組、10 新測試檔、10 修改檔、68 新測試

---

## 2. 版本資訊

| 項目 | 值 |
|---|---|
| 版本號 | v1.3.0 |
| 發佈日期 | 2026-03-02 |
| 基準版本 | v1.2.0（`ee2afd1`） |
| Switchover 日期 | 2026-03-21（可配置） |
| EODHD 方案 | EOD + Intraday All World Extended ($29.99/月) |
| EODHD API 額度 | 100,000 calls/day（付費）、20 calls/day（免費測試） |

### 跨倉庫版本統一

| 倉庫 | 之前版本 | v1.3.0 |
|---|---|---|
| prop-firm-pilot | v1.2.0 | v1.3.0 |
| TradingAgents | 0.1.0 | 1.3.0 |
| qlib_market_scanner | 0.0.1 | 1.3.0 |

---

## 3. 遷移背景

### 為何遷移

1. **Alpha Vantage 不支援 FX 小時線**：v1.2.0 新增的多時間框架分析需要 1H/4H 數據，AV 的 FX 端點僅提供日線
2. **API 速率限制**：AV 免費方案 5 calls/min，付費方案仍受限嚴重；EODHD 付費方案 100,000 calls/day
3. **成本效益**：EODHD $29.99/月涵蓋 EOD + Intraday + 技術指標 + 新聞 + 基本面，AV 同等覆蓋需 $49.99/月+
4. **數據品質**：EODHD 覆蓋 70+ 交易所、150,000+ 股票、FX、加密貨幣

### 遷移策略：方案 A（純 EODHD）

| 數據類型 | 遷移前（AV） | 遷移後（EODHD） |
|---|---|---|
| FX/Stock OHLCV | AV `FX_DAILY` / `TIME_SERIES_DAILY` | EODHD `/api/eod/` |
| FX Intraday | ❌ 不可用 | EODHD `/api/intraday/?interval=1h` |
| 技術指標 | AV `RSI`, `SMA`, etc. | EODHD `/api/technical/` |
| 新聞/情緒 | AV `NEWS_SENTIMENT` | EODHD `/api/news` + `/api/sentiments` |
| 基本面 | AV（少用） | EODHD `/api/fundamentals/` |
| 內部交易 | finnhub（本地） | 保持不變（本地） |
| 期權 | AV `HISTORICAL_OPTIONS` | 保持 OpenBB/yfinance（未來再遷移） |

---

## 4. 功能總覽

### 4.1 TradingAgents — EODHD Vendor 模組（Tasks 1-6）

| 模組 | 檔案 | 功能 |
|---|---|---|
| **Common Utilities** | `eodhd_common.py` | API key 管理、符號轉換（`EURUSD` ↔ `EURUSD.FOREX`）、HTTP helper、日期過濾、Rate Limit 處理 |
| **Stock/FX Data** | `eodhd_stock.py` | 每日 OHLCV 數據（FX + 股票），CSV 格式輸出，與 AV `get_stock()` 介面一致 |
| **Technical Indicators** | `eodhd_indicator.py` | SMA、EMA、RSI、MACD、Bollinger Bands、ATR，含指標描述和 lookback 處理 |
| **News/Sentiment** | `eodhd_news.py` | 個股新聞、全球宏觀新聞（tag-based）、內部交易資訊 |
| **Fundamentals** | `eodhd_fundamentals.py` | 公司概覽、資產負債表、現金流量表、損益表 |
| **Re-export Module** | `eodhd.py` | 統一匯出所有 EODHD 函數供 `interface.py` 註冊 |

### 4.2 TradingAgents — Vendor 註冊

`interface.py` 的 `VENDOR_METHODS` 字典新增 EODHD 條目，覆蓋 10 個數據方法：

```python
VENDOR_METHODS = {
    "get_stock_data": {"alpha_vantage": ..., "yfinance": ..., "eodhd": get_eodhd_stock},
    "get_indicators": {"alpha_vantage": ..., "eodhd": get_eodhd_indicator},
    "get_fundamentals": {"alpha_vantage": ..., "eodhd": get_eodhd_fundamentals},
    "get_news": {"alpha_vantage": ..., "eodhd": get_eodhd_news},
    "get_global_news": {"alpha_vantage": ..., "eodhd": get_eodhd_global_news},
    # ... 以及 balance_sheet, cashflow, income_statement, insider_*
}
VENDOR_LIST = ["local", "yfinance", "openai", "google", "eodhd"]
```

### 4.3 prop-firm-pilot — Switchover 機制（Task 7）

`fx_analyst_config.py` 新增 `_get_primary_vendor()` 函數，根據日期自動選擇 vendor：

```python
def _get_primary_vendor() -> str:
    if os.getenv("EODHD_FORCE_PRIMARY") in ("1", "true", "yes"):
        return "eodhd"
    switchover_date = date.fromisoformat(os.getenv("EODHD_SWITCHOVER_DATE", "2026-03-21"))
    return "eodhd" if date.today() >= switchover_date else "alpha_vantage"
```

`build_agent_config()` 使用動態 vendor 設定 `data_vendors` 和 `tool_vendors`。

### 4.4 qlib_market_scanner — EODHD Fetcher（Tasks 8-9）

| 元件 | 檔案 | 功能 |
|---|---|---|
| **EODHDFXFetcher** | `fx_fetcher.py` | FX 日線數據獲取、符號轉換、重試邏輯、Rate Limit 處理 |
| **EODHD Stock Fetcher** | `eodhd_fetcher.py` | 股票日線數據獲取、`download_ticker()` / `download_universe()` 介面 |
| **Fetcher Priority** | `fx_fetcher.py` `download_universe()` | EODHD > Alpha Vantage > MockFetcher |
| **Runner Integration** | `runner.py` | Stock profile 使用 EODHD fetcher、`EODHDRateLimitError` 異常處理 |

---

## 5. 功能詳述

### 5.1 EODHD Common Utilities (`eodhd_common.py`)

**責任範圍**: 所有 EODHD 模組的共用基礎設施。

**符號轉換邏輯**:
- FX 符號（6 字元貨幣對）：`EURUSD` → `EURUSD.FOREX`
- 股票符號（無後綴）：`AAPL` → `AAPL.US`
- 已有後綴：`AAPL.US` → `AAPL.US`（直通）
- 斜線處理：`EUR/USD` → `EURUSD.FOREX`

**支援的 FX 貨幣**: AUD, CAD, CHF, EUR, GBP, JPY, NZD, USD, XAU, XAG, HKD, SGD, NOK, SEK, DKK, ZAR, TRY, MXN, PLN, CZK, HUF, CNY, INR, THB（24 種）

**HTTP Helper** (`_make_api_request`):
- 自動附加 `api_token` 參數
- Rate limit 偵測（HTTP 429）→ 拋出 `EODHDRateLimitError`
- JSON 解析和錯誤處理

**日期過濾** (`_filter_by_date_range`):
- 防禦性雙重過濾（API 應已過濾，但確保安全）

### 5.2 Stock/FX Data (`eodhd_stock.py`)

**API 端點**: `GET /api/eod/{SYMBOL}`

**輸出格式**: CSV 字串，與 Alpha Vantage `get_stock()` 完全相容：
```
timestamp,open,high,low,close,volume
2026-02-28,1.0835,1.0862,1.0810,1.0845,0
```

**關鍵設計**: FX 的 volume 欄位為 0（FX 無中心化成交量），但保留欄位以維持格式一致性。

### 5.3 Technical Indicators (`eodhd_indicator.py`)

**API 端點**: `GET /api/technical/{SYMBOL}?function={func}&period={period}`

**支援指標對照**:

| 內部名稱 | EODHD function | 預設 period | 說明 |
|---|---|---|---|
| `close_50_sma` | `sma` | 50 | 中期趨勢 |
| `close_200_sma` | `sma` | 200 | 長期趨勢 |
| `close_10_ema` | `ema` | 10 | 短期動量 |
| `macd` / `macds` / `macdh` | `macd` | — | 動量交叉 |
| `rsi` | `rsi` | 14 | 超買超賣 |
| `boll` / `boll_ub` / `boll_lb` | `bbands` | 20 | 波動通道 |
| `atr` | `atr` | 14 | 波動率 |
| `vwma` | ❌ 不可用 | — | 回傳說明訊息 |

**Lookback 處理**: 指標需要額外歷史數據來初始化（如 200-SMA 需要 200+ 天），模組自動將請求起始日期前推 `max(period, 250)` 天以確保數據充足。

### 5.4 News/Sentiment (`eodhd_news.py`)

**個股新聞** (`get_news`):
- 端點: `GET /api/news?s={SYMBOL}`
- 支援 FX 和股票
- 回傳 JSON `{"articles": [...], "source": "eodhd"}`

**全球宏觀新聞** (`get_global_news`):
- 端點: `GET /api/news?t={TAGS}`
- 預設標籤: `economy,monetary policy,federal reserve,central bank,inflation,gdp`
- 適用於宏觀分析師（macro analyst）

**內部交易** (`get_insider_transactions`, `get_insider_sentiment`):
- 透過 EODHD 實作，作為 vendor 路由的備選
- 實際生產環境中 `tool_vendors` 設定為 `local`（使用 finnhub）

### 5.5 Fundamentals (`eodhd_fundamentals.py`)

**API 端點**: `GET /api/fundamentals/{SYMBOL}`（單一請求返回所有基本面數據）

**效率優勢**: Alpha Vantage 需要 4 次 API 調用（overview + balance + cashflow + income），EODHD 只需 1 次。

**提供函數**:
- `get_fundamentals()` — 公司概覽（General + Highlights + Valuation）
- `get_balance_sheet()` — 資產負債表（quarterly/annual）
- `get_cashflow()` — 現金流量表
- `get_income_statement()` — 損益表

### 5.6 Vendor 註冊與路由

`VENDOR_METHODS` 新增 EODHD 為第三個主要 vendor（alongside `alpha_vantage` 和 `yfinance`），覆蓋 10 個方法：

| 方法 | EODHD 實作 | 備註 |
|---|---|---|
| `get_stock_data` | `get_eodhd_stock` | FX + 股票日線 |
| `get_indicators` | `get_eodhd_indicator` | 6 類技術指標 |
| `get_fundamentals` | `get_eodhd_fundamentals` | 公司概覽 |
| `get_balance_sheet` | `get_eodhd_balance_sheet` | 資產負債表 |
| `get_cashflow` | `get_eodhd_cashflow` | 現金流量表 |
| `get_income_statement` | `get_eodhd_income_statement` | 損益表 |
| `get_news` | `get_eodhd_news` | 個股新聞 |
| `get_global_news` | `get_eodhd_global_news` | 宏觀新聞 |
| `get_insider_sentiment` | `get_eodhd_insider_sentiment` | 內部交易情緒 |
| `get_insider_transactions` | `get_eodhd_insider_transactions` | 內部交易記錄 |

**未覆蓋**: `get_options`（繼續使用 OpenBB/yfinance）、`get_reddit_stock_sentiments`（使用 Google search）

**Fallback 機制**: `route_to_vendor()` 在 EODHD 調用失敗時自動嘗試下一個 vendor（例如 `alpha_vantage` → `yfinance`），`EODHDRateLimitError` 被加入例外捕捉鏈。

### 5.7 Switchover 機制

**設計**: 日期感知的自動切換，無需手動改代碼。

```
                    2026-03-21
          ┌──────────┼──────────┐
          │ AV 主要  │ EODHD 主要│
          │ EODHD 備  │ AV 備    │
          └──────────┼──────────┘
                   Switchover
```

**控制方式**:

| 環境變數 | 用途 | 預設值 |
|---|---|---|
| `EODHD_SWITCHOVER_DATE` | 切換日期（ISO 格式） | `2026-03-21` |
| `EODHD_FORCE_PRIMARY` | 強制使用 EODHD（`1`/`true`/`yes`） | 未設定 |

**優先順序**: `EODHD_FORCE_PRIMARY` > `EODHD_SWITCHOVER_DATE` > 預設日期

**影響範圍**: `build_agent_config()` 回傳的 `data_vendors.core_stock_apis`、`data_vendors.news_data`，以及 `tool_vendors` 中的 `get_global_news`、`get_news`、`get_indicators` 均動態設定為主要 vendor。

### 5.8 qlib_market_scanner FX Fetcher

**新增類**: `EODHDFXFetcher`（`fx_fetcher.py`）

**功能**:
- 符號轉換（`EURUSD` → `EURUSD.FOREX`）
- 每日 OHLCV 數據獲取
- 指數退避重試（3 次）
- Rate limit 處理（HTTP 429 → `EODHDRateLimitError`）
- API 呼叫間隔 1 秒（禮貌性限制，EODHD 付費額度充足）

**Fetcher 優先順序** (`download_universe`):
```
EODHD_API_KEY 存在? → EODHDFXFetcher
                ↓ 否
ALPHA_VANTAGE_API_KEY 存在? → AlphaVantageFXFetcher
                ↓ 否
                → MockFetcher
```

### 5.9 qlib_market_scanner Stock Fetcher

**新增檔案**: `eodhd_fetcher.py`（213 行）

**提供函數**:
- `download_ticker()` — 單一股票日線數據
- `download_universe()` — 批量下載，含 CSV 快取、重試、Rate Limit 處理
- `build_output_dir()` — 輸出路徑管理

**Runner Integration** (`runner.py`):
- Stock profile pipeline 優先使用 EODHD fetcher
- `EODHDRateLimitError` 加入異常處理鏈

---

## 6. 架構圖

### 6.1 數據流（切換後）

```
                           ┌─────────────────┐
                           │   EODHD API     │
                           │  (100K/day)     │
                           └────────┬────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
              ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
              │TradingAgents│  │qlib_market│  │prop-firm  │
              │ (LLM 決策) │  │ _scanner  │  │  -pilot   │
              │            │  │(量化信號) │  │(排程器)   │
              └─────┬──────┘  └─────┬─────┘  └─────┬─────┘
                    │               │               │
              vendor routing   fetcher priority  switchover logic
              (interface.py)   (fx_fetcher.py)   (fx_analyst_config.py)
                    │               │               │
              ┌─────▼───────────────▼───────────────▼─────┐
              │           route_to_vendor() fallback       │
              │  EODHD → Alpha Vantage → yfinance → local │
              └────────────────────────────────────────────┘
```

### 6.2 TradingAgents 模組結構

```
tradingagents/dataflows/
├── interface.py          # VENDOR_METHODS + route_to_vendor() [修改]
├── eodhd.py              # Re-export module [新增]
├── eodhd_common.py       # API key, symbol translation, HTTP [新增]
├── eodhd_stock.py        # Daily OHLCV (get_stock) [新增]
├── eodhd_indicator.py    # Technical indicators [新增]
├── eodhd_news.py         # News + sentiment [新增]
├── eodhd_fundamentals.py # Company fundamentals [新增]
├── alpha_vantage_*.py    # 既有 AV 模組 (保留)
├── finnhub_*.py          # 既有 finnhub 模組 (保留)
└── config.py             # set_config/get_config (未修改)
```

### 6.3 qlib_market_scanner 模組結構

```
src/data/
├── fx_fetcher.py         # TraderMade + AV + EODHD + Mock fetchers [修改]
├── eodhd_fetcher.py      # Stock universe fetcher [新增]
├── alpha_vantage_fetcher.py  # 既有 AV 股票 fetcher (保留)
└── ...

src/pipeline/
├── runner.py             # Pipeline runner — EODHD integration [修改]
└── ...
```

---

## 7. 數據源對照表

### 切換前（AV 訂閱有效期內）

| 數據類型 | 主要 vendor | 備用 vendor | API 端點 |
|---|---|---|---|
| FX 日線 OHLCV | Alpha Vantage | EODHD（free tier 測試） | AV `FX_DAILY` |
| US 股票日線 | Alpha Vantage | yfinance | AV `TIME_SERIES_DAILY_ADJUSTED` |
| 技術指標 | Alpha Vantage | — | AV `SMA`, `RSI`, etc. |
| 新聞/情緒 | Alpha Vantage | — | AV `NEWS_SENTIMENT` |
| 基本面 | Alpha Vantage | OpenBB/yfinance | AV `OVERVIEW`, etc. |
| 內部交易 | finnhub (local) | — | finnhub API |
| 期權 | OpenBB/yfinance | — | yfinance/cboe |

### 切換後（2026-03-21 起）

| 數據類型 | 主要 vendor | 備用 vendor | API 端點 |
|---|---|---|---|
| FX 日線 OHLCV | **EODHD** | Alpha Vantage (free tier) | `/api/eod/EURUSD.FOREX` |
| FX 小時線 | **EODHD** | — | `/api/intraday/?interval=1h` |
| US 股票日線 | **EODHD** | yfinance | `/api/eod/AAPL.US` |
| 技術指標 | **EODHD** | Alpha Vantage (free tier) | `/api/technical/` |
| 新聞/情緒 | **EODHD** | — | `/api/news` + `/api/sentiments` |
| 基本面 | **EODHD** | OpenBB/yfinance | `/api/fundamentals/` |
| 內部交易 | finnhub (local) | EODHD | finnhub API |
| 期權 | OpenBB/yfinance | — | yfinance/cboe |

---

## 8. Switchover 機制

### 運作流程

```
啟動 → fx_analyst_config._get_primary_vendor()
  ├── EODHD_FORCE_PRIMARY=1? → return "eodhd"
  ├── date.today() >= EODHD_SWITCHOVER_DATE? → return "eodhd"
  └── 否則 → return "alpha_vantage"
        ↓
build_agent_config() 使用動態 vendor
  ├── data_vendors.core_stock_apis = primary
  ├── data_vendors.news_data = primary
  ├── tool_vendors.get_global_news = primary
  ├── tool_vendors.get_news = primary
  └── tool_vendors.get_indicators = primary
```

### 測試方法

**在切換日之前（3/21 前）進行 EODHD 測試**:

```bash
# 方法 1: 環境變數強制覆寫
EODHD_FORCE_PRIMARY=1 python -m src.main --config config/e8_one_5k_challenge.yaml

# 方法 2: 設定提前的切換日期
EODHD_SWITCHOVER_DATE=2026-03-01 python -m src.main --config config/e8_one_5k_challenge.yaml
```

**驗證切換是否生效**: 檢查日誌中的 vendor 路由訊息：
```
INFO | Using vendor 'eodhd' for get_stock_data
INFO | Using vendor 'eodhd' for get_indicators
```

---

## 9. 組態參考

### 環境變數（新增）

| 變數 | 說明 | 預設 | 必填 |
|---|---|---|---|
| `EODHD_API_KEY` | EODHD API 金鑰 | — | 是（切換後） |
| `EODHD_SWITCHOVER_DATE` | 數據源切換日期（ISO 格式） | `2026-03-21` | 否 |
| `EODHD_FORCE_PRIMARY` | 強制使用 EODHD | 未設定 | 否 |

### .env.example 新增內容

```env
# ── EODHD (eodhistoricaldata.com) ──────────────────────────────────
EODHD_API_KEY=           # Required after switchover date
# EODHD_SWITCHOVER_DATE=2026-03-21   # Override switchover date
# EODHD_FORCE_PRIMARY=1              # Force EODHD as primary vendor
```

### EODHD API 額度管理

| 方案 | 每日額度 | 備註 |
|---|---|---|
| Free Tier | 20 calls/day | 限 demo tickers: AAPL.US, EURUSD.FOREX, AMZN.US |
| EOD+Intraday All World Extended | 100,000 calls/day | 覆蓋所有數據類型 |

**預估每日消耗**（4 個 FX 貨幣對、正常運行）:

| 調用類型 | 每次掃描 | 每日掃描次數 | 每日總量 |
|---|---|---|---|
| Stock/FX OHLCV | 4 | 6-12 | 24-48 |
| 技術指標（6 類 × 4 對） | 24 | 6-12 | 144-288 |
| 新聞 | 5 | 6-12 | 30-60 |
| 全球新聞 | 1 | 6-12 | 6-12 |
| **合計** | | | **~200-400** |

100,000 額度遠超實際需求，無需擔心限額問題。

---

## 10. 修改檔案清單

### TradingAgents

| 檔案 | 動作 | 說明 |
|---|---|---|
| `tradingagents/dataflows/eodhd_common.py` | **新增** | API key、符號轉換、HTTP helper、Rate Limit 處理（208 行） |
| `tradingagents/dataflows/eodhd_stock.py` | **新增** | 每日 OHLCV — `get_stock()`（65 行） |
| `tradingagents/dataflows/eodhd_indicator.py` | **新增** | 技術指標 — `get_indicator()`（150 行） |
| `tradingagents/dataflows/eodhd_news.py` | **新增** | 新聞/情緒 — `get_news()`, `get_global_news()`（209 行） |
| `tradingagents/dataflows/eodhd_fundamentals.py` | **新增** | 基本面 — 4 個函數（121 行） |
| `tradingagents/dataflows/eodhd.py` | **新增** | Re-export 統一入口（24 行） |
| `tradingagents/dataflows/interface.py` | **修改** | `VENDOR_METHODS` 新增 EODHD、`VENDOR_LIST` 新增 `"eodhd"`、import EODHD 模組 |
| `.env.example` | **修改** | 新增 `EODHD_API_KEY` |
| `pyproject.toml` | **修改** | 版本號 → 1.3.0 |
| `tests/test_eodhd_common.py` | **新增** | 17 tests — API key、符號轉換、HTTP mock |
| `tests/test_eodhd_stock.py` | **新增** | 4 tests — get_stock CSV 輸出 |
| `tests/test_eodhd_indicator.py` | **新增** | 7 tests — 指標請求與格式化 |
| `tests/test_eodhd_news.py` | **新增** | 3 tests — 新聞/全球新聞 |
| `tests/test_eodhd_fundamentals.py` | **新增** | 4 tests — 基本面函數 |
| `tests/test_eodhd_registration.py` | **新增** | 12 tests — VENDOR_METHODS 註冊驗證 |
| `tests/test_eodhd_integration.py` | **新增** | 5 integration tests — 真實 API 調用（需 API key，否則 skip） |

### qlib_market_scanner

| 檔案 | 動作 | 說明 |
|---|---|---|
| `src/data/eodhd_fetcher.py` | **新增** | EODHD 股票數據 fetcher（213 行） |
| `src/data/fx_fetcher.py` | **修改** | 新增 `EODHDFXFetcher` 類、`download_universe()` 新增 EODHD 優先邏輯 |
| `src/pipeline/runner.py` | **修改** | Stock profile 使用 EODHD fetcher、異常處理 |
| `.env.example` | **修改** | 新增 `EODHD_API_KEY` |
| `pyproject.toml` | **修改** | 版本號 → 1.3.0 |
| `tests/test_eodhd_fx_fetcher.py` | **新增** | 4 tests — FX fetcher 單元測試 |
| `tests/test_eodhd_stock_fetcher.py` | **新增** | 4 tests — Stock fetcher 單元測試 |
| `tests/test_eodhd_integration.py` | **新增** | 4 integration tests（需 API key，否則 skip） |

### prop-firm-pilot

| 檔案 | 動作 | 說明 |
|---|---|---|
| `src/decision/fx_analyst_config.py` | **修改** | 新增 `_get_primary_vendor()`、`build_agent_config()` 動態 vendor 設定 |
| `.env.example` | **修改** | 新增 EODHD 相關環境變數 |
| `pyproject.toml` | **修改** | 版本號 → 1.3.0 |
| `tests/test_switchover.py` | **新增** | 4 tests — 切換邏輯測試 |

---

## 11. 測試覆蓋

### TradingAgents（58 passed, 5 skipped）

| 測試檔案 | 測試數 | 說明 |
|---|---|---|
| `test_eodhd_common.py` | 17 | API key、符號轉換（FX/股票/斜線/大小寫）、反向轉換、HTTP mock、Rate Limit |
| `test_eodhd_stock.py` | 4 | CSV 輸出格式、空數據處理、API 錯誤處理 |
| `test_eodhd_indicator.py` | 7 | 各指標請求、lookback 日期計算、不支援指標處理、VWMA fallback |
| `test_eodhd_news.py` | 3 | 個股新聞、全球新聞、錯誤處理 |
| `test_eodhd_fundamentals.py` | 4 | 公司概覽、資產負債表、現金流量表、空數據處理 |
| `test_eodhd_registration.py` | 12 | VENDOR_METHODS 註冊完整性、VENDOR_LIST 包含 eodhd |
| `test_eodhd_integration.py` | 5 (skip) | 真實 API 調用 — 無 API key 時自動 skip |

### qlib_market_scanner（19 passed, 4 skipped）

| 測試檔案 | 測試數 | 說明 |
|---|---|---|
| `test_eodhd_fx_fetcher.py` | 4 | 符號轉換、日線數據、重試邏輯、Rate Limit |
| `test_eodhd_stock_fetcher.py` | 4 | download_ticker、download_universe、CSV 快取 |
| `test_eodhd_integration.py` | 4 (skip) | 真實 API 調用 — 無 API key 時自動 skip |

### prop-firm-pilot（701 passed, 1 failed）

| 測試檔案 | 測試數 | 說明 |
|---|---|---|
| `test_switchover.py` | 4 | 日期前 → AV、日期後 → EODHD、強制覆寫、自訂日期 |

**已知失敗**（非本版本引入）：
- `test_prop_firm_guard_e8_one.py::TestE8OneConfig::test_config_loads_correctly` — 預期 `symbols == ["EURUSD"]` 但實際包含 4 個貨幣對，此為 v1.2.0 前已存在的問題

---

## 12. 已知限制與未來工作

### 已知限制

1. **Free Tier 限制**：在 3/21 前只能用 EODHD free tier 測試（20 calls/day，限 demo tickers），無法完整測試所有貨幣對
2. **FX 小時線未整合**：EODHD 支援 1H intraday 但本版本尚未在 qlib_market_scanner 的 FX pipeline 中整合（日線 → 小時線轉換需額外工作）
3. **VWMA 不可用**：EODHD Technical API 不提供 VWMA 指標，回傳說明訊息而非數據
4. **期權數據未遷移**：繼續使用 OpenBB/yfinance，未來考慮購買 EODHD Options 套件
5. **Alpha Vantage 未移除**：AV 代碼完整保留作為 fallback，切換後降級為備用

### 未來工作

1. **FX Intraday Integration**：在 qlib_market_scanner 中整合 EODHD 1H 數據，支援 v1.2.0 的多時間框架分析
2. **AV Free Tier Fallback 驗證**：確認 AV 免費方案（5 calls/min）是否足以作為可靠 fallback
3. **EODHD Options Data**：評估 EODHD Options 套件（額外費用），替代 OpenBB/yfinance
4. **API 額度監控**：新增 EODHD API 調用計數器和預警機制
5. **移除 AV 硬編碼**：在確認 EODHD 穩定運行數週後，考慮移除 AV 相關代碼以簡化維護
6. **修復 pre-existing test failure**：`test_prop_firm_guard_e8_one.py` 的 symbols 斷言

---

## 13. 相依性變更

| 套件 | 倉庫 | 變更 | 說明 |
|---|---|---|---|
| `requests` | TradingAgents | 已存在 | EODHD 模組使用同步 HTTP（匹配既有 AV 模組模式） |
| `python-dateutil` | TradingAgents | 已存在 | `eodhd_indicator.py` 使用 `relativedelta` 計算 lookback |
| `httpx` | qlib_market_scanner | 未使用 | qlib_market_scanner 的 EODHD fetcher 使用 `requests`（同步） |

**無新增第三方依賴** — 所有 EODHD 模組使用各倉庫已有的 `requests` / `httpx` 庫。

**跨 Repo 相依**：三個倉庫版本號統一為 1.3.0，部署時需同步更新。

---

> **報告結束** — PropFirmPilot v1.3.0 EODHD 數據源遷移版本
