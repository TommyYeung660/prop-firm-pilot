# WebSocket Live Probe And REST Fallback Guard Design

## 背景

v1.4.1 上線後，production 仍出現重複的 `MarketDataHub: REST fallback for <symbol> 1m`，
而且每輪都重新抓取當日數百根 `1min` bars。

本輪即時驗證得到兩個關鍵事實：

- 同一套 `EODHDFXWebSocketClient` 在本地可穩定收到 `EURUSD / GBPUSD / USDJPY / AUDUSD` 即時 ticks
- EODHD REST `1min` 當日資料的最新 bar 可能明顯落後當前時間，導致 warm cache 長時間被判定 stale

這代表問題不是單純「WebSocket parser 壞掉」，而是：

1. 需要一個可重跑的 live probe 腳本，能在 target 環境直接回答 websocket 是否真的有流量
2. 需要避免 stale REST `1m` tail 沒有進展時，每次 fallback 都重新抓完整個今日窗口

## 目標

- 新增 live diagnostics script，直接驗證：
  - websocket 授權是否成功
  - 每個 symbol 是否收到 ticks
  - per-symbol tick count / max gap / latest tick age
  - REST `1min` 最新 bar 時間與 age
- 在不改變既有策略 / compliance 行為的前提下，為 `MarketDataHub` 加入 REST fallback no-progress guard：
  - 若最新 REST `1m` bar 沒有前進，短時間內不要重複整段刷新
  - 仍可回用既有 warm cache quote，避免無限重抓

## 設計

### 1. Live Probe

新增 `src/diagnostics/eodhd_websocket_live.py` 放可測試的 probe helpers：

- `summarize_tick_events()`：根據 websocket event 序列計算每個 symbol 的 count / max gap / latest age
- `summarize_rest_bars()`：計算 REST `1min` bar 的 rows / latest_bar_time / latest_bar_age_sec
- `probe_websocket()`：用現有 `EODHDFXWebSocketClient` 訂閱 symbols 並收集指定秒數的 events

新增 CLI：

- `scripts/check_eodhd_websocket_live.py`

輸出內容聚焦：

- websocket 授權與 samples
- per-symbol websocket summary
- per-symbol REST `1min` lag summary

### 2. REST Fallback No-Progress Guard

在 `MarketDataHub` 內新增 REST refresh state：

- 以 `(symbol, timeframe)` 為 key 記錄最近一次 refresh 的：
  - `attempted_at`
  - `latest_bar_at`

新增邏輯：

- 若 warm cache 已 stale，先檢查是否最近才 refresh 過
- 若最新 cached / fetched bar timestamp 沒有前進，且仍在 cooldown 內，就跳過再次 refresh
- 對 quote fallback 直接回用既有 warm cache，而不是每次重抓當日整段 `1min`

### 3. 診斷訊息

`MarketDataHub` 的 REST fallback warning 補上：

- `latest_rest_bar_time`
- `latest_rest_bar_age_sec`

這樣 production 看到 degraded 時，可以直接分辨：

- websocket 真的 stale
- 還是 REST provider 自己就 lag 很久

## 範圍

本次只處理：

- websocket live probe
- `MarketDataHub` 的 repeated REST fallback 壓制
- fallback warning 診斷強化

本次不處理：

- 更換 websocket provider
- 重寫整個 market-data orchestration
- 調整策略 / tactical gate / compliance 門檻

## 驗收標準

- 有一個可在 target env 重跑的 `check_eodhd_websocket_live.py`
- 測試能覆蓋：
  - websocket event summary 計算
  - stale REST `1m` no-progress 時不會連續重抓
- 實際驗證時，`MarketDataHub` 在同一個 stale `1m` tail 下，不再每次 `get_quote()` 都重抓當日整段資料
