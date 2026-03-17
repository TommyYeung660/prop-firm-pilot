# Market Data Diagnostics Design

**Goal:** 在不改動既有 entry guard 決策邏輯的前提下，補齊 prod 診斷資訊，讓 operator 能直接分辨 websocket quote 健康、websocket closed bars 是否已累積，以及 market-data hub 啟動後已運行多久。

**Context**

- 目前 `Scheduler` 與 `OperationalMetrics` 只能看到 `MarketDataHub.feed_status()` 的摘要欄位：websocket state、forced stale symbols、warm cache keys。
- 生產環境已觀察到 `ws_state=healthy`，但 entry guard 仍因 `bars_5m_unavailable` 擋下 symbol。
- 既有資訊不足以判斷當下是 websocket 尚未累積出 closed `5m/1h` bars、market-data hub 剛重啟仍在暖機，或 REST fallback 本身 stale。

**Approach Options**

1. Extend existing feed status payload
   - 在 `MarketDataHub.feed_status()` 直接加上 hub 初始化時間、uptime、以及每個 symbol 的 websocket closed bar counts。
   - 優點：改動最小，現有 metrics snapshot 與 scheduler log payload 可直接重用。
   - 缺點：`feed_status()` payload 會稍微變大。

2. Dedicated diagnostics helper
   - 新增獨立 helper 組裝 market-data diagnostics，讓 `feed_status()` 與 scheduler 各自呼叫。
   - 優點：責任更明確。
   - 缺點：這次需求只需要少量欄位，會引入不必要抽象。

**Chosen Design**

採用方案 1，直接擴充 `MarketDataHub.feed_status()`。

1. Hub lifecycle diagnostics
   - `MarketDataHub` 初始化時記錄 `initialized_at`。
   - `feed_status()` 回傳：
     - `initialized_at`
     - `uptime_seconds`

2. Websocket closed-bar counts
   - `FXTickAggregator` 新增只讀 diagnostics 方法，輸出每個 symbol 在 `1m/5m/1h` 的 closed bar 數量。
   - `feed_status()` 回傳：
     - `websocket_closed_bar_counts: {symbol: {"1m": int, "5m": int, "1h": int}}`

3. Scheduler propagation
   - 保持 `Scheduler._build_metrics_snapshot()` 行為不變，直接帶出擴充後的 `feed_status()`。
   - scanner 被 market-data entry guard 擋下時，structured payload 補上：
     - `market_data_initialized_at`
     - `market_data_uptime_seconds`
     - `websocket_closed_bar_counts`
   - 不改 entry guard 的 allow/block 邏輯，只增加可觀測性。

4. Testing
   - `tests/data/test_market_data_hub.py`
     - 驗證 `feed_status()` 含 `initialized_at`、`uptime_seconds`、symbol-level closed bar counts。
   - `tests/test_scheduler.py`
     - 驗證 metrics snapshot 帶出新增欄位。
     - 驗證 scanner guard block log/event payload 帶出新增 diagnostics。

**Non-Goals**

- 不在這次處理 EODHD provider stale root cause。
- 不改 market-data freshness semantics 或 entry guard blocking 規則。
- 不新增新的 metrics sink 或 dashboard schema。
