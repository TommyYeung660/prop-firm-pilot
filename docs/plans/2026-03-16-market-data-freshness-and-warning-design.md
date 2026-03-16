# Market Data Freshness And Warning Design

**Goal:** 修正 tactical market-data freshness 對 closed bars 的判定偏差，並降低 stale warning 噪音，同時保留足夠的診斷資訊區分 provider 過舊與本地 freshness 判定。

**Context**

- `MarketDataHub` 目前用 bar 的 `datetime` 直接判定 freshness。
- 對 websocket aggregator 產出的 closed `5m/1h` bars，`datetime` 代表 bucket open time，不是 close time。
- 這會讓剛完成的 `1h` closed bar 在 live tactical 流程中被過早判成 stale。
- `Scheduler._sanitize_tactical_bars()` 會對相同 stale bars 每個 evaluation cycle 都重打 warning，形成高噪音。

**Design**

1. Closed-bar freshness semantics
   - 保持 bars 的 `datetime` 欄位仍代表 open time，避免破壞既有 OHLC schema。
   - 新增共用 freshness helper，對 `1m/5m/1h` bars 用 `open_time + timeframe_duration` 作為 effective close time。
   - `MarketDataHub` 的 freshness 判定改為使用 effective close time。
   - `Scheduler` 的 tactical stale-bar sanitize 也改用 effective close time。

2. Warning throttling
   - 對 stale tactical warning 做 keyed stateful throttling。
   - 規則：內容改變時立即記錄；內容相同時只保留低頻 heartbeat。
   - key 至少包含 `symbol/timeframe/source/reason/latest_bar_open_time`，避免不同 stale 狀態被錯誤合併。

3. Instrumentation
   - `MarketDataHub` REST fallback warning 額外記錄：
     - latest bar open time
     - implied latest bar close time
     - age by close
   - tactical stale warning 也同時記錄 open/close time，讓 operator 能直接判斷是 provider 本身舊，還是 freshness semantics 問題。

4. Testing
   - 新增測試證明：
     - websocket aggregator 的 closed `1h` bar 以 close time 計算時仍應視為 fresh
     - tactical sanitize 不應錯殺 close-time 仍新鮮的 `1h` bar
     - 重複 stale warning 應被節流
     - fallback instrumentation 要輸出 open/close freshness 資訊

**Non-Goals**

- 不在這次處理 websocket keepalive / handshake timeout 根因。
- 不改 provider API 取數窗口或 EODHD 供應端行為。
