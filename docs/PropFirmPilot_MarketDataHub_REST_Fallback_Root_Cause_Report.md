# MarketDataHub REST Fallback 根因報告

> **報告日期**: 2026-03-13
> 
> **調查範圍**: `v1.4.5a`、`v1.4.6b`、`v1.4.6c` 之後仍持續出現的 `MarketDataHub` REST fallback 問題
> 
> **基準版本**: 使用者於 `2026-03-13` pull 到 `origin/main = 6a565b1` 後，在本機執行 `uv run python -m src.main --config config/e8_one_5k_challenge.yaml --scheduler`
> 
> **定位**: 這份文件不是修復方案，而是針對現有代碼實作與 runtime 行為的根因分析

---

## 1. 執行摘要

這次問題的結論很明確：

- **這不是同一個 `1m` quote fallback bug 被連修三次仍沒修好**
- **真正持續存在的是另一條沒有被前幾版涵蓋到的 `5m/1h` bar fallback 路徑**
- **問題本質是結構性缺口，不是單點 typo 或單一條件判斷漏掉**

截至 `2026-03-13` 這次 runtime log，可見：

- `NZDUSD` 在 `14:19:57`、`14:21:58`、`14:23:59`、`14:26:00`、`14:28:01`、`14:30:02`、`14:32:04`、`14:34:05` 持續出現 `REST fallback for NZDUSD 1h`
- 其中 log 顯示 `ws_state=healthy`，但 `latest_rest_bar_time=2026-03-13T02:00:00+00:00`
- 這代表 **WebSocket 連線健康，不等於 `MarketDataHub` 已經有可用的 closed `1h` bars**

根因不是單一點，而是四個條件同時成立：

1. `MarketDataHub` 的 no-progress suppression **只做了 `1m` quote path，沒有做 `5m/1h` bar path**
2. `get_bars()` 對 `5m/1h` 一旦判定 stale，就會**每次都直接 refresh REST**
3. `FXTickAggregator.close_elapsed_bars()` **runtime 根本沒有被呼叫**
4. scheduler 的 position monitor 每 **120 秒**就會對有持倉 symbol 重跑 tactical data fetch

再加上外部事實：

- EODHD REST intraday bars 在當前環境下確實可能落後數小時

於是系統就形成一個穩定重現的迴圈：

> open position 存在  
> → position monitor 每 120 秒跑一次 tactical exit  
> → `_fetch_tactical_data()` 每次都要抓 `5m` / `1h` bars  
> → hub 拿不到 fresh closed websocket rollup bars  
> → warm cache 又是 lagging REST bars  
> → `get_bars()` 每次直接 refresh REST  
> → 又拿回同一個 stale tail  
> → 下一輪 120 秒後重複

---

## 2. 本次觀測到的現象

使用者提供的 `2026-03-13` runtime log 顯示：

- 啟動後 `EODHDFXWebSocketClient` 成功連線
- `ScannerBridge` 在 `14:17:54` 因 `--benchmark` 與舊版 scanner CLI 不相容而 fallback，這是 `v1.4.6c` 已處理的另一個問題
- `NZDUSD` / `USDCHF` live probe 之前已驗證 websocket 有 ticks
- 但 `MarketDataHub` 仍在 tactical / position monitor 路徑上持續刷：
  - `REST fallback for NZDUSD 1h`
  - 有時還伴隨 `REST fallback for NZDUSD 5m`

最關鍵的觀測值是：

- `ws_state=healthy`
- `latest_rest_bar_time=2026-03-13T02:00:00+00:00`
- `latest_rest_bar_age_sec` 持續增加

這說明當前問題不是：

- WebSocket 完全掛掉
- parser 壞掉
- subscription 沒生效

而是：

- **high-timeframe closed bars 的可用性判定，和 websocket 連線健康，是兩回事**

---

## 3. 直接根因

### 3.1 `1m` 有 suppression，`5m/1h` 沒有

`MarketDataHub._should_refresh_rest_cache()` 明確只對 `1m` 生效：

- [src/data/market_data_hub.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/data/market_data_hub.py):261
- [src/data/market_data_hub.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/data/market_data_hub.py):267

關鍵邏輯是：

```python
if timeframe != "1m":
    return True
```

也就是說：

- `1m`：有 cooldown + same-tail no-progress suppression
- `5m` / `1h`：永遠允許 refresh

這就是為什麼 `v1.4.1`、`v1.4.5a`、`v1.4.6b` 之後，你仍然可以看到 `5m/1h` fallback 持續重刷。

### 3.2 `get_bars()` 對 stale bar path 直接 refresh，沒有任何抑制

`MarketDataHub.get_bars()` 的流程是：

1. 先看 websocket closed bars
2. 沒有或不 fresh，就看 warm cache
3. warm cache 也不 fresh，就直接 `_refresh_rest_cache()`

對應代碼：

- [src/data/market_data_hub.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/data/market_data_hub.py):144
- [src/data/market_data_hub.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/data/market_data_hub.py):174

這條 path 沒有：

- no-progress suppression
- cooldown gate
- 「REST 也沒變新就暫停再試」的邏輯

因此只要 `5m/1h` 被判 stale，就會每次 refresh。

### 3.3 bar freshness 模型是單一固定秒數，沒有依 timeframe 調整

`_bars_are_fresh()` 用單一 `bar_cache_max_age_seconds` 判定所有 timeframe：

- [src/data/market_data_hub.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/data/market_data_hub.py):220
- [src/data/market_data_hub.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/data/market_data_hub.py):226

目前預設值是：

- `bar_cache_max_age_seconds = 3600`

但 scheduler 建 hub 時，根本沒有從 config 傳任何 bar freshness / cooldown 參數：

- [src/scheduler/scheduler.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/scheduler/scheduler.py):281
- [src/scheduler/scheduler.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/scheduler/scheduler.py):287

而 `WebSocketConfig` 本身只有：

- `stale_after_seconds`
- `quote_ttl_seconds`
- `warmup_*_bars`

沒有：

- bar cache freshness per timeframe
- REST refresh cooldown per timeframe

參考：

- [src/config.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/config.py):503
- [src/config.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/config.py):520

這代表目前 `5m` / `1h` 的 freshness 模型既是硬編碼，又是單一全域值。

### 3.4 warmup 先吃 REST，websocket rollup bars 剛啟動時本來就還沒形成

scheduler 啟動 market-data hub 的順序是：

1. 建立 `FXTickAggregator`
2. 建立 `EODHDFXWebSocketClient`
3. 建立 `MarketDataHub`
4. **先 `warmup()`**
5. 再 `create_task(self._websocket_client.run())`

參考：

- [src/scheduler/scheduler.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/scheduler/scheduler.py):271
- [src/scheduler/scheduler.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/scheduler/scheduler.py):291

這個順序本身不是錯，但在 EODHD REST lag 的前提下會導致：

- 啟動當下 warm cache 的 `5m/1h` 就已經是 stale
- websocket 剛起來時只有 live ticks，沒有既有 closed `5m/1h` bars

所以 cold start 的高時間框架 bar path，天然就會先掉進 REST fallback。

---

## 4. 更深一層的結構問題

### 4.1 `FXTickAggregator.close_elapsed_bars()` 在 runtime 沒有接線

這是本次最重要的 deeper root cause。

`FXTickAggregator` 提供了時間驅動的封 bar 方法：

- [src/data/fx_tick_aggregator.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/data/fx_tick_aggregator.py):125

但我在 repo 內做搜尋：

```text
rg -n "close_elapsed_bars\\(" src tests -S
```

結果顯示：

- runtime `src/` 沒有任何地方呼叫它
- 只有測試在呼叫

這意味著實際運行時：

- `1m` closed bar 主要靠「下一分鐘第一個 tick」觸發 finalize
- `5m` closed bar 主要靠「下一個 5 分鐘 bucket 的第一個 1m bar」觸發 finalize
- `1h` closed bar 主要靠「下一個整點 bucket 的第一個 1m bar」觸發 finalize

也就是說，high-timeframe closed bars 的 materialization 不是「時間到了就關」，而是「要等後面又有新 tick 進來跨 bucket 才會關」。

相關代碼：

- `add_tick()` 只在 1m bucket 切換時 finalize `1m`
  - [src/data/fx_tick_aggregator.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/data/fx_tick_aggregator.py):90
  - [src/data/fx_tick_aggregator.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/data/fx_tick_aggregator.py):96
- `5m/1h` rollup 只在下一個 rollup bucket 出現時 finalize 舊 bar
  - [src/data/fx_tick_aggregator.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/data/fx_tick_aggregator.py):162
  - [src/data/fx_tick_aggregator.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/data/fx_tick_aggregator.py):168

### 4.2 這直接解釋了為什麼 `ws_state=healthy` 還是 fallback

`EODHDFXWebSocketClient.get_status()` 的 `healthy` 只表示：

- 連線還在
- 最近 ticks 沒有 stale

它**不保證**：

- `FXTickAggregator` 已經產出 closed `5m` bars
- `FXTickAggregator` 已經產出 closed `1h` bars

而 `MarketDataHub.get_bars()` 真正看的不是 websocket state，而是：

- `websocket_bars.empty`
- `_bars_are_fresh(websocket_bars)`

參考：

- [src/data/market_data_hub.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/data/market_data_hub.py):151
- [src/data/market_data_hub.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/data/market_data_hub.py):157

所以看到下面這種 log 組合，從程式邏輯上完全成立：

- `ws_state=healthy`
- 但 `get_bars("NZDUSD", "1h", ...)` 還是 fallback

因為：

- websocket feed 是活的
- 但 hub 沒有可用的 closed `1h` bar

這不是 log 自相矛盾，而是兩個不同層級的健康狀態。

---

## 5. 為什麼會固定每 120 秒重現

position monitor loop 的基礎頻率預設是 `120` 秒：

- [src/config.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/config.py):193
- [src/config.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/config.py):194

而 loop 每輪只要有 open positions，就會跑 tactical exit cycle：

- [src/scheduler/scheduler.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/scheduler/scheduler.py):1768
- [src/scheduler/scheduler.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/scheduler/scheduler.py):1793

`_run_tactical_exit_cycle()` 對每個 open position 都會呼叫 `_fetch_tactical_data()`：

- [src/scheduler/scheduler.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/scheduler/scheduler.py):2747
- [src/scheduler/scheduler.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/scheduler/scheduler.py):2768

而 `_fetch_tactical_data()` 每次都會同時抓：

- `get_bars(symbol, "5m", ...)`
- `get_bars(symbol, "1h", ...)`

參考：

- [src/scheduler/scheduler.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/scheduler/scheduler.py):1309
- [src/scheduler/scheduler.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/scheduler/scheduler.py):1320

因此在本次 log 中：

- `NZDUSD` 有 active position
- position monitor 每 120 秒跑一次
- 每輪都要抓 `1h` bars
- `1h` path 沒有 suppression

就自然形成 `14:19:57` → `14:21:58` → `14:23:59` 這種近似 121 秒的固定循環。

### 5.1 `tactical.exit.evaluation_interval_seconds=60` 目前沒有實際生效

配置裡雖然有：

- [src/config.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/src/config.py):422

但我在 repo 內搜尋 `evaluation_interval_seconds`，只找到：

- config definition
- config test

沒有任何 runtime 使用點。

這表示 tactical exit 實際 cadence 不是由 `tactical.exit.evaluation_interval_seconds` 控制，而是由 position monitor loop 的 `120s` 控制。

這不是本次 REST fallback 的主根因，但它解釋了：

- 為什麼你看到的重刷節奏是約兩分鐘
- 而不是配置裡看起來的 60 秒

---

## 6. 版本修補為什麼沒解決這個問題

### 6.1 `v1.4.5a`

`v1.4.5a` 修的是：

- stale REST `1m` fallback 不應再合成可用 quote 往下游漏
- same-direction re-entry guard

沒有改到：

- `get_bars()` 的 `5m/1h` refresh loop

### 6.2 `v1.4.6b`

`v1.4.6b` 修的是：

- tactical freshness 若 hub 只有 bars、沒有 quote timestamp，回退到 MatchTrader quote

它解掉的是：

- entry gate 被 `data_freshness` 卡死

它沒解的是：

- `5m/1h` bar fallback 重刷

也就是說：

- `v1.4.6b` 修的是 **「不要錯誤 block 交易」**
- 不是 **「不要重複拉 stale bars」**

### 6.3 `v1.4.6c`

`v1.4.6c` 只修 scanner CLI `--benchmark` 向下相容：

- [docs/PropFirmPilot_changelog.md](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/docs/PropFirmPilot_changelog.md):78
- [docs/PropFirmPilot_changelog.md](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/docs/PropFirmPilot_changelog.md):95

與 `MarketDataHub` 無關。

### 6.4 `v1.4.1` 當時的修補目標本來就只鎖定 `1m`

從設計文件可以直接看出，當時 scope 是：

- live websocket probe
- stale REST **`1m`** no-progress guard

參考：

- [docs/plans/2026-03-12-websocket-rest-fallback-design.md](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/docs/plans/2026-03-12-websocket-rest-fallback-design.md)

也就是說，今天看到 `5m/1h` 問題仍在，不是「那次修復完全失敗」，而是：

- **那次根本沒有修到這條路徑**

---

## 7. 測試缺口

現有測試明顯只保護了 `1m quote fallback` 路徑。

### 7.1 `MarketDataHub` 測試只驗 `1m` no-progress suppression

現有測試：

- [tests/data/test_market_data_hub.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/tests/data/test_market_data_hub.py):311
- [tests/data/test_market_data_hub.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/tests/data/test_market_data_hub.py):403

這些測的是：

- `get_quote()` 在 stale `1m` tail 無進展時，不要重複 refresh

但 repo 裡沒有測：

- `get_bars("5m")` 在 stale same-tail 時是否應 suppress
- `get_bars("1h")` 在 stale same-tail 時是否應 suppress
- cold-start + websocket healthy + no closed rollup bar 的真實互動

### 7.2 scheduler 測試把 hub 整個 mock 掉了

相關測試：

- [tests/test_scheduler.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/tests/test_scheduler.py):3880
- [tests/test_scheduler.py](C:/Users/tommy.yeung/CursorProjects/prop-firm/prop-firm-pilot/tests/test_scheduler.py):4036

這些測試證明：

- `_fetch_tactical_data()` 會吃 hub 回傳的 bars

但因為 hub 被 mock 掉，所以完全測不到：

- aggregator 何時真的產出 closed `5m/1h` bars
- warm cache 是不是 stale
- `get_bars()` 是否在 live runtime 形成 120 秒 refresh loop

因此這個 bug 能一路穿過三個版本，是有測試覆蓋缺口支撐的。

---

## 8. 我實際做的最小重現

### 8.1 驗證 `1m` 路徑有 suppression，但 `5m/1h` 沒有

我用本地 stub provider 直接實例化 `MarketDataHub`，強制 `NZDUSD` 走 stale REST fallback，結果：

```text
{'1min': 1, '5min': 2, '1h': 2}
```

意思是：

- 連續兩次 `get_quote()`：`1m` 只打一次 REST
- 連續兩次 `get_bars("5m")`：打兩次 REST
- 連續兩次 `get_bars("1h")`：打兩次 REST

這與目前代碼完全一致，也與你看到的 production log 模式一致。

### 8.2 驗證 aggregator 的 time-based close 在 runtime 沒接線

我另外用最小腳本驗證：

- 沒有 `close_elapsed_bars()` 時，`5m/1h` closed bars 不會因為「時間到了」自動 materialize
- 一旦手動呼叫 `close_elapsed_bars(now=...)`，closed `5m/1h` bars 才會出現

這直接證明：

- runtime 若沒接這條線，高時間框架 bars availability 會依賴後續 tick 跨 bucket

---

## 9. 根因分級

### R1 — 主根因：`5m/1h` bar fallback path 從來沒有做 no-progress suppression

這是最核心、最直接的根因。

### R2 — 主根因：aggregator 的 elapsed-bar close 邏輯存在，但 runtime 未接線

這導致 websocket `5m/1h` closed bars 的產出延遲或缺失，比設計上更脆弱。

### R3 — 主根因：bar freshness / cooldown 缺乏 per-timeframe 配置與治理

現在是硬編碼單一 freshness 秒數，且 scheduler 沒把相關參數配置化帶進 hub。

### R4 — 外部促成因素：EODHD REST intraday provider lag

這不是 repo 內代碼造成，但它把上述缺口放大成 production 級問題。

### R5 — 測試缺口：只保了 `1m quote path`，沒保 `5m/1h bar path`

這使得 bug 在 hotfix 後仍能留在主線。

---

## 10. 一句話定義

**MarketDataHub 的 repeated REST fallback 並不是「同一個 `1m` bug 修三次還修不好」，而是 `5m/1h` bar fallback、aggregator 封 bar、scheduler polling cadence、以及 EODHD REST lag 共同形成的一個結構性重刷迴圈。**

---

## 11. 附帶發現（非本報告主題）

本次使用者 runtime log 顯示：

- 已 pull 到 `v1.4.6c` tag
- 但啟動 log 仍是 `PropFirmPilot v1.4.6b starting`

這是另一個獨立問題，與本報告主題無直接因果關係，但表示 `v1.4.6c` 的 runtime version identity 可能沒有同步更新。
