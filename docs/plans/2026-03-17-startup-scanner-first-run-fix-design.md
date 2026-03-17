# Startup Scanner First-Run Fix Design

**Date:** `2026-03-17`

**Goal:** 解決 production 啟動後第一輪 scanner 很容易因 `5m` tactical bars 尚未形成而失效的問題，讓首輪 daily signal 不再白跑，並且不必依賴下一次 scanner cadence 才重新獲得交易機會。

## Context

- `v1.5.0_beta_2` 已把 stale REST intraday bars 視為不可用，這個方向是正確的，因為 production 已證實 `EODHD REST` 可能長時間停在數小時前的 bar。
- 同時，production 與本地診斷也證實：
  - `EODHD WebSocket` quote feed 可以是健康的
  - `FXTickAggregator` 需要等到下一個 `5m` close 後，才會產生第一根 closed `5m` bar
  - 在 sidecar 剛啟動、尚未跨過第一個 `5m` close 的窗口內，`MarketDataHub.get_bars("5m")` 會落回 stale REST 或空資料
- 現行行為把這種「market-data sidecar 尚在暖機」與「真的沒有可交易 market data」混成同一個 scanner-stage block，導致：
  - 第一輪 scanner 候選直接丟失
  - 若沒有 volatility/news 觸發提早 rescan，就要等 `30min` 或 `60min` 後的下一輪 scanner
  - quiet session 下首輪信號的有效性與交易時效被 scheduler cadence 綁死

## Problem Statement

目前系統在啟動後的前幾分鐘存在一個結構性空窗：

1. scanner 已能產生 `1d` 候選信號
2. websocket 已有即時 tick，quote freshness 正常
3. 但 closed `5m` bar 尚未形成
4. scanner loop 直接把 symbol 視為 `market-data entry guard` 失敗
5. 該 symbol 不建立 intent，因此後續也不會進入 tactical retry

這不是 tactical gate 太嚴，而是責任分層錯了：本來應該由 tactical pending 去吸收的「短暫資料未就緒」，被提前在 scanner loop 直接丟棄。

## User-Approved Direction

使用者已明確選擇方案 `A`：

- 保留第一輪 scanner 候選
- 直接建立 intent
- 交由 tactical pending 以 `5min` cadence 重試，等待第一根 websocket `5m` closed bar 形成

不採用以下路徑：

- 不在 scanner loop 層另做一套 symbol 保留池與 delayed rescan scheduler
- 不用「啟動後先整體延後 scanner」去繞過問題

## Alternatives Considered

### Option A: Keep first-run candidates and rely on tactical retry

- 優點：
  - 直接解決首輪 scanner 白跑
  - 重用現有 `TACTICAL_PENDING -> RETRY_PENDING -> READY_FOR_EXEC` 狀態機
  - 不把 scanner cadence 與啟動暖機耦合
- 缺點：
  - 需要把 scanner-stage 的 market-data pre-block 縮窄，避免與 tactical gate 重疊

### Option B: Scanner-stage deferred symbol queue

- 優點：
  - scanner / tactical 責任邊界表面上較乾淨
- 缺點：
  - 需要新 queue、新喚醒邏輯、新去重規則
  - 其本質仍是另一套 retry system，與現有 tactical pending 重複

### Option C: Delay first scanner until bars are warm

- 優點：
  - 實作最簡單
- 缺點：
  - 啟動後故意失去第一輪反應能力
  - 若市場在啟動當下已經有成熟 daily signal，仍被系統主動延遲

結論：採用 `Option A`。

## Design

### 1. Reclassify startup `5m` bar absence as tactical-pending, not scanner rejection

- scanner loop 不再因「`5m` bars 尚未 ready」直接丟棄 symbol
- scanner loop 的 market-data pre-check 只保留真正應在 scanner-stage fail-close 的條件，例如：
  - scanner bundle invalid / stale
  - quote 完全不可用
  - market-data sidecar 明確 unhealthy，且沒有任何可用 fallback
- 若僅僅缺少可用 `5m` bars，但：
  - quote 有效
  - websocket feed healthy 或至少正在產生 ticks
  - `1d` scanner signal 有效
  則仍建立 `TradeIntent`

### 2. Let tactical validation absorb the warmup gap

- intent 進入既有 LLM/tactical pipeline 後：
  - `tactical_validator` 看到 `bars_5min` 不可用
  - 回傳 `WAIT / RETRY_PENDING`
  - scheduler 進入 `_retry_tactical_pending()`
- retry cadence 沿用現有設定：
  - `interval_seconds = 300`
  - `max_retries = 12`
  - `expire_action = degrade`
- 這使「等待第一根 websocket `5m` close」自然成為 tactical lifecycle 的一部分，而不是 scanner lifecycle 的失敗

### 3. Narrow scanner-stage market-data guard to true hard blockers

scanner-stage guard 改成只擋以下狀況：

- 沒有 fresh quote，`latest_bar_time` 也不存在
- market-data hub 明確報告 feed unhealthy / degraded，且此 symbol 沒有可執行的 quote fallback
- scanner signal 本身 stale / invalid

scanner-stage guard 不再擋以下狀況：

- quote 有效，但 `bars_5min` 暫時空
- websocket 已在流，但尚未累積出 closed `5m` bars
- `bars_1h` 暫時不足，但 quote 與後續 retry 仍可用

### 4. Make the retry reason explicit in diagnostics

要把「首輪暖機等待中」與「真正 market-data 壞掉」分開記錄。

新增或標準化的 reason / event 類型應至少包含：

- `market_data.startup_5m_bar_pending`
- `market_data.feed_degraded`
- `market_data.quote_unavailable`
- `market_data.rest_stale_only`

對 operator 來說，最重要的是能從 event / alert 一眼看出：

- 這是正常暖機中的 retryable wait
- 還是供應商真的卡死、需要人工介入

### 5. Preserve idempotency and avoid duplicate intents

方案 A 的前提是仍沿用既有 intent 去重規則：

- 同 symbol 同 trade date 的 in-progress intent 不重複建立
- tactical pending 不應讓 scanner loop 在下一輪又再建一個同 symbol intent
- 若早期 scanner 候選已進入 `tactical_pending`，下一輪 scanner 應視為已有 active pipeline item，直接跳過

這部分主要沿用現有 `intent_exists()` / pipeline count / symbol active checks，不另開新 queue。

## Data Flow

新流程如下：

1. Scheduler 啟動 market-data sidecar
2. WebSocket 開始接 tick，但 closed `5m` bars 尚未形成
3. Scanner loop 產生 daily candidates
4. scanner-stage market-data pre-check 檢查：
   - quote 可用
   - signal fresh
   - feed 沒有進入真正 hard-fail
5. 符合條件則建立 `TradeIntent`
6. LLM worker 做 decision + tactical validation
7. tactical validation 發現 `bars_5min` 暫時不可用，回傳 `RETRY_PENDING`
8. `_retry_tactical_pending()` 每 `300s` 重試
9. 一旦下一個 `5m` close 後 websocket 聚出 bar，tactical validation 轉為正常評估
10. intent 進入 `ready_for_exec` 或按既有規則 `degrade/cancel`

## Error Handling

- `ScannerBridge` / stale signal：維持 fail-closed，不做放寬
- `quote` 不可用：維持 scanner-stage hard block
- `5m bars` 暫時不可用但 quote 可用：改成 tactical retry
- `WebSocket healthy but REST stale`：不再讓首輪 scanner candidates 直接消失
- `WebSocket degraded and quote missing`：仍然 hard block，不建立 intent

## Testing Strategy

1. `tests/test_scheduler.py`
   - 新增回歸測試：scanner first-run 在 quote 可用但 `bars_5min` 空時，仍會建立 intent
   - 驗證 intent 進入 `tactical_pending`
   - 驗證下一次 tactical retry 成功後轉為 `ready_for_exec`
2. `tests/data/test_market_data_hub.py`
   - 驗證 websocket ticks 已存在但尚未形成 closed `5m` bar 時，hub 行為仍可提供 quote，但 bar 為空
3. `tests/data/test_fx_websocket_client.py`
   - 確保 healthy websocket + fresh tick 狀態判定不回退
4. `tests/test_scheduler.py`
   - 保留真正 hard-fail case：
     - quote missing
     - feed degraded + no usable fallback
     - stale scanner bundle

## Acceptance Criteria

- 程式啟動後第一輪 scanner 不再因 `5m` bars 尚未形成而直接白跑
- 在 `quote fresh + websocket healthy + first 5m bar pending` 狀態下，系統會建立 intent 並進入 tactical retry
- 不必等待下一輪 scanner cadence，既有 intent 就能在下一個 `5m` close 後繼續往 execution 推進
- 真正的 market-data hard failures 仍然 fail-closed，不會因本次修復誤放行
- diagnostics 能清楚區分「startup warmup wait」與「provider / feed failure」

## Non-Goals

- 不改 `scanner_timeframe = 1d`
- 不改 daily scanner cadence 選型
- 不引入新的 pending queue 子系統
- 不把這次修復擴大成一般 market-data policy refactor
