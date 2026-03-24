# Market Data Mainline Decoupling Design

## Context

`v1.5.0_stable` 的 prod 行為已證實兩件事同時成立：

- `2026-03-23` prod log 內確實出現 `EODHD` websocket degraded、scanner entry guard 被 stale bars 擋住、以及 tactical 長時間卡在 `atr_regime` / stale market-data。
- `2026-03-24` live probe 又顯示 websocket 本身沒有在短時間內落入「prod 不可接受」：
  - 12 分鐘 probe：`10/10` symbols 都有 ticks
  - 最大 tick gap `8.402s`
  - reconnect / handshake incidents `0`
  - 沒有 `>90s` stale breach

因此問題不是「websocket 一定要整個刪掉」，而是「主線 admission / tactical semantics 仍然把 websocket 當成隱性 hard dependency」。

另外，`Live API + intraday API` 也不是可直接替換的完整方案：

- `EODHD realtime` quote 可用，但 age 在連續樣本中曾到 `93s~115s`
- `EODHD intraday 5m` 在多數 symbols 上仍可落後 `16~28` 分鐘
- `EODHD intraday 1h` 甚至可能回傳尚未 close 的當前小時 bucket

這代表系統需要的是：

1. websocket 去關鍵化
2. quote polling 補 ingest
3. intraday API 只做 warmup / backfill，不再承擔 live 主路徑

## Goal

把 market-data 主線改成「websocket 可用時加分，但失效時不拖垮 scanner / tactical / execution 主線」：

- `MarketDataHub` 不再因 `websocket.enabled=false` 而缺席
- scanner entry readiness 改看 quote / bar readiness，不再把 websocket state 當 block 前提
- tactical validator 對 fresh quote + missing `5m` bars 一律 `WAIT`，不再只對 `websocket_cache` 特判
- 新增 `realtime quote polling` sidecar，持續把 live quotes 餵進 `FXTickAggregator`
- websocket 保留為 auxiliary ingest source

## Decision

採用「保留 websocket，但降成非關鍵路徑」：

- `broker quote` 仍是 quote primary
- `EODHD realtime quote polling` 是 quote backup，同時也是 aggregator 的補 tick 路徑
- `FXTickAggregator` 產出的 closed bars 是 live bar continuity 的主來源
- `EODHD intraday` 保留做 startup warmup / missing-bar backfill

## Intended Runtime Behavior

### Case A: websocket healthy

- websocket 與 realtime polling 都可餵 aggregator
- hub 仍可從 warmup cache 先服務啟動早期 bars
- scanner / tactical 都不直接依賴 websocket health flag

### Case B: websocket degraded / disabled，但 broker quote 與 realtime polling 正常

- hub 仍初始化
- scanner readiness 仍可評估
- aggregator 仍會持續收到 live quote snapshots
- tactical 遇到缺 `5m` closed bar 時會 `WAIT`，而不是誤進 pass-through execution

### Case C: realtime polling 正常，但 intraday backfill stale

- scanner 只在 `quote_unavailable` / `trade_date_not_ready` / `5m truly unavailable` 時 block
- tactical 可先用 quote freshness +現有 bars 做 `WAIT`
- `1h` 仍依 tactical 自身容忍與 stale 清洗邏輯決定，而不是在 scanner 提前硬阻擋

## Scope

### Modify

- `src/scheduler/scheduler.py`
- `src/data/market_data_hub.py`
- `src/decision/tactical_validator.py`
- `tests/test_scheduler.py`
- `tests/data/test_market_data_hub.py`
- `tests/test_tactical_validator.py`

### Not In Scope

- `export`
- Telegram delivery / alert fan-out
- 直接移除 websocket client 實作
- 重新設計 tactical ATR / EMA / RSI 規則

## Key Design Choices

### 1. `MarketDataHub` 永遠由 scheduler 初始化

`websocket.enabled` 只決定是否啟 websocket ingest task，不決定 hub 本身是否存在。

### 2. 新增 realtime polling sidecar

scheduler 新增 polling loop，按固定 cadence 拉 `EODHD realtime` quote，並透過 hub 餵進 `FXTickAggregator`。

### 3. startup retry 改成 aggregator-driven

`market_data.startup_5m_bar_pending` 的成立條件改成：

- quote 可用
- 最新 `5m` bar 仍不足以交易
- aggregator 尚未產出第一根可用 closed `5m` bar

而不是要求 websocket state 必須 healthy。

### 4. tactical 對 missing `5m` bars 一律 WAIT

現有 tactical validator 只在 `quote_source == "websocket_cache"` 且 `bars_5min.empty` 時 `WAIT`。  
這會讓非 websocket quote 路徑在 bars 缺失時落入過度寬鬆的 pass-through。  
新行為會改成：只要 quote freshness 存在、但 `5m` bars 缺失，就 `WAIT`。

## Risks

### Risk 1: polling quote timestamp 不一定每輪前進

已觀察到 `AUDUSD/NZDUSD` 可停留 `90s+`。  
這代表 polling 不能被當成完美 tick stream，只能當 backup ingest。

### Risk 2: 1h rollup 需要持續 minute-level quote ingestion

如果 polling 長時間沒有新 timestamp，aggregator 的 1h completeness 仍可能不足。  
因此 `intraday warmup / backfill` 不能移除。

### Risk 3: source taxonomy 仍使用既有名稱

本次 patch 先修主線，不大改 provenance taxonomy，避免擴散到 metrics / diagnostics / optimization 報表。

## Verification

- `tests/data/test_market_data_hub.py`
  - websocket disabled / disconnected 仍可得到 startup retryable readiness
- `tests/test_tactical_validator.py`
  - fresh quote + missing `5m` bars 時會 `WAIT`
- `tests/test_scheduler.py`
  - websocket disabled 時 hub 仍初始化
  - realtime polling 會把 quotes 餵進 hub / aggregator
- Focused regression:
  - `uv run pytest tests/data/test_market_data_hub.py tests/test_tactical_validator.py tests/test_scheduler.py -q`
  - `uv run ruff check src/data/market_data_hub.py src/decision/tactical_validator.py src/scheduler/scheduler.py tests/data/test_market_data_hub.py tests/test_tactical_validator.py tests/test_scheduler.py`
