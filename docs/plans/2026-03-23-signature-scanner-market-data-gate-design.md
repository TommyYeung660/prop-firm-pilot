# Signature Scanner Market-Data Gate Design

## Context

`TradeLocker` 帳號在 2026-03-23 開市後可正常登入與取得即時 quote，但 `EODHD` 的 `5m/1h` REST bars 仍停在約 `02:00 UTC`。  
現行 scanner 會在 `MarketDataHub.get_entry_readiness()` 先用較嚴格的 freshness 規則阻擋：

- `1h` bar 只要超過 `bar_cache_max_age_seconds` 就直接回 `market_data.bars_1h_stale`
- `startup_5m_bar_pending` 的冷啟動放行條件又要求 `bars_1h_fresh=True`

結果是 scanner 在前面直接擋掉，根本到不了後面的 tactical layer。這和現有 tactical runtime 的容忍策略不一致：

- tactical 對 `1h` bar 允許更寬鬆的 stale 容忍
- tactical 對沒有可用 `1h` 的情況已有 pass-through / retry 路徑

## Goal

讓 scanner 的 market-data gate 與 tactical 層對齊：

- 不再把 `1h stale` 當作 scanner 的硬阻擋條件
- 只要 quote 可用、websocket 健康、且當日 `5m` bar 尚未由 websocket 補齊，就標記為 `startup_5m_bar_pending`
- 保留真正危險的阻擋：`quote_unavailable`、`broker_quote_unavailable`、`trade_date_not_ready`

## Options

### Option 1: 移除 scanner 的 `1h stale` 硬阻擋，保留 5m 冷啟動 retry

做法：

- 移除 `get_entry_readiness()` 內的 `bars_1h_stale` block
- `startup_5m_bar_pending` 不再依賴 `bars_1h_fresh`
- 當 `5m` REST 仍是同日但 stale，且 websocket 已健康連線但尚未產出 closed `5m` bar 時，回傳：
  - `entry_safe=True`
  - `requires_tactical_retry=True`
  - `pending_reason="market_data.startup_5m_bar_pending"`

優點：

- 最符合現有 tactical 行為
- 最小 patch
- 直接修復 Signature 開市初期「有 quote 但無法進場」的失配

缺點：

- scanner 會更早放行 intent，實際執行節奏會更多依賴 tactical retry

### Option 2: 保留 scanner `1h` block，但把 freshness 門檻對齊 tactical

做法：

- scanner 對 `1h` 也改成 4h 容忍

缺點：

- 仍然把 tactical concern 放在 scanner
- 兩層的 freshness 規則仍容易再度分叉

### Option 3: 對 Signature / TradeLocker 做 broker-specific 特判

缺點：

- 把資料源問題綁成 broker-specific hack
- 後續維護成本最高

## Decision

採用 Option 1。

## Intended Behavior

### Case A: fresh quote + fresh 5m + stale 1h

scanner 不阻擋，交由 tactical 依自己的 `1h` 容忍與 ATR 規則處理。

### Case B: fresh quote + same-day stale 5m + stale 1h + websocket healthy + 尚未有 closed 5m websocket bar

scanner 不阻擋，但標記 `startup_5m_bar_pending`，讓後續 tactical retry 等第一根 websocket `5m` 關閉。

### Case C: quote missing / broker quote missing / latest bars 還停在前一個 UTC trade date

scanner 仍然直接阻擋。

## Files

- Modify: `src/data/market_data_hub.py`
- Test: `tests/data/test_market_data_hub.py`

## Verification

- 新增/更新 `MarketDataHub.get_entry_readiness()` 測試
- 跑 `uv run pytest tests/data/test_market_data_hub.py -q`

