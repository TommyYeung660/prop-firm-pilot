# PropFirmPilot v1.4.0 Report

> **報告日期**: 2026-03-11  
> **版本**: v1.4.0  
> **基準版本**: v1.3.9c  
> **跨倉庫**: `prop-firm-pilot` + `TradingAgents`

---

## 1. 摘要

v1.4.0 將 production market-data path 升級為 **WebSocket-first**，並把平倉結果從 `prop-firm-pilot` 結構化回灌到 `TradingAgents` persistent memory，形成可被後續決策重新檢索的 learning loop。

本版採 **aggressive rollout**：

- 預設啟用 `websocket.enabled: true`
- 預設啟用 `websocket.use_as_primary_market_data: true`
- REST 不再作為日內主要行情來源，只保留 broker / news / cold-start backfill / degraded fallback

---

## 2. 目標完成度

### 2.1 WebSocket-first market data

已完成：

- EODHD FX WebSocket client
- tick -> quote / `1m` / `5m` / `1h` aggregation
- scheduler startup warmup + WebSocket sidecar lifecycle
- volatility monitor 改走 WebSocket quote first
- tactical fetch 改走 local aggregated bars first

### 2.2 Closed-trade learning loop

已完成：

- scheduler 平倉後輸出 structured reflection payload
- TradingAgents 反射流程可接受 structured payload 或 legacy `{symbol: pnl}`
- persistent memory 會保存 lesson recommendation 與 structured metadata
- decision-time retrieval 會把 `retrieved_trade_lessons` 注入 trader prompt

---

## 3. 架構變更

### 3.1 prop-firm-pilot

新增元件：

- `src/data/fx_websocket_client.py`
- `src/data/fx_tick_aggregator.py`
- `src/data/market_data_hub.py`

核心讀取路徑：

1. `Scheduler.start()` 先 `warmup()` market-data hub
2. WebSocket sidecar 開始接收 tick
3. `FXTickAggregator` 生成最新 quote 與 closed bars
4. `MarketDataHub` 依 symbol freshness 決定使用：
   - `websocket_cache`
   - `warmup_cache`
   - `rest_fallback`
5. volatility monitor / tactical fetch 統一從 hub 讀取

### 3.2 TradingAgents

新增 learning loop：

1. `Scheduler._build_reflection_payload()` 產生 structured trade outcome
2. `AgentBridge.reflect()` 直接把 payload 傳入 `TradingAgentsGraph.reflect_and_remember()`
3. `Reflector` 將 payload normalize，連同 situation / recommendation / metadata 一起寫入 persistent memory
4. `TradingAgentsGraph._retrieve_trade_lessons()` 以 symbol + context 查詢相似 lesson
5. `Propagator` 將 `retrieved_trade_lessons` 放入 state
6. `trader` prompt 在 decision-time 顯式引用 retrieved lessons

---

## 4. Fallback 與安全規則

### 4.1 Market data fallback

- 單一 symbol stale：只對該 symbol 走 `rest_fallback`
- feed 整體 degraded：hub 退回 REST-backed reads，但 broker workflows 不受阻塞
- cold-start：先用 REST backfill seed bars，再切到 WebSocket incremental updates
- tactical / volatility 只使用 closed bars，不使用 partial bars

### 4.2 Learning loop isolation

- reflection failure：只記 log，不可阻塞平倉 closing flow
- retrieval failure：退化為空 lesson block，不可阻塞 trader decision
- legacy `{symbol: pnl}` reflect input 仍可用，確保向後相容

---

## 5. 主要修改檔案

### 5.1 prop-firm-pilot

- `src/config.py`
- `config/e8_one_5k_challenge.yaml`
- `src/data/fx_websocket_client.py`
- `src/data/fx_tick_aggregator.py`
- `src/data/market_data_hub.py`
- `src/scheduler/scheduler.py`
- `src/scheduler/volatility_monitor.py`
- `src/decision/tactical_validator.py`
- `src/decision/agent_bridge.py`

### 5.2 TradingAgents

- `tradingagents/agents/utils/memory.py`
- `tradingagents/graph/reflection.py`
- `tradingagents/graph/trading_graph.py`
- `tradingagents/graph/propagation.py`
- `tradingagents/agents/utils/agent_states.py`
- `tradingagents/agents/trader/trader.py`

---

## 6. 驗證命令

### 6.1 prop-firm-pilot targeted tests

```bash
uv run pytest tests/test_config.py tests/data/test_fx_websocket_client.py tests/data/test_fx_tick_aggregator.py tests/data/test_market_data_hub.py tests/test_volatility_monitor.py tests/test_tactical_validator.py tests/test_tactical_integration.py tests/test_scheduler.py tests/test_agent_bridge_config.py -q
```

### 6.2 TradingAgents targeted tests

```bash
uv run pytest C:/Users/tommy.yeung/CursorProjects/TradingAgents/tests/test_memory_reflection.py C:/Users/tommy.yeung/CursorProjects/TradingAgents/tests/test_prompt_memory_injection.py -q
```

### 6.3 v1.3.9c regression set

```bash
uv run pytest tests/test_scheduler.py tests/test_decision_cache.py tests/test_decision_store.py tests/test_ab_model_switching.py tests/test_agent_bridge_config.py tests/scheduler/test_optimization_integration.py tests/monitor/test_trade_journal.py tests/monitor/test_equity_monitor.py tests/scheduler/test_news_event_trigger.py -q
```

### 6.4 Lint

```bash
uv run ruff check src tests
uv run ruff check C:/Users/tommy.yeung/CursorProjects/TradingAgents/tradingagents C:/Users/tommy.yeung/CursorProjects/TradingAgents/tests
```

### 6.5 驗證結果

- prop-firm-pilot targeted suites：`188 passed`
- TradingAgents targeted suites：`6 passed`
- v1.3.9c regression set：`214 passed`
- prop-firm-pilot lint：`All checks passed!`
- TradingAgents changed-file lint：`All checks passed!`
- 註：TradingAgents full-repo lint 未作為本次 release gate，因該 repo 尚有大量與 v1.4.0 無關的既有 lint debt

---

## 7. 已知限制

- WebSocket path 目前聚焦 FX quote / bar ingestion，broker execution 仍維持 MatchTrader REST
- learning loop retrieval 目前主注入點是 trader prompt，尚未擴散到 bull / bear / judge prompts
- REST fallback 仍依賴 EODHD historical endpoints，因此 fallback timeliness 不等同於 live tick feed

---

## 8. 結論

v1.4.0 已把系統從 REST-first 觀測模式推進到 **WebSocket-first + REST fallback**，並完成從 closed trade outcome 回到 future decision prompt 的 learning loop 閉環。這是一次跨 `prop-firm-pilot` 與 `TradingAgents` 的行為級升級，而不是單點 bugfix。
