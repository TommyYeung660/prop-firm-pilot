# PropFirmPilot v1.3.9b OODA Review

> 日期: 2026-03-11
> 基準文件: `docs/PropFirmPilot_v1.3.5_road_map.md`
> 審查範圍: `prop-firm-pilot` 現行程式碼 + `../../TradingAgents` 中與決策/記憶直接相關的實作
> 審查目標: 判斷目前 v1.3.9b 是否已實質符合 OODA（Observe → Orient → Decide → Act）模型，並評估進入 v1.4.0 前的缺口

---

## 一句話結論

`v1.3.9b` 已經具備 OODA 的外形，但仍不是完整閉環 OODA 系統。
目前是 **Observe / Decide / Act 有主幹，Orient 最弱**；更準確地說，是一個「能觀察、能決策、能執行，但學習記憶、事件驅動觀察、以及等待重判機制尚未閉合」的半閉環版本。

---

## 主要發現（依嚴重度排序）

### 1. Critical: Scheduler 模式下的 EquityMonitor 沒有接上告警與緊急平倉 callback

- `src/scheduler/scheduler.py:1113` 呼叫 `self._equity_monitor.start(...)` 時，只傳入 `get_equity`、`day_start_balance`、`initial_balance` 與 drawdown limit。
- 但 `src/monitor/equity_monitor.py:44-47` 的設計明確支援 `on_alert`、`on_emergency_close`、`on_equity_snapshot`。
- 對照 `src/main.py:351-352`，`monitor-only` 模式有正確傳入 `self.alert_service.drawdown_warning` 與 `client.close_all_positions`；scheduler 模式沒有。

**影響**

- 在 scheduler 主流程下，權益監控會「計算 drawdown」，但不會發送 drawdown alert，也不會執行 `close_all_positions()`。
- 這代表 OODA 的 `Act` 最後一道防線在 24/7 主模式裡沒有真正接通。

**判定**

- 這不是小缺口，而是風控執行鏈路未閉合。

### 2. High: Tactical WAIT 沒有走 `tactical_pending` / retry，而是直接取消 intent

- `src/decision_store/sqlite_store.py:449-455` 已支援 `claimed -> tactical_pending -> ready_for_exec`。
- `tests/test_decision_store.py:963-1052` 也有完整狀態轉移測試。
- 但 scheduler 真正落地只走 `src/scheduler/scheduler.py:840` 的 `mark_ready_for_exec()`。
- 當 tactical gate 非 `PASS` 時，scheduler 直接在 `src/scheduler/scheduler.py:823-836` 取消 intent。
- `tests/test_scheduler.py:3114` 甚至把這個「WAIT 直接取消」當成現行正確行為。

**影響**

- roadmap 中設計的「Observe 新 5min bar -> Orient -> 再 Decide 是否進場」沒有落地。
- `tactical.retry.*` config 與 `tactical_pending` 狀態目前屬於「有模型、有測試、但沒有主流程接線」。

**判定**

- 目前 tactical layer 比較像同步阻擋器，不是 OODA 式的等待重判層。

### 3. High: `StrategicDecisionCache` 已存在且有測試，但完全沒有接入 scheduler 決策路徑

- `src/scheduler/scheduler.py:131-132` 建立了 `StrategicDecisionCache`。
- `tests/test_decision_cache.py` 11 個測試都通過。
- 但 repo 搜尋結果顯示 scheduler 內沒有任何 `is_fresh()` / `get_cached()` / `store()` / `invalidate()` 呼叫。

**影響**

- roadmap v1.3.7 / v1.4.0 想解決的「同一 1D signal 被反覆送進 LLM」問題，在主決策層沒有真正被消除。
- 目前只有 `src/signal/scanner_bridge.py` 的 scanner pipeline cache 已接線；**Observe 端的 cache 有，Decide 端的 cache 沒有**。

**判定**

- Decide 階段仍會對相同戰略情境重做昂貴推理，OODA 的「Orient 後保留態勢理解」不足。

### 4. High: TradingAgents 反思記憶存在，但持久性與跨 session 連續性很弱

- `src/decision/agent_bridge.py:603` 會在平倉後呼叫 `reflect()`。
- `../../TradingAgents/tradingagents/graph/trading_graph.py:430` 的 `reflect_and_remember()` 會更新 bull / bear / trader / judge / risk manager 記憶。
- `../../TradingAgents/tradingagents/agents/trader/trader.py:27` 會在決策 prompt 前做 `memory.get_memories(...)`，並在 `:71` 把 past memories 注入 trader prompt。

但有兩個結構性問題：

- `../../TradingAgents/tradingagents/graph/trading_graph.py:107` 若未明確提供 `session_id`，會隨機生成 `uuid` suffix。
- `../../TradingAgents/tradingagents/graph/trading_graph.py:195-201` 以該 suffix 建立 memory collection 名稱。
- `src/decision/agent_bridge.py:323-349` 的 `_apply_ab_model()` 每次切模型都 rebuild graph，等於建立新的 random session。
- `../../TradingAgents/tradingagents/agents/utils/memory.py:37-38` 使用 `chromadb.Client(Settings(allow_reset=True))`，程式碼中看不到 persistent path 或磁碟目錄設定。

**影響**

- 同一進程內、同一 graph instance 期間，反思記憶可用。
- 但跨 restart、跨 AB model rebuild、甚至 fallback rebuild 時，記憶連續性很可能中斷。
- 這與 roadmap v1.4.0 所要求的「確認記憶持久化與真正學習閉環」還有明顯距離。

**判定**

- `Orient` 有能力，但不像穩定的長期記憶，更像短生命週期的 session memory。

### 5. Medium: Optimization / AB state 只在 daily summary 刷新，非啟動即生效

- `src/scheduler/scheduler.py:2017-2021` 只在 `_send_daily_summary()` 裡 `refresh_state()` 並 `set_ab_state(...)`。
- Scheduler 啟動時沒有先做一次 optimization state refresh。

**影響**

- 如果系統在 daily summary 時間之後才啟動，當日剩餘時間內可能都使用預設 thresholds 與空的 AB 狀態。
- 這使得 `Orient` 的「根據近期表現調整 gating / model routing」不是實時啟用，而是依賴一天一次的排程。

**判定**

- 存在學習引擎，但不是每次啟動都立即進入最新態。

### 6. Medium: Observe 仍以價格與排程為主，缺乏主動新聞事件觀察與低延遲事件流

- repo 中不存在 `news_event_trigger.py`。
- repo 中不存在 `fx_websocket_client.py`。
- `src/scheduler/scheduler.py:1933-1955` 的 volatility loop 只根據報價波動 `self._rescan_event.set()`。
- `../../TradingAgents` 的 news / macro 分析是決策當下才拉取資料，不是常駐事件監控。
- `src/scheduler/volatility_monitor.py:38` 還有一個 hardcoded 的全域 15 分鐘 trigger 間隔 `_global_min_interval_seconds = 900`。

**影響**

- 系統能「看到價格已經動了」，但仍看不到「新聞剛發、價格還沒完全動」。
- 多品種連續衝擊時，全域 trigger throttle 也可能壓制觀察敏感度。

**判定**

- `Observe` 有價格觀察，不具備 roadmap v1.4.0 要求的事件觀察。

### 7. Medium: 已抽取的結構化風控資訊與 journal 資料，尚未真正回流到決策核心

- `src/decision/agent_bridge.py:48-62` 有 `extract_risk_meta()`。
- `src/decision/agent_bridge.py:445` 會產出 `risk_meta`。
- 但 repo 搜尋顯示 `risk_meta` 沒有被 scheduler / execution / tactical validator 實際消費。

另外：

- scheduler / execution 在 journal 中主要寫入 `log_event()`，見 `src/scheduler/scheduler.py:2182`、`src/execution/engine.py:754`。
- `TradeJournal.get_closed_trades()` / `get_daily_returns()` 存在於 `src/monitor/trade_journal.py:69-126`，但 live scheduler 並不產生完整 `type=TRADE, status=CLOSED` 的結構化 trade record。

**影響**

- 系統有收集資料，但沒有充分把這些資料轉成可執行的 decision context。
- 這使 `Orient` 的資訊利用率偏低。

---

## OODA 對照評估

### O — Observe

**已實作**

- Qlib scanner 觀察日線信號，且有 stale signal guard：`src/signal/scanner_bridge.py:165-175, 361-396`
- 波動率監控觀察報價異常：`src/scheduler/scheduler.py:1933-1955`、`src/scheduler/volatility_monitor.py:54-87`
- Tactical validator 觀察 5min / 1h bars 與 spread / freshness：`src/scheduler/scheduler.py:926-1047`
- Position monitor 觀察 open positions、unrealized PnL、關閉事件：`src/scheduler/scheduler.py:1140-1178, 1233-1528`
- TradingAgents 內部在 decision 當下觀察 market / macro / news / social：`../../TradingAgents/tradingagents/graph/trading_graph.py:225-301`

**結論**

- `Observe` 主幹存在，且比 v1.3.5 強很多。
- 但它仍然是 **price-driven + schedule-driven**，不是 **event-driven**。
- 新聞事件、經濟日曆、WebSocket tick 流都還沒有主動觀察器。

**評價**

- `Observe = 中等偏強`

### O — Orient

**已實作**

- 多 agent 分析整合 market / macro / news / social 輸出為單一 decision state
- Best Day、circuit breaker、same-direction limit、low-confidence cooldown 都在修正「情境解讀」：`src/scheduler/scheduler.py`
- Threshold engine 根據歷史勝率調整 confidence 門檻：`src/optimize/thresholds.py`
- TradingAgents 有 reflection + memory retrieval：
  `../../TradingAgents/tradingagents/graph/trading_graph.py:430`
  `../../TradingAgents/tradingagents/agents/trader/trader.py:27, 71`

**不足**

- optimization state 不是啟動即 refresh
- historical PnL summary 沒有注入 prompt
- MemoryJournal 只是 Markdown 記錄，不是可查詢記憶
- Chroma memory persistence / session continuity 不穩定
- `risk_meta` 抽取了但沒進入後續 gate / execution logic

**結論**

- `Orient` 是目前最弱的一環。
- 系統已經會「記錄」與「反思」，但還沒有把這些結果穩定轉為每次決策前都可依賴的上下文。

**評價**

- `Orient = 偏弱`

### D — Decide

**已實作**

- Scanner -> intent -> LLM -> validated decision -> format_decision 的主鏈路完整：`src/scheduler/scheduler.py:540-861`
- HOLD 決策會取消 intent，並清掉同 symbol 的 stale ready intent：`src/scheduler/scheduler.py:853-895`
- Tactical gate 會在 strategic decision 之後做次級判定：`src/scheduler/scheduler.py:778-839`
- Compliance 與 execution-side best day gate 形成決策後的最後篩選：`src/execution/engine.py:148-221`

**不足**

- `StrategicDecisionCache` 未接線，重複決策問題仍在
- `tactical_pending` 未接線，WAIT 不會重判
- AB model routing 不是啟動即有效
- `risk_meta.max_same_day_attempts` 等資料沒有影響決策流

**結論**

- `Decide` 能做出結構化決定，但還缺少 stateful、成本敏感、可等待重判的決策控制。

**評價**

- `Decide = 中等`

### A — Act

**已實作**

- ExecutionEngine 會做 compliance、quote 檢查、random delay、下單、SL/TP、slippage 檢查：`src/execution/engine.py:119-430`
- Position monitor 可做 best-day close、breakeven、re-evaluation close：`src/scheduler/scheduler.py:1140-1178, 1530-1886`
- Close 後會回寫 store、alert、metrics、reflect：`src/scheduler/scheduler.py:1412-1489`

**不足**

- Scheduler mode 下 equity monitor 沒接 `on_alert` / `on_emergency_close`
- 沒有 roadmap v1.4.0 的分級反應（80% 減倉、90% 全平）
- 波動 trigger 不會立即強制一次 equity refresh
- Tactical WAIT 沒有「等下一根 bar 再 act」的延遲執行機制

**結論**

- `Act` 在下單與平倉細節上很完整，但在「極端事件風控執行」與「延遲進場重試」兩個方向還沒完全閉合。

**評價**

- `Act = 中等`

---

## 綜合判定

### 現況是否符合 OODA？

**部分符合，但不是完整符合。**

更精確地說：

- 有 `Observe -> Decide -> Act` 主鏈
- 有部分 `Orient` 能力
- 但 **Orient 不夠穩定、Observe 不夠主動、Decide 不夠 stateful、Act 在 scheduler 模式的風控鏈有斷線**

如果用一句話描述目前架構：

> `v1.3.9b` 是一個「具備 OODA 骨架，但尚未完成閉環與事件驅動化」的版本。

---

## 與 roadmap v1.4.0 的符合度

| v1.4.0 項目 | 現況 | 判定 |
|---|---|---|
| 歷史盈虧注入 LLM 提示詞 | 找不到在 `prop-firm-pilot` 注入 prompt 的實作 | ❌ 未實作 |
| `reflect_and_remember()` 啟用 | 已呼叫，但記憶 persistence / session continuity 不穩 | ⚠️ 部分實作 |
| `NewsEventTrigger` | repo 無此模組 | ❌ 未實作 |
| 緊急平倉增強 | 有 equity monitor 類別，但 scheduler 主模式未接 callback，且沒有 80%/90% 分級反應 | ⚠️ 部分實作 |
| WebSocket PoC | repo 無 `fx_websocket_client.py` | ❌ 未實作 |

---

## 我對 v1.4.0 實作前的建議優先序

### P0

1. 先修正 scheduler mode 的 equity monitor callback wiring
   否則風控的 `Act` 仍然不完整。

2. 把 tactical WAIT 改成真正的 `tactical_pending + retry + expire_action`
   否則 tactical layer 不符合 roadmap，也不符合 OODA 的再觀察重判。

3. 把 `StrategicDecisionCache` 接到 `_process_claimed_intent()`
   否則 v1.4.0 再加 news trigger 之後，LLM 重算成本只會更高。

### P1

4. 固定 TradingAgents session_id，並把 memory backend 改成可持久化 storage
   否則 `reflect_and_remember()` 的收益在 restart / model switch 後會大幅流失。

5. 在 scheduler 啟動時先 refresh optimization state + set_ab_state
   避免當日長時間跑在 default thresholds。

### P2

6. 再做 `NewsEventTrigger`、historical PnL prompt injection、volatility -> immediate equity refresh、分級減倉

---

## 本次審查使用的實證材料

### 直接檢查的核心檔案

- `src/scheduler/scheduler.py`
- `src/scheduler/volatility_monitor.py`
- `src/decision/tactical_validator.py`
- `src/decision/agent_bridge.py`
- `src/execution/engine.py`
- `src/monitor/equity_monitor.py`
- `src/monitor/trade_journal.py`
- `src/monitor/memory_journal.py`
- `src/optimize/optimization_engine.py`
- `src/optimize/thresholds.py`
- `src/signal/scanner_bridge.py`
- `../../TradingAgents/tradingagents/graph/trading_graph.py`
- `../../TradingAgents/tradingagents/agents/trader/trader.py`
- `../../TradingAgents/tradingagents/agents/utils/memory.py`
- `../../TradingAgents/tradingagents/graph/reflection.py`

### 針對性測試

已執行並通過：

- `uv run pytest tests/test_scheduler.py::TestEquityMonitorLoop -q` -> `2 passed`
- `uv run pytest tests/test_decision_store.py -k tactical_pending -q` -> `5 passed`
- `uv run pytest tests/test_decision_cache.py -q` -> `11 passed`
- `uv run pytest tests/test_scheduler.py -k "tactical_gate_blocks_intent_when_shadow_mode_off or tactical_gate_passes_in_shadow_mode" -q` -> `2 passed`

### 補充說明

- 本次沒有深入審查 `../../qlib_market_scanner` 內部因子與模型細節；對 OODA 的評估以 `ScannerBridge` 的使用方式與 scheduler 整合為主。
- 對 TradingAgents 的判斷，是基於本地可讀到的 source code，而不是只看 bridge 側註解。

---

## 最終結論

在進入 v1.4.0 之前，`v1.3.9b` 最需要先補的不是新功能數量，而是 **把既有 OODA 骨架中的幾個斷點接通**：

1. scheduler 下的風控執行 callback
2. tactical pending / retry 閉環
3. strategic decision cache
4. memory persistence 與 session continuity

這四件事補上後，再做 `NewsEventTrigger`、historical PnL prompt injection、WebSocket PoC，v1.4.0 才會從「功能堆疊」變成「真正更完整的 OODA 閉環」。
