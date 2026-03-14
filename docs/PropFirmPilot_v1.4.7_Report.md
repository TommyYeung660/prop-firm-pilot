# PropFirmPilot v1.4.7 — Tactical Entry Control Plane 情景報告

> **報告日期**: 2026-03-14  
> **版本**: v1.4.7（Tactical Entry Fixes & Optimization）  
> **基準版本**: v1.4.6d  
> **聚焦範圍**: 以 `prop-firm-pilot` 的 tactical entry control plane 為主，說明 `qlib_market_scanner`、`TradingAgents`、scheduler / tactical validator / journal / calibration snapshot 如何形成完整 entry 流程

---

## 目錄

### Part A — 報告定位

1. 報告目的
2. v1.4.7 的核心變化
3. 三倉庫角色分工

### Part B — 標準交易主流程

4. v1.4.7 標準 entry 流程
5. tactical 模組在主流程中的決策權限
6. `action`、`resolution` 與狀態轉移

### Part C — 十個實際運營型交易情景

7. 情景 1：歐盤突破型 EURUSD 多單，乾淨直通
8. 情景 2：紐約盤 GBPUSD 空單，先等 spread 收斂再進場
9. 情景 3：東京盤 AUDUSD 多單，freshness 恢復不了而 timeout
10. 情景 4：事件後 USDJPY 空單，mixed-source 仍可安全放行
11. 情景 5：資料只剩 REST fallback，先 WAIT 再重判通過
12. 情景 6：條件不完美但仍可做，走 controlled degrade 放行
13. 情景 7：完全沒有 tactical data，系統直接 skip
14. 情景 8：scanner 給候選，但 TradingAgents 最終 HOLD
15. 情景 9：shadow mode 觀察期，先記錄 tactical verdict 不立刻阻擋
16. 情景 10：日終 calibration snapshot，將所有 tactical friction 結構化

### Part D — 總結

17. v1.4.7 對交易流程的真正影響
18. 對 operator 與後續版本的意義

---

## 1. 報告目的

這份報告不是單純列出功能清單，而是用 **10 個貼近真實 FX 生產運營的交易情景**，去展示 `v1.4.7` 完整 entry 流程如何運作。

重點不是：

- scanner 有沒有出訊號
- TradingAgents 有沒有輸出 BUY / SELL / HOLD

而是：

- tactical 模組如何把「可交易」與「暫時不該進」區分開來
- tactical verdict 如何變成 deterministic、可回放、可校準的控制面
- 為什麼 `v1.4.7` 之後，entry 不再只是一個模糊的 if/else gate

以下情景都不是虛構 API，而是依目前 `v1.4.7` 已落地的實作路徑整理出的 **真實可發生運營型案例**。它們反映的是系統現在真的會怎麼走，而不是理想化流程圖。

---

## 2. v1.4.7 的核心變化

`v1.4.7` 將 tactical entry 從「執行前的最後一道檢查」升級為 **entry control plane**。

和舊版本相比，最重要的變化有四個：

1. 每次 tactical evaluation 都會產生結構化 verdict，而不只是 `PASS / WAIT / REJECT` 字串。
2. scheduler 改為依 `resolution` 來做狀態轉移，而不是手寫 scattered branching。
3. `TACTICAL_RESULT` event 現在帶有 reason code、gate breakdown、policy hints、provenance。
4. 每日會輸出 `TACTICAL_ENTRY_CALIBRATION_SNAPSHOT`，把整天的 tactical friction 匯總成可分析資料。

這表示 v1.4.7 的 tactical 模組不只是「阻擋器」，而是：

- entry timing adjudicator
- provenance recorder
- lifecycle policy driver
- calibration data source

---

## 3. 三倉庫角色分工

### 3.1 `qlib_market_scanner`

`qlib_market_scanner` 的工作是找出「值得進入 decision pipeline 的候選」。

它提供的不是直接下單指令，而是候選上下文，例如：

- `scanner_score`
- `scanner_confidence`
- `scanner_score_gap`
- `scanner_drop_distance`
- `scanner_topk_spread`

它回答的是：

- 哪些 symbol 今天值得看
- 哪些 ranking / score 值得送進下一層判斷

它**不負責**：

- 判定最終 BUY / SELL / HOLD
- 判定當下 5m/1h timing 是否適合進場
- 決定 execution state transition

### 3.2 `TradingAgents`

`TradingAgents` 的工作是把 scanner 候選轉成可執行的策略方向與風險語境。

它會根據 scanner context、當前市場資料、歷史 lessons、PnL context、market event context 等資訊，輸出：

- `BUY / SELL / HOLD`
- `risk_report`
- `final_state`

它回答的是：

- 這個候選現在應不應該做方向性下注
- 風險敘事是什麼
- 策略角度偏多、偏空或觀望

它**不負責**：

- 判斷當下 spread/freshness/ATR 是否允許進場
- 自行決定 broker execution 細節
- 覆寫 tactical 的 timing correctness

### 3.3 `prop-firm-pilot` 的 tactical entry control plane

`prop-firm-pilot` 的 tactical 模組在 `v1.4.7` 之後負責：

- 以 5m/1h 與 quote 資料判定「現在能不能進」
- 產生 `action + resolution + reason_code + provenance`
- 驅動 `claimed / tactical_pending / ready_for_exec / timed_out / cancelled`
- 記錄 tactical friction 與 calibration snapshot

它回答的是：

- 不是「做不做方向」，而是「現在進還是等、放棄、降級放行」

---

## 4. v1.4.7 標準 entry 流程

標準流程如下：

1. `qlib_market_scanner` 產生候選 signal。
2. scheduler 將候選寫入 `TradeIntent`，狀態為 `pending`。
3. LLM worker claim intent，狀態變成 `claimed`。
4. `TradingAgents` 輸出 `BUY / SELL / HOLD` 與 risk context。
5. 若為 `BUY / SELL`，scheduler 呼叫 tactical validator。
6. tactical validator 產生 `TacticalResult`：
   - `action`
   - `resolution`
   - `summary_reason_code`
   - hard / soft gate breakdown
   - `policy_hints`
   - `provenance`
7. scheduler 根據 `resolution` 轉移狀態：
   - `EXECUTE_NOW` -> `ready_for_exec`
   - `RETRY_PENDING` -> `tactical_pending`
   - `EXECUTE_DEGRADED` -> `ready_for_exec`
   - `SKIP_CANCEL` -> `cancelled`
   - `EXPIRE_TIMEOUT` -> `timed_out`
8. execution engine 只處理 `ready_for_exec`。
9. trade journal 寫入完整 `TACTICAL_RESULT`。
10. 日終由 `TACTICAL_ENTRY_CALIBRATION_SNAPSHOT` 彙總一整天 verdict 分布。

---

## 5. tactical 模組在主流程中的決策權限

tactical 模組的權限刻意被設計成「只控制 entry correctness，不控制 strategy thesis」。

它可以做的事：

- 延後進場
- 放棄進場
- 在 policy 允許時降級放行
- 提供 deterministic reason code

它不可以做的事：

- 將 `BUY` 改成 `SELL`
- 直接替代 TradingAgents
- 改寫 execution engine 規則

因此，v1.4.7 的 tactical 模組是**戰術執行控制面**，不是另一套策略引擎。

---

## 6. `action`、`resolution` 與狀態轉移

為了理解下面十個情景，先看三個關鍵層次：

### 6.1 `action`

這是 tactical judgement 的表面結論：

- `PASS`
- `WAIT`
- `REJECT`

### 6.2 `resolution`

這是 scheduler 真正消費的控制語義：

- `EXECUTE_NOW`
- `RETRY_PENDING`
- `EXECUTE_DEGRADED`
- `SKIP_CANCEL`
- `EXPIRE_TIMEOUT`

### 6.3 狀態意義

- `ready_for_exec`：entry 可以交給 execution engine
- `tactical_pending`：不是拒絕，而是等待條件恢復
- `cancelled`：系統決定不做這筆
- `timed_out`：不是不想做，而是等不到可接受的 timing

這一層，就是 v1.4.7 與舊版最大的差別。

---

## 7. 情景 1：歐盤突破型 EURUSD 多單，乾淨直通

### 情景背景

- session：London open
- symbol：`EURUSD`
- scanner 類型：突破型 ranking signal
- 市場條件：spread 正常、quote 新鮮、5m/1h momentum 一致

### 完整流程

1. `qlib_market_scanner` 給出 `EURUSD` 高分候選，`scanner_score=0.84`、`scanner_confidence=high`。
2. scheduler 建立 `TradeIntent`，LMM worker 將其 claim。
3. `TradingAgents` 根據 scanner context 與當前市場上下文，輸出 `BUY`，並附上 risk report。
4. tactical validator 拉到新鮮 quote 與 bars：
   - spread gate pass
   - ATR regime pass
   - data freshness pass
   - EMA momentum / RSI / candle quality 至少 2/3 通過
5. tactical verdict：
   - `action=PASS`
   - `resolution=EXECUTE_NOW`
   - `summary_reason_code=tactical.pass.all_gates_aligned`
6. scheduler 將 intent 從 `claimed` 轉為 `ready_for_exec`。
7. execution engine 開倉，倉位後續正常進入 `opened`。

### tactical 模組的影響

在這個案例裡，tactical 模組不是阻擋器，而是**證明這筆單在 entry timing 上是乾淨的**。

它的價值在於：

- 這不是盲信 scanner 或 LLM
- 這筆單之後能被回放，知道為什麼當時被直接放行
- 若未來這種 setup 績效差，可以從 calibration snapshot 反向看這種通過樣本的分布

---

## 8. 情景 2：紐約盤 GBPUSD 空單，先等 spread 收斂再進場

### 情景背景

- session：New York open 前後
- symbol：`GBPUSD`
- 市場條件：方向判斷偏空，但當下 spread 明顯放大

### 完整流程

1. scanner 將 `GBPUSD` 列為高排名候選。
2. `TradingAgents` 輸出 `SELL`，方向與 scanner 相符。
3. tactical validator 在第一次檢查時發現：
   - quote 新鮮
   - 5m/1h 方向可做
   - 但 `spread_ratio` 超過限制
4. tactical verdict：
   - `action=WAIT`
   - `resolution=RETRY_PENDING`
   - `summary_reason_code=spread.fail.ratio_too_wide`
5. scheduler 轉到 `tactical_pending`，等待下一次 tactical retry。
6. 第二次檢查時 spread 收斂，gate 改為 pass。
7. 第二次 verdict 變成：
   - `action=PASS`
   - `resolution=EXECUTE_NOW`
8. intent 從 `tactical_pending` 轉為 `ready_for_exec`，後續開倉。

### tactical 模組的影響

這是 v1.4.7 最典型的價值之一：**不是否決交易，而是把進場點從不健康時刻推遲到健康時刻**。

若沒有 tactical control plane，系統可能：

- 在高 spread 時直接進場
- 開局就吃掉不必要的交易成本

而 v1.4.7 讓這次等待是可解釋的、可記錄的、可量化的。

---

## 9. 情景 3：東京盤 AUDUSD 多單，freshness 恢復不了而 timeout

### 情景背景

- session：Tokyo session
- symbol：`AUDUSD`
- 市場條件：scanner 與 agent 都偏多，但行情時間戳一直不可靠

### 完整流程

1. scanner 發現 `AUDUSD` 條件符合。
2. `TradingAgents` 認為這筆單可以做 `BUY`。
3. tactical validator 在第一次檢查時發現：
   - bars 可能存在
   - 但 authoritative freshness basis 不可靠
   - quote timestamp 缺失或持續 stale
4. 第一次 verdict：
   - `action=WAIT`
   - `resolution=RETRY_PENDING`
   - `summary_reason_code=freshness.fail.timestamp_missing` 或對應 stale code
5. intent 進入 `tactical_pending`。
6. 多次 retry 後，freshness 仍沒恢復。
7. policy 最終將此意圖轉為：
   - `resolution=EXPIRE_TIMEOUT`
8. scheduler 將 intent 標為 `timed_out`，不再繼續送 execution。

### tactical 模組的影響

這個情景最能說明 v1.4.7 的 safety value：

- 它不是把「沒有足夠 entry 可信度」誤當成可以硬做
- 也不是用 generic failure 混過去
- 它會明確告訴你：這筆不是策略不想做，而是**等不到可接受的 tactical timing**

對 operator 來說，`timed_out` 和 `cancelled` 的分離非常重要。

---

## 10. 情景 4：事件後 USDJPY 空單，mixed-source 仍可安全放行

### 情景背景

- session：美盤重大消息後
- symbol：`USDJPY`
- 市場條件：quote 來自 websocket cache，但部分 bars 來自 fallback

### 完整流程

1. scanner 抓到消息後的順勢空頭候選。
2. `TradingAgents` 綜合事件背景與 intraday context，輸出 `SELL`。
3. tactical data 讀取時出現混合來源：
   - quote_source=`websocket_cache`
   - bars_5min_source=`rest_fallback`
   - bars_1h_source=`websocket_cache`
   - `data_source=mixed`
4. tactical gate 仍全部達標。
5. verdict：
   - `action=PASS`
   - `resolution=EXECUTE_NOW`
   - provenance 清楚記錄 `mixed`
6. 系統正常進場。

### tactical 模組的影響

這裡 tactical 模組做了兩件事：

1. 它沒有因為 source 不純就盲目拒單。
2. 它也沒有把 source 差異藏起來。

因此，這筆單雖然被允許進場，但日後若要檢討表現，可以準確知道：

- 這是 mixed-source pass
- 不是純 websocket path

這正是 `v1.4.7` 所說的 provenance completeness。

---

## 11. 情景 5：資料只剩 REST fallback，先 WAIT 再重判通過

### 情景背景

- session：流動性一般
- symbol：`NZDUSD`
- 市場條件：WebSocket feed 暫時不可用，只剩 REST fallback

### 完整流程

1. scanner 仍提出 `NZDUSD` 候選。
2. `TradingAgents` 給出 `BUY`。
3. 第一次 tactical check 時：
   - `quote_source=rest_fallback`
   - `bars_5min_source=rest_fallback`
   - `bars_1h_source=rest_fallback`
   - `data_source=rest_fallback`
4. 第一次 verdict 不是直接拒絕，而是：
   - `action=WAIT`
   - `resolution=RETRY_PENDING`
   - 常見原因可能是 spread 或 freshness 暫不理想
5. 第二次 retry 時，fallback 資料仍存在，但品質回到允許區間。
6. 第二次 verdict 轉為 `EXECUTE_NOW`，最後進場。

### tactical 模組的影響

這個案例展示了 v1.4.7 的 operational pragmatism：

- 不把 fallback 視為絕對禁止
- 也不把 fallback 視為與 websocket 完全等價

系統既能交易，也能留下清楚證據，讓日後統計 `rest_fallback_ratio` 與 `rest_fallback_wait_count`。

---

## 12. 情景 6：條件不完美但仍可做，走 controlled degrade 放行

### 情景背景

- session：美盤中段
- symbol：`EURUSD`
- 市場條件：方向仍然合理，但 soft gates 一直未完全達標

### 完整流程

1. scanner 提出高分候選。
2. `TradingAgents` 給出明確 `BUY`。
3. tactical validator 多次檢查後，hard gates 都正常，但 soft score 始終略低。
4. 系統先給出：
   - `action=WAIT`
   - `resolution=RETRY_PENDING`
5. 進入 `tactical_pending` 後，retry 幾次仍未變成理想 pass。
6. 若 policy 設定允許 degrade，最終轉成：
   - `resolution=EXECUTE_DEGRADED`
7. intent 轉回 `ready_for_exec`，並寫入 `TACTICAL_DEGRADED` event。

### tactical 模組的影響

這個情景是 v1.4.7 非常關鍵的新能力。

如果沒有 degrade，系統容易：

- 在邊界 setup 上 endless WAIT
- 增加 intent churn
- 造成「明明可做但永遠做不到」的 friction

如果沒有結構化 degrade，系統又會：

- 把降級放行和正常放行混在一起

而 v1.4.7 的做法是：

- 允許有限度放行
- 但把它標成一種獨立可審計路徑

---

## 13. 情景 7：完全沒有 tactical data，系統直接 skip

### 情景背景

- symbol：`GBPUSD`
- 市場條件：沒有 spread、沒有 bars、沒有可靠 freshness
- 來源：可能是極端資料缺口、初始化異常，或不完整的手動候選

### 完整流程

1. 候選進入 `claimed`。
2. `TradingAgents` 給出 `BUY` 或 `SELL`。
3. tactical validator 發現：
   - `current_spread=0`
   - `bars_5min` 空
   - `bars_1h` 空
   - `latest_bar_time=None`
4. tactical verdict：
   - `action=REJECT`
   - `resolution=SKIP_CANCEL`
   - `summary_reason_code=data.reject.no_tactical_inputs`
5. scheduler 將 intent 標為 `cancelled`，不會進 execution。

### tactical 模組的影響

這個情景的價值不是「阻擋一筆單」，而是建立一條明確規則：

- **No Data = No Trade**

而且它不再只是 log 中的一句 warning，而是標準化 verdict，可進 journal、metrics、snapshot。

---

## 14. 情景 8：scanner 給候選，但 TradingAgents 最終 HOLD

### 情景背景

- session：任何時段
- symbol：`XAUUSD`
- 市場條件：scanner 看起來有機會，但 agent 綜合判斷後選擇觀望

### 完整流程

1. `qlib_market_scanner` 仍然可以把 `XAUUSD` 放進候選池。
2. intent 被 claim 後送進 `TradingAgents`。
3. `TradingAgents` 綜合歷史記憶、market event、風險敘事後，輸出 `HOLD`。
4. scheduler 直接取消該 intent。
5. tactical validator 不會介入，因為 tactical 模組只處理 `BUY / SELL` entry timing。

### tactical 模組的影響

這個案例看似 tactical 沒發揮，但其實非常重要，因為它說明了**權責邊界被守住了**：

- scanner 負責「值得看」
- TradingAgents 負責「應不應做方向」
- tactical 模組負責「現在是不是好 timing」

v1.4.7 的價值之一，就是不讓 tactical 模組越權成為另一個策略引擎。

---

## 15. 情景 9：shadow mode 觀察期，先記錄 tactical verdict 不立刻阻擋

### 情景背景

- 版本 rollout 初期
- tactical policy 還在觀察，不希望立刻改 live entry 行為

### 完整流程

1. scanner 提供候選，`TradingAgents` 輸出 `BUY`。
2. tactical validator 得到一個偏保守的 verdict，例如：
   - `action=WAIT`
   - `resolution=RETRY_PENDING`
3. 但系統處於 `shadow_mode=True`。
4. scheduler 仍然會寫出完整 `TACTICAL_RESULT` event。
5. 不過這次不阻擋 live flow，intent 仍可繼續走向 `ready_for_exec`。

### tactical 模組的影響

這個情景不是交易 alpha，而是 release safety。

它讓 operator 可以先回答：

- tactical 模組如果真的 enforce，會擋下多少單
- 哪些 reason code 最常出現
- false WAIT 是否過高

也就是說，v1.4.7 的 tactical 模組不只服務 live trading，也服務**穩定上線本身**。

---

## 16. 情景 10：日終 calibration snapshot，將所有 tactical friction 結構化

### 情景背景

- 一整天內系統處理了多筆 `EURUSD / GBPUSD / USDJPY`
- 其中有：
  - 直接 pass
  - WAIT 後 pass
  - degrade 放行
  - timeout
  - rest fallback / mixed-source verdict

### 完整流程

1. 每一筆 tactical entry verdict 都已寫入 `TACTICAL_RESULT`。
2. daily summary 執行時，scheduler 先寫 `METRICS_SNAPSHOT`。
3. 接著呼叫 tactical entry aggregation：
   - 依 `symbol / session / regime` 分組
   - 計算 `pass_rate`
   - 計算 `wait_rate`
   - 計算 `degrade_rate`
   - 計算 `timeout_rate`
   - 計算 `rest_fallback_ratio`
   - 排出 `top_reason_codes`
4. 寫出 `TACTICAL_ENTRY_CALIBRATION_SNAPSHOT` event。

### tactical 模組的影響

這個情景是 v1.4.7 和更早版本最大的營運差異。

以前 operator 只能說：

- 今天感覺 tactical 很常卡
- 好像某幾個 pair 比較容易等太久

現在可以直接回答：

- 哪個 symbol 在哪個 session 最常 `RETRY_PENDING`
- 哪一類 reason code 排第一
- mixed-source pass 比例有多高
- degrade 是否只集中在某種 regime

這讓 tactical 模組第一次成為可校準的控制面，而不只是 runtime 黑盒。

---

## 17. v1.4.7 對交易流程的真正影響

如果把這十個情景濃縮成一句話，`v1.4.7` 做的不是「讓 entry 更保守」，而是：

**讓 entry 變得可解釋、可延後、可放棄、可降級、可統計。**

它對完整交易流程的影響可以總結成五點：

1. scanner 不再直接意味著可進場，它只是候選入口。
2. TradingAgents 不再單獨承擔 entry correctness，tactical control plane 補上最後一公里。
3. `WAIT` 不再是模糊狀態，而是可追蹤的 tactical friction。
4. `degrade` 不再是隱性放行，而是顯性、可審計、可統計的 release path。
5. 日終 calibration snapshot 讓 tactical 問題首次能被量化，而不是靠肉眼讀 log。

---

## 18. 對 operator 與後續版本的意義

對 operator 而言，v1.4.7 最重要的不是多了幾個欄位，而是多了三種能力：

- 你可以回放每一筆 entry 為什麼進、為什麼等、為什麼跳過。
- 你可以區分「策略不想做」和「戰術 timing 等不到」。
- 你可以把一天的 tactical friction 轉成 calibration input。

對後續版本而言，v1.4.7 的意義是把 entry control plane 做穩，讓：

- `v1.4.8` 可以專心做 exit hardening
- `v1.4.9` 可以進一步看 tactical control plane 與 capital efficiency 的整合

也就是說，這一版不是在追更多交易，而是在建立一條**值得信任的 entry 管線**。
