# PropFirmPilot v1.4.1 — Production Incident Report + Remediation Priorities

> **報告日期**: 2026-03-12  
> **目標版本**: v1.4.1（Planned）  
> **基準版本**: v1.4.0  
> **觀測窗口**: 2026-03-11 16:33:07 至 2026-03-12 13:58:32（UTC+8）  
> **資料來源**: `prod_logs_20260312_v1.4.0/` production log bundle  
> **定位**: 這不是新功能版本規劃，而是以 production incident review 為基礎的 reliability / observability hardening release 定義

---

## 1. 執行摘要

這次 production run 沒有出現 crash、hard breach、emergency close 或 broker-side 致命錯誤，系統整體可用性仍屬可接受水準；但從 v1.4.0 的產品承諾來看，實際運行品質只可評為**中等**，不適合直接視為穩定完成版。

本次 incident review 的結論是：

- **v1.4.1 應定義為 reliability / observability hardening release**
- **不應再加入新功能主題**
- **優先修復 learning data correctness、runtime version identity、market-data degradation visibility**

整體評估如下：

| 面向 | 評估 |
|---|---|
| Availability | 高 |
| Risk / Compliance Safety | 高 |
| Execution Stability | 中高 |
| Market Data Quality | 中 |
| Observability / Postmortem Quality | 中低 |
| Strategy Throughput | 低 |
| 綜合 | **6/10** |

---

## 2. 觀測範圍與限制

### 2.1 實際可觀測窗口

雖然使用者指定的窗口是到 `2026-03-12 14:11`，但這份 bundle 的實際結束時間是：

- `raw/logs/prop_firm_pilot.log` 最後一筆：`2026-03-12 13:58:32`
- `INDEX.md` 打包時間：`2026-03-12 05:58:57 UTC` = `2026-03-12 13:58:57 UTC+8`

因此本報告的 runtime assessment 以 `2026-03-12 13:58:32` 為準。

### 2.2 資料來源

- 主 runtime log：`prod_logs_20260312_v1.4.0/raw/logs/prop_firm_pilot.log`
- 交易 journal：`prod_logs_20260312_v1.4.0/raw/data/trade_journal_e8_one_5k.jsonl`
- learning / memory journal：`prod_logs_20260312_v1.4.0/raw/MEMORY_E8_ONE_5K/2026-03-11.md`
- bundle manifest：`prod_logs_20260312_v1.4.0/INDEX.md`

### 2.3 重要說明

本報告以觀測窗口為主，但若同一個 production bundle 中存在會影響 v1.4.0 主打能力的**版本級缺陷**，即使發生時間略早於窗口，也會被納入 v1.4.1 的 remediation scope。最典型案例是 learning loop 的結果配對污染。

---

## 3. 觀測摘要

### 3.1 Runtime 統計

| 指標 | 數值 |
|---|---:|
| Log lines in window | 2131 |
| `INFO` | 2067 |
| `WARNING` | 64 |
| `ERROR` / `CRITICAL` | 0 |
| Scan starts | 33 |
| Early scans | 7 |
| Janitor cycles | 23 |
| Telegram commands handled | 17 |
| WebSocket failures | 17 |
| EODHD fetch retries | 2 |
| MatchTrader retries | 1 |
| Cooldown skips | 25 |

### 3.2 Trading / Decision 統計

以窗口化 `trade_journal` 計算：

| 指標 | 數值 |
|---|---:|
| `INTENT_CREATED` | 8 |
| `INTENT_CANCELLED` | 8 |
| New trade opened | 0 |
| Trade closed | 0 |
| `EQUITY_CHECK_ON_DEMAND` | 7 |
| `SAFE` checks | 7 |

補充觀測：

- 既有 `AUDUSD` 持倉在窗口內被重評估 **11 次**
- 平均決策延遲約 **11.95 分鐘**
- `worst_pct` 由 **0.2796%** 下降至 **0.1343%**

### 3.3 Market Data 效率摘要

窗口內觀測到：

- `EODHD` fetch events: **1646**
- fetched rows total: **4,397,485**
- 並多次重抓相同兩天窗口的 `1min` bars

這代表 v1.4.0 雖然名義上是 WebSocket-first，但 production 上仍大量退化為 REST-heavy 行為。

---

## 4. Incident 分級總覽

| ID | Priority | Incident | 影響 |
|---|---|---|---|
| I-01 | P0 | Reflection / memory result mapping 污染 | 直接污染 closed-trade learning loop，屬 correctness defect |
| I-02 | P1 | Runtime version identity mismatch | 版本辨識失真，release audit 與 postmortem 會誤判 |
| I-03 | P1 | WebSocket-first degraded into REST-heavy polling | 降低資料即時性與效率，掩蓋資料面退化 |
| I-04 | P1 | Telegram polling circuit degraded twice | 操作面可用性下降，控制面有短時降級 |
| I-05 | P2 | Strategy throughput stuck in cooldown / cancel loop | 沒有新增成交，產能偏低但未見失控 |
| I-06 | P2 | Log bundle manifest incomplete / references missing | 影響 incident review 與後續分析效率 |

---

## 5. Incident 詳細說明

### I-01 — P0: Reflection / Memory Result Mapping 污染

**現象**

同一份 memory journal 中，decision context 寫的是 `AUDUSD SELL`，但最後 `Trade Result` 卻掛上：

- `Symbol: EURUSD`
- `PnL: 14.9`
- `Reason: tp_hit`

**證據**

- `raw/MEMORY_E8_ONE_5K/2026-03-11.md`
- 該檔開頭為 `## 03:40:39 UTC - AUDUSD SELL`
- 同檔結尾 `Trade Result` 顯示 `EURUSD`

**影響**

- 這不是單純報表瑕疵，而是 **learning loop correctness defect**
- 後續 `retrieved_trade_lessons` 可能把錯誤 outcome 當成正確案例回灌到未來決策
- 會讓 v1.4.0 的「closed-trade learning」主打能力失真

**初步判斷**

反射寫回階段很可能沒有對 `symbol / intent_id / position_id / close event` 做足夠嚴格的 identity binding。

**v1.4.1 修復要求**

- reflection outcome 必須以 `position_id` 為主鍵，`intent_id + symbol` 為交叉驗證
- 任一 identity mismatch 時，**不得寫入 lesson / memory journal**
- mismatch 必須記錄為 warning/error metric，而不是 silent fallback

**完成標準**

- 新增 regression test，覆蓋 cross-symbol mismatch case
- 新增 guard 後，任何 `AUDUSD` decision 不可能再落入 `EURUSD` close result
- production log 中能看到 mismatch 被攔下而非被寫入

---

### I-02 — P1: Runtime Version Identity Mismatch

**現象**

- bundle manifest 標示版本為 `v1.4.0`
- runtime 啟動 log 卻寫 `PropFirmPilot v1.3.9 starting`
- `src/main.py` 中也仍然硬編碼 `v1.3.9`

**證據**

- `prod_logs_20260312_v1.4.0/INDEX.md`
- `prod_logs_20260312_v1.4.0/raw/logs/prop_firm_pilot.log`
- `src/main.py`

**影響**

- 會直接誤導 release audit、incident attribution、rollback 判斷
- 讓生產封包版本與 runtime 身分脫節
- 在多版本 hotfix 時尤其危險

**v1.4.1 修復要求**

- 建立單一版本來源，供 runtime、packer、report、changelog 共用
- 禁止在 `main.py` 再保留手寫版本字串

**完成標準**

- runtime start log、log pack version、文件版本一致
- CI 或 release script 對版本不一致會 fail-fast

---

### I-03 — P1: WebSocket-First Degraded into REST-Heavy Polling

**現象**

窗口內發生：

- **17 次** EODHD WebSocket failure
- **1646 次** REST fetch
- **4,397,485 rows** fetched
- 多次在 1 分鐘內重抓同樣 `2026-03-10 to 2026-03-12` 的 `1min` bars

典型故障型態：

- `keepalive ping timeout`
- `timed out during opening handshake`

**影響**

- 資料路徑雖未完全失效，但已明顯偏離 `v1.4.0` 的 WebSocket-first 承諾
- 造成不必要的網路成本、處理延遲與 degraded-state 隱藏
- 高波動時可能降低 tactical / volatility 反應品質

**初步判斷**

目前 fallback 行為偏粗，可能在 websocket 不健康時直接退回「大區間歷史重抓」，而不是 bounded incremental refill。

**v1.4.1 修復要求**

- REST fallback 改為 incremental backfill，不可反覆全段重抓
- 對 websocket health 引入明確狀態：`healthy / degraded / disconnected`
- 對 fallback 次數、rows fetched、stale duration 建立 metrics 與 alert threshold
- 對 repeated reconnect 增加更清楚的 bounded backoff / jitter / recovery logging

**完成標準**

- 同一 symbol 不再反覆全量重抓兩天 `1min` bars
- 可以在 prod log 明確辨識當前 market-data source 與 degraded 原因
- fetch volume 與 rows fetched 顯著下降

---

### I-04 — P1: Telegram Polling Circuit Degraded Twice

**現象**

窗口內 Telegram polling 至少兩次進入 circuit open：

- `2026-03-11 21:35:32`
- `2026-03-11 22:32:45`

前後伴隨多次：

- `ReadTimeout`
- `ConnectTimeout`

雖然其後仍有成功處理 `/profit` 指令，代表不是 hard-down，但確實有 degraded period。

**影響**

- operator command latency 上升
- incident 發生時，人工觀測 / 控制面可能短暫失靈

**v1.4.1 修復要求**

- 將 telegram failure / circuit-open 記入正式 operational metrics
- 明確記錄 half-open / recovered transition
- circuit open 時應有更清楚的自我告警

**完成標準**

- metrics snapshot 能看到 telegram degrade / recover 次數
- circuit open 不再是 log-only 事件

---

### I-05 — P2: Strategy Throughput Stuck in Cooldown / Cancel Loop

**現象**

窗口內：

- `INTENT_CREATED = 8`
- `INTENT_CANCELLED = 8`
- `EURUSD` low-confidence cooldown skip = **25**
- 新開倉 = **0**
- 平倉 = **0**

實際上系統幾乎只在重評估既有 `AUDUSD SELL` 倉位。

**影響**

- 不是安全性事故，但代表 production throughput 非常低
- 目前難分辨這是合理保守，還是 threshold / cooldown tuning 過度抑制

**v1.4.1 修復要求**

- 先補 observability，不要先直接調鬆 threshold
- 對 cooldown skip 建立 shadow outcome 分析，回看被取消訊號若放行的表現
- 區分「策略刻意保守」與「配置錯殺」

**完成標準**

- 能用 metrics / report 回答：cooldown 造成的 skip 是否有成本
- v1.4.1 不以提高交易頻率為目標，而是以「可診斷」為目標

---

### I-06 — P2: Log Bundle Manifest Incomplete / References Missing

**現象**

`INDEX.md` 指向多個不存在的檔案，例如：

- `summary/decisions_summary.md`
- `summary/telegram_summary.md`
- `raw/telegram_messages.json`

實際 bundle 中沒有這些檔案。

**影響**

- 直接降低 postmortem 效率
- 讓 incident review 還要先做 manifest 校對
- 使 bundle 本身的可信度下降

**v1.4.1 修復要求**

- packer 僅列出實際存在檔案
- 或在缺檔時 fail-fast，不得生成錯誤 manifest

**完成標準**

- `INDEX.md` 不再指向缺失檔案
- log bundle 可直接用於 incident review，無需先人工驗證 manifest

---

## 6. 修復優先序（P0 / P1 / P2）

### P0 — 必須先完成，否則不應宣稱 v1.4.1 已完成

1. 修復 reflection / memory result identity mapping
2. 為 learning loop 加上 mismatch guard 與 regression tests

### P1 — v1.4.1 主體修復範圍

1. 建立單一版本來源，修正 runtime / pack / docs version drift
2. 將 market-data fallback 從粗粒度歷史重抓改為 bounded incremental refill
3. 對 websocket degraded state 建立正式 metrics / alert / source logging
4. 強化 Telegram polling degradation observability 與 recoverability

### P2 — 應納入 v1.4.1，但可排在 P0 / P1 之後

1. 對 cooldown / cancel loop 建立 shadow analysis 與診斷報表
2. 修正 log bundle packer manifest 完整性
3. 補齊 postmortem 需要的 summary artifacts 或移除不存在引用

> 2026-03-12 worktree 狀態：已補上 structured `SCANNER_SKIP` diagnostics、deterministic decisions / telegram fallback summaries，以及 dynamic `INDEX.md` summary listing；正式 release 狀態仍待其餘使用者核定項目完成後再更新。

---

## 7. 建議的 v1.4.1 交付定義

v1.4.1 應被定義為：

> **Production Reliability / Observability Hardening for v1.4.0**

不建議在此版本夾帶新的策略、風控或功能擴張。此版的成功標準不是「功能更多」，而是：

- learning data 不再可被污染
- runtime 版本資訊可被信任
- websocket degraded state 可被正確觀測
- REST fallback 成本被明顯收斂
- operator control-plane 降級事件可被追蹤
- log bundle 可直接支援 postmortem

---

## 8. v1.4.1 驗收標準

### 8.1 Correctness

- 無 cross-symbol reflection / lesson write
- version string 在 runtime、bundle、docs 一致

### 8.2 Observability

- log / metrics 能區分 websocket `healthy / degraded / disconnected`
- telegram circuit open / recover 有明確 metric
- prod log bundle manifest 不再失真

### 8.3 Efficiency

- 不再反覆全量重抓兩天 `1min` bars
- market-data fallback 成本顯著下降

### 8.4 Safety

- 不降低既有 compliance guard
- 不因 observability hardening 改動 broker execution safety boundary

---

## 9. 不在 v1.4.1 範圍內的事項

以下項目不建議在 v1.4.1 一起處理：

- 新策略邏輯
- portfolio-level 新風控
- 重大 prompt / analyst 架構重寫
- WebSocket 功能擴張到新的 provider
- 以「提高交易頻率」為目標的 threshold 調整

---

## 10. 一句話定義

**v1.4.1 不是新功能版，而是把 v1.4.0 從「可運行」推進到「可被信任地運行」的修復版。**
