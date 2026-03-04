
# PropFirmPilot v1.3.5 — 發展路線圖

> **報告日期**: 2026-03-03  
> **當前版本**: v1.3.5（EODHD Intraday Dual-Timeframe: 1D Trend + 4H Entry）  
> **涵蓋範圍**: prop-firm-pilot · qlib_market_scanner · TradingAgents 三項目協作  
> **帳戶**: E8 Markets One-Phase $5,000 Challenge  

---

## 目錄

1. [情境優化 — 突發市場事件反應能力](#1-情境優化--突發市場事件反應能力)
2. [戰術執行模組 — 實盤進場時機驗證](#2-戰術執行模組--實盤進場時機驗證)
3. [學習優化 — 勝率改善與反饋迴圈](#3-學習優化--勝率改善與反饋迴圈)
4. [想法回顧 — v1.0 路線圖 vs 實際進度](#4-想法回顧--v10-路線圖-vs-實際進度)
5. [運行速度優化 — WebSocket 與延遲改善](#5-運行速度優化--websocket-與延遲改善)
6. [統一路線圖](#6-統一路線圖)

---

## 1. 情境優化 — 突發市場事件反應能力

### 1.1 問題場景

FX 市場的突發事件（地緣政治危機、央行緊急聲明、非農數據爆冷）可在 **數分鐘內** 引發 100–300 pips 的劇烈波動。以 E8 Markets $5,000 帳戶為例：

- **日內回撤上限**: 4%（$200）
- **最大回撤上限**: 6%（$300，trailing HWM）
- **單筆風險**: 0.5%（$25）

一次 NFP 數據爆冷可能在 **3 分鐘內** 觸發止損，若系統反應過慢，可能在逆勢持倉尚未平倉時已逼近回撤上限。

### 1.2 現有反應管道分析

以下是 v1.3.5 從市場事件到執行動作的完整延遲鏈：

```
市場事件發生（t = 0）
  │
  ├─[1] 波動率監控器偵測 ─────── 0–60s（輪詢間隔 60s）
  │     src/scheduler/volatility_monitor.py
  │     · 每 60s 透過 MatchTrader REST API 取得 bid/ask
  │     · 計算 30 分鐘滾動價格變動百分比
  │     · 閾值: 0.3%，冷卻: 900s（15 分鐘）
  │
  ├─[2] 觸發重新掃描 ──────────── 即時（asyncio.Event）
  │     scheduler.py: self._rescan_event.set()
  │
  ├─[3] Scanner 執行 ──────────── 5–10s（Qlib pipeline）
  │     qlib_market_scanner subprocess
  │
  ├─[4] LLM Worker 接收 Intent ── 0–30s（輪詢間隔 30s）
  │     scheduler.py: _llm_worker_loop()
  │
  ├─[5] LLM 分析師取得數據 ───── 5–15s（EODHD REST 同步呼叫）
  │     TradingAgents: news/macro/market analysts
  │
  ├─[6] LLM 推理與決策 ────────── 10–30s（多 Agent 辯論）
  │
  ├─[7] 執行迴圈處理 ──────────── 0–10s（輪詢間隔 10s）
  │
  ├─[8] 合規檢查 + 報價 ────────── 1–2s
  │
  ├─[9] 隨機延遲（反偵測）─────── 0.5–3s
  │
  └─[10] 訂單送出 ──────────────── 0.5–2s
```

| 場景 | 總延遲 | 說明 |
|------|:------:|------|
| **最佳** | ~25s | 波動觸發剛好命中 + Worker 空閒 |
| **典型** | ~81s | 各環節取平均值 |
| **最差** | ~5.2 小時 | 靜默時段 + 波動未達閾值 → 等待排程掃描 |

### 1.3 現有機制的盲點

| 盲點 | 說明 | 嚴重度 |
|------|------|:------:|
| **僅偵測價格波動，不偵測新聞事件** | 波動監控器只看 mid price 變動，無法在價格尚未波動時預判即將到來的衝擊 | 🔴 高 |
| **新聞感知是被動的** | TradingAgents 的 news/macro analyst 僅在 LLM 被觸發時才抓取新聞，不會主動監控 | 🔴 高 |
| **波動監控輪詢太慢** | 60s 間隔意味著最多 1 分鐘的偵測延遲 | 🟡 中 |
| **靜默時段掃描間隔過長** | 亞洲盤 4 小時一次掃描，若波動未達 0.3% 閾值則完全不掃描 | 🟡 中 |
| **冷卻期過長** | 波動觸發後 15 分鐘內不再觸發，連續衝擊會被忽略 | 🟡 中 |
| **LLM 數據工具為同步阻塞** | TradingAgents 使用 `requests.get()` 而非 async httpx | 🟢 低 |

### 1.4 改進路線圖

#### Phase A — 快速配置調優（v1.3.6，1–2 天）

無需改架構，僅調整參數即可顯著改善：

| 參數 | 現值 | 建議值 | 效果 |
|------|:----:|:------:|------|
| `volatility_poll_interval_seconds` | 60 | 15 | 偵測延遲從 60s → 15s |
| `volatility_cooldown_seconds` | 900 | 300 | 冷卻從 15 分鐘 → 5 分鐘 |
| `volatility_threshold_pct` | 0.3 | 0.2 | 更敏感的觸發閾值 |
| `llm_poll_interval_seconds` | 30 | 10 | Intent 等待時間減少 2/3 |

**預估效果**：典型場景延遲從 ~81s → ~40s。

#### Phase B — 新聞事件觸發器（v1.4.0，1–2 週）

新增 `NewsEventTrigger` 模組，獨立於價格波動監控：

```
                    ┌─────────────────────┐
                    │  NewsEventTrigger   │
                    │                     │
  EODHD News API ──┤  · 每 5 分鐘輪詢    │
  (or RSS feeds)   │  · 關鍵詞過濾:      │──→ rescan_event.set()
                    │    NFP, CPI, FOMC,  │
                    │    war, sanctions,   │
                    │    rate decision     │
                    └─────────────────────┘
```

**實作要點**：
1. 新模組 `src/scheduler/news_event_trigger.py`
2. 輪詢 EODHD `/api/news` + 關鍵詞匹配（支援正則）
3. 偵測到高影響新聞 → `rescan_event.set()` + Telegram 告警
4. 獨立冷卻期（避免同一事件重複觸發）

#### Phase C — 緊急平倉增強（v1.4.0，同步）

目前 equity monitor 在回撤達 90% 時觸發緊急平倉，但有 60s 輪詢延遲：

| 改進 | 說明 |
|------|------|
| **波動觸發 → 立即刷新權益** | 波動監控器偵測到異常時，額外觸發一次 equity 檢查 |
| **分級反應** | 80% → 縮小倉位（平掉最大虧損倉），90% → 全部平倉 |
| **Trailing Stop 動態收緊** | 波動率飆升時自動收緊止損距離 |

#### Phase D — 即時新聞串流（v2.0.0，長期）

升級為 **事件驅動架構**，徹底消除輪詢延遲：

1. **WebSocket 新聞源**：接入 EODHD WebSocket 或專業新聞 API（Bloomberg B-PIPE、Refinitiv）
2. **事件分類器**：NLP 模型即時判斷新聞影響等級（Low / Medium / High / Critical）
3. **優先通道**：Critical 級別新聞 → 繞過正常排隊 → 直接送入 LLM 分析 + 執行
4. **經濟日曆整合**：預先知道 NFP、FOMC 等高影響事件的發佈時間，在事件前 5 分鐘自動提高監控頻率

---

## 2. 戰術執行模組 — 實盤進場時機驗證

### 2.1 問題背景

2026-03-03 生產日誌揭示了一個關鍵架構盲點：

```
17:47–18:44  系統開設 3 筆 SELL AUDUSD（Qlib score=0.5423, TradingAgents=SELL）
18:34–05:51  波動監控反覆觸發（AUDUSD ±0.31%）→ 重新掃描
             → 相同的 Qlib 1D score（0.5423，日線級別，24h 內不變）
             → 相同的 TradingAgents SELL 決定
             → 新建 Intent → 被 BEST_DAY_RULE 拒絕
             → 此循環重複 15+ 次，每次浪費 ~10 分鐘 LLM 算力
```

**核心問題**：

| 問題 | 說明 | 影響 |
|------|------|:----:|
| **1D+4H 訊號日內靜態** | Qlib score=0.5423 在整個交易日內完全不變，4H TradingAgents 每次都返回相同結論 | 🔴 算力浪費 |
| **無低週期進場驗證** | 系統在決定 SELL 後立即執行，不檢查 5min/1min 是否正處於逆向反彈 | 🔴 差進場 |
| **無決策快取** | 每次波動觸發都重跑完整 Qlib+LLM 管道（~10 分鐘），即使結果必然相同 | 🔴 資源浪費 |
| **無 Intent 去重** | 已有相同方向的 Intent 在排隊/待驗證時，系統仍生成新的重複 Intent | 🟡 狀態混亂 |

**將軍與士兵的比喻**：

- 現有系統：將軍（1D Qlib + 4H TradingAgents）下達 SELL 命令 → 士兵盲目衝鋒
- 改進目標：將軍決定方向 → **士兵觀察實盤（5min/1H），判斷是否是進攻的好時機**
- 士兵**永不推翻**將軍的方向，只決定 **何時執行**（或判斷時機已過，放棄執行）

### 2.2 與現有路線圖功能重疊分析

| 現有功能 | 版本 | 是否重疊 | 說明 |
|---------|:----:|:-------:|------|
| 波動監控提速 15s | v1.3.6 | ❌ | 加速偵測，不涉及進場判斷 |
| 冷卻期縮短 5min | v1.3.6 | ❌ | 調整觸發頻率 |
| 反思迴圈啟用 | v1.3.6 | ❌ | 學習過去決策 |
| 歷史盈虧注入 LLM | v1.4.0 | ❌ | 改善決策品質 |
| NewsEventTrigger | v1.4.0 | ❌ | 偵測新聞事件 |
| WebSocket PoC | v1.4.0 | ⚠️ 互補 | 可為戰術層提供實時 tick 數據 |
| 動態 SL/TP | v1.5.0 | ⚠️ 互補 | 戰術層決定「是否進場」，動態 SL/TP 決定「進場後的目標」 |

**結論：無直接重疊。** 戰術模組填補的是「策略決策 → 訂單執行」之間的空白驗證層。

### 2.3 現有決策管道與插入點

```
Scanner (1D Qlib) → Intent 生成 → LLM Worker 認領
  → AgentBridge.decide() (4H TradingAgents, ~9 min)
  → format_decision() (混合置信度)
  → mark_ready_for_exec()          ← ⚡ 戰術驗證插入點
  → ExecutionEngine 認領
  → PropFirmGuard 合規檢查
  → MatchTrader 下單
```

**插入位置**：在 `Scheduler._process_claimed_intent()` 中，`format_decision()` 之後、`mark_ready_for_exec()` 之前。

**新增 Intent 狀態**：`pending → claimed → **tactical_pending** → ready_for_exec → executing → opened`

### 2.4 技術方案

#### 2.4.1 架構設計 — Hard/Soft 雙層門檻

基於生產環境的過濾退化研究（2 層確認可將勝率從 45% 提升至 62%，但 3 層確認會因錯失 60% 機會導致夏普比率退化），戰術門檻採用 **Hard/Soft 分離**，避免過度過濾導致系統永不交易：

```
                    Strategic Decision (BUY/SELL)
                            │
                    ┌───────┴───────┐
                    │  Hard Gates   │ ← 全部必須通過（AND）
                    │  · Spread     │    交易成本/安全/資料品質
                    │  · ATR Regime │
                    │  · Data Fresh │
                    └───────┬───────┘
                            │ 通過
                    ┌───────┴───────┐
                    │  Soft Gates   │ ← 評分制，達標即可
                    │  · EMA 動能   │    進場品質訊號
                    │  · RSI 狀態   │
                    │  · 蠟燭品質   │
                    └───────┬───────┘
                            │
                    ┌───────┴───────┐
                    │  PASS / WAIT  │
                    │  / REJECT     │
                    └───────────────┘
```

#### 2.4.2 Hard Gates（硬門檻 — 全部必須通過）

| Gate | 指標 | 條件 | 理由 |
|------|------|------|------|
| **Spread** | 當前 bid-ask spread | < 2× 該幣對典型 spread | 避免低流動性時段進場（亞盤交易 EUR 等） |
| **ATR Regime** | 1H ATR(14) vs rolling median | 0.5× < 當前 ATR < 2.5× median | 避免死寂市場（假突破）和極端波動（不可控） |
| **Data Freshness** | 最新 5min bar 時間戳 | 距今 < 10 分鐘 | 確保不基於陳舊數據做決策 |

#### 2.4.3 Soft Gates（軟門檻 — 評分制）

| Gate | 指標 | 條件（得分 +1） | 說明 |
|------|------|----------------|------|
| **動能方向** | 5min EMA(8) vs EMA(21) | BUY: EMA(8) > EMA(21)；SELL: EMA(8) < EMA(21) | 短期動能與策略方向一致 |
| **RSI 狀態** | 5min RSI(14) | BUY: RSI < 70；SELL: RSI > 30 | 未進入極端區域（加分項，非硬門檻） |
| **蠟燭品質** | 最近 5min bar body/range ratio | body_ratio > 0.3 | 價格行為有方向性（非十字星） |

**通過條件**：Soft Gates 得分 ≥ 2/3（至少 2 個軟門檻通過即可）。

⚠️ **為什麼 RSI 不是硬門檻**：5min RSI 在強趨勢日會把你擋在趨勢延伸段外。生產日誌中 AUDUSD 的 SELL 方向是正確的，如果 RSI 是硬門檻，系統可能在最好的趨勢日反而不交易。

#### 2.4.4 決策快取與 Intent 去重（同步實施）

戰術模組必須搭配以下兩個機制才能徹底解決生產問題：

**Strategic Decision Cache**：
- 同一 `symbol + direction` 在下一根 4H bar close 之前，不重跑 TradingAgents
- 波動觸發時只跑 tactical validation，不重跑 LLM（因為結果不會變）
- 預估節省：每次波動觸發從 ~10 分鐘 → ~10 秒（僅抓 5min 數據 + 計算指標）

**Intent 去重**：
- 同一 symbol 在 `tactical_pending` 期間，禁止生成新的同向 Intent
- 若已有倉位或剛平倉（< 30 分鐘），自動加入最短冷卻期避免撞上 Best Day Rule

#### 2.4.5 Retry 機制

| 參數 | 值 | 說明 |
|------|:--:|------|
| 檢查時機 | 對齊 5min bar close + ±10s jitter | 避免用未完成 K 線；jitter 避免多幣對同步打 API |
| 自適應頻率 | 低波動: 每 10–15min；正常: 每 5min；高波動: 每 5min | 基於 ATR 相對均值的倍率 |
| 最大等待時間 | 60 分鐘（12 次 × 5 分鐘） | 超過後 Intent 標記為 `tactical_expired` |
| 過期處理 | 降級門檻模式 | 到期不強制執行；改為只保留 Hard Gates + 放寬 Soft 為 1/3 |
| 方向變化 | 立即取消 | 若策略訊號在等待期間翻轉，立即 cancel Intent |

#### 2.4.6 Shadow Mode 上線策略

```
Phase 1: Shadow Mode（1–2 週）
  · 戰術模組正常運行並記錄所有 Gate 結果
  · 但不實際阻擋任何交易（僅記錄 + Telegram 通知）
  · 收集分佈數據：各 Gate 通過率、被擋交易的事後表現
  · 使用 DuckDB/JSONL 全量記錄每次檢查的 Gate 值

Phase 2: 門檻校準
  · 基於 Phase 1 數據分析：
    - 如果某 Gate 80% 時間都失敗 → 該 Gate 閾值不適合此幣對/時段
    - 如果被擋交易事後勝率 > 未被擋交易 → Gate 邏輯有誤
  · Walk-forward 比較：同一批策略訊號下，有/無 tactical 的成交品質差異

Phase 3: 正式啟用
  · 切換為真正阻擋模式
  · 持續記錄並每週回顧 Gate 分佈
```

#### 2.4.7 新增模組與配置

**新模組**: `src/decision/tactical_validator.py`

```python
class TacticalValidator:
    """戰術進場驗證器 — 低週期數據確認策略方向的執行時機。

    在 AgentBridge.decide() 返回 BUY/SELL 後、mark_ready_for_exec() 前執行。
    使用 5min/1H 數據驗證進場條件，不推翻策略方向，只決定執行時機。

    Usage:
        validator = TacticalValidator(config, data_fetcher)
        result = await validator.validate(intent, side="SELL")
        if result.passed:
            store.mark_ready_for_exec(intent.id)
    """

    async def validate(self, intent: TradeIntent, side: str) -> TacticalResult: ...
    async def _check_hard_gates(self, symbol: str, side: str) -> list[GateResult]: ...
    async def _check_soft_gates(self, symbol: str, side: str) -> list[GateResult]: ...
    async def _fetch_tactical_data(self, symbol: str) -> TacticalData: ...
```

**新模組**: `src/scheduler/decision_cache.py`

```python
class StrategicDecisionCache:
    """策略決策快取 — 同一 symbol+direction 在 4H 內不重跑 LLM。

    Usage:
        cache = StrategicDecisionCache()
        if cache.is_fresh(symbol, direction):
            # 跳過 LLM，直接進入 tactical validation
        else:
            decision = await agents.decide(...)
            cache.store(symbol, decision)
    """
```

**配置結構**:

```yaml
tactical:
  enabled: true                    # 全局開關
  shadow_mode: true                # Phase 1: 僅記錄不阻擋
  hard_gates:
    spread_max_multiplier: 2.0     # 當前 spread < 2× 典型值
    atr_min_ratio: 0.5             # ATR > 0.5× median
    atr_max_ratio: 2.5             # ATR < 2.5× median
    atr_period: 14
    atr_timeframe: "1h"
    data_max_age_seconds: 600      # 最新數據 < 10 分鐘
  soft_gates:
    min_score: 2                   # 至少 2/3 軟門檻通過
    ema_fast: 8
    ema_slow: 21
    ema_timeframe: "5min"
    ema_lookback_bars: 50
    rsi_period: 14
    rsi_overbought: 70
    rsi_oversold: 30
    candle_min_body_ratio: 0.3
  retry:
    interval_seconds: 300          # 5 分鐘（對齊 bar close）
    max_retries: 12                # 最長等待 1 小時
    expire_action: "degrade"       # 過期後降級門檻
    jitter_seconds: 10             # ±10s 隨機偏移
  decision_cache:
    ttl_seconds: 14400             # 4 小時 TTL
  intent_dedup:
    cooldown_after_close_seconds: 1800  # 平倉後 30 分鐘冷卻
```

### 2.5 預期效果

| 指標 | v1.3.5 現況 | v1.3.7 預期 | 說明 |
|------|:----------:|:----------:|------|
| LLM 重複調用 | 15+次/日（相同結果） | 1–2 次/日 | 決策快取消除 stale signal 重算 |
| 每次波動觸發成本 | ~10 分鐘 LLM | ~10 秒 tactical check | 跳過 LLM，只跑低週期指標 |
| Intent 重複生成 | 無限制 | 同向去重 | Intent 去重機制 |
| 逆勢進場 | 無檢查 | Hard+Soft Gate 過濾 | 避免在反彈中做空 |
| 平均 MAE（最大不利變動） | 未追蹤 | 預期降低 30–50% | 更好的進場時機 → 更小的逆向波動 |
| Telegram 告警 | 僅合規拒絕 | Gate 結果通知 | 全透明的戰術決策記錄 |

⚠️ **勝率影響**：不以勝率提升為主要指標。戰術模組的核心價值是**降低 MAE（最大不利變動）**、**改善平均風險回報比（R）**、以及**消除無效的 LLM 重算**。勝率可能小幅上升（+1–5pp），但更顯著的改善將體現在回撤控制和資源效率上。

### 2.6 實施計畫（v1.3.7，3–5 天）

| 天數 | 工作項 | 交付物 |
|:----:|--------|--------|
| D1 | TacticalValidator 核心 + Hard Gates | `src/decision/tactical_validator.py`，含 Spread/ATR/Data Freshness 三個硬門檻 |
| D1 | Soft Gates 實作 | EMA 動能 + RSI + 蠟燭品質評分邏輯 |
| D2 | StrategicDecisionCache + Intent 去重 | `src/scheduler/decision_cache.py`，修改 `scheduler.py` |
| D2 | Scheduler 整合 | 在 `_process_claimed_intent()` 插入 tactical 驗證 + 新狀態 `tactical_pending` |
| D3 | Shadow Mode + DuckDB 記錄 | Gate 結果全量寫入 DuckDB，Telegram 通知 |
| D3 | 配置結構 + 測試 | YAML 配置、單元測試、整合測試 |
| D4–5 | Shadow Mode 觀測 + 門檻校準 | 分析 Gate 通過率分佈，校準閾值 |

## 3. 學習優化 — 勝率改善與反饋迴圈

### 2.1 現有學習能力清單

基於三個項目的完整分析，以下是所有學習/優化/記憶模組的現狀：

| 模組 | 項目 | 狀態 | 說明 |
|------|------|:----:|------|
| **OptimizationEngine** | prop-firm-pilot | ✅ 運作中 | 勝率追蹤（14 天回望）、動態信心閾值、A/B 測試狀態 |
| **TradeStats** | prop-firm-pilot | ✅ 運作中 | 全局 + 每幣對勝率計算、PnL 聚合 |
| **Thresholds** | prop-firm-pilot | ✅ 運作中 | 根據勝率動態調整 min_confidence（勝率 < 45% → 提高至 0.65） |
| **MemoryJournal** | prop-firm-pilot | ⚠️ 僅記錄 | 每日 Markdown 日誌（MEMORY/YYYY-MM-DD.md），**無自動檢索** |
| **TradeJournal** | prop-firm-pilot | ⚠️ 僅記錄 | JSONL append-only 日誌，**無內建勝率分析** |
| **Reflection Engine** | TradingAgents | ⚠️ 未確認啟用 | 反思 bull/bear researcher、trader、judge、risk manager 決策 |
| **FinancialSituationMemory** | TradingAgents | ⚠️ 未確認啟用 | ChromaDB + OpenAI embeddings 向量記憶，支援語意相似度搜尋 |
| **Qlib Backtest** | qlib_market_scanner | ✅ 運作中 | IC/IR/Rank IC 指標計算，TopK+Dropout 策略回測 |
| **RD-Agent Factor Loader** | qlib_market_scanner | ✅ 運作中 | 載入 YAML 因子檔案，支援 IC IR 閾值過濾 |
| **RdAgentBridge** | prop-firm-pilot | ⚠️ 橋接存在 | subprocess 呼叫 qlib_rd_agent，核心邏輯未完整 |
| **qlib_rd_agent** | qlib_rd_agent | ❌ 未完成 | 專案結構存在，AI 因子發現核心邏輯未知 |

### 2.2 學習迴圈缺口分析

目前系統的學習能力呈 **斷裂狀態** — 各模組獨立運作但缺乏閉合的反饋迴圈：

```
                    ❌ 無閉環
                    ↗          ↘
  TradeJournal ──→ 記錄 PnL     OptimizationEngine ──→ 動態閾值
                    ↑                                    ↓
                    │          MemoryJournal              │
                    │          （僅 Markdown，           │
                    │            無語意檢索）             │
                    │                                    ↓
  TradingAgents ──→ 決策 ←──────── 閾值過濾 ←── min_confidence
        ↓
  FinancialSituationMemory  ←── reflect_and_remember()
  （ChromaDB 向量記憶）          ⚠️ 未確認是否在生產啟用
```

**關鍵缺口**：

| 缺口 | 影響 | 優先級 |
|------|------|:------:|
| **TradingAgents 反思迴圈未確認啟用** | LLM 不會從過去的錯誤決策中學習 | 🔴 P0 |
| **MemoryJournal 無語意檢索** | 歷史交易洞察無法自動注入決策流程 | 🔴 P0 |
| **TradeJournal → LLM 提示詞無連接** | LLM 不知道近期盈虧，無法避免重複犯錯 | 🔴 P0 |
| **RD-Agent 因子發現管道未完成** | 無法自動發現新 Alpha 因子，模型退化風險 | 🟡 P1 |
| **A/B 測試無自動路由** | 需手動切換模型，無法持續對比效果 | 🟡 P1 |

### 2.3 改進路線圖

#### Phase A — 啟用 TradingAgents 反思迴圈（v1.4.0，2–3 天）

**目標**：確認並啟用 `reflect_and_remember()` 方法。

1. **驗證調用路徑**：檢查 `agent_bridge.py` 是否在交易結束後呼叫 `graph.reflect_and_remember(returns_losses)`
2. **接入盈虧數據**：從 TradeJournal 讀取最近交易的 PnL 作為 `returns_losses` 參數
3. **向量記憶持久化**：確認 ChromaDB 存儲路徑配置正確，不會因重啟而遺失

**預期效果**：LLM 在後續決策中能自動檢索 "上次類似情境的決策結果"。

#### Phase B — 歷史盈虧注入 LLM 提示詞（v1.4.0，3–5 天）

這正是 v1.0 路線圖中 v1.1.0 計畫的 "歷史盈虧注入 LLM 提示詞"：

```python
# agent_bridge.py — 決策前注入最近交易表現
recent_trades = trade_journal.get_recent(days=7, symbol=symbol)
performance_context = f"""
## 近 7 天 {symbol} 交易表現
- 勝率: {recent_trades.win_rate:.1%}
- 平均盈利: +{recent_trades.avg_win:.1f} pips
- 平均虧損: -{recent_trades.avg_loss:.1f} pips
- 最近 3 筆: {recent_trades.last_3_summary}
- 教訓: {recent_trades.lessons_learned}
"""
# 注入到 market_analyst 的 system prompt
```

#### Phase C — MemoryJournal 語意升級（v1.5.0，1 週）

將 MemoryJournal 從 "被動 Markdown 日誌" 升級為 "可查詢的向量記憶"：

1. **遷移到 ChromaDB**：複用 TradingAgents 的 `FinancialSituationMemory` 架構
2. **自動嵌入**：每筆交易的決策推理、市場狀態、結果 → embedding → ChromaDB
3. **決策前查詢**：新交易前，自動搜尋 "最相似的歷史情境" 並注入提示詞

#### Phase D — RD-Agent 因子發現自動化（v2.0.0，長期）

```
每週六 03:00 UTC
  │
  ├─[1] prop-firm-pilot 觸發 RdAgentBridge.trigger_full_cycle()
  │
  ├─[2] qlib_rd_agent 執行 AI 因子發現
  │     · 基於最近 4 週交易數據
  │     · 嘗試 100+ 因子表達式
  │     · 保留 IC > 0.03, IR > 1.0 的因子
  │
  ├─[3] 輸出 discovered_factors.yaml → Dropbox 同步
  │
  └─[4] qlib_market_scanner 週一自動載入新因子
        · load_rdagent_factors() 已實作
        · 舊因子 vs 新因子 A/B 對比
```

### 2.4 勝率目標

| 指標 | v1.3.5 現況 | v1.4.0 目標 | v2.0.0 目標 |
|------|:----------:|:----------:|:----------:|
| Qlib Scanner IR | 1.179 (1D) | > 1.2 | > 1.5 |
| 實際勝率 | ~55% (回測) | > 58% | > 62% |
| 每月盈利目標 | 9%（$450） | 12%（$600） | 15%（$750） |
| 最大連續虧損 | 未追蹤 | < 5 筆 | < 4 筆 |

---

## 4. 想法回顧 — v1.0 路線圖 vs 實際進度

### 3.1 v1.0 §5.5 原始路線圖

v1.0 報告在 2026 年初規劃了以下版本路線：

```
v1.0.0 ─── 全自動 FX 交易系統正式版
v1.1.0 ─── LLM 回饋優化
v1.2.0 ─── Streamlit Dashboard
v1.3.0 ─── 多帳號管理
v2.0.0 ─── 進階交易策略
v3.0.0 ─── 實盤 $50k 帳號
```

### 3.2 實際進度對照表

| 原始計畫 | 計畫內容 | 實際實施版本 | 狀態 | 說明 |
|---------|---------|:----------:|:----:|------|
| **v1.0.0** | 全自動 FX 交易系統 | v1.0.0 | ✅ 完成 | 24/7 Scheduler、Alpha Vantage、3 倉位並行、548 測試 |
| **v1.1.0** | LLM 回饋優化 | — | ⏳ 部分完成 | |
| | □ 歷史盈虧注入 LLM 提示詞 | — | ❌ 未實施 | OptimizationEngine 追蹤勝率但未注入提示詞 |
| | □ 動態信心閾值 | v1.3.0 | ✅ 已實施 | `thresholds.py` 根據勝率動態調整 min_confidence |
| | □ 交易品質統計報告 | v1.3.0 | ✅ 已實施 | `trade_stats.py` 計算勝率、PnL 聚合 |
| **v1.2.0** | Streamlit Dashboard | — | ❌ 未開始 | 優先處理了數據源遷移和多時間框架 |
| | □ Portfolio Overview | — | ❌ | |
| | □ Trade History + Memory 瀏覽 | — | ❌ | |
| | □ 實時 Log 查看 | — | ❌ | |
| **v1.3.0** | 多帳號管理 | — | ❌ 未實施 | v1.3.0 實際用於交易業績修復 |
| | □ 帳號切換 | — | ❌ | |
| | □ 獨立數據隔離 | — | ❌ | |
| | □ 統一 Telegram 通知 | — | ❌ | Telegram Bot 已有但僅支援單帳號 |
| **v2.0.0** | 進階交易策略 | — | ⏳ 大幅提前 | |
| | □ 日內信號 (4H/1H) | **v1.3.5** | ✅ 已實施 | EODHD intraday 4H 聚合 + dual-timeframe |
| | □ 動態 SL/TP (ATR-based) | — | ❌ 未實施 | TradingAgents 計算 ATR 但未接入 SL/TP |
| | □ 相關性檢測 | — | ❌ 未實施 | |
| | □ 週末因子進化 | — | ⚠️ 橋接存在 | RdAgentBridge 已寫，核心未完成 |
| **v3.0.0** | 實盤 $50k 帳號 | — | ❌ 未開始 | 尚在 Phase 1 $5k 挑戰賽中 |
| | □ Phase 1 & 2 通過 | — | ❌ | 當前 Phase 1 進行中 |
| | □ 風控參數優化 | — | ❌ | |
| | □ 生產監控告警升級 | — | ❌ | |

### 3.3 計畫外的重大工作（v1.0 未預見）

| 版本 | 實際工作 | 價值 |
|------|---------|------|
| **v1.2.0** | 多時間框架分析啟用修復 | 解決 v1.2.0 prod 虧損 1% 問題 |
| **v1.3.0** | EODHD 數據源遷移（取代 Alpha Vantage） | AV 已無法滿足 intraday 需求 |
| **v1.3.5** | EODHD Intraday Dual-Timeframe | 1D Trend + 4H Entry，大幅提升入場精度 |
| **v1.3.5** | macro analyst 整合 | 補全 FX 分析的央行政策/經濟指標維度 |
| **v1.3.5** | config 架構清理 | 移除 DataConfig、ScheduleConfig、stale fields |
| **v1.3.5** | 5 項生產 Hotfix | intraday tool binding、ATR duplicate keys、EODHD None bars、Qlib 4h freq |

### 3.4 路線偏移分析

**核心發現**：v1.0 的路線圖是 **功能導向**（Dashboard、多帳號），但實際發展被 **交易效果導向** 拉動。

| 偏移原因 | 說明 |
|---------|------|
| **Alpha Vantage 退場** | v1.0 假設 AV 是穩定數據源，實際上 AV 對 FX intraday 支持不足，被迫遷移至 EODHD |
| **日內信號提前** | 原計畫 v2.0.0 才做 4H/1H，但 v1.2.0 prod 虧損迫使提前在 v1.3.5 實施 |
| **Dashboard 延後** | 交易效果改善的優先級遠高於可視化 Dashboard |
| **多帳號管理延後** | 單帳號尚未穩定盈利，多帳號管理意義不大 |

**教訓**：路線圖應以 **交易 P&L 改善** 為核心驅動力，功能性需求（Dashboard、多帳號）待核心策略穩定後再推進。

### 3.5 修訂後的版本優先級

基於 v1.3.5 的實際經驗，重新排序：

```
v1.3.5 (當前) ─── EODHD Dual-Timeframe + macro analyst ✅
  │
  ├── v1.3.6 ────── 快速配置調優（情境優化 Phase A）
  │                 □ 波動監控 15s 輪詢 + 5 分鐘冷卻
  │                 □ LLM Worker 10s 輪詢
  │                 □ 反思迴圈啟用驗證
  │
  ├── v1.3.7 ────── 戰術執行模組 + 決策快取 ⬅️ NEW
  │                 □ TacticalValidator（Hard/Soft 雙層門檻）
  │                 □ StrategicDecisionCache（4H TTL）
  │                 □ Intent 去重 + 平倉冷卻
  │                 □ Shadow Mode → 門檻校準 → 正式啟用
  │
  ├── v1.4.0 ────── 學習迴圈閉合 + 新聞觸發器
  │                 □ 歷史盈虧注入 LLM 提示詞（原 v1.1.0 計畫）
  │                 □ TradingAgents reflect_and_remember 啟用
  │                 □ NewsEventTrigger 模組
  │                 □ 緊急平倉增強
  │
  ├── v1.5.0 ────── 記憶系統升級
  │                 □ MemoryJournal → ChromaDB 向量記憶
  │                 □ 決策前自動檢索相似歷史情境
  │                 □ 動態 SL/TP (ATR-based)
  │
  ├── v2.0.0 ────── 即時數據 + 因子進化
  │                 □ EODHD WebSocket 即時報價
  │                 □ MatchTrader 權益 WebSocket（若可用）
  │                 □ RD-Agent 週末因子發現自動化
  │                 □ 相關性檢測
  │
  ├── v2.5.0 ────── Streamlit Dashboard（原 v1.2.0 計畫）
  │                 □ Portfolio Overview + Equity Curve
  │                 □ Trade History + Memory 瀏覽
  │                 □ 實時 Log + Scanner 信號可視化
  │
  └── v3.0.0 ────── 實盤 $50k 帳號
                    □ $5k Phase 1 通過
                    □ 多帳號管理（原 v1.3.0 計畫）
                    □ 風控參數優化
                    □ 生產監控告警升級
```

---

## 5. 運行速度優化 — WebSocket 與延遲改善

### 4.1 現有架構：全輪詢模式

v1.3.5 的所有數據獲取均為 **HTTP REST 輪詢**：

| 數據流 | 方式 | 頻率 | 平均延遲 | 日 API 調用 |
|--------|------|:----:|:-------:|:----------:|
| 權益監控 | MatchTrader REST | 每 60s | 30–60s | ~1,440 |
| 波動監控 | MatchTrader REST | 每 60s | 30–60s | ~5,760（4 pairs） |
| Scanner 數據 | EODHD REST | 每 1–4h | 30–240 min | ~48 |
| LLM 新聞數據 | EODHD REST | 按需（同步） | 1–3s/call | 不定 |

**無任何 WebSocket 或串流連接。**

### 4.2 EODHD WebSocket 技術規格

EODHD 提供 FX WebSocket 串流端點：

| 項目 | 規格 |
|------|------|
| **端點** | `wss://ws.eodhistoricaldata.com/ws/forex?api_token={KEY}` |
| **數據類型** | Tick-by-tick bid/ask 報價（非 OHLCV bar） |
| **延遲** | < 50ms |
| **FX Pairs** | 1100+ |
| **連接限制** | 每連接最多 50 個符號 |
| **API 配額** | WebSocket 連接 **不消耗** REST API 調用配額 |
| **數據來源** | VWAP 聚合自 100+ OTC 來源 |

**訊息格式**：
```json
{
  "s": "EURUSD",      // symbol
  "a": 1.086751,     // ask
  "b": 1.08665,      // bid
  "dc": 0.21,         // daily change %
  "dd": 0.0023,       // daily difference
  "t": 1725198451165  // epoch ms
}
```

**訂閱方式**：
```json
{"action": "subscribe", "symbols": "EURUSD,GBPUSD,USDJPY,AUDUSD"}
```

### 4.3 方案限制

⚠️ **當前方案不支持 WebSocket**：

| 方案 | WebSocket | 月費 |
|------|:---------:|:----:|
| End Of Day（當前） | ❌ | — |
| Forex and Crypto Live（當前） | ❌ | — |
| EOD+Intraday — All World Extended | ✅ | $29.99 |
| All-In-One | ✅ | $69.99 |

**建議**：升級至 EOD+Intraday — All World Extended（$29.99/月），即可獲得 WebSocket 支持。

### 4.4 WebSocket vs 輪詢比較

| 指標 | 輪詢（現況） | WebSocket（目標） | 改善倍數 |
|------|:----------:|:---------------:|:--------:|
| 報價延遲 | 30–60s | < 50ms | **600–1200×** |
| 波動偵測 | 30–60s | < 1s | **60×** |
| API 調用 | ~7,200/天 | 0（不消耗配額） | **∞** |
| 連接數 | 每次新建 | 1 條持久連接 | 更穩定 |

### 4.5 其他即時數據供應商

| 供應商 | 延遲 | FX pairs | 月費 | 現有 API Key |
|--------|:----:|:--------:|:----:|:----------:|
| **EODHD** | < 50ms | 1100+ | $29.99 | ✅ |
| **TraderMade** | < 50ms | 60+ | 需查詢（Streaming API 獨立 key） | ✅（REST） |
| **Polygon.io** | < 20ms | 數百 | $199+ | ❌ |
| **OANDA** | 未知 | 38,000+ | 需 OANDA 帳戶 | ❌ |

**建議**：優先使用 EODHD WebSocket（已有帳戶，成本最低），TraderMade 作為備援。

### 4.6 MatchTrader WebSocket

**研究結果**：MatchTrader **不提供公開 WebSocket API**。

- 官方 Platform API 僅支持 RESTful 操作（下單、查詢倉位、帳戶管理）
- 無 WebSocket 端點用於即時價格或權益串流
- **變通方案**：使用 EODHD WebSocket 獲取即時價格，MatchTrader REST 僅用於交易執行和權益查詢

### 4.7 實施路線圖

#### Phase A — WebSocket 客戶端 PoC（v1.4.0，2–3 天）

```python
# 新模組: src/data/fx_websocket_client.py

class EODHDWebSocketClient:
    """EODHD FX WebSocket 即時報價客戶端。

    使用 WebSocket 持久連接接收 tick-by-tick bid/ask 報價，
    取代 REST API 輪詢，延遲從 60s 降至 < 50ms。

    Usage:
        client = EODHDWebSocketClient(api_token="...", symbols=["EURUSD"])
        client.on_tick(handle_tick)
        await client.run()
    """
    URL = "wss://ws.eodhistoricaldata.com/ws/forex"

    def __init__(self, api_token: str, symbols: list[str]):
        self.api_token = api_token
        self.symbols = symbols
        self._callbacks: list[Callable] = []

    async def run(self) -> None:
        """連接並持續接收報價，斷線自動重連。"""
        ...

    async def _reconnect_loop(self) -> None:
        """指數退避重連：2^attempt 秒（最大 300s）。"""
        ...
```

#### Phase B — 整合到現有架構（v1.5.0，1 週）

| 整合點 | 改動 |
|--------|------|
| **volatility_monitor.py** | 從 WebSocket tick 計算即時波動率，取代 REST 輪詢 |
| **equity_monitor.py** | 結合 WebSocket 報價 + MatchTrader REST 權益，提高刷新頻率 |
| **scheduler.py** | WebSocket 觸發模式：tick 異常 → rescan_event |

#### Phase C — TradingAgents async 化（v2.0.0，2 週）

目前 TradingAgents 的所有 API 呼叫使用 **同步 `requests.get()`**：

| 檔案 | 改動 |
|------|------|
| `eodhd_common.py` | `requests.get()` → `httpx.AsyncClient` |
| `eodhd_news.py` | 同上 |
| `eodhd_intraday_indicator.py` | 同上 |
| 所有 `*_analyst.py` tool 定義 | 支持 async tool execution |

**預期效果**：LLM 分析師數據獲取延遲從 5–15s → 2–5s（並行請求）。

### 4.8 延遲改善路線總結

```
                        現況            Phase A         Phase B         Phase C
                        v1.3.5          v1.4.0          v1.5.0          v2.0.0
                        ──────          ──────          ──────          ──────
波動偵測延遲            30–60s          < 1s            < 50ms          < 50ms
Scanner 觸發            ~81s            ~40s            ~15s            ~10s
LLM 數據獲取            5–15s           5–15s           5–15s           2–5s
端到端（典型）           ~81s            ~40s            ~25s            ~15s
端到端（最佳）           ~25s            ~12s            ~5s             ~3s
```

---

## 6. 統一路線圖

### 6.1 版本時間線

```
2026 Q1 (當前)
  │
  ├─ v1.3.5 ✅ ─── EODHD Dual-Timeframe + macro analyst（Phase 4 測試中）
  │
  ├─ v1.3.6 ────── 快速配置調優                               [1–2 天]
  │                · 波動監控 15s + 冷卻 5 分鐘
  │                · LLM Worker 10s 輪詢
  │                · TradingAgents 反思迴圈驗證
  │
  ├─ v1.3.7 ────── 戰術執行模組 + 決策快取                    [3–5 天] ⬅️ NEW
  │                · TacticalValidator（Hard/Soft 雙層門檻）
  │                · StrategicDecisionCache（同 symbol+dir 4H 不重跑 LLM）
  │                · Intent 去重 + 平倉冷卻
  │                · Shadow Mode 觀測 → 門檻校準 → 正式啟用
  │
2026 Q2
  │
  ├─ v1.4.0 ────── 學習迴圈 + 新聞觸發 + WebSocket PoC        [2–3 週]
  │                · 歷史盈虧注入 LLM 提示詞
  │                · NewsEventTrigger 模組
  │                · EODHD WebSocket 客戶端 PoC
  │                · 緊急平倉增強
  │
  ├─ v1.5.0 ────── 記憶升級 + WebSocket 整合                  [3–4 週]
  │                · MemoryJournal → ChromaDB
  │                · WebSocket 整合到 volatility/equity monitor
  │                · 動態 SL/TP (ATR-based)
  │
2026 Q3
  │
  ├─ v2.0.0 ────── 即時數據 + 因子進化 + async 化              [4–6 週]
  │                · TradingAgents async httpx 遷移
  │                · RD-Agent 週末因子自動化
  │                · 相關性檢測
  │                · 即時新聞串流
  │
2026 Q4
  │
  ├─ v2.5.0 ────── Streamlit Dashboard                        [2–3 週]
  │                · Portfolio + Equity Curve
  │                · Trade History + Memory 瀏覽
  │                · 實時 Log 可視化
  │
  └─ v3.0.0 ────── $50k 實盤帳號                              [持續]
                   · $5k Phase 1 通過
                   · 多帳號管理
                   · 風控參數優化
```

### 6.2 優先級矩陣

| 改進項目 | 交易 P&L 影響 | 實施難度 | ROI | 建議版本 |
|---------|:----------:|:------:|:---:|:-------:|
| 波動監控提速（15s） | 🟡 中 | 🟢 低 | ⭐⭐⭐ | v1.3.6 |
| TradingAgents 反思啟用 | 🔴 高 | 🟢 低 | ⭐⭐⭐ | v1.3.6 |
| **戰術執行模組** | **🔴 高** | **🟡 中** | **⭐⭐⭐** | **v1.3.7** |
| **決策快取 + Intent 去重** | **🔴 高** | **🟢 低** | **⭐⭐⭐** | **v1.3.7** |
| 歷史盈虧注入 LLM | 🔴 高 | 🟡 中 | ⭐⭐⭐ | v1.4.0 |
| NewsEventTrigger | 🔴 高 | 🟡 中 | ⭐⭐⭐ | v1.4.0 |
| WebSocket PoC | 🟡 中 | 🟡 中 | ⭐⭐ | v1.4.0 |
| MemoryJournal 向量化 | 🟡 中 | 🟡 中 | ⭐⭐ | v1.5.0 |
| 動態 SL/TP | 🔴 高 | 🔴 高 | ⭐⭐ | v1.5.0 |
| RD-Agent 因子發現 | 🟡 中 | 🔴 高 | ⭐ | v2.0.0 |
| TradingAgents async | 🟢 低 | 🟡 中 | ⭐ | v2.0.0 |
| Streamlit Dashboard | 🟢 無 | 🟡 中 | ⭐ | v2.5.0 |

### 6.3 里程碑定義

| 里程碑 | 達成條件 | 目標日期 |
|--------|---------|:-------:|
| **M0.5: 戰術層上線** | Shadow Mode 完成 + 門檻校準 + 正式啟用 | v1.3.7 |
| **M1: Phase 1 通過** | $5k 帳戶達到 9% 利潤目標（$450） | 2026 Q2 |
| **M2: 學習迴圈閉合** | LLM 能自動從歷史交易學習 | v1.4.0 |
| **M3: 即時反應** | 端到端延遲 < 30s（典型場景） | v1.5.0 |
| **M4: 自動因子進化** | RD-Agent 週末自動發現新因子 | v2.0.0 |
| **M5: $50k 實盤** | Phase 1 & 2 通過，進入 $50k 帳戶 | v3.0.0 |

---

> **PropFirmPilot v1.3.5** — 從全輪詢架構邁向事件驅動的即時交易系統。戰術執行模組（v1.3.7）將填補策略決策與訂單執行之間的關鍵空白。
