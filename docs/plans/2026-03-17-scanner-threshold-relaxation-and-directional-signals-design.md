# Scanner Threshold Relaxation And Directional Signals Design

**Goal:** 先以低風險方式放寬 live scanner gating，驗證 `TradingAgents + tactical entry` 是否能承接更多 scanner 召回；之後再把 scanner 改成顯式輸出 long/short candidates，修正目前「單邊 ranking 卻由下游猜方向」的語義落差。

**Context**

- `qlib_market_scanner` 目前使用 binary classification label，語義是「forward return > 0」，因此現行 `score` 本質上偏向多頭 ranking，而不是多空對稱分數。
- `prop-firm-pilot` 目前會把 scanner 輸出的 symbol candidates 建成 intent，再交由 `TradingAgents` 產生 `BUY/SELL/HOLD`。實務上已出現高分 scanner candidate 被 LLM 判成 `SELL` 的情況，代表 scanner 與下游方向語義並未對齊。
- 目前 live gating 實際上讀的是 optimizer 產出的 `optimization_state`，而不是固定 config。若只手改 state file，scheduler 在啟動與日更時會重新覆蓋，無法形成穩定策略。
- 目前 `signals.json` 的 metadata `topk` 與 `universe` 代表的是「該次輸出結果」，而不是「策略配置」。例如當天只輸出兩筆 candidates，就會看到 `topk: 2` 與僅兩個 symbols 的 `universe`，容易誤導下游與 operator。

**Approach Options**

1. 擴成 14 個鏡像貨幣對，沿用現有單邊 ranking
   - 優點：scanner 主體改動較小，看起來能間接表達 short bias。
   - 缺點：會複製同一 FX 曝險，污染 Top-K、cooldown、same-direction guard、風控與績效統計；execution/broker symbol 支援也不保證乾淨。

2. 先放寬 gating，再把 scanner 改成顯式 long/short 輸出
   - 優點：可先驗證增加召回是否真的帶來更多有效交易，再逐步修正方向語義；風險可控，觀測結果容易歸因。
   - 缺點：需要分兩階段跨 repo 改動，schema 需要做版本升級與相容設計。

3. 維持現有 scanner，只限制 LLM 不可逆向 scanner
   - 優點：pilot 改動較少。
   - 缺點：scanner 仍沒有真正的 short candidate 來源，只是禁止下游逆向，結構問題仍存在。

**Chosen Design**

採用方案 2，拆成 `Phase 1` 與 `Phase 2`。

1. `Phase 1`: runtime threshold override
   - 在 `prop-firm-pilot` 增加明確的 manual gating override config。
   - 當 override 開啟時，LLM pre-filter 與 post-filter 一律優先使用 override，不吃 optimizer 動態 state。
   - 初始策略固定為:
     - `min_confidence = medium`
     - `min_blended_confidence = 0.55`
   - 其他流程完全不變：
     - scanner candidate generation 不變
     - `TradingAgents` 不變
     - tactical entry / execution / compliance 不變
   - 所有 scanner-gating 決策與 journal/metrics 額外記錄：
     - effective threshold values
     - threshold source: `override` 或 `dynamic`

2. `Phase 2`: scanner 顯式輸出 direction-aware candidates
   - `qlib_market_scanner` 保持 7 個實際可交易 FX pairs，不擴成 14 個鏡像 symbols。
   - scanner 改成同時輸出：
     - `long_candidates`
     - `short_candidates`
   - 每筆 candidate 必須帶 `side`，方向成為 scanner 正式契約的一部分。
   - `prop-firm-pilot` 增加 `scanner_side` ingestion 與 persistence，scheduler 以 `(symbol, side)` 作為 candidate / idempotency / same-direction 判斷粒度。
   - `TradingAgents` 的角色改為「確認或否決某個方向的 candidate」，而不是替 scanner 猜方向。

3. Signal schema 演進
   - 新 schema version 升級為 `fx_signal_v2`。
   - CSV 至少新增：
     - `side`
     - 必要時新增 `side_rank`
   - JSON metadata 明確區分配置值與實際輸出值：
     - `configured_universe`
     - `configured_topk_long`
     - `configured_topk_short`
     - `published_long_count`
     - `published_short_count`
   - 若保留過渡期 `candidates` 單一列表，仍需包含 `side`，避免下游再做方向推測。

4. 相容策略
   - `Phase 1` 不改 scanner schema。
   - `Phase 2` 期間，pilot 與 scanner 同時支援 `v1` 與 `v2`。
   - `ScannerBridge`、DB schema、`TradeIntent` 先能接受 `scanner_side`，但對舊資料允許 `NULL` / `legacy` fallback。
   - 等整條 live path 穩定吃 `v2` 後，再把「LLM 不可逆向 scanner side」升為硬規則。

5. Rollout
   - `Phase 1`
     - 啟用 override `medium / 0.55`
     - 觀測 3 到 5 個交易日
     - 核對：
       - `Intent Created` 數
       - LLM pre-filter cancel rate
       - tactical reject / retry rate
       - opened trades 數
       - 依 scanner confidence 分層的 win rate / pnl
   - `Phase 2`
     - 發布 `fx_signal_v2`
     - 先雙軌相容，後收斂為 side-aware hard contract

6. 回滾
   - `Phase 1`
     - 關閉 override，立即回到 optimizer dynamic thresholds。
   - `Phase 2`
     - 關閉 `require_side`
     - pilot 回退到 `v1` parsing
     - scanner 暫時保留 `v1` export 直到 live 驗證穩定

**Non-Goals**

- 不在這次把 FX universe 改成 14 個鏡像貨幣對。
- 不在 `Phase 1` 調整 tactical entry、execution sizing、compliance 規則。
- 不在 `Phase 2` 重新設計 broker symbol mapping 或新增新的交易標的。
- 不在這次直接實作新的 short alpha 模型；`Phase 2` 先定義契約與 downstream 接法。
