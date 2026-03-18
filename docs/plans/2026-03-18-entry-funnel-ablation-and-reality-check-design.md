# Entry Funnel Ablation And Reality Check Design

> **Date:** 2026-03-19
>
> **Document Role:** 定義 `scanner / LLM / tactical / no-trade` 四條 entry funnel 的比較框架，避免把任何單一層誤當成已被證明的 FX intraday alpha engine。

---

## 1. Goal

這份設計要回答三個問題：

1. `scanner -> tactical` 是否至少比 `no-trade` 更有保留價值。
2. `scanner -> LLM(confirm/veto) -> tactical` 是否真的比 `scanner -> tactical` 多帶來增量價值。
3. `tactical-only` 是否有足夠證據可以取代 scanner admission，而不是只是在 timing 上看起來比較乾淨。

---

## 2. Pipeline Matrix

| Label | Runtime Mode | Meaning |
| --- | --- | --- |
| `A` | `scanner_tactical` | scanner 提供 candidate 與方向，跳過 LLM，tactical 只負責 timing / safety |
| `B` | `scanner_llm_tactical` | scanner 提供方向，LLM 只允許 confirm / veto / abstain，tactical 只負責 timing / safety |
| `C` | `tactical_only` | tactical 必須自行承擔 admission / direction / timing，屬於受控實驗 |
| `D` | `no_trade` | 不開新單，只保留 admission / observation evidence，作為現實基線 |

---

## 3. Metrics

### 3.1 Economic Summary

- `net_pnl`
- `expectancy_per_opened_trade`
- `profit_factor`
- `max_drawdown`

### 3.2 Funnel Summary

- `scanner_candidates`
- `intents_created`
- `opened_count`
- `intent_creation_rate`
- `opened_trade_rate`
- `intent_to_open_rate`

### 3.3 Churn Summary

- `llm_veto_rate`
- `llm_cancels`
- `tactical_wait_then_expire_rate`
- `no_trade_count`

---

## 4. Decision Rules

1. 若 `A` 沒有穩定優於 `D`，代表 scanner admission 目前沒有足夠證據支持 production 保留，應考慮降級為 shadow / research-only。
2. 若 `B` 沒有穩定優於 `A`，LLM 不應被視為 entry edge 層，較合理定位是 `confirm / veto / no-trade`。
3. 若 `C` 沒有優於 `A`，就不能把 `tactical-only` 當成更好的簡化方案。
4. 若 `A/B/C` 都沒有優於 `D`，整條 entry funnel 目前沒有足夠證據支持 live trading expansion。

---

## 5. Runtime Positioning

### 5.1 `v1.5.0 stable` Boundary

這份 ablation / reality check 不是 `v1.5.0 stable` 的 minimum gate。

`v1.5.0 stable` 仍應優先收斂：

- contract closure
- tactical correctness
- entry / exit reliability
- exposure / memory / validation acceptance

### 5.2 `v1.5.x` Validation Role

這份工作比較適合被視為：

- `v1.5.x` validation accumulation
- bounded productization decision support
- alpha reality check before deeper intraday expansion

### 5.3 Tactical-Only Guardrail

`tactical-only` 在目前設計中不是新預設架構，只是受控實驗。

原因很直接：

- tactical 目前是 timing / safety layer
- 它沒有被證明可獨立生成方向 edge
- 若沒有 A/B 對照，直接把它升格成 default 很容易誤判

---

## 6. Deliverables

最小交付應包括：

1. bounded runtime mode 設計
2. 可比較的 funnel / churn / outcome snapshot
3. deterministic diagnostics entry point
4. 明確 recommendation rule，而不是主觀敘事

---

## 7. Success Criteria

這份設計成功的條件是：

- operator 能直接理解 A/B/C/D 各自代表什麼
- downstream 不會把 LLM 或 tactical 誤當成 free alpha engine
- 文件已明確標示這是 `v1.5.x` validation 工作，而不是 `v1.5.0 stable` gate
