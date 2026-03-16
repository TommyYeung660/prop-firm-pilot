# PropFirmPilot v1.5.0 System Design

> 日期: 2026-03-16  
> 範圍: scanner contract、downstream ingestion gate、research-to-live consistency

## 1. 目標

`v1.5.0` 把 `qlib_market_scanner` 視為 `prop-firm-pilot` 的正式上游研究引擎。這版系統設計要解決三件事：

1. 凍結一版 FX 研究 baseline，避免上下游各自漂移。
2. 把 scanner 輸出明確定義成 versioned contract，而不是只有鬆散的 CSV。
3. 在 live ingestion 前建立 gate，拒收 invalid、stale、degraded bundle。

## 2. 上游研究基線

PropFirmPilot `v1.5.0` 預設接受以下上游設定：

- `profile = fx`
- `scanner_version = v1.5.0`
- `label_version = cost_aware_directional_return_v1`
- `signal schema = fx_signal_v1`
- `metrics schema = fx_metrics_v1`
- `bundle version = fx_bundle_v1`

凍結 universe：

- `EURUSD`
- `GBPUSD`
- `USDJPY`
- `AUDUSD`
- `NZDUSD`
- `USDCAD`
- `USDCHF`

研究 cadence matrix：

- `1d`
- `1h`
- `4h+1h`
- `1d+1h`

## 3. Contract Artifacts

### 3.1 `signals.csv`

`signals.csv` 是 downstream ingest 的主表。必要欄位：

- `datetime`
- `instrument`
- `score`
- `rank`
- `score_gap`
- `drop_distance`
- `topk_spread`
- `confidence`
- `weight`
- `profile`
- `scanner_version`
- `schema_version`
- `cadence`
- `label_version`
- `regime_label`
- `market_date`

### 3.2 `signals.json`

`signals.json` 提供 agent 與外部消費者較穩定的 JSON 表示，metadata 至少要有：

- `scanner_version`
- `schema_version`
- `profile`
- `cadence`
- `label_version`
- `universe`

### 3.3 `metrics.json`

`metrics.json` 在 `v1.5.0` 不再只保留 signal/backtest 摘要。必要 section：

- `signal`
- `confidence`
- `backtest`
- `research`
- `regime`
- `validation`

### 3.4 `manifest.json`

`manifest.json` 是 bundle 索引與 gate 入口。必要欄位：

- `bundle_version`
- `scanner_version`
- `schema_versions`
- `research_run_id`
- `config_fingerprint`
- `generated_at`
- `data_date_range`
- `universe`
- `cadence`
- `label_version`
- `validation.status`

## 4. Downstream Ingestion Flow

1. `ScannerBridge.run_pipeline()` 執行上游 scanner，或在可恢復失敗時回退到既有 `signals.csv`。
2. `ScannerBridge.load_signals_from_file()` 先解析 `manifest.json` 與 `metrics.json`，再讀取 `signals.csv`。
3. 若 contract 驗證失敗，bridge 直接拒收 bundle，不會產生任何 `ScannerSignal`。
4. `Scheduler._scanner_loop()` 只會對通過 gate 的 `ScannerSignal` 建立 `TradeIntent`。
5. `TradeIntent` 持久化 scanner metadata，讓後續 decision、execution、journal 與 post-trade review 能對回 research context。

## 5. Validation Gates

### 5.1 Scanner Output Gate

由上游 `qlib_market_scanner` 驗證：

- `signals.csv` 是否含 v1.5.0 必要 metadata 欄位
- `metrics.json` 是否含 `research/regime/validation`
- `manifest.json` 是否含 `schema_versions/research_run_id`

### 5.2 Pilot Ingestion Gate

由 `ScannerBridge` 驗證：

- `manifest.json` 必須存在且可解析
- `schema_versions.signals_csv` 必須是 `fx_signal_v1`
- `scanner_version` 必須是 `v1.5.0`
- `validation.status` 不可為 `degraded`
- `signals.csv` 每列的 `profile/schema_version/scanner_version/cadence/label_version` 必須與 manifest 對齊

### 5.3 Research-to-Live Consistency Gate

由 `TradeIntent` 與 journal 支撐：

- `scanner_version`
- `scanner_schema_version`
- `scanner_market_date`
- `scanner_label_version`

這些欄位讓 live trade 可以回對到研究 bundle 與 market date，避免回測與實盤語義混在一起。

## 6. Rejection Semantics

PropFirmPilot `v1.5.0` 對 scanner bundle 使用顯式 reason code：

| Reason code | 觸發條件 | 行為 |
| --- | --- | --- |
| `scanner.contract.invalid` | 缺 manifest、schema/version 不支援、required 欄位缺失、row metadata 與 manifest 不一致 | 拒收 bundle，不建立 intents |
| `scanner.bundle.degraded` | `validation.status = degraded` | 拒收 bundle，不建立 intents |
| `scanner.bundle.stale` | bundle 自報 stale，或 `chosen_date` 相對 `target_date` 超出 `max_signal_age_days` | 拒收 bundle，不建立 intents |

`Scheduler` 會把這些拒收結果記成 `SCANNER_BUNDLE_REJECTED` event，讓 ops 與 diagnostics bundle 能追蹤 scanner 入口失敗原因。

## 7. Operational Notes

- `signals.csv` 仍是最終 fallback artifact，但從 `v1.5.0` 起，CSV 不再被視為獨立可信來源，必須與 `manifest.json` 一起驗證。
- 若 scanner subprocess 失敗但已有完整且有效的 bundle，bridge 可回退到既有輸出；若 bundle gate 不過，仍然拒收。
- stale 判定以明確日期比較為主，不使用模糊的「今天/昨天」語義。

## 8. 關聯文件

- `../README.md`
- `../PropFirmPilot_v1.5.0_Profitability_Outlook_Report.md`
- `../Hybrid_EA_LLM_Architecture_zh-TW.md`
- 上游 `qlib_market_scanner/docs/plans/2026-03-16-v1.5.0-fx-alpha-research-design.md`
