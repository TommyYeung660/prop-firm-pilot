# PropFirmPilot

`prop-firm-pilot` 是面向 prop firm FX 帳戶的執行與控制平面。它接收上游 `qlib_market_scanner` 的研究輸出，經過合約驗證、策略決策、風控檢查與 broker 執行，形成可審計的交易生命週期。

## v1.5.0 焦點

`v1.5.0` 把 `qlib_market_scanner` 視為正式上游，並凍結一版可供 live ingestion 的 scanner contract。這一版 downstream 文件與程式預設接受下列研究基線：

- `profile`: `fx`
- `scanner_version`: `v1.5.0`
- `signal schema`: `fx_signal_v1`
- `label_version`: `cost_aware_directional_return_v1`
- `FX baseline`: `EURUSD`, `GBPUSD`, `USDJPY`, `AUDUSD`, `NZDUSD`, `USDCAD`, `USDCHF`
- `research cadences`: `1d`, `1h`, `4h+1h`, `1d+1h`

## v1.5.0 Scanner Contract

PropFirmPilot 在 live ingest 前預期 scanner bundle 至少包含以下 artifact：

| Artifact | 用途 | 重要欄位 |
| --- | --- | --- |
| `signals.csv` | 候選信號主表 | `profile`, `scanner_version`, `schema_version`, `cadence`, `label_version`, `regime_label`, `market_date` |
| `signals.json` | 給 agent / 外部工具的 JSON 候選輸出 | `metadata.scanner_version`, `metadata.schema_version`, `metadata.universe` |
| `metrics.json` | 研究與驗證摘要 | `signal`, `confidence`, `backtest`, `research`, `regime`, `validation` |
| `manifest.json` | bundle 索引與契約版本 | `bundle_version`, `schema_versions`, `research_run_id`, `cadence`, `label_version`, `validation.status` |

目前 downstream 接受的 scanner contract 版本：

- `scanner_version = v1.5.0`
- `schema_versions.signals_csv = fx_signal_v1`
- `schema_versions.metrics_json = fx_metrics_v1`
- `bundle_version = fx_bundle_v1`

## Validation Gates

PropFirmPilot 對 scanner bundle 使用三層 gate：

1. Scanner output gate
   上游先驗證 `signals.csv`、`metrics.json`、`manifest.json` 是否具備必要欄位與 section。
2. Pilot ingestion gate
   `ScannerBridge` 在讀取 `signals.csv` 前，會驗 `manifest/schema/scanner_version/validation.status`，並拒收 unsupported、missing、stale、degraded bundle。
3. Research-to-live consistency gate
   `TradeIntent` 會持久化 `scanner_version`、`scanner_schema_version`、`scanner_market_date`、`scanner_label_version`，供 journal、回放與 live-vs-research 對帳使用。

顯式拒收原因碼：

- `scanner.contract.invalid`
- `scanner.bundle.stale`
- `scanner.bundle.degraded`

## 文件入口

- 系統設計與 contract: `docs/architecture/system-design.md`
- 文件索引: `docs/README.md`
- `v1.5.0` 長報告: `docs/PropFirmPilot_v1.5.0_Profitability_Outlook_Report.md`
- 既有中文架構藍圖: `docs/Hybrid_EA_LLM_Architecture_zh-TW.md`
