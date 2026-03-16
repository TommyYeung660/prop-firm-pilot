# PropFirmPilot v1.5.0 Cross-Repo Change Note

> **日期**: `2026-03-16`
>
> **上游 repo**: `qlib_market_scanner`
>
> **下游 repo**: `prop-firm-pilot`
>
> **對應版本**: `v1.5.0`

## 1. 摘要

`qlib_market_scanner` 在 `v1.5.0` 已完成 FX cadence 量化研究與正式決策。結論是:

> `prop-firm-pilot` 在 `v1.5.0` 的正式 FX scanner cadence 維持 `1d`。

這不是一次 runtime 行為翻轉，而是把原本已在使用的 `1d` 預設，提升為有研究產物、輸出契約與 validation gate 支撐的正式版本結論。

## 2. 這次什麼沒有變

- `prop-firm-pilot` 的 FX scanner 預設 cadence 先前已是 `1d`
- `v1.5.0` 不需要改動 `scanner_timeframe` 預設值
- 美股分析 cadence 不受這次 FX 研究結論影響，仍維持 stock-scoped 治理

## 3. 這次真正新增的是什麼

- 上游 `qlib_market_scanner` 已把 `1d`、`1h`、`4h+1h`、`1d+1h` 做完整量化比較
- 正式決策 artifact 已凍結:
  - `../qlib_market_scanner/outputs/experiments/v150_fx_matrix_full_v3/cadence_decision.json`
  - `../qlib_market_scanner/outputs/experiments/v150_fx_matrix_full_v3/cadence_scorecard.csv`
- 上游已明確定義提供給 `prop-firm-pilot` 的輸出契約與驗證 gate，將 cadence 決策從「操作預設」提升為「research-governed release contract」

## 4. 對 prop-firm-pilot v1.5.0 的直接影響

- FX scanner ingestion 契約維持 `1d`
- 不需要因為 `v1.5.0` 去切換成 `1h` 或 hybrid cadence
- 若未來要提升到 `1h`，必須由上游以新一輪研究與 gate 證據重新決策

## 5. 參考文件

- 上游正式結果: `../qlib_market_scanner/docs/reports/2026-03-16-v1.5.0-fx-cadence-selection-results.md`
- 上游 cross-repo note: `../qlib_market_scanner/docs/reports/2026-03-16-v1.5.0-cross-repo-change-note.md`
- 本 repo 主報告: `docs/PropFirmPilot_v1.5.0_Profitability_Outlook_Report.md`
