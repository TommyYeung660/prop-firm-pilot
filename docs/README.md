# 文件索引

本目錄收錄 `prop-firm-pilot` 的報告、架構說明、操作手冊與研究記錄。`v1.5.0` 起，文件入口以 scanner contract 與 cross-repo validation gate 為主軸。

## v1.5.0 核心文件

- `architecture/system-design.md`
  `v1.5.0` 的 scanner contract、artifact schema、ingestion gate 與 rejection semantics。
- `PropFirmPilot_v1.5.0_Profitability_Outlook_Report.md`
  長期盈利可行性與 `16.x` 工作包的總報告。
- `Hybrid_EA_LLM_Architecture_zh-TW.md`
  既有 async scheduler / decision store / execution control plane 架構藍圖。

## 研究與計畫

- `plans/2026-03-15-v1.5.0-profitability-outlook-design.md`
- `plans/2026-03-15-v1.5.0-profitability-outlook.md`
- `research/v1.5.0_profitability_matrix.md`
- `research/v1.5.0_profitability_source_notes.md`

## 上游對照

`v1.5.0` 的 scanner research 與 contract 凍結在上游 `qlib_market_scanner` 完成。對照文件：

- `qlib_market_scanner/docs/plans/2026-03-16-v1.5.0-fx-alpha-research-design.md`
- `qlib_market_scanner/docs/plans/2026-03-16-v1.5.0-fx-alpha-research-implementation-plan.md`

## 閱讀順序

1. 先讀 `architecture/system-design.md`，理解 contract 與 validation gate。
2. 再讀 `PropFirmPilot_v1.5.0_Profitability_Outlook_Report.md`，掌握 `v1.5.0` 的研究結論與工作包。
3. 若要追實作細節，再看 `Hybrid_EA_LLM_Architecture_zh-TW.md` 與 `plans/`。
