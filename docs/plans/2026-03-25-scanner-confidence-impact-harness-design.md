# Scanner Confidence Impact Harness Design

## Context

`qlib_market_scanner` 最新 FX confidence retune 沒有改動 `signals.csv` schema，也沒有新增
`prop-firm-pilot` ingestion 所需欄位；契約面仍維持 `confidence in {high, medium, low}` 與
既有 `fx_signal_v2` 路徑。

但 `prop-firm-pilot` 不是把 `confidence` 當純 metadata，而是直接將其餵進：

- scheduler pre/post LLM threshold gating
- blended confidence scoring
- execution-side bounded capital uplift

這意味著 scanner 的 confidence 分布一旦改變，即使 parser 不壞，也會直接影響
`intent` 通過率、side mix、以及每筆風險配置。

目前已用上游 formal-study artifact 手工驗證到這個風險存在，但這個分析流程還是 ad-hoc
腳本，無法在下游 repo 內重跑、留檔、或讓 operator/研究者重複比較不同 bundle。

## Goal

在 `prop-firm-pilot` 內建立一個可重複執行的 diagnostics harness，用來比較
baseline scanner bundle 與 retuned scanner bundle 對下游 confidence gate / capital
allocation 的靜態影響。

## Decision

採用一個「純函式分析器 + 小型 CLI 包裝」的 diagnostics 模組：

- 新增 `src/diagnostics/analyze_scanner_confidence_impact.py`
- 核心分析函式只處理 dataframe 與簡單 config values
- CLI 負責讀取 `alpha_candidates.csv`、account config、以及輸出格式

這跟 repo 內既有 `analyze_entry_funnel_ablation.py` / `analyze_preview_bundle.py` 風格一致，
也最容易在本地研究、CI、或 incident review 中重跑。

## Input Contract

### Required Inputs

- `baseline` candidates CSV
- `retuned` candidates CSV

這兩份 CSV 都預期來自 `qlib_market_scanner` 的 `alpha_candidates.csv`，至少包含：

- `datetime`
- `instrument`
- `side`
- `alpha_score`
- `alpha_confidence`
- `publish_status`
- `alpha_rank`

### Optional Inputs

- `--config config/<account>.yaml`

當提供 config 時，harness 會讀取：

- `scanner.topk`
- `scanner.topk_short`
- `scheduler.llm_threshold_override`
- `execution.default_risk_pct`
- `execution.max_risk_pct`
- `execution.max_positions`

若沒有提供 config，使用 repo 內預設的 `AppConfig` defaults。

## Analysis Model

### Row Alignment

用 `(datetime, instrument, side)` 對 baseline / retuned 做 inner join。

不嘗試做 fuzzy match，也不嘗試補 missing rows。任何 unmatched row 都應進 summary，
因為 row mismatch 本身也是研究風險。

### Candidate Selection Scope

先只分析 `publish_status == "published"` 且位於下游 account 實際 consumption 範圍內的 rows：

- long: `alpha_rank <= scanner.topk`
- short: `alpha_rank <= scanner.topk_short`

原因是這最接近 `prop-firm-pilot` 會看到的 live scanner candidate 集。

### Downstream Impact Metrics

對 baseline / retuned 各自計算：

- confidence label distribution
- side-level confidence distribution
- prefilter blended score distribution
- pass/fail under current `llm_threshold_override` or dynamic default thresholds
- capital uplift factor distribution

### Delta Metrics

另外輸出 baseline → retuned 的差異：

- count / rate delta
- number of rows whose confidence label changed
- rows newly passing prefilter
- rows newly entering `medium/high` uplift tiers

## Output Contract

### JSON

機器可讀 summary，頂層包含：

- `analysis_context`
- `row_alignment`
- `baseline_summary`
- `retuned_summary`
- `delta_summary`
- `sample_changed_rows`

### Markdown

人類可讀報告，至少包含：

- account / threshold context
- row alignment summary
- confidence distribution before/after
- pass-rate before/after
- uplift distribution before/after
- key interpretation bullets

## Error Handling

- 缺必要欄位：raise `ValueError`
- config 載入失敗：直接 propagate
- 沒有任何 matched rows：仍輸出空但結構完整的 summary，不直接 crash
- `topk_short = 0` 時 short rows 會被自然排除

## Testing Strategy

使用 TDD，新增 `tests/diagnostics/test_analyze_scanner_confidence_impact.py`，覆蓋：

1. 能正確對齊 baseline / retuned rows
2. 能依 config 的 `topk/topk_short` 篩出 live consumption 範圍
3. 能計算 prefilter 通過率與 capital uplift 分布
4. 能輸出空但完整的 summary 給 zero-match case
5. CLI JSON / markdown 都可正常輸出

## Out of Scope

- 不直接修改 scheduler threshold
- 不做 live broker replay
- 不做 tactical / LLM decision 真實模擬
- 不直接讀 scanner `manifest.json` / `metrics.json`

這個 harness 只負責回答一件事：在既有 `prop-firm-pilot` 配置下，
scanner confidence retune 會把下游 confidence gate 與 risk uplift 推到哪裡。
