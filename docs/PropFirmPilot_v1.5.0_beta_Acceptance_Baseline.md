# PropFirmPilot v1.5.0_beta 驗收基線

> **文件日期**: `2026-03-16`
>
> **驗收對象**: `v1.5.0_beta`
>
> **驗收範圍**: `prop-firm-pilot` + `qlib_market_scanner`
>
> **文件性質**: cross-repo beta acceptance baseline

---

## 1. 文件目的與適用範圍

這份文件定義 `v1.5.0_beta` 什麼條件下可被視為「beta 可接受基線」，以及哪些條件屬於 release blocker。

它不是 stable release certificate，也不是盈利證明。它只回答：

- 這次 beta 的 cross-repo contract 是否已正確落地
- 這次 beta 的 release governance 是否可被驗證
- 這次 beta 是否具備進入後續 market-open validation window 的資格

## 2. 驗收原則

### 2.1 證據優先

只接受可落地的本地證據：

- git commit
- version helper 輸出
- frozen artifact
- targeted verification
- smoke logs
- DecisionStore / runtime metadata

### 2.2 Cross-repo 一致性優先

`v1.5.0_beta` 必須以 repo pair 驗收，而不是單 repo 驗收。

### 2.3 不擴張 release 主張

以下說法不屬於本文件可批准的範圍：

- 已證明長期盈利
- 已完成 stable GA 驗收
- 已完成 full production certification

## 3. Blocker 定義

下列任一項成立，即視為 `v1.5.0_beta` 驗收失敗或不得放行：

1. `prop-firm-pilot` 與 `qlib_market_scanner` 的版本識別不一致
2. 上游 frozen research artifact 缺失或無法對應 `1d` canonical cadence 決議
3. 下游 scanner ingestion contract 無法接受 `v1.5.0_beta` bundle
4. 下游出現 `SCANNER_BUNDLE_REJECTED` 且無法透過配置或正確部署消除
5. targeted verification 失敗
6. runtime baseline 出現未經核准的 FX cadence 漂移
7. 文件與 changelog / roadmap 口徑互相衝突，導致 operator 無法判斷 beta 真實邊界

## 4. Baseline A: 版本與分支一致性

### 驗收條件

- `prop-firm-pilot` 位於 `main`
- `qlib_market_scanner` 位於 `main`
- 兩個 repo 的 commit pair 屬於同一個已核准的 `v1.5.0_beta` release wave
- 兩邊 `display_version` 都為 `1.5.0_beta`
- 兩邊 release tag helper 都輸出 `v1.5.0_beta`

### 建議證據

```powershell
git rev-parse --short HEAD
```

```powershell
@'
from src.version import get_app_version, get_release_tag
print(get_app_version())
print(get_release_tag())
'@ | uv run python -
```

### 驗收判定

- 全部成立: `PASS`
- 任一版本字串或 commit pair 不一致: `BLOCKER`

## 5. Baseline B: Upstream research artifact freeze

### 驗收條件

- `qlib_market_scanner` 正式結果文件存在
- `cadence_decision.json` 存在
- `cadence_scorecard.csv` 存在
- 上游正式結論明確指出 canonical FX cadence 為 `1d`
- beta 部署沒有把 `1h` 或 hybrid cadence 誤當作 release default

### 必備參考

- `../qlib_market_scanner/docs/reports/2026-03-16-v1.5.0-fx-cadence-selection-results.md`
- `../qlib_market_scanner/docs/reports/2026-03-16-v1.5.0-cross-repo-change-note.md`
- `../qlib_market_scanner/outputs/experiments/v150_fx_matrix_full_v3/cadence_decision.json`
- `../qlib_market_scanner/outputs/experiments/v150_fx_matrix_full_v3/cadence_scorecard.csv`

若工作樹內沒有這兩個 artifact，需先 restore 或 regenerate 到正式路徑，否則不得簽核 beta。

### 驗收判定

- artifact 與文件都存在且口徑一致: `PASS`
- artifact 缺失、或 cadence 結論不清楚: `BLOCKER`

## 6. Baseline C: Downstream scanner contract 相容性

### 驗收條件

- `prop-firm-pilot` ingestion gate 接受 `v1.5.0` 與 `v1.5.0_beta`
- `schema_version = "fx_signal_v1"`
- required signal columns 完整
- `config/e8_one_5k_challenge.yaml` 仍維持 `scanner_timeframe: "1d"`
- runtime bundle 會從 `outputs/*` 讀取自己的 `manifest.json` / `metrics.json`，不回退混讀 legacy `data/shared_export/*`
- 最新 intent / metadata 可追蹤：
  - `scanner_version`
  - `scanner_schema_version`
  - `scanner_market_date`
  - `scanner_label_version`

### 必備參考

- `src/signal/scanner_bridge.py`
- `config/e8_one_5k_challenge.yaml`

### 建議證據

```powershell
sqlite3 data/decisions.db "SELECT scanner_version, scanner_schema_version, scanner_market_date, scanner_label_version FROM intents ORDER BY created_at DESC LIMIT 5;"
```

### 驗收判定

- contract 條件完整且首次 smoke 可成功 ingest: `PASS`
- bundle 被拒收或 metadata 無法落盤: `BLOCKER`

## 7. Baseline D: Targeted verification evidence

### 7.1 `prop-firm-pilot`

目前 beta 基線已知證據：

- `uv run pytest tests/test_scanner_bridge.py tests/test_scheduler.py tests/test_version.py -q` -> `169 passed`
- `uv run ruff check src/signal/scanner_bridge.py tests/test_scanner_bridge.py tests/test_scheduler.py tests/test_version.py` -> `All checks passed!`

### 7.2 `qlib_market_scanner`

本次 beta 至少應保留 upstream targeted unit suite 通過證據。建議重跑：

```powershell
uv run pytest tests/unit/test_duckdb_store.py tests/unit/test_profile_cadence_isolation.py tests/unit/test_fx_alpha_matrix_experiment.py tests/unit/test_qlib_workflow_dataset_freq.py tests/unit/test_qlib_workflow_scorecard.py tests/unit/test_validate_outputs.py tests/unit/test_export_contract.py tests/unit/test_signals_csv_schema.py -q
```

beta 基線口徑：

- `uv run pytest tests/unit -q` -> `68 passed`
- `uv run ruff check src/main.py src/data/shared_export.py src/pipeline/runner.py src/pipeline/qlib_workflow.py tests/unit/test_runner.py tests/unit/test_shared_export.py` -> `All checks passed!`

### 7.3 版本 sanity

兩個 repo 都應能重現：

- `get_app_version() == "1.5.0_beta"`
- `get_release_tag() == "v1.5.0_beta"`

### 驗收判定

- 兩邊 targeted verification 可通過或已有等價證據: `PASS`
- 任一邊 targeted verification 無法重現: `BLOCKER`

## 8. Baseline E: Runtime / operator smoke evidence

### 驗收條件

- `prop-firm-pilot` monitor-only 或 scheduler smoke 可正常啟動
- 第一輪 scanner ingest 未出現 `SCANNER_BUNDLE_REJECTED`
- 第一輪 scanner ingest 未出現 `contract validation failed: manifest missing schema_versions`
- operator 可從日誌或資料表辨識 `1.5.0_beta` 與 scanner metadata
- 沒有未經核准的 `1h` cadence 切換

### 建議 smoke 順序

1. `qlib_market_scanner` 先做 `1d` runtime smoke
2. `prop-firm-pilot` 做 `--monitor-only` smoke
3. 若環境允許，再做第一輪 scheduler / market-open ingest 驗證

scanner smoke 至少應能保留以下證據：

- `uv run python -m src.main --profile fx --interval 1d --topk 3 --benchmark EURUSD`
- log 出現 `Validation OK`
- `outputs/manifest.json` 存在且 `schema_versions` 完整
- `outputs/metrics/metrics.json.validation.status == "passed"`

### 驗收判定

- smoke 正常，且沒有 contract rejection: `PASS`
- smoke 一開始就出現 contract blocker: `BLOCKER`

## 9. Baseline F: 文件與 release 口徑對齊

### 驗收條件

以下文件必須能互相對齊，不得互相矛盾：

- `docs/PropFirmPilot_changelog.md`
- `docs/PropFirmPilot_v1.4.0_road_map.md`
- `docs/PropFirmPilot_v1.5.0_Cross_Repo_Change_Note.md`
- `docs/PropFirmPilot_v1.5.0_Profitability_Outlook_Report.md`
- `docs/PropFirmPilot_v1.5.0_beta_Deployment_Manual.md`
- `docs/PropFirmPilot_v1.5.0_beta_Acceptance_Baseline.md`

最低一致口徑：

- `v1.5.0_beta` 已合入 `main`
- canonical FX cadence 為 `1d`
- 這次 beta 升級的是 contract / validation / research governance
- `v1.5.0 stable` 仍待後續 validation closure

### 驗收判定

- 文件口徑一致: `PASS`
- 任一核心敘事互相衝突: `BLOCKER`

## 10. 驗收輸出物清單

完成 beta 驗收時，至少應保存：

1. 兩個 repo 的 `HEAD` commit 證據
2. 兩個 repo 的 version helper 輸出
3. 上游 frozen artifact 存在證據
4. `prop-firm-pilot` targeted verification 輸出
5. `qlib_market_scanner` targeted verification 輸出
6. 第一段 monitor-only 或 scheduler smoke log
7. 第一筆成功 ingest 的 scanner metadata 證據

## 11. 最終簽核判定格式

建議使用以下格式保存驗收結論：

```text
Release: v1.5.0_beta
Date: 2026-03-16
Outcome: PASS | CONDITIONAL PASS | FAIL
PropFirmPilot HEAD: <sha>
QlibMarketScanner HEAD: <sha>
Version Identity: PASS | FAIL
Artifact Freeze: PASS | FAIL
Scanner Contract: PASS | FAIL
Targeted Verification: PASS | FAIL
Runtime Smoke: PASS | FAIL
Docs Alignment: PASS | FAIL
Open Risks: <short summary>
Next Gate: market-open validation / stable closure
```

若結果不是 `PASS`，必須明確指出 blocker，而不是只寫「待觀察」。
