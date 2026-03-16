# PropFirmPilot v1.5.0_beta 部署手冊

> **文件日期**: `2026-03-16`
>
> **部署目標**: `v1.5.0_beta`
>
> **文件性質**: cross-repo beta deployment manual
>
> **涵蓋 repo**: `prop-firm-pilot` + `qlib_market_scanner`
>
> **當前 beta 基線**:
> - 以 `main` 上已核准的 `v1.5.0_beta` commit pair 為準
> - 文件內的驗證步驟不硬編碼單一 SHA，避免部署手冊在下一個修補 commit 後立即過期

---

## 1. 文件目的與版本邊界

這份手冊描述的是 `v1.5.0_beta` 的跨 repo 部署流程，不是 `v1.5.0 stable` 的 GA 手冊。

本次 beta 的真正目標是：

- 凍結上游 `qlib_market_scanner` 的 FX release cadence 決策
- 讓下游 `prop-firm-pilot` 在 `main` 正式吸收 scanner bundle contract 與 version metadata
- 為後續 `v1.5.0 stable` 建立可驗證的 beta integration baseline

本次 beta **不是** 以下事情：

- 不是把 FX scanner runtime cadence 從 `1d` 切到 `1h`
- 不是宣稱已完成 full production certification
- 不是用單次 smoke 取代 multi-day market-open validation

## 2. 本次 beta 的跨 repo 釋出定義

`v1.5.0_beta` 由兩個 repo 共同構成：

### 2.1 Upstream: `qlib_market_scanner`

- `version = "1.5.0b0"`
- `display_version = "1.5.0_beta"`
- canonical FX cadence 已凍結為 `1d`
- 正式決策 artifact:
  - `outputs/experiments/v150_fx_matrix_full_v3/cadence_decision.json`
  - `outputs/experiments/v150_fx_matrix_full_v3/cadence_scorecard.csv`

### 2.2 Downstream: `prop-firm-pilot`

- `version = "1.5.0b0"`
- `display_version = "1.5.0_beta"`
- scanner ingestion gate 接受 `v1.5.0` 與 `v1.5.0_beta`
- FX scanner runtime baseline 維持 `config/e8_one_5k_challenge.yaml` 的 `scanner_timeframe: "1d"`

### 2.3 Cross-repo contract

下游必須能接受並驗證上游輸出的 versioned scanner bundle。至少包含：

- `signals.csv`
- `manifest.json`
- `metrics.json`

runtime ingestion 使用的 bundle family 必須固定為：

- `outputs/signals/signals.csv`
- `outputs/manifest.json`
- `outputs/metrics/metrics.json`

不要讓 runtime `outputs/*` 回退去讀 legacy `data/shared_export/*` sidecars。兩者可共存，但不能混讀。

下游 contract 關鍵條件：

- `schema_version = "fx_signal_v1"`
- `scanner_version in {"v1.5.0", "v1.5.0_beta"}`
- required signal columns 完整
- validation status 為 passing 狀態

## 3. 部署前置條件

### 3.1 工具與環境

- Python `3.10`
- `uv`
- 本機 sibling repo 佈局可用

建議路徑：

- `C:\Users\tommy.yeung\CursorProjects\qlib_market_scanner`
- `C:\Users\tommy.yeung\CursorProjects\prop-firm\prop-firm-pilot`
- `C:\Users\tommy.yeung\CursorProjects\TradingAgents`

### 3.2 環境變數

`prop-firm-pilot` 至少需要：

- `MATCHTRADER_API_URL`
- `MATCHTRADER_USERNAME`
- `MATCHTRADER_PASSWORD`
- `ITICK_API_KEY`
- `TRADERMADE_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `DROPBOX_APP_KEY`
- `DROPBOX_APP_SECRET`
- `DROPBOX_REFRESH_TOKEN`

`qlib_market_scanner` 若要做 Dropbox export 或外部資料驗證，亦需對應 credentials。

### 3.3 工作樹要求

部署前應確認：

- 目標 branch 是 `main`
- 不要從混有未審查 runtime 修改的 dirty worktree 直接部署
- 若存在本地文件或研究中產物，必須先確認它們不會改變 runtime 行為

## 4. Release 輸入物與應凍結 artifact

部署 `v1.5.0_beta` 前，至少要確認以下輸入物存在且口徑一致：

### 4.1 Upstream 研究與決策文件

- `../qlib_market_scanner/docs/reports/2026-03-16-v1.5.0-fx-cadence-selection-results.md`
- `../qlib_market_scanner/docs/reports/2026-03-16-v1.5.0-cross-repo-change-note.md`

### 4.2 Downstream 對齊文件

- `docs/PropFirmPilot_v1.5.0_Cross_Repo_Change_Note.md`
- `docs/PropFirmPilot_v1.5.0_Profitability_Outlook_Report.md`
- `docs/PropFirmPilot_changelog.md`
- `docs/PropFirmPilot_v1.4.0_road_map.md`

### 4.3 凍結 artifact

- `../qlib_market_scanner/outputs/experiments/v150_fx_matrix_full_v3/cadence_decision.json`
- `../qlib_market_scanner/outputs/experiments/v150_fx_matrix_full_v3/cadence_scorecard.csv`

這些 artifact 的用途是支撐 release governance。它們不是要求 runtime 每次啟動都重跑 research matrix。

若目前 checkout 不包含這兩個檔案，必須先從正式 artifact source 還原，或用相同研究設定重建到相同 output root，再繼續 beta 部署。

## 5. 建議部署順序

請依這個順序執行：

1. 先部署 `qlib_market_scanner`
2. 再部署 `prop-firm-pilot`
3. 最後做 cross-repo smoke checks
4. 開啟 beta 驗證窗口，保留第一批 market-open 證據

不要反過來先升 `prop-firm-pilot`，再讓它去碰一個未確認版本的 scanner bundle。

## 6. `qlib_market_scanner` 部署步驟

### 6.1 更新到 beta 基線

在 `qlib_market_scanner` repo 執行：

```powershell
git switch main
git pull --ff-only
git rev-parse --short HEAD
```

期望：

- `HEAD` 為本次核准的 `v1.5.0_beta` upstream commit
- 若不是文件撰寫當下的初始 candidate，也必須是同一 release wave 內經核准的 descendant commit

### 6.2 安裝依賴

```powershell
uv sync
```

若沿用既有虛擬環境，也至少要確認依賴與 `pyproject.toml` 一致。

### 6.3 驗證版本

```powershell
@'
from src.version import get_app_version, get_release_tag
print(get_app_version())
print(get_release_tag())
'@ | uv run python -
```

期望輸出：

- `1.5.0_beta`
- `v1.5.0_beta`

### 6.4 確認研究 artifact 凍結

```powershell
Test-Path outputs/experiments/v150_fx_matrix_full_v3/cadence_decision.json
Test-Path outputs/experiments/v150_fx_matrix_full_v3/cadence_scorecard.csv
```

兩者都必須為 `True`。

若其中任一項為 `False`，這不是可忽略警告，而是 beta release governance blocker。

### 6.5 做 scanner 煙霧驗證

```powershell
uv run python -m src.main --profile fx --interval 1d --topk 3 --benchmark EURUSD
```

部署預期：

- scanner 可在 `1d` cadence 正常產生輸出
- CLI 可直接接受 `--benchmark`
- runtime output 會寫出 `outputs/signals/signals.csv`、`outputs/manifest.json`、`outputs/metrics/metrics.json`
- output validation 會出現 `Validation OK`
- 不要把這一步改成 `1h` 或 composite cadence
- 這一步的目的只是驗證 beta runtime baseline，而不是重新做 cadence promotion

## 7. `prop-firm-pilot` 部署步驟

### 7.1 更新到 beta 基線

在 `prop-firm-pilot` repo 執行：

```powershell
git switch main
git pull --ff-only
git rev-parse --short HEAD
```

期望：

- `HEAD` 為本次核准的 `v1.5.0_beta` downstream commit
- 若不是文件撰寫當下的初始 candidate，也必須是同一 release wave 內經核准的 descendant commit

### 7.2 安裝依賴

```powershell
uv sync --all-extras
```

### 7.3 驗證版本

```powershell
@'
from src.version import get_app_version, get_release_tag
print(get_app_version())
print(get_release_tag())
'@ | uv run python -
```

期望輸出：

- `1.5.0_beta`
- `v1.5.0_beta`

### 7.4 確認 scanner baseline 未漂移

部署前至少確認以下條件：

- `config/e8_one_5k_challenge.yaml` 仍為 `scanner_timeframe: "1d"`
- scanner repo 路徑可由 `scanner.project_path` 正確指向 sibling `../../qlib_market_scanner`

### 7.5 執行安全 smoke

建議先做不開新倉的 smoke：

```powershell
uv run python -m src.main --config config/e8_one_5k_challenge.yaml --monitor-only
```

若部署環境使用 scheduler 模式，則在 monitor-only smoke 後再切換到正式運行模式。

## 8. 部署後 smoke checks

### 8.1 版本 smoke

確認兩邊 runtime 版本都顯示：

- `1.5.0_beta`
- `v1.5.0_beta`

### 8.2 Scanner contract smoke

第一次 scanner ingest 後，必須確認：

- 未出現 `SCANNER_BUNDLE_REJECTED`
- 未出現 `contract validation failed: manifest missing schema_versions`
- pilot 沒有因 scanner version drift 而拒收 bundle
- bundle metadata 中的 `schema_version` 為 `fx_signal_v1`
- FX cadence 仍是 `1d`

### 8.3 Runtime metadata smoke

若使用 scheduler / DecisionStore，應確認最新 intent 已能看到：

- `scanner_version`
- `scanner_schema_version`
- `scanner_market_date`
- `scanner_label_version`

可用 SQLite 快速檢查：

```powershell
sqlite3 data/decisions.db "SELECT scanner_version, scanner_schema_version, scanner_market_date, scanner_label_version FROM intents ORDER BY created_at DESC LIMIT 5;"
```

### 8.4 Operator smoke

第一輪 beta smoke 至少要保存：

- version helper 輸出
- 第一個成功 ingest 的 scanner bundle 證據
- 第一段 monitor-only 或 scheduler 啟動日誌

## 9. 常見失敗模式與處置

### 9.1 版本不一致

症狀：

- pilot 顯示 `1.5.0_beta`
- scanner 仍是舊版，或 bundle 帶入不被接受的 `scanner_version`

處置：

- 先停下 deployment
- 對齊上游 scanner 版本後再重試
- 不要讓新版 pilot 長時間配舊 scanner bundle 運行

### 9.2 遺失 `manifest.json` / `metrics.json`

症狀：

- scanner 可能成功輸出 `signals.csv`
- 但 pilot ingest 被 contract gate 拒收

處置：

- 檢查 scanner runtime sidecars 是否真的寫在 `outputs/manifest.json` 與 `outputs/metrics/metrics.json`
- 確認 runtime `outputs/*` 沒有錯誤回退到 legacy `data/shared_export/*`
- 若只有 `signals.csv` 存在而 sidecars 缺失，視為 release blocker，而不是可忽略 warning

### 9.3 Cadence 漂移

症狀：

- operator 想把 runtime 改成 `1h`
- 或把 composite cadence 當作 beta default

處置：

- 視為 release governance drift
- 直接退回 `1d`
- 若要升頻，必須走新的 research decision，不在這次 beta 內處理

### 9.4 研究 artifact 缺失

症狀：

- runtime 可跑
- 但 `cadence_decision.json` / `cadence_scorecard.csv` 不存在或口徑不一致

處置：

- 視為 beta release governance blocker
- 先補齊 artifact freeze，再繼續 release 驗收

## 10. 回滾原則

### 10.1 回滾以「cross-repo pair」為單位

不要只回滾其中一個 repo。這次 beta 是跨 repo 契約釋出，回滾也必須回到上一個已驗證的 pair。

### 10.2 不允許混搭回滾

以下做法都不建議：

- 保留 `prop-firm-pilot v1.5.0_beta`，但回到 pre-beta scanner contract
- 保留 `qlib_market_scanner v1.5.0_beta`，但用舊版 pilot 嘗試解讀新 bundle metadata

### 10.3 回滾觸發條件

任一條件成立即可啟動回滾評估：

- scanner bundle 持續被 pilot 拒收
- runtime baseline 發生未經核准的 cadence drift
- targeted verification 無法重現通過
- beta 首輪 smoke 產生 release-critical regression

## 11. Beta 驗證窗口與產出要求

`v1.5.0_beta` 合入 `main` 後，至少應保留以下驗證產出：

- 兩個 repo 的 `HEAD` commit 記錄
- 兩個 repo 的 version helper 輸出
- 上游 frozen artifact 存在證據
- 下游第一次成功 ingest 的 bundle 證據
- 第一段 monitor-only 或 scheduler smoke 日誌
- 首輪 market-open validation 的摘要結論

正確的 beta 口徑應該是：

- `v1.5.0_beta` 已完成 contract freeze 與 mainline integration
- `v1.5.0 stable` 仍需要額外的 market-open validation 與 acceptance closure
