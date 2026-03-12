# Prod Bundle Dropbox And Run-Log Design

> **日期**: 2026-03-13
> **範圍**: `scripts/pack_prod_logs.py`、新 `unpack_prod_logs.py`、Dropbox 同步、runtime log naming
> **目標**: 讓 production diagnostics bundle 更完整可分析，並把 zip 搬運流程自動化到 Dropbox，同時把 runtime log 切成 per-run 時間線

---

## 現況與問題

目前 `scripts/pack_prod_logs.py` 已能收集：

- runtime log
- trade journal
- decision SQLite DB
- HWM / optimization state
- memory markdown
- eval results
- Telegram export
- LLM / deterministic fallback summaries
- `INDEX.md`

這對「看見發生了什麼」已經有基本價值，但對「讓 LLM 完整分析 production 上某一次運行程序的情況」仍不夠完整。主要缺口是：

1. **缺少 config snapshot**
   - bundle 沒有打包 account YAML、`config/default.yaml`、或 merged config snapshot
   - LLM 難以確認當次 prod 到底用的是哪組 symbols / risk / scheduler / websocket 參數

2. **缺少 bundle manifest**
   - 目前只有 `INDEX.md`
   - 缺少 machine-readable metadata，例如：
     - account name
     - release tag / version
     - config path
     - bundle generated time
     - days cutoff
     - included log files
     - git commit / branch（若可取得）

3. **單一累積 log 檔污染時間線**
   - `logs/prop_firm_pilot.log` 會跨多次 run 持續增長
   - 同一份 log 可能混入前一版或前一輪啟動的事件
   - 這對版本切窗、事故追查、LLM 時序推理都不利

4. **bundle 搬運流程純手工**
   - zip 生成後仍要人工搬進開發場
   - 這增加延遲，也容易帶來版本對錯包

---

## 設計目標

- `pack prod` 完成本地打包後，自動上傳 Dropbox
- 新增 `unpack prod` 腳本，自動下載 Dropbox 最新 zip 並解壓到 repo 根目錄
- bundle 內增加足夠 metadata，讓 LLM 可以對 prod 執行狀態做更完整分析
- runtime log 改為每次啟動一個新檔，保留同一 run 內的 size rotation
- 不破壞現有 packer 行為：本地 zip 仍是第一產物

---

## 非目標

- 不改動交易邏輯
- 不把 Dropbox 同步做成背景 daemon
- 不引入複雜 artifact database 或遠端 index service
- 不將 `.env` 敏感內容直接打包進 bundle

---

## 方案比較

### 方案 A：在現有 `pack_prod_logs.py` 上擴充，另加 `unpack_prod_logs.py`

做法：

- 在現有 packer 補 manifest / config snapshot
- pack 成功後即上傳 Dropbox
- 新增獨立 unpack 腳本下載最新 zip
- runtime log 改成每次啟動新 base file

優點：

- 改動最小
- 可沿用既有測試與 packer 結構
- 對使用者操作流程改變最少

缺點：

- `pack_prod_logs.py` 會再變大一些

### 方案 B：重構為 diagnostics bundle library + thin scripts

優點：

- 架構更乾淨

缺點：

- 本次需求是「盡快可用」
- 過度工程，交付速度慢

### 方案 C：只加 Dropbox 搬運，不補 bundle metadata / per-run logs

優點：

- 工期最短

缺點：

- 搬運雖解決，但 LLM 分析完整性問題仍存在

---

## 選定方案

採用 **方案 A**。

理由：

- 能以最小改動同時解三個痛點：分析完整性、搬運麻煩、log 時間線污染
- 保留現有 `pack_prod_logs.py` 入口，降低使用成本
- 測試面可集中在既有 `tests/test_pack_prod_logs.py` 與新增 `unpack` / logging 測試

---

## Dropbox 路徑設計

遠端固定路徑：

`/prop-firm-pilot/prod_logs/<account_name>/`

其中 `<account_name>` 來自 config，例如：

- `e8_one_5k_challenge`

上傳檔名維持既有 zip naming：

- `prod_logs_<YYYYMMDD>_<version>.zip`

例如：

- `/prop-firm-pilot/prod_logs/e8_one_5k_challenge/prod_logs_20260313_v1.4.5a.zip`

---

## Dropbox 失敗策略

採用使用者指定策略：

- pack 先本地成功產出 zip
- Dropbox upload 失敗時：
  - 保留本地 zip
  - 腳本回傳非零 exit code

這樣使用者可立即手動補上傳，不會失去本地 artifact。

---

## Unpack 腳本設計

新增 `scripts/unpack_prod_logs.py`。

行為：

1. 載入 `.env`
2. 讀 account config，取得 `account_name`
3. 連 Dropbox 列出 `/prop-firm-pilot/prod_logs/<account_name>/`
4. 找出最新 zip
   - 以 Dropbox `server_modified` 為準
5. 下載 zip 到 repo 根目錄
6. 若同名 `prod_logs_*` 目錄已存在，先刪掉
7. 解壓到 repo 根目錄
8. 保留 zip 檔

這樣 unpack 永遠拉最新 diagnostics bundle，且能覆蓋舊同名目錄。

---

## Bundle 完整性設計

### 新增 `bundle_manifest.json`

除了 `INDEX.md`，再新增 machine-readable manifest，至少包含：

- `account_name`
- `config_path`
- `release_tag`
- `app_version`
- `generated_at_utc`
- `days`
- `date_range`
- `bundle_folder`
- `zip_name`
- `git_commit`（若可取得）
- `git_branch`（若可取得）
- `included_logs`
- `included_data_files`
- `included_summary_files`

### 新增 config snapshot

在 bundle 中加入：

- `raw/config/default.yaml`
- `raw/config/<account-config>.yaml`
- `raw/config/merged_config.yaml`

注意：

- 只打包 YAML config，不打包 `.env`
- merged config 只來自 YAML merge，不含 secrets

這樣 LLM 才能把事故與實際 runtime 參數對齊。

---

## Runtime Log Naming 設計

採用使用者指定模式：

- 每次程序啟動都建立新的 log file
- 檔名格式：
  - `logs/prop_firm_pilot_<YYYYMMDD>_<HHMMSS>_<release-tag>.log`

例如：

- `logs/prop_firm_pilot_20260313_091530_v1.4.5a.log`

實作上不改 YAML schema：

- config 仍保留 `logging.file` 作為 base path，例如 `logs/prop_firm_pilot.log`
- `setup_logging()` 啟動時，把它轉換成實際 run-specific log path
- 同一 run 內仍沿用 loguru rotation / retention

### 為何不改成子資料夾模式

- 使用者已選擇單檔命名模式
- 對既有 `logs/` 目錄與 packer glob 相容性較高
- 不需要額外處理每 run 子目錄層級

---

## 對 `pack_prod_logs.py` 的影響

`pack_prod_logs.py` 不再只依賴單一 `config.logging.file`。

新的 log 收集邏輯應：

- 優先掃描 `logs/` 內符合 `prop_firm_pilot_*.log*` 的檔案
- 回退兼容舊式 `prop_firm_pilot.log*`
- 只收 cutoff 內檔案
- 在 manifest 中明確記錄收進 bundle 的 log 檔名

這樣既兼容舊 log，又支援新 run-based log。

---

## 驗證策略

至少覆蓋以下測試面：

- `pack_prod_logs.py`
  - config snapshot 存在
  - bundle manifest 內容正確
  - Dropbox remote path 計算正確
  - upload 失敗時仍保留本地 zip 且回傳失敗
- `unpack_prod_logs.py`
  - 能選出 Dropbox 最新 zip
  - 同名資料夾存在時先刪再解壓
- `setup_logging()`
  - 產生 run-specific log path
  - 檔名包含 timestamp 與 release tag

---

## 預期結果

完成後：

- 使用者執行 `pack prod` 後，不需手動搬 zip
- 使用者執行 `unpack prod` 後，可直接在開發場拿到最新 prod bundle
- 每次 prod run 都有明確時間線，不再共用單一增長 log
- LLM 看到 bundle 時，能同時拿到：
  - log
  - trade journal
  - DB
  - memory
  - Telegram
  - config snapshot
  - manifest metadata

這才足以接近「完整分析 prod 上運行程序情況」的要求

