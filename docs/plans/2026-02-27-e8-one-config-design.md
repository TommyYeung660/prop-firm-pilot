# E8 One 5K Config 調整設計

> **日期**: 2026-02-27  
> **範圍**: `config/e8_one_5k_challenge.yaml`  
> **目標**: 依照 E8 One 5K 需求，調整 symbols、風險參數欄位與資料隔離路徑

---

## 目標

- **symbols** 只保留 `EURUSD`、`XAUUSD`，避免 scanner/執行階段混用未配置品種。
- 修正風險欄位名稱：將 `execution.default_risk_per_trade` 改為正確的
  `execution.default_risk_pct`（0.5%）。
- **資料隔離**：為 E8 One 5K 獨立決策 DB、trade journal、memory、optimization state，
  避免與其他帳號混用。

## 非目標

- 不改動 `default.yaml` 或其他帳號配置。
- 不修改程式碼邏輯（僅配置層級調整）。

## 變更內容（預期）

在 `config/e8_one_5k_challenge.yaml` 新增/覆寫：

- `symbols: [EURUSD, XAUUSD]`
- `execution.default_risk_pct: 0.005`
- `decision_store.db_path: data/decisions_e8_one_5k.db`
- `monitor.trade_journal_path: data/trade_journal_e8_one_5k.jsonl`
- `monitor.memory_dir: MEMORY_E8_ONE_5K`
- `optimization.state_path: data/optimization_state_e8_one_5k.json`

## 影響

- 只影響 E8 One 5K 帳號配置，其他帳號不受影響。
- 新的資料路徑會自動建立新檔案與目錄，不影響舊資料。
- 風險計算將採用 0.5%（正確欄位）。

## 驗證方式

- 透過既有 `tests/test_prop_firm_guard_e8_one.py` 追加斷言，確保
  symbols、風險欄位、與路徑覆寫生效。
- 執行單一測試檔確認配置讀取正確。

---
