# Operations Runbook : prop-firm-pilot Scheduler

這份手冊供操作員在生產環境運行 prop-firm-pilot Hybrid EA+LLM 調度器時使用。

### 1. Prerequisites & Environment Setup
系統要求 Python 3.10 並使用 uv 套件管理器進行依賴管理。

必須在 `.env` 檔案中設置以下變數：
* `MATCHTRADER_API_URL` : MatchTrader REST API 基礎網址
* `MATCHTRADER_USERNAME` : 經紀商帳號電子郵件
* `MATCHTRADER_PASSWORD` : 經紀商帳號密碼
* `ITICK_API_KEY` : iTick FX 數據供應商金鑰
* `TRADERMADE_API_KEY` : TraderMade FX 數據供應商金鑰
* `TELEGRAM_BOT_TOKEN` : Telegram 機器人 API 權杖，用於發送警報
* `TELEGRAM_CHAT_ID` : Telegram 聊天或群組 ID，用於接收警報
* `LLM_API_KEY` : TradingAgents LLM 的 API 金鑰
* `LLM_BASE_URL` : LLM 終端節點網址
* `DROPBOX_APP_KEY`, `DROPBOX_APP_SECRET`, `DROPBOX_REFRESH_TOKEN` : 用於 Dropbox 同步

配置文件路徑為 `config/default.yaml`，運行時會與 `config/e8_signature_50k.yaml` 合併。

### 2. Starting the Scheduler
使用以下指令啟動 24/7 調度器模式：
`python -m src.main --config config/e8_signature_50k.yaml --scheduler`

啟動時系統執行以下程序：
1. 加載配置與環境變數。
2. 登入 MatchTrader API 建立連線。
3. 執行 `recover_stale_claims()`，將前次當機遺留的 `claimed` 狀態意圖重新回收。
4. 啟動 5 個非同步工作程序：掃描器 (Scanner, 每 4 小時)、LLM 工人 (Poll, 每 30 秒)、執行引擎 (Execution, 每 10 秒)、清理工 (Janitor, 每 10 分鐘)、權益監控 (Equity, 每 60 秒)。

傳統每日循環模式：`python -m src.main --config config/e8_signature_50k.yaml`
僅監控模式（不開新倉）：`python -m src.main --config config/e8_signature_50k.yaml --monitor-only`

### 3. Stopping the Scheduler (Graceful Shutdown)
請發送 `SIGINT` (Ctrl+C) 或 `SIGTERM` 信號。調度器會觸發 `stop()` 流程。
在 Windows 系統，僅支援 `Ctrl+C` 進行優雅關閉。

關閉流程如下：
* `_running` 標記改為 False。
* 所有工人循環會在下一次迭代時退出。
* 權益監控停止，SQLite 存儲關閉。
* 處於 `claimed` 狀態的意圖將在下次啟動時由 `recover_stale_claims()` 自動復原。
* 除非絕對必要，否則不要使用強制結束程序 (kill -9)，以免跳過清理步驟。

### 4. Scheduler Worker Cadences
工作程序頻率可在 YAML 配置中的 `scheduler:` 區段修改：

| 工作程序 | 預設間隔 | 配置鍵值 | 描述 |
| :--- | :--- | :--- | :--- |
| Scanner | 4 小時 (14400s) | scanner_interval_seconds | 執行 qlib_market_scanner 子程序 |
| LLM Worker | 30 秒 | llm_poll_interval_seconds | 認領待處理意圖並調用 TradingAgents |
| Execution Engine | 10 秒 | execution_poll_interval_seconds | 透過 MatchTrader 執行就緒意圖 |
| Janitor | 10 分鐘 (600s) | janitor_interval_seconds | 回收過期認領並清理舊意圖 |
| Equity Monitor | 60 秒 | equity_poll_interval_seconds | 抓取帳戶權益以進行回撤警報 |
| LLM Worker Count | 1 | llm_worker_count | 並行執行的 LLM 工人數量 |

### 5. Decision State Machine
意圖狀態流轉如下：
```
pending → claimed → ready_for_exec → executing → opened → closed
                                                ↘ rejected
                                                ↘ failed
         claimed → timed_out (透過 Janitor)
         claimed → cancelled (LLM 錯誤或 HOLD 決定)
         pending → cancelled
```
* 認領存活時間 (Claim TTL)：30 分鐘 (配置鍵：`decision_store.claim_ttl_minutes`)。
* 意圖保留天數：7 天 (配置鍵：`decision_store.intent_retention_days`)。

### 6. Monitoring : Dashboard Queries
以下為 `DecisionStore` 常用監控方法對應的 SQL 查詢：

a) **管道狀態** : `store.get_pipeline_status()`
```sql
SELECT status, COUNT(*) as cnt
FROM intents
WHERE status IN ('pending', 'claimed', 'ready_for_exec', 'executing', 'opened')
GROUP BY status;
```

b) **每日摘要** : `store.get_daily_summary('2026-02-16')`
```sql
SELECT status, COUNT(*) as cnt FROM intents WHERE trade_date = '2026-02-16' GROUP BY status;
```

c) **七日成功率** : `store.get_success_rate(7)`
```sql
SELECT
  COUNT(*) as total,
  SUM(CASE WHEN status IN ('opened', 'closed') THEN 1 ELSE 0 END) as success,
  SUM(CASE WHEN status IN ('rejected', 'failed') THEN 1 ELSE 0 END) as failure
FROM intents
WHERE status IN ('opened', 'closed', 'rejected', 'failed')
  AND created_at >= datetime('now', '-7 days');
```

d) **單一商品統計** : `store.get_symbol_stats(7)`
```sql
SELECT symbol, COUNT(*) as total,
  SUM(CASE WHEN status IN ('opened', 'closed') THEN 1 ELSE 0 END) as opened,
  SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected,
  SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
FROM intents WHERE created_at >= datetime('now', '-7 days')
GROUP BY symbol ORDER BY total DESC;
```

e) **今日 API 調用數** : `store.get_api_call_count()`
```sql
SELECT call_count FROM api_calls WHERE call_date = date('now');
```

### 7. Monitoring : Telegram Alerts
系統會針對以下事件自動發送警報：意圖建立、交易執行、合規拒絕、工人錯誤、啟動恢復。
需要在 `.env` 中正確填寫 `TELEGRAM_BOT_TOKEN` 與 `TELEGRAM_CHAT_ID`。
若變數為空，警報服務會安靜停用。訊息格式使用 HTML。

### 8. Common Failure Scenarios & Recovery

a) **過期認領 (LLM 工人中途崩潰)**
* 徵兆：意圖卡在 `claimed` 狀態。
* 修復：啟動時會自動回收；Janitor 每 10 分鐘也會處理過期項目。
* 手動修復：
```sql
UPDATE intents SET status = 'timed_out' WHERE status = 'claimed' AND expires_at < datetime('now');
```

b) **API 額度耗盡 (每日 2000 次)**
* 徵兆：PropFirmGuard 拒絕交易並顯示 "API budget exceeded"。
* 檢查：`SELECT call_count FROM api_calls WHERE call_date = date('now');`
* 對策：增加 `equity_poll_interval_seconds` 的秒數。60 秒間隔每天僅權益監控就會消耗 1440 次請求。

c) **合規拒絕 (Compliance Rejections)**
* 徵兆：意圖變更為 `rejected`。
* 檢查：`SELECT id, symbol, compliance_snapshot FROM intents WHERE status = 'rejected' ORDER BY created_at DESC LIMIT 10;`
* 原因：每日回撤接近 85% 安全線、達到最大回撤限制或單日獲利上限 ($1,600)。這是安全系統正常運作，請勿強行覆蓋。

d) **MatchTrader 登入失敗**
* 檢查 `.env` 憑證。
* 確認 API 網址可連通。
* 檢查經紀商是否有維護視窗。

e) **掃描器子程序超時**
* 徵兆：日誌顯示錯誤並發送警報。
* 檢查 `../../qlib_market_scanner` 是否存在，且 `uv run` 可正常執行。
* 掃描器錯誤不會停止其他工人運作。

f) **重複意圖預防**
* 意圖使用唯一的 `idempotency_key` (`symbol:trade_date:source`)。
* 若掃描器同一天運行多次，重複意圖會被忽略。

### 9. SQLite Database Maintenance
* **路徑** : `data/decisions.db`
* **模式** : 使用 WAL (Write-Ahead Logging) 支援併發讀取。
* **備份指令** :
`sqlite3 data/decisions.db ".backup data/decisions_backup.db"`
* **磁碟空間回收 (WAL Checkpoint)** :
`sqlite3 data/decisions.db "PRAGMA wal_checkpoint(TRUNCATE);"`
* **完整性檢查** :
`sqlite3 data/decisions.db "PRAGMA integrity_check;"`
* **查看結構** :
`sqlite3 data/decisions.db ".schema"`

Janitor 會自動刪除超過 7 天的終端狀態意圖。

### 10. E8 Markets Compliance Summary

| 規則 | 限制 | 安全邊際 (85%) | 影響 |
| :--- | :--- | :--- | :--- |
| 每日回撤 | 當日初始餘額 5% ($2,500) | $2,125 | 軟限制，當日暫停交易 |
| 最大回撤 | 初始資金 8% ($4,000) | $3,400 | 硬限制，帳號終止 |
| 單日获利上限 | 獲利目標 40% ($1,600) | $1,360 | 單日獲利不可超過此金額 |
| API 請求 | 每日 2000 次 | 預留 50 次備用 | 由 SQLite 持久化計數並限制 |
| 反高頻交易 | 持倉小於 1 分鐘佔比 < 50% | 不適用 | 套用隨機延遲開倉 |

### 11. Log Files
* **路徑** : `logs/prop_firm_pilot.log`
* **輪轉** : 10 MB
* **保留** : 30 天
* **過濾關鍵字** :
    * `Scanner loop:` : 掃描器活動
    * `LLM worker` : LLM 處理
    * `Execution loop:` : 交易執行
    * `Janitor loop:` : 清理活動
    * `Scheduler:` : 生命週期事件
    * `Engine:` : 執行引擎決策
    * `PropFirmGuard:` : 合規檢查

### 12. Quick Reference Commands
```bash
# 啟動 24/7 調度器
python -m src.main --config config/e8_signature_50k.yaml --scheduler

# 啟動單次每日循環
python -m src.main --config config/e8_signature_50k.yaml

# 僅監控模式
python -m src.main --config config/e8_signature_50k.yaml --monitor-only

# 執行測試
uv run pytest -v

# Lint 檢查
uv run ruff check src/ tests/

# 資料庫狀態速查
sqlite3 data/decisions.db "SELECT status, COUNT(*) FROM intents GROUP BY status;"

# 今日 API 使用量
sqlite3 data/decisions.db "SELECT call_count FROM api_calls WHERE call_date = date('now');"

# 近期錯誤查詢
sqlite3 data/decisions.db "SELECT id, symbol, status, execution_error FROM intents WHERE status IN ('failed', 'rejected') ORDER BY created_at DESC LIMIT 10;"
```
