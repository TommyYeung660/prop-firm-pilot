# 2026-02-27 TradingAgents LLM 設定下放設計

## 背景
`prop-firm-pilot` 目前在自身設定中固定 `deep_think_llm` / `quick_think_llm`，並傳給 TradingAgents。新的需求是 **所有 LLM 設定都由 TradingAgents 內部 `.env` 控制**（含主/備援與模型升級），`prop-firm-pilot` 不再提供或覆寫模型設定。

## 目標
- `prop-firm-pilot` 不再傳入 `deep_think_llm` / `quick_think_llm`。
- 啟動時 **讀取 TradingAgents 專案的 `.env`**，僅覆寫 LLM 相關環境變數。
- LLM 設定統一由 TradingAgents 內部管理（RightCodes / Volcengine / AIHUBMIX 等）。
- 其餘 prop-firm 的 `.env` 內容不受影響。

## 非目標
- 不改 TradingAgents 內部邏輯。
- 不改 prop-firm 的交易流程、合規與執行邏輯。
- 不改資料來源、FX 研究策略、A/B 測試邏輯。

## 設計概述
### 1) 移除 LLM 設定輸入
- 從 `src/config.py` 的 `AgentsConfig` 移除 `deep_think_llm` / `quick_think_llm`。
- 從 `config/default.yaml` 與 `config/default.yaml.example` 移除對應欄位。
- `build_agent_config()` 不再接收或傳入 LLM 模型欄位。

### 2) TradingAgents `.env` 載入策略
- 在 `AgentBridge._ensure_loaded()`：
  - 解析 `agents_path/.env`（TradingAgents 的 `.env`）
  - **只覆寫** LLM 相關環境變數：
    - `RIGHTCODE_*`, `VOLCENGINE_*`, `AIHUBMIX_*`, `LLM_*`
  - 其他 key 不動，避免覆寫 prop-firm 的 Telegram / Broker 設定。
- 若 `.env` 不存在或解析失敗：記錄 warning，但不阻斷啟動（允許系統層注入）。

### 3) 最終資料流
- `prop-firm-pilot` 啟動時先載入自身 `.env`。
- `AgentBridge` 載入 TradingAgents `.env`（僅 LLM keys），再 import `tradingagents.default_config`。
- TradingAgents 依自己 `.env` 決定主/備援與模型升級策略。

## 測試與驗證
- 單元測試：
  - 驗證 `AgentBridge` 在 `agents_path/.env` 存在時，會寫入指定 LLM keys。
  - 驗證不覆寫非 LLM keys。
- 整合測試：
  - `prop-firm-pilot` 啟動 → TradingAgents 正常載入且使用內部 LLM 配置。

## 風險與緩解
- **風險**：TradingAgents `.env` 不存在 → LLM 無法初始化。
  - **緩解**：warn 並允許系統層環境變數注入。
- **風險**：不小心覆寫 prop-firm 其他 `.env`。
  - **緩解**：僅覆寫特定前綴 key。

## 遷移說明
- 將 LLM 設定移至 TradingAgents `.env`，移除 prop-firm 的 LLM 設定欄位。
- 後續模型升級（如 `gpt-5.3` / `glm-5.0`）只需修改 TradingAgents `.env`。
