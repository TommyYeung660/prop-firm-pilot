# 混合 EA+LLM 架構藍圖

> **版本**: 1.0-zh  
> **日期**: 2026-02-16  
> **狀態**: 草案 — 實施前等待批准  
> **範圍**: 將 `prop-firm-pilot` 從順序執行的每日循環系統，轉變為異步的多層交易管線，實現信號生成、LLM 分析與交易執行的解耦。

---

## 1. 問題陳述

目前的 `PropFirmPilot.run_daily_cycle()` 是**完全同步且順序執行**的：

```
fetch_data → run_scanner (最多 10 分鐘) → LLM decide per symbol (每個 10 分鐘以上) → execute
```

這造成了三個關鍵問題：

1. **總延遲**: 掃描器 (10 分鐘) + 每個商品的 LLM (10 分鐘以上 × 3 個商品) = 在第一筆交易執行前需耗時 **40 分鐘以上**。對於日線 (D1) 信號這尚可接受，但這阻礙了系統支持日內 (4H/1H) 信號的可能性。

2. **緊密耦合**: 如果掃描器失敗，就不會做出任何決策。如果 LLM 失敗，就不會執行任何交易。系統沒有重試機制、沒有後備方案，也沒有部分執行的功能。

3. **缺乏持久化**: 決策僅在 `run_daily_cycle()` 執行期間存在於記憶體中。如果進程在循環中途崩潰，所有工作都會遺失。

### 設計目標

| 目標 | 約束 |
|------|-----------|
| 將掃描器、LLM 和執行解耦為獨立的異步 Worker | 單機、單進程 (asyncio) |
| 將所有決策持久化以應對崩潰 | SQLite (不使用 Redis/Kafka) |
| 支持未來日內信號 (4H/1H) 且無需重寫代碼 | 模組化調度器 |
| 絕不削弱合規檢查 | PropFirmGuard 仍是唯一的關卡 |
| 漸進式遷移 — 現有代碼保持運作 | 在舊模組旁添加新模組 |

---

## 2. 架構概覽

### 三層設計

```
                    ┌─────────────────────────────────┐
                    │             策略層               │
                    │                                  │
                    │  ┌──────────┐    ┌────────────┐  │
                    │  │  掃描器  │    │ LLM Worker │  │
                    │  │ (4h 週期) │    │ (異步池)    │  │
                    │  └────┬─────┘    └──────┬─────┘  │
                    │       │                 │        │
                    └───────┼─────────────────┼────────┘
                            │      寫入       │ 讀取+寫入
                            ▼                 ▼
                    ┌─────────────────────────────────┐
                    │      決策存儲 (Decision Store)   │
                    │                                  │
                    │  intents 表    ←→ decisions 表    │
                    │  WAL 模式, 單寫入者安全           │
                    └───────────────┬──────────────────┘
                                    │ 讀取
                                    ▼
                    ┌─────────────────────────────────┐
                    │             執行層               │
                    │                                  │
                    │  ┌───────────┐  ┌─────────────┐  │
                    │  │   執行    │  │  PropFirm   │  │
                    │  │   引擎    │──│    Guard    │  │
                    │  └─────┬─────┘  └─────────────┘  │
                    │        │                         │
                    │        ▼                         │
                    │  ┌─────────────┐                  │
                    │  │ MatchTrader │                  │
                    │  │   Client    │                  │
                    │  └─────────────┘                  │
                    └─────────────────────────────────┘
                                    │
                    ┌───────────────┴──────────────────┐
                    │             監控層               │
                    │  EquityMonitor + Janitor + Alerts │
                    └─────────────────────────────────┘
```

### 數據流 (正常路徑)

```
1. 掃描器運行 → 為前 K 個商品生成 ScannerSignal[]
2. 為每個信號在 SQLite 中 插入 (INSERT) 一個 TradeIntent (status=pending)
3. LLM Worker 認領一個待處理意圖 (status=claimed, claim_ts=現在)
4. LLM 調用 TradingAgents.propagate() → 獲得 BUY/SELL/HOLD
5. 若為 BUY/SELL: 將意圖更新為 ready_for_exec 並附帶決策詳情
   若為 HOLD: 將意圖更新為 cancelled
6. 執行引擎獲取 ready_for_exec 狀態的意圖
7. 運行 PropFirmGuard.check_all() → 通過 (PASS) 或 拒絕 (REJECT)
8. 若通過: MatchTraderClient.open_position() → 意圖 status=opened
   若拒絕: 意圖 status=rejected 並記錄合規原因
9. Janitor 定期清理過期或舊的意圖
```

---

## 3. 決策狀態機

```
                    ┌─────────┐
                    │ pending  │ ← 掃描器創建意圖
                    └────┬────┘
                         │ LLM Worker 認領
                         ▼
                    ┌─────────┐     claim_ttl 過期
                    │ claimed  │ ──────────────────────► timed_out
                    └────┬────┘
                         │ LLM 返回 BUY/SELL
                         ▼
                ┌────────────────┐
                │ ready_for_exec │
                └───────┬────────┘
                        │ 執行引擎獲取
                        ▼
                    ┌───────────┐
                    │ executing │
                    └─────┬─────┘
                     ╱    │    ╲
                    ╱     │     ╲
                   ▼      ▼      ▼
              ┌────────┐ ┌────────┐ ┌────────┐
              │ opened │ │rejected│ │ failed │
              └───┬────┘ └────────┘ └────────┘
                  │ 持倉關閉 (TP/SL/手動)
                  ▼
              ┌────────┐
              │ closed │
              └────────┘

特殊轉換:
  pending  → cancelled  (用戶取消, 掃描器使信號失效)
  claimed  → cancelled  (LLM 返回 HOLD)
  claimed  → timed_out  (claim_ttl 超時, Janitor 回收)
```

### 狀態值 (Status Values)

| 狀態 | 擁有者 | 描述 |
|--------|-------|-------------|
| `pending` | Scanner | 意圖已創建，等待 LLM 分析 |
| `claimed` | LLM Worker | Worker 正在積極分析此意圖 |
| `ready_for_exec` | LLM Worker | LLM 決定 BUY/SELL，等待執行 |
| `executing` | Execution Engine | 引擎已獲取，正在調用 MatchTrader |
| `opened` | Execution Engine | 持倉成功開啟 |
| `rejected` | Execution Engine | PropFirmGuard 拒絕了該交易 |
| `failed` | Execution Engine | MatchTrader API 錯誤 |
| `cancelled` | 各種 | 明確取消 (HOLD、用戶手動、過時) |
| `timed_out` | Janitor | 認領 TTL 過期，回收為 pending 或 cancelled |
| `closed` | Monitor/Manual | 持倉已關閉 (觸及 TP/SL 或手動) |

---

## 4. 數據模型

### 4.1 TradeIntent (新文件: `src/decision/schemas.py`)

```python
class TradeIntent(BaseModel):
    """由掃描器發現的交易機會，等待 LLM 評估。"""

    id: str = Field(default_factory=lambda: uuid4().hex[:16])
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trade_date: str = Field(description="交易日期 YYYY-MM-DD")
    symbol: str = Field(description="外匯貨幣對，例如 EURUSD")

    # 掃描器輸出
    scanner_score: float = 0.0
    scanner_confidence: str = "medium"
    scanner_score_gap: float = 0.0
    scanner_drop_distance: float = 0.0
    scanner_topk_spread: float = 0.0

    # LLM 決策 (認領後填寫)
    suggested_side: Literal["BUY", "SELL", "HOLD"] | None = None
    suggested_sl_pips: float | None = None
    suggested_tp_pips: float | None = None
    agent_risk_report: str = ""
    agent_state_json: str = ""  # 來自 TradingAgents 的 final_state JSON 序列化結果

    # 生命周期
    source: Literal["scanner", "manual"] = "scanner"
    status: Literal[
        "pending", "claimed", "ready_for_exec", "executing",
        "opened", "rejected", "failed", "cancelled", "timed_out", "closed"
    ] = "pending"
    claim_worker_id: str | None = None
    claim_ts: datetime | None = None
    claim_ttl_minutes: int = 30
    expires_at: datetime | None = None
    idempotency_key: str = Field(default_factory=lambda: uuid4().hex[:12])

    # 執行結果 (執行後填寫)
    position_id: str | None = None
    executed_at: datetime | None = None
    execution_error: str | None = None
    compliance_snapshot: str = ""  # 合規檢查結果的 JSON
```

### 4.2 DecisionRecord (新文件: `src/decision/schemas.py`)

```python
class DecisionRecord(BaseModel):
    """不可變的審計記錄，連結意圖與執行結果。"""

    intent_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    claimed_at: datetime | None = None
    decided_at: datetime | None = None
    executed_at: datetime | None = None
    closed_at: datetime | None = None

    status: str = ""
    order_id: str | None = None
    position_id: str | None = None
    failure_reason: str = ""

    # 用於交易後分析的快照
    compliance_snapshot: str = ""
    execution_meta: str = ""  # JSON: entry_price, volume, sl, tp 等
```

---

## 5. SQLite 決策存儲

### 5.1 Schema (新文件: `src/decision_store/sqlite_store.py`)

```sql
-- 啟用 WAL 模式以支持寫入時的並發讀取
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS intents (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,     -- ISO 8601
    trade_date      TEXT NOT NULL,
    symbol          TEXT NOT NULL,

    -- 掃描器數據
    scanner_score       REAL DEFAULT 0,
    scanner_confidence  TEXT DEFAULT 'medium',
    scanner_score_gap   REAL DEFAULT 0,
    scanner_drop_distance REAL DEFAULT 0,
    scanner_topk_spread REAL DEFAULT 0,

    -- LLM 決策
    suggested_side      TEXT,          -- BUY/SELL/HOLD 或 NULL
    suggested_sl_pips   REAL,
    suggested_tp_pips   REAL,
    agent_risk_report   TEXT DEFAULT '',
    agent_state_json    TEXT DEFAULT '',

    -- 生命周期
    source              TEXT DEFAULT 'scanner',
    status              TEXT DEFAULT 'pending',
    claim_worker_id     TEXT,
    claim_ts            TEXT,
    claim_ttl_minutes   INTEGER DEFAULT 30,
    expires_at          TEXT,
    idempotency_key     TEXT UNIQUE,

    -- 執行
    position_id         TEXT,
    executed_at         TEXT,
    execution_error     TEXT,
    compliance_snapshot TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_intents_status ON intents(status);
CREATE INDEX IF NOT EXISTS idx_intents_trade_date ON intents(trade_date);
CREATE INDEX IF NOT EXISTS idx_intents_symbol_date ON intents(symbol, trade_date);

CREATE TABLE IF NOT EXISTS decisions (
    intent_id       TEXT PRIMARY KEY REFERENCES intents(id),
    created_at      TEXT NOT NULL,
    claimed_at      TEXT,
    decided_at      TEXT,
    executed_at     TEXT,
    closed_at       TEXT,

    status          TEXT DEFAULT '',
    order_id        TEXT,
    position_id     TEXT,
    failure_reason  TEXT DEFAULT '',

    compliance_snapshot TEXT DEFAULT '',
    execution_meta     TEXT DEFAULT ''
);
```

### 5.2 DecisionStore API

```python
class DecisionStore:
    """由 SQLite 支持的交易意圖與決策存儲。

    在單寫入者、多讀取者模式 (WAL 模式) 下是線程安全的。
    所有方法均為同步方法 — 由 async 代碼經由 asyncio.to_thread() 調用。
    """

    def __init__(self, db_path: str = "data/decisions.db") -> None: ...

    # ── 意圖生命周期 ────────────────────────────────────────────────
    def insert_intent(self, intent: TradeIntent) -> None: ...
    def claim_next_pending(self, worker_id: str) -> TradeIntent | None: ...
    def update_intent_decision(
        self, intent_id: str, side: str, sl_pips: float, tp_pips: float,
        risk_report: str, state_json: str,
    ) -> None: ...
    def mark_ready_for_exec(self, intent_id: str) -> None: ...
    def mark_executing(self, intent_id: str) -> None: ...
    def mark_opened(self, intent_id: str, position_id: str) -> None: ...
    def mark_rejected(self, intent_id: str, reason: str) -> None: ...
    def mark_failed(self, intent_id: str, error: str) -> None: ...
    def mark_cancelled(self, intent_id: str, reason: str) -> None: ...
    def mark_closed(self, intent_id: str) -> None: ...

    # ── 查詢 ────────────────────────────────────────────────────────
    def get_pending_intents(self) -> list[TradeIntent]: ...
    def get_ready_intents(self) -> list[TradeIntent]: ...
    def get_active_positions(self) -> list[TradeIntent]: ...
    def get_intent(self, intent_id: str) -> TradeIntent | None: ...
    def get_intents_by_date(self, trade_date: str) -> list[TradeIntent]: ...

    # ── 認領管理 ────────────────────────────────────────────────────
    def recycle_expired_claims(self) -> int: ...
    def cleanup_old_intents(self, retention_days: int = 7) -> int: ...

    # ── 冪等性 ──────────────────────────────────────────────────────
    def intent_exists(self, symbol: str, trade_date: str, source: str) -> bool: ...
```

---

## 6. 調度器設計

### 6.1 多週期編排器 (新文件: `src/scheduler/scheduler.py`)

調度器取代 `PropFirmPilot.run_daily_cycle()` 成為頂層入口點。它管理多個以不同頻率運行的異步 Worker。

```python
class Scheduler:
    """管理掃描器、LLM Worker 和執行引擎的異步編排器。

    用法:
        scheduler = Scheduler(config, decision_store)
        await scheduler.start()  # 持續運行直至中斷
    """

    async def start(self) -> None:
        """將所有 Worker 作為並發的 asyncio 任務啟動。"""
        await asyncio.gather(
            self._scanner_loop(),
            self._llm_worker_loop(worker_id="llm-0"),
            self._execution_loop(),
            self._janitor_loop(),
            self._equity_monitor_loop(),
        )
```

### 6.2 Worker 頻率

| Worker | 週期 | 描述 |
|--------|-------|-------------|
| **Scanner Loop** | 每 4 小時 (可配置) | 運行 `ScannerBridge.run_pipeline()`，為每個前 K 個信號插入 `TradeIntent` |
| **LLM Worker(s)** | 持續進行 (每 30 秒輪詢) | 認領 pending 意圖，調用 `AgentBridge.decide()`，並以結果更新意圖 |
| **Execution Engine** | 持續進行 (每 10 秒輪詢) | 獲取 `ready_for_exec` 意圖，進行合規檢查，並經由 MatchTrader 執行 |
| **Janitor** | 每 10 分鐘 | 回收過期的認領，清理舊意圖 (>7 天) |
| **Equity Monitor** | 每 60 秒 | 輪詢帳戶權益，觸發回撤告警 |

### 6.3 Scanner Loop

```python
async def _scanner_loop(self) -> None:
    while True:
        try:
            signals = await asyncio.to_thread(
                self._scanner.run_pipeline,
                date=today_str(),
                tickers=self._config.symbols,
            )
            for signal in signals[:self._config.scanner.topk]:
                # 冪等性: 若該商品在該日期已存在意圖，則跳過
                if self._store.intent_exists(signal.instrument, today_str(), "scanner"):
                    continue

                intent = TradeIntent(
                    trade_date=today_str(),
                    symbol=signal.instrument,
                    scanner_score=signal.score,
                    scanner_confidence=signal.confidence,
                    scanner_score_gap=signal.score_gap,
                    scanner_drop_distance=signal.drop_distance,
                    scanner_topk_spread=signal.topk_spread,
                    source="scanner",
                    expires_at=now_utc() + timedelta(hours=4),
                )
                self._store.insert_intent(intent)
                logger.info("Scanner: 已為 {} 創建意圖", signal.instrument)
        except Exception as e:
            logger.error("掃描器循環錯誤: {}", e)

        await asyncio.sleep(self._config.scheduler.scanner_interval_seconds)
```

### 6.4 LLM Worker Loop

```python
async def _llm_worker_loop(self, worker_id: str) -> None:
    while True:
        intent = await asyncio.to_thread(
            self._store.claim_next_pending, worker_id
        )
        if intent is None:
            await asyncio.sleep(30)  # 無工作，等待
            continue

        try:
            qlib_data = {
                "score": intent.scanner_score,
                "signal_strength": intent.scanner_confidence,
                "confidence": intent.scanner_confidence,
                "score_gap": intent.scanner_score_gap,
                "drop_distance": intent.scanner_drop_distance,
                "topk_spread": intent.scanner_topk_spread,
            }

            decision = await asyncio.to_thread(
                self._agents.decide,
                symbol=intent.symbol,
                trade_date=intent.trade_date,
                qlib_data=qlib_data,
            )

            if decision.is_actionable:
                await asyncio.to_thread(
                    self._store.update_intent_decision,
                    intent.id, decision.decision,
                    sl_pips=50.0, tp_pips=100.0,  # 未來將來自 DecisionFormatter
                    risk_report=decision.risk_report,
                    state_json=json.dumps(decision.final_state, default=str),
                )
                await asyncio.to_thread(self._store.mark_ready_for_exec, intent.id)
            else:
                await asyncio.to_thread(
                    self._store.mark_cancelled, intent.id, "LLM 決定 HOLD"
                )
        except Exception as e:
            logger.error("LLM Worker {}: 處理意圖 {} 時出錯: {}", worker_id, intent.id, e)
            await asyncio.to_thread(
                self._store.mark_failed, intent.id, str(e)
            )
```

### 6.5 Execution Engine Loop

```python
async def _execution_loop(self) -> None:
    while True:
        intents = await asyncio.to_thread(self._store.get_ready_intents)
        for intent in intents:
            await asyncio.to_thread(self._store.mark_executing, intent.id)

            # 建立交易計劃以進行合規檢查
            trade_plan = self._build_trade_plan(intent)
            account_snapshot = await self._get_account_snapshot()

            # 合規關卡
            result = self._guard.check_all(trade_plan, account_snapshot)
            if not result.passed:
                await asyncio.to_thread(
                    self._store.mark_rejected, intent.id, result.reason
                )
                continue

            # 隨機延遲 (防止重複策略檢測)
            delay = self._guard.add_random_delay()
            await asyncio.sleep(delay)

            # 執行
            try:
                order = await self._matchtrader.open_position(
                    symbol=intent.symbol,
                    side=intent.suggested_side,
                    volume=trade_plan.volume,
                )
                if order.success:
                    await asyncio.to_thread(
                        self._store.mark_opened, intent.id, order.position_id
                    )
                else:
                    await asyncio.to_thread(
                        self._store.mark_failed, intent.id, order.message
                    )
            except Exception as e:
                await asyncio.to_thread(
                    self._store.mark_failed, intent.id, str(e)
                )

        await asyncio.sleep(10)
```

---

## 7. 時間預算

### 7.1 每個週期的用時 (4 小時掃描週期)

| 階段 | 持續時間 | 備註 |
|-------|----------|-------|
| 掃描器子進程 | 2-10 分鐘 | 數據下載 + LightGBM 訓練 |
| 意圖插入 | <1 秒 | SQLite 寫入，可忽略不計 |
| 每個商品的 LLM | 5-15 分鐘 | TradingAgents.propagate() + LLM API 調用 |
| 合規檢查 | <100 毫秒 | 純計算，無 I/O |
| 交易執行 | 1-3 秒 | MatchTrader REST API 調用 |
| **3 個商品的總計** | **17-48 分鐘** | 但掃描器與 LLM 是並發運行的 |

### 7.2 滑點分析

| 信號時間框架 | LLM 延遲影響 | 建議 |
|------------------|-------------------|----------------|
| D1 (日線) | **低** — 主要外匯對每日移動約 50-100 pips。10 分鐘延遲約 2-5 pips 移動。 | 對於目前系統是安全的 |
| 4H | **低至中** — 取決於時段 (倫敦開盤波動較大)。 | 使用限價單是可接受的 |
| 1H | **中** — 考慮為已知模式預先計算 LLM 決策。 | 未來：增加緊急程度分類 |
| ≤15M | **高** — 會破壞策略。不要對 1 小時以下的信號使用 LLM。 | 此架構不支持 |

### 7.3 E8 Markets API 預算

| 操作 | 請求數/交易 | 每日預算 (2000) |
|-----------|---------------|---------------------|
| 登錄 + 餘額檢查 | 2 | 固定 |
| 開啟持倉 | 1 | 每筆交易 |
| 設置 SL/TP (修改) | 1 | 每筆交易 |
| 關閉持倉 | 1 | 每筆交易 |
| 權益輪詢 (60 秒) | ~1440 | 24 小時 × 60 |
| **可用於交易** | ~550 | 2000 - 1440 - 10 預留 |
| **每日最高交易數** | ~137 | 550 / 每筆交易 4 次調用 |

**結論**: 在 60 秒權益輪詢的情況下，我們有預算進行每日約 137 筆交易。目前的策略目標是每日 1-3 筆交易 — 完全在限制範圍內。若將權益輪詢減少到每 5 分鐘一次，可用於交易的預算將增加到約 1700 次。

---

## 8. 配置擴展

### 新配置區段 (添加至 `src/config.py`)

```python
class DecisionStoreConfig(BaseModel):
    """SQLite 決策存儲設置。"""
    db_path: str = "data/decisions.db"
    claim_ttl_minutes: int = 30
    intent_retention_days: int = 7

class SchedulerConfig(BaseModel):
    """多週期調度器設置。"""
    scanner_interval_seconds: int = 14400   # 4 小時
    llm_poll_interval_seconds: int = 30
    execution_poll_interval_seconds: int = 10
    janitor_interval_seconds: int = 600     # 10 分鐘
    llm_worker_count: int = 1               # 以 1 個開始，可擴展至 3 個
    equity_poll_interval_seconds: int = 60
```

### 更新後的 AppConfig

```python
class AppConfig(BaseModel):
    # ... 現有欄位 ...
    decision_store: DecisionStoreConfig = DecisionStoreConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
```

---

## 9. 新模組地圖

### 需創建的文件

| 文件 | 用途 | 行數 (估計) |
|------|---------|-------------|
| `src/decision/schemas.py` | TradeIntent, DecisionRecord Pydantic 模型 | ~80 |
| `src/decision_store/__init__.py` | 包初始化 | ~1 |
| `src/decision_store/sqlite_store.py` | 包含所有 SQL 操作的 DecisionStore 類 | ~250 |
| `src/decision_store/janitor.py` | 清理過期認領與舊意圖的 TTL 清道夫 | ~60 |
| `src/scheduler/__init__.py` | 包初始化 | ~1 |
| `src/scheduler/scheduler.py` | 多週期異步編排器 | ~200 |
| `src/execution/engine.py` | 執行引擎 Worker (合規 + 執行) | ~120 |

### 需修改的文件

| 文件 | 更改 | 風險 |
|------|--------|------|
| `src/config.py` | 添加 `DecisionStoreConfig`, `SchedulerConfig` | 低 — 僅增加內容 |
| `src/main.py` | 在現有的 `--daily` 和 `--monitor-only` 旁添加 `--scheduler` 模式 | 低 — 新的 CLI 標誌 |
| `src/decision/agent_bridge.py` | 在 `decide()` 周圍添加 `async decide_async()` 封裝 | 低 — 僅封裝 |
| `src/signal/scanner_bridge.py` | 無需更改 — 調度器直接調用 | 無 |
| `src/compliance/prop_firm_guard.py` | 無需更改 — 執行引擎直接調用 | 無 |

### 無需修改的文件 (安全關鍵，無需更改)

| 文件 | 原因 |
|------|--------|
| `src/compliance/drawdown_monitor.py` | 純計算，由 PropFirmGuard 原樣使用 |
| `src/compliance/best_day_tracker.py` | 純計算，原樣使用 |
| `src/execution/matchtrader_client.py` | 現有 API 已足夠；冪等性在意圖層級處理 |
| `src/execution/position_sizer.py` | 由執行引擎原樣使用 |

---

## 10. 實施階段

### 階段 2A: 基礎建設 (預計 1-2 天)

**目標**: 建立持久化層與數據模型。系統仍使用舊的每日循環，但決策會被記錄。

1. 創建 `src/decision/schemas.py` — TradeIntent, DecisionRecord 模型
2. 創建 `src/decision_store/sqlite_store.py` — 具備所有 CRUD 操作的完整 DecisionStore
3. 在 `src/config.py` 中添加 `DecisionStoreConfig` 和 `SchedulerConfig`
4. 為 DecisionStore 編寫單元測試 (插入、認領、狀態轉換、冪等性)
5. 為 TradeIntent 序列化編寫單元測試

**驗證**: `pytest tests/test_decision_store.py` 通過，SQLite 資料庫以正確的 Schema 創建。

### 階段 2B: 異步管線 (預計 2-3 天)

**目標**: 調度器取代每日循環。所有三個 Worker 都在運行。

1. 創建 `src/scheduler/scheduler.py` — 掃描器循環、LLM Worker 循環、執行循環
2. 創建 `src/execution/engine.py` — 合規檢查 + 執行流程
3. 創建 `src/decision_store/janitor.py` — 過期認領回收、舊意圖清理
4. 在 `src/decision/agent_bridge.py` 中添加 `async decide_async()`
5. 在 `src/main.py` 中添加 `--scheduler` 模式 (保留 `--daily` 以確保向下兼容)
6. 集成測試: 掃描器 → 意圖 → LLM → 執行 → 狀態轉換

**驗證**: `python -m src.main --config config/e8_signature_50k.yaml --scheduler` 啟動所有 Worker，並端到端處理模擬信號。

### 階段 2C: 強化 (預計 2-3 天)

**目標**: 具備監控、告警與優雅停機功能的生產就緒版本。

1. 針對以下情況添加 Telegram 告警：意圖創建、交易執行、合規拒絕、錯誤發生
2. 優雅停機 (SIGINT/SIGTERM → 取消 Worker → 沖刷待處理寫入)
3. 啟動恢復 (檢查崩潰會話中的 `claimed` 意圖 → 回收)
4. 速率限制器細化 (在 SQLite 中追蹤每日 MatchTrader API 調用數)
5. 用於監控的儀表板查詢輔助工具 (今日意圖、成功率等)
6. 運維操作手冊文檔

**驗證**: 在循環中途強制停止進程，重啟，驗證無重複交易 (冪等性)，驗證陳舊認領被回收。

---

## 11. 遷移策略

新的調度器模式與現有的每日循環模式**並存**。無破壞性更改。

```
# 舊方法 (仍可運作，未更改)
python -m src.main --config config/e8_signature_50k.yaml

# 新方法 (調度器模式)
python -m src.main --config config/e8_signature_50k.yaml --scheduler

# 僅監控 (仍可運作，未更改)
python -m src.main --config config/e8_signature_50k.yaml --monitor-only
```

一旦調度器模式在生產環境中通過驗證，舊的 `run_daily_cycle()` 就可以被棄用 (但不會移除 — 它可以作為較簡單的後備方案)。

---

## 12. 風險分析

| 風險 | 嚴重性 | 緩解措施 |
|------|----------|------------|
| 負載下的 SQLite 寫入爭用 | 低 | WAL 模式 + `busy_timeout=5000`。每次僅一個寫入者。我們的寫入率 < 1次/秒。 |
| LLM Worker 崩潰導致意圖永久處於 "claimed" 狀態 | 中 | Janitor 會回收超過 `claim_ttl_minutes` (30 分鐘) 的認領。啟動恢復也會進行檢查。 |
| 重啟時產生重複交易 | 低 | 每個意圖有 `idempotency_key`。插入前進行 `intent_exists()` 檢查。`mark_executing` 是原子操作。 |
| 超過 MatchTrader 速率限制 | 中 | 執行引擎追蹤每日 API 調用。PropFirmGuard.check_api_request_budget() 作為執行前關卡。若有需要，減少權益輪詢頻率。 |
| 掃描器子進程阻塞事件循環 | 低 | 已封裝在 `asyncio.to_thread()` 中。子進程設置了 600 秒超時。 |

---

## 13. 關鍵設計決策

### 為什麼選擇 SQLite，而非 Redis/Kafka?

- 單機部署 — 無需分佈式消息傳遞
- SQLite WAL 模式可處理我們的並發需求 (1 個寫入者，N 個讀取者)
- 寫入率極低 (每秒 < 1 次寫入) — SQLite 每日可處理數百萬次
- 崩潰恢復功能內建於 SQLite (ACID 事務)
- 零運維開銷 — 無需管理額外的服務
- 數據自動持久化，便於交易後分析

### 為什麼不修改 TradingAgents 內部代碼?

- TradingAgents 是具備自身發布週期的獨立研究項目
- `propagate()` 已經接受 `qlib_data` — 接口已足夠
- 我們從外部控制超時 (asyncio timeout 封裝)
- 修改會導致產生難以維護的分支 (fork)

### 為什麼 PropFirmGuard 留在執行層?

- Guard 必須在執行時檢查**最新**的帳戶狀態，而非決策時
- 帳戶餘額在 LLM 決策與執行之間會發生變化 (其他交易、權益波動)
- Guard 是最後的安全關卡 — 必須盡可能靠近執行端
- 若將其提前會造成 TOCTOU (檢查時間與使用時間不一致) 漏洞

### 為什麼掃描間隔為 4 小時?

- D1 信號每日更改一次 — 4 小時可捕捉新的日線柱狀圖，並提供額外 5 次檢查
- 掃描耗時 2-10 分鐘 — 每 4 小時運行一次並非浪費
- 與主要外匯交易時段 (悉尼、東京、倫敦、紐約) 對齊
- 可經由 `SchedulerConfig.scanner_interval_seconds` 配置

---

## 14. 附錄: E8 Markets 約束摘要

| 規則 | 限制 | 安全邊際 (85%) | 影響 |
|------|-------|---------------------|--------|
| 每日回撤 | 日初餘額的 5% ($2,500) | $2,125 | 軟違規 → 當日暫停交易 |
| 最大回撤 | 初始餘額的 8% ($4,000) | $3,400 | 硬違規 → 帳戶終止 |
| 最佳交易日規則 | 獲利目標的 40% ($1,600) | $1,360 | 單日獲利不得超過此金額 |
| API 請求 | 2,000次/日 | 預留 50 次應對緊急情況 | 包含所有請求 (修改 TP/SL) |
| 反高頻交易 (Anti-HFT) | 持倉少於 1 分鐘的交易不得超過 50% | 不適用 | 隨機延遲 + 持倉手數偏移 |
| 重複策略 | 禁止在多個帳戶使用相同 EA | 不適用 | 對持倉手數進行隨機偏移 |

---

## 15. 開放性問題

1. **權益輪詢頻率**: 60 秒 = 每日 1440 次 API 調用。我們是否應將其減少至 5 分鐘 (每日 288 次調用) 以騰出預算？或者若可用則改用 WebSocket？

2. **LLM Worker 數量**: 從 1 個開始，但若 3 個商品需要分析且每個耗時 10 分鐘，順序執行需 30 分鐘。我們是否應從 3 個 Worker 開始 (每個商品一個)？

3. **掃描器輸出緩存**: 若掃描器每 4 小時運行一次，但 D1 柱狀圖每日僅更改一次，掃描器是否應檢測到無變化並跳過重新訓練？

4. **持倉監控**: 目前系統會開啟持倉，但不會主動監控關閉條件 (除了權益監控外)。執行引擎是否也應輪詢未平倉持倉的 TP/SL 狀態，並將意圖狀態更新為 `closed`？

---

*藍圖結束*
