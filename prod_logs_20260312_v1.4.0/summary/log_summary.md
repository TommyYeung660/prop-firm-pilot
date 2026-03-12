以下是根據日誌生成的**結構化摘要**：

```json
{
  "system": "prop-firm-pilot",
  "time_range": {
    "start": "2026-03-12 11:22:52.223",
    "end": "2026-03-12 13:58:32.504"
  },
  "overall_status": "stable_with_minor_connectivity_warnings",
  "key_events": [
    {
      "time": "2026-03-12 11:22:52.223",
      "type": "scan_skip",
      "severity": "info",
      "component": "scheduler/scanner",
      "details": "Intraday rescan skipped because daily model scores unchanged until candle close."
    },
    {
      "time": "2026-03-12 11:23:23.952",
      "type": "janitor_cleanup",
      "severity": "info",
      "component": "decision_store/janitor",
      "details": "Cleaned 2 old intents (retention=7d)."
    },
    {
      "time": "2026-03-12 11:43:36.977",
      "type": "telegram_command",
      "severity": "info",
      "component": "telegram_bot",
      "details": "Received command /profit."
    },
    {
      "time": "2026-03-12 11:43:23.964",
      "type": "janitor_cleanup",
      "severity": "info",
      "component": "decision_store/janitor",
      "details": "Cleaned 3 old intents (retention=7d)."
    },
    {
      "time": "2026-03-12 12:12:13.586",
      "type": "telegram_command",
      "severity": "info",
      "component": "telegram_bot",
      "details": "Received command /profit."
    },
    {
      "time": "2026-03-12 12:43:47.522",
      "type": "agent_decision_start",
      "severity": "info",
      "component": "decision/agent_bridge",
      "details": "AgentBridge started deciding on AUDUSD for 2026-03-11."
    },
    {
      "time": "2026-03-12 12:57:00.581",
      "type": "agent_decision_result",
      "severity": "info",
      "component": "decision/agent_bridge",
      "details": "AUDUSD decision = SELL."
    },
    {
      "time": "2026-03-12 12:57:00.582",
      "type": "position_reevaluation",
      "severity": "info",
      "component": "scheduler/position_monitor",
      "details": "Re-evaluation confirmed open position W5765145699145536 (AUDUSD) with decision SELL."
    },
    {
      "time": "2026-03-12 12:59:11.065",
      "type": "websocket_failure",
      "severity": "warning",
      "component": "fx_websocket_client",
      "details": "EODHD FX websocket connection failed due to keepalive ping timeout; reconnect initiated."
    },
    {
      "time": "2026-03-12 12:59:23.071",
      "type": "websocket_failure",
      "severity": "warning",
      "component": "fx_websocket_client",
      "details": "EODHD FX websocket connection failed during opening handshake timeout; reconnect initiated."
    },
    {
      "time": "2026-03-12 12:59:49.443",
      "type": "volatility_trigger",
      "severity": "info",
      "component": "volatility_monitor",
      "details": "EURUSD moved -0.86% in 30min window, triggering early scan."
    },
    {
      "time": "2026-03-12 12:59:49.461",
      "type": "candidate_skipped",
      "severity": "warning",
      "component": "scheduler/scanner",
      "details": "EURUSD skipped due to low-confidence cooldown after 2 consecutive cancels."
    },
    {
      "time": "2026-03-12 13:14:50.983",
      "type": "volatility_trigger",
      "severity": "info",
      "component": "volatility_monitor",
      "details": "USDJPY moved 0.68% in 30min window, triggering early scan."
    },
    {
      "time": "2026-03-12 13:14:50.998",
      "type": "candidate_skipped",
      "severity": "warning",
      "component": "scheduler/scanner",
      "details": "EURUSD again skipped due to low-confidence cooldown after 2 consecutive cancels."
    },
    {
      "time": "2026-03-12 13:29:16.184",
      "type": "telegram_command",
      "severity": "info",
      "component": "telegram_bot",
      "details": "Received command /profit."
    },
    {
      "time": "2026-03-12 13:58:32.327",
      "type": "shutdown_start",
      "severity": "info",
      "component": "__main__",
      "details": "Shutdown signal received; graceful shutdown initiated."
    },
    {
      "time": "2026-03-12 13:58:32.504",
      "type": "shutdown_complete",
      "severity": "info",
      "component": "__main__",
      "details": "Shutdown completed successfully."
    }
  ],
  "data_pipeline": {
    "status": "active",
    "provider": "EODHD",
    "symbols": [
      "EURUSD",
      "GBPUSD",
      "USDJPY",
      "AUDUSD"
    ],
    "interval": "1min",
    "fetch_pattern": {
      "behavior": "high-frequency repeated polling",
      "row_counts_observed": {
        "EURUSD": 1847,
        "GBPUSD": 1845,
        "USDJPY": 1850,
        "AUDUSD": 1847
      }
    },
    "assessment": "Data fetch pipeline remained operational throughout most of the session."
  },
  "scanner_and_signal_state": {
    "cached_signals_used": true,
    "intraday_rescan_behavior": "Skipped because daily model scores unchanged until candle close.",
    "active_candidates": [
      "EURUSD"
    ],
    "suppressed_candidates": [
      {
        "symbol": "EURUSD",
        "reason": "low-confidence cooldown",
        "context": "2 consecutive cancels"
      }
    ]
  },
  "trading_decisions": {
    "new_trade_entries_detected": 0,
    "open_position_reevaluations": [
      {
        "position_id": "W5765145699145536",
        "symbol": "AUDUSD",
        "decision": "SELL",
        "result": "confirmed"
      }
    ],
    "agent_decisions": [
      {
        "symbol": "AUDUSD",
        "decision": "SELL",
        "decision_started_at": "2026-03-12 12:43:47.522",
        "decision_completed_at": "2026-03-12 12:57:00.581"
      }
    ]
  },
  "maintenance": {
    "janitor_cycles": [
      {
        "time": "2026-03-12 11:23:23.953",
        "recycled": 0,
        "cleaned": 2
      },
      {
        "time": "2026-03-12 11:33:23.959",
        "recycled": 0,
        "cleaned": 2
      },
      {
        "time": "2026-03-12 11:43:23.966",
        "recycled": 0,
        "cleaned": 3
      }
    ],
    "intent_retention": "7d"
  },
  "user_interaction": {
    "telegram_commands": [
      {
        "time": "2026-03-12 11:43:36.977",
        "command": "/profit"
      },
      {
        "time": "2026-03-12 12:12:13.586",
        "command": "/profit"
      },
      {
        "time": "2026-03-12 13:29:16.184",
        "command": "/profit"
      }
    ]
  },
  "warnings": [
    {
      "time": "2026-03-12 12:59:11.065",
      "component": "fx_websocket_client",
      "message": "Connection failed due to keepalive ping timeout."
    },
    {
      "time": "2026-03-12 12:59:23.071",
      "component": "fx_websocket_client",
      "message": "Connection failed due to opening handshake timeout."
    },
    {
      "time": "2026-03-12 12:59:49.461",
      "component": "scheduler/scanner",
      "message": "EURUSD skipped due to low-confidence cooldown."
    },
    {
      "time": "2026-03-12 13:14:50.998",
      "component": "scheduler/scanner",
      "message": "EURUSD skipped due to low-confidence cooldown."
    },
    {
      "time": "2026-03-12 13:58:32.328",
      "component": "telegram_bot",
      "message": "getUpdates failed: ReadError() during shutdown."
    }
  ],
  "errors_detected": [],
  "shutdown": {
    "graceful": true,
    "workers_cancelled_cleanly": true,
    "details": [
      "Execution loop cancelled during sleep",
      "Janitor loop cancelled during sleep",
      "Equity monitor stopped",
      "Position monitor loop cancelled during sleep",
      "Daily summary loop cancelled during sleep",
      "LLM workers llm-0 and llm-1 cancelled",
      "Volatility monitor loop cancelled during sleep",
      "Telegram bot stopped",
      "Scanner loop cancelled during sleep"
    ]
  },
  "operational_assessment": {
    "availability": "high",
    "trading_activity": "low",
    "decision_activity": "limited_to_AUDUSD_reevaluation",
    "main_risks_observed": [
      "Repeated websocket instability around 12:59",
      "EURUSD candidate repeatedly suppressed by low-confidence cooldown",
      "Heavy repeated full-range 1min bar fetches may indicate inefficient polling"
    ],
    "recommended_followups": [
      "Review EODHD websocket stability and reconnect/backoff behavior.",
      "Investigate whether repeated full historical fetches for the same 4 symbols are excessive and can be optimized with incremental updates.",
      "Review EURUSD low-confidence cooldown logic and root cause of consecutive cancels.",
      "Check whether AUDUSD decision latency (~13 minutes from decide start to result) is acceptable for production."
    ]
  }
}
```

**簡短結論：**
- 系統整體運行穩定，無致命錯誤。
- 主要交易相關事件是 **AUDUSD 持倉重評估後維持 SELL**。
- **EURUSD** 多次因低信心冷卻被跳過，未進一步執行。
- **12:59 左右 EODHD websocket 出現兩次連線異常**，但系統有自動重連。
- 最後系統在 **13:58** 完成**優雅關機**。  

如果你要，我也可以進一步幫你輸出成：
1. **表格版 incident report**  
2. **給工程團隊的 SRE 重點摘要**  
3. **給交易團隊的交易行為摘要**