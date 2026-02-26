# Memory Feedback Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 3.2/3.3/3.4 memory and optimization features (memory logging, daily optimization state, dynamic confidence thresholds, A/B testing, and feedback injection) without touching compliance rules.

**Architecture:** Add an `optimize` package to compute stats and write `optimization_state.json`. Scheduler reads this state for LLM pre/post filtering and A/B routing. MemoryJournal/TradeJournal are extended to log decisions and outcomes for feedback loops.

**Tech Stack:** Python 3.10, Pydantic v2, loguru, pytest, uv.

---

### Task 1: Optimization State Models + IO

**Files:**
- Create: `src/optimize/optimization_state.py`
- Create: `src/optimize/__init__.py`
- Modify: `src/config.py`
- Test: `tests/optimize/test_state_io.py`

**Step 1: Write the failing test**

```python
from pathlib import Path

from src.optimize.optimization_state import load_state, save_state, OptimizationState


def test_load_state_missing_returns_default(tmp_path: Path) -> None:
    state = load_state(tmp_path / "missing.json")
    assert isinstance(state, OptimizationState)
    assert state.version == "1.0"


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = OptimizationState(version="1.0")
    save_state(path, state)
    loaded = load_state(path)
    assert loaded.version == "1.0"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/optimize/test_state_io.py -v`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

```python
"""
Optimization state models and file IO.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal

from loguru import logger
from pydantic import BaseModel, Field


class Thresholds(BaseModel):
    min_confidence: Literal["low", "medium", "high"] = "medium"
    min_blended_confidence: float = 0.55


class ABTestState(BaseModel):
    model_a: str = "volcengine/glm-4.7"
    model_b: str = "gpt-5.2"
    ratio: float = 0.5
    counts: Dict[str, int] = Field(default_factory=dict)
    pnl_by_model: Dict[str, float] = Field(default_factory=dict)


class OptimizationState(BaseModel):
    version: str = "1.0"
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    pnl_lookback_days: int = 7
    winrate_lookback_days: int = 14
    global_thresholds: Thresholds = Field(default_factory=Thresholds)
    symbol_thresholds: Dict[str, Thresholds] = Field(default_factory=dict)
    ab_test: ABTestState = Field(default_factory=ABTestState)
    feedback_pnl: Dict[str, float] = Field(default_factory=dict)
    risk_per_trade_suggestion: float | None = None
    llm_cost_stats: Dict[str, Any] = Field(default_factory=dict)
    factor_contributions: Dict[str, Any] = Field(default_factory=dict)


def load_state(path: str | Path) -> OptimizationState:
    file_path = Path(path)
    if not file_path.exists():
        return OptimizationState()
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return OptimizationState(**data)
    except Exception as e:
        logger.warning("OptimizationState: failed to load {}, using default ({})", file_path, e)
        return OptimizationState()


def save_state(path: str | Path, state: OptimizationState) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(state.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/optimize/test_state_io.py -v`
Expected: PASS

**Step 5: Commit**

Commit via GitHub MCP (`mcp__github__create_or_update_file` for new/modified files).

---

### Task 2: Stats Extraction (DecisionStore + TradeJournal)

**Files:**
- Create: `src/optimize/trade_stats.py`
- Modify: `src/monitor/trade_journal.py`
- Test: `tests/optimize/test_trade_stats.py`

**Step 1: Write the failing test**

```python
from src.optimize.trade_stats import build_pnl_feedback, compute_win_rates


def test_compute_win_rates_empty(store) -> None:
    result = compute_win_rates(store, days=14)
    assert result["global"] == 0.0


def test_build_pnl_feedback_from_store(store_with_closed_trades) -> None:
    feedback = build_pnl_feedback(store_with_closed_trades, None, days=7)
    assert "EURUSD" in feedback
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/optimize/test_trade_stats.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict

from loguru import logger

from src.decision_store.sqlite_store import DecisionStore
from src.monitor.trade_journal import TradeJournal


def compute_win_rates(store: DecisionStore, days: int = 14) -> Dict[str, float]:
    intents = store.get_closed_intents(days=days)
    wins = defaultdict(int)
    totals = defaultdict(int)
    for intent in intents:
        pnl = intent.realized_pnl or 0.0
        totals[intent.symbol] += 1
        if pnl > 0:
            wins[intent.symbol] += 1
    result: Dict[str, float] = {}
    total_global = sum(totals.values())
    win_global = sum(wins.values())
    result["global"] = win_global / total_global if total_global > 0 else 0.0
    for sym, total in totals.items():
        result[sym] = wins[sym] / total if total > 0 else 0.0
    return result


def build_pnl_feedback(
    store: DecisionStore,
    journal: TradeJournal | None,
    days: int = 7,
) -> Dict[str, float]:
    pnl_by_symbol: Dict[str, float] = defaultdict(float)
    intents = store.get_closed_intents(days=days)
    for intent in intents:
        pnl_by_symbol[intent.symbol] += float(intent.realized_pnl or 0.0)
    if journal is not None:
        for entry in journal.get_closed_trades(days=days):
            pnl_by_symbol[entry.get("symbol", "")] += float(entry.get("pnl", 0.0))
    return dict(pnl_by_symbol)
```

Add to `TradeJournal`:

```python
from datetime import datetime, timezone, timedelta

    def get_closed_trades(self, days: int = 7) -> list[dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        results = []
        if not self._path.exists():
            return results
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") != "TRADE" or entry.get("status") != "CLOSED":
                    continue
                ts = entry.get("timestamp")
                if ts and datetime.fromisoformat(ts) >= cutoff:
                    results.append(entry)
        return results
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/optimize/test_trade_stats.py -v`
Expected: PASS

**Step 5: Commit**

Commit via GitHub MCP.

---

### Task 3: Dynamic Threshold Calculator

**Files:**
- Create: `src/optimize/thresholds.py`
- Test: `tests/optimize/test_thresholds.py`

**Step 1: Write the failing test**

```python
from src.optimize.thresholds import compute_thresholds


def test_stepwise_thresholds_low_winrate() -> None:
    result = compute_thresholds(global_win_rate=0.40, symbol_win_rates={})
    assert result["global"].min_confidence == "high"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/optimize/test_thresholds.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
from typing import Dict

from src.optimize.optimization_state import Thresholds


def _stepwise_threshold(win_rate: float) -> Thresholds:
    if win_rate < 0.45:
        return Thresholds(min_confidence="high", min_blended_confidence=0.65)
    if win_rate > 0.55:
        return Thresholds(min_confidence="low", min_blended_confidence=0.45)
    return Thresholds(min_confidence="medium", min_blended_confidence=0.55)


def _adjust_blended(base: float, delta: float) -> float:
    return max(0.30, min(0.80, base - delta))


def compute_thresholds(
    * ,
    global_win_rate: float,
    symbol_win_rates: Dict[str, float],
) -> Dict[str, Thresholds]:
    global_threshold = _stepwise_threshold(global_win_rate)
    result: Dict[str, Thresholds] = {"global": global_threshold}
    for symbol, win_rate in symbol_win_rates.items():
        delta = win_rate - global_win_rate
        adj = 0.05 if delta >= 0.05 else -0.05 if delta <= -0.05 else 0.0
        result[symbol] = Thresholds(
            min_confidence=global_threshold.min_confidence,
            min_blended_confidence=_adjust_blended(global_threshold.min_blended_confidence, adj),
        )
    return result
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/optimize/test_thresholds.py -v`
Expected: PASS

**Step 5: Commit**

Commit via GitHub MCP.

---

### Task 4: A/B Routing + Stats

**Files:**
- Create: `src/optimize/ab_testing.py`
- Test: `tests/optimize/test_ab_routing.py`

**Step 1: Write the failing test**

```python
from src.optimize.ab_testing import choose_model


def test_choose_model_deterministic() -> None:
    a = choose_model("abc123", 0.5, "m1", "m2")
    b = choose_model("abc123", 0.5, "m1", "m2")
    assert a == b
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/optimize/test_ab_routing.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
import hashlib
from typing import Dict

from src.optimize.optimization_state import ABTestState


def choose_model(intent_id: str, ratio: float, model_a: str, model_b: str) -> str:
    h = hashlib.sha256(intent_id.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    return model_a if bucket < ratio else model_b


def update_ab_stats(state: ABTestState, model_id: str, pnl: float | None) -> None:
    state.counts[model_id] = state.counts.get(model_id, 0) + 1
    if pnl is not None:
        state.pnl_by_model[model_id] = state.pnl_by_model.get(model_id, 0.0) + pnl
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/optimize/test_ab_routing.py -v`
Expected: PASS

**Step 5: Commit**

Commit via GitHub MCP.

---

### Task 5: OptimizationEngine Aggregation

**Files:**
- Create: `src/optimize/optimization_engine.py`
- Test: `tests/optimize/test_engine.py`

**Step 1: Write the failing test**

```python
from src.optimize.optimization_engine import OptimizationEngine


def test_engine_refresh_creates_state(tmp_path, store, journal) -> None:
    engine = OptimizationEngine(store, journal, state_path=tmp_path / "state.json")
    state = engine.refresh_state()
    assert state.generated_at
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/optimize/test_engine.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.decision_store.sqlite_store import DecisionStore
from src.monitor.trade_journal import TradeJournal
from src.optimize.ab_testing import update_ab_stats
from src.optimize.optimization_state import OptimizationState, save_state
from src.optimize.thresholds import compute_thresholds
from src.optimize.trade_stats import build_pnl_feedback, compute_win_rates


class OptimizationEngine:
    """Aggregate trade stats and write optimization state."""

    def __init__(
        self,
        store: DecisionStore,
        journal: TradeJournal | None,
        state_path: str | Path,
        pnl_days: int = 7,
        win_days: int = 14,
    ) -> None:
        self._store = store
        self._journal = journal
        self._state_path = Path(state_path)
        self._pnl_days = pnl_days
        self._win_days = win_days

    def refresh_state(self) -> OptimizationState:
        win_rates = compute_win_rates(self._store, days=self._win_days)
        thresholds = compute_thresholds(
            global_win_rate=win_rates.get("global", 0.0),
            symbol_win_rates={k: v for k, v in win_rates.items() if k != "global"},
        )
        state = OptimizationState(
            pnl_lookback_days=self._pnl_days,
            winrate_lookback_days=self._win_days,
            global_thresholds=thresholds["global"],
            symbol_thresholds={k: v for k, v in thresholds.items() if k != "global"},
            feedback_pnl=build_pnl_feedback(self._store, self._journal, days=self._pnl_days),
        )
        save_state(self._state_path, state)
        logger.info("OptimizationEngine: state updated at {}", state.generated_at)
        return state
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/optimize/test_engine.py -v`
Expected: PASS

**Step 5: Commit**

Commit via GitHub MCP.

---

### Task 6: Scheduler Daily Optimization Update

**Files:**
- Modify: `src/scheduler/scheduler.py`
- Modify: `src/main.py`
- Test: `tests/scheduler/test_optimization_integration.py`

**Step 1: Write the failing test**

```python
from src.scheduler.scheduler import Scheduler


def test_daily_summary_triggers_optimization(mocker, scheduler):
    scheduler._optimization_engine = mocker.Mock()
    scheduler._send_daily_summary("2026-02-12")
    scheduler._optimization_engine.refresh_state.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scheduler/test_optimization_integration.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# scheduler.py (init)
self._optimization_engine = optimization_engine
self._optimization_state = None

# scheduler.py
async def _send_daily_summary(self, date_str: str) -> None:
    if self._optimization_engine is not None:
        try:
            self._optimization_state = await asyncio.to_thread(
                self._optimization_engine.refresh_state
            )
        except Exception as e:
            logger.warning("Optimization refresh failed: {}", e)
    # existing summary logic...
```

Update `main._run_scheduler` to construct `OptimizationEngine` and pass into `Scheduler`.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/scheduler/test_optimization_integration.py -v`
Expected: PASS

**Step 5: Commit**

Commit via GitHub MCP.

---

### Task 7: LLM Pre/Post Filters Using Optimization State

**Files:**
- Modify: `src/scheduler/scheduler.py`
- Test: `tests/scheduler/test_llm_thresholds.py`

**Step 1: Write the failing test**

```python
from src.optimize.optimization_state import Thresholds
from src.scheduler.scheduler import Scheduler


def test_pre_filter_blocks_low_confidence(scheduler):
    thresholds = Thresholds(min_confidence="high", min_blended_confidence=0.65)
    assert scheduler._passes_threshold("low", 0.4, thresholds) is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scheduler/test_llm_thresholds.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# scheduler.py (helper)
_conf_map = {"high": 0.9, "medium": 0.6, "low": 0.3}

@staticmethod
def _passes_threshold(confidence: str, blended: float, thresholds: Thresholds) -> bool:
    if _conf_map.get(confidence, 0.0) < _conf_map.get(thresholds.min_confidence, 0.0):
        return False
    return blended >= thresholds.min_blended_confidence

# scheduler.py in _process_claimed_intent before LLM
blended = 0.6 * _conf_map.get(intent.scanner_confidence, 0.5) + 0.4 * intent.scanner_score
thresholds = self._get_thresholds_for_symbol(intent.symbol)
if not self._passes_threshold(intent.scanner_confidence, blended, thresholds):
    await self._cancel_intent_safe(... reason="LLM pre-filter: low confidence")
    return

# scheduler.py after decision/format_decision
if not self._passes_threshold(intent.scanner_confidence, formatted.confidence_score, thresholds):
    await self._cancel_intent_safe(... reason="LLM post-filter: low confidence")
    return
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/scheduler/test_llm_thresholds.py -v`
Expected: PASS

**Step 5: Commit**

Commit via GitHub MCP.

---

### Task 8: MemoryJournal Decision Logging + Close Append

**Files:**
- Modify: `src/monitor/memory_journal.py`
- Modify: `src/scheduler/scheduler.py`
- Test: `tests/monitor/test_memory_journal.py`

**Step 1: Write the failing test**

```python
from src.monitor.memory_journal import MemoryJournal


def test_log_decision_and_append_result(tmp_path):
    journal = MemoryJournal(tmp_path)
    journal.log_decision(symbol="EURUSD", side="HOLD", decision="HOLD")
    journal.append_trade_result(symbol="EURUSD", pnl=12.3, reason="tp_hit")
    content = (tmp_path / "2026-02-26.md").read_text(encoding="utf-8")
    assert "HOLD" in content
    assert "pnl" in content
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/monitor/test_memory_journal.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# memory_journal.py
    def log_decision(self, *, symbol: str, side: str, decision: str, context: dict[str, Any] | None = None) -> None:
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S UTC")
        lines = [f"## {time_str} - {symbol} {decision}", ""]
        lines.append("### Decision Context")
        lines.append("")
        lines.append(f"- **Symbol**: {symbol}")
        lines.append(f"- **Side**: {side}")
        lines.append(f"- **Decision**: {decision}")
        if context:
            for k, v in context.items():
                lines.append(f"- **{k}**: {v}")
        lines.append("")
        lines.append("---\n")
        self._append_to_file(self._memory_dir / f"{date_str}.md", "\n".join(lines))

    def append_trade_result(self, *, symbol: str, pnl: float, reason: str) -> None:
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        lines = ["### Trade Result", "", f"- **Symbol**: {symbol}", f"- **PnL**: {pnl}", f"- **Reason**: {reason}", "", "---\n"]
        self._append_to_file(self._memory_dir / f"{date_str}.md", "\n".join(lines))
```

Wire in scheduler:
- After LLM decision → `memory_journal.log_decision(...)`
- On position close → `memory_journal.append_trade_result(...)`

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/monitor/test_memory_journal.py -v`
Expected: PASS

**Step 5: Commit**

Commit via GitHub MCP.

---

### Task 9: TradeJournal Pipeline Event Logging

**Files:**
- Modify: `src/monitor/trade_journal.py`
- Modify: `src/scheduler/scheduler.py`
- Test: `tests/monitor/test_trade_journal.py`

**Step 1: Write the failing test**

```python
from src.monitor.trade_journal import TradeJournal


def test_log_event_appends(tmp_path):
    path = tmp_path / "trade_journal.jsonl"
    journal = TradeJournal(path)
    journal.log_event("LLM_DECISION", {"symbol": "EURUSD", "decision": "BUY"})
    content = path.read_text(encoding="utf-8")
    assert "LLM_DECISION" in content
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/monitor/test_trade_journal.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# trade_journal.py
    def log_event(self, event_type: str, details: dict[str, Any] | None = None) -> None:
        entry = {"type": event_type, **(details or {})}
        self._append(entry)
```

Wire in scheduler:
- Intent created
- LLM decision (include model_id, latency_ms)
- Opened / Closed / Rejected / Failed

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/monitor/test_trade_journal.py -v`
Expected: PASS

**Step 5: Commit**

Commit via GitHub MCP.

---

**Plan complete and saved to `docs/plans/2026-02-26-memory-feedback-optimization.md`. Two execution options:**

1. Subagent-Driven (this session)
2. Parallel Session (separate)

Which approach?
