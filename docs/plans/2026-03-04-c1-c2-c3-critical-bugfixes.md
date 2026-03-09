# C1-C2-C3 Critical Bugfixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix three critical production bugs: C1 (HOLD→BUY mapping), C2 (LLM refusal→SELL mapping), C3 (infinite retry loop after compliance rejection).

**Architecture:** Three surgical fixes — (1) Add risk_report cross-validation and LLM refusal detection in `agent_bridge.py`, (2) No changes needed in `decision_formatter.py` (the real bug is upstream in agent_bridge), (3) Add `rejected` status to `intent_exists()` query with a configurable cooldown window in `sqlite_store.py`.

**Tech Stack:** Python 3.10, pytest, SQLite, loguru, Pydantic v2

---

## Root Cause Analysis

### C1: HOLD decision incorrectly mapped to BUY
- **Location:** `src/decision/agent_bridge.py:236` — `decision=decision` trusts `propagate()` blindly
- **Problem:** TradingAgents `propagate()` returns `decision="BUY"` even when the risk_report contains "FINAL TRANSACTION PROPOSAL: HOLD"
- **Fix:** After `propagate()` returns, parse risk_report for the actual "FINAL TRANSACTION PROPOSAL" and override `decision` if it contradicts

### C2: LLM refusal incorrectly mapped to SELL
- **Location:** Same as C1 — `src/decision/agent_bridge.py:236`
- **Problem:** When GPT refuses to give trading advice (e.g. outputs "我無法依照你的要求提供明確買/賣/持有指令"), the `decision` variable from `propagate()` can still be "SELL" or "BUY"
- **Fix:** Detect refusal patterns in risk_report text and force `decision="HOLD"` when detected

### C3: Infinite retry loop after compliance rejection
- **Location:** `src/decision_store/sqlite_store.py:788-805` — `intent_exists()` only checks `pending`, `claimed`, `ready_for_exec`
- **Problem:** After Best Day compliance rejects an intent (→ status `rejected`), `intent_exists()` returns False, so scanner creates a NEW intent for the same symbol → LLM evaluates again → rejected again → loop burns LLM tokens for hours
- **Fix:** Add a new method `has_recent_rejection()` that checks for recently-rejected intents within a cooldown window, and call it from the scanner loop before creating new intents

---

## Task 1: C1+C2 Fix — Risk Report Cross-Validation & Refusal Detection in AgentBridge

**Files:**
- Modify: `src/decision/agent_bridge.py:234-247` (add validation after propagate)
- Test: `tests/test_agent_bridge_decision_validation.py` (new file)

### Step 1: Write failing tests for risk_report cross-validation and refusal detection

Create `tests/test_agent_bridge_decision_validation.py`:

```python
"""
Tests for AgentBridge decision validation — ensures risk_report
cross-validation and LLM refusal detection work correctly.

Guards against C1 (HOLD→BUY mapping) and C2 (LLM refusal→SELL mapping)
production bugs found in v1.3.5 prod run (2026-03-03).
"""

import pytest

from src.decision.agent_bridge import AgentDecision, validate_decision


# ── C1: Risk report says HOLD but decision says BUY/SELL ──────────────────


class TestRiskReportCrossValidation:
    """When risk_report contains 'FINAL TRANSACTION PROPOSAL: HOLD',
    decision must be overridden to HOLD regardless of propagate() return."""

    def test_hold_in_report_overrides_buy(self) -> None:
        risk_report = (
            "After careful analysis... FINAL TRANSACTION PROPOSAL: HOLD\n"
            "Risk is too high for entry."
        )
        result = validate_decision(
            raw_decision="BUY", risk_report=risk_report, symbol="EURUSD"
        )
        assert result == "HOLD"

    def test_hold_in_report_overrides_sell(self) -> None:
        risk_report = "... FINAL TRANSACTION PROPOSAL: HOLD ..."
        result = validate_decision(
            raw_decision="SELL", risk_report=risk_report, symbol="EURUSD"
        )
        assert result == "HOLD"

    def test_buy_in_report_matches_buy_decision(self) -> None:
        risk_report = "... FINAL TRANSACTION PROPOSAL: BUY ..."
        result = validate_decision(
            raw_decision="BUY", risk_report=risk_report, symbol="EURUSD"
        )
        assert result == "BUY"

    def test_sell_in_report_matches_sell_decision(self) -> None:
        risk_report = "... FINAL TRANSACTION PROPOSAL: SELL ..."
        result = validate_decision(
            raw_decision="SELL", risk_report=risk_report, symbol="EURUSD"
        )
        assert result == "SELL"

    def test_no_proposal_in_report_trusts_decision(self) -> None:
        """When risk_report has no FINAL TRANSACTION PROPOSAL, trust propagate()."""
        risk_report = "General analysis without a clear proposal."
        result = validate_decision(
            raw_decision="BUY", risk_report=risk_report, symbol="EURUSD"
        )
        assert result == "BUY"

    def test_empty_risk_report_trusts_decision(self) -> None:
        result = validate_decision(
            raw_decision="SELL", risk_report="", symbol="EURUSD"
        )
        assert result == "SELL"

    def test_report_buy_but_decision_sell_uses_report(self) -> None:
        """When report says BUY but propagate says SELL, trust the report."""
        risk_report = "... FINAL TRANSACTION PROPOSAL: BUY ..."
        result = validate_decision(
            raw_decision="SELL", risk_report=risk_report, symbol="EURUSD"
        )
        assert result == "BUY"


# ── C2: LLM refusal detection ──────────────────────────────────────────────


class TestLLMRefusalDetection:
    """When LLM refuses to give trading advice, decision must be HOLD."""

    def test_chinese_refusal_pattern(self) -> None:
        risk_report = "我無法依照你的要求提供明確買/賣/持有指令"
        result = validate_decision(
            raw_decision="SELL", risk_report=risk_report, symbol="EURUSD"
        )
        assert result == "HOLD"

    def test_english_unable_pattern(self) -> None:
        risk_report = "I'm unable to provide specific trading recommendations."
        result = validate_decision(
            raw_decision="BUY", risk_report=risk_report, symbol="EURUSD"
        )
        assert result == "HOLD"

    def test_english_cannot_pattern(self) -> None:
        risk_report = "I cannot provide financial advice or trading signals."
        result = validate_decision(
            raw_decision="BUY", risk_report=risk_report, symbol="EURUSD"
        )
        assert result == "HOLD"

    def test_disclaimer_pattern(self) -> None:
        risk_report = (
            "As an AI language model, I cannot give specific buy or sell "
            "recommendations for financial instruments."
        )
        result = validate_decision(
            raw_decision="SELL", risk_report=risk_report, symbol="EURUSD"
        )
        assert result == "HOLD"

    def test_not_financial_advisor_pattern(self) -> None:
        risk_report = "I'm not a financial advisor and cannot recommend trades."
        result = validate_decision(
            raw_decision="BUY", risk_report=risk_report, symbol="EURUSD"
        )
        assert result == "HOLD"

    def test_normal_report_not_flagged(self) -> None:
        """A real analysis report should NOT trigger refusal detection."""
        risk_report = (
            "Based on technical analysis, EURUSD shows bullish momentum. "
            "RSI at 62, MACD crossing above signal line. "
            "FINAL TRANSACTION PROPOSAL: BUY"
        )
        result = validate_decision(
            raw_decision="BUY", risk_report=risk_report, symbol="EURUSD"
        )
        assert result == "BUY"

    def test_hold_decision_stays_hold(self) -> None:
        """HOLD from propagate() should stay HOLD even without refusal."""
        risk_report = "Market is ranging, no clear direction."
        result = validate_decision(
            raw_decision="HOLD", risk_report=risk_report, symbol="EURUSD"
        )
        assert result == "HOLD"


# ── Edge cases ──────────────────────────────────────────────────────────────


class TestValidateDecisionEdgeCases:
    """Edge cases for decision validation."""

    def test_case_insensitive_proposal(self) -> None:
        risk_report = "Final Transaction Proposal: hold"
        result = validate_decision(
            raw_decision="BUY", risk_report=risk_report, symbol="EURUSD"
        )
        assert result == "HOLD"

    def test_refusal_overrides_even_with_buy_proposal(self) -> None:
        """If refusal is detected, force HOLD even if there's a BUY proposal."""
        risk_report = (
            "I cannot provide trading advice. "
            "FINAL TRANSACTION PROPOSAL: BUY"
        )
        result = validate_decision(
            raw_decision="BUY", risk_report=risk_report, symbol="EURUSD"
        )
        assert result == "HOLD"

    def test_non_standard_decision_becomes_hold(self) -> None:
        """Garbage decision values from propagate() should become HOLD."""
        result = validate_decision(
            raw_decision="MAYBE", risk_report="some text", symbol="EURUSD"
        )
        assert result == "HOLD"

    def test_none_decision_becomes_hold(self) -> None:
        result = validate_decision(
            raw_decision=None, risk_report="some text", symbol="EURUSD"
        )
        assert result == "HOLD"
```

### Step 2: Run tests to verify they fail

Run: `uv run pytest tests/test_agent_bridge_decision_validation.py -v`
Expected: FAIL — `validate_decision` does not exist yet.

### Step 3: Implement `validate_decision()` in agent_bridge.py

Add the following function and modify `decide()` to call it.

**Add `validate_decision()` function** (insert after `AgentDecision` class, before `MockTradingGraph`):

```python
import re

# ── LLM Refusal Detection Patterns ──────────────────────────────────────────

_REFUSAL_PATTERNS: list[re.Pattern[str]] = [
    # Chinese refusal patterns
    re.compile(r"我無法", re.IGNORECASE),
    re.compile(r"無法提供.*(?:建議|指令|推薦)", re.IGNORECASE),
    # English refusal patterns
    re.compile(r"I(?:'m| am) unable to provide", re.IGNORECASE),
    re.compile(r"I cannot (?:provide|give|offer|recommend)", re.IGNORECASE),
    re.compile(r"(?:not|neither) a financial advisor", re.IGNORECASE),
    re.compile(r"cannot (?:give|provide|offer) (?:specific |)(?:trading |financial )", re.IGNORECASE),
    re.compile(r"as an AI(?:\s+language)?\s+model", re.IGNORECASE),
]

# Regex to extract "FINAL TRANSACTION PROPOSAL: BUY/SELL/HOLD" from risk_report
_PROPOSAL_RE = re.compile(
    r"FINAL\s+TRANSACTION\s+PROPOSAL\s*:\s*(BUY|SELL|HOLD)",
    re.IGNORECASE,
)


def validate_decision(
    raw_decision: str | None,
    risk_report: str,
    symbol: str,
) -> Literal["BUY", "SELL", "HOLD"]:
    """Cross-validate LLM decision against risk_report content.

    Applies three safety layers:
    1. Normalize non-standard decision values → HOLD.
    2. Detect LLM refusal patterns → force HOLD.
    3. Extract FINAL TRANSACTION PROPOSAL from risk_report → override if mismatch.

    Args:
        raw_decision: Decision string from TradingAgents propagate() (may be None/garbage).
        risk_report: Full risk report text from TradingAgents.
        symbol: FX pair (for logging).

    Returns:
        Validated decision: "BUY", "SELL", or "HOLD".
    """
    # Layer 1: Normalize
    normalized = str(raw_decision).upper().strip() if raw_decision else "HOLD"
    if normalized not in ("BUY", "SELL", "HOLD"):
        logger.warning(
            "validate_decision: non-standard decision '{}' for {} → HOLD",
            raw_decision,
            symbol,
        )
        normalized = "HOLD"

    # Layer 2: Refusal detection (highest priority — overrides everything)
    for pattern in _REFUSAL_PATTERNS:
        if pattern.search(risk_report):
            if normalized != "HOLD":
                logger.warning(
                    "validate_decision: LLM refusal detected for {} "
                    "(decision was '{}', pattern={}), forcing HOLD",
                    symbol,
                    normalized,
                    pattern.pattern,
                )
            return "HOLD"

    # Layer 3: Cross-validate with FINAL TRANSACTION PROPOSAL
    match = _PROPOSAL_RE.search(risk_report)
    if match:
        proposal = match.group(1).upper()
        if proposal != normalized:
            logger.warning(
                "validate_decision: decision/report mismatch for {} "
                "(propagate='{}', report='{}'), using report value",
                symbol,
                normalized,
                proposal,
            )
            return proposal  # type: ignore[return-value]

    return normalized  # type: ignore[return-value]
```

**Modify `decide()` method** at line ~234-247 — replace the `AgentDecision(...)` construction:

Change from:
```python
result = AgentDecision(
    symbol=symbol,
    decision=decision,
    final_state=final_state if isinstance(final_state, dict) else {},
    risk_report=risk_report,
)
```

To:
```python
validated_decision = validate_decision(
    raw_decision=decision,
    risk_report=risk_report,
    symbol=symbol,
)
result = AgentDecision(
    symbol=symbol,
    decision=validated_decision,
    final_state=final_state if isinstance(final_state, dict) else {},
    risk_report=risk_report,
)
```

### Step 4: Run tests to verify they pass

Run: `uv run pytest tests/test_agent_bridge_decision_validation.py -v`
Expected: ALL PASS

### Step 5: Run existing tests to ensure no regressions

Run: `uv run pytest tests/test_agent_bridge_config.py -v`
Expected: ALL PASS (existing tests unaffected)

### Step 6: Lint check

Run: `uv run ruff check src/decision/agent_bridge.py tests/test_agent_bridge_decision_validation.py`
Expected: No errors

### Step 7: Commit

```bash
git add src/decision/agent_bridge.py tests/test_agent_bridge_decision_validation.py
git commit -m "fix(C1+C2): add risk_report cross-validation and LLM refusal detection in AgentBridge

Fixes two critical production bugs from v1.3.5:
- C1: HOLD in risk_report but propagate() returned BUY → now overridden
- C2: LLM refusal text but propagate() returned SELL → now forced to HOLD

Three-layer validation: normalize → refusal detect → cross-validate proposal."
```

---

## Task 2: C3 Fix — Compliance Rejection Cooldown in Scanner Loop

**Files:**
- Modify: `src/decision_store/sqlite_store.py` (add `has_recent_rejection()`)
- Modify: `src/scheduler/scheduler.py:282-294` (call `has_recent_rejection()` before creating intents)
- Test: `tests/test_rejection_cooldown.py` (new file)

### Step 1: Write failing tests for rejection cooldown

Create `tests/test_rejection_cooldown.py`:

```python
"""
Tests for compliance rejection cooldown — prevents infinite retry loops.

Guards against C3 production bug: scanner creates new intent → LLM evaluates →
compliance rejects (Best Day) → scanner creates new intent → infinite loop
burning LLM tokens for hours.

The fix adds has_recent_rejection() to DecisionStore, which the scanner loop
checks before creating new intents.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.decision.schemas import TradeIntent
from src.decision_store.sqlite_store import DecisionStore


TRADE_DATE = "2026-03-03"
SYMBOL = "EURUSD"
SOURCE = "scanner"


@pytest.fixture
def store(tmp_path) -> DecisionStore:
    db_path = f"{tmp_path}/test_cooldown.db"
    s = DecisionStore(db_path=db_path)
    yield s  # type: ignore[misc]
    s.close()


def _create_rejected_intent(
    store: DecisionStore,
    symbol: str = SYMBOL,
    trade_date: str = TRADE_DATE,
    rejected_ago_minutes: int = 0,
) -> TradeIntent:
    """Insert an intent and force it to rejected status with a specific executed_at time."""
    intent = TradeIntent(
        trade_date=trade_date,
        symbol=symbol,
        scanner_score=0.85,
        scanner_confidence="high",
        source=SOURCE,
    )
    store.insert_intent(intent)
    rejected_at = datetime.now(timezone.utc) - timedelta(minutes=rejected_ago_minutes)
    store._conn.execute(
        "UPDATE intents SET status = 'rejected', executed_at = ? WHERE id = ?",
        (rejected_at.isoformat(), intent.id),
    )
    store._conn.commit()
    return intent


# ── has_recent_rejection basic behavior ────────────────────────────────────


class TestHasRecentRejection:
    """has_recent_rejection() returns True when a rejected intent exists
    within the cooldown window."""

    def test_recently_rejected_blocks(self, store: DecisionStore) -> None:
        """Intent rejected 5 minutes ago with 60-min cooldown → blocked."""
        _create_rejected_intent(store, rejected_ago_minutes=5)
        assert store.has_recent_rejection(SYMBOL, TRADE_DATE, cooldown_minutes=60) is True

    def test_old_rejection_does_not_block(self, store: DecisionStore) -> None:
        """Intent rejected 120 minutes ago with 60-min cooldown → not blocked."""
        _create_rejected_intent(store, rejected_ago_minutes=120)
        assert store.has_recent_rejection(SYMBOL, TRADE_DATE, cooldown_minutes=60) is False

    def test_no_rejection_does_not_block(self, store: DecisionStore) -> None:
        """No rejected intents → not blocked."""
        assert store.has_recent_rejection(SYMBOL, TRADE_DATE, cooldown_minutes=60) is False

    def test_different_symbol_not_blocked(self, store: DecisionStore) -> None:
        """Rejected EURUSD should not block GBPUSD."""
        _create_rejected_intent(store, symbol="EURUSD", rejected_ago_minutes=5)
        assert store.has_recent_rejection("GBPUSD", TRADE_DATE, cooldown_minutes=60) is False

    def test_different_date_not_blocked(self, store: DecisionStore) -> None:
        """Rejected intent for a different date should not block."""
        _create_rejected_intent(store, trade_date="2026-03-02", rejected_ago_minutes=5)
        assert store.has_recent_rejection(SYMBOL, TRADE_DATE, cooldown_minutes=60) is False

    def test_non_rejected_statuses_not_counted(self, store: DecisionStore) -> None:
        """Only 'rejected' status should trigger cooldown, not cancelled/failed."""
        intent = TradeIntent(
            trade_date=TRADE_DATE,
            symbol=SYMBOL,
            scanner_score=0.85,
            scanner_confidence="high",
            source=SOURCE,
        )
        store.insert_intent(intent)
        now_str = datetime.now(timezone.utc).isoformat()
        store._conn.execute(
            "UPDATE intents SET status = 'cancelled', executed_at = ? WHERE id = ?",
            (now_str, intent.id),
        )
        store._conn.commit()
        assert store.has_recent_rejection(SYMBOL, TRADE_DATE, cooldown_minutes=60) is False

    def test_zero_cooldown_always_allows(self, store: DecisionStore) -> None:
        """With cooldown_minutes=0, even recent rejections don't block."""
        _create_rejected_intent(store, rejected_ago_minutes=1)
        assert store.has_recent_rejection(SYMBOL, TRADE_DATE, cooldown_minutes=0) is False

    def test_edge_of_cooldown_window(self, store: DecisionStore) -> None:
        """Intent rejected exactly at cooldown boundary → not blocked (exclusive)."""
        _create_rejected_intent(store, rejected_ago_minutes=60)
        assert store.has_recent_rejection(SYMBOL, TRADE_DATE, cooldown_minutes=60) is False
```

### Step 2: Run tests to verify they fail

Run: `uv run pytest tests/test_rejection_cooldown.py -v`
Expected: FAIL — `has_recent_rejection` does not exist yet.

### Step 3: Implement `has_recent_rejection()` in sqlite_store.py

Add the following method to `DecisionStore` class, after `intent_exists()` (line ~806):

```python
def has_recent_rejection(
    self,
    symbol: str,
    trade_date: str,
    cooldown_minutes: int = 60,
) -> bool:
    """Check if this symbol+date has a recently rejected intent.

    Used by the scanner loop to avoid re-creating intents that will be
    immediately rejected by compliance (e.g., Best Day Rule), preventing
    infinite retry loops that burn LLM tokens.

    Args:
        symbol: FX pair (e.g. "EURUSD").
        trade_date: Trading date string (e.g. "2026-03-03").
        cooldown_minutes: Reject intents created within this many minutes
            of the most recent rejection. 0 disables the cooldown.

    Returns:
        True if a rejected intent exists within the cooldown window.
    """
    if cooldown_minutes <= 0:
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=cooldown_minutes)
    cutoff_str = _dt_to_str(cutoff)
    row = self._conn.execute(
        """SELECT 1 FROM intents
           WHERE symbol = :symbol
             AND trade_date = :td
             AND status = 'rejected'
             AND executed_at IS NOT NULL
             AND executed_at > :cutoff
           LIMIT 1""",
        {"symbol": symbol, "td": trade_date, "cutoff": cutoff_str},
    ).fetchone()
    return row is not None
```

### Step 4: Run tests to verify they pass

Run: `uv run pytest tests/test_rejection_cooldown.py -v`
Expected: ALL PASS

### Step 5: Integrate into scanner loop

**Modify `src/scheduler/scheduler.py`** — in `_scanner_loop()`, after the `intent_exists` check (line ~289-294), add a `has_recent_rejection` check:

Find the block (lines ~282-294):
```python
# Idempotency: skip if an in-progress intent already exists
exists = await asyncio.to_thread(
    self._store.intent_exists,
    signal.instrument,
    today,
    "scanner",
)
if exists:
    logger.info(
        "Scanner loop: in-progress intent exists for {}, skipping",
        signal.instrument,
    )
    continue
```

**Insert AFTER this block** (before the `intent = TradeIntent(...)` line):

```python
# C3 fix: Skip symbols with recent compliance rejection (cooldown)
rejection_cooldown = getattr(
    self._config.scheduler, "rejection_cooldown_minutes", 120
)
recently_rejected = await asyncio.to_thread(
    self._store.has_recent_rejection,
    signal.instrument,
    today,
    cooldown_minutes=rejection_cooldown,
)
if recently_rejected:
    logger.warning(
        "Scanner loop: {} was rejected within {}min cooldown, "
        "skipping to avoid retry loop",
        signal.instrument,
        rejection_cooldown,
    )
    continue
```

### Step 6: Add config field for rejection_cooldown_minutes

**Modify `src/config.py`** — add `rejection_cooldown_minutes` to `SchedulerConfig`:

Find the `SchedulerConfig` class and add:
```python
rejection_cooldown_minutes: int = Field(
    default=120,
    description="Minutes to wait after a compliance rejection before retrying the same symbol. "
    "Prevents infinite scanner → LLM → rejection loops.",
)
```

### Step 7: Run full test suite

Run: `uv run pytest tests/test_rejection_cooldown.py tests/test_intent_exists.py tests/test_scheduler.py -v`
Expected: ALL PASS

### Step 8: Lint check

Run: `uv run ruff check src/decision_store/sqlite_store.py src/scheduler/scheduler.py src/config.py tests/test_rejection_cooldown.py`
Expected: No errors

### Step 9: Commit

```bash
git add src/decision_store/sqlite_store.py src/scheduler/scheduler.py src/config.py tests/test_rejection_cooldown.py
git commit -m "fix(C3): add compliance rejection cooldown to prevent infinite retry loops

Adds has_recent_rejection() to DecisionStore — scanner loop now checks
if a symbol was recently rejected by compliance before creating a new intent.
Default cooldown: 120 minutes (configurable via rejection_cooldown_minutes).

Fixes 7-hour infinite loop in v1.3.5 prod where Best Day rejection caused
scanner → LLM → reject → scanner cycles, burning LLM tokens."
```

---

## Task 3: Integration Verification

### Step 1: Run full test suite

Run: `uv run pytest -v`
Expected: ALL existing + new tests PASS

### Step 2: Lint entire codebase

Run: `uv run ruff check src/ tests/`
Expected: No errors (or only pre-existing ones)

### Step 3: Format check

Run: `uv run ruff format --check src/ tests/`
Expected: No formatting issues

### Step 4: Final commit (if format changes needed)

```bash
uv run ruff format src/ tests/
git add -u
git commit -m "style: format after C1-C3 bugfixes"
```
