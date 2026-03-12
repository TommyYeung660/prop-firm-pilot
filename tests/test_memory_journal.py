"""Tests for MemoryJournal."""

from datetime import datetime, timezone

import pytest

from src.compliance.prop_firm_guard import TradePlan
from src.decision.agent_bridge import AgentDecision
from src.monitor.memory_journal import MemoryJournal

# ── Mock Classes ─────────────────────────────────────────────────────────────


class MockSignal:
    """Mock signal object with qlib scanner data."""

    def __init__(self, instrument: str, score: float, confidence: str, score_gap: float):
        self.instrument = instrument
        self.score = score
        self.confidence = confidence
        self.score_gap = score_gap


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def mock_datetime(monkeypatch):
    """Mock datetime.now() to return a fixed timestamp."""

    class MockDatetime:
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 2, 20, 14, 30, 45, tzinfo=tz or timezone.utc)

    monkeypatch.setattr("src.monitor.memory_journal.datetime", MockDatetime)


@pytest.fixture
def trade_plan() -> TradePlan:
    """Create a mock TradePlan."""
    return TradePlan(
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
        stop_loss=1.0850,
        take_profit=1.0950,
        risk_amount=100.0,
    )


@pytest.fixture
def signal():
    """Create a mock signal object."""
    return MockSignal(
        instrument="EURUSD",
        score=0.85,
        confidence="high",
        score_gap=0.12,
    )


@pytest.fixture
def agent_decision() -> AgentDecision:
    """Create a mock AgentDecision."""
    return AgentDecision(
        symbol="EURUSD",
        decision="BUY",
        final_state={
            "trader_investment_plan": "Strong bullish signal with high confidence",
            "market_sentiment": "bullish",
        },
        risk_report="Risk acceptable: R:R ratio 1:2, stop loss at key support level",
    )


# ── Tests ───────────────────────────────────────────────────────────────────


def test_memory_journal_creates_directory(tmp_path):
    """Test that MemoryJournal creates directory if it doesn't exist."""
    memory_dir = tmp_path / "MEMORY"
    assert not memory_dir.exists()

    MemoryJournal(memory_dir)

    assert memory_dir.exists()
    assert memory_dir.is_dir()


def test_log_trade_decision_creates_file(
    tmp_path, mock_datetime, trade_plan, signal, agent_decision
):
    """Test that log_trade_decision creates the daily Markdown file."""
    memory_dir = tmp_path / "MEMORY"
    journal = MemoryJournal(memory_dir)

    journal.log_trade_decision(trade_plan, signal, agent_decision)

    expected_file = memory_dir / "2026-02-20.md"
    assert expected_file.exists()
    assert expected_file.is_file()


def test_log_trade_decision_formats_correctly(
    tmp_path, mock_datetime, trade_plan, signal, agent_decision
):
    """Test that log_trade_decision formats the Markdown correctly."""
    memory_dir = tmp_path / "MEMORY"
    journal = MemoryJournal(memory_dir)

    journal.log_trade_decision(trade_plan, signal, agent_decision)

    file_path = memory_dir / "2026-02-20.md"
    content = file_path.read_text(encoding="utf-8")

    # Check heading
    assert "## 14:30:45 UTC - EURUSD BUY" in content

    # Check Trade Details section
    assert "### Trade Details" in content
    assert "**Symbol**: EURUSD" in content
    assert "**Side**: BUY" in content
    assert "**Volume**: 0.1" in content
    assert "**Stop Loss**: 1.085" in content
    assert "**Take Profit**: 1.095" in content
    assert "**Risk Amount**: $100.00" in content

    # Check Scanner Signal section
    assert "### Scanner Signal (Qlib)" in content
    assert "**Instrument**: EURUSD" in content
    assert "**Score**: 0.85" in content
    assert "**Confidence**: high" in content
    assert "**Score Gap**: 0.12" in content

    # Check TradingAgents Reasoning section
    assert "### TradingAgents Reasoning" in content
    assert "**Decision**: BUY" in content
    assert "**Risk Report**:" in content
    assert "Risk acceptable: R:R ratio 1:2" in content
    assert "**Final State**:" in content
    assert "Strong bullish signal with high confidence" in content


def test_log_multiple_trades_appends_to_same_file(
    tmp_path, mock_datetime, trade_plan, signal, agent_decision
):
    """Test that multiple trades are appended to the same daily file."""
    memory_dir = tmp_path / "MEMORY"
    journal = MemoryJournal(memory_dir)

    # Log first trade
    journal.log_trade_decision(trade_plan, signal, agent_decision)

    # Create second trade
    trade_plan2 = TradePlan(
        symbol="GBPUSD",
        side="SELL",
        volume=0.15,
        stop_loss=1.2650,
        take_profit=1.2550,
        risk_amount=150.0,
    )
    signal2 = MockSignal(
        instrument="GBPUSD",
        score=0.72,
        confidence="medium",
        score_gap=0.08,
    )
    agent_decision2 = AgentDecision(
        symbol="GBPUSD",
        decision="SELL",
        final_state={"market_sentiment": "bearish"},
        risk_report="Moderate risk: stop at resistance",
    )

    # Log second trade
    journal.log_trade_decision(trade_plan2, signal2, agent_decision2)

    file_path = memory_dir / "2026-02-20.md"
    content = file_path.read_text(encoding="utf-8")

    # Both trades should be present
    assert "## 14:30:45 UTC - EURUSD BUY" in content
    assert "## 14:30:45 UTC - GBPUSD SELL" in content
    assert content.count("### Trade Details") == 2
    assert content.count("---") == 2


def test_signal_without_score_gap(tmp_path, mock_datetime, trade_plan, agent_decision):
    """Test formatting when signal has no score_gap."""
    memory_dir = tmp_path / "MEMORY"
    journal = MemoryJournal(memory_dir)

    # Signal without score_gap
    class SignalNoGap:
        instrument = "EURUSD"
        score = 0.85
        confidence = "high"

    signal = SignalNoGap()
    # Explicitly set score_gap to None
    signal.score_gap = None

    journal.log_trade_decision(trade_plan, signal, agent_decision)

    file_path = memory_dir / "2026-02-20.md"
    content = file_path.read_text(encoding="utf-8")

    # Should not include Score Gap line
    assert "**Score Gap**:" not in content
    assert "**Score**: 0.85" in content


def test_agent_decision_empty_fields(tmp_path, mock_datetime, trade_plan, signal):
    """Test formatting when agent decision has empty fields."""
    memory_dir = tmp_path / "MEMORY"
    journal = MemoryJournal(memory_dir)

    agent_decision = AgentDecision(
        symbol="EURUSD",
        decision="HOLD",
        final_state={},
        risk_report="",
    )

    journal.log_trade_decision(trade_plan, signal, agent_decision)

    file_path = memory_dir / "2026-02-20.md"
    content = file_path.read_text(encoding="utf-8")

    # Decision should still be present
    assert "**Decision**: HOLD" in content

    # Empty risk report and state should NOT show sections (skip empty)
    assert "**Risk Report**:" not in content
    assert "**Final State**:" not in content


def test_log_decision_and_append_result(tmp_path, mock_datetime):
    """Log a decision and append trade result to the same daily file."""
    memory_dir = tmp_path / "MEMORY"
    journal = MemoryJournal(memory_dir)

    journal.log_decision(
        symbol="EURUSD",
        side="HOLD",
        decision="HOLD",
        context={"intent_id": "intent-1", "reason": "low_confidence"},
    )
    journal.append_trade_result(
        intent_id="intent-1",
        position_id="pos-1",
        symbol="EURUSD",
        pnl=12.3,
        reason="tp_hit",
    )

    file_path = memory_dir / "2026-02-20.md"
    content = file_path.read_text(encoding="utf-8")

    assert "## 14:30:45 UTC - EURUSD HOLD" in content
    assert "### Decision Context" in content
    assert "**Decision**: HOLD" in content
    assert "**reason**: low_confidence" in content
    assert "### Trade Result" in content
    assert "**Intent ID**: intent-1" in content
    assert "**Position ID**: pos-1" in content
    assert "**PnL**: 12.3" in content
    assert "**Reason**: tp_hit" in content


def test_append_trade_result_uses_anchored_decision_file_across_days(tmp_path, monkeypatch):
    """Trade results should append to the decision file identified by intent_id."""
    memory_dir = tmp_path / "MEMORY"

    class MockDatetime:
        current = datetime(2026, 2, 20, 14, 30, 45, tzinfo=timezone.utc)

        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return cls.current
            return cls.current.astimezone(tz)

    monkeypatch.setattr("src.monitor.memory_journal.datetime", MockDatetime)

    journal = MemoryJournal(memory_dir)
    journal.log_decision(
        symbol="EURUSD",
        side="BUY",
        decision="BUY",
        context={"intent_id": "intent-anchor", "reason": "cross_day_close"},
    )

    MockDatetime.current = datetime(2026, 2, 21, 9, 5, 0, tzinfo=timezone.utc)
    journal.append_trade_result(
        intent_id="intent-anchor",
        position_id="pos-cross-day",
        symbol="EURUSD",
        pnl=7.5,
        reason="tp_hit",
    )

    day_one_file = memory_dir / "2026-02-20.md"
    day_two_file = memory_dir / "2026-02-21.md"
    day_one_content = day_one_file.read_text(encoding="utf-8")

    assert "### Trade Result" in day_one_content
    assert "**Intent ID**: intent-anchor" in day_one_content
    assert "**Position ID**: pos-cross-day" in day_one_content
    assert "**PnL**: 7.5" in day_one_content
    assert not day_two_file.exists()


def test_append_trade_result_skips_when_symbol_mismatch(tmp_path, monkeypatch):
    """Trade result must not be written when symbol does not match anchored decision."""
    memory_dir = tmp_path / "MEMORY"

    class MockDatetime:
        current = datetime(2026, 2, 20, 14, 30, 45, tzinfo=timezone.utc)

        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return cls.current
            return cls.current.astimezone(tz)

    monkeypatch.setattr("src.monitor.memory_journal.datetime", MockDatetime)

    journal = MemoryJournal(memory_dir)
    journal.log_decision(
        symbol="EURUSD",
        side="SELL",
        decision="SELL",
        context={"intent_id": "intent-mismatch", "reason": "symbol_guard"},
    )

    journal.append_trade_result(
        intent_id="intent-mismatch",
        position_id="pos-mismatch",
        symbol="AUDUSD",
        pnl=-3.2,
        reason="sl_hit",
    )

    file_path = memory_dir / "2026-02-20.md"
    content = file_path.read_text(encoding="utf-8")

    assert "## 14:30:45 UTC - EURUSD SELL" in content
    assert "### Trade Result" not in content
