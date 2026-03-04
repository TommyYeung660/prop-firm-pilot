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
        result = validate_decision(raw_decision="BUY", risk_report=risk_report, symbol="EURUSD")
        assert result == "HOLD"

    def test_hold_in_report_overrides_sell(self) -> None:
        risk_report = "... FINAL TRANSACTION PROPOSAL: HOLD ..."
        result = validate_decision(raw_decision="SELL", risk_report=risk_report, symbol="EURUSD")
        assert result == "HOLD"

    def test_buy_in_report_matches_buy_decision(self) -> None:
        risk_report = "... FINAL TRANSACTION PROPOSAL: BUY ..."
        result = validate_decision(raw_decision="BUY", risk_report=risk_report, symbol="EURUSD")
        assert result == "BUY"

    def test_sell_in_report_matches_sell_decision(self) -> None:
        risk_report = "... FINAL TRANSACTION PROPOSAL: SELL ..."
        result = validate_decision(raw_decision="SELL", risk_report=risk_report, symbol="EURUSD")
        assert result == "SELL"

    def test_no_proposal_in_report_trusts_decision(self) -> None:
        """When risk_report has no FINAL TRANSACTION PROPOSAL, trust propagate()."""
        risk_report = "General analysis without a clear proposal."
        result = validate_decision(raw_decision="BUY", risk_report=risk_report, symbol="EURUSD")
        assert result == "BUY"

    def test_empty_risk_report_trusts_decision(self) -> None:
        result = validate_decision(raw_decision="SELL", risk_report="", symbol="EURUSD")
        assert result == "SELL"

    def test_report_buy_but_decision_sell_uses_report(self) -> None:
        """When report says BUY but propagate says SELL, trust the report."""
        risk_report = "... FINAL TRANSACTION PROPOSAL: BUY ..."
        result = validate_decision(raw_decision="SELL", risk_report=risk_report, symbol="EURUSD")
        assert result == "BUY"


# ── C2: LLM refusal detection ──────────────────────────────────────────────


class TestLLMRefusalDetection:
    """When LLM refuses to give trading advice, decision must be HOLD."""

    def test_chinese_refusal_pattern(self) -> None:
        risk_report = "我無法依照你的要求提供明確買/賣/持有指令"
        result = validate_decision(raw_decision="SELL", risk_report=risk_report, symbol="EURUSD")
        assert result == "HOLD"

    def test_english_unable_pattern(self) -> None:
        risk_report = "I'm unable to provide specific trading recommendations."
        result = validate_decision(raw_decision="BUY", risk_report=risk_report, symbol="EURUSD")
        assert result == "HOLD"

    def test_english_cannot_pattern(self) -> None:
        risk_report = "I cannot provide financial advice or trading signals."
        result = validate_decision(raw_decision="BUY", risk_report=risk_report, symbol="EURUSD")
        assert result == "HOLD"

    def test_disclaimer_pattern(self) -> None:
        risk_report = (
            "As an AI language model, I cannot give specific buy or sell "
            "recommendations for financial instruments."
        )
        result = validate_decision(raw_decision="SELL", risk_report=risk_report, symbol="EURUSD")
        assert result == "HOLD"

    def test_not_financial_advisor_pattern(self) -> None:
        risk_report = "I'm not a financial advisor and cannot recommend trades."
        result = validate_decision(raw_decision="BUY", risk_report=risk_report, symbol="EURUSD")
        assert result == "HOLD"

    def test_normal_report_not_flagged(self) -> None:
        """A real analysis report should NOT trigger refusal detection."""
        risk_report = (
            "Based on technical analysis, EURUSD shows bullish momentum. "
            "RSI at 62, MACD crossing above signal line. "
            "FINAL TRANSACTION PROPOSAL: BUY"
        )
        result = validate_decision(raw_decision="BUY", risk_report=risk_report, symbol="EURUSD")
        assert result == "BUY"

    def test_hold_decision_stays_hold(self) -> None:
        """HOLD from propagate() should stay HOLD even without refusal."""
        risk_report = "Market is ranging, no clear direction."
        result = validate_decision(raw_decision="HOLD", risk_report=risk_report, symbol="EURUSD")
        assert result == "HOLD"


# ── Edge cases ──────────────────────────────────────────────────────────────


class TestValidateDecisionEdgeCases:
    """Edge cases for decision validation."""

    def test_case_insensitive_proposal(self) -> None:
        risk_report = "Final Transaction Proposal: hold"
        result = validate_decision(raw_decision="BUY", risk_report=risk_report, symbol="EURUSD")
        assert result == "HOLD"

    def test_refusal_overrides_even_with_buy_proposal(self) -> None:
        """If refusal is detected, force HOLD even if there's a BUY proposal."""
        risk_report = "I cannot provide trading advice. FINAL TRANSACTION PROPOSAL: BUY"
        result = validate_decision(raw_decision="BUY", risk_report=risk_report, symbol="EURUSD")
        assert result == "HOLD"

    def test_non_standard_decision_becomes_hold(self) -> None:
        """Garbage decision values from propagate() should become HOLD."""
        result = validate_decision(raw_decision="MAYBE", risk_report="some text", symbol="EURUSD")
        assert result == "HOLD"

    def test_none_decision_becomes_hold(self) -> None:
        result = validate_decision(raw_decision=None, risk_report="some text", symbol="EURUSD")
        assert result == "HOLD"
