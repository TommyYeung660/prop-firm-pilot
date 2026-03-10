"""
Tests for LLM structured risk field extraction (P2.7).

Validates:
1. RiskMeta fields are extracted from risk_report text
2. entry_style extraction (e.g. "breakout", "pullback", "momentum")
3. avoid_zone extraction (price ranges)
4. trigger_zone extraction
5. invalid_if extraction (invalidation conditions)
6. max_same_day_attempts extraction
7. Graceful fallback when fields not present
"""

from src.decision.agent_bridge import RiskMeta, extract_risk_meta


class TestExtractRiskMeta:
    """Tests for extract_risk_meta() parsing."""

    def test_extract_entry_style(self) -> None:
        report = "ENTRY STYLE: pullback\nSome other text"
        meta = extract_risk_meta(report)
        assert meta.entry_style == "pullback"

    def test_extract_avoid_zone(self) -> None:
        report = "AVOID ZONE: 1.0750 - 1.0780\nMore text"
        meta = extract_risk_meta(report)
        assert meta.avoid_zone == "1.0750 - 1.0780"

    def test_extract_trigger_zone(self) -> None:
        report = "TRIGGER ZONE: 1.0820 - 1.0850"
        meta = extract_risk_meta(report)
        assert meta.trigger_zone == "1.0820 - 1.0850"

    def test_extract_invalid_if(self) -> None:
        report = "INVALID IF: price breaks above 1.0900"
        meta = extract_risk_meta(report)
        assert meta.invalid_if == "price breaks above 1.0900"

    def test_extract_max_attempts(self) -> None:
        report = "MAX SAME DAY ATTEMPTS: 1"
        meta = extract_risk_meta(report)
        assert meta.max_same_day_attempts == 1

    def test_all_fields_present(self) -> None:
        report = (
            "ENTRY STYLE: momentum\n"
            "AVOID ZONE: 1.0750 - 1.0780\n"
            "TRIGGER ZONE: 1.0820 - 1.0850\n"
            "INVALID IF: price breaks above 1.0900\n"
            "MAX SAME DAY ATTEMPTS: 2\n"
            "FINAL TRANSACTION PROPOSAL: SELL"
        )
        meta = extract_risk_meta(report)
        assert meta.entry_style == "momentum"
        assert meta.avoid_zone == "1.0750 - 1.0780"
        assert meta.trigger_zone == "1.0820 - 1.0850"
        assert meta.invalid_if == "price breaks above 1.0900"
        assert meta.max_same_day_attempts == 2

    def test_empty_report(self) -> None:
        meta = extract_risk_meta("")
        assert meta.entry_style is None
        assert meta.avoid_zone is None
        assert meta.trigger_zone is None
        assert meta.invalid_if is None
        assert meta.max_same_day_attempts is None

    def test_garbage_report(self) -> None:
        meta = extract_risk_meta("Random text with no structured fields at all")
        assert meta.entry_style is None

    def test_case_insensitive(self) -> None:
        report = "entry style: Breakout\navoid zone: 1.0750-1.0780"
        meta = extract_risk_meta(report)
        assert meta.entry_style == "Breakout"
        assert meta.avoid_zone == "1.0750-1.0780"


class TestRiskMetaModel:
    """Tests for RiskMeta Pydantic model."""

    def test_default_construction(self) -> None:
        meta = RiskMeta()
        assert meta.entry_style is None
        assert meta.avoid_zone is None
        assert meta.trigger_zone is None
        assert meta.invalid_if is None
        assert meta.max_same_day_attempts is None

    def test_partial_construction(self) -> None:
        meta = RiskMeta(entry_style="pullback", max_same_day_attempts=2)
        assert meta.entry_style == "pullback"
        assert meta.avoid_zone is None
        assert meta.max_same_day_attempts == 2
