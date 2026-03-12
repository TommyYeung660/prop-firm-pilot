"""Tests for production log packer fallback summaries and INDEX generation."""

from pathlib import Path

from scripts import pack_prod_logs


def test_build_decisions_fallback_summary_reports_cooldown_and_follow_up() -> None:
    trade_content = "\n".join(
        [
            '{"type":"INTENT_CREATED","symbol":"EURUSD","intent_id":"i1","timestamp":"2026-03-11T01:00:00+00:00"}',
            (
                '{"type":"INTENT_CANCELLED","symbol":"EURUSD","intent_id":"i1",'
                '"reason":"LLM pre-filter: low confidence",'
                '"timestamp":"2026-03-11T01:01:00+00:00"}'
            ),
            '{"type":"SCANNER_SKIP","symbol":"EURUSD","reason":"low_confidence_cooldown","timestamp":"2026-03-11T01:02:00+00:00"}',
            '{"type":"TRADE_OPENED","symbol":"EURUSD","intent_id":"i2","timestamp":"2026-03-11T03:00:00+00:00"}',
            '{"type":"TRADE_CLOSED","symbol":"EURUSD","intent_id":"i2","pnl":12.5,"timestamp":"2026-03-11T05:00:00+00:00"}',
        ]
    )

    summary = pack_prod_logs._build_decisions_fallback_summary(trade_content)

    assert "INTENT_CREATED" in summary
    assert "INTENT_CANCELLED" in summary
    assert "SCANNER_SKIP" in summary
    assert "low_confidence_cooldown" in summary
    assert "LLM pre-filter: low confidence" in summary
    assert "12.50" in summary


def test_collect_summary_listing_only_includes_existing_files(tmp_path: Path) -> None:
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir()
    (summary_dir / "log_summary.md").write_text("log", encoding="utf-8")
    (summary_dir / "decisions_summary.md").write_text("decisions", encoding="utf-8")

    listing = pack_prod_logs._collect_summary_file_listing(summary_dir)

    assert any("summary/log_summary.md" in line for line in listing)
    assert any("summary/decisions_summary.md" in line for line in listing)
    assert all("summary/telegram_summary.md" not in line for line in listing)


def test_write_index_uses_actual_summary_listing(tmp_path: Path) -> None:
    index_path = tmp_path / "INDEX.md"
    summary_listing = [
        "- summary/log_summary.md (10 bytes)",
        "- summary/decisions_summary.md (20 bytes)",
    ]
    raw_listing = ["- raw/logs/prop_firm_pilot.log (100 bytes)"]

    pack_prod_logs._write_index(
        index_path=index_path,
        version="v1.4.1",
        date_range="2026-03-11 to 2026-03-12",
        timestamp="2026-03-12 06:00:00 UTC",
        summary_listing=summary_listing,
        raw_listing=raw_listing,
    )

    content = index_path.read_text(encoding="utf-8")
    assert "summary/log_summary.md" in content
    assert "summary/decisions_summary.md" in content
    assert "summary/telegram_summary.md" not in content
