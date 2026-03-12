"""Tests for production log packer fallback summaries and bundle metadata."""

import json
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
        version="v1.4.2",
        date_range="2026-03-11 to 2026-03-12",
        timestamp="2026-03-12 06:00:00 UTC",
        summary_listing=summary_listing,
        raw_listing=raw_listing,
    )

    content = index_path.read_text(encoding="utf-8")
    assert "summary/log_summary.md" in content
    assert "summary/decisions_summary.md" in content
    assert "summary/telegram_summary.md" not in content


def test_build_dropbox_bundle_dir_uses_account_name() -> None:
    remote_dir = pack_prod_logs._build_dropbox_bundle_dir("e8_one_5k_challenge")

    assert remote_dir == "/prop-firm-pilot/prod_logs/e8_one_5k_challenge"


def test_write_config_snapshots_writes_default_account_and_merged(tmp_path: Path) -> None:
    default_config_path = tmp_path / "config" / "default.yaml"
    account_config_path = tmp_path / "config" / "e8_one_5k_challenge.yaml"
    default_config_path.parent.mkdir(parents=True, exist_ok=True)
    default_config_path.write_text("symbols:\n  - EURUSD\n", encoding="utf-8")
    account_config_path.write_text("account_name: e8_one_5k_challenge\n", encoding="utf-8")

    raw_config_dir = tmp_path / "raw" / "config"
    merged_config = {
        "account_name": "e8_one_5k_challenge",
        "symbols": ["EURUSD", "GBPUSD"],
    }

    pack_prod_logs._write_config_snapshots(
        raw_config_dir=raw_config_dir,
        default_config_path=default_config_path,
        account_config_path=account_config_path,
        merged_config=merged_config,
    )

    assert (raw_config_dir / "default.yaml").exists()
    assert (raw_config_dir / "e8_one_5k_challenge.yaml").exists()
    merged_snapshot = raw_config_dir / "merged_config.yaml"
    assert merged_snapshot.exists()
    assert "GBPUSD" in merged_snapshot.read_text(encoding="utf-8")


def test_write_bundle_manifest_records_metadata_and_included_files(tmp_path: Path) -> None:
    manifest_path = tmp_path / "bundle_manifest.json"

    pack_prod_logs._write_bundle_manifest(
        manifest_path=manifest_path,
        account_name="e8_one_5k_challenge",
        config_path="config/e8_one_5k_challenge.yaml",
        version="v1.4.5a",
        app_version="1.4.5a",
        generated_at_utc="2026-03-13T10:00:00+00:00",
        days=7,
        date_range="2026-03-06 to 2026-03-13",
        bundle_folder="prod_logs_20260313_v1.4.5a",
        zip_name="prod_logs_20260313_v1.4.5a.zip",
        git_commit="abc123",
        git_branch="main",
        included_logs=["raw/logs/prop_firm_pilot_20260313_091530_v1.4.5a.log"],
        included_data_files=["raw/data/trade_journal_e8_one_5k.jsonl"],
        included_config_files=[
            "raw/config/default.yaml",
            "raw/config/e8_one_5k_challenge.yaml",
            "raw/config/merged_config.yaml",
        ],
        included_summary_files=["summary/log_summary.md"],
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["account_name"] == "e8_one_5k_challenge"
    assert payload["config_path"] == "config/e8_one_5k_challenge.yaml"
    assert payload["version"] == "v1.4.5a"
    assert payload["git_commit"] == "abc123"
    assert payload["included_logs"] == ["raw/logs/prop_firm_pilot_20260313_091530_v1.4.5a.log"]
    assert payload["included_config_files"] == [
        "raw/config/default.yaml",
        "raw/config/e8_one_5k_challenge.yaml",
        "raw/config/merged_config.yaml",
    ]


def test_upload_bundle_zip_uses_expected_dropbox_path(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, str] = {}
    zip_path = tmp_path / "prod_logs_20260313_v1.4.5a.zip"
    zip_path.write_text("zip", encoding="utf-8")

    class _FakeClient:
        def upload_file(self, local_path: Path, remote_path: str) -> None:
            captured["local_path"] = str(local_path)
            captured["remote_path"] = remote_path

    monkeypatch.setattr("scripts.pack_prod_logs.DropboxArtifactsClient", lambda: _FakeClient())

    remote_path = pack_prod_logs._upload_bundle_zip(
        zip_path=zip_path,
        account_name="e8_one_5k_challenge",
    )

    assert remote_path == (
        "/prop-firm-pilot/prod_logs/e8_one_5k_challenge/prod_logs_20260313_v1.4.5a.zip"
    )
    assert captured["local_path"] == str(zip_path)
    assert captured["remote_path"] == remote_path


def test_upload_bundle_zip_keeps_local_zip_when_upload_fails(tmp_path: Path, monkeypatch) -> None:
    zip_path = tmp_path / "prod_logs_20260313_v1.4.5a.zip"
    zip_path.write_text("zip", encoding="utf-8")

    class _FakeClient:
        def upload_file(self, local_path: Path, remote_path: str) -> None:
            raise RuntimeError("upload failed")

    monkeypatch.setattr("scripts.pack_prod_logs.DropboxArtifactsClient", lambda: _FakeClient())

    try:
        pack_prod_logs._upload_bundle_zip(
            zip_path=zip_path,
            account_name="e8_one_5k_challenge",
        )
    except RuntimeError as exc:
        assert str(exc) == "upload failed"
    else:
        raise AssertionError("Expected upload failure to propagate")

    assert zip_path.exists()


def test_load_log_content_uses_run_specific_logs_when_base_log_missing(tmp_path: Path) -> None:
    cutoff = pack_prod_logs.datetime(2000, 1, 1, 0, 0, tzinfo=pack_prod_logs.timezone.utc)
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    run_log = logs_dir / "prop_firm_pilot_20260313_091530_v1.4.5a.log"
    run_log.write_text("run-specific content", encoding="utf-8")

    content = pack_prod_logs._load_log_content(
        log_file=logs_dir / "prop_firm_pilot.log",
        cutoff=cutoff,
        max_chars=1000,
    )

    assert "run-specific content" in content
