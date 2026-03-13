import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from src.ops.dropbox_artifacts import DropboxArtifactEntry


def test_unpack_script_runs_via_direct_path_invocation() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "unpack_prod_logs.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Download and unpack latest prod bundle" in result.stdout


def test_select_latest_bundle_uses_server_modified() -> None:
    from scripts.unpack_prod_logs import _select_latest_bundle

    entries = [
        DropboxArtifactEntry(
            name="prod_logs_20260312_v1.4.5.zip",
            path_display="/prop-firm-pilot/prod_logs/e8/prod_logs_20260312_v1.4.5.zip",
            server_modified=datetime(2026, 3, 12, 10, 0, tzinfo=timezone.utc),
        ),
        DropboxArtifactEntry(
            name="prod_logs_20260313_v1.4.5a.zip",
            path_display="/prop-firm-pilot/prod_logs/e8/prod_logs_20260313_v1.4.5a.zip",
            server_modified=datetime(2026, 3, 13, 10, 0, tzinfo=timezone.utc),
        ),
    ]

    latest = _select_latest_bundle(entries)

    assert latest.name == "prod_logs_20260313_v1.4.5a.zip"


def test_extract_bundle_zip_replaces_existing_directory(tmp_path: Path) -> None:
    from scripts.unpack_prod_logs import _extract_bundle_zip

    zip_path = tmp_path / "prod_logs_20260313_v1.4.5a.zip"
    target_dir = tmp_path / "prod_logs_20260313_v1.4.5a"
    target_dir.mkdir()
    (target_dir / "stale.txt").write_text("old", encoding="utf-8")

    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("INDEX.md", "fresh-index")
        archive.writestr("raw/logs/run.log", "fresh-log")

    extracted_dir = _extract_bundle_zip(zip_path=zip_path, repo_root=tmp_path)

    assert extracted_dir == target_dir
    assert not (target_dir / "stale.txt").exists()
    assert (target_dir / "INDEX.md").read_text(encoding="utf-8") == "fresh-index"
    assert (target_dir / "raw" / "logs" / "run.log").read_text(encoding="utf-8") == "fresh-log"


def test_download_latest_bundle_downloads_to_repo_root_and_extracts(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts.unpack_prod_logs import _download_latest_bundle

    archive_name = "prod_logs_20260313_v1.4.5a.zip"
    remote_path = f"/prop-firm-pilot/prod_logs/e8_one_5k_challenge/{archive_name}"

    class _FakeClient:
        def list_zip_files(self, remote_dir: str) -> list[DropboxArtifactEntry]:
            assert remote_dir == "/prop-firm-pilot/prod_logs/e8_one_5k_challenge"
            return [
                DropboxArtifactEntry(
                    name=archive_name,
                    path_display=remote_path,
                    server_modified=datetime(2026, 3, 13, 10, 0, tzinfo=timezone.utc),
                )
            ]

        def download_file(self, source_remote_path: str, local_path: Path) -> None:
            assert source_remote_path == remote_path
            with zipfile.ZipFile(local_path, "w") as archive:
                archive.writestr("INDEX.md", "fresh-index")

    monkeypatch.setattr("scripts.unpack_prod_logs.DropboxArtifactsClient", lambda: _FakeClient())

    local_zip, extracted_dir = _download_latest_bundle(
        account_name="e8_one_5k_challenge",
        repo_root=tmp_path,
    )

    assert local_zip == tmp_path / archive_name
    assert extracted_dir == tmp_path / archive_name.replace(".zip", "")
    assert extracted_dir.joinpath("INDEX.md").read_text(encoding="utf-8") == "fresh-index"
