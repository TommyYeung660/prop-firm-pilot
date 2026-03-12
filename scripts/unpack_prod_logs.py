"""
Download and unpack the latest production diagnostics bundle from Dropbox.

Usage:
    python scripts/unpack_prod_logs.py --config config/e8_one_5k_challenge.yaml
"""

import argparse
import shutil
import zipfile
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from scripts.pack_prod_logs import (
    _build_dropbox_bundle_dir,
    _load_merged_config,
    _resolve_account_name,
)
from src.ops.dropbox_artifacts import DropboxArtifactEntry, DropboxArtifactsClient


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and unpack latest prod bundle")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to account config YAML (merged with config/default.yaml)",
    )
    return parser.parse_args()


def _select_latest_bundle(entries: list[DropboxArtifactEntry]) -> DropboxArtifactEntry:
    """Return the latest bundle by Dropbox server_modified timestamp."""
    if not entries:
        raise RuntimeError("No prod bundle zip files found in Dropbox folder")
    return max(entries, key=lambda entry: entry.server_modified)


def _extract_bundle_zip(zip_path: Path, repo_root: Path) -> Path:
    """Extract a bundle zip into repo root, replacing an existing same-name folder."""
    target_dir = repo_root / zip_path.stem
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_dir)
    logger.info("Extracted bundle {} -> {}", zip_path, target_dir)
    return target_dir


def _download_latest_bundle(
    account_name: str,
    repo_root: Path,
    client: DropboxArtifactsClient | None = None,
) -> tuple[Path, Path]:
    """Download the latest Dropbox bundle for an account and extract it."""
    dropbox_client = client or DropboxArtifactsClient()
    remote_dir = _build_dropbox_bundle_dir(account_name)
    latest = _select_latest_bundle(dropbox_client.list_zip_files(remote_dir))
    local_zip = repo_root / latest.name
    dropbox_client.download_file(latest.path_display, local_zip)
    extracted_dir = _extract_bundle_zip(local_zip, repo_root)
    return local_zip, extracted_dir


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    load_dotenv()

    config_path = Path(args.config)
    config = _load_merged_config(config_path)
    account_name = _resolve_account_name(config, config_path)

    local_zip, extracted_dir = _download_latest_bundle(
        account_name=account_name,
        repo_root=Path(".").resolve(),
    )
    logger.info("Latest prod bundle downloaded to {}", local_zip)
    logger.info("Latest prod bundle extracted to {}", extracted_dir)


if __name__ == "__main__":
    main()
