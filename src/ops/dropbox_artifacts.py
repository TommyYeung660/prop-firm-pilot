"""
Dropbox artifact sync helpers for diagnostics bundles.

Provides a minimal wrapper around the Dropbox SDK for uploading, listing,
and downloading production diagnostics archives.

Usage:
    client = DropboxArtifactsClient()
    client.upload_file(Path("bundle.zip"), "/prop-firm-pilot/prod_logs/acct/bundle.zip")
"""

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import dropbox
from dropbox.exceptions import ApiError
from dropbox.files import (
    FileMetadata,
    FolderMetadata,
    GetMetadataError,
    LookupError,
    WriteMode,
)
from loguru import logger


@dataclass
class DropboxArtifactEntry:
    """Remote artifact metadata needed for latest-bundle selection."""

    name: str
    path_display: str
    server_modified: datetime


class DropboxArtifactsClient:
    """Thin Dropbox client for production artifact upload and retrieval.

    Usage:
        client = DropboxArtifactsClient()
        client.upload_file(local_path, remote_path)
    """

    def __init__(
        self,
        app_key: str | None = None,
        app_secret: str | None = None,
        refresh_token: str | None = None,
    ) -> None:
        self._app_key = app_key or os.getenv("DROPBOX_APP_KEY", "").strip()
        self._app_secret = app_secret or os.getenv("DROPBOX_APP_SECRET", "").strip()
        self._refresh_token = refresh_token or os.getenv("DROPBOX_REFRESH_TOKEN", "").strip()
        if not self._app_key or not self._app_secret or not self._refresh_token:
            raise RuntimeError("Dropbox credentials missing in environment")
        self._client = dropbox.Dropbox(
            oauth2_refresh_token=self._refresh_token,
            app_key=self._app_key,
            app_secret=self._app_secret,
        )

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        """Upload a local file to Dropbox, creating parent folders if needed."""
        self._ensure_remote_dir(str(Path(remote_path).parent).replace("\\", "/"))
        with local_path.open("rb") as handle:
            self._client.files_upload(
                handle.read(),
                remote_path,
                mode=WriteMode.overwrite,
            )
        logger.info("Dropbox: uploaded {} -> {}", local_path, remote_path)

    def list_zip_files(self, remote_dir: str) -> list[DropboxArtifactEntry]:
        """List zip artifacts in a Dropbox folder."""
        entries: list[DropboxArtifactEntry] = []
        result = self._client.files_list_folder(remote_dir)
        while True:
            for item in result.entries:
                if isinstance(item, FileMetadata) and item.name.endswith(".zip"):
                    entries.append(
                        DropboxArtifactEntry(
                            name=item.name,
                            path_display=item.path_display or f"{remote_dir}/{item.name}",
                            server_modified=item.server_modified,
                        )
                    )
            if not result.has_more:
                break
            result = self._client.files_list_folder_continue(result.cursor)
        return entries

    def download_file(self, remote_path: str, local_path: Path) -> None:
        """Download a Dropbox file to a local path."""
        _metadata, response = self._client.files_download(remote_path)
        local_path.write_bytes(response.content)
        logger.info("Dropbox: downloaded {} -> {}", remote_path, local_path)

    def _ensure_remote_dir(self, remote_dir: str) -> None:
        """Create the remote folder path if it does not already exist."""
        if not remote_dir or remote_dir == "/":
            return
        current = ""
        for part in [piece for piece in remote_dir.split("/") if piece]:
            current = f"{current}/{part}"
            try:
                metadata = self._client.files_get_metadata(current)
                if isinstance(metadata, FolderMetadata):
                    continue
            except ApiError as exc:
                if self._is_not_found(exc):
                    self._client.files_create_folder_v2(current)
                    continue
                raise

    @staticmethod
    def _is_not_found(exc: ApiError) -> bool:
        """Return True when the Dropbox API error represents a missing path."""
        error = getattr(exc, "error", None)
        if isinstance(error, GetMetadataError):
            if not error.is_path():
                return False
            lookup_error = error.get_path()
            return isinstance(lookup_error, LookupError) and lookup_error.is_not_found()
        return False
