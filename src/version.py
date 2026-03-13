"""
Shared application version helpers sourced from the repo pyproject.

Keeps runtime logs, scripts, and release tooling aligned to one version
definition without duplicating hardcoded strings across modules.

Usage:
    version = get_app_version()
    tag = get_release_tag()
"""

import re
from pathlib import Path

_VERSION_PATTERN = re.compile(r'^version\s*=\s*"([^"]+)"\s*$')
_DISPLAY_VERSION_PATTERN = re.compile(r'^display_version\s*=\s*"([^"]+)"\s*$')


def _read_declared_versions(path: Path) -> tuple[str | None, str | None]:
    """Read packaging and display versions from pyproject without extra deps."""
    packaging_version: str | None = None
    display_version: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if packaging_version is None:
            packaging_match = _VERSION_PATTERN.match(line)
            if packaging_match:
                packaging_version = packaging_match.group(1)
                continue
        if display_version is None:
            display_match = _DISPLAY_VERSION_PATTERN.match(line)
            if display_match:
                display_version = display_match.group(1)
    return packaging_version, display_version


def get_app_version(pyproject_path: Path | None = None) -> str:
    """Return the user-facing release version declared in pyproject.toml."""
    path = pyproject_path or Path(__file__).resolve().parents[1] / "pyproject.toml"
    packaging_version, display_version = _read_declared_versions(path)
    if display_version is not None:
        return display_version
    if packaging_version is not None:
        return packaging_version
    raise RuntimeError(f"Unable to resolve project version from {path}")


def get_release_tag(pyproject_path: Path | None = None) -> str:
    """Return the version formatted as a release tag (e.g. v1.4.1)."""
    return f"v{get_app_version(pyproject_path)}"
