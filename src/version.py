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


def get_app_version(pyproject_path: Path | None = None) -> str:
    """Return the semantic version declared in the project pyproject.toml."""
    path = pyproject_path or Path(__file__).resolve().parents[1] / "pyproject.toml"
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = _VERSION_PATTERN.match(raw_line.strip())
        if match:
            return match.group(1)
    raise RuntimeError(f"Unable to resolve project version from {path}")


def get_release_tag(pyproject_path: Path | None = None) -> str:
    """Return the version formatted as a release tag (e.g. v1.4.1)."""
    return f"v{get_app_version(pyproject_path)}"
