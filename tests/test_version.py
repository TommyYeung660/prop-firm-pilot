"""Tests for shared runtime version resolution and packer validation."""

from pathlib import Path

import pytest

from scripts import pack_prod_logs
from src.version import get_app_version, get_release_tag


def test_get_app_version_matches_pyproject() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    version_line = next(
        line
        for line in pyproject.read_text(encoding="utf-8").splitlines()
        if line.startswith("version")
    )
    expected = version_line.split("=", 1)[1].strip().strip('"')

    assert get_app_version() == expected
    assert get_release_tag() == f"v{expected}"


def test_pack_prod_logs_defaults_to_current_release_tag() -> None:
    assert pack_prod_logs._resolve_version(None) == get_release_tag()


def test_pack_prod_logs_rejects_mismatched_explicit_version() -> None:
    with pytest.raises(ValueError, match="does not match current project version"):
        pack_prod_logs._resolve_version("v9.9.9")
