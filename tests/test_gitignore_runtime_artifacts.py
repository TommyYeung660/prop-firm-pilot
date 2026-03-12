"""Regression tests for ignored runtime artifacts."""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _check_ignore(path: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "check-ignore", "-v", path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("path", "expected_pattern"),
    [
        (".sisyphus/plans/example.md", ".sisyphus/"),
        ("data/decisions_e8_one_5k.db", "data/*"),
        ("data/optimization_state_e8_one_5k.json", "data/*"),
        ("data/trade_journal_e8_one_5k.jsonl", "data/*"),
        ("data/alpha158_fx_ic_ir_report.csv", "data/*"),
        ("prod_logs_20260312_v1.4.0/INDEX.md", "prod_logs_*/"),
        ("nunl", "nunl"),
    ],
)
def test_runtime_artifacts_are_ignored(path: str, expected_pattern: str) -> None:
    result = _check_ignore(path)

    assert result.returncode == 0, result.stderr or result.stdout
    assert expected_pattern in result.stdout


def test_source_tree_under_src_data_is_not_ignored() -> None:
    result = _check_ignore("src/data/fx_duckdb_store.py")

    assert result.returncode == 1, result.stdout
