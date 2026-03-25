import os
from pathlib import Path

from src.diagnostics.analyze_preview_bundle import _choose_main_log
from src.version import get_release_tag


def test_choose_main_log_prefers_current_release_tag_over_latest_generic_log(
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    release_log = log_dir / f"prop_firm_pilot_20260319_092004_{get_release_tag()}.log"
    generic_log = log_dir / "prop_firm_pilot.log"
    release_log.write_text("release", encoding="utf-8")
    generic_log.write_text("generic", encoding="utf-8")
    os.utime(release_log, (1_710_000_000, 1_710_000_000))
    os.utime(generic_log, (1_710_000_100, 1_710_000_100))

    chosen = _choose_main_log(log_dir)

    assert chosen == release_log
