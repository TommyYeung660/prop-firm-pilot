from datetime import datetime, timezone
from pathlib import Path

from src.config import AppConfig
from src.main import setup_logging


def test_build_run_log_path_embeds_timestamp_and_release_tag() -> None:
    from src.main import _build_run_log_path

    base_path = Path("logs/prop_firm_pilot.log")
    now = datetime(2026, 3, 13, 9, 15, 30, tzinfo=timezone.utc)

    path = _build_run_log_path(base_path, now=now, release_tag="v1.4.5a")

    assert path.name == "prop_firm_pilot_20260313_091530_v1.4.5a.log"
    assert path.parent == Path("logs")
    assert path != base_path


def test_setup_logging_uses_run_specific_log_path(tmp_path, monkeypatch) -> None:
    captured_sinks: list[object] = []

    def _fake_remove() -> None:
        return None

    def _fake_add(sink, **kwargs):  # noqa: ANN001, ANN003
        captured_sinks.append(sink)
        return len(captured_sinks)

    monkeypatch.setattr("src.main.logger.remove", _fake_remove)
    monkeypatch.setattr("src.main.logger.add", _fake_add)
    monkeypatch.setattr(
        "src.main._build_run_log_path",
        lambda base_path, now=None, release_tag=None: tmp_path
        / "logs"
        / "prop_firm_pilot_20260313_091530_v1.4.5a.log",
    )

    config = AppConfig()
    config.logging.file = str(tmp_path / "logs" / "prop_firm_pilot.log")

    setup_logging(config)

    assert len(captured_sinks) == 2
    assert captured_sinks[1] == tmp_path / "logs" / "prop_firm_pilot_20260313_091530_v1.4.5a.log"
