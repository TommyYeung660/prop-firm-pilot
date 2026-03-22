"""Tests for configuration defaults and validation."""

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import AppConfig, ScannerConfig, SchedulerConfig


def _read_repo_file(relative_path: str) -> str:
    """Read a repository file as UTF-8 text for docs regression checks."""
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / relative_path).read_text(encoding="utf-8")


def test_scheduler_config_llm_worker_count_default():
    """v1.2.0: Default LLM worker count should be 2."""
    config = SchedulerConfig()
    assert config.llm_worker_count == 2


def test_scheduler_config_entry_funnel_mode_default() -> None:
    config = SchedulerConfig()
    assert config.entry_funnel_mode == "scanner_llm_tactical"


def test_scheduler_config_allows_ablation_modes() -> None:
    config = SchedulerConfig(
        entry_funnel_mode="scanner_tactical",
        ablation_shadow_modes=["scanner_llm_tactical", "no_trade"],
    )
    assert config.entry_funnel_mode == "scanner_tactical"
    assert config.ablation_shadow_modes == ["scanner_llm_tactical", "no_trade"]


def test_scanner_config_benchmark_default():
    """P1.5: Scanner benchmark should default to broad FX benchmark, not a symbol-specific value."""
    config = ScannerConfig()
    assert config.benchmark == "FX"


def test_e8_one_5k_scanner_benchmark_from_config():
    """P1.5: e8_one_5k_challenge should explicitly pin scanner benchmark."""
    from src.config import load_config

    config = load_config("config/e8_one_5k_challenge.yaml")
    assert config.scanner.benchmark == "FX"


def test_scanner_config_topk_short_default():
    """Side-aware short export should stay opt-in by default at the model level."""
    config = ScannerConfig()
    assert config.topk_short == 0


def test_e8_one_5k_scanner_topk_short_from_config():
    """e8_one_5k_challenge should explicitly activate one short-side candidate."""
    from src.config import load_config

    config = load_config("config/e8_one_5k_challenge.yaml")
    assert config.scanner.topk_short == 1


def test_scheduler_config_reeval_interval_default():
    """v1.2.0: Default reeval interval should be 2h (7200s)."""
    config = SchedulerConfig()
    assert config.reeval_interval_seconds == 7200


def test_e8_one_5k_session_aware_config():
    """e8_one_5k_challenge should have session-aware cadence enabled."""
    from src.config import load_config

    config = load_config("config/e8_one_5k_challenge.yaml")
    assert config.scheduler.session_aware_enabled is True
    assert config.scheduler.active_session_interval_seconds == 1800
    assert config.scheduler.volatility_trigger_enabled is True
    assert config.scheduler.volatility_threshold_pct == 0.5


def test_default_config_includes_macro_analyst():
    """v1.3.0: macro analyst should be enabled by default."""
    config = AppConfig()
    assert "macro" in config.agents.selected_analysts


def test_scanner_config_max_signal_age_days_default():
    """v1.3.0: Default max_signal_age_days should be 2."""
    config = ScannerConfig()
    assert config.max_signal_age_days == 2


def test_scheduler_config_v136_tuning_defaults():
    """v1.3.6: Verify tuned parameter defaults."""
    config = SchedulerConfig()
    assert config.llm_poll_interval_seconds == 10
    assert config.volatility_poll_interval_seconds == 15
    assert config.volatility_cooldown_seconds == 300
    assert config.volatility_threshold_pct == 0.2


def test_e8_one_5k_v136_tuned_params():
    """v1.3.6: e8_one_5k_challenge YAML should have tuned v1.3.6 values."""
    from src.config import load_config

    config = load_config("config/e8_one_5k_challenge.yaml")
    assert config.scheduler.llm_poll_interval_seconds == 10
    assert config.scheduler.volatility_poll_interval_seconds == 30
    assert config.scheduler.volatility_cooldown_seconds == 1800
    assert config.scheduler.volatility_threshold_pct == 0.5


def test_scheduler_threshold_override_defaults():
    """Task 1: Scheduler should expose default manual LLM threshold override config."""
    config = SchedulerConfig()
    override = config.llm_threshold_override
    assert override.enabled is False
    assert override.min_confidence == "medium"
    assert override.min_blended_confidence == 0.55


def test_e8_one_5k_threshold_override_from_config():
    """Task 1: e8_one_5k_challenge should enable manual LLM threshold override."""
    from src.config import load_config

    config = load_config("config/e8_one_5k_challenge.yaml")
    override = config.scheduler.llm_threshold_override
    assert override.enabled is True
    assert override.min_confidence == "medium"
    assert override.min_blended_confidence == 0.55


@pytest.mark.parametrize("invalid_value", [-0.01, 1.01])
def test_scheduler_threshold_override_rejects_out_of_range_blended_confidence(
    invalid_value: float,
) -> None:
    """Task 1 follow-up: min_blended_confidence must stay within 0.0..1.0."""
    with pytest.raises(ValidationError):
        SchedulerConfig(
            llm_threshold_override={
                "enabled": True,
                "min_confidence": "medium",
                "min_blended_confidence": invalid_value,
            }
        )


class TestTacticalConfig:
    """Verify TacticalConfig loads with defaults and from YAML."""

    def test_default_tactical_config(self) -> None:
        from src.config import AppConfig, TacticalConfig

        config = AppConfig()
        assert hasattr(config, "tactical")
        assert isinstance(config.tactical, TacticalConfig)
        assert config.tactical.enabled is True
        assert config.tactical.shadow_mode is True

    def test_tactical_hard_gates_defaults(self) -> None:
        from src.config import TacticalConfig

        tc = TacticalConfig()
        assert tc.hard_gates.spread_max_multiplier == 2.0
        assert tc.hard_gates.atr_min_ratio == 0.5
        assert tc.hard_gates.atr_max_ratio == 2.5
        assert tc.hard_gates.atr_period == 14
        assert tc.hard_gates.atr_timeframe == "1h"
        assert tc.hard_gates.data_max_age_seconds == 600

    def test_tactical_soft_gates_defaults(self) -> None:
        from src.config import TacticalConfig

        tc = TacticalConfig()
        assert tc.soft_gates.min_score == 2
        assert tc.soft_gates.ema_fast == 8
        assert tc.soft_gates.ema_slow == 21
        assert tc.soft_gates.rsi_period == 14
        assert tc.soft_gates.candle_min_body_ratio == 0.3

    def test_tactical_retry_defaults(self) -> None:
        from src.config import TacticalConfig

        tc = TacticalConfig()
        assert tc.retry.interval_seconds == 300
        assert tc.retry.max_retries == 12
        assert tc.retry.expire_action == "degrade"
        assert tc.retry.jitter_seconds == 10

    def test_tactical_cache_defaults(self) -> None:
        from src.config import TacticalConfig

        tc = TacticalConfig()
        assert tc.decision_cache.ttl_seconds == 14400

    def test_tactical_dedup_defaults(self) -> None:
        from src.config import TacticalConfig

        tc = TacticalConfig()
        assert tc.intent_dedup.cooldown_after_close_seconds == 1800

    def test_tactical_exit_defaults(self) -> None:
        from src.config import TacticalConfig

        tc = TacticalConfig()
        assert tc.exit.enabled is True
        assert tc.exit.evaluation_interval_seconds == 60
        assert tc.exit.breakeven_activation_r == 0.3
        assert tc.exit.partial_close_ratio == 0.5
        assert tc.exit.defensive_exit_loss_r == -0.35
        assert tc.exit.defensive_exit_require_strong_candle is True
        assert tc.exit.use_llm_exception_path is True


def test_breakeven_activation_pct_from_config():
    """v1.3.9: breakeven_activation_pct should be 0.3 in e8_one_5k_challenge."""
    from src.config import load_config

    config = load_config("config/e8_one_5k_challenge.yaml")
    assert config.scheduler.breakeven_activation_pct == 0.3


def test_config_loads_websocket_primary_market_data_block():
    """v1.4.0: websocket-first market data settings should load from YAML."""
    from src.config import load_config

    config = load_config("config/e8_one_5k_challenge.yaml")
    assert config.websocket.enabled is True
    assert config.websocket.use_as_primary_market_data is True
    assert "EURUSD" in config.websocket.symbols


def test_e8_one_5k_first_batch_jpy_crosses_are_enabled_everywhere():
    """JPY first batch must stay aligned across symbols, websocket, and instruments."""
    from src.config import load_config

    config = load_config("config/e8_one_5k_challenge.yaml")
    expected_symbols = [
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "AUDUSD",
        "NZDUSD",
        "USDCAD",
        "USDCHF",
        "EURJPY",
        "AUDJPY",
        "CADJPY",
    ]

    assert config.symbols == expected_symbols
    assert config.websocket.symbols == expected_symbols
    assert list(config.instruments.keys()) == expected_symbols
    assert config.instruments["EURJPY"].pip_size == 0.01
    assert config.instruments["AUDJPY"].pip_size == 0.01
    assert config.instruments["CADJPY"].pip_size == 0.01


@pytest.mark.parametrize("backend", ["matchtrader", "tradelocker"])
def test_execution_config_accepts_supported_broker_backend(backend: str) -> None:
    from src.config import ExecutionConfig

    config = ExecutionConfig(broker_backend=backend)
    assert config.broker_backend == backend


def test_execution_config_broker_backend_defaults_to_matchtrader() -> None:
    from src.config import ExecutionConfig

    config = ExecutionConfig()
    assert config.broker_backend == "matchtrader"


def test_app_config_loads_tradelocker_env_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADELOCKER_API_URL", "https://api.test-tradelocker.com")
    monkeypatch.setenv("TRADELOCKER_EMAIL", "tester@example.com")
    monkeypatch.setenv("TRADELOCKER_PASSWORD", "secret")
    monkeypatch.setenv("TRADELOCKER_SERVER", "demo")
    monkeypatch.setenv("TRADELOCKER_ACCOUNT_ID", "acct-001")

    from src.config import AppConfig

    config = AppConfig()
    assert config.tradelocker.api_url == "https://api.test-tradelocker.com"
    assert config.tradelocker.email == "tester@example.com"
    assert config.tradelocker.password == "secret"
    assert config.tradelocker.server == "demo"
    assert config.tradelocker.account_id == "acct-001"


def test_ops_runbook_covers_tradelocker_env_and_backend_startup() -> None:
    content = _read_repo_file("docs/ops_runbook.md")
    normalized = content.lower()

    for env_var in [
        "TRADELOCKER_API_URL",
        "TRADELOCKER_EMAIL",
        "TRADELOCKER_PASSWORD",
        "TRADELOCKER_SERVER",
        "TRADELOCKER_ACCOUNT_ID",
    ]:
        assert env_var in content

    assert "execution.broker_backend: tradelocker" in normalized
    assert "tradeLocker-first".lower() in normalized
    assert "known limitations" in normalized


def test_runbook_does_not_reference_nonexistent_literal_config_files() -> None:
    content = _read_repo_file("docs/ops_runbook.md")
    repo_root = Path(__file__).resolve().parents[1]
    referenced_paths = re.findall(r"config/[A-Za-z0-9_./-]+\.ya?ml", content)
    missing = [
        relative_path
        for relative_path in referenced_paths
        if not (repo_root / relative_path).exists()
    ]
    assert not missing, f"runbook references missing config file(s): {missing}"


def test_env_example_tradelocker_api_url_includes_backend_api_path() -> None:
    content = _read_repo_file(".env.example")
    match = re.search(r"^TRADELOCKER_API_URL=(?P<url>\S+)$", content, flags=re.MULTILINE)
    assert match is not None, "TRADELOCKER_API_URL must exist in .env.example"

    url = match.group("url")
    assert "/backend-api" in url, (
        "TRADELOCKER_API_URL in .env.example must include '/backend-api' "
        "to match runtime endpoint shape"
    )


def test_readme_mentions_backend_selection_and_tradelocker_path() -> None:
    content = _read_repo_file("README.md")
    normalized = content.lower()

    assert "broker_backend" in normalized
    assert "tradelocker" in normalized
    assert "matchtrader" in normalized
    assert "e8 signature" in normalized


def test_v150_roadmap_marks_tradelocker_runtime_integration_status() -> None:
    content = _read_repo_file("docs/PropFirmPilot_v1.5.0_road_map.md")
    normalized = content.lower()

    assert "tradelocker-first" in normalized
    assert "runtime startup" in normalized
    assert "task 5" in normalized
