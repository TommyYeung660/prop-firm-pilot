"""Tests for configuration defaults and validation."""

from src.config import AppConfig, ScannerConfig, SchedulerConfig


def test_scheduler_config_llm_worker_count_default():
    """v1.2.0: Default LLM worker count should be 2."""
    config = SchedulerConfig()
    assert config.llm_worker_count == 2


def test_scheduler_config_reeval_interval_default():
    """v1.2.0: Default reeval interval should be 2h (7200s)."""
    config = SchedulerConfig()
    assert config.reeval_interval_seconds == 7200


def test_e8_one_5k_session_aware_config():
    """e8_one_5k_challenge should have session-aware cadence enabled."""
    from src.config import load_config

    config = load_config("config/e8_one_5k_challenge.yaml")
    assert config.scheduler.session_aware_enabled is True
    assert config.scheduler.active_session_interval_seconds == 3600
    assert config.scheduler.volatility_trigger_enabled is True
    assert config.scheduler.volatility_threshold_pct == 0.3


def test_default_config_includes_macro_analyst():
    """v1.3.0: macro analyst should be enabled by default."""
    config = AppConfig()
    assert "macro" in config.agents.selected_analysts


def test_scanner_config_max_signal_age_days_default():
    """v1.3.0: Default max_signal_age_days should be 2."""
    config = ScannerConfig()
    assert config.max_signal_age_days == 2
