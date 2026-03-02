"""Tests for configuration defaults and validation (v1.2.0)."""

from src.config import SchedulerConfig


def test_scheduler_config_llm_worker_count_default():
    """v1.2.0: Default LLM worker count should be 2."""
    config = SchedulerConfig()
    assert config.llm_worker_count == 2


def test_scheduler_config_reeval_interval_default():
    """v1.2.0: Default reeval interval should be 2h (7200s)."""
    config = SchedulerConfig()
    assert config.reeval_interval_seconds == 7200
