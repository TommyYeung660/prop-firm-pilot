"""Integration test: verify tactical config loads from YAML correctly."""

from src.config import load_config


class TestTacticalConfigFromYAML:
    def test_e8_one_loads_tactical_config(self) -> None:
        config = load_config("config/e8_one_5k_challenge.yaml")
        assert config.tactical.enabled is True
        assert config.tactical.shadow_mode is False
        assert config.tactical.hard_gates.spread_max_multiplier == 2.0
        assert config.tactical.soft_gates.min_score == 2
        assert config.tactical.decision_cache.ttl_seconds == 14400
        assert config.tactical.intent_dedup.cooldown_after_close_seconds == 1800
        assert config.tactical.exit.enabled is True
        assert config.tactical.exit.modify_cooldown_seconds == 300
        assert config.tactical.exit.tp_reprice_cooldown_seconds == 600
