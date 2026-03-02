import os
import pytest
from unittest.mock import patch
from datetime import date


class TestSwitchover:
    def test_before_switchover_uses_alpha_vantage(self):
        """Before March 21, config should use alpha_vantage as primary."""
        with patch("src.decision.fx_analyst_config.date") as mock_date:
            mock_date.today.return_value = date(2026, 3, 15)
            mock_date.fromisoformat = date.fromisoformat
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            from src.decision.fx_analyst_config import build_agent_config

            config = build_agent_config()
            assert config["data_vendors"]["core_stock_apis"] == "alpha_vantage"
            assert config["data_vendors"]["news_data"] == "alpha_vantage"

    def test_after_switchover_uses_eodhd(self):
        """On or after March 21, config should use eodhd as primary."""
        with patch("src.decision.fx_analyst_config.date") as mock_date:
            mock_date.today.return_value = date(2026, 3, 21)
            mock_date.fromisoformat = date.fromisoformat
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            from src.decision.fx_analyst_config import build_agent_config

            config = build_agent_config()
            assert config["data_vendors"]["core_stock_apis"] == "eodhd"
            assert config["data_vendors"]["news_data"] == "eodhd"

    def test_env_override_switchover_date(self):
        """EODHD_SWITCHOVER_DATE env var should override default."""
        with patch.dict(os.environ, {"EODHD_SWITCHOVER_DATE": "2026-04-01"}, clear=False):
            with patch("src.decision.fx_analyst_config.date") as mock_date:
                mock_date.today.return_value = date(2026, 3, 25)
                mock_date.fromisoformat = date.fromisoformat
                mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
                from src.decision.fx_analyst_config import build_agent_config

                config = build_agent_config()
                # Still before 4/1, so should be AV
                assert config["data_vendors"]["core_stock_apis"] == "alpha_vantage"

    def test_env_force_eodhd(self):
        """EODHD_FORCE_PRIMARY=1 should always use EODHD regardless of date."""
        with patch.dict(os.environ, {"EODHD_FORCE_PRIMARY": "1"}, clear=False):
            from src.decision.fx_analyst_config import build_agent_config

            config = build_agent_config()
            assert config["data_vendors"]["core_stock_apis"] == "eodhd"
