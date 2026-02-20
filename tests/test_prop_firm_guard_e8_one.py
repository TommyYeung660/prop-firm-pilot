import pytest

from src.compliance.prop_firm_guard import AccountSnapshot, PropFirmGuard, TradePlan
from src.config import load_config


@pytest.fixture
def e8_one_config():
    return load_config("config/e8_one_5k_challenge.yaml")


@pytest.fixture
def e8_one_guard(e8_one_config):
    return PropFirmGuard(
        config=e8_one_config.compliance,
        execution_config=e8_one_config.execution,
        instruments=e8_one_config.instruments,
    )


def _snapshot(**kwargs) -> AccountSnapshot:
    default_kwargs = {
        "balance": 5000.0,
        "equity": 5000.0,
        "margin": 0.0,
        "free_margin": 5000.0,
        "day_start_balance": 5000.0,
        "initial_balance": 5000.0,
        "open_positions": 0,
        "daily_pnl": 0.0,
        "total_pnl": 0.0,
        "equity_high_water_mark": 5000.0,
    }
    default_kwargs.update(kwargs)
    return AccountSnapshot(**default_kwargs)


def _trade(risk: float = 25.0, tp_pips: float = 100.0) -> TradePlan:
    return TradePlan(
        symbol="EURUSD",
        side="BUY",
        volume=0.25,  # e.g. 0.25 lots = $25 at 10 pips SL
        stop_loss=1.0900,
        take_profit=tp_pips,
        risk_amount=risk,
    )


class TestE8OneConfig:
    def test_config_loads_correctly(self, e8_one_config):
        """Verify the e8_one_5k_challenge.yaml config matches the Dashboard limitations."""
        assert e8_one_config.account.initial_balance == 5000
        assert e8_one_config.compliance.daily_drawdown_limit == 0.04
        assert e8_one_config.compliance.max_drawdown_limit == 0.06
        assert e8_one_config.compliance.profit_target == 0.09
        assert e8_one_config.compliance.best_day_ratio == 0.40
        assert e8_one_config.compliance.best_day_limit == 180
        assert e8_one_config.compliance.drawdown_type == "dynamic"


class TestE8OneDrawdownRules:
    def test_daily_drawdown_passes(self, e8_one_guard):
        # 5000 * 4% * 85% = 200 * 0.85 = 170 maximum safe daily loss
        account = _snapshot(equity=4850.0, daily_pnl=-150.0)
        trade = _trade(risk=10.0)  # $10 risk. 150 + 10 = 160 < 170.
        result = e8_one_guard.check_daily_drawdown(trade, account)
        assert result.passed is True

    def test_daily_drawdown_rejects(self, e8_one_guard):
        account = _snapshot(equity=4840.0, daily_pnl=-160.0)
        trade = _trade(risk=20.0)  # 160 + 20 = 180 > 170 safe limit
        result = e8_one_guard.check_daily_drawdown(trade, account)
        assert result.passed is False
        assert result.rule_name == "DAILY_DRAWDOWN"

    def test_max_drawdown_passes(self, e8_one_guard):
        # 6% of HWM. If HWM is 5000, 6% is 300. Safe is 300 * 85% = 255.
        account = _snapshot(equity=4760.0, total_pnl=-240.0, equity_high_water_mark=5000.0)
        trade = _trade(risk=10.0)  # 240 + 10 = 250 < 255
        result = e8_one_guard.check_max_drawdown(trade, account)
        assert result.passed is True

    def test_max_drawdown_rejects(self, e8_one_guard):
        account = _snapshot(equity=4750.0, total_pnl=-250.0, equity_high_water_mark=5000.0)
        trade = _trade(risk=10.0)  # 250 + 10 = 260 > 255 safe limit
        result = e8_one_guard.check_max_drawdown(trade, account)
        assert result.passed is False
        assert result.rule_name == "MAX_DRAWDOWN_DYNAMIC"

    def test_trailing_drawdown_rejects_from_high_water_mark(self, e8_one_guard):
        # HWM = 5200. Max DD is 6% of 5000 (initial) = 300. Wait, is it 6% of initial or HWM?
        # Let's check config logic. It's usually 6% of initial. 5000 * 0.06 = 300.
        # So floor moves up: 5200 - 300 = 4900.
        # Safe limit is 85% of 300 = 255.
        # If HWM is 5200, floor is 4900, safe floor is 4945.
        account = _snapshot(equity=4950.0, total_pnl=-50.0, equity_high_water_mark=5200.0)
        trade = _trade(risk=20.0)  # Projected loss: 250 + 20 = 270 > 265.2 (Safe Limit)
        result = e8_one_guard.check_max_drawdown(trade, account)
        assert result.passed is False


class TestE8OneBestDayRule:
    def test_best_day_passes(self, e8_one_guard):
        # Limit is 180. Safe limit is 180 * 0.85 = 153.
        account = _snapshot(daily_pnl=100.0)
        trade = _trade(
            risk=20.0, tp_pips=50.0
        )  # profit = 50 * 10 * 0.25 = 125. 100+125 = 225 > 153!
        # wait, let's lower the tp_pips
        trade = _trade(risk=10.0, tp_pips=10.0)  # profit = 10 * 10 * 0.25 = 25. 100+25 = 125 < 153
        result = e8_one_guard.check_best_day_rule(trade, account)
        assert result.passed is True

    def test_best_day_rejects(self, e8_one_guard):
        account = _snapshot(daily_pnl=140.0)
        # Even a small trade puts it over 153
        trade = _trade(risk=10.0, tp_pips=10.0)  # profit = 25. 140+25 = 165 > 153
        result = e8_one_guard.check_best_day_rule(trade, account)
        assert result.passed is False
        assert result.rule_name == "BEST_DAY_RULE"
