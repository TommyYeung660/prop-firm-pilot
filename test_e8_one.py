import pytest
from src.config import load_config
from src.compliance.prop_firm_guard import (
    PropFirmGuard,
    AccountSnapshot,
    TradePlan,
    ComplianceResult,
)


def test_e8_one_dynamic_drawdown():
    config = load_config("config/e8_one_5k_challenge.yaml")
    guard = PropFirmGuard(config.compliance, config.execution, config.instruments)

    # 1. Initial State
    account = AccountSnapshot(
        balance=5000,
        equity=5000,
        day_start_balance=5000,
        initial_balance=5000,
        equity_high_water_mark=5000,
        daily_pnl=0,
        total_pnl=0,
    )

    trade = TradePlan(
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
        stop_loss=1.0500,
        take_profit=1.0520,
        risk_amount=10,
    )

    res = guard.check_all(trade, account)
    assert res.passed, f"Should pass initially: {res.reason}"

    # 2. Daily Pause Check (2% of 5000 = $100. Stop at 85% = $85)
    # Let's say we lost $80 today, risk is $10 -> $90 loss projected, which is > $85.
    account_dd = AccountSnapshot(
        balance=5000,
        equity=4920,
        day_start_balance=5000,
        initial_balance=5000,
        equity_high_water_mark=5000,
        daily_pnl=-80,
        total_pnl=-80,
    )
    res = guard.check_daily_drawdown(trade, account_dd)
    assert not res.passed, "Should fail daily drawdown stop limit"
    assert res.rule_name == "DAILY_DRAWDOWN"

    # 3. Dynamic Max Drawdown Check (6% of HWM. If HWM is 5200, floor is 5200 * (1 - 0.06) = 4888)
    # Stop at 85% of 6% = 5.1%
    account_hwm = AccountSnapshot(
        balance=5200,
        equity=4940,  # Loss from HWM is 5200 - 4940 = 260.
        day_start_balance=5200,
        initial_balance=5000,
        equity_high_water_mark=5200,  # HWM moved up
        daily_pnl=0,
        total_pnl=200,
    )
    # Max allowed loss from HWM = 5200 * 0.06 = 312.
    # Safe limit = 312 * 0.85 = 265.2.
    # Current loss = 260. Risk = 10 -> Projected = 270 > 265.2.
    res = guard.check_max_drawdown(trade, account_hwm)
    assert not res.passed, "Should fail dynamic max drawdown"
    assert res.rule_name == "MAX_DRAWDOWN_DYNAMIC"

    print("All tests passed!")


if __name__ == "__main__":
    test_e8_one_dynamic_drawdown()
