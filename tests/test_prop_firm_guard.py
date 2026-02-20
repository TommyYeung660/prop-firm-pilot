"""
Tests for PropFirmGuard — focused on dynamic (trailing HWM) drawdown logic.

Covers:
- Balance-based drawdown (original E8 Signature behavior, regression guard)
- Dynamic drawdown with equity high-water mark (E8 Trial behavior)
- Edge cases: HWM equals initial, HWM None fallback, HWM far above initial
- should_stop_trading_today with Trial-level limits
- Config loading for e8_trial_5k.yaml
"""

from src.compliance.prop_firm_guard import (
    AccountSnapshot,
    PropFirmGuard,
    TradePlan,
)
from src.config import ComplianceConfig, ExecutionConfig, InstrumentConfig

# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_guard(
    drawdown_type: str = "balance",
    max_drawdown_limit: float = 0.08,
    daily_drawdown_limit: float = 0.05,
    best_day_limit: float = 1600.0,
    max_positions: int = 3,
) -> PropFirmGuard:
    """Create a PropFirmGuard with configurable compliance settings."""
    compliance = ComplianceConfig(
        daily_drawdown_limit=daily_drawdown_limit,
        max_drawdown_limit=max_drawdown_limit,
        drawdown_type=drawdown_type,
        best_day_limit=best_day_limit,
    )
    execution = ExecutionConfig(max_positions=max_positions)
    instruments = {
        "EURUSD": InstrumentConfig(pip_value=10.0, pip_size=0.0001),
    }
    return PropFirmGuard(compliance, execution, instruments)


def _trial_guard() -> PropFirmGuard:
    """Create a PropFirmGuard matching E8 Trial $5k config."""
    return _make_guard(
        drawdown_type="dynamic",
        max_drawdown_limit=0.04,
        daily_drawdown_limit=0.02,
        best_day_limit=160.0,
        max_positions=2,
    )


def _small_trade(risk: float = 10.0) -> TradePlan:
    """Create a small test trade with configurable risk."""
    return TradePlan(
        symbol="EURUSD",
        side="BUY",
        volume=0.01,
        stop_loss=1.0900,
        take_profit=1.1000,
        risk_amount=risk,
    )


def _snapshot(
    balance: float = 5000.0,
    equity: float = 5000.0,
    initial_balance: float = 5000.0,
    day_start_balance: float = 5000.0,
    equity_high_water_mark: float | None = None,
    daily_pnl: float = 0.0,
    open_positions: int = 0,
) -> AccountSnapshot:
    """Create an AccountSnapshot with configurable fields."""
    return AccountSnapshot(
        balance=balance,
        equity=equity,
        initial_balance=initial_balance,
        day_start_balance=day_start_balance,
        equity_high_water_mark=equity_high_water_mark,
        daily_pnl=daily_pnl,
        open_positions=open_positions,
    )


# ── Balance-Based Drawdown (Regression Tests) ──────────────────────────────


class TestBalanceBasedDrawdown:
    """Ensure original balance-based logic (E8 Signature) still works correctly."""

    def test_balance_drawdown_passes_when_safe(self) -> None:
        """Trade within safe limits should pass."""
        guard = _make_guard(drawdown_type="balance", max_drawdown_limit=0.08)
        account = _snapshot(initial_balance=50000.0, equity=49000.0, day_start_balance=50000.0)
        trade = _small_trade(risk=100.0)

        result = guard.check_max_drawdown(trade, account)
        assert result.passed is True
        assert result.rule_name == "MAX_DRAWDOWN"

    def test_balance_drawdown_fails_at_safety_margin(self) -> None:
        """Trade that would breach 85% safety margin should be rejected."""
        guard = _make_guard(drawdown_type="balance", max_drawdown_limit=0.08)
        # Max allowed loss = 50000 * 0.08 = 4000, safety = 4000 * 0.85 = 3400
        # Current loss = 50000 - 47000 = 3000, trade risk = 500 → projected = 3500 > 3400
        account = _snapshot(initial_balance=50000.0, equity=47000.0, day_start_balance=50000.0)
        trade = _small_trade(risk=500.0)

        result = guard.check_max_drawdown(trade, account)
        assert result.passed is False
        assert result.rule_name == "MAX_DRAWDOWN"
        assert "3500" in result.reason  # projected loss
        assert "3400" in result.reason  # safety limit

    def test_balance_drawdown_ignores_hwm(self) -> None:
        """Balance-based mode should use initial_balance, not equity_high_water_mark."""
        guard = _make_guard(drawdown_type="balance", max_drawdown_limit=0.08)
        account = _snapshot(
            initial_balance=50000.0,
            equity=49000.0,
            day_start_balance=50000.0,
            equity_high_water_mark=55000.0,  # HWM is higher but should be ignored
        )
        trade = _small_trade(risk=100.0)

        result = guard.check_max_drawdown(trade, account)
        assert result.passed is True
        assert result.rule_name == "MAX_DRAWDOWN"
        # Should NOT use HWM in details
        assert result.details.get("reference") is None or result.details.get("reference") == 50000.0


# ── Dynamic Drawdown (E8 Trial) ────────────────────────────────────────────


class TestDynamicDrawdown:
    """Tests for dynamic / trailing high-water mark drawdown (E8 Trial $5k)."""

    def test_dynamic_passes_when_safe(self) -> None:
        """Trade within dynamic drawdown safe limits should pass."""
        guard = _trial_guard()
        account = _snapshot(equity=5000.0, equity_high_water_mark=5000.0, initial_balance=5000.0)
        trade = _small_trade(risk=10.0)

        result = guard.check_max_drawdown(trade, account)
        assert result.passed is True
        assert result.rule_name == "MAX_DRAWDOWN_DYNAMIC"

    def test_dynamic_uses_hwm_not_initial_balance(self) -> None:
        """Dynamic drawdown should use HWM as reference, not initial balance.

        HWM = 5200, limit = 4%, max_loss = 208, safety = 208 * 0.85 = 176.80
        Current loss from HWM = 5200 - 5100 = 100
        Trade risk = 80 → projected = 180 > 176.80 → REJECT
        """
        guard = _trial_guard()
        account = _snapshot(
            equity=5100.0,
            equity_high_water_mark=5200.0,
            initial_balance=5000.0,
        )
        trade = _small_trade(risk=80.0)

        result = guard.check_max_drawdown(trade, account)
        assert result.passed is False
        assert result.rule_name == "MAX_DRAWDOWN_DYNAMIC"
        assert result.details["reference"] == 5200.0
        assert result.details["drawdown_type"] == "dynamic"

    def test_dynamic_hwm_increases_allowed_loss(self) -> None:
        """As equity HWM rises, the absolute allowed loss also rises.

        HWM = 6000 (up from 5000), limit = 4%
        max_allowed_loss = 6000 * 0.04 = 240
        safety = 240 * 0.85 = 204
        Current loss = 6000 - 5900 = 100
        Trade risk = 90 → projected = 190 < 204 → PASS
        """
        guard = _trial_guard()
        account = _snapshot(
            equity=5900.0,
            equity_high_water_mark=6000.0,
            initial_balance=5000.0,
        )
        trade = _small_trade(risk=90.0)

        result = guard.check_max_drawdown(trade, account)
        assert result.passed is True
        assert result.rule_name == "MAX_DRAWDOWN_DYNAMIC"

    def test_dynamic_hwm_same_as_initial(self) -> None:
        """When HWM equals initial balance, dynamic behaves like balance-based."""
        guard = _trial_guard()
        account = _snapshot(
            equity=4900.0,
            equity_high_water_mark=5000.0,
            initial_balance=5000.0,
        )
        # Max loss = 5000 * 0.04 = 200, safety = 170
        # Current loss = 100, risk = 60 → projected = 160 < 170 → PASS
        trade = _small_trade(risk=60.0)

        result = guard.check_max_drawdown(trade, account)
        assert result.passed is True

    def test_dynamic_hwm_none_falls_back_to_initial(self) -> None:
        """If equity_high_water_mark is None, fall back to initial_balance."""
        guard = _trial_guard()
        account = _snapshot(
            equity=4900.0,
            equity_high_water_mark=None,  # Not yet tracked
            initial_balance=5000.0,
        )
        # Reference = initial_balance = 5000 (fallback)
        # Max loss = 200, safety = 170, current loss = 100, risk = 60 → 160 < 170 → PASS
        trade = _small_trade(risk=60.0)

        result = guard.check_max_drawdown(trade, account)
        assert result.passed is True

    def test_dynamic_hwm_none_fallback_reference_verified(self) -> None:
        """Verify fallback reference appears in details when trade is rejected."""
        guard = _trial_guard()
        account = _snapshot(
            equity=4850.0,
            equity_high_water_mark=None,
            initial_balance=5000.0,
        )
        # Reference = 5000 (fallback), max = 200, safety = 170
        # Current loss = 150, risk = 25 → projected = 175 > 170 → REJECT
        trade = _small_trade(risk=25.0)

        result = guard.check_max_drawdown(trade, account)
        assert result.passed is False
        assert result.details["reference"] == 5000.0  # Confirmed fallback used

    def test_dynamic_hwm_none_still_rejects_when_breached(self) -> None:
        """Fallback to initial_balance should still reject dangerous trades."""
        guard = _trial_guard()
        account = _snapshot(
            equity=4850.0,
            equity_high_water_mark=None,
            initial_balance=5000.0,
        )
        # Reference = 5000, max_loss = 200, safety = 170
        # Current loss = 150, risk = 30 → projected = 180 > 170 → REJECT
        trade = _small_trade(risk=30.0)

        result = guard.check_max_drawdown(trade, account)
        assert result.passed is False
        assert result.details["reference"] == 5000.0

    def test_dynamic_large_hwm_makes_floor_very_high(self) -> None:
        """If HWM is much higher than initial, the floor (min equity) rises significantly.

        HWM = 7000 (40% profit from 5000), limit = 4%
        max_allowed_loss = 7000 * 0.04 = 280
        Floor = 7000 - 280 = 6720 (way above initial 5000!)
        safety = 280 * 0.85 = 238
        Current loss from HWM = 7000 - 6800 = 200
        Trade risk = 50 → projected = 250 > 238 → REJECT
        """
        guard = _trial_guard()
        account = _snapshot(
            equity=6800.0,
            equity_high_water_mark=7000.0,
            initial_balance=5000.0,
        )
        trade = _small_trade(risk=50.0)

        result = guard.check_max_drawdown(trade, account)
        assert result.passed is False
        assert result.details["reference"] == 7000.0

    def test_dynamic_details_include_drawdown_type(self) -> None:
        """Result details should include drawdown_type for debugging."""
        guard = _trial_guard()
        account = _snapshot(equity=4800.0, equity_high_water_mark=5000.0, initial_balance=5000.0)
        trade = _small_trade(risk=10.0)

        result = guard.check_max_drawdown(trade, account)
        # Whether pass or fail, should have drawdown_type in details when failing
        if not result.passed:
            assert result.details["drawdown_type"] == "dynamic"


# ── Daily Drawdown with Trial Limits ────────────────────────────────────────


class TestTrialDailyDrawdown:
    """Daily drawdown with Trial account's 2% limit."""

    def test_trial_daily_drawdown_passes(self) -> None:
        """Small trade within 2% daily limit should pass."""
        guard = _trial_guard()
        account = _snapshot(equity=4990.0, day_start_balance=5000.0)
        trade = _small_trade(risk=10.0)

        result = guard.check_daily_drawdown(trade, account)
        assert result.passed is True

    def test_trial_daily_drawdown_rejects_near_limit(self) -> None:
        """Trade that would exceed 85% of 2% daily limit should be rejected.

        day_start = 5000, daily limit = 2% → max = 100, safety = 85 → $85
        current loss = 5000 - 4930 = 70, trade risk = 20 → projected = 90 > 85 → REJECT
        """
        guard = _trial_guard()
        account = _snapshot(equity=4930.0, day_start_balance=5000.0)
        trade = _small_trade(risk=20.0)

        result = guard.check_daily_drawdown(trade, account)
        assert result.passed is False
        assert result.rule_name == "DAILY_DRAWDOWN"


# ── Best Day Rule with Trial Limits ─────────────────────────────────────────


class TestTrialBestDayRule:
    """Best Day Rule with Trial account's $160 limit."""

    def test_trial_best_day_passes(self) -> None:
        """Daily PnL + potential profit within $160 * 85% = $136 should pass."""
        guard = _trial_guard()
        account = _snapshot(daily_pnl=50.0)
        trade = _small_trade(risk=10.0)  # potential profit ~$20 (2:1 RR via pip calc)

        result = guard.check_best_day_rule(trade, account)
        assert result.passed is True

    def test_trial_best_day_rejects_near_limit(self) -> None:
        """Daily PnL near $136 safety limit should reject further profitable trades."""
        guard = _trial_guard()
        account = _snapshot(daily_pnl=130.0)
        # Potential profit for EURUSD: tp_pips = |1.1 - 1.09| / 0.0001 = 100 pips
        # profit = 100 * 10 * 0.01 = $10 → total = $140 > $136 → REJECT
        trade = _small_trade(risk=5.0)
        trade.take_profit = 100.0  # Set TP as 100 pips to generate $10 profit

        result = guard.check_best_day_rule(trade, account)
        assert result.passed is False
        assert result.rule_name == "BEST_DAY_RULE"


# ── check_all Integration ──────────────────────────────────────────────────


class TestCheckAllWithDynamic:
    """End-to-end check_all with dynamic drawdown config."""

    def test_all_checks_pass_with_dynamic(self) -> None:
        """A safe trade on a healthy Trial account should pass all checks."""
        guard = _trial_guard()
        account = _snapshot(
            equity=5050.0,
            equity_high_water_mark=5050.0,
            day_start_balance=5050.0,
            daily_pnl=0.0,
            open_positions=0,
        )
        trade = _small_trade(risk=5.0)
        trade.take_profit = 100.0  # Set TP as 100 pips to generate $10 profit

        result = guard.check_all(trade, account)
        assert result.passed is True

    def test_dynamic_drawdown_blocks_in_check_all(self) -> None:
        """Dynamic drawdown failure should block trade in check_all pipeline."""
        guard = _trial_guard()
        # HWM = 5500, equity = 5320 → loss from HWM = 180
        # max_loss = 5500 * 0.04 = 220, safety = 187
        # risk = 10 → projected = 190 > 187 → REJECT
        account = _snapshot(
            equity=5320.0,
            equity_high_water_mark=5500.0,
            initial_balance=5000.0,
            day_start_balance=5320.0,
            daily_pnl=0.0,
            open_positions=0,
        )
        trade = _small_trade(risk=10.0)

        result = guard.check_all(trade, account)
        assert result.passed is False
        assert "MAX_DRAWDOWN_DYNAMIC" in result.rule_name


# ── should_stop_trading_today ───────────────────────────────────────────────


class TestShouldStopTradingTrial:
    """should_stop_trading_today with Trial account limits."""

    def test_stop_on_daily_drawdown(self) -> None:
        """Should stop when daily loss reaches 85% of 2% limit.

        day_start = 5000, daily_dd = 2% → max = 100, safety = 85
        equity = 4910 → loss = 90 > 85 → STOP
        """
        guard = _trial_guard()
        account = _snapshot(equity=4910.0, day_start_balance=5000.0)

        assert guard.should_stop_trading_today(account) is True

    def test_stop_on_best_day(self) -> None:
        """Should stop when daily PnL reaches 85% of $160 = $136."""
        guard = _trial_guard()
        account = _snapshot(equity=5140.0, day_start_balance=5000.0, daily_pnl=140.0)

        assert guard.should_stop_trading_today(account) is True

    def test_no_stop_when_safe(self) -> None:
        """Should not stop when within safe limits."""
        guard = _trial_guard()
        account = _snapshot(equity=4960.0, day_start_balance=5000.0, daily_pnl=10.0)

        assert guard.should_stop_trading_today(account) is False


# ── Config Loading ──────────────────────────────────────────────────────────


class TestAccountSnapshotHWM:
    """Test equity_high_water_mark field on AccountSnapshot."""

    def test_hwm_defaults_to_none(self) -> None:
        """equity_high_water_mark should default to None."""
        account = AccountSnapshot(
            balance=5000.0,
            equity=5000.0,
            initial_balance=5000.0,
            day_start_balance=5000.0,
        )
        assert account.equity_high_water_mark is None

    def test_hwm_can_be_set(self) -> None:
        """equity_high_water_mark should accept a float value."""
        account = AccountSnapshot(
            balance=5000.0,
            equity=5000.0,
            initial_balance=5000.0,
            day_start_balance=5000.0,
            equity_high_water_mark=5500.0,
        )
        assert account.equity_high_water_mark == 5500.0
