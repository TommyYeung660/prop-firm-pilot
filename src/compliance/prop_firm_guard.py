"""
E8 Markets Prop Firm compliance engine — safety-critical module.

Enforces all E8 account rules before any trade is executed:
1. Daily drawdown limit (Soft Breach → Daily Pause)
2. Max drawdown limit (Hard Breach → account terminated)
   - Balance-based (E8 Signature $50k): fixed floor from initial balance
   - Dynamic / trailing (E8 Trial $5k): floor tracks equity high-water mark
3. 40% Best Day Rule (single day profit cap)
4. Position count limit
5. API request budget (2000/day)

ALL checks must pass before a trade is placed. One failure = trade rejected.
"""

import random
from datetime import datetime, timezone
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field

from src.config import ComplianceConfig, ExecutionConfig, InstrumentConfig

# ── Data Models ─────────────────────────────────────────────────────────────


class ComplianceResult(BaseModel):
    """Result of a compliance check."""

    passed: bool
    reason: str = ""
    rule_name: str = ""
    details: dict[str, Any] = {}


class TradePlan(BaseModel):
    """Proposed trade awaiting compliance approval."""

    symbol: str
    side: Literal["BUY", "SELL"]
    volume: float
    stop_loss: float
    take_profit: float
    risk_amount: float = Field(description="Max dollars at risk if SL is hit")


class AccountSnapshot(BaseModel):
    """Current account state for compliance evaluation."""

    balance: float
    equity: float
    margin: float = 0.0
    free_margin: float = 0.0
    day_start_balance: float = Field(description="Balance at the start of trading day")
    initial_balance: float = Field(description="Initial account balance ($50k for E8 Signature)")
    open_positions: int = 0
    daily_pnl: float = Field(default=0.0, description="Realized + unrealized PnL for today")
    total_pnl: float = Field(default=0.0, description="Total PnL since account start")
    equity_high_water_mark: float | None = Field(
        default=None,
        description=(
            "Highest equity ever reached. Required for dynamic drawdown accounts. "
            "If None, falls back to initial_balance for dynamic drawdown calculations."
        ),
    )


# ── PropFirmGuard ───────────────────────────────────────────────────────────


class PropFirmGuard:
    """E8 Markets compliance engine — supports Signature and Trial accounts.

    Usage:
        guard = PropFirmGuard(config, execution_config, instruments)
        result = guard.check_all(trade_plan, account_snapshot)
        if not result.passed:
            reject_trade(result.reason)
    """

    def __init__(
        self,
        config: ComplianceConfig,
        execution_config: ExecutionConfig,
        instruments: dict[str, InstrumentConfig],
    ) -> None:
        self._config = config
        self._exec_config = execution_config
        self._instruments = instruments
        self._daily_api_calls = 0
        self._daily_api_reset_date: str = ""

    # ── Main Check ──────────────────────────────────────────────────────

    def check_all(self, trade: TradePlan, account: AccountSnapshot) -> ComplianceResult:
        """Run ALL compliance checks. Returns first failure or PASS.

        Check order is deliberate — cheapest checks first, most critical last.
        """
        checks = [
            self.check_api_request_budget,
            lambda t, a: self.check_position_limit(a),
            self.check_daily_drawdown,
            self.check_max_drawdown,
            self.check_best_day_rule,
        ]

        for check_fn in checks:
            result = check_fn(trade, account)
            if not result.passed:
                logger.warning(
                    "Compliance REJECTED: {} — {} ({})",
                    result.rule_name,
                    result.reason,
                    trade.symbol,
                )
                return result

        logger.info("Compliance PASSED for {} {} {}", trade.side, trade.symbol, trade.volume)
        return ComplianceResult(passed=True, rule_name="ALL", reason="All checks passed")

    # ── Individual Checks ───────────────────────────────────────────────

    def check_daily_drawdown(self, trade: TradePlan, account: AccountSnapshot) -> ComplianceResult:
        """E8 rule: (day_start_balance - equity) must not exceed 5% of day_start_balance.

        Also checks if adding this trade's risk would breach the safety margin.
        """
        limit = self._config.daily_drawdown_limit
        stop_ratio = self._config.daily_drawdown_stop

        max_allowed_loss = account.day_start_balance * limit
        current_loss = max(0.0, account.day_start_balance - account.equity)
        projected_loss = current_loss + trade.risk_amount
        safe_limit = max_allowed_loss * stop_ratio

        if projected_loss >= safe_limit:
            return ComplianceResult(
                passed=False,
                rule_name="DAILY_DRAWDOWN",
                reason=(
                    f"Projected daily loss ${projected_loss:.2f} would exceed "
                    f"safety limit ${safe_limit:.2f} "
                    f"({stop_ratio:.0%} of ${max_allowed_loss:.2f} max)"
                ),
                details={
                    "current_loss": current_loss,
                    "trade_risk": trade.risk_amount,
                    "projected_loss": projected_loss,
                    "safe_limit": safe_limit,
                    "hard_limit": max_allowed_loss,
                },
            )

        return ComplianceResult(passed=True, rule_name="DAILY_DRAWDOWN")

    def check_max_drawdown(self, trade: TradePlan, account: AccountSnapshot) -> ComplianceResult:
        """E8 rule: max drawdown check — supports balance-based and dynamic (trailing HWM).

        Balance-based (E8 Signature $50k):
            Floor = initial_balance × (1 - max_drawdown_limit)
            Loss measured from initial_balance.

        Dynamic / Trailing (E8 Trial $5k):
            Floor = equity_high_water_mark × (1 - max_drawdown_limit)
            Loss measured from equity_high_water_mark. Floor only moves UP.
        """
        limit = self._config.max_drawdown_limit
        stop_ratio = self._config.max_drawdown_stop
        drawdown_type = self._config.drawdown_type

        # Determine reference point based on drawdown type
        if drawdown_type == "dynamic":
            # Dynamic: trailing high-water mark — floor rises with equity peaks
            reference = account.equity_high_water_mark or account.initial_balance
            rule_label = "MAX_DRAWDOWN_DYNAMIC"
        else:
            # Balance-based (default): fixed floor from initial balance
            reference = account.initial_balance
            rule_label = "MAX_DRAWDOWN"

        max_allowed_loss = reference * limit
        current_loss = max(0.0, reference - account.equity)
        projected_loss = current_loss + trade.risk_amount
        safe_limit = max_allowed_loss * stop_ratio

        if projected_loss >= safe_limit:
            return ComplianceResult(
                passed=False,
                rule_name=rule_label,
                reason=(
                    f"Projected total loss ${projected_loss:.2f} would exceed "
                    f"safety limit ${safe_limit:.2f} "
                    f"({stop_ratio:.0%} of ${max_allowed_loss:.2f} max, "
                    f"type={drawdown_type}, ref=${reference:.2f})"
                ),
                details={
                    "drawdown_type": drawdown_type,
                    "reference": reference,
                    "current_loss": current_loss,
                    "trade_risk": trade.risk_amount,
                    "projected_loss": projected_loss,
                    "safe_limit": safe_limit,
                    "hard_limit": max_allowed_loss,
                },
            )

        return ComplianceResult(passed=True, rule_name=rule_label)

    def check_best_day_rule(self, trade: TradePlan, account: AccountSnapshot) -> ComplianceResult:
        """E8 rule: daily PnL + potential profit must not exceed best_day_limit.

        best_day_limit = profit_target × initial_balance × best_day_ratio
        For $50k Signature: $50,000 × 8% × 40% = $1,600/day
        """
        best_day_limit = self._config.best_day_limit
        stop_ratio = self._config.best_day_stop
        safe_limit = best_day_limit * stop_ratio

        # Calculate potential profit from this trade (TP distance × volume × pip_value)
        potential_profit = self._estimate_potential_profit(trade)
        projected_daily_pnl = account.daily_pnl + potential_profit

        if projected_daily_pnl >= safe_limit:
            return ComplianceResult(
                passed=False,
                rule_name="BEST_DAY_RULE",
                reason=(
                    f"Projected daily PnL ${projected_daily_pnl:.2f} would exceed "
                    f"Best Day safety limit ${safe_limit:.2f} "
                    f"({stop_ratio:.0%} of ${best_day_limit:.2f})"
                ),
                details={
                    "current_daily_pnl": account.daily_pnl,
                    "potential_profit": potential_profit,
                    "projected_daily_pnl": projected_daily_pnl,
                    "safe_limit": safe_limit,
                    "hard_limit": best_day_limit,
                },
            )

        return ComplianceResult(passed=True, rule_name="BEST_DAY_RULE")

    def check_position_limit(
        self, account: AccountSnapshot, _trade: TradePlan | None = None
    ) -> ComplianceResult:
        """Check max concurrent positions."""
        max_pos = self._exec_config.max_positions

        if account.open_positions >= max_pos:
            return ComplianceResult(
                passed=False,
                rule_name="POSITION_LIMIT",
                reason=f"Already at max positions ({account.open_positions}/{max_pos})",
                details={"current": account.open_positions, "max": max_pos},
            )

        return ComplianceResult(passed=True, rule_name="POSITION_LIMIT")

    def check_api_request_budget(
        self, _trade: TradePlan | None = None, _account: AccountSnapshot | None = None
    ) -> ComplianceResult:
        """Check daily API request budget (2000/day, reserve 50 for emergencies)."""
        self._maybe_reset_daily_counters()

        limit = self._config.daily_api_request_limit
        reserve = 50
        remaining = limit - self._daily_api_calls

        if remaining <= reserve:
            return ComplianceResult(
                passed=False,
                rule_name="API_REQUEST_BUDGET",
                reason=(
                    f"API budget nearly exhausted ({self._daily_api_calls}/{limit} used, "
                    f"{remaining} remaining, {reserve} reserved)"
                ),
                details={
                    "used": self._daily_api_calls,
                    "limit": limit,
                    "remaining": remaining,
                },
            )

        return ComplianceResult(passed=True, rule_name="API_REQUEST_BUDGET")

    # ── Utility Methods ─────────────────────────────────────────────────

    def add_random_delay(self) -> float:
        """Return a random delay (seconds) to prevent duplicate strategy detection.

        The caller should await asyncio.sleep(delay).
        """
        delay = random.uniform(
            self._exec_config.random_delay_min,
            self._exec_config.random_delay_max,
        )
        logger.debug("PropFirmGuard: random delay {:.2f}s", delay)
        return delay

    def should_stop_trading_today(self, account: AccountSnapshot) -> bool:
        """Check if we should stop all new trades for today.

        Returns True if:
        - Daily PnL >= best_day_limit * best_day_stop (approaching Best Day cap)
        - Daily drawdown >= daily_drawdown_limit * daily_drawdown_stop (approaching DD cap)
        """
        # Best Day check
        best_day_limit = self._config.best_day_limit * self._config.best_day_stop
        if account.daily_pnl >= best_day_limit:
            logger.info(
                "PropFirmGuard: STOP trading — daily PnL ${:.2f} >= Best Day safety ${:.2f}",
                account.daily_pnl,
                best_day_limit,
            )
            return True

        # Daily drawdown check
        dd_limit = account.day_start_balance * self._config.daily_drawdown_limit
        dd_safe = dd_limit * self._config.daily_drawdown_stop
        current_loss = max(0.0, account.day_start_balance - account.equity)
        if current_loss >= dd_safe:
            logger.info(
                "PropFirmGuard: STOP trading — daily loss ${:.2f} >= DD safety ${:.2f}",
                current_loss,
                dd_safe,
            )
            return True

        return False

    def record_api_call(self) -> None:
        """Increment daily API call counter."""
        self._maybe_reset_daily_counters()
        self._daily_api_calls += 1

    def reset_daily_counters(self) -> None:
        """Explicitly reset daily counters (called at day start)."""
        self._daily_api_calls = 0
        self._daily_api_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        logger.debug("PropFirmGuard: daily counters reset")

    # ── Private ─────────────────────────────────────────────────────────

    def _maybe_reset_daily_counters(self) -> None:
        """Auto-reset daily counters if date has changed."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_api_reset_date:
            self._daily_api_calls = 0
            self._daily_api_reset_date = today

    def _estimate_potential_profit(self, trade: TradePlan) -> float:
        """Estimate potential profit from take profit pips.

        This is a rough estimate for Best Day Rule checking.
        """
        inst = self._instruments.get(trade.symbol)
        if inst is None:
            return trade.risk_amount * 2  # Assume 1:2 RR as fallback

        pip_value = inst.pip_value

        if trade.take_profit <= 0:
            return trade.risk_amount * 2

        # trade.take_profit is stored as pips in TradePlan
        tp_pips = trade.take_profit

        return tp_pips * pip_value * trade.volume
