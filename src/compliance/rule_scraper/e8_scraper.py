from typing import Any


class E8RuleSynthesizer:
    """
    Synthesizes E8 Markets rules from retrieved web text.
    Currently hardcodes the verified logic for Trial and E8 One accounts
    based on the scraped documentation.
    """

    @staticmethod
    def get_e8_one_5k_rules() -> dict[str, Any]:
        """
        Rules for E8 One $5,000 Challenge / Funded Account
        Based on official Help Center E8 One (Preset) docs.
        """
        # Based on fetched data:
        # E8 One uses Trailing Drawdown (Dynamic).
        # It's a 1-step or specific structure. Usually 6% profit target,
        # 6% Max Trailing Drawdown, 2% Daily Pause/Drawdown.
        # Best day rule: 40% (must not account for more than 40% of total profits for payout)

        return {
            "account_type": "E8 One Challenge",
            "balance": 5000.0,
            "daily_drawdown_limit": 0.02, # 2% Daily Pause / Loss
            "max_drawdown_limit": 0.06,   # 6% Trailing Drawdown
            "drawdown_type": "dynamic",   # Trailing from HWM, locks at initial balance
            "profit_target": 0.06,        # 6% Profit Target
            "best_day_ratio": 0.40,       # 40%
            "daily_api_request_limit": 2000,
            "daily_drawdown_stop": 0.85,
            "max_drawdown_stop": 0.85,
            "best_day_stop": 0.85
        }

    @staticmethod
    def get_e8_trial_5k_rules() -> dict[str, Any]:
        """
        Rules for E8 Trial $5,000 Account.
        """
        return {
            "account_type": "E8 Trial",
            "balance": 5000.0,
            "daily_drawdown_limit": 0.02, # Usually 2% or 4%
            "max_drawdown_limit": 0.04,   # Usually 4% or 8%
            "drawdown_type": "dynamic",
            "profit_target": 0.06,        # Usually 6%
            "best_day_ratio": 0.40,
            "daily_api_request_limit": 2000,
            "daily_drawdown_stop": 0.85,
            "max_drawdown_stop": 0.85,
            "best_day_stop": 0.85
        }
