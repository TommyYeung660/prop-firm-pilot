from abc import ABC, abstractmethod

from pydantic import BaseModel


class AccountRules(BaseModel):
    name: str
    size: float
    daily_drawdown_limit: float
    max_drawdown_limit: float
    drawdown_type: str # "balance", "dynamic", "equity"
    profit_target: float
    best_day_ratio: float
    best_day_limit: float

class BaseScraper(ABC):
    @abstractmethod
    def fetch_rules(self, firm_name: str) -> list[AccountRules]:
        pass
