"""Tests for broker factory backend routing."""

import sys
import types
from typing import Any

from src.config import AppConfig
from src.execution.broker_factory import build_broker_client


def test_broker_factory_returns_matchtrader_for_matchtrader_backend(
    monkeypatch: Any,
) -> None:
    class FakeMatchTraderClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(
        "src.execution.matchtrader_client.MatchTraderClient",
        FakeMatchTraderClient,
    )
    monkeypatch.setenv("MATCHTRADER_API_URL", "https://mtr.test")
    monkeypatch.setenv("MATCHTRADER_USERNAME", "user@example.com")
    monkeypatch.setenv("MATCHTRADER_PASSWORD", "secret")
    monkeypatch.setenv("MATCHTRADER_BROKER_ID", "2")
    monkeypatch.setenv("MATCHTRADER_ACCOUNT_ID", "acct-001")

    config = AppConfig()
    config.execution.broker_backend = "matchtrader"

    broker = build_broker_client(config, store=None)
    assert broker.__class__.__name__ == "FakeMatchTraderClient"


def test_broker_factory_returns_tradelocker_for_tradelocker_backend(
    monkeypatch: Any,
) -> None:
    fake_module = types.ModuleType("src.execution.tradelocker_client")

    class TradeLockerClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    fake_module.TradeLockerClient = TradeLockerClient
    monkeypatch.setitem(sys.modules, "src.execution.tradelocker_client", fake_module)

    config = AppConfig()
    config.execution.broker_backend = "tradelocker"
    config.tradelocker.api_url = "https://tl.test"
    config.tradelocker.email = "user@example.com"
    config.tradelocker.password = "secret"
    config.tradelocker.server = "demo"
    config.tradelocker.account_id = "acct-001"

    broker = build_broker_client(config, store=None)
    assert broker.__class__.__name__ == "TradeLockerClient"
