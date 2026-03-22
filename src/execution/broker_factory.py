"""
Broker client factory for selecting execution backend at runtime.

This module builds a broker client using AppConfig.execution.broker_backend.
TradeLocker import is intentionally lazy to keep Task 3 independent from
the concrete TradeLocker client implementation.

Usage:
    from src.execution.broker_factory import build_broker_client
    broker = build_broker_client(config, store=decision_store)
"""

import os
from typing import Any

from src.config import AppConfig


def build_broker_client(config: AppConfig, store: Any = None) -> Any:
    """Build broker client for the configured backend."""
    if config.execution.broker_backend == "matchtrader":
        from src.execution.matchtrader_client import MatchTraderClient

        return MatchTraderClient(
            base_url=os.getenv("MATCHTRADER_API_URL", ""),
            email=os.getenv("MATCHTRADER_USERNAME", ""),
            password=os.getenv("MATCHTRADER_PASSWORD", ""),
            broker_id=os.getenv("MATCHTRADER_BROKER_ID", "2"),
            account_id=os.getenv("MATCHTRADER_ACCOUNT_ID") or None,
            daily_request_limit=config.compliance.daily_api_request_limit,
            store=store,
        )

    if config.execution.broker_backend == "tradelocker":
        from src.execution.tradelocker_client import TradeLockerClient

        return TradeLockerClient(
            api_url=config.tradelocker.api_url,
            email=config.tradelocker.email,
            password=config.tradelocker.password,
            server=config.tradelocker.server,
            account_id=config.tradelocker.account_id,
            store=store,
        )

    raise ValueError(f"Unsupported broker backend: {config.execution.broker_backend}")
