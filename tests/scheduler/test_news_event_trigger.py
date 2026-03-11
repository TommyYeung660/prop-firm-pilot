"""
Tests for news-driven rescan trigger.
"""

from datetime import datetime, timezone

import httpx
import pytest
import respx

from src.scheduler.news_event_trigger import NewsEventTrigger


@pytest.mark.asyncio
async def test_check_once_triggers_on_new_high_impact_macro_headline() -> None:
    """Fresh macro headlines should trigger a rescan exactly once."""
    trigger = NewsEventTrigger(
        api_key="test-key",
        keywords=["federal reserve", "cpi"],
        lookback_minutes=30,
        max_headlines=10,
    )
    now = datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc)
    payload = {
        "feed": [
            {
                "title": "Federal Reserve signals emergency liquidity move",
                "summary": "Markets repriced rate path after the surprise headline.",
                "time_published": "20260311T115500",
                "url": "https://example.com/fed-liquidity",
            }
        ]
    }

    with respx.mock:
        respx.get("https://www.alphavantage.co/query").respond(200, json=payload)
        async with httpx.AsyncClient() as client:
            triggered, headlines = await trigger.check_once(client=client, now=now)
            triggered_again, headlines_again = await trigger.check_once(client=client, now=now)

    assert triggered is True
    assert headlines[0]["title"] == payload["feed"][0]["title"]
    assert triggered_again is False
    assert headlines_again == []


@pytest.mark.asyncio
async def test_check_once_ignores_stale_or_irrelevant_headlines() -> None:
    """Old or non-macro headlines should not trigger."""
    trigger = NewsEventTrigger(
        api_key="test-key",
        keywords=["federal reserve", "cpi"],
        lookback_minutes=30,
        max_headlines=10,
    )
    now = datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc)
    payload = {
        "feed": [
            {
                "title": "Corporate earnings roundup",
                "summary": "Irrelevant for FX macro trigger.",
                "time_published": "20260311T115500",
                "url": "https://example.com/earnings",
            },
            {
                "title": "Federal Reserve interview archive",
                "summary": "Old headline outside trigger window.",
                "time_published": "20260311T090000",
                "url": "https://example.com/old-fed",
            },
        ]
    }

    with respx.mock:
        respx.get("https://www.alphavantage.co/query").respond(200, json=payload)
        async with httpx.AsyncClient() as client:
            triggered, headlines = await trigger.check_once(client=client, now=now)

    assert triggered is False
    assert headlines == []
