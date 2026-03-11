"""
Macro news trigger — polls headline feeds and flags fresh high-impact events.

Uses Alpha Vantage NEWS_SENTIMENT to detect newly published macro headlines
that should force an early scheduler rescan.

Usage:
    trigger = NewsEventTrigger(api_key="...", keywords=["federal reserve", "cpi"])
    async with httpx.AsyncClient() as client:
        triggered, headlines = await trigger.check_once(client, now=datetime.now(timezone.utc))
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from loguru import logger

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
GLOBAL_NEWS_TOPICS = "economy_monetary,economy_macro,financial_markets"


class NewsEventTrigger:
    """Detect fresh macro headlines that should trigger an early rescan.

    Usage:
        trigger = NewsEventTrigger(api_key="...")
        triggered, headlines = await trigger.check_once(client, now)
    """

    def __init__(
        self,
        api_key: str,
        keywords: list[str],
        lookback_minutes: int = 30,
        max_headlines: int = 10,
        cooldown_seconds: int = 300,
    ) -> None:
        self._api_key = api_key
        self._keywords = [keyword.lower() for keyword in keywords]
        self._lookback_minutes = lookback_minutes
        self._max_headlines = max_headlines
        self._cooldown_seconds = cooldown_seconds
        self._seen_ids: set[str] = set()
        self._last_trigger_at: datetime | None = None

    async def check_once(
        self,
        client: httpx.AsyncClient,
        now: datetime,
    ) -> tuple[bool, list[dict[str, str]]]:
        """Fetch the latest headlines and return newly seen trigger-worthy items."""
        if self._last_trigger_at is not None:
            elapsed = (now - self._last_trigger_at).total_seconds()
            if elapsed < self._cooldown_seconds:
                return False, []

        response = await client.get(
            ALPHA_VANTAGE_URL,
            params={
                "function": "NEWS_SENTIMENT",
                "topics": GLOBAL_NEWS_TOPICS,
                "sort": "LATEST",
                "limit": str(self._max_headlines),
                "apikey": self._api_key,
            },
        )
        response.raise_for_status()
        payload = response.json()
        feed = payload.get("feed", []) if isinstance(payload, dict) else []

        fresh_headlines: list[dict[str, str]] = []
        cutoff = now - timedelta(minutes=self._lookback_minutes)
        for item in feed:
            headline = self._normalize_headline(item)
            if headline is None:
                continue
            published_at = self._parse_published_at(headline["published_at"])
            if published_at is None or published_at < cutoff:
                continue
            if not self._matches_keywords(headline):
                continue
            headline_id = headline["id"]
            if headline_id in self._seen_ids:
                continue
            self._seen_ids.add(headline_id)
            fresh_headlines.append(headline)

        if fresh_headlines:
            self._last_trigger_at = now
            logger.info(
                "NewsEventTrigger: {} fresh headline(s) matched macro keywords",
                len(fresh_headlines),
            )
            return True, fresh_headlines

        return False, []

    def _matches_keywords(self, headline: dict[str, str]) -> bool:
        """Check whether title or summary contains any configured keyword."""
        text = f"{headline['title']} {headline['summary']}".lower()
        return any(keyword in text for keyword in self._keywords)

    def _normalize_headline(self, item: Any) -> dict[str, str] | None:
        """Normalize provider payload into the scheduler's trigger format."""
        if not isinstance(item, dict):
            return None
        title = str(item.get("title", "")).strip()
        if not title:
            return None
        published_at = str(item.get("time_published", "")).strip()
        url = str(item.get("url", "")).strip()
        headline_id = url or f"{published_at}:{title}"
        return {
            "id": headline_id,
            "title": title,
            "summary": str(item.get("summary", "")).strip(),
            "published_at": published_at,
            "url": url,
        }

    @staticmethod
    def _parse_published_at(value: str) -> datetime | None:
        """Parse Alpha Vantage timestamps into UTC datetimes."""
        if not value:
            return None
        try:
            return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            logger.debug("NewsEventTrigger: invalid time_published '{}'", value)
            return None
