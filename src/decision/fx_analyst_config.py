"""
FX-specific analyst configuration for TradingAgents.

Configures which analysts are active, which data sources they use,
and how their prompts are adapted for FX trading.
"""

import os
from datetime import date
from typing import Any

# FX-appropriate analysts (removed: fundamentals, options — not applicable to FX)
FX_ANALYSTS = ["macro", "market", "news", "social"]

# Analyst-specific data vendor mapping for FX
FX_DATA_VENDORS: dict[str, dict[str, str]] = {
    "market": {
        "source": "itick",
        "data_type": "OHLCV daily bars",
        "focus": "Technical analysis, price action, support/resistance levels",
    },
    "news": {
        "source": "alpha_vantage",
        "data_type": "FX-related news",
        "focus": "Central bank decisions, NFP, CPI, trade balance, geopolitics",
    },
    "social": {
        "source": "web_search",
        "data_type": "Social sentiment",
        "focus": "Retail sentiment, COT positioning, market consensus",
    },
}

# Key FX macro events to watch (for news analyst prompts)
FX_KEY_EVENTS = [
    "Fed interest rate decision",
    "ECB interest rate decision",
    "BOJ interest rate decision",
    "RBA interest rate decision",
    "Non-Farm Payrolls (NFP)",
    "CPI (US, EU, UK, JP, AU)",
    "GDP releases",
    "PMI Manufacturing/Services",
    "Retail Sales",
    "Trade Balance",
    "Central bank speeches",
    "Geopolitical events",
]

# FX pair characteristics for agent context
FX_PAIR_CONTEXT: dict[str, dict[str, Any]] = {
    "EURUSD": {
        "description": "Euro vs US Dollar — most liquid pair",
        "key_drivers": ["ECB/Fed rate differential", "EU GDP", "US NFP"],
        "avg_daily_range_pips": 70,
        "session_bias": "London + NY overlap",
    },
    "GBPUSD": {
        "description": "British Pound vs US Dollar — cable",
        "key_drivers": ["BOE decisions", "Brexit effects", "UK CPI"],
        "avg_daily_range_pips": 90,
        "session_bias": "London session",
    },
    "USDJPY": {
        "description": "US Dollar vs Japanese Yen — safe haven",
        "key_drivers": ["BOJ policy", "US Treasury yields", "risk sentiment"],
        "avg_daily_range_pips": 75,
        "session_bias": "Tokyo + NY",
    },
    "AUDUSD": {
        "description": "Australian Dollar vs US Dollar — commodity currency",
        "key_drivers": ["RBA decisions", "China PMI", "commodity prices"],
        "avg_daily_range_pips": 65,
        "session_bias": "Sydney + London",
    },
    "NZDUSD": {
        "description": "New Zealand Dollar vs US Dollar — risk-sensitive commodity currency",
        "key_drivers": ["RBNZ decisions", "dairy prices", "China demand"],
        "avg_daily_range_pips": 60,
        "session_bias": "Sydney + London",
    },
    "USDCAD": {
        "description": "US Dollar vs Canadian Dollar — oil-correlated",
        "key_drivers": ["BoC decisions", "crude oil prices", "US/Canada trade"],
        "avg_daily_range_pips": 70,
        "session_bias": "NY session",
    },
    "USDCHF": {
        "description": "US Dollar vs Swiss Franc — safe haven mirror",
        "key_drivers": ["SNB policy", "USD strength", "European risk sentiment"],
        "avg_daily_range_pips": 60,
        "session_bias": "London + NY",
    },
    "EURJPY": {
        "description": "Euro vs Japanese Yen — rate differential and risk-on cross",
        "key_drivers": ["ECB/BOJ policy gap", "European risk sentiment", "carry demand"],
        "avg_daily_range_pips": 95,
        "session_bias": "London + Tokyo handoff",
    },
    "AUDJPY": {
        "description": "Australian Dollar vs Japanese Yen — high-beta carry cross",
        "key_drivers": ["RBA stance", "China growth proxies", "global risk appetite"],
        "avg_daily_range_pips": 105,
        "session_bias": "Sydney + Tokyo",
    },
    "CADJPY": {
        "description": "Canadian Dollar vs Japanese Yen — oil-linked cyclical cross",
        "key_drivers": ["BoC policy", "crude oil", "broad risk sentiment"],
        "avg_daily_range_pips": 85,
        "session_bias": "NY + Tokyo",
    },
    "GBPJPY": {
        "description": "British Pound vs Japanese Yen — high-volatility carry cross",
        "key_drivers": ["BOE/BOJ divergence", "risk sentiment", "UK data"],
        "avg_daily_range_pips": 140,
        "session_bias": "London + Tokyo handoff",
    },
    "NZDJPY": {
        "description": "New Zealand Dollar vs Japanese Yen — carry trade proxy",
        "key_drivers": ["RBNZ stance", "global risk appetite", "dairy exports"],
        "avg_daily_range_pips": 80,
        "session_bias": "Sydney + Tokyo",
    },
    "EURGBP": {
        "description": "Euro vs British Pound — tight-range European cross",
        "key_drivers": ["ECB/BOE rate differential", "EU/UK trade", "Brexit legacy"],
        "avg_daily_range_pips": 45,
        "session_bias": "London session",
    },
    "EURAUD": {
        "description": "Euro vs Australian Dollar — wide-range cross with commodity sensitivity",
        "key_drivers": ["ECB/RBA policy gap", "China PMI", "EU risk sentiment"],
        "avg_daily_range_pips": 100,
        "session_bias": "London + Sydney overlap",
    },
    "EURCHF": {
        "description": "Euro vs Swiss Franc — low-volatility range-bound cross",
        "key_drivers": ["ECB/SNB policy", "European stability", "safe haven flows"],
        "avg_daily_range_pips": 45,
        "session_bias": "London session",
    },
    "EURCAD": {
        "description": "Euro vs Canadian Dollar — rate differential and oil cross",
        "key_drivers": ["ECB/BoC policy gap", "crude oil", "EU/Canada trade"],
        "avg_daily_range_pips": 80,
        "session_bias": "London + NY",
    },
    "GBPAUD": {
        "description": "British Pound vs Australian Dollar — high-volatility cross",
        "key_drivers": ["BOE/RBA divergence", "commodity prices", "risk sentiment"],
        "avg_daily_range_pips": 130,
        "session_bias": "London + Sydney",
    },
    "GBPCAD": {
        "description": "British Pound vs Canadian Dollar — oil and rate differential cross",
        "key_drivers": ["BOE/BoC policy", "crude oil", "UK/Canada data"],
        "avg_daily_range_pips": 100,
        "session_bias": "London + NY",
    },
    "GBPCHF": {
        "description": "British Pound vs Swiss Franc — European cross with safe haven element",
        "key_drivers": ["BOE/SNB policy gap", "European risk", "CHF safe haven"],
        "avg_daily_range_pips": 90,
        "session_bias": "London session",
    },
    "AUDNZD": {
        "description": "Australian Dollar vs New Zealand Dollar — tight regional cross",
        "key_drivers": ["RBA/RBNZ differential", "commodity prices", "trans-Tasman data"],
        "avg_daily_range_pips": 50,
        "session_bias": "Sydney session",
    },
    "XAUUSD": {
        "description": "Gold vs US Dollar — safe haven / inflation hedge",
        "key_drivers": ["Real yields", "USD strength", "geopolitics", "inflation"],
        "avg_daily_range_pips": 200,
        "session_bias": "London + NY",
    },
}

# ── EODHD Switchover Configuration ───────────────────────────────────────
# Before switchover date: use Alpha Vantage. On/after: use EODHD.
# Override via env: EODHD_SWITCHOVER_DATE (ISO format), EODHD_FORCE_PRIMARY=1
_DEFAULT_SWITCHOVER_DATE = "2026-03-21"


def _get_primary_vendor() -> str:
    """Determine primary data vendor based on switchover date."""
    force = os.getenv("EODHD_FORCE_PRIMARY", "").strip() in ("1", "true", "yes")
    if force:
        return "eodhd"
    switchover_str = os.getenv("EODHD_SWITCHOVER_DATE", _DEFAULT_SWITCHOVER_DATE)
    switchover_date = date.fromisoformat(switchover_str)
    if date.today() >= switchover_date:
        return "eodhd"
    return "alpha_vantage"


def build_agent_config(
    output_language: str = "繁體中文",
    memory_path: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Build TradingAgents config dict for FX trading.

    Returns:
        Config dict ready to pass to TradingAgentsGraph().
    """
    primary = _get_primary_vendor()
    config: dict[str, Any] = {
        "output_language": output_language,
        "market_type": "fx",  # newly added for TradingAgents integration
        "data_vendors": {
            "core_stock_apis": primary,
            "news_data": primary,
        },
        # Tool-level vendor overrides (takes precedence over category-level)
        # - get_news: primary vendor's news API
        # - get_global_news: primary vendor's global news API
        # - get_indicators: primary vendor's technical indicator APIs
        # - get_insider_*: always local (finnhub)
        "tool_vendors": {
            "get_global_news": primary,
            "get_news": primary,
            "get_indicators": primary,
            "get_insider_sentiment": "local",
            "get_insider_transactions": "local",
        },
        # FX context injection
        "fx_mode": True,
        "fx_pairs": list(FX_PAIR_CONTEXT.keys()),
        "fx_key_events": FX_KEY_EVENTS,
    }
    if memory_path:
        config["memory_path"] = memory_path
    if session_id:
        config["session_id"] = session_id
    return config


def get_pair_context(symbol: str) -> dict[str, Any]:
    """Get FX pair context for agent prompts."""
    return FX_PAIR_CONTEXT.get(
        symbol,
        {
            "description": f"{symbol} — FX pair",
            "key_drivers": ["Unknown"],
            "avg_daily_range_pips": 50,
            "session_bias": "Unknown",
        },
    )
