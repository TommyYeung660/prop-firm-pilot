"""
FX-specific analyst configuration for TradingAgents.

Configures which analysts are active, which data sources they use,
and how their prompts are adapted for FX trading.
"""

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
    "XAUUSD": {
        "description": "Gold vs US Dollar — safe haven / inflation hedge",
        "key_drivers": ["Real yields", "USD strength", "geopolitics", "inflation"],
        "avg_daily_range_pips": 200,
        "session_bias": "London + NY",
    },
}


def build_agent_config(
    deep_think_llm: str = "volcengine/glm-4.7",
    quick_think_llm: str = "volcengine/glm-4.7",
    output_language: str = "繁體中文",
) -> dict[str, Any]:
    """Build TradingAgents config dict for FX trading.

    Returns:
        Config dict ready to pass to TradingAgentsGraph().
    """
    return {
        "deep_think_llm": deep_think_llm,
        "quick_think_llm": quick_think_llm,
        "output_language": output_language,
        "market_type": "fx",  # newly added for TradingAgents integration
        "data_vendors": {
            "core_stock_apis": "itick",
            "news_data": "alpha_vantage",
        },
        # FX context injection
        "fx_mode": True,
        "fx_pairs": list(FX_PAIR_CONTEXT.keys()),
        "fx_key_events": FX_KEY_EVENTS,
    }


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
