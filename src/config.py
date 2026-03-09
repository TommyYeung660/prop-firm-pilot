"""
Pydantic-based configuration system for prop-firm-pilot.

Loads configuration from YAML files with environment variable overrides.
Usage:
    from src.config import load_config
    config = load_config("config/e8_signature_50k.yaml")
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

# ── FX Instrument Config ────────────────────────────────────────────────────


class InstrumentConfig(BaseModel):
    """Per-instrument trading parameters."""

    pip_value: float = Field(description="Dollar value per pip for 1 standard lot")
    pip_size: float = Field(description="Pip size (0.0001 for most FX, 0.01 for JPY/XAU)")
    min_lot: float = Field(default=0.01, description="Minimum lot size")
    max_lot: float = Field(default=50.0, description="Maximum lot size")
    avg_spread_pips: float = Field(default=1.5, description="Average spread in pips")


# ── Sub-configs ─────────────────────────────────────────────────────────────


class AccountConfig(BaseModel):
    """Prop firm account information."""

    broker: str = "E8 Markets"
    plan: str = "Signature"
    initial_balance: float = 50_000
    currency: str = "USD"


class ComplianceConfig(BaseModel):
    """E8 Markets compliance rules — safety-critical, do not modify lightly."""

    daily_drawdown_limit: float = Field(
        default=0.05, description="5% daily drawdown (Soft Breach → Daily Pause)"
    )
    max_drawdown_limit: float = Field(
        default=0.08, description="8% max drawdown (Hard Breach → account terminated)"
    )
    profit_target: float = Field(default=0.08, description="8% profit target")
    best_day_ratio: float = Field(default=0.40, description="40% Best Day Rule")
    best_day_limit: float = Field(
        default=1600.0, description="profit_target * initial_balance * best_day_ratio"
    )
    daily_api_request_limit: int = Field(default=2000, description="API request limit per day")
    drawdown_type: Literal["balance", "equity", "dynamic"] = Field(
        default="balance",
        description=(
            "balance: fixed floor based on initial_balance (E8 Signature). "
            "equity: floor based on current equity. "
            "dynamic: trailing high-water mark — floor rises with equity peaks (E8 Trial)."
        ),
    )

    # Safety margins — stop trading before hitting hard limits
    daily_drawdown_stop: float = Field(
        default=0.85, description="Stop new trades at 85% of daily drawdown limit"
    )
    max_drawdown_stop: float = Field(
        default=0.85, description="Stop new trades at 85% of max drawdown limit"
    )
    best_day_stop: float = Field(
        default=0.85, description="Stop new trades at 85% of Best Day limit"
    )
    hwm_state_path: str = Field(
        default="data/hwm_state.json",
        description="File path for HighWaterMarkTracker persistence (dynamic drawdown only)",
    )


class ScannerConfig(BaseModel):
    """Bridge config for qlib_market_scanner."""

    project_path: str = "../../qlib_market_scanner"
    topk: int = 3
    max_signal_age_days: int = 2  # v1.3.0: reject signals older than N days


class AgentsConfig(BaseModel):
    """Bridge config for TradingAgents."""

    project_path: str = "../../TradingAgents"
    selected_analysts: list[str] = ["market", "news", "social", "macro"]
    output_language: str = "繁體中文"


class ExecutionConfig(BaseModel):
    """Trade execution parameters."""

    max_positions: int = 3
    default_risk_pct: float = 0.01
    max_risk_pct: float = 0.02
    random_delay_min: float = 0.5
    random_delay_max: float = 3.0
    position_offset_pct: float = 0.10
    max_slippage_pips: float = 2.0


class MonitorConfig(BaseModel):
    """Equity monitoring and alerting."""

    equity_check_interval: int = 60
    drawdown_alert_pct: float = 0.80
    auto_close_pct: float = 0.90
    trade_journal_path: str = "data/trade_journal.jsonl"
    memory_dir: str = "MEMORY"


class OptimizationConfig(BaseModel):
    """Optimization and feedback loop settings."""

    state_path: str = "data/optimization_state.json"
    pnl_lookback_days: int = 7
    winrate_lookback_days: int = 14
    ab_model_a: str = "rightcodes/gpt-5.4"
    ab_model_b: str = "volcengine/kimi-k2.5"
    ab_ratio: float = 0.5


class DecisionStoreConfig(BaseModel):
    """SQLite decision store settings for the Hybrid EA+LLM pipeline."""

    db_path: str = "data/decisions.db"
    claim_ttl_minutes: int = Field(default=30, description="Max minutes for LLM to process a claim")
    intent_retention_days: int = Field(
        default=7, description="Days to keep old intents before cleanup"
    )


class MarketHoursConfig(BaseModel):
    """FX market hours and weekend closure settings (per-account).

    Times are in UTC (winter time baseline). When dst_auto is enabled,
    close/open times auto-adjust for DST based on server_timezone.

    Typical FX: close Friday 22:00 UTC (winter), open Sunday 22:00 UTC (winter).
    E8 server = Europe/Athens (UTC+2 winter, UTC+3 summer).
    In summer, 22:00 winter-UTC shifts to 21:00 actual-UTC.
    """

    enabled: bool = Field(default=False, description="Enable weekend market closure handling")
    close_day: str = Field(default="Friday", description="Day market closes (e.g., Friday)")
    close_time_utc: str = Field(
        default="22:00", description="Market close time in UTC winter-baseline (HH:MM)"
    )
    open_day: str = Field(default="Sunday", description="Day market opens (e.g., Sunday)")
    open_time_utc: str = Field(
        default="22:00", description="Market open time in UTC winter-baseline (HH:MM)"
    )
    force_close_before_weekend: bool = Field(
        default=False, description="Force-close all positions before weekend"
    )
    force_close_minutes_before: int = Field(
        default=15, description="Minutes before market close to force-close positions"
    )

    # DST auto-adjustment
    dst_auto: bool = Field(
        default=False, description="Auto-adjust market hours for DST based on server_timezone"
    )
    server_timezone: str = Field(
        default="Europe/Athens",
        description="IANA timezone of the broker server (e.g., Europe/Athens for E8 Markets)",
    )


class SchedulerConfig(BaseModel):
    """Async scheduler cadences for the Hybrid EA+LLM pipeline."""

    scanner_interval_seconds: int = Field(default=14400, description="Scanner cycle interval (4h)")
    llm_poll_interval_seconds: int = Field(default=10, description="LLM worker poll (v1.3.6)")
    execution_poll_interval_seconds: int = Field(default=10, description="Execution engine poll")
    janitor_interval_seconds: int = Field(default=600, description="Janitor cleanup cycle (10min)")
    llm_worker_count: int = Field(default=2, description="Number of concurrent LLM workers")
    equity_poll_interval_seconds: int = Field(
        default=60, description="Equity monitor poll interval"
    )
    position_monitor_interval_seconds: int = Field(
        default=120, description="Position close detection poll interval"
    )
    daily_summary_hour_utc: int = Field(
        default=22, description="UTC hour to send daily summary (0-23)"
    )
    breakeven_activation_pct: float = Field(
        default=0.5,
        description="Move SL to breakeven when profit reaches this fraction of TP distance",
    )
    reeval_interval_seconds: int = Field(
        default=7200, description="Re-evaluate open positions via LLM every N seconds (2h)"
    )
    reeval_min_hold_seconds: int = Field(
        default=3600,
        description="Minimum seconds a position must be held before first re-evaluation (1h)",
    )
    market_hours: MarketHoursConfig = Field(
        default_factory=MarketHoursConfig,
        description="Weekend market closure settings",
    )

    # v1.2.0: Session-aware scanning cadence
    session_aware_enabled: bool = Field(
        default=False, description="Enable session-aware scanner interval adjustment"
    )
    active_session_interval_seconds: int = Field(
        default=3600, description="Scanner interval during active sessions (London/NY overlap, 1h)"
    )
    quiet_session_interval_seconds: int = Field(
        default=14400, description="Scanner interval during quiet hours (Asia, 4h)"
    )
    london_open_utc: int = Field(
        default=7, description="London session open hour (UTC winter-baseline)"
    )
    london_close_utc: int = Field(
        default=16, description="London session close hour (UTC winter-baseline)"
    )
    ny_open_utc: int = Field(
        default=12, description="New York session open hour (UTC winter-baseline)"
    )
    ny_close_utc: int = Field(
        default=21, description="New York session close hour (UTC winter-baseline)"
    )

    # DST auto-adjustment for sessions
    session_dst_auto: bool = Field(
        default=False, description="Auto-adjust London/NY session hours for DST"
    )
    london_timezone: str = Field(
        default="Europe/London", description="IANA timezone for London session DST calculation"
    )
    ny_timezone: str = Field(
        default="America/New_York", description="IANA timezone for New York session DST calculation"
    )
    # v1.2.0: Volatility-triggered scans
    volatility_trigger_enabled: bool = Field(
        default=False, description="Enable volatility-triggered scanner re-scans"
    )
    volatility_threshold_pct: float = Field(
        default=0.2, description="Price change % to trigger re-scan (v1.3.6: 0.3→0.2%)"
    )
    volatility_window_minutes: int = Field(
        default=30, description="Rolling window for price change calculation (minutes)"
    )
    volatility_poll_interval_seconds: int = Field(
        default=15, description="How often to check quote prices for volatility (v1.3.6: 60→15s)"
    )
    volatility_cooldown_seconds: int = Field(
        default=300, description="Min seconds between volatility-triggered scans (v1.3.6: 5min)"
    )

    # v1.3.5: Compliance rejection cooldown
    rejection_cooldown_minutes: int = Field(
        default=120,
        description="Minutes to wait after a compliance rejection before retrying the same symbol. "
        "Prevents infinite scanner → LLM → rejection loops.",
    )

    # v1.2.0: Multi-timeframe analysis
    multi_timeframe_enabled: bool = Field(
        default=False, description="Enable multi-timeframe entry timing"
    )
    entry_timeframe: str = Field(
        default="4h", description="Shorter timeframe for entry timing (4h or 1h)"
    )
    intraday_lookback_days: int = Field(
        default=90, description="Days of intraday data to fetch for entry analysis"
    )
    # v1.3.5: Dual-timeframe separation
    scanner_timeframe: str = Field(
        default="1d", description="Timeframe for Qlib scanner pipeline (1d recommended)"
    )
    agent_timeframe: str = Field(
        default="4h", description="Timeframe for TradingAgents entry analysis (4h recommended)"
    )


# ── Tactical Execution Config (v1.3.7) ──────────────────────────────────


class TacticalHardGatesConfig(BaseModel):
    """Hard gate thresholds — ALL must pass for tactical approval."""

    spread_max_multiplier: float = Field(
        default=2.0, description="Max spread as multiplier of typical spread"
    )
    atr_min_ratio: float = Field(
        default=0.5, description="Min ATR ratio vs rolling median (avoid dead market)"
    )
    atr_max_ratio: float = Field(
        default=2.5, description="Max ATR ratio vs rolling median (avoid extreme volatility)"
    )
    atr_period: int = Field(default=14, description="ATR lookback period")
    atr_timeframe: str = Field(default="1h", description="Timeframe for ATR calculation")
    data_max_age_seconds: int = Field(
        default=600, description="Max age of latest bar data in seconds"
    )


class TacticalSoftGatesConfig(BaseModel):
    """Soft gate thresholds — scoring system, min_score out of 3 must pass."""

    min_score: int = Field(default=2, description="Minimum soft gates to pass (out of 3)")
    ema_fast: int = Field(default=8, description="Fast EMA period for momentum check")
    ema_slow: int = Field(default=21, description="Slow EMA period for momentum check")
    ema_timeframe: str = Field(default="5min", description="Timeframe for EMA calculation")
    ema_lookback_bars: int = Field(default=50, description="Number of bars to fetch for EMA")
    rsi_period: int = Field(default=14, description="RSI lookback period")
    rsi_overbought: int = Field(default=70, description="RSI overbought threshold")
    rsi_oversold: int = Field(default=30, description="RSI oversold threshold")
    candle_min_body_ratio: float = Field(
        default=0.3, description="Min candle body/range ratio for directional quality"
    )


class TacticalRetryConfig(BaseModel):
    """Retry parameters for WAIT results."""

    interval_seconds: int = Field(default=300, description="Retry interval (5 min)")
    max_retries: int = Field(default=12, description="Max retries (1 hour total)")
    expire_action: Literal["degrade", "cancel"] = Field(
        default="degrade", description="Action on expiry: degrade thresholds or cancel"
    )
    jitter_seconds: int = Field(default=10, description="Random jitter ±seconds on retry timing")


class TacticalDecisionCacheConfig(BaseModel):
    """Strategic decision cache settings."""

    ttl_seconds: int = Field(default=14400, description="Cache TTL in seconds (4 hours)")


class TacticalIntentDedupConfig(BaseModel):
    """Intent deduplication settings."""

    cooldown_after_close_seconds: int = Field(
        default=1800, description="Cooldown after position close before allowing same-symbol intent"
    )


class TacticalConfig(BaseModel):
    """v1.3.7: Tactical execution module configuration.

    Controls the Hard/Soft dual-layer gate system that validates entry timing
    using low-timeframe (5min/1H) data before executing strategic decisions.

    Usage:
        config = TacticalConfig(shadow_mode=True)
        assert = config.hard_gates.spread_max_multiplier == 2.0
    """

    enabled: bool = Field(default=True, description="Enable tactical validation module")
    shadow_mode: bool = Field(
        default=True,
        description="Shadow mode: log gate results but never block trades (Phase 1)",
    )
    hard_gates: TacticalHardGatesConfig = Field(default_factory=TacticalHardGatesConfig)
    soft_gates: TacticalSoftGatesConfig = Field(default_factory=TacticalSoftGatesConfig)
    retry: TacticalRetryConfig = Field(default_factory=TacticalRetryConfig)
    decision_cache: TacticalDecisionCacheConfig = Field(default_factory=TacticalDecisionCacheConfig)
    intent_dedup: TacticalIntentDedupConfig = Field(default_factory=TacticalIntentDedupConfig)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    file: str = "logs/prop_firm_pilot.log"
    rotation: str = "10 MB"
    retention: str = "30 days"


# ── Root Config ─────────────────────────────────────────────────────────────


class AppConfig(BaseModel):
    """Root configuration for prop-firm-pilot."""

    symbols: list[str] = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD"]
    account: AccountConfig = AccountConfig()
    compliance: ComplianceConfig = ComplianceConfig()
    scanner: ScannerConfig = ScannerConfig()
    agents: AgentsConfig = AgentsConfig()
    execution: ExecutionConfig = ExecutionConfig()
    monitor: MonitorConfig = MonitorConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    logging: LoggingConfig = LoggingConfig()
    instruments: dict[str, InstrumentConfig] = {}
    decision_store: DecisionStoreConfig = DecisionStoreConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    tactical: TacticalConfig = Field(default_factory=TacticalConfig)


# ── Config Loading ──────────────────────────────────────────────────────────


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    config_path: str | Path,
    default_path: str | Path | None = None,
) -> AppConfig:
    """Load configuration from YAML, merging with defaults.

    Args:
        config_path: Path to account-specific config (e.g. e8_signature_50k.yaml).
        default_path: Path to default config. Auto-detected if None.

    Returns:
        Fully resolved AppConfig instance.
    """
    config_path = Path(config_path)

    # Auto-detect default config location
    if default_path is None:
        default_path = config_path.parent / "default.yaml"

    # Load default config
    base_data: dict[str, Any] = {}
    if Path(default_path).exists():
        with open(default_path, encoding="utf-8") as f:
            base_data = yaml.safe_load(f) or {}

    # Load account-specific config
    override_data: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            override_data = yaml.safe_load(f) or {}

    # Merge: account config overrides defaults
    merged = _deep_merge(base_data, override_data)

    return AppConfig(**merged)
