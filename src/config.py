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


class DataConfig(BaseModel):
    """FX data acquisition settings."""

    interval: str = "1d"
    lookback_days: int = 730
    provider: Literal["itick", "tradermade", "alpha_vantage"] = "itick"
    duckdb_path: str = "data/fx_prices.duckdb"
    qlib_binary_dir: str = "data/qlib_binary"


class ScannerConfig(BaseModel):
    """Bridge config for qlib_market_scanner."""

    project_path: str = "../../qlib_market_scanner"
    topk: int = 3
    n_drop: int = 1
    enable_rdagent_factors: bool = True
    min_factor_ic_ir: float = 0.5


class AgentsConfig(BaseModel):
    """Bridge config for TradingAgents."""

    project_path: str = "../../TradingAgents"
    selected_analysts: list[str] = ["market", "news", "social"]
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
    ab_model_a: str = "volcengine/glm-4.7"
    ab_model_b: str = "gpt-5.2"
    ab_ratio: float = 0.5


class ScheduleConfig(BaseModel):
    """Daily cycle scheduling (UTC)."""

    daily_cycle: str = "06:00"
    equity_monitor: str = "always"
    rdagent_trigger: str = "weekend"


class DecisionStoreConfig(BaseModel):
    """SQLite decision store settings for the Hybrid EA+LLM pipeline."""

    db_path: str = "data/decisions.db"
    claim_ttl_minutes: int = Field(default=30, description="Max minutes for LLM to process a claim")
    intent_retention_days: int = Field(
        default=7, description="Days to keep old intents before cleanup"
    )




class MarketHoursConfig(BaseModel):
    """FX market hours and weekend closure settings (per-account).

    Times are in UTC. Typical FX: close Friday 22:00 UTC, open Sunday 22:00 UTC.
    These correspond to 17:00 EST / 17:00 EST (US Eastern).
    """

    enabled: bool = Field(default=False, description="Enable weekend market closure handling")
    close_day: str = Field(default="Friday", description="Day market closes (e.g., Friday)")
    close_time_utc: str = Field(
        default="22:00", description="Market close time in UTC (HH:MM)"
    )
    open_day: str = Field(default="Sunday", description="Day market opens (e.g., Sunday)")
    open_time_utc: str = Field(
        default="22:00", description="Market open time in UTC (HH:MM)"
    )
    force_close_before_weekend: bool = Field(
        default=False, description="Force-close all positions before weekend"
    )
    force_close_minutes_before: int = Field(
        default=15, description="Minutes before market close to force-close positions"
    )
class SchedulerConfig(BaseModel):
    """Async scheduler cadences for the Hybrid EA+LLM pipeline."""

    scanner_interval_seconds: int = Field(default=14400, description="Scanner cycle interval (4h)")
    llm_poll_interval_seconds: int = Field(default=30, description="LLM worker poll interval")
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
    london_open_utc: int = Field(default=7, description="London session open hour (UTC)")
    london_close_utc: int = Field(default=16, description="London session close hour (UTC)")
    ny_open_utc: int = Field(default=12, description="New York session open hour (UTC)")
    ny_close_utc: int = Field(default=21, description="New York session close hour (UTC)")

    # v1.2.0: Volatility-triggered scans
    volatility_trigger_enabled: bool = Field(
        default=False, description="Enable volatility-triggered scanner re-scans"
    )
    volatility_threshold_pct: float = Field(
        default=0.3, description="Price change % to trigger re-scan (0.3 = 0.3%)"
    )
    volatility_window_minutes: int = Field(
        default=30, description="Rolling window for price change calculation (minutes)"
    )
    volatility_poll_interval_seconds: int = Field(
        default=60, description="How often to check quote prices for volatility (seconds)"
    )
    volatility_cooldown_seconds: int = Field(
        default=900, description="Min seconds between volatility-triggered scans (15min)"
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
    data: DataConfig = DataConfig()
    scanner: ScannerConfig = ScannerConfig()
    agents: AgentsConfig = AgentsConfig()
    execution: ExecutionConfig = ExecutionConfig()
    monitor: MonitorConfig = MonitorConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    schedule: ScheduleConfig = ScheduleConfig()
    logging: LoggingConfig = LoggingConfig()
    instruments: dict[str, InstrumentConfig] = {}
    decision_store: DecisionStoreConfig = DecisionStoreConfig()
    scheduler: SchedulerConfig = SchedulerConfig()


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
