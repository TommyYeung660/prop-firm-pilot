"""
Pydantic-based configuration system for prop-firm-pilot.

Loads configuration from YAML files with environment variable overrides.
Usage:
    from src.config import load_config
    config = load_config("config/e8_signature_50k.yaml")
"""

from pathlib import Path
from typing import Any, Dict, List, Literal

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
    drawdown_type: Literal["balance", "equity"] = Field(
        default="balance", description="E8 uses balance-based drawdown"
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
    selected_analysts: List[str] = ["market", "news", "social"]
    deep_think_llm: str = "volcengine/glm-4.7"
    quick_think_llm: str = "volcengine/glm-4.7"
    output_language: str = "繁體中文"


class ExecutionConfig(BaseModel):
    """Trade execution parameters."""

    max_positions: int = 3
    default_risk_pct: float = 0.01
    max_risk_pct: float = 0.02
    random_delay_min: float = 0.5
    random_delay_max: float = 3.0
    position_offset_pct: float = 0.10


class MonitorConfig(BaseModel):
    """Equity monitoring and alerting."""

    equity_check_interval: int = 60
    drawdown_alert_pct: float = 0.80
    auto_close_pct: float = 0.90
    trade_journal_path: str = "data/trade_journal.jsonl"


class ScheduleConfig(BaseModel):
    """Daily cycle scheduling (UTC)."""

    daily_cycle: str = "06:00"
    equity_monitor: str = "always"
    rdagent_trigger: str = "weekend"


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    file: str = "logs/prop_firm_pilot.log"
    rotation: str = "10 MB"
    retention: str = "30 days"


# ── Root Config ─────────────────────────────────────────────────────────────


class AppConfig(BaseModel):
    """Root configuration for prop-firm-pilot."""

    symbols: List[str] = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD"]
    account: AccountConfig = AccountConfig()
    compliance: ComplianceConfig = ComplianceConfig()
    data: DataConfig = DataConfig()
    scanner: ScannerConfig = ScannerConfig()
    agents: AgentsConfig = AgentsConfig()
    execution: ExecutionConfig = ExecutionConfig()
    monitor: MonitorConfig = MonitorConfig()
    schedule: ScheduleConfig = ScheduleConfig()
    logging: LoggingConfig = LoggingConfig()
    instruments: Dict[str, InstrumentConfig] = {}


# ── Config Loading ──────────────────────────────────────────────────────────


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
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
    base_data: Dict[str, Any] = {}
    if Path(default_path).exists():
        with open(default_path, "r", encoding="utf-8") as f:
            base_data = yaml.safe_load(f) or {}

    # Load account-specific config
    override_data: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            override_data = yaml.safe_load(f) or {}

    # Merge: account config overrides defaults
    merged = _deep_merge(base_data, override_data)

    return AppConfig(**merged)
