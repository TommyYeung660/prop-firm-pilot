# AGENTS.md — prop-firm-pilot

Fully automated FX trading system for E8 Markets prop firm accounts.
Python 3.10, async-first, Pydantic v2 config, loguru logging, DuckDB storage.

## Build & Run

```bash
# Install dependencies (uv preferred)
uv sync --all-extras

# Run daily trading cycle
python -m src.main --config config/e8_trial_5k.yaml

# Run with date override (backtesting)
python -m src.main --config config/e8_trial_5k.yaml --date 2026-02-12

# Monitor-only mode (no new trades)
python -m src.main --config config/e8_trial_5k.yaml --monitor-only
```

## Lint & Format

```bash
# Lint (ruff checks E, F, I, N, W, UP rules)
uv run ruff check src/ tests/
uv run ruff check src/ tests/ --fix    # auto-fix

# Format
uv run ruff format src/ tests/
```

Ruff config: `line-length = 100`, `target-version = "py310"`.

## Test

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_foo.py

# Run a single test function
uv run pytest tests/test_foo.py::test_bar

# Run with coverage
uv run pytest --cov=src

# Verbose output
uv run pytest -v
```

pytest config: `asyncio_mode = "auto"`, `testpaths = ["tests"]`.
Use `pytest-asyncio` for async tests — no `@pytest.mark.asyncio` decorator needed.
Use `respx` to mock `httpx` HTTP calls (not `responses` or `aiohttp`).

## Architecture

```
src/
├── main.py              # PropFirmPilot orchestrator (daily cycle entry point)
├── config.py            # Pydantic AppConfig, YAML loading with deep merge
├── compliance/          # E8 Markets rule enforcement (SAFETY-CRITICAL)
│   ├── prop_firm_guard.py    # Pre-trade compliance checks (all must pass)
│   ├── drawdown_monitor.py   # Real-time drawdown tracking (pure calc, no I/O)
│   └── best_day_tracker.py   # 40% Best Day Rule tracker
├── data/                # FX data acquisition & storage
│   ├── fx_data_fetcher.py    # Async multi-provider OHLCV fetcher (TraderMade, iTick)
│   ├── fx_duckdb_store.py    # DuckDB local cache for price data
│   └── fx_to_qlib.py         # Qlib binary format converter
├── decision/            # Multi-agent BUY/SELL/HOLD decisions
│   ├── agent_bridge.py       # Bridge to external TradingAgents project
│   ├── decision_formatter.py # Raw decision → execution-ready format
│   └── fx_analyst_config.py  # FX-specific analyst & pair configs
├── execution/           # Trade execution via MatchTrader REST API
│   ├── matchtrader_client.py # Async REST client (JWT auth, rate limiting, retries)
│   ├── order_manager.py      # Order lifecycle (open/close/reject tracking)
│   └── position_sizer.py     # Risk-based lot calculation with random offset
├── monitor/             # Equity monitoring & alerting
│   ├── equity_monitor.py     # Async background equity polling loop
│   ├── alert_service.py      # Telegram Bot API notifications
│   └── trade_journal.py      # Append-only JSONL trade log
├── research/            # Weekend factor discovery
│   └── rdagent_bridge.py     # Bridge to qlib_rd_agent subprocess
└── signal/              # Scanner signal processing
    ├── scanner_bridge.py     # Bridge to qlib_market_scanner subprocess
    └── signal_formatter.py   # Signal classification & filtering
```

## Code Style

### Imports
- stdlib → third-party → local (`from src.xxx import ...`), enforced by ruff `I` rule.
- Use `from __future__ import annotations` is NOT used; project uses `X | None` union syntax directly (Python 3.10+).
- Import types from `typing`: `Any, Dict, List, Literal, Tuple`.

### Type Annotations
- ALL function signatures must have full type annotations (params + return).
- Use `X | None` instead of `Optional[X]`.
- Use `dict[str, Any]` in new code, but existing code uses `Dict[str, Any]` — match surrounding context.
- Use `Literal["BUY", "SELL", "HOLD"]` for constrained string types.
- Pydantic `BaseModel` for data structures; use `Field(description=...)` for docs.
- Pydantic `model_config = {"populate_by_name": True}` when using aliases.

### Naming
- `snake_case` for functions, methods, variables, modules.
- `PascalCase` for classes.
- `UPPER_SNAKE_CASE` for module-level constants and SQL strings.
- Private methods/attrs prefixed with `_` (e.g., `self._config`, `def _ensure_auth`).
- File names match their primary class: `matchtrader_client.py` → `MatchTraderClient`.

### Module Docstrings
Every module starts with a triple-quote docstring explaining purpose and usage:
```python
"""
Brief description — what this module does.

Longer explanation of responsibilities and context.

Usage:
    obj = MyClass(...)
    result = obj.do_thing()
"""
```

### Class Docstrings
Every class has a docstring with brief description and `Usage:` example.

### Section Comments
Use visual separator comments to organize code sections:
```python
# ── Section Name ────────────────────────────────────────────────────
```

### Logging
- Use `loguru.logger` everywhere — never `print()` or stdlib `logging`.
- Use `{}` placeholders (not f-strings): `logger.info("Got {} items", count)`.
- Log levels: `debug` (internals), `info` (actions), `warning` (recoverable), `error` (failures), `critical` (system down).

### Error Handling
- Define custom exception hierarchies per module (e.g., `MatchTraderError` → `MatchTraderAuthError`).
- Never use bare `except:`. Always catch specific exceptions.
- Log errors before re-raising or returning error results.
- Return result objects (e.g., `OrderResult(success=False, message=...)`) instead of raising for expected failures.
- Use `ComplianceResult(passed=bool, reason=str, rule_name=str)` pattern for pass/fail checks.

### Async Patterns
- Use `httpx.AsyncClient` for HTTP (never `requests` or `aiohttp`).
- Use `async with` context managers for client lifecycle.
- Retry with exponential backoff: `wait = 2 ** attempt`.
- Rate limiting is critical — respect API limits (MatchTrader: 2000/day, iTick: 5/min).

### Configuration
- All config via Pydantic `BaseModel` subclasses in `src/config.py`.
- YAML config files in `config/` — `default.yaml` merged with account-specific YAML.
- Secrets in `.env` (never committed). Access via `os.getenv()`.
- Config loaded once at startup via `load_config()`.

## Safety-Critical Rules

**The compliance module (`src/compliance/`) is SAFETY-CRITICAL.** Incorrect changes can cause real financial loss (account termination). Key constraints:

- **Daily drawdown**: 5% of day-start balance (Soft Breach → pause)
- **Max drawdown**: 8% of initial balance (Hard Breach → account terminated)
- **Best Day Rule**: No single day profit > $1,600 (40% of 8% of $50k)
- **Safety margins**: Stop trading at 85% of each limit, not 100%
- **All compliance checks must pass before any trade is executed**

When modifying compliance code: add tests first, verify math carefully, never weaken safety margins.

## External Dependencies

This project bridges two sibling repos via subprocess calls:
- `../../qlib_market_scanner` — Qlib-based FX signal scanner (run via `uv run`)
- `../../TradingAgents` — Multi-agent LLM decision engine (imported dynamically)
- `../../qlib_rd_agent` — Weekend factor research (run via subprocess)

These are NOT installed as packages — they are invoked as separate processes or imported via `sys.path` manipulation. Mock them in tests.

## Environment Variables

Required secrets (see `.env.example`):
- `MATCHTRADER_API_URL`, `MATCHTRADER_USERNAME`, `MATCHTRADER_PASSWORD` — broker API
- `ITICK_API_KEY`, `TRADERMADE_API_KEY` — FX data providers
- `DROPBOX_APP_KEY`, `DROPBOX_APP_SECRET`, `DROPBOX_REFRESH_TOKEN` — inter-project sync
- `LLM_API_KEY`, `LLM_BASE_URL` — LLM for TradingAgents
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` — alert notifications

**Never commit `.env` or hardcode secrets.**
