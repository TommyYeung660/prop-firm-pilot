"""
EODHD FX Intraday Data Completeness Checker

Diagnoses whether EODHD data quality issues are causing tactical gate
(atr_regime, data_freshness) failures that block trade execution.

Checks:
  1. API connectivity & response structure
  2. Bar counts vs expected (gaps, missing bars)
  3. Null/zero field analysis (open, high, low, close, volume)
  4. Timestamp freshness (latest bar age)
  5. ATR regime simulation (mirrors TacticalValidator logic)
  6. Time gap analysis (missing hourly/5min slots)

Usage:
    python scripts/check_eodhd_data.py
    python scripts/check_eodhd_data.py --symbols EURUSD GBPUSD
    python scripts/check_eodhd_data.py --symbols EURUSD --verbose
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd

# ── Add project root to path ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file if present
from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env")

from src.data.fx_data_fetcher import EodhdProvider  # noqa: E402
from src.decision.tactical_validator import compute_atr  # noqa: E402


# ── Config (mirrors scheduler._fetch_tactical_data) ─────────────────────────

DEFAULT_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
ATR_PERIOD = 14
ATR_MIN_RATIO = 0.5
ATR_MAX_RATIO = 2.5
DATA_MAX_AGE_SECONDS = 600  # 10 minutes

# Scheduler fetches: 6h of 5min bars, 30h of 1h bars
LOOKBACK_5MIN_HOURS = 6
LOOKBACK_1H_HOURS = 30

# Expected bar counts (approximate, FX market hours only)
# 5min: 6h × 12 bars/h = 72 bars (if market open)
# 1h: 30h = 30 bars (if market open)
EXPECTED_5MIN_BARS_MIN = 30  # Minimum acceptable
EXPECTED_1H_BARS_MIN = 15  # ATR(14) needs at least 15


def _fmt_age(seconds: float) -> str:
    """Format seconds as human-readable age string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"


def _check_nulls_zeros(df: pd.DataFrame, interval: str) -> list[str]:
    """Check for null or zero values in OHLC columns."""
    issues: list[str] = []
    ohlc_cols = ["open", "high", "low", "close"]

    for col in ohlc_cols:
        if col not in df.columns:
            issues.append(f"  [!] Missing column: {col}")
            continue
        null_count = df[col].isna().sum()
        zero_count = (df[col] == 0).sum()
        if null_count > 0:
            issues.append(f"  [!] {interval} {col}: {null_count} null values")
        if zero_count > 0:
            issues.append(f"  [!] {interval} {col}: {zero_count} zero values")

    return issues


def _check_time_gaps(df: pd.DataFrame, interval: str) -> list[str]:
    """Check for unexpected time gaps in bar data."""
    issues: list[str] = []
    if len(df) < 2 or "datetime" not in df.columns:
        return issues

    diffs = df["datetime"].diff().dropna()

    if interval == "5min":
        expected_gap = pd.Timedelta(minutes=5)
        max_acceptable = pd.Timedelta(minutes=15)  # Allow up to 3× gap
    else:  # 1h
        expected_gap = pd.Timedelta(hours=1)
        max_acceptable = pd.Timedelta(hours=3)  # Allow up to 3× gap

    big_gaps = diffs[diffs > max_acceptable]
    if len(big_gaps) > 0:
        issues.append(f"  [!] {interval}: {len(big_gaps)} gaps > {max_acceptable}")
        for idx in big_gaps.index[:5]:  # Show first 5
            gap_start = df["datetime"].iloc[idx - 1]
            gap_end = df["datetime"].iloc[idx]
            gap_size = big_gaps[idx]
            issues.append(f"    Gap: {gap_start} -> {gap_end} ({gap_size})")
        if len(big_gaps) > 5:
            issues.append(f"    ... and {len(big_gaps) - 5} more gaps")

    return issues


def _simulate_atr_regime(df_1h: pd.DataFrame) -> dict:
    """Simulate ATR regime gate logic from TacticalValidator."""
    result = {
        "has_enough_data": False,
        "current_atr": float("nan"),
        "median_atr": float("nan"),
        "atr_ratio": float("nan"),
        "would_pass": False,
        "detail": "",
    }

    if df_1h.empty:
        result["detail"] = "No 1H data available (pass-through in validator)"
        return result

    if len(df_1h) < ATR_PERIOD + 1:
        result["detail"] = f"Only {len(df_1h)} bars, need {ATR_PERIOD + 1} for ATR({ATR_PERIOD})"
        return result

    current_atr = compute_atr(df_1h, period=ATR_PERIOD)
    if pd.isna(current_atr):
        result["detail"] = "ATR computation returned NaN"
        return result

    result["has_enough_data"] = True
    result["current_atr"] = current_atr

    # Compute median ATR (same as TacticalValidator.check_hard_gates)
    all_atrs: list[float] = []
    for i in range(ATR_PERIOD + 1, len(df_1h) + 1):
        window = df_1h.iloc[:i]
        a = compute_atr(window, period=ATR_PERIOD)
        if not pd.isna(a):
            all_atrs.append(a)

    median_atr = float(pd.Series(all_atrs).median()) if all_atrs else current_atr
    result["median_atr"] = median_atr

    ratio = current_atr / median_atr if median_atr > 0 else float("inf")
    result["atr_ratio"] = ratio
    result["would_pass"] = ATR_MIN_RATIO < ratio < ATR_MAX_RATIO
    result["detail"] = (
        f"ATR ratio={ratio:.3f}, range=[{ATR_MIN_RATIO}, {ATR_MAX_RATIO}] -> "
        f"{'PASS' if result['would_pass'] else 'FAIL'}"
    )
    return result


async def check_symbol(
    provider: EodhdProvider,
    symbol: str,
    client: httpx.AsyncClient,
    verbose: bool = False,
) -> dict:
    """Run all data completeness checks for a single symbol."""
    now = datetime.now(timezone.utc)
    start_5min = (now - timedelta(hours=LOOKBACK_5MIN_HOURS)).date()
    start_1h = (now - timedelta(hours=LOOKBACK_1H_HOURS)).date()
    end_date = now.date()

    result = {
        "symbol": symbol,
        "check_time": now.isoformat(),
        "issues": [],
        "bars_5min_count": 0,
        "bars_1h_count": 0,
        "latest_5min_age_s": None,
        "latest_1h_age_s": None,
        "atr_regime": {},
        "status": "UNKNOWN",
    }

    # ── Fetch 5min bars ──────────────────────────────────────────────────
    try:
        bars_5min = await provider.fetch_bars(symbol, start_5min, end_date, client, interval="5min")
    except Exception as e:
        result["issues"].append(f"  [X] 5min fetch failed: {e}")
        bars_5min = pd.DataFrame()

    # ── Fetch 1h bars ────────────────────────────────────────────────────
    try:
        bars_1h = await provider.fetch_bars(symbol, start_1h, end_date, client, interval="1h")
    except Exception as e:
        result["issues"].append(f"  [X] 1h fetch failed: {e}")
        bars_1h = pd.DataFrame()

    result["bars_5min_count"] = len(bars_5min)
    result["bars_1h_count"] = len(bars_1h)

    # ── Bar count check ──────────────────────────────────────────────────
    if len(bars_5min) < EXPECTED_5MIN_BARS_MIN:
        result["issues"].append(
            f"  [!] 5min bars: {len(bars_5min)} < {EXPECTED_5MIN_BARS_MIN} expected minimum"
        )
    if len(bars_1h) < EXPECTED_1H_BARS_MIN:
        result["issues"].append(
            f"  [!] 1h bars: {len(bars_1h)} < {EXPECTED_1H_BARS_MIN} expected minimum "
            f"(ATR({ATR_PERIOD}) needs ≥{ATR_PERIOD + 1})"
        )

    # ── Freshness check ──────────────────────────────────────────────────
    if not bars_5min.empty and "datetime" in bars_5min.columns:
        latest_5min = bars_5min["datetime"].max()
        age_5min = (now - latest_5min.to_pydatetime().replace(tzinfo=timezone.utc)).total_seconds()
        result["latest_5min_age_s"] = age_5min
        if age_5min > DATA_MAX_AGE_SECONDS:
            result["issues"].append(
                f"  [!] 5min latest bar age: {_fmt_age(age_5min)} > {DATA_MAX_AGE_SECONDS}s limit"
            )

    if not bars_1h.empty and "datetime" in bars_1h.columns:
        latest_1h = bars_1h["datetime"].max()
        age_1h = (now - latest_1h.to_pydatetime().replace(tzinfo=timezone.utc)).total_seconds()
        result["latest_1h_age_s"] = age_1h
        if age_1h > 7200:  # 1h bars can be up to 2h old normally
            result["issues"].append(f"  [!] 1h latest bar age: {_fmt_age(age_1h)} (>2h old)")

    # ── Null/zero analysis ───────────────────────────────────────────────
    if not bars_5min.empty:
        result["issues"].extend(_check_nulls_zeros(bars_5min, "5min"))
    if not bars_1h.empty:
        result["issues"].extend(_check_nulls_zeros(bars_1h, "1h"))

    # ── Time gap analysis ────────────────────────────────────────────────
    if not bars_5min.empty:
        result["issues"].extend(_check_time_gaps(bars_5min, "5min"))
    if not bars_1h.empty:
        result["issues"].extend(_check_time_gaps(bars_1h, "1h"))

    # ── ATR regime simulation ────────────────────────────────────────────
    atr_result = _simulate_atr_regime(bars_1h)
    result["atr_regime"] = atr_result

    if atr_result.get("has_enough_data") and not atr_result.get("would_pass"):
        result["issues"].append(f"  [X] ATR regime would FAIL: {atr_result['detail']}")

    # ── Overall status ───────────────────────────────────────────────────
    critical_issues = [i for i in result["issues"] if "[X]" in i]
    warning_issues = [i for i in result["issues"] if "[!]" in i]

    if critical_issues:
        result["status"] = "CRITICAL"
    elif warning_issues:
        result["status"] = "WARNING"
    else:
        result["status"] = "OK"

    # ── Verbose output ───────────────────────────────────────────────────
    if verbose and not bars_5min.empty:
        print(f"\n  [5min bars -- last 5 rows]")
        print(bars_5min.tail(5).to_string(index=False))
    if verbose and not bars_1h.empty:
        print(f"\n  [1h bars -- last 5 rows]")
        print(bars_1h.tail(5).to_string(index=False))

    return result


async def main(symbols: list[str] | None = None, verbose: bool = False) -> None:
    """Run EODHD data completeness check for all configured symbols."""
    api_key = os.getenv("EODHD_API_KEY", "")
    if not api_key:
        print("[X] EODHD_API_KEY not set. Set it in .env or environment.")
        sys.exit(1)

    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    provider = EodhdProvider(api_key=api_key)
    now = datetime.now(timezone.utc)

    print("=" * 70)
    print(f"EODHD FX Intraday Data Completeness Report")
    print(f"Time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')} (weekday: {now.strftime('%A')})")
    print(f"Symbols: {', '.join(symbols)}")
    print(
        f"Config: ATR period={ATR_PERIOD}, ATR range=[{ATR_MIN_RATIO}, {ATR_MAX_RATIO}], "
        f"freshness limit={DATA_MAX_AGE_SECONDS}s"
    )
    print("=" * 70)

    # Check if FX market is likely open
    weekday = now.weekday()  # 0=Mon, 6=Sun
    hour = now.hour
    if weekday == 5 or (weekday == 4 and hour >= 22) or (weekday == 6 and hour < 22):
        print("\n[!] FX market is likely CLOSED (weekend). Data may be stale.\n")

    results = []
    async with httpx.AsyncClient() as client:
        for symbol in symbols:
            print(f"\n-- {symbol} {'-' * (55 - len(symbol))}")
            result = await check_symbol(provider, symbol, client, verbose=verbose)
            results.append(result)

            # Print result
            status_icon = {"OK": "[OK]", "WARNING": "[!]", "CRITICAL": "[X]"}.get(result["status"], "?")
            print(f"  Status: {status_icon} {result['status']}")
            print(
                f"  5min bars: {result['bars_5min_count']} "
                f"(lookback: {LOOKBACK_5MIN_HOURS}h, need >={EXPECTED_5MIN_BARS_MIN})"
            )
            print(
                f"  1h bars:   {result['bars_1h_count']} "
                f"(lookback: {LOOKBACK_1H_HOURS}h, need >={EXPECTED_1H_BARS_MIN})"
            )

            if result["latest_5min_age_s"] is not None:
                print(f"  Latest 5min bar age: {_fmt_age(result['latest_5min_age_s'])}")
            else:
                print(f"  Latest 5min bar age: N/A (no data)")

            if result["latest_1h_age_s"] is not None:
                print(f"  Latest 1h bar age:   {_fmt_age(result['latest_1h_age_s'])}")
            else:
                print(f"  Latest 1h bar age:   N/A (no data)")

            atr = result["atr_regime"]
            if atr.get("has_enough_data"):
                print(
                    f"  ATR regime: ratio={atr['atr_ratio']:.3f} "
                    f"(current={atr['current_atr']:.6f}, median={atr['median_atr']:.6f}) "
                    f"-> {'PASS' if atr['would_pass'] else 'FAIL'}"
                )
            else:
                print(f"  ATR regime: {atr.get('detail', 'N/A')}")

            if result["issues"]:
                print(f"  Issues ({len(result['issues'])}):")
                for issue in result["issues"]:
                    print(issue)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    ok_count = sum(1 for r in results if r["status"] == "OK")
    warn_count = sum(1 for r in results if r["status"] == "WARNING")
    crit_count = sum(1 for r in results if r["status"] == "CRITICAL")
    print(f"  [OK] OK: {ok_count}  [!] Warning: {warn_count}  [X] Critical: {crit_count}")

    # Diagnosis for atr_regime / data_freshness gate failures
    print(f"\n-- Gate Failure Diagnosis --")

    any_atr_fail = any(
        r["atr_regime"].get("has_enough_data") and not r["atr_regime"].get("would_pass")
        for r in results
    )
    any_atr_nodata = any(
        not r["atr_regime"].get("has_enough_data") and r["bars_1h_count"] > 0 for r in results
    )
    any_freshness_fail = any(
        r.get("latest_5min_age_s") is not None and r["latest_5min_age_s"] > DATA_MAX_AGE_SECONDS
        for r in results
    )

    if any_atr_fail:
        print("  [X] atr_regime: Would FAIL for some symbols.")
        print("    -> ATR ratio outside [0.5, 2.5] -- market may be abnormally quiet or volatile.")
        print("    -> This is a MARKET CONDITION, not a data issue.")
    elif any_atr_nodata:
        print("  [!] atr_regime: Insufficient 1H bars for ATR computation.")
        print("    -> EODHD may not be returning enough historical bars.")
        print("    -> Check if API subscription includes intraday data.")
    else:
        print("  [OK] atr_regime: Would PASS for all symbols (data sufficient, ratio in range).")

    print()
    print("  NOTE: data_freshness gate uses MatchTrader quote timestamp, NOT EODHD bar time.")
    print("  If data_freshness fails, check MatchTrader API connectivity & quote availability.")

    if any_freshness_fail:
        print(f"\n  [!] EODHD bar staleness detected (>10min) -- but this does NOT directly")
        print(f"    cause data_freshness gate failure (which uses broker quote timestamp).")
        print(f"    However, stale EODHD bars affect ATR/EMA/RSI accuracy.")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EODHD FX data completeness checker")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help=f"FX pairs to check (default: {DEFAULT_SYMBOLS})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show raw bar data samples",
    )
    args = parser.parse_args()
    asyncio.run(main(symbols=args.symbols, verbose=args.verbose))
