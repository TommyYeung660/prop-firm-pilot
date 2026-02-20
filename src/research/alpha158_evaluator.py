"""
Alpha158 factor-level time-series IC/IR evaluation for FX data.

Computes ~28 Alpha158 factor types (Kbar + Price + Rolling) using pure pandas,
calculates time-series IC/IR per factor per instrument, and classifies factors
as effective/weak/dead based on their predictive power.

Usage:
    evaluator = Alpha158Evaluator(data_dir="../../qlib_market_scanner/data/qlib_fx/")
    results = evaluator.run()
    evaluator.save_csv(results, "alpha158_ic_report.csv")
    print(evaluator.generate_report(results))
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field
from scipy import stats

# ── Data Structures ────────────────────────────────────────────────────────


class FactorResult(BaseModel):
    """Result of IC/IR evaluation for a single factor."""

    model_config = {"populate_by_name": True}

    name: str = Field(description="Factor name")
    ic_mean: float = Field(description="Mean IC across time")
    ic_std: float = Field(description="Std of IC series")
    ic_ir: float = Field(description="IC_mean / IC_std (Information Ratio)")
    ic_positive_ratio: float = Field(description="% of positive IC values")
    category: Literal["effective", "weak", "dead"] = Field(description="Factor classification")
    depends_on_volume: bool = Field(default=False, description="Factor depends on volume data")
    instruments_detail: dict[str, float] = Field(
        default_factory=dict, description="Per-instrument IC mean"
    )


# ── Qlib Binary Data Loader ────────────────────────────────────────────────


class QlibBinLoader:
    """Load Qlib binary .bin data files into pandas DataFrames.

    The qlib_fx directory structure:
    ├── calendars/day.txt       # List of trading dates
    ├── instruments/all.txt    # Instrument list with date ranges
    └── features/
        └── EURUSD/
            ├── open.day.bin
            ├── high.day.bin
            ├── low.day.bin
            ├── close.day.bin
            ├── volume.day.bin
            └── adj_close.day.bin
    """

    def __init__(self, data_dir: str | Path) -> None:
        """Initialize the loader with qlib_fx root directory.

        Args:
            data_dir: Path to qlib_fx root directory (contains features/, calendars/, instruments/).
        """
        self._data_dir = Path(data_dir).resolve()
        self._features_dir = self._data_dir / "features"

    def load_calendar(self) -> list[str]:
        """Read calendars/day.txt and return list of date strings.

        Returns:
            List of date strings in YYYY-MM-DD format.
        """
        calendar_path = self._data_dir / "calendars" / "day.txt"
        if not calendar_path.exists():
            logger.error("Calendar file not found: {}", calendar_path)
            raise FileNotFoundError(f"Calendar file not found: {calendar_path}")

        with open(calendar_path, encoding="utf-8") as f:
            dates = [line.strip() for line in f if line.strip()]

        logger.info("Loaded {} calendar dates", len(dates))
        return dates

    def load_instruments(self) -> dict[str, tuple[str, str]]:
        """Read instruments/all.txt and return instrument date ranges.

        Format: instrument<TAB>start_date<TAB>end_date

        Returns:
            Dict mapping instrument name to (start_date, end_date) tuple.
        """
        instruments_path = self._data_dir / "instruments" / "all.txt"
        if not instruments_path.exists():
            logger.error("Instruments file not found: {}", instruments_path)
            raise FileNotFoundError(f"Instruments file not found: {instruments_path}")

        instruments = {}
        with open(instruments_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    instrument, start_date, end_date = parts[0], parts[1], parts[2]
                    instruments[instrument] = (start_date, end_date)

        logger.info("Loaded {} instruments", len(instruments))
        return instruments

    def load_ohlcv(self, instrument: str) -> pd.DataFrame:
        """Read all .bin files for an instrument and return OHLCV DataFrame.

        Binary format: np.fromfile(path, dtype=np.float32)
        First value is 0.0 padding, skip it: data[1:]

        Returns:
            DataFrame with columns: open, high, low, close, volume, adj_close, vwap
            Index is calendar dates.
        """
        instrument_dir = self._features_dir / instrument
        if not instrument_dir.exists():
            logger.error("Instrument directory not found: {}", instrument_dir)
            raise FileNotFoundError(f"Instrument directory not found: {instrument_dir}")

        # Load calendar first
        calendar = self.load_calendar()

        # Load each feature file
        features = {}
        for feature in ["open", "high", "low", "close", "volume", "adj_close"]:
            bin_path = instrument_dir / f"{feature}.day.bin"
            if not bin_path.exists():
                logger.error("Feature file not found: {}", bin_path)
                raise FileNotFoundError(f"Feature file not found: {bin_path}")

            # Load binary data (float32), skip first padding value
            data = np.fromfile(bin_path, dtype=np.float32)[1:]
            if len(data) != len(calendar):
                logger.warning(
                    "Feature {} has {} values, calendar has {}",
                    feature,
                    len(data),
                    len(calendar),
                )
            features[feature] = data

        # Create DataFrame
        df = pd.DataFrame(features, index=calendar)
        df.index.name = "date"
        df.index = pd.to_datetime(df.index)

        # Calculate VWAP = (open + high + low + close) / 4
        df["vwap"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0

        logger.debug("Loaded {} OHLCV data for {}", len(df), instrument)
        return df

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all instruments and return dict of DataFrames.

        Returns:
            Dict mapping instrument name to its OHLCV DataFrame.
        """
        instruments = self.load_instruments()
        all_data = {}

        for instrument in instruments:
            try:
                df = self.load_ohlcv(instrument)
                all_data[instrument] = df
            except FileNotFoundError as e:
                logger.error("Failed to load {}: {}", instrument, e)

        logger.info("Loaded {} instruments successfully", len(all_data))
        return all_data


# ── Alpha158 Factor Calculator ─────────────────────────────────────────────


class Alpha158Calculator:
    """Calculate Alpha158 factors from OHLCV data using pure pandas.

    Implements the exact same formulas as Qlib's Alpha158DL.
    Volume-dependent factors are skipped (FX volume data is all 1.0).
    """

    ROLLING_WINDOWS = [5, 10, 20, 30, 60]

    def compute_kbar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Kbar factors (9 factors).

        Returns:
            DataFrame with 9 kbar factor columns.
        """
        open_p = df["open"]
        high = df["high"]
        low = df["low"]
        close = df["close"]
        hilo = high - low + 1e-12

        result = pd.DataFrame(index=df.index)
        result["KMID"] = (close - open_p) / open_p
        result["KLEN"] = (high - low) / open_p
        result["KMID2"] = (close - open_p) / hilo
        result["KUP"] = (high - np.maximum(open_p, close)) / open_p
        result["KUP2"] = (high - np.maximum(open_p, close)) / hilo
        result["KLOW"] = (np.minimum(open_p, close) - low) / open_p
        result["KLOW2"] = (np.minimum(open_p, close) - low) / hilo
        result["KSFT"] = (2 * close - high - low) / open_p
        result["KSFT2"] = (2 * close - high - low) / hilo

        return result

    def compute_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Price factors (4 factors).

        Returns:
            DataFrame with 4 price factor columns.
        """
        result = pd.DataFrame(index=df.index)
        result["OPEN0"] = df["open"] / df["close"]
        result["HIGH0"] = df["high"] / df["close"]
        result["LOW0"] = df["low"] / df["close"]
        result["VWAP0"] = df["vwap"] / df["close"]

        return result

    def _linear_regression(
        self, series: pd.Series, window: int
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Perform linear regression on rolling window.

        Returns:
            Tuple of (beta, rsqr, residual) series.
            Beta is the slope / value, R2 is the R-squared, residual is the residual / value.
        """

        def fit_slope(s):
            if np.any(np.isnan(s)):
                return np.nan
            try:
                x_local = np.arange(len(s))
                coeffs = np.polyfit(x_local, s, 1)
                return coeffs[0]  # slope
            except (np.linalg.LinAlgError, TypeError):
                return np.nan

        def fit_r2(s):
            if np.any(np.isnan(s)):
                return np.nan
            try:
                x_local = np.arange(len(s))
                coeffs = np.polyfit(x_local, s, 1)
                y_pred = np.polyval(coeffs, x_local)
                ss_res = np.sum((s - y_pred) ** 2)
                ss_tot = np.sum((s - np.mean(s)) ** 2)
                if ss_tot == 0:
                    return np.nan
                return 1 - ss_res / ss_tot
            except (np.linalg.LinAlgError, TypeError):
                return np.nan

        def fit_residual(s):
            if np.any(np.isnan(s)):
                return np.nan
            try:
                x_local = np.arange(len(s))
                coeffs = np.polyfit(x_local, s, 1)
                y_pred = np.polyval(coeffs, x_local)
                return s[-1] - y_pred[-1]  # residual at last point
            except (np.linalg.LinAlgError, TypeError):
                return np.nan

        beta = series.rolling(window, min_periods=2).apply(fit_slope, raw=False)
        rsqr = series.rolling(window, min_periods=2).apply(fit_r2, raw=False)
        residual = series.rolling(window, min_periods=2).apply(fit_residual, raw=False)

        return beta, rsqr, residual

    def compute_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all rolling factors.

        Each factor type is computed for each window size.
        Total factors: 20 types × 5 windows = 100 factors.

        Returns:
            DataFrame with all rolling factor columns.
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]

        result = pd.DataFrame(index=df.index)

        for d in self.ROLLING_WINDOWS:
            # ROC: Rate of Change
            result[f"ROC{d}"] = close.shift(d) / close

            # MA: Moving Average
            result[f"MA{d}"] = close.rolling(d, min_periods=2).mean() / close

            # STD: Standard Deviation
            result[f"STD{d}"] = close.rolling(d, min_periods=2).std() / close

            # BETA, RSQR, RESI: Linear regression
            beta, rsqr, residual = self._linear_regression(close, d)
            result[f"BETA{d}"] = beta / close
            result[f"RSQR{d}"] = rsqr
            result[f"RESI{d}"] = residual / close

            # MAX, MIN
            result[f"MAX{d}"] = high.rolling(d, min_periods=1).max() / close
            result[f"MIN{d}"] = low.rolling(d, min_periods=1).min() / close

            # QTLUQUANTILE (0.8)
            result[f"QTLU{d}"] = close.rolling(d, min_periods=1).quantile(0.8) / close

            # QTLDQUANTILE (0.2)
            result[f"QTLD{d}"] = close.rolling(d, min_periods=1).quantile(0.2) / close

            # RANK: Percentile rank of current value in rolling window
            def rank_func(s):
                if len(s) < 2:
                    return 0.5
                return (s.iloc[-1] > s[:-1]).sum() / (len(s) - 1)

            result[f"RANK{d}"] = close.rolling(d, min_periods=2).apply(rank_func, raw=False)

            # RSV: Relative Strength Index variant
            max_high = high.rolling(d, min_periods=1).max()
            min_low = low.rolling(d, min_periods=1).min()
            result[f"RSV{d}"] = (close - min_low) / (max_high - min_low + 1e-12)

            # IMAX, IMIN
            def argmax_func(s):
                if len(s) < 2:
                    return 0.0
                return np.argmax(s.values) / len(s)

            def argmin_func(s):
                if len(s) < 2:
                    return 0.0
                return np.argmin(s.values) / len(s)

            result[f"IMAX{d}"] = high.rolling(d, min_periods=2).apply(argmax_func, raw=False)
            result[f"IMIN{d}"] = low.rolling(d, min_periods=2).apply(argmin_func, raw=False)
            result[f"IMXD{d}"] = result[f"IMAX{d}"] - result[f"IMIN{d}"]

            # CNTP, CNTN, CNTD: Count of positive/negative changes
            # Matches Qlib: Mean($close>Ref($close,1), d) — fraction of up-days
            up_flag = (close > close.shift(1)).astype(float)
            down_flag = (close < close.shift(1)).astype(float)

            result[f"CNTP{d}"] = up_flag.rolling(d, min_periods=2).mean()
            result[f"CNTN{d}"] = down_flag.rolling(d, min_periods=2).mean()
            result[f"CNTD{d}"] = result[f"CNTP{d}"] - result[f"CNTN{d}"]

            # SUMP, SUMN, SUMD: Sum of positive/negative changes
            # Matches Qlib: Sum(Greater($close-Ref($close,1), 0), d) / (Sum(Abs(...), d) + eps)
            price_diff = close - close.shift(1)
            pos_change = price_diff.clip(lower=0)  # Greater(diff, 0)
            neg_change = (-price_diff).clip(lower=0)  # Greater(Ref-close, 0)
            abs_change = price_diff.abs()

            sump = pos_change.rolling(d, min_periods=2).sum()
            sumn = neg_change.rolling(d, min_periods=2).sum()
            sum_abs = abs_change.rolling(d, min_periods=2).sum()

            result[f"SUMP{d}"] = sump / (sum_abs + 1e-12)
            result[f"SUMN{d}"] = sumn / (sum_abs + 1e-12)
            result[f"SUMD{d}"] = (sump - sumn) / (sum_abs + 1e-12)

        return result

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all Alpha158 factors from OHLCV data.

        Combines Kbar, Price, and Rolling factors.

        Args:
            df: DataFrame with columns: open, high, low, close, volume, adj_close, vwap

        Returns:
            DataFrame with all factor columns.
        """
        logger.info("Computing Alpha158 factors for {} rows", len(df))

        kbar = self.compute_kbar(df)
        price = self.compute_price(df)
        rolling = self.compute_rolling(df)

        result = pd.concat([kbar, price, rolling], axis=1)

        logger.info("Computed {} total factors", result.shape[1])
        return result


# ── Alpha158 Evaluator ─────────────────────────────────────────────────────


class Alpha158Evaluator:
    """Evaluate Alpha158 factor IC/IR on FX data using time-series IC.

    Uses rolling Spearman rank correlation between factor values and forward returns.
    Averages IC across instruments to get a robust time-series IC estimate.

    Usage:
        evaluator = Alpha158Evaluator(data_dir="../../qlib_market_scanner/data/qlib_fx/")
        results = evaluator.run()
        evaluator.save_csv(results, "alpha158_ic_report.csv")
        print(evaluator.generate_report(results))
    """

    def __init__(
        self,
        data_dir: str | Path,
        forward_period: int = 1,
        ic_window: int = 20,
    ) -> None:
        """Initialize the evaluator.

        Args:
            data_dir: Path to qlib_fx root directory.
            forward_period: Number of days ahead for return calculation (default 1).
            ic_window: Rolling window size for IC calculation (default 20).
        """
        self._data_dir = Path(data_dir).resolve()
        self._forward_period = forward_period
        self._ic_window = ic_window
        self._loader = QlibBinLoader(data_dir)
        self._calculator = Alpha158Calculator()

        logger.info(
            "Alpha158Evaluator initialized with forward_period={}, ic_window={}",
            forward_period,
            ic_window,
        )

    def run(self) -> list[FactorResult]:
        """Run full evaluation pipeline.

        Returns:
            List of FactorResult objects, one per factor.
        """
        logger.info("Starting Alpha158 evaluation pipeline")

        # Load all instruments
        all_data = self._loader.load_all()
        if not all_data:
            logger.error("No instruments loaded")
            return []

        instrument_names = list(all_data.keys())
        logger.info("Loaded {} instruments: {}", len(instrument_names), ", ".join(instrument_names))

        # Compute factors and forward returns for each instrument
        factor_data: dict[str, pd.DataFrame] = {}
        return_data: dict[str, pd.Series] = {}

        for instrument, df in all_data.items():
            logger.info("Processing instrument: {}", instrument)

            # Compute factors
            factors = self._calculator.compute_all(df)

            # Compute forward returns
            forward_returns = self._compute_forward_returns(df)

            # Align indices (remove NaN from forward returns at the end)
            valid_idx = forward_returns.dropna().index
            factors = factors.loc[valid_idx]
            forward_returns = forward_returns.loc[valid_idx]

            factor_data[instrument] = factors
            return_data[instrument] = forward_returns

        # Get factor names from first instrument
        factor_names = list(factor_data[instrument_names[0]].columns)
        logger.info("Evaluating {} factors", len(factor_names))

        results = []

        # Evaluate each factor
        for factor_name in factor_names:
            logger.debug("Evaluating factor: {}", factor_name)

            # Collect factor values and returns across all instruments
            factor_values = {inst: factor_data[inst][factor_name] for inst in instrument_names}

            # Compute time-series IC
            ic_mean, ic_std, ic_positive_ratio, per_instrument_ic = self._compute_timeseries_ic(
                factor_values, return_data
            )

            # Calculate IC IR
            ic_ir = ic_mean / ic_std if ic_std != 0 else 0.0

            # Classify factor
            category = self._classify_factor(ic_mean)

            # Check if factor depends on volume (skip these in FX)
            depends_on_volume = any(
                prefix in factor_name
                for prefix in ["CORR", "CORD", "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN", "VSUMD"]
            )

            result = FactorResult(
                name=factor_name,
                ic_mean=float(ic_mean),
                ic_std=float(ic_std),
                ic_ir=float(ic_ir),
                ic_positive_ratio=float(ic_positive_ratio),
                category=category,
                depends_on_volume=depends_on_volume,
                instruments_detail={k: float(v) for k, v in per_instrument_ic.items()},
            )
            results.append(result)

        logger.info("Evaluation complete: {} factors evaluated", len(results))
        return results

    def _compute_forward_returns(self, df: pd.DataFrame) -> pd.Series:
        """Compute forward log returns.

        Returns:
            Series of log returns: log(close.shift(-forward_period) / close)
        """
        close = df["close"]
        future_close = close.shift(-self._forward_period)
        forward_returns = np.log(future_close / close)

        return forward_returns

    def _compute_timeseries_ic(
        self,
        factor_values: dict[str, pd.Series],
        forward_returns: dict[str, pd.Series],
    ) -> tuple[float, float, float, dict[str, float]]:
        """Compute rolling Spearman IC for each instrument and average.

        For each instrument:
        1. Compute rolling Spearman correlation between factor and forward return
        2. Average IC series across instruments
        3. Compute mean, std, and positive ratio of the averaged IC series

        Args:
            factor_values: Dict mapping instrument name to factor values Series.
            forward_returns: Dict mapping instrument name to forward returns Series.

        Returns:
            Tuple of (ic_mean, ic_std, ic_positive_ratio, per_instrument_ic_dict).
        """
        ic_series_list = []
        per_instrument_ic = {}

        for instrument in factor_values:
            factor = factor_values[instrument]
            returns = forward_returns[instrument]

            # Align indices
            common_idx = factor.index.intersection(returns.index)
            factor = factor.loc[common_idx]
            returns = returns.loc[common_idx]

            # Remove NaN
            valid_mask = ~(factor.isna() | returns.isna())
            factor = factor[valid_mask]
            returns = returns[valid_mask]

            if len(factor) < self._ic_window + 1:
                logger.warning("Insufficient data for {} in instrument {}", factor.name, instrument)
                per_instrument_ic[instrument] = 0.0
                continue

            # Compute rolling Spearman correlation
            def rolling_spearman(s_factor, s_returns, window):
                """Compute rolling Spearman correlation."""
                ic_values = []
                for i in range(window, len(s_factor)):
                    window_factor = s_factor.iloc[i - window : i].values
                    window_return = s_returns.iloc[i - window : i].values

                    if len(window_factor) < 3 or len(window_return) < 3:
                        ic_values.append(np.nan)
                        continue

                    try:
                        corr, _ = stats.spearmanr(window_factor, window_return)
                        ic_values.append(corr)
                    except (ValueError, RuntimeWarning):
                        ic_values.append(np.nan)

                return np.array(ic_values)

            ic_series = rolling_spearman(factor, returns, self._ic_window)
            ic_series_list.append(pd.Series(ic_series))

            # Store per-instrument mean IC
            mean_ic = np.nanmean(ic_series)
            per_instrument_ic[instrument] = mean_ic

        # Average IC series across instruments
        if not ic_series_list:
            return 0.0, 0.0, 0.0, per_instrument_ic

        # Find minimum length
        min_len = min(len(s) for s in ic_series_list)

        # Truncate all to same length and average
        truncated_series = [s.iloc[-min_len:] for s in ic_series_list]
        stacked = pd.concat(truncated_series, axis=1)
        averaged_ic = stacked.mean(axis=1)

        # Compute stats
        ic_mean = averaged_ic.mean()
        ic_std = averaged_ic.std()
        ic_positive_ratio = (
            (averaged_ic > 0).sum() / len(averaged_ic) if len(averaged_ic) > 0 else 0.0
        )

        return ic_mean, ic_std, ic_positive_ratio, per_instrument_ic

    def _classify_factor(self, ic_mean: float) -> Literal["effective", "weak", "dead"]:
        """Classify factor based on absolute IC mean.

        Args:
            ic_mean: Mean IC value.

        Returns:
            One of "effective", "weak", "dead".
        """
        abs_ic = abs(ic_mean)
        if abs_ic > 0.02:
            return "effective"
        elif abs_ic > 0.01:
            return "weak"
        else:
            return "dead"

    def generate_report(self, results: list[FactorResult]) -> str:
        """Generate console summary string.

        Args:
            results: List of FactorResult objects.

        Returns:
            Formatted report string.
        """
        lines = []

        lines.append("═" * 63)
        lines.append("  Alpha158 FX Factor IC/IR Report")
        lines.append(f"  Method: Time-series IC (rolling Spearman, window={self._ic_window})")
        lines.append("═" * 63)
        lines.append("")

        # Group by category
        effective = [r for r in results if r.category == "effective" and not r.depends_on_volume]
        weak = [r for r in results if r.category == "weak" and not r.depends_on_volume]
        dead = [r for r in results if r.category == "dead" and not r.depends_on_volume]
        volume_factors = [r for r in results if r.depends_on_volume]

        # Sort by IC IR within each category
        effective.sort(key=lambda x: x.ic_ir, reverse=True)
        weak.sort(key=lambda x: x.ic_ir, reverse=True)
        dead.sort(key=lambda x: x.ic_mean, reverse=True)

        # Helper to format category
        def format_category(name: str, factors: list[FactorResult]) -> list[str]:
            if not factors:
                return []

            cat_lines = []
            cat_lines.append(f"Category: {name}")
            cat_lines.append("─" * 63)

            for r in factors:
                ic_sign = "+" if r.ic_mean >= 0 else ""
                ic_pct = f"{r.ic_positive_ratio:.0%}"
                cat_lines.append(
                    f"  {r.name:8s}  IC={ic_sign}{r.ic_mean:.3f}  IR={r.ic_ir:.2f}  IC+={ic_pct}"
                )

            cat_lines.append("")
            return cat_lines

        lines.extend(format_category("EFFECTIVE (|IC| > 0.02)", effective))
        lines.extend(format_category("WEAK (0.01 < |IC| ≤ 0.02)", weak))
        lines.extend(format_category("DEAD (|IC| ≤ 0.01)", dead))

        # Volume factors note
        if volume_factors:
            lines.append(
                f"Skipped {len(volume_factors)} volume-dependent factors (FX volume is fake):"
            )
            lines.append("  " + ", ".join([r.name for r in volume_factors[:10]]))
            if len(volume_factors) > 10:
                lines.append(f"  ... and {len(volume_factors) - 10} more")
            lines.append("")

        # Summary
        total = len(results) - len(volume_factors)
        n_eff, n_weak, n_dead = len(effective), len(weak), len(dead)
        lines.append(
            f"Summary: {n_eff} effective, {n_weak} weak, {n_dead} dead out of {total} total factors"
        )

        return "\n".join(lines)

    def save_csv(self, results: list[FactorResult], output_path: str | Path) -> None:
        """Save results to CSV file.

        Args:
            results: List of FactorResult objects.
            output_path: Path to output CSV file.
        """
        output_path = Path(output_path)

        # Prepare data for CSV
        rows = []
        for r in results:
            row = {
                "name": r.name,
                "ic_mean": r.ic_mean,
                "ic_std": r.ic_std,
                "ic_ir": r.ic_ir,
                "ic_positive_ratio": r.ic_positive_ratio,
                "category": r.category,
                "depends_on_volume": r.depends_on_volume,
            }
            # Add per-instrument IC
            for inst, ic in r.instruments_detail.items():
                row[f"ic_{inst}"] = ic
            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by category and IC IR
        category_order = {"effective": 0, "weak": 1, "dead": 2}
        df["category_order"] = df["category"].map(category_order)
        df = df.sort_values(["category_order", "ic_ir"], ascending=[True, False])
        df = df.drop(columns=["category_order"])

        df.to_csv(output_path, index=False)
        logger.info("Saved results to {}", output_path)
