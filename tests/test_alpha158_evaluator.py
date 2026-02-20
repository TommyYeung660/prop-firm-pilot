"""
Tests for Alpha158Evaluator — factor computation and IC/IR evaluation.

Covers:
- QlibBinLoader: reading binary data and calendar files
- Alpha158Calculator: Kbar, Price, Rolling factor computation
- Alpha158Evaluator: IC/IR calculation, classification, report generation

Uses synthetic data (no real FX data required). Binary files are written
to tmp_path for QlibBinLoader tests.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research.alpha158_evaluator import (
    Alpha158Calculator,
    Alpha158Evaluator,
    FactorResult,
    QlibBinLoader,
)

# ── Helpers ─────────────────────────────────────────────────────────────────


def make_synthetic_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing.

    Creates realistic-looking price data with a slight upward trend
    and random noise.

    Args:
        n: Number of trading days.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with open, high, low, close, volume, adj_close, vwap columns.
    """
    rng = np.random.default_rng(seed)

    # Generate close prices with random walk + slight trend
    returns = rng.normal(0.0002, 0.01, n)
    close = 1.10 * np.cumprod(1 + returns)

    # Generate OHLC from close
    spread = rng.uniform(0.001, 0.005, n)
    open_p = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_p, close) * (1 + spread)
    low = np.minimum(open_p, close) * (1 - spread)
    volume = np.ones(n, dtype=np.float32)  # Fake volume for FX
    adj_close = close.copy()
    vwap = (open_p + high + low + close) / 4.0

    dates = pd.bdate_range("2022-01-01", periods=n)
    return pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "adj_close": adj_close,
            "vwap": vwap,
        },
        index=dates,
    )


def write_qlib_bin_data(
    data_dir: Path,
    instruments: dict[str, pd.DataFrame],
    calendar: list[str],
) -> None:
    """Write synthetic data in Qlib binary format.

    Creates the directory structure and .bin files that QlibBinLoader expects.

    Args:
        data_dir: Root directory for qlib_fx data.
        instruments: Dict mapping instrument name to OHLCV DataFrame.
        calendar: List of date strings for the calendar file.
    """
    # Create directories
    (data_dir / "calendars").mkdir(parents=True, exist_ok=True)
    (data_dir / "instruments").mkdir(parents=True, exist_ok=True)

    # Write calendar
    with open(data_dir / "calendars" / "day.txt", "w") as f:
        for date in calendar:
            f.write(f"{date}\n")

    # Write instruments
    with open(data_dir / "instruments" / "all.txt", "w") as f:
        for name in instruments:
            f.write(f"{name}\t{calendar[0]}\t{calendar[-1]}\n")

    # Write feature files
    for name, df in instruments.items():
        inst_dir = data_dir / "features" / name
        inst_dir.mkdir(parents=True, exist_ok=True)

        for col in ["open", "high", "low", "close", "volume", "adj_close"]:
            # Qlib format: first value is 0.0 padding, then real data
            values = np.concatenate(
                [np.array([0.0], dtype=np.float32), df[col].values.astype(np.float32)]
            )
            values.tofile(inst_dir / f"{col}.day.bin")


# ── QlibBinLoader Tests ────────────────────────────────────────────────────


class TestQlibBinLoader:
    """Tests for binary data loading."""

    @pytest.fixture
    def loader_data(self, tmp_path: Path) -> tuple[QlibBinLoader, pd.DataFrame]:
        """Create a QlibBinLoader with synthetic data."""
        df = make_synthetic_ohlcv(100)
        calendar = [d.strftime("%Y-%m-%d") for d in df.index]
        instruments = {"EURUSD": df, "GBPUSD": df}
        data_dir = tmp_path / "qlib_fx"
        write_qlib_bin_data(data_dir, instruments, calendar)
        loader = QlibBinLoader(data_dir)
        return loader, df

    def test_load_calendar(self, loader_data: tuple[QlibBinLoader, pd.DataFrame]) -> None:
        """Calendar loads correct number of dates."""
        loader, df = loader_data
        calendar = loader.load_calendar()
        assert len(calendar) == len(df)
        assert calendar[0] == df.index[0].strftime("%Y-%m-%d")

    def test_load_instruments(self, loader_data: tuple[QlibBinLoader, pd.DataFrame]) -> None:
        """Instruments file loads correctly."""
        loader, _ = loader_data
        instruments = loader.load_instruments()
        assert "EURUSD" in instruments
        assert "GBPUSD" in instruments
        assert len(instruments) == 2

    def test_load_ohlcv(self, loader_data: tuple[QlibBinLoader, pd.DataFrame]) -> None:
        """OHLCV data loads with correct shape and VWAP is computed."""
        loader, df = loader_data
        result = loader.load_ohlcv("EURUSD")
        assert len(result) == len(df)
        assert "vwap" in result.columns
        assert "open" in result.columns
        assert "close" in result.columns
        # Verify VWAP calculation
        expected_vwap = (result["open"] + result["high"] + result["low"] + result["close"]) / 4.0
        pd.testing.assert_series_equal(result["vwap"], expected_vwap, check_names=False)

    def test_load_all(self, loader_data: tuple[QlibBinLoader, pd.DataFrame]) -> None:
        """load_all returns all instruments."""
        loader, _ = loader_data
        all_data = loader.load_all()
        assert len(all_data) == 2
        assert "EURUSD" in all_data
        assert "GBPUSD" in all_data

    def test_missing_instrument_raises(
        self, loader_data: tuple[QlibBinLoader, pd.DataFrame]
    ) -> None:
        """Loading non-existent instrument raises FileNotFoundError."""
        loader, _ = loader_data
        with pytest.raises(FileNotFoundError):
            loader.load_ohlcv("NONEXISTENT")


# ── Alpha158Calculator Tests ───────────────────────────────────────────────


class TestAlpha158Calculator:
    """Tests for factor computation."""

    @pytest.fixture
    def calc(self) -> Alpha158Calculator:
        return Alpha158Calculator()

    @pytest.fixture
    def df(self) -> pd.DataFrame:
        return make_synthetic_ohlcv(200)

    def test_kbar_shape(self, calc: Alpha158Calculator, df: pd.DataFrame) -> None:
        """Kbar produces 9 factor columns."""
        result = calc.compute_kbar(df)
        assert result.shape[1] == 9
        assert result.shape[0] == len(df)

    def test_kbar_names(self, calc: Alpha158Calculator, df: pd.DataFrame) -> None:
        """Kbar factor names match expected."""
        result = calc.compute_kbar(df)
        expected = [
            "KMID",
            "KLEN",
            "KMID2",
            "KUP",
            "KUP2",
            "KLOW",
            "KLOW2",
            "KSFT",
            "KSFT2",
        ]
        assert list(result.columns) == expected

    def test_kbar_no_nan_at_start(self, calc: Alpha158Calculator, df: pd.DataFrame) -> None:
        """Kbar factors should have no NaN (no lookback needed)."""
        result = calc.compute_kbar(df)
        assert result.iloc[0].notna().all()

    def test_price_shape(self, calc: Alpha158Calculator, df: pd.DataFrame) -> None:
        """Price produces 4 factor columns."""
        result = calc.compute_price(df)
        assert result.shape[1] == 4

    def test_price_names(self, calc: Alpha158Calculator, df: pd.DataFrame) -> None:
        """Price factor names match expected."""
        result = calc.compute_price(df)
        expected = ["OPEN0", "HIGH0", "LOW0", "VWAP0"]
        assert list(result.columns) == expected

    def test_rolling_factor_count(self, calc: Alpha158Calculator, df: pd.DataFrame) -> None:
        """Rolling produces 21 factor types × 5 windows = 105 factors."""
        result = calc.compute_rolling(df)
        # 21 types: ROC, MA, STD, BETA, RSQR, RESI, MAX, MIN,
        #           QTLU, QTLD, RANK, RSV, IMAX, IMIN, IMXD,
        #           CNTP, CNTN, CNTD, SUMP, SUMN, SUMD
        # × 5 windows = 105
        assert result.shape[1] == 105

    def test_compute_all_total(self, calc: Alpha158Calculator, df: pd.DataFrame) -> None:
        """compute_all returns kbar(9) + price(4) + rolling(105) = 118."""
        result = calc.compute_all(df)
        assert result.shape[1] == 118

    def test_kbar_kmid_formula(self, calc: Alpha158Calculator, df: pd.DataFrame) -> None:
        """KMID = (close - open) / open matches manual calc."""
        result = calc.compute_kbar(df)
        expected = (df["close"] - df["open"]) / df["open"]
        pd.testing.assert_series_equal(result["KMID"], expected, check_names=False)

    def test_price_open0_formula(self, calc: Alpha158Calculator, df: pd.DataFrame) -> None:
        """OPEN0 = open / close."""
        result = calc.compute_price(df)
        expected = df["open"] / df["close"]
        pd.testing.assert_series_equal(result["OPEN0"], expected, check_names=False)

    def test_rolling_roc_formula(self, calc: Alpha158Calculator, df: pd.DataFrame) -> None:
        """ROC5 = close.shift(5) / close."""
        result = calc.compute_rolling(df)
        expected = df["close"].shift(5) / df["close"]
        # Compare non-NaN values
        valid = expected.notna()
        pd.testing.assert_series_equal(result["ROC5"][valid], expected[valid], check_names=False)

    def test_rolling_ma_formula(self, calc: Alpha158Calculator, df: pd.DataFrame) -> None:
        """MA5 = close.rolling(5).mean() / close."""
        result = calc.compute_rolling(df)
        expected = df["close"].rolling(5, min_periods=2).mean() / df["close"]
        valid = expected.notna()
        pd.testing.assert_series_equal(result["MA5"][valid], expected[valid], check_names=False)


# ── Alpha158Evaluator Tests ────────────────────────────────────────────────


class TestAlpha158Evaluator:
    """Tests for IC/IR evaluation pipeline."""

    @pytest.fixture
    def evaluator_setup(self, tmp_path: Path) -> Alpha158Evaluator:
        """Create evaluator with synthetic data (2 instruments)."""
        n = 200
        df1 = make_synthetic_ohlcv(n, seed=42)
        df2 = make_synthetic_ohlcv(n, seed=99)
        calendar = [d.strftime("%Y-%m-%d") for d in df1.index]
        instruments = {"EURUSD": df1, "GBPUSD": df2}
        data_dir = tmp_path / "qlib_fx"
        write_qlib_bin_data(data_dir, instruments, calendar)
        return Alpha158Evaluator(
            data_dir=data_dir,
            forward_period=1,
            ic_window=20,
        )

    def test_run_returns_results(self, evaluator_setup: Alpha158Evaluator) -> None:
        """run() returns a non-empty list of FactorResult."""
        results = evaluator_setup.run()
        assert len(results) > 0
        assert all(isinstance(r, FactorResult) for r in results)

    def test_result_fields_valid(self, evaluator_setup: Alpha158Evaluator) -> None:
        """Each FactorResult has valid field values."""
        results = evaluator_setup.run()
        for r in results:
            assert isinstance(r.name, str) and len(r.name) > 0
            assert isinstance(r.ic_mean, float)
            assert isinstance(r.ic_std, float)
            assert isinstance(r.ic_ir, float)
            assert 0.0 <= r.ic_positive_ratio <= 1.0
            assert r.category in ("effective", "weak", "dead")

    def test_classification_thresholds(self, evaluator_setup: Alpha158Evaluator) -> None:
        """Factor classification follows defined thresholds."""
        evaluator = evaluator_setup
        assert evaluator._classify_factor(0.03) == "effective"
        assert evaluator._classify_factor(-0.03) == "effective"
        assert evaluator._classify_factor(0.015) == "weak"
        assert evaluator._classify_factor(-0.015) == "weak"
        assert evaluator._classify_factor(0.005) == "dead"
        assert evaluator._classify_factor(0.0) == "dead"

    def test_generate_report_contains_sections(self, evaluator_setup: Alpha158Evaluator) -> None:
        """Report contains expected section headers."""
        results = evaluator_setup.run()
        report = evaluator_setup.generate_report(results)
        assert "Alpha158 FX Factor IC/IR Report" in report
        assert "Summary:" in report

    def test_save_csv(self, evaluator_setup: Alpha158Evaluator, tmp_path: Path) -> None:
        """save_csv creates a valid CSV file."""
        results = evaluator_setup.run()
        csv_path = tmp_path / "report.csv"
        evaluator_setup.save_csv(results, csv_path)
        assert csv_path.exists()

        # Read and validate CSV
        df = pd.read_csv(csv_path)
        assert "name" in df.columns
        assert "ic_mean" in df.columns
        assert "ic_ir" in df.columns
        assert "category" in df.columns
        assert len(df) == len(results)

    def test_instruments_detail_populated(self, evaluator_setup: Alpha158Evaluator) -> None:
        """Each result has per-instrument IC details."""
        results = evaluator_setup.run()
        for r in results:
            assert len(r.instruments_detail) > 0
            assert "EURUSD" in r.instruments_detail
            assert "GBPUSD" in r.instruments_detail


# ── FactorResult Model Tests ───────────────────────────────────────────────


class TestFactorResult:
    """Tests for the FactorResult Pydantic model."""

    def test_create_valid(self) -> None:
        """FactorResult can be created with valid data."""
        result = FactorResult(
            name="ROC5",
            ic_mean=0.03,
            ic_std=0.05,
            ic_ir=0.6,
            ic_positive_ratio=0.62,
            category="effective",
        )
        assert result.name == "ROC5"
        assert result.category == "effective"

    def test_default_values(self) -> None:
        """FactorResult defaults are correct."""
        result = FactorResult(
            name="TEST",
            ic_mean=0.0,
            ic_std=0.0,
            ic_ir=0.0,
            ic_positive_ratio=0.0,
            category="dead",
        )
        assert result.depends_on_volume is False
        assert result.instruments_detail == {}

    def test_invalid_category_raises(self) -> None:
        """Invalid category raises validation error."""
        with pytest.raises(Exception):
            FactorResult(
                name="TEST",
                ic_mean=0.0,
                ic_std=0.0,
                ic_ir=0.0,
                ic_positive_ratio=0.0,
                category="invalid",  # type: ignore[arg-type]
            )
