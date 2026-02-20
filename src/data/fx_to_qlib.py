"""
FX DataFrame → Qlib binary format converter.

Converts pandas DataFrames (from fx_data_fetcher) into the binary format
that Microsoft Qlib expects for qlib.init(provider_uri=...).

Qlib binary directory structure:
    output_dir/
      instruments/
        all.txt              # instrument list with date ranges
      calendars/
        day.txt              # trading calendar (one date per line)
      features/
        EURUSD/
          open.day.bin       # float32 binary
          open.day.meta      # JSON metadata
          high.day.bin / .meta
          low.day.bin / .meta
          close.day.bin / .meta
          volume.day.bin / .meta
          factor.day.bin / .meta  # adj factor = 1.0 for FX
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# Features to export. "factor" is synthetic (always 1.0 for FX).
FEATURE_COLUMNS = ["open", "high", "low", "close", "volume", "factor"]


def convert_to_qlib_binary(
    data: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> Path:
    """Convert FX DataFrames to Qlib binary format.

    Args:
        data: Dict of symbol -> DataFrame. Each DataFrame must have columns:
              datetime (or date), open, high, low, close, volume.
        output_dir: Root directory for Qlib binary data.

    Returns:
        Path to the output directory.
    """
    output_dir = Path(output_dir)

    # Create directory structure
    instruments_dir = output_dir / "instruments"
    calendars_dir = output_dir / "calendars"
    features_dir = output_dir / "features"

    instruments_dir.mkdir(parents=True, exist_ok=True)
    calendars_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    # Build unified trading calendar from all symbols
    all_dates: set[pd.Timestamp] = set()
    processed_data: dict[str, pd.DataFrame] = {}

    for symbol, df in data.items():
        if df.empty:
            logger.warning("Qlib converter: skipping {} (empty DataFrame)", symbol)
            continue

        df = _prepare_dataframe(df)
        processed_data[symbol] = df
        all_dates.update(df["datetime"].tolist())

    if not all_dates:
        logger.error("Qlib converter: no data to convert")
        return output_dir

    # Sort calendar
    calendar = sorted(all_dates)
    cal_index = {dt: i for i, dt in enumerate(calendar)}

    # Write calendar
    _write_calendar(calendars_dir, calendar)

    # Write instruments
    _write_instruments(instruments_dir, processed_data, calendar)

    # Write features for each symbol
    for symbol, df in processed_data.items():
        _write_symbol_features(features_dir, symbol, df, cal_index, len(calendar))

    logger.info(
        "Qlib converter: wrote {} symbols, {} calendar days to {}",
        len(processed_data),
        len(calendar),
        output_dir,
    )
    return output_dir


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame for Qlib conversion."""
    df = df.copy()

    # Ensure datetime column
    if "datetime" not in df.columns and "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"])
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])

    # Normalize to date-level precision (remove time component)
    df["datetime"] = df["datetime"].dt.normalize()

    # Ensure volume
    if "volume" not in df.columns:
        df["volume"] = 0

    # Add adj factor (always 1.0 for FX — no splits/dividends)
    df["factor"] = 1.0

    # Drop duplicates, sort by date
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    return df


def _write_calendar(calendars_dir: Path, calendar: list[pd.Timestamp]) -> None:
    """Write calendars/day.txt — one date per line, YYYY-MM-DD."""
    cal_path = calendars_dir / "day.txt"
    with open(cal_path, "w", encoding="utf-8") as f:
        for dt in calendar:
            f.write(dt.strftime("%Y-%m-%d") + "\n")

    logger.debug("Qlib converter: wrote calendar ({} days)", len(calendar))


def _write_instruments(
    instruments_dir: Path,
    data: dict[str, pd.DataFrame],
    calendar: list[pd.Timestamp],
) -> None:
    """Write instruments/all.txt — one line per instrument with date range."""
    all_path = instruments_dir / "all.txt"
    with open(all_path, "w", encoding="utf-8") as f:
        for symbol, df in sorted(data.items()):
            min_date = df["datetime"].min().strftime("%Y-%m-%d")
            max_date = df["datetime"].max().strftime("%Y-%m-%d")
            f.write(f"{symbol}\t{min_date}\t{max_date}\n")

    logger.debug("Qlib converter: wrote instruments ({} symbols)", len(data))


def _write_symbol_features(
    features_dir: Path,
    symbol: str,
    df: pd.DataFrame,
    cal_index: dict[pd.Timestamp, int],
    cal_length: int,
) -> None:
    """Write binary feature files for a single symbol."""
    symbol_dir = features_dir / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    # Map DataFrame dates to calendar indices
    dates_list = df["datetime"].tolist()
    mapped_indices = [cal_index[dt] for dt in dates_list if dt in cal_index]
    if not mapped_indices:
        logger.warning("Qlib converter: no dates matched calendar for {}", symbol)
        return
    start_index = min(mapped_indices)
    end_index = max(mapped_indices)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            logger.warning(
                "Qlib converter: column '{}' missing for {}, filling with NaN", col, symbol
            )
            values = np.full(cal_length, np.nan, dtype=np.float32)
        else:
            # Create full-length array (NaN for missing days)
            values = np.full(cal_length, np.nan, dtype=np.float32)
            datetime_col: list[Any] = df["datetime"].tolist()
            col_values: list[Any] = df[col].tolist()
            for i in range(len(datetime_col)):
                idx = cal_index.get(datetime_col[i])
                if idx is not None:
                    values[idx] = np.float32(col_values[i])

        # Trim to actual data range
        trimmed = values[start_index : end_index + 1]

        # Write .bin file (raw float32 bytes)
        bin_path = symbol_dir / f"{col}.day.bin"
        trimmed.tofile(str(bin_path))

        # Write .meta file (JSON with start/end indices)
        meta_path = symbol_dir / f"{col}.day.meta"
        meta = {
            "start_index": start_index,
            "end_index": end_index,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    logger.debug(
        "Qlib converter: wrote {} features for {} (indices {}-{})",
        len(FEATURE_COLUMNS),
        symbol,
        start_index,
        end_index,
    )


def init_qlib_with_fx(qlib_dir: str | Path) -> None:
    """Initialize Qlib with FX binary data.

    Args:
        qlib_dir: Path to the Qlib binary data directory
                  (the output of convert_to_qlib_binary).
    """
    import qlib
    from qlib.config import REG_CN

    qlib_dir = Path(qlib_dir).resolve()

    if not qlib_dir.exists():
        raise FileNotFoundError(f"Qlib data directory not found: {qlib_dir}")

    qlib.init(
        provider_uri=str(qlib_dir),
        region=REG_CN,  # Region doesn't matter for custom data
    )

    logger.info("Qlib initialized with FX data from {}", qlib_dir)
