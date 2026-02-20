"""
DuckDB-based FX data cache — stores historical OHLCV data locally
to avoid redundant API calls and enable fast Qlib format conversion.

Schema mirrors the qlib_market_scanner's DuckDB store but adapted for FX:
- No adj_close (FX has no corporate actions)
- Volume = tick count (not real volume)
- Factor = 1.0 always (no splits/dividends)
"""

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
from loguru import logger

# DuckDB schema for FX daily bars
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS fx_daily (
    symbol      VARCHAR NOT NULL,
    date        DATE NOT NULL,
    open        DOUBLE NOT NULL,
    high        DOUBLE NOT NULL,
    low         DOUBLE NOT NULL,
    close       DOUBLE NOT NULL,
    volume      BIGINT DEFAULT 0,
    provider    VARCHAR DEFAULT 'unknown',
    fetched_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, date)
)
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_fx_daily_symbol_date
ON fx_daily (symbol, date)
"""


class FxDuckDbStore:
    """DuckDB-based cache for FX daily OHLCV data.

    Usage:
        store = FxDuckDbStore("data/fx_prices.duckdb")
        store.upsert("EURUSD", df)          # Save fetched data
        df = store.read("EURUSD", start, end)  # Read cached data
        missing = store.get_missing_dates("EURUSD", start, end)  # Find gaps
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self._db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables and indexes if not exist."""
        self._conn.execute(CREATE_TABLE_SQL)
        self._conn.execute(CREATE_INDEX_SQL)
        logger.debug("FxDuckDbStore: schema initialized at {}", self._db_path)

    def upsert(self, symbol: str, df: pd.DataFrame, provider: str = "unknown") -> int:
        """Insert or update daily bars for a symbol.

        Args:
            symbol: FX pair (e.g. "EURUSD").
            df: DataFrame with columns: datetime/date, open, high, low, close, volume.
            provider: Data provider name for audit trail.

        Returns:
            Number of rows upserted.
        """
        if df.empty:
            return 0

        # Normalize DataFrame
        df = df.copy()
        df["symbol"] = symbol
        df["provider"] = provider

        # Ensure date column
        if "datetime" in df.columns:
            df["date"] = pd.to_datetime(df["datetime"]).dt.date
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date

        # Ensure volume column
        if "volume" not in df.columns:
            df["volume"] = 0

        # Select only needed columns
        cols = ["symbol", "date", "open", "high", "low", "close", "volume", "provider"]
        df = pd.DataFrame(df[cols])

        # Upsert using INSERT OR REPLACE
        self._conn.execute("BEGIN TRANSACTION")
        try:
            # Delete existing rows for this symbol+date range
            min_date = df["date"].min()
            max_date = df["date"].max()
            self._conn.execute(
                "DELETE FROM fx_daily WHERE symbol = ? AND date BETWEEN ? AND ?",
                [symbol, min_date, max_date],
            )

            # Insert new rows
            self._conn.execute(
                "INSERT INTO fx_daily (symbol, date, open, high, low, close, volume, provider) "
                "SELECT symbol, date, open, high, low, close, volume, provider FROM df"
            )
            self._conn.execute("COMMIT")

            count = len(df)
            logger.info(
                "FxDuckDbStore: upserted {} rows for {} ({} to {})",
                count,
                symbol,
                min_date,
                max_date,
            )
            return count

        except Exception as e:
            self._conn.execute("ROLLBACK")
            logger.error("FxDuckDbStore: upsert failed for {}: {}", symbol, e)
            raise

    def read(
        self,
        symbol: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Read cached daily bars for a symbol.

        Args:
            symbol: FX pair.
            start_date: Start date (inclusive). None = all.
            end_date: End date (inclusive). None = all.

        Returns:
            DataFrame with columns: date, open, high, low, close, volume.
        """
        query = "SELECT date, open, high, low, close, volume FROM fx_daily WHERE symbol = ?"
        params: list[str | date] = [symbol]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date ASC"
        result = self._conn.execute(query, params).fetchdf()

        logger.debug("FxDuckDbStore: read {} rows for {}", len(result), symbol)
        return result

    def read_all_symbols(
        self,
        symbols: list[str],
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Read cached data for multiple symbols."""
        return {sym: self.read(sym, start_date, end_date) for sym in symbols}

    def get_date_range(self, symbol: str) -> tuple[date | None, date | None]:
        """Get the min and max dates available for a symbol.

        Returns:
            Tuple of (min_date, max_date). Both None if no data.
        """
        result = self._conn.execute(
            "SELECT MIN(date), MAX(date) FROM fx_daily WHERE symbol = ?",
            [symbol],
        ).fetchone()

        if result and result[0]:
            return result[0], result[1]
        return None, None

    def get_missing_dates(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> list[date]:
        """Find dates missing from the cache (for incremental fetching).

        Note: Only checks weekdays (Mon-Fri) since FX markets close on weekends.

        Returns:
            List of missing dates.
        """
        # Get all dates we have
        existing = self._conn.execute(
            "SELECT DISTINCT date FROM fx_daily WHERE symbol = ? AND date BETWEEN ? AND ? ORDER BY date",
            [symbol, start_date, end_date],
        ).fetchdf()

        existing_dates = set(existing["date"].tolist()) if not existing.empty else set()

        # Generate all weekdays in range
        all_dates = pd.bdate_range(start=start_date, end=end_date)
        all_dates_set = {d.date() for d in all_dates}

        missing = sorted(all_dates_set - existing_dates)
        return missing

    def row_count(self, symbol: str | None = None) -> int:
        """Get total row count, optionally filtered by symbol."""
        if symbol:
            result = self._conn.execute(
                "SELECT COUNT(*) FROM fx_daily WHERE symbol = ?", [symbol]
            ).fetchone()
        else:
            result = self._conn.execute("SELECT COUNT(*) FROM fx_daily").fetchone()
        return result[0] if result else 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
        logger.debug("FxDuckDbStore: connection closed")

    def __enter__(self) -> "FxDuckDbStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
