"""Tests for equity snapshots (Phase 2.8) — insert_equity_snapshot() and get_equity_history()."""

import pytest

from src.decision_store.sqlite_store import DecisionStore

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: object) -> DecisionStore:
    """Create a DecisionStore with a temporary database."""
    db_path = f"{tmp_path}/test_equity.db"
    s = DecisionStore(db_path=db_path)
    yield s
    s.close()


# ── Test Insert Equity Snapshot ──────────────────────────────────────────────


class TestInsertEquitySnapshot:
    """Tests for insert_equity_snapshot() method."""

    def test_insert_uses_write_lock(self, store: DecisionStore) -> None:
        """Snapshot writes should use the shared write lock."""

        class TrackingLock:
            def __init__(self) -> None:
                self.enter_count = 0

            def __enter__(self):
                self.enter_count += 1
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

        tracking_lock = TrackingLock()
        store._write_lock = tracking_lock

        store.insert_equity_snapshot(equity=50000.0, daily_dd_pct=0.02, max_dd_pct=0.05)

        assert tracking_lock.enter_count == 1

    def test_insert_single_snapshot(self, store: DecisionStore) -> None:
        """Insert and retrieve a single snapshot."""
        store.insert_equity_snapshot(
            equity=50000.0,
            daily_dd_pct=0.02,
            max_dd_pct=0.05,
            balance=50100.0,
            open_positions=2,
        )

        history = store.get_equity_history(hours=24)

        assert len(history) == 1
        row = history[0]
        assert row["equity"] == 50000.0
        assert row["balance"] == 50100.0
        assert row["daily_dd_pct"] == 0.02
        assert row["max_dd_pct"] == 0.05
        assert row["open_positions"] == 2
        assert "timestamp" in row

    def test_insert_multiple_snapshots_ordering(self, store: DecisionStore) -> None:
        """Multiple snapshots should be returned in ASC order by timestamp."""
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)

        # Insert 3 snapshots and manually set timestamps in a known order
        store.insert_equity_snapshot(
            equity=50000.0, daily_dd_pct=0.01, max_dd_pct=0.03, balance=50000.0
        )
        store._conn.execute(
            "UPDATE equity_snapshots SET timestamp = ? WHERE id = 1",
            ((now - timedelta(hours=3)).isoformat(),),
        )
        store._conn.commit()

        store.insert_equity_snapshot(
            equity=50100.0, daily_dd_pct=0.02, max_dd_pct=0.04, balance=50100.0
        )
        store._conn.execute(
            "UPDATE equity_snapshots SET timestamp = ? WHERE id = 2",
            ((now - timedelta(hours=2)).isoformat(),),
        )
        store._conn.commit()

        store.insert_equity_snapshot(
            equity=50200.0, daily_dd_pct=0.03, max_dd_pct=0.05, balance=50200.0
        )
        # id=3 keeps the current timestamp (most recent)

        history = store.get_equity_history(hours=24)
        assert len(history) == 3
        # Verify ASC ordering by timestamp
        assert history[0]["equity"] == 50000.0
        assert history[1]["equity"] == 50100.0
        assert history[2]["equity"] == 50200.0

    def test_default_values_for_optional_params(self, store: DecisionStore) -> None:
        """Default values for optional params should be set correctly."""
        store.insert_equity_snapshot(
            equity=50000.0,
            daily_dd_pct=0.02,
            max_dd_pct=0.05,
        )

        history = store.get_equity_history(hours=24)

        assert len(history) == 1
        row = history[0]
        assert row["balance"] is None
        assert row["open_positions"] == 0

    def test_table_creation_is_idempotent(self, store: DecisionStore) -> None:
        """Table creation should be idempotent."""
        # Call _ensure_tables() twice - should not raise
        store._ensure_tables()
        store._ensure_tables()

        # Verify can still insert and query
        store.insert_equity_snapshot(equity=50000.0, daily_dd_pct=0.02, max_dd_pct=0.05)
        history = store.get_equity_history(hours=24)
        assert len(history) == 1

    def test_negative_drawdown_values(self, store: DecisionStore) -> None:
        """Negative drawdown values should be stored correctly."""
        store.insert_equity_snapshot(
            equity=50000.0,
            daily_dd_pct=-0.01,
            max_dd_pct=-0.05,
        )

        history = store.get_equity_history(hours=24)
        assert len(history) == 1
        assert history[0]["daily_dd_pct"] == -0.01
        assert history[0]["max_dd_pct"] == -0.05

    def test_large_open_positions_count(self, store: DecisionStore) -> None:
        """Large open positions count should be stored correctly."""
        store.insert_equity_snapshot(
            equity=50000.0,
            daily_dd_pct=0.02,
            max_dd_pct=0.05,
            open_positions=100,
        )

        history = store.get_equity_history(hours=24)
        assert len(history) == 1
        assert history[0]["open_positions"] == 100


# ── Test Get Equity History ─────────────────────────────────────────────────


class TestGetEquityHistory:
    """Tests for get_equity_history() method."""

    def test_empty_history_returns_empty_list(self, store: DecisionStore) -> None:
        """Empty history should return an empty list."""
        history = store.get_equity_history(hours=24)
        assert history == []

    def test_hours_filter_excludes_old_snapshots(self, store: DecisionStore) -> None:
        """Old snapshots should be excluded by hours filter."""
        # Insert a snapshot
        store.insert_equity_snapshot(equity=50000.0, daily_dd_pct=0.02, max_dd_pct=0.05)

        # Set timestamp to 25 hours ago using direct SQL
        store._conn.execute(
            """UPDATE equity_snapshots
               SET timestamp = '2025-01-01T00:00:00'
               WHERE id = 1"""
        )
        store._conn.commit()

        # Query with 24-hour window - should return 0 results
        history = store.get_equity_history(hours=24)
        assert len(history) == 0

    def test_hours_filter_includes_recent_snapshots(self, store: DecisionStore) -> None:
        """Recent snapshots should be included by hours filter."""
        store.insert_equity_snapshot(equity=50000.0, daily_dd_pct=0.02, max_dd_pct=0.05)

        # Set timestamp to 10 hours ago using direct SQL (within 24-hour window)
        # Use a timestamp that's ISO format and recent enough
        from datetime import datetime, timedelta, timezone

        recent_ts = (datetime.now(timezone.utc) - timedelta(hours=10)).isoformat()
        store._conn.execute(
            f"""UPDATE equity_snapshots
               SET timestamp = '{recent_ts}'
               WHERE id = 1"""
        )
        store._conn.commit()

        history = store.get_equity_history(hours=24)
        assert len(history) == 1

    def test_return_dicts_with_correct_keys(self, store: DecisionStore) -> None:
        """Returned dicts should have correct keys."""
        expected_keys = {
            "timestamp",
            "equity",
            "balance",
            "daily_dd_pct",
            "max_dd_pct",
            "open_positions",
        }

        store.insert_equity_snapshot(
            equity=50000.0, daily_dd_pct=0.02, max_dd_pct=0.05, balance=50100.0
        )

        history = store.get_equity_history(hours=24)
        assert len(history) == 1
        assert set(history[0].keys()) == expected_keys

    def test_multiple_snapshots_all_have_correct_keys(self, store: DecisionStore) -> None:
        """All snapshots in history should have correct keys."""
        expected_keys = {
            "timestamp",
            "equity",
            "balance",
            "daily_dd_pct",
            "max_dd_pct",
            "open_positions",
        }

        for i in range(5):
            store.insert_equity_snapshot(
                equity=50000.0 + i * 100,
                daily_dd_pct=0.02 + i * 0.01,
                max_dd_pct=0.05 + i * 0.01,
            )

        history = store.get_equity_history(hours=24)
        assert len(history) == 5

        for row in history:
            assert set(row.keys()) == expected_keys

    def test_hours_parameter_different_values(self, store: DecisionStore) -> None:
        """Different hours parameters should filter correctly."""
        # Insert 3 snapshots with different timestamps
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        timestamps = [
            (now - timedelta(hours=2)).isoformat(),
            (now - timedelta(hours=12)).isoformat(),
            (now - timedelta(hours=48)).isoformat(),
        ]

        for i, ts in enumerate(timestamps):
            store.insert_equity_snapshot(
                equity=50000.0 + i * 100,
                daily_dd_pct=0.02,
                max_dd_pct=0.05,
            )
            # Update the timestamp to control the exact time
            store._conn.execute(
                f"""UPDATE equity_snapshots
                   SET timestamp = '{ts}'
                   WHERE id = {i + 1}"""
            )
            store._conn.commit()

        # 1-hour window: should return 0 (closest is 2h old)
        history_1h = store.get_equity_history(hours=1)
        assert len(history_1h) == 0

        # 24-hour window: should return 2 snapshots (2h and 12h old)
        history_24h = store.get_equity_history(hours=24)
        assert len(history_24h) == 2

        # 72-hour window: should return all 3 snapshots
        history_72h = store.get_equity_history(hours=72)
        assert len(history_72h) == 3

    def test_zero_equity_value(self, store: DecisionStore) -> None:
        """Zero equity should be stored and retrieved correctly."""
        store.insert_equity_snapshot(equity=0.0, daily_dd_pct=0.0, max_dd_pct=0.0)

        history = store.get_equity_history(hours=24)
        assert len(history) == 1
        assert history[0]["equity"] == 0.0
        assert history[0]["daily_dd_pct"] == 0.0
        assert history[0]["max_dd_pct"] == 0.0
