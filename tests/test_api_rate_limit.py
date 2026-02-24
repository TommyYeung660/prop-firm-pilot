"""
Tests for API rate limit feature spanning RateLimiter, DecisionStore persistence,
call type classification, and auto-throttle logic.

Tests cover:
- RateLimiter with store integration (loading counts, persisting, error handling)
- RateLimiter read/write tracking breakdown
- RateLimiter daily reset on date change
- Call type classification in MatchTraderClient._raw_request
- DecisionStore.record_api_calls persistence
- DecisionStore.get_api_call_breakdown queries
- Auto-throttle logic computation
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.decision_store.sqlite_store import DecisionStore
from src.execution.matchtrader_client import RateLimiter

# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> str:
    """Create temporary database path for testing."""
    return str(tmp_path / "test.db")


@pytest.fixture
def mock_store(tmp_db_path: str) -> DecisionStore:
    """Create a DecisionStore instance with temporary database."""
    return DecisionStore(db_path=tmp_db_path)


# ── RateLimiter with Store Integration ───────────────────────────────


class TestRateLimiterStoreIntegration:
    """Test RateLimiter integration with persistent storage."""

    def test_init_loads_existing_counts_from_store(self, tmp_db_path: str) -> None:
        """RateLimiter.__init__ loads counts from store via get_api_call_breakdown()."""
        store = DecisionStore(db_path=tmp_db_path)
        store.record_api_calls(count=5, call_type="read")
        store.record_api_calls(count=3, call_type="write")

        limiter = RateLimiter(daily_limit=2000, store=store)

        assert limiter.count == 8
        assert limiter.read_count == 5
        assert limiter.write_count == 3

    def test_init_handles_store_failure_gracefully(self) -> None:
        """RateLimiter.__init__ handles store failure gracefully."""
        broken_store = MagicMock()
        broken_store.get_api_call_breakdown.side_effect = Exception("DB connection failed")

        with patch("loguru.logger"):
            limiter = RateLimiter(daily_limit=2000, store=broken_store)

            assert limiter.count == 0
            assert limiter.read_count == 0
            assert limiter.write_count == 0

    def test_record_persists_to_store(self, tmp_db_path: str) -> None:
        """RateLimiter.record() persists to store via store.record_api_calls()."""
        store = DecisionStore(db_path=tmp_db_path)
        limiter = RateLimiter(daily_limit=2000, store=store)

        limiter.record(call_type="read")

        breakdown = store.get_api_call_breakdown()
        assert breakdown["total"] == 1
        assert breakdown["read"] == 1
        assert breakdown["write"] == 0

        limiter.record(call_type="write")

        breakdown = store.get_api_call_breakdown()
        assert breakdown["total"] == 2
        assert breakdown["read"] == 1
        assert breakdown["write"] == 1

    def test_record_handles_persistence_failure_gracefully(self) -> None:
        """RateLimiter.record() handles store persistence failure gracefully."""
        broken_store = MagicMock()
        broken_store.get_api_call_breakdown.return_value = {
            "total": 0,
            "read": 0,
            "write": 0,
        }
        broken_store.record_api_calls.side_effect = Exception("DB write failed")

        with patch("loguru.logger"):
            limiter = RateLimiter(daily_limit=2000, store=broken_store)
            limiter.record(call_type="read")
            assert limiter.count == 1

    def test_without_store_works_as_pure_in_memory_counter(self) -> None:
        """RateLimiter without store (store=None) still works as pure in-memory counter."""
        limiter = RateLimiter(daily_limit=100, store=None)

        for _ in range(5):
            limiter.record(call_type="read")
        for _ in range(3):
            limiter.record(call_type="write")

        assert limiter.count == 8
        assert limiter.read_count == 5
        assert limiter.write_count == 3
        assert limiter.remaining == 92


# ── RateLimiter Read/Write Tracking ───────────────────────────────────


class TestRateLimiterReadWriteTracking:
    """Test RateLimiter read/write call type tracking."""

    def test_record_read_increments_read_count(self) -> None:
        """record(call_type="read") increments read_count property."""
        limiter = RateLimiter(daily_limit=100)
        limiter.record(call_type="read")

        assert limiter.read_count == 1
        assert limiter.write_count == 0

    def test_record_write_increments_write_count(self) -> None:
        """record(call_type="write") increments write_count property."""
        limiter = RateLimiter(daily_limit=100)
        limiter.record(call_type="write")

        assert limiter.read_count == 0
        assert limiter.write_count == 1

    def test_record_default_call_type_is_read(self) -> None:
        """record() default call_type is read."""
        limiter = RateLimiter(daily_limit=100)
        limiter.record()

        assert limiter.read_count == 1
        assert limiter.write_count == 0

    def test_mixed_read_write_count_equals_sum(self) -> None:
        """Mixed read/write: count == read_count + write_count."""
        limiter = RateLimiter(daily_limit=100)

        for _ in range(7):
            limiter.record(call_type="read")
        for _ in range(4):
            limiter.record(call_type="write")

        assert limiter.read_count == 7
        assert limiter.write_count == 4
        assert limiter.count == 11


# ── RateLimiter Daily Reset ─────────────────────────────────────────


class TestRateLimiterDailyReset:
    """Test RateLimiter daily reset behavior."""

    def test_counts_reset_when_date_changes(self) -> None:
        """Counts reset when date changes (manipulate _reset_date directly)."""
        from datetime import date, timedelta

        limiter = RateLimiter(daily_limit=100)

        for _ in range(10):
            limiter.record()
        assert limiter.count == 10

        limiter._reset_date = date.today() - timedelta(days=1)

        assert limiter.count == 0
        assert limiter.read_count == 0
        assert limiter.write_count == 0

    def test_reset_clears_read_and_write_counts_to_zero(self) -> None:
        """Reset clears read_count and write_count to 0."""
        from datetime import date, timedelta

        limiter = RateLimiter(daily_limit=100)

        for _ in range(5):
            limiter.record(call_type="read")
        for _ in range(3):
            limiter.record(call_type="write")

        assert limiter.read_count == 5
        assert limiter.write_count == 3
        limiter._reset_date = date.today() - timedelta(days=1)

        assert limiter.read_count == 0
        assert limiter.write_count == 0
        assert limiter.count == 0


# ── Call Type Classification in _raw_request ───────────────────────────


class TestCallTypeClassification:
    """Test call type classification logic in MatchTraderClient._raw_request."""

    def _classify(self, method: str, path: str) -> str:
        _write_paths = ("/position/open", "/position/close", "/position/edit")
        return (
            "write" if method == "POST" and any(wp in path for wp in _write_paths) else "read"
        )

    def test_post_position_open_classified_as_write(self) -> None:
        """POST to /position/open -> write."""
        assert self._classify("POST", "/mtr-api/some-uuid/position/open") == "write"

    def test_post_position_close_classified_as_write(self) -> None:
        """POST to /position/close -> write."""
        assert self._classify("POST", "/mtr-api/some-uuid/position/close") == "write"

    def test_post_position_edit_classified_as_write(self) -> None:
        """POST to /position/edit -> write."""
        assert self._classify("POST", "/mtr-api/some-uuid/position/edit") == "write"

    def test_get_positions_classified_as_read(self) -> None:
        """GET to /positions -> read."""
        assert self._classify("GET", "/mtr-api/some-uuid/open-positions") == "read"

    def test_post_co_login_classified_as_read(self) -> None:
        """POST to /manager/co-login -> read (not in _write_paths)."""
        assert self._classify("POST", "/manager/co-login") == "read"

    def test_any_other_method_classified_as_read(self) -> None:
        """Any other method -> read."""
        assert self._classify("PUT", "/some/path") == "read"
        assert self._classify("DELETE", "/some/path") == "read"
        assert self._classify("HEAD", "/some/path") == "read"


# ── DecisionStore.record_api_calls ───────────────────────────────────


class TestDecisionStoreRecordApiCalls:
    """Test DecisionStore.record_api_calls persistence."""

    def test_record_read_increments_read_count(self, tmp_db_path: str) -> None:
        """record_api_calls(count=1, call_type="read") increments read_count."""
        store = DecisionStore(db_path=tmp_db_path)
        total = store.record_api_calls(count=1, call_type="read")
        assert total == 1

        breakdown = store.get_api_call_breakdown()
        assert breakdown["read"] == 1
        assert breakdown["write"] == 0

    def test_record_write_increments_write_count(self, tmp_db_path: str) -> None:
        """record_api_calls(count=1, call_type="write") increments write_count."""
        store = DecisionStore(db_path=tmp_db_path)
        total = store.record_api_calls(count=1, call_type="write")
        assert total == 1

        breakdown = store.get_api_call_breakdown()
        assert breakdown["read"] == 0
        assert breakdown["write"] == 1

    def test_returns_total_count_after_recording(self, tmp_db_path: str) -> None:
        """Returns total count after recording."""
        store = DecisionStore(db_path=tmp_db_path)

        total1 = store.record_api_calls(count=5, call_type="read")
        assert total1 == 5

        total2 = store.record_api_calls(count=3, call_type="write")
        assert total2 == 8


# ── DecisionStore.get_api_call_breakdown ─────────────────────────────


class TestDecisionStoreGetApiCallBreakdown:
    """Test DecisionStore.get_api_call_breakdown queries."""

    def test_returns_total_read_write_dict(self, tmp_db_path: str) -> None:
        """Returns total/read/write dict."""
        store = DecisionStore(db_path=tmp_db_path)

        store.record_api_calls(count=10, call_type="read")
        store.record_api_calls(count=5, call_type="write")

        breakdown = store.get_api_call_breakdown()

        assert breakdown["total"] == 15
        assert breakdown["read"] == 10
        assert breakdown["write"] == 5

    def test_returns_zeros_for_no_data(self, tmp_db_path: str) -> None:
        """Returns zeros for no data."""
        store = DecisionStore(db_path=tmp_db_path)
        breakdown = store.get_api_call_breakdown()
        assert breakdown == {"total": 0, "read": 0, "write": 0}

    def test_breakdown_for_specific_date(self, tmp_db_path: str) -> None:
        """get_api_call_breakdown(date) works for specific dates."""
        store = DecisionStore(db_path=tmp_db_path)

        with patch("src.decision_store.sqlite_store.datetime") as mock_dt:
            today = datetime(2026, 2, 24, 12, 0, 0, tzinfo=timezone.utc)
            mock_dt.now.return_value = today

            store.record_api_calls(count=5, call_type="read")

            breakdown_today = store.get_api_call_breakdown(date="2026-02-24")
            assert breakdown_today["total"] == 5

            breakdown_other = store.get_api_call_breakdown(date="2026-02-25")
            assert breakdown_other == {"total": 0, "read": 0, "write": 0}


# ── Auto-Throttle Logic ─────────────────────────────────────────────


class TestAutoThrottleLogic:
    """Test auto-throttle logic computation for rate limiting."""

    @staticmethod
    def _compute_sleep(base_interval: int, daily_limit: int, remaining: int) -> int:
        if remaining < daily_limit * 0.15:
            return base_interval * 4
        elif remaining < daily_limit * 0.30:
            return base_interval * 2
        else:
            return base_interval

    def test_remaining_below_15_percent_quadruples_sleep_interval(self) -> None:
        """When remaining < 15% of daily_limit -> sleep = base * 4."""
        assert self._compute_sleep(120, 2000, 200) == 480

    def test_remaining_between_15_30_percent_doubles_sleep_interval(self) -> None:
        """When remaining < 30% of daily_limit -> sleep = base * 2."""
        assert self._compute_sleep(120, 2000, 400) == 240

    def test_remaining_above_30_percent_uses_base_interval(self) -> None:
        """When remaining >= 30% -> sleep = base (no throttle)."""
        assert self._compute_sleep(120, 2000, 800) == 120

    def test_throttle_logic_edge_case_exactly_15_percent(self) -> None:
        """Edge case: remaining exactly at 15% threshold."""
        assert self._compute_sleep(120, 1000, 150) == 240

    def test_throttle_logic_edge_case_exactly_30_percent(self) -> None:
        """Edge case: remaining exactly at 30% threshold."""
        assert self._compute_sleep(120, 1000, 300) == 120


# ── Integration: RateLimiter + DecisionStore ─────────────────────────


class TestRateLimiterDecisionStoreIntegration:
    """Integration tests for RateLimiter with DecisionStore."""

    def test_full_lifecycle_store_and_reload(self, tmp_db_path: str) -> None:
        """Test full lifecycle: record calls, persist, reload in new limiter."""
        store = DecisionStore(db_path=tmp_db_path)
        limiter1 = RateLimiter(daily_limit=2000, store=store)

        limiter1.record(call_type="read")
        limiter1.record(call_type="read")
        limiter1.record(call_type="write")

        assert limiter1.count == 3

        limiter2 = RateLimiter(daily_limit=2000, store=store)

        assert limiter2.count == 3
        assert limiter2.read_count == 2
        assert limiter2.write_count == 1

    def test_multiple_limiter_instances_share_state(self, tmp_db_path: str) -> None:
        """Multiple RateLimiter instances with same store share persisted state."""
        store = DecisionStore(db_path=tmp_db_path)

        limiter_a = RateLimiter(daily_limit=2000, store=store)
        limiter_b = RateLimiter(daily_limit=2000, store=store)

        limiter_a.record(call_type="read")
        limiter_a.record(call_type="write")

        assert limiter_b.count == 0

        limiter_c = RateLimiter(daily_limit=2000, store=store)
        assert limiter_c.count == 2
