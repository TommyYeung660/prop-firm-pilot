"""
Tests for trade close retrospection quality (P3.9).

Validates:
1. ClosedPosition model accepts close_reason field
2. Resolution path is tracked and logged in TRADE_CLOSED events
3. Resolution path correctly identifies each fallback source
"""

from src.execution.matchtrader_client import ClosedPosition


class TestClosedPositionCloseReason:
    """Tests for ClosedPosition.close_reason field."""

    def test_close_reason_default_empty(self):
        """close_reason defaults to empty string."""
        cp = ClosedPosition(position_id="P1", symbol="EURUSD")
        assert cp.close_reason == ""

    def test_close_reason_from_broker(self):
        """close_reason is populated when broker provides it."""
        cp = ClosedPosition(
            position_id="P1",
            symbol="EURUSD",
            closeReason="STOP_LOSS",
        )
        assert cp.close_reason == "STOP_LOSS"

    def test_close_reason_alias(self):
        """close_reason works with both alias and field name."""
        cp = ClosedPosition(
            position_id="P1",
            symbol="EURUSD",
            close_reason="TAKE_PROFIT",
        )
        assert cp.close_reason == "TAKE_PROFIT"


class TestResolutionPathTracking:
    """Tests for resolution path tracking in _handle_position_closed.

    These tests verify the resolution_path logic by testing the
    helper function that determines the path label.
    """

    def test_broker_api_path(self):
        """When broker API returns data, resolution_path is 'broker_api'."""
        from src.scheduler.close_resolution import determine_resolution_path

        assert (
            determine_resolution_path(
                matched=True,
                used_execution_meta=False,
                used_best_day=False,
                used_reeval=False,
                used_last_known=False,
            )
            == "broker_api"
        )

    def test_execution_meta_path(self):
        """When execution_meta provides data, resolution_path is 'execution_meta'."""
        from src.scheduler.close_resolution import determine_resolution_path

        assert (
            determine_resolution_path(
                matched=False,
                used_execution_meta=True,
                used_best_day=False,
                used_reeval=False,
                used_last_known=False,
            )
            == "execution_meta"
        )

    def test_best_day_path(self):
        """When best_day_close provides PnL, resolution_path is 'best_day_close'."""
        from src.scheduler.close_resolution import determine_resolution_path

        assert (
            determine_resolution_path(
                matched=False,
                used_execution_meta=False,
                used_best_day=True,
                used_reeval=False,
                used_last_known=False,
            )
            == "best_day_close"
        )

    def test_reeval_path(self):
        """When reeval_close provides PnL, resolution_path is 'reeval_close'."""
        from src.scheduler.close_resolution import determine_resolution_path

        assert (
            determine_resolution_path(
                matched=False,
                used_execution_meta=False,
                used_best_day=False,
                used_reeval=True,
                used_last_known=False,
            )
            == "reeval_close"
        )

    def test_last_known_path(self):
        """When last_known_profit provides PnL, resolution_path is 'last_known_profit'."""
        from src.scheduler.close_resolution import determine_resolution_path

        assert (
            determine_resolution_path(
                matched=False,
                used_execution_meta=False,
                used_best_day=False,
                used_reeval=False,
                used_last_known=True,
            )
            == "last_known_profit"
        )

    def test_unknown_path(self):
        """When nothing provides data, resolution_path is 'unknown'."""
        from src.scheduler.close_resolution import determine_resolution_path

        assert (
            determine_resolution_path(
                matched=False,
                used_execution_meta=False,
                used_best_day=False,
                used_reeval=False,
                used_last_known=False,
            )
            == "unknown"
        )
