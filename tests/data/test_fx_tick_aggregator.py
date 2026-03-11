"""Tests for tick-to-bar aggregation."""

from datetime import datetime, timezone

from src.data.fx_tick_aggregator import FXTickAggregator
from src.data.fx_websocket_client import WebSocketTick


def _tick(symbol: str, bid: float, ask: float, dt: datetime) -> WebSocketTick:
    return WebSocketTick(
        symbol=symbol,
        bid=bid,
        ask=ask,
        timestamp_ms=int(dt.timestamp() * 1000),
    )


def test_aggregator_closes_one_minute_bar_from_ticks() -> None:
    agg = FXTickAggregator()
    agg.add_tick(_tick("EURUSD", 1.10, 1.11, datetime(2026, 3, 11, 12, 0, 5, tzinfo=timezone.utc)))
    agg.add_tick(_tick("EURUSD", 1.12, 1.13, datetime(2026, 3, 11, 12, 0, 40, tzinfo=timezone.utc)))

    agg.close_elapsed_bars(now=datetime(2026, 3, 11, 12, 1, 1, tzinfo=timezone.utc))
    bar = agg.get_closed_bars("EURUSD", "1m", limit=1)[0]

    assert bar.open == 1.105
    assert bar.high == 1.125
    assert bar.low == 1.105
    assert bar.close == 1.125


def test_aggregator_rolls_up_closed_one_minute_bars_into_five_minute_bar() -> None:
    agg = FXTickAggregator()

    for minute, price in enumerate([1.10, 1.11, 1.12, 1.13, 1.14]):
        agg.add_tick(
            _tick(
                "EURUSD",
                price,
                price + 0.01,
                datetime(2026, 3, 11, 12, minute, 10, tzinfo=timezone.utc),
            )
        )

    agg.close_elapsed_bars(now=datetime(2026, 3, 11, 12, 5, 1, tzinfo=timezone.utc))
    bar = agg.get_closed_bars("EURUSD", "5m", limit=1)[0]

    assert bar.open == 1.105
    assert bar.close == 1.145


def test_get_closed_bars_ignores_active_partial_bar() -> None:
    agg = FXTickAggregator()
    agg.add_tick(_tick("EURUSD", 1.10, 1.11, datetime(2026, 3, 11, 12, 0, 5, tzinfo=timezone.utc)))

    assert agg.get_closed_bars("EURUSD", "1m", limit=10) == []


def test_latest_quote_returns_last_bid_ask() -> None:
    agg = FXTickAggregator()
    agg.add_tick(_tick("EURUSD", 1.12, 1.13, datetime(2026, 3, 11, 12, 0, 40, tzinfo=timezone.utc)))

    quote = agg.latest_quote("EURUSD")

    assert quote is not None
    assert quote["bid"] == 1.12
    assert quote["ask"] == 1.13


def test_gap_handling_closes_previous_bar_without_synthesizing_missing_minutes() -> None:
    agg = FXTickAggregator()
    agg.add_tick(_tick("EURUSD", 1.10, 1.11, datetime(2026, 3, 11, 12, 0, 5, tzinfo=timezone.utc)))
    agg.add_tick(_tick("EURUSD", 1.15, 1.16, datetime(2026, 3, 11, 12, 3, 5, tzinfo=timezone.utc)))

    bars = agg.get_closed_bars("EURUSD", "1m", limit=10)

    assert len(bars) == 1
    assert bars[0].start_time == datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc)
