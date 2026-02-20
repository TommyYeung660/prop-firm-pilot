"""
Tests for src/execution/instrument_registry.py — Bidirectional symbol mapper.

Tests cover the InstrumentRegistry class: factory construction, bidirectional
mapping (to_broker / to_config), safe fallback, info lookup, and edge cases
like case-insensitive matching and dot-suffix handling.
"""

from unittest.mock import AsyncMock

import pytest

from src.execution.instrument_registry import InstrumentRegistry
from src.execution.matchtrader_client import InstrumentInfo

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_instrument(
    symbol: str,
    type_: str = "FOREX",
    leverage: float = 100.0,
    session_open: bool = True,
    volume_min: float = 0.01,
    volume_max: float = 50.0,
) -> InstrumentInfo:
    """Create a minimal InstrumentInfo for testing."""
    return InstrumentInfo(
        symbol=symbol,
        type=type_,
        leverage=leverage,
        sessionOpen=session_open,
        volumeMin=volume_min,
        volumeMax=volume_max,
    )


def _make_registry(
    config_to_broker: dict[str, str] | None = None,
    broker_to_config: dict[str, str] | None = None,
    instruments: dict[str, InstrumentInfo] | None = None,
    untradeable: list[str] | None = None,
) -> InstrumentRegistry:
    """Create an InstrumentRegistry directly (bypassing async factory)."""
    return InstrumentRegistry(
        config_to_broker=config_to_broker or {},
        broker_to_config=broker_to_config or {},
        instruments=instruments or {},
        untradeable=untradeable or [],
    )


# ── to_broker Tests ─────────────────────────────────────────────────────────


class TestToBroker:
    """Tests for InstrumentRegistry.to_broker()."""

    def test_maps_config_to_broker(self) -> None:
        """Should return broker symbol with dot suffix."""
        registry = _make_registry(
            config_to_broker={"EURUSD": "EURUSD.", "GBPUSD": "GBPUSD."},
        )
        assert registry.to_broker("EURUSD") == "EURUSD."
        assert registry.to_broker("GBPUSD") == "GBPUSD."

    def test_raises_key_error_for_unknown_symbol(self) -> None:
        """Should raise KeyError for symbols not in registry."""
        registry = _make_registry(config_to_broker={"EURUSD": "EURUSD."})
        with pytest.raises(KeyError, match="XAUUSD"):
            registry.to_broker("XAUUSD")

    def test_error_message_includes_available(self) -> None:
        """KeyError message should list available symbols."""
        registry = _make_registry(config_to_broker={"EURUSD": "EURUSD."})
        with pytest.raises(KeyError, match="Available"):
            registry.to_broker("NOPE")


# ── to_config Tests ─────────────────────────────────────────────────────────


class TestToConfig:
    """Tests for InstrumentRegistry.to_config()."""

    def test_maps_broker_to_config(self) -> None:
        """Should return config symbol without dot suffix."""
        registry = _make_registry(
            broker_to_config={"EURUSD.": "EURUSD", "GBPUSD.": "GBPUSD"},
        )
        assert registry.to_config("EURUSD.") == "EURUSD"
        assert registry.to_config("GBPUSD.") == "GBPUSD"

    def test_raises_key_error_for_unknown_broker_symbol(self) -> None:
        """Should raise KeyError for broker symbols not in registry."""
        registry = _make_registry(broker_to_config={"EURUSD.": "EURUSD"})
        with pytest.raises(KeyError, match="XAUUSD"):
            registry.to_config("XAUUSD.")


# ── to_config_safe Tests ────────────────────────────────────────────────────


class TestToConfigSafe:
    """Tests for InstrumentRegistry.to_config_safe()."""

    def test_returns_config_symbol_when_mapped(self) -> None:
        """Should return config symbol when broker symbol is in registry."""
        registry = _make_registry(broker_to_config={"EURUSD.": "EURUSD"})
        assert registry.to_config_safe("EURUSD.") == "EURUSD"

    def test_strips_dot_when_not_mapped(self) -> None:
        """Should strip trailing dot as fallback for unmapped symbols."""
        registry = _make_registry()  # Empty
        assert registry.to_config_safe("XAUUSD.") == "XAUUSD"

    def test_returns_as_is_when_no_dot(self) -> None:
        """Should return symbol unchanged if no trailing dot."""
        registry = _make_registry()
        assert registry.to_config_safe("XAUUSD") == "XAUUSD"


# ── get_info Tests ──────────────────────────────────────────────────────────


class TestGetInfo:
    """Tests for InstrumentRegistry.get_info()."""

    def test_returns_instrument_info(self) -> None:
        """Should return InstrumentInfo for a mapped config symbol."""
        info = _make_instrument("EURUSD.", leverage=100.0, volume_min=0.01)
        registry = _make_registry(instruments={"EURUSD": info})
        result = registry.get_info("EURUSD")
        assert result is not None
        assert result.symbol == "EURUSD."
        assert result.leverage == 100.0
        assert result.volume_min == 0.01

    def test_returns_none_for_unknown(self) -> None:
        """Should return None for symbols not in registry."""
        registry = _make_registry()
        assert registry.get_info("UNKNOWN") is None


# ── Properties Tests ────────────────────────────────────────────────────────


class TestProperties:
    """Tests for InstrumentRegistry properties."""

    def test_tradeable_symbols(self) -> None:
        """Should return list of config symbols that are tradeable."""
        registry = _make_registry(
            config_to_broker={"EURUSD": "EURUSD.", "GBPUSD": "GBPUSD."},
        )
        assert set(registry.tradeable_symbols) == {"EURUSD", "GBPUSD"}

    def test_untradeable_symbols(self) -> None:
        """Should return list of symbols that could not be mapped."""
        registry = _make_registry(untradeable=["XAUUSD", "BTCUSD"])
        assert registry.untradeable_symbols == ["XAUUSD", "BTCUSD"]

    def test_broker_symbols(self) -> None:
        """Should return list of broker-side symbols."""
        registry = _make_registry(
            broker_to_config={"EURUSD.": "EURUSD", "GBPUSD.": "GBPUSD"},
        )
        assert set(registry.broker_symbols) == {"EURUSD.", "GBPUSD."}

    def test_is_tradeable_true(self) -> None:
        """Should return True for tradeable config symbols."""
        registry = _make_registry(config_to_broker={"EURUSD": "EURUSD."})
        assert registry.is_tradeable("EURUSD") is True

    def test_is_tradeable_false(self) -> None:
        """Should return False for untradeable symbols."""
        registry = _make_registry(config_to_broker={"EURUSD": "EURUSD."})
        assert registry.is_tradeable("XAUUSD") is False

    def test_len(self) -> None:
        """Should return count of tradeable symbols."""
        registry = _make_registry(
            config_to_broker={"EURUSD": "EURUSD.", "GBPUSD": "GBPUSD."},
        )
        assert len(registry) == 2

    def test_repr(self) -> None:
        """Should include tradeable/untradeable counts."""
        registry = _make_registry(
            config_to_broker={"EURUSD": "EURUSD."},
            untradeable=["XAUUSD"],
        )
        r = repr(registry)
        assert "tradeable=1" in r
        assert "untradeable=1" in r


# ── Factory Tests ───────────────────────────────────────────────────────────


class TestFromMatchTrader:
    """Tests for InstrumentRegistry.from_matchtrader() async factory."""

    async def test_maps_dot_suffix_symbols(self) -> None:
        """Should match 'EURUSD' config → 'EURUSD.' broker."""
        client = AsyncMock()
        client.get_effective_instruments.return_value = [
            _make_instrument("EURUSD.", type_="FOREX"),
            _make_instrument("GBPUSD.", type_="FOREX"),
        ]

        registry = await InstrumentRegistry.from_matchtrader(client, ["EURUSD", "GBPUSD"])

        assert registry.to_broker("EURUSD") == "EURUSD."
        assert registry.to_broker("GBPUSD") == "GBPUSD."
        assert registry.to_config("EURUSD.") == "EURUSD"
        assert len(registry) == 2

    async def test_exact_match_without_dot(self) -> None:
        """Should match when broker symbol has no dot suffix."""
        client = AsyncMock()
        client.get_effective_instruments.return_value = [
            _make_instrument("XAUUSD", type_="CFD"),
        ]

        registry = await InstrumentRegistry.from_matchtrader(client, ["XAUUSD"])

        assert registry.to_broker("XAUUSD") == "XAUUSD"
        assert len(registry) == 1

    async def test_marks_unmatched_as_untradeable(self) -> None:
        """Should record symbols not found in broker instruments."""
        client = AsyncMock()
        client.get_effective_instruments.return_value = [
            _make_instrument("EURUSD.", type_="FOREX"),
        ]

        registry = await InstrumentRegistry.from_matchtrader(client, ["EURUSD", "XYZABC"])

        assert registry.is_tradeable("EURUSD") is True
        assert registry.is_tradeable("XYZABC") is False
        assert "XYZABC" in registry.untradeable_symbols

    async def test_case_insensitive_matching(self) -> None:
        """Should match symbols case-insensitively as fallback."""
        client = AsyncMock()
        client.get_effective_instruments.return_value = [
            _make_instrument("eurusd.", type_="FOREX"),
        ]

        registry = await InstrumentRegistry.from_matchtrader(client, ["EURUSD"])

        assert registry.to_broker("EURUSD") == "eurusd."
        assert len(registry) == 1

    async def test_empty_config_symbols(self) -> None:
        """Should handle empty config symbols list."""
        client = AsyncMock()
        client.get_effective_instruments.return_value = [
            _make_instrument("EURUSD.", type_="FOREX"),
        ]

        registry = await InstrumentRegistry.from_matchtrader(client, [])

        assert len(registry) == 0
        assert registry.tradeable_symbols == []
        assert registry.untradeable_symbols == []

    async def test_empty_effective_instruments(self) -> None:
        """Should mark all config symbols as untradeable when broker has none."""
        client = AsyncMock()
        client.get_effective_instruments.return_value = []

        registry = await InstrumentRegistry.from_matchtrader(client, ["EURUSD", "GBPUSD"])

        assert len(registry) == 0
        assert set(registry.untradeable_symbols) == {"EURUSD", "GBPUSD"}

    async def test_preserves_instrument_info(self) -> None:
        """Should store InstrumentInfo accessible via get_info()."""
        inst = _make_instrument("EURUSD.", type_="FOREX", leverage=200.0, volume_min=0.1)
        client = AsyncMock()
        client.get_effective_instruments.return_value = [inst]

        registry = await InstrumentRegistry.from_matchtrader(client, ["EURUSD"])

        info = registry.get_info("EURUSD")
        assert info is not None
        assert info.leverage == 200.0
        assert info.volume_min == 0.1
        assert info.type == "FOREX"
