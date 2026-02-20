"""
Instrument registry — maps config symbols to MatchTrader broker symbols.

E8 Markets account 950552 uses dot-suffix symbols (e.g. "EURUSD." instead
of "EURUSD"). This registry is built at startup from get_effective_instruments()
and provides bidirectional mapping between config symbols (base) and broker
symbols (with suffix).

Also validates that all configured trading symbols are actually tradeable
on the connected account.

Usage:
    registry = await InstrumentRegistry.from_matchtrader(client, config_symbols)
    broker_sym = registry.to_broker("EURUSD")      # → "EURUSD."
    config_sym = registry.to_config("EURUSD.")      # → "EURUSD"
    info = registry.get_info("EURUSD")               # → InstrumentInfo
"""

from loguru import logger

from src.execution.matchtrader_client import InstrumentInfo, MatchTraderClient


class InstrumentRegistryError(Exception):
    """Error during instrument registry initialization."""


class InstrumentRegistry:
    """Bidirectional symbol mapper between config and MatchTrader broker symbols.

    Built at startup from the broker's effective instruments list. Validates
    that all configured symbols are tradeable before the scheduler starts.

    Usage:
        registry = await InstrumentRegistry.from_matchtrader(client, ["EURUSD", "GBPUSD"])
        broker_sym = registry.to_broker("EURUSD")  # → "EURUSD."
    """

    def __init__(
        self,
        config_to_broker: dict[str, str],
        broker_to_config: dict[str, str],
        instruments: dict[str, InstrumentInfo],
        untradeable: list[str],
    ) -> None:
        self._config_to_broker = config_to_broker
        self._broker_to_config = broker_to_config
        self._instruments = instruments  # keyed by config symbol
        self._untradeable = untradeable

    # ── Factory ─────────────────────────────────────────────────────────

    @classmethod
    async def from_matchtrader(
        cls,
        client: MatchTraderClient,
        config_symbols: list[str],
    ) -> "InstrumentRegistry":
        """Build registry from live MatchTrader effective instruments.

        Fetches the account's effective instruments, matches each config symbol
        to its broker counterpart (exact match or dot-suffix), and logs any
        symbols that cannot be traded.

        Args:
            client: Authenticated MatchTraderClient.
            config_symbols: Symbols from AppConfig.symbols (e.g. ["EURUSD", "GBPUSD"]).

        Returns:
            InstrumentRegistry ready for bidirectional lookups.
        """
        effective = await client.get_effective_instruments()

        # Build lookup: strip trailing dots/suffixes for matching
        # Broker symbols may be "EURUSD.", "EURUSD", or other patterns
        broker_lookup: dict[str, InstrumentInfo] = {}
        for inst in effective:
            broker_lookup[inst.symbol] = inst
            # Also index by stripped version for fuzzy matching
            stripped = inst.symbol.rstrip(".")
            if stripped != inst.symbol:
                broker_lookup[stripped] = inst

        config_to_broker: dict[str, str] = {}
        broker_to_config: dict[str, str] = {}
        instruments: dict[str, InstrumentInfo] = {}
        untradeable: list[str] = []

        for config_sym in config_symbols:
            # Try exact match first, then with dot suffix
            info = broker_lookup.get(config_sym)
            if info is None:
                info = broker_lookup.get(f"{config_sym}.")
            if info is None:
                # Try case-insensitive
                for key, val in broker_lookup.items():
                    if key.upper().rstrip(".") == config_sym.upper():
                        info = val
                        break

            if info is not None:
                config_to_broker[config_sym] = info.symbol
                broker_to_config[info.symbol] = config_sym
                instruments[config_sym] = info
                logger.info(
                    "InstrumentRegistry: {} → {} (type={}, leverage={}, "
                    "lot={}-{}, session_open={})",
                    config_sym,
                    info.symbol,
                    info.type,
                    info.leverage,
                    info.volume_min,
                    info.volume_max,
                    info.session_open,
                )
            else:
                untradeable.append(config_sym)
                logger.warning(
                    "InstrumentRegistry: {} NOT FOUND in effective instruments — "
                    "this symbol CANNOT be traded on this account",
                    config_sym,
                )

        logger.info(
            "InstrumentRegistry: {}/{} symbols mapped, {} untradeable",
            len(config_to_broker),
            len(config_symbols),
            len(untradeable),
        )

        return cls(config_to_broker, broker_to_config, instruments, untradeable)

    # ── Public API ──────────────────────────────────────────────────────

    def to_broker(self, config_symbol: str) -> str:
        """Convert a config symbol to its broker symbol.

        Args:
            config_symbol: Symbol from config (e.g. "EURUSD").

        Returns:
            Broker symbol (e.g. "EURUSD.").

        Raises:
            KeyError: If the symbol is not in the registry.
        """
        if config_symbol not in self._config_to_broker:
            raise KeyError(
                f"Symbol '{config_symbol}' not in registry. "
                f"Available: {list(self._config_to_broker.keys())}"
            )
        return self._config_to_broker[config_symbol]

    def to_config(self, broker_symbol: str) -> str:
        """Convert a broker symbol back to its config symbol.

        Args:
            broker_symbol: Symbol from MatchTrader (e.g. "EURUSD.").

        Returns:
            Config symbol (e.g. "EURUSD").

        Raises:
            KeyError: If the symbol is not in the registry.
        """
        if broker_symbol not in self._broker_to_config:
            raise KeyError(
                f"Broker symbol '{broker_symbol}' not in registry. "
                f"Available: {list(self._broker_to_config.keys())}"
            )
        return self._broker_to_config[broker_symbol]

    def to_config_safe(self, broker_symbol: str) -> str:
        """Convert a broker symbol to config symbol, falling back to stripping dots.

        Unlike to_config(), this never raises — used for position monitoring where
        we may encounter broker symbols not in our config.

        Args:
            broker_symbol: Symbol from MatchTrader (e.g. "EURUSD.").

        Returns:
            Config symbol (e.g. "EURUSD") or the stripped broker symbol as fallback.
        """
        if broker_symbol in self._broker_to_config:
            return self._broker_to_config[broker_symbol]
        return broker_symbol.rstrip(".")

    def get_info(self, config_symbol: str) -> InstrumentInfo | None:
        """Get full InstrumentInfo for a config symbol.

        Args:
            config_symbol: Symbol from config (e.g. "EURUSD").

        Returns:
            InstrumentInfo from MatchTrader, or None if not found.
        """
        return self._instruments.get(config_symbol)

    @property
    def tradeable_symbols(self) -> list[str]:
        """Config symbols that are tradeable on this account."""
        return list(self._config_to_broker.keys())

    @property
    def untradeable_symbols(self) -> list[str]:
        """Config symbols that could NOT be matched to broker instruments."""
        return list(self._untradeable)

    @property
    def broker_symbols(self) -> list[str]:
        """All broker-side symbols in the registry."""
        return list(self._broker_to_config.keys())

    def is_tradeable(self, config_symbol: str) -> bool:
        """Check if a config symbol is tradeable on this account."""
        return config_symbol in self._config_to_broker

    def __len__(self) -> int:
        return len(self._config_to_broker)

    def __repr__(self) -> str:
        return (
            f"InstrumentRegistry(tradeable={len(self._config_to_broker)}, "
            f"untradeable={len(self._untradeable)})"
        )
