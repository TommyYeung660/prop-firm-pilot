"""
Execution Engine — processes trade intents that are ready for execution.

Reads ready_for_exec intents from the DecisionStore, runs PropFirmGuard
compliance checks, calculates position size, and executes trades via
MatchTraderClient. All state transitions are persisted atomically.

This module is called by the Scheduler's execution loop. All methods are
synchronous (store ops) or async (API calls), with the caller responsible
for asyncio.to_thread() wrapping where needed.

Usage:
    engine = ExecutionEngine(store, guard, matchtrader, sizer, config)
    await engine.execute_ready_intents()
"""

import asyncio
import json
from typing import Any, Literal

from loguru import logger

from src.compliance.prop_firm_guard import (
    AccountSnapshot,
    ComplianceResult,
    PropFirmGuard,
    TradePlan,
)
from src.config import AppConfig
from src.decision.decision_formatter import DEFAULT_SL_TP
from src.decision.schemas import TradeIntent
from src.decision_store.sqlite_store import DecisionStore
from src.execution.instrument_registry import InstrumentRegistry
from src.execution.matchtrader_client import MatchTraderClient
from src.execution.position_sizer import PositionSizer
from src.monitor.alert_service import AlertService

# ── Exceptions ──────────────────────────────────────────────────────────────


class ExecutionEngineError(Exception):
    """Base exception for execution engine errors."""


class ComplianceRejectedError(ExecutionEngineError):
    """Trade rejected by compliance checks."""


# ── ExecutionEngine ─────────────────────────────────────────────────────────


class ExecutionEngine:
    """Processes ready_for_exec intents through compliance and execution.

    Responsibilities:
    - Poll DecisionStore for ready_for_exec intents
    - Build TradePlan from intent fields + position sizing
    - Run PropFirmGuard.check_all() compliance gate
    - Execute via MatchTraderClient.open_position()
    - Update DecisionStore with outcome (opened/rejected/failed)

    Usage:
        engine = ExecutionEngine(store, guard, matchtrader, sizer, config)
        await engine.execute_ready_intents()
    """

    def __init__(
        self,
        store: DecisionStore,
        guard: PropFirmGuard,
        matchtrader: MatchTraderClient,
        sizer: PositionSizer,
        config: AppConfig,
        alert_service: AlertService | None = None,
        instrument_registry: InstrumentRegistry | None = None,
    ) -> None:
        self._store = store
        self._guard = guard
        self._matchtrader = matchtrader
        self._sizer = sizer
        self._config = config
        self._alert_service = alert_service
        self._registry = instrument_registry

    # ── Public API ──────────────────────────────────────────────────────

    async def execute_ready_intents(self) -> int:
        """Process all ready_for_exec intents through compliance and execution.

        Returns:
            Number of intents processed (regardless of outcome).
        """
        intents = await asyncio.to_thread(self._store.get_ready_intents)
        if not intents:
            return 0

        logger.info("ExecutionEngine: found {} ready intents", len(intents))
        processed = 0

        for intent in intents:
            try:
                await self._execute_single_intent(intent)
            except Exception as e:
                logger.error(
                    "ExecutionEngine: unexpected error processing intent {}: {}",
                    intent.id,
                    e,
                )
            processed += 1

        return processed

    # ── Internal Pipeline ───────────────────────────────────────────────

    async def _execute_single_intent(self, intent: TradeIntent) -> None:
        """Execute a single intent through the full pipeline.

        Pipeline:
        1. Mark as executing
        2. Build TradePlan (position sizing + SL/TP)
        3. Get AccountSnapshot from broker
        4. Run compliance checks
        5. Apply random delay (anti-duplicate-strategy)
        6. Execute trade via MatchTrader API (using broker symbol from registry)
        7. Update store with outcome
        8. Send Telegram alert
        """
        intent_id = intent.id
        symbol = intent.symbol
        side = intent.suggested_side

        # Pre-check: must have a valid BUY/SELL side
        if side not in ("BUY", "SELL"):
            logger.error(
                "ExecutionEngine: intent {} has invalid side '{}', skipping", intent_id, side
            )
            return

        # Resolve broker symbol (e.g. EURUSD → EURUSD.)
        broker_symbol = self._resolve_broker_symbol(symbol)

        # Step 1: Mark as executing
        try:
            await asyncio.to_thread(self._store.mark_executing, intent_id)
        except Exception as e:
            logger.error("ExecutionEngine: cannot mark {} executing: {}", intent_id, e)
            return

        # Step 2: Build TradePlan
        try:
            account_snapshot = await self._get_account_snapshot()
            trade_plan = self._build_trade_plan(intent, side, account_snapshot.equity)
        except Exception as e:
            logger.error("ExecutionEngine: failed to build trade plan for {}: {}", intent_id, e)
            await asyncio.to_thread(
                self._store.mark_failed, intent_id, f"Trade plan build error: {e}"
            )
            return

        # Step 3: Compliance gate
        compliance_result = self._guard.check_all(trade_plan, account_snapshot)
        compliance_snapshot = self._serialize_compliance(compliance_result, account_snapshot)

        if not compliance_result.passed:
            logger.warning(
                "ExecutionEngine: intent {} rejected by compliance: {}",
                intent_id,
                compliance_result.reason,
            )
            # Store compliance snapshot before rejecting
            await asyncio.to_thread(
                self._update_compliance_snapshot, intent_id, compliance_snapshot
            )
            await asyncio.to_thread(self._store.mark_rejected, intent_id, compliance_result.reason)
            await self._send_alert_rejection(symbol, side, compliance_result.reason)
            return

        # Step 4: Random delay (anti-duplicate-strategy detection by E8)
        delay = self._guard.add_random_delay()
        logger.debug("ExecutionEngine: applying {:.2f}s random delay for {}", delay, intent_id)
        await asyncio.sleep(delay)

        # Step 5: Execute trade (use broker symbol for API call)
        try:
            order = await self._matchtrader.open_position(
                symbol=broker_symbol,
                side=side,
                volume=trade_plan.volume,
            )

            # Store compliance snapshot regardless of outcome
            await asyncio.to_thread(
                self._update_compliance_snapshot, intent_id, compliance_snapshot
            )

            if order.success:
                await asyncio.to_thread(self._store.mark_opened, intent_id, order.position_id)
                logger.info(
                    "ExecutionEngine: intent {} opened as position {} ({} {} {:.2f} lots)",
                    intent_id,
                    order.position_id,
                    side,
                    broker_symbol,
                    trade_plan.volume,
                )
                await self._send_alert_opened(
                    symbol, side, trade_plan.volume, account_snapshot.equity, order.position_id
                )
            else:
                await asyncio.to_thread(self._store.mark_failed, intent_id, order.message)
                logger.warning(
                    "ExecutionEngine: intent {} execution failed: {}",
                    intent_id,
                    order.message,
                )
                await self._send_alert_failed(symbol, side, order.message)
        except Exception as e:
            logger.error("ExecutionEngine: API error on intent {}: {}", intent_id, e)
            await asyncio.to_thread(self._store.mark_failed, intent_id, str(e))
            await self._send_alert_failed(symbol, side, str(e))

    # ── Helpers ─────────────────────────────────────────────────────────

    def _build_trade_plan(
        self,
        intent: TradeIntent,
        side: Literal["BUY", "SELL"],
        account_equity: float,
    ) -> TradePlan:
        """Build a TradePlan from intent fields and position sizing.

        Uses the intent's suggested_sl_pips/tp_pips if set by the LLM worker,
        otherwise falls back to DEFAULT_SL_TP per instrument.

        Args:
            intent: The trade intent with decision fields populated.
            side: Validated BUY or SELL direction.
            account_equity: Current account equity for position sizing.

        Returns:
            TradePlan ready for compliance checking.
        """
        symbol = intent.symbol

        # SL/TP: prefer intent values, fallback to instrument defaults
        defaults = DEFAULT_SL_TP.get(symbol, {"sl_pips": 50, "tp_pips": 100})
        sl_pips = intent.suggested_sl_pips if intent.suggested_sl_pips else defaults["sl_pips"]
        tp_pips = intent.suggested_tp_pips if intent.suggested_tp_pips else defaults["tp_pips"]

        # Position sizing
        volume = self._sizer.calculate_volume(symbol, account_equity, sl_pips)
        risk_amount = self._sizer.calculate_risk_amount(symbol, volume, sl_pips)

        logger.debug(
            "ExecutionEngine: trade plan for {} — {} {:.2f} lots, "
            "SL={:.0f}p TP={:.0f}p risk=${:.2f}",
            symbol,
            side,
            volume,
            sl_pips,
            tp_pips,
            risk_amount,
        )

        return TradePlan(
            symbol=symbol,
            side=side,
            volume=volume,
            stop_loss=sl_pips,
            take_profit=tp_pips,
            risk_amount=risk_amount,
        )

    async def _get_account_snapshot(self) -> AccountSnapshot:
        """Fetch current account state from MatchTrader for compliance checks.

        Returns:
            AccountSnapshot with balance, equity, margin, and position count.
        """
        balance_info = await self._matchtrader.get_balance()
        positions = await self._matchtrader.get_open_positions()

        # Calculate daily PnL from open positions
        daily_pnl = sum(p.profit for p in positions)

        return AccountSnapshot(
            balance=balance_info.balance,
            equity=balance_info.equity,
            margin=balance_info.margin,
            free_margin=balance_info.free_margin,
            day_start_balance=balance_info.balance - daily_pnl,
            initial_balance=self._config.account.initial_balance,
            open_positions=len(positions),
            daily_pnl=daily_pnl,
            total_pnl=balance_info.balance - self._config.account.initial_balance,
        )

    def _update_compliance_snapshot(self, intent_id: str, snapshot_json: str) -> None:
        """Persist compliance check results on the intent for audit trail."""
        self._store._conn.execute(
            "UPDATE intents SET compliance_snapshot = :snap WHERE id = :id",
            {"snap": snapshot_json, "id": intent_id},
        )
        self._store._conn.commit()

    @staticmethod
    def _serialize_compliance(result: ComplianceResult, snapshot: AccountSnapshot) -> str:
        """Serialize compliance result and account snapshot for audit storage."""
        data: dict[str, Any] = {
            "passed": result.passed,
            "rule_name": result.rule_name,
            "reason": result.reason,
            "details": result.details,
            "account": {
                "balance": snapshot.balance,
                "equity": snapshot.equity,
                "margin": snapshot.margin,
                "free_margin": snapshot.free_margin,
                "day_start_balance": snapshot.day_start_balance,
                "initial_balance": snapshot.initial_balance,
                "open_positions": snapshot.open_positions,
                "daily_pnl": snapshot.daily_pnl,
                "total_pnl": snapshot.total_pnl,
            },
        }
        return json.dumps(data, default=str)

    # ── Symbol Resolution ───────────────────────────────────────────────

    def _resolve_broker_symbol(self, config_symbol: str) -> str:
        """Resolve a config symbol to its broker symbol via InstrumentRegistry.

        Falls back to using the config symbol as-is if no registry is set.

        Args:
            config_symbol: Symbol from config/intent (e.g. "EURUSD").

        Returns:
            Broker symbol (e.g. "EURUSD.") for use in MatchTrader API calls.
        """
        if self._registry is not None:
            try:
                return self._registry.to_broker(config_symbol)
            except KeyError:
                logger.warning(
                    "ExecutionEngine: symbol '{}' not in registry, using as-is",
                    config_symbol,
                )
        return config_symbol

    # ── Alert Helpers ───────────────────────────────────────────────────

    async def _send_alert_opened(
        self,
        symbol: str,
        side: str,
        volume: float,
        equity: float,
        position_id: str,
    ) -> None:
        """Send Telegram notification for a successfully opened trade."""
        if self._alert_service is not None:
            try:
                await self._alert_service.trade_opened(
                    symbol=symbol,
                    side=side,
                    volume=volume,
                    price=0.0,  # Filled from position query if needed
                    equity=equity,
                    position_id=position_id,
                )
            except Exception as e:
                logger.error("ExecutionEngine: alert failed for trade opened: {}", e)

    async def _send_alert_rejection(self, symbol: str, side: str, reason: str) -> None:
        """Send Telegram notification for a compliance rejection."""
        if self._alert_service is not None:
            try:
                await self._alert_service.compliance_rejection(symbol, side, reason)
            except Exception as e:
                logger.error("ExecutionEngine: alert failed for rejection: {}", e)

    async def _send_alert_failed(self, symbol: str, side: str, error: str) -> None:
        """Send Telegram notification for a failed trade execution."""
        if self._alert_service is not None:
            try:
                await self._alert_service.system_error(f"Trade failed: {side} {symbol} — {error}")
            except Exception as e:
                logger.error("ExecutionEngine: alert failed for error: {}", e)
