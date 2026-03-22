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
import math
from datetime import datetime, timezone
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
from src.execution.capital_allocator import (
    BoundedCapitalAllocator,
    CapitalAllocationDecision,
)
from src.execution.instrument_registry import InstrumentRegistry
from src.execution.matchtrader_client import MatchTraderClient
from src.execution.pip_value_resolver import (
    quote_to_reference_price,
    resolve_usd_pip_value_for_symbol,
)
from src.execution.portfolio_risk_guard import (
    OpenPositionRiskSnapshot,
    PortfolioRiskDecision,
    PortfolioRiskGuard,
)
from src.execution.position_sizer import PositionSizer
from src.monitor.alert_service import AlertService
from src.monitor.trade_journal import TradeJournal

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
        trade_journal: TradeJournal | None = None,
    ) -> None:
        self._store = store
        self._guard = guard
        self._matchtrader = matchtrader
        self._sizer = sizer
        self._config = config
        self._capital_allocator = BoundedCapitalAllocator(config.execution)
        self._portfolio_risk_guard = PortfolioRiskGuard(config.execution)
        self._alert_service = alert_service
        self._registry = instrument_registry
        self._trade_journal = trade_journal

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

        # Step 2: Build account snapshot
        try:
            broker_positions = await self._matchtrader.get_open_positions()
            account_snapshot = await self._get_account_snapshot(broker_positions=broker_positions)
        except Exception as e:
            logger.error(
                "ExecutionEngine: failed to build account snapshot for {}: {}",
                intent_id,
                e,
            )
            await asyncio.to_thread(
                self._store.mark_failed, intent_id, f"Account snapshot error: {e}"
            )
            return

        # Step 2.5: Execution-side Best Day hard gate (race-condition guard)
        best_day_block_reason = self._get_best_day_entry_block_reason(account_snapshot)
        if best_day_block_reason is not None:
            compliance_result = ComplianceResult(
                passed=False,
                rule_name="BEST_DAY_ENTRY_GATE",
                reason=best_day_block_reason,
            )
            compliance_snapshot = self._serialize_compliance(compliance_result, account_snapshot)
            await asyncio.to_thread(
                self._update_compliance_snapshot, intent_id, compliance_snapshot
            )
            await asyncio.to_thread(self._store.mark_rejected, intent_id, best_day_block_reason)
            await self._send_alert_rejection(symbol, side, best_day_block_reason)
            self._log_trade_event(
                "TRADE_REJECTED",
                {
                    "intent_id": intent_id,
                    "symbol": symbol,
                    "side": side,
                    "reason": best_day_block_reason,
                },
            )
            logger.warning(
                "ExecutionEngine: intent {} rejected by execution hard gate: {}",
                intent_id,
                best_day_block_reason,
            )
            return

        # Step 3: Build TradePlan
        try:
            pip_value_override = await self._resolve_live_pip_value_override(symbol)
            trade_plan, allocation = self._build_trade_plan(
                intent,
                side,
                account_snapshot.equity,
                account_snapshot.open_positions,
                pip_value_override=pip_value_override,
            )
        except Exception as e:
            logger.error("ExecutionEngine: failed to build trade plan for {}: {}", intent_id, e)
            await asyncio.to_thread(
                self._store.mark_failed, intent_id, f"Trade plan build error: {e}"
            )
            return

        # Step 3.5: Execution-side portfolio risk guard
        portfolio_risk_decision = await self._evaluate_portfolio_risk_decision(
            intent=intent,
            side=side,
            next_risk_pct=allocation.effective_risk_pct,
            broker_positions=broker_positions,
        )
        portfolio_risk_payload = portfolio_risk_decision.to_dict()

        if not portfolio_risk_decision.allowed:
            rejection_reason = portfolio_risk_decision.reason_code
            compliance_result = ComplianceResult(
                passed=False,
                rule_name="PORTFOLIO_RISK",
                reason=rejection_reason,
                details={"portfolio_risk": portfolio_risk_payload},
            )
            compliance_snapshot = self._serialize_compliance(
                compliance_result,
                account_snapshot,
                portfolio_risk_meta=portfolio_risk_payload,
            )
            execution_meta = self._build_execution_meta(
                fill_price=None,
                volume=trade_plan.volume,
                side=side,
                sl_price=None,
                tp_price=None,
                sl_pips=trade_plan.stop_loss,
                tp_pips=trade_plan.take_profit,
                pre_trade_bid=None,
                pre_trade_ask=None,
                slippage_pips=None,
                execution_latency_ms=None,
                random_delay_seconds=0.0,
                compliance_passed=False,
                order_raw_response={},
                risk_pct=allocation.effective_risk_pct,
                capital_allocation_meta=allocation.to_dict(),
                portfolio_risk_meta=portfolio_risk_payload,
            )
            await asyncio.to_thread(
                self._update_compliance_snapshot, intent_id, compliance_snapshot
            )
            await asyncio.to_thread(self._store.update_execution_meta, intent_id, execution_meta)
            await asyncio.to_thread(self._store.mark_rejected, intent_id, rejection_reason)
            await self._send_alert_rejection(symbol, side, rejection_reason)
            self._log_trade_event(
                "TRADE_REJECTED",
                {
                    "intent_id": intent_id,
                    "symbol": symbol,
                    "side": side,
                    "reason": rejection_reason,
                    "portfolio_risk": portfolio_risk_payload,
                },
            )
            logger.warning(
                "ExecutionEngine: intent {} rejected by portfolio risk guard: {}",
                intent_id,
                rejection_reason,
            )
            return

        # Step 4: Compliance gate
        compliance_result = self._guard.check_all(trade_plan, account_snapshot)
        compliance_snapshot = self._serialize_compliance(
            compliance_result,
            account_snapshot,
            portfolio_risk_meta=portfolio_risk_payload,
        )

        if not compliance_result.passed:
            logger.warning(
                "ExecutionEngine: intent {} rejected by compliance: {}",
                intent_id,
                compliance_result.reason,
            )
            execution_meta = self._build_execution_meta(
                fill_price=None,
                volume=trade_plan.volume,
                side=side,
                sl_price=None,
                tp_price=None,
                sl_pips=trade_plan.stop_loss,
                tp_pips=trade_plan.take_profit,
                pre_trade_bid=None,
                pre_trade_ask=None,
                slippage_pips=None,
                execution_latency_ms=None,
                random_delay_seconds=0.0,
                compliance_passed=False,
                order_raw_response={},
                risk_pct=allocation.effective_risk_pct,
                capital_allocation_meta=allocation.to_dict(),
                portfolio_risk_meta=portfolio_risk_payload,
            )
            # Store compliance snapshot before rejecting
            await asyncio.to_thread(
                self._update_compliance_snapshot, intent_id, compliance_snapshot
            )
            await asyncio.to_thread(self._store.update_execution_meta, intent_id, execution_meta)
            await asyncio.to_thread(self._store.mark_rejected, intent_id, compliance_result.reason)
            await self._send_alert_rejection(symbol, side, compliance_result.reason)
            self._log_trade_event(
                "TRADE_REJECTED",
                {
                    "intent_id": intent_id,
                    "symbol": symbol,
                    "side": side,
                    "reason": compliance_result.reason,
                },
            )
            return

        # Step 4.5: Pre-trade quote validation (slippage protection)
        relevant_price = None
        pre_trade_bid: float | None = None
        pre_trade_ask: float | None = None
        try:
            quote = await self._matchtrader.get_quote(broker_symbol)
            pre_trade_bid = quote.bid
            pre_trade_ask = quote.ask
            relevant_price = quote.ask if side == "BUY" else quote.bid
            logger.info(
                "ExecutionEngine: pre-trade quote for {} — bid={}, ask={}, using {}={:.5f}",
                broker_symbol,
                quote.bid,
                quote.ask,
                "ask" if side == "BUY" else "bid",
                relevant_price,
            )
        except Exception as e:
            relevant_price = None
            logger.warning(
                "ExecutionEngine: could not fetch pre-trade quote for {}: {} "
                "— proceeding without slippage check",
                broker_symbol,
                e,
            )

        # Step 5: Random delay (anti-duplicate-strategy detection by E8)
        delay = self._guard.add_random_delay()
        logger.debug("ExecutionEngine: applying {:.2f}s random delay for {}", delay, intent_id)
        await asyncio.sleep(delay)

        # Step 6: Execute trade (use broker symbol for API call)
        exec_start_time = asyncio.get_event_loop().time()
        try:
            order = await self._matchtrader.open_position(
                symbol=broker_symbol,
                side=side,
                volume=trade_plan.volume,
            )
            exec_latency_ms = (asyncio.get_event_loop().time() - exec_start_time) * 1000

            # Store compliance snapshot regardless of outcome
            await asyncio.to_thread(
                self._update_compliance_snapshot, intent_id, compliance_snapshot
            )

            if order.success:
                await asyncio.to_thread(self._store.mark_opened, intent_id, order.position_id)
                self._log_trade_event(
                    "TRADE_OPENED",
                    {
                        "intent_id": intent_id,
                        "symbol": symbol,
                        "side": side,
                        "position_id": order.position_id,
                        "volume": trade_plan.volume,
                    },
                )
                logger.info(
                    "ExecutionEngine: intent {} opened as position {} ({} {} {:.2f} lots)",
                    intent_id,
                    order.position_id,
                    side,
                    broker_symbol,
                    trade_plan.volume,
                )

                # Post-trade slippage check
                if relevant_price is not None and relevant_price > 0:
                    fill_price = self._extract_open_price(order.raw_response)
                    if fill_price is not None:
                        instrument = self._config.instruments.get(symbol)
                        if instrument is not None:
                            pip_size = instrument.pip_size
                            max_slippage = self._config.execution.max_slippage_pips * pip_size
                            slippage = abs(fill_price - relevant_price)
                            if slippage > max_slippage:
                                logger.warning(
                                    "ExecutionEngine: SLIPPAGE ALERT on {} — "
                                    "fill={:.5f} vs quote={:.5f}, slippage={:.5f} > "
                                    "max={:.5f} ({:.1f} pips)",
                                    symbol,
                                    fill_price,
                                    relevant_price,
                                    slippage,
                                    max_slippage,
                                    slippage / pip_size,
                                )
                                await self._send_alert_failed(
                                    symbol,
                                    side,
                                    f"Slippage alert: {slippage / pip_size:.1f} pips "
                                    f"(max: {self._config.execution.max_slippage_pips})",
                                )
                            else:
                                logger.debug(
                                    "ExecutionEngine: slippage OK for {} — "
                                    "{:.5f} vs {:.5f} ({:.1f} pips, max {})",
                                    symbol,
                                    fill_price,
                                    relevant_price,
                                    slippage / pip_size,
                                    self._config.execution.max_slippage_pips,
                                )
                # Set SL/TP on the opened position
                sl_price, tp_price = await self._set_sl_tp_on_position(
                    position_id=order.position_id,
                    broker_symbol=broker_symbol,
                    config_symbol=symbol,
                    side=side,
                    volume=trade_plan.volume,
                    sl_pips=trade_plan.stop_loss,
                    tp_pips=trade_plan.take_profit,
                    raw_response=order.raw_response,
                )

                # Extract fill price for alert
                fill_price = self._extract_open_price(order.raw_response)
                if fill_price is None:
                    fill_price = await self._fetch_position_open_price(order.position_id)
                await self._send_alert_opened(
                    symbol,
                    side,
                    trade_plan.volume,
                    account_snapshot.equity,
                    order.position_id,
                    sl_price,
                    tp_price,
                    fill_price,
                )
                # Build and persist execution_meta
                meta_fill_price = self._extract_open_price(order.raw_response)
                slippage_pips_val: float | None = None
                if relevant_price is not None and meta_fill_price is not None:
                    meta_instrument = self._config.instruments.get(symbol)
                    if meta_instrument is not None:
                        slippage_pips_val = (
                            abs(meta_fill_price - relevant_price) / meta_instrument.pip_size
                        )

                # v1.3.9: Extract AB model_id from agent state
                _ab_model_id = ""
                if intent.agent_state_json:
                    try:
                        _state = json.loads(intent.agent_state_json)
                        _ab_model_id = _state.get("_model_id", "")
                    except (json.JSONDecodeError, AttributeError):
                        pass

                execution_meta = self._build_execution_meta(
                    fill_price=meta_fill_price,
                    volume=trade_plan.volume,
                    side=side,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    sl_pips=trade_plan.stop_loss,
                    tp_pips=trade_plan.take_profit,
                    pre_trade_bid=pre_trade_bid,
                    pre_trade_ask=pre_trade_ask,
                    slippage_pips=slippage_pips_val,
                    execution_latency_ms=exec_latency_ms,
                    random_delay_seconds=delay,
                    compliance_passed=True,
                    order_raw_response=order.raw_response,
                    model_id=_ab_model_id,
                    risk_pct=allocation.effective_risk_pct,
                    capital_allocation_meta=allocation.to_dict(),
                    portfolio_risk_meta=portfolio_risk_payload,
                )
                await asyncio.to_thread(
                    self._store.update_execution_meta, intent_id, execution_meta
                )
            else:
                await asyncio.to_thread(self._store.mark_failed, intent_id, order.message)
                self._log_trade_event(
                    "TRADE_FAILED",
                    {
                        "intent_id": intent_id,
                        "symbol": symbol,
                        "side": side,
                        "reason": order.message,
                    },
                )
                logger.warning(
                    "ExecutionEngine: intent {} execution failed: {}",
                    intent_id,
                    order.message,
                )
                await self._send_alert_failed(symbol, side, order.message)
        except Exception as e:
            logger.error("ExecutionEngine: API error on intent {}: {}", intent_id, e)
            await asyncio.to_thread(self._store.mark_failed, intent_id, str(e))
            self._log_trade_event(
                "TRADE_FAILED",
                {
                    "intent_id": intent_id,
                    "symbol": symbol,
                    "side": side,
                    "reason": str(e),
                },
            )
            await self._send_alert_failed(symbol, side, str(e))

    # ── Helpers ─────────────────────────────────────────────────────────

    def _build_trade_plan(
        self,
        intent: TradeIntent,
        side: Literal["BUY", "SELL"],
        account_equity: float,
        open_positions: int,
        pip_value_override: float | None = None,
    ) -> tuple[TradePlan, CapitalAllocationDecision]:
        """Build a TradePlan from intent fields and position sizing.

        Uses the intent's suggested_sl_pips/tp_pips if set by the LLM worker,
        otherwise falls back to DEFAULT_SL_TP per instrument.

        Args:
            intent: The trade intent with decision fields populated.
            side: Validated BUY or SELL direction.
            account_equity: Current account equity for position sizing.

        Returns:
            Tuple of compliance-ready TradePlan and capital allocation decision.
        """
        symbol = intent.symbol

        # SL/TP: prefer intent values, fallback to instrument defaults
        defaults = DEFAULT_SL_TP.get(symbol, {"sl_pips": 50, "tp_pips": 100})
        sl_pips = intent.suggested_sl_pips if intent.suggested_sl_pips else defaults["sl_pips"]
        tp_pips = intent.suggested_tp_pips if intent.suggested_tp_pips else defaults["tp_pips"]

        # Position sizing
        allocation = self._capital_allocator.allocate_entry_risk(
            open_positions=open_positions,
            scanner_confidence=intent.scanner_confidence,
        )
        size_kwargs: dict[str, float] = {"risk_pct_override": allocation.effective_risk_pct}
        risk_kwargs: dict[str, float] = {}
        if pip_value_override is not None:
            size_kwargs["pip_value_override"] = pip_value_override
            risk_kwargs["pip_value_override"] = pip_value_override
        volume = self._sizer.calculate_volume(symbol, account_equity, sl_pips, **size_kwargs)
        risk_amount = self._sizer.calculate_risk_amount(symbol, volume, sl_pips, **risk_kwargs)

        logger.debug(
            "ExecutionEngine: trade plan for {} — {} {:.2f} lots, "
            "SL={:.0f}p TP={:.0f}p risk=${:.2f} risk_pct={:.2%}",
            symbol,
            side,
            volume,
            sl_pips,
            tp_pips,
            risk_amount,
            allocation.effective_risk_pct,
        )

        return (
            TradePlan(
                symbol=symbol,
                side=side,
                volume=volume,
                stop_loss=sl_pips,
                take_profit=tp_pips,
                risk_amount=risk_amount,
            ),
            allocation,
        )

    async def _resolve_live_pip_value_override(self, symbol: str) -> float | None:
        """Resolve live USD pip values for JPY-quoted pairs."""
        instrument = self._config.instruments.get(symbol)
        if instrument is None or not symbol.endswith("JPY"):
            return None

        try:
            if symbol == "USDJPY":
                quote = await self._matchtrader.get_quote(self._resolve_broker_symbol(symbol))
                symbol_price = quote_to_reference_price(quote.bid, quote.ask)
                resolved = resolve_usd_pip_value_for_symbol(
                    symbol,
                    static_pip_value=instrument.pip_value,
                    symbol_price=symbol_price,
                )
            else:
                usd_jpy_symbol = self._resolve_broker_symbol("USDJPY")
                quote = await self._matchtrader.get_quote(usd_jpy_symbol)
                usd_jpy_price = quote_to_reference_price(quote.bid, quote.ask)
                resolved = resolve_usd_pip_value_for_symbol(
                    symbol,
                    static_pip_value=instrument.pip_value,
                    usd_jpy_price=usd_jpy_price,
                )
        except Exception as e:
            logger.warning(
                "ExecutionEngine: pip-value override unavailable for {}: {}. "
                "Using static pip value {}",
                symbol,
                e,
                instrument.pip_value,
            )
            return instrument.pip_value

        logger.debug(
            "ExecutionEngine: resolved pip value for {} -> ${:.4f}/pip/lot",
            symbol,
            resolved,
        )
        return resolved

    async def _get_account_snapshot(
        self,
        broker_positions: list[Any] | None = None,
    ) -> AccountSnapshot:
        """Fetch current account state from MatchTrader for compliance checks.

        Returns:
            AccountSnapshot with balance, equity, margin, and position count.
        """
        balance_info = await self._matchtrader.get_balance()
        positions = (
            broker_positions
            if broker_positions is not None
            else await self._matchtrader.get_open_positions()
        )

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_intents = self._store.get_intents_by_date(today)
        realized_pnl = sum(
            (intent.realized_pnl or 0.0) for intent in today_intents if intent.status == "closed"
        )
        unrealized_pnl = sum(p.profit for p in positions)
        daily_pnl = realized_pnl + unrealized_pnl

        return AccountSnapshot(
            balance=balance_info.balance,
            equity=balance_info.equity,
            margin=balance_info.margin,
            free_margin=balance_info.free_margin,
            # balance already includes realized results, not floating PnL
            day_start_balance=balance_info.balance - realized_pnl,
            initial_balance=self._config.account.initial_balance,
            open_positions=len(positions),
            daily_pnl=daily_pnl,
            total_pnl=balance_info.balance - self._config.account.initial_balance,
        )

    def _get_best_day_entry_block_reason(self, snapshot: AccountSnapshot) -> str | None:
        """Return rejection reason when actual daily PnL says no new entries."""
        safe_limit = self._config.compliance.best_day_limit * self._config.compliance.best_day_stop
        pause_threshold = safe_limit * 0.90
        if snapshot.daily_pnl < pause_threshold:
            return None
        return (
            "Best Day entry gate active: "
            f"actual daily PnL ${snapshot.daily_pnl:.2f} >= pause threshold ${pause_threshold:.2f} "
            f"(safe limit ${safe_limit:.2f})"
        )

    def _update_compliance_snapshot(self, intent_id: str, snapshot_json: str) -> None:
        """Persist compliance check results on the intent for audit trail."""
        self._store._conn.execute(
            "UPDATE intents SET compliance_snapshot = :snap WHERE id = :id",
            {"snap": snapshot_json, "id": intent_id},
        )
        self._store._conn.commit()

    @staticmethod
    def _serialize_compliance(
        result: ComplianceResult,
        snapshot: AccountSnapshot,
        portfolio_risk_meta: dict[str, Any] | None = None,
    ) -> str:
        """Serialize compliance result and account snapshot for audit storage."""
        details: dict[str, Any] = dict(result.details or {})
        if portfolio_risk_meta is not None:
            details["portfolio_risk"] = portfolio_risk_meta
        data: dict[str, Any] = {
            "passed": result.passed,
            "rule_name": result.rule_name,
            "reason": result.reason,
            "details": details,
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

    @staticmethod
    def _build_execution_meta(
        *,
        fill_price: float | None,
        volume: float,
        side: str,
        sl_price: float | None,
        tp_price: float | None,
        sl_pips: float,
        tp_pips: float,
        pre_trade_bid: float | None,
        pre_trade_ask: float | None,
        slippage_pips: float | None,
        execution_latency_ms: float | None,
        random_delay_seconds: float,
        compliance_passed: bool,
        order_raw_response: dict[str, Any],
        model_id: str = "",
        risk_pct: float | None = None,
        capital_allocation_meta: dict[str, Any] | None = None,
        portfolio_risk_meta: dict[str, Any] | None = None,
    ) -> str:
        """Build execution metadata JSON string for persistence."""
        data: dict[str, Any] = {
            "fill_price": fill_price,
            "volume": volume,
            "side": side,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "pre_trade_bid": pre_trade_bid,
            "pre_trade_ask": pre_trade_ask,
            "slippage_pips": slippage_pips,
            "execution_latency_ms": execution_latency_ms,
            "random_delay_seconds": random_delay_seconds,
            "compliance_passed": compliance_passed,
            "order_raw_response": order_raw_response,
        }
        if model_id:
            data["model_id"] = model_id
        if risk_pct is not None:
            data["risk_pct"] = risk_pct
        if capital_allocation_meta:
            data["capital_allocation"] = capital_allocation_meta
        if portfolio_risk_meta is not None:
            data["portfolio_risk"] = portfolio_risk_meta
        return json.dumps(data, default=str)

    async def _evaluate_portfolio_risk_decision(
        self,
        *,
        intent: TradeIntent,
        side: Literal["BUY", "SELL"],
        next_risk_pct: float,
        broker_positions: list[Any],
    ) -> PortfolioRiskDecision:
        """Evaluate portfolio risk for the next entry; fail closed on any error."""
        try:
            open_snapshots = await self._build_open_position_risk_snapshots(broker_positions)
            return self._portfolio_risk_guard.evaluate_next_entry(
                next_symbol=intent.symbol,
                next_side=side,
                next_risk_pct=next_risk_pct,
                open_positions=open_snapshots,
            )
        except Exception as e:
            logger.error(
                "ExecutionEngine: portfolio risk evaluation failed for {}: {}",
                intent.id,
                e,
            )
            return PortfolioRiskDecision(
                allowed=False,
                reason_code="portfolio_risk.invalid_input",
                projected_total_open_risk_pct=0.0,
                projected_same_direction_positions=0,
                details={
                    "error": "portfolio_risk_evaluation_failed",
                    "message": str(e),
                },
            )

    async def _build_open_position_risk_snapshots(
        self,
        broker_positions: list[Any],
    ) -> list[OpenPositionRiskSnapshot]:
        """Map current broker open positions into risk snapshots for guard evaluation."""
        active_intents = self._store.get_active_positions()
        intents_by_position_id: dict[str, TradeIntent] = {
            str(intent.position_id): intent
            for intent in active_intents
            if intent.position_id
        }

        default_risk = max(float(self._config.execution.default_risk_pct), 0.0)
        conservative_fallback_risk = max(float(self._config.execution.max_risk_pct), default_risk)
        snapshots: list[OpenPositionRiskSnapshot] = []

        for broker_position in broker_positions:
            position_id = str(getattr(broker_position, "position_id", ""))
            matched_intent = intents_by_position_id.get(position_id)
            symbol = str(
                getattr(broker_position, "symbol", "")
                or (matched_intent.symbol if matched_intent is not None else "")
            )
            raw_side = str(
                getattr(broker_position, "side", "")
                or (matched_intent.suggested_side if matched_intent is not None else "")
            )
            side = raw_side.strip().upper()
            if not symbol or side not in {"BUY", "SELL"}:
                raise ValueError(
                    "unmapped_broker_position:"
                    f"{position_id}:symbol={symbol or '<missing>'}:side={side or '<missing>'}"
                )

            open_risk_pct = self._extract_execution_risk_pct(matched_intent)
            if open_risk_pct is None:
                open_risk_pct = conservative_fallback_risk

            snapshots.append(
                OpenPositionRiskSnapshot(
                    symbol=symbol,
                    side=side,
                    open_risk_pct=open_risk_pct,
                )
            )

        return snapshots

    def _extract_execution_risk_pct(self, intent: TradeIntent | None) -> float | None:
        """Extract risk_pct from decision execution_meta for an existing opened intent."""
        if intent is None:
            return None
        try:
            row = self._store._conn.execute(
                "SELECT execution_meta FROM decisions WHERE intent_id = :intent_id",
                {"intent_id": intent.id},
            ).fetchone()
            if row is None or not row["execution_meta"]:
                return None
            payload = json.loads(row["execution_meta"])
            risk_pct = float(payload.get("risk_pct"))
            if risk_pct <= 0.0 or not math.isfinite(risk_pct):
                return None
            return risk_pct
        except (TypeError, ValueError, json.JSONDecodeError, KeyError):
            return None

    # ── SL/TP Price Helpers ───────────────────────────────────────────────

    @staticmethod
    def _extract_open_price(raw_response: dict[str, Any]) -> float | None:
        """Extract the fill/open price from the open_position raw response."""
        for key in ("openPrice", "open_price", "price", "fillPrice", "open"):
            val = raw_response.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    continue
        return None

    async def _fetch_position_open_price(self, position_id: str) -> float | None:
        """Fetch open_price for a specific position from the broker."""
        try:
            positions = await self._matchtrader.get_open_positions()
            for p in positions:
                if str(p.position_id) == str(position_id):
                    return p.open_price
        except Exception as e:
            logger.warning(
                "ExecutionEngine: failed to fetch position {} for price: {}",
                position_id,
                e,
            )
        return None

    async def _set_sl_tp_on_position(
        self,
        position_id: str,
        broker_symbol: str,
        config_symbol: str,
        side: Literal["BUY", "SELL"],
        volume: float,
        sl_pips: float,
        tp_pips: float,
        raw_response: dict[str, Any],
    ) -> tuple[float | None, float | None]:
        """Calculate absolute SL/TP prices and set them on an opened position.

        Uses the fill price from the raw_response or fetches the position's open_price.
        Falls back gracefully if price cannot be determined.

        Returns:
            Tuple of (sl_price, tp_price) - None values if setting failed.
        """
        # Step 1: Get fill/open price
        open_price = self._extract_open_price(raw_response)
        if open_price is None or open_price <= 0:
            # Fallback: query the position directly
            open_price = await self._fetch_position_open_price(position_id)

        if open_price is None or open_price <= 0:
            logger.error(
                "ExecutionEngine: cannot determine open price for position {} — SL/TP NOT SET",
                position_id,
            )
            return None, None

        # Step 2: Get pip_size from config
        instrument = self._config.instruments.get(config_symbol)
        if instrument is None:
            logger.error(
                "ExecutionEngine: no instrument config for {} — SL/TP NOT SET",
                config_symbol,
            )
            return None, None
        pip_size = instrument.pip_size

        # Step 3: Calculate absolute prices
        if side == "BUY":
            sl_price = open_price - (sl_pips * pip_size)
            tp_price = open_price + (tp_pips * pip_size)
        else:  # SELL
            sl_price = open_price + (sl_pips * pip_size)
            tp_price = open_price - (tp_pips * pip_size)

        # Step 4: Round to instrument's price precision
        precision = 5  # default for most FX
        if self._registry is not None:
            info = self._registry.get_info(config_symbol)
            if info is not None:
                precision = info.price_precision
        sl_price = round(sl_price, precision)
        tp_price = round(tp_price, precision)

        # Step 5: Modify position to set SL/TP
        try:
            result = await self._matchtrader.modify_position(
                position_id=position_id,
                symbol=broker_symbol,
                side=side,
                volume=volume,
                sl=sl_price,
                tp=tp_price,
            )
            if result.success:
                logger.info(
                    "ExecutionEngine: SL/TP set on position {} — SL={:.{}f} TP={:.{}f} "
                    "(from open_price={:.{}f}, sl_pips={}, tp_pips={})",
                    position_id,
                    sl_price,
                    precision,
                    tp_price,
                    precision,
                    open_price,
                    precision,
                    sl_pips,
                    tp_pips,
                )
                return sl_price, tp_price
            else:
                logger.error(
                    "ExecutionEngine: failed to set SL/TP on position {}: {}",
                    position_id,
                    result.message,
                )
                return None, None
        except Exception as e:
            logger.error(
                "ExecutionEngine: error setting SL/TP on position {}: {}",
                position_id,
                e,
            )
            return None, None

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

    def _log_trade_event(self, event_type: str, details: dict[str, Any]) -> None:
        """Safely append an event to TradeJournal if configured."""
        if self._trade_journal is None:
            return
        try:
            self._trade_journal.log_event(event_type, details)
        except Exception as e:
            logger.warning("TradeJournal: failed to log {}: {}", event_type, e)

    # ── Alert Helpers ───────────────────────────────────────────────────

    async def _send_alert_opened(
        self,
        symbol: str,
        side: str,
        volume: float,
        equity: float,
        position_id: str,
        sl_price: float | None = None,
        tp_price: float | None = None,
        price: float | None = None,
    ) -> None:
        """Send Telegram notification for a successfully opened trade.

        Args:
            symbol: Trading instrument name.
            side: "BUY" or "SELL".
            volume: Position volume in lots.
            equity: Current account equity.
            position_id: ID of the opened position.
            sl_price: Stop loss price (optional, for logging).
            tp_price: Take profit price (optional, for logging).
            price: Actual fill price from broker (optional, defaults to 0.0).
        """
        if self._alert_service is not None:
            try:
                await self._alert_service.trade_opened(
                    symbol=symbol,
                    side=side,
                    volume=volume,
                    price=price if price is not None else 0.0,
                    sl=sl_price,
                    tp=tp_price,
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
