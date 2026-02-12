"""
Bridge to TradingAgents — invokes the multi-agent decision engine
with scanner signals and returns BUY/SELL/HOLD decisions.
"""

import importlib
import sys
import random
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

from loguru import logger


class AgentDecision:
    """Structured result from TradingAgents' propagate()."""

    def __init__(
        self,
        symbol: str,
        decision: Literal["BUY", "SELL", "HOLD"],
        final_state: Dict[str, Any],
        risk_report: str = "",
    ) -> None:
        self.symbol = symbol
        self.decision = decision
        self.final_state = final_state
        self.risk_report = risk_report

    @property
    def is_actionable(self) -> bool:
        return self.decision in ("BUY", "SELL")

    def __repr__(self) -> str:
        return f"AgentDecision({self.symbol}, {self.decision})"


class MockTradingGraph:
    """Mock for TradingAgentsGraph when import fails or dependencies missing."""

    def __init__(self, *args, **kwargs):
        pass

    def propagate(self, company_name: str, trade_date: str, qlib_data: Any = None):
        logger.warning(f"MockTradingGraph: simulating decision for {company_name}")
        # Random decision: 40% BUY, 40% SELL, 20% HOLD
        r = random.random()
        decision = "HOLD"
        if r < 0.4:
            decision = "BUY"
        elif r < 0.8:
            decision = "SELL"

        return {}, decision

    def reflect_and_remember(self, *args):
        pass


class AgentBridge:
    """Bridge to TradingAgents multi-agent decision engine.

    Imports TradingAgentsGraph from the TradingAgents project and
    calls propagate() with scanner signals.

    Usage:
        bridge = AgentBridge(
            agents_path="../../TradingAgents",
            selected_analysts=["market", "news", "social"],
            config={...},
        )
        decision = bridge.decide("EURUSD", "2026-02-12", qlib_data={...})
        if decision.is_actionable:
            # Execute trade
    """

    def __init__(
        self,
        agents_path: str | Path,
        selected_analysts: List[str] | None = None,
        config: Dict[str, Any] | None = None,
    ) -> None:
        self._agents_path = Path(agents_path).resolve()
        self._selected_analysts = selected_analysts or ["market", "news", "social"]
        self._config = config or {}
        self._graph: Any = None  # Lazy-loaded TradingAgentsGraph

    def _ensure_loaded(self) -> None:
        """Lazy-load TradingAgentsGraph on first use."""
        if self._graph is not None:
            return

        # Add TradingAgents to sys.path
        agents_str = str(self._agents_path)
        if agents_str not in sys.path:
            sys.path.insert(0, agents_str)
            logger.debug("AgentBridge: added {} to sys.path", agents_str)

        try:
            module = importlib.import_module("graph.trading_graph")  # Try relative to project root
            # Or absolute if installed as package
            # module = importlib.import_module("tradingagents.graph.trading_graph")

            # Since we added agents_path to sys.path, "graph.trading_graph" might work
            # if TradingAgents has that structure. Let's try flexible import.
            if not module:
                module = importlib.import_module("tradingagents.graph.trading_graph")

            graph_cls = getattr(module, "TradingAgentsGraph")

            self._graph = graph_cls(
                selected_analysts=self._selected_analysts,
                config=self._config,
            )
            logger.info(
                "AgentBridge: loaded TradingAgentsGraph (analysts={})",
                self._selected_analysts,
            )
        except Exception as e:
            logger.warning(
                "AgentBridge: failed to import TradingAgentsGraph ({}), falling back to Mock.", e
            )
            self._graph = MockTradingGraph()

    def decide(
        self,
        symbol: str,
        trade_date: str,
        qlib_data: Dict[str, Any] | None = None,
    ) -> AgentDecision:
        """Run multi-agent decision for a single symbol.

        Args:
            symbol: FX pair (e.g. "EURUSD").
            trade_date: Date string (e.g. "2026-02-12").
            qlib_data: Scanner signal data dict for injection.

        Returns:
            AgentDecision with BUY/SELL/HOLD and full state.
        """
        self._ensure_loaded()

        logger.info("AgentBridge: deciding on {} for {}", symbol, trade_date)

        try:
            final_state, decision = self._graph.propagate(
                company_name=symbol,
                trade_date=trade_date,
                qlib_data=qlib_data,
            )

            # Extract risk report from final state if available
            risk_report = ""
            if isinstance(final_state, dict):
                risk_report = final_state.get("risk_report", "")

            result = AgentDecision(
                symbol=symbol,
                decision=decision,
                final_state=final_state if isinstance(final_state, dict) else {},
                risk_report=risk_report,
            )

            logger.info(
                "AgentBridge: {} → {} (state keys: {})",
                symbol,
                decision,
                list(final_state.keys()) if isinstance(final_state, dict) else "N/A",
            )
            return result

        except Exception as e:
            logger.error("AgentBridge: propagate() failed for {}: {}", symbol, e)
            return AgentDecision(
                symbol=symbol,
                decision="HOLD",
                final_state={"error": str(e)},
                risk_report=f"Error during agent decision: {e}",
            )

    def decide_batch(
        self,
        signals: List[Dict[str, Any]],
        trade_date: str,
    ) -> List[AgentDecision]:
        """Run decisions for multiple symbols sequentially.

        Args:
            signals: List of signal dicts with "instrument" and qlib_data fields.
            trade_date: Date string.

        Returns:
            List of AgentDecision for each signal.
        """
        results = []
        for signal in signals:
            symbol = signal.get("instrument", signal.get("symbol", ""))
            if not symbol:
                logger.warning("AgentBridge: skipping signal with no instrument")
                continue

            qlib_data = {
                "score": signal.get("score", 0),
                "signal_strength": signal.get("signal_strength", "MODERATE"),
                "confidence": signal.get("confidence", "medium"),
                "score_gap": signal.get("score_gap", 0),
                "drop_distance": signal.get("drop_distance", 0),
                "topk_spread": signal.get("topk_spread", 0),
            }

            decision = self.decide(symbol, trade_date, qlib_data)
            results.append(decision)

        actionable = sum(1 for d in results if d.is_actionable)
        logger.info(
            "AgentBridge: batch complete — {}/{} actionable",
            actionable,
            len(results),
        )
        return results

    def reflect(self, returns_losses: Dict[str, float]) -> None:
        """Feed realized PnL back to TradingAgents for memory updates.

        Calls reflect_and_remember() which updates all agent memories
        to improve future decisions.

        Args:
            returns_losses: Dict of symbol -> realized PnL.
        """
        self._ensure_loaded()

        if not hasattr(self._graph, "reflect_and_remember"):
            logger.warning("AgentBridge: reflect_and_remember() not available")
            return

        try:
            self._graph.reflect_and_remember(returns_losses)
            logger.info("AgentBridge: reflected on {} results", len(returns_losses))
        except Exception as e:
            logger.error("AgentBridge: reflect failed: {}", e)
