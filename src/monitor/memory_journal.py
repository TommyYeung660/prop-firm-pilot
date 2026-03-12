"""
Trade memory journal — Markdown logs of all trading decisions for analysis.

Creates daily Markdown files in MEMORY:/{YYYY-MM-DD}.md containing
detailed information about each trade decision including:
- Trade plan details (symbol, side, volume, SL, TP, risk)
- Scanner signal (Qlib score, confidence, rank, score_gap)
- TradingAgents reasoning (decision, risk report, final state)
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

# ── Memory Journal ────────────────────────────────────────────────────────


class MemoryJournal:
    """Markdown trade memory journal for human-readable decision logs.

    Each day's trades are appended to a single Markdown file named
    {YYYY-MM-DD}.md in the specified directory.

    Usage:
        journal = MemoryJournal("MEMORY")
        journal.log_trade_decision(trade_plan, signal, agent_decision)
    """

    def __init__(self, memory_dir: str | Path) -> None:
        """Initialize memory journal directory.

        Args:
            memory_dir: Path to directory where Markdown files will be stored.
        """
        self._memory_dir = Path(memory_dir)
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        # v1.3.9a: Track last decision per symbol for diff computation
        self._last_decisions: dict[str, dict[str, Any]] = {}
        # v1.4.1: Track decision anchor file by intent_id for safe result append.
        self._decision_files: dict[str, Path] = {}
        logger.debug("MemoryJournal: initialized at {}", self._memory_dir)

    # ── Public Methods ─────────────────────────────────────────────────────

    def log_trade_decision(self, trade_plan: Any, signal: Any, agent_decision: Any) -> None:
        """Log a trade decision to today's Markdown file.

        Appends a formatted trade block containing:
        - Trade heading with timestamp, symbol, and side
        - Trade Details (symbol, side, volume, SL, TP, risk)
        - Scanner Signal (score, confidence, score_gap)
        - TradingAgents Reasoning (decision, risk report, final state)

        Args:
            trade_plan: TradePlan object with symbol, side, volume, stop_loss,
                        take_profit, risk_amount attributes.
            signal: Object with instrument, score, confidence, score_gap attributes.
            agent_decision: AgentDecision object with decision, risk_report,
                             final_state attributes.
        """
        # Get current UTC timestamp
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S UTC")

        # Build Markdown content
        content = self._format_trade_block(time_str, trade_plan, signal, agent_decision)

        # Append to daily file
        file_path = self._memory_dir / f"{date_str}.md"
        self._append_to_file(file_path, content)

        logger.info(
            "MemoryJournal: logged trade memory for {} ({})",
            trade_plan.symbol,
            date_str,
        )

    def log_decision(
        self,
        *,
        symbol: str,
        side: str,
        decision: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log an LLM decision (including HOLD) to today's Markdown file.

        v1.3.9a: Computes a diff against the previous decision for the same
        symbol so reviewers can quickly see what changed between cycles.
        """
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S UTC")

        ctx = context or {}
        # v1.3.9a: Compute diff vs last decision for this symbol
        current_snapshot = {"side": side, "decision": decision, **ctx}
        diff = self._compute_diff(symbol, current_snapshot)

        content = self._format_decision_block(
            time_str=time_str,
            symbol=symbol,
            side=side,
            decision=decision,
            context=ctx,
            diff=diff,
        )
        # Update last-known decision for next diff
        self._last_decisions[symbol] = current_snapshot

        file_path = self._memory_dir / f"{date_str}.md"
        self._append_to_file(file_path, content)
        intent_id = str(ctx.get("intent_id") or "").strip()
        if intent_id:
            self._decision_files[intent_id] = file_path
        logger.info("MemoryJournal: logged LLM decision for {} ({})", symbol, date_str)

    def append_trade_result(
        self,
        *,
        intent_id: str,
        position_id: str,
        symbol: str,
        pnl: float,
        reason: str,
    ) -> None:
        """Append a trade result block to the anchored decision file."""
        now = datetime.now(timezone.utc)
        time_str = now.strftime("%H:%M:%S UTC")

        file_path = self._find_file_for_intent(intent_id)
        if file_path is None:
            logger.warning(
                "MemoryJournal: no decision anchor for intent {} (symbol={}), skipping result",
                intent_id,
                symbol,
            )
            return

        if not self._file_has_matching_anchor(file_path, intent_id=intent_id, symbol=symbol):
            logger.warning(
                "MemoryJournal: symbol mismatch for intent {} (symbol={}), skipping result",
                intent_id,
                symbol,
            )
            return

        content = self._format_trade_result_block(
            time_str=time_str,
            intent_id=intent_id,
            position_id=position_id,
            symbol=symbol,
            pnl=pnl,
            reason=reason,
        )
        self._append_to_file(file_path, content)
        logger.info("MemoryJournal: appended trade result for {} ({})", symbol, file_path.name)

    # ── Private Methods ────────────────────────────────────────────────────

    def _format_trade_block(
        self, time_str: str, trade_plan: Any, signal: Any, agent_decision: Any
    ) -> str:
        """Format a trade decision as a Markdown block.

        Args:
            time_str: UTC timestamp string (HH:MM:SS UTC).
            trade_plan: TradePlan object.
            signal: Signal object.
            agent_decision: AgentDecision object.

        Returns:
            Markdown formatted string.
        """
        lines = []

        # Heading
        lines.append(f"## {time_str} - {trade_plan.symbol} {trade_plan.side}")
        lines.append("")

        # Trade Details
        lines.append("### Trade Details")
        lines.append("")
        lines.append(f"- **Symbol**: {trade_plan.symbol}")
        lines.append(f"- **Side**: {trade_plan.side}")
        lines.append(f"- **Volume**: {trade_plan.volume}")
        lines.append(f"- **Stop Loss**: {trade_plan.stop_loss}")
        lines.append(f"- **Take Profit**: {trade_plan.take_profit}")
        lines.append(f"- **Risk Amount**: ${trade_plan.risk_amount:.2f}")
        lines.append("")

        # Scanner Signal (Qlib)
        lines.append("### Scanner Signal (Qlib)")
        lines.append("")
        lines.append(f"- **Instrument**: {getattr(signal, 'instrument', 'N/A')}")
        lines.append(f"- **Score**: {getattr(signal, 'score', 'N/A')}")
        lines.append(f"- **Confidence**: {getattr(signal, 'confidence', 'N/A')}")
        score_gap = getattr(signal, "score_gap", None)
        if score_gap is not None:
            lines.append(f"- **Score Gap**: {score_gap}")
        lines.append("")

        # TradingAgents Reasoning
        lines.append("### TradingAgents Reasoning")
        lines.append("")
        lines.append(f"**Decision**: {agent_decision.decision}")
        lines.append("")

        if agent_decision.risk_report:
            lines.append("**Risk Report**:")
            lines.append("```")
            lines.append(agent_decision.risk_report)
            lines.append("```")
            lines.append("")

        if agent_decision.final_state:
            lines.append("**Final State**:")
            lines.append("```")
            lines.append(str(agent_decision.final_state))
            lines.append("```")
            lines.append("")

        # Separator
        lines.append("---")
        lines.append("")

        return "\n".join(lines)

    def _format_decision_block(
        self,
        *,
        time_str: str,
        symbol: str,
        side: str,
        decision: str,
        context: dict[str, Any],
        diff: dict[str, tuple[Any, Any]] | None = None,
    ) -> str:
        """Format a generic decision block for scheduler LLM outputs.

        Args:
            diff: Optional dict of {key: (old_value, new_value)} showing what
                  changed vs the previous decision for this symbol.
        """
        lines = []
        lines.append(f"## {time_str} - {symbol} {decision}")
        lines.append("")
        lines.append("### Decision Context")
        lines.append("")
        lines.append(f"- **Symbol**: {symbol}")
        lines.append(f"- **Side**: {side}")
        lines.append(f"- **Decision**: {decision}")
        for key, value in context.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")

        # v1.3.9a: Diff vs previous decision for same symbol
        if diff:
            lines.append("### Δ Changes vs Previous Decision")
            lines.append("")
            for key, (old_val, new_val) in diff.items():
                lines.append(f"- **{key}**: `{old_val}` → `{new_val}`")
            lines.append("")
        elif diff is not None:
            # diff was computed but empty — identical to previous
            lines.append("> ℹ️ No changes vs previous decision for this symbol.")
            lines.append("")

        lines.append("---")
        lines.append("")
        return "\n".join(lines)

    def _format_trade_result_block(
        self,
        *,
        time_str: str,
        intent_id: str,
        position_id: str,
        symbol: str,
        pnl: float,
        reason: str,
    ) -> str:
        """Format a trade result append block."""
        lines = []
        lines.append("### Trade Result")
        lines.append("")
        lines.append(f"- **Time**: {time_str}")
        lines.append(f"- **Intent ID**: {intent_id}")
        lines.append(f"- **Position ID**: {position_id}")
        lines.append(f"- **Symbol**: {symbol}")
        lines.append(f"- **PnL**: {pnl}")
        lines.append(f"- **Reason**: {reason}")
        lines.append("")
        lines.append("---")
        lines.append("")
        return "\n".join(lines)

    def _find_file_for_intent(self, intent_id: str) -> Path | None:
        """Resolve the Markdown file that contains the decision anchor for intent_id."""
        cached = self._decision_files.get(intent_id)
        if cached is not None and cached.exists():
            return cached

        for file_path in sorted(self._memory_dir.glob("*.md")):
            if self._file_has_intent_anchor(file_path, intent_id):
                self._decision_files[intent_id] = file_path
                return file_path
        return None

    def _file_has_matching_anchor(self, file_path: Path, *, intent_id: str, symbol: str) -> bool:
        """Check that a file contains a decision block for the given intent and symbol."""
        for block in self._read_blocks(file_path):
            if self._block_has_intent_anchor(block, intent_id) and self._block_has_symbol(
                block, symbol
            ):
                return True
        return False

    def _file_has_intent_anchor(self, file_path: Path, intent_id: str) -> bool:
        """Check whether any block in file_path references the given intent_id."""
        return any(
            self._block_has_intent_anchor(block, intent_id)
            for block in self._read_blocks(file_path)
        )

    def _read_blocks(self, file_path: Path) -> list[str]:
        """Read a Markdown file and split it into journal blocks."""
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.error("MemoryJournal: failed to read {}: {}", file_path, e)
            return []
        return [block.strip() for block in text.split("\n---\n") if block.strip()]

    def _block_has_intent_anchor(self, block: str, intent_id: str) -> bool:
        """Check whether a block contains the exact intent_id anchor."""
        return f"- **intent_id**: {intent_id}" in block

    def _block_has_symbol(self, block: str, symbol: str) -> bool:
        """Check whether a block contains the exact symbol line."""
        return f"- **Symbol**: {symbol}" in block

    def _compute_diff(
        self, symbol: str, current: dict[str, Any]
    ) -> dict[str, tuple[Any, Any]] | None:
        """Compare current decision snapshot to the last one for this symbol.

        Returns:
            None if no prior decision exists (first time).
            Empty dict if nothing changed.
            Dict of {key: (old_value, new_value)} for changed fields.
        """
        prev = self._last_decisions.get(symbol)
        if prev is None:
            return None

        # Keys that are verbose / non-comparable (e.g. risk_report, final_state)
        skip_keys = {"risk_report", "final_state"}
        all_keys = sorted((set(prev) | set(current)) - skip_keys)

        diff: dict[str, tuple[Any, Any]] = {}
        for key in all_keys:
            old_val = prev.get(key)
            new_val = current.get(key)
            if old_val != new_val:
                diff[key] = (old_val, new_val)
        return diff

    def _append_to_file(self, file_path: Path, content: str) -> None:
        """Append content to a file.

        Args:
            file_path: Path to the file.
            content: Content to append.
        """
        try:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(content)
        except OSError as e:
            logger.error("MemoryJournal: failed to write to {}: {}", file_path, e)
            raise
