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
