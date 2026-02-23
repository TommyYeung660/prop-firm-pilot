"""
Automated trial readiness assessment for E8 Markets.

Evaluates 8 criteria to determine if the trading system is ready
to transition from trial to live account. Reads closed trade data
from the SQLite decision store.

Usage:
    python -m scripts.assess_trial_readiness --db data/decisions.db
    python -m scripts.assess_trial_readiness --db data/decisions.db --json
"""

import argparse
from datetime import datetime, timedelta, timezone

from loguru import logger
from pydantic import BaseModel

from src.decision_store.sqlite_store import DecisionStore

# ── Pydantic Models ────────────────────────────────────────────────────────


class CriterionResult(BaseModel):
    """Result of a single readiness criterion check."""

    name: str
    passed: bool
    actual: float | str
    threshold: float | str
    detail: str = ""


class ReadinessReport(BaseModel):
    """Complete readiness assessment report."""

    assessed_at: datetime
    db_path: str
    criteria: list[CriterionResult]
    overall_ready: bool
    summary: str


# ── TrialReadinessAssessor ──────────────────────────────────────────────────


class TrialReadinessAssessor:
    """Assesses trial account readiness against 8 criteria.

    Uses raw SQL queries against the decision store to compute metrics.
    Handles NULL realized_pnl values gracefully by reporting insufficient data.

    Usage:
        assessor = TrialReadinessAssessor("data/decisions.db")
        report = assessor.assess(lookback_days=14)
        print(assessor.format_report(report))
    """

    def __init__(self, db_path: str) -> None:
        """Initialize the assessor with a decision store.

        Args:
            db_path: Path to the SQLite decision store database.
        """
        self._store = DecisionStore(db_path)
        self._daily_api_limit = 2000  # MatchTrader daily API call limit

    def assess(self, lookback_days: int = 14) -> ReadinessReport:
        """Run all 8 readiness criteria checks.

        Args:
            lookback_days: Number of days to look back for assessment.

        Returns:
            ReadinessReport with results for all 8 criteria.
        """
        logger.info("Starting trial readiness assessment ({}d lookback)", lookback_days)
        assessed_at = datetime.now(timezone.utc)
        criteria: list[CriterionResult] = []

        # ── Criterion 1: Stable Days ──────────────────────────────
        criteria.append(self._check_stable_days(lookback_days))

        # ── Criterion 2: Profitable Days ─────────────────────────
        criteria.append(self._check_profitable_days(lookback_days))

        # ── Criterion 3: Win Rate ────────────────────────────────
        criteria.append(self._check_win_rate(lookback_days))

        # ── Criterion 4: Max Day Loss ──────────────────────────────
        criteria.append(self._check_max_day_loss(lookback_days))

        # ── Criterion 5: Avg Daily PnL ─────────────────────────────
        criteria.append(self._check_avg_daily_pnl(lookback_days))

        # ── Criterion 6: Compliance Warnings ───────────────────────
        criteria.append(self._check_compliance_warnings(lookback_days))

        # ── Criterion 7: API Usage ────────────────────────────────
        criteria.append(self._check_api_usage(lookback_days))

        # ── Criterion 8: Minimum Trades ───────────────────────────
        criteria.append(self._check_min_trades(lookback_days))

        overall_ready = all(c.passed for c in criteria)
        summary = self._generate_summary(criteria, overall_ready)

        logger.info("Assessment complete: overall_ready={}", overall_ready)

        return ReadinessReport(
            assessed_at=assessed_at,
            db_path=self._store._db_path,
            criteria=criteria,
            overall_ready=overall_ready,
            summary=summary,
        )

    # ── Criterion Check Methods ──────────────────────────────────────────────

    def _check_stable_days(self, lookback_days: int) -> CriterionResult:
        """Check 1: System ran ≥14 consecutive days without crash."""
        threshold = 14
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        cutoff_str = cutoff.isoformat()

        # Count unique trade dates in the lookback period
        row = self._store._conn.execute(
            """SELECT COUNT(DISTINCT trade_date) as days
               FROM intents
               WHERE created_at >= :cutoff""",
            {"cutoff": cutoff_str},
        ).fetchone()

        actual = row["days"] if row else 0
        passed = actual >= threshold
        detail = f"System active on {actual} of {lookback_days} days"

        return CriterionResult(
            name="stable_days",
            passed=passed,
            actual=float(actual),
            threshold=float(threshold),
            detail=detail,
        )

    def _check_profitable_days(self, lookback_days: int) -> CriterionResult:
        """Check 2: ≥10 of last 14 days were profitable."""
        threshold = 10
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        cutoff_str = cutoff.isoformat()

        # Get daily PnL from intents table (using trade_date grouping)
        rows = self._store._conn.execute(
            """SELECT trade_date, SUM(realized_pnl) as daily_pnl
               FROM intents
               WHERE status = 'closed'
                 AND created_at >= :cutoff
                 AND realized_pnl IS NOT NULL
               GROUP BY trade_date""",
            {"cutoff": cutoff_str},
        ).fetchall()

        profitable_days = sum(1 for r in rows if r["daily_pnl"] and r["daily_pnl"] > 0)
        passed = profitable_days >= threshold
        detail = f"{profitable_days} profitable days out of {len(rows)} days with trades"

        return CriterionResult(
            name="profitable_days",
            passed=passed,
            actual=float(profitable_days),
            threshold=float(threshold),
            detail=detail,
        )

    def _check_win_rate(self, lookback_days: int) -> CriterionResult:
        """Check 3: Win rate ≥50% across all closed trades."""
        threshold = 0.50
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        cutoff_str = cutoff.isoformat()

        # Count winning vs losing trades
        row = self._store._conn.execute(
            """SELECT
                  COUNT(*) as total,
                  SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins
               FROM intents
               WHERE status = 'closed'
                 AND created_at >= :cutoff
                 AND realized_pnl IS NOT NULL""",
            {"cutoff": cutoff_str},
        ).fetchone()

        total = row["total"] if row and row["total"] else 0
        wins = row["wins"] if row and row["wins"] else 0

        if total == 0:
            return CriterionResult(
                name="win_rate",
                passed=False,
                actual="insufficient data",
                threshold=threshold,
                detail="No closed trades with PnL data in lookback period",
            )

        win_rate = wins / total
        passed = win_rate >= threshold
        detail = f"{wins} wins / {total} trades = {win_rate:.1%}"

        return CriterionResult(
            name="win_rate",
            passed=passed,
            actual=round(win_rate, 4),
            threshold=threshold,
            detail=detail,
        )

    def _check_max_day_loss(self, lookback_days: int) -> CriterionResult:
        """Check 4: No single day lost more than $50."""
        threshold = 50.0
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        cutoff_str = cutoff.isoformat()

        # Find the worst daily loss
        row = self._store._conn.execute(
            """SELECT trade_date, SUM(realized_pnl) as daily_pnl
               FROM intents
               WHERE status = 'closed'
                 AND created_at >= :cutoff
                 AND realized_pnl IS NOT NULL
               GROUP BY trade_date
               ORDER BY daily_pnl ASC
               LIMIT 1""",
            {"cutoff": cutoff_str},
        ).fetchone()

        if not row or row["daily_pnl"] is None:
            return CriterionResult(
                name="max_day_loss",
                passed=False,
                actual="insufficient data",
                threshold=threshold,
                detail="No closed trades with PnL data to assess",
            )

        worst_loss = row["daily_pnl"]
        passed = worst_loss >= -threshold  # Loss is negative, so >= -50
        detail = f"Worst daily loss: ${worst_loss:.2f} on {row['trade_date']}"

        return CriterionResult(
            name="max_day_loss",
            passed=passed,
            actual=worst_loss,
            threshold=-threshold,
            detail=detail,
        )

    def _check_avg_daily_pnl(self, lookback_days: int) -> CriterionResult:
        """Check 5: Average daily PnL < $120 (avoid Best Day Rule risk)."""
        threshold = 120.0
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        cutoff_str = cutoff.isoformat()

        # Get average daily PnL (only positive days matter for Best Day Rule)
        row = self._store._conn.execute(
            """SELECT AVG(daily_pnl) as avg_pnl
               FROM (
                   SELECT trade_date, SUM(realized_pnl) as daily_pnl
                   FROM intents
                   WHERE status = 'closed'
                     AND created_at >= :cutoff
                     AND realized_pnl IS NOT NULL
                   GROUP BY trade_date
               )""",
            {"cutoff": cutoff_str},
        ).fetchone()

        if not row or row["avg_pnl"] is None:
            return CriterionResult(
                name="avg_daily_pnl",
                passed=False,
                actual="insufficient data",
                threshold=threshold,
                detail="No closed trades with PnL data to assess",
            )

        avg_pnl = row["avg_pnl"]
        passed = abs(avg_pnl) < threshold
        detail = f"Average daily PnL: ${avg_pnl:.2f}"

        return CriterionResult(
            name="avg_daily_pnl",
            passed=passed,
            actual=round(avg_pnl, 2),
            threshold=threshold,
            detail=detail,
        )

    def _check_compliance_warnings(self, lookback_days: int) -> CriterionResult:
        """Check 6: Zero compliance warnings in last 7 days."""
        threshold = 0
        check_days = min(7, lookback_days)
        cutoff = datetime.now(timezone.utc) - timedelta(days=check_days)
        cutoff_str = cutoff.isoformat()

        # Check compliance_snapshot for warning indicators
        row = self._store._conn.execute(
            """SELECT COUNT(*) as warnings
               FROM intents
               WHERE created_at >= :cutoff
                 AND compliance_snapshot IS NOT NULL
                 AND (
                     compliance_snapshot LIKE '%warning%'
                     OR compliance_snapshot LIKE '%breach%'
                     OR compliance_snapshot LIKE '%failed%'
                 )""",
            {"cutoff": cutoff_str},
        ).fetchone()

        warnings = row["warnings"] if row else 0
        passed = warnings == threshold
        detail = f"{warnings} compliance warning(s) in last {check_days} days"

        return CriterionResult(
            name="compliance_warnings",
            passed=passed,
            actual=float(warnings),
            threshold=float(threshold),
            detail=detail,
        )

    def _check_api_usage(self, lookback_days: int) -> CriterionResult:
        """Check 7: Average daily API calls < 80% of limit (2000)."""
        threshold_ratio = 0.80
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        cutoff_str = cutoff.isoformat()

        # Get average daily API calls
        row = self._store._conn.execute(
            """SELECT AVG(call_count) as avg_calls
               FROM api_calls
               WHERE call_date >= date(:cutoff)""",
            {"cutoff": cutoff_str},
        ).fetchone()

        if not row or row["avg_calls"] is None:
            return CriterionResult(
                name="api_usage",
                passed=False,
                actual="insufficient data",
                threshold=threshold_ratio * self._daily_api_limit,
                detail="No API call data in lookback period",
            )

        avg_calls = row["avg_calls"]
        ratio = avg_calls / self._daily_api_limit
        passed = ratio < threshold_ratio
        detail = (f"Average: {avg_calls:.0f} calls/day "
                  f"({ratio:.1%} of {self._daily_api_limit} limit)")

        return CriterionResult(
            name="api_usage",
            passed=passed,
            actual=round(ratio, 4),
            threshold=threshold_ratio,
            detail=detail,
        )

    def _check_min_trades(self, lookback_days: int) -> CriterionResult:
        """Check 8: At least 20 closed trades in assessment period."""
        threshold = 20
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        cutoff_str = cutoff.isoformat()

        row = self._store._conn.execute(
            """SELECT COUNT(*) as trades
               FROM intents
               WHERE status = 'closed'
                 AND created_at >= :cutoff""",
            {"cutoff": cutoff_str},
        ).fetchone()

        trades = row["trades"] if row else 0
        passed = trades >= threshold
        detail = f"{trades} closed trades in {lookback_days} days"

        # Check if most trades are missing PnL data
        if trades >= 1:
            row_pnl = self._store._conn.execute(
                """SELECT COUNT(*) as trades_with_pnl
                   FROM intents
                   WHERE status = 'closed'
                     AND created_at >= :cutoff
                     AND realized_pnl IS NULL""",
                {"cutoff": cutoff_str},
            ).fetchone()
            missing_pnl = row_pnl["trades_with_pnl"] if row_pnl else 0
            if missing_pnl > trades * 0.5:
                return CriterionResult(
                    name="min_trades",
                    passed=False,
                    actual=f"insufficient data ({missing_pnl}/{trades} trades missing PnL)",
                    threshold=threshold,
                    detail="Run system longer with PnL tracking enabled",
                )

        return CriterionResult(
            name="min_trades",
            passed=passed,
            actual=float(trades),
            threshold=float(threshold),
            detail=detail,
        )

    # ── Report Formatting ─────────────────────────────────────────────────────

    def _generate_summary(
        self, criteria: list[CriterionResult], overall_ready: bool
    ) -> str:
        """Generate a human-readable summary."""
        passed_count = sum(1 for c in criteria if c.passed)
        total_count = len(criteria)

        if overall_ready:
            return (
                f"✓ READY: All {total_count} criteria passed. "
                f"System cleared for live account transition."
            )
        else:
            failed = [c.name for c in criteria if not c.passed]
            return (
                f"✗ NOT READY: {passed_count}/{total_count} criteria passed. "
                f"Failed: {', '.join(failed)}. "
                f"Address issues before transitioning."
            )

    def format_report(self, report: ReadinessReport) -> str:
        """Format report as human-readable terminal output."""
        lines = [
            "",
            "═" * 80,
            "  E8 Markets Trial Readiness Assessment",
            "═" * 80,
            f"  Assessed: {report.assessed_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"  Database: {report.db_path}",
            "",
            "  Criteria Results:",
            "─" * 80,
        ]

        for crit in report.criteria:
            status = "✓ PASS" if crit.passed else "✗ FAIL"
            lines.append(f"  [{status}] {crit.name}")

            if isinstance(crit.actual, str):
                actual_str = crit.actual
            else:
                actual_str = f"{crit.actual}"
            if isinstance(crit.threshold, str):
                threshold_str = crit.threshold
            else:
                threshold_str = f"{crit.threshold}"

            lines.append(f"    Actual: {actual_str}  |  Threshold: {threshold_str}")
            if crit.detail:
                lines.append(f"    Detail: {crit.detail}")
            lines.append("")

        lines.append("─" * 80)
        lines.append(f"  {report.summary}")
        lines.append("═" * 80)
        lines.append("")

        return "\n".join(lines)

    def format_telegram(self, report: ReadinessReport) -> str:
        """Format report as HTML message for Telegram alerts."""
        emoji = "✅" if report.overall_ready else "❌"
        status = "READY" if report.overall_ready else "NOT READY"

        lines = [
            f"<b>{emoji} E8 Trial Readiness: {status}</b>\n",
            f"<i>Assessed: {report.assessed_at.strftime('%Y-%m-%d %H:%M UTC')}</i>\n",
            "",
            "<b>Criteria:</b>",
        ]

        for crit in report.criteria:
            check = "✅" if crit.passed else "❌"
            lines.append(f"{check} <b>{crit.name}</b>")

            if isinstance(crit.actual, str):
                actual_str = crit.actual
            else:
                actual_str = f"{crit.actual}"
            if isinstance(crit.threshold, str):
                threshold_str = crit.threshold
            else:
                threshold_str = f"{crit.threshold}"

            lines.append(
                f"   Actual: <code>{actual_str}</code> "
                f"| Threshold: <code>{threshold_str}</code>"
            )
            if crit.detail:
                lines.append(f"   <i>{crit.detail}</i>")
            lines.append("")

        lines.append("<b>Summary:</b>")
        lines.append(report.summary)

        return "\n".join(lines)


# ── CLI Entry Point ───────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for trial readiness assessment."""
    parser = argparse.ArgumentParser(
        description="Assess E8 Markets trial account readiness"
    )
    parser.add_argument(
        "--db",
        default="data/decisions.db",
        help="Path to decisions database (default: data/decisions.db)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report as JSON instead of human-readable text",
    )
    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Output report as HTML for Telegram alerts",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=14,
        help="Lookback period in days (default: 14)",
    )
    args = parser.parse_args()

    assessor = TrialReadinessAssessor(args.db)
    report = assessor.assess(lookback_days=args.lookback)

    if args.json:
        print(report.model_dump_json(indent=2))
    elif args.telegram:
        print(assessor.format_telegram(report))
    else:
        print(assessor.format_report(report))


if __name__ == "__main__":
    main()
