"""
SQLite-backed store for trade intents and decision audit records.

Provides persistence for the Hybrid EA+LLM pipeline's decision state machine.
Uses WAL journal mode for concurrent read access during writes. All methods
are synchronous — callers use asyncio.to_thread() from async code.

Usage:
    store = DecisionStore("data/decisions.db")
    store.insert_intent(intent)
    claimed = store.claim_next_pending("llm-0")
"""

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.decision.schemas import VALID_TRANSITIONS, DecisionRecord, TradeIntent

# ── SQL Statements ──────────────────────────────────────────────────────────

CREATE_INTENTS_TABLE = """
CREATE TABLE IF NOT EXISTS intents (
    id                  TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,
    trade_date          TEXT NOT NULL,
    symbol              TEXT NOT NULL,

    -- Scanner data
    scanner_score       REAL DEFAULT 0,
    scanner_confidence  TEXT DEFAULT 'medium',
    scanner_score_gap   REAL DEFAULT 0,
    scanner_drop_distance REAL DEFAULT 0,
    scanner_topk_spread REAL DEFAULT 0,

    -- LLM decision
    suggested_side      TEXT,
    suggested_sl_pips   REAL,
    suggested_tp_pips   REAL,
    agent_risk_report   TEXT DEFAULT '',
    agent_state_json    TEXT DEFAULT '',

    -- Lifecycle
    source              TEXT DEFAULT 'scanner',
    status              TEXT DEFAULT 'pending',
    claim_worker_id     TEXT,
    claim_ts            TEXT,
    claim_ttl_minutes   INTEGER DEFAULT 30,
    expires_at          TEXT,
    idempotency_key     TEXT UNIQUE,

    -- Execution
    position_id         TEXT,
    executed_at         TEXT,
    execution_error     TEXT,
    compliance_snapshot TEXT DEFAULT '',

    -- Close result
    realized_pnl        REAL,
    exit_price          REAL,
    exit_reason         TEXT,
    hold_duration_seconds INTEGER
)
"""

CREATE_DECISIONS_TABLE = """
CREATE TABLE IF NOT EXISTS decisions (
    intent_id           TEXT PRIMARY KEY REFERENCES intents(id),
    created_at          TEXT NOT NULL,
    claimed_at          TEXT,
    decided_at          TEXT,
    executed_at         TEXT,
    closed_at           TEXT,

    status              TEXT DEFAULT '',
    order_id            TEXT,
    position_id         TEXT,
    failure_reason      TEXT DEFAULT '',
    execution_meta      TEXT DEFAULT '',
    realized_pnl        REAL,
    exit_reason         TEXT
)
"""

CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_intents_status ON intents(status);
CREATE INDEX IF NOT EXISTS idx_intents_trade_date ON intents(trade_date);
CREATE INDEX IF NOT EXISTS idx_intents_symbol_date ON intents(symbol, trade_date);
"""

CREATE_API_CALLS_TABLE = """
CREATE TABLE IF NOT EXISTS api_calls (
    call_date   TEXT NOT NULL,
    call_count  INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (call_date)
)
"""

CREATE_EQUITY_SNAPSHOTS_TABLE = """
CREATE TABLE IF NOT EXISTS equity_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    equity          REAL NOT NULL,
    balance         REAL,
    daily_dd_pct    REAL NOT NULL DEFAULT 0,
    max_dd_pct      REAL NOT NULL DEFAULT 0,
    open_positions  INTEGER DEFAULT 0
)
"""

CREATE_EQUITY_SNAPSHOTS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_snapshots(timestamp);
"""

INSERT_INTENT_SQL = """
INSERT INTO intents (
    id, created_at, trade_date, symbol,
    scanner_score, scanner_confidence, scanner_score_gap,
    scanner_drop_distance, scanner_topk_spread,
    suggested_side, suggested_sl_pips, suggested_tp_pips,
    agent_risk_report, agent_state_json,
    source, status, claim_worker_id, claim_ts,
    claim_ttl_minutes, expires_at, idempotency_key,
    position_id, executed_at, execution_error, compliance_snapshot,
    realized_pnl, exit_price, exit_reason, hold_duration_seconds
) VALUES (
    :id, :created_at, :trade_date, :symbol,
    :scanner_score, :scanner_confidence, :scanner_score_gap,
    :scanner_drop_distance, :scanner_topk_spread,
    :suggested_side, :suggested_sl_pips, :suggested_tp_pips,
    :agent_risk_report, :agent_state_json,
    :source, :status, :claim_worker_id, :claim_ts,
    :claim_ttl_minutes, :expires_at, :idempotency_key,
    :position_id, :executed_at, :execution_error, :compliance_snapshot,
    :realized_pnl, :exit_price, :exit_reason, :hold_duration_seconds
)
"""

INSERT_DECISION_SQL = """
INSERT INTO decisions (
    intent_id, created_at, status
) VALUES (:intent_id, :created_at, :status)
"""


# ── Helper Functions ────────────────────────────────────────────────────────


def _dt_to_str(dt: datetime | None) -> str | None:
    """Convert datetime to ISO 8601 string, or None."""
    if dt is None:
        return None
    return dt.isoformat()


def _str_to_dt(s: str | None) -> datetime | None:
    """Parse ISO 8601 string to datetime, or None."""
    if not s:
        return None
    return datetime.fromisoformat(s)


def _row_to_intent(row: sqlite3.Row) -> TradeIntent:
    """Convert a database row to a TradeIntent model."""
    data = dict(row)
    # Convert ISO strings back to datetime
    for dt_field in ("created_at", "claim_ts", "expires_at", "executed_at"):
        if data.get(dt_field):
            data[dt_field] = datetime.fromisoformat(data[dt_field])
        else:
            data[dt_field] = None
    return TradeIntent(**data)


def _row_to_decision(row: sqlite3.Row) -> DecisionRecord:
    """Convert a database row to a DecisionRecord model."""
    data = dict(row)
    for dt_field in ("created_at", "claimed_at", "decided_at", "executed_at", "closed_at"):
        if data.get(dt_field):
            data[dt_field] = datetime.fromisoformat(data[dt_field])
        else:
            data[dt_field] = None
    return DecisionRecord(**data)


# ── DecisionStore ───────────────────────────────────────────────────────────


class DecisionStoreError(Exception):
    """Base exception for DecisionStore operations."""


class InvalidTransitionError(DecisionStoreError):
    """Raised when an intent status transition violates the state machine."""


class DecisionStore:
    """SQLite-backed store for trade intents and decisions.

    Thread-safe for single-writer, multiple-reader pattern (WAL mode).
    All methods are synchronous — called from async code via asyncio.to_thread().

    Usage:
        store = DecisionStore("data/decisions.db")
        store.insert_intent(intent)
        claimed = store.claim_next_pending("llm-0")
    """

    def __init__(self, db_path: str = "data/decisions.db") -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = self._create_connection()
        self._ensure_tables()

    def _create_connection(self) -> sqlite3.Connection:
        """Create a SQLite connection with WAL mode and row factory."""
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_tables(self) -> None:
        """Create tables and indexes if they don't exist."""
        self._conn.executescript(CREATE_INTENTS_TABLE)
        self._conn.executescript(CREATE_DECISIONS_TABLE)
        self._conn.executescript(CREATE_INDEXES)
        self._conn.executescript(CREATE_API_CALLS_TABLE)
        self._conn.executescript(CREATE_EQUITY_SNAPSHOTS_TABLE)
        self._conn.executescript(CREATE_EQUITY_SNAPSHOTS_INDEX)
        self._conn.commit()
        self._ensure_columns()
        logger.debug("Decision store tables ensured at {}", self._db_path)


    def _ensure_columns(self) -> None:
        """Idempotent migration: add columns that may not exist in older DBs."""
        migrations = [
            ("intents", "realized_pnl", "REAL"),
            ("intents", "exit_price", "REAL"),
            ("intents", "exit_reason", "TEXT"),
            ("intents", "hold_duration_seconds", "INTEGER"),
            ("decisions", "realized_pnl", "REAL"),
            ("decisions", "exit_reason", "TEXT"),
            ("api_calls", "read_count", "INTEGER DEFAULT 0"),
            ("api_calls", "write_count", "INTEGER DEFAULT 0"),
        ]
        for table, col, col_type in migrations:
            try:
                self._conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists
        self._conn.commit()
    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
        logger.debug("Decision store connection closed")

    # ── Intent Lifecycle ────────────────────────────────────────────

    def insert_intent(self, intent: TradeIntent) -> None:
        """Insert a new trade intent and its companion decision record.

        Args:
            intent: TradeIntent with status='pending'.

        Raises:
            DecisionStoreError: If the idempotency_key already exists.
        """
        params = {
            "id": intent.id,
            "created_at": _dt_to_str(intent.created_at),
            "trade_date": intent.trade_date,
            "symbol": intent.symbol,
            "scanner_score": intent.scanner_score,
            "scanner_confidence": intent.scanner_confidence,
            "scanner_score_gap": intent.scanner_score_gap,
            "scanner_drop_distance": intent.scanner_drop_distance,
            "scanner_topk_spread": intent.scanner_topk_spread,
            "suggested_side": intent.suggested_side,
            "suggested_sl_pips": intent.suggested_sl_pips,
            "suggested_tp_pips": intent.suggested_tp_pips,
            "agent_risk_report": intent.agent_risk_report,
            "agent_state_json": intent.agent_state_json,
            "source": intent.source,
            "status": intent.status,
            "claim_worker_id": intent.claim_worker_id,
            "claim_ts": _dt_to_str(intent.claim_ts),
            "claim_ttl_minutes": intent.claim_ttl_minutes,
            "expires_at": _dt_to_str(intent.expires_at),
            "idempotency_key": intent.idempotency_key,
            "position_id": intent.position_id,
            "executed_at": _dt_to_str(intent.executed_at),
            "execution_error": intent.execution_error,
            "compliance_snapshot": intent.compliance_snapshot,
            "realized_pnl": intent.realized_pnl,
            "exit_price": intent.exit_price,
            "exit_reason": intent.exit_reason,
            "hold_duration_seconds": intent.hold_duration_seconds,
        }
        try:
            self._conn.execute(INSERT_INTENT_SQL, params)
            # Create companion decision record
            self._conn.execute(
                INSERT_DECISION_SQL,
                {
                    "intent_id": intent.id,
                    "created_at": _dt_to_str(intent.created_at),
                    "status": intent.status,
                },
            )
            self._conn.commit()
            logger.info(
                "Inserted intent {} for {} on {}", intent.id, intent.symbol, intent.trade_date
            )
        except sqlite3.IntegrityError as e:
            self._conn.rollback()
            raise DecisionStoreError(f"Duplicate intent: {e}") from e

    def claim_next_pending(self, worker_id: str) -> TradeIntent | None:
        """Atomically claim the oldest pending intent for a worker.

        Sets status=claimed, records claim_worker_id, claim_ts, and expires_at.

        Args:
            worker_id: Identifier for the LLM worker (e.g., "llm-0").

        Returns:
            The claimed TradeIntent, or None if no pending intents exist.
        """
        now = datetime.now(timezone.utc)
        row = self._conn.execute(
            "SELECT * FROM intents WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        if row is None:
            return None

        intent = _row_to_intent(row)
        ttl = intent.claim_ttl_minutes
        expires_at = now + timedelta(minutes=ttl)

        self._conn.execute(
            """UPDATE intents
               SET status = 'claimed',
                   claim_worker_id = :worker_id,
                   claim_ts = :claim_ts,
                   expires_at = :expires_at
               WHERE id = :id AND status = 'pending'""",
            {
                "worker_id": worker_id,
                "claim_ts": _dt_to_str(now),
                "expires_at": _dt_to_str(expires_at),
                "id": intent.id,
            },
        )
        self._conn.execute(
            """UPDATE decisions
               SET status = 'claimed', claimed_at = :claimed_at
               WHERE intent_id = :intent_id""",
            {"claimed_at": _dt_to_str(now), "intent_id": intent.id},
        )
        self._conn.commit()

        # Return the updated intent
        intent.status = "claimed"
        intent.claim_worker_id = worker_id
        intent.claim_ts = now
        intent.expires_at = expires_at
        logger.info("Worker {} claimed intent {} ({})", worker_id, intent.id, intent.symbol)
        return intent

    def update_intent_decision(
        self,
        intent_id: str,
        side: str,
        sl_pips: float,
        tp_pips: float,
        risk_report: str,
        state_json: str,
    ) -> None:
        """Update a claimed intent with the LLM decision.

        Args:
            intent_id: Intent to update.
            side: "BUY", "SELL", or "HOLD".
            sl_pips: Suggested stop-loss in pips.
            tp_pips: Suggested take-profit in pips.
            risk_report: Risk analysis text from TradingAgents.
            state_json: JSON-serialized final_state from TradingAgents.
        """
        now = datetime.now(timezone.utc)
        self._conn.execute(
            """UPDATE intents
               SET suggested_side = :side,
                   suggested_sl_pips = :sl_pips,
                   suggested_tp_pips = :tp_pips,
                   agent_risk_report = :risk_report,
                   agent_state_json = :state_json
               WHERE id = :id AND status = 'claimed'""",
            {
                "side": side,
                "sl_pips": sl_pips,
                "tp_pips": tp_pips,
                "risk_report": risk_report,
                "state_json": state_json,
                "id": intent_id,
            },
        )
        self._conn.execute(
            """UPDATE decisions SET decided_at = :decided_at WHERE intent_id = :intent_id""",
            {"decided_at": _dt_to_str(now), "intent_id": intent_id},
        )
        self._conn.commit()
        logger.info("Updated intent {} with decision: {}", intent_id, side)

    def mark_ready_for_exec(self, intent_id: str) -> None:
        """Transition intent from claimed → ready_for_exec."""
        self._transition(intent_id, "claimed", "ready_for_exec")

    def mark_executing(self, intent_id: str) -> None:
        """Transition intent from ready_for_exec → executing."""
        self._transition(intent_id, "ready_for_exec", "executing")

    def mark_opened(self, intent_id: str, position_id: str) -> None:
        """Transition intent from executing → opened with position details."""
        now = datetime.now(timezone.utc)
        updated = self._conn.execute(
            """UPDATE intents
               SET status = 'opened',
                   position_id = :position_id,
                   executed_at = :executed_at
               WHERE id = :id AND status = 'executing'""",
            {
                "position_id": position_id,
                "executed_at": _dt_to_str(now),
                "id": intent_id,
            },
        ).rowcount
        if not updated:
            raise InvalidTransitionError(
                f"Cannot mark {intent_id} as opened: not in 'executing' state"
            )
        self._conn.execute(
            """UPDATE decisions
               SET status = 'opened',
                   position_id = :position_id,
                   executed_at = :executed_at
               WHERE intent_id = :intent_id""",
            {
                "position_id": position_id,
                "executed_at": _dt_to_str(now),
                "intent_id": intent_id,
            },
        )
        self._conn.commit()
        logger.info("Intent {} opened with position {}", intent_id, position_id)

    def mark_rejected(self, intent_id: str, reason: str) -> None:
        """Transition intent from executing → rejected with compliance reason."""
        now = datetime.now(timezone.utc)
        updated = self._conn.execute(
            """UPDATE intents
               SET status = 'rejected',
                   execution_error = :reason,
                   executed_at = :executed_at
               WHERE id = :id AND status = 'executing'""",
            {
                "reason": reason,
                "executed_at": _dt_to_str(now),
                "id": intent_id,
            },
        ).rowcount
        if not updated:
            raise InvalidTransitionError(
                f"Cannot mark {intent_id} as rejected: not in 'executing' state"
            )
        self._conn.execute(
            """UPDATE decisions
               SET status = 'rejected',
                   failure_reason = :reason,
                   executed_at = :executed_at
               WHERE intent_id = :intent_id""",
            {
                "reason": reason,
                "executed_at": _dt_to_str(now),
                "intent_id": intent_id,
            },
        )
        self._conn.commit()
        logger.info("Intent {} rejected: {}", intent_id, reason)

    def mark_failed(self, intent_id: str, error: str) -> None:
        """Transition intent from executing → failed with error details."""
        now = datetime.now(timezone.utc)
        updated = self._conn.execute(
            """UPDATE intents
               SET status = 'failed',
                   execution_error = :error,
                   executed_at = :executed_at
               WHERE id = :id AND status = 'executing'""",
            {
                "error": error,
                "executed_at": _dt_to_str(now),
                "id": intent_id,
            },
        ).rowcount
        if not updated:
            raise InvalidTransitionError(
                f"Cannot mark {intent_id} as failed: not in 'executing' state"
            )
        self._conn.execute(
            """UPDATE decisions
               SET status = 'failed',
                   failure_reason = :error,
                   executed_at = :executed_at
               WHERE intent_id = :intent_id""",
            {
                "error": error,
                "executed_at": _dt_to_str(now),
                "intent_id": intent_id,
            },
        )
        self._conn.commit()
        logger.warning("Intent {} failed: {}", intent_id, error)

    def mark_cancelled(self, intent_id: str, reason: str) -> None:
        """Cancel an intent from pending or claimed state."""
        now = datetime.now(timezone.utc)
        updated = self._conn.execute(
            """UPDATE intents
               SET status = 'cancelled',
                   execution_error = :reason,
                   executed_at = :executed_at
               WHERE id = :id AND status IN ('pending', 'claimed')""",
            {
                "reason": reason,
                "executed_at": _dt_to_str(now),
                "id": intent_id,
            },
        ).rowcount
        if not updated:
            raise InvalidTransitionError(
                f"Cannot cancel {intent_id}: not in 'pending' or 'claimed' state"
            )
        self._conn.execute(
            """UPDATE decisions
               SET status = 'cancelled', failure_reason = :reason
               WHERE intent_id = :intent_id""",
            {"reason": reason, "intent_id": intent_id},
        )
        self._conn.commit()
        logger.info("Intent {} cancelled: {}", intent_id, reason)

    def mark_closed(
        self,
        intent_id: str,
        *,
        realized_pnl: float | None = None,
        exit_price: float | None = None,
        exit_reason: str | None = None,
        hold_duration_seconds: int | None = None,
    ) -> None:
        """Transition intent from opened → closed with optional PnL data."""
        now = datetime.now(timezone.utc)
        updated = self._conn.execute(
            """UPDATE intents
               SET status = 'closed',
                   realized_pnl = :realized_pnl,
                   exit_price = :exit_price,
                   exit_reason = :exit_reason,
                   hold_duration_seconds = :hold_duration_seconds
               WHERE id = :id AND status = 'opened'""",
            {
                "realized_pnl": realized_pnl,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "hold_duration_seconds": hold_duration_seconds,
                "id": intent_id,
            },
        ).rowcount
        if not updated:
            raise InvalidTransitionError(f"Cannot close {intent_id}: not in 'opened' state")
        self._conn.execute(
            """UPDATE decisions
               SET status = 'closed',
                   closed_at = :closed_at,
                   realized_pnl = :realized_pnl,
                   exit_reason = :exit_reason
               WHERE intent_id = :intent_id""",
            {
                "closed_at": _dt_to_str(now),
                "realized_pnl": realized_pnl,
                "exit_reason": exit_reason,
                "intent_id": intent_id,
            },
        )
        self._conn.commit()
        logger.info(
            "Intent {} closed (pnl={}, reason={})",
            intent_id, realized_pnl, exit_reason,
        )

    def update_execution_meta(self, intent_id: str, meta_json: str) -> None:
        """Persist execution metadata (latency, slippage, prices) on the decision record.
        Args:
            intent_id: The intent whose decision record to update.
            meta_json: JSON string with execution details.
        """
        self._conn.execute(
            "UPDATE decisions SET execution_meta = :meta WHERE intent_id = :id",
            {"meta": meta_json, "id": intent_id},
        )
        self._conn.commit()
        logger.debug("Updated execution_meta for intent {}", intent_id)
    # ── Queries ─────────────────────────────────────────────────────

    def get_pending_intents(self) -> list[TradeIntent]:
        """Get all intents with status='pending', ordered by creation time."""
        rows = self._conn.execute(
            "SELECT * FROM intents WHERE status = 'pending' ORDER BY created_at ASC"
        ).fetchall()
        return [_row_to_intent(r) for r in rows]

    def get_ready_intents(self) -> list[TradeIntent]:
        """Get all intents with status='ready_for_exec', ordered by creation time."""
        rows = self._conn.execute(
            "SELECT * FROM intents WHERE status = 'ready_for_exec' ORDER BY created_at ASC"
        ).fetchall()
        return [_row_to_intent(r) for r in rows]

    def get_active_positions(self) -> list[TradeIntent]:
        """Get all intents with status='opened' (currently open positions)."""
        rows = self._conn.execute(
            "SELECT * FROM intents WHERE status = 'opened' ORDER BY executed_at ASC"
        ).fetchall()
        return [_row_to_intent(r) for r in rows]

    def get_closed_intents(self, days: int = 7) -> list[TradeIntent]:
        """Get closed intents from the last N days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_str = _dt_to_str(cutoff)
        rows = self._conn.execute(
            """SELECT * FROM intents
               WHERE status = 'closed' AND created_at >= :cutoff
               ORDER BY created_at DESC""",
            {"cutoff": cutoff_str},
        ).fetchall()
        return [_row_to_intent(r) for r in rows]

    def get_intent(self, intent_id: str) -> TradeIntent | None:
        """Get a single intent by ID."""
        row = self._conn.execute(
            "SELECT * FROM intents WHERE id = :id", {"id": intent_id}
        ).fetchone()
        if row is None:
            return None
        return _row_to_intent(row)

    def get_intents_by_date(self, trade_date: str) -> list[TradeIntent]:
        """Get all intents for a specific trading date."""
        rows = self._conn.execute(
            "SELECT * FROM intents WHERE trade_date = :td ORDER BY created_at ASC",
            {"td": trade_date},
        ).fetchall()
        return [_row_to_intent(r) for r in rows]

    def get_decision(self, intent_id: str) -> DecisionRecord | None:
        """Get the decision audit record for an intent."""
        row = self._conn.execute(
            "SELECT * FROM decisions WHERE intent_id = :id", {"id": intent_id}
        ).fetchone()
        if row is None:
            return None
        return _row_to_decision(row)

    # ── Claim Management ────────────────────────────────────────────

    def recycle_expired_claims(self) -> int:
        """Find claimed intents past their TTL and set them to timed_out.

        Returns:
            Number of intents recycled.
        """
        now = datetime.now(timezone.utc)
        now_str = _dt_to_str(now)
        cursor = self._conn.execute(
            """UPDATE intents
               SET status = 'timed_out'
               WHERE status = 'claimed'
                 AND expires_at IS NOT NULL
                 AND expires_at < :now""",
            {"now": now_str},
        )
        count = cursor.rowcount
        if count > 0:
            # Also update decision records
            self._conn.execute(
                """UPDATE decisions
                   SET status = 'timed_out',
                       failure_reason = 'Claim TTL expired'
                   WHERE intent_id IN (
                       SELECT id FROM intents WHERE status = 'timed_out'
                   ) AND status = 'claimed'"""
            )
            self._conn.commit()
            logger.warning("Recycled {} expired claims", count)
        return count

    def cleanup_old_intents(self, retention_days: int = 7) -> int:
        """Delete intents older than retention_days in terminal states.

        Only deletes intents in terminal states: closed, rejected, failed,
        cancelled, timed_out.

        Returns:
            Number of intents deleted.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        cutoff_str = _dt_to_str(cutoff)
        terminal_states = ("closed", "rejected", "failed", "cancelled", "timed_out")
        placeholders = ",".join(f"'{s}'" for s in terminal_states)

        # Delete decisions first (FK constraint)
        self._conn.execute(
            f"""DELETE FROM decisions
                WHERE intent_id IN (
                    SELECT id FROM intents
                    WHERE status IN ({placeholders})
                      AND created_at < :cutoff
                )""",
            {"cutoff": cutoff_str},
        )
        cursor = self._conn.execute(
            f"""DELETE FROM intents
                WHERE status IN ({placeholders})
                  AND created_at < :cutoff""",
            {"cutoff": cutoff_str},
        )
        count = cursor.rowcount
        if count > 0:
            self._conn.commit()
            logger.info("Cleaned up {} old intents (retention={}d)", count, retention_days)
        return count

    # ── Idempotency ─────────────────────────────────────────────────

    def intent_exists(self, symbol: str, trade_date: str, source: str) -> bool:
        """Check if an *in-progress* intent already exists for this symbol+date+source.

        Only intents that are still being processed (pending, claimed, ready_for_exec)
        block new intent creation.  Completed intents (opened, closed) and terminal
        intents (cancelled, timed_out, failed) do NOT block — the scanner is free to
        create a fresh intent for the same symbol after a previous trade completes.
        """
        row = self._conn.execute(
            """SELECT 1 FROM intents
               WHERE symbol = :symbol
                 AND trade_date = :td
                 AND source = :source
                 AND status IN ('pending', 'claimed', 'ready_for_exec')
               LIMIT 1""",
            {"symbol": symbol, "td": trade_date, "source": source},
        ).fetchone()
        return row is not None

    # ── API Call Tracking (Rate Limiter) ────────────────────────────

    def record_api_calls(self, count: int = 1, call_type: str = "read") -> int:
        """Record API calls for today and return the new total.
        Persists API call counts in SQLite so they survive restarts.

        Args:
            count: Number of API calls to record (default 1).
            call_type: "read" or "write" for tracking breakdown.
        Returns:
            Updated total API call count for today.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        now_str = _dt_to_str(datetime.now(timezone.utc))
        self._conn.execute(
            """INSERT INTO api_calls (call_date, call_count, updated_at)
               VALUES (:date, :count, :now)
               ON CONFLICT(call_date) DO UPDATE
               SET call_count = call_count + :count,
                   updated_at = :now""",
            {"date": today, "count": count, "now": now_str},
        )
        # Update read/write breakdown
        if call_type == "write":
            self._conn.execute(
                """UPDATE api_calls SET write_count = write_count + :count
                   WHERE call_date = :date""",
                {"count": count, "date": today},
            )
        else:
            self._conn.execute(
                """UPDATE api_calls SET read_count = read_count + :count
                   WHERE call_date = :date""",
                {"count": count, "date": today},
            )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT call_count FROM api_calls WHERE call_date = :date",
            {"date": today},
        ).fetchone()
        return row["call_count"] if row else count

    def get_api_call_count(self, date: str | None = None) -> int:
        """Get API call count for a given date (default: today).

        Args:
            date: Date string YYYY-MM-DD. Defaults to today UTC.

        Returns:
            Number of API calls recorded for that date.
        """
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row = self._conn.execute(
            "SELECT call_count FROM api_calls WHERE call_date = :date",
            {"date": date},
        ).fetchone()
        return row["call_count"] if row else 0


    def get_api_call_breakdown(self, date: str | None = None) -> dict[str, int]:
        """Get API call breakdown (total/read/write) for a date.

        Args:
            date: Date string YYYY-MM-DD. Defaults to today UTC.

        Returns:
            Dictionary with total, read, and write counts.
        """
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row = self._conn.execute(
            "SELECT call_count, read_count, write_count FROM api_calls WHERE call_date = :date",
            {"date": date},
        ).fetchone()
        if row is None:
            return {"total": 0, "read": 0, "write": 0}
        return {
            "total": row["call_count"],
            "read": row["read_count"] or 0,
            "write": row["write_count"] or 0,
        }

    # ── Equity Snapshots ────────────────────────────────────────────

    def insert_equity_snapshot(
        self,
        equity: float,
        daily_dd_pct: float,
        max_dd_pct: float,
        balance: float | None = None,
        open_positions: int = 0,
    ) -> None:
        """Persist a periodic equity snapshot for drawdown analysis.
        Called by the EquityMonitor callback to build an equity curve.
        Args:
            daily_dd_pct: Daily drawdown as fraction of limit consumed (0.0-1.0+).
            max_dd_pct: Max drawdown as fraction of limit consumed (0.0-1.0+).
            balance: Current account balance (optional).
            open_positions: Number of currently open positions.
        """
        now_str = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT INTO equity_snapshots
               (timestamp, equity, balance, daily_dd_pct, max_dd_pct, open_positions)
               VALUES (:ts, :equity, :balance, :daily_dd, :max_dd, :positions)""",
            {
                "ts": now_str,
                "equity": equity,
                "balance": balance,
                "daily_dd": daily_dd_pct,
                "max_dd": max_dd_pct,
                "positions": open_positions,
            },
        )
        self._conn.commit()

    def get_equity_history(self, hours: int = 24) -> list[dict[str, Any]]:
        """Query equity snapshots from the last N hours.

        Returns:
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        cutoff_str = cutoff.isoformat()
        rows = self._conn.execute(
            """SELECT timestamp, equity, balance, daily_dd_pct, max_dd_pct, open_positions
               FROM equity_snapshots
               WHERE timestamp >= :cutoff
               ORDER BY timestamp ASC""",
            {"cutoff": cutoff_str},
        ).fetchall()
        return [dict(row) for row in rows]
    # ── Dashboard Query Helpers ─────────────────────────────────────

    def get_daily_summary(self, trade_date: str | None = None) -> dict[str, int]:
        """Get a summary of intent statuses for a given date.

        Args:
            trade_date: Date string YYYY-MM-DD. Defaults to today UTC.

        Returns:
            Dict mapping status → count, e.g. {"pending": 2, "opened": 1}.
        """
        if trade_date is None:
            trade_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = self._conn.execute(
            """SELECT status, COUNT(*) as cnt
               FROM intents
               WHERE trade_date = :td
               GROUP BY status""",
            {"td": trade_date},
        ).fetchall()
        return {row["status"]: row["cnt"] for row in rows}

    def get_success_rate(self, days: int = 7) -> dict[str, float | int]:
        """Calculate trade success rate over the last N days.

        Success = opened or closed. Failure = rejected or failed.
        Intents still in pipeline (pending, claimed, etc.) are excluded.

        Args:
            days: Lookback window in days (default 7).

        Returns:
            Dict with total, success, failure counts and success_rate (0.0-1.0).
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_str = _dt_to_str(cutoff)
        row = self._conn.execute(
            """SELECT
                 COUNT(*) as total,
                 SUM(CASE WHEN status IN ('opened', 'closed') THEN 1 ELSE 0 END) as success,
                 SUM(CASE WHEN status IN ('rejected', 'failed') THEN 1 ELSE 0 END) as failure
               FROM intents
               WHERE status IN ('opened', 'closed', 'rejected', 'failed')
                 AND created_at >= :cutoff""",
            {"cutoff": cutoff_str},
        ).fetchone()
        total = row["total"] or 0
        success = row["success"] or 0
        failure = row["failure"] or 0
        rate = success / total if total > 0 else 0.0
        return {
            "total": total,
            "success": success,
            "failure": failure,
            "success_rate": round(rate, 4),
        }

    def get_symbol_stats(self, days: int = 7) -> list[dict[str, str | int]]:
        """Get per-symbol intent statistics over the last N days.

        Args:
            days: Lookback window in days (default 7).

        Returns:
            List of dicts with symbol, total, opened, rejected, failed counts.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_str = _dt_to_str(cutoff)
        rows = self._conn.execute(
            """SELECT
                 symbol,
                 COUNT(*) as total,
                 SUM(CASE WHEN status IN ('opened', 'closed') THEN 1 ELSE 0 END) as opened,
                 SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected,
                 SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
               FROM intents
               WHERE created_at >= :cutoff
               GROUP BY symbol
               ORDER BY total DESC""",
            {"cutoff": cutoff_str},
        ).fetchall()
        return [
            {
                "symbol": row["symbol"],
                "total": row["total"],
                "opened": row["opened"] or 0,
                "rejected": row["rejected"] or 0,
                "failed": row["failed"] or 0,
            }
            for row in rows
        ]

    def get_pipeline_status(self) -> dict[str, int]:
        """Get current pipeline status — how many intents in each state.

        Unlike get_daily_summary (date-filtered), this shows ALL active intents
        across all dates. Useful for real-time monitoring.

        Returns:
            Dict mapping status → count for all non-terminal intents,
            plus a 'total_active' key.
        """
        rows = self._conn.execute(
            """SELECT status, COUNT(*) as cnt
               FROM intents
               WHERE status IN ('pending', 'claimed', 'ready_for_exec', 'executing', 'opened')
               GROUP BY status"""
        ).fetchall()
        result = {row["status"]: row["cnt"] for row in rows}
        result["total_active"] = sum(result.values())
        return result

    # ── Internal Helpers ────────────────────────────────────────────

    def _transition(self, intent_id: str, from_status: str, to_status: str) -> None:
        """Generic single-status transition with validation."""
        if to_status not in VALID_TRANSITIONS.get(from_status, []):
            raise InvalidTransitionError(f"Invalid transition: {from_status} → {to_status}")
        updated = self._conn.execute(
            "UPDATE intents SET status = :to WHERE id = :id AND status = :from_s",
            {"to": to_status, "id": intent_id, "from_s": from_status},
        ).rowcount
        if not updated:
            raise InvalidTransitionError(
                f"Cannot transition {intent_id}: not in '{from_status}' state"
            )
        self._conn.execute(
            "UPDATE decisions SET status = :to WHERE intent_id = :id",
            {"to": to_status, "id": intent_id},
        )
        self._conn.commit()
        logger.debug("Intent {} transitioned: {} → {}", intent_id, from_status, to_status)
