# Tactical Exit Re-entry Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the current production loop where tactical exits close positions for tiny profit or tactical loss and the scanner recreates the same-symbol trade minutes later.

**Architecture:** Keep the current tactical-exit state machine, but add three focused controls: a real same-symbol post-close cooldown at scanner admission, `sl-like` tactical loss counters for existing circuit breakers, and minimum hold/profit gates before `severe_tactical_reversal` may full-close. Preserve current close-reason taxonomy so analytics still show tactical reasons while scanner breakers operate on normalized loss semantics.

**Tech Stack:** Python 3.10, sqlite3, pytest, asyncio, Pydantic v2

---

## File Map

- Modify: `src/config.py`
  Responsibility: add tactical-exit threshold config for severe reversal gating.
- Modify: `config/e8_signature_50k_challenge.yaml`
  Responsibility: explicitly set the new severe-reversal thresholds in the production account config seen in the 2026-03-26 to 2026-03-30 logs.
- Modify: `src/decision/tactical_exit_rules.py`
  Responsibility: add hold-time context and gate `severe_tactical_reversal` full closes.
- Modify: `src/scheduler/scheduler.py`
  Responsibility: populate `hold_seconds` in tactical snapshots, enforce recent-close cooldown at scanner admission, and switch loss-breaker queries to normalized `sl-like` counters.
- Modify: `src/decision_store/sqlite_store.py`
  Responsibility: provide store queries for recent-close cooldown and normalized `sl-like` loss counts.
- Test: `tests/test_config.py`
- Test: `tests/test_tactical_exit_rules.py`
- Test: `tests/test_tactical_exit_scheduler.py`
- Test: `tests/test_decision_store.py`
- Test: `tests/test_scheduler.py`

## Task 1: Gate `severe_tactical_reversal` by Hold Time and Profit Threshold

**Files:**
- Modify: `src/config.py`
- Modify: `config/e8_signature_50k_challenge.yaml`
- Modify: `src/decision/tactical_exit_rules.py`
- Modify: `src/scheduler/scheduler.py`
- Test: `tests/test_config.py`
- Test: `tests/test_tactical_exit_rules.py`
- Test: `tests/test_tactical_exit_scheduler.py`

- [ ] **Step 1: Write the failing config and tactical-exit tests**

```python
# tests/test_config.py
def test_tactical_exit_defaults(self) -> None:
    from src.config import TacticalConfig

    tc = TacticalConfig()
    assert tc.exit.enabled is True
    assert tc.exit.evaluation_interval_seconds == 60
    assert tc.exit.breakeven_activation_r == 0.3
    assert tc.exit.partial_close_ratio == 0.5
    assert tc.exit.defensive_exit_loss_r == -0.35
    assert tc.exit.defensive_exit_require_strong_candle is True
    assert tc.exit.severe_reversal_min_hold_seconds == 900
    assert tc.exit.severe_reversal_min_r == 0.5
    assert tc.exit.use_llm_exception_path is True


def test_e8_signature_50k_config_loads_as_runnable_tradelocker_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config import load_config

    monkeypatch.delenv("TRADINGAGENTS_ENABLED", raising=False)
    config = load_config("config/e8_signature_50k_challenge.yaml")

    assert config.tactical.exit.severe_reversal_min_hold_seconds == 900
    assert config.tactical.exit.severe_reversal_min_r == 0.5
```

```python
# tests/test_tactical_exit_rules.py
def _make_snapshot(
    *,
    current_price: float = 1.1035,
    sl_price: float | None = 1.0980,
    tp_price: float | None = 1.1080,
    unrealized_r: float = 0.35,
    hold_seconds: int | None = None,
    partial_close_done: bool = False,
    bars_5min: pd.DataFrame | None = None,
    bars_1h: pd.DataFrame | None = None,
) -> TacticalExitSnapshot:
    return TacticalExitSnapshot(
        position_id="POS-1",
        symbol="EURUSD",
        side="BUY",
        open_price=1.1000,
        current_price=current_price,
        volume=0.10,
        sl_price=sl_price,
        tp_price=tp_price,
        original_sl_price=1.0980,
        original_tp_price=1.1080,
        unrealized_r=unrealized_r,
        partial_close_done=partial_close_done,
        hold_seconds=hold_seconds,
        bars_5min=bars_5min if bars_5min is not None else pd.DataFrame(),
        bars_1h=bars_1h if bars_1h is not None else pd.DataFrame(),
        last_tactical_exit_at=datetime(2026, 3, 12, 8, 0, tzinfo=timezone.utc),
    )


def test_severe_reversal_requires_min_hold_time_before_exit_now() -> None:
    snapshot = _make_snapshot(
        current_price=1.0985,
        unrealized_r=0.9,
        hold_seconds=120,
        bars_5min=_make_failed_buy_5min_bars(),
        bars_1h=_make_trending_1h_bars(),
    )

    decision = choose_tactical_exit(snapshot, TacticalExitConfig())

    assert decision.action == "PARTIAL_CLOSE"
    assert decision.reason == "profit_protection_partial_close"


def test_severe_reversal_requires_min_r_before_exit_now() -> None:
    snapshot = _make_snapshot(
        current_price=1.0988,
        unrealized_r=0.4,
        hold_seconds=1800,
        bars_5min=_make_failed_buy_5min_bars(),
        bars_1h=_make_trending_1h_bars(),
    )

    decision = choose_tactical_exit(snapshot, TacticalExitConfig())

    assert decision.action == "MOVE_TO_BREAKEVEN"
    assert decision.reason == "breakeven_threshold_reached"


def test_severe_reversal_still_exits_after_min_hold_and_profit_threshold() -> None:
    snapshot = _make_snapshot(
        current_price=1.0985,
        unrealized_r=0.9,
        hold_seconds=1800,
        bars_5min=_make_failed_buy_5min_bars(),
        bars_1h=_make_trending_1h_bars(),
    )

    decision = choose_tactical_exit(snapshot, TacticalExitConfig())

    assert decision.action == "EXIT_NOW"
    assert decision.reason == "severe_tactical_reversal"
```

```python
# tests/test_tactical_exit_scheduler.py
@pytest.mark.asyncio
async def test_run_tactical_exit_cycle_populates_hold_seconds_in_snapshot(
    scheduler: Scheduler,
) -> None:
    scheduler._fetch_tactical_data = AsyncMock(return_value=TacticalData())
    scheduler._handle_tactical_exit_evaluation = AsyncMock()
    scheduler._tactical_exit_manager = MagicMock()
    scheduler._tactical_exit_manager.evaluate_position.return_value = TacticalExitEvaluation(
        decision=TacticalExitDecision(
            action="HOLD",
            state="INITIAL_RISK",
            reason="no_tactical_exit_action",
        )
    )

    position = _make_position()
    intent = _make_opened_intent()
    intent.executed_at = datetime.now(timezone.utc) - timedelta(minutes=20)

    await scheduler._run_tactical_exit_cycle([position], [intent])

    snapshot = scheduler._tactical_exit_manager.evaluate_position.call_args.kwargs["snapshot"]
    assert snapshot.hold_seconds is not None
    assert snapshot.hold_seconds >= 1200
```

- [ ] **Step 2: Run the targeted tests to verify they fail first**

Run:

```bash
uv run pytest tests/test_tactical_exit_rules.py -k severe_reversal
uv run pytest tests/test_tactical_exit_scheduler.py -k hold_seconds
uv run pytest tests/test_config.py -k "tactical_exit_defaults or e8_signature_50k_config_loads_as_runnable_tradelocker_account"
```

Expected:

```text
FAILED tests/test_tactical_exit_rules.py::test_severe_reversal_requires_min_hold_time_before_exit_now
FAILED tests/test_tactical_exit_rules.py::test_severe_reversal_requires_min_r_before_exit_now
FAILED tests/test_tactical_exit_scheduler.py::test_run_tactical_exit_cycle_populates_hold_seconds_in_snapshot
FAILED tests/test_config.py::test_tactical_exit_defaults
FAILED tests/test_config.py::test_e8_signature_50k_config_loads_as_runnable_tradelocker_account
```

- [ ] **Step 3: Implement the minimal config, snapshot, and rule changes**

```python
# src/config.py
class TacticalExitConfig(BaseModel):
    ...
    severe_reversal_min_hold_seconds: int = Field(
        default=900,
        description="Minimum hold time before severe tactical reversal may full-close",
    )
    severe_reversal_min_r: float = Field(
        default=0.5,
        description="Minimum unrealized R before severe tactical reversal may full-close",
    )
```

```yaml
# config/e8_signature_50k_challenge.yaml
tactical:
  ...
  exit:
    ...
    severe_reversal_min_hold_seconds: 900
    severe_reversal_min_r: 0.5
```

```python
# src/decision/tactical_exit_rules.py
@dataclass
class TacticalExitSnapshot:
    ...
    unrealized_r: float
    hold_seconds: int | None = None
    partial_close_done: bool = False


def _is_severe_reversal(
    snapshot: TacticalExitSnapshot,
    context: ExitSignalContext,
    config: TacticalExitConfig,
) -> bool:
    if context.ema_aligned is None:
        return False
    if (snapshot.hold_seconds or 0) < config.severe_reversal_min_hold_seconds:
        return False
    if snapshot.unrealized_r < config.severe_reversal_min_r:
        return False
    return (
        snapshot.unrealized_r > 0
        and context.ema_aligned is False
        and (context.adverse_rsi or context.opposing_candle_strong)
    )


def choose_tactical_exit(
    snapshot: TacticalExitSnapshot,
    config: TacticalExitConfig,
) -> TacticalExitDecision:
    context = build_exit_signal_context(snapshot)
    state = classify_tactical_exit_state(snapshot, config, context=context)

    if _is_severe_reversal(snapshot, context, config):
        return TacticalExitDecision(
            action="EXIT_NOW",
            state="PROFIT_PROTECTION",
            reason="severe_tactical_reversal",
            requires_llm_exception=True,
        )
    ...
```

```python
# src/scheduler/scheduler.py
def _build_tactical_exit_snapshot(
    self,
    pos: Any,
    intent: TradeIntent,
    tactical_data: TacticalData,
) -> TacticalExitSnapshot:
    meta = self._load_execution_meta_dict(intent)
    hold_seconds = None
    opened_at = intent.executed_at or intent.created_at
    if opened_at is not None:
        hold_seconds = int((self._now_utc() - opened_at).total_seconds())
    return TacticalExitSnapshot(
        ...
        unrealized_r=self._compute_tactical_exit_unrealized_r(pos, meta),
        hold_seconds=hold_seconds,
        partial_close_done=bool(meta.get("partial_close_done", False)),
        ...
    )
```

- [ ] **Step 4: Re-run the targeted tests and the surrounding tactical/config suites**

Run:

```bash
uv run pytest tests/test_tactical_exit_rules.py
uv run pytest tests/test_tactical_exit_scheduler.py -k "hold_seconds or llm_reeval_only_runs_for_exception_cases"
uv run pytest tests/test_config.py -k "tactical_exit_defaults or e8_signature_50k_config_loads_as_runnable_tradelocker_account"
```

Expected:

```text
... passed
```

- [ ] **Step 5: Commit**

```bash
git add src/config.py config/e8_signature_50k_challenge.yaml src/decision/tactical_exit_rules.py src/scheduler/scheduler.py tests/test_config.py tests/test_tactical_exit_rules.py tests/test_tactical_exit_scheduler.py
git commit -m "fix: gate severe tactical reversal exits"
```

## Task 2: Add Recent-Close Cooldown and `sl-like` Loss Queries to DecisionStore

**Files:**
- Modify: `src/decision_store/sqlite_store.py`
- Test: `tests/test_decision_store.py`

- [ ] **Step 1: Write the failing DecisionStore tests**

```python
# tests/test_decision_store.py
class TestRecentCloseCooldownAndSlLikeLosses:
    def _open_and_close(
        self,
        store: DecisionStore,
        symbol: str,
        *,
        pnl: float,
        exit_reason: str,
    ) -> TradeIntent:
        intent = TradeIntent(trade_date="2026-03-27", symbol=symbol)
        store.insert_intent(intent)
        store.claim_next_pending("llm-0")
        store.update_intent_decision(intent.id, "SELL", 20.0, 40.0, "report", "{}")
        store.mark_ready_for_exec(intent.id)
        store.mark_executing(intent.id)
        store.mark_opened(intent.id, position_id=f"POS-{intent.id[:8]}")
        store.mark_closed(intent.id, realized_pnl=pnl, exit_reason=exit_reason)
        closed = store.get_intent(intent.id)
        assert closed is not None
        return closed

    def test_has_recent_closed_intent_for_symbol_returns_true_within_cooldown(
        self, store: DecisionStore
    ) -> None:
        self._open_and_close(
            store,
            "AUDUSD",
            pnl=2.64,
            exit_reason="severe_tactical_reversal",
        )

        assert store.has_recent_closed_intent_for_symbol(
            "AUDUSD",
            cooldown_seconds=1800,
        ) is True

    def test_has_recent_closed_intent_for_symbol_returns_false_outside_cooldown(
        self, store: DecisionStore
    ) -> None:
        closed = self._open_and_close(
            store,
            "AUDUSD",
            pnl=2.64,
            exit_reason="severe_tactical_reversal",
        )
        store._conn.execute(
            "UPDATE decisions SET closed_at = ? WHERE intent_id = ?",
            ("2026-03-27T00:00:00+00:00", closed.id),
        )
        store._conn.commit()

        assert store.has_recent_closed_intent_for_symbol(
            "AUDUSD",
            cooldown_seconds=1800,
        ) is False

    def test_count_sl_like_losses_today_includes_initial_risk_structure_failure(
        self, store: DecisionStore
    ) -> None:
        self._open_and_close(
            store,
            "AUDUSD",
            pnl=-86.13,
            exit_reason="initial_risk_structure_failure",
        )

        assert store.count_sl_like_losses_today("2026-03-27") == 1
        assert store.count_symbol_sl_like_losses_today("AUDUSD", "2026-03-27") == 1

    def test_count_sl_like_losses_today_excludes_positive_severe_tactical_reversal(
        self, store: DecisionStore
    ) -> None:
        self._open_and_close(
            store,
            "AUDUSD",
            pnl=17.48,
            exit_reason="severe_tactical_reversal",
        )

        assert store.count_sl_like_losses_today("2026-03-27") == 0
        assert store.count_symbol_sl_like_losses_today("AUDUSD", "2026-03-27") == 0
```

- [ ] **Step 2: Run the targeted tests to verify they fail first**

Run:

```bash
uv run pytest tests/test_decision_store.py -k "recent_closed_intent_for_symbol or sl_like_losses"
```

Expected:

```text
FAILED tests/test_decision_store.py::TestRecentCloseCooldownAndSlLikeLosses::test_has_recent_closed_intent_for_symbol_returns_true_within_cooldown
FAILED tests/test_decision_store.py::TestRecentCloseCooldownAndSlLikeLosses::test_count_sl_like_losses_today_includes_initial_risk_structure_failure
```

- [ ] **Step 3: Implement the new store helpers**

```python
# src/decision_store/sqlite_store.py
SL_LIKE_EXIT_REASONS = (
    "sl_hit",
    "initial_risk_structure_failure",
    "severe_tactical_reversal",
)


def has_recent_closed_intent_for_symbol(
    self,
    symbol: str,
    *,
    cooldown_seconds: int,
) -> bool:
    if cooldown_seconds <= 0:
        return False
    cutoff = _dt_to_str(datetime.now(timezone.utc) - timedelta(seconds=cooldown_seconds))
    row = self._conn.execute(
        """SELECT 1
           FROM decisions d
           JOIN intents i ON i.id = d.intent_id
           WHERE i.symbol = :symbol
             AND i.status = 'closed'
             AND d.status = 'closed'
             AND d.closed_at IS NOT NULL
             AND d.closed_at > :cutoff
           LIMIT 1""",
        {"symbol": symbol, "cutoff": cutoff},
    ).fetchone()
    return row is not None


def count_sl_like_losses_today(self, trade_date: str) -> int:
    row = self._conn.execute(
        """SELECT COUNT(*) AS cnt FROM intents
           WHERE trade_date = :td
             AND status = 'closed'
             AND COALESCE(realized_pnl, 0) < 0
             AND exit_reason IN (
                 'sl_hit',
                 'initial_risk_structure_failure',
                 'severe_tactical_reversal'
             )""",
        {"td": trade_date},
    ).fetchone()
    return row["cnt"] if row else 0


def count_symbol_sl_like_losses_today(self, symbol: str, trade_date: str) -> int:
    row = self._conn.execute(
        """SELECT COUNT(*) AS cnt FROM intents
           WHERE symbol = :symbol
             AND trade_date = :td
             AND status = 'closed'
             AND COALESCE(realized_pnl, 0) < 0
             AND exit_reason IN (
                 'sl_hit',
                 'initial_risk_structure_failure',
                 'severe_tactical_reversal'
             )""",
        {"symbol": symbol, "td": trade_date},
    ).fetchone()
    return row["cnt"] if row else 0
```

- [ ] **Step 4: Re-run the targeted tests and the full store suite**

Run:

```bash
uv run pytest tests/test_decision_store.py -k "recent_closed_intent_for_symbol or sl_like_losses"
uv run pytest tests/test_decision_store.py
```

Expected:

```text
... passed
```

- [ ] **Step 5: Commit**

```bash
git add src/decision_store/sqlite_store.py tests/test_decision_store.py
git commit -m "fix: add recent-close cooldown and sl-like loss counters"
```

## Task 3: Wire Scanner Admission to Cooldown and `sl-like` Breakers

**Files:**
- Modify: `src/scheduler/scheduler.py`
- Test: `tests/test_scheduler.py`

- [ ] **Step 1: Write the failing scheduler regression tests**

```python
# tests/test_scheduler.py
async def test_scanner_skips_symbol_with_recent_close_cooldown(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
) -> None:
    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )
    closed = _advance_intent_to_closed(
        store,
        Scheduler._today_str(),
        symbol="AUDUSD",
        realized_pnl=17.48,
    )
    store._conn.execute(
        "UPDATE intents SET exit_reason = ? WHERE id = ?",
        ("severe_tactical_reversal", closed.id),
    )
    store._conn.execute(
        "UPDATE decisions SET exit_reason = ?, closed_at = ? WHERE intent_id = ?",
        ("severe_tactical_reversal", datetime.now(timezone.utc).isoformat(), closed.id),
    )
    store._conn.commit()

    mock_scanner.run_pipeline.return_value = [_make_mock_signal("AUDUSD", score=0.91)]
    await _run_loop_once(sched, sched._scanner_loop())

    audusd_intents = [i for i in store.get_intents_by_date(Scheduler._today_str()) if i.symbol == "AUDUSD"]
    assert len(audusd_intents) == 1
    assert audusd_intents[0].status == "closed"


async def test_scanner_symbol_loss_breaker_counts_initial_risk_structure_failure(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
) -> None:
    config.scheduler.symbol_loss_limit = 1
    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )
    closed = _advance_intent_to_closed(
        store,
        Scheduler._today_str(),
        symbol="AUDUSD",
        realized_pnl=-86.13,
    )
    store._conn.execute(
        "UPDATE intents SET exit_reason = ?, realized_pnl = ? WHERE id = ?",
        ("initial_risk_structure_failure", -86.13, closed.id),
    )
    store._conn.execute(
        "UPDATE decisions SET exit_reason = ?, realized_pnl = ? WHERE intent_id = ?",
        ("initial_risk_structure_failure", -86.13, closed.id),
    )
    store._conn.commit()

    mock_scanner.run_pipeline.return_value = [_make_mock_signal("AUDUSD", score=0.91)]
    await _run_loop_once(sched, sched._scanner_loop())

    audusd_intents = [i for i in store.get_intents_by_date(Scheduler._today_str()) if i.symbol == "AUDUSD"]
    assert len(audusd_intents) == 1
    assert audusd_intents[0].status == "closed"


async def test_scanner_daily_loss_breaker_counts_initial_risk_structure_failure(
    config: AppConfig,
    store: DecisionStore,
    mock_scanner: MagicMock,
    mock_agents: MagicMock,
    mock_engine: AsyncMock,
    mock_matchtrader: AsyncMock,
) -> None:
    config.scheduler.daily_sl_hit_limit = 1
    sched = Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
    )
    closed = _advance_intent_to_closed(
        store,
        Scheduler._today_str(),
        symbol="AUDUSD",
        realized_pnl=-98.88,
    )
    store._conn.execute(
        "UPDATE intents SET exit_reason = ?, realized_pnl = ? WHERE id = ?",
        ("initial_risk_structure_failure", -98.88, closed.id),
    )
    store._conn.execute(
        "UPDATE decisions SET exit_reason = ?, realized_pnl = ? WHERE intent_id = ?",
        ("initial_risk_structure_failure", -98.88, closed.id),
    )
    store._conn.commit()

    mock_scanner.run_pipeline.return_value = [_make_mock_signal("GBPUSD", score=0.91)]
    await _run_loop_once(sched, sched._scanner_loop())

    symbols = {i.symbol for i in store.get_intents_by_date(Scheduler._today_str())}
    assert symbols == {"AUDUSD"}
```

- [ ] **Step 2: Run the targeted tests to verify they fail first**

Run:

```bash
uv run pytest tests/test_scheduler.py -k "recent_close_cooldown or initial_risk_structure_failure"
```

Expected:

```text
FAILED tests/test_scheduler.py::test_scanner_skips_symbol_with_recent_close_cooldown
FAILED tests/test_scheduler.py::test_scanner_symbol_loss_breaker_counts_initial_risk_structure_failure
FAILED tests/test_scheduler.py::test_scanner_daily_loss_breaker_counts_initial_risk_structure_failure
```

- [ ] **Step 3: Implement the scheduler admission changes**

```python
# src/scheduler/scheduler.py
recent_close_cooldown = self._config.tactical.intent_dedup.cooldown_after_close_seconds
if recent_close_cooldown > 0:
    recently_closed = await asyncio.to_thread(
        self._store.has_recent_closed_intent_for_symbol,
        signal.instrument,
        cooldown_seconds=recent_close_cooldown,
    )
    if recently_closed:
        self._log_trade_event(
            "SCANNER_SKIP",
            {
                "symbol": signal.instrument,
                "reason": "recent_close_cooldown",
                "cooldown_seconds": recent_close_cooldown,
            },
        )
        logger.warning(
            "Scanner loop: {} closed within {}s cooldown, skipping re-entry",
            signal.instrument,
            recent_close_cooldown,
        )
        continue

daily_sl_count = await asyncio.to_thread(
    self._store.count_sl_like_losses_today,
    today,
)
...
symbol_sl_count = await asyncio.to_thread(
    self._store.count_symbol_sl_like_losses_today,
    signal.instrument,
    today,
)
```

- [ ] **Step 4: Re-run the targeted tests and the surrounding scheduler suites**

Run:

```bash
uv run pytest tests/test_scheduler.py -k "recent_close_cooldown or initial_risk_structure_failure"
uv run pytest tests/test_scheduler.py -k "tactical or active_position or recent_close_cooldown or initial_risk_structure_failure"
```

Expected:

```text
... passed
```

- [ ] **Step 5: Commit**

```bash
git add src/scheduler/scheduler.py tests/test_scheduler.py
git commit -m "fix: block rapid re-entry after tactical closes"
```

## Verification Checklist

- `uv run pytest tests/test_config.py -k "tactical_exit_defaults or e8_signature_50k_config_loads_as_runnable_tradelocker_account"`
- `uv run pytest tests/test_tactical_exit_rules.py`
- `uv run pytest tests/test_tactical_exit_scheduler.py`
- `uv run pytest tests/test_decision_store.py`
- `uv run pytest tests/test_scheduler.py -k "tactical or active_position or recent_close_cooldown or initial_risk_structure_failure"`
- `uv run ruff check src/config.py src/decision/tactical_exit_rules.py src/decision_store/sqlite_store.py src/scheduler/scheduler.py tests/test_config.py tests/test_tactical_exit_rules.py tests/test_tactical_exit_scheduler.py tests/test_decision_store.py tests/test_scheduler.py`

## Expected Production Outcomes

- A symbol closed moments ago will not be recreated by the next scanner cycle.
- Tactical losses with negative realized PnL will trip the same symbol and daily loss breakers as stop losses.
- `severe_tactical_reversal` will no longer full-close positions that are only a few minutes old or only trivially profitable.
