"""
Tests for TradeJournal event logging.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import (
    AccountConfig,
    AppConfig,
    ComplianceConfig,
    DecisionStoreConfig,
    MonitorConfig,
    SchedulerConfig,
)
from src.decision.agent_bridge import AgentDecision
from src.decision.schemas import TradeIntent
from src.decision.tactical_validator import GateResult, TacticalResult
from src.decision_store.sqlite_store import DecisionStore
from src.monitor.trade_journal import TradeJournal
from src.scheduler.scheduler import Scheduler


def test_log_event_appends(tmp_path) -> None:
    path = tmp_path / "trade_journal.jsonl"
    journal = TradeJournal(path)

    journal.log_event("LLM_DECISION", {"symbol": "EURUSD", "decision": "BUY"})

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    entry = json.loads(lines[-1])
    assert entry["type"] == "LLM_DECISION"
    assert entry["symbol"] == "EURUSD"
    assert entry["decision"] == "BUY"


@pytest.fixture
def store(tmp_path: object) -> DecisionStore:
    db_path = f"{tmp_path}/test_trade_journal.db"
    s = DecisionStore(db_path=db_path)
    yield s  # type: ignore[misc]
    s.close()


@pytest.fixture
def config() -> AppConfig:
    return AppConfig(
        account=AccountConfig(initial_balance=50000),
        compliance=ComplianceConfig(),
        scheduler=SchedulerConfig(
            scanner_interval_seconds=0,
            llm_poll_interval_seconds=0,
            execution_poll_interval_seconds=0,
            janitor_interval_seconds=0,
            llm_worker_count=1,
            equity_poll_interval_seconds=0,
            position_monitor_interval_seconds=0,
            daily_summary_hour_utc=22,
        ),
        decision_store=DecisionStoreConfig(),
        monitor=MonitorConfig(),
    )


@pytest.fixture
def trade_journal(tmp_path) -> TradeJournal:
    return TradeJournal(tmp_path / "trade_journal.jsonl")


@pytest.fixture
def mock_agents() -> MagicMock:
    agents = MagicMock()
    agents.using_mock = False
    agents.decide.return_value = AgentDecision(
        symbol="EURUSD",
        decision="BUY",
        final_state={"test": True},
        risk_report="test",
    )
    return agents


@pytest.fixture
def scheduler(
    config: AppConfig,
    store: DecisionStore,
    mock_agents: MagicMock,
    trade_journal: TradeJournal,
) -> Scheduler:
    mock_scanner = MagicMock()
    mock_engine = AsyncMock()
    mock_matchtrader = AsyncMock()
    return Scheduler(
        config=config,
        store=store,
        scanner=mock_scanner,
        agents=mock_agents,
        engine=mock_engine,
        matchtrader=mock_matchtrader,
        trade_journal=trade_journal,
    )


class TestSchedulerTradeJournalIntegration:
    async def test_llm_decision_logged(self, scheduler: Scheduler, store: DecisionStore) -> None:
        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.85,
            scanner_confidence="high",
        )
        store.insert_intent(intent)
        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None

        await scheduler._process_claimed_intent("llm-0", claimed)

        path = scheduler._trade_journal._path
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        events = [json.loads(line) for line in lines]
        assert any(e.get("type") == "LLM_DECISION" for e in events)

    async def test_tactical_result_logged_with_reason_code_and_provenance(
        self,
        scheduler: Scheduler,
        store: DecisionStore,
    ) -> None:
        scheduler._config.tactical.enabled = True
        scheduler._config.tactical.shadow_mode = True

        intent = TradeIntent(
            trade_date=Scheduler._today_str(),
            symbol="EURUSD",
            scanner_score=0.85,
            scanner_confidence="high",
        )
        store.insert_intent(intent)
        claimed = store.claim_next_pending("llm-0")
        assert claimed is not None

        tactical_result = TacticalResult(
            action="WAIT",
            resolution="RETRY_PENDING",
            summary_reason_code="spread.fail.ratio_too_wide",
            detail="Spread too wide",
            hard_gates=[
                GateResult(
                    gate_name="spread",
                    passed=False,
                    status="FAIL",
                    reason_code="spread.fail.ratio_too_wide",
                    detail="spread_ratio=3.33, limit=2.0x",
                )
            ],
            provenance={"data_source": "rest_fallback", "quote_source": "rest_fallback"},
            policy_hints={"retryable": True},
        )

        with patch.object(
            scheduler,
            "_run_tactical_validation",
            new_callable=AsyncMock,
        ) as mock_tac:
            mock_tac.return_value = tactical_result
            await scheduler._process_claimed_intent("llm-0", claimed)

        path = scheduler._trade_journal._path
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        events = [json.loads(line) for line in lines]
        event = next(e for e in events if e.get("type") == "TACTICAL_RESULT")
        assert event["resolution"] == "RETRY_PENDING"
        assert event["summary_reason_code"] == "spread.fail.ratio_too_wide"
        assert event["provenance"]["data_source"] == "rest_fallback"
        assert event["hard_gates"][0]["reason_code"] == "spread.fail.ratio_too_wide"
