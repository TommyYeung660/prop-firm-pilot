"""Tests for rule-based entry-funnel ablation diagnostics."""

from src.diagnostics.analyze_entry_funnel_ablation import analyze_ablation


def test_analyze_entry_funnel_ablation_emits_mode_labels_and_summaries() -> None:
    snapshots = [
        {
            "entry_funnel_mode": "scanner_tactical",
            "date": "2026-03-14",
            "net_pnl": 120.0,
            "profit_factor": 1.4,
            "max_drawdown": 40.0,
            "scanner_candidates": 10,
            "intents_created": 8,
            "opened_count": 4,
            "llm_vetoes": 0,
            "llm_cancels": 0,
            "tactical_waits": 2,
            "tactical_expires": 1,
            "no_trade_count": 2,
        },
        {
            "entry_funnel_mode": "scanner_llm_tactical",
            "date": "2026-03-14",
            "net_pnl": 160.0,
            "profit_factor": 1.8,
            "max_drawdown": 30.0,
            "scanner_candidates": 10,
            "intents_created": 6,
            "opened_count": 3,
            "llm_vetoes": 2,
            "llm_cancels": 1,
            "tactical_waits": 1,
            "tactical_expires": 0,
            "no_trade_count": 3,
        },
        {
            "entry_funnel_mode": "tactical_only",
            "date": "2026-03-14",
            "net_pnl": 40.0,
            "profit_factor": 1.1,
            "max_drawdown": 25.0,
            "scanner_candidates": 0,
            "intents_created": 2,
            "opened_count": 2,
            "llm_vetoes": 0,
            "llm_cancels": 0,
            "tactical_waits": 1,
            "tactical_expires": 0,
            "no_trade_count": 0,
        },
        {
            "entry_funnel_mode": "no_trade",
            "date": "2026-03-14",
            "net_pnl": 0.0,
            "profit_factor": 1.0,
            "max_drawdown": 0.0,
            "scanner_candidates": 8,
            "intents_created": 0,
            "opened_count": 0,
            "llm_vetoes": 0,
            "llm_cancels": 0,
            "tactical_waits": 0,
            "tactical_expires": 0,
            "no_trade_count": 8,
        },
    ]

    result = analyze_ablation(snapshots)

    assert result["mode_labels"]["A"]["mode"] == "scanner_tactical"
    assert result["mode_labels"]["B"]["mode"] == "scanner_llm_tactical"
    assert result["mode_labels"]["C"]["mode"] == "tactical_only"
    assert result["mode_labels"]["D"]["mode"] == "no_trade"
    assert result["economic_summary"]["B"]["net_pnl"] == 160.0
    assert result["funnel_summary"]["A"]["opened_trade_rate"] == 0.4
    assert result["churn_summary"]["B"]["llm_veto_rate"] == 0.2
    assert result["churn_summary"]["A"]["tactical_wait_then_expire_rate"] == 0.5


def test_analyze_entry_funnel_ablation_recommends_llm_downgrade_when_b_not_better_than_a() -> None:
    snapshots = [
        {
            "entry_funnel_mode": "scanner_tactical",
            "date": "2026-03-14",
            "net_pnl": 150.0,
            "profit_factor": 1.6,
            "max_drawdown": 45.0,
            "scanner_candidates": 12,
            "intents_created": 9,
            "opened_count": 5,
            "llm_vetoes": 0,
            "llm_cancels": 0,
            "tactical_waits": 2,
            "tactical_expires": 0,
            "no_trade_count": 1,
        },
        {
            "entry_funnel_mode": "scanner_llm_tactical",
            "date": "2026-03-14",
            "net_pnl": 90.0,
            "profit_factor": 1.1,
            "max_drawdown": 55.0,
            "scanner_candidates": 12,
            "intents_created": 8,
            "opened_count": 4,
            "llm_vetoes": 3,
            "llm_cancels": 2,
            "tactical_waits": 3,
            "tactical_expires": 2,
            "no_trade_count": 4,
        },
        {
            "entry_funnel_mode": "tactical_only",
            "date": "2026-03-14",
            "net_pnl": 80.0,
            "profit_factor": 1.2,
            "max_drawdown": 35.0,
            "scanner_candidates": 0,
            "intents_created": 3,
            "opened_count": 3,
            "llm_vetoes": 0,
            "llm_cancels": 0,
            "tactical_waits": 1,
            "tactical_expires": 0,
            "no_trade_count": 0,
        },
        {
            "entry_funnel_mode": "no_trade",
            "date": "2026-03-14",
            "net_pnl": 0.0,
            "profit_factor": 1.0,
            "max_drawdown": 0.0,
            "scanner_candidates": 12,
            "intents_created": 0,
            "opened_count": 0,
            "llm_vetoes": 0,
            "llm_cancels": 0,
            "tactical_waits": 0,
            "tactical_expires": 0,
            "no_trade_count": 12,
        },
    ]

    result = analyze_ablation(snapshots)

    assert result["recommendation"] == "downgrade_llm_to_confirm_veto"


def test_analyze_entry_funnel_ablation_recommends_no_trade_shadow_mode_when_all_modes_fail() -> None:
    snapshots = [
        {
            "entry_funnel_mode": "scanner_tactical",
            "date": "2026-03-14",
            "net_pnl": -40.0,
            "profit_factor": 0.8,
            "max_drawdown": 60.0,
            "scanner_candidates": 9,
            "intents_created": 7,
            "opened_count": 4,
            "llm_vetoes": 0,
            "llm_cancels": 0,
            "tactical_waits": 2,
            "tactical_expires": 1,
            "no_trade_count": 1,
        },
        {
            "entry_funnel_mode": "scanner_llm_tactical",
            "date": "2026-03-14",
            "net_pnl": -10.0,
            "profit_factor": 0.9,
            "max_drawdown": 35.0,
            "scanner_candidates": 9,
            "intents_created": 6,
            "opened_count": 3,
            "llm_vetoes": 1,
            "llm_cancels": 1,
            "tactical_waits": 2,
            "tactical_expires": 1,
            "no_trade_count": 3,
        },
        {
            "entry_funnel_mode": "tactical_only",
            "date": "2026-03-14",
            "net_pnl": -5.0,
            "profit_factor": 0.95,
            "max_drawdown": 20.0,
            "scanner_candidates": 0,
            "intents_created": 2,
            "opened_count": 2,
            "llm_vetoes": 0,
            "llm_cancels": 0,
            "tactical_waits": 1,
            "tactical_expires": 1,
            "no_trade_count": 0,
        },
        {
            "entry_funnel_mode": "no_trade",
            "date": "2026-03-14",
            "net_pnl": 0.0,
            "profit_factor": 1.0,
            "max_drawdown": 0.0,
            "scanner_candidates": 9,
            "intents_created": 0,
            "opened_count": 0,
            "llm_vetoes": 0,
            "llm_cancels": 0,
            "tactical_waits": 0,
            "tactical_expires": 0,
            "no_trade_count": 9,
        },
    ]

    result = analyze_ablation(snapshots)

    assert result["recommendation"] == "return_to_no_trade_shadow_mode"
