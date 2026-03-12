"""
Tests for operational metrics collection (P3.11 + P3.12).

Validates:
1. OperationalMetrics counts LLM decisions (success/cancel/error)
2. Tracks tactical gate results (pass/block)
3. Tracks SL/TP hit counts
4. Tracks API retry counts (MatchTrader, Telegram)
5. get_summary() returns a complete snapshot
6. reset() clears all counters
"""

import pytest

from src.monitor.operational_metrics import OperationalMetrics


class TestOperationalMetrics:
    """Tests for OperationalMetrics counter class."""

    def test_initial_state_all_zero(self):
        """Fresh metrics have all counters at zero."""
        m = OperationalMetrics()
        summary = m.get_summary()
        assert summary["llm_success"] == 0
        assert summary["llm_cancel"] == 0
        assert summary["llm_error"] == 0
        assert summary["tactical_pass"] == 0
        assert summary["tactical_block"] == 0
        assert summary["sl_hits"] == 0
        assert summary["tp_hits"] == 0
        assert summary["api_retries"] == 0
        assert summary["telegram_failures"] == 0

    def test_record_llm_success(self):
        m = OperationalMetrics()
        m.record_llm_result("success")
        m.record_llm_result("success")
        assert m.get_summary()["llm_success"] == 2

    def test_record_llm_cancel(self):
        m = OperationalMetrics()
        m.record_llm_result("cancel")
        assert m.get_summary()["llm_cancel"] == 1

    def test_record_llm_error(self):
        m = OperationalMetrics()
        m.record_llm_result("error")
        assert m.get_summary()["llm_error"] == 1

    def test_llm_success_rate_with_data(self):
        m = OperationalMetrics()
        m.record_llm_result("success")
        m.record_llm_result("success")
        m.record_llm_result("cancel")
        m.record_llm_result("error")
        # 2 success / 4 total = 0.5
        assert m.get_summary()["llm_success_rate"] == pytest.approx(0.5)

    def test_llm_success_rate_no_data(self):
        m = OperationalMetrics()
        assert m.get_summary()["llm_success_rate"] == 0.0

    def test_record_tactical_pass(self):
        m = OperationalMetrics()
        m.record_tactical_result(passed=True)
        assert m.get_summary()["tactical_pass"] == 1

    def test_record_tactical_block(self):
        m = OperationalMetrics()
        m.record_tactical_result(passed=False)
        assert m.get_summary()["tactical_block"] == 1

    def test_tactical_block_rate(self):
        m = OperationalMetrics()
        m.record_tactical_result(passed=True)
        m.record_tactical_result(passed=False)
        m.record_tactical_result(passed=False)
        # 2 blocks / 3 total = 0.667
        assert m.get_summary()["tactical_block_rate"] == pytest.approx(2 / 3, rel=1e-2)

    def test_record_trade_close_sl(self):
        m = OperationalMetrics()
        m.record_trade_close("sl_hit")
        m.record_trade_close("sl_hit")
        assert m.get_summary()["sl_hits"] == 2

    def test_record_trade_close_tp(self):
        m = OperationalMetrics()
        m.record_trade_close("tp_hit")
        assert m.get_summary()["tp_hits"] == 1

    def test_record_trade_close_manual(self):
        m = OperationalMetrics()
        m.record_trade_close("manual_close")
        assert m.get_summary()["manual_closes"] == 1

    def test_record_api_retry(self):
        m = OperationalMetrics()
        m.record_api_retry("matchtrader")
        m.record_api_retry("matchtrader")
        m.record_api_retry("telegram")
        assert m.get_summary()["api_retries"] == 3
        assert m.get_summary()["matchtrader_retries"] == 2
        assert m.get_summary()["telegram_retries"] == 1

    def test_record_telegram_failure(self):
        m = OperationalMetrics()
        m.record_telegram_failure()
        m.record_telegram_failure()
        assert m.get_summary()["telegram_failures"] == 2

    def test_record_telegram_poll_degradation_transitions(self):
        m = OperationalMetrics()
        m.record_telegram_poll_failure()
        m.record_telegram_poll_failure()
        m.record_telegram_poll_circuit_open()
        m.record_telegram_poll_probe()
        m.record_telegram_poll_recovery()

        summary = m.get_summary()
        assert summary["telegram_poll_failures"] == 2
        assert summary["telegram_poll_circuit_opens"] == 1
        assert summary["telegram_poll_probe_polls"] == 1
        assert summary["telegram_poll_circuit_recoveries"] == 1

    def test_reset_clears_all(self):
        m = OperationalMetrics()
        m.record_llm_result("success")
        m.record_tactical_result(passed=True)
        m.record_trade_close("sl_hit")
        m.record_api_retry("matchtrader")
        m.record_telegram_failure()
        m.record_telegram_poll_failure()
        m.record_telegram_poll_circuit_open()
        m.reset()
        summary = m.get_summary()
        assert summary["llm_success"] == 0
        assert summary["tactical_pass"] == 0
        assert summary["sl_hits"] == 0
        assert summary["api_retries"] == 0
        assert summary["telegram_failures"] == 0
        assert summary["telegram_poll_failures"] == 0
        assert summary["telegram_poll_circuit_opens"] == 0

    def test_get_summary_returns_dict(self):
        """get_summary returns a plain dict suitable for JSON serialization."""
        m = OperationalMetrics()
        summary = m.get_summary()
        assert isinstance(summary, dict)
        # All values should be int or float
        for v in summary.values():
            assert isinstance(v, (int, float))
