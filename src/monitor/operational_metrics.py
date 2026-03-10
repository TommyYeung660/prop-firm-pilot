"""
Operational metrics collector for system health monitoring.

Aggregates counters for LLM decisions, tactical gates, trade outcomes,
API retries, and Telegram failures. Exposed via daily summary and
JSONL trade journal.

Usage:
    metrics = OperationalMetrics()
    metrics.record_llm_result("success")
    metrics.record_api_retry("matchtrader")
    summary = metrics.get_summary()  # dict of all counters
"""

from loguru import logger


class OperationalMetrics:
    """In-memory operational metrics collector.

    Usage:
        metrics = OperationalMetrics()
        metrics.record_llm_result("success")
        metrics.record_tactical_result(passed=True)
        metrics.record_trade_close("sl_hit")
        metrics.record_api_retry("matchtrader")
        summary = metrics.get_summary()
    """

    def __init__(self) -> None:
        self._llm_success: int = 0
        self._llm_cancel: int = 0
        self._llm_error: int = 0
        self._tactical_pass: int = 0
        self._tactical_block: int = 0
        self._sl_hits: int = 0
        self._tp_hits: int = 0
        self._manual_closes: int = 0
        self._api_retries: int = 0
        self._matchtrader_retries: int = 0
        self._telegram_retries: int = 0
        self._telegram_failures: int = 0

    def record_llm_result(self, result: str) -> None:
        """Record an LLM decision result.

        Args:
            result: One of 'success', 'cancel', 'error'.
        """
        if result == "success":
            self._llm_success += 1
        elif result == "cancel":
            self._llm_cancel += 1
        elif result == "error":
            self._llm_error += 1
        else:
            logger.warning("OperationalMetrics: unknown LLM result '{}'", result)

    def record_tactical_result(self, *, passed: bool) -> None:
        """Record a tactical gate evaluation result."""
        if passed:
            self._tactical_pass += 1
        else:
            self._tactical_block += 1

    def record_trade_close(self, exit_reason: str) -> None:
        """Record a trade closure by exit reason."""
        if exit_reason == "sl_hit":
            self._sl_hits += 1
        elif exit_reason == "tp_hit":
            self._tp_hits += 1
        else:
            self._manual_closes += 1

    def record_api_retry(self, source: str) -> None:
        """Record an API retry event.

        Args:
            source: 'matchtrader' or 'telegram'.
        """
        self._api_retries += 1
        if source == "matchtrader":
            self._matchtrader_retries += 1
        elif source == "telegram":
            self._telegram_retries += 1

    def record_telegram_failure(self) -> None:
        """Record a Telegram send failure (after all retries)."""
        self._telegram_failures += 1

    def get_summary(self) -> dict[str, int | float]:
        """Return a snapshot of all metrics as a plain dict."""
        llm_total = self._llm_success + self._llm_cancel + self._llm_error
        tactical_total = self._tactical_pass + self._tactical_block
        return {
            "llm_success": self._llm_success,
            "llm_cancel": self._llm_cancel,
            "llm_error": self._llm_error,
            "llm_success_rate": (self._llm_success / llm_total if llm_total > 0 else 0.0),
            "tactical_pass": self._tactical_pass,
            "tactical_block": self._tactical_block,
            "tactical_block_rate": (
                self._tactical_block / tactical_total if tactical_total > 0 else 0.0
            ),
            "sl_hits": self._sl_hits,
            "tp_hits": self._tp_hits,
            "manual_closes": self._manual_closes,
            "api_retries": self._api_retries,
            "matchtrader_retries": self._matchtrader_retries,
            "telegram_retries": self._telegram_retries,
            "telegram_failures": self._telegram_failures,
        }

    def reset(self) -> None:
        """Reset all counters (call at day boundary)."""
        self._llm_success = 0
        self._llm_cancel = 0
        self._llm_error = 0
        self._tactical_pass = 0
        self._tactical_block = 0
        self._sl_hits = 0
        self._tp_hits = 0
        self._manual_closes = 0
        self._api_retries = 0
        self._matchtrader_retries = 0
        self._telegram_retries = 0
        self._telegram_failures = 0
