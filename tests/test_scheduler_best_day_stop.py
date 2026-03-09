"""Test that scanner loop stops retrying after Best Day pause is triggered."""

import inspect


def test_scheduler_has_best_day_pause_attribute() -> None:
    """Scheduler should have a _best_day_paused_today attribute."""
    from src.scheduler.scheduler import Scheduler

    source = inspect.getsource(Scheduler.__init__)
    assert "_best_day_paused_today" in source, (
        "Scheduler.__init__ should initialize _best_day_paused_today"
    )


def test_best_day_pause_flag_resets_on_rollover() -> None:
    """_maybe_rollover_best_day_tracker should reset _best_day_paused_today."""
    from src.scheduler.scheduler import Scheduler

    source = inspect.getsource(Scheduler._maybe_rollover_best_day_tracker)
    assert "_best_day_paused_today" in source, (
        "_maybe_rollover_best_day_tracker should reset _best_day_paused_today"
    )


def test_scanner_loop_checks_pause_flag_before_should_pause() -> None:
    """Scanner loop should check the daily flag BEFORE calling _should_pause_new_entries."""
    from src.scheduler.scheduler import Scheduler

    source = inspect.getsource(Scheduler._scanner_loop)
    flag_pos = source.index("_best_day_paused_today")
    pause_pos = source.index("_should_pause_new_entries")
    assert flag_pos < pause_pos, "Daily flag check should come BEFORE _should_pause_new_entries"
