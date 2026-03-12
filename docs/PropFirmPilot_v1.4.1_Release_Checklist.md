# PropFirmPilot v1.4.1 — Final Release Checklist

> **Checklist Date**: 2026-03-12
> **Release Target**: `v1.4.1`
> **Status**: Ready for release candidate
> **Note**: Engineering scope is closed in this worktree. Git commit, tag, deploy, and post-deploy bundle capture were not executed here.

---

## 1. Release Identity

- [x] `pyproject.toml` version is `1.4.1`
- [x] `src/version.py` is the single shared version source
- [x] runtime startup path uses shared version helper instead of hardcoded release text
- [x] `scripts/pack_prod_logs.py` defaults to shared release tag and fail-fast on explicit `--version` drift

## 2. Scope Closure

### P0 — Learning Data Correctness

- [x] `MemoryJournal` decision/result anchoring uses `intent_id`
- [x] trade result append requires `intent_id + position_id + symbol`
- [x] decision/result identity mismatch is rejected instead of silently writing cross-symbol outcome data
- [x] regression coverage added for cross-symbol / identity mismatch paths

### P1 — Reliability / Observability Hardening

- [x] runtime / packer / release docs now converge on one version source
- [x] `TelegramBotHandler` writes polling failure, circuit-open, probe, and recovery transitions into shared `OperationalMetrics`
- [x] `EODHDFXWebSocketClient.get_status()` exposes `healthy / degraded / disconnected`
- [x] `MarketDataHub` degraded path reuses REST-backed cache first and only refreshes incrementally when cache is stale
- [x] `Scheduler` `METRICS_SNAPSHOT` includes market-data feed status

### P2 — Diagnostics / Bundle Hardening

- [x] cooldown paths emit structured `SCANNER_SKIP` events for `recent_rejection_cooldown` and `low_confidence_cooldown`
- [x] `scripts/pack_prod_logs.py` generates deterministic `decisions_summary.md` fallback
- [x] `scripts/pack_prod_logs.py` generates deterministic `telegram_summary.md` fallback
- [x] `INDEX.md` only lists summary files that actually exist in the bundle

## 3. Release Artifacts

- [x] Incident report: `docs/PropFirmPilot_v1.4.1_Incident_Report.md`
- [x] Changelog entry: `docs/PropFirmPilot_changelog.md`
- [x] P0 plan: `docs/plans/2026-03-12-v1.4.1-p0-memory-identity.md`
- [x] P1 plan: `docs/plans/2026-03-12-v1.4.1-p1-reliability-hardening.md`
- [x] P2 plan: `docs/plans/2026-03-12-v1.4.1-p2-diagnostics-bundle-hardening.md`

## 4. Fresh Verification Evidence

Verified on `2026-03-12` with fresh command output from this worktree:

- [x] Targeted release regression suite:

```bash
uv run pytest tests/test_version.py tests/test_operational_metrics.py tests/test_alert_service.py tests/data/test_fx_websocket_client.py tests/data/test_market_data_hub.py tests/test_memory_journal.py tests/test_scheduler.py tests/test_volatility_monitor.py tests/test_pack_prod_logs.py tests/monitor/test_trade_journal.py -q
```

Result: `232 passed in 56.41s`

- [x] Changed-scope lint:

```bash
uv run ruff check src/version.py src/main.py src/monitor/memory_journal.py src/monitor/operational_metrics.py src/monitor/telegram_bot.py src/data/fx_websocket_client.py src/data/market_data_hub.py src/scheduler/scheduler.py scripts/pack_prod_logs.py tests/test_version.py tests/test_pack_prod_logs.py tests/test_operational_metrics.py tests/test_alert_service.py tests/data/test_fx_websocket_client.py tests/data/test_market_data_hub.py tests/test_memory_journal.py tests/test_scheduler.py tests/test_volatility_monitor.py tests/monitor/test_trade_journal.py
```

Result: `All checks passed!`

- [x] Shared version resolution sanity check:

```bash
@'
from src.version import get_app_version, get_release_tag
from scripts.pack_prod_logs import _resolve_version
print(get_app_version())
print(get_release_tag())
print(_resolve_version(None))
'@ | uv run python -
```

Result:

```text
1.4.1
v1.4.1
v1.4.1
```

## 5. Operational Acceptance

- [x] postmortem can distinguish websocket feed state from `METRICS_SNAPSHOT`
- [x] Telegram polling degradation is no longer log-only; it is visible in shared operational metrics
- [x] cooldown / cancel-loop diagnosis can be replayed from structured `SCANNER_SKIP` events
- [x] bundle generation no longer depends on LLM summaries to produce usable `decisions` and `telegram` summaries
- [x] `v1.4.1` remains a reliability / observability hardening release, not a new strategy or risk-policy expansion

## 6. Manual Release Ops Not Executed Here

- [ ] Review worktree diff and stage release commit
- [ ] Create git commit for `v1.4.1`
- [ ] Create release tag `v1.4.1`
- [ ] Deploy target runtime
- [ ] Capture first post-deploy startup log confirming `v1.4.1`
- [ ] Generate fresh production bundle after deployment for release audit

## 7. Release Decision

Engineering acceptance for `v1.4.1` is complete in this worktree. The release is ready to cut as a release candidate once the remaining git / deployment steps are approved and executed.
