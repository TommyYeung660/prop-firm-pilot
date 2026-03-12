# Prod Bundle Dropbox And Run-Log Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make production diagnostics bundles more complete for LLM analysis, upload them automatically to Dropbox after packing, add a new unpack script that downloads the latest bundle from Dropbox, and switch runtime logging to a new log file per process start.

**Architecture:** Extend the existing `scripts/pack_prod_logs.py` instead of replacing it. Add a small reusable Dropbox helper module for upload/list/download, emit a machine-readable `bundle_manifest.json` plus config snapshots into the bundle, create a new `scripts/unpack_prod_logs.py` entrypoint, and update `setup_logging()` to derive a run-specific log filename from the configured base path and release tag.

**Tech Stack:** Python 3.10, loguru, pathlib, zipfile, dropbox SDK, pytest, ruff

---

### Task 1: Lock run-based logging behavior in tests

**Files:**
- Create: `tests/test_main_logging.py`
- Modify: `src/main.py`

**Step 1: Write the failing test**

Add tests proving that:

- `setup_logging()` derives a run log path from the configured base file
- the derived path contains a UTC timestamp and release tag
- the derived filename is not the raw `prop_firm_pilot.log`

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_main_logging.py -q`

Expected: FAIL because `setup_logging()` currently writes directly to the fixed path from config.

**Step 3: Write minimal implementation**

- Add a helper in `src/main.py` that converts `logs/prop_firm_pilot.log` into:
  - `logs/prop_firm_pilot_<YYYYMMDD>_<HHMMSS>_<release-tag>.log`
- Keep rotation and retention behavior unchanged

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_main_logging.py -q`

Expected: PASS

---

### Task 2: Add failing packer tests for manifest, config snapshot, and Dropbox path

**Files:**
- Modify: `tests/test_pack_prod_logs.py`
- Modify: `scripts/pack_prod_logs.py`

**Step 1: Write the failing tests**

Add tests proving that:

- packer can build the Dropbox folder path:
  - `/prop-firm-pilot/prod_logs/<account_name>/`
- bundle manifest includes version, config path, account name, and included files
- packer emits config snapshots for:
  - `raw/config/default.yaml`
  - `raw/config/<account-config>.yaml`
  - `raw/config/merged_config.yaml`

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pack_prod_logs.py -k "manifest or config or dropbox" -q`

Expected: FAIL because these helpers and outputs do not exist yet.

**Step 3: Write minimal implementation**

- Add helper functions to:
  - resolve account name
  - build Dropbox artifact path
  - write config snapshots
  - write `bundle_manifest.json`
- Keep existing `INDEX.md` behavior intact

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pack_prod_logs.py -k "manifest or config or dropbox" -q`

Expected: PASS

---

### Task 3: Add Dropbox helper and pack upload flow

**Files:**
- Create: `src/ops/dropbox_artifacts.py`
- Modify: `scripts/pack_prod_logs.py`
- Modify: `tests/test_pack_prod_logs.py`

**Step 1: Write the failing tests**

Add tests proving that:

- packer calls the Dropbox uploader with the expected remote path after zip creation
- upload failure raises an error after local zip already exists
- upload success preserves the local zip and returns the remote path

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pack_prod_logs.py -k "upload" -q`

Expected: FAIL because no Dropbox helper or upload step exists yet.

**Step 3: Write minimal implementation**

- Implement a small Dropbox helper module using the existing `dropbox` dependency
- Support:
  - token refresh from `.env`
  - upload file
  - list folder
  - download file
- In `pack_prod_logs.py`, upload after zip creation
- If upload fails:
  - keep local zip
  - raise a runtime error so script exits non-zero

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pack_prod_logs.py -k "upload" -q`

Expected: PASS

---

### Task 4: Add the new unpack script with overwrite semantics

**Files:**
- Create: `scripts/unpack_prod_logs.py`
- Create: `tests/test_unpack_prod_logs.py`
- Modify: `src/ops/dropbox_artifacts.py`

**Step 1: Write the failing tests**

Add tests proving that:

- unpack selects the latest zip by Dropbox `server_modified`
- if the target extracted folder already exists, it is removed first
- the zip is downloaded to repo root and then extracted

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_unpack_prod_logs.py -q`

Expected: FAIL because the unpack script does not exist yet.

**Step 3: Write minimal implementation**

- Add `scripts/unpack_prod_logs.py`
- Reuse the Dropbox helper for list and download
- Resolve the account folder from config
- Download latest zip to repo root
- Delete existing same-name extracted directory
- Extract zip into repo root

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_unpack_prod_logs.py -q`

Expected: PASS

---

### Task 5: Verify targeted suites and artifact behavior

**Files:**
- Modify: `docs/PropFirmPilot_changelog.md`
- Modify: `scripts/pack_prod_logs.py`
- Modify: `src/main.py`
- Create/Modify: `src/ops/dropbox_artifacts.py`, `scripts/unpack_prod_logs.py`

**Step 1: Run targeted tests**

Run: `uv run pytest tests/test_pack_prod_logs.py tests/test_unpack_prod_logs.py tests/test_main_logging.py tests/test_version.py -q`

Expected: PASS

**Step 2: Run targeted lint**

Run: `uv run ruff check src/main.py src/ops/dropbox_artifacts.py scripts/pack_prod_logs.py scripts/unpack_prod_logs.py tests/test_pack_prod_logs.py tests/test_unpack_prod_logs.py tests/test_main_logging.py`

Expected: PASS

**Step 3: Update changelog**

- Add an entry describing:
  - per-run logging
  - Dropbox upload/download flow
  - richer prod bundle metadata and config snapshots

**Step 4: Summarize residual limits**

Explicitly note:

- bundle still excludes `.env` secrets by design
- Dropbox latest bundle selection uses remote modified time, not semantic version ordering

