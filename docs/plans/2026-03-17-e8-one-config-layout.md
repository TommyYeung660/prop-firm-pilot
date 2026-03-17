# E8 One Config Layout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在同一份 `config/e8_one_5k_challenge.yaml` 內，把內容整理為高頻常調與低頻基礎兩大區塊，並以等價驗證保證所有配置值不變。

**Architecture:** 這次不改任何 Python 程式與 schema，只重排 YAML 中的 top-level section 順序與註解。為避免誤動數值，實作前先保存目前 YAML 解析結果，實作後再重新解析並比較兩者是否完全相等。

**Tech Stack:** YAML, Python 3.10, Pydantic config loader

---

### Task 1: Capture baseline config structure

**Files:**
- Modify: `config/e8_one_5k_challenge.yaml`
- Create: temporary baseline snapshot outside repo

**Step 1: Save parsed baseline**

Run:

```bash
uv run python -c "import json, yaml, pathlib, os; data=yaml.safe_load(pathlib.Path('config/e8_one_5k_challenge.yaml').read_text(encoding='utf-8')); pathlib.Path(os.environ['TEMP']).joinpath('e8_one_5k_challenge_baseline.json').write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2), encoding='utf-8')"
```

**Step 2: Confirm snapshot exists**

Run:

```bash
Get-Item "$env:TEMP\\e8_one_5k_challenge_baseline.json"
```

**Step 3: Commit**

No commit in this task; snapshot is temporary.

### Task 2: Reorganize YAML into two sections

**Files:**
- Modify: `config/e8_one_5k_challenge.yaml`

**Step 1: Rewrite top-level layout**

- Keep `account_name` at top as metadata.
- Move high-frequency sections next:
  - `symbols`
  - `scanner`
  - `execution`
  - `websocket`
  - `scheduler`
  - `tactical`
- Move low-frequency sections after that:
  - `account`
  - `compliance`
  - `decision_store`
  - `monitor`
  - `optimization`
  - `agents`
  - `instruments`

**Step 2: Preserve comments while improving section labels**

- Add explicit headers for:
  - `高頻常調參數`
  - `低頻基礎參數`
- Keep field-level Chinese annotations.

**Step 3: Commit**

```bash
git add config/e8_one_5k_challenge.yaml
git commit -m "docs: reorganize e8 one config for tuning workflow"
```

### Task 3: Verify structural equivalence

**Files:**
- Modify: `config/e8_one_5k_challenge.yaml`

**Step 1: Load config through application loader**

Run:

```bash
uv run python -c "from src.config import load_config; cfg = load_config('config/e8_one_5k_challenge.yaml'); print(cfg.account.initial_balance); print(cfg.scheduler.quiet_session_interval_seconds); print(len(cfg.symbols))"
```

Expected:
- YAML loads successfully
- Current tuned values remain intact

**Step 2: Compare parsed YAML with baseline snapshot**

Run:

```bash
uv run python -c "import json, yaml, pathlib, os, sys; current=yaml.safe_load(pathlib.Path('config/e8_one_5k_challenge.yaml').read_text(encoding='utf-8')); baseline=json.loads(pathlib.Path(os.environ['TEMP']).joinpath('e8_one_5k_challenge_baseline.json').read_text(encoding='utf-8')); print(current == baseline); sys.exit(0 if current == baseline else 1)"
```

Expected: `True`

**Step 3: Review diff**

Run:

```bash
git diff -- config/e8_one_5k_challenge.yaml
```

**Step 4: Commit**

```bash
git add config/e8_one_5k_challenge.yaml
git commit -m "docs: split e8 one config into tuning and baseline sections"
```
