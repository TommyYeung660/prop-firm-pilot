# E8 Signature Risk Cap Adjustment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 將 `e8_signature_50k_challenge` 的 scanner confidence uplift 風險上限由 1.0% 下修到 0.75%，降低 retuned confidence 分佈導致的稀疏部位平均曝險擴張。

**Architecture:** 這次採 config-only 方案，不修改 scheduler、execution 或 diagnostics 邏輯。先收斂帳號級 `max_risk_pct`，再用既有 `analyze_scanner_confidence_impact` harness 驗證新設定下的 exposure context。

**Tech Stack:** YAML config、Python CLI diagnostics、pytest/uv

---

### Task 1: 調整帳號風險上限

**Files:**
- Modify: `config/e8_signature_50k_challenge.yaml`

**Step 1: 修改配置**

把 `execution.max_risk_pct` 從 `0.01` 改為 `0.0075`，並更新註解說明這是保守 uplift cap。

**Step 2: 檢查 diff**

Run: `git diff -- config/e8_signature_50k_challenge.yaml`
Expected: 只看到 `max_risk_pct` 與相鄰註解的最小變更

### Task 2: 重跑 impact harness

**Files:**
- Modify: `config/e8_signature_50k_challenge.yaml`
- Use: `src/diagnostics/analyze_scanner_confidence_impact.py`

**Step 1: 執行診斷**

Run: `uv run python -m src.diagnostics.analyze_scanner_confidence_impact --baseline-candidates ..\..\..\..\qlib_market_scanner\.worktrees\fx-confidence-calibration-study\tmp\formal_fx_calibration_run\runtime_from_existing_pred\signals\alpha_candidates.csv --retuned-candidates ..\..\..\..\qlib_market_scanner\.worktrees\fx-confidence-calibration-study\tmp\formal_fx_calibration_run\runtime_from_existing_pred_retuned\signals\alpha_candidates.csv --config config\e8_signature_50k_challenge.yaml --format json`
Expected: `analysis_context.max_risk_pct` 反映為 `0.0075`

**Step 2: 回報結果**

整理新的風險上下文與仍未改變的 live-impact 特性，明確標示這次只收斂 capital uplift 上限，沒有更動候選集或信心分桶。
