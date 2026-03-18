# `v1.5.0` Roadmap Alignment Note Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 產出一頁繁體中文筆記，交叉比對 `v1.5.0` roadmap、FX 算法 summary 與目前實作狀態，回答方向與實現程度是否一致。

**Architecture:** 先抽出 summary 的核心主張，再對照 roadmap 的版本策略與程式中的實作證據，最後把結果壓縮成「已對齊 / 未對齊 / 下一步」三段。文件重點是判斷與證據，不是重寫 roadmap。

**Tech Stack:** Markdown, local repo docs, source inspection, `git diff --check`

---

### Task 1: 固定對齊框架

**Files:**
- Create: `docs/plans/2026-03-18-roadmap-alignment-vs-fx-summary-design.md`
- Create: `docs/plans/2026-03-18-roadmap-alignment-vs-fx-summary.md`

**Step 1: 凍結比較對象**

- `docs/PropFirmPilot_v1.5.0_road_map.md`
- `docs/research/2026-03-18_fx_quant_algorithms_summary_zh-TW.md`
- relevant implementation evidence in `src/`

**Step 2: 凍結輸出形狀**

- 一句話結論
- 已對齊
- 未對齊
- 對齊分數
- 下一步

**Step 3: 保存設計與計畫**

將框架寫入 `docs/plans/2026-03-18-roadmap-alignment-vs-fx-summary-design.md`
與 `docs/plans/2026-03-18-roadmap-alignment-vs-fx-summary.md`。

### Task 2: 撰寫正式筆記

**Files:**
- Create: `docs/research/2026-03-18_v1.5.0_roadmap_alignment_vs_fx_summary_zh-TW.md`

**Step 1: 寫一句話結論**

明確回答是否對齊，避免模糊措辭。

**Step 2: 寫已對齊段落**

聚焦：

- validation-first
- contract-first
- cost-aware baseline
- LLM bounded role

**Step 3: 寫未對齊段落**

聚焦：

- exposure / aggregate open-risk guard
- memory quality gate
- live-vs-research consistency
- 尚未明確收斂成 simple-ML / factor-first productization

**Step 4: 寫下一步**

用 3-5 條行動項，對應 roadmap 中最該補的缺口。

### Task 3: 基本驗證

**Files:**
- Modify: `docs/research/2026-03-18_v1.5.0_roadmap_alignment_vs_fx_summary_zh-TW.md`

**Step 1: 檢查 Markdown**

Run:

```bash
git diff --check -- docs/plans/2026-03-18-roadmap-alignment-vs-fx-summary-design.md docs/plans/2026-03-18-roadmap-alignment-vs-fx-summary.md docs/research/2026-03-18_v1.5.0_roadmap_alignment_vs_fx_summary_zh-TW.md
```

Expected: no whitespace or conflict-marker errors

**Step 2: 檢查工作樹**

Run:

```bash
git status --short
```

Expected: only intended research-note docs plus any unrelated pre-existing changes
