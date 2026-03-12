# Roadmap Resequencing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Re-sequence the `v1.4.x` roadmap so tactical entry/exit stabilization becomes `v1.4.6` and `v1.4.7`, while the unfinished broader `v1.4.5` scope moves to `v1.4.8`.

**Architecture:** This is a documentation-only change. Update the formal roadmap narrative, timeline, priority matrix, and milestones, then align the changelog with an `Unreleased` planning section that reflects the new sequence without rewriting shipped history.

**Tech Stack:** Markdown, local git CLI verification, `rg`

---

### Task 1: Rewrite the roadmap sequence

**Files:**
- Modify: `docs/PropFirmPilot_v1.4.0_road_map.md`

**Step 1: Snapshot the current roadmap references**

Run: `rg -n "v1\.4\.5|v1\.4\.6|v1\.4\.7|v1\.4\.8|tactical" docs/PropFirmPilot_v1.4.0_road_map.md`
Expected: existing `v1.4.5` references appear, while `v1.4.6` to `v1.4.8` are missing or incomplete.

**Step 2: Rewrite section 4**

Introduce:

- `v1.4.6` for tactical entry fixes and optimization
- `v1.4.7` for tactical exit fixes and optimization
- `v1.4.8` for the deferred broader `v1.4.5` scope

**Step 3: Rewrite timeline, matrix, and milestones**

Make the version line, ROI matrix, and milestone table consistent with the new sequence.

**Step 4: Review for stale scope assignment**

Run: `rg -n "v1\.4\.5" docs/PropFirmPilot_v1.4.0_road_map.md`
Expected: only intentional historical references remain.

### Task 2: Align the changelog with the new roadmap

**Files:**
- Modify: `docs/PropFirmPilot_changelog.md`

**Step 1: Add an `Unreleased` planning section**

Describe the roadmap resequencing without marking unreleased work as shipped.

**Step 2: Add planned scope bullets**

Document:

- `v1.4.6` tactical entry fixes and optimization
- `v1.4.7` tactical exit fixes and optimization
- `v1.4.8` deferred broader decision-quality scope

**Step 3: Verify the changelog remains historically accurate**

Run: `rg -n "## \[Unreleased\]|v1\.4\.6|v1\.4\.7|v1\.4\.8" docs/PropFirmPilot_changelog.md`
Expected: one `Unreleased` section with the new planned versions.

### Task 3: Verify the final document set

**Files:**
- Modify: `docs/PropFirmPilot_v1.4.0_road_map.md`
- Modify: `docs/PropFirmPilot_changelog.md`
- Create: `docs/plans/2026-03-13-roadmap-resequence-design.md`
- Create: `docs/plans/2026-03-13-roadmap-resequence.md`

**Step 1: Review the diff**

Run: `git diff -- docs/PropFirmPilot_v1.4.0_road_map.md docs/PropFirmPilot_changelog.md docs/plans/2026-03-13-roadmap-resequence-design.md docs/plans/2026-03-13-roadmap-resequence.md`
Expected: only documentation changes and version resequencing edits.

**Step 2: Confirm working tree status**

Run: `git status --short`
Expected: only the intended documentation files are modified or added.
