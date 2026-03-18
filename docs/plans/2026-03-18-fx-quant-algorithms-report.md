# FX 量化分析算法研究報告 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 產出一份繁體中文研究型 Markdown 報告，總結現時 FX 量化分析常見算法與其有效性，並用簡單易明的方式說明其適用條件與常見失效原因。

**Architecture:** 先用官方市場結構材料確認 FX 市場現況，再用近年學術與開放取用論文整理主流算法類別，最後把結果壓縮成一份可快速閱讀的研究摘要。結論以「成本後、樣本外、regime 敏感度、資料門檻」四個維度統一比較。

**Tech Stack:** Markdown, web research, BIS, IMF, RBA, academic journals, arXiv, `git diff --check`

---

### Task 1: 固定研究框架

**Files:**
- Create: `docs/plans/2026-03-18-fx-quant-algorithms-design.md`
- Create: `docs/plans/2026-03-18-fx-quant-algorithms-report.md`

**Step 1: 鎖定報告定位**

- 研究導向
- 繁體中文
- 簡單易明
- 不做獲利承諾

**Step 2: 鎖定評估維度**

- 樣本外預測力
- 成本後可行性
- regime 穩定性
- 可解釋性
- 資料與基建門檻

**Step 3: 保存設計文件**

將上述定位與結構寫入 `docs/plans/2026-03-18-fx-quant-algorithms-design.md`。

### Task 2: 蒐集高品質來源

**Files:**
- Create: `docs/research/2026-03-18_fx_quant_algorithms_summary_zh-TW.md`

**Step 1: 蒐集官方市場結構來源**

至少使用：

- BIS 2025 FX turnover / market structure
- BIS execution algorithms report
- IMF 或 RBA 關於匯率可預測性的近期研究

**Step 2: 蒐集算法與有效性來源**

至少覆蓋：

- 技術分析 / 趨勢
- 因子 / carry / momentum / value
- 機器學習
- 深度學習
- 強化學習
- 新聞或 LLM 融合
- order flow / LOB

**Step 3: 只保留可支撐結論的來源**

避免把低品質回測文章或行銷材料當成主證據。

### Task 3: 撰寫正式報告

**Files:**
- Create: `docs/research/2026-03-18_fx_quant_algorithms_summary_zh-TW.md`

**Step 1: 寫市場背景**

先解釋 FX 市場為何難做，以及為何「可預測」不等於「可交易」。

**Step 2: 寫算法總覽表**

用表格壓縮各類方法的核心概念、資料需求與目前有效性判斷。

**Step 3: 寫分節評估**

逐類說明：

- 它是什麼
- 為什麼有人用
- 目前證據怎樣
- 主要失效點是什麼

**Step 4: 寫結論**

回答：

- 目前哪些方法最有研究價值
- 哪些方法最容易被高估
- 讀者應如何看待論文中的「有效」

### Task 4: 基本驗證

**Files:**
- Modify: `docs/research/2026-03-18_fx_quant_algorithms_summary_zh-TW.md`

**Step 1: 檢查 Markdown**

Run:

```bash
git diff --check -- docs/plans/2026-03-18-fx-quant-algorithms-design.md docs/plans/2026-03-18-fx-quant-algorithms-report.md docs/research/2026-03-18_fx_quant_algorithms_summary_zh-TW.md
```

Expected: no whitespace or conflict-marker errors

**Step 2: 檢查工作樹**

Run:

```bash
git status --short
```

Expected: only intended docs plus any pre-existing unrelated changes
