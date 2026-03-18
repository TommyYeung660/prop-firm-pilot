# `v1.5.0` Roadmap 與 FX 算法 Summary 對齊分析設計

**日期:** `2026-03-18`

**Goal:** 將「`v1.5.0` roadmap 是否對齊 FX 量化算法 summary」的判斷整理成一頁繁體中文研究筆記，供後續 roadmap 討論與版本優先級校準使用。

**Positioning:** 這不是新 roadmap，也不是 marketing 摘要，而是一份交叉比對筆記。它要回答的不是「系統好不好」，而是「目前方向是否符合研究上較合理的 FX 量化方法論」。

---

## 1. 文件範圍

- 對照文件：
  - `docs/PropFirmPilot_v1.5.0_road_map.md`
  - `docs/research/2026-03-18_fx_quant_algorithms_summary_zh-TW.md`
  - `README.md`
- 補充證據：
  - `src/signal/scanner_bridge.py`
  - `src/scheduler/scheduler.py`
  - `src/execution/capital_allocator.py`
  - `src/execution/engine.py`
  - `src/decision/agent_bridge.py`

---

## 2. 核心判斷方式

用三層來判斷：

1. `方向是否對齊`
   也就是 roadmap 的方法論，是否符合 summary 對現時 FX 量化的保守看法。
2. `實作是否已落地`
   也就是 roadmap 說要做的事情，有多少已經變成程式與契約。
3. `缺口是否被正確認知`
   也就是尚未完成的部分，有沒有被 roadmap 誤判成已完成。

---

## 3. 正式筆記結構

1. 一句話結論
2. 已對齊的地方
3. 部分對齊或未對齊的地方
4. 對齊分數
5. 建議下一步

---

## 4. 寫作原則

- 只寫高訊號結論
- 用白話描述，但保留技術準確性
- 每個主要判斷都附本 repo 內證據點
- 不重新展開整份 roadmap
