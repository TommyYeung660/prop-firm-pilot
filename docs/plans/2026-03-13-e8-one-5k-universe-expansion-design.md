# E8 One 5K Universe Expansion Design

> **日期**: 2026-03-13
> **範圍**: `config/e8_one_5k_challenge.yaml`
> **目標**: 擴大 E8 One 5K 的 major FX universe，提升候選分散度與資金利用率，同時維持既有風險邊界不變

---

## 背景

目前 `e8_one_5k_challenge.yaml` 已配置 4 個品種：

- `EURUSD`
- `GBPUSD`
- `USDJPY`
- `AUDUSD`

在現時配置下，使用者觀察到大部分情況仍只會交易單一貨幣對，代表雖然 `max_positions=5`，但可交易 universe 對 scanner / downstream decision flow 仍偏窄，導致候選集中與資金利用率偏低。

---

## 設計目標

- 將 E8 One 5K universe 擴充為 7 個 major pairs：
  - `EURUSD`
  - `GBPUSD`
  - `USDJPY`
  - `AUDUSD`
  - `NZDUSD`
  - `USDCAD`
  - `USDCHF`
- 讓 websocket 觀測層與 execution/tactical 層使用同一份 symbols universe
- 在不提高單筆風險、不放寬 tactical gate、不增加 `max_positions` 的前提下，提高同日候選覆蓋面
- 補上 config-level regression assertions，避免之後被其他配置調整默默收窄回 4 個品種

---

## 非目標

- 不改動 runtime 邏輯
- 不改動 optimization threshold / confidence gate
- 不引入 cross pairs
- 不調整 correlation / portfolio guard
- 不修改其他 account config

---

## 方案比較

### 方案 A：擴到 7 個 majors

新增 `NZDUSD`、`USDCAD`、`USDCHF`，並同步補齊 `websocket.symbols` 與 `instruments`。

優點：

- 改動最小
- 仍以主流美元對為主，流動性與 spread 假設較穩定
- 在沒有完整 correlation guard 前，比含 crosses 的方案更保守

缺點：

- 候選分散度提升有限於美元主導 universe

### 方案 B：擴到 10-12 個，含 cross pairs

在 majors 之外加入 `EURJPY`、`GBPJPY`、`EURGBP`、`AUDJPY` 等。

優點：

- 候選數與分散度最高

缺點：

- 在尚未完成 correlation / portfolio guard 前，風險共振更明顯
- 需要更多 instrument spread / pip assumptions

### 方案 C：只擴 symbols，不調整 scanner top-k

優點：

- 改動更少

缺點：

- 預設 `scanner.topk=3` 仍會限制每輪最多 3 個候選
- 即使 universe 增大，資金利用率改善可能有限

---

## 選定方案

採用 **方案 A**，並一併在 E8 One 5K YAML 覆寫：

- `symbols`
- `websocket.symbols`
- `instruments`
- `scanner.topk: 5`

理由：

- 對使用者問題的直接修復是「擴大可交易 universe」
- `scanner.topk` 若不一起提高，新增 symbols 的效益會被上游候選數截斷
- `max_positions` 已經是 `5`，因此將 `scanner.topk` 提高到 `5` 與現有容量一致，不額外提高 account 風險上限

---

## 具體變更

在 `config/e8_one_5k_challenge.yaml`：

- `symbols` 由 4 個擴充為 7 個 majors
- `websocket.symbols` 同步擴充為相同 7 個 majors
- 新增 `scanner.topk: 5`
- 在 `instruments` 補上：
  - `NZDUSD`
  - `USDCAD`
  - `USDCHF`

新增 instrument 參數時，遵守現有 YAML 結構：

- `pip_size`
- `pip_value`
- `avg_spread_pips`
- `min_lot`
- `max_lot`

---

## 風險與限制

- 這次變更只擴充 candidate universe，不保證每輪都會產生多品種交易
- 若 optimization thresholds 仍明顯偏向單一 symbol，後續仍可能看到集中現象
- 因尚未上線 correlation guard，7 個 majors 是在可接受風險下的保守擴張；暫不納入 crosses

---

## 驗證方式

- 更新 config regression test，確認：
  - `symbols` 為 7 個 majors
  - `websocket.symbols` 為相同 7 個 majors
  - `scanner.topk == 5`
  - 新增 instrument keys 存在
- 執行 targeted pytest 驗證 config 載入

