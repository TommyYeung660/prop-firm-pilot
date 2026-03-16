# PropFirmPilot v1.5.0 — 長期盈利可行性與工作展望報告

> **報告日期**: 2026-03-15
>
> **版本目標**: `v1.5.0`
>
> **報告定位**: 混合證據型、量化研究標準導向、保守審查
>
> **核心問題**: 本系統在 `v1.5.0` 的實現，是否具備成為長期盈利交易系統的必要條件？
>
> **目前結論等級**: `具備局部合理性，但距離長期盈利系統仍有關鍵缺口`
>
> **外部來源代號**: `A*` 與 `O*` 見 `docs/research/v1.5.0_profitability_source_notes.md`

---

## 目錄

### Part A — 報告定位與結論邊界

1. 這份報告要回答什麼
2. 什麼叫長期盈利交易系統
3. 本報告的結論邊界

### Part B — 長期盈利系統的通用判準

4. Alpha 可得性
5. Execution 可實現性
6. Risk / Portfolio 可控性
7. Learning / Adaptation 可持續性
8. Research / Validation 嚴謹性

### Part C — 開源系統與實務對照

9. 為什麼要看開源平台
10. Qlib、LEAN、Freqtrade、FinRL、TradingAgents 給了什麼對照

### Part D — 對本系統 `v1.5.0` 的逐項審查

11. `prop-firm-pilot`
12. `qlib_market_scanner`
13. `TradingAgents`
14. 五維度評分

### Part E — `v1.5.0` 工作展望

15. 必做功能工作包
16. 必做研究工作包
17. 必做驗證工作包

### Part F — 保守結論與 Go / No-Go Gate

18. 最終結論
19. Go / No-Go Gate
20. `v1.5.0 -> v1.5.x` 的實務路線

---

## 1. 這份報告要回答什麼

這份報告不是 marketing 文案，也不是週末寫給自己看的樂觀 roadmap。它要回答的是一個更嚴格、也更直接的問題:

> `v1.5.0` 的實現，是否具備成為長期盈利交易系統的必要條件？

這裡的 **長期盈利交易系統**，不是指某幾天賺錢，也不是指在 prop firm 規則下短期存活，而是指:

- 有合理的 **Alpha（超額報酬來源，扣除成本後仍能重複捕捉的優勢）**
- 這個 alpha 能在真實執行中留下足夠期望值
- 組合層級風險可控，而不只是單筆單子看起來安全
- 系統能跨 **regime（市場狀態，例如趨勢、盤整、高波動）** 存活
- 研究與驗證流程足以支撐保守而可信的結論

---

## 2. 什麼叫長期盈利交易系統

在量化研究標準下，長期盈利從來不是「某個模型很聰明」的同義詞。它更接近以下五件事同時成立:

1. 有可持續、可成本後保留的 alpha。
2. 有足夠好的 execution，避免 **implementation shortfall（理論價格與實際成交價格差造成的績效折損）** 吃掉優勢。
3. 有組合層級的風控與資金配置，而不只是單筆止損。
4. 有記憶、適應與校準能力，但不把噪音誤當學習。
5. 有嚴格的樣本外驗證，能對抗過擬合與 selection bias。

任何一條缺得太多，都不應宣稱是長期盈利系統。

---

## 3. 本報告的結論邊界

這份報告有三個邊界必須先固定:

### 3.1 不把開源平台當成盈利證據

Qlib、LEAN、Freqtrade、FinRL、TradingAgents 都是很有價值的參照物，但它們證明的是「成熟系統通常具備哪些底盤」，不是「用了它們就能穩定賺錢」。

### 3.2 不把本地已有 control plane 誤判成 alpha 證明

本系統目前最強的是 control plane，也就是:

- 市場資料
- tactical entry / exit
- journal
- close reconciliation
- monitoring
- memory hooks

這些都很重要，但它們主要回答的是「能不能可靠運作」，不是「是否已證明能長期盈利」。

### 3.3 不在週末替 live evidence 補空白

今天是 2026-03-15，週日，FX 市場休市。這代表本報告不能拿 fresh live run 充當證據，只能根據:

- 本地三倉庫代碼
- 現有 roadmap / report
- 學術與官方來源

做保守審查。

---

## 4. Alpha 可得性

**判準**: 系統是否有合理機制，持續找出成本後仍有價值的機會。

依 `A4`，機器學習可以從市場資料中抽出非線性訊號，但前提是:

- 樣本外成立
- 特徵不是 data-mined noise
- 成本後仍保留優勢

放到本系統，alpha 來源主要來自 `qlib_market_scanner`，而不是 `TradingAgents`。這是合理的，因為:

- scanner 負責候選排序
- tactical layer 負責進場時機過濾
- LLM 層較適合補充情境風險與例外判斷，不適合承擔主要 alpha 證明

這條線目前的核心問題，不是完全沒有 alpha 假說，而是 **證據成熟度太低**。

---

## 5. Execution 可實現性

**判準**: 系統在真實點差、滑點、延遲、broker 約束下，是否仍能保留正期望。

依 `A5`，execution 不是後處理，而是策略的一部分。理論上正確的信號，若在實盤成交時失真太大，一樣會變負期望。

本系統在這條線上的正面訊號比其他維度更強，因為 `prop-firm-pilot` 已經有:

- pre-trade quote 檢查
- post-trade slippage 檢查
- `exec_latency_ms`、`slippage_pips`、price metadata 落盤
- close control plane 與 broker read-back reconciliation
- write-budget awareness

這代表系統不是只會「下單」，而是開始把 execution integrity 做成正式能力。

但 execution 的證據仍不夠成熟，原因是:

- 目前更像是在證明「可以保護自己」
- 還不是在證明「可以長期穩定保留 alpha」

---

## 6. Risk / Portfolio 可控性

**判準**: 系統是否能從組合層級控制回撤、集中度、相關性與資金利用效率。

依 `A6`，長期盈利一定是 portfolio 問題，不只是單筆問題。這也是本系統目前最明顯的硬缺口之一。

現有能力主要集中在:

- prop-firm 合規
- drawdown protection
- best-day protection
- reduce exposure
- per-trade stop-based sizing

這些保護很有價值，但它們仍偏帳戶安全與單筆控制，不是真正的 portfolio construction。換句話說，系統目前較接近:

- `不要死`

還不是:

- `在可控風險下，把資金有效配置到多筆低相關機會`

---

## 7. Learning / Adaptation 可持續性

**判準**: 系統能否從過去交易中穩定學習，並跨 regime 做自我修正，而不是只是累積更多文本與噪音。

依 `A3`，市場環境會變，所以適應能力是必要條件。但必要不代表自動有效。若記憶層沒有品質控制，自適應很容易退化成 overreaction。

本系統在這條線上的進展很值得肯定:

- 有 `historical_pnl_context`
- 有 `market_event_context`
- 有 `retrieved_trade_lessons`
- 有 reflection hooks
- 有 entry calibration snapshot

但目前仍缺少單一、穩定的 trade-memory contract。這意味著系統雖然已經「開始學」，卻還沒有證明自己「學得乾淨、學得對、學得可重現」。

---

## 8. Research / Validation 嚴謹性

**判準**: 是否具備足夠嚴格的研究流程，能防止過擬合、selection bias 與錯誤歸因。

依 `A1`、`A2`，任何看起來很強的交易系統，只要缺少:

- walk-forward
- out-of-sample
- ablation
- multiple-testing control
- cost modeling

它的結論都不應被高估。

本系統的好消息是，`qlib_market_scanner` 已經不只是隨手回測。它真的有:

- walk-forward 腳本
- cost / turnover sensitivity
- dynamic cost

壞消息則是，這些還沒有被提升成「整體系統 release gate」。目前看到的是局部研究能力，不是 system-wide validation discipline。

---

## 9. 為什麼要看開源平台

開源平台不能替你賺錢，但能暴露一個事實:

> 成熟交易系統通常不是先問「模型多聰明」，而是先把 research-to-live 的契約、成本、保護、審計、回放、配置做穩。

因此本報告看開源平台，不是拿它們背書，而是看它們共同重視什麼。

---

## 10. Qlib、LEAN、Freqtrade、FinRL、TradingAgents 給了什麼對照

### 10.1 Qlib

依 `O1`，Qlib 的價值在於:

- 把量化研究流程做成正式底盤
- 把資料、模型、回測、online serving 放在同一生態

對本系統的啟示是:

- `qlib_market_scanner` 的方向合理
- 但真正要學的是研究紀律，不是只學輸出分數

### 10.2 LEAN

依 `O2`，LEAN 的成熟點在於 research / backtest / live 共用一套 engine 與資料契約。這很重要，因為長期盈利常死在:

- backtest 與 live 的語義不一致
- 研究環境和 production 環境是兩套系統

對本系統的啟示是:

- `v1.5.0` 不能只補 feature，還要補 cross-repo contract 與 validation gate

### 10.3 Freqtrade

依 `O3`，Freqtrade 很清楚地把:

- backtesting
- hyperopt
- protections
- dry-run / live

分成可操作的工作流。對本系統的啟示是:

- 一個能實盤使用的系統，保護與驗證必須產品化，而不是散在多份報告裡

### 10.4 FinRL

依 `O4`，FinRL 提醒我們:

- adaptive / RL 路線可以研究
- 但 benchmark、environment、evaluation protocol 必須先穩

對本系統的啟示是:

- 不要太早把「會學習」誤當「會盈利」

### 10.5 TradingAgents

依 `O5`，TradingAgents 本身就明講是研究用途，而且表現受非確定性因素影響。對本系統的直接結論是:

- LLM layer 可以是 decision overlay
- 但不應該被當成長期盈利可信度的主要證據

---

## 11. `prop-firm-pilot`

### 11.1 目前最強的部分

`prop-firm-pilot` 現在最成熟的，不是 alpha，而是 execution-and-ops core:

- async scheduler
- tactical entry / exit control plane
- close reconciliation
- journal / diagnostics
- account protection
- memory hooks

這很重要，因為一個會崩潰、會寫錯 close reason、會丟 execution metadata 的系統，不值得談盈利。

### 11.2 目前最大的缺口

它最大的缺口也很清楚:

- position sizing 仍以單筆交易為主
- 缺 exposure budget
- 缺 correlation guard
- 缺 capital allocator
- 缺 integrated validation gate

所以它已經很像 production control plane，但還不像完整的 long-term portfolio system。

---

## 12. `qlib_market_scanner`

### 12.1 正面判讀

三個 repo 裡，最接近正式量化研究底盤的是 scanner repo。因為它至少已經有:

- FX profile
- 1D / 1H / 4H path
- dynamic cost
- walk-forward experiment
- cost-turnover sensitivity
- richer signal schema

這些都不是表面工程，而是真正和 research rigor 有關的能力。

### 12.2 保守判讀

但要非常保守地說:

- `qlib_market_scanner` 的內建 FX research baseline 仍只有 4 個 pairs；雖然 `prop-firm-pilot` runtime 已可透過 `--tickers` 覆寫到 7 個 pairs，但研究基線與實盤 universe 仍未完全對齊
- intraday 支援帶有 compatibility mapping
- 目前更像 feasibility / scaffolding
- 還不是 robust cross-regime alpha proof

因此 scanner repo 最合理的評價是:

- **有研究基礎**
- **還沒有盈利級證據**

---

## 13. `TradingAgents`

### 13.1 正面判讀

TradingAgents 對本系統的價值不在於「神奇預測」，而在於:

- 能整合高維上下文
- 能注入 historical / market-event / retrieved lessons
- 能提供比單一 rule engine 更豐富的風險敘事

### 13.2 保守判讀

但它同時也是本系統最難被嚴格驗證的一層:

- README 明講研究用途
- 預設仍是 stock-first
- 很多 fundamentals / options / insider 結構仍在
- prompt / tool / memory 的非確定性高
- 還沒有看到對 intraday FX 的嚴格 latency-budget 與 ablation protocol

所以對長期盈利而言，TradingAgents 現階段比較像:

- 有用的 decision overlay

而不是:

- 可單獨承擔盈利可信度的主引擎

---

## 14. 五維度評分

| 維度 | 系統能力分 | 證據成熟度分 | 核心判讀 |
|---|:---:|:---:|---|
| Alpha 可得性 | 2 | 1 | 有 scanner 研究底盤，但沒有夠強的 FX OOS 證據 |
| Execution 可實現性 | 3 | 2 | control plane 強，live expectancy 證據仍不足 |
| Risk / Portfolio 可控性 | 2 | 1 | 帳戶保護不錯，portfolio construction 明顯不足 |
| Learning / Adaptation 可持續性 | 2 | 1 | learning loop 已閉合，但記憶契約不穩 |
| Research / Validation 嚴謹性 | 2 | 1 | 有局部研究能力，缺整體 release gate |

### 14.1 這組分數代表什麼

這組分數說明的是:

- 本系統不是空的，也不是玩具
- 但它最成熟的是運作控制面
- 距離「可以被保守地稱為長期盈利候選系統」還差兩類東西:
  - 硬能力缺口
  - 驗證成熟度缺口

---

## 15. 必做功能工作包

若 `v1.5.0` 要真正朝長期系統邁進，以下功能不是可選優化，而是必要工作包。

### 15.1 Portfolio / Exposure Control

- 建立 base / quote currency exposure budget
- 建立同向集中暴露上限
- 建立相關性分群與 setup budget
- 讓 sizing 從單筆 stop-based，升級到 portfolio-aware sizing

### 15.2 Cross-Repo Contract Freeze

- 凍結 scanner output schema
- 凍結 agent risk output schema
- 凍結 pilot ingestion / memory payload schema

### 15.3 Trade Memory Quality Gate

- 統一 trade-memory schema
- 區分 raw event、reflection、retrieval 三層
- 不完整或不一致的 close 事件不得直接進長期記憶

---

## 16. 必做研究工作包

### 16.1 FX Alpha Research

- 擴大並凍結 `qlib_market_scanner` 的 FX research baseline，不要只停在內建 4 個 pairs；並與 `prop-firm-pilot` 目前的 7-pair runtime universe 對齊
- 研究 `1d + 1h`、`4h + 1h`、`1h` 等 hybrid cadence
- 重寫更貼近 FX 的 label，而不是沿用 stock-ish 假設
- 補 regime segmentation

### 16.2 LLM Layer Ablation

- 比較:
  - scanner only
  - scanner + tactical
  - scanner + tactical + TradingAgents
- 如果加了 LLM 沒有穩定提升，就不能把它當核心盈利來源

### 16.3 Memory Ablation

- 比較有無 retrieved lessons
- 比較不同 retrieval key:
  - symbol
  - session
  - regime
  - setup
  - outcome type

若 memory 不能穩定改善決策品質，就應降級它在 production 的權重。

---

## 17. 必做驗證工作包

### 17.1 Integrated Walk-Forward

需要的不是只有 scanner repo 的 walk-forward，而是整體系統的 walk-forward:

- scanner ranking
- tactical gating
- cost model
- execution assumption
- portfolio rule

### 17.2 Live-vs-Research Consistency Gate

至少要建立以下對照:

- research 預估 spread vs live observed spread
- research 預估 hold profile vs live hold profile
- expected entry/exit logic vs actual journal trail

### 17.3 Selection-Bias Control

依 `A1`、`A2`，在 `v1.5.0` 之後的正式研究報告中，應加入類似以下 gate:

- 多次試驗紀錄
- 明確的 model / config registry
- 保守 Sharpe 解讀
- 對 strategy family 的 overfitting 風險審查

---

## 18. 最終結論

### 18.1 直接答案

如果問題是:

> 本系統在 `v1.5.0` 能不能被保守地視為長期盈利交易系統？

我的答案是:

> **目前不能。**

更準確的結論等級是:

> **具備局部合理性，但距離長期盈利系統仍有關鍵缺口。**

### 18.2 為什麼不是更差

因為本系統已經具備以下真實底盤:

- execution integrity
- tactical control plane
- audit trail
- learning hooks
- scanner research scaffold

這些都讓它比一般「策略 + 下單腳本」更接近可持續演進的系統。

### 18.3 為什麼不能更高

因為以下條件仍未同時成立:

- 成本後 alpha 的整體 OOS 證據
- portfolio-level 資金配置能力
- LLM layer 的嚴格可驗證穩定性
- trade-memory quality gate
- integrated research-to-live validation gate

---

## 19. Go / No-Go Gate

### 19.1 No-Go

以下任一項未完成前，不應宣稱具備長期盈利能力，也不應擴大實盤資金規模:

1. 沒有 integrated walk-forward
2. 沒有 exposure / correlation budget
3. 沒有 LLM ablation
4. 沒有 trade-memory quality gate
5. 沒有 live-vs-research consistency review

### 19.2 Go to Small-Scale Controlled Live

若要把 `v1.5.0` 當成小規模受控 live candidate，至少應達成:

- scanner cadence 研究完成並固定一版
- close / entry journal schema 穩定
- portfolio exposure guard 上線
- 連續一段受控 live run 沒有重複 P0 / P1 correctness incident
- live slippage / latency / rejection stats 可追蹤

### 19.3 Go to Broader Capital

若未來要從 prop-firm 過渡到更一般的實盤使用，還要再加:

- 更長期 live sample
- 更寬鬆市場條件下的容量檢查
- broker / venue 差異驗證
- 更完整的 portfolio construction 與 capital allocation

---

## 20. `v1.5.0 -> v1.5.x` 的實務路線

最務實的路線不是急著宣稱成功，而是:

1. 把 `v1.5.0` 做成第一個真正有 research gate 的 stable milestone
2. 讓 `v1.5.x` 專注在 bugfix、微調與 validation accumulation
3. 等 integrated evidence 成熟後，再決定是否把結論上調到:
   - `具備成為長期盈利候選系統的條件，但仍缺關鍵驗證`

---

## 總結

本系統最值得肯定的，不是它現在已經能被稱為長期盈利，而是它已經長成一個 **有機會被嚴格驗證** 的交易系統雛形。

這兩者差很多。

一個只有故事的系統，不值得研究。

一個只有 control plane、沒有 alpha 的系統，也不值得擴張。

而本系統目前所處的位置，是兩者之間:

- 它已有足夠多的工程底盤，值得繼續做 `v1.5.0`
- 但它還沒有足夠多的證據，值得宣稱自己已能長期盈利

因此，`v1.5.0` 最合理的目標，不是「證明成功」，而是:

> **把本系統提升為一個可以被保守、可重現、可審計地驗證的長期盈利候選系統。**
