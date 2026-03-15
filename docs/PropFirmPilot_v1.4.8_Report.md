# PropFirmPilot v1.4.8 — Close Control Plane 細顆粒情景報告

> **報告日期**: 2026-03-15  
> **版本**: v1.4.8（Close Control Plane）  
> **基準版本**: v1.4.7  
> **聚焦範圍**: 以 `prop-firm-pilot` 的 close control plane 為主，說明 tactical exit、drawdown de-risk、emergency close、LLM re-eval、Best Day Rule、external close reconciliation，如何與 `qlib_market_scanner`、`TradingAgents`、journal / memory / scheduler 串成完整交易閉環

---

## 目錄

### Part A — 報告定位

1. 報告目的
2. 為什麼這不是 live 驗證報告
3. 三倉庫角色分工

### Part B — Close Control Plane 標準主流程

4. v1.4.8 標準 close 主流程
5. close-domain 的三層主語義
6. `submitted`、`verified`、`pending_reconcile` 與 `final_close_reason`

### Part C — 十二個細顆粒交易情景

7. 情景 1：EURUSD 倫敦盤突破多單，`+1.0R` 先 partial close，餘倉 trail 後 TP
8. 情景 2：GBPUSD 紐約盤回撤空單，先 move to breakeven，回抽後打到保本止損
9. 情景 3：USDJPY 趨勢延續多單，trail 連續上移兩次，最後保利出場
10. 情景 4：AUDUSD 震盪盤多單，reprice TP 後縮短持倉時間獲利出場
11. 情景 5：NZDUSD 反轉風險升高，tactical 直接 full close
12. 情景 6：EURUSD 高波動回吐，modify 成功但 verify mismatch，最後仍被原始 SL 打掉
13. 情景 7：GBPUSD drawdown 進入 DANGER，系統做 reduce exposure partial close
14. 情景 8：帳戶進入 CRITICAL，emergency close 同時處理兩筆 USDCHF 倉位
15. 情景 9：EURUSD 持倉期間 LLM re-eval 翻向，觸發 reeval close
16. 情景 10：GBPJPY 觸發 Best Day Rule 保護，主動鎖定日內浮盈
17. 情景 11：AUDUSD 出現 broker-side / manual close，系統以 external detected close 對賬
18. 情景 12：USDCAD 外部關倉且 broker closed row 延遲，靠 execution meta 與 last-known profit 完成最終收斂

### Part D — 可持續學習 / Audit Trail / 後續基線

19. v1.4.8 對交易流程的真正影響
20. control plane 的可持續學習性
21. audit trail：一筆單如何被完整回放
22. 對 v1.4.9 與 v1.5.0 的邏輯基線
23. 術語速覽

---

## 1. 報告目的

這份報告不是 changelog，也不是單純把 `CloseControlPlane`、`CloseReconciler`、`Scheduler` 的程式碼逐段翻譯。它的目的，是把 `v1.4.8` 的 **Close Control Plane（平倉控制平面，指所有平倉動作共用的一套調度、驗證、對賬與審計層）** 用 operator 能直接消化的方式講清楚。

更準確地說，這份報告要回答四個問題：

1. `qlib_market_scanner`、`TradingAgents`、`prop-firm-pilot` 在 close-domain 各自負責什麼。
2. tactical exit、drawdown de-risk、emergency close、Best Day Rule、LLM re-eval、external close，現在是否都進了同一條 close execution 管線。
3. `trigger_source`、`action_kind`、`final_close_reason` 為什麼必須拆開來看，而不能再用單一 `exit_reason` 混寫。
4. 為什麼 `v1.4.8` 不是只讓系統「更會平倉」，而是讓整個 control plane 更容易學習、回放、修補。

因此，下面的 12 個情景不是虛構故事，而是依照目前 `main` 上 `v1.4.8` 的實際語義，整理出的 **高真實度規則型運營情景**。它們說明的是系統現在「會怎麼走」，不是週一市場打開後「已經驗證過一定如此」。

---

## 2. 為什麼這不是 live 驗證報告

2026-03-15 是週日，FX 市場休市。這表示：

- 目前不能用即時市場去驗證 `v1.4.8` 的 runtime 行為
- 也不應把這份文件寫成「已經在週一實盤證明有效」的語氣

所以本報告的定位，是 **rule-driven operational baseline report（規則驅動的運營基線報告）**。它依據的是：

- 已落地的 `CloseIntent` / `CloseOutcome` / `CloseReconciliation` 資料模型
- `CloseControlPlane.execute()` 對 `modify_only`、`partial_close`、`full_close` 的實際返回語義
- `CloseReconciler.reconcile()` 對 `trigger_source`、`action_kind`、`final_close_reason`、`resolution_path` 的實際決定順序
- `Scheduler` 目前已寫出的 `CLOSE_CONTROL_EVENT` 與 `TRADE_CLOSED` payload

這種報告有一個很重要的用途：在週一 2026-03-16 實盤前，先把大家對系統行為的預期固定下來。等市場重開後，operator 可以直接拿 journal 去對照：

- 哪些情景真的發生了
- 哪些欄位與報告一致
- 哪些地方是 `v1.4.9` 必須補的 bug 或微調

換句話說，這份文件不是要替代 live evidence，而是要先定義 **我們預期 close control plane 應該長成什麼樣**。

---

## 3. 三倉庫角色分工

### 3.1 `qlib_market_scanner`

`qlib_market_scanner` 的角色是 **候選掃描器**。它負責回答：

- 哪些 symbol 在當下值得送進 decision pipeline
- 哪些標的具備 trend、breakout、pullback、range reversion 之類的研究價值

它不負責回答：

- 最終是 `BUY`、`SELL` 還是 `HOLD`
- 倉位建立後要不要移動止損
- 倉位應該如何被關掉

### 3.2 `TradingAgents`

`TradingAgents` 是 **多代理 LLM 決策模組**，主要負責把 scanner 候選轉成策略方向與風險敘事。它回答的是：

- 這筆候選現在該不該做方向下注
- 若做，是 `BUY`、`SELL`，還是 `HOLD`
- 當前 thesis 的風險來源是什麼

它不直接碰 broker write，也不決定 close-domain 的 canonical reason。

### 3.3 `prop-firm-pilot` 的 close control plane

`prop-firm-pilot` 在 `v1.4.8` 之後，負責的是 **close-domain 的統一操作與對賬**：

- tactical exit 需要 `MOVE_TO_BREAKEVEN`、`TRAIL_SL`、`REPRICE_TP`、`PARTIAL_CLOSE`、`EXIT_NOW` 時，會先被映射成 `CloseIntent（平倉意圖，系統要送出的結構化 close 命令）`
- `CloseControlPlane.execute()` 負責把這些命令轉成 `CloseOutcome（執行結果，描述 broker write 與回讀狀態）`
- position monitor 偵測到部位消失後，再由 `CloseReconciler（平倉對賬器，負責把 broker 事實與 fallback 資料收斂成單一最終結論）` 產生 canonical `final_close_reason`

這個分工意味著：

- scanner 負責找機會
- `TradingAgents` 負責判斷要不要下注
- close control plane 負責把已建立的風險敘事，真正轉成一套可回放的 close lifecycle

---

## 4. v1.4.8 標準 close 主流程

標準主流程如下：

1. `qlib_market_scanner` 產生候選，提供 symbol 與 setup 類型。
2. `TradingAgents` 經過多代理討論後，輸出 `BUY / SELL / HOLD` 與風險敘事。
3. 若為 `BUY / SELL`，系統先經過 `v1.4.7` 的 entry control plane，確認 timing、spread、freshness、volatility 等條件，才建立倉位。
4. 倉位建立後，`Scheduler` 持續監看：
   - tactical exit 條件
   - drawdown 狀態
   - Best Day Rule
   - re-evaluation reversal
   - broker-side / manual close
5. 任一 close-domain 觸發出現時，`Scheduler` 不再自己直接拼 broker calls，而是先建立 `CloseIntent`。
6. `CloseControlPlane.execute()` 根據 `action_kind` 分三類處理：
   - `modify_only`
   - `partial_close`
   - `full_close`
7. control plane 回傳 `CloseOutcome`，並立刻寫出 `CLOSE_CONTROL_EVENT`。
8. 若是 `modify_only`：
   - broker modify 失敗 -> `skipped / not_needed`
   - modify 成功但 verify 失敗 -> `verify_failed / mismatch`
   - verify 成功 -> `accepted / verified`
9. 若是 `partial_close` 或 `full_close`：
   - broker close 成功只代表 `submitted / pending_reconcile`
   - 不代表這筆 trade 已經 canonical 關閉
10. position monitor 偵測 position 消失後，`CloseReconciler.reconcile()` 會合併：
   - broker closed row
   - pending `CloseOutcome`
   - `execution_meta`
   - `_best_day_close_positions`
   - `_reevaluation_close_positions`
   - `_last_known_profit`
11. reconciler 產生：
   - `trigger_source`
   - `action_kind`
   - `final_close_reason`
   - `resolution_path`
12. `Scheduler` 再寫出 `TRADE_CLOSED`，更新 store、memory、metrics、alerts。

這一條流程的核心意義是：**平倉不再只是 broker API 成功或失敗，而是一個有命令、有執行、有回讀、有最終對賬的完整 lifecycle。**

---

## 5. close-domain 的三層主語義

理解 `v1.4.8`，最重要的是先拆開三層主語義。

### 5.1 `trigger_source`

`trigger_source（觸發來源，誰要求這次 close-domain 動作）` 回答的是「誰先按下了 close 按鈕」。

例如：

- `tactical_exit`
- `reduce_exposure`
- `emergency_close`
- `best_day_close`
- `reeval_close`
- `manual_or_broker`

### 5.2 `action_kind`

`action_kind（動作類型，系統這次試圖做什麼）` 回答的是「系統對 broker 想做哪一種操作」。

例如：

- `modify_only`
- `partial_close`
- `full_close`
- `external_detected_close`

### 5.3 `final_close_reason`

`final_close_reason（最終平倉原因，這筆倉位最後是怎麼結束的）` 回答的是「倉位終局被怎麼定性」。

例如：

- `tp_hit`
- `sl_hit`
- `best_day_close`
- `reeval_close`
- `manual_close`
- `broker_stopout`

這三層必須分開，因為它們不是同一件事。

最典型的例子，是 tactical `modify_only` 成功把 stop 移到保本位。此時：

- `trigger_source=tactical_exit`
- `action_kind=modify_only`
- 最後倉位若被移動後的 stop 打掉，`final_close_reason` 仍可能是 `sl_hit`

也就是說，**誰發起了操作**，和 **倉位最後怎麼死掉或獲利結束**，在 close-domain 裡本來就是兩件不同的事。

---

## 6. `submitted`、`verified`、`pending_reconcile` 與 `final_close_reason`

`v1.4.8` 的另一個關鍵，是把 broker write 狀態與最終 close reason 分開。

| 動作類型 | control plane 可直接知道的事 | 當下不能直接知道的事 |
| --- | --- | --- |
| `modify_only` | modify 是否送達、回讀是否一致 | 這次修改最後是否真的改變了交易終局 |
| `partial_close` | close request 是否已送出 | 剩餘倉位最後會怎麼關 |
| `full_close` | close request 是否已送出 | broker closed row 何時可見、最終對賬結果 |
| `external_detected_close` | position 消失了 | 到底是 manual、broker-side 還是某種 fallback 路徑 |

因此：

- `accepted / verified` 只代表 modify 類操作已被 broker 正確接受
- `submitted / pending_reconcile` 只代表 close request 已送出，尚未 canonical 結案
- `final_close_reason` 只能由 reconciler 在 position 真正消失後決定

這個分離非常重要，因為沒有它，系統就會犯兩種錯：

1. 把「modify 成功」誤寫成「這筆 trade 已安全收斂」
2. 把「close request 已送出」誤寫成「倉位已確定如何結束」

下面 12 個情景，都要用這個視角來看。

---

## 7. 情景 1：EURUSD 倫敦盤突破多單，`+1.0R` 先 partial close，餘倉 trail 後 TP

### 情景背景

- session：London open
- `qlib_market_scanner` 給出 breakout 候選
- `TradingAgents` 維持 `BUY` thesis，理由是歐盤開盤後的趨勢延續與美元偏弱

### 操作數字

| 欄位 | 數值 | 欄位 | 數值 |
| --- | --- | --- | --- |
| entry price | `1.08420` | initial SL | `1.08300` |
| initial TP | `1.08600` | lot | `0.12` |
| 5m realized vol | `0.10%` | 1m ATR | `2.1 pips` |
| 5m ATR | `6.4 pips` | spread | `0.7 pips` |
| partial close volume | `0.06 lot` | breakeven trigger | `+12.0 pips (+1.0R)` |
| trailing level | `1.08422 -> 1.08512 -> 1.08548` | close price | `1.08600` |
| realized pnl | `+$17.2` | trigger_source | `tactical_exit` |
| action_kind | `partial_close` | execution_status | `submitted` |
| readback_status | `pending_reconcile` | final_close_reason | `tp_hit` |
| resolution_path | `broker_api` | | |

### 完整流程

1. 09:08，scanner 將 `EURUSD` 列為突破型候選；`TradingAgents` 輸出 `BUY`。
2. 09:12，entry control plane 放行後在 `1.08420` 建立 `0.12 lot` 多單，初始風險是 `12 pips`。
3. 09:27，價格推進到 `1.08540`，剛好達到 `+1.0R`。tactical exit 產生 `PARTIAL_CLOSE`，`Scheduler` 建立 `CloseIntent(trigger_source="tactical_exit", action_kind="partial_close")`，對 `0.06 lot` 送 broker close。
4. close control plane 只知道這個 partial close 已經 `submitted / pending_reconcile`。它不會說「整筆 trade 已結束」，因為剩餘 `0.06 lot` 還在場內。
5. 同時，tactical exit 對剩餘倉位做保護：先把 `breakeven（保本位，止損移到接近進場價）` 抬到 `1.08422`，之後再依 1m / 5m 波動把 `trailing stop（移動止損，價格朝有利方向走時跟著上移保護位）` 推到 `1.08512`、`1.08548`。
6. 09:44，價格打到原始 TP `1.08600`，broker 關掉剩餘部位。position monitor 抓到 closed row 後，reconciler 看到 close reason 靠近 TP，將 terminal outcome 定為 `tp_hit`。

### tactical 模組的影響

這個案例很能說明 `v1.4.8` 的語義：

- `trigger_source=tactical_exit`、`action_kind=partial_close` 記錄的是關鍵 tactical 操作
- `final_close_reason=tp_hit` 記錄的是倉位終局

兩者不衝突，反而一起構成了完整事實。日後 operator 可以直接回答：

- tactical partial close 是否常把好單過早切掉
- 在 `+1.0R` 先收一半、餘倉 trail 的組合，是否比完全持有更穩

---

## 8. 情景 2：GBPUSD 紐約盤回撤空單，先 move to breakeven，回抽後打到保本止損

### 情景背景

- session：New York open 後 30 分鐘
- scanner 給出 pullback short 候選
- `TradingAgents` 維持 `SELL`，理由是美元修復而英鎊回抽失敗

### 操作數字

| 欄位 | 數值 | 欄位 | 數值 |
| --- | --- | --- | --- |
| entry price | `1.27180` | initial SL | `1.27310` |
| initial TP | `1.26920` | lot | `0.10` |
| 5m realized vol | `0.14%` | 1m ATR | `3.0 pips` |
| 5m ATR | `8.2 pips` | spread | `1.1 pips` |
| partial close volume | `N/A` | breakeven trigger | `+10.0 pips` |
| trailing level | `N/A` | close price | `1.27179` |
| realized pnl | `-$0.9` | trigger_source | `tactical_exit` |
| action_kind | `modify_only` | execution_status | `accepted` |
| readback_status | `verified` | final_close_reason | `sl_hit` |
| resolution_path | `broker_api` | | |

### 完整流程

1. 14:36，系統在 `1.27180` 建立 `GBPUSD` 空單，初始 SL 在 `1.27310`。
2. 14:49，價格先跌到 `1.27080`，已達 `+10 pips`，符合 `MOVE_TO_BREAKEVEN` 觸發門檻。
3. tactical exit 產生 `modify_only` 指令，請 broker 把止損改到 `1.27178`。`CloseControlPlane.execute()` 對這種操作會先 `modify_position()`，再 `verify_sl_tp()`。
4. 這次 verify 通過，所以 `execution_status=accepted`、`readback_status=verified`。也就是說，保本位真的已寫入 broker。
5. 15:02，價格回抽到 `1.27179`，剛好打到新的保本止損。position monitor 偵測部位消失後，reconciler 會把 `breakeven_sl` 視為可被分類的 stop 位，因此 terminal outcome 是 `sl_hit`，不是 `manual_close`。

### tactical 模組的影響

這裡最值得注意的，不是系統有沒有賺錢，而是：

- tactical 的 `MOVE_TO_BREAKEVEN` 已被明確寫成 `modify_only`
- broker verify 通過後，倉位之後若被保本位打掉，能被準確分類為 `sl_hit`

這讓 `v1.4.8` 可以統計：

- breakeven 觸發過早是否導致好單被洗出
- breakeven 在高 1m ATR 環境中是否更容易造成微幅虧損或零附近結束

---

## 9. 情景 3：USDJPY 趨勢延續多單，trail 連續上移兩次，最後保利出場

### 情景背景

- session：紐約盤中段
- scanner 顯示 trend continuation
- `TradingAgents` 維持 `BUY`，理由是美債殖利率支撐日圓弱勢

### 操作數字

| 欄位 | 數值 | 欄位 | 數值 |
| --- | --- | --- | --- |
| entry price | `148.420` | initial SL | `148.180` |
| initial TP | `148.980` | lot | `0.08` |
| 5m realized vol | `0.12%` | 1m ATR | `3.6 pips` |
| 5m ATR | `11.5 pips` | spread | `0.9 pips` |
| partial close volume | `N/A` | breakeven trigger | `+21.0 pips` |
| trailing level | `148.470 -> 148.610` | close price | `148.608` |
| realized pnl | `+$10.1` | trigger_source | `tactical_exit` |
| action_kind | `modify_only` | execution_status | `accepted` |
| readback_status | `verified` | final_close_reason | `sl_hit` |
| resolution_path | `broker_api` | | |

### 完整流程

1. 倉位在 `148.420` 建立後，市場沿趨勢推進，但 1m ATR 維持在 `3.6 pips` 左右，代表回吐速度不慢。
2. 當浮盈超過 `+21 pips`，系統先把保護位抬到 `148.470`，將初始風險的一部分收回。
3. 再往上走一段後，tactical exit 依 5m ATR 與局部 swing low，把 stop 再抬到 `148.610`。
4. 這兩次操作都屬於 `modify_only`，因此 control plane 的責任，是確認 broker write 已被接受與驗證，而不是直接定義最終平倉原因。
5. 後續價格在 `148.608` 回落，被第二次 trailing stop 打掉。reconciler 看到 close price 靠近 `trailing_sl`，會將 terminal outcome 定為 `sl_hit`。

### tactical 模組的影響

這個例子說明了一個常被誤解的點：

- 「保利出場」在倉位管理上是成功的
- 但 canonical `final_close_reason` 仍可能是 `sl_hit`

這不是錯，而是資料模型在回答不同問題：

- tactical 上，這次 trail 是正確保護
- terminal 上，broker 確實是用 stop 機制把部位關掉

---

## 10. 情景 4：AUDUSD 震盪盤多單，reprice TP 後縮短持倉時間獲利出場

### 情景背景

- session：亞洲盤尾段
- scanner 抓到區間底部反彈
- `TradingAgents` 給出 `BUY`，但把 thesis 定位為 range bounce，而不是 trend expansion

### 操作數字

| 欄位 | 數值 | 欄位 | 數值 |
| --- | --- | --- | --- |
| entry price | `0.65420` | initial SL | `0.65300` |
| initial TP | `0.65680` | lot | `0.14` |
| 5m realized vol | `0.08%` | 1m ATR | `1.4 pips` |
| 5m ATR | `4.6 pips` | spread | `0.8 pips` |
| partial close volume | `N/A` | breakeven trigger | `+8.4 pips` |
| trailing level | `N/A` | close price | `0.65558` |
| realized pnl | `+$19.1` | trigger_source | `tactical_exit` |
| action_kind | `modify_only` | execution_status | `accepted` |
| readback_status | `verified` | final_close_reason | `tp_hit` |
| resolution_path | `broker_api` | | |

### 完整流程

1. 原始 TP 在 `0.65680`，對應的是完整趨勢延伸目標。
2. 但進場後 20 分鐘，5m realized vol 只維持在 `0.08%`，而上方區間壓力沒有被有效吞掉。
3. tactical exit 判定這不是趨勢擴張，而是區間回補，因此發出 `REPRICE_TP`，把 TP 下修到 `0.65560`。
4. 這是一個典型 `modify_only`：broker modify 成功，verify 也成功，所以 `execution_status=accepted`、`readback_status=verified`。
5. 之後價格在 `0.65558` 被新 TP 收掉。reconciler 看到 close price 靠近 `dynamic_tp`，將 terminal outcome 定為 `tp_hit`。

### tactical 模組的影響

這個案例的核心，不是系統把 TP 改小，而是把策略 thesis 從「等趨勢拉滿」改成「先把區間能拿的先拿掉」。

對 operator 來說，可學習的地方是：

- `REPRICE_TP` 是否能顯著縮短持倉時間
- 在低 5m ATR、低 realized vol 的盤面中，提早收斂 TP 是否優於硬等原始目標

---

## 11. 情景 5：NZDUSD 反轉風險升高，tactical 直接 full close

### 情景背景

- session：紐約盤尾段
- scanner 與 `TradingAgents` 原本給的是 `BUY`
- 持倉後出現快速反向 orderflow，tactical exit 判斷原 thesis 已明顯失去效率

### 操作數字

| 欄位 | 數值 | 欄位 | 數值 |
| --- | --- | --- | --- |
| entry price | `0.60640` | initial SL | `0.60510` |
| initial TP | `0.60890` | lot | `0.11` |
| 5m realized vol | `0.16%` | 1m ATR | `2.8 pips` |
| 5m ATR | `7.9 pips` | spread | `1.0 pips` |
| partial close volume | `N/A` | breakeven trigger | `not reached` |
| trailing level | `N/A` | close price | `0.60640` |
| realized pnl | `$0.0` | trigger_source | `tactical_exit` |
| action_kind | `full_close` | execution_status | `submitted` |
| readback_status | `pending_reconcile` | final_close_reason | `manual_close` |
| resolution_path | `broker_api` | | |

### 完整流程

1. 進場後 10 分鐘，1m ATR 從 `2.0` 擴到 `2.8 pips`，而 bid-side 深度連續走弱。
2. tactical exit 不再選擇移動止損，而是直接發出 `EXIT_NOW`。`Scheduler` 會建立 `CloseIntent(trigger_source="tactical_exit", action_kind="full_close")`。
3. `CloseControlPlane.execute()` 對 `full_close` 的語義很保守：只要 broker close request 成功，就回 `submitted / pending_reconcile`。
4. position monitor 之後在 broker closed row 中看到這筆單以近乎打平的價格離場，且 close price 不靠近 SL / TP。
5. 現行 reconciler 並不會把 tactical full close 自動保留為 terminal reason；在這種 close price 不近 SL/TP、PnL 也近乎零的情況，它最後會落到 `manual_close`。

### tactical 模組的影響

這是 `v1.4.8` 很值得直接說清楚的一個現況：

- tactical `full_close` 已經有 `trigger_source=tactical_exit`
- 但 terminal `final_close_reason` 目前不一定保留 tactical 語義

也就是說，這個情景不是在說系統做錯，而是在提醒：

- `v1.4.8` 已把 close control plane 做穩
- 但 tactical full close 的 terminal classification，仍然是 `v1.4.9` 值得微調的地方

---

## 12. 情景 6：EURUSD 高波動回吐，modify 成功但 verify mismatch，最後仍被原始 SL 打掉

### 情景背景

- session：倫敦盤中段
- scanner 與 `TradingAgents` 都支持 `BUY`
- 倉位先進入浮盈，但隨後波動突然放大

### 操作數字

| 欄位 | 數值 | 欄位 | 數值 |
| --- | --- | --- | --- |
| entry price | `1.08960` | initial SL | `1.08840` |
| initial TP | `1.09200` | lot | `0.13` |
| 5m realized vol | `0.18%` | 1m ATR | `3.8 pips` |
| 5m ATR | `10.6 pips` | spread | `0.9 pips` |
| partial close volume | `N/A` | breakeven trigger | `+10.8 pips` |
| trailing level | `attempted 1.08992` | close price | `1.08838` |
| realized pnl | `-$15.8` | trigger_source | `tactical_exit` |
| action_kind | `modify_only` | execution_status | `verify_failed` |
| readback_status | `mismatch` | final_close_reason | `sl_hit` |
| resolution_path | `broker_api` | | |

### 完整流程

1. 倉位進入 `+0.9R` 附近時，tactical exit 試圖把 stop 從 `1.08840` 抬到 `1.08992`。
2. broker modify call 回報 success，但 `verify_sl_tp()` 沒有在 read-back 中看到預期值，因此 `CloseControlPlane.execute()` 回 `verify_failed / mismatch`。
3. 這個結果的關鍵不是「broker modify 一定沒成功」，而是「系統不能把它當成已驗證成功」。
4. 因為 verify mismatch，系統不會把新的 tactical 保護位當作已可信寫入。也就是說，`execution_meta` 不應更新成新的 trailing level。
5. 幾分鐘後價格急跌到 `1.08838`，實際打到的仍是原始 SL 區域。reconciler 看到 broker close reason / close price 靠近舊 stop，將 terminal outcome 定為 `sl_hit`。

### tactical 模組的影響

這個情景是 `v1.4.8` 最核心的工程價值之一：

- 過去系統很容易把「modify API 回 success」誤以為保護單已生效
- 現在 control plane 明確區分：
  - broker write success
  - read-back verified
  - verify mismatch

對 operator 來說，這意味著可以把問題準確分成兩類：

1. tactical 規則本身不佳
2. tactical 規則本來對，但 broker read-back integrity 不夠

---

## 13. 情景 7：GBPUSD drawdown 進入 DANGER，系統做 reduce exposure partial close

### 情景背景

- session：紐約盤前後
- scanner 原本給的是高信心 `BUY`
- 倉位建立後，帳戶層級 drawdown 進入 `DANGER`

### 操作數字

| 欄位 | 數值 | 欄位 | 數值 |
| --- | --- | --- | --- |
| entry price | `1.26840` | initial SL | `1.26710` |
| initial TP | `1.27100` | lot | `0.18` |
| 5m realized vol | `0.15%` | 1m ATR | `2.9 pips` |
| 5m ATR | `8.1 pips` | spread | `1.2 pips` |
| partial close volume | `0.09 lot` | breakeven trigger | `deferred` |
| trailing level | `1.26940` on remaining size | close price | `1.27092` |
| realized pnl | `+$19.3 total` | trigger_source | `reduce_exposure` |
| action_kind | `partial_close` | execution_status | `submitted` |
| readback_status | `pending_reconcile` | final_close_reason | `tp_hit` |
| resolution_path | `broker_api` | | |

### 完整流程

1. 帳戶 drawdown 逼近 danger band 時，系統不直接全平，而是先做 `reduce_exposure（降風險減倉，先把部位縮小以降低帳戶暴露）`。
2. `Scheduler` 建立 `CloseIntent(trigger_source="reduce_exposure", action_kind="partial_close")`，對 `0.09 lot` 送出部分平倉。
3. control plane 只回 `submitted / pending_reconcile`。這一步仍然不代表整筆 trade 結案。
4. 部分平倉後，剩餘 `0.09 lot` 的風險顯著下降，tactical exit 才有空間把 trailing 保護位往上拉到 `1.26940`。
5. 市場回升後，剩餘部位在 `1.27092` 附近出場，terminal outcome 被 reconciler 定為 `tp_hit`。

### tactical 模組的影響

這個情景展示了 close control plane 與 compliance / equity layer 的接合方式：

- `reduce_exposure` 處理的是帳戶風險，不是方向 thesis
- 但它仍然透過同一套 close control plane 寫入一致的 audit trail

因此，之後能直接分析：

- 先 partial close 再讓剩餘倉位跑完，是否比直接 emergency flat 更有效
- 哪一種 drawdown 區間最常需要 de-risk

---

## 14. 情景 8：帳戶進入 CRITICAL，emergency close 同時處理兩筆 USDCHF 倉位

### 情景背景

- 帳戶權益快速惡化，drawdown 狀態進入 `CRITICAL`
- 系統在同一 symbol 上有兩筆未結倉位，需要立刻平掉
- 這裡的重點不是方向是否正確，而是 `emergency_close` 的 child intents 是否一致

### 倉位數字

| 欄位 | USDCHF-A | USDCHF-B |
| --- | --- | --- |
| entry price | `0.90160` | `0.90110` |
| initial SL | `0.90020` | `0.89980` |
| initial TP | `0.90420` | `0.90340` |
| lot | `0.10` | `0.08` |
| 5m realized vol | `0.17%` | `0.17%` |
| 1m ATR | `2.7 pips` | `2.7 pips` |
| 5m ATR | `7.5 pips` | `7.5 pips` |
| spread | `1.0 pips` | `1.0 pips` |
| close price | `0.90078` | `0.90074` |
| realized pnl | `-$8.8` | `-$3.2` |

| canonical 欄位 | 數值 |
| --- | --- |
| partial close volume | `N/A` |
| breakeven trigger | `ignored under emergency` |
| trailing level | `ignored under emergency` |
| trigger_source | `emergency_close` |
| action_kind | `full_close` |
| execution_status | `submitted` |
| readback_status | `pending_reconcile` |
| final_close_reason | `emergency_close` |
| resolution_path | `broker_api` |

### 完整流程

1. drawdown monitor 宣告 `CRITICAL` 後，策略優先級會完全讓位給帳戶生存。
2. `Scheduler` 不是發一個抽象的 `close_all` 給 broker，而是對每一筆 position 展開 child `CloseIntent`，每筆都帶 `trigger_source="emergency_close"`、`action_kind="full_close"`。
3. broker close request 送出後，control plane 對每一筆都回 `submitted / pending_reconcile`。
4. position monitor 之後分別看到兩筆 position 消失。因為 `trigger_source=emergency_close` 在 reconciler 的優先序最高，所以 terminal outcome 直接定為 `emergency_close`，不會被 PnL sign 改寫成 `sl_hit` 或 `tp_hit`。

### tactical 模組的影響

這個情景說明了一個設計取向：

- tactical exit 是在優化單筆 trade
- `emergency_close` 是在保命

兩者都走同一套 close control plane，但 `emergency_close` 在 final reason 上擁有明確優先權，這讓日後統計不會把「系統為了保帳戶而強制平倉」誤讀成單純 stop loss。

---

## 15. 情景 9：EURUSD 持倉期間 LLM re-eval 翻向，觸發 reeval close

### 情景背景

- session：美盤中段
- 原始倉位是 `BUY`
- 新一輪 `TradingAgents` re-evaluation 發現 thesis 反轉，策略方向改判為 `SELL`

### 操作數字

| 欄位 | 數值 | 欄位 | 數值 |
| --- | --- | --- | --- |
| entry price | `1.08720` | initial SL | `1.08590` |
| initial TP | `1.08980` | lot | `0.12` |
| 5m realized vol | `0.11%` | 1m ATR | `2.2 pips` |
| 5m ATR | `6.8 pips` | spread | `0.8 pips` |
| partial close volume | `N/A` | breakeven trigger | `not used` |
| trailing level | `not used` | close price | `1.08786` |
| realized pnl | `+$7.9` | trigger_source | `reeval_close` |
| action_kind | `full_close` | execution_status | `submitted` |
| readback_status | `pending_reconcile` | final_close_reason | `reeval_close` |
| resolution_path | `broker_api` | | |

### 完整流程

1. 倉位仍在小幅浮盈時，LLM re-eval 判斷原本的上行 thesis 不再成立。
2. `Scheduler` 建立 `CloseIntent(trigger_source="reeval_close", action_kind="full_close")`，直接要求 broker 平倉。
3. 這個請求在 control plane 看起來只是 `submitted / pending_reconcile`。
4. 真正重要的是 position 消失之後的 canonical classification。由於 `reeval_close` 在 reconciler 的優先權高於 broker close reason 與 PnL sign，所以 terminal outcome 最終定為 `reeval_close`。

### tactical 模組的影響

這裡的重點是：`TradingAgents` 的 re-eval 不是直接改 store，而是透過 close control plane 落到可審計的操作路徑。這代表日後可以回看：

- 哪些 pair 最常因 re-eval 翻向而提早平倉
- re-eval close 是否真的減少了 thesis 失效後的回吐

---

## 16. 情景 10：GBPJPY 觸發 Best Day Rule 保護，主動鎖定日內浮盈

### 情景背景

- 帳戶日內已累積不錯的浮盈
- `Best Day Rule（最佳日獲利規則，避免單日利潤過度集中造成 prop rule 風險）` 接近保護閾值
- 系統選擇把浮盈倉位先關掉，而不是繼續暴露

### 操作數字

| 欄位 | 數值 | 欄位 | 數值 |
| --- | --- | --- | --- |
| entry price | `191.420` | initial SL | `191.760` |
| initial TP | `190.720` | lot | `0.06` |
| 5m realized vol | `0.19%` | 1m ATR | `5.6 pips` |
| 5m ATR | `17.8 pips` | spread | `1.8 pips` |
| partial close volume | `N/A` | breakeven trigger | `already beyond` |
| trailing level | `optional but ignored here` | close price | `191.040` |
| realized pnl | `+$15.2` | trigger_source | `best_day_close` |
| action_kind | `full_close` | execution_status | `submitted` |
| readback_status | `pending_reconcile` | final_close_reason | `best_day_close` |
| resolution_path | `broker_api` | | |

### 完整流程

1. position monitor 發現這筆 `GBPJPY` 空單已有明顯浮盈，而帳戶 best-day exposure 已接近 guard。
2. 系統不是等 TP，也不是等 tactical trailing，而是直接由 risk/compliance 層發起 `best_day_close`。
3. `CloseIntent` 經 close control plane 送出 broker close request。
4. 位置消失後，即使 broker close price 看起來很像正常獲利出場，reconciler 仍會優先保留 `best_day_close` 作為 terminal outcome。

### tactical 模組的影響

這個案例說明 `v1.4.8` 將「帳戶規則保護」也收進了同一套 close audit schema。之後可以直接回答：

- Best Day Rule 每週觸發幾次
- 這些被保護性關掉的倉位，若硬持有到原始 TP，歷史上平均會多賺還是回吐

---

## 17. 情景 11：AUDUSD 出現 broker-side / manual close，系統以 external detected close 對賬

### 情景背景

- 倉位原本由 scanner 與 `TradingAgents` 正常建立
- 之後不是 tactical、不是 reeval、不是 best-day、也不是 emergency 發起
- position monitor 單純發現部位不見了

### 操作數字

| 欄位 | 數值 | 欄位 | 數值 |
| --- | --- | --- | --- |
| entry price | `0.65180` | initial SL | `0.65050` |
| initial TP | `0.65460` | lot | `0.10` |
| 5m realized vol | `0.07%` | 1m ATR | `1.2 pips` |
| 5m ATR | `4.2 pips` | spread | `0.8 pips` |
| partial close volume | `N/A` | breakeven trigger | `not used` |
| trailing level | `not used` | close price | `0.65180` |
| realized pnl | `$0.0` | trigger_source | `manual_or_broker` |
| action_kind | `external_detected_close` | execution_status | `N/A` |
| readback_status | `N/A` | final_close_reason | `manual_close` |
| resolution_path | `broker_api` | | |

### 完整流程

1. position monitor 輪詢時，發現 `AUDUSD` position 已不在 open positions 內。
2. 因為系統內沒有 pending `CloseOutcome` 可對應，reconciler 會先落一個預設：
   - `trigger_source=manual_or_broker`
   - `action_kind=external_detected_close`
3. 若 broker closed row 顯示 close price 幾乎等於 entry，且不靠近 SL / TP、PnL 也近乎零，final classification 會落到 `manual_close`。

### tactical 模組的影響

這個案例的重要性不在 tactical，而在 close control plane 的邊界清楚了：

- 不是所有關倉都一定來自系統主動命令
- 但即使是外部關倉，系統也有一套 canonical schema 去收斂它

---

## 18. 情景 12：USDCAD 外部關倉且 broker closed row 延遲，靠 execution meta 與 last-known profit 完成最終收斂

### 情景背景

- 倉位原本是正常 `SELL`
- position monitor 發現部位消失，但 broker closed row 在前幾次重試都還沒出現
- 這是 `v1.4.8` reconciliation 能力最能體現的情景之一

### 操作數字

| 欄位 | 數值 | 欄位 | 數值 |
| --- | --- | --- | --- |
| entry price | `1.35280` | initial SL | `1.35410` |
| initial TP | `1.35020` | lot | `0.09` |
| 5m realized vol | `0.09%` | 1m ATR | `1.7 pips` |
| 5m ATR | `5.3 pips` | spread | `0.9 pips` |
| partial close volume | `N/A` | breakeven trigger | `not used` |
| trailing level | `not used` | close price | `1.35020` |
| realized pnl | `+$17.1` | trigger_source | `manual_or_broker` |
| action_kind | `external_detected_close` | execution_status | `N/A` |
| readback_status | `N/A` | final_close_reason | `tp_hit` |
| resolution_path | `execution_meta` | | |

### 完整流程

1. position monitor 首先看到 position 消失，但 broker API 在 `2s / 4s / 8s / 12s / 16s` 重試窗口內都還沒回 closed row。
2. 此時系統先從 `_last_known_profit` 拿到最後一次輪詢到的浮盈，約 `+$17.1`。
3. 因為 fallback PnL 為正，舊邏輯先把 exit reason 重新推成 `tp_hit`。
4. 接著，`execution_meta` 中保留了開倉時的 `tp_price=1.35020`，所以 close price 能用 execution meta 補齊。
5. reconciler 看到：
   - 沒有 pending close outcome
   - 沒有 broker closed row
   - 有正向 PnL
   - 有 `execution_meta` 可補 close price
6. 最後輸出：
   - `trigger_source=manual_or_broker`
   - `action_kind=external_detected_close`
   - `final_close_reason=tp_hit`
   - `resolution_path=execution_meta`

### tactical 模組的影響

這個案例展示了 close control plane 的真正韌性：

- 沒有 broker closed row，不等於系統就只能寫成 unknown
- `execution_meta` 與 `last_known_profit` 可以一起把 final classification 補回來

同時也揭示了一個細節：雖然 PnL 來自 `last_known_profit`，但 `resolution_path` 仍會落在 `execution_meta`，因為當前 reconciler 的 path 優先序是先記錄哪條路徑真正補齊了 close facts。

---

## 19. v1.4.8 對交易流程的真正影響

如果把上面 12 個情景濃縮成一句話，`v1.4.8` 做的不是「多幾個平倉規則」，而是：

**把平倉從 scattered broker calls，升級成一套可命名、可驗證、可對賬、可回放的 control plane。**

它對整個交易流程的真正影響有六點：

1. close-domain 不再只有 `exit_reason` 這個模糊欄位，而是明確拆成 `trigger_source`、`action_kind`、`final_close_reason`。
2. tactical modify 不再因為 broker API 回 success 就被視為真的完成，`verify_failed / mismatch` 現在有獨立語義。
3. partial close 與 full close 不再混成同一件事，剩餘倉位的終局能獨立被對賬。
4. `emergency_close`、`best_day_close`、`reeval_close` 這些高優先級 close，可以在 terminal classification 上保留自身語義。
5. external / manual close 不再只能靠 operator 猜，而是有固定的 fallback reconciliation 路徑。
6. `v1.4.7` 的 entry control plane 與 `v1.4.8` 的 close control plane 現在可以前後呼應，形成完整交易閉環。

---

## 20. control plane 的可持續學習性

這一版真正讓 `v1.4.9`、`v1.5.0` 有基線的地方，不只是 close 做穩，而是 **整個 control plane 變得可學習**。

### 20.1 `CLOSE_CONTROL_EVENT` 是操作層樣本

每次 close-domain 操作，現在至少都能留下：

- `trigger_source`
- `action_kind`
- `execution_status`
- `readback_status`
- `reason_code`
- `broker_success`
- `broker_message`

這意味著日後可以問：

- 哪種 `reason_code` 最常導致 `verify_failed`
- 哪個 pair 最常出現 `partial_close` 後仍然回吐
- `MOVE_TO_BREAKEVEN` 在高 1m ATR 時段是否過於頻繁

### 20.2 `TRADE_CLOSED` 是終局層樣本

當 position 最終關掉後，現在又會留下：

- `pnl`
- `reason`
- `trigger_source`
- `action_kind`
- `final_close_reason`
- `resolution_path`

這讓系統之後可以把「做了什麼 close-domain 動作」和「最後倉位怎麼結束」對起來。

### 20.3 scanner / `TradingAgents` / close control plane 可以被串起來學

一筆 trade 現在可以沿著這條鏈路被 join 起來：

1. scanner 候選特徵
2. `TradingAgents` thesis
3. entry control plane verdict
4. open trade metadata
5. close control events
6. final trade close

這表示 `v1.5.0` 若要做更成熟的 memory quality、policy calibration、pair-specific tactical tuning，已經有足夠乾淨的資料骨架，不必再先回頭補 close schema。

---

## 21. audit trail：一筆單如何被完整回放

最適合拿來示範 audit trail 的，是上面的 **情景 6**。這筆單的價值，在於它剛好穿過了 `modify_only -> verify mismatch -> final SL` 的全鏈路。

### 21.1 replay path

1. `TradeIntent` 已開倉：`EURUSD BUY @ 1.08960`
2. tactical exit 判斷應該把保護位抬到 `1.08992`
3. `Scheduler` 建立：

```json
{
  "trigger_source": "tactical_exit",
  "action_kind": "modify_only",
  "position_id": "POS-20481",
  "intent_id": "INT-20481",
  "symbol": "EURUSD",
  "reason_code": "atr_trailing_stop_improved"
}
```

4. `CloseControlPlane.execute()` 回：

```json
{
  "trigger_source": "tactical_exit",
  "action_kind": "modify_only",
  "execution_status": "verify_failed",
  "readback_status": "mismatch"
}
```

5. `Scheduler` 立刻寫 `CLOSE_CONTROL_EVENT`
6. 後續 broker closed row 顯示 close price 在 `1.08838`，接近原始 SL `1.08840`
7. `CloseReconciler.reconcile()` 回：

```json
{
  "trigger_source": "tactical_exit",
  "action_kind": "modify_only",
  "final_close_reason": "sl_hit",
  "resolution_path": "broker_api"
}
```

8. `Scheduler` 再寫 `TRADE_CLOSED`
9. store 的 `exit_reason` 最後被固定為 `sl_hit`
10. memory / reflection 看到的是：
    - 這筆單曾有 tactical intervention
    - intervention 的 broker read-back 並未驗證成功
    - 最終仍被 stop 打掉

### 21.2 為什麼這條 audit trail 有用

如果沒有這條鏈路，operator 看到的只會是：

- `EURUSD closed, loss`

但有了 `v1.4.8` 之後，可以準確回答：

- 誰先發起 close-domain 行為
- 系統到底是 modify、partial close 還是 full close
- broker write 是否已被 verify
- 最後 terminal reason 是什麼
- 這次 close facts 是靠 broker API、execution meta，還是 last-known profit 補回來

這就是 audit trail 不再只是 log，而是 **可重放的運營事實鏈**。

---

## 22. 對 v1.4.9 與 v1.5.0 的邏輯基線

### 22.1 對 v1.4.9 的直接 baseline

`v1.4.9` 已預留為 `v1.4.7` 與 `v1.4.8` 的 bugfix / 微調，因此最應直接承接的是：

1. `verify mismatch` 的容忍與定位
   - 是否有某些 symbol 的 price precision 或 broker rounding 需要更細修正
2. tactical full close 的 terminal semantics
   - 是否要讓 `tactical_exit` 在某些 full-close 場景保留更明確的終局分類
3. partial close 的 journal / analytics 細節
   - 是否需要額外持久化 closed volume、remaining volume、weighted pnl
4. `manual_or_broker` 的分類風險
   - 哪些 external close 被現在的 PnL sign fallback 過度簡化
5. operator alert wording
   - `verify_failed`、`pending_reconcile`、`execution_meta` fallback 是否要在通知中說得更直接

### 22.2 對 v1.5.0 的邏輯 baseline

若 `v1.4.9` 把 close-domain 的 bug 與邊界微調補好，`v1.5.0` 就可以往更高一層做：

1. entry / close control plane 的聯動 policy
   - 哪些 entry friction 類型應配哪種 exit policy
2. memory quality 提升
   - 將 `trigger_source / action_kind / final_close_reason / resolution_path` 納入 lesson extraction
3. exposure budget
   - 不只看單筆 trade，而是看同時持倉的組合風險怎麼驅動 de-risk 或 emergency 行為
4. cross-repo contract freeze
   - 固定 scanner、`TradingAgents`、`prop-firm-pilot` 之間的欄位契約，讓 runtime learning 更穩

也就是說，`v1.4.8 report` 的真正價值，不是把目前 close logic 說完，而是替後續版本定義一條清楚的優先順序：

- `v1.4.9` 修 close-domain 邊界
- `v1.5.0` 再把整條 trading control plane 做成更成熟的 learning system

---

## 23. 術語速覽

- `Close Control Plane`：平倉控制平面。所有 close-domain 命令共用的一套執行、驗證、對賬與審計層。
- `CloseIntent`：平倉意圖。系統準備送去 broker 的結構化 close 命令。
- `CloseOutcome`：執行結果。描述 broker write 是否送達，以及 read-back 是否驗證成功。
- `CloseReconciler`：平倉對賬器。把 broker closed row、execution meta、fallback PnL 等資訊合併成單一最終結論。
- `trigger_source`：觸發來源。是誰要求這次 close-domain 動作。
- `action_kind`：動作類型。系統這次對 broker 嘗試做什麼。
- `execution_status`：執行狀態。broker write 在 control plane 看起來是成功、略過、還是 verify 失敗。
- `readback_status`：回讀狀態。系統回頭驗證 broker 狀態時，是否與期望一致。
- `pending_reconcile`：待對賬。close request 已送出，但還沒等到 canonical final close。
- `final_close_reason`：最終平倉原因。倉位在 terminal state 被怎麼定性。
- `resolution_path`：解析路徑。reconciler 主要靠哪條資料來源把 close facts 補齊。
- `partial close`：部分平倉。只先關掉一部分倉位。
- `full close`：全平。把當前 position 全部關掉。
- `modify_only`：只改保護單，不直接關倉。
- `breakeven`：保本位。把 stop 移到進場價附近，盡量避免好單翻成虧損。
- `trailing stop`：移動止損。價格朝有利方向移動時，跟著調整保護位。
- `reprice TP`：重設止盈。把原本的 TP 改到更貼近當前市場結構的位置。
- `realized volatility`：已實現波動率。過去一小段時間內，價格實際走了多大。
- `ATR`：平均真實波幅。常用來估計近期波動區間大小。
- `execution_meta`：執行附加資料。開倉或後續操作時保留下來的價格、volume、保護單資訊。

---

如果把這份報告再濃縮一次，`v1.4.8` 的關鍵不是「平倉更花俏」，而是：

**從現在開始，系統終於能用同一套語言說明一筆單是誰想關、怎麼關、最後怎麼結束。**
