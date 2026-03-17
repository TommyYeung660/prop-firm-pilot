# E8 One Config Layout Design

**Goal:** 在同一份 `config/e8_one_5k_challenge.yaml` 內，把內容重新整理成「高頻常調參數」與「低頻基礎參數」兩大區塊，讓日常調參時能先看到最常改的部分，同時保留既有載入邏輯與所有參數值不變。

**Context**

- 目前這份 YAML 已有完整中文註解，但區塊順序仍偏向程式 schema，而不是實際維運調參流程。
- 使用者日常最常改的，主要是掃描效率、交易節奏、行情容忍度、tactical gate 與標的清單。
- 低頻才會碰的，是帳號規則、資金基線、風控規則、儲存路徑、instrument 規格等。

**Approach Options**

1. 保持現有順序，只加「高頻速查表」註解
   - 優點：最小 diff。
   - 缺點：閱讀與維護體驗改善有限，不符合使用者要的兩大區塊。

2. 在同一份 YAML 內重排成兩大區塊
   - 優點：不改載入邏輯、不改值，但閱讀路徑明顯改善。
   - 缺點：diff 較大，需要額外做等價驗證避免誤改值。

3. 拆成兩份 YAML 再做 merge
   - 優點：概念上最乾淨。
   - 缺點：需要動配置載入流程，風險不必要。

**Chosen Design**

採用方案 2，在同一份 YAML 內重排。

1. 區塊安排
   - 檔案最前面保留 metadata 與閱讀說明。
   - 第一大區塊為「高頻常調參數」：
     - `symbols`
     - `scanner`
     - `execution`
     - `websocket`
     - `scheduler`
     - `tactical`
   - 第二大區塊為「低頻基礎參數」：
     - `account`
     - `compliance`
     - `decision_store`
     - `monitor`
     - `optimization`
     - `agents`
     - `instruments`

2. 不變項
   - 不更改任何 key 名稱。
   - 不更改任何參數值。
   - 不更改 YAML 結構型別。
   - 不更改 `load_config()` 行為。

3. 註解策略
   - 延續現有中文註解。
   - 在兩大區塊加上用途說明，讓使用者知道哪些是日常常調、哪些通常不要動。
   - 共用欄位註解保留集中寫法，避免 instrument 區過度冗長。

4. 驗證
   - 在重排前，先將目前 YAML 解析成 baseline dict。
   - 重排後重新解析。
   - 驗證兩份 dict 完全相等，確認只有順序與註解改變。

**Non-Goals**

- 不處理 scanner retry / market-data retry 行為。
- 不調整任何現有數值。
- 不把這份 YAML 拆成多檔。
