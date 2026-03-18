# 現時 FX 量化分析算法與有效性總結

**更新日期:** `2026-03-18`

## 一頁結論

如果只看一句話：`現時 FX 最可用的量化分析，不是追求最複雜模型，而是用成本意識、regime 意識與嚴格驗證，把簡單模型、因子訊號與風控組合起來。`

再講得更直接一點：

- `短線純價格預測` 依然很難，很多主要貨幣對在短期看起來仍接近 random walk。
- `中期、條件式、regime-aware` 的預測比「任何時候都能預測」更可信。
- `簡單可解釋模型` 近年的實證裡，往往比黑盒深度模型更耐交易成本。
- `order flow / limit order book` 類方法可能很強，但通常只對有專有資料與低延遲基建的團隊真正可用。
- `深度學習、強化學習、LLM` 現在更像增強器，不太像已證明穩定勝出的萬能核心。

一句話結論：`現時真正有用的，不是最炫的算法，而是最能穿過交易成本、資料偏差與市場 regime 變化的算法。`

## 1. 先理解 FX 市場為什麼難做

截至 `2025-09-30`，BIS 公布 `2025 Triennial Survey` 顯示，全球 OTC FX 日均成交約 `9.6 兆美元`，比 `2022` 的 `7.5 兆美元` 再增長，市場高度電子化、分散化，而且美元仍在 `89.2%` 的交易一側出現。這代表兩件事：

1. 市場很深，但也很競爭。
2. 小訊號很容易被交易成本、滑點和其他參與者吃掉。

BIS 對 FX execution algorithm 的研究也指出，演算法執行已是成熟工具，能幫助參與者在碎片化市場中找流動性，但同時也提高了對資料、風控與執行監控的要求。

一句話結論：`FX 不是沒有訊號，而是訊號通常很薄，執行要求很高。`

## 2. 「有效」在 FX 裡是什麼意思

在 FX，`預測準` 不等於 `可賺錢`。一個模型要算「有效」，至少要過四關：

1. `樣本外有效`
不是只在訓練期或同一段歷史資料裡好看。

2. `成本後有效`
扣掉 spread、commission、swap、滑點之後，還有沒有正期望。

3. `跨 regime 不會完全失靈`
疫情、升息週期、避險潮、央行干預來時，策略不能整個翻車。

4. `可重複驗證`
不是只靠一篇回測、單一貨幣對、或一組剛好調中的參數。

`2025` 的一篇 FX 機器學習研究特別重要，因為它把時間序列交叉驗證、動態交易成本與未來期間驗證都放進去；結果顯示，很多模型在「不算成本」時看起來都行，但一進入真實成本框架，差距立刻拉開。

一句話結論：`FX 最常見的錯覺，是把「預測分數好看」誤當成「交易可行」。`

## 3. 主流算法總覽

| 算法類別 | 簡單說法 | 常用資料 | 目前有效性判斷 |
| --- | --- | --- | --- |
| 技術分析 / 趨勢追蹤 | 看價格是否延續既有方向 | OHLCV、技術指標 | `有條件有效`，在部分貨幣與時段仍有訊號，但 edge 較容易隨時間衰退 |
| 均值回歸 / 價值型 | 看匯率是否偏離長期合理區間 | 實質匯率、PPP、利差 | `中期較有用`，短線通常不強，較適合當慢因子 |
| Carry / Momentum / Value 因子 | 用利差、相對強弱、估值排序貨幣 | 利率、遠期點、過去報酬、實質匯率 | `仍是核心研究基線`，但公開 alpha 有被壓縮跡象 |
| 統計時間序列模型 | 用 ARIMA、GARCH、狀態轉換模型抓波動與 regime | 價格、報酬、波動 | `適合當基線與風控工具`，單獨做 alpha 通常有限 |
| 樹模型 / 線性 ML | Logistic、Random Forest、XGBoost 之類 | 技術面、宏觀面、衍生特徵 | `目前實務感最強`，尤其簡單可解釋模型在成本後表現不差 |
| 深度學習 | LSTM、GRU、CNN、Transformer | 長序列價格、宏觀、文本 | `研究很多，但穩定超額證據仍混合` |
| 強化學習 | 讓 agent 自己學進出場或倉位 | 價格、狀態特徵、reward 設計 | `前沿但偏實驗性`，最依賴模擬環境與 reward 設計 |
| 新聞 / 情緒 / LLM 融合 | 把新聞、評論、宏觀文本轉成訊號 | 新聞、研究報告、社群文本 | `可作增強層`，目前較像輔助特徵，不像穩定主引擎 |
| Order Flow / LOB | 看買賣盤與成交流如何推動短線價格 | Tick、成交、委託簿 | `理論和實務都可能強`，但資料與基建門檻最高 |

一句話結論：`現在最值得看的是「簡單 ML + 因子 + regime / risk overlay」，不是單押某一種神奇模型。`

## 4. 各類算法的現時有效性

### 4.1 技術分析 / 趨勢追蹤

這類方法的核心很簡單：`如果價格已經在動，就先假設它短時間還會沿著原方向走。`

近年仍有研究認為，技術交易規則在 FX 不算完全失效。`2024` 發表的一篇大型研究檢查超過 `21,000` 條規則、`30` 個貨幣，指出 developed 與 emerging currencies 都仍能看到某些可預測性與獲利性；但它同時也指出，這種獲利性在成熟市場隨時間有減弱現象。

怎樣理解這件事比較準：

- 它不是說「均線交叉永遠有用」。
- 它比較像是說，`市場在部分階段仍會留下趨勢或行為偏誤痕跡`。
- 真正難的是避免 data snooping，也就是從一大堆規則裡碰巧挑中歷史贏家。

一句話結論：`技術分析沒有死，但它比較像條件式 edge，不像永久聖杯。`

### 4.2 均值回歸、價值型與基本面模型

這類方法的想法是：`匯率雖然短期很吵，但如果偏離經濟基本面太遠，之後可能慢慢回來。`

這也是為什麼很多 FX 研究喜歡看：

- 實質匯率
- 購買力平價（PPP）
- 利差
- 殖利率曲線
- 不確定性與媒體關注度

近期官方研究比早年更不悲觀。IMF `2024` 的工作論文指出，匯率可以同時有長期近似 random walk 的特性，也存在 `中期可預測性`；它在 `2000-2024`、`9` 個自由浮動通膨目標國家的資料上，樣本外表現優於 random walk。`2025` 的 RBA 研究也指出，用更高頻、即時可得資料時，某些 real effective exchange rates 的確具有可預測性。

所以現在更合理的說法不是「基本面完全沒用」，而是：

- `短線` 很多主要貨幣對仍很難靠宏觀模型打贏隨機漫步。
- `中期` 或特定 regime 下，基本面與利差資訊會變得比較有用。

一句話結論：`基本面不是拿來做每一根 K 線預測，而是比較像中期定錨與 regime 判斷工具。`

### 4.3 Carry / Momentum / Value 因子

這是 FX 量化最經典的一組方法：

- `Carry`：買高利率貨幣、賣低利率貨幣
- `Momentum`：買近期強勢、賣近期弱勢
- `Value`：買被低估、賣被高估

它們至今仍然是研究與實務的核心 baseline，因為：

- 結構清楚
- 好解釋
- 易做組合
- 能和風控、波動目標、風險平價結合

但要小心一個很重要的現實。`2022` 的 open-access 研究〈Are carry, momentum and value still there in currencies?〉指出，這三種貨幣 predictability 與 mispricing 有關，但在相關異常被公開、被市場學會之後，樣本外 Sharpe 明顯下降，代表 `公開 alpha 會被擠壓`。

這對今天的實務含義是：

- 這些因子仍適合當 `研究骨架`
- 但不應假設它們今天還有早年一樣的超額報酬
- 更合理的做法是把它們做成 `多因子、風險控制、動態配置`

一句話結論：`貨幣因子法仍重要，但現在比較像底盤，不像單獨就能輕鬆賺錢的引擎。`

### 4.4 統計時間序列模型

這類包括：

- `ARIMA`
- `GARCH`
- `狀態空間模型`
- `Markov / regime switching`

它們的優點不是最會賺，而是：

- 很適合當基線模型
- 很適合做波動預測
- 很適合做 regime 切換偵測

在 FX 上，很多團隊會把這類模型放在：

- 波動目標控制
- 倉位縮放
- 停損與風險限額
- 濾掉極端市場環境

也就是說，它們往往不是 alpha 主角，但很常是系統穩定性的關鍵配角。

一句話結論：`統計模型在 FX 裡最可靠的角色，通常是風控與基線，不是單打獨鬥的預言機。`

### 4.5 機器學習：Logistic、Tree-based、XGBoost 類

如果只看這兩年「比較接近實務」的研究，這一類很值得重視。

`2025` 的一篇開放取用研究，對 `8` 個貨幣對、`2018-2023` 的資料，用時間序列交叉驗證和真實交易成本測試 `7` 種模型後，得出的重點很清楚：

- 簡單、可解釋的模型不但沒有輸，反而常常更好
- 用 profit-aware 的 loss function 去選模型，比只看分類準確率更合理
- 一旦納入成本，新興市場貨幣對的可行性會急速下降
- 拿方向預測模型直接兼做風險模型，會出大問題

這篇研究中，`Logistic Regression + profit-aware optimization` 的風險調整後表現優於較複雜模型，代表現在最務實的方向之一，可能不是更深的網路，而是：

- 更好的特徵
- 更合理的損失函數
- 更嚴格的 walk-forward 驗證

一句話結論：`現時最有實務味的 FX ML，不是黑盒越深越好，而是簡單模型配合正確目標函數。`

### 4.6 深度學習：LSTM、GRU、CNN、Transformer

這是目前研究數量最多的一群。

常見原因很簡單：

- FX 是時間序列
- 深度學習擅長非線性
- 大家希望它能捕捉傳統模型抓不到的結構

系統性回顧顯示，`ANN / RNN / LSTM` 仍是文獻裡的主流。`2024` 的 Transformer 類研究在多個 NZD 相關匯率資料集上，報告 `TFT` 在預測誤差上表現最佳。另一篇 `2024` 研究也指出，把 complexity measures 加進去後，`LSTM` 與 `GRU` 在 FX 波動預測上能進一步提升準確度。

但深度學習在 FX 的真正問題不是「能不能把誤差壓低」，而是：

- 可不可以穩定樣本外
- 會不會過度交易
- 成本後是否仍划算
- 是否真的比簡單模型多賺到足夠的 alpha

目前公開證據比較像是：

- `做 forecasting` 常有幫助
- `做 trading` 是否長期更好，證據仍混合

一句話結論：`深度學習在 FX 是有研究價值的強工具，但現在還不能直接等同於更高可交易性。`

### 4.7 強化學習（RL）

RL 的吸引力很大，因為它不是只做預測，而是直接學：

- 何時進場
- 何時出場
- 倉位大小怎樣調

近兩年 RL 在金融研究很熱，`2025` 已有接受於 ACM Computing Surveys 的 survey 整理了 `167` 篇相關文獻。FX 方向也有新研究，例如 `2024` 有工作把 `risk aversion` 和 `size-dependent fees` 放進 EUR/USD 的 RL 交易框架。

RL 的優點是它理論上更接近交易決策本身；但它的難點也最大：

- reward 很容易設錯
- 模擬環境未必像真市場
- 訓練不穩定
- 很容易把微小歷史偏差學成錯誤政策

因此，現階段比較穩妥的看法是：

- `RL 很有研究前景`
- `但在 FX 上仍偏實驗性`
- 最適合拿來做 sizing / execution / policy layer 的研究，不太像已被廣泛證明的穩定主策略

一句話結論：`RL 在 FX 很前沿，但離「成熟、穩定、可放心上線」仍有距離。`

### 4.8 新聞、文本、情緒與 LLM 融合

這一類方法的想法是：`價格資料不夠，還要把新聞、評論、宏觀敘事一起讀進來。`

`2024-2025` 已經看到一些 EUR/USD 研究，把新聞文本、情緒分數、甚至 LLM 做出的文本分類結果，與 Bi-LSTM 或 LSTM 類模型融合。這些研究通常都得到一個相似結論：

- `文本 + 價格` 往往比 `只有價格` 更好

但現在這個方向有兩個限制：

1. 公開研究多集中在單一貨幣對
2. 真實交易中的延遲、資料清洗與事件去重，比論文難很多

所以比較穩健的看法是：

- 文本與 LLM 類訊號 `適合做增強`
- 目前還不適合當成「只靠文字就能穩定打 FX」的核心敘事

一句話結論：`LLM 在 FX 現在最像加分項，不像單獨能站住腳的主因子。`

### 4.9 Order Flow / Limit Order Book

這類方法看的不是「收盤價長怎樣」，而是：

- 誰在主動買
- 誰在主動賣
- 委託簿哪邊在堆量
- 流動性是不是突然抽乾

在理論上，這是最貼近短線價格形成機制的方法之一。BIS 對 FX execution algorithm 的研究也說明，現在的 FX 市場非常依賴電子執行、內部撮合和多流動池存取，這使得微結構訊號更重要，也更難手工處理。

`2024` 的一篇 LOB 研究在哥倫比亞銀行間 USD/COP 市場，用限價簿動態做秒級預測與交易，得到正面回測結果。這類研究的含義通常不是「人人都該做 LOB」，而是：

- `如果你拿得到高品質 order flow / LOB 資料，短線 edge 可能存在`
- `但如果你拿不到，這條路對多數團隊並不現實`

一句話結論：`order flow 類方法可能最強，但也最不普及，門檻遠高於一般價格型模型。`

## 5. 現時哪些方向比較值得做

如果目標是做嚴肅研究，而不是追熱門名詞，現在比較值得優先做的通常是：

### 5.1 簡單模型 + 好特徵 + 成本意識

先把：

- 線性模型
- tree-based 模型
- 少量穩定特徵
- 利差 / 波動 / regime 變數

做好，再談更複雜模型。

### 5.2 多因子而不是單因子

把 `carry + momentum + value + volatility control` 放在一起，比單押其中一個可靠。

### 5.3 Regime-aware 架構

近期研究越來越清楚地指出，FX 可預測性常常是 `分段出現` 的。也就是說，判斷「現在是不是可交易 regime」本身，就是策略的一部分。

### 5.4 Alpha 和 Risk 分開建模

`2025` 的機器學習研究明確指出，方向模型拿來兼做 VaR 風險控制會嚴重失準。這點很重要。

一句話結論：`現時最實際的路線，是把 alpha、regime、risk 當三個模組，不要想一個模型包辦全部。`

## 6. 現時最容易被高估的方向

### 6.1 純深度學習神話

很多論文能把 RMSE 做得很漂亮，但沒有把 turnover、成本、滑點與 live drift 算清楚。

### 6.2 純 RL 神話

很多 RL 成果高度依賴 reward 設計與模擬環境，一到真實市場就可能變樣。

### 6.3 單一貨幣對、短樣本、高度調參回測

這是 FX 研究最常見的幻覺來源之一。

### 6.4 只做 train/test percentage split

系統性回顧指出，早期大量研究只做簡單切分，驗證不足，結論可信度有限。

一句話結論：`在 FX，最危險的不是模型太差，而是驗證太鬆。`

## 7. 最後判斷

如果把「現時 FX 量化分析哪些算法還值得研究」濃縮成一句話：

`最值得優先研究的是：簡單可解釋 ML、經典貨幣因子、regime 模型、波動/風控模型，以及在有資料條件下的 order flow。`

`最值得保留但不要過度神化的是：深度學習、強化學習、新聞/LLM 融合。`

`最不應再做的，是忽略交易成本、忽略 regime、忽略樣本外，只追求回測好看。`

## 來源

1. BIS, `OTC foreign exchange turnover in April 2025`, 2025-09-30  
   https://www.bis.org/statistics/rpfx25_fx.htm
2. BIS Markets Committee, `FX execution algorithms and market functioning`, 2020-10-30  
   https://www.bis.org/publ/mktc13.htm
3. Ayitey Junior et al., `Forex market forecasting using machine learning: Systematic Literature Review and meta-analysis`, Journal of Big Data, 2023  
   https://doi.org/10.1186/s40537-022-00676-2
4. López-Herrera et al., `Directional forecasting for eight forex pairs against the US dollar using machine learning techniques`, Discover Artificial Intelligence, 2025  
   https://doi.org/10.1007/s44163-025-00424-4
5. Lu and Zhao, `Prediction of Currency Exchange Rate Based on Transformers`, Algorithms, 2024  
   https://doi.org/10.3390/a17080332
6. Katsiampa et al., `Forecasting Forex Market Volatility Using Deep Learning Models and Complexity Measures`, Journal of Risk and Financial Management, 2024  
   https://doi.org/10.3390/jrfm17120557
7. Bakker, `Reconciling Random Walks and Predictability: A Dual-Component Model of Exchange Rate Dynamics`, IMF Working Paper 2024/252  
   https://doi.org/10.5089/9798400295034.001
8. McCarthy and Snudden, `Forecasts of Period-average Exchange Rates: Insights from Real-time Daily Data`, RBA Research Discussion Paper 2025-09  
   https://www.rba.gov.au/publications/rdp/2025/2025-09.html
9. Beckmann, Kerkemeier, and Kruse-Becher, `Regime-specific exchange rate predictability`, Journal of Economic Dynamics and Control, 2025  
   https://doi.org/10.1016/j.jedc.2025.105095
10. Hutchinson et al., `Are carry, momentum and value still there in currencies?`, International Review of Financial Analysis, 2022  
    https://doi.org/10.1016/j.irfa.2022.102245
11. Hsu, Taylor, and Wang, `Technical Trading: Is it Still Beating the Foreign Exchange Market?`, 2024 working paper  
    https://www.cicfconf.org/sites/default/files/paper_412.pdf
12. Pippas, Ludvig, and Turkay, `The Evolution of Reinforcement Learning in Quantitative Finance: A Survey`, accepted by ACM Computing Surveys in 2025  
    https://doi.org/10.48550/arXiv.2408.10932
13. Monaco et al., `Exploiting Risk-Aversion and Size-dependent fees in FX Trading with Fitted Natural Actor-Critic`, 2024  
    https://doi.org/10.48550/arXiv.2410.23294
14. Ding et al., `EUR-USD Exchange Rate Forecasting Based on Information Fusion with Large Language Models and Deep Learning Methods`, 2024/2025  
    https://doi.org/10.48550/arXiv.2408.13214
15. Leon et al., `Deep Heterogeneous AutoML Trend Prediction Model for Algorithmic Trading in the USD/COP Colombian FX Market Through Limit Order Book (LOB)`, SN Computer Science, 2024  
    https://doi.org/10.1007/s42979-024-02930-1
