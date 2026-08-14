DMIPS（Dhrystone Million Instructions Per Second）是==衡量處理器**整數計算與邏輯控制能力**的基準測試指標==。它是透過執行標準化的 Dhrystone 程式，計算每秒鐘執行了多少個百萬條指令而得出。此指標經常用於評估微控制器與嵌入式系統的運算效能。 [1](https://www.weathink.com/news/hangye/26.html), [2](https://baike.baidu.com/item/dmips/1822619), [3](https://zhuanlan.zhihu.com/p/2027680245851891231), [4](https://www.cnblogs.com/lingjiajun/p/11833843.html)

核心運算公式與概念

- **公式計算**：\(DMIPS = \frac{\text{每秒執行Dhrystone次數}}{1757}\)
- **基準標準**：該指標是以 1970 年代的 DEC VAX 11/780 小型機（1 DMIPS）為基準進行相對比較。
- **效能評估**：單核效能通常以 DMIPS/MHz 表示，代表該晶片在 1 MHz 下每秒能執行的相對指令數。 [1](https://www.eet-china.com/mp/a208372.html), [2](https://developer.arm.com/documentation/ka001236/1-0/), [3](https://electronics.stackexchange.com/questions/517114/what-is-dmips-mhz)

常見架構的 DMIPS/MHz 效能

不同處理器架構的每兆赫茲效能各有不同： [1](https://zh.wikipedia.org/zh-tw/ARM%E8%99%95%E7%90%86%E5%99%A8%E5%85%A7%E6%A0%B8%E5%88%97%E8%A1%A8)

- **ARM Cortex-M4**：約 1.25 DMIPS/MHz
- **ARM Cortex-A7**：約 1.75 DMIPS/MHz
- **ARM Cortex-A9**：約 2.50 DMIPS/MHz
- **ARM Cortex-A53**：約 2.30 DMIPS/MHz [1](https://zh.wikipedia.org/zh-tw/ARM%E8%99%95%E7%90%86%E5%99%A8%E5%85%A7%E6%A0%B8%E5%88%97%E8%A1%A8), [2](https://electronics.stackexchange.com/questions/517114/what-is-dmips-mhz), [3](https://www.eet-china.com/mp/a208372.html)

如何計算晶片的總 DMIPS

若要計算一顆晶片的總整數算力，可將核心數、時脈（MHz）與單核的 DMIPS/MHz 數值相乘。 [1](https://www.eet-china.com/mp/a208372.html)

- **計算範例**：若某雙核心 CPU 採用 Cortex-A53 架構（2.3 DMIPS/MHz），且時脈設定為 1.6 GHz（1600 MHz）。
- **計算式**：\(2 \text{ (核)} \times 1600 MHz \times 2.3 \text{ (DMIPS/MHz)} = 7360 \text{ DMIPS}\) [1](https://www.eet-china.com/mp/a208372.html)

限制與發展

雖然它簡短且易於評估，但由於程式碼規模較小，測試結果極易受到編譯器最佳化策略的影響。在現代複雜的計算場景中，它通常僅作為參考，現代系統還會搭配評估快取與記憶體系統的 [CoreMark](https://www.eet-china.com/mp/a208372.html) 或處理浮點運算的 Whetstone 等指標進行綜合分析。