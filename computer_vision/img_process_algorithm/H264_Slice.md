1. 基本概念
在 H.264 / AVC (Advanced Video Coding) 標準裡，一個 slice 是一個圖像（frame）或場（field）的一部分，它可以獨立解碼。換句話說，slice 是影像的一個“最小可解碼單位”。
- Frame / Picture → 可以包含多個 slice
- Slice → 包含一組 macroblock (通常是 16x16 pixels)
- Macroblock → 編碼的基本單位，包含亮度和色度資訊

好處：
- 可以 分段解碼（parallel decoding / error resilience）
- 有利於網路傳輸時的 丟包容忍
- 適合 多核心/GPU 並行解碼

2. Slice 類型
H.264 定義了幾種 slice：
   1. I-slice (Intra slice)
    - 全部 macroblock 都是 Intra-coded
    - 不依賴其他 slice 的資料
    - 適合新 GOP (Group of Pictures) 開頭
   2. P-slice (Predictive slice)
    - 巨集區塊依賴 前一個 frame 的資料
    - 進行運動補償 (motion compensation)
   3. B-slice (Bi-directional slice)
    - 巨集區塊可以依賴 前後兩個 frame 的資料
    - 壓縮效率最高，但解碼延遲較大
   4. SP / SI slices
   - 專門用於特定場景或切換 (Switching / Scene Change)
   - 不常用於一般播放
3. Slice 與 Macroblock 的關係
一個 slice 可以是整張 frame，也可以只是一部分：

``` bash
Frame
 ├─ Slice 0
 │   ├─ Macroblock 0
 │   ├─ Macroblock 1
 │   └─ …
 └─ Slice 1
     ├─ Macroblock 0
     ├─ Macroblock 1
     └─ …
```

注意： slice 可以跨行或跨列排列 macroblock，這取決於 slice 結構設定（slice_map）。

4. 為什麼要切 slice？
- 容錯：傳輸途中丟掉一個 slice，不會影響整張圖
- 並行解碼：GPU / 多核 CPU 可以同時解碼不同 slice
- 位元率控制：可以精確控制每個 slice 的位元量

5. 在 H.264 bitstream 中
Slice 是 NAL (Network Abstraction Layer) 單位的一部分：

```bash
[Start code][NAL header][Slice header][Macroblock data]
```

- NAL header：告訴 slice 的類型
- Slice header：包含 slice id、frame number、QP、motion vector info 等
- Macroblock data：實際的壓縮資訊

[H264 Slice（片）概念詳解](https://blog.csdn.net/u011487024/article/details/153402405)
這篇文章深入探討了 H.264 視訊編碼中的「Slice（片）」概念，並詳細說明了其在編碼過程中的作用、種類、封裝方式、優缺點以及實際應用場景。

🔍 Slice 是什麼？

在 H.264 中，Slice 是將一幀影像劃分為多個獨立編碼區域的基本單位。每個 Slice 由多個巨集區塊（Macroblock）組成，並且其編碼過程相互獨立。這樣的設計有助於錯誤隔離和並行處理。

🧩 Slice 的主要作用
1. 錯誤隔離：每個 Slice 的編碼過程相互獨立，當某個 Slice 出現錯誤時，錯誤被限制在該 Slice 內部，減少花屏的範圍。
2. 並行處理：多個 Slice 可以同時進行編解碼，提高處理速度，支持多線程並行處理。
3. 靈活分片：根據網路狀況和編碼需求靈活分片，實現更精細的碼率控制和質量優化。

🧪 Slice 的類型
- I Slice（帧内编码片）：僅包含 I 巨集區塊，使用帧內預測進行編碼，無需依賴其他幀的數據，常用於關鍵幀。
- P Slice（單向帧間编码片）：包含 P 巨集區塊或 I 巨集區塊，使用前向參考幀進行預測，依賴前面的幀，常用於普通預測幀。
- B Slice（雙向帧間编码片）：包含 B 巨集區塊或 I 巨集區塊，使用前向和後向參考幀進行預測，依賴前後幀，壓縮效率最高。
- SP Slice（切換 P 片）：包含 P 巨集區塊或 I 巨集區塊，用於視頻流之間的高效切換，應用於碼流切換、網路適應等場景。
- SI Slice（切換 I 片）：僅包含 SI 巨集區塊，用於視頻流之間的高效切換，應用於碼流切換、錯誤恢復等場景。

📦 在 NALU 中的封裝
Slice 被封裝在 NALU（Network Abstraction Layer Unit）中，NALU 結構包括 NALU 頭部和 NALU 載荷。Slice 的類型在 NALU 中由 nal_unit_type 字段表示，例如：

- IDR Slice：值為 5，即時解碼刷新片。
- P Slice：值為 1，預測片。
- B Slice：值為 2，雙向預測片。
- SP Slice：值為 3，切換 P 片。
- SI Slice：值為 4，切換 I 片。

⚖️ 優缺點分析
優點：

- ✅ 錯誤隔離：限制錯誤傳播範圍，提高容錯能力。
- ✅ 並行處理：支持多線程編碼，提高處理速度。
- ✅ 靈活分片：自適應網路狀況，精細碼率控制。

缺點：

- ❌ 增加開銷：每個 Slice 需要頭部信息，增加碼流大小。
- ❌ 降低效率：減少可參考信息，可能降低壓縮效率。
- ❌ 複雜度增加：編碼器複雜度提高，解碼器需要處理分片。

🌐 實際應用場景
- 網路傳輸：錯誤恢復、漸進傳輸、重傳機制。
- 實時編碼：並行編碼、延遲控制、質量平衡。
- 存儲優化：隨機訪問、部分解碼、容錯存儲。

🧠 最佳實踐建議
- 分片策略：根據網路狀況和內容複雜度調整 Slice 數量，平衡錯誤隔離和編碼效率。
- 編碼優化：合理設置 Slice 大小，考慮並行度，優化預測模式。
- 解碼優化：充分利用多核處理器，實現 robust 的錯誤恢復機制，合理管理內存。

🔚 總結
H.264 中的 Slice 概念是現代視訊編碼的重要特性，它通過將幀分割為獨立的編碼單元，實現了錯誤隔離、並行處理和靈活控制。雖然 Slice 分片會增加一定的編碼開銷，但在網路傳輸、實時編碼和存儲優化等場景中，其帶來的優勢遠大於成本。合理使用 Slice 分片技術，可以顯著提升 H.264 編碼系統的性能和可靠性。