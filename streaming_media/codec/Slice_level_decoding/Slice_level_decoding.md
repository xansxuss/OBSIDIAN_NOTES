---
title: "Slice_level_decoding"
source: "https://gemini.google.com/app/2484fcb5e897af64?hl=zh-TW"
author:
published:
created: 2026-08-27
description: "Gemini conversation with 2 messages"
tags:
  - "clippings"
---
**Slice Level Decoding** 是影像與視訊編解碼技術（如 H.264/AVC、H.265/HEVC、H.266/VVC 及 JPEG/JPEG 2000）中的一個關鍵架構層級。

視訊碼流（Bitstream）的階層結構通常為： **Sequence (序列) → Picture/Frame (影格) → Slice (切片) → Coding Tree Unit/Macroblock (編碼單元/巨區塊) → Block (方塊)** 。

Slice Level Decoding 即為針對「Slice」這一層級所進行的標頭解析（Header Parsing）與資料解碼過程。

### 1\. Slice 的核心概念與目的

Slice 是影像中一個獨立可解碼的區域，包含數個連續或特定區域的 Macroblock (MB) 或 Coding Tree Unit (CTU)。

- **錯誤隔離（Error Resiliency）：** Slice 之間是相互獨立的。若某個 Slice 在傳輸過程中封包遺失或損壞，解碼器仍能獨立解碼其他 Slice，防止錯誤擴散至整頁影格。
- **平行處理（Parallel Processing）：** 多核心 CPU/GPU 可將同一 Frame 切分為多個 Slice，並分派給不同執行緒同時進行 Slice Level Decoding。
- **傳輸單元匹配（MTU Matching）：** 在網路傳輸（如 RTP/UDP）時，通常會將 Slice 的大小限制在網路 MTU 範圍內，避免 IP 封包被拆分。

### 2\. Slice Level Decoding 的執行流程

Slice 解碼主要分為兩個階段： **Slice Header 解析** 與 **Slice Data 解碼** 。

#### 階段一：Slice Header Parsing（標頭解析）

解碼器首先讀取 Slice Header 的語法元素（Syntax Elements），以取得該 Slice 的解碼 Context：

1. **Slice Type（切片類型）：**
	- **I-Slice：** 僅使用幀內預測（Intra Prediction）。
		- **P-Slice：** 使用單向幀間預測（Inter Prediction）。
		- **B-Slice：** 使用雙向幀間預測。
2. **Frame Number / POC (Picture Order Count)：** 確認目前 Slice 屬於哪一個 Picture 以及顯示順序。
3. **Reference Picture List Construction：** 建立參考影格清單（RefPicList0, RefPicList1），供 P/B Slice 進行運動補償。
4. **Quantization Parameter (QP) Delta：** 取得該 Slice 的基礎量化參數偏移量。
5. **Deblocking Filter Parameters：** 去區塊濾波器的配置參數。

#### 階段二：Slice Data Decoding（內容解碼）

解析完 Header 後，解碼器進入迴圈，依序處理 Slice 內的所有 Block：

1. **Entropy Decoding（熵解碼）：** 使用 CABAC 或 CAVLC 將 Bitstream 解碼為殘差係數（Residual Coefficients）、運動向量（Motion Vectors, MVs）與預測模式。
2. **Inverse Quantization & Inverse Transform (IQ/IT)：** 將殘差係數進行反量化與反轉換（如 IDCT），還原出空間域的殘差訊號。
3. **Prediction Reconstruction（預測重建）：**
	- **Intra：** 根據周邊已解碼像素進行幀內預測。
		- **Inter：** 根據 Motion Vector 與 Reference Picture List 進行幀間運動補償。
4. **Reconstruction：** 將預測訊號與殘差訊號相加，重建原始像素。
5. **In-loop Filtering（環路濾波）：** 執行 Deblocking Filter、SAO (Sample Adaptive Offset) 等，減少邊界區塊效應。

### 3\. 不同編碼標準的比較

| 特性 | H.264 / AVC | H.265 / HEVC | H.266 / VVC |
| --- | --- | --- | --- |
| **基本結構單元** | Macroblock (16x16) | CTU (最大 64x64) | CTU (最大 128x128) |
| **平行機制** | Slice | Slice, Tile, Wavefront (WPP) | Slice, Tile, Subpicture |
| **Slice 獨立性** | 完全獨立（不跨 Slice 參考） | 可搭配 Dependent Slice Header | 引進 Subpicture，增強區域裁切解碼 |

### 4\. 工程實作注意事項

- **Context State 重置：** 熵解碼器（如 CABAC）在每個 Slice 開頭都必須重置（Reset）其機率模型狀態，確保解碼獨立性。
- **記憶體配置：** 多執行緒解碼時，Slice Level Decoding 需要注意 Reference Frame 的 Read Lock 機制，避免 Race Condition。
- **無 STL / 低延遲需求（如 C++ 實作）：** 在高吞吐量的多媒體管線中，Slice Header 的解析通常採用 bit-stream reader 進行 Shift & Mask 操作，避免動態記憶體配置（Zero-Allocation）。