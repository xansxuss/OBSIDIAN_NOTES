---
title: "CUDA kernel學習路線圖"
source: "https://claude.ai/chat/e58c11d8-226f-4cee-99b8-d120955e3f47"
author:
published:
created: 2026-07-06
description: "Claude conversation with 2 messages"
tags:
  - "clippings"
---
CUDA kernel 的學習路線圖，由淺入深:

### 階段一:基礎觀念與環境建置

- 了解 GPU 與 CPU 架構的差異(SIMT 模型、記憶體階層、warp 概念)
- 安裝 CUDA Toolkit,確認 `nvcc` 編譯環境正常
- 寫第一個 kernel:`vector add`,理解 `<<<blocks, threads>>>` 的啟動語法
- 搞懂 `threadIdx` 、 `blockIdx` 、 `blockDim` 、 `gridDim` 這幾個內建變數怎麼算出全域索引

### 階段二:記憶體模型

這塊是 CUDA 效能的核心,建議花較多時間:

- Global memory、Shared memory、Register、Constant memory、Texture memory 的差異與用途
- Coalesced memory access(合併存取)的原理,這直接影響頻寬利用率
- Shared memory 的 bank conflict 問題
- 練習題:矩陣乘法(先寫 naive 版本,再用 shared memory 做 tiling 優化,比較效能差異)

### 階段三:同步與執行模型

- `__syncthreads()` 的用途與限制
- Warp 的概念、warp divergence 對效能的影響
- Occupancy(佔用率)的概念,如何用 register 數量與 shared memory 用量去估算
- 用 Nsight Compute 或 `nvprof` / `nsys` 實際量測 kernel 效能瓶頸

### 階段四:進階優化技巧

- Loop unrolling、指令層級平行化
- Warp-level primitives(如 `__shfl_sync` 系列函式),減少對 shared memory 的依賴
- Atomic operations 與其效能代價
- Stream 與非同步操作,理解如何讓運算與資料傳輸重疊(overlap)
- Multi-GPU 的基本概念(如果有需求的話)

### 階段五:實戰專案

建議挑幾個經典題目動手做,每個都能對應到不同優化技巧:

- Reduction(歸約運算,學習 tree-based reduction 與 warp shuffle 優化)
- Histogram(學習 atomic 操作與 shared memory 私有化技巧)
- Convolution / Stencil 計算(學習 shared memory tiling 與 halo 區處理)
- 有餘力的話可以研究 cuBLAS 或 cuDNN 內部某些 kernel 的實作思路,理解業界做法

### 學習資源建議

- NVIDIA 官方的《CUDA C++ Programming Guide》一定要讀,是最權威的參考資料
- Mark Harris 的 reduction 優化系列文章,是經典入門教材
- 《Programming Massively Parallel Processors》這本書系統性很強,推薦當教科書讀