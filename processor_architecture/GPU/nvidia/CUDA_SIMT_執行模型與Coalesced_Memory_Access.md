---
title: CUDA SIMT 執行模型與 Coalesced Memory Access
source: 
author: 
published: 
created: 2026-09-17
description: 整理 CUDA warp 為單位的 SIMT 執行模型、divergence 成因，以及與其密切相關的記憶體合併存取原理
categorization: processor_architecture/GPU
tags:
  - CUDA
  - SIMT
  - warp-divergence
  - coalesced-memory-access
  - GPU-architecture
---

## 1. SIMT 是什麼、為什麼不是 SIMD

- CUDA 採用 **SIMT（Single Instruction, Multiple Threads）**：同一個 warp（32 個執行緒）在同一時刻執行相同指令，只是各自處理不同資料。
- 傳統 **SIMD（Single Instruction, Multiple Data）** 需要程式設計師明確把資料打包成向量暫存器（例如 `__m256`），指令直接對整個向量操作。
- CUDA 的差異在於「把這件事藏起來」：kernel 程式碼看起來像單一執行緒跑序列化邏輯，但硬體實際上把 32 個執行緒綁成一個 warp，共用同一個指令發射器（instruction fetch/decode），每個週期發出同一條指令，每個執行緒各自用自己的暫存器與資料位址執行。

```cuda
__global__ void add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

「程式設計模型是純量、硬體執行是向量」的落差，就是 SIMT 想強調的重點。

## 2. Warp 排程與延遲隱藏

- 一個 SM（Streaming Multiprocessor）上會同時常駐多個 active warp，但實際執行單元（CUDA core）數量有限，warp scheduler 每個週期挑一個「已就緒」的 warp 發射指令。
- 當某 warp 因記憶體存取（如 global memory load）而停滯（stall），scheduler 立刻切去執行別的 warp。
- 這是 CUDA 用「大量平行度掩蓋延遲」（latency hiding）的核心機制，與 CPU 的 out-of-order execution 是完全不同哲學：
  - CPU：靠減少延遲（cache、預測執行、亂序）
  - GPU：靠隱藏延遲（大量 warp 交錯執行）

## 3. Warp Divergence

當 warp 內出現分支且各執行緒走向不同路徑：

```cuda
if (threadIdx.x % 2 == 0) {
    // path A
} else {
    // path B
}
```

硬體用 **execution mask（active mask）** 處理：

1. 先標記走 path A 的執行緒，其餘執行緒被遮蔽（predicated off，實際跑了但結果不生效），整個 warp 一起跑過 path A 指令。
2. 切換 mask，跑過 path B 指令，這次換 A 的執行緒被遮蔽。
3. 兩段跑完後才 reconverge（Volta 之後有 Independent Thread Scheduling，reconverge 時機更靈活，但分支序列化執行的本質不變）。

**代價**：divergence 不是「效能打折」，而是「兩條路徑的執行時間直接相加」。若 A、B 各要跑 10 個週期，整個 warp 要花 20 個週期，而非理想情況下平行跑完的 10 個週期。

### 常見實務影響

- 條件式依據 `threadIdx.x` 的低位元（如奇偶判斷）幾乎必定造成同一 warp 內分裂，是最差寫法。
- 條件式依據 `blockIdx.x` 不會有 divergence 問題，因為同一 block（進而同一 warp）的所有執行緒 `blockIdx.x` 相同，要嘛整個 warp 都進 if、要嘛都進 else。
- 資料相依的迴圈次數（data-dependent loop trip count）也會造成類似 divergence，因為執行緒還沒跑完迴圈前，warp 沒辦法整體往下走。
- 邊界檢查類分支（如 `if (idx < width * height)`）通常不會造成明顯 divergence，因為同一 warp 內連續的 32 個 `idx` 幾乎都落在同一邊，只有邊界那個 warp 會分裂，影響很小。

## 4. Coalesced Memory Access（合併記憶體存取）

Warp 整體發出的 load/store 位址是否連續，直接決定實際頻寬利用率，這是與 SIMT 同一套硬體邏輯下的另一半重點。

### 4.1 基本原理

- 當一個 warp 執行記憶體存取指令時，32 個執行緒會同時各自產生一個記憶體位址。
- GPU 的記憶體控制器會嘗試把這 32 個位址「合併」成盡量少的記憶體交易（memory transaction），每筆交易通常對齊在 32 / 64 / 128 bytes 的區塊邊界（依架構而異）。
- **理想情況（fully coalesced）**：32 個執行緒存取的位址剛好是連續且對齊的一段記憶體（例如 `idx` 對應到 `array[idx]`，`idx` 是連續遞增的 global thread index），這樣硬體可以用最少的交易數（例如 1～2 筆）完成整個 warp 的存取。
- **最差情況（strided / scattered access）**：32 個執行緒的位址彼此間隔很大或完全隨機，硬體必須拆成多達 32 筆獨立交易，實際頻寬利用率可能只剩理想值的幾分之一甚至更低。

### 4.2 常見造成非合併存取的模式

- **Stride 存取**：例如 `array[idx * stride]`，當 stride 不為 1 時，每個執行緒的位址間隔擴大，交易數隨之增加。
- **結構陣列（Array of Structures, AoS）存取單一欄位**：例如

  ```cuda
  struct Point { float x, y, z; };
  Point points[N];
  // 每個 thread 只讀 points[idx].y
  ```

  因為 `x, y, z` 交錯排列，同一個 warp 讀 `y` 欄位時位址不連續，造成 strided access。改成 **Structure of Arrays（SoA）**：

  ```cuda
  float xs[N], ys[N], zs[N];
  // 每個 thread 讀 ys[idx]
  ```

  可以讓同一 warp 的存取回到連續、可合併的模式。

- **轉置（transpose）類操作**：例如矩陣轉置時，讀取是連續的（row-major 連續讀），但寫入卻是跨步的（寫到 column-major 位置），導致讀寫其中一邊必然非合併，通常透過 shared memory 做 staging 來緩解。

### 4.3 與 warp divergence 的關係與差異

- Warp divergence 影響的是「指令發射的次數與序列化」，coalesced access 影響的是「同一次記憶體指令實際要花幾筆交易」。
- 兩者可以同時發生：一段程式碼可能既有分支導致 divergence，分支內的記憶體存取又剛好是非合併模式，效能會被雙重放大打折。
- 實務上最佳化的順序通常是：先確認記憶體存取模式（是否 coalesced），再看是否有可避免的 divergence，因為記憶體頻寬問題往往是更常見的瓶頸來源。

### 4.4 與目前 jetson_multimedia_api video_decode / 後處理 kernel 的關聯

- 若後處理 kernel（色彩轉換、scaling、resize）用 `idx = blockIdx.x * blockDim.x + threadIdx.x` 存取影像 buffer，且影像是連續排列的 row-major 格式，這類存取通常天生就是 coalesced 的良好案例。
- 若牽涉到 YUV 轉 RGB 這類跨 plane 存取（例如 NV12 的 Y plane 與 UV plane 分開排列），要注意 UV plane 的存取 pattern 是否因為 subsampling（如 4:2:0）造成跨步存取，這是實務上容易忽略的非合併存取來源。

## 5. 延伸閱讀方向

- Shared memory 與 bank conflict：warp 內執行緒同時存取 shared memory 時，若多個執行緒落在同一個 bank，也會造成類似「序列化交易」的效能損失，機制與 global memory coalescing 概念上呼應但硬體實作不同。
- Occupancy（SM 上同時可容納的 warp 數量）：與 latency hiding 直接相關，register 與 shared memory 用量會限制 occupancy 上限。

