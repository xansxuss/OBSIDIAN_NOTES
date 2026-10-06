---
title: CUDA C++ Programming Guide - Ch.2 Programming Model
source: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html
author: NVIDIA Corporation
published: 2024-10-01
created: 2026-09-22
description: CUDA 12.6 官方指南第2章，涵蓋 Kernel、Thread Hierarchy、Memory Hierarchy、Heterogeneous Programming 與 Compute Capability
categorization: processor_architecture/GPU/nvidia/CUDA/Guide
tags:
  - CUDA
  - GPU程式設計
  - Thread
  - Memory
  - Kernel
  - NVIDIA
  - 官方文件筆記
---

# Ch.2 Programming Model

> 版本：CUDA C++ Programming Guide 12.6
> 官方 PDF：[CUDA_C_Programming_Guide.pdf](https://docs.nvidia.com/cuda/archive/12.6.2/pdf/CUDA_C_Programming_Guide.pdf)
> 上一章：[[CUDA-Programming-Guide-01-Introduction]]
> 下一章：[[CUDA-Programming-Guide-03-ProgrammingInterface]]

---

## 2.1 Kernels（核心函式）

- CUDA C++ 用 `__global__` 修飾詞定義 **kernel**：
  - 呼叫一次，由 **N 個不同的 CUDA thread 平行執行 N 次**（一般 C++ 函式只執行一次）
- 每個 thread 有唯一 thread ID，透過內建變數（如 `threadIdx`）在 kernel 內取得
- 用 `<<<gridDim, blockDim>>>` **執行配置語法**（execution configuration）指定啟動幾個 thread

```cpp
// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    // ...
    // 啟動 1 個 block，每個 block 含 N 個 thread
    VecAdd<<<1, N>>>(A, B, C);
    // ...
}
```

**邏輯說明**：
- `VecAdd<<<1, N>>>(...)` → 1 個 block、N 個 thread，共 N 個 thread 並行執行
- 每個 thread 透過 `threadIdx.x` 取得自己的索引 `i`
- 每個 thread 只負責 `C[i] = A[i] + B[i]` 這一筆，N 筆加法被平行拆給 N 個 thread 同時完成

---

## 2.2 Thread Hierarchy（執行緒階層）

### 核心概念

- `threadIdx` 是**三維向量**（x, y, z），thread 可用 1D/2D/3D 方式組成 **thread block**
- 對應向量（1D）、矩陣（2D）、體積（3D）等資料結構

### Thread ID 計算公式

| Block 維度 | Thread ID |
|---|---|
| 1D（大小 Dx） | `x` |
| 2D（大小 Dx, Dy） | `x + y·Dx` |
| 3D（大小 Dx, Dy, Dz） | `x + y·Dx + z·Dx·Dy` |

### 硬體限制

- 每個 thread block **最多 1024 個 thread**（同一 block 內 thread 必須共用同一 SM 的有限資源）
- 多個 block 組成一個 **grid**（1D/2D/3D）
- grid 內 block 數量通常由**資料量大小**決定
- Block 索引 → `blockIdx`，Block 維度 → `blockDim`

### 多 Block 矩陣加法範例

```cpp
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
    // 全域座標 = block 偏移 + block 內偏移
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)          // 邊界檢查，防止越界存取
        C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    // ...
    dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads/block（常用選擇）
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    // ...
}
```

**邏輯說明**：
- `blockIdx.x * blockDim.x + threadIdx.x` → 把「第幾個 block」乘上「每 block 有幾個 thread」再加上「block 內第幾個 thread」，換算出全域矩陣座標 `(i, j)`
- `if (i < N && j < N)` 邊界檢查：若 N 不是 16 的倍數，最後一列/欄的 block 會有部分 thread 超出陣列範圍，需要用 if 擋掉避免記憶體越界

> 📎 **Figure 4 - Grid of Thread Blocks**
> Grid → Block → Thread 三層結構示意圖，含 blockIdx/threadIdx 索引對應關係
> 原始圖表：[官方 PDF 第 13 頁](https://docs.nvidia.com/cuda/archive/12.6.2/pdf/CUDA_C_Programming_Guide.pdf#page=13)

### 重要原則

- **Thread block 必須能獨立執行**：任意順序、平行或序列都要正確，這是讓 CUDA 程式能自動 scale 到不同 SM 數量 GPU 的根基
- Block 內 thread 可透過 **shared memory** 共享資料
- 用 `__syncthreads()` 做**屏障同步**（block 內所有 thread 都到達這行才能繼續）

---

### 2.2.1 Thread Block Clusters（CC 9.0+ 新增，Hopper 架構）

> ⚠️ RTX 4070 SUPER（Ada, CC 8.9）**不支援**此功能，需 CC 9.0 Hopper 以上。

- 在 thread block 之上新增一層可選階層：**Cluster**
- 同一 cluster 內的 block 保證排在同一個 **GPC**（GPU Processing Cluster）上執行
- Cluster 最多支援 **8 個 thread block**（portable cluster size）
- Cluster 內 block 可互相存取彼此的 shared memory → 稱為 **Distributed Shared Memory**

```cpp
// 啟用方式 1：編譯期，X 方向 2 個 block
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output)
{
}

// 啟用方式 2：執行期
cudaLaunchConfig_t config = {0};
config.gridDim = numBlocks;
config.blockDim = threadsPerBlock;

cudaLaunchAttribute attribute[1];
attribute[0].id = cudaLaunchAttributeClusterDimension;
attribute[0].val.clusterDim.x = 2;
attribute[0].val.clusterDim.y = 1;
attribute[0].val.clusterDim.z = 1;
config.attrs = attribute;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, cluster_kernel, input, output);
```

> 📎 **Figure 5 - Grid of Thread Block Clusters**
> Grid / Cluster / Block / Thread 四層結構示意圖（CC 9.0+）
> 原始圖表：[官方 PDF 第 15 頁](https://docs.nvidia.com/cuda/archive/12.6.2/pdf/CUDA_C_Programming_Guide.pdf#page=15)

---

## 2.3 Memory Hierarchy（記憶體階層）

| 記憶體類型 | 可見範圍 | 生命週期 | 速度 |
|---|---|---|---|
| **Local memory** | 單一 thread 私有 | 該 thread | 慢（實際在 global memory） |
| **Shared memory** | 同一 block 內所有 thread | 該 block | 快（近似 L1 cache） |
| **Global memory** | 所有 thread | 整個應用程式（跨 kernel 持續） | 慢 |
| **Constant memory** | 所有 thread，唯讀 | 整個應用程式 | 快（有快取） |
| **Texture memory** | 所有 thread，唯讀 | 整個應用程式 | 快（有快取，支援特殊定址/過濾） |

> 📎 **Figure 6 - Memory Hierarchy**
> Thread / Block / Grid 各層對應的記憶體空間示意圖
> 原始圖表：[官方 PDF 第 16 頁](https://docs.nvidia.com/cuda/archive/12.6.2/pdf/CUDA_C_Programming_Guide.pdf#page=16)

**補充**：
- CC 9.0+ 的 cluster 內，thread block 可透過 **Distributed Shared Memory** 互相存取彼此的 shared memory
- Global / Constant / Texture memory 跨 kernel 呼叫**持續存在**（persistent）

---

## 2.4 Heterogeneous Programming（異質運算）

- CUDA 假設 **host（CPU）** 與 **device（GPU）** 是兩個各自擁有獨立 DRAM 的處理器
- GPU 作為 host 的**協同處理器（coprocessor）**
- Host 程式需透過 **CUDA runtime API** 明確管理：
  - Device 記憶體的配置（`cudaMalloc`）與釋放（`cudaFree`）
  - Host ↔ Device 之間的資料搬移（`cudaMemcpy`）

> 📎 **Figure 7 - Heterogeneous Programming**
> Host/Device 分離式記憶體架構，串行碼在 CPU、平行碼在 GPU 執行示意圖
> 原始圖表：[官方 PDF 第 18 頁](https://docs.nvidia.com/cuda/archive/12.6.2/pdf/CUDA_C_Programming_Guide.pdf#page=18)

### Unified Memory（統一記憶體）

- 提供橫跨 host/device 的**共同位址空間（common address space）**
- CPU/GPU 都能存取同一份 managed memory，省去手動搬資料
- 細節見 [[CUDA-Programming-Guide-19-UnifiedMemory]]

---

## 2.5 Asynchronous SIMT Programming Model（非同步 SIMT 程式設計模型）

> 從 **NVIDIA Ampere 架構**開始支援（CC 8.0+）

- **非同步操作（asynchronous operation）**：由一個 CUDA thread 發起，但「像是」由另一個 thread 執行完成
- 需透過**同步物件**等待完成：
  - `cuda::barrier`
  - `cuda::pipeline`

### Thread Scope（同步範圍）

| Scope | 說明 |
|---|---|
| `thread_scope_thread` | 只有發起的那個 thread 自己同步 |
| `thread_scope_block` | 同一 thread block 內的 thread 同步 |
| `thread_scope_device` | 同一 GPU 裝置內的 thread 同步 |
| `thread_scope_system` | 同一系統內所有 CUDA/CPU thread 同步 |

> 這部分是進階效能優化機制（如 `memcpy_async`），細節見 [[CUDA-Programming-Guide-03-ProgrammingInterface]]

---

## 2.6 Compute Capability（運算能力版本）

- 每張 GPU 有一個 Compute Capability，格式為 `X.Y`（主版號.次版號）
- 用來識別 GPU 支援的硬體功能與指令集，可在 runtime 查詢

### 主版號對應架構

| 主版號 | 架構 |
|---|---|
| 9 | Hopper |
| 8 | Ampere（8.6 = Ampere desktop，8.9 = Ada Lovelace） |
| 7 | Volta（7.5 = Turing） |
| 6 | Pascal |
| 5 | Maxwell |
| 3 | Kepler |

> ⚠️ **Compute Capability ≠ CUDA 版本號**
> - CC 是**顯卡硬體**的版本（如 RTX 4070 SUPER = CC 8.9）
> - CUDA 12.6 是**軟體平台**的版本，兩者互相獨立

---

## 小結

- **Grid → Block → Thread** 三層結構 + `gridDim` / `blockIdx` / `blockDim` / `threadIdx` 四個內建變數 → 是所有 CUDA kernel 寫法的核心
- Thread block 之間互相獨立、順序不保證 → 是 CUDA 能 scale 到不同 GPU 規模的關鍵設計
- 記憶體階層（local / shared / global / constant / texture）的存取範圍與速度差異 → 直接影響效能優化策略（見 [[CUDA-Programming-Guide-05-PerformanceGuidelines]]）
- Cluster（CC 9.0+）與 Asynchronous SIMT（CC 8.0+）是進階功能，基礎開發不需要優先掌握

## 待整理

- [ ] [[CUDA-Programming-Guide-03-ProgrammingInterface]]（NVCC 編譯流程、CUDA Runtime API）
- [ ] [[CUDA-Programming-Guide-04-HardwareImplementation]]（SIMT Architecture、Hardware Multithreading）
- [ ] [[CUDA-Programming-Guide-05-PerformanceGuidelines]]（效能最大化策略）
