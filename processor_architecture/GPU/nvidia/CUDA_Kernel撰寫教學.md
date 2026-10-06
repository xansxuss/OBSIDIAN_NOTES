---
title: CUDA Kernel 撰寫完整教學
source: 
author: 
published: 
created: 2026-09-17
description: 從 GPU 架構、執行緒階層到記憶體最佳化的 CUDA kernel 撰寫完整教學
categorization: processor_architecture/GPU
tags:
  - CUDA
  - GPU
  - kernel
  - parallel-computing
  - CPP
---

# CUDA Kernel 撰寫完整教學

> 本筆記涵蓋 CUDA 程式設計的核心觀念，從硬體架構、執行緒模型、記憶體階層，到實際撰寫並最佳化 kernel。範例延伸自 [[video-codec-sdk-decode-bench]] 與 [[jetson-video-decode]] 這類需要高效能運算的場景，也適用於 [[license-plate-recognition]] 專案中影像前處理的 GPU 加速。

## 1. 基本概念與心智模型

### 1.1 CPU 與 GPU 的分工

CPU（Host）負責流程控制、記憶體配置、呼叫 kernel；GPU（Device）負責大量平行運算。CUDA 程式的基本流程：

1. Host 配置並初始化資料
2. 將資料從 Host 記憶體複製到 Device 記憶體
3. Host 呼叫 kernel，Device 執行平行運算
4. 將結果從 Device 複製回 Host
5. 釋放 Device 記憶體

### 1.2 SIMT 執行模型

CUDA 採用 **SIMT（Single Instruction, Multiple Threads）**：同一個 warp（32 個執行緒）中的所有執行緒，在同一時刻執行相同的指令，只是處理不同的資料。這與 CPU 的 SIMD 概念類似，但 CUDA 把「多執行緒」包裝成程式設計介面，讓每個執行緒看起來像獨立執行序列化程式碼，實際上底層硬體是以 warp 為單位排程。

理解這點很重要：**warp 內若發生分支（if/else）而各執行緒走向不同路徑，會造成 warp divergence**，因為硬體會依序執行每個分支路徑，讓不符合條件的執行緒閒置，效能因此下降。

## 2. 執行緒階層（Thread Hierarchy）

CUDA 的執行緒組織成三層：

```
Grid
 └── Block (可為 1D/2D/3D)
      └── Thread (可為 1D/2D/3D)
```

- **Thread**：最小執行單位。
- **Block**：一組執行緒，同一個 block 內的 thread 可以透過 shared memory 溝通，並用 `__syncthreads()` 同步。
- **Grid**：一個 kernel 呼叫產生的所有 block 集合。

### 2.1 內建變數

在 kernel 內可以直接使用以下內建變數計算全域索引：

| 變數 | 意義 |
|---|---|
| `threadIdx.x/y/z` | 目前執行緒在 block 內的座標 |
| `blockIdx.x/y/z` | 目前 block 在 grid 內的座標 |
| `blockDim.x/y/z` | 每個 block 內的 thread 數量 |
| `gridDim.x/y/z` | grid 內的 block 數量 |

計算「全域索引」（處理一維陣列時最常見）：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

這行的邏輯：`blockIdx.x * blockDim.x` 算出目前 block 之前，已經有多少個 thread 被分配掉了，再加上 `threadIdx.x`（本 block 內的偏移量），就得到這個 thread 在整個 grid 中對應到第幾筆資料。

## 3. 第一個 Kernel：向量相加

```cpp
// kernel 定義：__global__ 代表由 Host 呼叫、在 Device 上執行
__global__ void vecAdd(const float* A, const float* B, float* C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 因為 thread 數量通常會多於資料數量（無法整除的情況），
    // 一定要做邊界檢查，避免存取超出陣列範圍的記憶體
    if (idx < n)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    const int n = 1 << 20; // 約 100 萬筆
    size_t bytes = n * sizeof(float);

    // Host 記憶體
    float* h_A = new float[n];
    float* h_B = new float[n];
    float* h_C = new float[n];
    // ... 初始化 h_A, h_B ...

    // Device 記憶體
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Host -> Device 複製
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 決定 block/grid 大小
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // 呼叫 kernel：<<<gridDim, blockDim>>>
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // Device -> Host 複製（會自動同步等待 kernel 執行完成）
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
}
```

**邏輯重點說明：**

- `(n + threadsPerBlock - 1) / threadsPerBlock` 是「無條件進位」的整數除法寫法，確保就算 n 不是 256 的倍數，也會配置足夠的 block 涵蓋所有資料。
- `<<<blocksPerGrid, threadsPerBlock>>>` 這個三角括號語法是 CUDA 對 C++ 的擴充，用來指定 kernel 啟動時的 grid/block 維度。
- `cudaMemcpy` 呼叫預設是同步（blocking）的，因此不需要額外呼叫 `cudaDeviceSynchronize()` 才能確保 kernel 執行完畢；但如果要精確計時或使用 stream，仍需明確同步。

## 4. 記憶體階層（Memory Hierarchy）

這是 CUDA 效能最佳化最關鍵的一環：

| 記憶體類型 | 範圍 | 速度 | 生命週期 |
|---|---|---|---|
| Register | 單一 thread | 最快 | kernel 執行期間 |
| Shared Memory | 同一 block | 快（接近 L1 cache） | block 執行期間 |
| Local Memory | 單一 thread（實體上位於 global memory） | 慢 | kernel 執行期間 |
| Global Memory | 整個 grid | 慢（但頻寬大） | 整個程式生命週期 |
| Constant Memory | 整個 grid，唯讀 | 快（有 cache） | 整個程式生命週期 |
| Texture Memory | 整個 grid，唯讀 | 快（針對空間局部性最佳化） | 整個程式生命週期 |

### 4.1 Global Memory Coalescing（合併存取）

當同一個 warp 內的 32 個 thread 存取 global memory 時，如果位址是連續且對齊的，硬體可以把這些請求合併成少數幾筆記憶體交易，大幅提升頻寬利用率。反之，若存取模式跳躍（stride 很大）或不對齊，會拆成多筆交易，效能大幅下降。

```cpp
// 良好的合併存取：thread i 存取 data[i]，位址連續
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float val = data[idx];

// 不良存取：跳躍式存取（stride = width），容易造成非合併存取
float val2 = data[idx * width];
```

### 4.2 Shared Memory 與 Tiling：矩陣乘法範例

Shared memory 由同一個 block 內的所有 thread 共用，可用來暫存重複使用的資料，減少對 global memory 的重複讀取。以下是「分塊（tiling）」矩陣乘法的經典範例：

```cpp
#define TILE_SIZE 16

__global__ void matMulTiled(const float* A, const float* B, float* C, int N)
{
    // __shared__ 宣告的陣列，會配置在該 block 的 shared memory 中
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // 把 N x N 的矩陣切成多個 TILE_SIZE x TILE_SIZE 的區塊，逐塊搬運計算
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t)
    {
        // 每個 thread 負責搬一個元素進 shared memory
        tileA[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE_SIZE + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];

        // 關鍵：所有 thread 都要等 tile 資料搬運完成才能開始計算，
        // 否則會有 thread 讀到還沒被寫入的 shared memory
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
        {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // 計算完這個 tile 後，要等所有 thread 都用完 shared memory，
        // 才能讓下一輪迴圈覆寫 tileA/tileB，避免資料競爭
        __syncthreads();
    }

    if (row < N && col < N)
    {
        C[row * N + col] = sum;
    }
}
```

**邏輯重點說明：**

- 樸素版矩陣乘法中，每個輸出元素需要讀取一整列 A 與一整行 B，若矩陣很大，同樣的資料會被重複從 global memory 讀取非常多次。
- Tiling 的做法是：讓同一個 block 的 thread 合作，把一小塊 A、B 搬進 shared memory（一次搬運，多次使用），再用這塊資料算出對應的部分結果，逐塊累加。
- 兩次 `__syncthreads()` 缺一不可：第一次確保「資料搬完才能算」，第二次確保「這輪算完才能覆寫」。這是典型的 producer-consumer 同步模式。

## 5. 執行緒同步與 Atomic 操作

### 5.1 `__syncthreads()`

只能同步**同一個 block 內**的所有 thread，讓它們在同一個時間點都執行到這行才繼續往下走。不同 block 之間無法用這個方式同步（除非使用 cooperative groups 或分開多次 kernel 呼叫）。

### 5.2 Atomic 操作：Reduction 範例

當多個 thread 需要同時寫入同一塊記憶體（例如全域加總）時，必須使用 atomic 操作避免資料競爭（race condition）：

```cpp
__global__ void sumReduce(const float* input, float* result, int n)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 每個 thread 先把自己負責的資料載入 shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // block 內部做樹狀（tree-based）reduction：
    // 每一輪讓一半的 thread 把另一半的資料加進來，s 每輪減半
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 每個 block 的加總結果，只由該 block 的 0 號 thread 負責寫回，
    // 並用 atomicAdd 累加到全域結果，避免多個 block 同時寫入造成資料遺失
    if (tid == 0)
    {
        atomicAdd(result, sdata[0]);
    }
}
```

**邏輯重點說明：**

- 樹狀 reduction 的時間複雜度是 O(log n)，比起單一 thread 逐一累加的 O(n) 快很多。
- `s >>= 1` 等同於 `s /= 2`，每輪讓有效工作的 thread 數量減半，直到 s 為 0。
- `atomicAdd` 保證多個 block 同時執行「讀取-修改-寫回」時不會互相覆蓋彼此的結果，但因為是硬體序列化執行，過度使用 atomic 會造成效能瓶頸，通常只在最後彙總階段使用。

## 6. Kernel 啟動設定與 Occupancy

### 6.1 如何選擇 block size

- 每個 block 中的 thread 數量建議為 32 的倍數（因為一個 warp 是 32 個 thread，非 32 倍數會造成浪費）。
- 常見選擇：128、256、512。
- 實際最佳值需視 kernel 使用的暫存器數量與 shared memory 大小而定，這會影響「occupancy」（SM 上同時能駐留的 warp 比例）。

### 6.2 限制因素

一個 SM（Streaming Multiprocessor）能同時執行的 block/warp 數量，受限於三個資源：

1. Register 數量（每個 thread 用掉的暫存器越多，同時能跑的 thread 越少）
2. Shared memory 容量（每個 block 用掉的 shared memory 越多，同時能跑的 block 越少）
3. 硬體本身對每個 SM 的 block/warp 數量上限

可以用 `nvcc --ptxas-options=-v` 編譯查看每個 kernel 使用的暫存器與 shared memory 數量，或用 CUDA Occupancy Calculator / Nsight Compute 分析。

## 7. 錯誤檢查

CUDA API 呼叫大多回傳 `cudaError_t`，務必檢查回傳值，否則錯誤會被吃掉導致難以除錯：

```cpp
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                          \
        }                                                                 \
    } while (0)

// 使用方式
CUDA_CHECK(cudaMalloc(&d_A, bytes));
```

kernel 呼叫本身不會回傳 `cudaError_t`（因為 `<<<>>>` 語法是非同步的），要檢查 kernel 是否啟動成功、以及是否執行時發生錯誤，需要額外呼叫：

```cpp
vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
CUDA_CHECK(cudaGetLastError());       // 檢查啟動階段的錯誤（如設定參數不合法）
CUDA_CHECK(cudaDeviceSynchronize());  // 等待執行完成，並檢查執行期間的錯誤
```

## 8. Streams 與非同步執行

CUDA Stream 是一個「命令佇列」，同一個 stream 內的操作依序執行，不同 stream 之間可以並行（例如：一邊做資料搬運、一邊做運算），達到重疊（overlap）的效果。

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 使用 pinned memory（cudaMallocHost）才能真正做到非同步搬運與運算重疊
cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, stream1);
kernelA<<<grid, block, 0, stream1>>>(d_A, ...);

cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice, stream2);
kernelB<<<grid, block, 0, stream2>>>(d_B, ...);

cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);

cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

這在 [[video-codec-sdk-decode-bench]] 這類需要同時處理解碼與後處理的 pipeline 中特別有用：可以讓 GPU 在跑上一批資料的運算時，同時搬運下一批資料，隱藏 PCIe 傳輸延遲。

## 9. 編譯與執行

```bash
nvcc -O3 -arch=sm_86 main.cu -o main
./main
```

- `-arch=sm_86` 指定目標 GPU 的計算能力（Compute Capability），需依實際硬體調整（例如 RTX 40 系列為 `sm_89`，需查閱對應表）。
- `-O3` 開啟主機端程式碼最佳化；device 端程式碼最佳化則多半由 `nvcc` 內建的 `ptxas` 負責，可用 `-Xptxas -O3` 明確指定。

## 10. 常見陷阱整理

- **忘記邊界檢查**：thread 數量常大於資料量，未檢查 `idx < n` 會存取越界記憶體。
- **Warp divergence**：迴圈或條件判斷讓同一 warp 內的 thread 走不同分支，序列化執行造成效能下降。
- **忘記 `__syncthreads()`**：在使用 shared memory 的 producer-consumer 模式中漏放同步點，會讀到未寫入或已被覆寫的資料。
- **Bank conflict**：多個 thread 同時存取 shared memory 中同一個 memory bank 的不同位址，會造成存取序列化；可透過調整陣列維度（例如 padding）避開。
- **未使用 pinned memory**：一般 `malloc` 配置的 host 記憶體無法讓 `cudaMemcpyAsync` 真正非同步，需用 `cudaMallocHost` 配置。
- **過度使用 atomic**：所有 thread 集中對同一個位址做 atomic 操作，會造成硬體序列化，成為效能瓶頸。

## 11. 延伸方向

- Warp-level primitives（`__shfl_sync` 等），可在不使用 shared memory 的情況下於 warp 內交換資料。
- Cooperative Groups，提供比傳統 block/grid 更彈性的同步範圍。
- Tensor Core 與混合精度運算，適用於深度學習相關的矩陣運算加速，可與 [[license-plate-recognition]] 專案的模型推論加速方向連結。
- Nsight Systems / Nsight Compute 效能分析工具的實際操作。
