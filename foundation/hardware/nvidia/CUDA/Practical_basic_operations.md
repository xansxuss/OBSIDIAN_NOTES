在實作 CUDA 核心運算時，理解「執行緒配置（Thread Layout）」與「記憶體存取模式」是從向量加法跨越到矩陣乘法的關鍵。
以下兩個經典範例的實作重點與 CUDA 程式邏輯。
1. 向量加法 (Vector Addition)這是 CUDA 的入門實作，主要學習如何將一維陣列的索引對應到 GPU 的執行緒。核心邏輯在 CPU 中我們使用 for 迴圈，而在 CUDA 中，我們利用執行緒的索引來決定每個 Thread 負責處理哪一個元素。
- Index 計算： $i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$
- 邊界檢查：由於執行緒總數（Grid Size）通常是 Block Size 的整數倍，可能超過陣列長度 $N$，必須加上 if (i < n)。

Kernel 範例程式

``` cuda
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
```

2. 矩陣乘法 (Matrix Multiplication)矩陣乘法是運算密集型（Compute-bound）的核心，實作上會經歷從「單純版本」到「共享記憶體優化版」的演進。

基本版本 (Naive Implementation)將二維矩陣映射到 GPU 的 2D Grid 與 2D Block。每個執行緒負責計算結果矩陣 $C$ 中的一個元素 $C_{row, col}$。
- Row 索引： $row = \text{blockIdx.y} \times \text{blockDim.y} + \text{threadIdx.y}$Col 索引： $
- col = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$

Kernel 範例程式 (Naive)

``` cuda
__global__ void matrixMul(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

3. 進階：使用共享記憶體 (Tiled Matrix Multiplication)Naive 版本的主要瓶頸在於全域記憶體 (Global Memory) 的存取頻次過高。在上面的 for 迴圈中，同一個 Block 的執行緒會重複讀取相同的 $A$ 列與 $B$ 行。優化策略：將矩陣切成小塊（Tiles），大小通常等同於 Block 大小（如 $16 \times 16$ 或 $32 \times 32$）。將 Tile 預先載入到 Shared Memory（位於晶片上，延遲極低）。利用 __syncthreads() 確保 Block 內所有執行緒都完成資料載入後，再進行運算。為什麼這對效能至關重要？減少帶寬壓力：從 Global Memory 讀取的次數減少了 $Tile\_Width$ 倍。合併存取 (Coalesced Access)：確保 Global Memory 的存取符合記憶體對齊，提升吞吐量。

總結：從向量到矩陣的思維轉變

| 特性 | 向量加法 (1D) | 矩陣乘法 (2D) |
| --- | --- | --- |
| 執行緒維度 | dim3 block(256); | "dim3 block(16, 16);" |
| 索引計算 | 僅需 x 方向 | 需計算 row (y) 與 col (x) |
| 運算複雜度 | O(N) | O(N3) |
| 優化關鍵 | 隱藏記憶體延遲 | 善用 Shared Memory 減少重複讀取 |

