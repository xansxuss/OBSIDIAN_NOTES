### cuda dim3

dim3 是 CUDA 裡面用來定義 grid（網格）與 block（區塊）維度的結構，基本上就是一個最多可支援三維（x, y, z）的整數向量，用來告訴 GPU kernel 啟動時要開多少 thread。

🚀 定義

``` cpp
struct dim3 {
    unsigned int x, y, z;
    __host__ __device__
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
        : x(vx), y(vy), z(vz) {}
};
```

也就是說，你可以像這樣用：

``` cpp
dim3 block(32, 8, 1);
dim3 grid(16, 16, 1);
myKernel<<<grid, block>>>(...);
```

🧠 核心概念
當你呼叫 kernel 時：

``` cpp
kernel<<<grid, block>>>(args...);
```

其實會產生：
- grid.x * grid.y * grid.z 個 block
- 每個 block 裡面有 block.x * block.y * block.z 個 thread
整體 threads 數量 = gridDim * blockDim

📦 範例：2D 影像操作
假設你要處理一張影像（寬 640、高 480），你可以這樣配置：

``` cpp
int width = 640;
int height = 480;

dim3 block(32, 16);
dim3 grid((width + block.x - 1) / block.x,
          (height + block.y - 1) / block.y);

processImage<<<grid, block>>>(...);
```

這樣：

- 每個 block 有 32 × 16 = 512 threads
- grid 大小約為 (20, 30) → 共 600 blocks
總 threads 約 600 × 512 = 307,200，剛好能 cover 整張影像。

🧩 在 kernel 裡的座標計算

``` cpp
__global__ void processImage(uint8_t* img, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    img[idx] = 255 - img[idx]; // 反相處理
}
```

⚙️ 維度使用建議

| 用途           | 維度 | 說明                                         |
| ------------ | -- | ------------------------------------------ |
| 向量、列表        | 1D | `dim3 block(256); dim3 grid((N+255)/256);` |
| 影像 (H×W)     | 2D | `dim3 block(16, 16);`                      |
| 體積資料 (H×W×D) | 3D | `dim3 block(8, 8, 8);`                     |


🚧 小提醒：雖然 dim3 支援 z 維度，但大多數 GPU 的 blockDim.z 通常不超過 64（實際依 GPU compute capability 而定）。

**CUDA 的 dim3 來一次處理多張圖。**

🧩 目標情境

假設：
- 你有 N 張圖片
- 每張圖片大小為 W × H
- 你希望 kernel 能同時處理多張（例如 batch warpAffine、batch normalize、batch inference 前處理）

🚀 典型 dim3 配置
我們可以用 3D grid/block 來 mapping：

``` cpp
dim3 block(16, 16, 1);
dim3 grid(
    (W + block.x - 1) / block.x,   // X 對應 width
    (H + block.y - 1) / block.y,   // Y 對應 height
    N                              // Z 對應 batch
);
```

然後在 kernel 裡面取出座標：

``` cpp
__global__ void processBatch(
    uint8_t* src, float* dst,
    int W, int H, int N, int srcPitch, int dstPitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // width
    int y = blockIdx.y * blockDim.y + threadIdx.y; // height
    int n = blockIdx.z;                            // batch idx (image id)

    if (x >= W || y >= H || n >= N) return;

    // 每張圖在記憶體中的起始位移
    int srcOffset = n * srcPitch * H;
    int dstOffset = n * dstPitch * H;

    int idx = y * srcPitch + x;
    dst[dstOffset + idx] = 255 - src[srcOffset + idx];
}
```

🧠 重點說明

| 維度        | 對應          | 說明                      |
| --------- | ----------- | ----------------------- |
| `block.x` | 每張圖的寬度分工    | 一個 block 負責一小塊 16×16 區域 |
| `block.y` | 每張圖的高度分工    | 同上                      |
| `block.z` | 不使用（通常 = 1） | 因為一張圖通常是 2D 處理          |
| `grid.x`  | 圖片寬度分 block | `(W+bx-1)/bx`           |
| `grid.y`  | 圖片高度分 block | `(H+by-1)/by`           |
| `grid.z`  | batch index | `N`                     |

⚡ Example
假設：

``` cpp
N = 8;
W = 640;
H = 480;

dim3 block(32, 16);
dim3 grid((W + block.x - 1) / block.x,
          (H + block.y - 1) / block.y,
          N);
```

→
這會產生：
- grid.z = 8 → 一次處理 8 張圖
- 每張圖被切成 20 × 30 個 block
- 每個 block 512 threads
→ GPU 同時跑 8 * 20 * 30 * 512 = 2,457,600 threads（會自動由 SM 排程）

⚙️ 進階版：使用 shared memory per image
若每張圖需要各自的變換矩陣（例如 batch warpAffine），可這樣設計：

``` cpp
__global__ void batchWarpAffine(
    const uchar3* src, float* dst,
    const float* warpMatrices, // N × 6
    int W, int H, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;

    if (x >= W || y >= H || n >= N) return;

    __shared__ float M[6];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        const float* srcM = warpMatrices + n * 6;
        for (int i = 0; i < 6; ++i) M[i] = srcM[i];
    }
    __syncthreads();

    // warp
    float srcX = M[0]*x + M[1]*y + M[2];
    float srcY = M[3]*x + M[4]*y + M[5];

    // ... sample & write dst[n][y][x]
}
```

