影像 batch 不是「把圖貼在一起看起來像有多張」這種視覺效果，而是 把多張影像的資料放到一個連續記憶體區塊，形成一個可以給 DNN / CUDA kernel 一次性讀取的 batch buffer。這在 GPU pipeline 裡會很常用。

基本概念：Batch 不是 [`std::vector<cv::Mat>`](https://chatgpt.com/c/694a94bb-6b50-8324-9c19-d409f92431be)

在 host 端通常有：

``` cpp
std::vector<cv::Mat> imgs;   // 例如 N 張 H×W×C
```

這種 vector 只是 host 指標列表。
但 GPU kernel 不能直接用 ` std::vecto<cv::Mat> `，理由有三：

1. cv::Mat metadata + step 不連續

2. 每張 Mat 在 GPU 上要有自己 data pointer

3. GPU 一次 kernel launch 需要 device memory pointers

所以基本原則：

Step 1 — 先把影像上傳到 GPU（統一 memory layout）

常見做法是把 N 張圖先轉成 contiguous float buffer（或 uint8_t）：

假設：

``` txt
imgs[i] 是 H×W×3 BGR（uint8）
目標 batch buffer → float / normalized
```

那 buffer size 就是：

``` txt
N * H * W * C
```

而且要有一個一致的 layout，例如：

``` txt
batch[b][y][x][c]  (NHWC)
```

Step 2 — 設計 CUDA kernel

最直觀的映射是：

``` cpp
grid.x = ceil(W/tx)
grid.y = ceil(H/ty)
grid.z = N  <-- batch index
```

block 一般設 16×16（或 32×8）：

``` cpp
dim3 block(16, 16);
dim3 grid(
    (W + block.x - 1) / block.x,
    (H + block.y - 1) / block.y,
    N
);
```

在 kernel 裡：

``` cpp
int bx = blockIdx.x * blockDim.x  + threadIdx.x;
int by = blockIdx.y * blockDim.y  + threadIdx.y;
int b  = blockIdx.z;  // batch index
if (bx >= W || by >= H) return;

int idx = b * (H*W*C) + (by*W + bx)*C;
float3 pix;
pix.x = input[idx + 0];
pix.y = input[idx + 1];
pix.z = input[idx + 2];

// example operation
output[idx + 0] = pix.x * 0.0078125f;  // /128
output[idx + 1] = pix.y * 0.0078125f;
output[idx + 2] = pix.z * 0.0078125f;
```

簡單模型：

``` cpp
global_idx = b * H*W*C + y * W*C + x * C + c
```

Step 3 — 例子（OpenCV → GPU batch）

先把所有 imgs 轉到一塊 device buffer：

``` cpp
int N = imgs.size();
int H = imgs[0].rows;
int W = imgs[0].cols;
int C = imgs[0].channels();

size_t batch_bytes = (size_t)N * H * W * C * sizeof(uint8_t);

// allocate
uint8_t* d_batch;
cudaMalloc(&d_batch, batch_bytes);

// copy one by one
size_t single_bytes = (size_t)H * W * C * sizeof(uint8_t);

for (int i = 0; i < N; i++) {
    cudaMemcpy(
        d_batch + i * single_bytes,
        imgs[i].data,
        single_bytes,
        cudaMemcpyHostToDevice
    );
}
```


如果要一邊上傳一邊做 normalize，最簡單是：

1. 上傳成 uint8_t batch

2. kernel 裡 convert → float

3. output 到 float batch buffer

把 batch 給 DNN

如果你用 TensorRT / ONNX Runtime：

``` cpp
// assuming input buffer layout NHWC
bindings[input_index] = d_batch;
```

永遠不用一張一張 feed，直接把 batch 一次送入網路。

為什麼這種方式好

這種切法的直覺是：

- 每個 thread 處理一個 pixel

- grid.z 自然映射 batch

- memory 一次性連續可 coalesce

這嬉皮笑臉地對硬體說：

請把資料一次拉進來，不要左閃右閃

結果是 bandwidth 被你真正利用上，而不是被 warp shuffle / cache miss 吃掉。

進階改造

1. NHWC 轉 NCHW

    - 用另一個 kernel 轉

    - 或者在 DNN 引擎支援的 preprocessor 里做

2. zero-copy / mapped memory

    - 用 cudaHostRegister() 把 host buffer map 到 device

    - 影像讀取或 camera callback 直接收到 batch buffer

3. multi-stream pipeline

    - stream0 抓影像 → 上傳 batch

    - stream1 kernel preprocessing

    - stream2 DNN inference

總結（工程心智）

把 batch 看成：

``` cpp
batch_buffer[N][H][W][C]
```

→ 每個 thread = (b, y, x) 位置的像素
→ grid.z = batch index
→ idx 計算不要忘 step 和通道

就像處理一張大圖那樣，把那個「大」橫向延伸到 batch 這個維度。這樣的設計跟 DNN、GPU pipeline 朋友都能愉快握手。
