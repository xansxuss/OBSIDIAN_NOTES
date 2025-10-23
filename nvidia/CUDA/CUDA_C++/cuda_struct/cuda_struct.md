### cuda struct

如何設計一個能在 host/device 兩邊都安全使用的 CUDA 結構（struct），
而且要考慮實際上 GPU kernel、cudaMemcpy、甚至 unified memory / pinned memory 的相容性。

1. 基本原則：什麼樣的 struct 能用在 CUDA？

在 CUDA 裡，struct 只要滿足「POD（Plain Old Data）」條件，
就能安全地在 Host 與 Device 之間傳遞（例如透過 cudaMemcpy）。

✅ POD 條件：

- 沒有虛擬函式
- 沒有繼承
- 成員都是基本型別或其他 POD struct
- 沒有 STL 容器（std::vector, std::string 都不行）

2. 設計範例 1：最安全的 GPU 可用 struct
適合放進 kernel 使用、可直接 cudaMemcpy()。

``` cuda
struct DeviceTensor {
    float* data;   // device pointer
    int width;
    int height;
    int stride;
};
```

Host 端配置

``` cpp
DeviceTensor h_tensor;
h_tensor.width = 640;
h_tensor.height = 480;
h_tensor.stride = h_tensor.width * sizeof(float);

cudaMalloc(&h_tensor.data, h_tensor.width * h_tensor.height * sizeof(float));

// 傳給 GPU
DeviceTensor* d_tensor;
cudaMalloc(&d_tensor, sizeof(DeviceTensor));
cudaMemcpy(d_tensor, &h_tensor, sizeof(DeviceTensor), cudaMemcpyHostToDevice);
```

Kernel 端

``` cuda
__global__ void fillKernel(DeviceTensor tensor, float val) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < tensor.width && y < tensor.height)
        tensor.data[y * tensor.width + x] = val;
}
```

3. 設計範例 2：封裝 host/device 指標對應（binding）
可直接在 kernel 用

``` cuda
struct CudaBinding {
    void* host_ptr;     // pinned host memory
    void* device_ptr;   // device memory
    size_t bytes;
};
```

Host 初始化

``` cpp
CudaBinding buf{};
buf.bytes = 1024;

// pinned host memory + device mem
cudaHostAlloc(&buf.host_ptr, buf.bytes, cudaHostAllocMapped);
cudaMalloc(&buf.device_ptr, buf.bytes);
```

傳到 GPU

``` cpp
CudaBinding* d_binding;
cudaMalloc(&d_binding, sizeof(CudaBinding));
cudaMemcpy(d_binding, &buf, sizeof(CudaBinding), cudaMemcpyHostToDevice);
```

在 kernel 裡

``` cuda
__global__ void copyKernel(CudaBinding b) {
    uint8_t* p = (uint8_t*)b.device_ptr;
    for (int i = 0; i < b.bytes; ++i) p[i] = i % 256;
}
```

4. 計範例 3：RAII + Unified Memory Struct（Host/Device 共用）
最靈活，但有 unified memory 依賴（CUDA 6+）。

``` cpp
struct UnifiedBuffer {
    size_t size;
    float* data;

    __host__ UnifiedBuffer(size_t n) : size(n) {
        cudaMallocManaged(&data, n * sizeof(float));
    }

    __host__ ~UnifiedBuffer() {
        cudaFree(data);
    }
};
```

用法

``` cpp
__global__ void scale(UnifiedBuffer buf, float s) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < buf.size) buf.data[i] *= s;
}

int main() {
    UnifiedBuffer buf(1024);
    for (int i = 0; i < 1024; ++i) buf.data[i] = i;

    scale<<<4, 256>>>(buf, 0.5f);
    cudaDeviceSynchronize();

    std::cout << buf.data[0] << std::endl;
}
```

- 優點：host/device 指標共用，不需 cudaMemcpy
- 缺點：效能受 unified memory page migration 影響

5. 專業等級：Zero-copy + pinned + move-only 設計
這個版本是給你那種「GPU pipeline / TensorRT preproc / zero-copy」場景的：

``` cpp
struct CudaBuffer {
    void* host_ptr = nullptr;
    void* device_ptr = nullptr;
    size_t size = 0;

    CudaBuffer() = default;
    explicit CudaBuffer(size_t bytes) : size(bytes) {
        cudaHostAlloc(&host_ptr, bytes, cudaHostAllocMapped);
        cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept {
        *this = std::move(other);
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            release();
            host_ptr = other.host_ptr;
            device_ptr = other.device_ptr;
            size = other.size;
            other.host_ptr = nullptr;
            other.device_ptr = nullptr;
        }
        return *this;
    }

    ~CudaBuffer() { release(); }

    void release() {
        if (host_ptr) cudaFreeHost(host_ptr);
        host_ptr = nullptr;
        device_ptr = nullptr;
        size = 0;
    }
};
```

- CPU / GPU 共用同一塊 pinned memory（零拷貝）
- RAII 自動釋放
- 支援 move（放進 std::vector / thread-safe queue）

結論

| 使用情境             | 結構建議        | 記憶體類型          | 備註           |
| -------------------- | --------------- | ------------------- | -------------- |
| 單純 kernel 輸入輸出 | POD struct      | device-only         | 最快           |
| CPU-GPU 傳遞 binding | `CudaBinding`   | malloc + cudaMalloc | 傳統、安全     |
| 零拷貝管線           | `CudaBuffer`    | cudaHostAllocMapped | 最現代、最快   |
| 統一記憶體           | `UnifiedBuffer` | cudaMallocManaged   | 簡潔但效能不穩 |
