在 Linux 上「GPU 燒機」（stress test）聽起來有點像把顯示卡丟進魔法陣裡讓它發光發熱，但背後其實是很腳踏實地的工程邏輯：你用高負載的計算與記憶體操作去逼 GPU 持續在最大功耗、最大溫度、最大頻率下工作，看它會不會崩、會不會降頻、會不會冒出奇怪 artifacts（破圖）。

下面原理拆解：

---

### **GPU 燒機的核心原理：把所有硬體單元逼到極限**

GPU 本體可粗分幾個關鍵模組，每個都能被「燒」。

**1. SM / CUDA cores → 算力爆炸壓測**  
讓 GPU 的 shader / CUDA core 每一個運算單元都持續做 FMAD（浮點乘加）、矩陣乘、向量運算。  
典型做法：  
_用超大的矩陣乘法跑不停，比如 GEMM、Conv2D 模擬深度學習負載。_

結果：  
GPU 會拉到最高時脈、最高功耗。

---

**2. Memory controller + VRAM → 記憶體帶寬壓測**  
大量資料在 VRAM ↔ SM 間流動，像：

- 大量 texture fetch
    
- 巨量 global memory 隨機讀寫
    
- FFT / Sorting / huge batch GEMM（因為 memory-bound）
    

結果：  
顯示卡會進入「頻寬滿載」狀態，GDDR6X/GDDR6 變超燙。

---

**3. GPU Cache 層級 → L1/L2 燒法**  
產生大量 cache miss / cache thrash。  
例如：

- 不規律記憶體訪問
    
- 大量隨機取樣
    
- 手寫 kernel 去刻意破壞 locality
    

結果：  
L2 cache 控制器也會變一顆小火爐。

---

**4. Rasterizer / Texture Unit / ROP → 圖形管線壓測**  
OpenGL / Vulkan 的 FurMark 就是這種：

- 大量 fragment shader
    
- insane 的 texture sampling
    
- 瘋狂 rasterize 讓 ROP 吃到飽
    

結果：  
如果 GPU driver 或 VRM 有弱點，FurMark 會讓它現形。

---

**5. Power limits → 逼近 TDP、看 VRM 會不會燒**  
燒機同時也是測 VRM（電壓調節模組）的壓力測試。  
如果電源品質不好、溫度控制差，VRM ripple 會變高然後造成 GPU Crash。

---

### **Linux 燒 GPU 的常見方式**

這些工具本身就是用上述方法把硬體推到極限。

1. **CUDA stress test**
    

`cuda-gpustress`

或你自己寫一個無限大的 GEMM 迴圈。

2. **nvidia-smi + 深度學習框架**  
    跑超大 batch、無限迴圈的 matmul (PyTorch / TensorRT)。
    
3. **FurMark for Linux（Mesa + OpenGL）/ glmark2**  
    用 fragment shader 手撕 GPU。
    
4. **小型自製 CUDA kernel 爆打 GPU**  
    例如：
    
	```bash
	while(true) {     
		custom_kernel<<<grid, block>>>(data); }
	```

---

### **Linux 下 GPU 為什麼燒得特別「透明」？**

因為 Linux 不太會幫你藏 GPU 的真實狀態，你看到的就是底層行為：

- `nvidia-smi dmon` 可看 SM 負載、記憶體控制器、頻率、功耗
    
- `watch -n 0.5 nvidia-smi` 直接看溫度是否逐步上升
    
- 驅動不會像 Windows 那樣「保護」你，只要你沒超電壓，它就讓你玩
    

這也就是為啥雲端 GPU cluster 測壓都在 Linux 上做。

---

### **燒機實際在測什麼？**

這是工程重點：

1. **穩定性**：高溫會揭露壞 VRAM / 壞 SM / 壞 MOSFET。
    
2. **散熱系統**：風扇、散熱器、風道是否設計得好。
    
3. **降頻點**：GPU 在幾度開始 thermal throttle。
    
4. **功率限制行為**：Power Limit 是否會提早卡住效能。
    
5. **記憶體一致性**：過熱時 VRAM 是否會 bit flip。
    
6. **深度學習伺服器的可靠性**：連續跑 72hr 是否 crash。
    

像你做 YOLOv8/TensorRT pipeline，若 GPU 在高負載下容易出現 Segfault 或 kernel reset，那就代表 memory controller 或電源路徑的瑕疵。

---

### 總結一句原理

**GPU 燒機 = 把所有算力、頻寬、cache、電源、溫度推到硬體極限，逼它「露出弱點」的科學儀式。**

# 1) CUDA 純計算型燒機 kernel（C++ / nvcc）

這個範例把每個 thread 推入大量 FMAD（浮點乘加）迴圈，使用 `volatile` 防止編譯器把計算優化掉。把資料放在寄存器/演算單元，不受記憶體頻寬限制（compute-bound）。

**stress_compute.cu**

```cpp
// stress_compute.cu
// 用法：nvcc -O3 -arch=sm_80 -o stress_compute stress_compute.cu
// 執行：./stress_compute [seconds] [blocks] [threads_per_block]
// 範例：./stress_compute 300 256 1024

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

__global__ void heavy_fmad_kernel(int iterations, volatile float *sink) {
    // 每個 thread 有本地變數（寄存器）進行 FMAD
    float a = 1.1234567f + threadIdx.x;
    float b = 2.7654321f + blockIdx.x;
    float c = 0.0f;

    // 大量算術運算，避免記憶體存取
    for (int i = 0; i < iterations; ++i) {
        // 交錯使用不同常數避免簡單迴路優化
        a = a * 1.0000013f + 0.1234567f;
        b = b * 0.9999987f + 0.7654321f;
        c = fmaf(a, b, c); // fused multiply-add
        // 輕微寫回到全域 sink 以避免整個迴圈被認定為 dead code
        if ((i & 127) == 0) sink[(threadIdx.x + blockIdx.x) & 1023] = c;
    }

    // 最後把結果放回 sink（讓它看起來有副作用）
    sink[(threadIdx.x + blockIdx.x) & 1023] = c;
}

int main(int argc, char** argv) {
    int run_seconds = 120;
    int blocks = 256;
    int threads = 1024;
    if (argc > 1) run_seconds = atoi(argv[1]);
    if (argc > 2) blocks = atoi(argv[2]);
    if (argc > 3) threads = atoi(argv[3]);

    // iterations: 調整到讓每次 kernel 執行時間合理（太小反而開銷高）
    int iterations = 100000; // 可視 GPU compute capability 調整

    float *d_sink;
    cudaMalloc(&d_sink, sizeof(float) * 4096);

    auto start = std::chrono::steady_clock::now();
    while (true) {
        heavy_fmad_kernel<<<blocks, threads>>>(iterations, d_sink);
        // 非同步提交，為了逼出更高併發可以不每次 sync，但至少檢查錯誤
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
            break;
        }
        // 小幅同步檢查錯誤與防止 host 退出太快
        cudaDeviceSynchronize();

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start).count() >= run_seconds) break;
    }

    cudaFree(d_sink);
    printf("Done\n");
    return 0;
}
```

編譯：
```bash
nvcc -O3 -arch=sm_80 -o stress_compute stress_compute.cu
```

（`-arch=sm_80` 換成你的 GPU 架構，例如 `sm_75`、`sm_86` 等）

執行（範例，跑 5 分鐘）：

```bash
./stress_compute 300 512 512
```


要點說明：

- `iterations` 可以調整使 kernel 每次跑較久，減少 kernel-launch overhead。
    
- 多開幾個 blocks/threads 可逼滿 SM。thread block 数目要超過 GPU 的 active block capacity 來提升 occupancy。
    
- 若要更激進，可在 host 端以多個 CUDA stream 非同步提交多個 kernel。

# 2) PyTorch C++（libtorch）— 大型 matmul loop（拉爆算力 + VRAM）

這段使用 libtorch C++ frontend 做巨大的矩陣乘法（GEMM）在 CUDA 上，連續迴圈執行以達到長時間高負載。用 CMake 編譯。

**CMakeLists.txt**

``` cmake
cmake_minimum_required(VERSION 3.18)
project(torch_matmul_stress)

find_package(Torch REQUIRED) # 需要設定 LIBTORCH 路徑：-DCMAKE_PREFIX_PATH=/path/to/libtorch
set(CMAKE_CXX_STANDARD 17)

add_executable(torch_matmul_stress main.cpp)
target_link_libraries(torch_matmul_stress "${TORCH_LIBRARIES}")
set_property(TARGET torch_matmul_stress PROPERTY CXX_STANDARD 17)
```

main.cpp

```cpp
// main.cpp
// 用法：./torch_matmul_stress [iters] [N]
// 範例：./torch_matmul_stress 10000 8192

#include <torch/torch.h>
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    int64_t iters = 1000;
    int64_t N = 4096;
    if (argc > 1) iters = std::stoll(argv[1]);
    if (argc > 2) N = std::stoll(argv[2]);

    torch::Device device(torch::kCUDA, 0);
    std::cout << "Using device: " << device << "\n";
    std::cout << "Matrix size: " << N << " x " << N << ", iters: " << iters << "\n";

    // 建立巨型 tensor（float32）在 GPU（可能會需要大量 VRAM）
    auto A = torch::rand({N, N}, device, torch::kFloat32);
    auto B = torch::rand({N, N}, device, torch::kFloat32);

    // warming up
    for (int i = 0; i < 5; ++i) {
        auto C = torch::mm(A, B);
        C = C + 1.0;
    }
    cudaDeviceSynchronize();

    auto t0 = std::chrono::steady_clock::now();
    for (int64_t i = 0; i < iters; ++i) {
        auto C = torch::mm(A, B);
        // 強制 materialize 並同步，確保工作完成再下一次迴圈（更容易燒滿 GPU）
        C = C + 0.0;
        cudaDeviceSynchronize();

        if ((i+1) % 10 == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - t0).count();
            double iters_per_sec = (i+1) / (elapsed + 1e-9);
            std::cout << "iter " << (i+1) << " / " << iters << "  (iters/sec ~ " << iters_per_sec << ")\n";
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double total = std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count();
    std::cout << "Total time: " << total << " s\n";
    return 0;
}
```

編譯（假設你已下載 libtorch，設定 `LIBTORCH` 路徑）：

``` bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release -j
```

執行（注意 VRAM）：

``` bash
./torch_matmul_stress 1000 8192
```

注意：`N=8192` 會用掉非常多 VRAM（8192x8192 ≈ 268M floats ≈ 1.07 GB per matrix × 多倍），請先估算你的 GPU 記憶體容量，再調整 `N`。若 VRAM 不夠會 OOM。

要點：

- 把 `cudaDeviceSynchronize()` 放在迴圈裡會把每次運算強制等待完成，最大化單張卡佔用。
    
- 可以改成非同步（不 sync）以測試隊列與多 kernel 並行能力。
    
- 若要同時測試多卡，可使用 NCCL / 多進程或在程式中把 tensors 放到不同 device。
    

# 3) Linux 下監控 GPU 燒機的最佳指令（實戰小抄）

下面列出一組常用命令、要看什麼欄位，以及如何解讀。幾個一行指令和常見腳本片段提供。

## 即時監控（常用）

``` bash
# 基本狀態：溫度、利用率、功耗、頻率、記憶體使用
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,pstate,power.draw,clocks.current.graphics,clocks.current.memory --format=csv

# 連續監控（每秒）
watch -n 1 "nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.current.graphics --format=csv,noheader,nounits"

# 更簡潔：gpustat（需 pip 安裝）
gpustat -cp
# 或每秒更新：
watch -n 1 gpustat --no-color
```

解讀重點：

- `temperature.gpu`：溫度上升速度是重要指標，超過製造商建議就要停。常見安全門檻 ~85~95°C（依卡不同）。
    
- `utilization.gpu`：SM 利用率（高表示算力被用滿）。
    
- `utilization.memory` & `memory.used`：是否被 VRAM 或 memory-bandwidth 綁死。
    
- `pstate`：電源狀態，P0 是最高效能。若看到持續處於 P2/P8 代表被降頻或受限。
    
- `power.draw`：實際瓦數，觀察是否接近 TDP。
    

## 更細節的 telemetry

``` bash
# 多欄位詳細輸出，CSV logfile，適合記錄
nvidia-smi --query-gpu=timestamp,index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,pstate,power.draw,clocks.current.graphics,clocks.applications.graphics --format=csv -l 1 > gpu_log.csv
```

這樣可以後面拿去畫圖（溫度 vs 時間、功耗 vs 時間）。

## 追蹤驅動錯誤 / kernel reset

``` bash
# 檢查 kernel log，看有沒有 GPU 驅動或 TDR 重置
dmesg | egrep -i "NVRM|gpu|CUDA|Xid|GPU" -n

# /var/log/syslog 或 journald
sudo journalctl -k | grep -i nvrm
sudo journalctl -u display-manager --since "1 hour ago" | grep -i Xid
```

如果看到 `Xid` 或 `GPU shark` 類錯誤，代表 GPU driver 已經 detect 到錯誤或 reset。這通常是 VRAM ECC、IO errors、或電源/熱問題。

## 觀察功率/電壓/頻率（更進階）

``` bash
# 查詢電源限制
nvidia-smi -q -d POWER

# 查詢溫度與風扇速率
nvidia-smi -q -d TEMPERATURE
nvidia-smi -q -d FAN
```

## 實時視覺化工具（互動式）

- `nvtop`：互動式的 top-like 工具（可顯示每個進程 GPU 使用率與顯存）。  
    安裝（Ubuntu）：

``` bash
sudo apt install nvtop
nvtop
```

- `dcgmi` / `nvidia-smi dmon`：適合資料中心監控（`nvidia-smi dmon` 提供更密集的 telemetry）。
    

## 建議的監控腳本範例（每 1s 寫 CSV）

``` bash
#!/bin/bash
OUT="gpu_stress_log.csv"
echo "ts,index,temp,util_gpu,util_mem,mem_used,power,clk_gpu" > $OUT
while true; do
  ts=$(date --iso-8601=seconds)
  read -r line < <(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.current.graphics --format=csv,noheader,nounits)
  echo "${ts},${line}" >> $OUT
  sleep 1
done
```

後續用 Python/matplotlib 或 pandas 畫圖，找出 throttle 發生時刻。

# 額外小技巧與實務建議

- 開始前把 GPU power limit 與 fan policy 記錄下來：`nvidia-smi -q`。若要長時間 burn-in，視情況提高風扇轉速（或手動設定 power limit 以保護硬體）。
    
- 對筆電：不要在電池供電下進行燒機，電池可能鼓、膨脹、冒煙、死亡。
    
- 若發現 `Xid` 或 `GPU reset`，立刻停止燒機並檢查系統 log。連續 Xid 代表硬體層面可能已受損或電源/散熱不良。
    
- 若想壓記憶體頻寬，多做大尺寸隨機存取的 kernel 或用 `torch::mm` + 非暫存化 memory copy 測試（memory-bound）。
    
- 若想同時壓 compute 與 memory，混合 launch：一個 kernel 做大量 FMAD（compute-bound），另一個做巨大 memory streaming（memory-bound），交替或同時執行。

