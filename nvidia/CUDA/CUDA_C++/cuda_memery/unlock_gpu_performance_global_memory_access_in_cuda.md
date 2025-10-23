# Unlock GPU Performance：CUDA 裡的全域記憶體訪問優化

## 簡介  
在 CUDA 程式設計中，「全域記憶體（Global Memory）」是 GPU 裡 DRAM 的主要記憶體空間，對效能影響非常大。本文帶你從記憶體類型、訪問模式，到如何用工具檢測與優化，全方位看「為什麼 Global Memory 訪問方式」會成為效能瓶頸。

---

## 1. 全域記憶體是什麼？  
- 在 CUDA 設備上，各種記憶體種類（如：shared、register、global）有不同的存取範圍／生命週期／快取行為。
- Global Memory（也稱 device memory）是位於設備 DRAM 的主記憶體空間，類似 CPU 的 RAM。 
- 它可以被主機（host）與所有 kernel 裡的 thread 訪問。
- 配置方式包括：在 device 端使用 `__device__`、或是透過 `cudaMalloc()`／`cudaMallocManaged()` 在 runtime 動態分配。
- 範例流程：  
  1. host 分配、初始化 global memory  
  2. 傳資料至 device  
  3. kernel 執行，thread 從 global memory 讀／寫  
  4. host 從 device 拷回結果，釋放 memory

---

## 2. 記憶體訪問「共轭」（Coalesced） vs 「非共轭」（Uncoalesced）  
### 共轭訪問  
- 在 CUDA 中，threads 是以 warp（目前 GPU 為 32 threads）為單位執行。
- GPU 記憶體系統會把同一 warp 的訪問「合併（coalescing）」成最少數的記憶體交易（memory transactions）以提升頻寬效率。
- 範例：每個 thread 訪問連續的 4 byte 元素，才能在一個 warp 裡順序訪問連續記憶位置。那麼多個 thread 的 requests 能被合併成少數的 32-byte sector 交易。
### 非共轭訪問  
- 如果每個 thread 訪問之間有大跳（stride 非常大），例如每 32 thread 跳 32 個元素／128 bytes，那麼系統雖然讀了大量資料，但實際只有少部分被 thread 使用：效率極低。
- 作者透過分析工具（Nsight Compute）示範：在 coalesced 的情況下，每一記憶體請求對應約 4 個 32-byte sector；而在 uncoalesced 的情況下則是 32 個 sector → 效率大約差 8 倍。
---

## 3. 跳距（Stride）效應  
- 「Stride」是指 warp 中連續 thread 訪問的記憶位置之間的距離（以元素或 bytes 計）。
- 實驗中顯示：當 stride 從 0~31 增加時，有效帶寬（bandwidth）明顯下降。也就是說，跳得越大 ⇒ 合併效果越差 ⇒ 記憶體頻寬利用率越低。
- 對工程上而言：儘量避免讓 warp 裡的 threads 去訪問大距離、不連續的位置。
---

## 4. 多維陣列（矩陣）訪問場域  
- 在 CUDA 裡，我們常以 2D 或 3D thread block 處理 2D／3D 資料。但實體記憶體通常是線性（1D）配置：row-major（在 C++）是儲存方式。
- 若 threadIdx.x 變動最快（也就是 warp 裡 threads 在 x 方向連續），那麼設計 storage + 訪問模式時，要讓 threadIdx.x 的連續 thread 也訪問記憶體中**連續元素**（例如同一 row 裡的不同 column）以達成 coalescing。
- 作者給了兩個 kernel 範例：  
  - `coalesced_matrix_access`：row-major 序儲存、threadIdx.x 對應 column → 訪問連續 memory → 成功 coalesce。
  - `uncoalesced_matrix_access`：錯誤地用 `col * height + row`（相當於假設 column-major 存儲）→ threadIdx.x 增加時訪問間隔大、跳躍明顯 → 非共轭訪問。
- 在分析中：兩個 kernel 的「請求數量（requests）」一樣，但「sector 數量」差很大（一個為 33,554,432，另一個為 268,435,456）→ 差了 8 倍以上。
---

## 5. 總結／實務建議  
- 如果你在撰寫 GPU kernel，想抓最大效能，那麼「記憶體訪問模式設計」是不能忽略的重要環節。作者強調：Global Memory 的效率往往是整個 kernel 性能的瓶頸。
  1. 讓 warp 裡 threads 訪問 memory 時盡可能走**連續路徑**（最低跳距）  
  2. 減少 stride、避免 thread 間跳很大／訪問分散位置  
  3. 使用 Nsight Compute 等工具 **量化檢測**（如 sectors/request、每秒 bytes read、帶寬百分比）以確保你的訪問確實「共轭」  
- 作者提醒：這篇是原 2013 年文章的更新版本。
---

## 6. 觀察  
- 當你真的踏入大型 GPU 程式（尤其深度學習、HPC）時，你會發現「記憶體頻寬」常常比「運算量」更快成為瓶頸。這篇文章本質就是在對付這種類型的瓶頸。  
- 雖然範例比較基礎 (2 D、連續／間隔訪問)，但關鍵思維卻能套用到更複雜場景：例如當你處理 tiled 矩陣、跨 GPU 存取、或者更加複雜的記憶體層級 (如 L1/L2／cache)時，理解「訪問共轭 vs 非共轭」還是核心。  
- 有趣的地方：作者以「sectors per request」這種比較底層的 metric 來說明訪問效率。對我們 AI 影像工程師來說，這種低階細節常被忽略，但卻能帶來一兩成以上的效能優化，特別是在記憶體頻寬飽和的情況下。  
- 建議一個實作提醒：  
  - 在你做 CUDA kernel 時，**先畫出**threadIdx.x/y/z 對記憶體中資料的映射。確保這個映射是「threadIdx.x 快動、memory 連續」的模式。  
  - 若你處理影像資料 (2D／3D)，儘量設計讓每個 warp 的 threads 在記憶體 row（或 depth slice）中訪問連續元素，而不是跳來跳去。  
  - 使用 Nsight Compute 觀察 L1/L2 cache、DRAM bytes、sectors 等，做「先量化、再優化」而不是猜優化。  
  - 若你發現效能飽和、頻寬被卡住，不要先猜 kernel 運算量，而是先看 memory access pattern 是否共轭。

---

## 結語  
這篇文章提醒我們一件重要而容易被忽略的事：在 GPU 程式優化裡，「記憶體訪問方式」其實往往比「算很多浮點運算」還要來得關鍵。設計好 Global Memory 存取模式、減少跳躍、確保 coalescing，才能真正「解鎖」你的 GPU 效能。

程式碼片段 + 性能數據

# 程式碼片段與性能數據

## 程式碼片段  
### 共巢 (coalesced) 訪問範例  
```cpp
__global__ void coalesced_access(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // 每個 thread 訪問前後相鄰的 4 byte word
        output[tid] = input[tid] * 2.0f;
    }
}
```

這段程式碼中，每個 thread 對應一個 input[tid]／output[tid]，thread 0 → element 0、thread 1 → element 1 …，達到連續 4 byte 的訪問，有利於合併交易 (coalescing)。

``` cpp
__global__ void uncoalesced_access(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // 使用 stride = 32 的跳躍 (約 128 bytes)，使每個 thread 訪問的位置分散
        int scattered_index = (tid * 32) % n;
        output[tid] = input[scattered_index] * 2.0f;
    }
}
```

這段程式碼中，因為每個 thread 訪問的位置間隔很大（這裡簡化為「乘 32」），導致不同 thread 所訪問的記憶體位址散佈，在同一 warp 裡無法達成有效的合併。 

性能數據／關鍵指標
在共巢案例中：一個 warp (32 threads) 訪問 4 bytes／thread、連續排列的資料 → 可以用 4 個 32-byte sector（也就是 128 bytes）來滿足整個 warp。 
在非共巢案例中：因為散佈訪問，每個 thread 最可能落在不同的 32-byte sector，理論上可能需要 32 個 32-byte sector 來服務一個 warp。也就是說效率可能是共巢的約 1/8 左右。 
關鍵 metric：使用 Nsight Compute 觀察「sectors per request」指標。「sectors/request 越少」＝越好＝表示記憶體交易水平越低、帶寬利用率越高。 
[Global Memory Coalescing](https://giahuy04.medium.com/global-memory-coalescing-37a6f9d7e314)

觀察

這篇文章其實並沒有給出「某 GPU 型號執行這兩段程式碼的實際 秒數或 GB/s 數據」，但它教你看的是結構性 差異：連續 vs 跳躍訪問對交易數量（transaction count）／sector數量的影響。

對影像 AI 工程師來說：重點在於「在 kernel 裡設計 threadIdx.x 對應記憶體中連續元素」比什麼都還重要。如果跳得太大 → 可能就卡在記憶體頻寬而不是算力。

若把 “4 sectors／warp” vs “32 sectors／warp” 這種差異套用到實務，假設頻寬瓶頸的情況，你可能從 80–90% 利用率掉到 10–20% 利用率，效能就大幅下降。

建議：用 Nsight Compute 看「l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio」這類指標（或類似）來量化你 kernel 的記憶體存取效率。這樣才能從「覺得慢」變成「知道為什麼慢」。

[Unlock GPU Performance: Global Memory Access in CUDA](https://developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/)