NMS系列運算子
- nms_kernel.cpp
- nms_kernel.cu
NMS 是物件偵測（Object Detection）中不可或缺的後處理步驟。它的演算法核心邏輯非常直觀，但要在 C++ 中寫到**極致高效、不用 STL、且能平行化**，就需要非常深厚的記憶體與指標操作功力。

以下為你系統性拆解 NMS 的核心演算法邏輯、C++ 底層實作的研究痛點，以及進階的 NMS 變體。

## 📌 NMS 的核心演算法流程

在閱讀 `torchvision` 的 `nms_kernel.cpp` 之前，我們必須先釐清標準 NMS 的核心行為：

1. **輸入**：一堆候選邊界框（Bounding Boxes，包含 $x_1, y_1, x_2, y_2$）以及它們對應的置信度分數（Scores）。
    
2. **排序**：將所有 Bounding Boxes 依照 Scores 從大到小排序。
    
3. **壓制（Suppression）**：
    
    - 挑選目前 Score 最高的 Bounding Box $B$ 作為基準，將它保留（移入輸出列表）。
        
    - 計算 $B$ 與其餘所有 Bounding Boxes 的 **IoU（Intersection over Union，交併比）**。
        
    - 如果 IoU 大於設定的閾值（Threshold），代表重複性極高，直接將該 Bounding Box 剔除（壓制）。
        
    - 在未處理的 Box 中，重複上述步驟，直到所有 Box 都被處理完畢。
        

## 🛠️ C++ 底層實作的研究痛點（針對你的不使用 STL 目標）

當你打開 `torchvision/csrc/ops/cpu/nms_kernel.cpp` 時，你可以特別觀察官方如何處理以下高效能 C++ 的技術細節：

### 1. 拋棄 `std::vector` 的動態記憶體配置

`std::vector` 的 `push_back` 或 `erase` 會頻繁觸發記憶體重新配置（Reallocation）與資料搬移。在高效能算子中這是致命的。

- **研究重點**：觀察底層如何事先配置好一塊連續的記憶體（Raw Array），並使用「指標」或「狀態陣列（Mask/Suppressed Flags）」來標記哪些 Box 已經被壓制，而不是真的去刪除陣列元素。
    

### 2. 快取友善（Cache-Friendly）的資料佈局

Bounding Box 的資料一般是 `[N, 4]` 的形狀（張量）。

- **研究重點**：在計算 IoU 時，記憶體是按行存取（Row-major）還是按列存取（Column-major）？如何讓 CPU 的 L1/L2 快取命中率（Cache Hit Rate）最高？讀原始碼時，注意看它迴圈的巢狀順序（Nested Loop Order）。
    

### 3. 自行實作排序（不依賴 `std::sort`）

標準 NMS 需要頻繁對 Index 或 Score 進行排序。

- **實戰練習**：試著自己用 C 風格實作 **快速排序（Quick Sort）** 或 **堆積排序（Heap Sort）**。思考如何在排序時只更動「索引陣列（Index Array）」，而完全不去搬動沉重的 Bounding Box 座標資料。
    

## 🚀 進階 NMS 系列變體（強烈推薦一併研究）

除了標準 NMS，學界和業界為了改進效能與精準度，衍生出了幾個經典的變體算子。閱讀並實作這些變體，能大幅提升你對演算法變形與 C++ 架構設計的掌握度：

### 1. Soft-NMS

- **痛點**：標準 NMS 太過暴力。如果兩個同類別的物體真的高度重疊（例如：兩個人前後並排走），標準 NMS 會直接把後面的 Box 刪掉，導致漏檢。
    
- **改進邏輯**：它不直接刪除 IoU 大於閾值的 Box，而是**降低其置信度分數（Score Decoherence）**。IoU 越高，分數扣越多。
    
- **C++ 研究價值**：它的邏輯不需要進行「硬刪除」，而是動態更新分數。你可以研究它如何使用高斯函數（Gaussian）或線性函數來平滑地調整權重。
    

### 2. Fast NMS (Matrix NMS)

- **痛點**：傳統 NMS 是一個序列式的演算法（一邊挑選、一邊壓制），很難完全平行化。
    
- **改進邏輯**：預先計算所有 Box 兩兩之間的 IoU 矩陣，然後透過矩陣運算一次性找出需要被壓制的 Box。
    
- **C++ 研究價值**：這非常適合用來練習 **SIMD（單指令流多資料流，如 AVX2、NEON）** 指令集優化。如何利用 CPU 的向量化指令，一次計算 4 個或 8 個 IoU？
    

### 3. DIOU / CIOU NMS

- **改進邏輯**：傳統 NMS 只考慮 IoU（重疊面積）。但有時候兩個 Box 雖然重疊度相同，但中心點的距離或寬高比不同。DIOU NMS 在計算壓制指標時，加入了**兩個 Box 中心點距離**的懲罰項。
    
- **C++ 研究價值**：這涉及更多幾何運算（如計算歐幾里得距離、外接矩形邊長）。如何確保這些數學運算不會造成浮點數溢位（Overflow）或下溢（Underflow）？
    

## 🎯 你的下一步行動指南

如果你想開始動手，建議採取「由簡入繁」的逆向工程策略：

1. **純 C 實作**：自己開一個 `.cpp` 檔案，宣告 `float boxes[N][4]` 與 `float scores[N]`，不引進任何標頭檔（連 `<vector>`、`<algorithm>` 都不用），純手工寫出標準 NMS。
    
2. **比對原始碼**：寫完後，打開 `torchvision` 的 `nms_kernel.cpp`，對比官方的實作。看看官方在哪些邊界條件（例如：面積為 0、IoU 剛好等於閾值）做了防禦性程式設計（Defensive Programming）。
    
3. **效能榨乾**：嘗試引入 OpenMP（C++ 的平行化語法，不需要 STL 支援）到你的純 C 實作中，看看能不能做到多核心平行加速。

## 💡 階段一：掌握核心算子的底層實作

你貼出的編譯指令指向了兩個非常有研究價值的 C++ 檔案：

1. `nms_kernel.cpp` (Non-Maximum Suppression, 非極大值抑制)
    
2. `ps_roi_align_kernel.cpp` (Position-Sensitive ROI Align)
    

### 研究重點：

- **演算法的底層優化**：看看它們如何處理邊界條件？在沒有 STL 的情況下，它們如何精準操作指標（Pointer）與記憶體連續性，以達到極致的快取（Cache）利用率？
    
- **平行運算（Parallelism）**：在 CPU 算子中，PyTorch 通常會使用 `at::parallel_for` 來做多執行緒平行化。去觀察它們是如何切分資料、避免 Race Condition（競態條件）的。
    

## 💡 階段二：理解 Python 與 C++ 的橋樑（Pybind11）

在指令中你看到了 `-DPYBIND11_COMPILER_TYPE="_gcc"` 等參數。`torchvision` 封裝底層高效能 C++ 的核心就是 **pybind11**。

### 研究重點：

- **C++ 類別與函式的導出**：尋找原始碼中的 `PYBIND11_MODULE` 或 PyTorch 封裝的 `TORCH_LIBRARY`。
    
- **記憶體零拷貝（Zero-copy）**：研究 Python 的 `torch.Tensor` 是如何轉成 C++ 的 `at::Tensor`。在底層，它們其實共享同一塊記憶體（儲存區塊），理解這個機制能讓你寫出記憶體效率極高的系統。
    

## 💡 階段三：實戰學習方法——「逆向拆解與重構」

光看程式碼很容易流於走馬看花，強烈建議你用以下步驟進行「主動式學習」：

### 1. 簡化並抽離 (Stripping)

試著把 `nms_kernel.cpp` 從 `torchvision` 的複雜架構中單獨拿出來。去掉 PyTorch 的萬用 Tensor 封裝（`at::Tensor`），嘗試用最純粹的 **C 風格陣列（Raw Array）和指標**來重寫它。

> **這非常符合你不依賴標準庫 (STL) 的方向！** 你可以自己實作簡單的排序（如 Quick Sort）或記憶體管理，並手動為它加上 OpenMP 進行 CPU 多執行緒加速。

### 2. 效能對比 (Benchmarking)

寫一個簡單的 C++ `main()` 函式去跑你重寫的 NMS，並與 PyTorch 官方的效能進行對比（使用 `std::chrono` 或是 Linux 的 `perf` 工具）。

- 為什麼官方的比較快？
    
- 官方程式碼在編譯時用了哪些優化旗標（例如我們在指令中看到的 `-O2`、`-fwrapv`）？
    

### 3. 改寫為 CUDA 版本（進階）

指令中帶有 `-DWITH_CUDA`。當你搞懂 CPU 版本的 `nms_kernel.cpp` 後，去對照看 `nms_kernel.cu`（CUDA 版本）。

- 觀察同一個演算法，在 CPU 的執行緒模型（Thread model）與 GPU 的 Grid/Block/Thread 記憶體架構下，思維有何不同？
    
- 如何利用 GPU 的 Shared Memory（共享記憶體）來加速重複的邊界框（Bounding Box）重疊度（IoU）計算？
    

## 🛠️ 給 AI 工程師的專案推薦

除了 `torchvision` 之外，如果你想專攻 **C/C++ (少 STL) + Python** 的高效能混合編譯，以下開源專案也非常值得你 clone 下來讀：

1. **ggml / llama.cpp**
    
    - **特點**：幾乎**純 C/C++ 實作**，極度排斥複雜的 STL，完全專注於底層記憶體佈局、AVX/NEON 指令集優化。
        
    - **學習點**：如何用最樸素的 C 語言結構體與指標，撐起頂級的大語言模型（LLM）推論。
        
2. **vLLM (PagedAttention 核心部分)**
    
    - **特點**：精妙的 GPU 記憶體分頁管理（Pager）。
        
    - **學習點**：如何將作業系統的虛擬記憶體概念，搬到 GPU 上解決 LLM 顯存碎片化的問題。