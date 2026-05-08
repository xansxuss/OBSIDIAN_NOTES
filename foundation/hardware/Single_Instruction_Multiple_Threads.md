**SIMT (Single Instruction, Multiple Threads)** 是 NVIDIA 提出的執行模型，也是 CUDA 的核心設計哲學。

簡單來說，SIMT 是介於 **SIMD**（單指令流多資料流）與 **SPMD**（單程式多資料流）之間的混血架構。

---

##  SIMT 的核心特點

在 SIMT 架構中，雖然多個執行緒執行相同的指令，但每個執行緒擁有獨立的暫存器狀態、程式計數器（PC）以及呼叫堆疊。這使得它比傳統向量處理器（Vector Processor）更具彈性。

### 1. 執行緒束（Warp）的運作

- GPU 將執行緒分成群組，NVIDIA 稱之為 **Warp**（通常是 32 個執行緒）。
    
- 在硬體層級，一個 Warp 內的執行緒在同一時鐘週期內接收相同的指令。
    
- **優點**：極大化指令吞吐量，減少指令解碼的開銷。
    

### 2. 分支分歧（Branch Divergence）

這是 SIMT 最具代表性的特性。當程式碼中出現 `if-else` 時：

- 若 Warp 內的執行緒走不同的路徑，硬體會**序列化（Serialize）**這些路徑。
    
- 這會導致所謂的「遮罩（Masking）」機制：執行 `if` 路徑時，走 `else` 的執行緒會處於閒置狀態，反之亦然。這就是為什麼 AI 工程師在優化 Kernel 時，會極力避免 Warp Divergence。
    

### 3. 與 SIMD 的差異

|**特性**|**SIMD (如 AVX)**|**SIMT (如 CUDA)**|
|---|---|---|
|**程式編寫**|需手動處理向量暫存器與長度|以純量（Scalar）執行緒視角撰寫|
|**分支處理**|通常需手動套用遮罩|硬體自動處理分歧與同步|
|**記憶體存取**|向量存取|記憶體合併（Coalescing）機制|
在開發深度學習算子（Custom Ops）或優化推理模型時，理解 SIMT 有助於處理以下問題：

- **Memory Coalescing**：確保一個 Warp 內的執行緒存取連續的記憶體位址，以達到最高的頻寬利用率。
    
- **Latency Hiding**：SIMT 允許當一個 Warp 在等待記憶體 I/O 時，切換到另一個 Ready 的 Warp 執行，這也是 GPU 運算能力強大的原因——靠大量的執行緒來掩蓋延遲。
    
- **Occupancy**：如何分配暫存器與共享記憶體（Shared Memory），以讓更多的 Warp 在 Streaming Multiprocessor (SM) 上同時運作。