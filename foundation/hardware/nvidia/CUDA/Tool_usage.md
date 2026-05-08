進行效能調優（Performance Tuning）時，熟練掌握 NVIDIA 的 Nsight 系列工具是區別「一般開發者」與「資深優化專家」的關鍵。這兩款工具分別對應了不同的優化層級：Nsight Systems 負責宏觀的系統流程，而 Nsight Compute 則負責微觀的算子（Kernel）細節。

以下針對這兩項工具的使用重點與策略進行整理：

1. NVIDIA Nsight Systems (nsys)
定位：系統級追蹤（System-wide Trace） Nsight Systems 用於觀察 CPU、GPU、記憶體以及各種 API（CUDA, TensorRT, cuDNN）之間的交互作用，找出系統瓶頸（Bottleneck）。

- 關鍵使用情境：

    - 找出 CPU Bound 或 GPU Bound： 觀察 CPU Thread 的活動與 GPU Kernel 的執行時間軸。

    - 分析資料搬移： 檢查 cudaMemcpy 的頻率與資料量，判斷是否有頻繁且不必要的 HtoD（Host to Device）傳輸。

    - 多卡/多節點通訊： 追蹤 NCCL 調用，優化分布式訓練中的通訊延遲。

    - API 負載： 找出哪個 CUDA API 調用（如 cudaMalloc）造成了不必要的同步阻塞。

- 常用技巧：

    - 使用 NVIDIA Tools Extension (NVTX)：在程式碼中加入標記（Ranges），讓 Timeline 上能顯示對應的邏輯區塊（例如：Data Loading, Forward Pass, Loss Calculation）。

    - 關注 Overhead： 檢查 GPU 上的「空隙」（Gaps），這些空隙通常代表 CPU 處理太慢或是 API 同步造成的等待。

2. NVIDIA Nsight Compute (ncu)
定位：核心級剖析（Kernel-level Profiling） 當 Nsight Systems 鎖定了某個耗時最長的 Kernel 後，就輪到 Nsight Compute 出馬，進行深入的硬體利用率分析。

- 關鍵指標分析：

    - SOL (Speed of Light)： 顯示 Kernel 在運算單元（Compute）與記憶體頻寬（Memory）上的利用率。

    - Roofline Model： 直觀判斷 Kernel 是受限於「算力」（Compute Bound）還是「頻寬」（Memory Bound）。

    - Memory Hierarchy： 分析 L1/L2 Cache 的命中率，以及 Shared Memory 的使用效率。

    - Occupancy： 檢查活躍的 Warps 數量，評估 Register 使用量是否限制了 GPU 的平行度。

- 進階優化策略：

    - Coalesced Access： 檢查 Global Memory 存取是否合併，減少記憶體請求次數。

    - Bank Conflicts： 偵測 Shared Memory 的存取衝突。

    - Register Spilling： 檢查是否有變數溢位到 Local Memory（這會大幅降低效能）。

3. 工作流程建議 (Workflow)
在進行優化時，建議遵循 "Top-Down" 的邏輯：

1. 第一步：nsys 宏觀診斷

    - 執行 nsys profile -t cuda,nvtx,cudnn,cublas python train.py。

    - 找出 Timeline 上的瓶頸：是資料讀取太慢？還是某個 Kernel 執行太久？

2. 第二步：ncu 精準打擊

    - 針對耗時最長的 Kernel 執行 ncu -o profile_report -k <kernel_name> python train.py。

    - 查看 Memory Workload Analysis 與 Scheduler Statistics。

3. 第三步：程式碼修改與驗證

    - 根據建議修改 CUDA C++ 或 TensorRT 實作，然後再次回到第一步驗證整體效能提升。