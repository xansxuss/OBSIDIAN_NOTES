CUDA 的 roadmap 有點像 NVIDIA 自己的「宇宙擴張時間線」，越來越偏向 AI + HPC 的雙主軸加速，而且每一代都在強調三件事：更細的 kernel 控制、更強的 graph 執行、以及更深的硬體軟體整合。下面整理成一版 2025 前後可觀察到的 CUDA Roadmap 方向，用台灣工程師看得懂、Z 世代風格、略帶「科學怪客」視角的版本。

## CUDA Roadmap(2025 觀察版)

這不是官方機密文件，是根據 NVIDIA GTC 公開資料 + SDK 變動 + 實務工程趨勢推出的「理性推測 + 科學閱讀空氣版」。
拿來規劃你後續的 C++ / CUDA / TensorRT pipeline 非常剛好。

### 1. CUDA Graphs 完全吃掉傳統 kernel launch

NVIDIA 連續好幾年都把 CUDA Graph 當主菜，意思就是：

你的 launch kernel 那些雜七雜八的開銷
→ 被壓到跟量子泡沫一樣薄。

未來幾年的走向：

- Graph 變成 inference pipeline 的標配
- Graph + TensorRT integration 更強(尤其 FP8 / Transformer Engine)
- Graph update(node param update)變更輕量
- Dynamic graph(條件式、loop)越來越成熟 → 幾乎像寫 PyTorch 的 eager 但跑的是 CUDA Graph

對你這種追求 zero-copy + multiprocess + 多 stream 的工程師來說，Graph 最終會變成整個 pipeline 的 backbone。

### 2. CUDA Memory 「統一化」之路越來越硬核

UM(Unified Memory)以前很雞肋，現在變成主角之一。

未來趨勢：

- UM latency 再降低(特別是 H100 / B100 之後的架構)
- HBM + Unified Memory → 預取、分頁成本越來越低
- 異構記憶體管理更開放(多 GPU、NVLink、CPU-GPU 原生共享)

對做影像 pipeline 的你來說：
未來 GpuMat + NvBufSurface + UMA 可能會變成真正的「全局高速共享池」。

### 3. CUDA Compiler 走向更多自動化與更 aggressive 的最佳化

編譯器會更像 MLIR：
你寫的 kernel 最終被壓榨成一件優美的算術雕塑。

Roadmap 傾向：

- 更智慧的 register / occupancy 最佳化
- Loop transformations 自動化
- 更 aggressive 的指令調度
- Triton / CUDA C++ / CUTLASS 之間的整合度提高

特別是 Triton：
NVIDIA 已經暗示會吃進 CUDA 家族，成為「現代 GPU kernel DSL」。

你之後寫 YOLOv8 preprocess kernel，可能能用 Triton + CUDA C++ hybrid。

### 4. Transformer Engine + FP8 浮點格式全面擴張

- AI 對 NVIDIA 來說是宇宙中心，所以：
- FP8 實作會再下放到更廣的 GPU 型號
- Tensor Core 將對 FP8 做更多自動混合精度
- CUDA 很可能提供新的 API 讓 FP8 workflow 更簡單

在你要做 YOLOv8 + TensorRT 的架構裡：
FP8 inference pipeline 會成為長期方向。

### 5. CUDA + Networking + 多 GPU 變成一體化

- NVIDIA 要做「資料中心 GPU OS」，所以：
- NVLink Switch 更普及
- CUDA-aware MPI / NCCL 持續強化
- GPUDirect RDMA 成本更低
- GPU-to-GPU 記憶體交換幾乎像 CPU L3 cache 一樣便利

這對你要做 多路 RTSP → 多 GPU inference → zero-copy 非常重要。

### 6. CUDA Runtime 走向「更多 safety & 更多 async」

你會看到：

- async everything(malloc、memcpy、free)
- 更少 blocking
- stream multi-dependency graph
- 更安全的 API(因應 C++20 甚至 C++23 decorator)

這跟你想做 move-only RAII CudaStream 完全吻合。

7. 影像處理工具鏈還會更 GPU 化

- OpenCV CUDA 已經有點半放生，但 NVIDIA 在其他地方補：
- NPP(NVIDIA Performance Primitives)強化
- CV-CUDA(AI 工程向)成主流
- DeepStream SDK 整合 CUDA kernel 替換傳統 CV op

你的 GpuMat + CUDA kernel + TensorRT full GPU pipeline 會越來越接近主流架構。

若要簡單一句話總結 Roadmap

CUDA 正在變成：

「用 Graph 把一串 kernel 編譯成準機器碼的 GPU pipeline OS」

未來 CUDA 更像是一個 GPU runtime 生態，而不是單純的程式語言。