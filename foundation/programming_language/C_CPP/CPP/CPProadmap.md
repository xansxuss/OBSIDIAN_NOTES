### 🌱 第一章：語言基本生存技能

先把「能在野外活下去」的技能點起來，不需要花俏，穩定就好。
包含：

- C++ 語法與物件模型（class / struct / constructor / destructor）
- pointers（指標）＋ references（參照）到底差在哪
- vector / map / unordered_map / string 這些 STL 基本工具箱
- RAII（Resource Acquisition Is Initialization）：C++ 世界的生命線
- 值語意 vs 參考語意，複製、移動（move semantics）為何存在

#### **小提醒：這階段不需要碰大量 meta programming，不然會直接被模板炸到 PTSD。**

### 🚀 第二章：進階語言機制（也就是你開始比較像工程師的階段）

從「寫 C with classes」進化成真正用 C++ 風格思考。

- templates：泛型編程的核心（function template / class template）
- lambda：把行為當物件帶著走
- std::unique_ptr、std::shared_ptr、std::weak_ptr
- move constructor / move assignment operator（move-only 型別設計）
- enum class、constexpr、inline variable
- C++17 / C++20 的語法糖（structured binding、if constexpr、ranges）

#### **在這裡，你會突然理解「C++ 那種怪怪的哲學」。**

### ⚙️ 第三章：記憶體管理與效能

這章是整個 C++ 宇宙最硬的一塊，但也是區分 初階 vs 專業 的地方。

- stack / heap / static segment 的實際運作
- memory alignment（對齊）
- cache-friendly coding：資料布局比演算法更重要
- allocator（客製記憶體配置器）
- smart pointer 的陷阱與正確用法
- zero-copy、move-only API、避免不必要的複製

這章直接變主線任務。

### 🧰 第四章：C++ 工程化與大型專案治理

能寫不代表能「做成產品」。
很多 C++ 新手卡在工程化這段。

- CMake（基本 target、library、include directory、install、find_package）
- shared library vs static library
- linker / ABI / symbol visibility
- cross-compilation（x86 build 到 ARM），Yocto、OpenWRT 會很常遇到
- 套件管理（vcpkg、conan）
- logging（spdlog）、config、CLI、module schema

寫 AI 工程、嵌入式、影像系統——這段直接是營養打滿。

### 🧪 第五章：現代 C++ 與並行

真正把 C++ 當武器用的時候。

- std::thread / mutex / lock_guard
- condition variable
- thread pool 設計
- async / future / promise
- lock-free 結構的基本認知（真的難）
- C++20 coroutine（做網路 I/O 和 pipeline 變很好用）

### 🔌 第六章：和硬體與外部世界接軌

你走的是影像 AI + CUDA + 嵌入式流派，所以這章是你的專屬 buff。

- OpenCV（CPU vs CUDA）
- GStreamer（多串流、多管線）
- CUDA（kernel、shared memory、stream、event）
- TensorRT（engine build、context、bindings、zero-copy）
- FFmpeg（影像解碼）
- Linux system programming（epoll、thread affinity、NUMA）

這裡的學習曲線像是帶刺的螺旋樓梯，但是踩上去很帥。

### 🔬 第七章：最佳化、低延遲、Real-time 系統

這章是大佬級玩家會玩的。

- lock contention 減少
- memory pool / ring buffer
- NUMA-aware scheduling
- 多 stream GPU pipeline
- zero-copy buffer reuse（你已經在做）
- SIMD（SSE/AVX）、intrinsics
- profiling（Perf、Nsight、VTune）

當你需要「毫秒級任務」，這章會變成你的日常武器庫。

🚀 第八章：C++ 在 AI 與視覺系統的全套應用（你現在的主線）

把前面所有技能組合起來，變成高效能 AI 系統：

- YOLO GPU 前處理（CUDA kernel）
- TensorRT 推論
- 多 camera pipeline、thread pool、non-blocking queue
- GStreamer + CUDA zero-copy
- pybind11 將整套 pipeline 封成 Python 模組
- 自製 C++ library（支援 find_package）

TensorRT + gstreamer + CUDA專案規模基本就是這章的世界。