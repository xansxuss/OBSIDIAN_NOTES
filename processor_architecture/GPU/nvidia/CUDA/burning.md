先定義一件事：你到底要燒什麼
GPU 不只一塊矽，燒錯地方等於白忙。

常見目標有幾種：

SM / CUDA core：算力、時脈穩定度

VRAM：頻寬、錯誤、溫升

L2 / memory fabric：長時間壓力

PCIe / NVLink：傳輸與同步

Driver + runtime 穩定性：memory leak、context crash

真實工作負載：例如你那套 OpenCV + CUDA + TensorRT + zero-copy pipeline

第一個原則：
燒機一定要貼近實際 workload。
核心架構思路（高層）

一個好的燒機程式，結構會長這樣：

- 初始化 GPU（context、device、clock lock）

- 配置「可放大」的 workload

- 多 stream / 多 kernel / 多 buffer 並行

- 可選的 host ↔ device 資料搬移

- 監控 + 記錄

- 可長時間跑（數小時～數天）

不是一次 kernel launch 就結束那種。

SM / 算力燒法（最乾淨）

目標：讓 SM 一直滿載、不被 memory 卡住。

思路：

- 設計 高 arithmetic intensity 的 kernel

- 少 global memory，多 register / shared memory

- 大量 FMA、matrix-like 操作

典型套路：

自- 己寫一個假的 GEMM / convolution

- 或者 unrolled 的 FMA loop

- grid size > SM 數量 × 好幾倍，避免 scheduler 空轉

關鍵不是算什麼，而是：

- occupancy 拉滿

- kernel runtime 長（避免 launch overhead）

VRAM / 記憶體燒法

目標：讓 HBM / GDDR 一直喘。

思路：

- 分配接近 VRAM 上限的 buffer

- 不斷做大範圍 read/write

- pattern 要變，避免 cache 全命中

實務做法：

- stride 掃描

- random access（但要注意不要被 L2 吃掉）

- memcpy device-to-device loop

這一類很容易燒出：

- 記憶體錯誤

- driver 掛掉

- ECC error（如果有）

PCIe / Host ↔ Device 燒法

這個常被忽略，但在實際系統超重要。

思路：

- pinned memory

- async memcpy

- 多 stream 同時傳

如果你在做影像串流（RTSP、DMA-BUF 那種），
這一段其實比純 CUDA kernel 還現實。

- 多 Stream + 同步地獄（穩定性測試）

- 真正會出事的地方在這。

- 燒機不該是：

while(true) launch kernel

而是：

- N 個 stream

- 每個 stream 不同 workload

- 交錯 event / sync

- 偶爾做 realloc / free

這會逼出：

- race condition

- driver bug

- 你自己 RAII 包裝寫爛的地方 😈

溫度 / 功耗控制

燒機一定要「可量測」。

至少要能：

- 定期呼叫 NVML

- 記錄：

    - temperature

    - power draw

    - clock throttle reason

    - memory usage

重點不是即時顯示，而是：
log 下來，事後看曲線。

溫度穩定 ≠ 沒問題
有時是 power throttling 在偷偷救場。

失敗條件要明確

燒機程式一定要定義「什麼叫 fail」：

- kernel launch error

- CUDA API timeout

- 計算結果錯誤（checksum / reference）

- NVML 回報 XID error

- performance 明顯掉速

沒有 fail 定義的燒機，只是自我安慰。

為什麼不直接用現成工具？

不是不能用（如 stress-ng、cuda-sample、burn），
但工程師常忽略一件事：

現成工具不會幫你測「你自己的 pipeline」。

你現在做的東西如果是：

- zero-copy

- GpuMat / DMA-BUF

- TensorRT

- multi-stream inference

那你最該燒的，就是那整條。

一句工程師版總結

燒機不是「把 GPU 用到 100%」，
而是：

用你未來最怕出問題的方式，
把它連續折磨 72 小時，
還要能留下證據。