### NUMA

UMA（Non-Uniform Memory Access，非一致性記憶體存取架構）是一種多處理器（multi-processor）系統架構設計，用來解決 SMP（Symmetric Multi-Processing） 在 CPU 數量增加時產生的記憶體瓶頸問題。

🧠 基本概念

在 NUMA 架構下：

系統中的每個 CPU socket（或 node）都有 自己的本地記憶體（local memory）。

不同 socket 的記憶體仍然可以被所有 CPU 存取，但：

存取自己 node 的記憶體 → 快

存取別的 node 的記憶體 → 慢（非一致）

所以叫 非一致性（Non-Uniform）。

🔍 對比 SMP（UMA）

| 架構                                   | 記憶體架構                 | 延遲  | 範例                                    |
| ------------------------------------ | --------------------- | --- | ------------------------------------- |
| **UMA (Uniform Memory Access)**      | 所有 CPU 共用一塊記憶體        | 一致  | 傳統雙核心桌機                               |
| **NUMA (Non-Uniform Memory Access)** | 每個 CPU 有自己的記憶體區域，仍可互訪 | 不一致 | 多 socket 伺服器，例如 AMD EPYC / Intel Xeon |

⚙️ NUMA 的硬體實作

以一台 2-socket 伺服器為例：
- Node 0：CPU0 + Memory Bank0
- Node 1：CPU1 + Memory Bank1

兩個 node 間透過 QPI / UPI（Intel） 或 Infinity Fabric（AMD） 連接。
每個 CPU 可以直接存取自己的記憶體，也能透過 interconnect 存取另一個 node 的記憶體（但延遲較高）。

💡 為什麼你應該在意 NUMA？

當你的應用程式：
- 吃大量記憶體（像 TensorRT、OpenCV CUDA pipeline）
- 使用多線程或多 process（尤其是 pin 在不同 CPU 上）
- 跑在多 socket server（例如 Jetson AGX Orin NX 也是 pseudo-NUMA 架構）

那就會遇到：
「記憶體在哪裡」這件事，比你想像的更重要。

🚀 NUMA-aware 最佳實踐

1. 記憶體與 CPU affinity 綁定（NUMA binding）
- Linux 下可用：

``` bash
numactl --cpunodebind=0 --membind=0 ./your_program
```

- 或在程式裡：

``` cpp
numa_run_on_node(0);
numa_set_membind(numa_all_nodes_ptr);
```

2. 多 socket 系統中盡量避免跨 node 存取
- 對大矩陣運算或深度學習推論尤為關鍵。
- 尤其在 TensorRT / OpenMP / ThreadPool 設計中，建議每個 node 有自己的一組工作集與 buffer。
3. 監控 NUMA 效率
可用：

``` bash
numastat
numactl --hardware
lscpu | grep "NUMA"
```

- 或在 /sys/devices/system/node/ 檢查每個 node 的 memory 分佈。

🧩 延伸：NUMA + GPU
在 GPU + CPU 混合系統（像 Jetson、DGX、或多 GPU server）：
- GPU 通常與特定 NUMA node 綁定（PCIe slot 對應）。
- 理想策略：
  - CPU thread 與 GPU 位於同一 NUMA node。
  - 讓 CPU-GPU DMA 傳輸最短，減少跨 node PCIe latency。
例如：

``` bash
nvidia-smi topo -m
```

可查看 GPU 與 NUMA node 的對應關係。

🧬 一句話總結
NUMA 讓多核心 CPU 可以更有效率地共享記憶體，但也要求程式設計者更聰明地「靠近資料」。