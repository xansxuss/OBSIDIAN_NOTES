在開發高效能運算（HPC）或深度學習模型時，**CUDA Stream** 是實現硬體併發（Concurrency）的核心機制。

簡單來說，`cudaStream_t` 是一個**處理程序句柄（Handle）**，代表 GPU 上的一個指令序列。

---

## 核心概念：為什麼需要 Stream？

在預設情況下，CUDA 會使用 **預設串流（Default Stream/Null Stream）**。這是一個同步串流，所有的任務（Kernel 啟動、記憶體拷貝）都會按順序排隊執行。

使用自定義的 `cudaStream_t` 可以達成以下目標：

1. **計算與傳輸重疊（Overlap of Copy and Kernel Execution）**：當 GPU 在執行 Kernel A 時，可以同時透過 PCIe 進行 Kernel B 的資料傳輸。
    
2. **併發 Kernel 執行**：如果 GPU 資源充足（例如運算單元 SM 未佔滿），多個小型的 Kernel 可以同時在同一個 GPU 上跑。
    

---

## 基本操作與 API

以下是處理串流的標準生命週期：

## 1. 宣告與建立

```C++
cudaStream_t stream;
cudaStreamCreate(&stream); // 建立非阻塞的串流
```

_註：若要更細緻控制，可使用 `cudaStreamCreateWithFlags`。_

## 2. 在串流中執行任務

大部分的 CUDA API 都有非同步版本（Async），並接受 `stream` 作為參數：

- **記憶體拷貝**：`cudaMemcpyAsync(dst, src, size, kind, stream);`
    
- **Kernel 啟動**：`kernel<<<grid, block, sharedMem, stream>>>(args);`
    

## 3. 同步與銷毀

由於 `Async` 函式會立即回傳給 CPU，你需要確保任務完成：

- **串流同步**：`cudaStreamSynchronize(stream);`（等待該串流完成）
    
- **銷毀**：`cudaStreamDestroy(stream);`
    

---

## 併發模式圖解

當你使用多個串流時，理想的時序會如下圖所示，藉此極大化 GPU 利用率：

- **序列執行 (Default Stream)**：`Copy H2D` -> `Kernel` -> `Copy D2H`
    
- **併發執行 (Multiple Streams)**：當 Stream 1 在跑 `Kernel` 時，Stream 2 已經在跑 `Copy H2D`。
    

---

## 注意事項（工程師坑位提醒）

1. **分頁鎖定記憶體 (Pinned Memory)**： 要實現 `cudaMemcpyAsync` 的非同步傳輸，主機端的記憶體必須使用 `cudaHostAlloc` 或 `cudaMallocHost` 分配。一般的 `malloc` 是無法達成傳輸重疊的。
    
2. **預設串流的阻塞特性**： Legacy Default Stream 會跟所有其他串流同步。如果你混合使用預設串流與自定義串流，可能會導致原本想做的非同步操作失效。
    
    - _建議：_ 在編譯時加入 `--default-stream per-thread` 或 `#define CUDA_API_PER_THREAD_DEFAULT_STREAM`。
        
3. **多串流的資源競爭**： 雖然串流在邏輯上是並行的，但底層硬體資源（如 Copy Engine 或 SM）是有限的。如果 Kernel 太大佔滿了 SM，即便開了多串流，Kernel 之間仍會變成排隊執行。

