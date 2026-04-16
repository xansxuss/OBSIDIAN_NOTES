
`cudaStreamCreateWithFlags` 是優化系統效能（尤其是解決隱性同步問題）的關鍵。

這個 API 的定義如下：

$$cudaError\_t \ cudaStreamCreateWithFlags(cudaStream\_t \ *pStream, \ unsigned \ int \ flags)$$

主要的 `flags` 選項決定了這個串流如何與其他串流（特別是「預設串流」）互動。

---

## 核心 Flags 詳解

目前最常用的兩個 Flag 分別是：

## 1. `cudaStreamDefault` (0x00)

這與 `cudaStreamCreate` 的行為完全一致。

- **特性**：此串流中發出的任務，會受到「舊版預設串流（Legacy Default Stream）」的同步影響。
    
- **影響**：如果程式中其他地方使用了預設串流（未指定串流的 Kernel 呼叫），可能會導致這個自定義串流的非同步任務被迫停下來等待。
    

## 2. `cudaStreamNonBlocking` (0x01)

這是高效能非同步開發最推薦的選項。

- **特性**：此串流**不會**與預設串流（NULL stream）進行隱式同步。
    
- **優勢**：它允許該串流內的任務與預設串流中的任務真正併發執行。如果你正在開發一個需要極低延遲或是多模型併發推論的系統，應優先選用此 Flag。
    

---

## 為什麼要用 Non-Blocking？

在 CUDA 的執行邏輯中，**預設串流（Legacy Default Stream）具有「序列化」的魔力**。任何在預設串流中執行的操作，都會等待所有先前的串流完成，並且會阻塞後續所有串流的操作。

使用 `cudaStreamNonBlocking` 可以打破這種全域鎖定，讓你的自定義串流像一條獨立的快速道路，不受主幹道（預設串流）塞車的影響。

---

## 進階技巧：搭配 Priority（優先權）

如果你除了想控制同步行為，還想讓某些任務「插隊」（例如：處理高優先權的推論請求），你應該使用更強大的：

`cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority)`

- **Priority 數值**：通常 0 代表預設優先權，負數代表較高優先權（例如 -1）。
    
- **查詢範圍**：你可以透過 `cudaDeviceGetStreamPriorityRange(&low, &high)` 取得當前硬體支援的級別。
    

---

## 實作範例

``` C++
cudaStream_t stream;
// 建立一個完全不被預設串流阻塞的串流
cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

if (err != cudaSuccess) {
    fprintf(stderr, "Failed to create stream: %s\n", cudaGetErrorString(err));
}

// 執行 Kernel (非同步)
myKernel<<<grid, block, 0, stream>>>(data);

// 善後
cudaStreamDestroy(stream);
```

---

##  Debug 小錦囊

如果你發現即便用了 `cudaStreamNonBlocking`，Kernel 之間還是沒有重疊（Overlap），請檢查以下兩點：

1. **硬體限制**：GPU 是否有足夠的 **Compute Engines** 或 **Copy Engines**？（較舊的卡只有一個 Copy Engine，無法同時做 H2D 與 D2H）。
    
2. **相依性**：是否在不經意間呼叫了 `cudaDeviceSynchronize()` 或 `cudaStreamSynchronize()`？這會強制拉回 CPU 端等待。