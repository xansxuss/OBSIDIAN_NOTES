在CUDA 中，Shared Memory (共享記憶體) 是優化效能的核心。它的存取延遲比 Global Memory 低了約 100 倍，頻寬也高得多，運作方式類似於 CPU 的 L1 Cache，但由工程師手動控制。
以下是針對優化 Shared Memory 存取的關鍵技巧與實作策略：

1. Shared Memory 的基本存取模式
在撰寫 Kernel 時，我們會將資料從 Global Memory 搬移到 Shared Memory，進行多次計算後，再寫回 Global Memory。
典型流程：

1. 將資料從 Global Memory 載入 Shared Memory（__syncthreads 確保同步）。
2. 在 Shared Memory 中進行頻繁的計算與資料交換。
3. 計算結果寫回 Global Memory。

2. 避免 Bank Conflict (銀行衝突)Shared Memory 被劃分為 32 個 Banks，對應一個 Warp 中的 32 個 Threads。
- 理想狀況： 每個 Thread 存取不同的 Bank，此時存取是並行的。
- 衝突狀況： 多個 Thread 存取同一個 Bank 中的「不同」位址，存取會變成序列化，導致效能下降。
優化技巧：Padding
當你在處理 2D 矩陣時，如果矩陣寬度剛好是 32 的倍數，時常會發生 Bank Conflict。解決方法： 在宣告 Shared Memory 時多加一列（例如 [TILE_SIZE][TILE_SIZE + 1]），強迫資料錯開，使存取落在不同 Bank。

``` Cpp 
// 避免 Bank Conflict 的 Padding 技巧
__shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; 
```

3. Tiled Matrix Multiplication (分塊矩陣乘法)
這是 Shared Memory 最經典的應用。透過將矩陣切成小塊 (Tiles)，每個 Thread Block 只負責一塊，可以大幅減少對 Global Memory 的重複讀取。
- 優勢： 若 Tile 大小為 $B \times B$，Global Memory 的存取次數可降低為原本的 $1/B$。

4. 使用 __copy_async (CUDA 11+)在較新的 NVIDIA 架構（如 Ampere 及之後）中，可以使用硬體支援的 Asynchronous Copy。
- 傳統方式： 資料需經過 Register 才能到 Shared Memory。
- 優化方式： 資料直接從 Global Memory 搬到 Shared Memory，不占用 Register，且搬運過程中 Thread 可以繼續做其他運算。

5. 減少 Register Spilling雖然 Shared Memory 很快，但它的容量有限（通常每個 Block 為 48KB 到 100KB+）。
- 平衡點： 如果 Shared Memory 用太多，會限制每個 SM (Streaming Multiprocessor) 能同時執行的 Block 數量（降低 Occupancy）。
- 策略： 檢查 nvcc --resource-usage 的輸出，確保 Shared Memory 的使用量不會導致 Active Warps 減少。

總結與建議對 AI 工程師來說，優化卷積神經網路 (CNN) 或 Transformer 的運算子時，Shared Memory 是必經之路。建議你在實作時：
1. 優先確認存取模式：確保 Warp 內的 Threads 存取連續的 Global Memory（Coalesced Access）再進到 Shared Memory。
2. 善用工具：使用 NVIDIA Nsight Compute 觀察 Shared Overheads 與 Bank Conflicts 比例。