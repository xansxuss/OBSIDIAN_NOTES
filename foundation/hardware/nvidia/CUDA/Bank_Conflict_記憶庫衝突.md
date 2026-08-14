**Bank Conflict**（記憶庫衝突）是一種常見的 GPU 效能瓶頸，尤其在 CUDA 或 OpenCL 等平行運算編程中經常出現。 [1](https://modal.com/gpu-glossary/perf/bank-conflict), [2](https://stackoverflow.com/questions/3841877/what-is-a-bank-conflict-doing-cuda-opencl-programming)
核心概念與運作原理

為了實現極高的並行讀寫頻寬，GPU 的共享記憶體（Shared Memory）被劃分為多個獨立的記憶庫（Banks），允許平行存取。 [1](https://modal.com/gpu-glossary/perf/bank-conflict)

- **理想狀態**：若一個 Warp（執行緒束，通常為 32 個執行緒）內的執行緒同時存取**不同**的 Bank，系統能以單一記憶體事務（Memory Transaction）同時處理，速度極快。
- **發生衝突**：若多個執行緒同時請求**同一個** Bank 內的不同記憶體地址，存取動作就會被強制序列化（串行化），大幅延遲執行時間並降低頻寬。 [1](https://github.com/PaddleJitLab/CUDATutorial/blob/develop/docs/09_optimize_reduce/02_bank_conflict/README.md), [2](https://modal.com/gpu-glossary/perf/bank-conflict)

常見情境與解決方案

以下是幾種常見的 Bank Conflict 情況及其優化方式：

1. **跨距存取（Strided Access）**
    - **問題**：若執行緒以特定跨距（如步長為 32 的倍數）存取陣列，所有執行緒會對應到同一個 Bank，導致 32 路衝突。
    - **解決方法**：將資料佈局轉置（Transpose）或加上適當的偏移量。 [1](https://modal.com/gpu-glossary/perf/bank-conflict)[CUDA Programming](https://www.cs.nthu.edu.tw/~cherung/teaching/2010gpucell/CUDA04.pdf)
- **結構體陣列（AoS）**
    - **問題**：若將多個屬性綁在同一個結構體中（Array of Structures），在讀取單一屬性時會跳躍存取，容易引發衝突。
    - **解決方法**：改為結構體陣列（Structure of Array, SoA）或將資料結構平面化，確保連續執行緒存取連續記憶體。 [1](https://www.cs.nthu.edu.tw/~cherung/teaching/2010gpucell/CUDA04.pdf)
- **記憶體填充（Padding）**
    - **問題**：在二維陣列處理中，跨行存取很容易集中在特定 Bank。
    - **解決方法**：在宣告共享記憶體時，手動加入些許虛擬空間（Padding），讓記憶體地址錯開，避免映射到同一個 Bank。 [1](https://modal.com/gpu-glossary/perf/bank-conflict),  [2](https://www.cs.nthu.edu.tw/~cherung/teaching/2010gpucell/CUDA04.pdf)

---

_註：若 Warp 內的所有執行緒存取的是**同一個地址**（Broadcast），硬體會直接將數據廣播，此情況並不會觸發 Bank Conflict。_ [1](https://modal.com/gpu-glossary/perf/bank-conflict)