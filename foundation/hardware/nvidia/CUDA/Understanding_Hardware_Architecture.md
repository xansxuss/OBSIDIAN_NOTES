## Understanding Hardware Architecture


1. 執行配置的映射：Grid $\rightarrow$ Block $\rightarrow$ Thread這部分描述的是「任務如何分配到硬體核心」。

| 邏輯層級 (Software) | 映射硬體 (Hardware) | 說明 |
| --- | --- | --- |
| Grid | GPU (Whole Device) | 整個 Kernel 啟動時的範圍。 |
| Block | Streaming Multiprocessor (SM) | 一個 Block 會被分配到單一個 SM 執行。SM 的資源（如 Shared Memory）由該 SM 上的所有 Block 共享。 |
| Thread | CUDA Core / SP | 最小執行單元。多個 Thread 組成 Warp（通常是 32 個），這是硬體排程與執行的最小單位。 |

核心觀念
- SM 是關鍵： 一個 Block 只能在一顆 SM 上執行，不能跨 SM；但一顆 SM 可以同時容納多個 Blocks（視暫存器與記憶體資源而定）。
- Warp 執行： 硬體實際上是以 Warp 為單位進行指令派發。如果你的 Thread 索引沒寫好導致 Warp Divergence，效能會大幅下降。

2. 記憶體架構的映射：Global -> Shared -> Register這部分決定了「資料存取的延遲（Latency）與頻寬」。


| 記憶體類型 | 硬體位置 | 存取速度 | 作用範圍與特性 |
| --- | --- | --- |
| Register | SM 內的暫存器檔案 | 極快 (0 cycle) | Thread 私有。數量有限，用太多會降低 occupancy（可同時執行的執行緒數量）。 |
| Shared Memory | SM 內的 L1/SRAM | 很快 (低延遲) | Block 內共享。主要用於 Thread 間通訊或作為 User-managed cache，需注意 Bank Conflict。 |
| Global Memory | GPU 顯存 (VRAM) | 慢 (400-800 cycles) | 整個 Grid 可見。重點在於 Coalesced Access（合併存取），確保一次指令能抓取連續地址資料。 |

3. 綜合映射關係圖
將兩者結合，你會發現一個清晰的層級對應：

    1. Thread ↔ Register: 每個執行緒有自己的私有空間，運算最快，但空間最小。

    2. Block ↔ Shared Memory: 同一個 Block 裡的 Thread 可以透過這塊區域交換資料，這也是 CUDA 程式碼優化的「兵家必爭之地」。

    3. Grid ↔ Global Memory: 跨 Block 的資料交換必須回到 Global Memory（或是透過 L2 Cache），這會帶來巨大的延遲。

4. 給 AI 工程師的優化心法
在開發模型推理或自定義 Op（如 FlashAttention 實作）時，請記住以下映射原則：

    - 避免 Register Spilling: 如果你的 Kernel 變數太多，編譯器會把變數丟到 Local Memory（實體上是 Global Memory），這會讓速度掉進深淵。

    - 善用 Shared Memory 進行 Tiling: 將資料從 Global 搬進 Shared，再讓 Block 內的 Thread 重複使用，減少對 VRAM 的頻寬壓力。

    - Occupancy 平衡: 增加每個 Block 的 Thread 數或 Shared Memory 用量，雖然能提高單個 Block 的能力，但可能導致 SM 能同時執行的 Block 變少，反而降低了隱藏延遲（Latency Hiding）的能力。