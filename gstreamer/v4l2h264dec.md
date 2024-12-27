image_format : 
    SINK 範本 (sink):
      - Capabilities： 只接收 video/x-h264 格式的影像，要求為 byte-stream 格式，並且 alignment 設置為 au（即編碼數據對齊單元）。
    SRC 
      - 輸出的影像為 video/x-raw 格式。
      - format： NV12_16L32S。
元素屬性：
    - automatic-request-sync-point-flags：
        用於設置自動請求同步點的標誌。同步點標誌有兩個選項：
            discard-input：丟棄輸入數據。
            corrupt-output：將損壞的輸出丟棄。
    - automatic-request-sync-points：
        設置是否在需要的情況下自動請求同步點。預設值為 false，表示不會自動請求。
    - capture-io-mode：
        設置捕獲 I/O 模式，指定視頻數據的輸入方式。可選的模式包括：
            auto：自動選擇模式。
            rw：讀寫模式。
            mmap：內存映射模式。
            userptr：使用者指針模式。
            dmabuf：DMA 緩衝區模式。
            dmabuf-import：導入 DMA 緩衝區模式。
    - output-io-mode：
        設置輸出端的 I/O 模式，與 sink 端口對應，默認為 auto
            auto (0)：自動選擇 I/O 模式，根據系統或硬體環境自動決定使用哪種模式。
            rw (1)：讀寫模式，表示使用常規的讀寫方式，這是一種通用的 I/O 模式。
            mmap (2)：使用內存映射（memory mapping）進行 I/O。這種模式將設備的緩衝區映射到進程的內存中，進行高速的數據讀寫。
            userptr (3)：使用使用者空間指標（user pointer）進行 I/O。在這種模式下，使用者可以提供一個緩衝區的指標，並讓 V4L2 驅動直接操作這些緩衝區。
            dmabuf (4)：使用 DMA 緩衝區（Direct Memory Access Buffer）。這是一種高效的記憶體共享方式，可以讓不同的硬體或進程共享緩衝區，避免了複製數據的開銷。
            dmabuf-import (5)：導入現有的 DMA 緩衝區。這個模式允許導入來自其他進程或設備的 DMA 緩衝區，進行共享和處理。
