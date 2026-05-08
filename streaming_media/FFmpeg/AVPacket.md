### AVPacket (壓縮後的資料數據包)

`AVPacket` 主要用來存放編碼後（Compressed）的數據，例如從影片檔案（如 .mp4）讀取出來的 H.264 位元流。

- **用途**：包含一個或多個編碼後的影格。
    
- **關鍵屬性**：包含 `pts` (Presentation Timestamp)、`dts` (Decoding Timestamp) 以及指向壓縮數據的指標。
    
- **記憶體管理**：`av_packet_alloc()` 僅配置結構本體。實際存放資料的緩衝區通常是透過 `av_read_frame()` 填入，最後必須使用 `av_packet_free(&packet)` 釋放。