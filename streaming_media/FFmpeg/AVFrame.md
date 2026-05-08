### AVFrame (解碼後的原始影格)

`AVFrame` 用來存放解碼後（Uncompressed）的原始數據，例如 YUV 或 RGB 格式的影像，或是 PCM 格式的音訊。

- **用途**：代表一幀完整的影像或一段音訊採樣。
    
- **關鍵屬性**：`data` 指標陣列（存放不同 Plane 的數據）、`linesize` (跨度/步長)、`width`、`height`、`format` 等。
    
- **記憶體管理**：同樣地，`av_frame_alloc()` 只配置結構。實際的 Pixel Buffer 需透過解碼器（`avcodec_receive_frame`）填入或手動透過 `av_frame_get_buffer` 配置。釋放時請用 `av_frame_free(&pFrame)`。