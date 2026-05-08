在 FFmpeg 解碼流程中，從解碼器（Decoder）提取原始影像資料（Raw Data）的關鍵步驟。

當呼叫 `avcodec_receive_frame` 時，實際上是在執行解碼管線（Pipeline）的後半段。這通常與 `avcodec_send_packet` 成對出現。

### 1. 函數邏輯與回傳值處理

這個函數的行為是非同步的（以 Buffer 的角度來看），其回傳值決定了下一步的操作：

- **`0` (成功)**：表示解碼器成功輸出一影格，資料已填入 `pFrame`。
    
- **`AVERROR(EAGAIN)`**：表示目前解碼器內部的資料不足以輸出一幀 Frame。你需要先呼叫 `avcodec_send_packet` 送入更多資料，再回來嘗試 receive。
    
- **`AVERROR_EOF`**：解碼器已經完全沖刷（Flushed）完畢，不會再有任何影像輸出。這通常發生在影片結尾。
    
- **`AVERROR(EINVAL)`**：解碼器未開啟或狀態錯誤。

### 2. 標準解碼迴圈 (C/C++ 實作範例)

針對 `pFrame`，典型的實作邏輯如下：

```C++
int ret = 0;
// 1. 將壓縮過的 packet 送入解碼器
ret = avcodec_send_packet(pCodecCtx, packet);
if (ret < 0) {
    // 錯誤處理
    return ret;
}

// 2. 嘗試從解碼器拿出解碼後的 frame (一個 packet 可能解出多個 frame)
while (ret >= 0) {
    ret = avcodec_receive_frame(pCodecCtx, pFrame);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        break; 
    } else if (ret < 0) {
        // 解碼過程發生錯誤
        return ret;
    }

    // 成功拿到 pFrame，這裡可以進行 YUV -> RGB 轉換或渲染
    // process_image(pFrame);

    // 重要：使用完畢後要重置 Frame，以便下一次 receive 重複使用
    av_frame_unref(pFrame); 
}
```

### 開發細節：記憶體與效能

- **Buffer Ownership**：`avcodec_receive_frame` 會讓 `pFrame->data` 指向解碼器內部管理的 Buffer。這就是為什麼在迴圈最後必須呼叫 `av_frame_unref`，它會將引用計數減一，而不是直接釋放 `AVFrame` 結構本身。
    
- **多執行緒解碼**：如果在 `pCodecCtx` 中設定了 `thread_count > 1`，`receive_frame` 的順序通常會由 FFmpeg 內部維護，確保 PTS (Presentation Timestamp) 的順序是正確的。
    
- **不使用 STL 的考量**：在處理高性能影像串流時，避免 STL 是明智的。可以考慮實作一個簡單的 **Object Pool** 來管理 `AVFrame`，避免頻繁的 `alloc` 與 `free` 造成的記憶體碎片。