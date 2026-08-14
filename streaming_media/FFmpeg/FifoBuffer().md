在處理多媒體串流時，AVFifo（在較新版本中為 AVFifoBuffer 的後繼者）是一個非常實用的環形佇列（Circular Queue）資料結構。它主要用於緩衝原始資料（如 PCM 音訊或 YUV 影像），協調生產者（Producer）與消費者（Consumer）之間處理速度不一致的問題。

以下是針對 AVFifo 的核心概念與 C 語言實作重點：

1. 核心特性
環形緩衝區：自動處理記憶體回繞，不需手動搬移資料。

自動擴展：可設定在空間不足時自動增長空間。

高效能：避免頻繁的 malloc 與 free，適合高效能 AI 影像處理流水線。

2. 常用 API 與實作流程
在 FFMPEG 5.x 之後，建議使用新的 AVFifo 相關函式（定義於 <libavutil/fifo.h>）：

初始化與銷毀
``` cpp
// 建立一個 FIFO，每個元素大小為 1 byte，初始容量為 1024
AVFifo *fifo = av_fifo_alloc2(1024, 1, AV_FIFO_FLAG_AUTO_GROW);

// 釋放記憶體
av_fifo_freep2(&fifo);
```

寫入資料 (Producer)

```cpp
uint8_t *data = /* 你的影像或音訊資料 */;
size_t data_size = 512;

// 檢查可用空間
if (av_fifo_can_write(fifo) >= data_size) {
    av_fifo_write(fifo, data, data_size);
}
```

讀取資料 (Consumer)

```cpp
uint8_t output_buf[512];
// 從 FIFO 讀取並移除資料
av_fifo_read(fifo, output_buf, sizeof(output_buf));

// 如果只想看資料但不移除 (Peek)
av_fifo_peek(fifo, output_buf, sizeof(output_buf), 0);
```

### 3. 進階操作：直接存取記憶體

為了追求極致效能，有時我們想避免中間緩衝區的拷貝。`av_fifo_write_from_cb` 和 `av_fifo_read_to_cb` 允許你傳入回呼函式（Callback），直接在 FIFO 的內部記憶體上操作。

### 4. 開發建議

- **執行緒安全**：`AVFifo` 本身**並非** Thread-safe。在多執行緒環境（例如：一個 Thread 解碼，另一個 Thread 進行 AI 推論）中，必須搭配 `pthread_mutex` 或 C++11 的 `std::mutex` 使用。
    
- **非標準函式庫思維**：由於你傾向不使用 C++ 標準函式庫（STL），`AVFifo` 是替代 `std::deque` 或 `std::queue` 處理 Byte stream 的極佳選擇，它更貼近底層且記憶體佈局可控。
    
- **音訊處理同步**：在處理音訊時，由於取樣率（Sample Rate）固定，`AVFifo` 常被用來累積足夠長度的資料（例如 20ms 的音訊幀）再送入推論模型。