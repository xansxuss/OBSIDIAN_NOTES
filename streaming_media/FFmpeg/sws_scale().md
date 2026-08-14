
在影片處理或電腦視覺開發中，sws_scale 是 FFmpeg (libswscale) 函式庫裡最核心的函式。它主要負責處理圖像內容的縮放（Scaling）、像素格式轉換（Pixel Format Conversion）以及色彩空間變換（Color Space Conversion）。

## 核心功能
- 縮放 (Scaling)：調整影像的寬度與高度。

- 像素格式轉換：例如將 YUV420P 轉換為 RGB24，以便在螢幕上顯示。

- 色彩空間變換：例如處理 BT.601 與 BT.709 之間的顏色差異。

## 函式原型

```cpp
int sws_scale(struct SwsContext *c, 
              const uint8_t *const srcSlice[], const int srcStride[], 
              int srcSliceY, int srcSliceH, 
              uint8_t *const dst[], const int dstStride[]);
```

參數詳解：
- `c`: 指向 SwsContext 的指標，這需要事先透過 sws_getContext 或 sws_getCachedContext 初始化。

- `srcSlice[]`: 輸入來源的資料指標陣列（例如 Y, U, V 分量的記憶體位址）。

- `srcStride[]`: 輸入影像每行（Line）的位元組跨度（Stride/Pitch）。

- `srcSliceY`: 開始處理的行索引，通常傳入 0。

- `srcSliceH`: 處理的行數（影像高度）。

- `dst[]`: 輸出目標的資料指標陣列。

- `dstStride[]`: 輸出影像的每行位元組跨度。

## 基本使用流程
在 C++ 環境下，建議的實作步驟如下：

1. 初始化 Context：設定來源與目標的寬高及格式。

```cpp
struct SwsContext* sws_ctx = sws_getContext(
    srcW, srcH, srcFormat,
    dstW, dstH, dstFormat,
    SWS_BILINEAR, // 縮放演算法（雙線性插值）
    NULL, NULL, NULL
);
```

2. 執行轉換：呼叫 sws_scale。

3. 釋放資源：

```cpp
sws_freeContext(sws_ctx);
```

## 開發建議
- 效能優化 (Performance)：
	`sws_scale` 在 CPU 上執行，如果影像解析度極高（如 4K），可能會成為效能瓶頸。在 AI 應用中，通常會考慮使用硬體加速（如 NVIDIA 的 NPP 或 OpenCV 的 CUDA 模組）來取代。

	如果需要重複轉換相同規格的影像，務必重複使用同一個 SwsContext，避免頻繁分配記憶體。

- 記憶體對齊 (Alignment)：
	在配置 dst 緩衝區時，建議使用 av_image_alloc 或 posix_memalign 以確保記憶體是對齊的，這能觸發 CPU 的 SIMD 指令集加速。