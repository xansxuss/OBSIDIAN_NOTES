---
title: "ffmpeg 引用計數複製 "
source: "https://gemini.google.com/app/bc08b89d8e8905ee"
author:
published:
created: 2026-06-22
description: "Gemini conversation with 2 messages"
tags:
  - "clippings"
---
在 FFmpeg 的開發中，處理影音資料（如 `AVFrame` 和 `AVPacket` ）時，為了避免記憶體複製（Deep Copy）帶來的效能開銷，FFmpeg 廣泛使用了引用計數（Reference Counting）機制。

如果你想「複製」一個對象，但實際上只增加引用計數、共享底層的資料記憶體，應該使用 FFmpeg 提供的高效 API，而不是自己去寫 `memcpy` 或手動賦值。

## 核心 API 與用法

針對 `AVFrame` 和 `AVPacket` ，FFmpeg 提供了對應的引用計數複製函式：

### 1\. AVFrame 的引用計數複製

如果你想建立一個新的 `AVFrame` ，並讓它指向與來源（src）相同的影像/音訊資料：

```
// 初始化目標 frame
AVFrame *dst = av_frame_alloc();
if (!dst) {
    // 錯誤處理
}

// 進行引用計數複製
// 這會複製 src 的屬性（如 pts, width, height），並增加底層 buffer 的引用計數
int ret = av_frame_ref(dst, src);
if (ret < 0) {
    // 錯誤處理
    av_frame_free(&dst);
}
```

> ⚠️ **注意：** 當你不再使用 `dst` 時，必須呼叫 `av_frame_unref(dst)` 來減少引用計數。如果計數歸零，FFmpeg 會自動釋放底層的記憶體。最後再呼叫 `av_frame_free(&dst)` 釋放結構體本體。

### 2\. AVPacket 的引用計數複製

對於編碼後的壓縮封包（Packet），邏輯完全相同：

```
AVPacket *dst = av_packet_alloc();
if (!dst) {
    // 錯誤處理
}

// 進行引用計數複製
int ret = av_packet_ref(dst, src);
if (ret < 0) {
    // 錯誤處理
    av_packet_free(&dst);
}
```

> ⚠️ **注意：** 使用完畢後，需呼叫 `av_packet_unref(dst)` ，最後呼叫 `av_packet_free(&dst)` 。

## 記憶體架構概念

當你呼叫 `av_frame_ref(dst, src)` 時，背後的記憶體架構如下圖所示：

- `AVFrame` 結構體本身會複製一份（獨立的 `pts` 、 `width` 、 `height` 等中介資料）。
- 但儲存實際影音資料的 `AVBufferRef` （ `dst->buf[0]` ）會指向同一個底層的 `AVBuffer` 。
- 底層 `AVBuffer` 的 `refcount` （引用計數）會加 1。

## 什麼時候該用「硬複製（Deep Copy）」？

如果你需要 **修改** 複製出來的影像內容（例如寫入浮水印、剪裁），就 **不能** 使用引用計數複製，因為這會同時污染到原始的 `src` 資料。

這時候你必須配置新的記憶體並進行實體複製：

```
AVFrame *dst = av_frame_alloc();
dst->format = src->format;
dst->width  = src->width;
dst->height = src->height;

// 配置全新的實體記憶體
av_frame_get_buffer(dst, 0); 

// 實體複製影像資料
av_frame_copy(dst, src); 

// 複製中介資料 (pts, flags 等)
av_frame_copy_props(dst, src);
```

## 常見陷阱

1. **忘記呼叫 `av_frame_unref`** ：會導致底層的大塊記憶體（如 4K 影像 buffer）無法被釋放，造成嚴重的 Memory Leak。
2. **直接 `dst = src;`**：這是指標賦值，並非引用計數複製。這會導致兩個指標指向同一個 `AVFrame` 結構體，當其中一方呼叫 `av_frame_free` 後，另一方就會變成懸空指標（Dangling Pointer），導致 Crash。

[^1]: 
