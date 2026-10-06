---
title: "NvBufSurface API reference"
source: "https://claude.ai/chat/a76aa8e8-9a87-41ad-907d-70d4815adb4a"
author:
published:
created: 2026-07-03
description: "Claude conversation with 2 messages"
tags:
  - "clippings"
---
**NvBufSurface** 是 NVIDIA 多媒體 API（常見於 Jetson 平台與 DeepStream SDK）裡用來管理「緩衝表面（buffer surface）」的核心資料結構，主要用途是統一處理影像/視訊幀在不同記憶體型態之間的配置、映射與轉換。

#### 核心概念

**1\. 用途**  
NvBufSurface 讓開發者可以用同一套 API，處理位於不同記憶體空間的影像資料，例如：

- CPU（一般 malloc 記憶體）
- CUDA device memory
- NVMM（NVIDIA 硬體專用記憶體，通常用於 VIC、ISP、編解碼器之間零拷貝傳輸）

**2\. 主要結構**

```c
typedef struct NvBufSurface {
    uint32_t batchSize;
    uint32_t numFilled;
    bool isContiguous;
    NvBufSurfaceMemType memType;
    NvBufSurfaceParams *surfaceList; // 每個 batch 中各幀的參數
} NvBufSurface;
```

`NvBufSurfaceParams` 裡則包含寬高、色彩格式（NV12、RGBA 等）、每個 plane 的 pitch、data pointer 等資訊。

**3\. 常用 API**

- `NvBufSurfaceCreate()` / `NvBufSurfaceDestroy()` ：配置與釋放
- `NvBufSurfaceMap()` / `NvBufSurfaceUnMap()` ：把硬體記憶體映射成 CPU 可存取的指標
- `NvBufSurfaceSyncForCpu()` / `NvBufSurfaceSyncForDevice()` ：在 CPU 與裝置記憶體之間同步資料（cache coherency）
- `NvBufSurfTransform()` ：進行縮放、裁切、色彩空間轉換等操作，通常透過 VIC 或 GPU 硬體加速

#### 常見情境

在 DeepStream pipeline 裡，GStreamer 的 buffer（ `GstBuffer` ）通常會夾帶一個 `NvBufSurface` ，讓 plugin（例如 `nvinfer` 、 `nvvideoconvert` ）能直接存取硬體記憶體，避免不必要的 CPU-GPU 資料搬移。
