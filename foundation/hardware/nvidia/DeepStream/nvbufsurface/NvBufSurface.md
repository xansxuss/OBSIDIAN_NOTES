---
title: NvBufSurface
source: https://gemini.google.com/app/61b28cd7dced2d91
author:
published:
created: 2026-07-09
description: Gemini conversation with 4 messages
tags:
  - nvidia
  - jetson
  - hardware_codec
---
`NvBufSurface` 是 NVIDIA 在 **DeepStream SDK** 以及 **Jetson Linux (L4T)** 多媒體 API 中非常核心的一個 C/C++ 結構體。

簡單來說，它是 NVIDIA 用來 **在記憶體中配置與管理影像/視訊緩衝區（Buffer）的硬體加速架構** 。不論是硬體解碼後的影片畫面、TensorRT 模型的輸入，還是進行影像縮放（Scaling）或色彩空間轉換（Color Space Conversion）後的資料，大多都是以 `NvBufSurface` 的形式在處理。

## 核心設計理念與優勢

1. **零拷貝（Zero-Copy）與硬體加速** ： 傳統的影像處理（如 OpenCV）通常在 CPU 記憶體中進行，若要交給 GPU 處理，需要進行繁重的記憶體複製（Memory Copy）。 `NvBufSurface` 支援硬體直接存取，影像資料可以在解碼器（NVDEC）、GPU、縮放器（VIC/NvMedia）以及顯示器（NVENC/Display）之間直接傳遞，完全不需要經過 CPU 搬運，極大化提升吞吐量。
2. **跨平台統一介面** ： 同一個結構體同時支援 NVIDIA 的 **dGPU（獨立顯示卡，如 RTX、A100）** 與 **Jetson（嵌入式系統，如 Orin、Xavier）** 。
3. **支援多批次處理（Batching）** ： 在 AI 推理中，我們常需要同時餵給模型多張影像（Batch）。 `NvBufSurface` 結構體內部設計了 `surfaceList` 陣列，可以同時管理一個 Batch 中的多個 `NvBufSurfaceParams` （個別畫面的結構體）。

## 重要欄位解析

在 `nvbufsurface.h` 中，這個結構體的定義大致如下（以下為概念架構，不使用 C++ 標準函式庫）：

``` C
typedef struct _NvBufSurface {
  uint32_t gpuId;              // GPU 的設備 ID
  uint32_t batchSize;          // 這一組 Surface 包含了多少張影像 (Batch 大小)
  uint32_t numFilled;          // 目前有效、有填入影像資料的數量
  NvBufSurfaceMemType memType; // 記憶體類型（例如 CUDA Device, CUDA Unified, NVBUF_MEM_SURFACE_ARRAY 等）
  NvBufSurfaceParams *surfaceList; // 指向各個影像詳細參數的陣列指標
} NvBufSurface;
```

而 `surfaceList` 裡面的每一張影像 `NvBufSurfaceParams` 則包含了：

- 寬度（ `width` ）、高度（ `height` ）
- 色彩格式（ `colorFormat` ，如 NV12, RGBA, YUV420）
- `dataPtr` ：指向實際記憶體位址的指標。
- `planeParams` ：影像各個平面（Plane）的步長（Pitch）、偏移量（Offset）與大小。這在處理 NV12（Y 平面與 UV 平面分離）這種格式時非常關鍵。

## 常見的記憶體類型 (NvBufSurfaceMemType)

根據你部署的硬體平台不同，選擇的記憶體類型也大不相同：

| 記憶體類型 | 說明 | 適用平台 |
| --- | --- | --- |
| `NVBUF_MEM_CUDA_DEVICE` | 純 GPU 記憶體（Device Memory），CPU 無法直接存取。 | dGPU (如伺服器/桌機) |
| `NVBUF_MEM_CUDA_PINNED` | 鎖定分頁的 CPU 記憶體（Pinned Memory），可加速 CPU-GPU 傳輸。 | dGPU |
| `NVBUF_MEM_SURFACE_ARRAY` | Jetson 專屬的硬體區塊記憶體（Block Linear），適合硬體解碼與轉碼。 | Jetson (嵌入式) |
| `NVBUF_MEM_HANDLE` | 運用系統的原生 Handle（如 Linux 的 dma-buf fd）來進行零拷貝分享。 | Jetson / 跨進程 |

## 常用 API 函式

NVIDIA 提供了一系列的 C 語言函式來管理這些 Surface，通常以 `NvBufSurface*` 開頭：

- **`NvBufSurfaceCreate()`** ：配置記憶體，建立一個 `NvBufSurface` 實例。
- **`NvBufSurfaceDestroy()`** ：釋放記憶體。
- **`NvBufSurfaceMap()`** ：將 GPU 或是硬體緩衝區對應（Map）到 CPU 的記憶體空間，讓 CPU 暫時可以存取（讀寫）裡面的像素資料。
- **`NvBufSurfaceUnMap()`** ：結束 CPU 存取，解除對應。
- **`NvBufSurfaceCopy()`** ：在兩個 Surface 之間複製資料。
- **`NvBufSurfaceMemSet()`** ：初始化或清空 Surface 的數值（類似 `memset` ）。

## C/C++ 開發注意事項（針對非標準庫環境）

在開發上不使用標準函式庫，操作 `NvBufSurface` 時有幾點建議：

1. **記憶體生命週期** ： `NvBufSurfaceCreate` 分配的是底層驅動與硬體級別的實體記憶體，請務必嚴格遵循 `Create` 與 `Destroy` 的配對，否則會導致嚴重的 GPU 記憶體洩漏（Memory Leak）。
2. **自訂指標操作** ：若要在自訂演算法中直接修改影像像素，必須先呼叫 `NvBufSurfaceMap` ，並透過 `surfaceList[i].planeParams` 算準每個 Plane 的 `pitch` （每行位元組數，通常包含硬體對齊，不能直接用 `width * channels` 計算），再用純 C 的指標偏移去操作 `dataPtr` 。

---

這裡提供一個實用的 C 語言範例，展示如何在不依賴 C++ 標準函式庫（STL）的情況下，正確配置 `NvBufSurface` 、將其對應（Map）到 CPU 空間進行影像資料處理（例如填入自訂顏色），最後安全地釋放記憶體。

這個流程是 NVIDIA DeepStream 自訂外掛（Plugin）或 Jetson 多媒體開發中最基礎且核心的操作。

### NvBufSurface 配置與指標操作範例

``` C
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "nvbufsurface.h"

// 假設我們在 Jetson 平台或 dGPU 上配置一個 RGBA 格式的緩衝區
#define IMG_WIDTH   1920
#define IMG_HEIGHT  1080
#define BATCH_SIZE  1

int main() {
    NvBufSurface *surf = NULL;
    
    // 1. 設定配置參數
    NvBufSurfaceCreateParams create_params;
    create_params.gpuId = 0;
    create_params.width = IMG_WIDTH;
    create_params.height = IMG_HEIGHT;
    create_params.size = 0; // 設為 0，系統會根據 width, height, colorFormat 自動計算所需大小
    create_params.colorFormat = NVBUF_COLOR_FORMAT_RGBA; // 使用 RGBA 格式
    
#if defined(__aarch64__)
    // Jetson 嵌入式平台：使用統一記憶體或硬體表面陣列以達到零拷貝
    create_params.memType = NVBUF_MEM_SURFACE_ARRAY; 
#else
    // dGPU 獨立顯示卡平台：使用主機與裝置可同時存取的 Pinned/Unified 記憶體以便 CPU 存取
    create_params.memType = NVBUF_MEM_CUDA_PINNED;
#endif

    // 2. 建立 NvBufSurface 實例
    if (NvBufSurfaceCreate(&surf, BATCH_SIZE, &create_params) != 0) {
        // 取代標準庫的錯誤處理，直接輸出至 stderr
        fprintf(stderr, "Error: Failed to create NvBufSurface.\n");
        return -1;
    }

    // 3. 將 GPU/硬體記憶體對應（Map）到 CPU 存取空間
    // 參數 -1, -1 代表對應此 Batch 中的所有 Surfaces 以及所有影像平面 (Planes)
    if (NvBufSurfaceMap(surf, -1, -1, NVBUF_MAP_READ_WRITE) != 0) {
        fprintf(stderr, "Error: Failed to map NvBufSurface to CPU space.\n");
        NvBufSurfaceDestroy(surf);
        return -1;
    }

    // 4. 透過純指標操作影像像素（此處以第一張圖為例）
    NvBufSurfaceParams *surface_params = &surf->surfaceList[0];
    
    // 取得實際的記憶體指標與 Pitch
    uint8_t *pixel_data = (uint8_t *)surface_params->mappedAddr.addr[0];
    uint32_t pitch = surface_params->planeParams.pitch[0];
    uint32_t height = surface_params->planeParams.height[0];

    // 核心安全法則：硬體配置的 pitch 通常會大於 width * bytes_per_pixel（因為硬體記憶體對齊限制）
    // 絕對不能使用傳統的 \`width * 4\` 當作一行的跨度，必須使用驅動回傳的 pitch。
    for (uint32_t y = 0; y < height; y++) {
        // 計算當前行（Row）的起始指標
        uint8_t *row_ptr = pixel_data + (y * pitch);
        
        for (uint32_t x = 0; x < IMG_WIDTH; x++) {
            // RGBA 格式，每個像素佔用 4 位元組
            uint32_t pixel_idx = x * 4;
            
            row_ptr[pixel_idx + 0] = 255; // Red
            row_ptr[pixel_idx + 1] = 0;   // Green
            row_ptr[pixel_idx + 2] = 0;   // Blue
            row_ptr[pixel_idx + 3] = 255; // Alpha
        }
    }

    // 5. 確保資料同步（Sync）
    // 當 CPU 修改了記憶體後，若使用的是有快取（Cache）的記憶體類型，
    // 必須呼叫 Sync 確保資料寫回硬體，以免後續 GPU 或硬體單元讀到舊資料。
    NvBufSurfaceSyncForDevice(surf, -1, -1);

    // 6. 解除對應（Unmap）與釋放記憶體
    NvBufSurfaceUnMap(surf, -1, -1);
    NvBufSurfaceDestroy(surf);

    printf("Successfully allocated, modified, and released NvBufSurface.\n");
    return 0;
}
```

### 技術細節與防坑指南

1. **`pitch` 與 `width` 的差異（關鍵）**
	程式碼中特別強調了 `pitch` 。硬體為了加速記憶體存取（例如 64-byte 或 128-byte 對齊），在影像每一行的末尾可能會補上一些空白位元組（Padding）。如果你直接用 `row_ptr = pixel_data + (y * width * 4)` ，影像畫面將會出現嚴重的斜向撕裂或破圖。
2. **多平面（Multi-plane）格式處理**
	上面的範例使用的是單一平面的 `RGBA` 。如果你處理的是視訊常見的 **`NV12`** 格式，它會包含兩個平面：
	- `surf->surfaceList[0].mappedAddr.addr[0]` 指向 **Y 平面** （亮度，大小為 $Width \times Height$ ）。
		- `surf->surfaceList[0].mappedAddr.addr[1]` 指向 **UV 平面** （色彩，交錯排列，大小為 $Width \times \frac{Height}{2}$ ）。
		- 操作時，這兩個平面的 `pitch` 和 `height` 都必須分別從 `planeParams.pitch[i]` 與 `planeParams.height[i]` 取得。

這個範例符合在 Jetson 或伺服器端開發高效能 pipeline 的硬體存取邏輯。

#### CUDA Kernel 使用 NvBufSurface

在 GPU 上利用 CUDA Kernel 直接處理 NvBufSurface，是 NVIDIA 平台上實現端到端（End-to-End）硬體加速的最強大手段。

這個流程的關鍵在於：不要使用 NvBufSurfaceMap（那是給 CPU 用的），而是要直接拿 NvBufSurface 裡面的 Device 指標（dataPtr） 或透過 EGLImage（在 Jetson 上）直接餵給 CUDA Kernel。這樣影像資料就能完全留在 GPU 記憶體中，達到零複製（Zero-Copy）的極致效能。

以下為您提供一個完全不使用 C++ 標準函式庫、純 C/C++ 與 CUDA 的實務範例。我們以常見的 RGBA 影像 為例，在 CUDA Kernel 中將影像調暗（亮度減半）。

1. CUDA Kernel 原始碼 (kernel.cu)

``` C
#include <cuda_runtime.h>
#include <stdint.h>

// CUDA Kernel：將 RGBA 影像亮度減半
__global__ void dimImageKernel(uint8_t* pDevData, int width, int height, int pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 檢查是否超出影像邊界
    if (x < width && y < height) {
        // 使用硬體回傳的 pitch 定位到正確的行 (Row)
        uint8_t* pRow = pDevData + (y * pitch);
        
        // RGBA 格式，每個像素 4 位元組
        int pixelOffset = x * 4;

        // 修改 R, G, B 通道，保持 Alpha 不變
        pRow[pixelOffset + 0] = pRow[pixelOffset + 0] / 2; // R
        pRow[pixelOffset + 1] = pRow[pixelOffset + 1] / 2; // G
        pRow[pixelOffset + 2] = pRow[pixelOffset + 2] / 2; // B
    }
}

// C 介面封裝函式，方便主程式呼叫
extern "C" void launchDimImageKernel(void* pDevData, int width, int height, int pitch, cudaStream_stream) {
    // 定義 2D Block 與 Grid 大小
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // 使用非同步 Stream 啟動 Kernel，提升 Pipeline 吞吐量
    dimImageKernel<<<gridDim, blockDim, 0, stream>>>(
        (uint8_t*)pDevData, width, height, pitch
    );
}
```

2. 主程式原始碼 (main.cpp)

``` C 
#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "nvbufsurface.h"

// 宣告外部 CUDA 函式
extern "C" void launchDimImageKernel(void* pDevData, int width, int height, int pitch, cudaStream_t stream);

#define IMG_WIDTH   1920
#define IMG_HEIGHT  1080

int main() {
    NvBufSurface *surf = NULL;
    cudaStream_t stream;
    
    // 建立 CUDA Stream 用於非同步操作
    cudaStreamCreate(&stream);

    // 1. 設定 NvBufSurface 配置參數
    NvBufSurfaceCreateParams create_params;
    create_params.gpuId = 0;
    create_params.width = IMG_WIDTH;
    create_params.height = IMG_HEIGHT;
    create_params.size = 0; 
    create_params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    
    // 記憶體類型選擇：
    // Jetson 建議使用 NVBUF_MEM_SURFACE_ARRAY 配合 EGLImage (若要寫回硬體單元) 
    // 或 NVBUF_MEM_CUDA_DEVICE / NVBUF_MEM_CUDA_UNIFIED
    #if defined(__aarch64__)
    create_params.memType = NVBUF_MEM_CUDA_DEVICE; 
    #else
    create_params.memType = NVBUF_MEM_CUDA_DEVICE; // dGPU 直接用純 GPU 記憶體
    #endif

    // 2. 配置 NvBufSurface
    if (NvBufSurfaceCreate(&surf, 1, &create_params) != 0) {
        fprintf(stderr, "Error: Failed to create NvBufSurface.\n");
        return -1;
    }

    // 3. 取得第一張影像的參數與 GPU 記憶體指標
    NvBufSurfaceParams *surface_params = &surf->surfaceList[0];
    
    // 關鍵點：核心驅動分配的硬體 Device 指標
    void* pGpuDeviceData = surface_params->dataPtr; 
    
    uint32_t width  = surface_params->planeParams.width[0];
    uint32_t height = surface_params->planeParams.height[0];
    uint32_t pitch  = surface_params->planeParams.pitch[0]; // 務必將 pitch 傳入 Kernel

    printf("Launching CUDA Kernel with Width: %d, Height: %d, Pitch: %d\n", width, height, pitch);

    // 4. 啟動 CUDA Kernel 直接在 GPU 記憶體上做運算 (Zero-Copy)
    launchDimImageKernel(pGpuDeviceData, width, height, pitch, stream);

    // 5. 同步 Stream，確保 GPU 運算完成
    cudaStreamSynchronize(stream);

    // 如果後續要交給其他硬體單元（如 NVENC 編碼器），通知硬體快取已更新
    NvBufSurfaceSyncForDevice(surf, 0, 0);

    // 6. 清理資源
    NvBufSurfaceDestroy(surf);
    cudaStreamDestroy(stream);

    printf("CUDA Kernel Processing Completed Successfully.\n");
    return 0;
}
```

💡 AI 工程師視角的架構細節（防坑必看）
1. 為什麼一定要傳遞 pitch 進入 Kernel？
不使用標準庫的底層開發對記憶體佈局要求極高。硬體為了對齊記憶體（Memory Alignment），pitch（每行的真實位元組數）通常會大於 width * 4。
在 Kernel 內定位像素時：

- 正確做法：uint8_t* pRow = pDevData + (y * pitch);

- 錯誤做法：uint8_t* pRow = pDevData + (y * width * 4);（這會導致影像隨著列數增加，畫面越來越往右偏移，最終形成嚴重的斜向撕裂）。

2. Jetson 上的特殊處理：NVBUF_MEM_SURFACE_ARRAY
在 Jetson 上將 memType 設定為 NVIDIA 預設高效能的 NVBUF_MEM_SURFACE_ARRAY（Block Linear 佈局），普通的 CUDA Kernel 是無法直接存取 dataPtr 的。

- 解決方案：必須透過 NvBufSurfaceMapEglImage() 將其轉換為 EGLImage，再透過 cudaGraphicsEGLRegisterImage 映射成 CUDA 可以讀寫的 cudaArray，最後在 Kernel 中使用 CUDA Texture (Surface Object) 來讀寫。

- 簡化做法：如果不想寫繁瑣的 EGL 映射程式碼，在配置時直接將 memType 指定為 NVBUF_MEM_CUDA_DEVICE 或 NVBUF_MEM_CUDA_UNIFIED（Pitch Linear 佈局），就能像上方範例一樣，直接拿 dataPtr 當作一般指標操作。

3. 多批次（Batch）處理的指標遞增
如果處理的是 DeepStream 傳進來的 Batch 資料（batchSize > 1），可以利用多個 CUDA Streams 平行啟動多個 Kernel，分別傳入 surf->surfaceList[i].dataPtr，即可完美吃滿 GPU 的平行運算能耐。