`NvBufSurface` 幾乎是 **NVIDIA Jetson / DeepStream / VPI / TensorRT** 生態系裡的「影像記憶體統一抽象層」。  
它是你在 **GStreamer、CUDA、VPI、OpenCV CUDA、TensorRT** 之間能做到 **零拷貝 (zero-copy)** 的關鍵。

---

## 🧩 基本概念

`NvBufSurface` 是 NVIDIA 自家的 **buffer container** 結構體，定義在：

``` cpp
#include "nvbufsurface.h"
```

這個結構會封裝 GPU 端的記憶體(NVMM, a.k.a. NVIDIA Memory Manager)，  
讓所有 NVIDIA 元件都能直接共享同一份資料，而不必來回 `cudaMemcpy`。

---

## 🧠 為什麼要用 NvBufSurface？

想像這樣的 pipeline：

``` bash
nvv4l2decoder → nvvideoconvert → appsink → OpenCV CUDA → TensorRT → nveglglessink
```

如果中間不用 `NvBufSurface`：

- 每個 plugin 都會 CPU-GPU 來回 copy(超慢)
- buffer metadata 不一致(不好管理)

但如果都用 `video/x-raw(memory:NVMM)` → 都是 `NvBufSurface`：

- 所有階段都在 GPU 記憶體上直接處理
- **零 CPU 參與、零拷貝、延遲極低**

---

## 🧱 結構定義

(取自 JetPack SDK 內部 header)

```cpp
typedef struct 
{     
guint numFilled;         // 有效的 surface 數     
guint batchSize;         // 批次大小(DeepStream 常用)     
NvBufSurfaceParams surfaceList[NVBUF_MAX_PLANES]; // 每個影像的參數     
NvBufSurfaceMemType memType; // 記憶體類型(NVBUF_MEM_DEFAULT/NVBUF_MEM_CUDA_DEVICE)     
int gpuId;               // GPU ID     
cudaStream_t cudaStream; // 可選，用於 async 操作 } NvBufSurface;`
```

```
每個 `surfaceList[i]` 包含：

```cpp 
typedef struct 
{     
unsigned int width;     
unsigned int height;     
unsigned int pitch;     
void* dataPtr;           // GPU 記憶體指標     
NvBufSurfaceColorFormat colorFormat; // e.g. NVBUF_COLOR_FORMAT_NV12     
NvBufSurfaceLayout layout;           // Pitch, BlockLinear, etc.     
NvBufSurfaceMappedAddr mappedAddr;   // CPU/GPU 映射資訊 } NvBufSurfaceParams;
```
---

## 🧩 記憶體類型(`memType`)

| Enum                      | 說明                      |
| ------------------------- | ----------------------- |
| `NVBUF_MEM_DEFAULT`       | 自動選擇(通常為 NVMM)          |
| `NVBUF_MEM_CUDA_DEVICE`   | CUDA GPU device memory  |
| `NVBUF_MEM_CUDA_PINNED`   | CUDA pinned host memory |
| `NVBUF_MEM_SURFACE_ARRAY` | Jetson 硬體影像 buffer      |
| `NVBUF_MEM_SYSTEM`        | 普通 CPU 記憶體              |

在 Jetson 上通常是：

> `video/x-raw(memory:NVMM)` → `NVBUF_MEM_SURFACE_ARRAY`

---

## ⚙️ 常見 API

|函式|功能|
|---|---|
|`NvBufSurfaceCreate()`|建立 GPU buffer|
|`NvBufSurfaceDestroy()`|釋放|
|`NvBufSurfaceMap()`|將 surface map 到 CPU 或 CUDA 空間|
|`NvBufSurfaceSyncForCpu()` / `NvBufSurfaceSyncForDevice()`|同步資料|
|`NvBufSurfaceFromFd()`|由 dma-buf fd 建立 surface(常見於 GStreamer)|
|`NvBufSurfaceMemSet()`|初始化記憶體|
|`NvBufSurfaceCopy()`|GPU-to-GPU 拷貝|
|`NvBufSurfaceMapEglImage()`|將 buffer 映射為 EGLImage(顯示用)|

---

## 🔄 與 GStreamer 整合

在 DeepStream 或一般 Jetson GStreamer 管線中，  
`NvBufSurface` 是每個 `video/x-raw(memory:NVMM)` buffer 的 **實體內容**。

例如：

``` bash
GstBuffer *buffer = gst_app_sink_pull_sample(appsink); NvBufSurface *surface = (NvBufSurface *) gst_buffer_get_nvds_surface(buffer);
```

或者自己手動從 `GstMemory` 提取：

``` bash
NvBufSurface *surface = (NvBufSurface *) gst_buffer_get_memory(buffer, 0);
```


---

## ⚡ OpenCV CUDA 包裝範例

以下示範如何將 `NvBufSurface` 直接包成 `cv::cuda::GpuMat`：

``` cpp
cv::cuda::GpuMat WrapNvBufSurface(NvBufSurface *surface) 
{     
if (!surface || surface->numFilled == 0)         
throw std::runtime_error("Invalid NvBufSurface");      
auto &src = surface->surfaceList[0];     
return cv::cuda::GpuMat(src.height, src.width, CV_8UC1, src.dataPtr, src.pitch); }
```

若格式為 NV12，要手動拆成 Y / UV plane：

``` cpp 
cv::cuda::GpuMat y(src.height, src.width, CV_8UC1, src.dataPtr, src.pitch); 
cv::cuda::GpuMat uv(src.height/2, src.width/2, CV_8UC2,                     
(uchar*)src.dataPtr + src.pitch * src.height, src.pitch);
```

---

## 🧩 與 CUDA 結合範例

``` cpp
NvBufSurfaceMap(surface, 0, 0, NVBUF_MAP_READ); NvBufSurfaceSyncForCpu(surface, 0, 0);  // 獲得 CUDA 指標 
void* devPtr = surface->surfaceList[0].dataPtr;  // 在 CUDA kernel 裡直接使用 
processKernel<<<grid, block>>>((uint8_t*)devPtr, width, height);  
NvBufSurfaceSyncForDevice(surface, 0, 0); 
NvBufSurfaceUnMap(surface, 0, 0);
```

---

## 🔬 與 EGL 顯示結合

若你要丟給 `nveglglessink`：

``` cpp
EGLImageKHR egl_image; 
NvBufSurfaceMapEglImage(surface, 0); 
egl_image = surface->surfaceList[0].mappedAddr.eglImage; 
eglDestroyImageKHR(egl_display, egl_image); 
NvBufSurfaceUnMapEglImage(surface, 0);
```

---

## ✅ 總結

|特性|說明|
|---|---|
|🌐 通用介面|可跨 GStreamer / CUDA / TensorRT / VPI|
|⚡ 高效能|支援 NVMM、DMA-BUF、零拷貝|
|🧱 統一格式|每個 buffer 都包含 metadata(color format, pitch, layout)|
|🧩 可 batch 化|`batchSize` 支援 DeepStream 批次推論|
|🧠 可同步 CPU/GPU|有 sync API 避免 race condition|
