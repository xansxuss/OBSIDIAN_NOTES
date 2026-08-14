``` C
/**
 * Specifies memory types for \ref NvBufSurface.
 */
typedef enum
{
  /** Specifies the default memory type, i.e. \ref NVBUF_MEM_CUDA_DEVICE
   for dGPU, \ref NVBUF_MEM_SURFACE_ARRAY for Jetson. Use \ref NVBUF_MEM_DEFAULT
   to allocate whichever type of memory is appropriate for the platform. */
  NVBUF_MEM_DEFAULT,
  /** Specifies CUDA Host memory type. */
  NVBUF_MEM_CUDA_PINNED,
  /** Specifies CUDA Device memory type. */
  NVBUF_MEM_CUDA_DEVICE,
  /** Specifies CUDA Unified memory type. */
  NVBUF_MEM_CUDA_UNIFIED,
  /** Specifies NVRM Surface Array type. Valid only for Jetson. */
  NVBUF_MEM_SURFACE_ARRAY,
  /** Specifies NVRM Handle type. Valid only for Jetson. */
  NVBUF_MEM_HANDLE,
  /** Specifies memory allocated by malloc(). */
  NVBUF_MEM_SYSTEM,
} NvBufSurfaceMemType;
```
這段程式碼是 NVIDIA DeepStream SDK 或 Jetson Linux (L4T) 多媒體架構中非常核心的 NvBufSurface 記憶體類型定義。
在 C 語言環境下定義了一個名為 NvBufSurfaceMemType 的型別。這在 Jetson 平台或 x86 dGPU 上開發高效能 AI 推論管線（Pipeline）時，如何管理硬體加速的緩衝區（Buffer）至關重要。
這幾種記憶體配置類型的底層硬體與系統行為：

記憶體類型詳細拆解
1. NVBUF_MEM_DEFAULT
- 行為： 自動選擇該硬體平台上最適合、效能最好的預設類型。

- 平台差異：
	- 在 x86 + 獨立顯示卡 (dGPU) 平台上：等同於 NVBUF_MEM_CUDA_DEVICE。
	- 在 Jetson 平台上：等同於 NVBUF_MEM_SURFACE_ARRAY。

- 建議： 如果您的程式需要跨平台（同時支援 Jetson 與 dGPU），用這個可以維持架構的彈性。

2. NVBUF_MEM_CUDA_PINNED
- 行為： 配置主機端（CPU）的頁面鎖定記憶體（Page-Locked / Pinned Memory）。
- 特點： 這種記憶體不會被作業系統交換（Swap）到硬碟。因為物理地址固定，當 CPU 要透過 PCIe 傳送資料到 GPU 時，可以啟動 DMA（直接記憶體存取），達到極高的傳輸頻寬。

3. NVBUF_MEM_CUDA_DEVICE
- 行為： 配置獨立顯示卡上的顯示記憶體（VRAM）。
- 特點： 只有 GPU 核心可以直接存取。CPU 無法直接讀寫它，必須透過 cudaMemcpy 等 API 進行搬移。在獨立顯卡平台上，AI 模型推論與影像解碼的資料通常都放在這裡以追求極致效能。

4. NVBUF_MEM_CUDA_UNIFIED
- 行為： 配置 CUDA 統一記憶體（Unified Memory / Managed Memory）。
- 特點： 系統會建立一個虛擬指標，CPU 和 GPU 都可以直接存取同一個記憶體位址。底層由驅動程式（透過 Page Fault 機制）在背景自動幫您同步、搬移資料。雖然開發上最直覺方便，但在 dGPU 上可能會有額外的驅動程式調度開銷。

5. NVBUF_MEM_SURFACE_ARRAY 與 NVBUF_MEM_HANDLE (Jetson 專用)
- 行為： 這兩者是 Jetson（硬體整合晶片 SoC）專屬 的特殊底層記憶體。
- 核心觀念：零複製 (Zero-Copy)
	- Jetson 採用統一記憶體架構（UMA），CPU 和 GPU 本來就共享同一塊物理記憶體。
	- NVBUF_MEM_SURFACE_ARRAY 是透過 NVIDIA 核心驅動程式（NVRM）配置的核心層緩衝區，專門用來對接硬體加速單元（例如 NVDEC 影片解碼器、VIC 影像縮放器、NVENC 編碼器以及 GPU）。
	- 這能確保影像在解碼、前處理（預處理）、硬體縮放、投餵給 AI 模型（TensorRT）時，完全不需要在記憶體中複製資料，只傳遞硬體控制把手（Handle）。

6. NVBUF_MEM_SYSTEM
- 行為： 單純透過標準 C 的 malloc() 在 CPU 端的系統記憶體（Heap）配置空間。
- 特點： 這是最傳統的記憶體。GPU 或者是 Jetson 的硬體加速引擎無法直接對其進行加速處理。通常只在需要將資料拉回純 CPU 處理邏輯（例如寫入一般檔案、使用純 CPU 的第三方函式庫）時才會用到。

#### 在撰寫 C/C++ 程式碼並呼叫 `NvBufSurfaceCreate` 配置影像緩衝區時：

- **在寫 Jetson 專案：** 務必優先選用 `NVBUF_MEM_DEFAULT` 或 `NVBUF_MEM_SURFACE_ARRAY`。這樣才能把 Jetson 的硬體編解碼與 NvMedia 效能榨到極致，避免 CPU 介入搬運資料造成樽頸（Bottleneck）。
- **資料跨硬體元件：** 當您需要把 OpenCV 處理完的 `cv::Mat`（通常是 `SYSTEM` 或 `PINNED` 記憶體）丟進 DeepStream 管線，或是反過來要把硬體影像拉出來時，就需要特別注意這幾種型別之間的轉換與快取同步（Cache sync）問題。