---
title: NvVideoDecoder 原始碼解析（V4L2 硬體解碼器封裝類別）
source:
author:
published:
created: 2026-09-04
description: 解析 Jetson Multimedia API 中 NvVideoDecoder 如何封裝 V4L2 M2M 介面操作 NVDEC 硬體解碼器
categorization: embedded
tags:
  - jetson
  - v4l2
  - nvdec
  - video-decode
  - embedded-linux
  - multimedia-api
---

## 背景概念

`NvVideoDecoder` 是 [[jetson-video-decode]] 研究中的核心類別，繼承自 `NvV4l2Element`（可對照 [[NvV4l2ElementPlane]] 的 plane wrapper），負責封裝對 Jetson 硬體解碼器（NVDEC）的 V4L2 M2M 操作。

- **V4L2 M2M（Memory-to-Memory）裝置**：有兩個 plane。
  - **Output plane（輸出平面）**：對驅動而言是「輸出」，但實際上是使用者**輸入**壓縮編碼資料（H.264、H.265…）的地方。
  - **Capture plane（擷取平面）**：解碼完成後，驅動把原始 YUV 影像放在這裡讓使用者**擷取**。
- 狀態機順序限制：**設定格式 → 要求 buffer → streamon**，順序不能顛倒，這也是程式碼中多組防呆巨集存在的原因。

## 類別設計重點（NvVideoDecoder.h）

### 工廠方法而非公開建構子
```cpp
static NvVideoDecoder *createVideoDecoder(const char *name, int flags = 0);
```
建構子放在 `private`，外部只能透過這個靜態工廠方法建立物件；失敗時回傳 `NULL`，屬於 C 風格的錯誤處理慣例（不丟例外）。

### 格式設定 API
- `setCapturePlaneFormat(pixfmt, width, height)`：設定解碼輸出的原始像素格式與解析度（呼叫 `VIDIOC_S_FMT`）。
- `setOutputPlaneFormat(pixfmt, sizeimage)`：設定輸入的編碼格式與單一 buffer 最大容量。

### 解碼行為控制（皆透過 `VIDIOC_S_EXT_CTRLS` 設定 V4L2 control）
- `setFrameInputMode()` / `disableCompleteFrameInputBuffer()`：允許輸入 buffer 不含完整一個 frame。
- `setSliceMode()`：啟用 HEVC slice-level 解碼。
- `disableDPB()`：關閉 Decoded Picture Buffer，用於 low-latency 場景。
- `getMinimumCapturePlaneBuffers()`：查詢擷取平面最少需要幾個 buffer（需在 resolution change event 後呼叫才有效）。
- `setSkipFrames()`：設定跳幀策略。
- `setMaxPerfMode()`：切換最大效能模式。
- `enableMetadataReporting()`：啟用輸出中繼資料回報。

### Metadata／HDR 查詢
- `checkifMasteringDisplayDataPresent()`、`MasteringDisplayData()`：HDR mastering display 資訊。
- `getMetadata()` / `getInputMetadata()`：依 `buffer_index` 查詢特定畫面的解碼／輸入中繼資料。
- `getSAR()`：取得 Sample Aspect Ratio（像素長寬比校正用）。

### Polling 中斷機制
- `DevicePoll()`：對驅動下 poll 動作。
- `SetPollInterrupt()` / `ClearPollInterrupt()`：讓應用程式能從外部中斷正在 blocking 的 poll，常用於程式結束時避免卡死。

### AV1 專屬
- `setAV1OperatingPoint()` / `getAV1NumOperatingPoints()`：AV1 的 operating point（類似 scalable coding 的分層）選擇與查詢。
- `enableAV1MVC()`：啟用多視角編碼（3D 影像）支援。
- `enableGDRStream()`：支援 GDR（Gradual Decoder Refresh，不依賴完整 I-frame 的漸進式刷新）串流。

### private 區段
```cpp
static const NvElementProfiler::ProfilerField valid_fields =
        NvElementProfiler::PROFILER_FIELD_TOTAL_UNITS |
        NvElementProfiler::PROFILER_FIELD_FPS;
```
用位元或組合旗標，告知內建 profiler 要統計「總處理單位數」與「FPS」。

## 實作重點（NvVideoDecoder.cpp）

### 巨集：統一 ioctl 回傳處理
```cpp
#define CHECK_V4L2_RETURN(ret, str)              \
    if (ret < 0) {                               \
        COMP_SYS_ERROR_MSG(str << ": failed");   \
        return -1;                               \
    } else {                                     \
        COMP_DEBUG_MSG(str << ": success");      \
        return 0;                                \
    }
```
因為巨集內直接 `return`，只能放在函式最後一行；許多方法本體只是組 struct，最後交給這個巨集發出 ioctl 並統一處理回傳與 log。

### 三組防呆巨集（對應狀態機規則）
- `RETURN_ERROR_IF_FORMATS_SET()`：確保尚未設定過 output plane 格式。
- `RETURN_ERROR_IF_BUFFERS_REQUESTED()`：確保兩個 plane 都尚未要求配置 buffer。
- `RETURN_ERROR_IF_FORMATS_NOT_SET()`：確保格式已經設定過。

### createVideoDecoder：裝置節點相容性處理
```cpp
if (access(DECODER_DEV, F_OK) == 0)
    dec = new NvVideoDecoder(name, DECODER_DEV, flags);
else if (access(DECODER_DEV_ALT, F_OK) == 0)
    dec = new NvVideoDecoder(name, DECODER_DEV_ALT, flags);
else
    return NULL;

if (dec->isInError())
{
    delete dec;
    return NULL;
}
```
先嘗試 `/dev/nvhost-nvdec`，找不到再嘗試新版路徑 `/dev/v4l2-nvdec`（不同 Jetson Linux 版本裝置節點命名不同）；建構後檢查 `isInError()`，有錯就刪除物件避免回傳半殘的實例。

### setCapturePlaneFormat：格式白名單與 plane 計算
```cpp
if (! ((pixfmt == V4L2_PIX_FMT_NV12M) || (pixfmt == V4L2_PIX_FMT_P010M) || ...))
{
    COMP_ERROR_MSG("Only NV12M, P010M, YUV420M, YUV422M, NV24M and NV24_10LE is supported");
    return -1;
}
```
只允許特定幾種 multi-planar 像素格式（`M` 後綴代表分開的記憶體平面）。

```cpp
NvBuffer::fill_buffer_plane_format(&num_bufferplanes, planefmts, width, height, pixfmt);
capture_plane.setBufferPlaneFormat(num_bufferplanes, planefmts);
```
依 `width`、`height`、`pixfmt` 自動算出需要幾個 plane、各自的 stride 與大小，再存入 `capture_plane` 物件供之後配置實體記憶體使用。

最後填入 `v4l2_format` 的 `pix_mp`（pixel format multi-planar）欄位，交給 `capture_plane.setFormat(format)` 發出真正的 `VIDIOC_S_FMT`。

### setOutputPlaneFormat：編碼格式檢查
以 `switch` 檢查是否為支援的編碼格式（H.264 / H.265 / VP8 / VP9 / AV1 / MPEG2 / MPEG4 / MJPEG），符合才記錄到 `output_plane_pixfmt` 並設定 buffer 大小 `sizeimage`。

### 其餘 control 類方法的共同寫法模式
`disableDPB()`、`setSliceMode()`、`setMaxPerfMode()` 等方法幾乎都遵循同一套模式：
1. 用防呆巨集檢查目前狀態是否允許呼叫。
2. `memset` 歸零 `v4l2_ext_control` 與 `v4l2_ext_controls` 結構，避免帶入垃圾值。
3. 填入 `control.id`（對應的 `V4L2_CID_xxx`）與 `control.value`（或 `control.string` 指向更複雜的資料結構）。
4. 呼叫 `setExtControls(ctrls)` 或 `getExtControls(ctrls)`（父類別方法），最後交給 `CHECK_V4L2_RETURN` 統一處理回傳。

## 待釐清 / 下一步

- `NvV4l2Element::setExtControls()` / `getExtControls()` 的實作細節尚未拆解。
- `NvBuffer::fill_buffer_plane_format()` 如何依像素格式計算 plane 數量與 stride，值得單獨深入。
