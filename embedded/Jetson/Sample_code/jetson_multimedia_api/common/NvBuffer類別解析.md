---
title: NvBuffer 類別解析（Jetson Multimedia API）
source: 
author: 
published: 
created: 2026-09-07
description: NvBuffer 是 Jetson Multimedia API 中管理 V4L2 buffer/plane 記憶體的核心類別，涵蓋 map/unmap、allocate/deallocate、參考計數與像素格式換算。
categorization: embedded
tags:
  - jetson
  - v4l2
  - multimedia-api
  - cpp
  - buffer-management
---

## 概述

`NvBuffer` 模仿 V4L2 的 `v4l2_buffer` 結構，用來描述一塊緩衝區，支援：

- 最多 [[MAX_PLANES]]（3）個 plane，對應 YUV 等多平面像素格式
- MMAP 記憶體對應（`map()` / `unmap()`）
- USERPTR 記憶體配置（`allocateMemory()` / `deallocateMemory()`）
- 執行緒安全的參考計數（`ref()` / `unref()`）

此類別與 [[NvV4l2ElementPlane]]、[[NvV4l2Element]] 同屬 jetson_multimedia_api 的核心元件，是 [[jetson-video-decode]] 研究專案的基礎資料結構。

## 巢狀型別

### NvBufferPlaneFormat（格式描述）

| 成員 | 功用 |
|---|---|
| `width` / `height` | plane 像素寬高 |
| `bytesperpixel` | 每像素位元組數 |
| `stride` | 每列實際佔用位元組數（可能因對齊而大於理論值） |
| `sizeimage` | 整個 plane 總位元組數 |

### NvBufferPlane（實際資料）

在 `NvBufferPlaneFormat` 基礎上，額外包含 `data`（指標）、`bytesused`、`fd`（MMAP/DMA 檔案描述子）、`mem_offset`、`length`。

格式資訊（不變規格）與執行期狀態（會變動的資料）分離，是常見的職責分離設計。

## 三種建構子

| 建構子 | 使用情境 | buf_type / memory_type |
|---|---|---|
| `NvBuffer(buf_type, memory_type, n_planes, fmt, index)` | 通用版，完全自訂 | 呼叫端指定 |
| `NvBuffer(pixfmt, width, height, index)` | Raw 像素格式（YUV/RGB），由 `fill_buffer_plane_format()` 自動推算 plane 配置 | 固定為 `V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE` + `V4L2_MEMORY_USERPTR` |
| `NvBuffer(size, index)` | 非 raw 格式（如 H.264/H.265 bitstream），單一 plane | 同上 |

`const` 成員（`buf_type`、`memory_type`、`index`）必須透過**初始化列**賦值，不能在建構子主體內用 `=` 賦值。

第二個建構子在呼叫 `fill_buffer_plane_format()` 後，額外手動補算：

```cpp
sizeimage = width * height * bytesperpixel;
stride    = width * bytesperpixel;
```

此計算**未考慮記憶體對齊**，是理論最小值。

## 解構子

```cpp
~NvBuffer()
{
    if (mapped) unmap();
    if (allocated) deallocateMemory();
    pthread_mutex_destroy(&ref_lock);
}
```

依旗標自動清理資源，符合 RAII 精神，但**不會自動釋放透過 `allocateMemory()` 配置的記憶體以外的外部資源**（例如若 `fd` 是從外部開啟，需呼叫端自行處理）。

## map() / unmap()（僅適用 MMAP）

`map()` 核心呼叫：

```cpp
mmap(NULL, planes[j].length, PROT_READ | PROT_WRITE,
     MAP_SHARED, planes[j].fd, planes[j].mem_offset);
```

- `MAP_SHARED`：對應的是核心驅動配置的實體記憶體，非複製。
- 失敗判斷用 `MAP_FAILED`（非 `NULL`），這是 `mmap` API 特有慣例。

**已知瑕疵：** 迴圈中若某個 plane 對應失敗即直接 `return -1`，但先前已成功 mmap 的 plane 未被 rollback（unmap），存在資源洩漏風險。

`unmap()` 邏輯對稱，逐一 `munmap()` 後清空指標。

## allocateMemory() / deallocateMemory()（僅適用 USERPTR）

```cpp
planes[j].length = MAX(sizeimage, width * bytesperpixel * height);
planes[j].data = new unsigned char [planes[j].length];
```

取兩種計算方式的較大值，避免配置不足。

**已知瑕疵：** `new` 失敗時是丟出 `std::bad_alloc` 例外，而非回傳 `MAP_FAILED`；程式碼中 `if (planes[j].data == MAP_FAILED)` 這行**永遠不會成立**，是原始碼複製貼上 `map()` 邏輯時未修正的殘留錯誤。若要走 C 風格重寫，建議改用 `malloc()` + `NULL` 檢查，或 `new(std::nothrow)` + `NULL` 檢查。

`deallocateMemory()` 對稱使用 `delete[]`（因配置時是 `new[]`，型別須對應，否則為未定義行為）。

## ref() / unref()：執行緒安全參考計數

```cpp
int ref()
{
    pthread_mutex_lock(&ref_lock);
    ref_count = ++this->ref_count;
    pthread_mutex_unlock(&ref_lock);
    return ref_count;
}
```

- `lock → 修改 → 讀出到區域變數 → unlock → 回傳`，避免多執行緒 race condition。
- `unref()` 有下限保護（`ref_count > 0` 才遞減），因 `ref_count` 是 `uint32_t`，無保護會在 0 時 underflow 成極大正數。
- 區域變數 `ref_count` 與成員 `this->ref_count` 同名（shadowing），可讀性上建議改名。

## fill_buffer_plane_format()：像素格式對照表

靜態函式，依 `raw_pixfmt` 換算所需 plane 數與各 plane 的 width/height/bytesperpixel：

| pixfmt | plane 數 | 說明 |
|---|---|---|
| YUV420M / YVU420M | 3 | 4:2:0，U/V 寬高各減半 |
| YUV422M | 3 | 4:2:2，U/V 寬減半 |
| YUV422RM | 3 | 4:2:2 變形，U/V 高減半 |
| NV12M | 2 | Y + 交錯 UV（bytesperpixel=2） |
| GREY | 1 | 純灰階 |
| YUYV/YVYU/UYVY/VYUY | 1 | packed 格式 |
| ABGR32/XRGB32 | 1 | 32-bit 全彩 |
| P010M | 2 | 10-bit HDR |
| NV24M/NV24_10LE | 2 | 4:4:4 全取樣 |

**已知瑕疵：** 呼叫端（第二個建構子）未檢查此函式的回傳值，遇不支援格式時 `n_planes`/`fmt` 可能處於未定義狀態仍繼續執行。

## 整體歸納與重寫方向

| 職責 | 對應函式 |
|---|---|
| 建構初始化 | 三個建構子多載 |
| 資源自動釋放 | 解構子（RAII） |
| MMAP 對應/解除 | map() / unmap() |
| USERPTR 配置/釋放 | allocateMemory() / deallocateMemory() |
| 執行緒安全計數 | ref() / unref() |
| 像素格式換算 | fill_buffer_plane_format()（static） |

若後續要改寫成不使用標準函式庫的版本，優先處理三項瑕疵：
1. `map()` 失敗時未 rollback 已成功的 plane
2. `allocateMemory()` 對 `new` 失敗的錯誤檢查邏輯錯誤
3. `fill_buffer_plane_format()` 回傳值未被檢查
