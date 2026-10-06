---
title: NvV4l2Element 逐行解析 — Jetson Multimedia API V4L2 M2M 基底類別
source:
author:
published:
created: 2026-09-04
description: 拆解 jetson_multimedia_api 中 NvV4l2Element 這個 V4L2 M2M 裝置抽象基底類別的每一行邏輯
categorization: embedded/Jetson
tags:
  - jetson-multimedia-api
  - V4L2
  - V4L2-M2M
  - cpp
  - video-decode
  - embedded-linux
---

## 概觀

`NvV4l2Element` 是 [[jetson_multimedia_api]] 中所有 V4L2 M2M（memory-to-memory，例如硬體編解碼器）元件的**抽象基底類別**。`NvVideoDecoder`、`NvVideoEncoder` 皆繼承自它。它本身不能被直接實例化（建構子為 `protected`），負責：

1. 開啟 V4L2 裝置節點，並驗證裝置是否具備 `V4L2_CAP_VIDEO_M2M_MPLANE` 能力。
2. 持有並初始化 [[NvV4l2ElementPlane]]（`output_plane` / `capture_plane`）。
3. 提供控制項讀寫、事件訂閱/取出、錯誤查詢等 V4L2 元件共通操作。

## 標頭檔（NvV4l2Element.h）結構

- 繼承自 [[NvElement]]，取得 `is_in_error`、`profiler` 等共用機制。
- 依賴 `NvV4l2ElementPlane.h`（plane 抽象）與 `v4l2_nv_extensions.h`（NVIDIA 專屬控制項擴充）。
- 虛擬解構子 `~NvV4l2Element()`：確保透過基底類別指標刪除子類別物件時，正確呼叫到子類別解構子。

### Public 介面（對應標準 V4L2 ioctl）

| 方法 | 對應 ioctl | 用途 |
|---|---|---|
| `subscribeEvent()` | `VIDIOC_SUBSCRIBE_EVENT` | 訂閱裝置事件（如解析度改變 `V4L2_EVENT_SOURCE_CHANGE`） |
| `dqEvent()` | `VIDIOC_DQEVENT` | 從佇列取出已發生的事件，支援逾時 |
| `setControl()` / `getControl()` | `VIDIOC_S_CTRL` / `VIDIOC_G_CTRL` | 單一整數控制項讀寫 |
| `setExtControls()` / `getExtControls()` | `VIDIOC_S_EXT_CTRLS` / `VIDIOC_G_EXT_CTRLS` | 多個或含額外資料結構的擴充控制項 |
| `isInError()` | — | 虛擬函式，回傳整體錯誤狀態（含兩個 plane） |
| `abort()` | 間接呼叫 `STREAMOFF` | 終止兩個 plane 的串流，緩衝區歸還應用程式 |
| `waitForIdle()` | — | 虛擬函式，基底類別未實作，需子類別覆寫 |
| `enableProfiling()` | — | 啟用效能剖析，須在設定 plane 格式**之前**呼叫 |

### 關鍵成員

- `NvV4l2ElementPlane output_plane` / `capture_plane`：public，分別對應輸出方向（餵資料）與擷取方向（取結果）。
- `int fd`：protected，V4L2 裝置檔案描述符，子類別可直接使用。
- `uint32_t output_plane_pixfmt / capture_plane_pixfmt`：記錄兩個 plane 目前的像素格式，作為「是否已設定格式」的旗標。
- `void *app_data`：public，不透明指標，供應用程式掛載自訂 context，類別本身不解讀內容。
- 建構子為 `protected` → 強制只能透過子類別建立，是典型抽象基底類別設計。

## 實作重點（NvV4l2Element.cpp）

### 全域 mutex

```cpp
pthread_mutex_t initializer_mutex = PTHREAD_MUTEX_INITIALIZER;
```
跨所有實例共用的全域鎖，專門保護 `v4l2_open()` 呼叫本身——因為 `libv4l2` 函式庫本身在多執行緒同時開啟裝置時存在已知的競爭條件（race condition），在應用層加鎖來規避。

### 建構子流程

1. 初始化列中，`output_plane` / `capture_plane` 建構時就把（尚未賦值的）`fd` 成員以**參照**方式傳入——依賴 C++ 成員初始化順序取決於「宣告順序」而非「初始化列順序」，`fd` 宣告在兩個 plane 之前，因此可行。之後 `fd` 被賦值時，plane 內部持有的參照也會同步看到新值。
2. 上鎖 → `v4l2_open(dev_node, flags | O_RDWR)`（強制加上 `O_RDWR`）→ 解鎖。失敗則設 `is_in_error = 1` 並提早 `return`（此時解構子仍會被呼叫做資源清理）。
3. `VIDIOC_QUERYCAP` 查詢裝置能力，失敗則設錯誤並提早結束。
4. 用位元遮罩檢查 `caps.capabilities & V4L2_CAP_VIDEO_M2M_MPLANE`，確認裝置支援 multi-planar M2M，這是硬體編解碼器的必要能力。

### 解構子

先呼叫兩個 plane 的 `deinitPlane()`（因為 plane 清理動作本身仍需要用到 `fd`），再檢查 `fd != -1` 才呼叫 `v4l2_close()`，避免對未成功開啟或已是 -1 的 fd 做關閉動作。

### dqEvent() 輪詢邏輯

一個 `do-while` 輪詢迴圈：
- 成功（`ret == 0`）→ 印除錯訊息，`while` 判斷因 `ret` 為 0（falsy）而結束。
- 失敗且非 `EAGAIN`（真正錯誤）→ `break`。
- 失敗且是 `EAGAIN`、但 `max_wait_ms--`（後置遞減，判斷用遞減前的值）已倒數到 0 → 印警告、`break`（逾時放棄）。
- 其餘情況 → `usleep(1000)` 睡 1 毫秒後重試。
- 迴圈結束條件額外加上 `output_plane.getStreamStatus() || capture_plane.getStreamStatus()`：只要任一 plane 仍在串流狀態才繼續等，兩者都已停止（如呼叫過 `abort()`）就算 `ret` 非 0 也會跳出，避免對已終止裝置無窮等待。

### setControl / getControl

`setControl()` 建立 `v4l2_control`、填入 id 與值後呼叫 `VIDIOC_S_CTRL`。`getControl()` 只填 id，呼叫 `VIDIOC_G_CTRL` 後透過參照把核心填入的值回傳給呼叫端。

### setExtControls / getExtControls

與單一控制項不同，直接使用呼叫端已組裝好的 `v4l2_ext_controls` 結構（通常內含控制項陣列），因為擴充控制項內容較複雜，組裝責任交給呼叫端。

### subscribeEvent

`memset()` 先把 `v4l2_event_subscription` 結構歸零（避免保留欄位帶有未初始化垃圾值），再填入 `type`/`id`/`flags`，呼叫 `VIDIOC_SUBSCRIBE_EVENT`。

### abort()

```cpp
ret |= output_plane.setStreamStatus(false);
ret |= capture_plane.setStreamStatus(false);
```
刻意用位元或 `|=` 而非提早 `return`，確保**兩個 plane 都會被嘗試停止**，不會因為第一個失敗就跳過第二個。

### isInError()

同樣用位元或合併三個錯誤來源（自身 `is_in_error`、`capture_plane`、`output_plane`），任一有錯即視為整體錯誤。

### enableProfiling()

利用 `output_plane_pixfmt` / `capture_plane_pixfmt` 這兩個旗標做前置條件檢查：只要任一 plane 已設定過格式，就拒絕啟用並印錯誤，藉此**強制執行**標頭檔註解所寫的「必須在設定格式前呼叫」規則。

## 待研究方向

- [[NvV4l2ElementPlane]] 的緩衝區管理與 mmap 細節（下一步逐行解析對象）
- 子類別（`NvVideoDecoder`）如何覆寫 `waitForIdle()`
- `V4L2_EVENT_SOURCE_CHANGE` 事件在解析度改變流程中的處理方式