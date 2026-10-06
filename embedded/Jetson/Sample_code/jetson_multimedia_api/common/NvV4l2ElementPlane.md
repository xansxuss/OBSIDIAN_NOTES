---
title: NvV4l2ElementPlane 原始碼解析
source:
author:
published:
created: 2026-09-04
description: Jetson Multimedia API 中 NvV4l2ElementPlane 類別的架構與核心方法解析，封裝 V4L2 output/capture plane 的 ioctl 操作
categorization: embedded/Jetson
tags:
  - Jetson
  - V4L2
  - multimedia-api
  - NvBuffer
  - video-decode
  - multithreading
---

## 概觀

`NvV4l2ElementPlane` 是 [[jetson-video-decode]] 範例中 `jetson_multimedia_api` 的核心 helper class，把 V4L2 M2M 裝置（例如硬體解碼器）其中一個 plane（Output 或 Capture）包裝成物件，避免每次手動組 `v4l2_buffer` + 呼叫 `ioctl()`。

一個 `NvV4l2Element`（如 decoder）內部各持有一個 output plane 與一個 capture plane 實體，兩者都是 `NvV4l2ElementPlane`。

### V4L2 buffer 生命週期

```
REQBUFS → QUERYBUF / EXPBUF → mmap() 或分配記憶體
   → QBUF → STREAMON → (迴圈) DQBUF ⇄ QBUF → STREAMOFF → REQBUFS(count=0)
```

類別裡幾乎每個 method 都對應上面某一步驟。

---

## 類別設計重點

- Constructor / Destructor 為 `private`，僅 `friend class NvV4l2Element` 可建立與銷毀 → 保證這個物件只能透過工廠角色的 `NvV4l2Element` 管理生命週期。
- Copy constructor / `operator=` 宣告但不實作 → 禁止複製（物件內部管理 `NvBuffer*` 陣列、pthread mutex/cond、裝置 fd，複製會導致資源衝突）。
- `int &fd`、`NvElementProfiler &v4l2elem_profiler` 用 **reference** 成員，因為 fd 生命週期由外層 `NvV4l2Element` 管理，plane 只是借用。
- `plane_lock` / `plane_cond` 是 `public` 的同步機制，保護所有跟 buffer queue 狀態相關的成員，也讓外層 `NvV4l2Element` 能做跨 plane 同步。

### 支援的三種記憶體型態（`enum v4l2_memory`）

| 型態 | 特性 |
|---|---|
| `V4L2_MEMORY_MMAP` | kernel 配置，userspace mmap 存取 |
| `V4L2_MEMORY_USERPTR` | userspace 自行配置記憶體，指標交給 kernel |
| `V4L2_MEMORY_DMABUF` | 用 DMA-BUF fd 共享記憶體，Jetson 上最常見（可跨 GPU/VIC/ISP 零拷貝） |

---

## 核心方法對照表

| 方法 | 對應 ioctl / 用途 |
|---|---|
| `getFormat` / `setFormat` | `VIDIOC_G_FMT` / `VIDIOC_S_FMT`，設定後同步回填 `n_planes`、`planefmts[]`（stride、sizeimage） |
| `getCrop` / `setSelection` | `VIDIOC_G_CROP` / `VIDIOC_S_SELECTION`（**唯一**需要把 `_MPLANE` type 轉成非 `_MPLANE` 的函式，V4L2 規格差異） |
| `reqbufs(mem_type, num)` | `VIDIOC_REQBUFS`；`num=0` 時走「釋放」路徑（delete 所有 `NvBuffer` 並清空陣列），身兼配置/釋放兩種功能 |
| `queryBuffer` | `VIDIOC_QUERYBUF`（僅 MMAP），取得 offset/length 供 `mmap()` 用 |
| `exportBuffer` | `VIDIOC_EXPBUF`，逐一 plane 匯出成 DMA-BUF fd |
| `setupPlane(mem_type, num, map, allocate)` | 整合 `reqbufs`+`queryBuffer`+`exportBuffer`+`map`/`allocate`，任一步失敗 `goto error` 呼叫 `deinitPlane()` 完整清理 |
| `deinitPlane()` | `setupPlane` 的反向：STREAMOFF → 等 DQ Thread 結束 → 依記憶體型態解除 map/釋放記憶體 → `reqbufs(mem_type, 0)` |
| `mapOutputBuffers` / `unmapOutputBuffers` | 僅 DMABUF 用，透過 `NvBufSurfaceFromFd` + `NvBufSurfaceMap/UnMap` 映射/解除映射 CPU 可存取位址 |
| `setStreamStatus(bool)` | `VIDIOC_STREAMON`/`STREAMOFF`，有冪等檢查；STREAMOFF 會把 `num_queued_buffers` 強制歸零 |
| `qBuffer` / `dqBuffer` | `VIDIOC_QBUF` / `VIDIOC_DQBUF`，依 `memory_type` 分支填 `userptr`/`fd`/`bytesused`；`dqBuffer` 用 `do...while` 重試，`EAGAIN` + `V4L2_BUF_FLAG_LAST` 代表 EOS |
| `waitAllBuffersQueued` / `waitAllBuffersDequeued` / `waitForDQThread` | 用 `pthread_cond_timedwait` + 絕對時間（`timeval`→`timespec` 換算）實作逾時等待，皆用 `while` 迴圈防範 spurious wakeup |
| `startDQThread` / `stopDQThread` / `dqThread` | 背景執行緒自動輪詢 `dqBuffer`，成功則呼叫使用者 `callback`；`stopDQThread` 在 **blocking 模式下無效**（ioctl 會卡住，`stop_dqthread` 旗標沒機會被檢查） |

---

## Profiler 埋點時機

- `qBuffer`：只有 **output plane** 呼叫 `v4l2elem_profiler.startProcessing()`（送進硬體開始計時）。
- `dqBuffer`：只有 **capture plane** 呼叫 `v4l2elem_profiler.finishProcessing()`（拿到結果結束計時）。
- 一組 output QBUF → capture DQBUF 才構成一次完整處理耗時。

---

## 容易踩坑 / 設計取捨

- `-1` 傳給 `uint32_t` 型別的等待時間參數（如 `waitForDQThread(-1)`、`dqBuffer(..., -1)`）會被隱式轉成 `0xFFFFFFFF`，效果等同「幾乎無限等待/重試」。
- DMA-BUF 的底層記憶體**不由這個類別擁有**：`deinitPlane`／`unmapOutputBuffers` 在 DMABUF 分支刻意不釋放記憶體，只解除映射或不做事，生命週期歸建立方管理。
- `getStreamStatus()` 讀取 `streamon` 沒有上鎖，屬於「粗略、非強一致性」查詢的刻意取捨。
- `startDQThread` 沒檢查 `pthread_create` 回傳值，理論上建立失敗時 `dqthread_running` 仍會被設為 `true`，狀態與實際不符。
- `reqbufs`/`setFormat` 內的 log 字串有寫死 "at output plane"/"at capture plane" 的小瑕疵（output/capture 共用同一份程式碼卻沒依實際 plane 客製訊息）。

---

## 延伸閱讀

- 對照 [[NvVideoDecoder]] 中 `NvVideoDecoder` 如何驅動 output/capture 兩個 `NvV4l2ElementPlane` 實體，走完整解碼流程。
- 可延伸研究 `NvBuffer` 類別（`buffers[]` 陣列元素型別）與 `NvBufSurface` API（DMA-BUF 映射細節）。
