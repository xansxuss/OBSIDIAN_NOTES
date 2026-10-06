---
title: NvElementProfiler 原始碼解析
source:
author:
published:
created: 2026-08-31
description: 拆解 Jetson Multimedia API 的 NvElementProfiler 類別，說明其延遲/FPS/遲到統計實作邏輯
categorization: embedded/jetson_multimedia_api
tags:
  - jetson
  - jetson-multimedia-api
  - profiling
  - cpp
  - pthread
  - multithreading
---

## 概述

`NvElementProfiler` 是 NVIDIA Jetson Multimedia API 框架中，供所有元件（decoder、encoder 等）共用的**效能量測輔助類別**，用來統計：

- 處理延遲（latency）：平均 / 最小 / 最大
- 平均處理速率（FPS）
- 遲到單元數（late units）
- 總處理單元數

此類別採用 **friend class 設計模式**，不開放外部直接建構，只能被 [[NvElement]] 內部持有並管理生命週期，屬於封裝良好的內部基礎設施。此檔案與正在研究的 [[jetson-video-decode]] 範例密切相關：`NvVideoDecoder` 等元件皆繼承自 `NvElement`，因此自動具備 profiling 能力，是效能調校（tuning）的重要資訊來源。

## 資料結構設計

### ProfilerField（bitmask 旗標）

```cpp
typedef int ProfilerField;
static const ProfilerField PROFILER_FIELD_NONE = 0;
static const ProfilerField PROFILER_FIELD_TOTAL_UNITS = 1;
static const ProfilerField PROFILER_FIELD_LATE_UNITS = 2;
static const ProfilerField PROFILER_FIELD_LATENCIES = 4;
static const ProfilerField PROFILER_FIELD_FPS = 8;
static const ProfilerField PROFILER_FIELD_ALL = (PROFILER_FIELD_FPS << 1) - 1;
```

- 每個欄位各佔一個位元，用 `|` 組合、`&` 檢查是否啟用。
- `PROFILER_FIELD_ALL` 用「最大旗標左移一位再減一」的技巧取得全部位元為 1 的遮罩，新增欄位時不必手動修改。

### NvElementProfilerData（對外結構）

| 欄位 | 說明 |
|---|---|
| `valid_fields` | 此元件支援哪些統計欄位 |
| `average/min/max_latency_usec` | 延遲統計（微秒） |
| `total_processed_units` | 累計處理總數 |
| `num_late_units` | 遲到單元數 |
| `average_fps` | 平均處理速率 |
| `profiling_time` | 總統計經過時間 |

### NvElementProfilerDataInternal（內部結構）

繼承自 `NvElementProfilerData`，額外增加：

- `start_time` / `stop_time`：第一次 / 最近一次完成處理的牆上時鐘時間
- `accumulated_time`：暫停（disable）期間之前累積的時間，讓重新啟用後可無縫接續統計
- `total_latency`：所有已處理單元的延遲總和，供計算平均延遲

## 核心 API

- `startProcessing()`：登記一個單元開始處理，回傳唯一 ID（遞增計數器產生，`id=0` 保留為特殊值）
- `finishProcessing(id, is_late)`：登記單元完成，依 ID（或 `id=0` 取佇列最早的一筆）查表算出延遲並累計
- `getProfilerData(data)`：即時計算平均延遲、平均 FPS 等統計結果並填入輸出結構
- `printProfilerData(out_stream)`：依 `valid_fields` 逐項格式化印出
- `enableProfiling(reset_data)` / `disableProfiling()`：啟用／暫停統計，暫停時把經過時間存入 `accumulated_time`

## 關鍵實作邏輯

### 配對式延遲計算

以 `std::map<uint64_t, struct timeval> unit_start_time_queue` 記錄「ID → 開始時間」：

- `startProcessing()` 插入一筆記錄，並用 `end()` 當插入提示位置，因為 ID 嚴格遞增、必落在最尾端，可將插入複雜度最佳化到接近 O(1)。
- `finishProcessing()` 依 ID 查表取出開始時間、算出延遲後即從 map 中移除該筆。

### 平均 FPS 為何用 (N-1)

`start_time` 記錄的是**第一個單元完成處理的時刻**，而非 profiler 啟用的瞬間。因此經過時間 `total_time` 只涵蓋「第 2 個到最後一個單元」之間的間隔，共 `(N-1)` 段，所以：

```cpp
average_fps = (total_processed_units - 1) * 1000000 / total_time;
```

### 可暫停續計（accumulated_time）

`disableProfiling()` 把當前這輪的 `stop_time - start_time` 累加進 `accumulated_time`，並將 `start_time`/`stop_time` 歸零；下次 `enableProfiling(false)`（不重置）時可從累積時間繼續，不遺漏、不重算。

### timeval 進位/借位正規化

`getProfilerData()` 計算 `profiling_time` 時，`tv_usec` 分開相加減可能產生負數或超過 1,000,000，需手動正規化：

```cpp
if (data.profiling_time.tv_usec < 0) {
    data.profiling_time.tv_usec += 1000000;
    data.profiling_time.tv_sec--;
}
if (data.profiling_time.tv_usec > 1000000) {
    data.profiling_time.tv_usec -= 1000000;
    data.profiling_time.tv_sec++;
}
```

### 執行緒安全

所有讀寫共享狀態的函式皆以單一 `pthread_mutex_t profiler_lock` 保護，並用巨集簡化：

```cpp
#define LOCK() pthread_mutex_lock(&profiler_lock)
#define UNLOCK() pthread_mutex_unlock(&profiler_lock)
#define RETURN_IF_DISABLED() \
    if (!enabled) { \
        UNLOCK(); \
        return; \
    }
```

屬於「單一鎖保護所有內部狀態」的簡單設計，因 profiler 呼叫頻率遠低於熱路徑（hot path），效能影響可接受。

### 禁止複製（non-copyable）

```cpp
NvElementProfiler(const NvElementProfiler& that);
void operator=(NvElementProfiler const&);
```

宣告於 private 但不實作，是 C++11 之前常見的禁止複製寫法（現代寫法為 `= delete`）。原因是類別內含 `pthread_mutex_t`，複製鎖會造成未定義行為。

### reset() 與 memset 的安全性

```cpp
memset(&data_int, 0, sizeof(data_int));
data_int.min_latency_usec = (uint64_t) -1;
```

- 用 `memset` 整塊歸零，因 `data_int` 全由基本型別（`uint64_t`、`float`、`timeval`）組成的 POD 結構，無指標或需建構管理的物件，可安全使用。
- `min_latency_usec` 設為 `(uint64_t)-1`（型別最大值），確保第一次比較 `latency < min_latency_usec` 必定成立，正確初始化最小值。