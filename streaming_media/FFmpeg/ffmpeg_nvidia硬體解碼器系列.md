---
title: "ffmpeg nvidia 硬體解碼器系列"
source: "https://gemini.google.com/app/e169ff60bd508dd5"
author:
published:
created: 2026-07-03
description: "Gemini conversation with 2 messages"
tags:
  - "clippings"
---
在 FFmpeg 中使用 NVIDIA 的硬體解碼技術，會因為 **獨立顯示卡 (dGPU)** 與 **嵌入式平台 (Jetson 系列)** 的底層架構不同，而使用完全不同的解碼器系列。

以下為您梳理這兩大陣營在 FFmpeg 中的硬體解碼器名稱、底層技術與使用範例：

## 一、 獨立顯示卡系列 (dGPU: GeForce, Quadro, RTX, Tesla)

在 PC、伺服器或工作站的 NVIDIA 顯卡上，FFmpeg 主要使用 **NVDEC** (NVIDIA Video Decoder) 技術。

### 1\. 核心解碼器名稱 (-c:v)

在 FFmpeg 中，對應不同編碼格式的硬體解碼器名稱如下：

- **H.264 / AVC:** `h264_cuvid`
- **H.265 / HEVC:** `hevc_cuvid`
- **VP9:** `vp9_cuvid`
- **AV1:** `av1_cuvid` (僅限較新的 GPU，如 RTX 30/40/50 系列)

### 2\. 兩種硬體加速宣告方式

在 dGPU 上使用硬體解碼，FFmpeg 提供兩種宣告路徑：

- **方式 A：使用 CUVID 解碼器 (舊式但直接)** 直接指定解碼器名稱。這會將解碼工作完全交給 NVDEC 晶片。
	```
	ffmpeg -c:v h264_cuvid -i input.mp4 -c:v h264_nvenc output.mp4
	```
- **方式 B：使用 `-hwaccel cuda` (新式，推薦)** 讓 FFmpeg 自動選擇最佳的硬體解碼器（底層仍是 NVDEC），好處是通用性高，且方便進行 GPU 記憶體內（零拷貝）的畫面縮放或像素格式轉換。
	```
	ffmpeg -hwaccel cuda -i input.mp4 -c:v h264_nvenc output.mp4
	```

## 二、 Jetson 嵌入式系列 (Jetson Nano, Xavier, Orin)

Jetson 系列的 SoC 雖然也包含 NVIDIA GPU，但其多媒體編解碼晶片（NVMedia/V4L2 介面）與 PC 端的 NVDEC 架構不同。 **原生官方版本的 FFmpeg 並不直接支援 Jetson 的硬體解碼器** 。

在 Jetson 上，通常需要透過開源社群維護的 **`jetson-ffmpeg`** （基於 **NVMPI** 或 **L4T Multimedia API** ）來編譯客製化的 FFmpeg。

### 1\. 核心解碼器名稱 (-c:v)

編譯好支援 Jetson 硬體加速的 FFmpeg 後，解碼器名稱會變成：

- **H.264 / AVC:** `h264_nvmpi`
- **H.265 / HEVC:** `hevc_nvmpi`

### 2\. 使用範例

在 Jetson 平台上，從攝影機或影片檔讀取並進行硬體解碼：

```
# 使用 nvmpi 硬體解碼 H.264 影片
ffmpeg -c:v h264_nvmpi -i input.mp4 -f null -
```

> **注意 (Orin 系列的新趨勢):** 在較新的 Jetson Orin 平台與較新版的 Jetpack 中，NVIDIA 官方更推薦使用 **DeepStream SDK** 或 **GStreamer (nvv4l2decoder)** 來處理視訊串流，因為這能達到最低的延遲與最佳的零拷貝 (Zero-copy) 效能。若堅持在 Jetson 使用 FFmpeg，務必確認使用的是有整合 NVMPI 的 patch 版本。

## 三、 dGPU 與 Jetson 差異對照表

| 特性 | 獨立顯示卡 (dGPU) | Jetson 系列 (嵌入式) |
| --- | --- | --- |
| **底層硬體技術** | NVDEC (CUDA 生態系) | NVMedia / L4T Multimedia API (V4L2) |
| **FFmpeg 官方支援** | 原生直接支援，不需額外打補丁 | 需編譯 `jetson-ffmpeg` (NVMPI) |
| **H.264 解碼器名稱** | `h264_cuvid` / `-hwaccel cuda` | `h264_nvmpi` |
| **H.265 解碼器名稱** | `hevc_cuvid` / `-hwaccel cuda` | `hevc_nvmpi` |
| **記憶體架構** | 主機記憶體與 GPU 記憶體獨立 (需 Host ↔ Device 傳輸) | 統一記憶體架構 (Unified Memory, 減少傳輸開銷) |
