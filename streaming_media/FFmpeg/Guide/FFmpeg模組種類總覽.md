---
title: FFmpeg 模組種類總覽
source: 
author: 
published: 
created: 2026-09-23
description: FFmpeg 七個核心函式庫的功能、資料結構與資料流說明。
categorization: streaming_media/FFmpeg
tags:
  - FFmpeg
  - 影音處理
  - 函式庫
  - libavformat
  - libavcodec
  - libavfilter
---

## 概述

FFmpeg 由 7 個核心函式庫組成，各自負責不同階段的媒體處理任務。整體資料流為：

```
輸入來源
   ↓
libavformat  (Demux) → AVPacket
   ↓
libavcodec   (Decode) → AVFrame
   ↓
libswscale / libswresample  (格式轉換)
   ↓
libavfilter  (濾鏡處理)
   ↓
libavcodec   (Encode) → AVPacket
   ↓
libavformat  (Mux) → 輸出檔案
```

---

## 核心函式庫

### libavformat
- 負責**容器格式**的封裝與解封裝
- Demux：將 `.mp4`、`.mkv`、`.ts` 等容器拆解為各串流
- Mux：將編碼資料重新打包成容器輸出
- 主要資料結構：`AVFormatContext`、`AVStream`、`AVPacket`

### libavcodec
- 負責**編碼與解碼**，是 FFmpeg 最核心的函式庫
- 解碼：`AVPacket`（壓縮）→ `AVFrame`（原始）
- 編碼：`AVFrame` → `AVPacket`
- 支援 H.264、H.265、VP9、AV1、AAC、MP3 等數百種 codec
- 整合硬體加速：NVDEC、VAAPI、VideoToolbox 等
- 相關筆記：[[codec]]、[[NALU]]

### libavfilter
- 負責**濾鏡圖（Filter Graph）**，以 DAG 串接多個濾鏡節點
- 常用濾鏡：`scale`、`overlay`、`drawtext`、`yadif`（去交錯）
- 輸入／輸出節點：`buffersrc` / `buffersink`

### libswscale
- 負責**影像縮放與像素格式轉換**
- 例：YUV420P → RGB24、1920×1080 → 1280×720
- 通常在 AVCodec 解碼後使用，將 `AVFrame` 格式轉換為顯示所需格式

### libswresample
- 負責**音訊重採樣與格式轉換**
- 取樣率轉換：44100 Hz → 48000 Hz
- 聲道布局：stereo → mono
- 樣本格式：`AV_SAMPLE_FMT_FLTP` → `AV_SAMPLE_FMT_S16`

### libavdevice
- 負責**裝置輸入輸出**，為 libavformat 的延伸
- 輸入裝置：`v4l2`（Linux）、`avfoundation`（macOS）、`dshow`（Windows）
- 輸出裝置：SDL 顯示視窗、OpenGL
- 需呼叫 `avdevice_register_all()` 啟用

### libavutil
- **共用工具函式庫**，所有其他函式庫皆依賴此庫
- 記憶體管理：`av_malloc`、`av_free`
- 時間戳記換算：`av_rescale_q`
- 資料結構：`AVDictionary`、`AVRational`、`AVBufferRef`
- Log 系統：`av_log`

---

## 相關筆記
- [[demuxer-module]] — 通用 demuxer 模組實作（FFmpeg API）
- [[codec]] — 編解碼格式說明
- [[low_latency_pipeline]] — 低延遲串流架構
- [[V4L2]] — Linux 攝影機裝置介面