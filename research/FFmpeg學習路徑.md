---
title: "FFmpeg學習路徑"
source: "https://claude.ai/chat/cee27659-217b-42cb-8ead-6a0dec33e5c2"
author:
published:
created: 2026-07-03
description: "Claude conversation with 4 messages"
tags:
  - "clippings"
---
學習 FFmpeg 是一個滿有系統性的過程，建議可以照下面的順序安排，由淺入深：

### 第一階段：命令列工具入門（1-2 週）

先熟悉 `ffmpeg` 、 `ffprobe` 、 `ffplay` 這三個命令列工具，不用碰程式碼：

- **基礎轉檔** ：容器格式轉換（mp4、mkv、avi 等）
- **編碼參數** ：認識 codec（H.264、H.265、VP9、AV1）、bitrate、CRF、preset
- **濾鏡（filter）** ：scale、crop、overlay、fps 等常用濾鏡語法
- **音訊處理** ：取樣率、聲道、音訊編碼（AAC、Opus）
- **ffprobe 分析** ：讀懂媒體檔案的 metadata、串流資訊

這階段的重點是建立「容器（container）」、「編碼（codec）」、「串流（stream）」這幾個核心概念的正確認知，之後看原始碼才不會混亂。

### 第二階段：架構概念（1 週）

理解 FFmpeg 內部的資料流動：

- **libavformat** ：負責解封裝（demux）／封裝（mux）
- **libavcodec** ：負責編碼／解碼
- **libavfilter** ：濾鏡圖（filter graph）
- **libavutil** ：共用工具函式
- **libswscale / libswresample** ：影像縮放與音訊重取樣

建議畫一張資料流程圖：檔案 → demux → packet → decode → frame → filter → encode → packet → mux → 檔案，把這個 pipeline 記熟。

### 第三階段：API 實作（依你的語言偏好，主力放 C）

因為你熟 C/C++，直接用 C 呼叫 libav\* 系列函式庫會比 Python binding 更能理解底層：

1. 寫一個最簡單的「開檔 → 印出串流資訊」的小程式（ `avformat_open_input` 、 `avformat_find_stream_info` ）
2. 進階到「解封裝＋解碼單一影格存成 PPM/BMP」
3. 再做「讀取→解碼→編碼→寫出」的轉檔小工具
4. 最後加上濾鏡（ `avfilter_graph` 系列 API）

這部分官方有 `doc/examples/` 目錄下的範例程式（如 `decode_video.c` 、 `transcoding.c` ）非常值得逐行研讀。

### 第四階段：整合應用

- 若要用 Python，可以搭配 `PyAV` 這種對 libav 做 binding 的套件，或單純用 `subprocess` 呼叫命令列工具，看你的專案需求
- 若目標是效能或嵌入式應用，就往硬體加速（VAAPI、NVENC、videotoolbox）方向深入

再來往下挖，看 FFmpeg 底層真正在做什麼事情。

### 核心資料結構的生命週期

FFmpeg 最重要的兩個資料容器是 `AVPacket` （壓縮資料）跟 `AVFrame` （未壓縮資料）。這兩者都用 **reference counting（參照計數）** 管理記憶體，底層是 `AVBufferRef` ：

- 每個 `AVPacket` / `AVFrame` 內部的資料緩衝區不是直接複製，而是透過 `av_buffer_ref()` 增加參照計數
- 當你呼叫 `av_packet_unref()` 或 `av_frame_unref()` ，只是把計數減一，真正歸零才會 `av_free()`
- 這樣設計是為了避免大量記憶體複製，尤其影像 frame 動輒好幾 MB

理解這個機制很重要，因為很多新手寫 C 程式時會忘記 unref，造成記憶體洩漏，或是誤以為淺拷貝（shallow copy）後的兩個 struct 互不影響（其實底層 buffer 是共用的）。

### Demux 到 Decode 的狀態機

`av_read_frame()` 底層其實在做的事情：

1. `AVFormatContext` 內部有個 `AVIOContext` ，負責跟實體檔案/網路串流做 I/O，本身有自己的緩衝區（預設 4KB 或自訂大小）
2. demuxer（例如 `mov` 、 `matroska` ）解析容器格式，把資料切成一個個 `AVPacket` ，並附上 `pts` 、 `dts` 、 `duration` ，這些時間戳記單位是該串流的 `time_base` （不是絕對秒數，要自己換算）
3. 丟進 decoder 前，packet 要先進 `avcodec_send_packet()` ，內部維護一個內部緩衝佇列，decoder 可能不會馬上吐出對應的 frame（因為 B-frame 需要重排序）
4. `avcodec_receive_frame()` 才拿到真正解碼完的 `AVFrame`

這個「send/receive 非同步」介面是刻意設計成這樣，因為一個 packet 可能對應零個、一個、或多個 frame（例如音訊一個 packet 可能包含多個 frame）。

### Filter Graph 的協商機制

`libavfilter` 內部是一個有向圖，節點之間協商（negotiate）格式時會做：

- 每個 filter pad 宣告自己支援的 pixel format / sample format 清單
- graph config 階段（ `avfilter_graph_config()` ）會跑一次類似 constraint solving 的過程，找出整條路徑都相容的格式
- 如果格式不相容，中間會自動插入 `auto-inserted filter` （像 `scale` 、 `format` 、 `aresample` ），這也是為什麼有時候你沒加的 filter 卻出現在 graph 裡

### 時間戳記與同步的底層邏輯

這是最多人搞混的地方：

- 每個 stream 有自己的 `time_base` （例如 1/90000），PTS 是「幾個 time\_base 單位」而不是秒數
- 換算成秒： `pts_seconds = pts * time_base.num / time_base.den`
- mux 輸出時要用 `av_packet_rescale_ts()` 把 packet 的 time\_base 從 decoder 的轉成 muxer 目標串流的 time\_base，這步驟漏掉是新手最常見的同步錯誤來源
- DTS（解碼順序）跟 PTS（顯示順序）不同，是因為 B-frame 需要參考未來的 frame 才能解碼，所以解碼順序跟顯示順序會不一致

### 多執行緒模型

`libavcodec` 的多執行緒藏在 `AVCodecContext.thread_count` 跟 `thread_type` 裡：

- `FF_THREAD_FRAME` ：不同 frame 平行解碼（有 latency，因為要 buffer 好幾張才能平行跑）
- `FF_THREAD_SLICE` ：單一 frame 內部切片平行解碼（latency 低）
- 這個機制對外是透明的，你呼叫 API 的方式完全一樣，但了解它才能解釋「為什麼加執行緒後 delay 變長」這種現象

---
