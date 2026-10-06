---
title: "Gradual_Decoder_Refresh"
source: "https://gemini.google.com/app/f3153dd12be9cf11?hl=zh-TW"
author:
published:
created: 2026-08-27
description: "Gemini conversation with 2 messages"
tags:
  - "clippings"
---
Gradual Decoder Refresh** (GDR，亦稱為 Intra Refresh) 是一種在視訊編碼（如 H.264/AVC、H.265/HEVC 及 H.266/VVC）中用於替代傳統 I 幀 (IDR) 的錯誤復原與隨機存取技術。

傳統視訊編碼會定期插入全新的 I 幀（Keyframe）來清理參考幀緩衝區，但這會產生極高的峰值碼率 (Bitrate Spike)。GDR 則將原本集中在單一 I 幀中的 Intra 宏區塊 (MB) 或編碼單元 (CU) 拆分，分散到連續的數個 P 幀或 B 幀中（例如以垂直或水平條狀的方式逐步刷新畫面）。

### 核心優點

- **消弭碼率峰值 (Bitrate Smoothing)** ：避免傳統 IDR 幀引起的暴增資料量，顯著穩定網路傳輸頻寬與 Latency。
- **極低延遲 (Ultra-Low Latency)** ：特別適合 RTSP 實時串流、視訊會議、雲端遊戲與無人機影像傳輸。
	docs.amd.com
- **快速錯誤復原** ：即使發生封包丟失 (Packet Loss)，解碼器也能在數個幀內逐步修復畫面失真，無需重新發送大型 IDR 幀。
	docs.amd.com

### 運作機制與限制

```
幀 N   : [ Intra 區域 ][   Inter 預測區域 (未刷新)   ]
幀 N+1 : [ 已刷新 ][ Intra 區域 ][   未刷新區域   ]
幀 N+2 : [      已刷新      ][ Intra 區域 ][未刷新]
```

1. **恢復點 (Recovery Point SEI)** ：編碼器會在 GDR 週期開頭插入 SEI 訊息，告訴解碼器「畫面將在多少幀後完全復原」。
2. **向量限制 (Motion Vector Constraints)** ：為避免未刷新的失真像素被參考運動向量 (MV) 污染到已刷新的區域（Leakage），編碼器會在 Intra Refresh 過程對運動向量與環路濾波 (Loop Filters) 施加運動邊界限制。

### 常見應用場景與配置範例

GDR 在各主流硬體編解碼器（如 NVIDIA NVENC、AMD VCU、Intel QuickSync 等）中皆有支援。

**以 GStreamer + Hardware H.264 Encoder 為例** ：

``` bash
gst-launch-1.0 filesrc location=input.yuv ! rawvideoparse width=1920 height=1080 format=nv12 framerate=30/1 \
  ! omxh264enc control-rate=constant gop-mode=low-delay-p gdr-mode=horizontal periodicity-idr=30 \
  ! video/x-h264, profile=high ! filesink location=output.h264
```

FFMPEG
1. ### **NVIDIA NVENC (H.264 / HEVC 硬體加速)**
	NVENC 對 GDR (Intra Refresh) 支援度最好且效能高，是 Low-Latency 串流的首選。
``` bash
# H264
ffmpeg -i input.mp4 -c:v h264_nvenc \
  -g 30 \
  -intra-refresh 1 \
  -forced-idr 0 \
  -bf 0 \
  -rc cbr -b:v 4M \
  -y output_gdr.264
# H265
ffmpeg -i input.mp4 -c:v hevc_nvenc \
  -g 30 \
  -intra-refresh 1 \
  -forced-idr 0 \
  -bf 0 \
  -rc cbr -b:v 4M \
  -y output_gdr.265
```
- **`-intra-refresh 1`**：開啟 NVENC 的 Intra Refresh (GDR) 機制。
    
- **`-g 30`**：指定刷新週期（Refresh Period），代表以 30 幀為一個完整的 GDR 輪替週期。
    
- **`-forced-idr 0`**：強制禁止編碼器在週期點插入 IDR 幀。
    
- **`-bf 0`**：關閉 B 幀（B 幀的雙向預測會破壞 Intra Refresh 的單向刷新約束，建議設為 0）。

2. 
	x264 (CPU 軟體編碼)
	libx264 透過 -intra-refresh 參數開啟 GDR。x264 預設以 Vertical Wave/Refresh Block 的方式向右掃描刷新。
	x265 (CPU HEVC 軟體編碼)
	libx265 需透過 x265-params 設定 intra-refresh。

``` bash
# H264
ffmpeg -i input.mp4 -c:v libx264 \
  -g 30 \
  -keyint_min 30 \
  -sc_threshold 0 \
  -bf 0 \
  -x264-params "intra-refresh=1:no-scenecut=1" \
  -y output_gdr.h264
# H265
ffmpeg -i input.mp4 -c:v libx265 \
  -g 30 \
  -keyint_min 30 \
  -bf 0 \
  -x265-params "intra-refresh=1:no-scenecut=1:keyint=30" \
  -y output_gdr.hevc
```
- -x264-params "intra-refresh=1"：告知 x264 啟用 Intra Refresh 並在週期開頭寫入 Recovery Point SEI。
- -sc_threshold 0 / no-scenecut=1：徹底禁用 Scene Cut (場景切換) 觸發的自動 IDR 幀插入。
- -keyint_min 30：強制最小 GOP 長度與 -g 一致，防止發送 IDR。

驗證 GDR 是否生效
重編碼完成後，可以透過 ffprobe 檢查碼流中的 Frame Type。如果設定成功，除了第一幀（Sequence Header 必須的 IDR/SPS/PPS 之外），後續將不會出現任何 IDR 幀，全數由 P 幀與內部 Intra Block 組成：

``` bash
ffprobe -select_streams v:0 -show_frames -show_entries frame=pict_type,key_frame,pkt_size output_gdr.264
```
成功特徵：第一幀為 pict_type=I, key_frame=1，之後所有幀皆為 pict_type=P, key_frame=0，且封包大小 (pkt_size) 極度均勻，沒有傳統 IDR 幀暴增突發 (Bitrate Spike) 的現象。

