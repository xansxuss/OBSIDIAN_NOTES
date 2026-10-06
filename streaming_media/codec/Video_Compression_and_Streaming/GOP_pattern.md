---
title: "GOP_pattern"
source: "https://gemini.google.com/app/09b655d9070c0ecd"
author:
published:
created: 2026-08-17
description: "Gemini conversation with 2 messages"
tags:
  - "clippings"
---
**You**

GOP pattern

---

**Gemini**

**GOP (Group of Pictures) Pattern** 是視訊編碼（如 H.264, H.265/HEVC, AV1）中用來定義圖像幀（Frames）排列順序與參考關係的結構模式。透過時間軸上的幀間壓縮（Temporal Compression），消除連續畫面間的冗餘資訊以降低碼率。

### 1\. 核心幀類型 (Frame Types)

在 GOP 結構中主要包含三種幀：

- **I-Frame (Intra-coded Picture / Keyframe)** ：
	- **獨立幀 / 關鍵幀** 。採用幀內壓縮，不依賴其他畫面即可完整解碼。
		- 資料量最大，作為整個 GOP 的隨機存取點（Random Access Point）與參考基準。
		CSDN博客
- **P-Frame (Predicted Picture)** ：
	- **前向預測幀** 。僅參考前方的 I 幀或 P 幀，僅記錄運動向量（Motion Vectors）與殘差（Residuals）。
- **B-Frame (Bi-directional Predicted Picture)** ：
	- **雙向預測幀** 。可同時參考前方的 I/P 幀與後方的 P/I 幀。
		- 壓縮率最高、資料量最小，但需要維護重排序緩衝區（Reordering Buffer），解碼延遲較高。
		CSDN博客

### 2\. 表示法與參數 (M & N)

GOP 的模式常用 $(M, N)$ 或 ffmpeg 參數來表示：

- **$N$ (GOP Length / Size)** ：兩個 I 幀之間的距離（即一個 GOP 包含的總幀數）。
	Wikipedia
- **$M$ (Anchor Distance)** ：兩個參考幀（I 或 P）之間的距離。例如 $M=3$ 代表每兩個 P/I 幀之間夾著 2 個 B 幀。
	Amazon.com

#### 常見 GOP Pattern 範例：

```
Display Order:  I0  B1  B2  P3  B4  B5  P6  B7  B8  P9  B10 B11 I12  (M=3, N=12)
Decode Order:   I0  P3  B1  B2  P6  B4  B5  P9  B7  B8  I12 B10 B11
```

### 3\. Open GOP vs. Closed GOP

| 特性 | Closed GOP | Open GOP |
| --- | --- | --- |
| **定義** | 內部幀 **不得參考** 其他 GOP 的畫面 | 允許 GOP 開頭的 B 幀參考上一個 GOP 的幀 |
| **關鍵幀** | 第一幀為 **IDR (Instantaneous Decoder Refresh)** | 第一幀通常為 Non-IDR I-frame |
| **壓縮效率** | 稍低 | **較高** （可省約 5%~7% 碼率） |
| **隨機切換/剪輯** | 支援任意 I 幀秒切、無失真切片 | 切片或 Seek 時前幾幀可能解碼失敗/花屏 |
| **應用場景** | HLS/DASH 串流切片、即時視訊會議、剪輯專案 | 檔案儲存、低碼率錄影、VOD 廣播 |

### 4\. 工程應用與配置策略

1. **低延遲串流（Live Streaming / WebRTC）** ：
	- **策略** ：小 GOP（如 $N=30$ 或 $1$ 秒內）、禁用或僅用 1 個 B 幀（B-pyramid off）、強制 Closed GOP。
		- **優點** ：降低 Channel-switching 延遲與 Buffer 佔用，避免雙向參考產生的解碼等待。
		Flussonic
2. **高壓縮比存取（VOD / Archiving）** ：
	- **策略** ：長 GOP（如 $N=250$ 或更長）、 $M=3 \sim 4$ （階層式 Hierarchical B-frames）、Open GOP。
		Wikipedia
		- **優點** ：極大化利用 B 幀壓縮率，壓低 Overall Bitrate。
		Flussonic
3. **場景切換偵測（Dynamic / Adaptive GOP）** ：
	- 當 Encoder 偵測到 Scene Cut 時，強制插入 IDR-frame 開啟新 GOP，避免跨場景的無效 Motion Estimation 浪費碼率。
		Wikipedia