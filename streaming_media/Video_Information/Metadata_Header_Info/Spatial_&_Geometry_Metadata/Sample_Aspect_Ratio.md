---
title: "Sample_Aspect_Ratio"
source: "https://gemini.google.com/app/467f2892adf622a7?hl=zh-TW"
author:
published:
created: 2026-08-27
description: "Gemini conversation with 2 messages"
tags:
  - "clippings"
---
**Sample Aspect Ratio (SAR)** ，中文常稱為 **樣本長寬比** 或 **像素比（Pixel Aspect Ratio, PAR）** ，是指數位影像中 **單一像素（Pixel）本身的寬度與高度比例** 。

在數位視訊編碼（如 H.264、H.265 / HEVC）與多媒體容器（如 MP4、MKV）中，SAR 是決定畫面如何正確拉伸與顯示的關鍵參數之一。

**核心概念與相關比例**

理解 SAR 時，通常會與 **DAR** 及 **PAR** 一起比較：

| 縮寫                   | 全名                   | 定義與說明                                                                        |
| -------------------- | -------------------- | ---------------------------------------------------------------------------- |
| **SAR**              | Sample Aspect Ratio  | **單一像素的寬高比** ($Width_{pixel} : Height_{pixel}$)。若像素為正方形，SAR 為 `1:1` 。        |
| **FAR / Storage AR** | Frame Aspect Ratio   | **儲存解析度的寬高比例** 。計算方式為 $寬解析度 : 高解析度$ （例如 $1920 \times 1080$ 的 FAR 為 `16:9` ）。 |
| **DAR**              | Display Aspect Ratio | **最終顯示在螢幕上的畫面長寬比** （例如螢幕顯示為 `16:9` 或 `4:3` ）。                                |

數學關係式如下：

$$
\text{DAR} = \text{FAR} \times \text{SAR} = \left(\frac{\text{Frame Width}}{\text{Frame Height}}\right) \times \left(\frac{\text{SAR}_{w}}{\text{SAR}_{h}}\right)
$$

**常見的 SAR 應用場景**

- **Square Pixels（正方形像素, SAR = 1:1）**
	- 現代高清視訊（1080p、4K 等）大多採用 `SAR 1:1` 。此時畫面解析度比例（FAR）即為最終顯示比例（DAR）。
- **Non-Square Pixels（長方形像素 / 變形像素）**
	- **傳統標清電視（SD / NTSC / PAL）：** 例如 NTSC DVD 解析度為 $720 \times 480$ （FAR 為 $3:2$ ）。若要顯示為 $16:9$ 的 DAR，其 SAR 會設定為 $32:27$ 。
		- **Anamorphic（變形鏡頭 / 擠壓儲存）：** 為了節省頻寬或記憶體，將 $16:9$ 的畫面橫向壓縮儲存於 $4:3$ 的畫格中，解碼播放時再透過設定 `SAR > 1` （例如 $4:3$ ）將畫面拉伸回正確的 $16:9$ 顯示。

**FFmpeg 中的語法與設定**

在 FFmpeg 處理影片時，常使用 `sample_aspect_ratio` 來調整 SAR：

- **檢視影片 SAR 指令：**
	```
	ffprobe -v error -show_entries stream=sample_aspect_ratio,display_aspect_ratio input.mp4
	```
- **修改影片 SAR 濾鏡：**
	```
	# 將 SAR 強制設為 1:1 (Square Pixels)
	ffmpeg -i input.mp4 -vf "setsar=1/1" -c:a copy output.mp4
	```