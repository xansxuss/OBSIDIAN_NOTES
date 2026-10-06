---
title: Display_Aspect_Ratio
source: https://gemini.google.com/app/a548db3ea6c6b5ec?hl=zh-TW
author:
published:
created: 2026-08-27
description:
tags:
---
**DAR** 是指 **Display Aspect Ratio（顯示長寬比）** 。

長寬比定義了影像顯示時的寬度與高度比例（例如 16:9 或 4:3）。在數位影像與影片處理中，DAR 與另外兩個常見的 Aspect Ratio 概念密切相關：

**核心概念比較**

- **DAR (Display Aspect Ratio，顯示長寬比)** ：影像最終在螢幕上呈現給使用者看時的物理寬高比例。常見如 $16:9$ 、 $4:3$ 。
- **SAR (Storage Aspect Ratio，儲存長寬比 / 畫素長寬比)** ：影像實際儲存的像素解析度比例（橫向像素數 / 縱向像素數）。例如 $1920 \times 1080$ 的 SAR 為 $16:9$ ； $720 \times 480$ 的 SAR 為 $3:2$ 。
- **PAR (Pixel Aspect Ratio，單一像素長寬比)** ：單一像素點本身的物理寬高比。標準電腦螢幕通常是正方形像素（PAR = $1:1$ ），但早期電視或特定影片格式（如 NTSC/PAL DVD）會使用長方形像素（PAR $\neq 1:1$ ）。

**三者數學關係**

$$
DAR = SAR \times PAR
$$
 
$$
\text{即：}\text{顯示長寬比} = \left(\frac{\text{影像寬度像素}}{\text{影像高度像素}}\right) \times \text{單一像素長寬比}
$$

**實務應用場景**

1. **影片解碼與渲染 (FFmpeg / OpenCV)** 在處理 NTSC DVD 影片時，原始儲存解析度可能為 $720 \times 480$ （SAR = $3:2$ ）。若 PAR 為 $8/9$ ，則計算出來的 DAR 為：
	$$
	\frac{720}{480} \times \frac{8}{9} = \frac{4}{3}
	$$
	播放器在渲染時會自動將影像拉伸為 $4:3$ 顯示，否則畫面中的人事物會看起來被壓扁或拉長。
2. **影像變形修復 (Anamorphic Processing)** 使用變形鏡頭拍攝或進行特定壓縮儲存時，影像會先經由非對稱縮放儲存。影像處理 Pipeline 必須讀取 Metadata 中的 PAR 與 DAR 資訊，才能還原為正確的比例進行物件檢測（Object Detection）或視覺分析。