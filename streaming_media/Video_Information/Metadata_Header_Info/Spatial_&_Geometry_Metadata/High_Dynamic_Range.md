---
title: High_Dynamic_Range
source: https://gemini.google.com/app/f9248e2292432633?hl=zh-TW
author:
published:
created: 2026-08-27
description: Gemini conversation with 4 messages
tags:
  - clippings
---
**HDR（High Dynamic Range，高動態範圍）** 是一種用於影像、影片與顯示技術中的處理標準，旨在擴展畫面中最亮與最暗區域之間的明暗比（動態範圍），使影像更接近人眼實境所見。

![[Pasted image 20260827165227.png]]
<span style="font-size: 12px">HDR 多曝光合成與色調對映原理. 資料來源：PetaPixel</span>

**核心應用領域與原理**

- **攝影與影像處理 (Photography)** ：透過曝光包圍（Exposure Bracketing）拍攝多張不同曝光度（欠曝、正常、過曝）的照片，經由演算法合成並透過色調對映（Tone Mapping），同時保留高光細節與暗部層次。
- **顯示器與電視技術 (Display Standards)** ：提升螢幕的峰值亮度（Peak Brightness, nits）與對比度，並支援更廣的色域（如 DCI-P3、Rec. 2020）。
	- **HDR10** ：開放標準，採用靜態元資料（Static Metadata）。
		- **HDR10+** ：支援動態元資料，依據逐幀（Frame-by-frame）調整亮度曲線。
		- **Dolby Vision** ：杜比影院級標準，支援高達 12-bit 色深與動態元資料動態對映。
		- **HLG (Hybrid Log-Gamma)** ：主要用於電視直播廣播，向下相容 SDR 螢幕。
- **計算攝影與電腦視覺 (Computational Photography & Vision)** ：在智慧型手機與鏡頭影像訊號處理器（ISP）中，利用單張 Sensor 多曝光讀出（如 DOL-HDR、Staggered HDR）即時合成 HDR 影像，消除動態鬼影（Ghosting Artifacts）。