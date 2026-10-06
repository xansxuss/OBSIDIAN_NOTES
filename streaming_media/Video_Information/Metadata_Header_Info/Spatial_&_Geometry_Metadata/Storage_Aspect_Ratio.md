---
title: "Storage_Aspect_Ratio"
source: "https://gemini.google.com/app/6b6fe37b06009157?hl=zh-TW"
author:
published:
created: 2026-08-27
description: "Gemini conversation with 2 messages"
tags:
  - "clippings"
---
**Storage Aspect Ratio (SAR)** 泛指影像資料在儲存媒體（檔案標頭或記憶體 Buffer）中 **實際像素陣列的點陣寬高比** （Pixel Array Width-to-Height Ratio）。

SAR 與影像顯示相關的另外兩個長寬比概念（PAR、DAR）共同決定了最終的畫面呈現：

![[Pasted image 20260827163948.png]]
<span style="font-size: 12px">常見長寬比例示意圖. 資料來源：petrroudny / Getty Images</span>

**核心概念公式**

影像處理與編解碼（如 H.264 / HEVC / MPEG-4）中，三者的核心轉換公式如下：

$$
\text{DAR} = \text{SAR} \times \text{PAR}
$$

| 名稱 | 全名 | 定義說明 |
| --- | --- | --- |
| **SAR** | Storage Aspect Ratio | 檔案實際儲存的 **水平像素數與垂直像素數之比** （ $\text{Width} / \text{Height}$ ）。 |
| **PAR** | Pixel Aspect Ratio | **單一像素點本身的幾何寬高比** （方型像素為 $1:1$ ；非正方形像素如 DVD/NTSC/PAL 則不為 $1$ ）。 |
| **DAR** | Display Aspect Ratio | **最終螢幕顯示給使用者看到的物理畫面寬高比** （例如 $16:9$ 或 $4:3$ ）。 |

**常見情境範例**

1. **正方形像素（Square Pixels, $\text{PAR} = 1:1$ ）：**
	- **1080p Full HD：** 儲存解析度 $1920 \times 1080 \Rightarrow \text{SAR} = 16:9$ 。因為 $\text{PAR} = 1:1$ ，最終 $\text{DAR} = 16:9$ 。
2. **非正方形像素（Non-Square Pixels / Anamorphic）：**
	- **NTSC DVD (4:3 DAR)：** 儲存解析度 $720 \times 480 \Rightarrow \text{SAR} = 3:2$ 。設定 $\text{PAR} = 8:9$ 後，計算出 $\text{DAR} = \frac{3}{2} \times \frac{8}{9} = \frac{4}{3}$ 。
		- **NTSC DVD (16:9 Widescreen DAR)：** 儲存解析度同樣為 $720 \times 480 \Rightarrow \text{SAR} = 3:2$ 。設定 $\text{PAR} = 32:27$ 後，計算出 $\text{DAR} = \frac{3}{2} \times \frac{32}{27} = \frac{16}{9}$ 。

**開發與記憶體對齊（Stride / Pitch）注意事宜**

- **SAR vs. Stride Padding：** 在硬體加速（如 CUDA / NVDEC / VAAPI）或 Memory Allocator 操作時，為了記憶體對齊（如 64-byte alignment），圖像列長度（Stride/Pitch）可能會大於實際有效寬度。計算 SAR 時， **必須以 Effective Width/Height 為準** ，而非 Memory Pitch。
- **Container Header (Container-level vs. Stream-level)：** MP4/MKV 容器層的 Aspect Ratio 設定有時會覆蓋 H.264/HEVC VUI (Visual Usability Information) 中的 `aspect_ratio_idc` ，寫入多媒體 pipeline 時需確認解析順序。
