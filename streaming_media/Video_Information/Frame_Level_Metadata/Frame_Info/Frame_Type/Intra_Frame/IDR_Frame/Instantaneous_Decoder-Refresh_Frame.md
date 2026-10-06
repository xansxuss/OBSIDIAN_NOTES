---
title: Instantaneous_Decoder-Refresh_Frame
source: https://gemini.google.com/app/b7ec11eef82ab455?hl=zh-TW
author:
published:
created: 2026-08-27
description: Gemini conversation with 2 messages
tags:
  - clippings
---
**IDR Frame（Instantaneous Decoder Refresh Frame，即時解碼器刷新幀）** 是 H.264/AVC 與 H.265/HEVC 等視訊編碼標準中的一種特殊 **I 幀（Intra Frame）** 。

其主要作用在於劃分獨立的解碼單元，強制解碼器清空參考幀緩衝區（Decoded Picture Buffer, DPB），防止錯誤持續擴散。
![[Pasted image 20260827171015.png]]
<span style="font-size: 12px">GOP 結構與 I 幀位置示意. 資料來源：ResearchGate</span>

**核心特性與運作機制**

- **DPB 刷新機制** ：當解碼器收到 IDR Frame 時，會立即清空 DPB 中所有先前的參考幀（Reference Frames），並重新初始化序列參數（SPS/PPS）。
- **嚴格的參考障壁（Reference Barrier）** ：IDR Frame 之後的所有 P 幀與 B 幀， **絕對不能** 跨越 IDR 幀去參考 IDR 之前的任何畫面。
- **解碼與播放切入點** ：因為具備完全獨立性，IDR 幀是視訊串流進行 **Seek（隨機跳轉）** 、 **切換頻道/串流** 或 **快進/快退** 的安全接入點。

**IDR Frame 與 一般 I Frame (Non-IDR I Frame) 的差異**

| 特性 | IDR Frame (IDR 幀) | Ordinary I-Frame (一般 I 幀) |
| --- | --- | --- |
| **內建圖像類型** | 自身全部為 Intra 內聯編碼 | 自身全部為 Intra 內聯編碼 |
| **DPB 清空** | **強制清空** DPB | **不清空** DPB |
| **跨障壁參考權限** | 後續 P/B 幀 **嚴禁參考** IDR 前的幀 | 後續 P/B 幀 **允許參考** 該 I 幀之前的幀 |
| **獨立解碼能力** | 完全獨立，可作為強勢 Seek Point | 不一定能獨立解碼（若後續 B/P 幀參考了舊幀） |
| **NAL Unit Type (H.264)** | `NALU Type 5` | `NALU Type 1` (Slice) |

**工程應用場合**

1. **實時串流與拖曳（Streaming & Seeking）** ：串流媒體（HLS / RTSP / WebRTC）會定期插入 IDR 幀（例如每 1~2 秒一個 GOP），確保播放器隨機 seek 時能立即渲染出正確畫面，不出現花屏。
2. **抗錯與阻斷遺失（Error Resiliency）** ：當網路封包遺失導致解碼器解碼失敗時，解碼器可拋棄當前 GOP，直接等待下一個 IDR 幀抵達即可恢復正常解碼。
3. **動態場景切換（Scene Change）** ：視訊切換畫面時，編碼器會主動插入 IDR 幀以切斷與舊場景的參考關係，節省編碼流量並提高畫質。