---
title: "自製NVIDIA硬體解編碼器"
source: "https://claude.ai/chat/e5821638-76cc-45ce-a5f9-f36266a05478"
author:
published: 2026-08-11
created: 2026-08-12
description: "Claude conversation with 7 messages"
tags:
  - "clippings"
---
### 先搞懂硬體架構

NVENC(編碼)跟NVDEC(解碼)是GPU晶片上 **獨立的ASIC區塊**,不佔用CUDA核心運算資源。它們各自由一顆基於 **Falcon微處理器** (NVIDIA自研的小型微控制器架構)的韌體驅動,透過MMIO暫存器與PCIe(dGPU)或內部匯流排(Tegra iGPU)跟主系統溝通。從Turing世代開始,大部分控制邏輯又被搬進了\*\*GSP(GPU System Processor)\*\*這個獨立協同處理器裡,韌體是加密簽章過的二進位檔,無法直接反組譯出明碼邏輯。

這代表:NVENC/NVDEC本身的 **韌體** 是黑盒子,但 **驅動如何跟這顆黑盒子溝通** (暫存器位址、命令佇列格式)是可以逆向的,nouveau就是這樣做的。

### 路線一:應用層 — 包自己的解編碼器framework

**dGPU(桌上型/筆電獨立顯示卡)**

- 底層用 NVIDIA Video Codec SDK(NVENC/NVDEC API,C介面),或者更省事直接吃 FFmpeg 的 `h264_nvenc` / `hevc_nvdec` 這類 hwaccel 後端
- 你在上面用 C++ 包一層自己的 session 管理、frame queue、zero-copy(CUDA/EGL interop 避免 GPU↔CPU 來回搬資料)
- 因為你偏好不用標準函式庫,這部分要自己寫記憶體池、環形緩衝區、執行緒同步原語,NVIDIA SDK本身是C ABI,跟STL沒有耦合,相容性沒問題

**iGPU(這裡指Tegra/Jetson系列的內建GPU,NVIDIA在x86平台沒有做iGPU)**

- 用 Jetson Linux(L4T)的多媒體API:`NvVideoEncoder` / `NvVideoDecoder`,底層是標準 V4L2 memory-to-memory(M2M)介面,跟桌上型的NVENC SDK是不同的一套API
- 需要處理 `libargus` (相機管線)或 DMA-BUF 做零拷貝

這條路線技術上一兩個月內可以做出堪用的自製編解碼器wrapper,是相對務實的起點。

### 路線二:驅動層 — 逆向NVDEC/NVENC介面(nouveau方向)

**現況**:nouveau對NVDEC已有部分支援,做法是從NVIDIA官方驅動裡 **萃取韌體blob** 直接載入(韌體本身沒辦法逆向,只能借用),真正逆向的是 **命令佇列格式、暫存器映射、初始化序列** 這些跟韌體溝通的協定。NVENC支援目前很零星,幾乎沒人做完整。

**門檻**:

- 要懂 Falcon ISA 跟 GSP 韌體載入流程
- 主要工具是 `envytools` (暫存器/命令格式的逆向文件與工具集)
- NVIDIA 自己開源了部分 kernel driver(open-gpu-kernel-modules),但使用者空間的關鍵邏輯與GSP韌體仍是closed binary,對逆向有幫助但不是全貌
- 這條路nouveau社群做了十幾年還沒完全做完,個人專案要有心理準備這是長期戰

### 建議的實際起步順序

1. 先在路線一把整個NVENC/NVDEC的資料流(bitstream in → 硬體排程 → decoded frame out)搞熟,理解session、surface、command buffer的概念
2. 同時开始讀 envytools 的暫存器文件跟 nouveau 的 NVDEC 驅動原始碼,對照你在路線一觀察到的行為(可以用 `nvidia-smi` 、 `strace` 、PCIe封包截取工具去看官方驅動實際送了什麼命令)
3. 兩邊互相驗證,會比單純啃文件有效率很多


這兩張圖分別對應「PCIe 獨立顯卡走的路徑」跟「Tegra SoC 內建 GPU 走的路徑」,兩者用的是同一顆NVDEC/NVENC ASIC設計,但資料怎麼進出這顆ASIC是完全不同的兩套機制。

先看 dGPU(桌上型/筆電獨顯)的版本:應用程式透過驅動API把bitstream送過PCIe匯流排,進到GPU上的command queue,再由Falcon/GSP韌體排程、驅動NVENC/NVDEC這顆固定功能ASIC實際跑編解碼,最後結果寫回VRAM,再透過PCIe傳回主機。

---


![[Pasted image 20260812111121.png]]
再看 Jetson(Tegra SoC)的版本:因為GPU跟CPU共用同一塊實體記憶體(unified memory),沒有PCIe這一段,應用程式改用**V4L2 memory-to-memory**這套Linux標準介面,透過ioctl把buffer排進kernel驅動的佇列,一樣是同一顆NVDEC/NVENC ASIC硬體在跑,但輸出直接寫進SoC共享DRAM,用DMA-BUF做zero-copy,不用像dGPU那樣把資料搬過PCIe來回。

![[Pasted image 20260812111138.png]]

兩張圖對照著看,關鍵差異就在中間那段搬資料的方式:dGPU要走PCIe這條實體匯流排,有額外的傳輸延遲,所以NVIDIA的SDK特別重視CUDA/EGL interop這類zero-copy技巧來避免無謂的來回搬運;Jetson因為是SoC統一記憶體架構,V4L2 M2M本身就是為了這種unified memory設計的介面,DMA-BUF直接讓CPU跟硬體區塊共用同一塊實體位址,天生就沒有PCIe那段開銷。