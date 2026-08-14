# 自製 NVIDIA 專用硬體解編碼器 Roadmap

適用範圍:dGPU(PCIe 獨立顯示卡)與 iGPU(Jetson/Tegra SoC)

---

## 專案定位

本專案分成兩條性質完全不同、但可以並行推進的路線:

1. **應用層開發**:呼叫 NVIDIA 公開的 NVENC/NVDEC API 或 V4L2 M2M 介面,包裝出自己的解編碼器 framework。技術成熟、有官方文件支援,可在數週到數月內做出堪用成果。
2. **驅動層逆向工程**:逆向 NVIDIA 未公開的 NVENC/NVDEC 韌體通訊協定(暫存器映射、Falcon 命令佇列格式),寫出開源驅動,類似 nouveau 專案的方向。技術門檻高,是長期投入型專案,沒有明確終點。

建議先把路線一做到堪用,同時開始累積路線二需要的觀察資料,再逐步切換重心,而不是兩條線同時全力硬推。

---

## 硬體架構背景

NVENC(編碼)與 NVDEC(解碼)是 GPU 晶片上**獨立的 ASIC 區塊**,不佔用 CUDA 核心運算資源。兩者各自由基於 **Falcon 微處理器**(NVIDIA 自研微控制器架構)的韌體驅動。自 Turing 世代起,控制邏輯多半被搬進獨立的 **GSP(GPU System Processor)** 協同處理器,韌體為加密簽章的二進位檔,無法直接反組譯。

- **韌體本身**是黑盒子,無法逆向出明碼邏輯。
- **驅動如何跟這顆黑盒子溝通**(暫存器位址、命令佇列格式)是可以逆向的,nouveau 專案就是這樣做的。

### dGPU 資料流

```
主機 (CPU)
  └─ 應用程式 + NVENC/NVDEC API
        │  PCIe 送入 bitstream
        ▼
┌─────────────────── dGPU ───────────────────┐
│  Command queue ──▶ NVENC/NVDEC ASIC          │
│       ▲                  │                    │
│  Falcon/GSP 韌體 ─────────┘                    │
│  (排程並控制 ASIC)         ▼                    │
│                     VRAM frame buffer         │
└────────────────────────────────────────────┘
        │  結果經 PCIe 傳回主機
        ▼
```

### Jetson(Tegra SoC / iGPU)資料流

```
應用程式 (V4L2 client)
  └─ libargus / NvVideoDecoder API
        │  ioctl VIDIOC_QBUF
        ▼
┌────────────── Tegra SoC (Jetson iGPU) ──────────────┐
│  V4L2 M2M 驅動 ──▶ Kernel buffer 佇列                  │
│  (/dev/nvhost-nvdec)   (OUTPUT/CAPTURE queue)          │
│                              │                          │
│                              ▼                          │
│                   NVDEC/NVENC 硬體區塊                  │
│                   (與 dGPU 同款 ASIC)                    │
│                              │                          │
│                              ▼                          │
│                   共享 SoC DRAM (DMA-BUF, zero-copy)     │
└──────────────────────────────────────────────────────┘
        │  結果透過 DMA-BUF fd 直接回傳應用程式
        ▼
```

**關鍵差異**:dGPU 要走 PCIe 實體匯流排,有額外傳輸延遲,因此 NVIDIA SDK 特別重視 CUDA/EGL interop 這類 zero-copy 技巧;Jetson 是 SoC 統一記憶體架構,V4L2 M2M 本身就是為 unified memory 設計的介面,DMA-BUF 讓 CPU 與硬體區塊共用同一塊實體位址,天生沒有 PCIe 開銷。

---

## 七階段開發路線

### 階段 1:dGPU 最小可行版本

用官方 NVENC/NVDEC SDK 包一個最簡單的 encode/decode session:讀一段 H.264/H.265 bitstream 進去,拿到解碼後 frame,或反過來把 raw frame 編碼輸出。先不管效能,目標是把 session 建立、command submit、frame 取回這條路徑走通。C++ 部分用裸指標跟自己的 ring buffer 管理 frame pool,避開 STL。

### 階段 2:Jetson 對應版本

在 Jetson 上用 V4L2 M2M API(NvVideoDecoder/NvVideoEncoder)重做一次同樣的最小流程。這步的重點是體會兩套 API 在語意上的差異:dGPU 是 session-based,Jetson 是 queue-based(OUTPUT/CAPTURE 兩個佇列),之後要包統一介面時才知道抽象層該怎麼切。

### 階段 3:整合成自己的 C++ framework

把前兩步的經驗抽成一套共用介面:session 管理、記憶體池、zero-copy(dGPU 用 CUDA/EGL interop,Jetson 用 DMA-BUF)。這是未來拿來測試、展示、甚至商用的成果,值得花時間打磨 API 設計,不用急著碰驅動層。

### 階段 4:蒐集官方驅動的實際行為

在熟悉的世代 GPU(建議先挑 Maxwell/Pascal,文件最齊全)上,用 strace、PCIe 封包截取工具、或直接讀 open-gpu-kernel-modules 裡開放的部分,記錄官方 driver 送了哪些命令、暫存器寫入序列長什麼樣子。這一步不用逆向,純粹先建立「正常運作時系統長怎樣」的基準線。

### 階段 5:啃 envytools 與 nouveau 現有成果

對照第 4 步蒐集到的行為,去讀 envytools 的暫存器文件跟 nouveau 的 NVDEC 驅動原始碼,理解 Falcon 命令佇列格式、初始化序列。目標是先看懂別人已經逆向出來的部分,不要一開始就想自己從零挖。

### 階段 6:嘗試繞過官方 userspace driver

在選定的舊世代 GPU 上,試著自己直接送最小的一組命令給硬體(例如觸發一次簡單的解碼),不透過 NVIDIA 官方 driver。這是驗證對格式理解正確與否的第一個真正里程碑,失敗很正常,GPU 掛掉重開機也正常。

⚠️ **風險提醒**:直接送裸命令有機會讓 GPU 進入異常狀態甚至觸發 TDR(Timeout Detection and Recovery)或系統當機,建議準備一台不放重要資料的測試機,不要在主力工作機上做這步。

### 階段 7:長期擴充與維護

確認舊世代跑得通之後,再往新世代(尤其 Turing 之後導入 GSP 韌體的部分)擴充。這條線本質上是持續逆向工程專案,心態上要當成像 nouveau 一樣的長期投入,而不是一次性任務。

---

## 時間預期

|階段|預估時程|性質|
|---|---|---|
|1-3(應用層)|1-2 個月|有官方文件支援,可預期時程|
|4-7(驅動層)|無明確終點|長期逆向工程,需持續投入|

**建議**:第 4-6 步先挑 Maxwell 或 Pascal 這種較舊世代,因為控制邏輯還沒被搬進 GSP 協同處理器,暫存器介面相對直接,文件跟社群成果也最完整。Turing 以後 NVIDIA 把大量控制邏輯藏進加密簽章的 GSP 韌體裡,逆向難度直接跳一個檔次。

---

## 參考工具與資源

- **NVIDIA Video Codec SDK**:官方 NVENC/NVDEC API(C 介面)
- **Jetson Linux (L4T) 多媒體 API**:`NvVideoEncoder`/`NvVideoDecoder`,底層為 V4L2 M2M
- **FFmpeg hwaccel 後端**:`h264_nvenc`/`hevc_nvdec` 可作為對照組
- **envytools**:暫存器與命令格式逆向文件、工具集
- **nouveau 專案**:開源驅動,已有部分 NVDEC 支援
- **open-gpu-kernel-modules**:NVIDIA 官方開源的部分 kernel driver(不含 GSP 韌體本體)