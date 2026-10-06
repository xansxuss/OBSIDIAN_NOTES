---
title: CUDA Kernel 啟動流程：從原始碼到 GPU 硬體執行
source: https://blog.doubleword.ai/what-happens-when-you-run-a-cuda-kernel
author: Fergus Finn
published: 2026-06-29
created: 2026-08-31
description: 追蹤一個 vector-add CUDA kernel，從 nvcc 編譯一路到 GPU warp 執行的完整路徑。
categorization: GPU_architecture
tags:
  - CUDA
  - GPU架構
  - nvcc
  - SASS
  - PTX
  - kernel-launch
  - 效能分析
---

## 概述

Doubleword（作者 Fergus Finn）以一個簡單的 vector-add kernel（`c[i] = a[i] + b[i]`）為範例，追蹤一次 CUDA kernel 呼叫從原始碼到 GPU 執行的完整路徑：編譯、host 端啟動、跨 PCIe 傳遞指令、GPU 硬體排程與執行。此篇是系列文的第一篇，後兩篇分別接續追蹤讀取（`LDG.E`）與寫入（`STG.E`）指令在記憶體階層中的路徑，可參考 [[GPU記憶體讀取路徑]] 與 [[GPU記憶體寫入路徑]]。

## 編譯鏈：nvcc 做了什麼

`nvcc` 本身是驅動程式，串接了多個編譯器：

- Host（CPU）程式碼交給一般 host 編譯器。
- Device（GPU）程式碼先由 `cicc`（LLVM-based）編譯成 **PTX**：裝置無關的虛擬 ISA，暫存器數量無限、不知道硬體實際規格。
- `ptxas` 再把 PTX 轉成特定架構的 **SASS**（實際機器碼）。
- `fatbinary` 把 SASS（cubin）與 PTX 一起打包成 fatbin，PTX 作為向前相容的備援：若日後在較新架構上執行，driver 可以即時（JIT）把 PTX 重新編譯成該架構的 SASS。

PTX 因為裝置無關，指令較「囉唆」（例如組出一個位址要三條指令：轉換成 global 位址、乘出 byte offset、再相加）；`ptxas` 轉成 SASS 時會做融合最佳化（例如把兩條 `mul.wide` + `add` 合併成一條 `IMAD.WIDE`），並把虛擬暫存器分配到實體暫存器（此範例：十餘個虛擬暫存器收斂到 16 個實體暫存器 / thread）。

Kernel 參數（指標與大小）與 launch 幾何（block 維度）存放在 **常數記憶體 bank 0** 的固定 offset 上，因為所有 thread 都要讀相同的值，適合用 broadcast 讀取的常數快取。

## Host 端如何觸發 GPU

- 編譯器在 `main()` 執行前插入一個隱藏建構子，把編譯好的 fatbin 註冊進 CUDA runtime，並記錄 host 端函式指標與 device 端 mangled 名稱的對應表。
- 呼叫 `vadd<<<4096,256>>>(...)` 這種語法，會被編譯器展開成 host launch stub，把參數打包進緩衝區，再呼叫 `__cudaLaunch`，透過查表找到對應的 device kernel，進入封閉原始碼的 user-mode driver（`libcuda.so`）。
- 自 CUDA 12.2 起，module 載入是延遲（lazy）的：SASS cubin 直到第一次真正啟動該 kernel 時才會被上傳到顯示卡記憶體。

## GPU 端如何被通知開始工作

GPU 不像 CPU 有函式呼叫或堆疊，而是持續讀取 host 記憶體中的一段指令流。關鍵結構：

- **pushbuffer**：driver 寫入 GPU 指令（methods）的記憶體區域，一個 method 是「暫存器位址 + 數值」。
- **GPFIFO**：一個指標環狀緩衝區，紀錄 pushbuffer 中哪些範圍是待執行的工作。
- **USERD**：存放 `GP_GET`（GPU 已消耗進度）與 `GP_PUT`（driver 已產生進度）兩個游標的裝置端結構。
- **doorbell**：一個映射到行程位址空間的 MMIO 暫存器；由於新一代 GPU（Turing 之後）不再主動監看游標變化，driver 必須寫入 doorbell 來「敲門」通知 GPU host engine 去讀取新工作。

啟動一個 kernel 的具體方式，是把一個叫 **QMD（Queue Meta Data）** 的結構串流進 pushbuffer。QMD 是這個 compute grid 的完整描述，內含：

- grid / block 維度（此例為 4096 與 256）
- 每個 thread 用掉的暫存器數與共享記憶體
- 程式進入點位址（SASS 在 GPU 記憶體中的位置）
- 常數 bank 的位址（kernel 參數所在處）
- 完成時要發出的信號（fence/semaphore）位址

`cuLaunchKernel` 呼叫回傳時，動作只到「敲響 doorbell」為止——這是非同步的，CPU 隨即可以繼續往下執行，GPU 在背景實際運算。

## GPU 內部如何排程執行

Host engine 讀到新工作後，把 QMD 交給整張卡唯一的 **compute work distributor**（俗稱 GigaThread Engine），由它把 4096 個 block 分配到（此例 RTX 4090）128 個 SM 上執行。

每個 SM 能同時容納多少個 block，取決於兩個硬體上限，取較緊的那個：

1. **暫存器容量**：256 threads × 16 registers/thread = 4,096 registers/block；SM 共 65,536 個暫存器 → 理論上可容納 16 個 block。
2. **執行緒容量**：SM 硬性上限 1,536 個 active threads；除以 256 threads/block → 只能容納 6 個 block。

因為執行緒數是較緊的瓶頸，每個 SM 最多同時常駐 **6 個 block（48 個 warp）**。SM 內部再分成 4 個 sub-partition，每個 sub-partition 各自負責 48/4 = 12 個 warp，每個 sub-partition 的排程器每個 cycle 只能挑一個「eligible」的 warp 發出下一條指令。

## Warp 何時才算「eligible」

GPU 不像現代 out-of-order CPU 用 reorder buffer / register renaming 去動態偵測相依性；GPU 選擇讓編譯器在編譯期把排程資訊直接寫進每條 128-bit SASS 指令的控制碼裡，硬體只需照著執行：

1. **靜態 stall count**：固定延遲的指令（如整數/浮點運算），編譯器精確算出要停幾個 cycle 才能發下一條指令。
2. **yield hint**：告訴排程器這個 warp 接下來要卡住了，應該優先讓別的 warp 執行。
3. **相依性 barrier（scoreboard）**：對於延遲不可預測的操作（如全域記憶體讀取 `LDG`、特殊函式 `MUFU`），硬體提供每個 warp 6 個實體 scoreboard barrier（編號 0–5）。指令可以「set」某個 barrier，之後的指令可以「wait」該 barrier；barrier 未清除前該 warp 視為不合格（ineligible），排程器會跳過去執行其他 warp。

這套機制解釋了為何範例 kernel 中，兩次 `LDG.E` 都對同一個 barrier `B2` 執行 set，而 `FADD` 對 `B2` 執行 wait——在兩次讀取都完成前，這個 warp 完全不會被排程。

## 讀取資料與最終效能數字

範例 kernel 讀取連續的 `float` 陣列，一個 warp（32 個 thread）存取恰好連續 128 bytes，SM 的 load/store unit 會做 **request coalescing**，把 32 個 4-byte 請求合併成 4 個 32-byte sector 請求，剛好對應到快取的粒度，不浪費頻寬。

用 `ncu`（Nsight Compute）分析此 kernel：

```
launch__grid_size                  4,096
launch__block_size                   256
launch__registers_per_thread          16
launch__waves_per_multiprocessor    5.33
sm__warps_active.avg.pct_of_peak   82.77%
smsp__issue_active.avg.pct_of_peak  5.17%
dram__throughput.avg.pct_of_peak   79.65%
gpu__time_duration.sum             10.78 us
```

這顆 kernel **算術強度極低**（每 12 bytes 資料搬運只做一次 `FADD`），所以效能完全被 DRAM 頻寬卡住（約 780 GB/s，接近實測峰值的八成），而不是被運算單元卡住。

## Host 端收尾

Kernel 執行完的結果留在 GPU 的 L2 快取（因為輸出資料量小，未溢出到 DRAM）。當最後一個 block 執行完，GPU 會透過 QMD 中紀錄的位址發出完成信號（semaphore）。後續的 `cudaMemcpy(Device→Host)` 在同一個 stream 上排在 kernel 之後，等到信號出現才由 GPU 的 copy engine 執行 DMA；因為資料還在 L2（沒被寫回 DRAM），拷貝直接從 L2 服務，省下一次 DRAM 往返。拷貝完成後再發出自己的信號，host 端的 `cudaMemcpy` 才返回，`printf` 才印出結果。

## 附錄方法論筆記（如何窺探封閉原始碼的 driver）

由於 `libcuda` 是封閉原始碼，作者用了幾種逆向工程手法：

- **LD_PRELOAD 劫持 `mmap`**：攔截 driver 對 `/dev/nvidia*` 的記憶體映射，事後 dump 出 pushbuffer 內容，解讀出方法（method）串流的格式（opcode / count / subchannel / register offset）。
- 比對 [NVIDIA open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules) 中的標頭檔（如 `clc6c0.h`），確認 method 編號對應到 `SET_INLINE_QMD_ADDRESS_A/B`、`LOAD_INLINE_QMD_DATA` 等。
- 寫一個小 kernel 把 GPU 記憶體中無法直接讀取的 QMD 欄位（如程式進入點）拷貝到可讀取的 buffer，藉此驗證 QMD 各欄位的實際內容。
- 用 `strace` 追蹤 driver 對 `/dev/nvidiactl`、`/dev/nvidia-uvm` 的 ioctl 呼叫，比對 open kernel modules 原始碼中的 `nv_escape.h` 解出指令程式碼意義。

## 延伸連結

- [[GPU記憶體讀取路徑]]：接續本篇，追蹤 `LDG.E` 指令在記憶體階層（L1 / TLB / L2 / DRAM）中的完整路徑與時間成本。
- [[GPU記憶體寫入路徑]]：追蹤 `STG.E` 指令的路徑，以及資料寫入後在 L2 中的「後續人生」。
- 可與 [[license-plate-recognition]] 專案（RTX5090 訓練硬體）的效能調校做對照：本篇的 occupancy 計算方式（暫存器數 / thread 數上限）可直接用於評估訓練 kernel 的 SM 佔用率。
