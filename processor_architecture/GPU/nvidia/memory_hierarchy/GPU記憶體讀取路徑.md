---
title: GPU 記憶體讀取路徑：一次 LDG.E 的完整旅程
source: https://blog.doubleword.ai/what-happens-when-a-gpu-reads-memory
author: Fergus Finn
published: 2026-08-13
created: 2026-08-31
description: 追蹤一條全域記憶體讀取指令（LDG.E）在 RTX 4090 上走過 L1、TLB、L2、DRAM 的完整路徑與時間成本。
categorization: GPU_architecture
tags:
  - CUDA
  - GPU架構
  - 記憶體階層
  - L1快取
  - L2快取
  - DRAM
  - 效能分析
  - 逆向工程
---

## 概述

本篇是 [[CUDA-Kernel啟動流程]] 的延伸，專注在同一個 vector-add kernel 中的一條指令：`LDG.E R4, [R4.64]`（讀取全域記憶體）。作者在自己的 RTX 4090 上，用時間量測與探針（probe）實驗，逆向工程出這條指令走過的每一層硬體單元，因為這些細節大多沒有公開文件。

## 整體路徑與時間成本

```
warp → coalescer → L1（~15.4 ns）→ TLB → crossbar → L2（~127.4 ns）→ 記憶體控制器 → DRAM（~255.4 ns）
```

一次完全 miss 到 DRAM 的往返總成本約 **255 ns（約 660 cycles，核心時脈鎖定 2.6 GHz）**，這段期間發出請求的 warp 全程被卡住（ineligible），但同一個 sub-partition 的其他 11 個 warp、同一顆 SM 的其他 36 個 warp、乃至整張卡另外 6096 個 warp 仍在正常運作，用大量平行度把這個延遲藏起來。

## 從暫存器到 coalescer

`LDG.E` 先從暫存器檔案讀出位址：一列暫存器同時存放 32 個 lane 的 `R4`，另一列存放 `R5`（因為位址是 64-bit，需要一對暫存器）。讀出位址後，指令交給 **load/store unit（LSU）**，LSU 做位址運算後，把請求送到 **coalescer**。

Coalescer 的工作是把每個 thread 各自要求的 4 bytes，合併成 L1 快取以 32-byte **sector** 為單位的最少請求數。此範例中 32 個 thread 存取連續的 `float` 陣列，剛好對應到 4 個連續 sector（128 bytes，恰為一條快取線）。

## L1 快取：虛擬定址、虛擬標記

L1 快取以 128-byte **line** 為單位組織，4-way set-associative。關鍵特性：**L1 用虛擬位址做索引與標記**（而非物理位址）。作者用「同一塊物理記憶體映射到兩個不同虛擬位址」的實驗證明這點：若用其中一個虛擬位址建立好一組會互相衝突的 eviction set，換成另一個虛擬別名去存取同一條物理線，衝突就會消失——代表索引是從虛擬位址算出來的。

Set 的選擇是位址某些 bit 的 **XOR parity** 組合（一種避免 2 的冪次跨步存取一直命中同一個 set 的雜湊設計），細節收錄在原文附錄，不在此贅述雜湊函式本身。此範例中資料是第一次載入，L1 必然 miss，請求繼續往下走。

## 位址轉譯：虛擬轉物理

L1 之後，硬體才需要開始處理「虛擬轉物理」的轉譯（因為 L2 是物理定址、物理標記的）。GPU 記憶體的虛實對應在 `cudaMalloc` 配置時就由 driver 寫入 GPU 端的分頁表（page table）。

SM 內建一個 **16 項、全關聯（fully associative）、共用於所有 warp** 的 TLB，用 LRU 汰換。首次載入必定 TLB miss，重新填入約需 **4.4 ns（11 cycles）**；有趣的是，這個 refill 成本在整張卡可映射的所有分頁上幾乎完全一致（誤差 < 0.1 ns），暗示下一層轉譯快取是全域共用、非常廉價的設計。

轉譯完成後，離開 SM 的請求變成「一條 128-byte line 的物理位址 + 想要的 sector 遮罩」，此例中一次請求涵蓋全部 4 個 sector，經由 **crossbar** 送往 L2。

## L2 快取：物理定址，跨 36 個 slice

L2 依物理位址被切成 **36 個 slice**（每個 2 MiB），由位址的複雜雜湊函式決定歸屬的 slice；任何 SM 都能存取任何 slice，36 個 slice 可並行服務，總頻寬因此是單一 slice 的 36 倍。

每個 slice 內部結構與 L1 類似：1024 個 set，每個 set 16-way set-associative（比 L1 的 4-way 更「胖」），線的大小同樣是 128 bytes。此範例中資料第一次被存取，L2 同樣 miss，請求落到 12 個記憶體控制器之一（每個控制器管理 3 個 slice，對應一顆 GDDR6X 顆粒）。

**L2 命中成本約 127 ns（約 330 cycles）**。

## DRAM：activate 與 column read

記憶體控制器要向 DRAM 顆粒下指令才能拿到資料。GDDR6X 的結構：

- 每顆晶片切成 2 個獨立 **channel**
- 每個 channel 有 16 個 **bank**（二維記憶胞陣列）
- 每個 bank 有 65,536 個 **row**，每個 row 為 1 KiB（32 個 32-byte column）

DRAM 一次只能「開啟」一個 row（**activate**，成本高），開啟後才能便宜地讀取該 row 內任意 column（**read**）。此範例的 4 個 sector 剛好是同一個 row 的 4 個 column，因此只需 1 次 activate + 4 次 read。

實體層面：DRAM 每個 cell 是一個電容加一個電晶體；activate 會驅動該 row 的字元線（wordline），把電容電荷透過位元線（bitline）送進感測放大器（sense amplifier），放大成穩定的數位訊號。Read 指令再從中選出所需 bit，透過 16 個 data pin 以 **PAM4** 訊號（每個符號 4 種電壓準位，帶 2 bit）序列化送出。

## 回程

資料反序列化後寫入 L2 該 slice 的 line，再經 crossbar 回到原本的 SM，填入 L1 對應 slot，最終寫入暫存器 `R4`。此時，發出讀取時設定的相依性 barrier 被清除，該 warp 重新變成 eligible，排程器下個 cycle 可以發出等待這筆資料的 `FADD`。

## 附錄方法論筆記

- **量測工具**：主要靠「pointer chase」（讓一串指標形成相依鏈，重複跳躍兩萬次取平均延遲）與 `ncu` 硬體計數器（取多次量測的斜率消除固定開銷）。
- **判斷 L1 是虛擬還是物理定址**：如前述的雙重映射實驗。
- **判斷 L2 slice 對應函式**：需要先能分辨「兩個位址是否落在同一 slice」，作者用同一個記憶體控制器頻寬會被共用 slice 的存取拖慢的特性做間接量測，再配合另一張晶片（L40S，同款 die 但啟用更多 slice）互相印證，反推出一個以 XOR parity 與 mod-9 計數器組成的函式。
- **DRAM row size**：利用「同 row 存取遠快於跨 row」的特性，掃描不同 offset 找出從哪個距離開始會跨 row，得出 row size 為 1 KiB。
- **DRAM refresh 的干擾**：約 2% 的存取會額外多花 ~210 ns，是 DRAM 定期刷新電容電荷造成的固定長度停頓。

## 延伸連結

- [[CUDA-Kernel啟動流程]]：本篇的前導文章，說明這條 `LDG.E` 指令所屬的 kernel 是如何被啟動、排程到 SM 上的。
- [[GPU記憶體寫入路徑]]：對稱地追蹤同一個 kernel 中 `STG.E`（寫入）指令的路徑，並說明資料寫入 L2 後如何、何時真正落地到 DRAM。
