---
title: GPU 記憶體寫入路徑：一次 STG.E 的旅程與後續人生
source: https://blog.doubleword.ai/what-happens-when-a-gpu-writes-memory
author: Fergus Finn
published: 2026-08-28
created: 2026-08-31
description: 追蹤一條全域記憶體寫入指令（STG.E）在 RTX 4090 上的路徑，以及資料寫入 L2 後如何、何時才真正落地到 DRAM。
categorization: GPU_architecture
tags:
  - CUDA
  - GPU架構
  - 記憶體階層
  - L1快取
  - L2快取
  - DRAM
  - 快取替換策略
  - memory-fence
  - 效能分析
---

## 概述

本篇接續 [[GPU記憶體讀取路徑]]，對稱地追蹤同一個 vector-add kernel 中負責寫回結果的指令：`STG.E [R6.64], R9`。核心洞見是讀與寫在「完成」的定義上完全不對稱：**讀取要等到資料真的到手才算做完，寫入卻是「發出去就算數」**，資料的實際落地是延後、非同步發生的。

## 路徑總覽

```
warp（發出約需 6 cycle）→ coalescer → L1（write-through，直接往下送）
→ crossbar → L2（收到 ack 約需 140 ns）→ 之後才視情況寫回 DRAM（on eviction）
```

`LDG.E` 全程要等 255 ns；`STG.E` 只需要約 6 cycle（2.3 ns）就能讓 warp 的角度上「完成」，可以馬上執行下一條指令，不需要等資料真正寫進任何地方。

## 離開 warp

`STG.E` 需要讀 3 列暫存器：組成 64-bit 位址的兩列，加上要寫入的資料本身那列（`LDG.E` 只需要讀位址的兩列）。LSU 送出「寫入這些位址」的指令、32-bit 有效 lane 遮罩，以及 32 個位址。

**發送速率上限**：一個 warp 大約每 6.1 cycle（2.3 ns）才能發出下一條 `STG.E`，跟同時有幾個 lane 在寫無關；SM 出口頻寬上限是每 cycle 32 bytes，若所有 warp 都在寫，會在這裡形成瓶頸。

Coalescer 把 32 個 4-byte 寫入合併成最少的 32-byte sector（此例連續 128 bytes → 4 個 sector，一條 line）。與讀取不同的是，寫入的每個 sector 會附帶一個 **byte mask**，標記這個 sector 中哪些 byte 真的要被覆寫。

## 通過 L1：write-through

L1 快取是 **write-through**：不論該 line 目前是否已經在 L1，sector、mask 與資料都會直接往下送到 L2，不會在 L1 端就算完成。作者透過三個實驗確認這個策略：

1. **miss 時是否在 L1 配置空間**：先寫入一條線，再馬上 pointer chase 讀它，若立刻命中就代表有配置——實驗結果是有，排除 write-around。
2. **hit 時是否保留 L1 副本**：warm up 一組 64 條線的 pointer chase 迴圈，每次跳躍前先寫入剛離開的那條線，64 步後繞回來若仍是 L1 速度，代表副本被保留、更新，而非被丟棄。
3. **是在寫入當下、還是等到被替換時才送到 L2**：用 `lts__t_requests_srcunit_tex_op_write` 計數器確認每次寫入都會立刻在 L2 端記一筆請求。

三者合起來確認是 **write-through with allocate & update**（原文列出的三選項之二）。

若該 write 需要在 L1 的 set 中騰出空間，舊資料以嚴格 LRU 順序讓位。

## 抵達 L2

Sector 經 crossbar 送到由物理位址雜湊決定的其中一個 L2 slice（雜湊函式與 [[GPU記憶體讀取路徑]] 中讀取路徑用的相同）。每次寫入只送一次請求（即使涵蓋整條 line 的 4 個 sector）。

- 若該 line 已存在於 L2：依 byte mask 覆寫對應 byte，並將該 sector 標記為 **dirty**。
- 若該 line 不存在：L2 需要找一個空位（可能需要先把別的 line 逐出到 DRAM），找到位置後寫入資料，並記錄哪些 byte 有效。

寫完後 L2 會回傳一個 **acknowledgement** 經 crossbar 送回 SM——**約需 140 ns**，這就是「完成」訊號被 LSU 消耗的時間點。此時發出這個 `STG.E` 的 warp 通常早已執行到別的地方（此範例中甚至整個 kernel 都已結束），資料實際上還沒到 DRAM，只是「dirty 地」留在 L2。

## 髒資料的後續人生：怎麼、何時真正落地 DRAM

L2 是整張卡所有記憶體流量的匯集點，空間終究會用完，需要替換策略決定誰該被清出去。每條 line 帶有：

- **RRPV（re-reference prediction value）**：0、1、2 三種狀態（數字愈小代表快取認為愈快會再被用到）。新插入的 line（不論讀或寫）一律先設為 **1**。
- **dirty mask**：標記這份資料是否只存在於 L2（尚未寫回 DRAM）。

替換策略運作方式：

1. 命中：讀命中把 RRPV 設回 0；寫命中則更新對應 sector 並維持/更新 dirty mask。
2. Miss 需要騰位：先在該 set 的 16 個 way 中尋找 RRPV=2 的 line；若找不到，把全部 RRPV 值加一再重找。找到多個候選時取最久未使用的一個。
   - 若該候選是 **dirty**：不會立刻踢出，而是先把它的 dirty sector 送去記憶體控制器寫回 DRAM（進入該 set 的 FIFO write-back buffer，每兩次 fill 才會從佇列中彈出一個），然後繼續找下一個乾淨的候選來真正騰出空間。
   - 找到乾淨的候選後，才真正踢出、讓新資料進駐（RRPV 設為 1）。

另外還有一條 **「≥8 dirty」清潔規則**：當一個 set 中髒 line 數量達到 8 條（滿額 16 條的一半）以上時，寫入發生前會先主動把該 set 中最久未寫入的髒 line 清乾淨。這是為了避免「一整個 set 全是髒資料，miss 一來就要連環觸發一大串同時寫回，瞬間塞爆記憶體控制器佇列，拖慢後續所有讀寫」的最壞情況——這條規則像個「工友」，平時就順手清潔，讓 DRAM 流量分散、平滑。

寫回 DRAM 時，控制器一樣要先 activate 該 row，再依 byte mask 逐個 sector 送出資料（GDDR6X 可以直接依 mask 只寫特定 byte；相較之下，有 ECC 的 HBM 因為以 codeword 為單位運作，做不到直接的 partial write，必須用「讀出、合併、再寫入」的方式間接完成）。

## 資料何時才「真的可見」：Fence 與 Scope

`STG.E` 本身是 fire-and-forget，若程式需要確保資料真的落地、其他人看得到，要靠 **fence** 指令，依可見範圍分三種：

| Fence | 可見範圍 | 等待點 | 成本 |
| --- | --- | --- | --- |
| `membar.cta` | 同一個 block（CTA） | 交給 L1（SM 內流量的匯集點）就算數 | 約 1 ns（3 cycle） |
| `membar.gl` | 整張晶片所有 SM | 收到 L2 的 ack | 約 140 ns（同 L2 命中時間） |
| `membar.sys` | 主機與其他裝置（跨 PCIe / NVLink） | 更複雜的跨系統排序 | 約 1 µs |

值得注意：即使 `membar.gl` 完成，代表資料「原則上」全域可見，其他 SM 若要真的讀到，仍必須主動繞過自己可能過時的 L1（例如用 `LDG.E.STRONG.GPU`，或 `ld.acquire.gpu` 編譯出的 `CCTL.IVALL` 指令去整個 invalidate 該 SM 的 L1）。Fence 本身也是雙向的：除了確保自己先前的寫入可見，也會 invalidate 自己的 L1，確保之後讀到的是別人寫入的最新資料。

範例 kernel 的 PTX 中並沒有寫任何 fence，但因為結尾接的是 `cudaMemcpyDeviceToHost`（需要 system 層級的可見性），driver 會在 kernel 結束時自動插入一個等同 `membar.sys` 的系統層級 membar，成本與量測到的 `membar.sys`（約 0.74 µs）一致。

## 附錄方法論筆記

- **判斷 write-through/write-back**：見上文三個實驗。
- **判斷 RRPV 插入值與替換演算法**：用「填滿一個 set、插入目標 line、再灌入干擾流量，數多少次會把目標擠掉」的存活曲線（survival curve）反推。例如：D 條髒 line 會讓目標多撐 `2D-1` 次干擾才被擠掉，暗示髒佇列每兩次 fill 才排出一個。
- **量測 fence 成本**：用一段迴圈重複發 N 次 store，比較「不加 fence」與「每次都加 fence」兩種情況下的耗時斜率差。
- **驗證系統層 membar 的插入點**：直接從 pushbuffer 中讀出 `CWD_MEMBAR_TYPE = L1_SYSMEMBAR` 欄位。

## 延伸連結

- [[CUDA-Kernel啟動流程]]：本系列第一篇，說明這個 kernel 從原始碼到啟動的完整路徑。
- [[GPU記憶體讀取路徑]]：對稱的讀取路徑分析，兩篇合看可以完整理解 L1/L2 快取一致性設計的全貌。
