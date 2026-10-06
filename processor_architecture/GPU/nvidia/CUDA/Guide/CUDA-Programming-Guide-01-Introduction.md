---
title: CUDA C++ Programming Guide - Ch.1 Introduction
source: https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html
author: NVIDIA Corporation
published: 2024-10-01
created: 2026-09-22
description: CUDA 12.6 官方指南第1章，說明 GPU 平行運算優勢、CUDA 平台定義與可擴展程式設計模型
categorization: processor_architecture/GPU/nvidia/CUDA/Guide
tags:
  - CUDA
  - GPU程式設計
  - 平行運算
  - NVIDIA
  - 官方文件筆記
---

# Ch.1 Introduction

> 版本：CUDA C++ Programming Guide 12.6
> 官方 PDF：[CUDA_C_Programming_Guide.pdf](https://docs.nvidia.com/cuda/archive/12.6.2/pdf/CUDA_C_Programming_Guide.pdf)
> 系列：[[CUDA-Programming-Guide-02-ProgrammingModel]] → [[CUDA-Programming-Guide-03-ProgrammingInterface]]

---

## 1.1 The Benefits of Using GPUs（為什麼用 GPU）

- GPU 在同樣的價格/功耗範圍內，**指令吞吐量**與**記憶體頻寬**都遠高於 CPU。
- 本質差異來自設計目標不同：
  - **CPU**：追求單一執行緒（thread）盡快執行完畢，同時只能跑數十個執行緒。
  - **GPU**：犧牲單執行緒效能，換取同時執行**數千個執行緒**，用大量平行運算把記憶體延遲「藏」起來。
- GPU 把更多電晶體資源用在**資料運算（ALU）**，而非快取與流程控制（branch prediction 等）。
- 適合平行度高的應用：影像處理、物理模擬、金融運算、生物資訊、深度學習等，不限於圖形繪製。

> 📎 **Figure 1 - The GPU Devotes More Transistors to Data Processing**
> CPU vs GPU 晶片資源分配對比示意圖（ALU 比例差異）
> 原始圖表：[官方 PDF 第 3 頁](https://docs.nvidia.com/cuda/archive/12.6.2/pdf/CUDA_C_Programming_Guide.pdf#page=3)

---

## 1.2 CUDA®: A General-Purpose Parallel Computing Platform and Programming Model

- 2006 年 11 月 NVIDIA 推出 **CUDA**（Compute Unified Device Architecture）：
  - 一個**通用平行運算平台與程式設計模型**
  - 利用 NVIDIA GPU 內部的平行運算引擎，以更有效率的方式解決複雜計算問題
- 開發者可用 **C++** 作為高階語言直接撰寫 GPU 程式
- 同時也支援 FORTRAN、DirectCompute、OpenACC 等其他語言/API/指令式（directive-based）介面

> 📎 **Figure 2 - GPU Computing Applications**
> CUDA 支援的語言與 API 生態系全貌示意圖
> 原始圖表：[官方 PDF 第 5 頁](https://docs.nvidia.com/cuda/archive/12.6.2/pdf/CUDA_C_Programming_Guide.pdf#page=5)

---

## 1.3 A Scalable Programming Model（可擴展的程式設計模型）

CUDA 的核心設計理念：**「寫一次程式，自動適應不同規模的硬體」**

### 三個核心抽象概念

1. **執行緒群組階層**（hierarchy of thread groups）
2. **共享記憶體**（shared memories）
3. **屏障同步**（barrier synchronization）

### 設計邏輯

- 把大問題拆成可**獨立**平行求解的子問題 → 由一個 **thread block** 負責
- 子問題內再拆成更細的部分 → 由 block 內所有 thread **協同**運算
- 每個 thread block 可被排到 GPU 上**任一個 SM（Streaming Multiprocessor）**執行：
  - 順序任意，可並行也可序列
  - 同一份編譯好的 CUDA 程式，在擁有不同 SM 數量的 GPU 上都可直接執行，SM 越多自動跑越快
  - **不需要重寫程式碼**，這是 CUDA 能橫跨整個 GPU 產品線（入門到旗艦）的關鍵原因

> 📎 **Figure 3 - Automatic Scalability**
> 同一份程式在不同 SM 數量 GPU 上自動平行擴展示意圖
> 原始圖表：[官方 PDF 第 7 頁](https://docs.nvidia.com/cuda/archive/12.6.2/pdf/CUDA_C_Programming_Guide.pdf#page=7)

---

## 1.4 Document Structure（文件架構）

本指南後續章節與對應筆記：

| 章節 | 內容 | 對應筆記 |
|---|---|---|
| Programming Model | CUDA 執行緒/記憶體模型 | [[CUDA-Programming-Guide-02-ProgrammingModel]] |
| Programming Interface | NVCC 編譯、Runtime API | [[CUDA-Programming-Guide-03-ProgrammingInterface]] |
| Hardware Implementation | SM 硬體、SIMT 架構 | [[CUDA-Programming-Guide-04-HardwareImplementation]] |
| Performance Guidelines | 效能優化策略 | [[CUDA-Programming-Guide-05-PerformanceGuidelines]] |
| C++ Language Extensions | `__global__`、`<<<>>>` 等語法 | [[CUDA-Programming-Guide-07-CppLanguageExtensions]] |
| Cooperative Groups | 執行緒群組同步原語 | [[CUDA-Programming-Guide-08-CooperativeGroups]] |
| Unified Memory Programming | 統一記憶體模型 | [[CUDA-Programming-Guide-19-UnifiedMemory]] |

---

## 小結

- GPU 因晶片架構設計，天生適合高平行度的計算任務，與 CPU 是互補而非競爭關係。
- CUDA 的可擴展性源自 thread block 的獨立執行設計，是後面所有效能優化概念的根基。
- 理解「為什麼 GPU 這樣設計」，比死背 API 更重要，本章是整本指南的世界觀基礎。
