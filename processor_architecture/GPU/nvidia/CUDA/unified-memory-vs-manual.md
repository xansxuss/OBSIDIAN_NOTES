---
title: CUDA Unified Memory 與手動記憶體管理比較
source: 
author: 
published: 
created: 2026-09-17
description: 比較 CUDA Unified Memory (cudaMallocManaged) 與傳統手動記憶體管理 (cudaMalloc + cudaMemcpy) 的機制、效能與適用場景
categorization: computer_vision
tags:
  - CUDA
  - GPU
  - memory-management
  - unified-memory
  - performance
---

## 概念定義

### Unified Memory（統一記憶體）
- 由 `cudaMallocManaged()` 配置，回傳一塊 CPU 與 GPU 都能直接存取的**單一虛擬位址空間**。
- 底層由驅動程式與硬體（page fault 機制）自動處理資料在 Host 與 Device 之間的搬移，不需要程式手動呼叫 `cudaMemcpy`。
- 需要 GPU 支援 Pascal（compute capability 6.x）以上架構，才有完整的 on-demand page migration 能力；更早期架構是用「整批預先複製」的方式模擬。

### 手動記憶體管理
- 分別用 `cudaMalloc()` 配置 Device 記憶體、`malloc()`（或 `cudaMallocHost()` 配置 pinned memory）配置 Host 記憶體。
- 資料搬移必須明確呼叫 `cudaMemcpy()` / `cudaMemcpyAsync()`，方向（H2D / D2H / D2D）與時機完全由開發者控制。

## 核心差異對照

| 項目 | Unified Memory | 手動管理 |
|---|---|---|
| API 複雜度 | 低（一個指標到處用） | 高（Host/Device 各一份指標） |
| 資料搬移時機 | 由 page fault 觸發，隱式、非同步不易掌控 | 明確呼叫，時機完全可控 |
| 效能可預測性 | 較差，首次存取有 page fault overhead | 較佳，可用 `cudaMemcpyAsync` + stream 重疊計算與傳輸 |
| Prefetch 最佳化 | 可用 `cudaMemPrefetchAsync()` 手動提示，減少 page fault | 不需要，本來就是顯式搬移 |
| 多 GPU / NUMA 情境 | 需注意 `cudaMemAdvise()` 設定存取模式，否則容易 thrashing | 完全自行控制，較不易誤用 |
| 除錯難度 | 資料何時實際搬移不透明，較難用工具追蹤 | 每次搬移都是明確呼叫，容易用 Nsight 等工具分析 |
| 程式碼可讀性 | 高，邏輯更接近一般 CPU 程式 | 需要額外處理雙緩衝區與同步 |
| 記憶體用量 | 可能因 driver 內部管理而略高 | 可精準控制配置大小 |

## 適用場景建議

### 適合用 Unified Memory
- 快速原型開發（prototype），先求邏輯正確再最佳化效能
- 資料結構複雜、含指標的巢狀結構（例如 tree、graph），手動搬移非常麻煩
- CPU/GPU 交替存取頻繁、但每次存取量不大的情境
- 教學或概念驗證（PoC）用途

### 適合用手動管理
- 對延遲（latency）與吞吐量（throughput）要求嚴格的即時系統，例如 [[license-plate-recognition]] 這類即時推論管線
- 需要精準重疊計算與傳輸（`cudaMemcpyAsync` + multiple streams）以榨乾硬體效能，類似 [[video-codec-sdk-decode-bench]] 中做的 FPS benchmark 情境
- 嵌入式平台（如 Jetson 系列）資源有限，記憶體配置需要精細控管，可對應 [[jetson-video-decode]] 的開發脈絡
- 需要避免 page fault 造成的效能抖動（jitter），對即時性要求高的系統

## 效能陷阱筆記

1. **First-touch page fault**：Unified Memory 第一次在 GPU 上存取尚未搬移的頁面時，會觸發 page fault 並暫停 kernel 執行去搬資料，這個 overhead 在頻繁小量存取時會被放大。
2. **忘記 prefetch**：若已知資料即將在 GPU 使用，應主動呼叫 `cudaMemPrefetchAsync()` 搬到對應裝置，否則會退化成邊跑邊搬的低效模式。
3. **手動管理下的 pinned memory**：Host 端若用一般 `malloc()` 而非 `cudaMallocHost()`，`cudaMemcpyAsync` 實際上不會是真非同步（因為 pageable memory 需要先複製到內部 staging buffer）。
4. **過度依賴 Unified Memory 做效能量測**：若要做像 AppDecPerf 這類精確 FPS 測試，混用 Unified Memory 可能讓 page fault 的隨機延遲汙染量測結果，建議 benchmark 情境優先採用手動管理。

## 簡易程式碼對照

```cpp
// Unified Memory 版本
float* data;
cudaMallocManaged(&data, N * sizeof(float));
// CPU 可直接讀寫
for (int i = 0; i < N; ++i) data[i] = i;
// GPU kernel 直接使用同一指標
kernel<<<blocks, threads>>>(data, N);
cudaDeviceSynchronize();
// 讀取結果也直接用同一指標
cudaFree(data);
```

```cpp
// 手動管理版本
float* h_data = (float*)malloc(N * sizeof(float));
float* d_data;
cudaMalloc(&d_data, N * sizeof(float));

for (int i = 0; i < N; ++i) h_data[i] = i;
cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

kernel<<<blocks, threads>>>(d_data, N);
cudaDeviceSynchronize();

cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

cudaFree(d_data);
free(h_data);
```

- Unified Memory 版本邏輯上少了顯式的 H2D / D2H 搬移呼叫，程式碼更簡潔，但實際搬移時機交給 driver 決定。
- 手動版本每一步搬移都寫得很清楚，方便對應到 profiling 工具上看到的時間軸。

## 結論

Unified Memory 在開發效率與程式碼可讀性上有明顯優勢，但在需要精準效能控制（尤其是即時系統或效能 benchmark）的場合，手動記憶體管理仍是主流做法。實務上常見策略是：開發初期用 Unified Memory 快速驗證邏輯，效能調校階段再視需要局部改回手動管理，或搭配 `cudaMemPrefetchAsync` / `cudaMemAdvise` 微調 Unified Memory 的行為，取得兩者的平衡。
