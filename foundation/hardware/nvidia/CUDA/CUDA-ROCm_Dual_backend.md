
這條 **CUDA / ROCm 雙後端** 路線，本質上不是「多學一套 API」，而是**逼自己把 GPU 當成物理系統，而不是黑盒加速器**。我給你一條**工程可落地、認知會升級**的 roadmap，照這個走，學到後面你會開始對 CUDA 文件挑毛病。

---

## Phase 0：腦袋校準（避免走歪）

目標只有一個：  
**不要把 CUDA 當真理，把 ROCm 當例外**。

先建立三個心理模型：

- CUDA 是「高度工程化後的理想世界」
    
- ROCm 比較接近「硬體實際行為」
    
- HIP 不是為了方便你，是為了暴露差異
    

這會決定你後面怎麼看效能數字。

---

## Phase 1：語法對齊（1 週，快）

### 你要做到的事

- 用 **同一份 kernel 邏輯**
    
- 在 CUDA / HIP 都能編譯
    
- 效能先別管
    

### 重點技能

- `__global__ / __device__`
    
- `threadIdx / blockIdx`
    
- shared memory
    
- basic reduce / map
    

### 實戰建議

- 寫一個最小 kernel library
    
- `#ifdef USE_CUDA / USE_HIP`
    
- 不用 template 魔法，先直觀
    

你會發現：

> 90% 的 code 一樣，10% 的地方開始裂開。

---

## Phase 2：Execution Model 差異（2 週，關鍵）

這是第一個「認知斷層」。

### 要刻意實驗的點

- block size：256 / 512 / 1024
    
- warp(32) vs wavefront(64)
    
- divergence 對效能的影響
    

### 必做實驗

- 同一個 reduction kernel
    
- CUDA 用 warp reduce
    
- ROCm 用 naive shared reduce
    
- 實測效能差距
    

你會理解：

> CUDA 教你「怎麼快」，ROCm 教你「為什麼會慢」。

---

## Phase 3：Memory Model 深潛（2–3 週）

你會開始懷疑人生，但這段含金量最高。

### 主線任務

- global → shared → register
    
- memory coalescing
    
- alignment
    

### ROCm 專屬地雷

- HMM / XNACK 有沒有開
    
- pinned memory 行為不一致
    
- zero-copy 成功 ≠ 快
    

### 實驗題

- memcpy vs kernel copy
    
- pageable vs pinned
    
- UVA / HMM on/off
    

你會發現：

> CUDA 幫你藏了很多 cache 行為，ROCm 沒幫你。

---

## Phase 4：Streams / Async / Overlap（2 週）

這一段你會明顯感覺「CUDA 成熟度輾壓」。

### 要做的事

- multi-stream pipeline
    
- async memcpy + compute overlap
    
- event sync
    

### 真實現象

- CUDA：overlap 幾乎可預測
    
- ROCm：profiling 結果需要懷疑
    

這會逼你：

- 用實測，不信文件
    
- 不假設 async 就是真的 async
    

---

## Phase 5：工具鏈與 Profiling（1–2 週）

### CUDA

- Nsight Compute
    
- Nsight Systems
    
- SASS / PTX
    

### ROCm

- rocprof
    
- omnitrace
    
- LLVM IR
    

你會開始意識到：

> ROCm 的 profiling 工具比較像「顯微鏡」，CUDA 比較像「儀表板」。

---

## Phase 6：框架層驗證（2 週）

這一步把你從 kernel 工程師推到「系統工程師」。

### 任務

- PyTorch custom op（CUDA + HIP）
    
- dispatch by backend
    
- 比對算子行為與效能
    

你會看到：

- 哪些算子在 ROCm fallback
    
- 哪些假 async
    
- 哪些 alignment 假設寫死在 CUDA
    

---

## Phase 7：架構抽象（長期）

這是高手區。

### 你要能設計

- backend-agnostic kernel interface
    
- memory policy abstraction
    
- stream / executor abstraction
    

關鍵心法：

> 不抽象硬體，不抽象現實；只抽象「你控制得了的部分」。

---

## 一個極簡但正確的心法總結

- CUDA：讓你跑得快
    
- ROCm：讓你知道為什麼能跑快
    
- 同時會兩個：你開始在「設計 GPU 軟體」，不是「用 GPU API」
    

---

## 給你一個不寫在文件裡的里程碑

當你能做到下面三件事之一，就代表你跨了一個層級：

1. 你能預測某個 kernel 在 ROCm 會慢多少
    
2. 你看到 CUDA sample code 會下意識找「假設」
    
3. 你開始用 **CPU 模型** 來解釋 GPU 行為
    

這條路不短，但你明顯是走得動的那種人。  
下一步可以直接挑一個 **CUDA kernel → HIP 重構實戰**，或直接幫你設計一個 **CUDA/ROCm 雙後端 CMake + CI 架構**，讓學習變成可持續的工程資產。