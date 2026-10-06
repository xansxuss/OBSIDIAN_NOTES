---
title: NVIDIA CuTe DSL 使用教學
source: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html
author: NVIDIA
published:
created: 2026-10-06
description: CUTLASS 4.x 的 Python DSL「CuTe DSL」的安裝方式、核心 decorator 與向量加法入門範例。
categorization: processor_architecture/GPU/nvidia/CUDA/CuTe_DSL
tags:
  - CuTeDSL
  - CUTLASS
  - CUDA
  - Python-DSL
  - GPU-kernel
  - JIT
  - MLIR
---

# NVIDIA CuTe DSL 使用教學

## 1. 簡介

- **CuTe DSL** 是 CUTLASS 4.x 推出的第一個 Python DSL，讓人用 Python 撰寫高效能 CUDA kernel。
- 定位為**低階程式模型**，與 CuTe C++ 的抽象一致：layout、tensor、hardware atom，並可完整控制 thread 與資料階層。
- 目標硬體：Ampere、Hopper、Blackwell 的 Tensor Core。
- 運作流程：Python → 自訂 IR → 透過 MLIR 與 ptxas 做 **JIT 編譯** → CUDA kernel。
- 優點：編譯速度快、metaprogramming 比 C++ 直覺、可與 [[PyTorch]] 直接整合（不需 glue code）。
- 限制：
  - 不取代 [[CUTLASS]] C++ 函式庫（2.x / 3.x API）。
  - 仍在 **public beta**，API 持續演進（PyPI 頁面預計 2026 夏季結束 beta，實際狀態需到官方文件確認）。
  - 官方文件另有 Limitations 章節，說明與 C++ 的差異。

> [!note] 英文用語
> 正確寫法為 **CuTe DSL**（中間有空格）、**NVIDIA**（全大寫）。

## 2. 環境需求與安裝

- 僅支援 **Linux**，Python 3.10 ~ 3.14（以 4.4 版文件為準）。
- NVIDIA 驅動版本需與對應的 CUDA Toolkit（12.9 或 13.1）一致。

```bash
# 若之前裝過舊版，先移除
pip uninstall nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-cu13 -y

# CUDA Toolkit 12.9
pip install nvidia-cutlass-dsl

# CUDA Toolkit 13.1
pip install "nvidia-cutlass-dsl[cu13]"

# 建議一併安裝（跑範例用）
pip install torch jupyter
```

- 要搭配 GitHub 最新範例時，clone 儲存庫後用對應 commit 的 `setup.sh`（`--cu12` 或 `--cu13`）安裝，避免版本不相容。
- 使用 Jupyter 時建議設定 `export PYTHONUNBUFFERED=1`。

## 3. 核心概念

| decorator      | 執行位置        | 用途                               |
| -------------- | ----------- | -------------------------------- |
| `@cute.kernel` | GPU（device） | 定義 kernel，等同 CUDA 的 `__global__` |
| `@cute.jit`    | CPU（host）   | 計算 grid / block 並啟動 kernel       |

### 3.1 重要 API

- `cute.arch.thread_idx()` / `block_idx()` / `block_dim()`：取得 `(x, y, z)` 三元組，對應 `threadIdx` / `blockIdx` / `blockDim`。
- `cute.size(tensor)`：取得元素總數。
- `from_dlpack(torch_tensor)`：透過 [[DLPack]] 零拷貝包裝成 CuTe tensor。
- `cute.compile(fn, *args)`：先編譯，之後重複呼叫不再編譯，適合 benchmark。
- `cute.printf(...)`：在 kernel 內列印（不要用 Python 內建 `print`，它只在 trace 階段執行一次）。

## 4. 範例：向量加法

對應 C++ 的 `C[i] = A[i] + B[i]`。

```python
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ---------- Device 端：每個 thread 處理一個元素 ----------
@cute.kernel
def vec_add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    # 取得 thread 在 block 內的索引、block 在 grid 內的索引、block 大小
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    # 全域索引，等同 blockIdx.x * blockDim.x + threadIdx.x
    idx = bidx * bdim + tidx

    # 邊界檢查：總 thread 數可能大於元素數
    if idx < cute.size(gC):
        gC[idx] = gA[idx] + gB[idx]


# ---------- Host 端：計算啟動配置並 launch ----------
@cute.jit
def vec_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    n = cute.size(mC)                       # 元素總數
    threads = 256                           # 每個 block 的 thread 數
    blocks = (n + threads - 1) // threads   # 無條件進位，確保涵蓋所有元素

    vec_add_kernel(mA, mB, mC).launch(
        grid=(blocks, 1, 1),
        block=(threads, 1, 1),
    )


if __name__ == "__main__":
    N = 1 << 20
    a = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)

    # 零拷貝包裝成 CuTe tensor
    mA, mB, mC = from_dlpack(a), from_dlpack(b), from_dlpack(c)

    # 先編譯一次，之後重複呼叫不需要再編譯
    compiled = cute.compile(vec_add, mA, mB, mC)
    compiled(mA, mB, mC)

    torch.cuda.synchronize()
    print("結果正確:", torch.allclose(c, a + b))
```

### 4.1 邏輯重點

1. `from_dlpack` 共用記憶體不複製資料，所以能與 PyTorch 無縫整合。
2. `cute.compile` 把 JIT 編譯與執行分開；直接呼叫 `vec_add(...)` 也能跑，但會隱含編譯。
3. `if idx < cute.size(gC)` 依賴執行期數值，DSL 會轉成 GPU 上的條件分支，不是 Python 層的分支。
4. 型別標註 `: cute.Tensor` 是必要的，DSL 靠它判斷參數類型。

> [!warning] 注意
> 此範例依官方文件寫法整理，**尚未實際在 GPU 上執行驗證**。beta 階段 API 可能變動，遇到錯誤請對照所安裝版本的文件。

## 5. 除錯技巧

- 用 `cute.printf` 在 kernel 內印值。
- 先用小的 N（例如 64）驗證正確性，再放大。
- `cute.compile` 可加選項，例如 `cute.GenerateLineInfo`（產生行號資訊）、`cute.OptimizationLevel(3)`（最佳化等級）。

## 6. 後續學習方向

- [[CuTe_Layout]] 與 Tensor 基礎（CuTe 最核心概念）
- tiled copy 改寫向量加法（向量化存取，提升頻寬利用率）
- tiled MMA 與 GEMM 入門範例
- 與自己研究的 [[Python DSL 產生 CUDA kernel]]（NVRTC + Driver API）做設計比較：CuTe DSL 走 MLIR + ptxas，兩者的 IR 與編譯路徑不同

## 7. 參考資料

- [[Python_DSL]]
- CUTLASS DSL Quick Start Guide（docs.nvidia.com）
- CUTLASS DSL Overview（docs.nvidia.com/cutlass/media/docs/pythonDSL/overview.html）
- PyPI：`nvidia-cutlass-dsl`
- GitHub：NVIDIA/cutlass
