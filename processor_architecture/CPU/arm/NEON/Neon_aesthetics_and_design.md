---
title: "Neon aesthetics and design"
source: "https://claude.ai/chat/72eafe3f-8a3b-447c-8dc4-21e3e1e9e87e"
author:
published:
created: 2026-07-02
description: "Claude conversation with 2 messages"
tags:
  - "clippings"
---
## ARM NEON 簡介

ARM NEON 是 ARM 架構下的 SIMD（單指令多資料，Single Instruction Multiple Data）擴充指令集，用來加速平行運算，常見於影像處理、音訊處理、機器學習推論等場景。

### 核心概念

- **暫存器** ：NEON 有 32 個 128-bit 向量暫存器（ `Q0` – `Q31` ），也可視為 64 個 64-bit（ `D0` – `D31` ）
- **資料型態** ：可將一個暫存器切成多筆較小的資料同時運算，例如一次處理 16 個 `int8_t` 、8 個 `int16_t` 、4 個 `int32_t` 或 4 個 `float32_t`
- **使用方式** ：
	1. **Intrinsics** （最常用）：透過 `<arm_neon.h>` 提供的 C/C++ 函式呼叫，編譯器會轉成對應組合語言
		2. **內嵌組合語言** ：直接寫 NEON 組語，控制力最高但較難維護
		3. **自動向量化** ：讓編譯器（如 GCC/Clang 開 `-O3` 加 `-mfpu=neon` 或 AArch64 預設支援）自動轉換迴圈

### 簡單範例（Intrinsics，兩個 float 陣列相加）

```c
#include <arm_neon.h>

void add_arrays(const float *a, const float *b, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vaddq_f32(va, vb);
        vst1q_f32(out + i, vc);
    }
    // 處理剩餘不足 4 個的元素
    for (; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}
```

編譯（AArch64 通常內建支援 NEON，不需額外旗標；32-bit ARM 需要 `-mfpu=neon` ）：

```bash
gcc -O2 -march=armv8-a add.c -o add
```

C++ 不使用標準函式庫，這段程式碼本身沒有依賴 `<vector>` 、 `<algorithm>` 之類的東西，純粹是 C 風格陣列加上 NEON intrinsics。
