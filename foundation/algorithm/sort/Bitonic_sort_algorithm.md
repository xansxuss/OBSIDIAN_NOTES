---
title: "Bitonic sort algorithm"
source: "https://claude.ai/chat/11de18e5-5737-4a0c-9a55-310fcd8dd62e"
author:
published:
created: 2026-07-07
description: "Claude conversation with 8 messages"
tags:
  - "clippings"
---
雙調排序（Bitonic Sort）是一種平行運算導向的排序演算法，特別適合在 GPU 或 SIMD 架構上實作，因為每一階段的比較都是固定樣式（oblivious），不需要依資料內容決定分支。

### 基本概念

1. **雙調序列（Bitonic Sequence）** ：先遞增後遞減（或反過來）的序列，例如 `1, 4, 7, 9, 8, 3, 2` 。
2. **雙調合併（Bitonic Merge）** ：把一個雙調序列切成兩半，兩兩比較後交換，就能得到兩個較短的雙調序列，遞迴下去最終得到排序好的序列。
3. **建構過程** ：從長度 2 開始，先做出小的雙調序列，再逐步合併成更大的雙調序列，直到整個陣列排序完成。
4. 演算法要求陣列長度為 **2 的冪次** ，若不是則通常補上無窮大（或無窮小）當作 padding。

時間複雜度是 O(n log²n)，比起 O(n log n) 的快速排序理論上較差，但因為比較樣式固定，非常適合平行硬體加速。

### C++ 實作（不使用標準函式庫）

以下版本只用原生陣列與手寫函式，沒有 `<algorithm>` 、 `<vector>` 等標準容器或演算法：

```cpp
#include <iostream>

// 手寫交換函式，取代 std::swap
void mySwap(int &a, int &b) {
    int tmp = a;
    a = b;
    b = tmp;
}

// 比較並依方向決定是否交換
// ascending 為 true 表示希望這段最後是遞增排列
void compAndSwap(int arr[], int i, int j, bool ascending) {
    if ((arr[i] > arr[j]) == ascending) {
        mySwap(arr[i], arr[j]);
    }
}

// 遞迴合併雙調序列
void bitonicMerge(int arr[], int low, int cnt, bool ascending) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++) {
            compAndSwap(arr, i, i + k, ascending);
        }
        bitonicMerge(arr, low, k, ascending);
        bitonicMerge(arr, low + k, k, ascending);
    }
}

// 遞迴建構並排序
void bitonicSort(int arr[], int low, int cnt, bool ascending) {
    if (cnt > 1) {
        int k = cnt / 2;
        // 前半段做遞增
        bitonicSort(arr, low, k, true);
        // 後半段做遞減，兩者合起來才是雙調序列
        bitonicSort(arr, low + k, k, false);
        // 合併整段
        bitonicMerge(arr, low, cnt, ascending);
    }
}

int main() {
    int arr[] = {3, 7, 4, 8, 6, 2, 1, 5}; // 長度必須是 2 的冪次
    int n = 8;

    bitonicSort(arr, 0, n, true); // true = 由小到大排序

    for (int i = 0; i < n; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << "\n";

    return 0;
}
```

**注意事項：**

- 這裡只用 `<iostream>` 做輸出，排序核心邏輯完全沒有依賴標準函式庫的容器或演算法。
- 陣列長度 `n` 必須是 2 的冪次（4、8、16、32...），否則邏輯會出錯，若你的資料長度不固定，需要額外處理 padding，我可以再幫你補上。
- 如果你想要完全平行化（例如用 pthread 或 OpenMP 展示每一階段的並行比較），我也可以幫你改寫成多執行緒版本。


### 在 CPU SIMD實現？

在 CPU 上要用 SIMD 實作雙調排序，核心概念是： **利用向量暫存器一次處理多筆資料的比較與交換** ，用 `min` / `max` 向量指令取代分支判斷，再搭配「排列（permute/shuffle）」指令把資料換到正確的位置。這裡示範用 AVX2（256-bit 暫存器，一次可放 8 個 int32）對 8 個元素做完整的雙調排序。

### 核心原理

原本 `compAndSwap` 是「比較後決定要不要交換」，在 SIMD 版本裡改成：

1. 用 `permutevar8x32` 把每個 lane 換到它要比較的「夥伴」位置，取得 `partner` 向量。
2. 同時算出 `vmin = min(v, partner)` 和 `vmax = max(v, partner)` 。
3. 依照這個 stage 預先算好的方向遮罩（哪些 lane 該取 min、哪些該取 max），用 `blendv` 選出結果。

因為每個 stage 的排列樣式和方向遮罩都是 **固定常數** ，可以事先算好，執行時完全沒有分支，非常適合 SIMD。

### C++ + AVX2 實作

```cpp
#include <immintrin.h>
#include <iostream>

// T/F 代表 blend 用的全 1 / 全 0 遮罩（int32 版本）
constexpr int T = -1;   // 0xFFFFFFFF
constexpr int F = 0;

// 對放在一個 __m256i 裡的 8 個 int32 做完整雙調排序（結果為遞增）
__m256i bitonicSort8(__m256i v) {
    __m256i idx, mask, partner, vmin, vmax;

    // Stage A：距離1，建構長度2的雙調區塊（asc,desc,asc,desc）
    idx  = _mm256_setr_epi32(1,0,3,2,5,4,7,6);
    mask = _mm256_setr_epi32(T,F,F,T,T,F,F,T);
    partner = _mm256_permutevar8x32_epi32(v, idx);
    vmin = _mm256_min_epi32(v, partner);
    vmax = _mm256_max_epi32(v, partner);
    v = _mm256_blendv_epi8(vmax, vmin, mask);

    // Stage B：距離2，合併成長度4的雙調區塊（block0 asc, block1 desc）
    idx  = _mm256_setr_epi32(2,3,0,1,6,7,4,5);
    mask = _mm256_setr_epi32(T,T,F,F,F,F,T,T);
    partner = _mm256_permutevar8x32_epi32(v, idx);
    vmin = _mm256_min_epi32(v, partner);
    vmax = _mm256_max_epi32(v, partner);
    v = _mm256_blendv_epi8(vmax, vmin, mask);

    // Stage C：距離1，收尾成兩個排好序的長度4區塊
    idx  = _mm256_setr_epi32(1,0,3,2,5,4,7,6);
    mask = _mm256_setr_epi32(T,F,T,F,F,T,F,T);
    partner = _mm256_permutevar8x32_epi32(v, idx);
    vmin = _mm256_min_epi32(v, partner);
    vmax = _mm256_max_epi32(v, partner);
    v = _mm256_blendv_epi8(vmax, vmin, mask);

    // Stage D：距離4，最終合併（全部遞增）
    idx  = _mm256_setr_epi32(4,5,6,7,0,1,2,3);
    mask = _mm256_setr_epi32(T,T,T,T,F,F,F,F);
    partner = _mm256_permutevar8x32_epi32(v, idx);
    vmin = _mm256_min_epi32(v, partner);
    vmax = _mm256_max_epi32(v, partner);
    v = _mm256_blendv_epi8(vmax, vmin, mask);

    // Stage E：距離2
    idx  = _mm256_setr_epi32(2,3,0,1,6,7,4,5);
    mask = _mm256_setr_epi32(T,T,F,F,T,T,F,F);
    partner = _mm256_permutevar8x32_epi32(v, idx);
    vmin = _mm256_min_epi32(v, partner);
    vmax = _mm256_max_epi32(v, partner);
    v = _mm256_blendv_epi8(vmax, vmin, mask);

    // Stage F：距離1，最終排序完成
    idx  = _mm256_setr_epi32(1,0,3,2,5,4,7,6);
    mask = _mm256_setr_epi32(T,F,T,F,T,F,T,F);
    partner = _mm256_permutevar8x32_epi32(v, idx);
    vmin = _mm256_min_epi32(v, partner);
    vmax = _mm256_max_epi32(v, partner);
    v = _mm256_blendv_epi8(vmax, vmin, mask);

    return v;
}

int main() {
    alignas(32) int data[8] = {5, 3, 8, 1, 9, 2, 7, 4};

    __m256i v = _mm256_load_si256(reinterpret_cast<__m256i*>(data));
    v = bitonicSort8(v);
    _mm256_store_si256(reinterpret_cast<__m256i*>(data), v);

    for (int i = 0; i < 8; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << "\n";

    return 0;
}
```

編譯時記得加上 `-mavx2` ：

```
g++ -O2 -mavx2 bitonic_simd.cpp -o bitonic_simd
```

### 說明

- **完全沒有分支** ：每個 stage 都是固定的 `permute → min/max → blend` ，這正是 SIMD 硬體最擅長的模式。
- **6 個 stage** 對應純量版本裡遞迴展開後的 6 次「比較層」（log₂8 × (log₂8+1) / 2 = 6）。
- 這裡只排序放在單一暫存器內的 8 筆資料。若陣列比 8 大很多，實務做法是：
	- **跨暫存器的長距離比較** ：用兩個暫存器直接做 `min` / `max` （例如 `vmin = min(regA, regB)`, `vmax = max(regA, regB)` ，兩個結果分別存回對應位置），處理距離較大的 stage。
		- **暫存器內的短距離比較** ：用像上面這樣的 permute + blend 處理距離較小的 stage。
		- 兩者交錯使用，就能把整個大陣列排序完成。

## 加入到NMS演算法裡？

把雙調排序用在 NMS（Non-Maximum Suppression）裡，最實際的做法分兩塊：

1. **排序階段** ：用雙調排序把偵測框依信賴分數由高到低排序。
2. **抑制階段** ：主迴圈（誰壓制誰）本質上是序列相依（sequential dependency），沒辦法完全平行化，但 **每一輪計算 IoU 的內層迴圈** 可以用 AVX2 一次算 8 組 IoU，這才是 SIMD 真正發揮效果的地方。

## 3. NMS 演算法的完整管線（Pipeline）

將 Bitonic Sort 整合進 NMS 後，高效能 CPU NMS 的完整流程如下：

**1. Score Threshold 過濾與補齊：**時間複雜度: O(N)。

遍歷所有 BBox，剔除分數低於閾值（如 0.25）的框。統計剩餘數量 $M$，並在陣列末端填補 `-FLT_MAX` 到最接近的 2 的冪次方（如 128 或 256）。

**2. SIMD Bitonic 排序：**時間複雜度: O(M log^2 M)。

使用上述的 Key-Value 同步 Bitonic Sort，對篩選後的 Score 與 Index 進行降序排序。排序完成後，Index 陣列的第一個元素即為最高分 BBox 的索引。

**3. SIMD IoU 計算與抑制（Suppression）：**時間複雜度: 最差 O(M^2)。

順序取出最高分的 BBox，並利用 SIMD（AVX2 可以同時計算 8 個 BBox 的 $IoU$）計算它與後續所有 BBox 的重疊面積。若 $IoU > 閾值$，則在 Mask 陣列中將該後續 BBox 標記為「已抑制」。

## 效能對比與實務收益

在邊緣運算裝置（Edge AI）或 CPU 推理伺服器上，引入 SIMD 優化後的 NMS 效能提升非常顯著：

| **階段**     | **傳統作法 (Scalar)**            | **AVX2 SIMD 加速作法**       | **效能優勢原因**      |
| ---------- | ---------------------------- | ------------------------ | --------------- |
| **排序階段**   | `std::sort` 頻繁觸發 CPU 分支預測失敗  | Bitonic Sort 無分支，流水線全速前進 | 記憶體連續存取 + 暫存器洗牌 |
| **IoU 計算** | 逐個框計算 `Intersection / Union` | 1 個指令同時算 8 個框的 $IoU$     | 向量化除法與乘法        |

### 資料結構與 IoU 計算

```cpp
#include <immintrin.h>
#include <iostream>
#include <cmath>

struct Box {
    float x1, y1, x2, y2, score;
    int idx;
};

// 純量版 IoU，debug 或驗證用
float iouScalar(const Box &a, const Box &b) {
    float ix1 = a.x1 > b.x1 ? a.x1 : b.x1;
    float iy1 = a.y1 > b.y1 ? a.y1 : b.y1;
    float ix2 = a.x2 < b.x2 ? a.x2 : b.x2;
    float iy2 = a.y2 < b.y2 ? a.y2 : b.y2;

    float iw = ix2 - ix1 > 0 ? ix2 - ix1 : 0;
    float ih = iy2 - iy1 > 0 ? iy2 - iy1 : 0;
    float inter = iw * ih;

    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);

    return inter / (areaA + areaB - inter + 1e-6f);
}
```

### 雙調排序：依 score 由大到小排序（處理任意長度）

雙調排序要求長度是 2 的冪次，所以用 `score = -無限大` 的假框當 padding，排序完再忽略它們。

```cpp
void mySwap(Box &a, Box &b) {
    Box tmp = a;
    a = b;
    b = tmp;
}

// ascending=false 表示希望這段排列成「score 遞減」
void compAndSwap(Box arr[], int i, int j, bool ascending) {
    bool shouldSwapOrder = (arr[i].score < arr[j].score); // score 越大越前面
    if (shouldSwapOrder == ascending) {
        mySwap(arr[i], arr[j]);
    }
}

void bitonicMerge(Box arr[], int low, int cnt, bool ascending) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++) {
            compAndSwap(arr, i, i + k, ascending);
        }
        bitonicMerge(arr, low, k, ascending);
        bitonicMerge(arr, low + k, k, ascending);
    }
}

void bitonicSort(Box arr[], int low, int cnt, bool ascending) {
    if (cnt > 1) {
        int k = cnt / 2;
        bitonicSort(arr, low, k, true);
        bitonicSort(arr, low + k, k, false);
        bitonicMerge(arr, low, cnt, ascending);
    }
}

int nextPow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}
```

### NMS 主體：排序 + SIMD 批次 IoU 抑制

```cpp
// 回傳保留下來的框數量，keepIdx 存放保留框在原始 boxes[] 的 idx
int nmsWithBitonicSort(Box boxes[], int n, float iouThreshold, int keepIdx[]) {
    int padded = nextPow2(n);
    Box *arr = new Box[padded];

    for (int i = 0; i < n; i++) arr[i] = boxes[i];
    for (int i = n; i < padded; i++) {
        arr[i].score = -1e30f; // 極小值 padding，排序後自然沉到最後
        arr[i].x1 = arr[i].y1 = arr[i].x2 = arr[i].y2 = 0;
        arr[i].idx = -1;
    }

    // 1. 用雙調排序依 score 由大到小排列
    bitonicSort(arr, 0, padded, false);

    bool *suppressed = new bool[n];
    for (int i = 0; i < n; i++) suppressed[i] = false;

    int keepCount = 0;

    // 2. 序列相依的抑制主迴圈
    for (int i = 0; i < n; i++) {
        if (suppressed[i]) continue;

        Box &cur = arr[i];
        keepIdx[keepCount++] = cur.idx;

        // 廣播目前這個框的座標到 AVX 暫存器
        __m256 cx1 = _mm256_set1_ps(cur.x1);
        __m256 cy1 = _mm256_set1_ps(cur.y1);
        __m256 cx2 = _mm256_set1_ps(cur.x2);
        __m256 cy2 = _mm256_set1_ps(cur.y2);
        __m256 cArea = _mm256_set1_ps((cur.x2 - cur.x1) * (cur.y2 - cur.y1));
        __m256 thresh = _mm256_set1_ps(iouThreshold);

        int j = i + 1;
        // 每次處理 8 個候選框
        for (; j + 8 <= n; j += 8) {
            float bx1[8], by1[8], bx2[8], by2[8], barea[8];
            for (int k = 0; k < 8; k++) {
                Box &b = arr[j + k];
                bx1[k] = b.x1; by1[k] = b.y1; bx2[k] = b.x2; by2[k] = b.y2;
                barea[k] = (b.x2 - b.x1) * (b.y2 - b.y1);
            }

            __m256 vbx1 = _mm256_loadu_ps(bx1);
            __m256 vby1 = _mm256_loadu_ps(by1);
            __m256 vbx2 = _mm256_loadu_ps(bx2);
            __m256 vby2 = _mm256_loadu_ps(by2);
            __m256 vbarea = _mm256_loadu_ps(barea);

            __m256 ix1 = _mm256_max_ps(cx1, vbx1);
            __m256 iy1 = _mm256_max_ps(cy1, vby1);
            __m256 ix2 = _mm256_min_ps(cx2, vbx2);
            __m256 iy2 = _mm256_min_ps(cy2, vby2);

            __m256 zero = _mm256_set1_ps(0.0f);
            __m256 iw = _mm256_max_ps(_mm256_sub_ps(ix2, ix1), zero);
            __m256 ih = _mm256_max_ps(_mm256_sub_ps(iy2, iy1), zero);
            __m256 inter = _mm256_mul_ps(iw, ih);

            __m256 unionArea = _mm256_sub_ps(_mm256_add_ps(cArea, vbarea), inter);
            __m256 iou = _mm256_div_ps(inter, unionArea);

            __m256 mask = _mm256_cmp_ps(iou, thresh, _CMP_GT_OQ); // iou > threshold
            int bits = _mm256_movemask_ps(mask);

            for (int k = 0; k < 8; k++) {
                if (bits & (1 << k)) suppressed[j + k] = true;
            }
        }

        // 剩下不足 8 個的尾巴用純量處理
        for (; j < n; j++) {
            if (!suppressed[j] && iouScalar(cur, arr[j]) > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }

    delete[] arr;
    delete[] suppressed;
    return keepCount;
}
```

### 測試

```cpp
int main() {
    Box boxes[6] = {
        {10, 10, 50, 50, 0.9f, 0},
        {12, 12, 48, 48, 0.85f, 1},   // 與框0高度重疊，會被抑制
        {100, 100, 150, 150, 0.8f, 2},
        {105, 105, 148, 148, 0.75f, 3}, // 與框2重疊，會被抑制
        {200, 200, 220, 220, 0.6f, 4},
        {11, 11, 49, 49, 0.55f, 5}     // 也與框0重疊
    };

    int keep[6];
    int n = nmsWithBitonicSort(boxes, 6, 0.5f, keep);

    std::cout << "保留框數量: " << n << "\n";
    for (int i = 0; i < n; i++) {
        std::cout << "idx=" << keep[i] << "\n";
    }
    return 0;
}
```

編譯：

```
g++ -O2 -mavx2 nms_bitonic.cpp -o nms_bitonic
```

### 幾個實務上需要注意的點

- **抑制迴圈無法完全平行化** ：因為第 i 輪要看前面哪些框還沒被抑制，這是資料相依（data dependency），雙調排序只能加速「排序」本身，抑制邏輯本質上仍是序列的。
- **這裡示範用 AVX2 一次比對 8 個候選框的 IoU** ，這是真正能省時間的部分，尤其框數很多時效果明顯。
- 若框的數量非常龐大（例如上萬個），建議額外做\*\*空間分群（spatial binning）\*\*先排除明顯不重疊的框，再進 NMS，這樣可以減少 IoU 計算次數。


## 跨暫存器大陣列雙調排序

### 設計概念

整個陣列切成多個 AVX2 暫存器（每個放 8 個 `int32`），依照 Batcher's Bitonic Sort 的標準迭代公式（用 XOR 決定比較夥伴）：

```
for k = 2; k <= n; k <<= 1        // 逐步建構更大的雙調序列
  for j = k>>1; j > 0; j >>= 1    // 合併步驟，距離從 k/2 遞減到 1
```

關鍵拆分成兩種情況：

- **`j >= 8`（跨暫存器）**：因為 `j` 是 8 的倍數，一定是「整個暫存器對整個暫存器」比較，方向對整個暫存器一致，直接對兩個暫存器做 `min`/`max` 交換，**完全不需要 permute**，速度最快。
- **`j < 8`（暫存器內）**：夥伴在同一個暫存器的不同 lane，要用 `permutevar8x32` 換位，再依方向決定每個 lane 要 `min` 還是 `max`（用 `blendv` 選擇）。

### 完整程式碼

cpp

```cpp
#include <immintrin.h>
#include <climits>

// ---------- 跨暫存器（j >= 8）比較交換 ----------
// 兩個要比較的元素分屬不同暫存器，且整個暫存器方向一致
void exchangeCrossRegister(__m256i regs[], int nRegs, int j, int k) {
    int regStride = j / 8;
    for (int r = 0; r < nRegs; r++) {
        int partner = r ^ regStride;
        if (partner <= r) continue; // 每組只處理一次

        bool ascending = (((r * 8) & k) == 0);

        __m256i a = regs[r];
        __m256i b = regs[partner];
        __m256i vmin = _mm256_min_epi32(a, b);
        __m256i vmax = _mm256_max_epi32(a, b);

        if (ascending) {
            regs[r] = vmin;
            regs[partner] = vmax;
        } else {
            regs[r] = vmax;
            regs[partner] = vmin;
        }
    }
}

// ---------- 暫存器內（j < 8）比較交換 ----------
void exchangeIntraRegister(__m256i regs[], int nRegs, int j, int k) {
    int idxArr[8];
    bool isLower[8];
    for (int lane = 0; lane < 8; lane++) {
        idxArr[lane] = lane ^ j;
        isLower[lane] = ((lane & j) == 0);
    }
    __m256i permIdx = _mm256_setr_epi32(
        idxArr[0], idxArr[1], idxArr[2], idxArr[3],
        idxArr[4], idxArr[5], idxArr[6], idxArr[7]);

    bool kSmall = (k < 8);
    bool ascLaneFixed[8];
    if (kSmall) {
        for (int lane = 0; lane < 8; lane++)
            ascLaneFixed[lane] = ((lane & k) == 0);
    }

    for (int r = 0; r < nRegs; r++) {
        __m256i v = regs[r];
        __m256i partner = _mm256_permutevar8x32_epi32(v, permIdx);
        __m256i vmin = _mm256_min_epi32(v, partner);
        __m256i vmax = _mm256_max_epi32(v, partner);

        int maskArr[8];
        if (kSmall) {
            // k < 8：方向會隨 lane 位置變化，是固定不隨暫存器改變的樣式
            for (int lane = 0; lane < 8; lane++)
                maskArr[lane] = (isLower[lane] == ascLaneFixed[lane]) ? -1 : 0;
        } else {
            // k >= 8：整個暫存器方向一致，依此翻轉固定樣式
            bool regAsc = (((r * 8) & k) == 0);
            for (int lane = 0; lane < 8; lane++) {
                bool takeMin = regAsc ? isLower[lane] : !isLower[lane];
                maskArr[lane] = takeMin ? -1 : 0;
            }
        }
        __m256i blendMask = _mm256_setr_epi32(
            maskArr[0], maskArr[1], maskArr[2], maskArr[3],
            maskArr[4], maskArr[5], maskArr[6], maskArr[7]);

        regs[r] = _mm256_blendv_epi8(vmax, vmin, blendMask);
    }
}

// n 必須是 2 的冪次，且是 8 的倍數（n >= 8）
void bitonicSortLargeAVX2(int *data, int n) {
    int nRegs = n / 8;
    __m256i *regs = new __m256i[nRegs];
    for (int r = 0; r < nRegs; r++)
        regs[r] = _mm256_loadu_si256((const __m256i*)(data + r * 8));

    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            if (j >= 8) exchangeCrossRegister(regs, nRegs, j, k);
            else exchangeIntraRegister(regs, nRegs, j, k);
        }
    }

    for (int r = 0; r < nRegs; r++)
        _mm256_storeu_si256((__m256i*)(data + r * 8), regs[r]);
    delete[] regs;
}

int nextPow2(int n) {
    int p = 8;
    while (p < n) p <<= 1;
    return p;
}
```

使用方式（記得非 2 冪次/非 8 倍數要補 `INT_MAX` padding）：

cpp

```cpp
int n = 1000;
int padded = nextPow2(n); // 補到 1024
int *arr = new int[padded];
// ... 填入資料 arr[0..n-1]，arr[n..padded-1] 填 INT_MAX
bitonicSortLargeAVX2(arr, padded);
// arr 現在是遞增排序（padding 值會沉到最後面）
```

編譯：`g++ -O2 -mavx2 file.cpp -o file`

### 驗證結果

我用插入排序當對照組，對 8、16、32、64、128、256、1000、4096 各種長度各跑 20 次隨機測試，全部一致：

```
size 8 tested
size 16 tested
...
size 4096 tested
ALL PASSED
```

### 效能重點

- **跨暫存器比較**（`j>=8`）完全沒有 permute，只有 `min`/`max`，是這個實作裡最便宜的部分，佔了大陣列排序中大多數的 stage 數（`log₂(n) × (log₂(n)+1)/2` 個 stage 裡，隨著 `n` 變大，跨暫存器 stage 佔比越來越高）。
- **暫存器內比較**（`j<8`）只發生在最後 3 個合併步驟（`j=4,2,1`），跟陣列大小無關，是固定成本。
- 若要接回上一輪的 NMS，只要把 `nmsWithBitonicSort` 裡呼叫的 `bitonicSort`（遞迴純量版）換成 `bitonicSortLargeAVX2`，並把 `Box` 拆成 SoA（score 陣列額外排序、或排序 index 陣列再重排 Box），就能無縫接上。

**這裡有一個概念上的細節，也是許多 C/C++ 工程師（甚至是資深工程師）有時會不小心誤解的地方：**

在 C/C++ 的 `<climits>` 與 `<cfloat>` 中，**`-FLT_MAX` 的確是代表「值最小的負數」**，但在處理 **SIMD 比較指令**時，通常更推薦使用 **`-INFINITY`（負無窮大）**，或者需要特別注意它的底層二進位行為。

以下為您釐清這兩個概念在 NMS 補齊（Padding）時的差異：

## 1. 浮點數的 `FLT_MAX` vs `FLT_MIN`

在 IEEE 754 浮點數標準中：

- **`FLT_MAX`**：大約是 3.40282×1038（最大正數）。
    
- **`-FLT_MAX`**：大約是 −3.40282×1038（絕對值最大、但數值最小的**有限負數**）。
    
- **`FLT_MIN`**：大約是 1.17549×10−38。注意！這**不是**負數，它是**大於零的最小正正規化數（Minimum normalized positive value）**。
    

所以在尋找「值最小的有限數」時，您說的 `-FLT_MAX` 在數學語意上完全正確。

## 2. 為什麼在 SIMD NMS 中，用 `-INFINITY` 比 `-FLT_MAX` 安全？

雖然 `-FLT_MAX` 在數學上是最小的有限數，但在實作高效能、無分支（Branchless）的 NMS 補齊時，使用 **`-INFINITY`**（可用 `-(float)INFINITY` 或底層十六進位 `0xFF800000` 表示）通常是更好的選擇，原因有兩個：

### 原因 A：避免無效 BBox 參與 IoU 計算

在 NMS 的第三步，我們會拿「目前最高分」的 BBox 去跟後面所有的 BBox 計算 IoU。 如果我們用 `-FLT_MAX` 來填補那些無效的 BBox：

1. 排序後，這些無效 BBox 的 Score 會被排到最後面。
    
2. 當前面的有效 BBox 依序處理時，NMS 依然會拿有效的 BBox 去跟這些 `-FLT_MAX` 的無效 BBox 計算 IoU。這會導致 CPU 白白浪費心力去算一些根本不存在的框。
    

**更好的做法：** 在過濾階段，直接把無效 BBox 的坐標設為 `NaN`，或是將分數設為 `-INFINITY`，並在 SIMD IoU 遮罩中，利用 `_mm256_cmp_ps` 的 `_CMP_ORD_Q`（檢查是否為有效浮點數）直接把他們排除，連 IoU 的數學公式都省得算。

### 原因 B：硬體比較指令的極端狀況

在 AVX2 中，`_mm256_min_ps` 處理 `-INFINITY` 的速度與普通浮點數一樣快。而且 `-INFINITY` 滿足一個完美的特性：**它小於任何可能的有限浮點數（包括誤差產生的極小負數）**。

如果您的模型在算分數時，因為數值不穩定（Numerical Instability）不小心吐出了包含 `-FLT_MAX` 甚至溢位的錯誤數值，使用 `-INFINITY` 作為 Padding Pad 就能百分之百保證這些無效資料永遠會被壓在最底層，不會干擾真正的資料。

> ### 💡 快速總結
> 
> - 如果您需要的是一個**有限的**最小負數：用 `-FLT_MAX`。
>     
> - 如果您在寫 **NMS 的 SIMD 補齊**：建議直接填 **`-INFINITY`**。這不僅能確保排序結果正確，還能利用浮點數硬體特性，直接在後續的 IoU 計算中快速將其過濾。

**在 CPU SIMD 實作中，「補齊（Padding）到 2 的冪次方」通常有兩種思維：一種是計算出目標長度後，用虛擬資料填滿陣列；另一種是進階的硬體無分支（Branchless）位元運算**。**

在處理如 NMS 篩選後的動態資料量時，我們需要極高效率的演算法來找出「大於或等於當前數量 $M$ 的最小 2 的冪次方」（即 $\ge M$ 的最小 $2^k$）。

## 1. 快速尋找 2 的冪次方（Bitwise Hacks）

在 C++ 中，最快且不依賴標準函式庫的方式是使用**位元運算**，或是現代 CPU（自 Haswell 架構起）硬體支援的 `__builtin_clz`（Count Leading Zeros，計算高位連續零個數）指令。

### 方法 A：純位元移位（通用、無分支）

透過連續的位元或運算（OR）將最高位元（MSB）以下的所有位置全部填滿 1，最後再加 1 就能翻轉成 2 的冪次方。

C++

```
inline uint32_t next_power_of_two(uint32_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}
```

### 方法 B：使用 CPU 內建指令（最快，僅需 1~2 個時脈週期）

利用 `__builtin_clz` 直接由硬體找出最高位元的位置，效率極高：

C++

```
inline uint32_t next_power_of_two_fast(uint32_t n) {
    if (n <= 1) return 1;
    // 32 減去高位零個數，得到最高位元的位置
    return 1_u32 << (32 - __builtin_clz(n - 1));
}
```

## 2. 在 NMS 記憶體佈局中實作補齊

在 NMS 的實務管線中，為了配合 AVX2 的 32-byte 對齊與結構體分離陣列（SoA），我們會預先配置一塊足夠大的記憶體（通常是模型最大 BBox 輸出量，例如 YOLO 的 8400 補齊到 8192 或 16384）。

當過濾完剩餘 $M$ 個有效框時，補齊的具體實作如下：

C++

```
// 假設有效資料已經緊密排列在陣列的前 M 個位置
void pad_to_power_of_two(float* scores, int* indices, uint32_t M) {
    uint32_t N = next_power_of_two_fast(M);
    
    // 如果已經是 2 的冪次方且大於 0，就不需要額外填補
    if (N == M) return;

    // 使用前一題建議的 -INFINITY 作為 Padding 值
    float pad_score = -(float)INFINITY; 
    int pad_index = -1; // 無效索引

    // 實務上這裡也可以用 SIMD 向量化寫入（一次填 8 個）
    for (uint32_t i = M; i < N; i++) {
        scores[i] = pad_score;
        indices[i] = pad_index;
    }
}
```

## 3. 進階：真的需要完全補齊嗎？（避免極端記憶體浪費）

補齊到 2 的冪次方雖然能讓 Bitonic Sort 運作得最完美，但會遇到一個**極端的效能陷阱**：

> **「邊界膨脹」問題：**
> 
> 假設經過 Score Threshold 過濾後，好死不死剩下了 **513** 個有效框。
> 
> 根據演算法，它會被強行補齊到 **1024**。這意味著您必須多花接近一倍的 CPU 時間，去對 511 個無效的 `-INFINITY` 進行排序比較。

### 業界的高效能折衷方案：以 8 為基底（SIMD Lane Alignment）

在 2026 年的主流高效能推理庫中，如果資料量大，我們不再強求將整體數組補齊到巨大的 $2^k$。相反地，我們只需要將資料量補齊到 **8 的倍數**（對齊 AVX2 暫存器大小）。

1. **局部排序：** 每 8 個元素一組，用前面寫好的 `bitonic_sort_8_nms` 暫存器內排序法全部排好。
    
2. **多路合併（Multi-way Merge）：** 接下來不再使用大跨度的 Bitonic 網路，而是改用 SIMD 最佳化的 **Merge Sort（歸併排序）** 網路將這幾個已經排好序的 8 元素區塊合併起來。
    

這樣既能享受 SIMD 暫存器內 Bitonic Sort 無分支的超高速優化，又能避免在資料量剛好跨越邊界時（如 513 變 1024）所帶來的無謂運算浪費。