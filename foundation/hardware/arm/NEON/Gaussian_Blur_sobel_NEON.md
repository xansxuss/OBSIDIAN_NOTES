---
title: Gaussian_Blur
source: https://claude.ai/chat/72eafe3f-8a3b-447c-8dc4-21e3e1e9e87e
author:
published:
created: 2026-07-02
description: Claude conversation with 6 messages
tags:
  - clippings
---
## 完整版 3x3 高斯模糊（含邊界處理）

### 高斯模糊

3x3 高斯核心常用權重：

```
1 2 1
2 4 2
1 2 1
```

總和為 16，所以最後要除以 16。

```c
#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>  // 只用 malloc/free，非 STL

// 邊界處理：複製邊緣像素 (replicate / clamp border)
static inline int clamp_idx(int idx, int max_idx) {
    if (idx < 0) return 0;
    if (idx > max_idx) return max_idx;
    return idx;
}

void gaussian_blur3x3_neon(const uint8_t *src, uint8_t *dst, int width, int height) {
    // 先算好每一列（row）的水平方向 1-2-1 加權和，存成 int16_t 暫存陣列
    int16_t *temp = (int16_t*)malloc(sizeof(int16_t) * width * height);

    // Step 1：水平方向 1-2-1 卷積
    for (int y = 0; y < height; y++) {
        const uint8_t *row = src + y * width;
        int16_t *trow = temp + y * width;

        int x = 1;
        for (; x + 8 <= width - 1; x += 8) {
            uint8x8_t left   = vld1_u8(row + x - 1);
            uint8x8_t center = vld1_u8(row + x);
            uint8x8_t right  = vld1_u8(row + x + 1);

            uint16x8_t sum = vaddl_u8(left, right);       // left + right
            uint16x8_t center16 = vmovl_u8(center);
            sum = vmlaq_n_u16(sum, center16, 2);           // + center*2

            vst1q_s16(trow + x, vreinterpretq_s16_u16(sum));
        }
        // 收尾（含邊界）
        for (; x < width - 1; x++) {
            trow[x] = row[x-1] + 2*row[x] + row[x+1];
        }
        // 左右邊界像素：直接複製鄰居處理
        trow[0] = row[0] + 2*row[0] + row[1 < width ? 1 : 0];
        trow[width-1] = row[width-2 >= 0 ? width-2 : 0] + 2*row[width-1] + row[width-1];
    }

    // Step 2：垂直方向 1-2-1 卷積 + 除以 16
    for (int y = 0; y < height; y++) {
        int y0 = clamp_idx(y - 1, height - 1);
        int y1 = y;
        int y2 = clamp_idx(y + 1, height - 1);

        const int16_t *r0 = temp + y0 * width;
        const int16_t *r1 = temp + y1 * width;
        const int16_t *r2 = temp + y2 * width;
        uint8_t *drow = dst + y * width;

        int x = 0;
        for (; x + 8 <= width; x += 8) {
            int16x8_t v0 = vld1q_s16(r0 + x);
            int16x8_t v1 = vld1q_s16(r1 + x);
            int16x8_t v2 = vld1q_s16(r2 + x);

            int16x8_t sum = vaddq_s16(v0, v2);
            sum = vmlaq_n_s16(sum, v1, 2);

            // 除以 16 並飽和轉回 uint8_t
            uint8x8_t result = vqshrun_n_s16(sum, 4); // shift right 4 = /16，並飽和
            vst1_u8(drow + x, result);
        }
        for (; x < width; x++) {
            int32_t v = r0[x] + 2*r1[x] + r2[x];
            v >>= 4;
            drow[x] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }

    free(temp);
}
```

**重點技巧** ： `vqshrun_n_s16` 一次做完「右移（除以 16）+ 有號轉無號 + 飽和」，是這段程式碼的核心關鍵字。

---

## Sobel 邊緣偵測

Sobel 用兩個方向的核心：

```
Gx:  -1 0 1      Gy:  -1 -2 -1
     -2 0 2            0  0  0
     -1 0 1            1  2  1
```

邊緣強度： `magnitude = sqrt(Gx² + Gy²)` ，實務上常用 `|Gx| + |Gy|` 近似，省掉開根號。

```c
void sobel_neon(const uint8_t *src, uint8_t *dst, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
        const uint8_t *row_prev = src + (y - 1) * width;
        const uint8_t *row_cur  = src + y * width;
        const uint8_t *row_next = src + (y + 1) * width;
        uint8_t *drow = dst + y * width;

        int x = 1;
        for (; x + 8 <= width - 1; x += 8) {
            // 讀取 3x3 鄰域所需的九組資料（左中右各一組 8 個像素）
            uint8x8_t p00 = vld1_u8(row_prev + x - 1);
            uint8x8_t p02 = vld1_u8(row_prev + x + 1);
            uint8x8_t p10 = vld1_u8(row_cur  + x - 1);
            uint8x8_t p12 = vld1_u8(row_cur  + x + 1);
            uint8x8_t p20 = vld1_u8(row_next + x - 1);
            uint8x8_t p22 = vld1_u8(row_next + x + 1);

            uint8x8_t p01 = vld1_u8(row_prev + x);
            uint8x8_t p21 = vld1_u8(row_next + x);

            // Gx = (p02 + 2*p12 + p22) - (p00 + 2*p10 + p20)
            int16x8_t gx_pos = vreinterpretq_s16_u16(
                vmlal_n_u8(vaddl_u8(p02, p22), p12, 2));
            int16x8_t gx_neg = vreinterpretq_s16_u16(
                vmlal_n_u8(vaddl_u8(p00, p20), p10, 2));
            int16x8_t gx = vsubq_s16(gx_pos, gx_neg);

            // Gy = (p20 + 2*p21 + p22) - (p00 + 2*p01 + p02)
            int16x8_t gy_pos = vreinterpretq_s16_u16(
                vmlal_n_u8(vaddl_u8(p20, p22), p21, 2));
            int16x8_t gy_neg = vreinterpretq_s16_u16(
                vmlal_n_u8(vaddl_u8(p00, p02), p01, 2));
            int16x8_t gy = vsubq_s16(gy_pos, gy_neg);

            // 近似 magnitude = |Gx| + |Gy|
            int16x8_t abs_gx = vabsq_s16(gx);
            int16x8_t abs_gy = vabsq_s16(gy);
            int16x8_t mag = vaddq_s16(abs_gx, abs_gy);

            uint8x8_t result = vqmovun_s16(mag); // 飽和轉 uint8_t
            vst1_u8(drow + x, result);
        }
        for (; x < width - 1; x++) {
            int gx = (row_prev[x+1] + 2*row_cur[x+1] + row_next[x+1])
                    - (row_prev[x-1] + 2*row_cur[x-1] + row_next[x-1]);
            int gy = (row_next[x-1] + 2*row_next[x] + row_next[x+1])
                    - (row_prev[x-1] + 2*row_prev[x] + row_prev[x+1]);
            int mag = abs(gx) + abs(gy);
            drow[x] = (uint8_t)(mag > 255 ? 255 : mag);
        }
    }
    // 最外圍一圈邊界：實務上通常填 0 或複製鄰近列
    for (int x = 0; x < width; x++) {
        dst[x] = 0;
        dst[(height-1)*width + x] = 0;
    }
    for (int y = 0; y < height; y++) {
        dst[y*width] = 0;
        dst[y*width + width-1] = 0;
    }
}
```

---

## 英文技術用語糾正

你這兩個詞其實用法都正確，我幫你補充精準度跟常見搭配：

**de-interleave** （動詞）

- 你的用法沒問題。這詞常用在「解交錯」資料排列，例如影像的 RGB packed → planar 轉換
- 常見搭配： `de-interleave the channels` 、 `de-interleaved data`
- 反義詞是 `interleave` （交錯排列），例如 `vld3` 是 de-interleave load， `vst3` 是 interleave store
- 小提醒：作名詞時要寫 `de-interleaving` （動名詞），不是 `de-interleave` 本身

**saturate** （動詞）

- 你的用法正確。在數位訊號/影像處理領域， `saturate` 專指「數值超過範圍時鎖在最大/最小值，而非溢位環繞」
- 形容詞是 `saturated` ：例如 `saturated arithmetic` （飽和運算）、 `the pixel value is saturated at 255`
- 名詞是 `saturation` ，但要小心：在色彩學裡 `saturation` 指「飽和度」（色彩鮮豔程度），跟這裡「數值飽和」是同一個字但完全不同意思，上下文要分清楚，否則容易搞混聽者
- 反義概念是 `overflow` （溢位）或 `wraparound` （環繞），你程式碼裡常會需要對比這兩者：
	- `wrapping addition` （環繞加法）→ 255 + 1 = 0
		- `saturating addition` （飽和加法）→ 255 + 1 = 255
