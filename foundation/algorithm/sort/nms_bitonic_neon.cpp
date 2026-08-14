// nms_bitonic_neon.cpp
//
// ARM64 NEON 加速版本：
//   1) Bitonic sort：依信心分數 (score) 由高到低排序
//   2) NMS (Non-Maximum Suppression)：以 NEON 向量化計算 IoU 做抑制
//
// 設計原則：不使用 C++ 標準函式庫容器/演算法（沒有 vector / algorithm /
// sort），只用最基本的 C 記憶體配置 (malloc/free) 與陣列操作。
//
// 編譯 (AArch64)：
//   aarch64-linux-gnu-gcc -O2 -march=armv8-a -o nms_bitonic_neon nms_bitonic_neon.cpp -lstdc++
//
// 執行環境需求：ARMv8-A (arm64)，NEON 為必備指令集，不需額外開關。

#include <arm_neon.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// ------------------------------------------------------------------
// 資料結構
// ------------------------------------------------------------------
struct Box {
    float x1, y1, x2, y2;   // 左上角 / 右下角座標
    float score;            // 信心分數
};

// ------------------------------------------------------------------
// 工具函式
// ------------------------------------------------------------------
static inline uint32_t next_pow2(uint32_t n) {
    uint32_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// ------------------------------------------------------------------
// Bitonic Sort（降冪，score 由大到小），score/idx 同步搬動
// n 必須是 2 的冪次
//
// 當比較距離 j >= 4 時，同一個 4 元素向量內的排序方向必定相同，
// 因此可以直接用 NEON 對 4 筆資料同時做「比較 + 交換」。
// j < 4 時距離小於向量寬度、會跨越向量邊界，改用純量處理。
// ------------------------------------------------------------------
static void bitonic_sort_desc(float* score, uint32_t* idx, uint32_t n) {
    for (uint32_t k = 2; k <= n; k <<= 1) {
        for (uint32_t j = k >> 1; j > 0; j >>= 1) {

            if (j >= 4) {
                for (uint32_t i = 0; i < n; i += 4) {
                    uint32_t l = i ^ j;
                    if (l < i) continue;              // 每對只處理一次

                    // 因為 j >= 4，這 4 個 i 的 (i & k) 必定一致
                    bool block_even = ((i & k) == 0);

                    float32x4_t vi_s = vld1q_f32(score + i);
                    float32x4_t vl_s = vld1q_f32(score + l);
                    uint32x4_t  vi_i = vld1q_u32(idx + i);
                    uint32x4_t  vl_i = vld1q_u32(idx + l);

                    // 目標是「降冪」：
                    //   block_even  -> 想要 score[i] >= score[l]，反之交換
                    //   !block_even -> 想要 score[i] <= score[l]，反之交換
                    uint32x4_t need_swap = block_even
                        ? vcltq_f32(vi_s, vl_s)   // i < l 時交換
                        : vcgtq_f32(vi_s, vl_s);  // i > l 時交換

                    float32x4_t new_i_s = vbslq_f32(need_swap, vl_s, vi_s);
                    float32x4_t new_l_s = vbslq_f32(need_swap, vi_s, vl_s);
                    uint32x4_t  new_i_i = vbslq_u32(need_swap, vl_i, vi_i);
                    uint32x4_t  new_l_i = vbslq_u32(need_swap, vi_i, vl_i);

                    vst1q_f32(score + i, new_i_s);
                    vst1q_f32(score + l, new_l_s);
                    vst1q_u32(idx + i, new_i_i);
                    vst1q_u32(idx + l, new_l_i);
                }
            } else {
                for (uint32_t i = 0; i < n; ++i) {
                    uint32_t l = i ^ j;
                    if (l <= i) continue;
                    bool block_even = ((i & k) == 0);
                    bool swap_needed = block_even ? (score[i] < score[l])
                                                   : (score[i] > score[l]);
                    if (swap_needed) {
                        float ts = score[i]; score[i] = score[l]; score[l] = ts;
                        uint32_t ti = idx[i]; idx[i] = idx[l]; idx[l] = ti;
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------
// NMS 核心：輸入已依信心分數由高到低排序的 SoA 座標陣列，
// 用 NEON 一次比對 4 個候選框的 IoU，回傳保留框在 sorted 陣列中的索引。
// ------------------------------------------------------------------
static uint32_t nms_neon(const float* x1, const float* y1,
                          const float* x2, const float* y2,
                          const float* area,
                          uint32_t n, float iou_thresh,
                          uint32_t* keep_out) {
    uint8_t* suppressed = (uint8_t*)calloc(n, sizeof(uint8_t));
    uint32_t keep_count = 0;

    const float32x4_t vzero   = vdupq_n_f32(0.0f);
    const float32x4_t veps    = vdupq_n_f32(1e-9f);
    const float32x4_t vthresh = vdupq_n_f32(iou_thresh);

    for (uint32_t i = 0; i < n; ++i) {
        if (suppressed[i]) continue;
        keep_out[keep_count++] = i;

        const float32x4_t vax1 = vdupq_n_f32(x1[i]);
        const float32x4_t vay1 = vdupq_n_f32(y1[i]);
        const float32x4_t vax2 = vdupq_n_f32(x2[i]);
        const float32x4_t vay2 = vdupq_n_f32(y2[i]);
        const float32x4_t vaarea = vdupq_n_f32(area[i]);

        uint32_t j = i + 1;
        for (; j + 4 <= n; j += 4) {
            float32x4_t vbx1 = vld1q_f32(x1 + j);
            float32x4_t vby1 = vld1q_f32(y1 + j);
            float32x4_t vbx2 = vld1q_f32(x2 + j);
            float32x4_t vby2 = vld1q_f32(y2 + j);
            float32x4_t vbarea = vld1q_f32(area + j);

            float32x4_t ix1 = vmaxq_f32(vax1, vbx1);
            float32x4_t iy1 = vmaxq_f32(vay1, vby1);
            float32x4_t ix2 = vminq_f32(vax2, vbx2);
            float32x4_t iy2 = vminq_f32(vay2, vby2);

            float32x4_t iw = vmaxq_f32(vsubq_f32(ix2, ix1), vzero);
            float32x4_t ih = vmaxq_f32(vsubq_f32(iy2, iy1), vzero);
            float32x4_t inter = vmulq_f32(iw, ih);

            float32x4_t uni = vaddq_f32(vsubq_f32(vaddq_f32(vaarea, vbarea), inter), veps);
            float32x4_t iou_v = vdivq_f32(inter, uni);

            uint32x4_t mask = vcgeq_f32(iou_v, vthresh); // IoU >= 門檻就要抑制

            uint32_t m[4];
            vst1q_u32(m, mask);
            for (int t = 0; t < 4; ++t) {
                if (m[t]) suppressed[j + t] = 1;
            }
        }
        // 尾端不足 4 筆，純量處理
        for (; j < n; ++j) {
            if (suppressed[j]) continue;
            float ix1v = x1[i] > x1[j] ? x1[i] : x1[j];
            float iy1v = y1[i] > y1[j] ? y1[i] : y1[j];
            float ix2v = x2[i] < x2[j] ? x2[i] : x2[j];
            float iy2v = y2[i] < y2[j] ? y2[i] : y2[j];
            float iw = ix2v - ix1v; if (iw < 0) iw = 0;
            float ih = iy2v - iy1v; if (ih < 0) ih = 0;
            float inter = iw * ih;
            float uni = area[i] + area[j] - inter + 1e-9f;
            if (inter / uni >= iou_thresh) suppressed[j] = 1;
        }
    }

    free(suppressed);
    return keep_count;
}

// ------------------------------------------------------------------
// 對外主流程：bitonic sort 排序 + NEON NMS
// keep_indices 存放「原始 boxes 陣列」中被保留框的索引，
// 已依信心分數由高到低排列。回傳實際保留的框數。
// ------------------------------------------------------------------
uint32_t nms_pipeline(const Box* boxes, uint32_t n, float iou_thresh,
                       uint32_t* keep_indices, uint32_t max_keep) {
    if (n == 0) return 0;

    uint32_t padded_n = next_pow2(n);

    float*    score = (float*)malloc(padded_n * sizeof(float));
    uint32_t* idx   = (uint32_t*)malloc(padded_n * sizeof(uint32_t));

    for (uint32_t i = 0; i < n; ++i) {
        score[i] = boxes[i].score;
        idx[i] = i;
    }
    // padding 補上極小分數與 sentinel 索引，排序後永遠沉到最後面
    for (uint32_t i = n; i < padded_n; ++i) {
        score[i] = -3.4e38f;
        idx[i] = 0xFFFFFFFFu;
    }

    bitonic_sort_desc(score, idx, padded_n);

    // 依排序結果重新排列成 SoA，方便 NEON 批次讀取
    float*    sx1 = (float*)malloc(n * sizeof(float));
    float*    sy1 = (float*)malloc(n * sizeof(float));
    float*    sx2 = (float*)malloc(n * sizeof(float));
    float*    sy2 = (float*)malloc(n * sizeof(float));
    float*    sarea = (float*)malloc(n * sizeof(float));
    uint32_t* sorted_orig_idx = (uint32_t*)malloc(n * sizeof(uint32_t));

    uint32_t valid = 0;
    for (uint32_t i = 0; i < padded_n && valid < n; ++i) {
        uint32_t oi = idx[i];
        if (oi == 0xFFFFFFFFu) continue; // 跳過 padding
        sx1[valid] = boxes[oi].x1;
        sy1[valid] = boxes[oi].y1;
        sx2[valid] = boxes[oi].x2;
        sy2[valid] = boxes[oi].y2;
        sarea[valid] = (boxes[oi].x2 - boxes[oi].x1) * (boxes[oi].y2 - boxes[oi].y1);
        sorted_orig_idx[valid] = oi;
        ++valid;
    }

    uint32_t* local_keep = (uint32_t*)malloc(n * sizeof(uint32_t));
    uint32_t keep_count = nms_neon(sx1, sy1, sx2, sy2, sarea, n, iou_thresh, local_keep);

    uint32_t out_count = (keep_count < max_keep) ? keep_count : max_keep;
    for (uint32_t i = 0; i < out_count; ++i) {
        keep_indices[i] = sorted_orig_idx[local_keep[i]];
    }

    free(score); free(idx);
    free(sx1); free(sy1); free(sx2); free(sy2); free(sarea);
    free(sorted_orig_idx); free(local_keep);

    return out_count;
}

// ------------------------------------------------------------------
// 簡單測試（僅用 cstdio，不使用 STL）
// ------------------------------------------------------------------
int main() {
    Box boxes[] = {
        {0.0f, 0.0f, 10.0f, 10.0f, 0.90f},  // 0：高分框
        {1.0f, 1.0f, 11.0f, 11.0f, 0.85f},  // 1：與 0 高度重疊 -> 應被抑制
        {50.0f, 50.0f, 60.0f, 60.0f, 0.75f},// 2：獨立區域，保留
        {0.5f, 0.5f, 10.5f, 10.5f, 0.60f},  // 3：與 0 重疊 -> 應被抑制
        {51.0f, 51.0f, 61.0f, 61.0f, 0.55f},// 4：與 2 重疊 -> 應被抑制
        {100.0f,100.0f,110.0f,110.0f,0.95f},// 5：獨立區域，最高分，保留
    };
    uint32_t n = sizeof(boxes) / sizeof(boxes[0]);

    uint32_t keep[16];
    uint32_t keep_n = nms_pipeline(boxes, n, 0.3f, keep, 16);

    printf("保留 %u 個框（依信心分數高到低）:\n", keep_n);
    for (uint32_t i = 0; i < keep_n; ++i) {
        uint32_t k = keep[i];
        printf("  原始索引 %u  score=%.2f  box=(%.1f,%.1f,%.1f,%.1f)\n",
               k, boxes[k].score, boxes[k].x1, boxes[k].y1, boxes[k].x2, boxes[k].y2);
    }
    return 0;
}
