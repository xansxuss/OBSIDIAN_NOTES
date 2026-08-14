// nms_bitonic_avx2.cpp
//
// x86-64 AVX2 加速版本（對應先前的 arm64 NEON 版本）：
//   1) Bitonic sort：依信心分數 (score) 由高到低排序
//   2) NMS (Non-Maximum Suppression)：用 AVX2 向量化計算 IoU 做抑制
//
// 與 NEON 版差異：暫存器寬度從 128-bit(4 個 float) 換成 256-bit(8 個 float)，
// 演算法邏輯完全相同，只是向量化的門檻與寬度改成 8。
//
// 設計原則：不使用 C++ 標準函式庫容器/演算法（沒有 vector / algorithm /
// sort），只用最基本的 C 記憶體配置 (malloc/free)。
//
// 編譯：
//   g++ -O2 -mavx2 -mfma -o nms_bitonic_avx2 nms_bitonic_avx2.cpp
//
// 備註：目標 CPU 若支援 AVX-512F，可以把寬度再拉到 16 個 float
// （__m512 + _mm512_*），架構完全一樣，只是把 8 換成 16、intrinsics 換前綴。

#include <immintrin.h>
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
// 比較距離 j >= 8 時，同一個 8 元素向量內排序方向必定相同，
// 用 AVX2 對 8 筆資料同時做「比較 + 交換」。
// j < 8 時距離小於向量寬度、會跨越向量邊界，改用純量處理。
// ------------------------------------------------------------------
static void bitonic_sort_desc(float* score, uint32_t* idx, uint32_t n) {
    for (uint32_t k = 2; k <= n; k <<= 1) {
        for (uint32_t j = k >> 1; j > 0; j >>= 1) {

            if (j >= 8) {
                for (uint32_t i = 0; i < n; i += 8) {
                    uint32_t l = i ^ j;
                    if (l < i) continue;              // 每對只處理一次

                    // 因為 j >= 8，這 8 個 i 的 (i & k) 必定一致
                    bool block_even = ((i & k) == 0);

                    __m256  vi_s = _mm256_loadu_ps(score + i);
                    __m256  vl_s = _mm256_loadu_ps(score + l);
                    __m256i vi_i = _mm256_loadu_si256((const __m256i*)(idx + i));
                    __m256i vl_i = _mm256_loadu_si256((const __m256i*)(idx + l));

                    // 目標是「降冪」：
                    //   block_even  -> 想要 score[i] >= score[l]，反之交換
                    //   !block_even -> 想要 score[i] <= score[l]，反之交換
                    __m256 need_swap = block_even
                        ? _mm256_cmp_ps(vi_s, vl_s, _CMP_LT_OQ)   // i < l 時交換
                        : _mm256_cmp_ps(vi_s, vl_s, _CMP_GT_OQ);  // i > l 時交換
                    // 32-bit lane 的比較遮罩每個 byte 高位元一致，可直接用於 epi8 blend
                    __m256i mask_i = _mm256_castps_si256(need_swap);

                    __m256  new_i_s = _mm256_blendv_ps(vi_s, vl_s, need_swap);
                    __m256  new_l_s = _mm256_blendv_ps(vl_s, vi_s, need_swap);
                    __m256i new_i_i = _mm256_blendv_epi8(vi_i, vl_i, mask_i);
                    __m256i new_l_i = _mm256_blendv_epi8(vl_i, vi_i, mask_i);

                    _mm256_storeu_ps(score + i, new_i_s);
                    _mm256_storeu_ps(score + l, new_l_s);
                    _mm256_storeu_si256((__m256i*)(idx + i), new_i_i);
                    _mm256_storeu_si256((__m256i*)(idx + l), new_l_i);
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
// 用 AVX2 一次比對 8 個候選框的 IoU，回傳保留框在 sorted 陣列中的索引。
// ------------------------------------------------------------------
static uint32_t nms_avx2(const float* x1, const float* y1,
                          const float* x2, const float* y2,
                          const float* area,
                          uint32_t n, float iou_thresh,
                          uint32_t* keep_out) {
    uint8_t* suppressed = (uint8_t*)calloc(n, sizeof(uint8_t));
    uint32_t keep_count = 0;

    const __m256 vzero   = _mm256_set1_ps(0.0f);
    const __m256 veps    = _mm256_set1_ps(1e-9f);
    const __m256 vthresh = _mm256_set1_ps(iou_thresh);

    for (uint32_t i = 0; i < n; ++i) {
        if (suppressed[i]) continue;
        keep_out[keep_count++] = i;

        const __m256 vax1 = _mm256_set1_ps(x1[i]);
        const __m256 vay1 = _mm256_set1_ps(y1[i]);
        const __m256 vax2 = _mm256_set1_ps(x2[i]);
        const __m256 vay2 = _mm256_set1_ps(y2[i]);
        const __m256 vaarea = _mm256_set1_ps(area[i]);

        uint32_t j = i + 1;
        for (; j + 8 <= n; j += 8) {
            __m256 vbx1 = _mm256_loadu_ps(x1 + j);
            __m256 vby1 = _mm256_loadu_ps(y1 + j);
            __m256 vbx2 = _mm256_loadu_ps(x2 + j);
            __m256 vby2 = _mm256_loadu_ps(y2 + j);
            __m256 vbarea = _mm256_loadu_ps(area + j);

            __m256 ix1 = _mm256_max_ps(vax1, vbx1);
            __m256 iy1 = _mm256_max_ps(vay1, vby1);
            __m256 ix2 = _mm256_min_ps(vax2, vbx2);
            __m256 iy2 = _mm256_min_ps(vay2, vby2);

            __m256 iw = _mm256_max_ps(_mm256_sub_ps(ix2, ix1), vzero);
            __m256 ih = _mm256_max_ps(_mm256_sub_ps(iy2, iy1), vzero);
            __m256 inter = _mm256_mul_ps(iw, ih);

            __m256 uni = _mm256_add_ps(
                _mm256_sub_ps(_mm256_add_ps(vaarea, vbarea), inter), veps);
            __m256 iou_v = _mm256_div_ps(inter, uni);

            __m256 mask = _mm256_cmp_ps(iou_v, vthresh, _CMP_GE_OQ); // IoU >= 門檻要抑制
            int bits = _mm256_movemask_ps(mask); // 每個 lane 對應一個位元

            for (int t = 0; t < 8; ++t) {
                if (bits & (1 << t)) suppressed[j + t] = 1;
            }
        }
        // 尾端不足 8 筆，純量處理
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
// 對外主流程：bitonic sort 排序 + AVX2 NMS
// keep_indices 存放「原始 boxes 陣列」中被保留框的索引，
// 已依信心分數由高到低排列。回傳實際保留的框數。
// ------------------------------------------------------------------
uint32_t nms_pipeline(const Box* boxes, uint32_t n, float iou_thresh,
                       uint32_t* keep_indices, uint32_t max_keep) {
    if (n == 0) return 0;

    uint32_t padded_n = next_pow2(n);
    if (padded_n < 8) padded_n = 8; // 確保至少一個 AVX2 向量寬度，方便對齊處理

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

    // 依排序結果重新排列成 SoA，方便 AVX2 批次讀取
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
    uint32_t keep_count = nms_avx2(sx1, sy1, sx2, sy2, sarea, n, iou_thresh, local_keep);

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
