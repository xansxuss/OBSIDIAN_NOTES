它是一個「線性變換 + 平移」的座標映射。在 2D 影像裡，你拿輸出影像的每一個像素 (x_d, y_d)，反推它在輸入影像中對應的位置 (x_s, y_s)，然後用插值把值「撈」回來。
數學骨架長這樣(2×3 矩陣)：

``` bash
| x_s | = | a b c | | x_d |
| y_s | = | d e f | | y_d |
```

``` markdown
x_s = a*x_d + b*y_d + c
y_s = d*x_d + e*y_d + f
```

***旋轉、縮放、平移、剪切(shear)***，全部都包在裡面。
唯一做不到的是***「透視」***，那是 warpPerspective(3×3)。

#### 為什麼一定要「反推」(backward mapping)

理論上你可以從 source 推到 destination，但工程師通常會選擇反過來：

- 正推(forward mapping)：會有洞(holes)，某些 dst pixel 沒人填
- 反推(backward mapping)：每個 dst pixel 都主動去 src 找值，保證滿版

所以 CUDA / OpenCV / TensorRT 裡看到的幾乎都是：

``` iterate dst → compute src → sample ```

#### 插值：最近鄰 vs 雙線性
反推後的 (x_s, y_s) 幾乎不會是整數，於是要插值：
- Nearest：最快，但會鋸齒、抖動
- Bilinear：取四個點，加權平均
權重是小數部分 (dx, dy)
數學上很乾淨，但在 CUDA 裡會冒出你看到的那些 warning(xHigh 沒用到)，本質就是「你宣告了四個角，卻只用了其中幾個」。

#### 邊界處理是魔鬼藏身處
當 (x_s, y_s) 跑到影像外：
- clamp：卡在邊界
- constant：補 0(YOLO 前處理最愛)
- mirror / wrap：電腦視覺學術派會用，工程派很少
YOLO / TensorRT pipeline 幾乎清一色：out-of-bound → 0
因為 deterministic、好 debug、不會污染特徵。

#### 為什麼 WarpAffine 是 YOLO 前處理的核心
因為它一次解決三件事：
1. resize
2. letterbox(縮放 + 平移)
3. 幾何對齊(bbox 可逆)
只要你保留那個 affine matrix，後處理就能把 bbox 再 warp 回原圖。
這也是為什麼大家寧願自己寫 CUDA kernel，也不想在 CPU 上慢慢玩。

#### 一個工程師視角的小吐槽
WarpAffine 聽起來像高等線代，實際上你在寫的就是：
對每個 pixel 做兩次 FMA + 一次 bilinear lookup
真正難的不是數學，是：
- memory layout(HWC / CHW)
- pitch
- batch
- stream safety
- 邊界條件
數學只佔 5%，其餘 95% 是「別踩雷」。




純 C++（不靠 OpenCV warpAffine）的最小可理解實作，把原理、工程細節一次對齊。這個版本是 CPU + bilinear + backward mapping，寫法跟你之後搬到 CUDA 幾乎一模一樣，只是把 for 迴圈換成 thread。

一、資料結構與假設

假設：

輸入影像：uint8_t、HWC、單通道（先簡化）

輸出影像：同樣格式

``` cpp
affine matrix：2x3，row-major

struct AffineMatrix {
    float m[6]; // [a b c d e f]
};
```

對應數學：

``` bash
x_s = a*x_d + b*y_d + c
y_s = d*x_d + e*y_d + f
```

二、Bilinear 插值（核心）

這段邏輯你在 CUDA kernel 已經寫過，只是這裡是 CPU 版。

``` cpp
inline float bilinearSample(
    const uint8_t* src,
    int srcW, int srcH,
    int srcStride,
    float x, float y)
{
    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    if (x0 < 0 || y0 < 0 || x1 >= srcW || y1 >= srcH)
        return 0.0f;  // out-of-bound → 0（YOLO style）

    float dx = x - x0;
    float dy = y - y0;

    float v00 = src[y0 * srcStride + x0];
    float v01 = src[y0 * srcStride + x1];
    float v10 = src[y1 * srcStride + x0];
    float v11 = src[y1 * srcStride + x1];

    float v0 = v00 + dx * (v01 - v00);
    float v1 = v10 + dx * (v11 - v10);
    return v0 + dy * (v1 - v0);
}
```

三、WarpAffine 主迴圈（backward mapping）

這就是整個 warpAffine 的靈魂。

``` cpp
void warpAffineCPU(
    const uint8_t* src,
    int srcW, int srcH, int srcStride,
    uint8_t* dst,
    int dstW, int dstH, int dstStride,
    const AffineMatrix& M)
{
    const float a = M.m[0];
    const float b = M.m[1];
    const float c = M.m[2];
    const float d = M.m[3];
    const float e = M.m[4];
    const float f = M.m[5];

    for (int y = 0; y < dstH; ++y) {
        for (int x = 0; x < dstW; ++x) {

            float srcX = a * x + b * y + c;
            float srcY = d * x + e * y + f;

            float val = bilinearSample(
                src, srcW, srcH, srcStride, srcX, srcY);

            dst[y * dstStride + x] =
                static_cast<uint8_t>(val + 0.5f);
        }
    }
}
```

四、Affine matrix 怎麼來（以 resize + letterbox 為例）

YOLO 前處理最常見：

``` cpp
AffineMatrix makeLetterboxAffine(
    int srcW, int srcH,
    int dstW, int dstH)
{
    float scale = std::min(
        dstW / (float)srcW,
        dstH / (float)srcH);

    float newW = srcW * scale;
    float newH = srcH * scale;

    float tx = (dstW - newW) * 0.5f;
    float ty = (dstH - newH) * 0.5f;

    AffineMatrix M;
    M.m[0] = 1.0f / scale;
    M.m[1] = 0.0f;
    M.m[2] = -tx / scale;
    M.m[3] = 0.0f;
    M.m[4] = 1.0f / scale;
    M.m[5] = -ty / scale;
    return M;
}
```

注意：
這個 matrix 是 dst → src，不是反的。
很多人第一個 bug 就死在這裡。

五、為什麼這份 C++ 很適合你現在的 CUDA 工作

因為：

- loop = CUDA thread
- (x, y) = blockIdx + threadIdx
- bilinearSample = device inline
- out-of-bound policy 已經跟 YOLO 對齊
- affine matrix layout 跟 TensorRT plugin 常用的一樣

