這三種格式（YU12、YV12、NV12）都屬於 YUV 4:2:0 的家族，也就是說——
👉 亮度（Y）是完整取樣的，
👉 色度（U/V）則每 2x2 像素共用一組值。

不過，它們的**記憶體排列方式（memory layout）**不同，這差異在 GPU / DMA / 硬體編解碼 / OpenCV / GStreamer 等場景下會非常關鍵。
以下是更深入的技術解析 👇

🧩 一、YU12（又稱 I420）
📖 定義

YU12 是最標準的 Planar YUV 4:2:0 格式，也常被稱為 I420（在 FFmpeg、OpenCV、GStreamer 等框架中幾乎等同）。

🧠 記憶體排列

依序存放三個平面（plane）：

``` bash
YYYYYYYYYYYY...
UUUU...
VVVV...
```

| 分量 | 尺寸            | 說明             |
| -- | ------------- | -------------- |
| Y  | W × H         | 每個像素都有         |
| U  | (W/2) × (H/2) | 每 2x2 像素共用一個 U |
| V  | (W/2) × (H/2) | 每 2x2 像素共用一個 V |

🧮 若畫面為 1920×1080：

``` bash
Y: 1920*1080 bytes
U: 960*540 bytes
V: 960*540 bytes
Total = 1920*1080*1.5 = 3,110,400 bytes
```

📦 範例（前幾行）

``` bash
[Y-plane]
YYYYYYYY
YYYYYYYY
...

[U-plane]
UUUU
UUUU
...

[V-plane]
VVVV
VVVV
...
```

🧩 二、YV12
📖 定義

YV12 與 YU12 幾乎一樣，唯U、V 平面的順序顛倒。
可視為「YU12 的姊妹版」。

🧠 記憶體排列

``` bash
YYYYYYYYYYYY...
VVVV...
UUUU...
```

| 分量 | 尺寸            | 說明      |
| -- | ------------- | ------- |
| Y  | W × H         | 同 YU12  |
| V  | (W/2) × (H/2) | 注意：V 在前 |
| U  | (W/2) × (H/2) | 注意：U 在後 |

🚩 關鍵差異

YU12：Y → U → V

YV12：Y → V → U

⚠️ 這一行的順序差異足以導致畫面偏色或整體色彩顛倒。
例如：若你誤把 YV12 當成 YU12 解碼，整個畫面會呈現「偏藍」或「偏紅」。

✅ 特點

常見於舊版的 Windows AVI、MPEG、一些古老的 VFW（Video for Windows）Codec。

許多硬體或 GPU decoder 不再原生支援。

🧩 三、NV12（Semi-Planar）
📖 定義

NV12 是 Intel（後來 NVIDIA、AMD 都支援）提出的 半平面（semi-planar） 格式。
亮度（Y）仍然是單獨平面，但 U、V 是交錯存放（interleaved） 的。

🧠 記憶體排列

``` bash
YYYYYYYYYYYY...
UVUVUVUVUVUV...
```

| 分量 | 尺寸                                | 說明                       |
| -- | --------------------------------- | ------------------------ |
| Y  | W × H                             | Luma plane               |
| UV | (W/2) × (H/2) × 2 bytes per pixel | Chroma interleaved plane |

例如：

``` bash
Y plane:
Y0 Y1 Y2 Y3 Y4 Y5 ...

UV plane:
U0 V0 U1 V1 U2 V2 ...
```

✅ 特點

- 硬體編解碼器的首選格式（Intel、NVIDIA、ARM Mali 都支援）
- DMA-friendly，適合零拷貝（zero-copy）影像處理。
- OpenCV / CUDA / GStreamer 中支援度極高：
    - GStreamer caps 通常寫成 "video/x-raw, format=NV12"
    - CUDA 支援 cv::cuda::cvtColor(GpuMat(NV12) → BGR)

⚙️ 四、比較表格

| 格式              | 結構型態        | U/V 排列 | 記憶體連續性  | GPU/硬體支援 | 備註      |
| --------------- | ----------- | ------ | ------- | -------- | ------- |
| **YU12 (I420)** | Planar      | U → V  | 3 個分開平面 | 高        | 最通用     |
| **YV12**        | Planar      | V → U  | 3 個分開平面 | 中        | 舊系統遺留格式 |
| **NV12**        | Semi-Planar | 交錯 UV  | 2 個平面   | 非常高      | 硬體原生支援  |

🧠 五、實務建議（工程角度）

| 應用                        | 建議格式        | 原因                  |
| ------------------------- | ----------- | ------------------- |
| 軟體編碼（x264、FFmpeg）         | YU12 / I420 | 通用，壓縮演算法最佳化         |
| GPU / CUDA / TensorRT 前處理 | NV12        | 硬體零拷貝支援最佳           |
| 相機 ISP（V4L2、GStreamer）    | NV12        | 最常見輸出格式             |
| 舊式 AVI / MPEG             | YV12        | 歷史相容性               |
| OpenCV CPU 轉換             | YU12        | 方便 `cv::cvtColor()` |
