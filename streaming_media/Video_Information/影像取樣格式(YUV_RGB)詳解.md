---
title: 影像取樣格式（YUV／RGB）詳解
source: 
author: 
published: 
created: 2026-09-07
description: 整理 YUV444/YUV420/YUV422/YUV411/YUV410 各家族格式的記憶體排列、bpp 與應用場景
categorization: streaming_media/Video_Information
tags:
  - YUV
  - 色彩空間
  - chroma-subsampling
  - V4L2
  - NVDEC
  - 影像格式
---

## 一、色彩空間與取樣的基本概念

### 為什麼用 YUV 而不是 RGB

人眼對「亮度」(Luminance) 的敏感度遠高於對「色度」(Chrominance) 的敏感度。YUV 色彩空間把影像拆成：

- **Y（Luma）**：亮度資訊，決定畫面的黑白細節
- **U（Cb）**：藍色色度差
- **V（Cr）**：紅色色度差

因為人眼對色度不敏感，色度分量可以用比亮度更低的取樣頻率儲存，這就是「Chroma Subsampling（色度取樣）」，可以大幅降低資料量而視覺上幾乎沒有差異。

### 取樣格式的命名規則 J:a:b

業界常用 `J:a:b` 來描述取樣比例（J 通常是 4）：

| 記法 | 意義 |
|---|---|
| 4:4:4 | 每個 Y 都有對應的 U、V，完全不做色度壓縮 |
| 4:2:2 | 水平方向每 2 個 Y 共用 1 組 UV |
| 4:2:0 | 水平、垂直方向每 2×2 個 Y 共用 1 組 UV |
| 4:1:1 | 水平方向每 4 個 Y 共用 1 組 UV |
| 4:1:0 | 水平、垂直方向每 4×4 個 Y 共用 1 組 UV |

---

## 二、RGB 與 YUV444（無色度壓縮）

### RGB（RGB24 / RGB888）

- 不是 YUV 家族，是直接的色彩三原色表示
- 每像素 R、G、B 各 8-bit，共 24 bpp，無壓縮無取樣
- 顯示器輸出、GPU 渲染、大部分電腦視覺演算法（OpenCV 預設處理）都是用這格式，因為運算上不需要先做色彩空間轉換
- 缺點：資料量比 YUV420 大整整 2 倍，不適合直接拿來做影片編碼或網路傳輸

### YUV444（4:4:4）

- 每個像素都有完整獨立的 Y、U、V 三個分量
- 資料量：每像素 3 bytes（若每分量 8-bit），即 24 bpp
- 沒有色度壓縮，畫質最好，但頻寬與儲存成本最高
- 常見於專業影像後製、色度鍵（去背）需要高精度色彩邊緣的場合

---

## 三、YUV420（4:2:0）家族 — 12 bpp

每 2×2 個 Y 像素共用 1 組 U、V，寬高各壓縮一半。這是消費性影片、視訊會議、H.264/H.265 編碼最常用的格式，因為它在畫質與頻寬間取得很好的平衡。

> 注意：YUV420 是一個「概念類別」，實際記憶體排列方式還細分成 **planar**（分平面）跟 **semi-planar**（半平面）好幾種變形。

### I420（又稱 YUV420p，full planar）

- 三個 plane **完全分開**：Y、U、V 各自獨立一塊記憶體

```
Plane 0：Y，完整解析度
Plane 1：U，寬高各為 Y 的一半
Plane 2：V，寬高各為 Y 的一半
```

- 資料量跟 NV12 一樣（12 bpp），差別只在 U、V 是否合併成一個 plane
- 是 FFmpeg（`AV_PIX_FMT_YUV420P`）、libx264/libx265 編碼器內部最常用的標準格式，因為完全分離的 plane 對軟體編碼演算法處理起來比較單純

### YV12

- I420 的變形，只是把 U plane 跟 V plane 順序對調（先 V 後 U），概念上跟 NV12/NV21 的關係一樣

### NV12（semi-planar，UV 交錯）

這是 NVIDIA（包含 Jetson）硬體解碼器 NVDEC 輸出的預設格式，也是拆解 NvBuffer 時最常遇到的格式。

記憶體排列（2 個 plane）：

```
Plane 0（Y plane）：完整解析度的 Y，逐 row 排列
Plane 1（UV plane）：U、V 交錯排列，解析度是 Y 的一半（寬高各一半）
  排列方式：U0 V0 U1 V1 U2 V2 ...
```

範例（4×4 影像）：

```
Y plane (4x4):
Y00 Y01 Y02 Y03
Y10 Y11 Y12 Y13
Y20 Y21 Y22 Y23
Y30 Y31 Y32 Y33

UV plane (2x2, 交錯):
U00 V00 U01 V01
U10 V10 U11 V11
```

- 「semi-planar」的意思就是：Y 獨立一個 plane，但 U、V 合併成一個 plane 用交錯方式儲存，不是完全分開的三個 plane
- 廣泛用於：NVDEC/NVENC 硬體編解碼、Android Camera HAL、大部分 GPU 影像處理管線

### NV21（semi-planar，VU 交錯）

- 跟 NV12 幾乎一模一樣，唯一差異是第二個 plane 的順序是 **V 在前、U 在後**：`V0 U0 V1 U1 ...`
- 常見於 Android Camera（Android 的 `ImageFormat.NV21` 是很多手機相機的原生輸出格式）
- 在做格式轉換（例如 NV12 → NV21）時，只要注意第二個 plane 的 byte 順序對調即可，Y plane 完全不用動

---

## 四、YUV422（4:2:2）家族 — 16 bpp

水平方向每 2 個 Y 共用 1 組 UV，垂直方向不做取樣（跟 Y 一樣密度）。色度只在水平方向被壓縮一半，資料量是每像素 16 bpp（2 bytes），介於 YUV444（24 bpp）跟 YUV420（12 bpp）之間。

### YUYV（又稱 YUY2，packed）

- **Packed（交錯式）** 格式，不是 planar，Y/U/V 全部混在同一個 buffer 裡
- 排列順序：`Y0 U0 Y1 V0  Y2 U1 Y3 V1 ...`
- 每 2 個像素共用一組 UV，屬於 4:2:2 取樣
- 常見於 USB Camera（UVC）、V4L2 capture 裝置的原始輸出格式，因為硬體 ISP 掃描像素時一次吐出交錯資料比較方便
- 記憶體上只有一個 plane，沒有分開的 Y plane 跟 UV plane

### UYVY（packed，順序不同）

- 跟 YUYV 幾乎一樣，只是 Y 跟 UV 的順序對調：`U0 Y0 V0 Y1 U1 Y2 V1 Y3 ...`
- 也就是每組 4 bytes 是 `U Y V Y`，而不是 `Y U Y V`
- 常見於類比視訊卡（analog capture card）、SDI 訊號、部分監視器/工業相機的原生輸出
- 跟 YUYV 轉換時只要注意 byte offset 對調，不需要重新計算色度值

### YVYU（packed，U/V 順序對調）

- 排列：`Y0 V0 Y1 U0 ...`
- 也是 YUYV 的變形，只是 U、V 位置互換
- 較少見，部分舊型 Windows DirectShow 相機驅動會用

### I422（YUV422P，full planar）

- 3 個 plane 完全分開：Y、U、V
- U、V plane 的**寬度**是 Y 的一半，但**高度**跟 Y 相同（這是跟 4:2:0 最大的差異，4:2:0 是寬高都減半）
- FFmpeg 對應格式：`AV_PIX_FMT_YUV422P`
- 用於需要比 4:2:0 更好色彩精度、又不想到 4:4:4 那麼大資料量的專業影音場合（例如廣電、ProRes 422 內部處理）

### NV16（semi-planar，UV 交錯）

- 2 個 plane：Y plane 完整解析度；UV plane 寬度減半、高度不變（因為是 4:2:2），交錯排列 `U0 V0 U1 V1 ...`
- 是 NV12 在 4:2:2 版本下的對應格式，部分 ISP/硬體編碼器會用

### NV61（semi-planar，VU 交錯）

- 跟 NV16 一樣，只是第二個 plane 順序對調成 `V0 U0 V1 U1 ...`
- 是 NV21 的 4:2:2 對應版本

---

## 五、更低取樣密度格式（少見）

### YUV411（4:1:1）

- 水平方向每 4 個 Y 共用 1 組 UV，垂直不取樣
- 資料量：12 bpp，跟 4:2:0 一樣，但壓縮方式不同（4:1:1 只在水平方向壓，4:2:0 是水平垂直各壓一半）
- 早期 DV（Digital Video，如 MiniDV 攝影機）常用，現在已經很少見
- 常見變形：Y41P（packed）

### YUV410（4:1:0）

- 水平垂直方向都是每 4 個 Y 共用 1 組 UV，色度取樣密度非常低
- 資料量最小，約 9 bpp
- 早期低頻寬視訊會議（如 H.263 早期版本）用過，現在幾乎絕跡

---

## 六、高位元深度與其他格式

### P010 / P016

- 這是 NV12 的 10-bit / 16-bit 版本，Plane 排列邏輯跟 NV12 完全一樣（Y plane + 交錯 UV plane），差別在每個 sample 用 **16-bit 容器**儲存（10-bit 有效資料 + 6-bit padding，或 16-bit 全用）
- 這是 HDR 影片（HDR10、Dolby Vision）、H.265/AV1 10-bit 編碼常用的輸出格式
- 在 Jetson 平台上，如果解碼 10-bit H.265 串流，NVDEC 輸出就會是 P010 而不是 NV12，這點在研究 `fill_buffer_plane_format` 時要特別注意 bytesperpixel 會變成 2（而不是 1）

### GBR / GBRP（RGB 的 planar 版本）

- 跟 RGB24 資料內容一樣，但拆成 3 個獨立 plane（G、B、R 各自一塊）
- FFmpeg 內部濾鏡處理常用，因為 planar 對某些演算法（如色彩調整）運算比較方便

---

## 七、總覽對照表

| 格式 | 取樣比例 | Plane 數 | Chroma 排列 | bpp | 典型使用場景 |
|---|---|---|---|---|---|
| RGB24 | 無取樣 | 1（交錯） | — | 24 | 顯示、OpenCV、GPU 渲染 |
| YUV444 | 4:4:4 | 3（planar） | 各自獨立 | 24 | 專業後製、去背 |
| YUYV/YUY2 | 4:2:2 | 1（packed） | Y U Y V 交錯 | 16 | USB Camera / V4L2 原始輸出 |
| UYVY | 4:2:2 | 1（packed） | U Y V Y 交錯 | 16 | 類比擷取卡、SDI |
| YVYU | 4:2:2 | 1（packed） | Y V Y U 交錯 | 16 | 少數舊型驅動 |
| I422 (YUV422P) | 4:2:2 | 3（planar） | Y / U / V 各自獨立 | 16 | 廣電、專業剪輯 |
| NV16 | 4:2:2 | 2（semi-planar） | Y / (UV 交錯) | 16 | 部分 ISP/硬體編碼 |
| NV61 | 4:2:2 | 2（semi-planar） | Y / (VU 交錯) | 16 | 少見 |
| I420 (YUV420p) | 4:2:0 | 3（planar） | Y / U / V 各自獨立 | 12 | FFmpeg、x264/x265 軟體編碼 |
| YV12 | 4:2:0 | 3（planar） | Y / V / U 各自獨立 | 12 | 與 I420 相同，順序相反 |
| NV12 | 4:2:0 | 2（semi-planar） | Y / (UV 交錯) | 12 | NVDEC/NVENC 硬體、Android |
| NV21 | 4:2:0 | 2（semi-planar） | Y / (VU 交錯) | 12 | Android Camera |
| YUV411 | 4:1:1 | 依變形而定 | — | 12 | 早期 DV 攝影機 |
| YUV410 | 4:1:0 | 依變形而定 | — | 9 | 早期低頻寬視訊會議 |
| P010/P016 | 4:2:0 | 2（semi-planar，16-bit） | Y / (UV 交錯，16-bit) | 15/24 | HDR、10-bit H.265/AV1 |

---

## 八、跟 [[NvVideoDecoder]] 與[[NvVideoEncoder]]的延伸重點

在拆解 `NvBuffer::fill_buffer_plane_format` 時，針對不同 pixel format 計算 plane 數量、每個 plane 的 stride（每 row 的實際 byte 數，通常會有 alignment padding）跟 bytesperpixel 的邏輯重點：

- **NV12/NV21**：2 個 plane，Y plane 的 bytesperpixel = 1，UV plane 的 bytesperpixel = 2（因為 U、V 交錯在同一個 byte 序列裡），且 UV plane 的 width/height 都要除以 2
- **I420/YV12**：3 個 plane，每個 plane bytesperpixel 都是 1，但 U、V plane 的 width/height 各除以 2
- **YUYV/UYVY**：只有 1 個 plane，但 bytesperpixel = 2（因為每 2 個像素共用 4 bytes）
- **4:2:2 系列（NV16/I422 等）**：chroma plane 只有寬度減半，高度跟 Y 相同 — 與 4:2:0 系列（寬高都減半）邏輯不同，容易寫錯
- **P010**：bytesperpixel = 2，且每個 plane 的 stride 計算要以 16-bit 為單位對齊，不能沿用 NV12 的 8-bit 邏輯直接套用

NVDEC 硬體解碼輸出通常是 NV12（方便硬體用 DMA 一次搬移整個交錯的 UV plane），但如果要餵給 x264/x265 這類軟體編碼器，通常還需要轉成 I420，這中間的轉換（NV12 → I420）本質上只是把交錯的 UV plane 拆成兩個獨立 plane，不涉及色彩空間轉換，運算量很小。
