影像頻域領域（Fourier space 🌀）——這是影像濾波中最「工程＋理論味」的部分。
完整拆解：高通 / 低通 / 帶通 / 帶阻濾波，包含：
🧠 原理與公式
⚙️ OpenCV 實作方式
🔍 優缺點
🧩 適用場景
⚡ 專業建議（從頻譜角度）

### 1. 頻域濾波的核心思想

在空間域（spatial domain）中，我們對像素做卷積；
而在頻域（frequency domain）中，我們對影像的頻譜分量進行操作。

👉 根據傅立葉轉換的卷積定理：
- 空間卷積 ⇔ 頻域乘法
- 空間濾波器 → 頻域遮罩 (mask)

### 2. 濾波類型總覽
| 類型                           | 通過頻率      | 阻擋頻率      | 用途        | 類比音訊範例   |
| ---------------------------- | --------- | --------- | --------- | -------- |
| **低通濾波 (Low-pass)**          | 低頻（平滑區）   | 高頻（細節/雜訊） | 去雜訊、模糊化   | 低音通過     |
| **高通濾波 (High-pass)**         | 高頻（邊緣/細節） | 低頻（平滑背景）  | 強化邊緣、銳化   | 高音通過     |
| **帶通濾波 (Band-pass)**         | 特定頻段      | 其他頻段      | 紋理分析、條紋特徵 | 均衡器某頻段強化 |
| **帶阻濾波 (Band-stop / Notch)** | 阻擋特定頻段    | 其他通過      | 移除週期性干擾   | 消除電波干擾噪聲 |

### 3. 頻域濾波數學模型

設影像傅立葉轉換為：

F(u,v)=F{f(x,y)}

濾波後的頻譜為：

G(u,v)=H(u,v)⋅F(u,v)

再逆轉換回空間域：

g(x,y)=F−1{G(u,v)}

其中 H(u,v) 是濾波器遮罩 (filter mask)。

### 4. OpenCV 實作步驟（通用模板）

``` cpp
// 1. 轉為灰階與 float
cv::Mat src = cv::imread("image.png", cv::IMREAD_GRAYSCALE);
cv::Mat padded;
int m = cv::getOptimalDFTSize(src.rows);
int n = cv::getOptimalDFTSize(src.cols);
cv::copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

// 2. 建立複數矩陣
cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
cv::Mat complexI;
cv::merge(planes, 2, complexI);

// 3. DFT
cv::dft(complexI, complexI);

// 4. 產生濾波遮罩 (H)
cv::Mat mask(padded.size(), CV_32F);
float D0 = 50; // 截止頻率
for (int i = 0; i < mask.rows; i++)
    for (int j = 0; j < mask.cols; j++) {
        float D = std::sqrt((i - mask.rows/2)*(i - mask.rows/2) + (j - mask.cols/2)*(j - mask.cols/2));
        mask.at<float>(i,j) = (D <= D0) ? 1.0f : 0.0f;  // Low-pass 範例
    }

// 5. 套用遮罩
cv::Mat channels[2];
cv::split(complexI, channels);
cv::multiply(channels[0], mask, channels[0]);
cv::multiply(channels[1], mask, channels[1]);
cv::merge(channels, 2, complexI);

// 6. 逆轉換
cv::idft(complexI, complexI);
cv::split(complexI, channels);
cv::normalize(channels[0], channels[0], 0, 1, cv::NORM_MINMAX);
cv::imshow("Filtered", channels[0]);
```

### 5. 濾波器類型細解
1. 低通濾波器 (LPF, Low-pass Filter)

| 項目 | 說明 |
| ---- | ---- |
| **原理** | 保留低頻（緩變區域），抑制高頻（邊緣與雜訊） |
| **遮罩例** | <br>理想低通：<br>$$ H(u,v) = \begin{cases} 1 & D(u,v) \le D_0 \ 0 & D(u,v) > D_0 \end{cases} $$<br><br>高斯低通：<br>$$ H(u,v) = e^{-\frac{D^2(u,v)}{2D_0^2}} $$ |
| **優點** | 去高頻雜訊、平滑圖像 |
| **缺點**   | 模糊邊緣、細節損失 |
| **應用場景** | - 模糊化 / 降噪<br> - 圖像金字塔前處理<br> - 影像降采樣 (anti-aliasing) |
2. 高通濾波器 (HPF, High-pass Filter)

| 項目 | 說明 |
| ---- | ---- |
| **原理** | 保留高頻（細節、邊緣），抑制低頻（背景） |
| **遮罩例** | <br>理想高通：<br>$$ H(u,v) = 1 - \text{LPF}(u,v) $$<br><br>高斯高通：<br>$$ H(u,v) = 1 - e^{-\frac{D^2(u,v)}{2D_0^2}} $$ |
| **優點** | 強化邊緣、對比明顯 |
| **缺點** | 可能放大雜訊、影像顫動感強 |
| **應用場景** | - 銳化影像<br> - 邊緣偵測<br> - 紋理分析 |
3. 帶通濾波器 (Band-pass Filter)

| 項目 | 說明 |
| ---- | ---- |
| **原理** | 僅保留介於兩個截止頻率 ( D_1 < D < D_2 ) 的頻率成分 |
| **遮罩例** | $$ H(u,v) = e^{-\frac{(D^2(u,v) - D_c^2)^2}{2 D_w^2}} $$（高斯帶通）|
| **優點** | 可針對特定紋理頻率進行分析 |
| **缺點** | 頻帶需人工設計，非通用 |
| **應用場景** | - 條紋分析<br> - 模式辨識（週期紋）<br> - 生物紋理（指紋 / 紡織瑕疵）|
4. 帶阻濾波器 (Band-stop / Notch Filter)

| 項目 | 說明 |
| ---- | ---- |
| **原理** | 阻擋特定頻率範圍，其他通過（與帶通相反） |
| **遮罩例** | $$ H(u,v) = 1 - \text{Band-pass}(u,v) $$ |
| **優點** | 消除週期性干擾（如正弦條紋、電波噪聲） |
| **缺點** | 需手動選中心頻率 |
| **應用場景** | - 去除週期干擾（如掃描線、摩爾紋）<br> - 光學影像修復 |

### 6. 濾波器頻譜視覺化

| 類型           | 頻譜遮罩示意       | 結果影像效果  |
| ------------ | ------------ | ------- |
| **低通 (LPF)** | 中央亮（通過低頻）    | 模糊、平滑   |
| **高通 (HPF)** | 四周亮（保邊）      | 銳化、強邊緣  |
| **帶通 (BPF)** | 環狀亮區         | 僅保留特定紋理 |
| **帶阻 (BSF)** | 中央 + 外圈亮，中間暗 | 移除特定紋理  |

### 7. 實務建議

| 目標          | 推薦濾波             | 說明                   |
| ----------- | ---------------- | -------------------- |
| 去高頻雜訊 / 平滑化 | 低通               | 比 GaussianBlur 更精確控制 |
| 銳化 / 邊緣強化   | 高通               | 可結合原圖 `unsharp mask` |
| 消除週期干擾      | 帶阻               | 精準 notch 掉干擾頻率       |
| 紋理分析 / 特徵提取 | 帶通               | 適合 Gabor 或 FFT 結合使用  |
| 實時系統        | ⚠️ 優先考慮空間濾波（效率高） | 頻域濾波偏慢，不適合即時影像       |


