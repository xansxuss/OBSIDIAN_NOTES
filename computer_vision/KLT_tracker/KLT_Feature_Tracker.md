---
title: "KLT Feature Tracker"
source: "https://gemini.google.com/app/4d42a94559704988"
author:
published:
created: 2026-07-03
description: "Gemini conversation with 2 messages"
tags:
  - "clippings"
---
**Kanade–Lucas–Tomasi (KLT) Feature Tracker** 是電腦視覺（Computer Vision）領域中非常經典且奠定基礎的稀疏特徵追蹤（Sparse Feature Tracking）演算法。它結合了 **Lucas-Kanade (LK) 光流法** 與 **Shi-Tomasi 角點偵測** 的核心思想。

對於撰寫 C/C++ 或 Python 的底層影像處理工程師來說，理解 KLT 的數學本質與實作架構至關重要。以下為您梳理 KLT 的核心原理與運作機制。

## 1\. 核心三大假設

KLT 特徵追蹤的數學推導建立在以下三個前提之上：

- **亮度恆定（Brightness Constancy）：** 同一個空間點在連續兩幀影像中的像素灰階值（Intensity）保持不變。
	Grokipedia
- **時間連續 / 微小移動（Temporal Persistence）：** 幀與幀之間的時間間隔極短，特徵點的位移非常微小。這允許我們使用泰勒展開式（Taylor expansion）進行線性逼近。
	Grokipedia
- **空間一致性（Spatial Coherence）：** 特徵點周圍的小視窗（Window，例如 $7 \times 7$ 或 $11 \times 11$ ）內的所有像素，都具有相同的運動向量。
	Grokipedia

## 2\. 特徵點選擇：什麼是「好追蹤的特徵」？

在 1994 年的論文中，Shi 和 Tomasi 提出了解決光流法中孔徑問題（Aperture Problem）的特徵選擇機制。

針對影像 $I$ 中的任意像素點，計算其局部視窗 $W$ 內的光流結構張量（Structure Tensor）或稱黑塞矩陣（Hessian Matrix） $G$ ：

$$
G = \sum_{W} \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}
$$

其中 $I_x, I_y$ 分別是像素在 $x$ 與 $y$ 方向的影像梯度（偏微分）。

- 對矩陣 $G$ 進行特徵值分解，得到兩個特徵值 $\lambda_1, \lambda_2$ （假設 $\lambda_1 \ge \lambda_2$ ）。
- **Shi-Tomasi 準則：** 當 $\min(\lambda_1, \lambda_2) > \theta$ （ $\theta$ 為設定的閾值）時，該點才被定義為「適合追蹤的特徵點」（通常是角點）。這確保了矩陣 $G$ 是可逆的（Invertible），不會出現退化（如平坦區域或邊緣空轉問題）。
	Grokipedia

## 3\. 追蹤演算法：Lucas-Kanade 迭代優化

給定前後兩幀影像 $I$ 與 $J$ ，目標是為每個特徵點尋找一個位移向量 $\mathbf{d} = [d_x, d_y]^T$ ，使得視窗內的平方誤差和（SSD）最小化：

$$
\epsilon(\mathbf{d}) = \sum_{W} \left[ J(\mathbf{x} + \mathbf{d}) - I(\mathbf{x}) \right]^2
$$

利用一階泰勒展開式將 $J(\mathbf{x} + \mathbf{d})$ 線性逼近為 $J(\mathbf{x}) + \nabla J \cdot \mathbf{d}$ ，並定義兩幀的時域差為 $I_t(\mathbf{x}) = J(\mathbf{x}) - I(\mathbf{x})$ ，經過偏微分為零的最小平方法（Least Squares）推導後，可得到經典的正規方程（Normal Equation）：

$$
G \mathbf{d} = \mathbf{b}
$$

其中 $\mathbf{b} = - \sum_{W} I_t \begin{bmatrix} I_x \\ I_y \end{bmatrix}$ 。

由於這是一個線性逼近，當真實位移稍大時需要透過 **Newton-Raphson 迭代法** 不斷更新 $\mathbf{d}$ ，直到 $\Delta \mathbf{d}$ 小於閾值或達到最大迭代次數。

## 4\. 金字塔實作（Pyramidal KLT）

標準的 KLT 只能處理像素級別的微小位移（通常小於 1~2 像素）。為了讓演算法具備應對大範圍運動的魯棒性（Robustness），實務上必定會引入影像金字塔（Image Pyramid）：

1. 建構高斯金字塔

Coarse-to-Fine

將原始影像進行高斯平滑並降採樣，建立多層解像度的金字塔（通常為 3-4 層）。

2. 頂層粗估（Top Level）

大尺度捕捉

在影像解像度最低的頂層，原本較大的物理位移會被按比例縮小，此時運行 LK 迭代，捕捉到粗略的運動趨勢。

3. 逐層傳遞與細化

向下投影

將上一層計算出的位移量放大兩倍，作為下一層（解像度較高）迭代的初始猜測值（Initial Guess）。

4. 底層精確解（Layer 0）

亞像素級微調

最終回到原始影像層（Level 0）進行最後的迭代優化，獲得具備次像素（Subpixel）精度的特徵點座標。

## 5\. C/C++ 與 Python 實作建議

- **Python / OpenCV 生態：** 通常直接調用 `cv2.goodFeaturesToTrack()` （內部實作即為 Shi-Tomasi 算子）挑選特徵，隨後送入 `cv2.calcOpticalFlowPyrLK()` 進行金字塔光流追蹤。
- **C/C++ 自行封裝（不使用標準函式庫與 OpenCV）：** 身為 AI 工程師，若需要極致效能或嵌入式佈署，您需要自行實作以下核心元件：
	1. **影像梯度運算元：** 建議使用 Scharr 或 Sobel 算子計算 $I_x, I_y$ 。
		2. **記憶體對齊（Memory Alignment）：** 計算矩陣 $G$ 與 $\mathbf{b}$ 時，利用 SIMD（如 SSE/AVX 或 ARM Neon）進行多像素並行累加。
		3. **雙線性插值（Bilinear Interpolation）：** 在迭代過程中， $\mathbf{x} + \mathbf{d}$ 經常落於非整數像素座標，必須實作快速的雙線性插值以讀取亞像素的灰階值與梯度。
		4. **2x2 矩陣反元素：** 由於 $G$ 僅為 $2 \times 2$ ，直接利用克萊姆法則（Cramer's Rule）計算逆矩陣，避免繁複的矩陣庫依賴。