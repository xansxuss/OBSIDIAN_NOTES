---
title: YOLOX 架構與各部件數學推導
source: 
author: 
published: 
created: 2026-09-08
description: YOLOX 各部件（backbone、neck、decoupled head、anchor-free、SimOTA）的架構與數學原理整理
categorization: computer_vision
tags:
  - YOLOX
  - object-detection
  - anchor-free
  - deep-learning
  - computer-vision
---

## 概述

YOLOX（Megvii, 2021）是 YOLO 系列的重要轉折點，核心變革是把 anchor-based 設計換成 anchor-free，並解決分類與回歸任務共用同一組卷積導致互相干擾的問題。整體資料流：

輸入影像 → CSPDarknet backbone → PAFPN neck → Decoupled head ×3 尺度 → 輸出預測（訓練時以 SimOTA 分配標籤）

與 [[license-plate-recognition]] 專案相關：目前使用的 YOLOv8n 已吸收 YOLOX 的 decoupled head、anchor-free、動態標籤分配（TaskAlignedAssigner）等設計，屬於同一世代架構哲學。

---

![[YOLOX架構詳解_p1.png]]

## Backbone：CSPDarknet

跟 YOLOv5 完全一樣，用 CSP（Cross Stage Partial）結構降低運算量同時保留梯度資訊。輸出三個不同深度的特徵圖（stride 8、16、32），分別對應小、中、大物件。這部分 YOLOX 沒有創新，是直接沿用既有設計。

給定輸入 $X \in \mathbb{R}^{C \times H \times W}$：

$$
X_1 = \text{Conv}_{1\times1}(X), \quad X_2 = \text{Conv}_{1\times1}(X)
$$

$$
X_1' = \text{Bottleneck}^{(n)}(X_1)
$$

$$
Y = \text{Conv}_{1\times1}\big(\text{Concat}(X_1', X_2)\big)
$$

其中 $\text{Bottleneck}^{(n)}$ 為 $n$ 層堆疊殘差單元：

$$
X^{(k+1)} = X^{(k)} + \text{Conv}_{3\times3}\big(\sigma(\text{BN}(\text{Conv}_{1\times1}(X^{(k)})))\big)
$$

$\sigma$ 為 SiLU 激活函數：$\sigma(x) = x \cdot \text{sigmoid}(x)$

$X_2$ 分支跳過殘差堆疊，直接在最後 concat，反向傳播時不會重複累積 $X_1$ 已算過的梯度（CSPNet 論文論點）。

下採樣（stride 2 卷積）：

$$
H_l = \left\lfloor \frac{H_{l-1}}{2} \right\rfloor
$$

640×640 輸入經 5 個 stage，輸出三個尺度：
- $C_3$：stride 8，80×80
- $C_4$：stride 16，40×40
- $C_5$：stride 32，20×20

---

## Neck：PAFPN

### Top-down（語意由深往淺傳）
Path Aggregation FPN，先做 top-down（把深層語意特徵往上採樣、融合到淺層），再做 bottom-up（把淺層高解析度資訊往下傳回深層）。雙向融合讓每個尺度的特徵圖都同時具備語意強度與空間解析度，這是它對小物件比較友善的原因之一——淺層特徵沒有被單向丟棄。

$$
P_5 = \text{Conv}_{1\times1}(C_5)
$$

$$
P_4 = \text{CSPLayer}\big(\text{Concat}(\text{Upsample}(P_5),\, C_4)\big)
$$

$$
P_3 = \text{CSPLayer}\big(\text{Concat}(\text{Upsample}(P_4),\, C_3)\big)
$$

Upsample 用最近鄰插值放大兩倍：

$$
F_{\text{up}}(i,j) = F\left(\left\lfloor \frac{i}{2} \right\rfloor, \left\lfloor \frac{j}{2} \right\rfloor\right)
$$

### Bottom-up（空間解析度由淺往深傳回，PAFPN 的 "PA" 部分，YOLOv3 只有 top-down）

$$
N_3 = P_3
$$

$$
N_4 = \text{CSPLayer}\big(\text{Concat}(\text{Downsample}(N_3),\, P_4)\big)
$$

$$
N_5 = \text{CSPLayer}\big(\text{Concat}(\text{Downsample}(N_4),\, P_5)\big)
$$

Downsample 用 stride 2 的 3×3 卷積（非池化）。$N_3, N_4, N_5$ 三尺度分別送進三個獨立 decoupled head。

---

## Head：Decoupled head

YOLOv3~v5 把分類（class）、物件性（objectness）、座標回歸（bbox regression）全部塞在同一組卷積輸出，YOLOX 作者發現這樣做會拖慢收斂速度，因為分類是「語意任務」、回歸是「空間任務」，兩者需要的特徵表示方向其實有衝突。

YOLOX 把 head 拆成三條獨立分支，各自用 1×1 conv 降維後接自己的卷積層，最後才把結果 concat 回去。論文實驗顯示這樣做能明顯加速收斂，也提升精度，代價是參數量和運算量略增。

每個尺度特徵圖 $F \in \mathbb{R}^{256 \times H \times W}$（先用 $\text{Conv}_{1\times1}$ 統一降到 256 channel），分成兩條獨立的 3×3 卷積分支：

$$
F_{\text{cls}} = \text{Conv}_{3\times3}\big(\text{Conv}_{3\times3}(F)\big), \quad F_{\text{reg}} = \text{Conv}_{3\times3}\big(\text{Conv}_{3\times3}(F)\big)
$$

三組輸出：

$$
\hat{p}_{\text{cls}} = \text{sigmoid}\big(\text{Conv}_{1\times1}(F_{\text{cls}})\big) \in \mathbb{R}^{H \times W \times K}
$$

$$
\hat{p}_{\text{obj}} = \text{sigmoid}\big(\text{Conv}_{1\times1}(F_{\text{reg}})\big) \in \mathbb{R}^{H \times W \times 1}
$$

$$
(t_x, t_y, t_w, t_h) = \text{Conv}_{1\times1}(F_{\text{reg}}) \in \mathbb{R}^{H \times W \times 4}
$$

$K$ 為類別數。分類、回歸拆開的動機：分類是語意任務、回歸是空間任務，兩者需要的特徵表示方向有衝突，拆開後收斂速度明顯加快、精度提升，代價是參數量與運算量略增。

### Loss

$$
\mathcal{L} = \frac{\mathcal{L}_{\text{cls}} + \lambda \, \mathcal{L}_{\text{reg}}}{N_{\text{pos}}} + \frac{\mathcal{L}_{\text{obj}}}{N_{\text{total}}}
$$

$N_{\text{pos}}$：正樣本數；$N_{\text{total}}$：所有預測格數；$\lambda$ 通常設 5。

$\mathcal{L}_{\text{cls}}$、$\mathcal{L}_{\text{obj}}$ 為 BCE：

$$
\text{BCE}(p, y) = -\big[y \log p + (1-y)\log(1-p)\big]
$$

$\mathcal{L}_{\text{reg}}$ 為 IoU loss（只對正樣本計算）：

$$
\mathcal{L}_{\text{IoU}} = 1 - \frac{|B_{\text{pred}} \cap B_{\text{gt}}|}{|B_{\text{pred}} \cup B_{\text{gt}}|}
$$

---

## Anchor-free 解碼公式

不再預先定義一堆 anchor box，每個 grid cell 只預測一組 bbox（中心點偏移 + 寬高），大幅減少預測數量（少了約 2/3），也不需要針對資料集手動聚類 anchor 尺寸。這對小物件是實質幫助，因為 anchor-based 設計常常因為 anchor 尺寸沒對準小物件而漏檢；anchor-free 用「中心點落在哪個 grid」直接判斷正負樣本，不受 anchor 形狀限制。

與 anchor-based（YOLOv3~v5）差異最直接之處：不再乘上預定義 anchor 寬高，直接用 stride 當唯一尺度基準。

對 grid 座標 $(i,j)$、stride $s$：

$$
b_x = (t_x + i) \times s, \qquad b_y = (t_y + j) \times s
$$

$$
b_w = e^{t_w} \times s, \qquad b_h = e^{t_h} \times s
$$

中心點不經過 sigmoid 限制在 0~1 再加 grid（YOLOv3 做法），YOLOX 直接回歸原始偏移量 $t_x, t_y$。少了 anchor 中介變數，每格只輸出一組框，預測數量降至約三分之一，也是收斂加快的原因之一。

---

## SimOTA 標籤分配

傳統作法用 IoU 門檻硬性判斷哪些預測框算正樣本，SimOTA 把這件事變成一個最佳運輸（optimal transport）問題：每個 ground truth 依照自己的大小動態決定要分配幾個正樣本（dynamic top-k），再用分類 loss + 回歸 loss 的組合成本去挑最匹配的預測位置。這能避免固定門檻對小物件或極端長寬比物件不友善的問題，因為門檻是動態算出來的，不是寫死的常數。

把標籤分配轉為最佳化問題，是完整 Optimal Transport（需 Sinkhorn-Knopp 迭代求解）的簡化版。

### 候選區域

對每個 ground truth $g_j$，先框出中心先驗區域：預測中心點落在 $g_j$ 框內，或落在以 $g_j$ 中心為圓心、半徑 $2.5s$ 的區域內，才有資格當候選正樣本。

### 配對成本

$$
c_{ij} = \mathcal{L}_{\text{cls}}(\hat{p}_i, g_j) + \lambda_c \, \mathcal{L}_{\text{reg}}(\hat{b}_i, g_j)
$$

### 動態 $k$ 值估計

取跟 $g_j$ IoU 最高的前 $q$ 個候選（例如 $q=10$），加總 IoU 後取整數作為該 gt 應分配的正樣本數：

$$
k_j = \text{round}\left(\sum_{i=1}^{q} \text{IoU}(\hat{b}_i, g_j)\right)
$$

物件面積大、與預測框普遍重疊度高的 gt，$k_j$ 自動變大；小物件因候選框 IoU 普遍偏低，$k_j$ 自動縮小，避免把品質不好的預測硬塞進正樣本。

### 分配規則

對每個 $g_j$，在候選集合中挑出成本 $c_{ij}$ 最小的 $k_j$ 個預測作為正樣本。若同一預測被多個 gt 選中，保留成本較小的配對，其餘視為負樣本。

「Sim」代表用動態 top-k 取代精確 OT 求解，犧牲一點最優性換取訓練速度。

---

## 小結：對小物件偵測的意義

- PAFPN 雙向融合讓淺層高解析度特徵不被單向丟棄
- Anchor-free 不受預定義 anchor 尺寸限制，避免小物件因 anchor 不匹配而漏檢
- SimOTA 動態 $k_j$ 讓小物件不會被固定 IoU 門檻排除在正樣本外

相關筆記：[[license-plate-recognition]]
