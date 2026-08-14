**Distribution Focal Loss (DFL)** 是一種專門用於目標檢測（Object Detection）中邊界框迴歸（Bounding Box Regression）的損失函數。它最早在 _Generalized Focal Loss (GFL)_ 論文中被提出，近年來在 YOLOv8、YOLOv11 等主流的 Anchor-Free 偵測模型中被廣泛採用。

## 為什麼需要 DFL？（為解決什麼痛點？）

在傳統的目標檢測中，邊界框迴歸通常被建模為一個**確定性的狄拉克 δ 分布（Dirac delta distribution）**。也就是說，模型直接預測一個連續的數值（例如利用 L1、Smooth L1 或是 IoU 損失 逼近 Ground Truth 座標 $y$）。

然而，這種作法忽略了以下問題：

1. **邊界的模糊性（Ambiguity）**：許多物體的邊界因為遮擋、陰影、光照、甚至是標記誤差（Label Noise），其實是**不確定、模糊**的。
    
2. **缺乏不確定性的表達能力**：強行要模型預測一個絕對精確的單點坐標，會導致模型在面對模糊邊界時難以收斂或產生不穩定的預測。
    ![[Pasted image 20260716173947.jpg]]
    GFL 論文圖示：模糊邊界的預測分布較平緩，清晰邊界則集中且尖銳. 資料來源：Less is More
DFL 的核心思想是：**不要直接預測一個連續值，而是去預測一個離散的「機率分布（Probability Distribution）」**。

## DFL 的運作機制

為了讓模型預測分布，DFL 將原本的迴歸問題轉化為**分類問題**。

### 1. 將空間離散化（Discretization）

假設邊界框坐標的預測範圍是 $[y_0, y_n]$（在 YOLO 中通常為 `reg_max`，例如 $[0, 16]$ 的區間）。我們將這個區間切成離散的整數點 $y_i \in \{y_0, y_1, \dots, y_n\}$。

模型對每個座標不再只輸出 1 個實數，而是輸出 $n+1$ 個值，並透過 Softmax 得到在這些離散點上的機率分布 $\{P(y_0), P(y_1), \dots, P(y_n)\}$。

最終的預測座標 $\hat{y}$ 則是這些點的**期望值（Expectation）**：

$$\hat{y} = \sum_{i=0}^{n} P(y_i) \cdot y_i$$

### 2. 引入 Focal Loss 思想

如果僅僅使用一般的期望值迴歸，模型可能會產生「多峰分布」（即雖然期望值剛好在 $y$，但預測出來的機率分布卻分散在很遠的兩個峰，這不符合邊界定位的物理意義）。

我們希望**機率分布能高度集中在 Ground Truth $y$ 的周邊**。因此，DFL 選擇 $y$ 左右相鄰的兩個離散整數點 $y_i$ 與 $y_{i+1}$（滿足 $y_i \le y \le y_{i+1}$），並強制模型**快速將機率聚焦在這兩點上**。

其公式定義為：

$$DFL(S_i, S_{i+1}) = - \left( (y_{i+1} - y)\log(S_i) + (y - y_i)\log(S_{i+1}) \right)$$

其中：

- $S_i = P(y_i)$ 且 $S_{i+1} = P(y_{i+1})$，為 Softmax 後的機率。
    
- $(y_{i+1} - y)$ 與 $(y - y_i)$ 作為距離權重，距離 $y$ 越近的整數點，被分配到的機率權重應該越高。
    

這本質上就是一種**加權的交叉熵損失（Weighted Cross Entropy Loss）**，促使網路將機率分布的「峰值」向 Ground Truth $y$ 靠攏。

## 程式碼實現（PyTorch 概念）

在 YOLO 中，DFL 通常實作為對分類特徵圖進行交叉熵計算。以下為簡化的實作邏輯：

Python

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistributionFocalLoss(nn.Module):
    def __init__(self):
        super(DistributionFocalLoss, self).__init__()

    def forward(self, pred_dist, target):
        """
        pred_dist: 預測的分布, 形狀為 (N, reg_max + 1)
        target: 真實連續座標, 形狀為 (N,)
        """
        # 尋找 target 左右鄰近的離散整數點
        target_left = target.long()          # y_i
        target_right = target_left + 1       # y_{i+1}
        
        # 計算權重 (離目標越近，權重越大)
        weight_left = target_right - target  # y_{i+1} - y
        weight_right = target - target_left  # y - y_i
        
        # 計算加權的 Cross Entropy Loss
        loss_left = F.cross_entropy(pred_dist, target_left, reduction='none')
        loss_right = F.cross_entropy(pred_dist, target_right, reduction='none')
        
        loss = loss_left * weight_left + loss_right * weight_right
        return loss.mean()
```

## 總結

- **優勢**：在遮擋、邊界模糊或小目標的場景下，DFL 能大幅提升邊界框的定位精確度（Localization Accuracy），提供更合理的置信度資訊。
    
- **搭配**：在實務上，DFL 通常與 **CIoU / GIoU 損失** 搭配使用，前者優化整體的重疊面積，後者（DFL）精細雕琢邊界的精準度。