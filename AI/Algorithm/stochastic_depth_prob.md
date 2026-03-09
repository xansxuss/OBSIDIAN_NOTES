`stochastic_depth_prob`（隨機深度機率）是訓練深層神經網路（如 ResNet 或 Vision Transformers）時一種非常有效的正規化（Regularization）技術，最早由 Gao Huang 等人在 2016 年提出。

簡單來說，它的核心思想是在訓練期間**隨機丟棄（Drop）整個卷積層或殘差區塊（Residual Blocks）**，但在測試時使用完整的網路。

---

## 運作機制與數學表達

在殘差網路中，一個典型的區塊輸出可以表示為：

$$H_l = \text{ReLU}(b_l \cdot f_l(H_{l-1}) + H_{l-1})$$

其中 $b_l$ 是一個伯努利隨機變數（Bernoulli random variable）：

- **訓練階段：** $b_l$ 以機率 $p_l$ 取值為 1（保留該層），以 $1 - p_l$ 取值為 0（跳過該層，直接恆等映射）。這裡的 $1 - p_l$ 就是你提到的 `stochastic_depth_prob`。
    
- **測試階段：** 為了保持數值期望值一致，每一層的輸出會根據存活機率進行縮放：$H_l = \text{ReLU}(p_l \cdot f_l(H_{l-1}) + H_{l-1})$。
    

### 機率分佈策略

通常 `stochastic_depth_prob` 不是全域固定的一個數值，而是採用 **Linear Decay（線性遞減）** 策略：

- **淺層：** 掉機率較低（接近 0），因為淺層特徵對後續提取至關重要。
    
- **深層：** 掉機率較高（例如 0.5），深層冗餘性較高，丟棄有助於泛化。
    

---

## 為什麼要用 Stochastic Depth？

1. **解決梯度消失：** 縮短了訓練時的反向傳播路徑，讓梯度更容易傳回淺層。
    
2. **訓練加速：** 訓練時因為跳過了部分運算，實際上的計算量會減少（類似 Dropout，但作用於層級）。
    
3. **強大的正規化：** 防止模型過度依賴某些特定的層，這在 ViT (Vision Transformer) 這種動輒數百層的架構中幾乎是標配，能顯著降低過擬合。
    
4. **模型集成效果：** 論文指出，這等同於在訓練過程中隱式地訓練了多個不同深度的模型組合。
    

---

## 工程實作參考 (PyTorch)

在 `torchvision` 或 `timm` 庫中，這通常被稱為 `drop_path`。如果你在實作 Transformer block，代碼邏輯大約如下：

```python
import torch
import torch.nn as nn

class StochasticDepth(nn.Module):
    def __init__(self, prob: float):
        super().__init__()
        self.prob = prob  # 這是掉層的機率 (drop probability)

    def forward(self, x):
        if not self.training or self.prob == 0.0:
            return x
        
        keep_prob = 1 - self.prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        # 產生 mask 並縮放以維持期望值
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = random_tensor.floor()
        return x.div(keep_prob) * binary_mask
```

---

## 常見參數建議

- **對於小型模型：** `stochastic_depth_prob` 通常設為 **0.1** 左右。
    
- **對於大型 Vision Transformer (ViT-L/H)：** 通常會設到 **0.4 或 0.5**。