
在處理二元分類（Binary Classification）問題時，處理數值穩定性（Numerical Stability）是至關重要的。`BCEWithLogitsLoss` 是 PyTorch 中將 `Sigmoid` 層與 `BCELoss`（Binary Cross Entropy）整合在一起的損失函數。

### 核心原理

這個損失函數直接接受模型的原始輸出（Logits），而不是經過激活函數後的機率值。

其數學公式如下：

$$L = -[y \cdot \log(\sigma(x)) + (1 - y) \cdot \log(1 - \sigma(x))]$$

其中 $\sigma(x)$ 是 Sigmoid 函數：

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

---

### 為什麼不分開用 Sigmoid + BCELoss？

在實務開發中，我們強烈建議使用 `BCEWithLogitsLoss`，原因有二：

1. **數值穩定性（Numerical Stability）**：
    
    當 Logits 值非常大或非常小時，先做 Sigmoid 再取 Log 容易產生浮點數溢位或下溢（Overflow/Underflow）。`BCEWithLogitsLoss` 利用了 **Log-Sum-Exp 技巧** 進行優化，計算過程更穩定。
    
2. **效能**：
    
    將兩個運算合併在一個運算子（Operator）中，減少了中間張量的產生，運算速度稍微快一些。
    

---

### 關鍵參數：`pos_weight`

這是 AI 工程師在處理**類別不平衡（Class Imbalance）**時的利器。如果你發現正樣本（Positive samples）太少，可以透過 `pos_weight` 來增加正樣本的損失權重。

- **用法**：`criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))`
    
- **效果**：這會讓模型在答錯正樣本時，受到的懲罰是負樣本的 5 倍。
    

---

### 程式碼範例

```python
import torch
import torch.nn as nn

# 假設 Batch Size = 3
input_logits = torch.randn(3, requires_grad=True) 
target = torch.empty(3).random_(2) # 二元標籤：0 或 1

criterion = nn.BCEWithLogitsLoss()
loss = criterion(input_logits, target)

loss.backward()

print(f"Logits: {input_logits}")
print(f"Loss: {loss.item()}")
```

---

### 總結比較

|**特性**|**BCELoss**|**BCEWithLogitsLoss**|
|---|---|---|
|**輸入內容**|機率值 (0~1)|原始 Logits (-inf ~ inf)|
|**內建激活**|無|內建 Sigmoid|
|**穩定性**|較差（易梯度消失或溢位）|**優（推薦使用）**|
|**類別權重**|僅支援 `weight`|支援 `weight` 與 `pos_weight`|