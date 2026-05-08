`nn.CrossEntropyLoss` 是 PyTorch 中處理分類問題最核心的損失函數。它實際上是將 `nn.LogSoftmax()` 和 `nn.NLLLoss()` (Negative Log Likelihood Loss) 包裝在一起的一個方便介面。

以下是針對開發與實作時需要注意的幾個核心要點：

---

### 1. 數學原理

`nn.CrossEntropyLoss` 執行的運算邏輯如下：

1. **Softmax**: 將 Raw Logits 轉換為機率分佈。
    
2. **Log**: 對機率取對數。
    
3. **NLLLoss**: 計算預測機率與真實標籤（Ground Truth）之間的距離。
    

對於單一樣本，其公式為：

$$loss(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right) = -x[class] + \log\left(\sum_j \exp(x[j])\right)$$

### 2. 輸入與輸出規範

在使用時，最常遇到的問題通常與 Tensor 的維度（Shape）有關：

- **Input (Logits)**: 模型最後一層**不需要**加 Softmax，直接傳入原始的預算分數。
    
    - 維度：$(N, C)$，其中 $N$ 是 Batch Size，$C$ 是類別數量。
        
    - 如果是電腦視覺中的語義分割（Semantic Segmentation），維度則為 $(N, C, d_1, d_2, ..., d_k)$。
        
- **Target (Labels)**:
    
    - **類別索引模式**：維度為 $(N)$，內容是型別為 `torch.long` 的類別索引（值在 $0$ 到 $C-1$ 之間）。
        
    - **機率分佈模式**（PyTorch 1.10+）：維度為 $(N, C)$，內容是每個類別的機率（可用於 Label Smoothing）。
        

---

### 3. 重要參數說明

這個類別有幾個強大的參數，在處理不平衡數據（Imbalanced Data）時非常實用：

|**參數**|**說明**|
|---|---|
|**`weight`**|傳入一個 1D Tensor，手動為每個類別指定權重。對於樣本數較少的類別，給予較高權重可提升模型重視度。|
|**`ignore_index`**|指定一個目標值，計算損失時會忽略該標籤。常用於 Padding 填充處（例如 NLP 中的 `<PAD>`）。|
|**`reduction`**|決定輸出的形式：`mean` (預設，取平均)、`sum` (加總) 或 `none` (回傳每個樣本的 Loss Tensor)。|

---

### 4. 程式碼範例 (PyTorch)

``` python
import torch
import torch.nn as nn

# 假設有 3 個類別，Batch size 為 2
loss_fn = nn.CrossEntropyLoss()

# 模型輸出的 Raw Logits (未經過 Softmax)
input_logits = torch.randn(2, 3, requires_grad=True)

# 真實標籤 (索引形式)
target = torch.tensor([1, 0], dtype=torch.long)

output = loss_fn(input_logits, target)
output.backward()

print(f"Logits:\n{input_logits}")
print(f"Loss: {output.item()}")
```

### 5. 常見坑點提醒

- **不要重複過 Softmax**：如果你在模型最後一層加了 `nn.Softmax`，再丟進 `nn.CrossEntropyLoss`，會導致數值計算不穩定且梯度計算結果與預期不符。