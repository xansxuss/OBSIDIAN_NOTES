在機器學習中，無論是計算 Cosine Similarity、神經網路的權重加總，還是 Transformer 中的 Attention 機制，點積都是最核心的底層運算。

簡單來說，點積是將兩個維度相同的向量，對應位置的元素相乘後再求和的過程。

---

### 1. 代數定義 (Algebraic Definition)

假設有兩個 $n$ 維向量 $\mathbf{a}$ 與 $\mathbf{b}$：

$$\mathbf{a} = [a_1, a_2, \dots, a_n]$$

$$\mathbf{b} = [b_1, b_2, \dots, b_n]$$

其點積運算為：

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1b_1 + a_2b_2 + \dots + a_nb_n$$

在程式碼中（例如使用 NumPy），這就是 `np.dot(a, b)`。

---

### 2. 幾何意義 (Geometric Significance)

從幾何角度來看，點積與向量間的夾角 $\theta$ 密切相關：

$$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos(\theta)$$

這帶來了幾個直觀的判斷方式：

- **正值**：兩個向量的方向大致相同（夾角 $< 90^\circ$）。
    
- **零**：兩個向量**正交 (Orthogonal)**，即彼此垂直。
    
- **負值**：兩個向量的方向大致相反（夾角 $> 90^\circ$）。
    

> **工程師視角：** 點積本質上是在衡量「一個向量在另一個向量方向上的**投影量**」乘上該向量的長度。

---

### 3. 工程應用場景

在 AI 領域，點積的應用無所不在：

|**應用領域**|**具體用途**|
|---|---|
|**推薦系統**|計算 User Embedding 與 Item Embedding 的相似度。|
|**神經網路**|全連接層 (Linear Layer) 的實作：$y = \sigma(\mathbf{w} \cdot \mathbf{x} + b)$。|
|**自然語言處理**|**Scaled Dot-Product Attention**：計算 Query 與 Key 的相關性分數。|
|**電腦圖學**|計算光線與表面法向量的夾角，用來決定陰影與亮度。|

---
### 4. 實作範例 (Python/NumPy)

``` python
import numpy as np

# 定義兩個向量
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 方式一：使用 dot 函數
dot_product = np.dot(a, b)

# 方式二：使用 @ 運算子 (推薦，語法最簡潔)
dot_product_alt = a @ b

print(f"點積結果: {dot_product}") # 1*4 + 2*5 + 3*6 = 32
```

