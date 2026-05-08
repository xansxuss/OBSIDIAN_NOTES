**Laurens van der Maaten 的 t-SNE** 是一個很有意思的東西。表面看起來像「把高維資料壓扁到 2D」，但實際上它做的事情更像是：

> 在低維空間裡重新安排點的位置，讓**鄰居關係**盡量跟高維世界一樣。

這個概念很接近物理模擬。想像每個資料點都是粒子，彼此之間有吸引力和排斥力。

---

## 直覺版：高維世界的「誰跟誰像」

假設你有一堆高維向量，比如：

- 人臉 embedding（512 維）
    
- CLIP image embedding（512 或 768 維）
    
- word embedding（300~1000 維）
    

在高維空間裡，距離近代表語義接近。

t-SNE 做的第一件事是：

把「距離」轉成 **機率**

意思是：

某個點 iii 把點 jjj 當鄰居的機率是多少。

核心公式是高斯分佈：

$$
p_{j|i} =
\frac{
\exp\left(-\frac{\|x_i-x_j\|^2}{2\sigma_i^2}\right)
}{
\sum_{k \ne i}
\exp\left(-\frac{\|x_i-x_k\|^2}{2\sigma_i^2}\right)
}
$$

意思：

距離越近 → 機率越高。

但裡面藏了幾個很有意思的幾何機關。

---

## 1. Gaussian kernel：距離 → 相似度

先看分子：

$$\exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma_i^2}\right)$$

這其實就是 **Gaussian kernel**。

直覺是：
```
距離 = 0      →  exp(0) = 1  
距離變大      →  exp(-something) → 越來越小  
距離很遠      →  接近 0
```
所以你可以把它想成：

相似度 = exp(-distance²)

像一個**模糊光暈**：

```
       xi  
      ●  
    ●  ●  
  ●      ●
```

越靠近 xix_ixi​ 的點，亮度越高。

---

## 2. 為什麼要除以分母？

分母

$$
\sum_{k \ne i} \exp\left(-\frac{\|x_i - x_k\|^2}{2\sigma_i^2}\right)
$$

只是做一件很普通但很重要的事：

**Normalization（正規化）**

讓所有機率加起來 = 1

$$
\sum_j p_{j|i} = 1
$$

所以它其實就是：

```
p(j|i) = similarity(i,j) / total_similarity(i,*)
```

也就是：

**「在所有鄰居裡，j 的相對權重是多少」**

---

## 3. σ_i 是真正的魔法

這裡有個非常精妙的設計：

$$
\sigma_i
$$

**每個點自己的 σ 都不同。**

為什麼？

因為資料密度不同。

想像兩種情況：

### 稀疏區域
```
x_i         x_j  
 ●-----------●
```

距離很遠  
如果 σ 太小

```
exp(-distance²) ≈ 0
```

整個機率會崩掉。

所以 **σ 必須變大**。

---

### 密集區域

```
● ● ● ●  
  xi
```
點很多很近  
如果 σ 太大

所有人都像鄰居。

所以 **σ 必須變小**。

---

### t-SNE 的做法

它會調整 σ 讓 **perplexity 固定**。

perplexity 大概可以理解成：

有效鄰居數

例如：

perplexity = 30

意思是

> 每個點大概有 **30 個鄰居**。

---

## 4. 這個公式其實在做一件更深的事

它不是單純算距離。

它在建構一個 **機率圖（probability graph）**：

xi → xj : p(j|i)

這其實是：

**高維 manifold 的局部幾何**

換句話說：

t-SNE 不在意

xi 和 xj 的實際距離

它只在意：

xi 覺得誰是鄰居

---

## 5. 為什麼 t-SNE 會形成 cluster

因為接下來它會做另一件事：

低維空間用 **Student-t distribution**

而不是 Gaussian。

也就是：

高維

Gaussian

低維

1 / (1 + distance²)

這會產生 famous 的：

**crowding problem solution**

結果就是：

同群 → 緊密  
不同群 → 拉很開

---

## 6. 一個很少人講的直覺

這個公式其實可以這樣理解：

每個點都在問：

> 「在我眼裡，誰是朋友？」

而 t-SNE 在做的事情是：

> 找一個 2D 空間  
> 讓大家的「朋友圈」盡量保持一樣。

所以最後的圖不是「距離圖」。

而是：

**社交網路圖。**

---

如果繼續往下挖，會連到一個很漂亮的 rabbit hole：

- t-SNE
    
- UMAP
    
- Riemannian manifold
    
- Information Geometry
    

最後你會發現一件很瘋的事：

**embedding 空間其實是在學資料的「曲率」。**

而這剛好跟你前面聊的  
sphere packing、語義 embedding、face embedding

全部會在同一條數學宇宙線上交會。

那條線的名字其實叫：

**高維幾何。**



---

## 然後事情開始變魔法

現在你把所有點丟到 **2D 空間**。

這時候重新定義一個「鄰居機率」。

但這次 **不用 Gaussian**，而是用 **Student-t 分布**：

$$
q_{ij} =
\frac{
(1 + \|y_i - y_j\|^2)^{-1}
}{
\sum_{k \ne l}
(1 + \|y_k - y_l\|^2)^{-1}
}
$$

為什麼？

因為 Student-t 有 **heavy tail**。

意思是：

遠距離的點不會被壓太近。

這是 t-SNE 能分出 cluster 的關鍵。

稍微拆一下直覺，這個式子其實在做：

distance → similarity

但和高維 Gaussian 不同：

高維：

exp(-distance²)

低維：

1 / (1 + distance²)

差別非常關鍵。

Gaussian 衰減：

distance ↑  
similarity → 0 (很快)

Student-t 衰減：

distance ↑  
similarity ↓ (但很慢)

所以遠距離點會有 **更強的排斥力**。  
這就是為什麼 t-SNE 的 cluster 會被「拉開」。

有趣的是，如果你把這些公式放到 **embedding 空間 / face embedding / CLIP embedding** 的角度看，它其實是在做一件很像 **局部流形幾何（manifold geometry）** 的事情。

換句話說：  
t-SNE 並不是單純降維，而是在試圖讓 **高維鄰居關係的幾何結構**在低維保留下來。

---

## 最後一步：優化

t-SNE 做的事情其實很單純：

最小化

$$
KL(P \,\|\, Q) = \sum_{i \ne j} p_{ij} \, \log \frac{p_{ij}}{q_{ij}}
$$
也就是：

$$
KL(P \,\|\, Q) = \sum_{i,j} p_{ij} \, \log \frac{p_{ij}}{q_{ij}}
$$

意思：

低維空間的鄰居機率 QQQ  
要盡量模仿高維空間的 PPP。

所以整個演算法其實是：

1️⃣ 算高維鄰居機率  
2️⃣ 隨機放到 2D  
3️⃣ gradient descent  
4️⃣ 讓鄰居關係對齊

### 直覺理解

1. pijp_{ij}pij​ → 高維空間的鄰居機率
    
2. qijq_{ij}qij​ → 低維空間的鄰居機率
    

KL 散度做的事情其實是：

> **「低維空間的點排列，跟高維空間的鄰居分布差多少？」**

- KL 越小 → 低維排列越忠實保留高維結構
    
- KL 越大 → 排列失真，鄰居關係被破壞
    

---

### t-SNE 使用這個公式的原因

- 高維用 Gaussian kernel → pijp_{ij}pij​
    
- 低維用 Student-t kernel → qijq_{ij}qij​
    
- 用 KL(P||Q) 來做梯度下降，把低維空間拉成 **保留局部鄰居的 2D/3D 視覺化圖**
    

所以整個 t-SNE 核心就是這三個公式的組合：

1. 高維鄰居分布 pijp_{ij}pij​
    
2. 低維鄰居分布 qijq_{ij}qij​
    
3. KL 散度 KL(P∣∣Q)KL(P||Q)KL(P∣∣Q) → 最終損失函數
    

---

---

## 為什麼 t-SNE 會出現漂亮 cluster？

因為它 **非常偏心 local structure**。

簡單講：

- 很在意「近鄰」
    
- 不太在意「遠距離」
    

所以會看到：

cluster cluster cluster

但 cluster 之間距離其實 **沒有物理意義**。

很多人會誤解：

> cluster A 比 cluster B 靠近 cluster C

這通常是假的。

---

## 一個很酷的幾何視角

如果你做的是：

- face embedding
    
- CLIP embedding
    
- language embedding
    

通常會先做

$$
\|x\| = 1
$$

也就是 **L2 normalization**。

這代表：

所有點都在

**高維球面上**

這時候 t-SNE 其實是在做：

> **sphere manifold → 2D map**

這就是為什麼：

- embedding
    
- sphere packing
    
- manifold learning
    

常常會在同一篇論文裡一起出現。

---

## 為什麼很多人現在改用 UMAP

t-SNE 有幾個問題：

1️⃣ 很慢  
2️⃣ global structure 會壞掉  
3️⃣ 每次跑結果不同

所以現在很多人用  
UMAP

UMAP 的哲學其實更幾何一點：

> 假設資料在一個 **Riemannian manifold**

然後把拓撲結構保留下來。

---

## 一個 AI 工程很少人注意的事

如果你的 embedding 是：

- ArcFace
    
- CLIP
    
- FaceNet
    

其實很多時候你看到的 t-SNE cluster

不是「模型學到的」

而是

**sphere geometry 本身造成的**。

高維球面本來就很容易出現 cluster illusion。

這也是為什麼你之前提到：

- 512 維 embedding
    
- hypersphere
    
- sphere packing
    

那條 rabbit hole 其實會一路挖到：

- **E8 lattice**
    
- **Leech lattice**
    
- error-correcting codes
    
- communication theory
    

看起來是不同領域，但其實都在解同一個問題：

> **高維空間如何放最多彼此分離的向量。**

這件事跟 embedding 的可分性、容量、泛化能力都有微妙關係。

---

最後留一個有點科幻、但很合理的想法：

如果語義空間真的接近  
**低曲率高維流形**，

那麼像 t-SNE / UMAP 其實只是

在畫一張

**宇宙地圖的投影圖**。

就像把地球攤平成世界地圖一樣。  
地圖很好看，但地球其實是彎的。