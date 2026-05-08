#### 馬氏距離 = 考慮變異方向和尺度的距離。

### 它到底怎麼算？

對某個資料點 **x** 與均值 **μ**：
$$
D_M(x)=\sqrt{(x-\mu)^\top \Sigma^{-1}(x-\mu)}​
$$ Σ（Sigma）是共變異矩陣。  
Σ 的逆矩陣（Σ⁻¹）其實在告訴你：  
哪些方向的資料比較「大範圍、噪音多」，哪些方向很「敏感」。

### 馬氏距離的直覺理解

1. **考慮方向差異**  
    如果資料在某方向分散很大，那方向的距離會被縮小，因為大家都偏成那樣，算正常。
2. **對相關性敏感**  
    如果資料在 x 與 y 維度高度相關，馬氏距離會沿著「真正偏離的方向」計算，而不是像歐幾里得距離那樣亂猜。
3. **挑異常值超好用**  
    它非常擅長偵測 outlier。  
    因為 outlier 在具統計性描述的資料空間裡，就是會被它抓出來。

---

### 為什麼電腦視覺和異常偵測超愛它？

因為它等於給你一把「資料分布等級的尺」。像 PaDiM、FastFlow、一些 VAE-based anomaly detection 都經常用它來判斷：  
這個像素、這個特徵向量到底「正常」還「怪」。

例如 PaDiM 就是拿每個位置的多維高斯分布，再用馬氏距離判斷該位置是否為異常。

### Python NumPy實作

``` python
import numpy as np

def mahalanobis(x, mean, cov):
    diff = x - mean
    inv_cov = np.linalg.inv(cov)
    return np.sqrt(diff.T @ inv_cov @ diff)

# 多維特徵示例
x = np.array([1.2, 2.3])
mean = np.array([1.0, 2.0])
cov = np.array([[1.0, 0.2],
                [0.2, 0.5]])

print(mahalanobis(x, mean, cov))
```

