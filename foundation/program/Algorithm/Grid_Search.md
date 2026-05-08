**網格搜尋 (Grid Search)**是一種「窮舉法」，透過遍歷所有給定的參數組合，來找出模型表現最好的那一組設定。

以下為 Grid Search 的核心邏輯與實務上的考量：

---

### 1. 運作原理

Grid Search 會定義一個參數的「網格」，針對每一種組合進行交叉驗證（Cross-Validation），最後回傳評估指標（如 Accuracy, F1-score）最高的參數。

假設你有兩個超參數需要調整：

- **Learning Rate:** $[0.1, 0.01]$
    
- **Batch Size:** $[16, 32, 64]$
    

Grid Search 會測試 $2 \times 3 = 6$ 種組合。如果搭配 **5-Fold Cross-Validation**，則總共會執行 $6 \times 5 = 30$ 次訓練模型與評估的過程。

---

### 2. 優缺點分析
|**優點**|**缺點**|
|---|---|
|**簡單直觀**：實作非常容易（如 Scikit-learn 的 `GridSearchCV`）。|**計算代價高**：隨著參數數量增加，組合數會呈指數級成長（維度災難）。|
|**保證找到全域最佳解**：只要範圍夠大、間隔夠細，一定能找到該網格內的最佳點。|**效率低落**：會浪費大量資源在顯然不佳的參數區域。|
|**適合平行運算**：各組實驗互相獨立，非常適合丟到叢集跑運算。|**依賴手動設定**：如果最佳解在網格範圍外，你永遠找不到。|

---

### 3. 實務建議

身為工程師，在處理大規模模型（如 LLM 或大型 CNN）時，通常不會直接無腦用 Grid Search。以下是幾種優化策略：

- **先粗後細 (Coarse-to-fine)**：第一輪先用大範圍、大間距的網格定位出大致的優質區域，第二輪再針對該區域進行精細搜尋。
    
- **搭配 Random Search**：實務經驗顯示，Random Search 在同樣的計算預算下，通常比 Grid Search 更能抓到重要參數的細微變化。
    
- **自動化調參**：如果專案預算與時間允許，建議轉向 **Bayesian Optimization** (如 Optuna 或 Ray Tune)，這類方法會根據之前的搜尋結果來決定下一步，效率遠高於 Grid Search。
    

---

### 4. Scikit-learn 快速實作範例

``` python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# 定義參數網格
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto']
}

# 宣告 GridSearchCV 物件
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# 開始練跑
grid_search.fit(X_train, y_train)

# 取得最佳參數
print(f"最佳參數: {grid_search.best_params_}")
```

