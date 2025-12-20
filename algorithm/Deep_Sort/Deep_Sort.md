## 1️⃣ Deep SORT 背景與定位

Deep SORT 是 **多目標追蹤 (MOT, Multi-Object Tracking)** 的經典演算法之一，是對 **SORT** 的改進版本：

- **SORT (Simple Online and Realtime Tracking)**：基於卡爾曼濾波器 + 匈牙利演算法做目標追蹤，速度快但缺乏強魯棒的外觀描述能力。
- **Deep SORT**：在 SORT 的基礎上加入了 **深度學習的外觀特徵 (Appearance Feature)**，大幅改善長時間遮擋或交叉情況下的 ID 保持能力。

核心目標：**在影片序列中為每個目標分配穩定且唯一的 ID，同時能線上處理**。

---

## 2️⃣ Deep SORT 的主要組件

Deep SORT 可以分為三個核心模組：

### (1) **檢測器 (Detector)**

- 輸入：單張影像。
- 輸出：多個 bounding box（位置信息） + 分數（confidence）。
- 常用 YOLOv4/v5/v8 或 Faster R-CNN 作檢測。
- Deep SORT 不做檢測，它只使用檢測結果。    

---

### (2) **外觀特徵提取 (Appearance Feature Extraction)**

- 使用 **卷積神經網路 (CNN)** 提取每個 bounding box 的外觀特徵向量。
- 網路通常經過行人 re-ID 或物件 re-ID 訓練。
- 特徵向量一般經過 L2 正規化：
	$f = \frac{f}{\|f\|_2}$
- 這使得相似目標在特徵空間的距離更小，便於後續匹配。

### (3) **卡爾曼濾波器 (Kalman Filter)**

- 用於追蹤每個目標的 **動態位置**。
    
- 狀態向量一般定義為：
    
    $x = [u, v, \gamma, h, \dot{u}, \dot{v}, \dot{\gamma}, \dot{h}]^T$
    - $u$,$v$：bbox 中心座標
    - $\gamma = w/h$：長寬比
    - $h$：高度
    - $\dot{u}, \dot{v}, \dot{\gamma}, \dot{h}$：速度
- 卡爾曼濾波器預測下一幀的位置，提供 **預測 bbox**。

### (4) **資料關聯 (Data Association)**

- Deep SORT 使用 **兩階段匹配 (two-stage matching)**：

1. **外觀距離匹配 (Appearance Distance Matching)**
    - 計算 **餘弦距離或歐氏距離**：
	    $d_{\text{appearance}}(i,j) = 1 - \frac{f_i \cdot f_j}{\|f_i\| \|f_j\|}$
    - 對於可信度高的匹配優先使用。
2. **運動距離匹配 (Motion/Gating Matching)**
    - 使用卡爾曼濾波器預測位置。
    - 只允許合理範圍內的 bbox 匹配（gating）：
	    $d_{\text{motion}}(i,j) = (z_j - \hat{z}_i)^T S_i^{-1} (z_j - \hat{z}_i)$

- 匹配算法：**匈牙利演算法 (Hungarian Algorithm)**，解決最佳一對一分配。

### (5) **軌跡管理 (Track Management)**

每個追蹤物件被稱為一個 **Track**，有以下狀態：

|狀態|說明|
|---|---|
|Tentative|新生成的 track，需要連續匹配幾幀才能確認|
|Confirmed|已確認有效 track|
|Deleted|長時間未匹配，刪除 track|

- **閾值**：
    - `max_age`：允許多少幀沒有匹配
    - `n_init`：新 track 需要連續幀數才變成 confirmed

---

## 3️⃣ Deep SORT 流程總結

整個 Deep SORT pipeline 可以用文字流程描述：

1. **檢測器**檢測當前影像 → 得到 bbox 與置信度。
2. **特徵提取器**計算每個 bbox 的 appearance feature。
3. **卡爾曼濾波器**預測每個 track 的下一幀位置。
4. **第一輪匹配**：
    - 使用 appearance distance + gating 進行匈牙利匹配
    - 匹配成功 → 更新 track
5. **第二輪匹配**：
    - 對未匹配的 detections 使用 IoU 與 motion gating 匹配未匹配 track
6. **軌跡管理**：
    - 新 track 建立 tentative
    - 長時間未匹配的 track 標記為 deleted
7. 回到下一幀，重複步驟 1～6。

---

## 4️⃣ Deep SORT 的核心優勢

1. **長時間遮擋可維持 ID**：因為引入 appearance feature。
2. **線上即時處理**：每幀只需一次卡爾曼更新 + 匈牙利匹配。
3. **靈活**：可替換檢測器和外觀特徵模型。

---

## 5️⃣ 總結公式與關鍵概念

- **卡爾曼濾波器狀態**：
    $xt∣t−1​=Fxt−1∣t−1$
    $P_{t|t-1} = F P_{t-1|t-1} F^T + Q$
- **餘弦距離**：
    $d_{\text{cos}}(f_i, f_j) = 1 - \frac{f_i \cdot f_j}{\|f_i\| \|f_j\|}$​
- **運動 gating (Mahalanobis)**：
    
    $d_{\text{motion}}(i,j) = (z_j - \hat{z}_i)^T S_i^{-1} (z_j - \hat{z}_i)$
- **匈牙利匹配**：
    - 求解成本矩陣最小總和分配
    - 可保證一對一匹配

## Deep SORT 流程圖
![[Deep_SORT流程圖.png]]