## 1️⃣ ByteTrack 的核心目標

ByteTrack 是一種 **高效能的線上多目標追蹤（Online MOT）演算法**，主要特點是：

1. 能夠 **追蹤所有檢測到的目標，包括低信心檢測（low-score detections）**。
2. 將 **檢測和追蹤分離**，讓追蹤更穩定、漏追率更低。
3. 使用 **簡單的匈牙利演算法 + Kalman Filter + IOU/距離匹配**，效率很高，幾乎無需額外訓練。

ByteTrack 在 MOT20/MOT17 上的表現通常比 DeepSORT 還要好，尤其是對 **低檢測信心目標** 的處理。

---

## 2️⃣ 流程拆解

ByteTrack 的流程大致可以分成以下幾個步驟：

### Step 1: 目標檢測

- 先使用 **YOLOv8、YOLOv5、或者任何檢測器**，得到當前幀的 bounding boxes 和 confidence score。
- 目標檢測結果會被分為兩類：
    1. **高信心檢測 (high-score detections)**  
        `score >= conf_thres`  
        用於主要追蹤。
    2. **低信心檢測 (low-score detections)**  
        `score < conf_thres`  
        避免直接丟掉，後續用於補充匹配。

---

### Step 2: 已存在軌跡預測

- 每一個已存在的目標軌跡會被 **Kalman Filter 預測**到下一幀的位置。
- Kalman Filter 會估算：
    - 目標中心座標 `(x, y)`
    - 寬高 `(w, h)`
    - 速度 `(vx, vy)`（可選）

---

### Step 3: 匹配高信心檢測

- **核心匹配演算法**：
    1. 計算 **IOU（Intersection over Union）或距離**矩陣  
        `cost_matrix[i, j] = 1 - IOU(track_i, detection_j)`
    2. 用 **匈牙利演算法（Hungarian Algorithm）**解決二分匹配問題。
- 匹配結果：
    - 成功匹配 → 更新 Kalman Filter，保持同一 ID
    - 無法匹配 → 標記為未匹配軌跡或未匹配檢測

---

### Step 4: 匹配低信心檢測

- 將 **尚未匹配的軌跡** 與 **低信心檢測** 再次匹配。
- 這一步是 ByteTrack 的關鍵創新：
    - **DeepSORT** 往往直接丟掉低分檢測 → 容易漏追。
    - **ByteTrack** 將低分檢測用於補充匹配 → 增加對遮擋或難檢測目標的追蹤穩定性。

---

### Step 5: 處理新增軌跡

- 對於仍然無法匹配的低信心檢測，**只有超過一定分數門檻才會生成新的軌跡**。
- 這樣避免產生太多假軌跡（false positive tracks）。

---

### Step 6: 處理消失軌跡

- 如果軌跡在多幀都沒有匹配到任何檢測，則視為 **軌跡消失**，會被刪除或標記為 inactive。
- 消失判斷通常依賴一個 **max_age** 參數（軌跡容忍幾幀未被檢測到）。

---

### Step 7: 輸出

- 每一幀最終輸出：
    `track_id, bbox(x1, y1, x2, y2), score, class_id`
- 同一個目標在多幀中保持相同 `track_id`。

---

## 3️⃣ ByteTrack 核心公式與邏輯

### 3.1 Kalman Filter 更新

假設目標狀態：

$$
x=[x,y,w,h,vx,vy]T\mathbf{x} = [x, y, w, h, vx, vy]^Tx=[x,y,w,h,vx,vy]T
$$

Kalman Filter 預測：

$$
xt∣t−1=Fxt−1∣t−1\mathbf{x}_{t|t-1} = F \mathbf{x}_{t-1|t-1}xt∣t−1​=Fxt−1∣t−1​
$$

更新：

$$
xt∣t=xt∣t−1+K(zt−Hxt∣t−1)\mathbf{x}_{t|t} = \mathbf{x}_{t|t-1} + K(z_t - H \mathbf{x}_{t|t-1})xt∣t​=xt∣t−1​+K(zt​−Hxt∣t−1​)
$$

- `z_t` 是檢測到的 bbox
- `K` 是 Kalman Gain

---

### 3.2 IOU 計算

IOU(A,B)=A∩BA∪BIOU(A, B) = \frac{A \cap B}{A \cup B}IOU(A,B)=A∪BA∩B​

---

### 3.3 匹配成本

$$
Ci,j=1−IOU(tracki,detectionj)C_{i,j} = 1 - IOU(track_i, detection_j)Ci,j​=1−IOU(tracki​,detectionj​)
$$
---

## 4️⃣ ByteTrack 的亮點

1. **Low-score detection tracking** → 減少漏追。
2. **簡單高效** → 只用匈牙利 + Kalman，不依賴 ReID 特徵。
3. **可直接套現成檢測器** → YOLOv8、YOLOv5 都能用。
4. **零訓練** → 主要是後處理演算法。

---

## 5️⃣ Python-like 流程 pseudocode

```python
high_dets, low_dets = split_detections(detections, conf_thres)
  # Step1: 預測現有軌跡位置 
for track in active_tracks:
	track.predict()  # Kalman Filter  
# Step2: 匹配高信心檢測 
matches, unmatched_tracks, unmatched_dets = match_tracks(high_dets, active_tracks)  
# Step3: 匹配低信心檢測 
matches2, unmatched_tracks, unmatched_low_dets = match_tracks(low_dets, unmatched_tracks)  
# Step4: 創建新軌跡 
for det in unmatched_low_dets:     
	if det.score > low_thres:         
		create_new_track(det)  
# Step5: 移除消失軌跡 
remove_old_tracks(max_age)
```

ByteTrack MOT pipeline diagram
![[ByteTrack MOT pipeline diagram.png]]

### IDS
ByteTrack 論文／常見評估指標中，ID Switchs（簡寫為 “IDS”）指的是「身份切換次數（identity switches 次數）」。
### ✅ 為什麼是這個意思

- 在多目標追蹤（MOT, Multi‑Object Tracking）的評估裡，一個重要指標是 “ID Switches / IDs / IDS”，用來衡量追蹤過程中同一個真實目標被錯誤地賦予不同 tracking‑ID 的次數。[MDPI](https://www.mdpi.com/2079-9292/13/15/3033?utm_source=chatgpt.com)
- “IDS = the number of identity switches that occur during the tracking process, meaning the same target is assigned different identities.” [MDPI](https://www.mdpi.com/2079-9292/13/15/3033?utm_source=chatgpt.com)
- 在原始 ByteTrack 論文裡，也把降低 “IDs” 當成它相較於傳統方法的一個主要改善重點。[ECVA](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf?utm_source=chatgpt.com)

### 🔍 在什麼情況會發生 “ID Switch”

- 當目標因遮擋、偵測分數低、遮蔽、模糊、快速移動等原因導致 detector 無法穩定偵測到同一目標時。
- 若追蹤器在之後重新偵測到該目標，但錯誤地把它當成「新目標」而非延續之前的 track，就會產生一個 identity switch（IDS +1）。
- 所以 “IDS 越低” 就代表追蹤器在維持目標 identity 穩定性上的表現越好。