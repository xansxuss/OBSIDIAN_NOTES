## 1️⃣ STrack 的角色定位

在 ByteTrack 中，整個追蹤流程大致分三個階段：

1. **檢測（Detection）**
    
    - 使用 YOLOv5/YOLOv8 之類的物件檢測器產生邊界框 (bbox) 與 confidence score。
        
    - 分為 **高分檢測（high-score）** 與 **低分檢測（low-score）**。ByteTrack 核心就是把低分檢測納入二次匹配，提升追蹤完整性。
        
2. **匹配（Matching）**
    
    - 將現有追蹤器 STrack 與當前檢測做匹配，使用 **IoU** 或 **Kalman Filter 預測位置**。
        
    - 匹配策略：先匹配高分檢測，再將未匹配的低分檢測匹配剩餘追蹤器。
        
3. **更新（Update / Life-cycle）**
    
    - 已匹配 STrack 更新狀態、位置、速度等。
        
    - 未匹配的 STrack 增加 `age`，直到超過 `max_age` 才刪除。
        

在這個流程中，**STrack** 就是追蹤單位，每個 STrack 都包含物體歷史信息與預測能力。

---

## 2️⃣ STrack 的核心資料結構

Python 版本（簡化版）：

``` python
class STrack:
    def __init__(self, tlwh, score, cls_id=None):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)  # bbox 格式: [x, y, w, h]
        self.score = score                             # 檢測分數
        self.cls_id = cls_id                           # 類別ID，可選
        self.kalman_filter = KalmanFilter()           # 內建卡爾曼濾波器
        self.is_activated = False                     # 是否已啟用
        self.track_id = 0                              # 唯一ID
        self.frame_id = 0                              # 最近更新幀
        self.start_frame = 0                           # 初始幀
        self.age = 0                                   # 生命週期計數
        self.time_since_update = 0                     # 自上次更新經過幀數
```

### 核心欄位解釋

- `tlwh`：邊界框位置（左上 x、y + 寬、高），方便轉成 `tlbr` 進行 IoU 計算。
    
- `score`：檢測信心分數，用來做匹配排序。
    
- `kalman_filter`：用來預測下一幀 bbox。
    
- `is_activated`：用來判斷這個 STrack 是否已經進入追蹤隊列。
    
- `track_id`：唯一識別碼，整個追蹤過程中不變。
    
- `age` / `time_since_update`：用於控制生命週期與刪除。
    

---

## 3️⃣ STrack 核心方法

### (1) `predict()`

使用卡爾曼濾波器預測下一幀位置：

``` python
def predict(self):
    self.kalman_filter.predict()
    self.tlwh = self.kalman_filter.get_state()
    self.age += 1
    self.time_since_update += 1
```

> 卡爾曼濾波器預測公式：
> 
> $\mathbf{x}_{t|t-1} = F \mathbf{x}_{t-1|t-1}$
> $P_{t|t-1} = F P_{t-1|t-1} F^T + Q$
> 
> - `x`：狀態向量 (中心 x, y, 寬, 高, 速度)
>     
> - `F`：狀態轉移矩陣
>     
> - `P`：誤差共變矩陣
>     
> - `Q`：過程噪聲
>     

### (2) `update(detection)`

將檢測結果與 STrack 匹配，更新卡爾曼濾波器與生命週期：

``` python
def update(self, detection):
    self.frame_id = detection.frame_id
    self.time_since_update = 0
    self.age += 1
    self.is_activated = True

    self.kalman_filter.update(detection.tlwh)
    self.tlwh = self.kalman_filter.get_state()
    self.score = detection.score
```

- `update` 的邏輯就是 **修正預測位置**，並同步更新 `tlwh` 與 `score`。
    
- 每次更新後，`time_since_update` 會歸零，生命週期保持活躍。
    

### (3) 邊界框格式轉換

ByteTrack 內部常用兩種 bbox 格式：

- **tlwh**: `[x, y, w, h]`
    
- **tlbr**: `[x1, y1, x2, y2]`
    

轉換公式：

$x1 = x, \quad y1 = y, \quad x2 = x + w, \quad y2 = y + h$

---

## 4️⃣ STrack 生命週期管理

每個 STrack 都有 **time_since_update** 與 **max_age**：

- 如果一個 STrack `time_since_update > max_age` → 刪除
    
- 如果匹配成功 → `time_since_update = 0`
    
- 透過這個策略，ByteTrack 可以「自動清理消失的物體」而不依賴 ReID。
    

---

## 5️⃣ 匹配策略與 STrack 的角色

1. **高分檢測匹配**
    
    - 先用 IoU 匹配現有活躍 STrack。
        
    - 已匹配 STrack → `update()`
        
    - 未匹配 STrack → 預測下一幀位置，保留於待匹配池。
        
2. **低分檢測匹配**
    
    - 對未匹配 STrack 再進行匹配，主要增加「檢測遺漏補全率」。
        

> ByteTrack 的亮點就在於 **把低分檢測納入二次匹配**，而 STrack 是追蹤單位，負責管理整個物體的生命週期。

---

## 6️⃣ 小結

- **STrack = ByteTrack 的追蹤單元**
    
- 核心職責：
    
    1. 保存物體歷史（位置、ID、分數）
        
    2. 卡爾曼濾波預測與更新
        
    3. 控制生命週期 (`age` / `time_since_update`)
        
    4. 支援 IoU / 匹配算法
        
- 在 ByteTrack 中，整個多目標追蹤就是對 STrack 做 **匹配、更新、預測、刪除** 的循環。