## 1. **核心哲學：不確定性的賽局**

卡爾曼濾波假設世界有兩種雜訊：

1. **系統雜訊**（process noise）：你的物體動起來會抖、加速度不是穩定的、速度會變來變去。    
2. **量測雜訊**（measurement noise）：感測器很雷：YOLO 偵測歪、雷達反射怪、IMU 有 bias。

濾波器做的事情，就是在這兩個不確定來源之間做一場「馬氏距離[[Mahalanobis_distance]]  vs 高斯分布」的辯論，  
最終選出「目前世界最可能的狀態」。

---

## 2. **模型長什麼樣：兩個方程式就決定一切**

### **(1) 狀態轉移方程式：世界怎麼運動**

$x_{t|t-1} = F x_{t-1|t-1} + B u_t + w_t$

• x 是狀態，例如 $[x, y, vx, vy]$。  
• F 是動態模型，一般 tracking 用「匀速模型」：

$F = \begin{bmatrix} 1&0&1&0\\ 0&1&0&1\\ 0&0&1&0\\ 0&0&0&1 \end{bmatrix}$

• $w_t$​ 是系統噪聲，通常為高斯分布：

$w_t \sim N(0, Q)$

---

### **(2) 觀測方程式：感測器怎麼看世界**

$z_t = H x_t + v_t$

• YOLO、相機、雷達量到的位置是 $z$。  
• $H$ 基本就是「從 state 裡抽出 x, y」的矩陣。

例如：

$H = \begin{bmatrix} 1&0&0&0\\ 0&1&0&0 \end{bmatrix}$

• 量測噪聲

$v_t \sim N(0, R)$

---

## 3. **運作流程：預測 → 更新（Predict → Update）**

### **(A) 預測階段**

預測 state：

$xt−1∣t−1x_{t|t-1} = F x_{t-1|t-1}$

預測誤差協方差：

$P_{t|t-1} = F P_{t-1|t-1} F^\top + Q$

這裡的 $P$ 是卡爾曼濾波的靈魂，代表「我對自己的不確定程度」。

---

### **(B) 更新階段：相信多少量測？相信多少模型？**

創新（innovation）：

$y_t = z_t - H x_{t|t-1}$

創新協方差：

$S_t = H P_{t|t-1} H^\top + R$

卡爾曼增益（決定相信誰比較多）：

$K_t = P_{t|t-1} H^\top S_t^{-1}$

更新最佳估計：

$x_{t|t} = x_{t|t-1} + K_t y_t$

更新不確定性：

$P_{t|t} = (I - K_t H) P_{t|t-1}$

這裡有個漂亮的意涵：

• 若感測器很不準（R 大）→ K 變小 → 較信任 model  
• 若 model 很不準（Q 大）→ K 變大 → 較信任感測器

這就是 Kalman filter 的智慧之處。

---

## 4. **直覺比喻（最重要的理解）**

假設你是個玩 RC 飛機的工程師：

• 你預測飛機下一秒位置（靠物理）  
• 你感測到當下位置（IMU + GPS）  
• 這兩個數字絕對不會一樣  
• Kalman 就是用高斯機率給你一個兩者之間的平均，但不是普通平均，是「加權後的數學最佳解」

加權由 Q 與 R 決定。  
整個濾波器就是在「預測可信度」與「感測可信度」之間找平衡。

---

## 5. **為什麼 tracking 都用匀速模型？**

因為：

1. 穩定、簡單、矩陣線性
2. 大部分物體的短期行為接近匀速
3. 影片 frame rate 高，物體在 1/30 秒內通常速度不劇變
4. 多物件追蹤（SORT / DeepSORT / ByteTrack）非常吃速度估計

你若用加速度模型（CV2）會更穩，但協方差會飆升，常常造成奇怪匹配問題。

---

## 6. **和 YOLO Tracking 的連結**

在 MOT 裡：

YOLO → Bounding box  
Kalman → 預測下一幀框  
Hungarian Assign → 匹配哪個框屬於哪個 track

卡爾曼濾波在 tracking 的作用：

• 在 YOLO 漏掉一幀時，仍能預測框  
• 平滑 jitter  
• 提供速度向量（很多 tracker 用速度做匹配權重）  
• 設計 gating：  
超過馬氏距離門檻就不配對（避免爆 ID）

沒有卡爾曼，多物件追蹤幾乎玩不下去。

---

## 7. **資料流動畫（文字示意）**

``` bash
┌──────────────┐
│ 上一幀最佳估計 x(t-1|t-1)│
└───────┬────────┘
		│ 透過 F 預測
        ▼            
┌───────────────────┐
│ 預測狀態 x(t|t-1)  │
└───────┬──────────┘
        │
        │ 比對 YOLO 感測值 z(t)
		▼       
┌────────────────────────────┐
│     創新 y(t)=z(t)-H x(t|t-1) │
└────────────┬──────────────┘
             │
             ▼
┌─────────────────────┐
│     計算卡爾曼增益 K │  
└──────┬──────────────┘
       ▼
┌────────────────────────────┐
│ 更新 x(t|t)=x(t|t-1)+K y(t) │
└────────────────────────────┘
```


### 卡爾曼濾波完整矩陣流程圖

``` bash
                    ┌───────────────────────────────────┐
                    │         上一幀最佳估計             │
                    │   x(t-1|t-1),  P(t-1|t-1)          │
                    └───────────────┬───────────────────┘
                                    │
                 【Predict】        │
                                    ▼
          ┌────────────────────────────────────────┐
          │ 狀態預測：                              │
          │   x(t|t-1) = F · x(t-1|t-1) + B·u       │
          └───────────────┬────────────────────────┘
                          │
                          ▼
          ┌────────────────────────────────────────┐
          │ 協方差預測：                            │
          │  P(t|t-1) = F · P(t-1|t-1) · Fᵀ + Q     │
          └───────────────┬────────────────────────┘
                          │
                          │
                          ▼
                ┌──────────────────────┐
                │ 感測量測值 z(t)       │
                │ (ex: YOLO bbox)      │
                └───────────┬──────────┘
                            │
                  【Update】│
                            ▼
        ┌────────────────────────────────────────────┐
        │ 創新（量測殘差）：                         │
        │   y(t) = z(t) - H · x(t|t-1)               │
        └────────────────┬───────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────┐
        │ 創新協方差：                               │
        │   S(t) = H · P(t|t-1) · Hᵀ + R              │
        └────────────────┬───────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────┐
        │ 卡爾曼增益：                               │
        │   K(t) = P(t|t-1) · Hᵀ · S(t)^{-1}          │
        └────────────────┬───────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────┐
        │ 更新狀態：                                 │
        │  x(t|t) = x(t|t-1) + K(t) · y(t)            │
        └────────────────┬───────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────┐
        │ 更新協方差：                               │
        │  P(t|t) = (I - K(t)·H) · P(t|t-1)           │
        └────────────────────────────────────────────┘
```

### 解讀指南

### 1. $x(t|t-1)$

模型預測出來的 state（未融合量測）

### 2.$P(t|t-1)$

模型預測的誤差（你對自己預測的信心）

### 3. $y(t)$

觀測與預測的落差（innovation）

### 4. $S(t)$

這個落差的可信程度（越大 = 不要太相信量測）

### 5. $K(t)$

「相信 model vs 相信 sensor」的權重分配器

### 6. $x(t|t)$

融合後的最佳估計（真正拿來做 tracking 的）

### 7. $P(t|t)$

新的信心指標（越小越確定）

## 1️⃣ 卡爾曼濾波完整原理

卡爾曼濾波是一種 **遞迴估計器**，適用於線性動態系統。它透過 **先驗預測 + 後驗更新** 來最小化狀態估計的均方誤差。

### 系統模型

線性動態系統：

$\mathbf{x}_k = \mathbf{F} \mathbf{x}_{k-1} + \mathbf{B} \mathbf{u}_k + \mathbf{w}_k, \quad \mathbf{w}_k \sim \mathcal{N}(0, \mathbf{Q})$
$\mathbf{z}_k = \mathbf{H} \mathbf{x}_k + \mathbf{v}_k, \quad \mathbf{v}_k \sim \mathcal{N}(0, \mathbf{R})$

- $\mathbf{x}_k$​ : 系統狀態向量（位置、速度…）
    
- $\mathbf{F}$ : 狀態轉移矩陣
    
- $\mathbf{B}$ : 控制矩陣
    
- $\mathbf{u}_k$​ : 控制輸入
    
- $\mathbf{w}_k$​ : 過程噪聲
    
- $\mathbf{z}_k$ : 觀測值
    
- $\mathbf{H}$ : 觀測矩陣
    
- $\mathbf{Q}$ : 過程噪聲協方差
    
- $\mathbf{R}$ : 觀測噪聲協方差
    

---

## 2️⃣ Kalman Filter 遞迴流程（矩陣形式）

### 預測步驟 (Predict)

$\hat{\mathbf{x}}_{k|k-1} = \mathbf{F} \hat{\mathbf{x}}_{k-1|k-1} + \mathbf{B} \mathbf{u}_k$
$\mathbf{P}_{k|k-1} = \mathbf{F} \mathbf{P}_{k-1|k-1} \mathbf{F}^T + \mathbf{Q}$

### 更新步驟 (Update)

$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}^T (\mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^T + \mathbf{R})^{-1}$
$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k (\mathbf{z}_k - \mathbf{H} \hat{\mathbf{x}}_{k|k-1})$
$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_{k|k-1}$

其中：

- $\hat{\mathbf{x}}_{k|k}$是更新後狀態
    
- $\mathbf{P}_{k|k}$是更新後協方差矩陣
    
- $\mathbf{K}_k$​ 是卡爾曼增益（Kalman Gain）
    

---

## 3️⃣ 流程圖示意

``` bash
  +----------------+
  | Previous state | x_{k-1|k-1}, P_{k-1|k-1}
  +----------------+
           |
           v
   [Predict Step]
 x_{k|k-1} = F x_{k-1|k-1} + B u_k
 P_{k|k-1} = F P_{k-1|k-1} F^T + Q
           |
           v
   [Update Step]
  K_k = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}
  x_{k|k} = x_{k|k-1} + K_k (z_k - H x_{k|k-1})
  P_{k|k} = (I - K_k H) P_{k|k-1}
           |
           v
  +----------------+
  | Updated state  | x_{k|k}, P_{k|k}
  +----------------+
           |
           v
       Next iteration
```

---

## 4️⃣ Python 最小範例（2D 位置 + 速度）

``` python
import numpy as np

dt = 1.0  # 時間間隔

# 狀態向量 [x, y, vx, vy]
x = np.array([0, 0, 1, 1], dtype=float)  

# 協方差矩陣
P = np.eye(4) * 500

# 狀態轉移矩陣
F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# 觀測矩陣 (只能觀測位置)
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

# 過程噪聲
Q = np.eye(4) * 0.01

# 觀測噪聲
R = np.eye(2) * 1.0

# 模擬觀測值
measurements = [np.array([i+np.random.randn()*0.5, i+np.random.randn()*0.5]) for i in range(1, 11)]

for z in measurements:
    # Predict
    x = F @ x
    P = F @ P @ F.T + Q

    # Update
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    x = x + K @ (z - H @ x)
    P = (np.eye(4) - K @ H) @ P

    print(f"Measurement: {z}, Estimate: {x[:2]}")
```

---

## 5️⃣ C++ 最小範例（Eigen 矩陣庫）

``` cpp
#include <Eigen/Dense>
#include <iostream>
#include <vector>

int main() {
    double dt = 1.0;

    Eigen::Vector4d x(0, 0, 1, 1); // [x, y, vx, vy]
    Eigen::Matrix4d P = Eigen::Matrix4d::Identity() * 500;
    Eigen::Matrix4d F;
    F << 1, 0, dt, 0,
         0, 1, 0, dt,
         0, 0, 1, 0,
         0, 0, 0, 1;
    Eigen::Matrix<double,2,4> H;
    H << 1,0,0,0,
         0,1,0,0;
    Eigen::Matrix4d Q = Eigen::Matrix4d::Identity() * 0.01;
    Eigen::Matrix2d R = Eigen::Matrix2d::Identity();

    std::vector<Eigen::Vector2d> measurements;
    for(int i=1; i<=10; ++i) {
        measurements.push_back(Eigen::Vector2d(i + ((double)rand()/RAND_MAX-0.5),
                                              i + ((double)rand()/RAND_MAX-0.5)));
    }

    for(auto& z : measurements) {
        // Predict
        x = F * x;
        P = F * P * F.transpose() + Q;

        // Update
        Eigen::Matrix2d S = H * P * H.transpose() + R;
        Eigen::Matrix<double,4,2> K = P * H.transpose() * S.inverse();
        x = x + K * (z - H * x);
        P = (Eigen::Matrix4d::Identity() - K * H) * P;

        std::cout << "Measurement: [" << z.transpose() << "], Estimate: [" << x(0) << ", " << x(1) << "]\n";
    }
}
```