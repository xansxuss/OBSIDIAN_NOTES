---
title: 等速與等加速度卡爾曼濾波器（CV / CA）整理與 Python 實作
source: 
author: 
published: 
created: 2026-10-01
description: 整理等速 (CV) 與等加速度 (CA) 卡爾曼濾波器的公式、差異與可接入 ByteTrack 的 Python 實作。
categorization: computer_vision/Kalman_Filter
tags:
  - Kalman-Filter
  - ByteTrack
  - 多目標追蹤
  - 運動模型
  - 等速模型
  - 等加速度模型
  - Python
---

# 等速與等加速度卡爾曼濾波器（CV / CA）

> 「變速」模型在文獻中通常稱為 **等加速度 (Constant Acceleration, CA)** 模型。本篇用同一套程式碼實作 CV 與 CA，介面與 [[ByteTrack]] 的 `KalmanFilter` 相容，可直接替換。

## 1. 卡爾曼濾波器在追蹤中的角色

多目標追蹤每一幀都做三件事：

1. **預測**：用濾波器把每條軌跡推到「這一幀應該在哪」。
2. **配對**：預測框和偵測框算 IoU 成本，用 [[Hungarian_Algorithm]] 分配（[[Deep_Sort]] 另外用 [[Mahalanobis_distance]] 做門檻過濾）。
3. **更新**：配到的偵測框修正濾波器狀態；沒配到的軌跡變 Lost，只靠預測往前推。

所以運動模型好不好，直接決定「預測框離真實框多遠」，也就決定能不能配得上。

## 2. 核心公式

狀態 $x$、共變異 $P$、轉移矩陣 $F$、量測矩陣 $H$、過程雜訊 $Q$、量測雜訊 $R$。

**預測 (predict)**

$$x' = F x \qquad P' = F P F^{T} + Q$$

**更新 (update)**

$$S = H P' H^{T} + R \qquad K = P' H^{T} S^{-1}$$

$$x = x' + K (z - H x') \qquad P = P' - K S K^{T}$$

- $z - Hx'$ 稱為**創新 (innovation)**，是「量測」與「預測」的差。
- $K$ 是卡爾曼增益：$Q$ 越大 → 越不信預測 → $K$ 越大 → 更快跟上量測；$R$ 越大 → 越不信量測 → $K$ 越小 → 軌跡更平滑。
- 穩態時，CV 的行為近似 α-β 濾波器，CA 近似 α-β-γ 濾波器。

## 3. ByteTrack 的狀態設計

量測 $z = (c_x, c_y, a, h)$：中心點、長寬比 $a = w/h$、高 $h$。用長寬比而不用寬，是因為同一個人在畫面中遠近縮放時，長寬比近似不變。

| 項目 | 係數 | 說明 |
|---|---|---|
| 位置雜訊 $w_p$ | $1/20$ | 乘上目前框高 $h$ |
| 速度雜訊 $w_v$ | $1/160$ | 乘上目前框高 $h$ |
| 長寬比 $a$ | Q: $10^{-2}$，R: $10^{-1}$ | 固定值 |
| 長寬比速度 $v_a$ | Q: $10^{-5}$ | 幾乎不允許變化 |

**所有雜訊都乘 $h$**：近的大目標像素位移大、雜訊也大；遠的小目標則小，同一組係數可通用。

## 4. 等速模型 (CV)

狀態（8 維）：$[c_x, c_y, a, h, v_x, v_y, v_a, v_h]$

$$F = \begin{bmatrix} I_4 & I_4 \\ 0 & I_4 \end{bmatrix}, \qquad c_x' = c_x + v_x$$

- 假設速度「近似不變」，速度的真實變化全部由 $Q$ 的速度項吸收。
- **優點**：參數少、直線運動很穩、計算量小。
- **缺點**：遇到突然轉向、停下、反彈，速度估計要花數幀才跟上，期間預測會偏離。

## 5. 等加速度模型 (CA)

狀態（10 維）：$[c_x, c_y, a, h, v_x, v_y, v_a, v_h, a_x, a_y]$

每個軸（以 $x$ 為例）的轉移：

$$\begin{bmatrix} c_x' \\ v_x' \\ a_x' \end{bmatrix} = \begin{bmatrix} 1 & 1 & \tfrac12 \\ 0 & 1 & 1 \\ 0 & 0 & \lambda \end{bmatrix} \begin{bmatrix} c_x \\ v_x \\ a_x \end{bmatrix}$$

- $\lambda$ 即程式裡的 `acc_decay`：$1.0$ 是標準 CA；小於 1（如 0.9）讓加速度逐幀衰減。
- **只有中心點 $(c_x, c_y)$ 有加速度**，長寬比與高度不建加速度。否則 Lost 軌跡的框大小會被加速度項拉著亂脹縮。
- 前 8 維與 CV 完全相同，所以 `STrack` 裡的 `mean[:4]`（取框）和 `mean[7] = 0`（Lost 時清掉高度速度）**都不用改**。
- **優點**：對平滑的加速、減速、轉彎追得上。
- **缺點**：加速度是速度的「微分」，對雜訊非常敏感；濾波器會把量測抖動誤當成加速度並往前外插，導致直線運動的預測反而更抖。Lost 久了還會被舊加速度帶到很遠。

## 6. CV 與 CA 比較

| 項目 | CV 等速 | CA 等加速度 |
|---|---|---|
| 狀態維度 | 8 | 10 |
| 額外參數 | 無 | `std_weight_acceleration`、`acc_decay` |
| 直線運動預測 | 較準 | 較抖（誤差約 +40%） |
| 平滑加速／減速 | 落後（持續偏移） | 追得上 |
| 瞬間反向（撞牆） | 偏差大，約數幀才修正 | 較好，但仍無法預知 |
| 雜訊敏感度 | 低 | 高 |
| Lost 外插風險 | 低 | 較高（可用 `acc_decay` 緩解） |
| 常見採用 | SORT、DeepSORT、ByteTrack | 少見，多用於雷達等機動目標 |

## 7. Python 實作

檔名 `kalman_filter_cv_ca.py`，只依賴 numpy。

```python
# -*- coding: utf-8 -*-
"""
等速 (Constant Velocity, CV) 與 等加速度 (Constant Acceleration, CA) 卡爾曼濾波器。

介面與 ByteTrack 的 KalmanFilter 相容，可直接傳入 BYTETracker(kalman_filter=...)：
    initiate(measurement)                -> (mean, covariance)
    predict(mean, covariance)            -> (mean, covariance)
    multi_predict(means, covariances)    -> (means, covariances)   向量化，一次預測 N 條軌跡
    project(mean, covariance)            -> (projected_mean, projected_covariance)
    update(mean, covariance, measurement)-> (mean, covariance)

量測向量：z = (cx, cy, a, h)       a = 寬 / 高 (長寬比)，h = 高
狀態向量（前 8 維兩個模型完全相同，所以 STrack 的 mean[:4]、mean[7] 不用改）：
    CV : [cx, cy, a, h, vx, vy, va, vh]                  8 維
    CA : [cx, cy, a, h, vx, vy, va, vh, ax, ay]          10 維
         (只有中心點 cx, cy 有加速度；長寬比與高度不建加速度，
          否則 Lost 的軌跡框大小會被加速度項帶著亂脹縮)
雜訊大小都乘上目前的框高 h，所以遠(小)的目標雜訊小、近(大)的目標雜訊大。
"""
import numpy as np


def _stack(*cols):
    """把多個形狀相同的陣列 (...) 疊成 (..., len(cols))，方便一次組出標準差向量。"""
    return np.stack(cols, axis=-1)


class KalmanFilterBase:
    MEAS_DIM = 4  # 量測維度：(cx, cy, a, h)

    def __init__(self, std_weight_position=1.0 / 20, std_weight_velocity=1.0 / 160):
        self.wp = std_weight_position       # 位置雜訊係數 (× h)
        self.wv = std_weight_velocity       # 速度雜訊係數 (× h)
        self.F = self._build_motion_matrix()               # 狀態轉移矩陣 (n, n)
        self.n = self.F.shape[0]                           # 狀態維度
        self.H = np.eye(self.MEAS_DIM, self.n)             # 量測矩陣 (4, n)：只取前 4 維

    # ---------- 子類別必須實作 ----------
    def _build_motion_matrix(self):
        raise NotImplementedError

    def _process_std(self, h):
        """過程雜訊 Q 的標準差，形狀 (..., n)。"""
        raise NotImplementedError

    def _init_std(self, h):
        """初始共變異 P0 的標準差，形狀 (..., n)。"""
        raise NotImplementedError

    # ---------- 共用邏輯 ----------
    def _measure_std(self, h):
        """量測雜訊 R 的標準差 (cx, cy, a, h)。長寬比 a 的 1e-1 沿用 ByteTrack 原值。"""
        p = self.wp * h
        return _stack(p, p, np.full_like(p, 1e-1), p)

    def initiate(self, measurement):
        """用第一個偵測建立軌跡：速度、加速度都從 0 開始，不確定度給大一點。"""
        z = np.asarray(measurement, dtype=np.float64)
        mean = np.r_[z, np.zeros(self.n - self.MEAS_DIM)]
        cov = np.diag(np.square(self._init_std(z[3])))
        return mean, cov

    def multi_predict(self, means, covariances):
        """預測步驟：x' = F x，P' = F P Fᵀ + Q。一次處理 N 條軌跡 (N, n) / (N, n, n)。"""
        means = np.atleast_2d(np.asarray(means, dtype=np.float64))
        covs = np.asarray(covariances, dtype=np.float64).reshape(len(means), self.n, self.n)
        std = self._process_std(means[:, 3])               # (N, n)；h 取預測前的高度
        idx = np.arange(self.n)
        Q = np.zeros_like(covs)
        Q[:, idx, idx] = np.square(std)                    # Q 為對角矩陣
        new_means = means @ self.F.T
        new_covs = self.F @ covs @ self.F.T + Q            # (n,n) @ (N,n,n) 會自動廣播
        return new_means, new_covs

    def predict(self, mean, covariance):
        """單一軌跡的預測，內部直接呼叫 multi_predict，確保兩者結果一致。"""
        m, c = self.multi_predict(mean[None, :], covariance[None, :, :])
        return m[0], c[0]

    def project(self, mean, covariance):
        """把狀態投影到量測空間：(H x, H P Hᵀ + R)。"""
        R = np.diag(np.square(self._measure_std(mean[3])))
        return self.H @ mean, self.H @ covariance @ self.H.T + R

    def update(self, mean, covariance, measurement):
        """更新步驟：K = P Hᵀ S⁻¹，x = x + K (z − H x)，P = P − K S Kᵀ。"""
        z = np.asarray(measurement, dtype=np.float64)
        proj_mean, S = self.project(mean, covariance)      # S：創新共變異 (4, 4)
        # S 對稱，K = P Hᵀ S⁻¹ = (S⁻¹ H P)ᵀ；用 solve 而非顯式求反矩陣，數值較穩定
        K = np.linalg.solve(S, self.H @ covariance).T      # (n, 4)
        new_mean = mean + K @ (z - proj_mean)
        new_cov = covariance - K @ S @ K.T
        return new_mean, new_cov


class KalmanFilterCV(KalmanFilterBase):
    """等速模型：假設速度近似不變，速度的變化全部交給過程雜訊 Q 吸收。"""

    def _build_motion_matrix(self):
        F = np.eye(8)
        for i in range(4):
            F[i, 4 + i] = 1.0                              # 位置 += 速度 × dt (dt = 1 幀)
        return F

    def _process_std(self, h):
        h = np.asarray(h, dtype=np.float64)
        p, v = self.wp * h, self.wv * h
        one = np.ones_like(h)
        return _stack(p, p, 1e-2 * one, p, v, v, 1e-5 * one, v)

    def _init_std(self, h):
        h = np.asarray(h, dtype=np.float64)
        p, v = 2 * self.wp * h, 10 * self.wv * h
        one = np.ones_like(h)
        return _stack(p, p, 1e-2 * one, p, v, v, 1e-5 * one, v)


class KalmanFilterCA(KalmanFilterBase):
    """等加速度模型：多一組 (ax, ay) 狀態，速度可以隨加速度線性變化。

    std_weight_acceleration : 加速度過程雜訊係數 (× h)，越大越能快速跟上機動，
                              但直線飛行時預測越抖
    acc_decay               : 每幀加速度衰減率。1.0 = 標準 CA；<1 (如 0.9) 讓加速度逐漸歸零，
                              避免 Lost 太久的軌跡被舊加速度外插到很遠
    """

    def __init__(self, std_weight_position=1.0 / 20, std_weight_velocity=1.0 / 160,
                 std_weight_acceleration=1.0 / 160, acc_decay=1.0):
        self.wa = std_weight_acceleration
        self.acc_decay = acc_decay
        super().__init__(std_weight_position, std_weight_velocity)

    def _build_motion_matrix(self):
        F = np.eye(10)
        for i in range(4):
            F[i, 4 + i] = 1.0                              # 位置 += 速度
        F[0, 8] = F[1, 9] = 0.5                            # 位置 += ½ · 加速度 · dt²
        F[4, 8] = F[5, 9] = 1.0                            # 速度 += 加速度 · dt
        F[8, 8] = F[9, 9] = self.acc_decay                 # 加速度保持（或衰減）
        return F

    def _process_std(self, h):
        h = np.asarray(h, dtype=np.float64)
        p, v, a = self.wp * h, self.wv * h, self.wa * h
        one = np.ones_like(h)
        return _stack(p, p, 1e-2 * one, p, v, v, 1e-5 * one, v, a, a)

    def _init_std(self, h):
        h = np.asarray(h, dtype=np.float64)
        p, v, a = 2 * self.wp * h, 10 * self.wv * h, 10 * self.wa * h
        one = np.ones_like(h)
        return _stack(p, p, 1e-2 * one, p, v, v, 1e-5 * one, v, a, a)
```

### 設計重點

- **共用基底 `KalmanFilterBase`**：predict／update 的數學只寫一次；CV 與 CA 只需要給 `F`、`_process_std`、`_init_std` 三樣東西，兩者因此不會出現不一致。
- **`predict` 直接呼叫 `multi_predict`**：單條與批次共用同一份程式碼，結果必然相同（自我測試有驗證）。
- **`multi_predict` 向量化**：`self.F @ covs @ self.F.T` 利用 numpy 廣播，一次處理 N 條軌跡，不用 Python 迴圈。
- **Q 是對角矩陣**：先建全零，再用 `Q[:, idx, idx] = std**2` 一次填入對角線。
- **`update` 用 `np.linalg.solve` 而不是 `inv`**：$S$ 對稱，$K = (S^{-1} H P)^{T}$，直接解線性方程組，數值較穩定，也不需要 scipy。
- **雜訊乘 `h`**：`_process_std(means[:, 3])` 取預測前的高度，與原版 ByteTrack 一致。
- **`initiate`**：速度、加速度從 0 開始，初始不確定度給大（位置 ×2、速度 ×10），讓前幾次更新能快速收斂。

## 8. 驗證與撞牆實驗

檔名 `kalman_bounce_experiment.py`。

```python
# -*- coding: utf-8 -*-
"""驗證與實驗：
1) selftest()      : 確認 CV 與原版 ByteTrack 公式一致、multi_predict 與 predict 一致、CA 能追上加速運動
2) run_trials()    : 球撞牆瞬間速度反向，比較各模型「預測框 vs 真實框」的 IoU
執行： python kalman_bounce_experiment.py
"""
import numpy as np
from kalman_filter_cv_ca import KalmanFilterCV, KalmanFilterCA


# ---------- 小工具 ----------
def xyah_to_tlbr(m):
    """(cx, cy, a, h) -> (x1, y1, x2, y2)"""
    cx, cy, a, h = m[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def tlbr_to_xyah(b):
    """(x1, y1, x2, y2) -> (cx, cy, a, h)"""
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / h, h])


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ---------- 1) 自我測試 ----------
class _ReferenceKF:
    """依原版 ByteTrack 公式獨立寫的對照組（只用 numpy），用來驗證 KalmanFilterCV。"""
    def __init__(self):
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, 4 + i] = 1.0
        self.H = np.eye(4, 8)
        self.wp, self.wv = 1 / 20, 1 / 160

    def initiate(self, z):
        h = z[3]
        std = [2 * self.wp * h] * 2 + [1e-2] + [2 * self.wp * h] + [10 * self.wv * h] * 2 + [1e-5] + [10 * self.wv * h]
        return np.r_[z, np.zeros(4)], np.diag(np.square(std))

    def predict(self, m, c):
        h = m[3]
        std = [self.wp * h] * 2 + [1e-2, self.wp * h] + [self.wv * h] * 2 + [1e-5, self.wv * h]
        return self.F @ m, self.F @ c @ self.F.T + np.diag(np.square(std))

    def update(self, m, c, z):
        h = m[3]
        R = np.diag(np.square([self.wp * h, self.wp * h, 1e-1, self.wp * h]))
        S = self.H @ c @ self.H.T + R
        K = c @ self.H.T @ np.linalg.inv(S)
        return m + K @ (z - self.H @ m), c - K @ S @ K.T


def selftest():
    rng = np.random.default_rng(0)
    cv, ref = KalmanFilterCV(), _ReferenceKF()

    # (1) CV 與原版公式逐步比對：initiate -> 10 次 predict/update
    z0 = np.array([300.0, 200.0, 0.8, 120.0])
    m1, c1 = cv.initiate(z0)
    m2, c2 = ref.initiate(z0)
    for _ in range(10):
        m1, c1 = cv.predict(m1, c1); m2, c2 = ref.predict(m2, c2)
        z = z0 + rng.normal(0, [2, 2, 0.02, 2])
        m1, c1 = cv.update(m1, c1, z); m2, c2 = ref.update(m2, c2, z)
    assert np.allclose(m1, m2, atol=1e-8) and np.allclose(c1, c2, atol=1e-8), "CV 與原版公式不一致"
    print("[OK] KalmanFilterCV 與原版 ByteTrack 公式一致")

    # (2) multi_predict 與逐一 predict 結果相同（CV、CA 都測）
    for kf in (cv, KalmanFilterCA()):
        ms, cs = zip(*[kf.initiate(rng.uniform([0, 0, 0.3, 30], [800, 400, 1.5, 200])) for _ in range(6)])
        ms, cs = np.array(ms), np.array(cs)
        bm, bc = kf.multi_predict(ms, cs)
        for i in range(6):
            sm, sc = kf.predict(ms[i], cs[i])
            assert np.allclose(bm[i], sm) and np.allclose(bc[i], sc)
    print("[OK] multi_predict 與 predict 一致")

    # (3) 等加速度運動：x = x0 + ½ a t²，CA 的預測誤差應明顯小於 CV
    def steady_error(kf):
        errs = []
        r = np.random.default_rng(1)
        x_true = lambda t: 100 + 0.5 * 0.3 * t * t               # a = 0.3 px/frame²
        m, c = kf.initiate([x_true(0), 100, 1.0, 50.0])
        for t in range(1, 80):
            m, c = kf.predict(m, c)
            if t > 40:
                errs.append(abs(m[0] - x_true(t)))
            m, c = kf.update(m, c, [x_true(t) + r.normal(0, 0.7), 100 + r.normal(0, 0.7), 1.0, 50.0])
        return np.mean(errs)
    e_cv, e_ca = steady_error(cv), steady_error(KalmanFilterCA())
    assert e_ca < e_cv
    print(f"[OK] 等加速度運動：CV 預測誤差 {e_cv:.2f} px，CA {e_ca:.2f} px")


# ---------- 2) 撞牆實驗 ----------
def simulate_bounce(r, v, n=60, tb=30):
    """球往左飛，第 tb 幀撞左牆，回傳每幀圓心 x 與撞牆幀。"""
    x, vx, xs, hit = r + v * (tb - 0.5), -v, [], None
    for t in range(n):
        x += vx
        if x - r < 0:
            x, vx = r, -vx
            hit = t if hit is None else hit
        xs.append(x)
    return np.array(xs), hit


def run_trials(make_kf, trials=1000, post=6, seed=0):
    """回傳 (直線預測誤差 px, 撞牆後最大誤差 px, IoU<0.5 比例%, IoU<0.3 比例%)。
    每個模型用同樣的亂數種子，所以吃到的雜訊完全一樣，比較才公平。"""
    steady, worst_err, worst_iou = [], [], []
    for i in range(trials):
        rng = np.random.default_rng(seed + i)
        r, v = rng.uniform(20, 32), rng.uniform(4, 8)
        xs, tb = simulate_bounce(r, v)
        kf = make_kf()
        boxes = [np.array([x - r, 100 - r, x + r, 100 + r]) for x in xs]
        noisy = [tlbr_to_xyah(b + rng.normal(0, 0.04 * r, 4)) for b in boxes]   # 框四邊各加雜訊
        m, c = kf.initiate(noisy[0])
        e_post, i_post = [], []
        for t in range(1, len(xs)):
            m, c = kf.predict(m, c)                                  # 先預測，再和真實框比
            err, ov = abs(m[0] - xs[t]), iou(xyah_to_tlbr(m), boxes[t])
            if 10 < t < tb - 2:
                steady.append(err)
            if tb <= t < tb + post:
                e_post.append(err); i_post.append(ov)
            m, c = kf.update(m, c, noisy[t])
        worst_err.append(max(e_post)); worst_iou.append(min(i_post))
    worst_iou = np.array(worst_iou)
    return np.mean(steady), np.mean(worst_err), (worst_iou < 0.5).mean() * 100, (worst_iou < 0.3).mean() * 100


if __name__ == "__main__":
    selftest()
    models = [
        ("CV 標準 (wv=1/160)",            lambda: KalmanFilterCV()),
        ("CV 速度雜訊 x4 (wv=1/40)",      lambda: KalmanFilterCV(std_weight_velocity=1 / 40)),
        ("CA (wa=1/160)",                 lambda: KalmanFilterCA(std_weight_acceleration=1 / 160)),
        ("CA (wa=1/40)",                  lambda: KalmanFilterCA(std_weight_acceleration=1 / 40)),
        ("CA (wa=1/160, 衰減0.9)",        lambda: KalmanFilterCA(std_weight_acceleration=1 / 160, acc_decay=0.9)),
    ]
    print(f"\n{'模型':<26}{'直線誤差px':>10}{'撞牆最大誤差px':>14}{'IoU<0.5':>9}{'IoU<0.3':>9}")
    for name, mk in models:
        s, w, a, b = run_trials(mk)
        print(f"{name:<26}{s:>10.2f}{w:>14.2f}{a:>8.1f}%{b:>8.1f}%")
```

### 自我測試結果

- CV 與依原版 ByteTrack 公式獨立寫的對照組逐步比對，差異小於 $10^{-8}$。
- `multi_predict` 與逐一 `predict` 完全一致。
- 等加速度運動（$a = 0.3$ px/幀²）：CV 預測誤差 **4.18 px**，CA **0.56 px**。

### 撞牆實驗結果

設定：球往牆飛、撞牆瞬間 x 軸速度反向；半徑 20–32 px、速度 4–8 px/幀；偵測框四邊各加 $0.04r$ 的高斯雜訊；1000 次試驗，各模型使用相同亂數。表中「IoU」是**撞牆後 6 幀內，預測框與真實框的最差 IoU**。

| 模型 | 直線誤差 (px) | 撞牆最大誤差 (px) | IoU < 0.5 | IoU < 0.3 |
|---|---|---|---|---|
| CV 標準 ($w_v = 1/160$) | 0.53 | 15.34 | 34.0% | 0.0% |
| CV 速度雜訊 ×4 ($w_v = 1/40$) | 0.61 | 12.13 | 7.0% | 0.0% |
| CA ($w_a = 1/160$) | 0.75 | 11.70 | 4.6% | 0.0% |
| CA ($w_a = 1/40$) | 0.93 | 10.43 | 1.8% | 0.0% |
| CA ($w_a = 1/160$，衰減 0.9) | 0.68 | 11.99 | 6.0% | 0.0% |

### 結論

- **IoU < 0.3 全部是 0%**：BYTETracker 的 `second_match_thresh` 設 0.7（約 IoU ≥ 0.3）時，撞牆後的低分框都能接回；預設 0.5 則有約三成的撞牆事件會失敗。
- **最划算的是只調高 CV 的速度雜訊**：直線誤差只增加約 15%，IoU < 0.5 的比例從 34% 降到 7%。CA 雖然更低（4.6%、1.8%），但直線誤差增加 40–75%，而真實場景絕大多數幀都是直線運動。
- **CA 真正的優勢在平滑加速**：等加速度測試中誤差從 4.18 降到 0.56 px，這是調 CV 速度雜訊做不到的。
- **預測無法消除撞牆當下的誤差**：反彈前沒有任何資訊能預告反向，最大誤差只能從 15 px 降到 10–12 px。
- 這是理想化的單軸實驗，沒有漏檢與多球互撞；請以自己的追蹤評估（IDSW、FN）為準。

## 9. 接入 ByteTrack

`BYTETracker` 的建構子有 `kalman_filter=` 參數，直接傳入實例即可：

```python
from kalman_filter_cv_ca import KalmanFilterCV, KalmanFilterCA

# 等速，速度雜訊調高（撞牆／急轉彎較多的場景）
tracker = BYTETracker(track_buffer=50, frame_rate=30,
                      kalman_filter=KalmanFilterCV(std_weight_velocity=1 / 40))

# 等加速度，加速度會逐幀衰減
tracker = BYTETracker(track_buffer=50, frame_rate=30,
                      kalman_filter=KalmanFilterCA(std_weight_acceleration=1 / 160, acc_decay=0.9))
```

- 對照原版公式驗證的是 `KalmanFilterCV`；若專案內自己的 `kalman_filter.py` 有改過雜訊係數，數值會不同。
- `STrack` 取框用 `mean[:4]`、Lost 時 `mean[7] = 0`，兩個模型都相容，不需要改 `bytetrack.py`。
- 這份實作假設 $dt = 1$ 幀。如果有跳幀，需要自己多呼叫 `predict` 幾次，或擴充成可傳入 `dt`。

## 10. 選擇與調參建議

1. **先用 CV**，這是 SORT 系列的預設，直線運動最穩。
2. 發現轉向、急停後軌跡斷掉：**先把 `std_weight_velocity` 調大**（1/160 → 1/80 → 1/40），每次重跑 IDSW／FN 確認直線誤差沒有惡化太多。
3. 目標有明顯的平滑加速或減速（例如車輛起步、煞車）：才考慮 CA，並搭配 `acc_decay` 約 0.9 抑制外插。
4. 想讓低分框更容易接回：調 `second_match_thresh`（0.5 → 0.7），代價是低分雜訊框搶走軌跡的風險。
5. 調參時固定亂數種子、用同一份資料，只改一個參數，才看得出因果。

## 相關筆記

- [[Kalman_Filter]]、[[ByteTrack]]、[[Deep_Sort]]
- [[Hungarian_Algorithm]]、[[Mahalanobis_distance]]
- [[ball_track_sim]]（球體碰牆軌跡模擬器，用已知真值測試追蹤邏輯）
