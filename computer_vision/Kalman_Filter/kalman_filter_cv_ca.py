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
