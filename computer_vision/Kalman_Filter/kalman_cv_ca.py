#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
等速 (Constant Velocity, CV) 與等加速度 (Constant Acceleration, CA) 卡爾曼濾波器。

同一個類別 MotionKalman，用 order 切換模型：
    order=1 -> CV：狀態 = [位置, 速度]
    order=2 -> CA：狀態 = [位置, 速度, 加速度]
位置可以是多維(n_pos)，例如 (cx, cy) 或 ByteTrack 的 (cx, cy, a, h)。

雜訊設計沿用 ByteTrack：雜訊標準差 = 係數 * scale，scale 通常取「框高 h」，
所以大物體的雜訊大、小物體的雜訊小(像素誤差與物體大小成比例)。

執行：python kalman_cv_ca.py     # 跑 2D 範例 + 撞牆實驗
"""
from math import factorial

import numpy as np


class MotionKalman:
    """線性卡爾曼濾波器。

    模型：   x_k = F x_{k-1} + w,   w ~ N(0, Q)     (狀態如何隨時間演化)
             z_k = H x_k + v,       v ~ N(0, R)     (量測如何由狀態產生)
    """

    def __init__(self, order=1, n_pos=1, dt=1.0, w_pos=1 / 20, w_vel=1 / 160, w_acc=1 / 160):
        """
        order : 1 = 等速(CV)；2 = 等加速度(CA)
        n_pos : 位置維度數。1 = 單軸；2 = (x, y)；4 = ByteTrack 的 (cx, cy, a, h)
        dt    : 兩次更新的時間間隔，影像追蹤通常以「幀」為單位，故為 1
        w_pos : 位置雜訊係數(同時用於量測雜訊 R)；實際標準差 = w_pos * scale
        w_vel : 速度過程雜訊係數。越大 = 越相信「速度會變」，濾波器反應越快、越不平滑
        w_acc : 加速度過程雜訊係數(只有 order=2 才用到)
        """
        if order not in (1, 2):
            raise ValueError("order 只能是 1(等速) 或 2(等加速度)")
        self.order, self.n_pos, self.dt = order, n_pos, dt
        self.m = order + 1                 # 階數：位置、速度、(加速度)
        self.n = n_pos * self.m            # 狀態總維度
        # 每個狀態分量的雜訊係數，例：CV 且 n_pos=2 -> [w_pos, w_pos, w_vel, w_vel]
        self.coef = np.repeat(np.array([w_pos, w_vel, w_acc][: self.m]), n_pos)
        self.meas_coef = w_pos

        # F：狀態轉移矩陣。區塊 (i, j) = dt^(j-i) / (j-i)! * I，也就是位置 += 速度*dt (+ 加速度*dt^2/2)
        F = np.eye(self.n)
        for i in range(self.m):
            for j in range(i + 1, self.m):
                F[i * n_pos:(i + 1) * n_pos, j * n_pos:(j + 1) * n_pos] = \
                    np.eye(n_pos) * dt ** (j - i) / factorial(j - i)
        self.F = F
        # H：量測矩陣。相機/偵測器只量得到位置，所以只取出狀態的前 n_pos 個分量
        self.H = np.zeros((n_pos, self.n))
        self.H[:, :n_pos] = np.eye(n_pos)
        self.x = None                      # 狀態估計
        self.P = None                      # 估計的共變異矩陣(不確定度)

    def initiate(self, z, scale=1.0):
        """用第一個量測 z 開啟濾波器。速度/加速度未知，先設 0 並給很大的不確定度。"""
        z = np.asarray(z, dtype=float).reshape(self.n_pos)
        self.x = np.zeros(self.n)
        self.x[: self.n_pos] = z
        std = self.coef * scale
        std[: self.n_pos] *= 2             # 位置初始標準差 = 2 * w_pos * scale
        std[self.n_pos:] *= 10             # 速度/加速度完全沒觀測過，初始標準差放大 10 倍
        self.P = np.diag(std ** 2)
        return self.x[: self.n_pos].copy()

    def predict(self, scale=1.0):
        """預測步：x⁻ = F x；P⁻ = F P Fᵀ + Q。回傳預測的位置。"""
        Q = np.diag((self.coef * scale) ** 2)   # 過程雜訊：每幀「模型可能不準」的程度
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q
        return self.x[: self.n_pos].copy()

    def update(self, z, scale=1.0):
        """更新步：用量測 z 修正預測。回傳修正後的位置。"""
        z = np.asarray(z, dtype=float).reshape(self.n_pos)
        R = np.eye(self.n_pos) * (self.meas_coef * scale) ** 2   # 量測雜訊
        y = z - self.H @ self.x                                   # 殘差(innovation)：量測 - 預測
        S = self.H @ self.P @ self.H.T + R                        # 殘差的共變異
        # 卡爾曼增益 K = P Hᵀ S⁻¹：S 越小(預測越準)K 越小，越相信預測；R 越小 K 越大，越相信量測
        K = np.linalg.solve(S, self.H @ self.P).T                 # S、P 皆對稱，故轉置後等於 P Hᵀ S⁻¹
        self.x = self.x + K @ y
        I_KH = np.eye(self.n) - K @ self.H
        # Joseph 形式：數學上等於 (I-KH)P，但數值上能保持 P 對稱且半正定
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return self.x[: self.n_pos].copy()

    @property
    def position(self):
        return self.x[: self.n_pos].copy()

    @property
    def velocity(self):
        return self.x[self.n_pos: 2 * self.n_pos].copy()


# ============================================================
# 範例 1：2D 點以固定速度移動，觀察濾波器如何估出速度
# ============================================================
def demo_2d(seed=0):
    rng = np.random.default_rng(seed)
    true_v = np.array([3.0, 1.0])                  # 真實速度 (px/幀)
    kf = MotionKalman(order=1, n_pos=2)
    pos = np.array([100.0, 50.0])
    h = 40.0                                       # 假設物體框高 40 像素
    kf.initiate(pos + rng.normal(0, 1.0, 2), scale=h)
    for _ in range(30):
        pos = pos + true_v
        kf.predict(scale=h)
        kf.update(pos + rng.normal(0, 1.0, 2), scale=h)   # 量測有 1 像素雜訊
    print(f"[2D 範例] 真實速度 {true_v}，30 幀後估計速度 {np.round(kf.velocity, 2)}")


# ============================================================
# 範例 2：球撞牆，速度瞬間反向 —— 比較各模型的「預測」誤差
# ============================================================
def bounce_trajectory(r, v, n=60, tb=30):
    """球朝左牆飛，第 tb 幀撞牆(與 ball_track_sim.py 的規則相同：位置夾回牆內、速度反號)。"""
    x, vx, xs, hit = r + v * (tb - 0.5), -v, [], None
    for t in range(n):
        x += vx
        if x - r < 0:
            x, vx = r, -vx
            hit = t if hit is None else hit
        xs.append(x)
    return np.array(xs), hit


def iou_1d(err, r):
    """同尺寸(寬 2r)的兩個框只在 x 方向差 |err| 時的 IoU(y 方向完全重疊)。"""
    w = 2 * r
    inter = np.clip(w - np.abs(err), 0, None)
    return inter / (2 * w - inter)


def bounce_demo(trials=2000, seed=0):
    rng = np.random.default_rng(seed)
    data = []                                      # 所有模型共用同一批軌跡與雜訊，比較才公平
    for _ in range(trials):
        r, v = rng.uniform(20, 32), rng.uniform(4, 8)
        xs, tb = bounce_trajectory(r, v)
        z = xs + rng.normal(0, 0.04 * r / np.sqrt(2), len(xs))   # 框中心的量測雜訊
        data.append((r, xs, z, tb))

    configs = [
        ("CV 標準 (w_vel=1/160)", dict(order=1, w_vel=1 / 160)),
        ("CV w_vel=1/80",         dict(order=1, w_vel=1 / 80)),
        ("CV w_vel=1/40",         dict(order=1, w_vel=1 / 40)),
        ("CV w_vel=1/20",         dict(order=1, w_vel=1 / 20)),
        ("CV w_vel=1/10",         dict(order=1, w_vel=1 / 10)),
        ("CA w_acc=1/160",        dict(order=2, w_acc=1 / 160)),
        ("CA w_acc=1/40",         dict(order=2, w_acc=1 / 40)),
        ("CA w_acc=1/10",         dict(order=2, w_acc=1 / 10)),
    ]
    print(f"\n[撞牆實驗] {trials} 次，量測每幀都有；統計撞牆後 6 幀內「預測框 vs 真實框」最差的 IoU")
    print(f"{'模型':<22}{'直線誤差px':>10}{'撞牆最大誤差px':>14}{'IoU<0.5':>9}{'IoU<0.3':>9}")
    for name, kw in configs:
        steady, worst_iou, worst_err = [], [], []
        for r, xs, z, tb in data:
            h = 2 * r
            kf = MotionKalman(n_pos=1, **kw)
            kf.initiate(z[0], scale=h)
            errs = {}
            for t in range(1, len(xs)):
                pred = kf.predict(scale=h)[0]              # 先預測(追蹤器拿這個位置去配對偵測)
                if 10 < t < tb - 2:
                    steady.append(abs(pred - xs[t]))       # 直線飛行階段的預測誤差
                if tb <= t < tb + 6:
                    errs[t] = pred - xs[t]                 # 撞牆後 6 幀的預測誤差
                kf.update(z[t], scale=h)                   # 再用量測更新
            e = np.array(list(errs.values()))
            worst_iou.append(iou_1d(e, r).min())
            worst_err.append(np.abs(e).max())
        worst_iou = np.array(worst_iou)
        print(f"{name:<22}{np.mean(steady):>10.2f}{np.mean(worst_err):>14.2f}"
              f"{(worst_iou < 0.5).mean() * 100:>8.1f}%{(worst_iou < 0.3).mean() * 100:>8.1f}%")


if __name__ == "__main__":
    demo_2d()
    bounce_demo()
