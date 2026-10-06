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
