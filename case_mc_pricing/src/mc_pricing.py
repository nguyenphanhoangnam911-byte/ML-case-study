"""Case study Monte Carlo Control: mô hình nhu cầu, MDP + quy hoạch động, bộ mô phỏng,
các biến thể MC Control và BatchMC. Trích nguyên văn từ Dynamic_Pricing_MC_Control.ipynb
(các ô 3, 6, 8, 10, 12, 22)."""


import bisect, random
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July",
                                      "August", "September", "October", "November", "December"], 1)}


# ----------------------------------------------------------------------------
# 1. Làm sạch dữ liệu và hiệu chỉnh nhu cầu
# ----------------------------------------------------------------------------
def load_bookings(path):
    """City Hotel, phòng loại A, booking không hủy, khách lẻ (Transient/Transient-Party),
    kênh bán có định giá linh hoạt: Online TA, Offline TA/TO (nhóm L) và Direct (nhóm B)."""
    df = pd.read_csv(path)
    c = df[(df.hotel == "City Hotel") & (df.reserved_room_type == "A") & (df.is_canceled == 0)
           & (df.adr > 0) & (df.adr < 1000)
           & df.customer_type.isin(["Transient", "Transient-Party"])
           & df.market_segment.isin(["Online TA", "Offline TA/TO", "Direct"])].copy()
    c["date"] = pd.to_datetime(dict(year=c.arrival_date_year, month=c.arrival_date_month.map(MONTHS),
                                    day=c.arrival_date_day_of_month))
    c["ym"] = c.date.dt.to_period("M")
    c["radr"] = c.adr / c.groupby("ym").adr.transform("median")    # giá tương đối theo mùa
    c["seg"] = np.where(c.market_segment == "Direct", "B", "L")
    return df, c


def booking_curves(c, T=30):
    """Số booking trung bình mỗi đêm theo lead_time (0..T-1) cho từng nhóm, và giá tương đối lịch sử."""
    n_nights = (c.date.max() - c.date.min()).days + 1
    w = c[c.lead_time < T]
    obs = {s: (w[w.seg == s].groupby("lead_time").size().reindex(range(T), fill_value=0) / n_nights).values
           for s in "LB"}
    p_hist = {s: float(w[w.seg == s].radr.median()) for s in "LB"}
    return obs, p_hist, n_nights


class Scenario:
    """Mô hình nhu cầu: mỗi ngày t (còn t ngày đến ngày nhận phòng) có số khách tiềm năng ~ Poisson(λ_t^seg).
    Khách mua nếu WTP ≥ giá; WTP ~ LogNormal(ln m_seg, σ_seg). m_seg được chọn sao cho tại giá lịch sử
    tỷ lệ chuyển đổi bằng `conv` (mặc định 50%)."""

    def __init__(self, obs, p_hist, ref_price=98.0, sigma=None, conv=0.5, demand_mult=1.0, shift_late=0.0):
        self.sigma = sigma or {"L": 0.25, "B": 0.5}
        self.ref = ref_price
        self.m = {s: p_hist[s] * np.exp(-self.sigma[s] * norm.ppf(1 - conv)) for s in "LB"}
        self.lam = {}
        for s in "LB":
            lam = obs[s] / self.prob_buy(s, p_hist[s] * ref_price)
            if shift_late > 0:   # dồn một phần nhu cầu sang 7 ngày cuối (xu hướng đặt sát ngày)
                moved = lam[7:] * shift_late
                lam = lam.copy(); lam[7:] -= moved
                lam[:7] += moved.sum() * lam[:7] / lam[:7].sum()
            self.lam[s] = lam * demand_mult

    def prob_buy(self, s, price_eur):
        return 1 - norm.cdf((np.log(price_eur / self.ref) - np.log(self.m[s])) / self.sigma[s])

    def rate(self, t, price_eur):
        """Cường độ khách MUA tại ngày t (t = số ngày còn lại, 1..T) với mức giá price."""
        return sum(self.lam[s][t - 1] * self.prob_buy(s, price_eur) for s in "LB")


# ----------------------------------------------------------------------------
# 2. MDP định giá và Dynamic Programming
# ----------------------------------------------------------------------------
class PricingMDP:
    """Trạng thái (t, x): còn t ngày, còn x phòng. Hành động: chọn một mức giá trong `prices`.
    Bán min(D, x) phòng, D ~ Poisson(rate(t, p)). Episode kết thúc khi t = 0 hoặc x = 0."""

    def __init__(self, scenario, prices=(85, 100, 115, 130, 150), T=30, C=10, dmax=80):
        self.sc, self.prices, self.T, self.C = scenario, np.array(prices, float), T, C
        self.nA = len(prices)
        self.rates = np.array([[scenario.rate(t, p) for p in prices] for t in range(1, T + 1)])  # [t-1, a]
        d = np.arange(dmax + 1)
        self.pmf = np.array([[poisson.pmf(d, self.rates[t, a]) for a in range(self.nA)] for t in range(T)])
        self.pmf[:, :, -1] += 1 - self.pmf.sum(axis=2)      # dồn đuôi vào dmax
        self.d = d

    def _q(self, t, x, V_prev):
        s = np.minimum(self.d, x)
        return np.array([np.sum(self.pmf[t - 1, a] * (self.prices[a] * s + V_prev[x - s])) for a in range(self.nA)])

    def solve(self):
        """Quy hoạch động lùi: trả về V[t][x], Q[t][x][a], chính sách tối ưu pi[t][x]."""
        T, C = self.T, self.C
        V = np.zeros((T + 1, C + 1)); Q = np.zeros((T + 1, C + 1, self.nA)); pi = np.zeros((T + 1, C + 1), int)
        for t in range(1, T + 1):
            for x in range(1, C + 1):
                Q[t, x] = self._q(t, x, V[t - 1]); pi[t, x] = int(np.argmax(Q[t, x])); V[t, x] = Q[t, x].max()
        return V, Q, pi

    def evaluate(self, pol):
        """Doanh thu kỳ vọng chính xác của chính sách xác định pol[t][x] từ (T, C)."""
        V = np.zeros(self.C + 1)
        for t in range(1, self.T + 1):
            Vn = np.zeros(self.C + 1)
            for x in range(1, self.C + 1):
                a = pol[t][x]; s = np.minimum(self.d, x)
                Vn[x] = np.sum(self.pmf[t - 1, a] * (self.prices[a] * s + V[x - s]))
            V = Vn
        return V[self.C]

    def occupancy(self, pol):
        """Xác suất ở trạng thái (t, x) khi theo pol từ (T, C) – dùng làm trọng số khi so chính sách."""
        occ = np.zeros((self.T + 1, self.C + 1)); occ[self.T, self.C] = 1.0
        for t in range(self.T, 0, -1):
            for x in range(1, self.C + 1):
                if occ[t, x] == 0: continue
                a = pol[t][x]
                for dd in range(self.d[-1] + 1):
                    p = self.pmf[t - 1, a, dd]
                    if p < 1e-15: continue
                    occ[t - 1, x - min(dd, x)] += occ[t, x] * p
        return occ

    def best_fixed(self):
        vals = [self.evaluate(np.full((self.T + 1, self.C + 1), a)) for a in range(self.nA)]
        return int(np.argmax(vals)), vals


# ----------------------------------------------------------------------------
# 3. Bộ mô phỏng nhanh (lấy mẫu Poisson bằng bảng CDF)
# ----------------------------------------------------------------------------
class FastPricingSim:
    def __init__(self, mdp, seed):
        self.mdp, self.rng = mdp, random.Random(seed)
        cdf = np.cumsum(mdp.pmf, axis=2); cdf[:, :, -1] = 1.0
        self.cdf = [[cdf[t, a].tolist() for a in range(mdp.nA)] for t in range(mdp.T)]
        self.prices = mdp.prices.tolist()

    def step(self, t, x, a):
        d = bisect.bisect_right(self.cdf[t - 1][a], self.rng.random())
        sold = d if d < x else x
        return t - 1, x - sold, self.prices[a] * sold

    def rollout(self, t, x, policy, first_action=None):
        ep = []
        first = True
        while t > 0 and x > 0:
            a = first_action if (first and first_action is not None) else policy(t, x)
            first = False
            t2, x2, r = self.step(t, x, a)
            ep.append((t, x, a, r)); t, x = t2, x2
        return ep


# ----------------------------------------------------------------------------
# 4. Thuật toán học
# ----------------------------------------------------------------------------
class Learner:
    def __init__(self, mdp, seed):
        self.mdp, self.sim = mdp, FastPricingSim(mdp, seed)
        self.rng = random.Random(seed + 7919)
        T, C, nA = mdp.T, mdp.C, mdp.nA
        self.Q = [[[0.0] * nA for _ in range(C + 1)] for _ in range(T + 1)]
        self.greedy = [[self.rng.randrange(nA) for _ in range(C + 1)] for _ in range(T + 1)]
        self.k = 0; self.revenue = []

    def refresh(self, t, x):
        row = self.Q[t][x]; m = max(row)
        best = [i for i, q in enumerate(row) if q == m]
        self.greedy[t][x] = best[0] if len(best) == 1 else self.rng.choice(best)

    def eps_act(self, t, x, eps):
        if self.rng.random() < eps:
            return self.rng.randrange(self.mdp.nA)
        return self.greedy[t][x]


class OnPolicyMC(Learner):
    """On-policy MC Control, ε-greedy. alpha=None: trung bình mẫu; alpha số: bước cố định."""
    def __init__(self, mdp, seed, eps_schedule, alpha=None):
        super().__init__(mdp, seed)
        self.eps_schedule, self.alpha = eps_schedule, alpha
        self.N = [[[0] * mdp.nA for _ in range(mdp.C + 1)] for _ in range(mdp.T + 1)]

    def run_episode(self):
        self.k += 1; eps = self.eps_schedule(self.k)
        ep = self.sim.rollout(self.mdp.T, self.mdp.C, lambda t, x: self.eps_act(t, x, eps))
        self.revenue.append(sum(e[3] for e in ep))
        update_mc(self, ep)


class MCExploringStarts(Learner):
    def __init__(self, mdp, seed):
        super().__init__(mdp, seed)
        self.N = [[[0] * mdp.nA for _ in range(mdp.C + 1)] for _ in range(mdp.T + 1)]
        self.alpha = None

    def run_episode(self):
        self.k += 1
        t0 = self.rng.randint(1, self.mdp.T); x0 = self.rng.randint(1, self.mdp.C); a0 = self.rng.randrange(self.mdp.nA)
        ep = self.sim.rollout(t0, x0, lambda t, x: self.greedy[t][x], first_action=a0)
        self.revenue.append(np.nan)   # ES không bắt đầu từ (T, C) nên không tính doanh thu thực tế
        update_mc(self, ep)


def update_mc(L, ep):
    """Mỗi (t, x) chỉ xuất hiện một lần trong Episode (t giảm dần) ⇒ First-Visit ≡ Every-Visit."""
    G = 0.0
    for t, x, a, r in reversed(ep):
        G += r                                   # γ = 1
        if L.alpha is None:
            L.N[t][x][a] += 1
            L.Q[t][x][a] += (G - L.Q[t][x][a]) / L.N[t][x][a]
        else:
            L.Q[t][x][a] += L.alpha * (G - L.Q[t][x][a])
        L.refresh(t, x)


class OffPolicyMC(Learner):
    """Off-policy MC Control, weighted IS. behavior(t, x) trả về (a, b(a|s)); mặc định ε_b-greedy theo Q."""
    def __init__(self, mdp, seed, eps_b=0.3):
        super().__init__(mdp, seed)
        self.C_ = [[[0.0] * mdp.nA for _ in range(mdp.C + 1)] for _ in range(mdp.T + 1)]
        self.eps_b = eps_b; self.tail = []

    def behavior(self, t, x):
        a = self.eps_act(t, x, self.eps_b)
        pb = self.eps_b / self.mdp.nA + (1 - self.eps_b if a == self.greedy[t][x] else 0)
        return a, pb

    def learn_from(self, ep, probs):
        G, W, used = 0.0, 1.0, 0
        for (t, x, a, r), pb in zip(reversed(ep), reversed(probs)):
            G += r
            self.C_[t][x][a] += W
            self.Q[t][x][a] += (W / self.C_[t][x][a]) * (G - self.Q[t][x][a])
            self.refresh(t, x); used += 1
            if a != self.greedy[t][x]:
                break
            W /= pb
        self.tail.append(used / len(ep))

    def run_episode(self):
        self.k += 1
        probs = []
        def pol(t, x):
            a, pb = self.behavior(t, x); probs.append(pb); return a
        ep = self.sim.rollout(self.mdp.T, self.mdp.C, pol)
        self.revenue.append(sum(e[3] for e in ep))
        self.learn_from(ep, probs)


def snapshot(L, V0, pi_star, occ):
    pol = L.greedy
    v = L.mdp.evaluate(pol)
    agree = sum(occ[t, x] * (pol[t][x] == pi_star[t][x]) for t in range(1, L.mdp.T + 1) for x in range(1, L.mdp.C + 1))
    agree /= occ[1:, 1:].sum()
    return v / V0, agree


# ----------------------------------------------------------------------------
# 5. Monte Carlo Control dạng batch: hai cách ước lượng Q từ cùng một dữ liệu
# ----------------------------------------------------------------------------
# Ý tưởng: ước lượng Monte Carlo Q(s,a) = trung bình các Return quan sát tại (s,a)
# chính là nghiệm bình phương tối thiểu của bài toán hồi quy G ~ one-hot(s,a).
# - Hướng 1 (không dùng scikit-learn): tính trung bình nhóm bằng np.bincount.
# - Hướng 2 (scikit-learn): OneHotEncoder + hồi quy tuyến tính bình phương tối thiểu (Ridge với α = 1e-10,
#   chỉ để ma trận khả nghịch khi có ô chưa được thăm; sai lệch so với trung bình mẫu < 1e-6).
# Cả hai dùng chung vòng lặp GPI: chạy K Episode theo chính sách ε-greedy hiện tại,
# ước lượng lại Q từ toàn bộ dữ liệu, rồi cải thiện chính sách (greedy theo Q).
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline


class BatchMC(Learner):
    """On-policy Monte Carlo Control dạng batch. estimator='numpy' hoặc 'sklearn'."""

    def __init__(self, mdp, seed, eps=0.1, K=1000, estimator="numpy"):
        super().__init__(mdp, seed)
        self.eps, self.K, self.estimator = eps, K, estimator
        self.n_cells = (mdp.T + 1) * (mdp.C + 1) * mdp.nA
        self.cells, self.returns = [], []
        if estimator == "sklearn":
            self.model = make_pipeline(
                OneHotEncoder(categories=[np.arange(self.n_cells)], sparse_output=True),
                Ridge(alpha=1e-10, fit_intercept=False, solver="cholesky"))
        # chính sách ban đầu: greedy xác định (argmax trên Q = 0 → mức giá đầu tiên) để hai hướng trùng nhau
        self.greedy = [[0] * (mdp.C + 1) for _ in range(mdp.T + 1)]
        self.fit_time = 0.0

    def cell(self, t, x, a):
        return (t * (self.mdp.C + 1) + x) * self.mdp.nA + a

    def run_episode(self):
        self.k += 1
        ep = self.sim.rollout(self.mdp.T, self.mdp.C, lambda t, x: self.eps_act(t, x, self.eps))
        self.revenue.append(sum(e[3] for e in ep))
        G = 0.0
        for t, x, a, r in reversed(ep):
            G += r
            self.cells.append(self.cell(t, x, a)); self.returns.append(G)
        if self.k % self.K == 0:
            self.evaluate_and_improve()

    def estimate_Q(self):
        cells = np.asarray(self.cells); y = np.asarray(self.returns)
        if self.estimator == "numpy":
            cnt = np.bincount(cells, minlength=self.n_cells)
            tot = np.bincount(cells, weights=y, minlength=self.n_cells)
            q = np.divide(tot, cnt, out=np.zeros(self.n_cells), where=cnt > 0)
        else:
            self.model.fit(cells.reshape(-1, 1), y)
            q = self.model[-1].coef_.copy()
        return q.reshape(self.mdp.T + 1, self.mdp.C + 1, self.mdp.nA)

    def evaluate_and_improve(self):
        import time as _t
        t0 = _t.perf_counter(); Qhat = self.estimate_Q(); self.fit_time += _t.perf_counter() - t0
        self.Qhat = Qhat
        g = np.round(Qhat, 4).argmax(axis=2)           # cải thiện chính sách: greedy theo Q
                                                       # (làm tròn 1e-4 € để hai hướng phá hòa giống nhau)
        self.greedy = g.tolist()
