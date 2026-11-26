"""Cài đặt từ đầu (chỉ dùng NumPy) của K-Means, Mini-Batch K-Means và K-Medoids (PAM).
Trích nguyên văn từ RFM_Customer_Clustering.ipynb (các ô 15, 30, 32, 34, 36, 42)."""
import time
from itertools import permutations
from typing import Dict, List, Literal, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATURES: List[str] = ["Recency", "Frequency", "Monetary"]


def compute_inertia(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """WCSS = tổng bình phương khoảng cách Euclid từ mỗi điểm tới tâm cụm của nó."""
    diff = X - centers[labels]
    return float(np.einsum("ij,ij->", diff, diff))


def compute_abs_cost(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """Tổng khoảng cách Euclid (không bình phương) từ điểm tới tâm cụm."""
    return float(np.linalg.norm(X - centers[labels], axis=1).sum())


def relabel_by_monetary(labels: np.ndarray, centers: np.ndarray, col: int = 2
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Đánh lại số hiệu cụm theo thứ tự tăng dần của tâm trên trục Monetary.

    Trả về (nhãn mới, tâm đã sắp xếp, order) với order[k_mới] = k_cũ, dùng để sắp xếp
    các thông tin đi kèm theo cụm (vd. medoid_indices_[order]).
    """
    order = np.argsort(centers[:, col])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return inverse[labels], centers[order], order


def profile_clusters(rfm_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Chân dung từng cụm (đơn vị gốc): kích thước, TRUNG VỊ R/F/M và trung bình M để đối chiếu."""
    tmp = rfm_df[FEATURES].copy()
    tmp["Cluster"] = labels
    g = tmp.groupby("Cluster")
    out = pd.DataFrame({
        "Size": g.size(),
        "Recency (median)": g["Recency"].median(),
        "Frequency (median)": g["Frequency"].median(),
        "Monetary (median)": g["Monetary"].median(),
        "Monetary (mean)": g["Monetary"].mean(),
    })
    return out.round(1)


def fit_timed(make_model: Callable[[], object], X: np.ndarray, repeats: int = 3
              ) -> Tuple[object, np.ndarray, float]:
    """Fit `repeats` lần (cùng random_state nên cùng kết quả); trả về model, nhãn, trung vị thời gian."""
    times = []
    for _ in range(repeats):
        model = make_model()
        t0 = time.perf_counter()
        labels = model.fit_predict(X)
        times.append(time.perf_counter() - t0)
    return model, labels, float(np.median(times))


def match_centers(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Ghép 1-1 tâm A[i] ↔ B[p(i)] với tổng khoảng cách nhỏ nhất; trả về khoảng cách từng cặp."""
    D = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=-1))
    rows = np.arange(len(A))
    best = min(permutations(range(len(B))), key=lambda p: D[rows, list(p)].sum())
    return D[rows, list(best)]


def summarize_results(results: Dict[str, dict], X: np.ndarray,
                      silhouette_fn: Callable[[np.ndarray, np.ndarray], float]) -> pd.DataFrame:
    """Bảng tổng hợp chỉ số cho một dict kết quả {tên: {labels, centers, time}}."""
    rows = []
    for name, res in results.items():
        labels, centers = res["labels"], res["centers"]
        rows.append({
            "Model": name,
            "Inertia (WCSS)": compute_inertia(X, labels, centers),
            "Abs. cost (ΣL2)": compute_abs_cost(X, labels, centers),
            "Silhouette": silhouette_fn(X, labels),
            "Time (s, median)": res["time"],
            "Cluster sizes": sorted(np.bincount(labels, minlength=centers.shape[0]).tolist(), reverse=True),
        })
    return pd.DataFrame(rows).set_index("Model")


def plot_clusters_1x3(X: np.ndarray, results: Dict[str, dict], suptitle: str,
                      axes_idx: Tuple[int, int] = (1, 2), zoom: bool = False) -> None:
    """Vẽ subplot 1x3: mỗi model một ô; tâm/medoid là dấu X đỏ đậm.

    axes_idx : cặp đặc trưng đem vẽ (0=R, 1=F, 2=M). F–M là nơi có outlier; R–M là nơi
               phần lớn các cụm thực sự tách nhau (nên cần xem cả hai).
    zoom     : True => thu hẹp khung nhìn vào 99% dữ liệu để thấy cấu trúc vùng dày đặc.
    """
    i, j = axes_idx
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.6), sharex=True, sharey=True)
    palette = plt.get_cmap("tab10")

    for ax, (name, res) in zip(axs, results.items()):
        labels, centers, _ = relabel_by_monetary(res["labels"], res["centers"])
        for k in range(centers.shape[0]):
            mask = labels == k
            ax.scatter(X[mask, i], X[mask, j], s=11, alpha=0.55, color=palette(k),
                       label=f"Cụm {k} (n={mask.sum()})")
        ax.scatter(centers[:, i], centers[:, j], marker="X", s=330, c="red",
                   edgecolors="black", linewidths=1.6, zorder=10,
                   label=res.get("center_name", "Tâm"))
        ax.set_title(f"{name}\n{res.get('subtitle', '')}", fontsize=11)
        ax.set_xlabel(f"{FEATURES[i]} (đã chuẩn hóa)")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axs[0].set_ylabel(f"{FEATURES[j]} (đã chuẩn hóa)")

    if zoom:
        for idx, setter in ((i, axs[0].set_xlim), (j, axs[0].set_ylim)):
            lo, hi = X[:, idx].min(), np.quantile(X[:, idx], 0.99)
            pad = 0.15 * (hi - lo)
            setter(lo - pad, hi + pad)

    view = f"{FEATURES[i]}–{FEATURES[j]}"
    fig.suptitle(f"{suptitle}  [{view}, " + ("zoom vào 99% dữ liệu]" if zoom else "toàn bộ dữ liệu, gồm outlier]"),
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def pairwise_sq_euclidean(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Ma trận bình phương khoảng cách Euclid (n×k) bằng phép nhân ma trận."""
    a2 = np.einsum("ij,ij->i", A, A)[:, None]
    b2 = np.einsum("ij,ij->i", B, B)[None, :]
    d2 = a2 + b2 - 2.0 * (A @ B.T)
    np.maximum(d2, 0.0, out=d2)            # chặn sai số làm tròn âm nhỏ
    return d2


def pairwise_manhattan(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Ma trận khoảng cách Manhattan (L1); vòng lặp chỉ chạy theo số chiều d (rất nhỏ)."""
    d1 = np.zeros((A.shape[0], B.shape[0]))
    for j in range(A.shape[1]):
        d1 += np.abs(A[:, j][:, None] - B[:, j][None, :])
    return d1


def pairwise_distance(A: np.ndarray, B: np.ndarray,
                      metric: Literal["euclidean", "manhattan"] = "euclidean") -> np.ndarray:
    """Ma trận khoảng cách (không bình phương) giữa mọi cặp (A_i, B_j)."""
    if metric == "euclidean":
        return np.sqrt(pairwise_sq_euclidean(A, B))
    if metric == "manhattan":
        return pairwise_manhattan(A, B)
    raise ValueError(f"metric không hỗ trợ: {metric!r}")


def one_hot(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Ma trận one-hot (n×K); dùng để cộng điểm theo cụm bằng một phép nhân ma trận."""
    oh = np.zeros((labels.shape[0], n_clusters))
    oh[np.arange(labels.shape[0]), labels] = 1.0
    return oh


def _check_X(X: np.ndarray, n_clusters: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X phải là mảng 2 chiều (n_samples, n_features).")
    if X.shape[0] < n_clusters:
        raise ValueError("Số mẫu phải >= n_clusters.")
    return X


class CustomKMeans:
    """K-Means (Lloyd) cài đặt bằng NumPy.

    Nguyên lý: luân phiên (1) gán mỗi điểm vào tâm gần nhất theo Euclid^2 và
    (2) cập nhật tâm = trung bình cụm. Mỗi bước giảm (không tăng) WCSS
    J = sum_i ||x_i - mu_{c_i}||^2 nên thuật toán hội tụ về cực tiểu địa phương.

    Parameters
    ----------
    n_clusters   : số cụm K.
    init         : 'random' (chọn K điểm ngẫu nhiên) hoặc 'k-means++'.
    n_init       : số lần chạy với khởi tạo khác nhau; giữ nghiệm có inertia nhỏ nhất.
    max_iter     : số vòng lặp tối đa mỗi lần chạy.
    tol          : ngưỡng dịch chuyển tâm (nhân với phương sai trung bình của dữ liệu).
    random_state : seed để tái lập kết quả.
    """

    def __init__(self, n_clusters: int = 4, init: Literal["random", "k-means++"] = "random",
                 n_init: int = 10, max_iter: int = 300, tol: float = 1e-4,
                 random_state: Optional[int] = None) -> None:
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = np.inf
        self.n_iter_: int = 0
        self.inertia_history_: List[float] = []

    # ---- 1. Khởi tạo -------------------------------------------------
    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Khởi tạo K tâm: ngẫu nhiên (Forgy) hoặc k-means++ (xác suất ∝ D(x)^2)."""
        n = X.shape[0]
        if self.init == "random":
            return X[rng.choice(n, size=self.n_clusters, replace=False)].copy()

        centers = np.empty((self.n_clusters, X.shape[1]))
        centers[0] = X[rng.integers(n)]
        closest_d2 = pairwise_sq_euclidean(X, centers[:1]).ravel()
        for c in range(1, self.n_clusters):          # lặp theo K, không theo n
            total = closest_d2.sum()
            probs = closest_d2 / total if total > 0 else None
            centers[c] = X[rng.choice(n, p=probs)]
            closest_d2 = np.minimum(closest_d2, pairwise_sq_euclidean(X, centers[c:c + 1]).ravel())
        return centers

    # ---- 2. Gán cụm --------------------------------------------------
    @staticmethod
    def _assign_clusters(X: np.ndarray, centroids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Trả về (nhãn, bình phương khoảng cách tới tâm gần nhất) cho mọi điểm."""
        d2 = pairwise_sq_euclidean(X, centroids)             # (n, K) một phép nhân ma trận
        labels = np.argmin(d2, axis=1)
        return labels, d2[np.arange(X.shape[0]), labels]

    # ---- 3. Cập nhật tâm ---------------------------------------------
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray, min_d2: np.ndarray) -> np.ndarray:
        """Tâm mới = trung bình các điểm thuộc cụm; tổng theo cụm tính bằng one-hot^T @ X."""
        onehot = one_hot(labels, self.n_clusters)
        counts = onehot.sum(axis=0)
        centroids = (onehot.T @ X) / np.maximum(counts, 1.0)[:, None]

        empty = np.flatnonzero(counts == 0)                  # xử lý cụm rỗng
        if empty.size:
            d2 = min_d2.copy()
            for k in empty:
                far = int(np.argmax(d2))
                centroids[k] = X[far]
                d2[far] = 0.0
        return centroids

    # ---- Một lần chạy ------------------------------------------------
    def _single_run(self, X: np.ndarray, rng: np.random.Generator
                    ) -> Tuple[np.ndarray, np.ndarray, float, int, List[float]]:
        centers = self._init_centroids(X, rng)
        tol_abs = self.tol * float(np.mean(np.var(X, axis=0)))
        history: List[float] = []
        prev_labels: Optional[np.ndarray] = None
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            labels, min_d2 = self._assign_clusters(X, centers)
            new_centers = self._update_centroids(X, labels, min_d2)
            shift = float(np.sum((new_centers - centers) ** 2))
            centers = new_centers
            history.append(float(min_d2.sum()))
            if prev_labels is not None and np.array_equal(labels, prev_labels):
                break                                         # phép gán không đổi => hội tụ
            prev_labels = labels
            if shift <= tol_abs:
                break                                         # tâm gần như đứng yên
        labels, min_d2 = self._assign_clusters(X, centers)
        return centers, labels, float(min_d2.sum()), n_iter, history

    # ---- API ---------------------------------------------------------
    def fit(self, X: np.ndarray) -> "CustomKMeans":
        X = _check_X(X, self.n_clusters)
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_init):
            centers, labels, inertia, n_iter, hist = self._single_run(X, rng)
            if inertia < self.inertia_:
                self.cluster_centers_, self.labels_ = centers, labels
                self.inertia_, self.n_iter_, self.inertia_history_ = inertia, n_iter, hist
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._assign_clusters(np.asarray(X, dtype=float), self.cluster_centers_)[0]

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_


class CustomMiniBatchKMeans:
    """Mini-Batch K-Means cài đặt bằng NumPy.

    Nguyên lý: mỗi vòng rút ngẫu nhiên một mini-batch, gán nhãn cho batch, rồi cập nhật
    tâm bằng trung bình di động có trọng số đếm (per-center count):
        mu_k <- (n_k * mu_k + S_k) / (n_k + b_k),   n_k <- n_k + b_k
    với b_k, S_k là số điểm và tổng điểm của batch trong cụm k. Learning rate
    eta_k = b_k / (n_k + b_k) giảm dần => tâm hội tụ về trung bình cụm.

    Parameters
    ----------
    n_clusters         : số cụm K.
    batch_size         : kích thước mini-batch.
    init               : 'random' hoặc 'k-means++'.
    n_init             : số lần chạy đầy đủ; chọn nghiệm có inertia (trên TOÀN BỘ dữ liệu) nhỏ nhất.
    max_iter           : số bước mini-batch tối đa mỗi lần chạy.
    max_no_improvement : dừng sớm nếu EWA của inertia theo batch không giảm trong ngần ấy bước.
    tol                : ngưỡng dịch chuyển tâm (nhân phương sai TB dữ liệu); 0 = tắt tiêu chí này.
    random_state       : seed để tái lập.
    """

    def __init__(self, n_clusters: int = 4, batch_size: int = 1024,
                 init: Literal["random", "k-means++"] = "random", n_init: int = 5,
                 max_iter: int = 300, max_no_improvement: int = 10, tol: float = 0.0,
                 random_state: Optional[int] = None) -> None:
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.max_no_improvement = max_no_improvement
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = np.inf
        self.n_steps_: int = 0
        self.counts_: Optional[np.ndarray] = None

    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # Tái sử dụng logic khởi tạo của CustomKMeans (cùng cách chọn tâm ban đầu).
        return CustomKMeans(n_clusters=self.n_clusters, init=self.init)._init_centroids(X, rng)

    @staticmethod
    def _assign_clusters(X: np.ndarray, centroids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d2 = pairwise_sq_euclidean(X, centroids)
        labels = np.argmin(d2, axis=1)
        return labels, d2[np.arange(X.shape[0]), labels]

    def _update_centroids(self, Xb: np.ndarray, labels_b: np.ndarray,
                          centers: np.ndarray, counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Cập nhật tâm theo trung bình di động (vectorized trên toàn batch)."""
        onehot = one_hot(labels_b, self.n_clusters)
        batch_counts = onehot.sum(axis=0)                    # b_k
        batch_sums = onehot.T @ Xb                           # S_k
        new_counts = counts + batch_counts                   # n_k + b_k

        new_centers = centers.copy()
        active = batch_counts > 0                            # chỉ cập nhật tâm có điểm trong batch
        new_centers[active] = (centers[active] * counts[active, None] + batch_sums[active]) \
                              / new_counts[active, None]
        return new_centers, new_counts

    def _single_run(self, X: np.ndarray, rng: np.random.Generator
                    ) -> Tuple[np.ndarray, np.ndarray, float, int, np.ndarray]:
        n = X.shape[0]
        bs = min(self.batch_size, n)
        centers = self._init_centroids(X, rng)
        counts = np.zeros(self.n_clusters)
        tol_abs = self.tol * float(np.mean(np.var(X, axis=0)))
        alpha = min(1.0, 2.0 * bs / (n + 1))                        # hệ số EWA giống sklearn
        ewa, best_ewa, no_improve, step = None, np.inf, 0, 0
        for step in range(1, self.max_iter + 1):
            batch_idx = rng.choice(n, size=bs, replace=False)       # rút mini-batch ngẫu nhiên
            Xb = X[batch_idx]
            labels_b, d2_b = self._assign_clusters(Xb, centers)
            batch_inertia = float(d2_b.mean())                      # inertia/điểm, tính TRƯỚC khi cập nhật
            new_centers, counts = self._update_centroids(Xb, labels_b, centers, counts)
            shift = float(np.sum((new_centers - centers) ** 2))
            centers = new_centers

            if self.tol > 0 and shift <= tol_abs:                   # (1) tâm gần như đứng yên
                break
            ewa = batch_inertia if ewa is None else (1 - alpha) * ewa + alpha * batch_inertia
            if ewa < best_ewa:
                best_ewa, no_improve = ewa, 0
            else:
                no_improve += 1
            if no_improve >= self.max_no_improvement:               # (2) inertia không cải thiện
                break
        labels, min_d2 = self._assign_clusters(X, centers)          # nhãn cuối trên toàn bộ dữ liệu
        return centers, labels, float(min_d2.sum()), step, counts

    def fit(self, X: np.ndarray) -> "CustomMiniBatchKMeans":
        X = _check_X(X, self.n_clusters)
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_init):
            centers, labels, inertia, steps, counts = self._single_run(X, rng)
            if inertia < self.inertia_:
                self.cluster_centers_, self.labels_, self.inertia_ = centers, labels, inertia
                self.n_steps_, self.counts_ = steps, counts
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._assign_clusters(np.asarray(X, dtype=float), self.cluster_centers_)[0]

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_


class CustomKMedoids:
    """K-Medoids (PAM, best-improvement swap) cài đặt bằng NumPy.

    Nguyên lý: tối thiểu hoá T = sum_i d(x_i, m_{c_i}) với mỗi medoid m_k là một điểm dữ liệu thật.
    Khởi tạo K medoid ngẫu nhiên; lặp: thử hoán đổi mỗi medoid với mỗi non-medoid, tính lại
    tổng cost tuyệt đối, giữ hoán đổi giảm cost nhiều nhất (các hoán đổi còn lại bị revert),
    dừng khi không còn hoán đổi nào cải thiện. Vì cost chỉ tăng tuyến tính theo khoảng cách,
    outlier không kéo được medoid như cách chúng kéo trung bình trong K-Means.

    Parameters
    ----------
    n_clusters   : số cụm K.
    metric       : 'euclidean' hoặc 'manhattan'.
    n_init       : số lần khởi tạo ngẫu nhiên; giữ nghiệm có cost thấp nhất.
    max_iter     : số lần swap tối đa mỗi lần chạy.
    chunk_size   : số ứng viên xử lý mỗi lượt (giới hạn bộ nhớ tạm cỡ chunk_size × N).
    random_state : seed để tái lập.
    """

    def __init__(self, n_clusters: int = 4, metric: Literal["euclidean", "manhattan"] = "euclidean",
                 n_init: int = 3, max_iter: int = 100, chunk_size: int = 512,
                 random_state: Optional[int] = None) -> None:
        self.n_clusters = n_clusters
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.chunk_size = chunk_size
        self.random_state = random_state
        self.medoid_indices_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = np.inf              # = tổng cost tuyệt đối
        self.n_iter_: int = 0
        self.cost_history_: List[float] = []

    # ---- Ma trận khoảng cách (đối xứng tuyệt đối, float32 để tiết kiệm RAM) ----
    def _distance_matrix(self, X: np.ndarray) -> np.ndarray:
        D = pairwise_distance(X, X, self.metric).astype(np.float32)
        np.fill_diagonal(D, 0.0)
        return np.maximum(D, D.T)                 # ép đối xứng chính xác sau khi làm tròn

    # ---- 1. Khởi tạo: medoid là điểm dữ liệu THỰC ----
    def _init_medoids(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(n, size=self.n_clusters, replace=False)

    # ---- 2. Gán cụm theo medoid gần nhất ----
    @staticmethod
    def _assign_clusters(D: np.ndarray, medoids: np.ndarray) -> Tuple[np.ndarray, float]:
        sub = D[:, medoids]                                         # (n, K)
        labels = np.argmin(sub, axis=1)
        cost = float(sub[np.arange(D.shape[0]), labels].sum(dtype=np.float64))
        return labels, cost

    # ---- Tham chiếu: hoán đổi 1 cặp -> tính cost (dùng để kiểm chứng bản vectorized) ----
    @staticmethod
    def _swap_cost_naive(D: np.ndarray, medoids: np.ndarray, slot: int, candidate: int) -> float:
        trial = medoids.copy()
        trial[slot] = candidate                                     # thử swap
        return float(D[:, trial].min(axis=1).sum(dtype=np.float64)) # revert là tự động: medoids gốc không đổi

    # ---- 3. Duyệt toàn bộ cặp swap (vectorized theo ứng viên) ----
    def _best_swap(self, D: np.ndarray, medoids: np.ndarray, current_cost: float
                   ) -> Tuple[float, int, int]:
        """Trả về (cost_tốt_nhất, slot, ứng_viên); slot = -1 nếu không swap nào giảm cost."""
        n = D.shape[0]
        non_medoids = np.setdiff1d(np.arange(n), medoids, assume_unique=True)
        best_cost, best_slot, best_cand = current_cost, -1, -1
        for slot in range(self.n_clusters):                         # lặp theo K (nhỏ)
            others = np.delete(medoids, slot)
            d_other = (D[:, others].min(axis=1) if others.size
                       else np.full(n, np.inf, dtype=D.dtype))      # khoảng cách khi bỏ medoid `slot`
            for start in range(0, non_medoids.size, self.chunk_size):
                cand = non_medoids[start:start + self.chunk_size]
                # D đối xứng => D[cand, :] chính là cột khoảng cách tới từng ứng viên (đọc theo hàng nhanh hơn)
                costs = np.minimum(d_other[None, :], D[cand, :]).sum(axis=1, dtype=np.float64)
                j = int(np.argmin(costs))
                if costs[j] < best_cost - 1e-9:
                    best_cost, best_slot, best_cand = float(costs[j]), slot, int(cand[j])
        return best_cost, best_slot, best_cand

    def _single_run(self, D: np.ndarray, rng: np.random.Generator
                    ) -> Tuple[np.ndarray, np.ndarray, float, int, List[float]]:
        medoids = self._init_medoids(D.shape[0], rng)
        _, cost = self._assign_clusters(D, medoids)
        history = [cost]
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            new_cost, slot, cand = self._best_swap(D, medoids, cost)
            if slot < 0:                                            # mọi swap đều bị revert => hội tụ
                break
            medoids[slot] = cand                                    # giữ swap tốt nhất
            cost = new_cost
            history.append(cost)
        labels, cost = self._assign_clusters(D, medoids)
        return medoids, labels, cost, n_iter, history

    # ---- API ----
    def fit(self, X: np.ndarray) -> "CustomKMedoids":
        X = _check_X(X, self.n_clusters)
        rng = np.random.default_rng(self.random_state)
        D = self._distance_matrix(X)
        for _ in range(self.n_init):
            medoids, labels, cost, n_iter, hist = self._single_run(D, rng)
            if cost < self.inertia_:
                self.medoid_indices_, self.labels_, self.inertia_ = medoids.copy(), labels, cost
                self.n_iter_, self.cost_history_ = n_iter, hist
        self.cluster_centers_ = X[self.medoid_indices_]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmin(pairwise_distance(np.asarray(X, dtype=float),
                                           self.cluster_centers_, self.metric), axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_


def silhouette_numpy(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette trung bình, vectorized: tổng khoảng cách tới từng cụm = D @ one_hot."""
    n, K = X.shape[0], int(labels.max()) + 1
    D = pairwise_distance(X, X, "euclidean")
    onehot = one_hot(labels, K)
    counts = onehot.sum(axis=0)
    sums = D @ onehot                                            # (n, K)
    idx = np.arange(n)
    a = sums[idx, labels] / np.maximum(counts[labels] - 1.0, 1.0)
    mean_other = sums / np.maximum(counts, 1.0)[None, :]
    mean_other[idx, labels] = np.inf                             # loại cụm của chính điểm đó
    b = mean_other.min(axis=1)
    s = (b - a) / np.maximum(np.maximum(a, b), 1e-12)
    s[counts[labels] == 1] = 0.0                                 # cụm 1 điểm => s = 0
    return float(s.mean())


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    """ARI giữa hai phân hoạch (bất biến với hoán vị nhãn): 1 = trùng khớp hoàn toàn."""
    ka, kb = int(a.max()) + 1, int(b.max()) + 1
    cont = np.zeros((ka, kb))
    np.add.at(cont, (a, b), 1.0)
    comb2 = lambda v: v * (v - 1.0) / 2.0
    sum_ij, sum_a, sum_b = comb2(cont).sum(), comb2(cont.sum(1)).sum(), comb2(cont.sum(0)).sum()
    total = comb2(float(len(a)))
    expected = sum_a * sum_b / total
    max_index = 0.5 * (sum_a + sum_b)
    return float((sum_ij - expected) / (max_index - expected)) if max_index != expected else 1.0
