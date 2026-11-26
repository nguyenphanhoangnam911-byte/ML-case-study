# Bảng ánh xạ lý thuyết → mã nguồn (case RFM)

Số dòng theo `case_rfm_clustering/src/rfm_scratch.py` (bản nộp). Sau mỗi lần sửa code, phải cập nhật lại bảng. Cột AI log ID điền theo `ai_log.md`.

| Công thức / bước | Thuật toán | Hàm / lớp | Dòng | Bằng chứng kiểm chứng | AI log ID |
|---|---|---|---|---|---|
| WCSS = Σ ‖x − μ‖² | K-Means, Mini-Batch | `compute_inertia` | 14–17 | Khớp WCSS sklearn trên cùng nhãn (báo cáo 7.1) | |
| Abs. cost = Σ ‖x − m‖ | K-Medoids | `compute_abs_cost` | 20–22 | 2.122,5 trùng sklearn-extra (7.1) | |
| Khoảng cách Euclid bình phương | K-Means, Mini-Batch | `pairwise_sq_euclidean` | 129–135 | Dùng trong bước gán; WCSS khớp sklearn | |
| Khoảng cách Euclid / Manhattan | K-Medoids | `pairwise_distance` | 146–153 | Medoid trùng sklearn-extra ở seed 42 (7.1) | |
| Khởi tạo ngẫu nhiên / k-means++ | K-Means | `CustomKMeans._init_centroids` | 205–219 | k-means++: WCSS 4.026,8 so với 4.027,6 của sklearn (seed 42) | |
| Gán điểm vào tâm gần nhất | K-Means | `CustomKMeans._assign_clusters` | 223–227 | Như trên | |
| Cập nhật tâm = trung bình cụm | K-Means | `CustomKMeans._update_centroids` | 230–243 | Như trên | |
| Vòng lặp Lloyd và điều kiện dừng | K-Means | `CustomKMeans._single_run` | 246–265 | WCSS hội tụ 4.044,49 (seed 42, khởi tạo ngẫu nhiên) | |
| Chọn nghiệm tốt nhất qua n_init | K-Means | `CustomKMeans.fit` | 268–276 | Như trên | |
| Cập nhật tâm theo batch, tốc độ học 1/số điểm | Mini-Batch | `CustomMiniBatchKMeans._update_centroids` | 334–346 | WCSS 4.399,09 (seed 42); dao động 4.332–5.820 qua seed 0–4 (7.1) | |
| Vòng batch và dừng sớm (trung bình mũ) | Mini-Batch | `CustomMiniBatchKMeans._single_run` | 348–376 | Như trên | |
| Ma trận khoảng cách N×N | K-Medoids | `CustomKMedoids._distance_matrix` | 431–434 | Medoid trùng sklearn-extra ở seed 42 | |
| Gán cụm theo medoid gần nhất | K-Medoids | `CustomKMedoids._assign_clusters` | 442–446 | Như trên | |
| Chi phí sau hoán đổi (medoid, ứng viên) | K-Medoids | `_swap_cost_naive`, `_best_swap` | 450–473 | Chi phí 2.122,466 (seed 42) | |
| Vòng hoán đổi PAM | K-Medoids | `CustomKMedoids._single_run` | 475–489 | Như trên | |
| Silhouette | Chỉ số | `silhouette_numpy` | 512–526 | 0,613285 trùng `silhouette_score` (nhãn K-Means, X) | |
| ARI | Chỉ số | `adjusted_rand_index` | 529–539 | 0,928586 trùng `adjusted_rand_score` đến 6 chữ số thập phân | |
| Ghép tâm 1-1 | Đối chiếu | `match_centers` | 65–70 | Dùng ở 7.1 | |

Đa số bằng chứng nằm ở báo cáo mục 7.1; kết quả nhiều seed ở `experiments/results/` (resA.json, resB.json).
