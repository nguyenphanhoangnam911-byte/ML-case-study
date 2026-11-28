# Nhật ký sử dụng AI (case RFM)

Mỗi lần AI đóng góp nội dung hoặc code được dùng trong bài có một mục log. ID dùng chung với bảng "AI prompting log" trong báo cáo (AI-00 đến AI-05 cho cả hai case); AI-00 đến AI-03 thuộc case này. Cột AI log ID trong [`theory_to_code_mapping.md`](theory_to_code_mapping.md) trỏ tới các ID này. Sinh viên giải thích được mọi dòng code và công thức trong bài.

## AI-00

| Trường | Nội dung |
|---|---|
| ID | AI-00 |
| Công cụ / ngày | Claude (claude.ai), 01/11/2026 |
| Mục đích | Viết khung notebook phân cụm RFM: tiền xử lý, ba thuật toán scikit-learn/sklearn-extra, ba lớp From Scratch bằng NumPy. |
| Đầu ra AI (tóm tắt) | Notebook RFM_Customer_Clustering.ipynb gồm 45 ô, 4 phần: tiền xử lý RFM; scikit-learn và sklearn-extra; From Scratch chỉ dùng NumPy (CustomKMeans, CustomMiniBatchKMeans, CustomKMedoids); so sánh hai cách. |
| Sinh viên đã chỉnh sửa gì | Chạy lại notebook trên máy, xử lý lỗi môi trường của scikit-learn-extra, dùng làm khung ban đầu. Các lỗi về dữ liệu và phương pháp được sửa ở AI-03. |
| Cách kiểm chứng | Chạy lại toàn bộ notebook; kiểm tra các hàm From Scratch theo bảng ánh xạ. |
| Vị trí dùng trong bài | `notebooks/RFM_Customer_Clustering.ipynb`, `src/rfm_scratch.py` (khung các lớp); báo cáo Mục 5. |

**Prompt (nguyên văn):**

> Đóng vai là một Senior Machine Learning Engineer chuyên về thuật toán Unsupervised Learning. Hãy viết cho tôi toàn bộ mã nguồn (code) cho một file Jupyter Notebook hoàn chỉnh, mô phỏng lại bài toán: "Phân cụm hành vi khách hàng bằng RFM Model".
>
> MỤC TIÊU:
> Huấn luyện và so sánh 3 thuật toán: K-Means, Mini-Batch K-Means, và K-Medoids.
> Tôi muốn code được chia làm 2 phần (2 cách tiếp cận) hoàn toàn độc lập trong cùng 1 Notebook:
> - Cách 1: Sử dụng thư viện Scikit-learn (Chuẩn công nghiệp, tối ưu production).
> - Cách 2: Code From Scratch chỉ dùng Numpy (Chứng minh hiểu rõ bản chất toán học đằng sau thuật toán).
>
> YÊU CẦU CẤU TRÚC JUPYTER NOTEBOOK:
> Hãy dùng các thẻ Markdown (#, ##, ###) để chia cấu trúc Notebook rõ ràng.
>
> ### PHẦN 1: TẢI DỮ LIỆU VÀ TIỀN XỬ LÝ
> - Sử dụng thư viện Pandas để đọc một file CSV .
> - Viết code (Data Engineering) để tổng hợp dữ liệu giao dịch thô này thành mô hình RFM (Recency, Frequency, Monetary) theo từng CustomerID.
> - Xử lý các giá trị dị thường cơ bản (như loại bỏ các Quantity bị âm).
> - CHÚ Ý CỐ TÌNH GIỮ LẠI (Không xóa) các khách hàng có mức chi tiêu (Monetary) cực kỳ lớn (Outliers) để quan sát khả năng chống nhiễu của K-Medoids.
> - Thực hiện chuẩn hóa 3 cột R, F, M bằng StandardScaler (vì thang đo của chúng hoàn toàn khác nhau).
>
> ### PHẦN 2: CÁCH 1 - THỰC CHIẾN VỚI SCIKIT-LEARN & SKLEARN-EXTRA
> Viết code huấn luyện 3 model:
> 1. KMeans từ sklearn.cluster
> 2. MiniBatchKMeans từ sklearn.cluster
> 3. KMedoids từ sklearn_extra.cluster (Sử dụng metric 'manhattan' hoặc 'euclidean').
> - Chạy fit_predict cho cả 3.
> - Tính Inertia (WCSS) hoặc Silhouette Score.
> - Vẽ 1 biểu đồ subplot (1x3) so sánh trực quan kết quả của 3 mô hình, in đậm vị trí tâm (Centroid/Medoid) bằng dấu X màu đỏ. (Phải thấy rõ tâm K-Means bị Outlier kéo lệch, còn K-Medoids thì không).
>
> ### PHẦN 3: CÁCH 2 - CODE FROM SCRATCH (CHỈ DÙNG NUMPY)
> Đây là phần quan trọng nhất. Hãy viết class (Object-Oriented) cho từng thuật toán mà KHÔNG import bất kỳ hàm clustering nào. Phải tự code các bước:
> 1. CustomKMeans:
> - Hàm _init_centroids (Khởi tạo ngẫu nhiên).
> - Hàm _assign_clusters (Tính Euclidean distance và gán nhãn).
> - Hàm _update_centroids (Tính trung bình - mean).
> - Hàm fit.
> 2. CustomMiniBatchKMeans:
> - Tương tự như trên nhưng trong vòng lặp phải trích xuất ngẫu nhiên 1 batch_size.
> - Cập nhật tâm theo cơ chế trung bình di động (learning rate hoặc cache tính toán trên batch).
> 3. CustomKMedoids: (Có thể dùng thuật toán PAM đơn giản hóa hoặc Alternating)
> - Hàm khởi tạo lấy ngẫu nhiên các điểm dữ liệu THỰC TẾ làm Medoid ban đầu.
> - Vòng lặp SWAP: Thử hoán đổi 1 medoid với 1 non-medoid, tính lại tổng chi phí tuyệt đối (Absolute cost). Nếu cost giảm thì giữ nguyên swap, nếu không thì revert.
>
> ### PHẦN 4: SO SÁNH VÀ ĐÁNH GIÁ (FROM SCRATCH)
> - Khởi tạo 3 class Custom vừa viết.
> - Chạy huấn luyện trên cùng tập dữ liệu.
> - Vẽ lại biểu đồ subplot (1x3) để chứng minh code From Scratch chạy ra kết quả tương đương (hoặc gần giống) với Scikit-learn.
>
> YÊU CẦU VỀ CODE:
> - Code phải sạch, có Type Hinting (typing), docstring mô tả nguyên lý toán học ngắn gọn ở mỗi class.
> - Đặt random_seed cố định để kết quả có thể tái lập (reproducible).
> - Ở các hàm tính khoảng cách trong Numpy (From scratch), hãy cố gắng tận dụng Vectorization (phép toán ma trận) thay vì dùng vòng lặp for lồng nhau để tối ưu tốc độ.
>
> Viết phần code thật chi tiết, không dùng placeholder như "Your code here".

## AI-01

| Trường | Nội dung |
|---|---|
| ID | AI-01 |
| Công cụ / ngày | Claude (claude.ai), 03/11/2026 |
| Mục đích | Rà soát thiết kế đánh giá của notebook (nhiều seed, chọn K, so sánh công bằng). |
| Đầu ra AI (tóm tắt) | Danh sách 12 điểm chưa ổn: Mini-Batch scratch chưa hội tụ; so sánh Mini-Batch sklearn và scratch không cùng điều kiện; chỉ chạy một seed và một K; hai bản K-Medoids không trùng medoid; chỉ số center drift tự đặt; chọn K = 4 thiếu căn cứ. |
| Sinh viên đã chỉnh sửa gì | Không dùng trực tiếp; dùng làm danh sách kiểm tra, các điểm được xử lý ở AI-03. |
| Cách kiểm chứng | Đối chiếu từng điểm với notebook. |
| Vị trí dùng trong bài | Thiết kế đánh giá ở báo cáo Mục 4. |

**Prompt (nguyên văn):**

> ủa vậy cái notebook này có những điểm nào chưa ổn vậy

## AI-02

| Trường | Nội dung |
|---|---|
| ID | AI-02 |
| Công cụ / ngày | Claude (claude.ai), 03/11/2026 |
| Mục đích | Rà soát lỗi tiền xử lý và phương pháp của notebook. |
| Đầu ra AI (tóm tắt) | Rà soát 43 ô theo ba mức độ. Lỗi nặng nhất: đơn hủy bị bỏ nhưng đơn mua gốc vẫn giữ nên Monetary bị thổi phồng (khách 12346 và 16446 hủy 100%); thiếu dòng import KMedoids; center drift thiên vị K-Medoids; chọn K không khớp Silhouette. |
| Sinh viên đã chỉnh sửa gì | Không dùng trực tiếp; các lỗi được sửa ở AI-03. |
| Cách kiểm chứng | Kiểm tra lại hai khách 12346 và 16446 trên dữ liệu gốc. |
| Vị trí dùng trong bài | Tiền xử lý RFM (Monetary ròng) và thí nghiệm bỏ outlier, báo cáo Mục 4. |

**Prompt (nguyên văn):**

> đây là 1 notebook về phân cụm hành vi khách hàng bằng thuật toán học máy k-means, hãy xem và đánh giá những chỗ chưa ổn trong notebook

## AI-03

| Trường | Nội dung |
|---|---|
| ID | AI-03 |
| Công cụ / ngày | Claude (claude.ai), 03/11/2026 |
| Mục đích | Sửa các lỗi đã nêu ở AI-01 và AI-02. |
| Đầu ra AI (tóm tắt) | Notebook sửa và chạy lại trên Online Retail (541.909 dòng): Monetary tính ròng sau khi trừ đơn hủy (còn 4.320 khách); thêm không gian X_log; thay center drift bằng thí nghiệm bỏ outlier đo ARI trên phần lõi; quét K có ghi chú K = 4 là lựa chọn nghiệp vụ; ghép tâm một-một; thêm k-means++ và Mini-Batch có điều kiện dừng giống sklearn; thêm silhouette_numpy và adjusted_rand_index. |
| Sinh viên đã chỉnh sửa gì | Chạy lại trên máy (scikit-learn 1.6.1), đối chiếu các số trong phần nhận xét với output; số liệu trong báo cáo lấy từ lần chạy lại này. |
| Cách kiểm chứng | Bốn kiểm tra ở báo cáo Mục 5.4: số liệu tái lập, tách module, chỉ số tự cài khớp sklearn (Silhouette 0,61646), lặp 5 seed. |
| Vị trí dùng trong bài | `src/rfm_scratch.py` (`_init_centroids`, `CustomMiniBatchKMeans._single_run`, `silhouette_numpy`, `adjusted_rand_index`, `match_centers`); `experiments/`; báo cáo Mục 4–7. |

**Prompt (nguyên văn):**

> oke vậy hãy sửa những lỗi m đã trình bày

