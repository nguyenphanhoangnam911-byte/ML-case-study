# Case study RFM: K-Means, Mini-Batch K-Means, K-Medoids

Phân cụm khách hàng từ dữ liệu giao dịch bằng mô hình Recency – Frequency – Monetary, so sánh ba thuật toán. Mỗi thuật toán có hai bản: scikit-learn (và `scikit-learn-extra` cho K-Medoids) và bản tự cài đặt chỉ dùng NumPy.

Notebook: [`notebooks/RFM_Customer_Clustering.ipynb`](notebooks/RFM_Customer_Clustering.ipynb)

Mã tự cài đặt: [`src/rfm_scratch.py`](src/rfm_scratch.py). Bản tự cài đặt K-Means, Mini-Batch K-Means, K-Medoids và các chỉ số (chỉ dùng NumPy). Script nhiều seed cho Mục 7.1: [`experiments/`](experiments/).

## 1. Dữ liệu

- Bộ dữ liệu: *Online Retail* (UCI Machine Learning Repository), 541.909 dòng × 8 cột, 01/12/2010 – 09/12/2011.
- Nguồn tải: UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/352/online+retail. Trích dẫn: Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33. Giấy phép: CC BY 4.0.
- Đặt file `data.csv` **cùng thư mục với notebook** (`notebooks/data.csv`). Notebook đọc với `encoding="latin1"`. File này bị `.gitignore` loại, không commit.

## 2. Môi trường

Python 3.13 (đã kiểm chứng). Cài thư viện đúng phiên bản:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Lưu ý quan trọng:

- Số liệu trong báo cáo được tái tạo với **scikit-learn 1.6.1**. Với scikit-learn 1.9.1, Mini-Batch K-Means của sklearn cho kết quả khác (xem mục 5). Vì vậy cần cài đúng `requirements.txt`.
- `scikit-learn-extra` đã ngừng bảo trì. Gói build được ở Python 3.13 trong lần kiểm chứng; với phiên bản Python/NumPy khác có thể không build được.
- `jinja2` là bắt buộc vì notebook dùng `DataFrame.style`.

## 3. Cách chạy

1. Mở notebook bằng Jupyter, chạy **Restart & Run All**.
2. Seed cố định: `SEED = 42`, số cụm `K = 4`.
3. Thời gian chạy: khoảng 2–3 phút cho toàn bộ notebook (đo lại ngày 28/11/2026: 137 giây trên máy 2 nhân CPU, Python 3.13, scikit-learn 1.6.1). K-Medoids chiếm phần lớn thời gian (khoảng 23–30 giây mỗi lần huấn luyện).

## 4. Nội dung notebook

| Phần | Nội dung |
|---|---|
| 1 | Tải dữ liệu, làm sạch, tổng hợp RFM, chuẩn hóa (`X` và `X_log`) |
| 2 | Cách 1: scikit-learn, thí nghiệm bỏ outlier |
| 3 | Cách 2: tự cài đặt bằng NumPy (`CustomKMeans`, `CustomMiniBatchKMeans`, `CustomKMedoids`) |
| 4 | So sánh bản scratch với scikit-learn, chân dung cụm, kết luận |

## 5. Kết quả tái lập và lưu ý

- Phần tất định (K-Means, K-Medoids, bản scratch) cho cùng kết quả ở mọi phiên bản đã thử.
- Mini-Batch K-Means của sklearn **phụ thuộc phiên bản thư viện**: với scikit-learn 1.6.1 WCSS = 6.413,1; với 1.9.1 WCSS = 6.349,7.
- Thời gian chạy phụ thuộc máy, chỉ mang tính tham khảo.

## 6. Tài liệu kiểm chứng

- [`docs/theory_to_code_mapping.md`](docs/theory_to_code_mapping.md): bảng ánh xạ công thức → mã nguồn → bằng chứng → AI log ID.
- [`docs/ai_log.md`](docs/ai_log.md): nhật ký sử dụng AI.

## 7. Mã nguồn tham khảo

Không có. Nhóm không sao chép hay chỉnh sửa mã nguồn từ nguồn bên ngoài. Mã nguồn được viết với sự hỗ trợ của AI (xem [`docs/ai_log.md`](docs/ai_log.md)) và chỉ gọi các thư viện công khai liệt kê trong `requirements.txt`.
