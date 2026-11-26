# Thí nghiệm nhiều seed (so sánh A và B, Mục 7.1)

- `exp_A.py`: so sánh A, bản tự cài đặt so với scikit-learn, seed 0–4 → `results/resA.json`.
- `exp_B.py`: so sánh B, bỏ 62 outlier (|z| > 3), seed 0–4 → `results/resB.json`.
- `exp_common.py`, `rfm_prep.py`: nạp `data.csv` và dựng ma trận X, X_log.

Chạy (môi trường theo `requirements.txt`, mỗi script khoảng 5–6 phút):

```bash
export RFM_DATA=/đường/dẫn/data.csv   # mặc định: notebooks/data.csv
python experiments/exp_A.py
python experiments/exp_B.py
```

`results/` chứa kết quả đã chạy trên môi trường ghim (NumPy 2.1.3, scikit-learn 1.6.1, scikit-learn-extra 0.3.0).
