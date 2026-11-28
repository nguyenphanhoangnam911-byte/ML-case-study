# Tiểu luận Machine Learning: mã nguồn cho hai case study

Repo này chứa mã nguồn, kết quả chạy và tài liệu kiểm chứng của hai case study trong báo cáo (Mục 4–7).

- Repo: https://github.com/nguyenphanhoangnam911-byte/ML-case-study
 
| Case | Thuật toán | Thư mục |
|---|---|---|
| Phân cụm hành vi khách hàng bằng mô hình RFM | K-Means, Mini-Batch K-Means, K-Medoids (PAM) | [`case_rfm_clustering/`](case_rfm_clustering/) |
| Định giá động phòng khách sạn | Monte Carlo Control (Exploring Starts, on-policy ε-greedy, GLIE, off-policy) | [`case_mc_pricing/`](case_mc_pricing/) |

Mỗi case có hai cách cài đặt: dùng thư viện, và tự cài đặt từ đầu. Hướng dẫn chạy nằm trong README của từng case.

## Cấu trúc

```
├── README.md
├── case_rfm_clustering/
│   ├── README.md
│   ├── requirements.txt
│   ├── notebooks/
│   ├── src/             # bản tự cài đặt (NumPy)
│   ├── experiments/     # script so sánh A, B nhiều seed (chỉ case RFM)
│   ├── results/
│   └── docs/            # bảng ánh xạ công thức → code, nhật ký AI
└── case_mc_pricing/
    └── (cấu trúc tương tự)
```

## Dữ liệu

Repo **không** chứa file dữ liệu. Nguồn và cách tải nằm trong README của từng case.

## Tài liệu tham khảo

Các nguồn dưới đây được trích trong báo cáo hoặc dùng trực tiếp trong mã nguồn (APA 7).

**Dữ liệu**
- Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief, 22*, 41–49. https://doi.org/10.1016/j.dib.2018.11.126
- Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33

**Phân cụm (case RFM)**
- Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. In *Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms* (pp. 1027–1035).
- Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. *Journal of Database Marketing & Customer Strategy Management, 19*(3), 197–208. https://doi.org/10.1057/dbm.2012.17
- Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of Classification, 2*(1), 193–218. https://doi.org/10.1007/BF01908075
- Kaufman, L., & Rousseeuw, P. J. (1990). *Finding groups in data: An introduction to cluster analysis*. Wiley. https://doi.org/10.1002/9780470316801
- Lloyd, S. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory, 28*(2), 129–137. https://doi.org/10.1109/TIT.1982.1056489
- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability* (Vol. 1, pp. 281–297). University of California Press.
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics, 20*, 53–65. https://doi.org/10.1016/0377-0427(87)90125-7
- Sculley, D. (2010). Web-scale k-means clustering. In *Proceedings of the 19th International Conference on World Wide Web* (pp. 1177–1178). https://doi.org/10.1145/1772690.1772862

**Học tăng cường (case Monte Carlo)**
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.

**Thư viện**
- Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature, 585*, 357–362. https://doi.org/10.1038/s41586-020-2649-2
- Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

**Mã nguồn tham khảo:** không có. Nhóm không sao chép hay chỉnh sửa mã nguồn từ nguồn bên ngoài; phần AI hỗ trợ được ghi trong `docs/ai_log.md` của từng case.
