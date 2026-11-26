# Bảng ánh xạ lý thuyết → mã nguồn (case Monte Carlo Control)

Số dòng theo `case_mc_pricing/src/mc_pricing.py` (bản nộp). Sau mỗi lần sửa code, phải cập nhật lại bảng. Nghiệm quy hoạch động (V*, Q*, π*) là đáp án chuẩn độc lập. Cột AI log ID điền theo `ai_log.md`.

| Công thức / bước | Biến thể | Hàm / lớp | Dòng | Bằng chứng kiểm chứng | AI log ID |
|---|---|---|---|---|---|
| P(mua) = P(WTP ≥ giá), WTP ~ LogNormal | Mô hình nhu cầu | `Scenario.prob_buy` | 62–63 | Bảng P(mua) theo giá; đường cong đặt phòng mô phỏng khớp dữ liệu (Hình M6.1) | |
| Cường độ khách mua λ(t, p) = Σ λts·Ps(mua \| p) | Mô hình nhu cầu | `Scenario.rate` | 65–67 | Như trên | |
| V_t(x) = max_p Σ_d P(D = d)·[p·min(d, x) + V_{t−1}(x − min(d, x))] | DP lùi (đáp án chuẩn) | `PricingMDP.solve` | 90–97 | V*(30, 10) = 1.021,0 € (đã chạy lại) | |
| Doanh thu kỳ vọng chính xác của một chính sách | Đánh giá | `PricingMDP.evaluate` | 99–108 | Giá cố định 100 €: 976,0 € | |
| Giá cố định tốt nhất | Đánh giá | `PricingMDP.best_fixed` | 123–125 | 976,0 € tại 100 € | |
| Lấy mẫu chuyển trạng thái D ~ Poisson, Episode | Mọi biến thể | `FastPricingSim.step`, `rollout` | 138–151 | Doanh thu chính sách học được chấm bằng DP | |
| Chọn hành động ε-greedy | On-policy, GLIE | `Learner.eps_act` | 171–174 | Như các dòng biến thể | |
| Cải thiện chính sách greedy (phá hòa ngẫu nhiên) | Mọi biến thể gia tăng | `Learner.refresh` | 166–169 | Như các dòng biến thể | |
| Q ← Q + (G − Q)/N; Q ← Q + α(G − Q); G cộng dồn ngược, γ = 1 | On-policy, ES, GLIE | `update_mc` | 205–215 | Khớp π* theo xác suất ghé thăm: 40,6% / 36,2% / 33,5% | |
| On-policy MC ε-greedy | On-policy | `OnPolicyMC` | 177–188 | 97,36% ± 0,25 so với tối ưu sau 100.000 Episode | |
| Exploring Starts: (t, x, a) khởi đầu ngẫu nhiên | ES | `MCExploringStarts` | 191–202 | 96,37% ± 0,58 | |
| GLIE: ε = 1000/(1000 + k) | GLIE | `OnPolicyMC` với `eps_schedule` | 177–188 | 95,75% ± 0,44 | |
| Weighted IS: C ← C + W; Q ← Q + (W/C)(G − Q); W ← W/b | Off-policy | `OffPolicyMC.learn_from` | 230–240 | 93,54% ± 0,63; 13,7% số bước dùng để học | |
| Hành vi ε_b-greedy và xác suất b(a\|s) | Off-policy | `OffPolicyMC.behavior`, `run_episode` | 225–228, 242–249 | Như dòng trên | |
| Ước lượng Q = trung bình Return theo ô (np.bincount) | Batch NumPy | `BatchMC.estimate_Q` | 305–314 | Q trùng bản scikit-learn: sai khác lớn nhất 1,15×10⁻⁷ € | |
| Ước lượng Q = hồi quy Return lên one-hot (Ridge) | Batch scikit-learn | `BatchMC.estimate_Q`, `__init__` | 278–292, 305–314 | Chính sách greedy giống hệt ở 1000/1000 lần cải thiện | |
| Cải thiện chính sách batch: argmax Q (làm tròn 4 chữ số) | Batch | `BatchMC.evaluate_and_improve` | 316–322 | Như dòng trên | |
| Đánh giá chính sách so với V* và π* | Mọi biến thể | `snapshot` | 252–257 | Dùng cho mọi con số ở 7.1 | |

