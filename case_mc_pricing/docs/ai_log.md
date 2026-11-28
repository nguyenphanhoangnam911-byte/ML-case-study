# Nhật ký sử dụng AI (case Monte Carlo Control)

Mỗi lần AI đóng góp nội dung hoặc code được dùng trong bài có một mục log. ID dùng chung với bảng "AI prompting log" trong báo cáo (AI-00 đến AI-05 cho cả hai case); AI-04 và AI-05 thuộc case này. Cột AI log ID trong [`theory_to_code_mapping.md`](theory_to_code_mapping.md) trỏ tới các ID này. Sinh viên giải thích được mọi dòng code và công thức trong bài.

## AI-04

| Trường | Nội dung |
|---|---|
| ID | AI-04 |
| Công cụ / ngày | Claude (claude.ai), trước 13/11/2026 (ngày không hiển thị trong ảnh chụp lịch sử chat) |
| Mục đích | Làm bản đầu của case study định giá phòng khách sạn trên `hotel_bookings.csv` kèm báo cáo. |
| Đầu ra AI (tóm tắt) | Bản đầu của case study định giá phòng khách sạn: bộ mô phỏng hiệu chỉnh từ dữ liệu, quy hoạch động, các biến thể Monte Carlo Control. Bản này còn chứa Q-learning và hai case study Blackjack, FrozenLake. |
| Sinh viên đã chỉnh sửa gì | Bỏ Q-learning, Blackjack, FrozenLake theo AI-05; giữ bộ mô phỏng, quy hoạch động và các biến thể Monte Carlo. |
| Cách kiểm chứng | Chạy lại notebook; đối chiếu V*(30, 10) = 1.021,0 € với quy hoạch động. |
| Vị trí dùng trong bài | `src/mc_pricing.py` (bộ mô phỏng, `PricingMDP`, các lớp MC Control); `notebooks/Dynamic_Pricing_MC_Control.ipynb`; báo cáo Mục 4–7. |

**Prompt (nguyên văn):**

> đây là bộ dữ liệu, hãy tiến hành làm case study này và viết báo cáo

## AI-05

| Trường | Nội dung |
|---|---|
| ID | AI-05 |
| Công cụ / ngày | Claude (claude.ai), 13/11/2026 |
| Mục đích | Rà soát và thu gọn case study về đúng Monte Carlo Control, làm theo hai hướng có và không dùng scikit-learn. |
| Đầu ra AI (tóm tắt) | Bỏ Q-learning và các thuật toán ra đời sau Monte Carlo; bỏ Blackjack và FrozenLake; giữ case study khách sạn làm chủ đạo; thêm hai hướng cài đặt bước ước lượng Q (NumPy batch và scikit-learn) cùng thí nghiệm TN0. |
| Sinh viên đã chỉnh sửa gì | Chạy lại notebook và đối chiếu số liệu với báo cáo. |
| Cách kiểm chứng | TN0: sai khác Q lớn nhất giữa NumPy và scikit-learn 1,15×10⁻⁷ €, chính sách giống 1000/1000; chạy lại toàn bộ notebook ngày 28/11/2026 không lỗi. |
| Vị trí dùng trong bài | `src/mc_pricing.py` (`BatchMC`, `estimate_Q`), notebook TN0–TN7; báo cáo Mục 4–7. |

**Prompt (nguyên văn):**

> đây là bài thuyết trình về thuật toán monte carlo thôi, q-learning là thuật toán ra đời sau nên tuyệt đối không được phép đưa vào bài thuyết trình lần này, rà soát lại những chỗ sai sót và dư thừa về Q-learning và những thuật toán ra đời sau và tiên tiến hơn monte carlo, và cả trong case study luôn, làm thuật toán này bằng 2 hướng sckit-learn và ko dùng sckit-learn, bỏ đi những phần liên quan đến 2 case study black jack và frozen lake vì không cần thiết nữa, chủ đạo sẽ là case study về hotel mình đang làm

