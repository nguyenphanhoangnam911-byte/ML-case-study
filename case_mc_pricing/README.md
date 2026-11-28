# Case study Monte Carlo Control: định giá động phòng khách sạn

Dùng Monte Carlo Control để học chính sách giá cho 10 phòng trong 30 ngày cuối trước ngày nhận phòng. Dữ liệu thật chỉ dùng để hiệu chỉnh bộ mô phỏng nhu cầu; agent học trên bộ mô phỏng. Chính sách tối ưu chính xác được tính bằng quy hoạch động để làm đáp án chuẩn.

Notebook: [`notebooks/Dynamic_Pricing_MC_Control.ipynb`](notebooks/Dynamic_Pricing_MC_Control.ipynb)

Mã tự cài đặt: [`src/mc_pricing.py`](src/mc_pricing.py). Bộ mô phỏng, quy hoạch động và các biến thể Monte Carlo Control (bản tự cài đặt, kèm bản batch scikit-learn).

## 1. Dữ liệu

- Bộ dữ liệu: *Hotel Booking Demand* (Antonio, de Almeida & Nunes, 2019), 119.390 dòng. Dùng City Hotel, phòng loại A.
- Nguồn tải: Kaggle, https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand (bản đăng lại của bộ dữ liệu gốc). Bài báo gốc: Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126.
- Đặt file `hotel_bookings.csv` **cùng thư mục với notebook** (`notebooks/hotel_bookings.csv`). File này bị `.gitignore` loại, không commit.

## 2. Môi trường

Python 3.13 (đã kiểm chứng).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Cách chạy

1. Mở notebook bằng Jupyter, chạy **Restart & Run All**.
2. Seed cố định (10 seed cho các thí nghiệm), 100.000 episode mỗi lần chạy.
3. Thời gian chạy: khoảng 20 phút cho toàn bộ notebook (đo lại ngày 28/11/2026: 1.219 giây trên máy 2 nhân CPU, Python 3.13). Phần lớn thời gian nằm ở các thí nghiệm 10 seed × 100.000 Episode.
4. Notebook ghi kết quả ra `pricing_results.npz` và các hình `figP*.png` trong thư mục chạy.
5. Hình từng bước của một Episode (`results/figP9_algorithm_steps.png`) vẽ bằng script riêng: `MC_DATA=/đường/dẫn/hotel_bookings.csv python make_figP9.py` (chạy trong thư mục `case_mc_pricing/`; mất vài giây).

## 4. Nội dung notebook

| Thí nghiệm | Câu hỏi |
|---|---|
| TN0 | NumPy và scikit-learn có cho cùng kết quả không? |
| TN1 | Định giá động hơn giá cố định bao nhiêu? |
| TN2 | So sánh các biến thể MC Control (ES, ε-greedy, GLIE, off-policy) |
| TN3 | Cái giá của việc học (regret) theo ε |
| TN4 | Học từ log giá lịch sử (off-policy) |
| TN5 | Cú sốc nhu cầu |
| TN7 | Vì sao Monte Carlo hội tụ chậm |
| TN6 | Độ bền với giả định |

## 5. Kết quả tái lập

Đã chạy lại với `hotel_bookings.csv`, không lỗi, số liệu khớp báo cáo: V*(30, 10) = 1.021,0 €; giá cố định tốt nhất 100 € (976,0 €); sai khác Q lớn nhất giữa NumPy và scikit-learn 1,15×10⁻⁷, chính sách giống hệt ở 1000/1000 lần cải thiện.

## 6. Lưu ý về "bản dùng thư viện"

scikit-learn không có thuật toán học tăng cường. Bản "scikit-learn" trong notebook chỉ thay **bước ước lượng Q** bằng hồi quy Ridge trên one-hot; môi trường, vòng lặp GPI và cải thiện chính sách vẫn là mã tự viết.

Vì vậy bản tự cài đặt bằng NumPy là bản chính. Bản scikit-learn đóng vai trò kiểm chứng: với one-hot của từng ô (s, a), hồi quy bình phương tối thiểu cho đúng trung bình Return theo ô, nên nếu hai bản cho cùng Q thì bước ước lượng Q của bản NumPy là đúng. Kết quả TN0 xác nhận điều này (sai khác Q lớn nhất 1,15×10⁻⁷ €, chính sách giống nhau 1000/1000 lần). Nhóm không dùng thư viện học tăng cường chuyên dụng vì các thư viện phổ biến chỉ cung cấp môi trường (Gymnasium) hoặc thuật toán học sâu (Stable-Baselines3), không có sẵn Monte Carlo Control dạng bảng, còn môi trường định giá ở đây là bộ mô phỏng do nhóm tự xây.

## 7. Tài liệu kiểm chứng

- [`docs/theory_to_code_mapping.md`](docs/theory_to_code_mapping.md): bảng ánh xạ công thức → mã nguồn → bằng chứng → AI log ID.
- [`docs/ai_log.md`](docs/ai_log.md): nhật ký sử dụng AI.

## 8. Mã nguồn tham khảo

Không có. Nhóm không sao chép hay chỉnh sửa mã nguồn từ nguồn bên ngoài. Mã nguồn được viết với sự hỗ trợ của AI (xem [`docs/ai_log.md`](docs/ai_log.md)) và chỉ gọi các thư viện công khai liệt kê trong `requirements.txt`.
