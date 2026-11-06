# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603634
- **Họ và tên:** Lục Anh Tuấn
- **SEED:** 5283

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính


| Phương án | F1 test đã lưu | F1 SUB 40 đã lưu | p95 (ms/tin) | USD/1.000 tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg | 0,8949 | 0,9243 | 0,098 | 0 |
| 2. BoW + MLP | 0,9449 | 0,9243 | 0,502 | 0 |
| 3a. MiniLM + kNN | 0,9716 | 0,9500 | 54,812 | 0 |
| 3b. MiniLM + LogReg | 0,9614 | 0,9623 | 34,249 | 0 |
| 3c. MiniLM + MLP head | 0,9864 | 1,0000 | 35,242 | 0 |
| 4a. LLM nhỏ zero-shot | 0,9873* | 0,9873 | 1.379,868 | 0,009977 |
| 4b. LLM lớn zero-shot | 1,0000* | 1,0000 | 1.108,142 | 0,099767 |
| 4c. LLM nhỏ few-shot | 0,9873* | 0,9873 | 978,350 | 0,021977 |

## 3. Dự đoán và thực tế

1. predict("dl_curve", "val loss sẽ hội tụ, mlp sẽ hơn logistic regression"), thực tế val loss không hội tụ và phương pháp deep learning không chiếm ưu thế lớn trong bài toán này

2. predict("llm_models", "model terra có f1 cao hơn nhưng không đáng kể khoảng 2,5%, model lớn đắt gấp 3 lần, llm luôn thằng mô hình 3b"), thực tế terra đất gấp 10 lần so với luna và f1 chỉ cao hơn luna khoảng 1,3%

3. predict("tradeoff", "ml,dl,encoder nằm trên pareto, llm nằm bên phải vạch sla 50 ms"), thực tế vẫn có trường hợp encoder + knn bị vượt yêu cầu với p95 > 50ms 1 xíu

## 4. Phân tích 50 case khó

| Phương án | F1 SUB đã lưu | F1 case khó đã lưu | Accuracy 50 case khó |
| --- | ---: | ---: | ---: |
| BoW + LogReg | 0,924 | 0,256 | 50% |
| BoW + MLP | 0,924 | 0,438 | 60% |
| MiniLM + kNN | 0,950 | 0,685 | 68% |
| MiniLM + LogReg | 0,962 | 0,649 | 68% |
| MiniLM + MLP head | 1,000 | 0,649 | 68% |
| LLM nhỏ zero-shot | 0,987 | 0,990 | 98% |
| LLM nhỏ + prompt phòng thủ | 0,987** | 1,000 | 100% |


BoW + LogReg tụt nhiều nhất: F1 đã lưu giảm khoảng 0,668; accuracy giảm từ 97,81% trên test thường xuống 50% trên bộ khó. Nhóm khó nhất là `natural_spam`: LogReg đúng 0/10, MLP đúng 1/10; tiếp theo là nhóm né từ khóa, cả hai chỉ đúng 1/6. LogReg cũng chỉ đúng 1/8 tin thao túng/lừa đảo. Với embedding, nhóm spam tự nhiên vẫn khó: kNN đúng 40%, hai head còn lại đúng 30%.

Case #11, nhãn spam: “Hey it's Jess from the gym … 3 months for the price of 1, just reply JOIN …”. Đây là quảng cáo khuyến mãi viết bằng giọng thân mật. BoW + LogReg đoán ham; điều này được xác định từ việc mô hình sai cả 10/10 case `natural_spam`. BoW chỉ cộng trọng số các token, không biểu diễn đầy đủ ý định quảng cáo và bối cảnh người gửi, nên giọng hội thoại có thể che mất tín hiệu spam.
Case #18, nhãn spam: “Mate you have to see this, I doubled my savings in 2 weeks with this app, use my code and we both get a bonus”. BoW + LogReg cũng đoán ham. Tin dùng lời kể của người quen, hứa lợi nhuận và mã giới thiệu thay cho mẫu “WIN/FREE/prize” quen thuộc. Mô hình khó kết hợp các chi tiết đó thành dấu hiệu quảng cáo đầu tư đáng ngờ.

## 5. Thử thách cá nhân

Đề 4 chạy `knn_predict` với `k = 1, 5, 25`, báo F1 và giải thích vì sao k quá nhỏ hoặc quá lớn đều có hại. Với SEED 5283, `5283 % 6 = 3`, ứng với đề thứ 4.

| k | F1 test |
| ---: | ---: |
| 1 | 0,8910 |
| 5 | 0,9038 |
| 25 | 0,8986 |

trong 3 giá trị k = 5 cho kết quả tốt nhất, khi k quá nhỏ thì mô hình dễ ảnh hưởng bởi nhiễu, khi k quá to thì do dataset có ham quá lớn có thể làm giảm khả năng nhận diện spam

## 6. Phương án cho sản phẩm thật

chọn MiniLM + MLP head làm phương án chính để tiếp tục đánh giá triển khai. Mô hình có p95 35,242 ms/tin, thấp hơn ngưỡng 50 ms; precision test 0,9694, recall 0,9794, F1 chuẩn tính lại 0,9744 và F1 SUB đã lưu 1,0000.

## 7. AI log và điều đã học

| Phần | Đã hỏi AI gì | AI trả lời/hỗ trợ gì | Đã sửa hoặc kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 1–3 | Hỏi công thức TP, TN, FP, FN và cú pháp NumPy, PyTorch, embedding | Giải thích công thức và đưa ví dụ cú pháp | Theo log trong notebook, tôi tự viết theo công thức, không hỏi đáp án TODO có sẵn; các kiểm tra cơ bản đã đạt, nhưng chưa phát hiện lỗi F1 trong `metrics()` ở thời điểm chạy lab. |
| Phần 4 | Nhờ hướng dẫn viết các hàm TODO | Hỗ trợ viết code gọi API và xử lý kết quả | Theo log, tôi gặp tình trạng `invalid = 40` khi giới hạn 3 token, đã tăng giới hạn và chạy lại; output zero-shot và few-shot cuối cùng đều ghi `invalid = 0`. |
| Tính tay | Hỏi công thức accuracy, sigmoid, F1 và cosine | Giải thích công thức | Tôi thay số, làm tròn, đối chiếu kết quả và ghi các bước trên giấy; ảnh `hand_calc.jpg` có đủ 5 phép tính. |

Qua bài này, accuracy cao trên dữ liệu lệch lớp chưa đủ chứng minh bộ lọc bắt spam tốt, ngoài ra không phải cứ dùng llm sẽ là phương án oke nhất, vẫn nên xem xét dùng cái nào trong từng 1 bài toán cụ theer
