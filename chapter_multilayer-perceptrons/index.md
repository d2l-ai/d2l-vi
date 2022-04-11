# Multilayer Perceptrons
:label:`chap_perceptrons`

Trong chương này, chúng tôi sẽ giới thiệu mạng thực sự * sâu * đầu tiên của bạn. Các mạng sâu đơn giản nhất được gọi là nhận thức đa lớp, và chúng bao gồm nhiều lớp tế bào thần kinh mỗi lớp được kết nối hoàn toàn với những người trong lớp bên dưới (từ đó chúng nhận được đầu vào) và những người ở trên (đến lượt nó, ảnh hưởng). Khi chúng tôi đào tạo các mô hình công suất cao, chúng tôi có nguy cơ vượt quá. Vì vậy, chúng tôi sẽ cần phải cung cấp giới thiệu nghiêm ngặt đầu tiên của bạn về các khái niệm về overfitting, underfitting, và lựa chọn mô hình. Để giúp bạn chống lại những vấn đề này, chúng tôi sẽ giới thiệu các kỹ thuật chính quy hóa như phân rã cân và bỏ học. Chúng tôi cũng sẽ thảo luận về các vấn đề liên quan đến tính ổn định số và khởi tạo tham số là chìa khóa để đào tạo thành công các mạng sâu. Trong suốt, chúng tôi mong muốn cung cấp cho bạn một nắm bắt vững chắc không chỉ về các khái niệm mà còn về việc thực hành sử dụng các mạng sâu. Vào cuối chương này, chúng tôi áp dụng những gì chúng tôi đã giới thiệu cho đến nay cho một trường hợp thực sự: dự đoán giá nhà. Chúng tôi đánh giá các vấn đề liên quan đến hiệu suất tính toán, khả năng mở rộng và hiệu quả của các mô hình của chúng tôi cho các chương tiếp theo.

```toc
:maxdepth: 2

mlp
mlp-scratch
mlp-concise
underfit-overfit
weight-decay
dropout
backprop
numerical-stability-and-init
environment
kaggle-house-price
```
