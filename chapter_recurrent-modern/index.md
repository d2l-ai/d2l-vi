# Mạng thần kinh tái phát hiện đại
:label:`chap_modern_rnn`

Chúng tôi đã giới thiệu những điều cơ bản về RNNs, có thể xử lý dữ liệu trình tự tốt hơn. Để trình diễn, chúng tôi đã triển khai các mô hình ngôn ngữ dựa trên RNN trên dữ liệu văn bản. Tuy nhiên, những kỹ thuật như vậy có thể không đủ cho các học viên khi họ phải đối mặt với một loạt các vấn đề học tập chuỗi ngày nay. 

Ví dụ, một vấn đề đáng chú ý trong thực tế là sự bất ổn số của RNNs. Mặc dù chúng tôi đã áp dụng các thủ thuật triển khai như cắt gradient, vấn đề này có thể được giảm bớt hơn nữa với các thiết kế phức tạp hơn của các mô hình trình tự. Cụ thể, RNN có cổng phổ biến hơn nhiều trong thực tế. Chúng tôi sẽ bắt đầu bằng cách giới thiệu hai trong số các mạng được sử dụng rộng rãi như vậy, cụ thể là các đơn vị định kỳ *ged* (Grus) và * bộ nhớ ngắn hạn dài* (LSTM). Hơn nữa, chúng tôi sẽ mở rộng kiến trúc RNN với một layer ẩn một chiều duy nhất đã được thảo luận cho đến nay. Chúng tôi sẽ mô tả các kiến trúc sâu với nhiều lớp ẩn, và thảo luận về thiết kế hai chiều với cả tính toán tái phát về phía trước và ngược. Những mở rộng như vậy thường được áp dụng trong các mạng tái phát hiện đại. Khi giải thích các biến thể RNN này, chúng tôi tiếp tục xem xét vấn đề mô hình hóa ngôn ngữ tương tự được giới thiệu trong :numref:`chap_rnn`. 

Trên thực tế, mô hình ngôn ngữ chỉ tiết lộ một phần nhỏ trong số những gì trình tự học có khả năng. Trong một loạt các vấn đề học tập trình tự, chẳng hạn như nhận dạng giọng nói tự động, văn bản sang giọng nói và dịch máy, cả đầu vào và đầu ra đều là các chuỗi có độ dài tùy ý. Để giải thích làm thế nào để phù hợp với loại dữ liệu này, chúng tôi sẽ lấy dịch máy làm ví dụ, và giới thiệu kiến trúc bộ mã hóa-giải mã dựa trên RNNs và tìm kiếm chùm tia để tạo chuỗi.

```toc
:maxdepth: 2

gru
lstm
deep-rnn
bi-rnn
machine-translation-and-dataset
encoder-decoder
seq2seq
beam-search
```
