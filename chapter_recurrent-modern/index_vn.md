<!--
# Modern Recurrent Neural Networks
-->

# Mạng Nơ-ron Hồi tiếp Hiện đại
:label:`chap_modern_rnn`

<!--
Although we have learned the basics of recurrent neural networks, they are not sufficient for a practitioner to solve today's sequence learning problems.
For instance, given the numerical unstability during gradient calculation, gated recurrent neural networks are much more common in practice.
We will begin by introducing two of such widely-used networks, namely gated recurrent units (GRUs) and long short term memory (LSTM),
with illustrations using the same language modeling problem as introduced in :numref:`chap_rnn`.
-->

Mặc dù đã biết về các kiến thức cơ bản của mạng nơ-ron hồi tiếp, chúng vẫn chưa đủ để ta giải quyết các bài toán học chuỗi hiện nay.
Ví dụ như RNN có hiện tượng bất ổn số học khi tính gradient, do đó các mạng nơ-ron hồi tiếp có cổng được sử dụng phổ biến hơn nhiều trong thực tiễn.
Chúng ta bắt đầu chương này bằng việc giới thiệu hai cấu trúc mạng phổ biến: nút hồi tiếp có cổng (*gated recurrent unit - GRU*) và bộ nhớ ngắn hạn dài (*long short term memory - LSTM*).
Chúng cũng sẽ được áp dụng minh họa trong cùng bài toán mô hình hóa ngôn ngữ đã được giới thiệu ở :numref:`chap_rnn`.

<!--
Furthermore, we will modify recurrent neural networks with a single undirectional hidden layer.
We will describe deep architectures, and discuss the bidirectional design with both forward and backward recursion.
They are frequently adopted in modern recurrent networks.
-->

Hơn nữa, chúng ta sẽ sửa đổi mạng nơ-ron hồi tiếp với một tầng ẩn đơn chiều.
Ta cũng sẽ mô tả các kiến trúc mạng sâu và thảo luận về thiết kế hai chiều (*bidirectional*) gồm cả hồi tiếp xuôi và ngược.
Chúng thường xuyên được sử dụng trong các mạng nơ-ron hồi tiếp hiện đại.


<!--
In fact, a large portion of sequence learning problems such as automatic speech recognition, 
text to speech, and machine translation, consider both inputs and outputs to be sequences of arbitrary length.
Finally, we will take machine translation as an example, and introduce the encoder-decoder architecture based on
recurrent neural networks and modern practices for such sequence to sequence learning problems.
-->

Trên thực tế, phần lớn các bài toán học chuỗi như nhận dạng giọng nói tự động, chuyển đổi văn bản thành giọng nói và dịch máy, đều có đầu vào và đầu ra là các chuỗi với chiều dài bất kì.
Cuối cùng, ta sẽ lấy bài toán dịch máy làm ví dụ để giới thiệu kiến trúc mã hóa - giải mã (*encoder-decoder*) dựa trên mạng nơ-ron hồi tiếp cùng các kỹ thuật hiện đại để giải quyết bài toán học từ chuỗi sang chuỗi.

```toc
:maxdepth: 2

gru_vn
lstm_vn
deep-rnn_vn
bi-rnn_vn
machine-translation-and-dataset_vn
encoder-decoder_vn
seq2seq_vn
beam-search_vn
```

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Văn Cường
* Phạm Minh Đức
* Phạm Hồng Vinh
* Lê Khắc Hồng Phúc
