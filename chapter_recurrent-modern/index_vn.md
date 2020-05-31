<!--
# Modern Recurrent Neural Networks
-->

# Mạng Nơ-ron Truy hồi Hiện đại
:label:`chap_modern_rnn`

<!--
Although we have learned the basics of recurrent neural networks, they are not sufficient for a practitioner to solve today's sequence learning problems.
For instance, given the numerical unstability during gradient calculation, gated recurrent neural networks are much more common in practice.
We will begin by introducing two of such widely-used networks, namely gated recurrent units (GRUs) and long short term memory (LSTM),
with illustrations using the same language modeling problem as introduced in :numref:`chap_rnn`.
-->

Dù chúng ta đã hiểu cơ bản về mạng nơ-ron truy hồi, chúng vẫn chưa đủ để giải quyết các vấn đề trong học chuỗi hiện nay.
Ví dụ, RNN có hiện tượng không ổn định giá trị số khi tính gradient, về điểm này thì các mạng nơ-ron truy hồi theo cổng được sử dụng phổ biến hơn.
Chúng ta bắt đầu chương này bằng việc giới thiệu hai cấu trúc mạng như vậy: nút truy hồi theo cổng (*gated recurrent unit - GRU*) và bộ nhớ ngắn hạn dài (*long short term memory - LSTM*), minh họa bằng cách sử dụng chúng để giải quyết bài toán mô hình hóa ngôn ngữ trong :numref:`chap_rnn`.

<!--
Furthermore, we will modify recurrent neural networks with a single undirectional hidden layer.
We will describe deep architectures, and discuss the bidirectional design with both forward and backward recursion.
They are frequently adopted in modern recurrent networks.
-->

Xa hơn, chúng ta sẽ thay đổi mạng nơ-ron truy hồi với một tầng ẩn vô hướng đơn.
Ta cũng sẽ mô ta các kiến trúc mạng sâu, và thảo luận thiết kế hai chiều (*bidirectional*) với việc truy hồi xuôi và ngược.
Chúng thường xuyên được sử dụng trong các mạng nơ-ron truy hồi hiện đại.


<!--
In fact, a large portion of sequence learning problems such as automatic speech recognition, 
text to speech, and machine translation, consider both inputs and outputs to be sequences of arbitrary length.
Finally, we will take machine translation as an example, and introduce the encoder-decoder architecture based on
recurrent neural networks and modern practices for such sequence to sequence learning problems.
-->

Trên thực tế, phần lớn các vấn đề trong học chuỗi như nhận dạng giọng nói tự động, chuyển đổi văn bản thành giọng nói và dịch máy, có đầu vào và đầu ra đều là chuỗi chiều dài bất kì.
Cuối cùng, ta sẽ lấy ví dụ về dịch máy, giới thiệu kiến trúc bộ biến đổi - giải biến đổi (*encoder-decoder*) dựa trên mạng nơ-ron truy hồi, và các phương pháp hiện đại để giải quyết các vấn đề học chuỗi trên.

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
