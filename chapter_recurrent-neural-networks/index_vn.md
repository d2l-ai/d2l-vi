<!--
# Recurrent Neural Networks
-->

# Mạng Nơ-ron Hồi tiếp
:label:`chap_rnn`

<!--
So far we encountered two types of data: generic vectors and images.
For the latter we designed specialized layers to take advantage of the regularity properties in them.
In other words, if we were to permute the pixels in an image, it would be much more difficult to reason about 
its content of something that would look much like the background of a test pattern in the times of analog TV.
-->

Cho đến nay, chúng ta đã gặp hai loại dữ liệu: các vector tổng quát và hình ảnh.
Với dữ liệu hình ảnh, ta đã thiết kế các tầng chuyên biệt nhằm tận dụng tính chính quy (*regularity property*) của hình ảnh.
Nói cách khác, nếu ta hoán vị các điểm ảnh trong một ảnh, ta sẽ thu được một bức ảnh trông giống như các khuôn mẫu kiểm tra (*test pattern*) hay thấy trong truyền hình analog, và rất khó để suy luận về nội dung của chúng.

<!--
Most importantly, so far we tacitly assumed that our data is generated i.i.d., i.e., independently and identically distributed, all drawn from some distribution.
Unfortunately, this is not true for most data.
For instance, the words in this paragraph are written in sequence, and it would be quite difficult to decipher its meaning if they were permuted randomly.
Likewise, image frames in a video, the audio signal in a conversation, or the browsing behavior on a website, all follow sequential order.
It is thus only reasonable to assume that specialized models for such data will do better at describing it and at solving estimation problems.
-->

Quan trọng hơn là cho đến thời điểm này, chúng ta đã ngầm định rằng dữ liệu được sinh ra từ những phân phối độc lập và giống hệt nhau (*independently and identically distributed - i.i.d.*).
Thật không may, điều này lại không đúng với hầu hết các loại dữ liệu.
Ví dụ, các từ trong đoạn văn này được viết theo một trình tự nhất định mà nếu bị hoán vị đi một cách ngẫu nhiên thì sẽ rất khó để giải mã ý nghĩa của chúng.
Tương tự với các khung hình trong video, tín hiệu âm thanh trong một cuộc hội thoại hoặc hành vi duyệt web, tất cả đều có cấu trúc tuần tự.
Do đó, hoàn toàn hợp lý khi ta giả định rằng các mô hình chuyên biệt cho những kiểu dữ liệu này sẽ giúp việc mô tả dữ liệu và giải quyết các bài toán ước lượng được tốt hơn.

<!--
Another issue arises from the fact that we might not only receive a sequence as an input but rather might be expected to continue the sequence.
For instance, the task could be to continue the series 2, 4, 6, 8, 10, ... 
This is quite common in time series analysis, to predict the stock market, the fever curve of a patient or the acceleration needed for a race car.
Again we want to have models that can handle such data.
-->

Một vấn đề nữa phát sinh khi chúng ta không chỉ nhận một chuỗi làm đầu vào mà còn muốn dự đoán những phần tử tiếp theo của chuỗi.
Ví dụ, bài toán có thể là dự đoán phần tử tiếp theo trong dãy 2, 4, 6, 8, 10, ...
Tác vụ này khá phổ biến trong phân tích chuỗi thời gian: để dự đoán thị trường chứng khoán, đường cong biểu hiện tình trạng sốt của bệnh nhân, hoặc gia tốc cần thiết cho một chiếc xe đua.
Một lần nữa, chúng ta muốn xây dựng các mô hình có thể xử lý ổn thỏa kiểu dữ liệu trên.

<!--
In short, while convolutional neural networks can efficiently process spatial information, recurrent neural networks are designed to better handle sequential information.
These networks introduce state variables to store past information, and then determine the current outputs, together with the current inputs.
-->

Tóm lại, trong khi các mạng nơ-ron tích chập có thể xử lý hiệu quả thông tin trên chiều không gian, thì các mạng nơ-ron hồi tiếp được thiết kế để xử lý thông tin tuần tự tốt hơn.
Các mạng này sử dụng các biến trạng thái để lưu trữ thông tin trong quá khứ, sau đó dựa vào chúng và các đầu vào hiện tại để xác định các đầu ra hiện tại.

<!--
Many of the examples for using recurrent networks are based on text data.
Hence, we will emphasize language models in this chapter.
After a more formal review of sequence data we discuss basic concepts of a language model and use this discussion as the inspiration for the design of recurrent neural networks.
Next, we describe the gradient calculation method in recurrent neural networks to explore problems that may be encountered in recurrent neural network training.
-->

Ở chương này, đa phần những ví dụ đề cập đến các mạng hồi tiếp đều dựa trên dữ liệu văn bản.
Vì vậy, chúng ta sẽ cùng đào sâu tìm hiểu những mô hình ngôn ngữ.
Sau khi tìm hiểu về dữ liệu chuỗi, ta sẽ thảo luận các khái niệm cơ bản của mô hình ngôn ngữ để làm bàn đạp cho việc thiết kế các mạng nơ-ron hồi tiếp.
Cuối cùng, ta sẽ tiến hành mô tả phương pháp tính toán gradient trong các mạng nơ-ron hồi tiếp để từ đó hiểu rõ hơn các vấn đề có thể gặp phải trong quá trình huấn luyện.

```toc
:maxdepth: 2

sequence_vn
text-preprocessing_vn
language-models-and-dataset_vn
rnn_vn
rnn-scratch_vn
rnn-gluon_vn
bptt_vn
```

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Văn Quang
* Lê Khắc Hồng Phúc
* Phạm Hồng Vinh
* Phạm Minh Đức
* Nguyễn Văn Cường