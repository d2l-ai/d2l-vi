<!--
# Recurrent Neural Networks
-->

# Mạng nơ-ron Truy hồi
:label:`chap_rnn`

<!--
So far we encountered two types of data: generic vectors and images.
For the latter we designed specialized layers to take advantage of the regularity properties in them.
In other words, if we were to permute the pixels in an image, it would be much more difficult to reason about 
its content of something that would look much like the background of a test pattern in the times of analog TV.
-->

Cho đến nay, chúng ta đã gặp hai loại dữ liệu: vector chung và hình ảnh.
Với dữ liệu hình ảnh, chúng ta đã thiết kế các tầng chuyên biệt để tận dụng các tính chất tương tự trong hình ảnh.
Nói cách khác, nếu chúng ta hoán vị các điểm ảnh trong một ảnh, sẽ rất khó để suy luận về 
nội dung của một thứ tương tự như nền trong ảnh mẫu trong thời đại truyền hình tương tự.

<!--
Most importantly, so far we tacitly assumed that our data is generated i.i.d., i.e., independently and identically distributed, all drawn from some distribution.
Unfortunately, this is not true for most data.
For instance, the words in this paragraph are written in sequence, and it would be quite difficult to decipher its meaning if they were permuted randomly.
Likewise, image frames in a video, the audio signal in a conversation, or the browsing behavior on a website, all follow sequential order.
It is thus only reasonable to assume that specialized models for such data will do better at describing it and at solving estimation problems.
-->

Quan trọng hơn cả, cho đến thời điểm này, chúng ta đã giả định rằng dữ liệu được sinh từ các phân phối độc lập và giống hệt nhau (viết tắt là i.i.d).
Thật không may, điều này không đúng với hầu hết các dữ liệu.
Ví dụ, các từ trong đoạn này được viết theo trình tự, và sẽ rất khó để giải mã nghĩa của chúng nếu các từ được hoán vị một cách ngẫu nhiên.
Tương tự, các khung hình ảnh trong video, tín hiệu âm thanh trong cuộc hội thoại, hoặc hành vi duyệt web, tất cả đều tuân theo thứ tựchuỗi.
Do đó, chỉ hợp lý khi giả định rằng các mô hình chuyên biệt cho những dữ liệu này sẽ giúp việc mô tả dữ liệu và giải quyết các bài toán ước lượng tốt hơn.

<!--
Another issue arises from the fact that we might not only receive a sequence as an input but rather might be expected to continue the sequence.
For instance, the task could be to continue the series 2, 4, 6, 8, 10, ... 
This is quite common in time series analysis, to predict the stock market, the fever curve of a patient or the acceleration needed for a race car.
Again we want to have models that can handle such data.
-->

Một vấn đề nữa nảy sinh khi chúng ta không chỉ nhận một chuỗi làm đầu vào mà còn muốn dự đoán những phần tử tiếp theo của chuỗi.
Ví dụ, bài toán có thể là dự đoán phần tử tiếp theo trong dãy 2, 4, 6, 8, 10, ...
Điều này khá phổ biến trong phân tích chuỗi thời gian, để dự đoán thị trường chứng khoán, đường cong biểu hiện sốt của bệnh nhân, hoặc khả gia tốc cần thiết cho một chiếc xe đua.
Một lần nữa, chúng ta muốn xây dựng các mô hình có thể xử lý dữ liệu trên.

<!--
In short, while convolutional neural networks can efficiently process spatial information, recurrent neural networks are designed to better handle sequential information.
These networks introduce state variables to store past information, and then determine the current outputs, together with the current inputs.
-->

Nói tóm lại, trong khi các mạng nơ-ron tích chập có thể xử lý hiệu quả những dữ liệu không gian, thì các mạng nơ-ron truy hồi được thiết kế để xử lý dữ liệu chuỗi tốt hơn.
Các mạng này giới thiệu các biến trạng thái để lưu trữ thông tin trong quá khứ và sau đó xác định các đầu ra hiện tại, cùng với các đầu vào hiện tại.

<!--
Many of the examples for using recurrent networks are based on text data.
Hence, we will emphasize language models in this chapter.
After a more formal review of sequence data we discuss basic concepts of a language model and use this discussion as the inspiration for the design of recurrent neural networks.
Next, we describe the gradient calculation method in recurrent neural networks to explore problems that may be encountered in recurrent neural network training.
-->

Rất nhiều ví dụ về các mạng truy hồi trong chương này dựa trên dữ liệu văn bản.
Do vậy, chúng tôi sẽ bàn nhiều hơn các mô hình ngôn ngữ trong chương này.
Sau khi xem xét về dữ liệu chuỗi, chúng tôi sẽ thảo luận về các khái niệm cơ bản của mô hình ngôn ngữ  để lấy nguồn cảm hứng thiết kế các mạng nơ-ron truy hồi.
Tiếp đến, chúng tôi sẽ mô tả phương pháp tính toán gradient trong các mạng nơ-ron truy hồi, từ đó hiểu rõ hơn các vấn đề có thể gặp phải trong việc huấn luyện mạng nơ-ron truy hồi.

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
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn. Ưu tiên kiểm tra danh sách phía dưới để điền tên đầy đủ của reviewer.

* Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md
-->


* Đoàn Võ Duy Thanh
* Nguyễn Văn Quang
