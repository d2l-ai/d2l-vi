# Mạng nơ-ron tái phát
:label:`chap_rnn`

Cho đến nay chúng tôi đã gặp hai loại dữ liệu: dữ liệu dạng bảng và dữ liệu hình ảnh. Đối với cái sau, chúng tôi đã thiết kế các lớp chuyên dụng để tận dụng sự đều đặn trong chúng. Nói cách khác, nếu chúng ta hoán vị các pixel trong một hình ảnh, sẽ khó khăn hơn nhiều để lý luận về nội dung của nó về một cái gì đó trông giống như nền của một mẫu thử nghiệm trong thời đại TV analog. 

Quan trọng nhất, cho đến nay chúng tôi ngầm cho rằng dữ liệu của chúng tôi đều được rút ra từ một số phân phối và tất cả các ví dụ được phân phối độc lập và giống hệt nhau (i.i.d.). Thật không may, điều này không đúng đối với hầu hết dữ liệu. Ví dụ, các từ trong đoạn này được viết theo thứ tự, và sẽ khá khó để giải mã ý nghĩa của nó nếu chúng được hoán vị ngẫu nhiên. Tương tự như vậy, khung hình ảnh trong video, tín hiệu âm thanh trong cuộc trò chuyện và hành vi duyệt web trên một trang web, tất cả đều theo thứ tự tuần tự. Do đó, hợp lý khi cho rằng các mô hình chuyên dụng cho dữ liệu như vậy sẽ làm tốt hơn trong việc mô tả chúng. 

Một vấn đề khác phát sinh từ thực tế là chúng ta có thể không chỉ nhận được một chuỗi như một đầu vào mà có thể được dự kiến sẽ tiếp tục trình tự. Ví dụ, nhiệm vụ có thể là tiếp tục loạt $2, 4, 6, 8, 10, \ldots$ Điều này khá phổ biến trong phân tích chuỗi thời gian, để dự đoán thị trường chứng khoán, đường cong sốt của bệnh nhân hoặc khả năng tăng tốc cần thiết cho một chiếc xe đua. Một lần nữa chúng tôi muốn có các mô hình có thể xử lý dữ liệu như vậy. 

Nói tóm lại, trong khi CNN có thể xử lý hiệu quả thông tin không gian, * mạng thần kinh định kỳ* (RNN) được thiết kế để xử lý thông tin tuần tự tốt hơn. RNNs giới thiệu các biến trạng thái để lưu trữ thông tin quá khứ, cùng với các đầu vào hiện tại, để xác định các đầu ra hiện tại. 

Nhiều ví dụ để sử dụng các mạng định kỳ dựa trên dữ liệu văn bản. Do đó, chúng tôi sẽ nhấn mạnh các mô hình ngôn ngữ trong chương này. Sau khi xem xét chính thức hơn về dữ liệu trình tự, chúng tôi giới thiệu các kỹ thuật thực tế để xử lý trước dữ liệu văn bản. Tiếp theo, chúng ta thảo luận về các khái niệm cơ bản về mô hình ngôn ngữ và sử dụng cuộc thảo luận này làm nguồn cảm hứng cho việc thiết kế RNNs. Cuối cùng, chúng tôi mô tả phương pháp tính toán gradient cho RNNs để khám phá các vấn đề có thể gặp phải khi đào tạo các mạng như vậy.

```toc
:maxdepth: 2

sequence
text-preprocessing
language-models-and-dataset
rnn
rnn-scratch
rnn-concise
bptt
```
