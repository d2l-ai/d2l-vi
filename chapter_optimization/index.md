# Thuật toán tối ưu hóa
:label:`chap_optimization`

Nếu bạn đọc cuốn sách theo thứ tự cho đến thời điểm này, bạn đã sử dụng một số thuật toán tối ưu hóa để đào tạo các mô hình học sâu. Chúng là những công cụ cho phép chúng tôi tiếp tục cập nhật các tham số mô hình và giảm thiểu giá trị của chức năng mất mát, như được đánh giá trên bộ đào tạo. Thật vậy, bất kỳ ai nội dung với việc xử lý tối ưu hóa như một thiết bị hộp đen để giảm thiểu các chức năng khách quan trong một cài đặt đơn giản cũng có thể nội dung chính mình với kiến thức rằng tồn tại một loạt các câu thần chú của một thủ tục như vậy (với tên như “SGD” và “Adam”). 

Tuy nhiên, để làm tốt, một số kiến thức sâu hơn là bắt buộc. Thuật toán tối ưu hóa rất quan trọng đối với việc học sâu. Một mặt, đào tạo một mô hình học sâu phức tạp có thể mất hàng giờ, ngày hoặc thậm chí vài tuần. Hiệu suất của thuật toán tối ưu hóa ảnh hưởng trực tiếp đến hiệu quả đào tạo của mô hình. Mặt khác, việc hiểu các nguyên tắc của các thuật toán tối ưu hóa khác nhau và vai trò của các siêu tham số của chúng sẽ cho phép chúng ta điều chỉnh các siêu tham số theo cách nhắm mục tiêu để cải thiện hiệu suất của các mô hình học sâu. 

Trong chương này, chúng tôi khám phá các thuật toán tối ưu hóa học sâu phổ biến trong chuyên sâu. Hầu như tất cả các vấn đề tối ưu hóa phát sinh trong học sâu đều là * nonconvex*. Tuy nhiên, việc thiết kế và phân tích các thuật toán trong bối cảnh các bài toán * lồi đã được chứng minh là rất hướng dẫn. Chính vì lý do đó mà chương này bao gồm một mồi về tối ưu hóa lồi và bằng chứng cho một thuật toán gốc gradient ngẫu nhiên rất đơn giản trên một chức năng khách quan lồi.

```toc
:maxdepth: 2

optimization-intro
convexity
gd
sgd
minibatch-sgd
momentum
adagrad
rmsprop
adadelta
adam
lr-scheduler
```
