<!--
# Optimization Algorithms
-->

# Thuật toán Tối ưu
:label:`chap_optimization`

<!--
If you read the book in sequence up to this point you already used a number of advanced optimization algorithms to train deep learning models.
They were the tools that allowed us to continue updating model parameters and to minimize the value of the loss function, as evaluated on the training set.
Indeed, anyone content with treating optimization as a black box device to minimize objective functions in a simple setting might well content oneself 
with the knowledge that there exists an array of incantations of such a procedure (with names such as "Adam", "NAG", or "SGD").
-->

Độc giả theo dõi đến chương này của cuốn sách hẳn đã sử dụng nhiều thuật toán tối ưu tiên tiến để huấn luyện các mô hình học sâu. 
Chúng là công cụ cho phép ta liên tục cập nhật các tham số của mô hình và cực tiểu hóa giá trị hàm mất mát khi đánh giá trên tập huấn luyện. 
Sự thật là có nhiều người hài lòng với việc coi các thuật toán tối ưu như một hộp đen ma thuật (với các câu thần chú như "Adam", "NAG", hoặc "SGD") có tác dụng cực tiểu hóa hàm mục tiêu.

<!--
To do well, however, some deeper knowledge is required.
Optimization algorithms are important for deep learning.
On one hand, training a complex deep learning model can take hours, days, or even weeks.
The performance of the optimization algorithm directly affects the model's training efficiency.
On the other hand, understanding the principles of different optimization algorithms and the role of their parameters will enable us 
to tune the hyperparameters in a targeted manner to improve the performance of deep learning models.
-->

Tuy nhiên, để làm tốt, ta cần các kiến thức sâu hơn. 
Các thuật toán tối ưu rất quan trọng trong học sâu. 
Một mặt, việc huấn luyện một mô hình học sâu phức tạp có thể mất hàng giờ, hàng ngày, thậm chí hàng tuần. 
Hoạt động của thuật toán tối ưu trực tiếp ảnh hưởng đến hiệu quả và tốc độ huấn luyện. 
Mặt khác, việc hiểu rõ nguyên lý của các thuật toán tối ưu khác nhau và vai trò của các tham số của chúng sẽ giúp ta điều chỉnh các siêu tham số một cách có chủ đích nhằm cải thiện hiệu suất của mô hình.

<!--
In this chapter, we explore common deep learning optimization algorithms in depth.
Almost all optimization problems arising in deep learning are *nonconvex*.
Nonetheless, the design and analysis of algorithms in the context of convex problems has proven to be very instructive.
It is for that reason that this section includes a primer on convex optimization and the proof for a very simple stochastic gradient descent algorithm on a convex objective function.
-->

Chương này đề cập sâu hơn đến các thuật toán tối ưu học sâu. Hầu hết tất cả các vấn đề tối ưu xuất hiện trong học sâu là *không lồi* (*nonconvex*). 
Tuy nhiên, kiến thức từ việc thiết kế và phân tích các thuật toán giải quyết bài toán tối ưu lồi đã được chứng tỏ là rất hữu ích khi áp dụng trong các bài toán không lồi.
Do vậy, phần này tập trung vào giới thiệu tối ưu lồi và chứng minh một thuật toán hạ gradient ngẫu nhiên (*stochastic gradient descent*) đơn giản áp dụng cho hàm mục tiêu lồi.

```toc
:maxdepth: 2

optimization-intro_vn
convexity_vn
gd_vn
sgd_vn
minibatch-sgd_vn
momentum_vn
adagrad_vn
rmsprop_vn
adadelta_vn
adam_vn
lr-scheduler_vn
```

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Văn Cường
* Lê Khắc Hồng Phúc
