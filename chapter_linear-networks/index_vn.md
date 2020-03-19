<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->

<!--
# Linear Neural Networks
-->

# Mạng nơ-ron Tuyến tính
:label:`chap_linear`

<!--
Before we get into the details of deep neural networks, we need to cover the basics of neural network training. 
In this chapter, we will cover the entire training process, including defining simple neural network architectures, handling data, specifying a loss function, and training the model. 
In order to make things easier to grasp, we begin with the simplest concepts. 
Fortunately, classic statistical learning techniques such as linear and logistic regression can be cast as *shallow* neural networks. 
Starting from these classic algorithms, we will introduce you to the basics, 
providing the basis for more complex techniques such as softmax regression (introduced at the end of this chapter) and multilayer perceptrons (introduced in the next chapter).
-->

Trước khi tìm hiểu chi tiết về mạng nơ-ron sâu, chúng ta cần nắm vững những kiến thức căn bản của việc huấn luyện mạng nơ-ron.
Chương này sẽ đề cập đến toàn bộ quá trình huấn luyện, bao gồm xác định kiến trúc mạng nơ-ron đơn giản, xử lý dữ liệu, chỉ rõ hàm mất mát và huấn luyện mô hình.
Để mọi thứ dễ dàng hơn, ta sẽ bắt đầu với một số khái niệm đơn giản nhất.
May thay, một số phương pháp học thống kê cổ điển như hồi quy tuyến tính, hồi quy logistic có thể được xem như những mạng nơ-ron *nông*.
Hãy bắt đầu bằng những thuật toán cổ điển này, chúng tôi sẽ giới thiệu những nội dung căn bản nhằm tạo nền tảng cho những kỹ thuật phức tạp hơn như Hồi quy Softmax (sẽ được giới thiệu ở cuối chương này) và Perceptron đa tầng (sẽ được giới thiệu ở chương sau).

```toc
:maxdepth: 2

linear-regression_vn
linear-regression-scratch_vn
linear-regression-gluon_vn
softmax-regression_vn
fashion-mnist_vn
softmax-regression-scratch_vn
softmax-regression-gluon_vn
```

<!-- ===================== Kết thúc dịch Phần 1 ==================== -->

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.
-->

* Đoàn Võ Duy Thanh
* Trần Hoàng Quân
* Vũ Hữu Tiệp
