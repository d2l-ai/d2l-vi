<!-- ===================== Bắt đầu dịch ==================== -->

<!--
# Multilayer Perceptrons
-->

# Perceptron Đa tầng
:label:`chap_perceptrons`

<!--
In this chapter, we will introduce your first truly *deep* networks.
The simplest deep networks are called multilayer perceptrons, and they consist of many layers of neurons each fully connected to those in the layer below 
(from which they receive input) and those above (which they, in turn, influence).
When we train high-capacity models we run the risk of overfitting.
Thus, we will need to provide your first rigorous introduction to the notions of overfitting, underfitting, and capacity control.
To help you combat these problems, we will introduce regularization techniques such as dropout and weight decay.
We will also discuss issues relating to numerical stability and parameter initialization that are key to successfully training deep networks.
Throughout, we focus on applying models to real data, aiming to give the reader a firm grasp not just of the concepts but also of the practice of using deep networks.
We punt matters relating to the computational performance, scalability and efficiency of our models to subsequent chapters.
-->

Trong chương này, chúng tôi sẽ giới thiệu mạng nơ-ron *sâu* thực sự đầu tiên của bạn.
Mạng nơ-ron sâu đơn giản nhất được gọi là perceptron đa tầng.
Nó gồm nhiều tầng nơ-ron, mỗi nơ-ron được kết nối đầy đủ với các nơ-ron khác ở tầng phía dưới (các nơ-ron cung cấp đầu vào) và tầng phía trên (các nơ-ron mà nó gây ảnh hưởng).
Khi huấn luyện các mô hình có độ phức tạp cao, ta sẽ có nguy cơ gặp vấn đề quá khớp.
Vì vậy, chúng tôi cần cung cấp cho bạn những hiểu biết ban đầu thật chặt chẽ với các khái niệm quá khớp, dưới khớp và kiểm soát độ phức tạp.
Nhằm giúp bạn giải quyết những vấn đề kể trên, chúng tôi sẽ giới thiệu những kỹ thuật điều chuẩn như dropout và suy giảm trọng số.
Ta cũng sẽ bàn đến các vấn đề liên quan tới sự ổn định số học và việc khởi tạo tham số, hai yếu tố chính giúp việc huấn luyện mạng nơ-ron sâu thành công.
Xuyên suốt chương này, chúng tôi tập trung vào việc áp dụng các mô hình cho dữ liệu thực tế, nhằm giúp độc giả không chỉ nắm vững được các khái niệm mà còn có thể thực hành sử dụng mạng nơ-ron sâu.
Những vấn đề liên quan tới hiệu năng tính toán, khả năng mở rộng và mức hiệu quả của mô hình sẽ được giới thiệu ở các chương sau.

```toc
:maxdepth: 2

mlp_vn
mlp-scratch_vn
mlp-gluon_vn
underfit-overfit_vn
weight-decay_vn
dropout_vn
backprop_vn
numerical-stability-and-init_vn
environment_vn
kaggle-house-price_vn
```

<!-- ===================== Kết thúc dịch ==================== -->

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.

* Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md.
-->

* Đoàn Võ Duy Thanh
* Phạm Minh Đức
