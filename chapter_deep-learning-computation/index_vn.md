<!--
# Deep Learning Computation
-->

# Tính toán Học sâu
:label:`chap_computation`

<!--
Alongside giant datasets and powerful hardware, great software tools have played an indispensable role in the rapid progress of deep learning.
Starting with the pathbreaking Theano library released in 2007, flexible open-source tools have enabled researchers to rapidly prototype models 
avoiding repetitive work when recycling standard components while still maintaining the ability to make low-level modifications.
Over time, deep learning's libraries have evolved to offer increasingly coarse abstractions.
Just as semiconductor designers went from specifying transistors to logical circuits to writing code, 
neural networks researchers have moved from thinking about the behavior of individual artificial neurons to conceiving of networks in terms of whole layers, 
and now often design architectures with far coarser *blocks* in mind.
-->

Ngoài các tập dữ liệu khổng lồ và phần cứng mạnh mẽ, không thể không nhắc tới vai trò quan trọng của các công cụ phần mềm tốt trong sự phát triển chóng mặt của học sâu.
Mở đầu với thư viện tiên phong Theano được phát hành vào năm 2007, các công cụ mã nguồn mở linh hoạt đã giúp các nhà nghiên cứu nhanh chóng thử nghiệm các mô hình bằng cách tránh việc bắt người dùng phải xây dựng lại các thành phần tiêu chuẩn nhưng vẫn cho phép việc thay đổi ở bậc thấp. 
Theo thời gian, các thư viện học sâu ngày càng phát triển để cung cấp tính trừu tượng cao hơn.
Tương tự với việc các nhà thiết kế chất bán dẫn đi từ việc chỉ rõ các lựa chọn bóng bán dẫn đến mạch logic để viết mã nguồn, các nhà nghiên cứu mạng nơ-ron sâu đã thay đổi từ việc nghĩ về hành vi của từng nơ-ron nhân tạo đơn lẻ sang việc xem xét cả một tầng trong mạng nơ-ron.
Giờ đây, họ thường thiết kế các kiến trúc mạng ở mức độ trừu tượng là các *khối*.

<!--
So far, we have introduced some basic machine learning concepts, ramping up to fully-functional deep learning models.
In the last chapter, we implemented each component of a multilayer perceptron from scratch and even showed how to leverage MXNet's Gluon library to roll out the same models effortlessly.
To get you that far that fast, we *called upon* the libraries, but skipped over more advanced details about *how they work*.
In this chapter, we will peel back the curtain, digging deeper into the key components of deep learning computation, 
namely model construction, parameter access and initialization, designing custom layers and blocks, reading and writing models to disk, and leveraging GPUs to achieve dramatic speedups.
These insights will move you from *end user* to *power user*, giving you the tools needed to combine the reap the benefits of a mature deep learning library, 
while retaining the flexibility to implement more complex models, including those you invent yourself!
While this chapter does not introduce any new models or datasets, the advanced modeling chapters that follow rely heavily on these techniques.
-->

Đến nay, chúng tôi đã giới thiệu một vài khái niệm học máy cơ bản, rồi tiến tới các mô hình học sâu.
Ở chương trước, ta đã lập trình từng thành phần của một perceptron đa tầng từ đầu và biết được cách tận dụng thư viện Gluon từ MXNet để xây dựng lại mô hình một cách dễ dàng hơn.
Để giúp bạn có những bước tiến xa hơn mức mong đợi, chúng tôi tập trung vào việc *sử dụng* các thư viện và không đề cập đến những chi tiết nâng cao hơn về *cách hoạt động của chúng*.
Trong chương này, chúng tôi sẽ vén tấm màn bí ẩn và đào sâu vào những yếu tố chính của tính toán học sâu; cụ thể là việc xây dựng mô hình, truy cập và khởi tạo tham số, thiết kế các tầng và khối tùy chỉnh, đọc và ghi mô hình lên ổ cứng và cuối cùng là tận dụng GPU nhằm đạt được tốc độ đáng kể.
Những hiểu biết này sẽ giúp bạn từ một *người dùng cuối* (*end user*) trở thành một *người dùng thành thạo* (*power user*), cung cấp cho bạn các công cụ cần thiết để gặt hái lợi ích của một thư viện học sâu trưởng thành, đồng thời giữ được sự linh hoạt để lập trình những mô hình phức tạp hơn, bao gồm cả những mô hình mà bạn tự phát minh!
Mặc dù chương này không giới thiệu bất cứ mô hình hay tập dữ liệu mới nào, các chương sau về mô hình nâng cao sẽ phụ thuộc rất nhiều vào những kỹ thuật sắp được nhắc đến. 

```toc
:maxdepth: 2

model-construction_vn
parameters_vn
deferred-init_vn
custom-layer_vn
read-write_vn
use-gpu_vn
```

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Phạm Minh Đức
