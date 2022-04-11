# Tính toán học sâu
:label:`chap_computation`

Bên cạnh các bộ dữ liệu khổng lồ và phần cứng mạnh mẽ, các công cụ phần mềm tuyệt vời đã đóng một vai trò không thể thiếu trong tiến bộ nhanh chóng của học sâu. Bắt đầu với thư viện Theano phá vỡ được phát hành vào năm 2007, các công cụ mã nguồn mở linh hoạt đã cho phép các nhà nghiên cứu nhanh chóng các mô hình nguyên mẫu, tránh làm việc lặp đi lặp lại khi tái chế các thành phần tiêu chuẩn trong khi vẫn duy trì khả năng thực hiện các sửa đổi cấp thấp. Theo thời gian, các thư viện của deep learning đã phát triển để cung cấp các trừu tượng ngày càng thô. Cũng giống như các nhà thiết kế bán dẫn đã đi từ chỉ định bóng bán dẫn đến các mạch logic sang viết mã, các nhà nghiên cứu mạng thần kinh đã chuyển từ suy nghĩ về hành vi của các tế bào thần kinh nhân tạo riêng lẻ sang hình thành mạng về toàn bộ lớp, và bây giờ thường thiết kế kiến trúc với thô hơn xa * khối* trong tâm trí. 

Cho đến nay, chúng tôi đã giới thiệu một số khái niệm máy học cơ bản, tăng cường các mô hình học sâu đầy đủ chức năng. Trong chương cuối, chúng tôi đã triển khai từng thành phần của MLP từ đầu và thậm chí cho thấy cách tận dụng các API cấp cao để triển khai các mô hình tương tự một cách dễ dàng. Để giúp bạn có được đến mức nhanh như vậy, chúng tôi * gọi lên* các thư viện, nhưng bỏ qua các chi tiết nâng cao hơn về * cách chúng hoạt động*. Trong chương này, chúng ta sẽ bóc lại bức màn, đào sâu hơn vào các thành phần chính của tính toán học sâu, cụ thể là xây dựng mô hình, truy cập và khởi tạo tham số, thiết kế các lớp và khối tùy chỉnh, đọc và ghi các mô hình vào đĩa và tận dụng GPU để đạt được tốc độ ấn tượng. Những thông tin chi tiết này sẽ chuyển bạn từ * người dùng cùng* sang *power user*, cung cấp cho bạn các công cụ cần thiết để gặt hái những lợi ích của thư viện deep learning trưởng thành trong khi vẫn giữ được sự linh hoạt để triển khai các mô hình phức tạp hơn, bao gồm cả những người bạn tự phát minh ra! Mặc dù chương này không giới thiệu bất kỳ mô hình hoặc bộ dữ liệu mới nào, nhưng các chương mô hình nâng cao theo sau phụ thuộc rất nhiều vào các kỹ thuật này.

```toc
:maxdepth: 2

model-construction
parameters
deferred-init
custom-layer
read-write
use-gpu
```
