#  Sơ bộ
:label:`chap_preliminaries`

Để bắt đầu học sâu, chúng ta sẽ cần phát triển một vài kỹ năng cơ bản. Tất cả machine learning đều liên quan đến việc trích xuất thông tin từ dữ liệu. Vì vậy, chúng ta sẽ bắt đầu bằng cách học các kỹ năng thực tế để lưu trữ, thao tác và xử lý dữ liệu trước. 

Hơn nữa, machine learning thường yêu cầu làm việc với các tập dữ liệu lớn, mà chúng ta có thể nghĩ là bảng, trong đó các hàng tương ứng với các ví dụ và các cột tương ứng với các thuộc tính. Đại số tuyến tính cho chúng ta một bộ kỹ thuật mạnh mẽ để làm việc với dữ liệu dạng bảng. Chúng tôi sẽ không đi quá xa vào cỏ dại mà tập trung vào cơ bản của các hoạt động ma trận và việc thực hiện chúng. 

Ngoài ra, học sâu là tất cả về tối ưu hóa. Chúng tôi có một mô hình với một số thông số và chúng tôi muốn tìm những thông số phù hợp với dữ liệu của chúng tôi* tốt nhất*. Xác định cách nào để di chuyển từng tham số ở mỗi bước của một thuật toán đòi hỏi một chút giải tích, sẽ được giới thiệu ngắn gọn. May mắn thay, gói `autograd` tự động tính toán sự khác biệt cho chúng tôi và chúng tôi sẽ đề cập đến nó tiếp theo. 

Tiếp theo, machine learning có liên quan đến việc đưa ra dự đoán: giá trị khả năng của một số thuộc tính không xác định là gì, với thông tin mà chúng ta quan sát? Để lý do nghiêm ngặt dưới sự không chắc chắn, chúng ta sẽ cần phải gọi ngôn ngữ xác suất. 

Cuối cùng, tài liệu chính thức cung cấp nhiều mô tả và ví dụ vượt ra ngoài cuốn sách này. Để kết thúc chương, chúng tôi sẽ chỉ cho bạn cách tra cứu tài liệu cho các thông tin cần thiết. 

Cuốn sách này đã giữ nội dung toán học ở mức tối thiểu cần thiết để có được sự hiểu biết đúng đắn về học sâu. Tuy nhiên, điều đó không có nghĩa là cuốn sách này là toán học miễn phí. Do đó, chương này cung cấp một giới thiệu nhanh chóng về toán học cơ bản và thường được sử dụng để cho phép bất cứ ai hiểu ít nhất *most* về nội dung toán học của cuốn sách. Nếu bạn muốn hiểu *tất cả* của nội dung toán học, xem xét thêm [online appendix on mathematics](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/index.html) là đủ.

```toc
:maxdepth: 2

ndarray
pandas
linear-algebra
calculus
autograd
probability
lookup-api
```
