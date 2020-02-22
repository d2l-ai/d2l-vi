<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->

<!--
#  Preliminaries
-->

# Sơ bộ
:label:`chap_preliminaries`

<!--
To get started with deep learning, we will need to develop a few basic skills.
All machine learning is concerned with extracting information from data.
So we will begin by learning the practical skills for storing, manipulating, and preprocessing data.
-->

Để bắt đầu với học sâu, ta sẽ cần nắm bắt một vài kỹ năng cơ bản.
Tất cả những vấn đề về học máy đều có liên quan đến việc trích xuất thông tin từ dữ liệu.
Vì vậy, chúng tôi sẽ bắt đầu bằng cách học các kỹ năng thực tế để lưu trữ, thao tác và xử lý dữ liệu.

<!--
Moreover, machine learning typically requires working with large datasets, which we can think of as tables, where the rows correspond to examples and the columns correspond to attributes.
Linear algebra gives us a powerful set of techniques for working with tabular data.
We will not go too far into the weeds but rather focus on the basic of matrix operations and their implementation.
-->

Hơn nữa, học máy thường yêu cầu làm việc với các tập dữ liệu lớn, mà chúng ta có thể coi như ở dạng bảng, trong đó các hàng tương ứng với các mẫu và các cột tương ứng với các thuộc tính.
Đại số tuyến tính cung cấp cho ta một tập kỹ thuật mạnh mẽ để làm việc với dữ liệu dạng bảng.
Chúng ta sẽ không đi quá sâu mà chỉ tập trung cơ bản vào các toán tử ma trận cơ bản và cách thực thi chúng.

<!--
Additionally, deep learning is all about optimization.
We have a model with some parameters and we want to find those that fit our data *the best*.
Determining which way to move each parameter at each step of an algorithm requires a little bit of calculus, which will be briefly introduced.
Fortunately, the `autograd` package automatically computes differentiation for us, and we will cover it next.
-->

Bên cạnh đó, học sâu luôn liên quan đến tối ưu hoá.
Chúng ta có một mô hình với bộ tham số và muốn tìm ra các tham số khớp với dữ liệu *nhất*.
Việc xác định cách điều chỉnh từng tham số ở mỗi bước trong thuật toán đòi hỏi một chút kiến thức về giải tích, mà sẽ được giới thiệu ngắn gọn dưới đây.
May thay, gói `autograd` sẽ tự động tính đạo hàm cho chúng ta, và sẽ được đề cập ngay sau đó.

<!-- ===================== Kết thúc dịch Phần 1 ==================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ==================== -->

<!--
Next, machine learning is concerned with making predictions: what is the likely value of some unknown attribute, given the information that we observe?
To reason rigorously under uncertainty we will need to invoke the language of probability.
-->

Kế tiếp, học máy liên quan đến việc đưa ra những dự đoán như: Xác định giá trị của một số thuộc tính chưa biết dựa trên thông tin quan sát được?
Để suy luận chặt chẽ dưới sự bất định, chúng ta sẽ cần tìm đến ngôn ngữ của xác suất.

<!--
In the end, the official documentation provides plenty of descriptions and examples that are beyond this book.
To conclude the chapter, we will show you how to look up documentation for the needed information.
-->

Cuối cùng, tài liệu tham khảo chính thức cung cấp rất nhiều mô tả và ví dụ nằm ngoài cuốn sách này.
Để kết thúc chương này, chúng tôi sẽ chỉ bạn cách tra cứu tài liệu tham khảo cho các thông tin cần thiết.

<!--
This book has kept the mathematical content to the minimum necessary to get a proper understanding of deep learning.
However, it does not mean that this book is mathematics free.
Thus, this chapter provides a rapid introduction to basic and frequently-used mathematics to allow anyone to understand at least *most* of the mathematical content of the book.
If you wish to understand *all* of the mathematical content, further reviewing :numref:`chap_appendix_math` should be sufficient.
-->

Cuốn sách này đã cung cấp nội dung toán học ở mức tối thiểu cần có để có được sự hiểu biết đúng đắn về học sâu.
Tuy nhiên, điều đó không đồng nghĩa rằng cuốn sách này không cần các kiến thức toán học.
Do vậy, chương này sẽ giới thiệu nhanh về các kiến thức toán học cơ bản và thông dụng, cho phép tất cả mọi người tối thiểu là sẽ hiểu được *hầu hết* nội dung toán trong quyển sách này.
Nếu bạn muốn hiểu *tất cả* nội dung toán học, hãy tham khảo thêm :numref:`chap_appendix_math`.

```toc
:maxdepth: 2

ndarray_vn
pandas_vn
linear-algebra_vn
calculus_vn
autograd_vn
probability_vn
lookup-api_vn
```


<!-- ===================== Kết thúc dịch Phần 2 ==================== -->

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
* Nguyễn Cảnh Thướng
* Vũ Hữu Tiệp
* Lê Khắc Hồng Phúc
* Phạm Hồng Vinh
