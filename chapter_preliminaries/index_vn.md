<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->

<!--
#  Preliminaries
-->

# *dịch tiêu đề phía trên*
:label:`chap_preliminaries`

<!--
To get started with deep learning, we will need to develop a few basic skills.
All machine learning is concerned with extracting information from data.
So we will begin by learning the practical skills for storing, manipulating, and preprocessing data.
-->

*dịch đoạn phía trên*

<!--
Moreover, machine learning typically requires working with large datasets, which we can think of as tables, where the rows correspond to examples and the columns correspond to attributes.
Linear algebra gives us a powerful set of techniques for working with tabular data.
We will not go too far into the weeds but rather focus on the basic of matrix operations and their implementation.
-->

*dịch đoạn phía trên*

<!--
Additionally, deep learning is all about optimization.
We have a model with some parameters and we want to find those that fit our data *the best*.
Determining which way to move each parameter at each step of an algorithm requires a little bit of calculus, which will be briefly introduced.
Fortunately, the `autograd` package automatically computes differentiation for us, and we will cover it next.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 1 ==================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ==================== -->

<!--
Next, machine learning is concerned with making predictions: what is the likely value of some unknown attribute, given the information that we observe?
To reason rigorously under uncertainty we will need to invoke the language of probability.
-->

Kế tiếp, học máy liên quan đến việc đưa ra những dự đoán như: Xác định giá trị của một số thuộc tính chưa biết dựa trên thông tin quan sát được?
Để suy luận chặt chẽ dưới sự không chắc chắn, chúng ta sẽ cần tìm đến ngôn ngữ của xác suất.

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
Do vậy, chương này cung cấp một giới thiệu nhanh về toán học cơ bản và thông dụng cho phép bất cứ ai có sự hiểu biết cơ bản *tối thiểu* về toán đều có thể tiếp cận được.
Nếu bạn muốn hiểu *tất cả* nội dung về toán học, hãy tham khảo thêm :numref:`chap_appendix_math`.

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

<!-- Phần 1 -->
*

<!-- Phần 2 -->
*
