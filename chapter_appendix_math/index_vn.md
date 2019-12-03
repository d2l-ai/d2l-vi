<!===================== Bắt đầu dịch Phần 1 ==================== -->

<!--
# Appendix: Mathematics for Deep Learning
-->

# Phụ lục: Toán học cho Học Sâu
:label:`chap_appendix_math`

<!--
**Brent Werness** (*Amazon*), **Rachel Hu** (*Amazon*), and authors of this book
-->

**Brent Werness** (*Amazon*), **Rachel Hu** (*Amazon*), và các tác giả của cuốn sách này


<!--
One of the wonderful parts of modern deep learning is the fact that much of it can be understood and used without a full understanding of the mathematics below it.  This is a sign that the field is maturing.  Just as most software developers no longer need to worry about the theory of computable functions, neither should deep learning practitioners need to worry about the theoretical foundations of maximum likelihood learning.
-->

Một trong những điểm tuyệt vời nhất của học sâu hiện đại là nó có thể được hiểu
và sử dụng mà không cần hiểu cặn kẽ nền tảng toán học đằng sau. Đây là một dấu
hiệu thể hiện lĩnh vực này đã trưởng này. Giống như hầu hết các nhà phát
triển phần mềm không cần bận tâm đến lý thuyết hàm số khả tính,
những người làm việc với học sâu cũng không cần bận tâm đến nền
tảng lý thuyết của học hợp lý cực đại (maximum likelihood).

<!--
But, we are not quite there yet.
-->

Tuy nhiên, chúng ta chưa thực sự gần với mức đó.

<!--
In practice, you will sometimes need to understand how architectural choices influence gradient flow, or the implicit assumptions you make by training with a certain loss function.  You might need to know what in the world entropy measures, and how it can help you understand exactly what bits-per-character means in your model.  These all require deeper mathematical understanding.
-->

Trên thực tế, bạn sẽ thi thoảng cần hiểu sự lựa chọn kiến trúc ảnh hưởng tới
dòng gradient như thế nào, hoặc những giả thiết ngầm khi huấn luyện với một
hàm mất mát cụ thể. Bạn có thể cần biết entropy đong đếm thứ gì trên thế giới,
và nó có thể giúp bạn hiểu chính xác số lượng bit trên một ký tự có ý nghĩa
như thế nào trong mô hình của bạn. Tất cả những điều này đòi hỏi những hiểu
biết toán học sâu hơn.

<!--
This appendix aims to provide you the mathematical background you need to understand the core theory of modern deep learning, but it is not exhaustive.  We will begin with examining linear algebra in greater depth.  We develop a geometric understanding of all the common linear algebraic objects and operations that will enable us to visualize the effects of various transformations on our data.  A key element is the development of the basics of eigen-decompositions.
-->

Phần phụ lục này nhằm cung cấp cho bạn nền tảng toán học cần thiết để hiểu
lý thuyết cốt lõi của học sâu hiện đại, nhưng đây không phải là toàn bộ kiến
thức cần thiết. Chúng ta sẽ bắt đầu xem xét đại số tuyến tính sâu hơn. Chúng tôi
phát triển ý nghĩa hình học của các đại lượng và toán tử đại số tuyến tính,
việc này cho phép chúng ta minh hoạ hiệu ứng của nhiều phép biến đổi dữ liệu.
Một thành phần chủ chốt là sự phát triển của các kiến thức nền tảng liên quan tới phân tích trị riêng.

<!--
We next develop the theory of differential calculus to the point that we can fully understand why the gradient is the direction of steepest descent, and why back-propagation takes the form it does.  Integral calculus is then discussed to the degree needed to support our next topic, probability theory.
-->

Tiếp theo, chúng ta phát triển lý thuyết giải tích vi phân để có thể hiểu cặn kẽ
tại sao gradient là hướng hạ dốc nhất, và tại sao lan truyền ngược
có công thức như vậy. Giải tích tích phân được thảo luận tiếp sau đó ở mức cần
thiết để hỗ trợ chủ đề tiếp theo -- lý thuyết xác suất.

<!--
Problems encountered in practice frequently are not certain, and thus we need a language to speak about uncertain things.  We review the theory of random variables and the most commonly encountered distributions so we may discuss models probabilistically.  This provides the foundation for the naive Bayes classifier, a probabilistic classification technique.
-->

Các vấn đề gặp phải trên thực tế thường không chắc chắn, và bởi vậy chúng ta cần
một ngôn ngữ để nói về những điều không chắc chắn. Chúng ta sẽ ôn tập lại lý
thuyết biến ngẫu nhiên và những phân phối thường gặp nhất để có thể
thảo luận các mô hình dưới góc nhìn xác suất. Việc này cung cấp nền tảng cho bộ phân loại
Naive Bayes, một phương pháp phân loại dựa trên xác suất.

<!--
Closely related to probability theory is the study of statistics.  While statistics is far too large a field to do justice in a short section, we will introduce fundamental concepts that all machine learning practitioners should be aware of, in particular: evaluating and comparing estimators, conducting hypothesis tests, and constructing confidence intervals.
-->

Liên quan mật thiết tới lý thuyết xác suất là lý thuyết thống kê. Trong khi
thống kê là một mảng quá lớn để ôn tập trong một mục ngắn, chúng tôi sẽ giới
thiệu các khái niệm cơ bản mà mọi người làm học máy cần biết, cụ thể: đánh giá
và so sánh các bộ ước lượng, thực hiện kiểm chứng thống kê,
và xây dựng khoảng tin cậy.

<!--
Last, we turn to the topic of information theory, which is the mathematical study of information storage and transmission.  This provides the core language by which we may discuss quantitatively how much information a model holds on a domain of discourse.
-->

Cuối cùng, chúng ta sẽ thảo luận chủ đề lý thuyết thông tin qua nghiên cứu toán
học về lưu trữ và truyền tải thông tin. Phần này cung cấp ngôn ngữ cơ bản ở đó
chúng ta thảo luận một cách định lượng lượng thông tin một mô hình hàm chứa.

<!--
Taken together, these form the core of the mathematical concepts needed to begin down the path towards a deep understanding of deep learning.
-->

Kết hợp lại, những kiến thức này định hình những khái niệm toán học cốt lõi cần
thiết để bắt đầu đi tới con đường hiểu sâu về học sâu.

```toc
:maxdepth: 2

geometry-linear-algebric-ops
eigen-decomposition
single-variable-calculus
multivariable-calculus
integral-calculus
random-variables
maximum-likelihood
distributions
naive-bayes
statistics
information-theory
```

<!===================== Kết thúc dịch Phần 1 ==================== -->

### Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Mỗi tên chỉ xuất hiện một lần: Nếu bạn đã dịch hoặc review phần 1 của trang này
thì không cần điền vào các phần sau nữa.
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.
-->

<!-- Phần 1 -->
* Vũ Hữu Tiệp
* Lê Khắc Hồng Phúc
