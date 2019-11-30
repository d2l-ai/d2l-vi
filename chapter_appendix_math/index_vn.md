<!===================== Bắt đầu dịch Phần 1 ==================== -->

<!--
# Appendix: Mathematics for Deep Learning
-->

# *dịch tiêu đề phía trên*
:label:`chap_appendix_math`

<!--
**Brent Werness** (*Amazon*), **Rachel Hu** (*Amazon*), and authors of this book
-->

*dịch đoạn phía trên*


<!--
One of the wonderful parts of modern deep learning is the fact that much of it can be understood and used without a full understanding of the mathematics below it.  This is a sign that the field is maturing.  Just as most software developers no longer need to worry about the theory of computable functions, neither should deep learning practitioners need to worry about the theoretical foundations of maximum likelihood learning.
-->

*dịch đoạn phía trên*

<!--
But, we are not quite there yet.
-->

*dịch đoạn phía trên*

<!--
In practice, you will sometimes need to understand how architectural choices influence gradient flow, or the implicit assumptions you make by training with a certain loss function.  You might need to know what in the world entropy measures, and how it can help you understand exactly what bits-per-character means in your model.  These all require deeper mathematical understanding.
-->

*dịch đoạn phía trên*

<!--
This appendix aims to provide you the mathematical background you need to understand the core theory of modern deep learning, but it is not exhaustive.  We will begin with examining linear algebra in greater depth.  We develop a geometric understanding of all the common linear algebraic objects and operations that will enable us to visualize the effects of various transformations on our data.  A key element is the development of the basics of eigen-decompositions.
-->

*dịch đoạn phía trên*

<!--
We next develop the theory of differential calculus to the point that we can fully understand why the gradient is the direction of steepest descent, and why back-propagation takes the form it does.  Integral calculus is then discussed to the degree needed to support our next topic, probability theory.
-->

*dịch đoạn phía trên*

<!--
Problems encountered in practice frequently are not certain, and thus we need a language to speak about uncertain things.  We review the theory of random variables and the most commonly encountered distributions so we may discuss models probabilistically.  This provides the foundation for the naive Bayes classifier, a probabilistic classification technique.
-->

*dịch đoạn phía trên*

<!--
Closely related to probability theory is the study of statistics.  While statistics is far too large a field to do justice in a short section, we will introduce fundamental concepts that all machine learning practitioners should be aware of, in particular: evaluating and comparing estimators, conducting hypothesis tests, and constructing confidence intervals.
-->

*dịch đoạn phía trên*

<!--
Last, we turn to the topic of information theory, which is the mathematical study of information storage and transmission.  This provides the core language by which we may discuss quantitatively how much information a model holds on a domain of discourse.
-->

*dịch đoạn phía trên*

<!--
Taken together, these form the core of the mathematical concepts needed to begin down the path towards a deep understanding of deep learning.
-->

*dịch đoạn phía trên*

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
*

<!-- Phần 2 -->
*

<!-- Phần 3 -->
*

<!-- Phần 4 -->
*

<!-- Phần 5 -->
*

<!-- Phần 6 -->
*
