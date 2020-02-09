<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Documentation
-->

# Tài liệu

<!--
Due to constraints on the length of this book, we cannot possibly introduce every single MXNet function and class (and you probably would not want us to). The API documentation and additional tutorials and examples provide plenty of documentation beyond the book. In this section we provide you with some guidance to exploring the MXNet API.
-->

Vì độ dài cuốn sách này có giới hạn, chúng tôi không thể giới thiệu hết tất cả các hàm và lớp của MXNet (và tốt nhất nên như vậy). Tài liệu API, các hướng dẫn và ví dụ sẽ cung cấp nhiều thông tin vượt ra khỏi nội dung cuốn sách. Trong chương này, chúng tôi sẽ cung cấp một vài chỉ dẫn để bạn có thể khám phá MXNet API.

<!--
## Finding All the Functions and Classes in a Module
-->

## Tra cứu tất cả các hàm và lớp trong một Module

<!--
In order to know which functions and classes can be called in a module, we invoke the `dir` function. For instance, we can query all properties in the `np.random` module as follows:
-->

Để biết những hàm/lớp nào có thể được gọi trong một module, chúng ta dùng hàm `dir`. Ví dụ, ta có thể lấy tất cả thuộc tính của module `np.random` bằng cách:

```{.python .input  n=1}
from mxnet import np
print(dir(np.random))
```

<!--
Generally, we can ignore functions that start and end with `__` (special objects in Python) or functions that start with a single `_`(usually internal functions). Based on the remaining function or attribute names, we might hazard a guess that this module offers various methods for generating random numbers, including sampling from the uniform distribution (`uniform`), normal distribution (`normal`), and multinomial distribution  (`multinomial`).
-->

Thông thường, ta có thể bỏ qua những hàm bắt đầu và kết thúc với `__` (các đối tượng đặc biệt trong Python) hoặc những hàm bắt đầu bằng `_` (thường là các hàm địa phương). Dựa trên những hàm và thuộc tính còn lại, ta có thể dự đoán rằng module này cung cấp những phương thức sinh số ngẫu nhiên, bao gồm lấy mẫu từ phân phối đều liên tục (`uniform`), phân phối chuẩn (`normal`) và phân phối chuẩn nhiều chiều (`multinomial`)

<!--
## Finding the Usage of Specific Functions and Classes
-->

## Tra cứu cách sử dụng một hàm hoặc một lớp cụ thể

<!--
For more specific instructions on how to use a given function or class, we can invoke the  `help` function. As an example, let's explore the usage instructions for `ndarray`'s `ones_like` function.
-->

Để tra cứu chi tiết cách sử dụng một hàm hoặc lớp nhất định, ta dùng hàm `help`. Ví dụ, để tra cứu cách sử dụng hàm `ones_like` với `ndarray`:

```{.python .input}
help(np.ones_like)
```

<!--
From the documentation, we can see that the `ones_like` function creates a new array with the same shape as the supplied `ndarray` and sets all the elements to `1`. Whenever possible, you should run a quick test to confirm your interpretation:
-->

Từ tài liệu, ta có thể thấy hàm `ones_like` tạo một mảng cùng kích thước với `ndarray` có các phần tử được gán giá trị bằng `1`. Nếu có thể, bạn nên chạy thử để hiểu rõ hơn

```{.python .input}
x = np.array([[0, 0, 0], [2, 2, 2]])
np.ones_like(x)
```

<!--
In the Jupyter notebook, we can use `?` to display the document in another window. For example, `np.random.uniform?` will create content that is almost identical to `help(np.random.uniform)`, displaying it in a new browser window. In addition, if we use two question marks, such as `np.random.uniform??`, the code implementing the function will also be displayed.
-->

Trong Jupyter notebook, ta có thể dùng `?` để mở tài liệu trong một cửa sổ khác. Ví dụ, `np.random.uniform?` sẽ in ra nội dung y hệt `help(np.random.uniform)` trong một cửa sổ trình duyệt mới. Ngoài ra, nếu chúng ta dùng dấu `?` hai lần như `np.random.uniform??` thì đoạn mã định nghĩa hàm cũng sẽ được in ra.

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## API Documentation
-->

## *dịch tiêu đề phía trên*

<!--
For further details on the API details check the MXNet website at  [http://mxnet.apache.org/](http://mxnet.apache.org/). You can find the details under the appropriate headings (also for programming languages other than Python).
-->

*dịch đoạn phía trên*

<!--
## Summary
-->

## *dịch tiêu đề phía trên*

<!--
* The official documentation provides plenty of descriptions and examples that are beyond this book.
* We can look up documentation for the usage of MXNet API by calling the `dir` and `help` functions, or checking the MXNet website.
-->

*dịch đoạn phía trên*


<!--
## Exercises
-->

## *dịch tiêu đề phía trên*

<!--
1. Look up `ones_like` and `autograd` on the MXNet website.
2. What are all the possible outputs after running `np.random.choice(4, 2)`?
3. Can you rewrite `np.random.choice(4, 2)` by using the `np.random.randint` function?
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2322)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2322)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

### Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.
-->

* Đoàn Võ Duy Thanh
<!-- Phần 1 -->
* Trần Hoàng Quân

<!-- Phần 2 -->
*
