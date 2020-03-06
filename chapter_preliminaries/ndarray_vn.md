<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Data Manipulation
-->

# Thao tác với Dữ liệu
:label:`sec_ndarray`

<!--
In order to get anything done, we need some way to store and manipulate data.
Generally, there are two important things we need to do with data: (i) acquire them; and (ii) process them once they are inside the computer.
There is no point in acquiring data without some way to store it, so let's get our hands dirty first by playing with synthetic data.
To start, we introduce the $n$-dimensional array (`ndarray`), MXNet's primary tool for storing and transforming data.
In MXNet, `ndarray` is a class and we call any instance "an `ndarray`".
-->

Muốn thực hiện bất cứ điều gì, chúng ta đều cần một cách nào đó để lưu trữ và thao tác với dữ liệu.
Thường sẽ có hai điều quan trọng chúng ta cần làm với dữ liệu: (i) thu thập; và (ii) xử lý sau khi đã có dữ liệu trên máy tính.
Sẽ thật vô nghĩa khi thu thập dữ liệu mà không có cách để lưu trữ nó, vậy nên trước tiên hãy cùng làm quen với dữ liệu tổng hợp.
Để bắt đầu, chúng tôi giới thiệu mảng $n$ chiều (`ndarray`) -- công cụ chính trong MXNET để lưu trữ và biến đổi dữ liệu.
Trong MXNet, `ndarray` là một lớp và mỗi thực thể của lớp đó là "một `ndarray`".

<!--
If you have worked with NumPy, the most widely-used scientific computing package in Python, then you will find this section familiar.
That's by design. We designed MXNet's `ndarray` to be an extension to NumPy's `ndarray` with a few killer features.
First, MXNet's `ndarray` supports asynchronous computation on CPU, GPU, and distributed cloud architectures, whereas NumPy only supports CPU computation.
Second, MXNet's `ndarray` supports automatic differentiation.
These properties make MXNet's `ndarray` suitable for deep learning.
Throughout the book, when we say `ndarray`, we are referring to MXNet's `ndarray` unless otherwise stated.
-->

Nếu bạn từng làm việc với NumPy, gói tính toán phổ biến nhất trong Python, bạn sẽ thấy mục này quen thuộc.
Việc này là có chủ đích.
Chúng tôi thiết kế `ndarray` trong MXNet là một dạng mở rộng của `ndarray` trong NumPy với một vài tính năng đặc biệt.
Thứ nhất, `ndarray` trong MXNet hỗ trợ tính toán phi đồng bộ <!-- TODO --> trên CPU, GPU, và các kiến trúc phân tán đám mây, trong khi NumPy chỉ hỗ trợ tính toán trên CPU.
Thứ hai, `ndaray` trong MXNet hỗ trợ tính vi phân tự động.
Những tính chất này khiến `ndarray` của MXNet phù hợp với học sâu.
Thông qua cuốn sách, nếu không nói gì thêm, chúng ta ngầm hiểu `ndarray` là `ndarray` của MXNet.

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Getting Started
-->

## Bắt đầu

<!--
In this section, we aim to get you up and running, equipping you with the the basic math and numerical computing tools that you will build on as you progress through the book.
Do not worry if you struggle to grok some of the mathematical concepts or library functions.
The following sections will revisit this material in the context practical examples and it will sink.
On the other hand, if you already have some background and want to go deeper into the mathematical content, just skip this section.
-->

Trong mục này, mục tiêu của chúng tôi là trang bị cho bạn các kiến thức toán cơ bản và cài đặt các công cụ tính toán mà bạn sẽ xây dựng dựa trên nó xuyên suốt cuốn sách này.
Đừng lo nếu bạn gặp khó khăn với các khái niệm toán khó hiểu hoặc các hàm trong thư viện tính toán.
Các mục tiếp theo sẽ nhắc lại những khái niệm này trong từng ngữ cảnh kèm theo ví dụ thực tiễn.
Mặt khác, nếu bạn đã có kiến thức nền tảng và muốn đi sâu hơn vào các nội dung toán, bạn có thể bỏ qua mục này.

<!--
To start, we import the `np` (`numpy`) and `npx` (`numpy_extension`) modules from MXNet.
Here, the `np` module includes functions supported by NumPy, while the `npx` module contains a set of extensions developed to empower deep learning within a NumPy-like environment.
When using `ndarray`, we almost always invoke the `set_np` function: this is for compatibility of `ndarray` processing by other components of MXNet.
-->

Để bắt đầu, ta cần khai báo mô-đun `np` (`numpy`) và `npx` (`numpy_extension`) từ MXNet.
Ở đây, mô-đun `np` bao gồm các hàm hỗ trợ bởi NumPy, trong khi mô-đun `npx` chứa một tập các hàm mở rộng được phát triển để hỗ trợ học sâu trong một môi trường giống với NumPy.
Khi sử dụng `ndarray`, ta luôn cần gọi hàm `set_np`: điều này nhằm đảm bảo sự tương thích của việc xử lý `ndarray` bằng các thành phần khác của MXNet.

```{.python .input  n=1}
from mxnet import np, npx
npx.set_np()
```

<!--
An `ndarray` represents a (possibly multi-dimensional) array of numerical values.
With one axis, an `ndarray` corresponds (in math) to a *vector*.
With two axes, an `ndarray` corresponds to a *matrix*.
Arrays with more than two axes do not have special mathematical names---we simply call them *tensors*.
-->

Một `ndarray` biểu diễn một mảng (có thể đa chiều) các giá trị số.
Với một trục, một `ndarray` tương ứng (trong toán) với một *vector*.
Với hai trục, một `ndarray` tương ứng với một *ma trận*.
Các mảng với nhiều hơn hai trục không có tên toán học cụ thể--chúng được gọi chung là *tensor*.

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
To start, we can use `arange` to create a row vector `x` containing the first $12$ integers starting with $0$, though they are created as floats by default.
Each of the values in an `ndarray` is called an *element* of the `ndarray`.
For instance, there are $12$ elements in the `ndarray` `x`.
Unless otherwise specified, a new `ndarray` will be stored in main memory and designated for CPU-based computation.
-->

Để bắt đầu, chúng ta sử dụng `arange` để tạo một vector hàng `x` chứa $12$ số nguyên đầu tiên bắt đầu từ $0$, nhưng được khởi tạo mặc định dưới dạng số thực.
Mỗi giá trị trong một `ndarray` được gọi là một *phần tử* của `ndarray` đó.
Như vậy, có $12$ phần tử trong `ndarray` `x`.
Nếu không nói gì thêm, một `ndarray` mới sẽ được lưu trong bộ nhớ chính và được tính toán trên CPU.

```{.python .input  n=2}
x = np.arange(12)
x
```

<!--
We can access an `ndarray`'s *shape* (the length along each axis) by inspecting its `shape` property.
-->

Chúng ta có thể lấy *kích thước* (độ dài theo mỗi trục) của `ndarray` bằng thuộc tính `shape`.

```{.python .input  n=3}
x.shape
```

<!--
If we just want to know the total number of elements in an `ndarray`, i.e., the product of all of the shape elements, we can inspect its `size` property.
Because we are dealing with a vector here, the single element of its `shape` is identical to its `size`.
-->

Nếu chỉ muốn biết tổng số phần tử của một `ndarray`, nghĩa là tích của tất cả các thành phần trong `shape`, ta có thể sử dụng thuộc tính `size`.
Vì ta đang làm việc với một vector, cả `shape` và `size` của nó đều chứa cùng một phần tử duy nhất.

```{.python .input  n=4}
x.size
```

<!--
To change the shape of an `ndarray` without altering either the number of elements or their values, we can invoke the `reshape` function.
For example, we can transform our `ndarray`, `x`, from a row vector with shape ($12$,) to a matrix with shape ($3$, $4$).
This new `ndarray` contains the exact same values, but views them as a matrix organized as $3$ rows and $4$ columns.
To reiterate, although the shape has changed, the elements in `x` have not.
Note that the `size` is unaltered by reshaping.
-->

Để thay đổi kích thước của một `ndarray` mà không làm thay đổi số lượng phần tử cũng như giá trị của chúng, ta có thể gọi hàm `reshape`.
Ví dụ, ta có thể biến đổi `ndarray` `x` trong ví dụ trên, từ một vector hàng với kích thước ($12$,) sang một ma trận với kích thước ($3$, $4$).
`ndarray` mới này chứa $12$ phần tử y hệt, nhưng được xem như một ma trận với $3$ hàng và $4$ cột.
Mặc dù kích thước thay đổi, các phần tử của `x` vẫn giữ nguyên.
Chú ý rằng `size` giữ nguyên khi thay đổi kích thước.

```{.python .input  n=5}
x = x.reshape(3, 4)
x
```
<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
Reshaping by manually specifying every dimension is unnecessary.
If our target shape is a matrix with shape (height, width), then after we know the width, the height is given implicitly.
Why should we have to perform the division ourselves?
In the example above, to get a matrix with $3$ rows, we specified both that it should have $3$ rows and $4$ columns.
Fortunately, `ndarray` can automatically work out one dimension given the rest.
We invoke this capability by placing `-1` for the dimension that we would like `ndarray` to automatically infer.
In our case, instead of calling `x.reshape(3, 4)`, we could have equivalently called `x.reshape(-1, 4)` or `x.reshape(3, -1)`.
-->

Việc chỉ định cụ thể mọi chiều khi thay đổi kích thước là không cần thiết.
Nếu kích thước mong muốn là một ma trận với kích thước (chiều_cao, chiều_rộng), thì sau khi biết chiều_rộng, chiều_cao có thể được ngầm suy ra.
Tại sao ta lại cần phải tự làm phép tính chia?
Trong ví dụ trên, để có được một ma trận với $3$ hàng, chúng ta phải chỉ định rõ rằng nó có $3$ hàng và $4$ cột.
May mắn thay, `ndarray` có thể tự động tính một chiều từ các chiều còn lại.
Ta có thể dùng chức năng này bằng cách đặt `-1` cho chiều mà ta muốn `ndarray` tự suy ra.
Trong trường hợp vừa rồi, thay vì gọi `x.reshape(3, 4)`, ta có thể gọi `x.reshape(-1, 4)` hoặc `x.reshape(3, -1)`.

<!--
The `empty` method grabs a chunk of memory and hands us back a matrix without bothering to change the value of any of its entries.
This is remarkably efficient but we must be careful because the entries might take arbitrary values, including very big ones!
-->

Phương thức `empty` lấy một đoạn bộ nhớ và trả về một ma trận mà không thay đổi các giá trị sẵn có tại đoạn bộ nhớ đó.
Việc này có hiệu quả tính toán đáng kể nhưng ta phải cẩn trọng bởi các phần tử đó có thể chứa bất kỳ giá trị nào, kể cả các số rất lớn!

```{.python .input  n=6}
np.empty((3, 4))
```

<!--
Typically, we will want our matrices initialized either with zeros, ones, some other constants, or numbers randomly sampled from a specific distribution.
We can create an `ndarray` representing a tensor with all elements set to $0$ and a shape of ($2$, $3$, $4$) as follows:
-->

Thông thường ta muốn khởi tạo các ma trận với các giá trị bằng không, bằng một, bằng hằng số nào đó hoặc bằng các mẫu ngẫu nhiên lấy từ một phân phối cụ thể.
Ta có thể tạo một `ndarray` biểu diễn một tensor với tất cả các phần tử bằng $0$ và có kích thước ($2$, $3$, $4$) như sau:

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

```{.python .input  n=7}
np.zeros((2, 3, 4))
```

<!--
Similarly, we can create tensors with each element set to 1 as follows:
-->

Tương tự, ta có thể tạo các tensor với các phần tử bằng 1 như sau:

```{.python .input  n=8}
np.ones((2, 3, 4))
```

<!--
Often, we want to randomly sample the values for each element in an `ndarray` from some probability distribution.
For example, when we construct arrays to serve as parameters in a neural network, we will typically inititialize their values randomly.
The following snippet creates an `ndarray` with shape ($3$, $4$).
Each of its elements is randomly sampled from a standard Gaussian (normal) distribution with a mean of $0$ and a standard deviation of $1$.
-->

Ta thường muốn lấy mẫu ngẫu nhiên cho mỗi phần tử trong một `ndarray` từ một phân phối xác suất.
Ví dụ, khi xây dựng các mảng để chứa các tham số của một mạng nơ-ron, ta thường khởi tạo chúng với các giá trị ngẫu nhiên.
Đoạn mã dưới đây tạo một `ndarray` có kích thước ($3$, $4$) với các phần tử được lấy mẫu ngẫu nhiên từ một phân phối Gauss (phân phối chuẩn) với trung bình bằng $0$ và độ lệch chuẩn $1$.

```{.python .input  n=10}
np.random.normal(0, 1, size=(3, 4))
```

<!--
We can also specify the exact values for each element in the desired `ndarray` by supplying a Python list (or list of lists) containing the numerical values.
Here, the outermost list corresponds to axis $0$, and the inner list to axis $1$.
-->

Ta cũng có thể khởi tạo giá trị cụ thể cho mỗi phần tử trong `ndarray` mong muốn bằng cách đưa vào một mảng Python (hoặc mảng của mảng) chứa các giá trị số.
Ở đây, mảng ngoài cùng tương ứng với trục $0$, và mảng bên trong tương ứng với trục $1$.

```{.python .input  n=9}
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
## Operations
-->

## Phép toán

<!--
This book is not about software engineering.
Our interests are not limited to simply reading and writing data from/to arrays.
We want to perform mathematical operations on those arrays.
Some of the simplest and most useful operations are the *elementwise* operations.
These apply a standard scalar operation to each element of an array.
For functions that take two arrays as inputs, elementwise operations apply some standard binary operator on each pair of corresponding elements from the two arrays. We can create an elementwise function from any function that maps from a scalar to a scalar.
-->

Cuốn sách này không nói về kỹ thuật phần mềm.
Chúng tôi không chỉ hứng thú với việc đơn giản đọc và ghi dữ liệu vào/từ các mảng mà còn muốn thực hiện các phép toán trên các mảng này.
Một vài phép toán đơn giản và hữu ích nhất là các phép toán tác động lên *từng phần tử* (*elementwise*).
Các phép toán này hoạt động như những phép toán chuẩn trên số vô hướng áp dụng lên từng phần tử của mảng.
Với những hàm nhận hai mảng đầu vào, phép toán theo từng thành phần được áp dụng trên từng cặp phần tử tương ứng của hai mảng.
Ta có thể tạo một hàm theo từng phần tử từ một hàm bất kỳ ánh xạ từ một số vô hướng tới một số vô hướng.

<!--
In mathematical notation, we would denote such a *unary* scalar operator (taking one input) by the signature $f: \mathbb{R} \rightarrow \mathbb{R}$.
This just mean that the function is mapping from any real number ($\mathbb{R}$) onto another.
Likewise, we denote a *binary* scalar operator (taking two real inputs, and yielding one output) by the signature $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$.
Given any two vectors $\mathbf{u}$ and $\mathbf{v}$ *of the same shape*, and a binary operator $f$, we can produce a vector $\mathbf{c} = F(\mathbf{u},\mathbf{v})$ by setting $c_i \gets f(u_i, v_i)$ for all $i$, where $c_i, u_i$, and $v_i$ are the $i^\mathrm{th}$ elements of vectors $\mathbf{c}, \mathbf{u}$, and $\mathbf{v}$.
Here, we produced the vector-valued $F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$ by *lifting* the scalar function to an elementwise vector operation.
-->

Trong toán học, ta ký hiệu một toán tử *đơn ngôi* vô hướng (lấy một đầu vào) bởi $f: \mathbb{R} \rightarrow \mathbb{R}$.
Điều này nghĩa là hàm số ánh xạ từ một số thực bất kỳ ($\mathbb{R}$) sang một số thực khác.
Tương tự, ta ký hiệu một toán tử *hai ngôi* vô hướng (lấy hai đầu vào, trả về một đầu ra) bởi $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$.
Cho trước hai vector bất kỳ $\mathbf{u}$ và $\mathbf{v}$ *với cùng kích thước*, và một toán tử hai ngôi $f$, ta có thể tính được một vector $\mathbf{c} = F(\mathbf{u},\mathbf{v})$ bằng cách tính $c_i \gets f(u_i, v_i)$ cho mọi $i$ với $c_i, u_i$, và $v_i$ là các phần tử thứ $i$ của vector $\mathbf{c}, \mathbf{u}$, và $\mathbf{v}$.
Ở đây, chúng ta tạo một hàm trả về vector $F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$ bằng cách áp dụng hàm $f$ lên từng phần tử.

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 7 ===================== -->

<!--
In MXNet, the common standard arithmetic operators (`+`, `-`, `*`, `/`, and `**`) have all been *lifted* to elementwise operations for any identically-shaped tensors of arbitrary shape.
We can call elementwise operations on any two tensors of the same shape.
In the following example, we use commas to formulate a $5$-element tuple, where each element is the result of an elementwise operation.
-->

Trong MXNet, các phép toán tiêu chuẩn (`+`, `-`, `*`, `/`, và `**`) là các phép toán theo từng phần tử trên các tensor đồng kích thước bất kỳ.
Ta có thể gọi những phép toán theo từng phần tử lên hai tensor đồng kích thước.
Trong ví dụ dưới đây, các dấu phẩy được sử dụng để tạo một tuple $5$ phần tử với mỗi phần tử là kết quả của một phép toán theo từng phần tử.

```{.python .input  n=11}
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

<!--
Many more operations can be applied elementwise, including unary operators like exponentiation.
-->

Rất nhiều các phép toán khác có thể được áp dụng theo từng phần tử, bao gồm các phép toán đơn ngôi như hàm mũ cơ số $e$.

```{.python .input  n=12}
np.exp(x)
```

<!--
In addition to elementwise computations, we can also perform linear algebra operations, including vector dot products and matrix multiplication.
We will explain the crucial bits of linear algebra (with no assumed prior knowledge) in :numref:`sec_linear-algebra`.
-->

Ngoài các phép tính theo từng phần tử, ta cũng có thể thực hiện các phép toán đại số tuyến tính, bao gồm tích vô hướng của hai vector và phép nhân ma trận.
Chúng ta sẽ giải thích những điểm quan trọng của đại số tuyến tính (mà không cần kiến thức nền tảng) trong :numref:`sec_linear-algebra`.

<!-- ===================== Kết thúc dịch Phần 7 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 8 ===================== -->

<!--
We can also *concatenate* multiple `ndarray`s together, stacking them end-to-end to form a larger `ndarray`.
We just need to provide a list of `ndarray`s and tell the system along which axis to concatenate.
The example below shows what happens when we concatenate two matrices along rows (axis $0$, the first element of the shape) vs. columns (axis $1$, the second element of the shape).
We can see that, the first output `ndarray`'s axis-$0$ length ($6$) is the sum of the two input `ndarray`s' axis-$0$ lengths ($3 + 3$); while the second output `ndarray`'s axis-$1$ length ($8$) is the sum of the two input `ndarray`s' axis-$1$ lengths ($4 + 4$).
-->

Ta cũng có thể *nối* nhiều `ndarray` với nhau, xếp chồng chúng lên nhau để tạo ra một `ndarray` lớn hơn.
Ta chỉ cần cung cấp một danh sách các `ndarray` và khai báo chúng được nối theo trục nào.
Ví dụ dưới đây thể hiện cách nối hai ma trận theo hàng (trục $0$, phần tử đầu tiên của kích thước) và theo cột (trục $1$, phần tử thứ hai của kích thước).
Ta có thể thấy rằng, cách thứ nhất tạo một `ndarray` với độ dài trục $0$ ($6$) bằng tổng các độ dài trục $0$ của hai `ndarray` đầu vào ($3 + 3$);
trong khi cách thứ hai tạo một `ndarray` với độ dài trục $1$ ($8$) bằng tổng các độ dài trục $1$ của hai `ndarray` đầu vào ($4 + 4$).

```{.python .input  n=14}
x = np.arange(12).reshape(3, 4)
y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([x, y], axis=0), np.concatenate([x, y], axis=1)
```

<!--
Sometimes, we want to construct a binary `ndarray` via *logical statements*.
Take `x == y` as an example.
For each position, if `x` and `y` are equal at that position, the corresponding entry in the new `ndarray` takes a value of $1$, meaning that the logical statement `x == y` is true at that position; otherwise that position takes $0$.
-->

Đôi khi, ta muốn tạo một `ndarray` nhị phân thông qua các *mệnh đề logic*.
Lấy `x == y` làm ví dụ.
Với mỗi vị trí, nếu giá trị của`x` và `y` tại vị trí đó bằng nhau thì phần tử tương ứng trong `ndarray` mới lấy giá trị $1$, nghĩa là mệnh đề logic `x == y` là đúng tại vị trí đó; ngược lại vị trí đó lấy giá trị $0$.

```{.python .input  n=15}
x == y
```

<!--
Summing all the elements in the `ndarray` yields an `ndarray` with only one element.
-->

Lấy tổng mọi phần tử trong một `ndarray` tạo ra một `ndarray` với chỉ một phần tử.

```{.python .input  n=16}
x.sum()
```

<!--
For stylistic convenience, we can write `x.sum()`as `np.sum(x)`.
-->

Ta cũng có thể thay `x.sum()` bởi `np.sum(x)`.

<!-- ===================== Kết thúc dịch Phần 8 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 9 ===================== -->

<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 4 - BẮT ĐẦU ===================================-->

<!--
## Broadcasting Mechanism
-->

## Cơ chế Lan truyền
<!-- bàn thêm từ này -->
<!--
In the above section, we saw how to perform elementwise operations on two `ndarray`s of the same shape. Under certain conditions, even when shapes differ, we can still perform elementwise operations by invoking the *broadcasting mechanism*.
These mechanisms work in the following way:
First, expand one or both arrays by copying elements appropriately so that after this transformation, the two `ndarray`s have the same shape.
Second, carry out the elementwise operations on the resulting arrays.
-->

Trong mục trên, ta đã thấy cách thực hiện các phép toán theo từng phần tử với hai `ndarray` đồng kích thước.
Trong những điều kiện nhất định, thậm chí khi kích thước khác nhau, ta vẫn có thể thực hiện các phép toán theo từng phần tử bằng cách sử dụng *cơ chế lan truyền* (_broadcasting mechanism_).
Cơ chế này hoạt động như sau:
Thứ nhất, mở rộng một hoặc cả hai mảng bằng cách lặp lại các phần tử một cách hợp lý sao cho sau phép biến đổi này, hai `ndarray` có cùng kích thước.
Thứ hai, thực hiện các phép toán theo từng phần tử với hai mảng mới này.

<!--
In most cases, we broadcast along an axis where an array initially only has length $1$, such as in the following example:
-->

Trong hầu hết các trường hợp, chúng ta lan truyền một mảng theo trục có độ dài ban đầu là $1$, như ví dụ dưới đây:

```{.python .input  n=17}
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

<!--
Since `a` and `b` are $3\times1$ and $1\times2$ matrices respectively, their shapes do not match up if we want to add them.
We *broadcast* the entries of both matrices into a larger $3\times2$ matrix as follows: for matrix `a` it replicates the columns and for matrix `b` it replicates the rows before adding up both elementwise.
-->

Vì `a` và `b` là các ma trận có kích thước lần lượt là $3\times1$ và $1\times2$, kích thước của chúng không khớp nếu ta muốn thực hiện phép cộng.
Ta *lan truyền* các phần tử của cả hai ma trận thành các ma trận $3\times2$ như sau: lặp lại các cột trong ma trận `a` và các hàng trong ma trận `b` trước khi cộng chúng theo từng phần tử.

```{.python .input  n=18}
a + b
```

<!-- ===================== Kết thúc dịch Phần 9 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 10 ===================== -->

<!--
## Indexing and Slicing
-->
## Chỉ số và Cắt chọn mảng

<!--
Just as in any other Python array, elements in an `ndarray` can be accessed by index.
As in any Python array, the first element has index $0$ and ranges are specified to include the first but *before* the last element.
As in standard Python lists, we can access elements according to their relative position to the end of the list by using negative indices.
-->

Cũng giống như trong bất kỳ mảng Python khác, các phần tử trong một `ndarray` có thể được truy cập theo chỉ số.
Tương tự, phần tử đầu tiên có chỉ số $0$ và khoảng được cắt chọn bao gồm phần tử đầu tiên nhưng *không tính* phần tử cuối cùng. <!-- người dịch tự sửa để tránh lặp từ -->
Và trong các danh sách Python tiêu chuẩn, chúng ta có thể truy cập các phần tử theo vị trí đếm ngược từ cuối danh sách bằng cách sử dụng các chỉ số âm.

<!--
Thus, `[-1]` selects the last element and `[1:3]` selects the second and the third elements as follows:
-->
Vì vậy, `[-1]` chọn phần tử cuối cùng và `[1:3]` chọn phần tử thứ hai và phần tử thứ ba như sau:

```{.python .input  n=19}
x[-1], x[1:3]
```

<!--
Beyond reading, we can also write elements of a matrix by specifying indices.
-->
Ngoài việc đọc, chúng ta cũng có thể viết các phần tử của ma trận bằng cách chỉ định các chỉ số.

```{.python .input  n=20}
x[1, 2] = 9
x
```

<!--
If we want to assign multiple elements the same value, we simply index all of them and then assign them the value.
For instance, `[0:2, :]` accesses the first and second rows, where `:` takes all the elements along axis $1$ (column).
While we discussed indexing for matrices, this obviously also works for vectors and for tensors of more than $2$ dimensions.
-->
Nếu chúng ta muốn gán cùng một giá trị cho nhiều phần tử, chúng ta chỉ cần trỏ đến tất cả các phần tử đó và gán giá trị cho chúng.
Chẳng hạn, `[0:2 ,:]` truy cập vào hàng thứ nhất và thứ hai, trong đó `:` lấy tất cả các phần tử dọc theo trục $1$ (cột).
Ở đây chúng ta đã thảo luận về cách truy cập vào ma trận, nhưng tất nhiên phương thức này cũng áp dụng cho các vector và tensor với nhiều hơn $2$ chiều.

```{.python .input  n=21}
x[0:2, :] = 12
x
```

<!-- ===================== Kết thúc dịch Phần 10 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 11 ===================== -->

<!-- ========================================= REVISE PHẦN 4 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 5 - BẮT ĐẦU ===================================-->

<!--
## Saving Memory
-->

## Tiết kiệm Bộ nhớ

<!--
In the previous example, every time we ran an operation, we allocated new memory to host its results.
For example, if we write `y = x + y`, we will dereference the `ndarray` that `y` used to point to and instead point `y` at the newly allocated memory.
In the following example, we demonstrate this with Python's `id()` function, which gives us the exact address of the referenced object in memory.
After running `y = y + x`, we will find that `id(y)` points to a different location.
That is because Python first evaluates `y + x`, allocating new memory for the result and then makes `y` point to this new location in memory.
-->

Ở ví dụ trước, mỗi khi chạy một phép tính, chúng ta sẽ cấp phát bộ nhớ mới để lưu trữ kết quả của lượt chạy đó. 
Cụ thể hơn, nếu viết `y = x + y`, ta sẽ ngừng tham chiếu đến `ndarray` mà `y` đã chỉ đến trước đó và thay vào đó gán `y` vào bộ nhớ được cấp phát mới.
Trong ví dụ tiếp theo, chúng ta sẽ minh họa việc này với hàm `id()` của Python - hàm cung cấp địa chỉ chính xác của một đối tượng được tham chiếu trong bộ nhớ. 
Sau khi chạy `y = y + x`, chúng ta nhận ra rằng `id(y)` chỉ đến một địa chỉ khác. 
Đó là bởi vì Python trước hết sẽ tính `y + x`, cấp phát bộ nhớ mới cho kết quả trả về và gán `y` vào địa chỉ mới này trong bộ nhớ.

```{.python .input  n=22}
before = id(y)
y = y + x
id(y) == before
```

<!--
This might be undesirable for two reasons.
First, we do not want to run around allocating memory unnecessarily all the time.
In machine learning, we might have hundreds of megabytes of parameters and update all of them multiple times per second.
Typically, we will want to perform these updates *in place*.
Second, we might point at the same parameters from multiple variables.
If we do not update in place, this could cause that discarded memory is not released, and make it possible for parts of our code to inadvertently reference stale parameters.
-->

Đây có thể là điều không mong muốn vì hai lý do.
Thứ nhất, không phải lúc nào chúng ta cũng muốn cấp phát bộ nhớ không cần thiết.
Trong học máy, ta có thể có đến hàng trăm megabytes tham số và cập nhật tất cả chúng nhiều lần mỗi giây, và thường thì ta muốn thực thi các cập nhật này *tại chỗ*.
Thứ hai, chúng ta có thể trỏ đến cùng tham số từ nhiều biến khác nhau.
Nếu không cập nhật tại chỗ, các bộ nhớ đã bị loại bỏ sẽ không được giải phóng, dẫn đến khả năng một số chỗ trong mã nguồn sẽ vô tình tham chiếu lại các tham số cũ. 

<!--
Fortunately, performing in-place operations in MXNet is easy.
We can assign the result of an operation to a previously allocated array with slice notation, e.g., `y[:] = <expression>`.
To illustrate this concept, we first create a new matrix `z` with the same shape as another `y`, using `zeros_like` to allocate a block of $0$ entries.
-->

May mắn thay, ta có thể dễ dàng thực hiện các phép tính tại chỗ với MXNet.
Chúng ta có thể gán kết quả của một phép tính cho một mảng đã được cấp phát trước đó bằng ký hiệu cắt chọn (*slice notation*), ví dụ, `y[:] = <expression>`. 
Để minh họa khái niệm này, đầu tiên chúng ta tạo một ma trận mới `z` có cùng kích thước với ma trận `y`, sử dụng `zeros_like` để gán giá trị khởi tạo bằng $0$. 

```{.python .input  n=23}
z = np.zeros_like(y)
print('id(z):', id(z))
z[:] = x + y
print('id(z):', id(z))
```

<!--
If the value of `x` is not reused in subsequent computations, we can also use `x[:] = x + y` or `x += y` to reduce the memory overhead of the operation.
-->

Nếu các tính toán tiếp theo không tái sử dụng giá trị của `x`, chúng ta có thể viết `x[:] = x + y` hoặc `x += y` để giảm thiểu việc sử dụng bộ nhớ không cần thiết trong quá trình tính toán.

```{.python .input  n=24}
before = id(x)
x += y
id(x) == before
```

<!-- ===================== Kết thúc dịch Phần 11 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 12 ===================== -->

<!-- ========================================= REVISE PHẦN 5 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 6 - BẮT ĐẦU ===================================-->

<!--
## Conversion to Other Python Objects
-->

## Chuyển đổi sang các Đối Tượng Python Khác

<!--
Converting an MXNet `ndarray` to a NumPy `ndarray`, or vice versa, is easy.
The converted result does not share memory.
This minor inconvenience is actually quite important: when you perform operations on the CPU or on GPUs, you do not want MXNet to halt computation, waiting to see whether the NumPy package of Python might want to be doing something else with the same chunk of memory.
The `array` and `asnumpy` functions do the trick.
-->

Chuyển đổi một MXNet `ndarray` sang NumPy `ndarray` hoặc ngược lại là khá đơn giản.
Tuy nhiên, kết quả của phép chuyển đổi này không chia sẻ bộ nhớ với đối tượng cũ.
Điểm bất tiện này tuy nhỏ nhưng lại khá quan trọng: khi bạn thực hiện các phép tính trên CPU hoặc GPUs, bạn không muốn MXNet dừng việc tính toán để chờ xem liệu gói Numpy của Python có sử dụng cùng bộ nhớ đó để làm việc khác không. 
Hàm `array` và `asnumpy` sẽ giúp bạn giải quyết vấn đề này. 

```{.python .input  n=25}
a = x.asnumpy()
b = np.array(a)
type(a), type(b)
```

<!--
To convert a size-$1$ `ndarray` to a Python scalar, we can invoke the `item` function or Python's built-in functions.
-->

Để chuyển đổi một mảng `ndarray` chứa một phần tử sang số vô hướng Python, ta có thể gọi hàm `item` hoặc các hàm có sẵn trong Python.

```{.python .input}
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

<!-- ===================== Kết thúc dịch Phần 12 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 13 ===================== -->

<!--
## Summary
-->

## Tổng kết

<!--
* MXNet's `ndarray` is an extension to NumPy's `ndarray` with a few killer advantages that make it suitable for deep learning.
* MXNet's `ndarray` provides a variety of functionalities including basic mathematics operations, broadcasting, indexing, slicing, memory saving, and conversion to other Python objects.
-->

* MXNet `ndarray` là phần mở rộng của NumPy `ndarray` với một số ưu thế vượt trội giúp cho nó phù hợp với học sâu. 
* MXNet `ndarray` cung cấp nhiều chức năng bao gồm các phép toán cơ bản, cơ chế lan truyền (*broadcasting*), chỉ số (*indexing*), cắt chọn (*slicing*), tiết kiệm bộ nhớ và khả năng chuyển đổi sang các đối tượng Python khác.

<!--
## Exercises
-->

## Bài tập

<!--
1. Run the code in this section. Change the conditional statement `x == y` in this section to `x < y` or `x > y`, and then see what kind of `ndarray` you can get.
2. Replace the two `ndarray`s that operate by element in the broadcasting mechanism with other shapes, e.g., three dimensional tensors. Is the result the same as expected?
-->

1. Chạy đoạn mã nguồn trong mục này. Thay đổi điều kiện mệnh đề `x == y` sang `x < y` hoặc `x > y`, sau đó kiểm tra dạng của `ndarray` nhận được.
2. Thay hai `ndarray` trong phép tính theo từng phần tử ở phần cơ chế lan truyền (*broadcasting mechanism*) với các `ndarray` có kích thước khác, ví dụ như tensor ba chiều. Kết quả có giống như bạn mong đợi hay không?

<!-- ===================== Kết thúc dịch Phần 13 ===================== -->

<!-- ========================================= REVISE PHẦN 6 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2316)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2315)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

<!--
![](../img/qr_ndarray.svg)
-->

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
* Vũ Hữu Tiệp
* Lê Khắc Hồng Phúc
* Phạm Hồng Vinh
* Trần Thị Hồng Hạnh
* Phạm Minh Đức
* Lê Đàm Hồng Lộc
* Nguyễn Lê Quang Nhật
