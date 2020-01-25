<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Data Manipulation
-->

# Xử lý Dữ liệu
:label:`sec_ndarray`

<!--
In order to get anything done, we need some way to store and manipulate data.
Generally, there are two important things we need to do with data: (i) acquire them; and (ii) process them once they are inside the computer.
There is no point in acquiring data without some way to store it, so let's get our hands dirty first by playing with synthetic data.
To start, we introduce the $n$-dimensional array (`ndarray`), MXNet's primary tool for storing and transforming data.
In MXNet, `ndarray` is a class and we call any instance "an `ndarray`".
-->

Để làm bất cứ điều gì, chúng ta cần một cách để lưu trữ và xử lý dữ liệu.
Thông thường, có ha điều quan trọng chúng ta cần làm với dữ liệu: (i) thu thập và (ii) xử lý sau khi đã có dữ liệu trên máy tính.
Thật vô lý khi thu thập dữ liệu mà không có cách để lưu trữ nó, vậy trước tiên hãy làm quen với dữ liệu tổng hợp.
Để bắt đầu, chúng tôi giới thiệu mảng $n$ chiều (`ndarray`) -- công cụ chính trong MXNET để lưu trữ và biến đổi dữ liệu.
Trong MXNet, `ndarray` là một lớp và một *instance* <!-- tìm xem các sách lập trình gọi instance là gì --> là "một `ndarray`".

<!--
If you have worked with NumPy, the most widely-used scientific computing package in Python, then you will find this section familiar.
That's by design. We designed MXNet's `ndarray` to be an extension to NumPy's `ndarray` with a few killer features.
First, MXNet's `ndarray` supports asynchronous computation on CPU, GPU, and distributed cloud architectures, whereas NumPy only supports CPU computation.
Second, MXNet's `ndarray` supports automatic differentiation.
These properties make MXNet's `ndarray` suitable for deep learning.
Throughout the book, when we say `ndarray`, we are referring to MXNet's `ndarray` unless otherwise stated.
-->

Nếu bạn từng làm việc với NumPy, gói tính toán phổ biến nhất trong Python, bạn sẽ thấy mục này quen thuộc.
Việc này có chủ đích.
Chúng tôi thiết kế các `ndarray` trong MXNet là một dạng mở rộng của các `ndarray` trong NumPy với một vài tính chất đặc biệt.
Thứ nhất, `ndarray` trong MXNet hỗ trợ tính toán phi đồng bộ <!-- TODO --> trên CPU, GPU, và các kiến trúc phân tán đám mây, trong khi NumPy chỉ hỗ trợ tính toán trên CPU.
Thứ hai, `ndaray` trong MXNet hỗ trợ tính đạo hàm tự động.
Những tính chất này khiến `ndarray` của MXNet phù hợp với học sâu.
Thông qua cuốn sách, nếu không nói gì thêm, chúng ta ngầm hiểu `ndarray` là `ndarray` của MXNet.

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Getting Started
-->

## *dịch tiêu đề phía trên*

<!--
In this section, we aim to get you up and running, equipping you with the the basic math and numerical computing tools that you will build on as you progress through the book.
Do not worry if you struggle to grok some of the mathematical concepts or library functions.
The following sections will revisit this material in the context practical examples and it will sink.
On the other hand, if you already have some background and want to go deeper into the mathematical content, just skip this section.
-->

*dịch đoạn phía trên*

<!--
To start, we import the `np` (`numpy`) and `npx` (`numpy_extension`) modules from MXNet.
Here, the `np` module includes functions supported by NumPy, while the `npx` module contains a set of extensions developed to empower deep learning within a NumPy-like environment.
When using `ndarray`, we almost always invoke the `set_np` function: this is for compatibility of `ndarray` processing by other components of MXNet.
-->

*dịch đoạn phía trên*

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

*dịch đoạn phía trên*

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

*dịch đoạn phía trên*

```{.python .input  n=2}
x = np.arange(12)
x
```

<!--
We can access an `ndarray`'s *shape* (the length along each axis) by inspecting its `shape` property.
-->

*dịch đoạn phía trên*

```{.python .input  n=3}
x.shape
```

<!--
If we just want to know the total number of elements in an `ndarray`, i.e., the product of all of the shape elements, we can inspect its `size` property.
Because we are dealing with a vector here, the single element of its `shape` is identical to its `size`.
-->

*dịch đoạn phía trên*

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

*dịch đoạn phía trên*

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

*dịch đoạn phía trên*

<!--
The `empty` method grabs a chunk of memory and hands us back a matrix without bothering to change the value of any of its entries.
This is remarkably efficient but we must be careful because the entries might take arbitrary values, including very big ones!
-->

*dịch đoạn phía trên*

```{.python .input  n=6}
np.empty((3, 4))
```

<!--
Typically, we will want our matrices initialized either with zeros, ones, some other constants, or numbers randomly sampled from a specific distribution.
We can create an `ndarray` representing a tensor with all elements set to $0$ and a shape of ($2$, $3$, $4$) as follows:
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

```{.python .input  n=7}
np.zeros((2, 3, 4))
```

<!--
Similarly, we can create tensors with each element set to 1 as follows:
-->

*dịch đoạn phía trên*

```{.python .input  n=8}
np.ones((2, 3, 4))
```

<!--
Often, we want to randomly sample the values for each element in an `ndarray` from some probability distribution.
For example, when we construct arrays to serve as parameters in a neural network, we will typically inititialize their values randomly.
The following snippet creates an `ndarray` with shape ($3$, $4$).
Each of its elements is randomly sampled from a standard Gaussian (normal) distribution with a mean of $0$ and a standard deviation of $1$.
-->

*dịch đoạn phía trên*

```{.python .input  n=10}
np.random.normal(0, 1, size=(3, 4))
```

<!--
We can also specify the exact values for each element in the desired `ndarray` by supplying a Python list (or list of lists) containing the numerical values.
Here, the outermost list corresponds to axis $0$, and the inner list to axis $1$.
-->

*dịch đoạn phía trên*

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

## *dịch tiêu đề phía trên*

<!--
This book is not about software engineering.
Our interests are not limited to simply reading and writing data from/to arrays.
We want to perform mathematical operations on those arrays.
Some of the simplest and most useful operations are the *elementwise* operations.
These apply a standard scalar operation to each element of an array.
For functions that take two arrays as inputs, elementwise operations apply some standard binary operator on each pair of corresponding elements from the two arrays. We can create an elementwise function from any function that maps from a scalar to a scalar.
-->

*dịch đoạn phía trên*

<!--
In mathematical notation, we would denote such a *unary* scalar operator (taking one input) by the signature $f: \mathbb{R} \rightarrow \mathbb{R}$.
This just mean that the function is mapping from any real number ($\mathbb{R}$) onto another.
Likewise, we denote a *binary* scalar operator (taking two real inputs, and yielding one output) by the signature $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$.
Given any two vectors $\mathbf{u}$ and $\mathbf{v}$ *of the same shape*, and a binary operator $f$, we can produce a vector $\mathbf{c} = F(\mathbf{u},\mathbf{v})$ by setting $c_i \gets f(u_i, v_i)$ for all $i$, where $c_i, u_i$, and $v_i$ are the $i^\mathrm{th}$ elements of vectors $\mathbf{c}, \mathbf{u}$, and $\mathbf{v}$.
Here, we produced the vector-valued $F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$ by *lifting* the scalar function to an elementwise vector operation.
-->

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 7 ===================== -->

*dịch đoạn phía trên*

<!--
In MXNet, the common standard arithmetic operators (`+`, `-`, `*`, `/`, and `**`) have all been *lifted* to elementwise operations for any identically-shaped tensors of arbitrary shape.
We can call elementwise operations on any two tensors of the same shape.
In the following example, we use commas to formulate a $5$-element tuple, where each element is the result of an elementwise operation.
-->

*dịch đoạn phía trên*

```{.python .input  n=11}
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

<!--
Many more operations can be applied elementwise, including unary operators like exponentiation.
-->

*dịch đoạn phía trên*

```{.python .input  n=12}
np.exp(x)
```

<!--
In addition to elementwise computations, we can also perform linear algebra operations, including vector dot products and matrix multiplication.
We will explain the crucial bits of linear algebra (with no assumed prior knowledge) in :numref:`sec_linear-algebra`.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 7 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 8 ===================== -->

<!--
We can also *concatenate* multiple `ndarray`s together, stacking them end-to-end to form a larger `ndarray`.
We just need to provide a list of `ndarray`s and tell the system along which axis to concatenate.
The example below shows what happens when we concatenate two matrices along rows (axis $0$, the first element of the shape) vs. columns (axis $1$, the second element of the shape).
We can see that, the first output `ndarray`'s axis-$0$ length ($6$) is the sum of the two input `ndarray`s' axis-$0$ lengths ($3 + 3$); while the second output `ndarray`'s axis-$1$ length ($8$) is the sum of the two input `ndarray`s' axis-$1$ lengths ($4 + 4$).
-->

*dịch đoạn phía trên*

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

*dịch đoạn phía trên*

```{.python .input  n=15}
x == y
```

<!--
Summing all the elements in the `ndarray` yields an `ndarray` with only one element.
-->

*dịch đoạn phía trên*

```{.python .input  n=16}
x.sum()
```

<!--
For stylistic convenience, we can write `x.sum()`as `np.sum(x)`.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 8 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 9 ===================== -->

<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 4 - BẮT ĐẦU ===================================-->

<!--
## Broadcasting Mechanism
-->

## *dịch tiêu đề phía trên*

<!--
In the above section, we saw how to perform elementwise operations on two `ndarray`s of the same shape. Under certain conditions, even when shapes differ, we can still perform elementwise operations by invoking the *broadcasting mechanism*.
These mechanisms work in the following way:
First, expand one or both arrays by copying elements appropriately so that after this transformation, the two `ndarray`s have the same shape.
Second, carry out the elementwise operations on the resulting arrays.
-->

*dịch đoạn phía trên*

<!--
In most cases, we broadcast along an axis where an array initially only has length $1$, such as in the following example:
-->

*dịch đoạn phía trên*

```{.python .input  n=17}
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

<!--
Since `a` and `b` are $3\times1$ and $1\times2$ matrices respectively, their shapes do not match up if we want to add them.
We *broadcast* the entries of both matrices into a larger $3\times2$ matrix as follows: for matrix `a` it replicates the columns and for matrix `b` it replicates the rows before adding up both elementwise.
-->

*dịch đoạn phía trên*

```{.python .input  n=18}
a + b
```

<!-- ===================== Kết thúc dịch Phần 9 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 10 ===================== -->

<!--
## Indexing and Slicing
-->

## *dịch tiêu đề phía trên*

<!--
Just as in any other Python array, elements in an `ndarray` can be accessed by index.
As in any Python array, the first element has index $0$ and ranges are specified to include the first but *before* the last element.
As in standard Python lists, we can access elements according to their relative position to the end of the list by using negative indices.
-->

*dịch đoạn phía trên*

<!--
Thus, `[-1]` selects the last element and `[1:3]` selects the second and the third elements as follows:
-->

*dịch đoạn phía trên*

```{.python .input  n=19}
x[-1], x[1:3]
```

<!--
Beyond reading, we can also write elements of a matrix by specifying indices.
-->

*dịch đoạn phía trên*

```{.python .input  n=20}
x[1, 2] = 9
x
```

<!--
If we want to assign multiple elements the same value, we simply index all of them and then assign them the value.
For instance, `[0:2, :]` accesses the first and second rows, where `:` takes all the elements along axis $1$ (column).
While we discussed indexing for matrices, this obviously also works for vectors and for tensors of more than $2$ dimensions.
-->

*dịch đoạn phía trên*

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

## *dịch tiêu đề phía trên*

<!--
In the previous example, every time we ran an operation, we allocated new memory to host its results.
For example, if we write `y = x + y`, we will dereference the `ndarray` that `y` used to point to and instead point `y` at the newly allocated memory.
In the following example, we demonstrate this with Python's `id()` function, which gives us the exact address of the referenced object in memory.
After running `y = y + x`, we will find that `id(y)` points to a different location.
That is because Python first evaluates `y + x`, allocating new memory for the result and then makes `y` point to this new location in memory.
-->

*dịch đoạn phía trên*

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

*dịch đoạn phía trên*

<!--
Fortunately, performing in-place operations in MXNet is easy.
We can assign the result of an operation to a previously allocated array with slice notation, e.g., `y[:] = <expression>`.
To illustrate this concept, we first create a new matrix `z` with the same shape as another `y`, using `zeros_like` to allocate a block of $0$ entries.
-->

*dịch đoạn phía trên*

```{.python .input  n=23}
z = np.zeros_like(y)
print('id(z):', id(z))
z[:] = x + y
print('id(z):', id(z))
```

<!--
If the value of `x` is not reused in subsequent computations, we can also use `x[:] = x + y` or `x += y` to reduce the memory overhead of the operation.
-->

*dịch đoạn phía trên*

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

## *dịch tiêu đề phía trên*

<!--
Converting an MXNet `ndarray` to a NumPy `ndarray`, or vice versa, is easy.
The converted result does not share memory.
This minor inconvenience is actually quite important: when you perform operations on the CPU or on GPUs, you do not want MXNet to halt computation, waiting to see whether the NumPy package of Python might want to be doing something else with the same chunk of memory.
The `array` and `asnumpy` functions do the trick.
-->

*dịch đoạn phía trên*

```{.python .input  n=25}
a = x.asnumpy()
b = np.array(a)
type(a), type(b)
```

<!--
To convert a size-$1$ `ndarray` to a Python scalar, we can invoke the `item` function or Python's built-in functions.
-->

*dịch đoạn phía trên*

```{.python .input}
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

<!-- ===================== Kết thúc dịch Phần 12 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 13 ===================== -->

<!--
## Summary
-->

## *dịch tiêu đề phía trên*

<!--
* MXNet's `ndarray` is an extension to NumPy's `ndarray` with a few killer advantages that make it suitable for deep learning.
* MXNet's `ndarray` provides a variety of functionalities including basic mathematics operations, broadcasting, indexing, slicing, memory saving, and conversion to other Python objects.
-->

*dịch đoạn phía trên*


<!--
## Exercises
-->

## *dịch tiêu đề phía trên*

<!--
1. Run the code in this section. Change the conditional statement `x == y` in this section to `x < y` or `x > y`, and then see what kind of `ndarray` you can get.
2. Replace the two `ndarray`s that operate by element in the broadcasting mechanism with other shapes, e.g., three dimensional tensors. Is the result the same as expected?
-->

*dịch đoạn phía trên*

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
* Vũ Hữu Tiệp

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

<!-- Phần 7 -->
*

<!-- Phần 8 -->
*

<!-- Phần 9 -->
*

<!-- Phần 10 -->
*

<!-- Phần 11 -->
*

<!-- Phần 12 -->
*

<!-- Phần 13 -->
*
