<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->

<!--
# Linear Algebra
-->

# Đại số tuyến tính
:label:`sec_linear-algebra`

<!--
Now that you can store and manipulate data, let's briefly review the subset of basic linear algebra that you will need to understand and implement most of models covered in this book.
Below, we introduce the basic mathematical objects, arithmetic, and operations in linear algebra, expressing each both through mathematical notation and the corresponding implementation in code.
-->

Bây giờ bạn đã có thể lưu trữ và xử lý dữ liệu, hãy cùng nhanh chóng ôn lại những kiến thức đại số tuyến tính cần thiết để hiểu và lập trình hầu hết các mô hình được nhắc tới trong quyển sách này.
Dưới đây, chúng ta giới thiệu các đối tượng toán học, số học, phép tính trong đại số tuyến tính, biểu diễn chúng dưới ký hiệu toán học và mã nguồn. 


<!--
## Scalars
-->

## Số vô hướng

<!--
If you never studied linear algebra or machine learning, then your past experience with math probably consisted of thinking about one number at a time.
And, if you ever balanced a checkbook or even paid for dinner at a restaurant then you already know how to do basic things like adding and multiplying pairs of numbers.
For example, the temperature in Palo Alto is $52$ degrees Fahrenheit.
Formally, we call values consisting of just one numerical quantity *scalars*.
If you wanted to convert this value to Celsius (the metric system's more sensible temperature scale), you would evaluate the expression $c = \frac{5}{9}(f - 32)$, setting $f$ to $52$.
In this equation, each of the terms---$5$, $9$, and $32$---are scalar values.
The placeholders $c$ and $f$ are called *variables* and they represented unknown scalar values.
-->

Nếu bạn chưa từng học đại số tuyến tính hay học máy, có lẽ bạn mới chỉ từng làm việc với từng con số riêng lẻ.
Nếu bạn đã từng phải cân bằng sổ chi tiêu hoặc chỉ đơn giản là trả tiền cho bữa ăn, bạn chắc chắn là đã biết cách thực hiện các phép tính cơ bản như cộng trừ nhân chia các cặp số.
Ví dụ, nhiệt độ tại Palo Alto là $52$ độ Fahrenheit.
Chúng ta gọi các giá trị mà chỉ bao gồm một số duy nhất là *vô hướng* (*scalar*).
Nếu bạn muốn chuyển giá trị nhiệt độ trên sang độ Celsius (thang đo nhiệt độ hợp lý hơn theo hệ mét), bạn sẽ phải tính biểu thức $c = \frac{5}{9}(f - 32)$ với giá trị $f$ bằng $52$.
Trong phương trình trên, mỗi số hạng --- $5$, $9$ và $32$ --- là các số vô hướng.
Các ký hiệu số hạng $c$ và $f$ được gọi là *biến* và chúng biễu diễn các giá trị số vô hướng chưa biết.

<!--
In this book, we adopt the mathematical notation where scalar variables are denoted by ordinary lower-cased letters (e.g., $x$, $y$, and $z$).
We denote the space of all (continuous) *real-valued* scalars by $\mathbb{R}$.
For expedience, we will punt on rigorous definitions of what precisely *space* is, but just remember for now that the expression $x \in \mathbb{R}$ is a formal way to say that $x$ is a real-valued scalar.
The symbol $\in$ can be pronounced "in" and simply denotes membership in a set.
Analogously, we could write $x, y \in \{0, 1\}$ to state that $x$ and $y$ are numbers whose value can only be $0$ or $1$.
-->

Trong quyển sách này, chúng ta sẽ theo quy ước ký hiệu các biến vô hướng bằng các chữ cái được viết thường (chẳng hạn $x$, $y$ và $z$).
Chúng ta ký hiệu không gian (liên tục) của tất cả các *số thực* vô hướng là $\mathbb{R}$.
Vì tính thiết thực, chúng ta sẽ bỏ qua việc định nghĩa chặt chẽ *không gian* là gì, bây giờ bạn chỉ cần nhớ biểu thức $x \in \mathbb{R}$ là một cách nói chính quy rằng $x$ là một số thực vô hướng.
Ký hiệu $\in$ đọc là "thuộc" và chỉ đơn giản là biểu diễn mối quan hệ của phần tử trong một tập hợp.
Tương tự, ta có thể viết $x, y \in \{0, 1\}$ để ký hiệu rằng $x$ và $y$ là các số mà chỉ có thể nhận giá trị $0$ hoặc $1$.

<!--
In MXNet code, a scalar is represented by an `ndarray` with just one element.
In the next snippet, we instantiate two scalars and perform some familiar arithmetic operations with them, namely addition, multiplication, division, and exponentiation.
-->

Trong mã nguồn MXNet, một số vô hướng được biễu diễn bằng `ndarray` với chỉ một phần tử.
Trong đoạn mã dưới đây, chúng ta khởi tạo hai số vô hướng và thực hiện các phép tính cộng, trừ, nhân, chia, lũy thừa quen thuộc với chúng.

```{.python .input  n=1}
from mxnet import np, npx
npx.set_np()

x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

<!-- =================== Kết thúc dịch Phần 1 ==================== -->

<!-- =================== Bắt đầu dịch Phần 2 ==================== -->

<!--
## Vectors
-->

## Vector

<!--
You can think of a vector as simply a list of scalar values.
We call these values the *elements* (*entries* or *components*) of the vector.
When our vectors represent examples from our dataset, their values hold some real-world significance.
For example, if we were training a model to predict the risk that a loan defaults, we might associate each applicant with a vector whose components correspond to their income, length of employment, number of previous defaults, and other factors.
If we were studying the risk of heart attacks hospital patients potentially face, we might represent each patient by a vector whose components capture their most recent vital signs, cholesterol levels, minutes of exercise per day, etc.
In math notation, we will usually denote vectors as bold-faced, lower-cased letters (e.g., $\mathbf{x}$, $\mathbf{y}$, and $\mathbf{z})$.
-->

Bạn có thể nghĩ vector đơn giản là một dãy các số vô hướng.
Chúng ta gọi các giá trị đó là *phần tử* (*thành phần*) của vector.
Khi các vector biễu diễn cho mẫu trong tập dữ liệu, các giá trị của nó mang theo những ý nghĩa từ thực tế.
Ví dụ, nếu chúng ta huấn luyện một mô hình dự đoán rủi ro vỡ nợ, chúng ta có thể gán cho mỗi ứng viên một vector với các thành phần tương ứng với thu nhập của họ, thời gian làm việc, số lần vỡ nợ trước đó và các yếu tố khác.
Nếu chúng ta đang học về rủi ro bị đau tim của bệnh nhân trong bệnh viện, ta có thể biểu diễn mỗi bệnh nhân bằng một vector với phần tử mang các thông tin về chỉ số sức khỏe, nồng độ cholesterol, số phút tập thể dục mỗi ngày, v.v.
Trong ký hiệu toán học, chúng ta thường ký hiệu vector bằng chữ cái đậm nét viết thường (ví dụ $\mathbf{x}$, $\mathbf{y}$, và $\mathbf{z})$.

<!--
In MXNet, we work with vectors via $1$-dimensional `ndarray`s.
In general `ndarray`s can have arbitrary lengths, subject to the memory limits of your machine.
-->

Trong MXNet, chúng ta làm việc với vector thông qua các `ndarray` $1$-chiều.
Tổng quát lên, `ndarray` có thể có chiều dài bất kỳ, giới hạn bởi bộ nhớ máy tính.

```{.python .input  n=2}
x = np.arange(4)
x
```

<!--
We can refer to any element of a vector by using a subscript.
For example, we can refer to the $i^\mathrm{th}$ element of $\mathbf{x}$ by $x_i$.
Note that the element $x_i$ is a scalar, so we do not bold-face the font when referring to it.
Extensive literature considers column vectors to be the default orientation of vectors, so does this book.
In math, a vector $\mathbf{x}$ can be written as
-->

Một phần tử bất kỳ trong vector có thể được ký hiệu sử dụng chỉ số dưới.
Ví dụ ta đề cập tới phần tử thứ $i$ của $\mathbf{x}$ bằng $x_i$.
Lưu ý rằng phần từ $x_i$ là một số vô hướng nên nó không cần được in đậm.
Có rất nhiều tài liệu tham khảo xem vector cột là chiều mặc định của vector, quyển sách này cũng vậy.
Trong toán học, một vector có thể được viết như sau

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\x_{2}  \\ \vdots  \\x_{n}\end{bmatrix},$$
:eqlabel:`eq_vec_def`


<!--
where $x_1, \ldots, x_n$ are elements of the vector.
In code, we access any element by indexing into the `ndarray`.
-->

trong đó $x_1, \ldots, x_n$ là các phần tử của vector.
Trong mã nguồn, chúng ta sử dụng chỉ số để truy cập tới các phần tử trong `ndarray`.

```{.python .input  n=3}
x[3]
```

<!-- =================== Kết thúc dịch Phần 2 ==================== -->

<!-- =================== Bắt đầu dịch Phần 3 ==================== -->

<!--
### Length, Dimensionality, and Shape
-->

### Độ dài, Chiều, và Kích thước

<!--
Let's revisit some concepts from :numref:`sec_ndarray`.
A vector is just an array of numbers.
And just as every array has a length, so does every vector.
In math notation, if we want to say that a vector $\mathbf{x}$ consists of $n$ real-valued scalars, we can express this as $\mathbf{x} \in \mathbb{R}^n$.
The length of a vector is commonly called the *dimension* of the vector.
-->

Hãy quay lại với những khái niệm từ :numref:`sec_ndarray`.
Một vector đơn thuần là một dãy các số.
Và mỗi dãy số đều có độ dài, mỗi vector cũng tương tự như vậy.
Trong ký hiệu toán học, nếu ta muốn nói rằng một vector $\mathbf{x}$ chứa $n$ các số thực vô hướng, ta có thể biểu diễn nó bằng $\mathbf{x} \in \mathbb{R}^n$.
Độ dài của một vector còn được gọi là số **chiều** của vector.

<!--
As with an ordinary Python array, we can access the length of an `ndarray` by calling Python's built-in `len()` function.
-->

Cũng giống như một dãy thông thường trong Python, chúng ta có thể xem độ dài của của một `ndarray` bằng cách gọi hàm `len()` có sẵn của Python. 

```{.python .input  n=4}
len(x)
```

<!--
When an `ndarray` represents a vector (with precisely one axis), we can also access its length via the `.shape` attribute.
The shape is a tuple that lists the length (dimensionality) along each axis of the `ndarray`.
For `ndarray`s with just one axis, the shape has just one element.
-->

Khi một `ndarray` biễu diễn một vector (với chính xác một trục), ta cũng có thể xem độ dài của nó qua thuộc tính `.shape` (kích thước).
Kích thước là một `tuple` liệt kê độ dài (số chiều) dọc theo mỗi trục của `ndarray`.
Với các `ndarray` có duy nhất một trục, kích thước của nó chỉ có một phần tử.

```{.python .input  n=5}
x.shape
```

<!--
Note that the word "dimension" tends to get overloaded in these contexts and this tends to confuse people.
To clarify, we use the dimensionality of a *vector* or an *axis* to refer to its length, i.e., the number of elements of a vector or an axis.
However, we use the dimensionality of an `ndarray` to refer to the number of axes that an `ndarray` has.
In this sense, the dimensionality of an `ndarray`'s some axis will be the length of that axis.
-->

Lưu ý rằng ở đây, từ "chiều" thường có rất nhiều nghĩa trong các ngữ cảnh và điều này thường khiến mọi người bị nhầm lẫn.
Để làm rõ, chúng ta dùng số chiều của một *vector* hoặc một *trục* để chỉ độ dài của nó, tức là số phần tử trong một vector, hay một trục.
Tuy nhiên, chúng ta sử dụng số chiều của một `ndarray` để chỉ số trục của `ndarray` đó.
Theo nghĩa này, chiều của một trục của một `ndarray` là độ dài của trục đó. 

<!-- =================== Kết thúc dịch Phần 3 ==================== -->

<!-- =================== Bắt đầu dịch Phần 4 ==================== -->

<!--
## Matrices
-->

## Ma trận

<!--
Just as vectors generalize scalars from order $0$ to order $1$, matrices generalize vectors from order $1$ to order $2$.
Matrices, which we will typically denote with bold-faced, capital letters (e.g., $\mathbf{X}$, $\mathbf{Y}$, and $\mathbf{Z}$), are represented in code as `ndarray`s with $2$ axes.
-->

Cũng giống như vector khái quát số vô hướng từ bậc $0$ sang bậc $1$, ma trận khái quát vector từ bậc $1$ sang bậc $2$.
Ma trận thường được ký hiệu với ký tự hoa và được in đậm (ví dụ: $\mathbf{X}$, $\mathbf{Y}$, và $\mathbf{Z}$).
Ma trận được biểu diễn bằng các `ndarray` với $2$ trục khi viết mã.

<!--
In math notation, we use $\mathbf{A} \in \mathbb{R}^{m \times n}$ to express that the matrix $\mathbf{A}$ consists of $m$ rows and $n$ columns of real-valued scalars.
Visually, we can illustrate any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ as a table, where each element $a_{ij}$ belongs to the $i^{\mathrm{th}}$ row and $j^{\mathrm{th}}$ column:
-->

Trong ký hiệu toán học, ta dùng $\mathbf{A} \in \mathbb{R}^{m \times n}$ để biểu thị một ma trận $\mathbf{A}$ $m$ hàng và $n$ cột các giá trị số thực.
Về mặt hình ảnh, ta có thể minh họa bất kỳ ma trận $\mathbf{A} \in \mathbb{R}^{m \times n}$ như một bảng biểu mà mỗi phần tử $a_{ij}$ nằm ở dòng thứ $i$ và cột thứ $j$ của bảng:   

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`


<!--
For any $\mathbf{A} \in \mathbb{R}^{m \times n}$, the shape of $\mathbf{A}$ is ($m$, $n$) or $m \times n$.
Specifically, when a matrix has the same number of rows and columns, its shape becomes a square; thus, it is called a *square matrix*.
-->

Với bất kỳ ma trận $\mathbf{A} \in \mathbb{R}^{m \times n}$, kích thước của ma trận $\mathbf{A}$ là ($m$, $n$) hay $m \times n$.
Trong trường hợp đặc biệt, nếu một ma trận có số dòng bằng số cột, dạng của nó là một hình vuông; vì vậy, nó được gọi là một *ma trận vuông* (*square matrix*).

<!--
We can create an $m \times n$ matrix in MXNet by specifying a shape with two components $m$ and $n$ when calling any of our favorite functions for instantiating an `ndarray`.
-->

Ta có thể tạo một ma trận $m \times n$ trong MXNet bằng cách khai báo kích thước của nó với hai thành phần $m$ và $n$ khi sử dụng bất kỳ hàm khởi tạo `ndarray` nào mà ta thích.

```{.python .input  n=6}
A = np.arange(20).reshape(5, 4)
A
```

<!--
We can access the scalar element $a_{ij}$ of a matrix $\mathbf{A}$ in :eqref:`eq_matrix_def` by specifying the indices for the row ($i$) and column ($j$), such as $[\mathbf{A}]_{ij}$.
When the scalar elements of a matrix $\mathbf{A}$, such as in :eqref:`eq_matrix_def`, are not given, we may simply use the lower-case letter of the matrix $\mathbf{A}$ with the index subscript, $a_{ij}$, to refer to $[\mathbf{A}]_{ij}$.
To keep notation simple, commas are inserted to separate indices only when necessary, such as $a_{2, 3j}$ and $[\mathbf{A}]_{2i-1, 3}$.
-->

Ta có thể truy cập phần tử vô hướng $a_{ij}$ của ma trận $\mathbf{A}$ trong :eqref:`eq_matrix_def` bằng cách khai báo chỉ số dòng ($i$) và chỉ số cột ($j$), như là $[\mathbf{A}]_{ij}$.
Khi thành phần vô hướng của ma trận $\mathbf{A}$, như trong :eqref:`eq_matrix_def`, vẫn chưa biết, ta có thể sử dụng ký tự viết thường của ma trận $\mathbf{A}$ với các chỉ số ghi dưới, $a_{ij}$, để chỉ thành phần $[\mathbf{A}]_{ij}$.
Nhằm giữ cho các ký hiệu đơn giản, dấu phẩy thường được thêm vào để phân tách các chỉ số khi cần thiết, như $a_{2, 3j}$ và $[\mathbf{A}]_{2i-1, 3}$.

<!--
Sometimes, we want to flip the axes.
When we exchange a matrix's rows and columns, the result is called the *transpose* of the matrix.
Formally, we signify a matrix $\mathbf{A}$'s transpose by $\mathbf{A}^\top$ and if $\mathbf{B} = \mathbf{A}^\top$, then $b_{ij} = a_{ji}$ for any $i$ and $j$.
Thus, the transpose of $\mathbf{A}$ in :eqref:`eq_matrix_def` is a $n \times m$ matrix:
-->

Đôi khi, ta muốn hoán đổi các trục.
Khi ta hoán đổi các dòng với các cột của ma trận, kết quả có được là *chuyển vị* (*transpose*) của ma trận đó.
Về lý thuyết, chuyển vị của ma trận $\mathbf{A}$ được ký hiệu là $\mathbf{A}^\top$ và nếu $\mathbf{B} = \mathbf{A}^\top$ thì $b_{ij} = a_{ji}$ với mọi $i$ và $j$.
Do đó, chuyển vị của $\mathbf{A}$ trong :eqref:`eq_matrix_def` là một ma trận $n \times m$:

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

<!--
In code, we access a matrix's transpose via the `T` attribute.
-->

Khi lập trình, ta lấy chuyển vị của một ma trận thông qua thuộc tính `T`.

```{.python .input  n=7}
A.T
```

<!--
As a special type of the square matrix, a *symmetric matrix* $\mathbf{A}$ is equal to its transpose: $\mathbf{A} = \mathbf{A}^\top$.
-->

Là một biến thể đặc biệt của ma trận vuông, *ma trận đối xứng* (*symmetric matrix*) $\mathbf{A}$ có chuyển vị bằng chính nó: $\mathbf{A} = \mathbf{A}^\top$.

```{.python .input}
B = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```{.python .input}
B == B.T
```

<!--
Matrices are useful data structures: they allow us to organize data that have different modalities of variation.
For example, rows in our matrix might correspond to different houses (data points), while columns might correspond to different attributes.
This should sound familiar if you have ever used spreadsheet software or have read :numref:`sec_pandas`.
Thus, although the default orientation of a single vector is a column vector, in a matrix that represents a tabular dataset, it is more conventional to treat each data point as a row vector in the matrix.
And, as we will see in later chapters, this convention will enable common deep learning practices.
For example, along the outermost axis of an `ndarray`, we can access or enumerate minibatches of data points, or just data points if no minibatch exists.
-->

Ma trận là một cấu trúc dữ liệu hữu ích: chúng cho phép ta tổ chức dữ liệu có nhiều phương thức biến thể khác nhau.
Ví dụ, các dòng trong ma trận của chúng ta có thể tượng trưng cho các căn nhà khác nhau (các điểm dữ liệu), còn các cột có thể tượng trưng cho những thuộc tính khác nhau của ngôi nhà.
Bạn có thể thấy quen thuộc với điều này nếu đã từng sử dụng các phần mềm lập bảng tính hoặc đã đọc :numref:`sec_pandas`.
Do đó, mặc dù một vector đơn lẻ có hướng mặc định là một vector cột, trong một ma trận biểu thị một tập dữ liệu bảng biểu, sẽ tiện lợi hơn nếu ta coi một điểm dữ liệu như một vector dòng trong ma trận.
Chúng ta sẽ thấy ở những chương sau, quy ước này sẽ cho phép sự xuất hiện của những kỹ thuật học sâu thông dụng.
Ví dụ, với trục ngoài cùng của `ndarray`, ta có thể truy cập hay duyệt qua các batch nhỏ của những điểm dữ liệu hoặc chỉ đơn thuần là các điểm dữ liệu nếu không có batch nhỏ nào cả. 

<!-- =================== Kết thúc dịch Phần 4 ==================== -->

<!-- =================== Bắt đầu dịch Phần 5 ==================== -->

<!--
## Tensors
-->

## Tensor

<!--
Just as vectors generalize scalars, and matrices generalize vectors, we can build data structures with even more axes. 
Tensors give us a generic way of describing `ndarray`s with an arbitrary number of axes. 
Vectors, for example, are first-order tensors, and matrices are second-order tensors.
Tensors are denoted with capital letters of a special font face (e.g., $\mathsf{X}$, $\mathsf{Y}$, and $\mathsf{Z}$) and their indexing mechanism (e.g., $x_{ijk}$ and $[\mathsf{X}]_{1, 2i-1, 3}$) is similar to that of matrices.
-->

Cũng giống như vector khái quát hoá số vô hướng và ma trận khái quát hoá vector, ta có thể xây dựng những cấu trúc dữ liệu với thậm chí nhiều trục hơn.
Tensor cho chúng ta một phương pháp tổng quát để miêu tả các `ndarray` với số trục bất kỳ.
Ví dụ, vector là các tensor bậc một còn ma trận là các tensor bậc hai.
Tensor được ký hiệu với ký tự viết hoa sử dụng một mặt chữ đặc biệt (ví dụ: $\mathsf{X}$, $\mathsf{Y}$, và $\mathsf{Z}$) và có cơ chế truy vấn (ví dụ: $x_{ijk}$ and $[\mathsf{X}]_{1, 2i-1, 3}$) giống như ma trận.

<!--
Tensors will become more important when we start working with images, which arrive as `ndarray`s with 3 axes corresponding to the height, width, and a *channel* axis for stacking the color channels (red, green, and blue). 
For now, we will skip over higher order tensors and focus on the basics.
-->

Tensor sẽ trở nên rất quan trọng khi ta bắt đầu làm việc với hình ảnh, thường được biểu diễn dưới dạng `ndarray` với 3 trục tương ứng với chiều cao, chiều rộng và một trục *kênh* (*channel*) để xếp chồng các kênh màu (đỏ, xanh lá và xanh dương).
Tạm thời, ta sẽ bỏ qua các tensor bậc cao hơn và tập trung vào những thứ cơ bản trước. 

```{.python .input  n=9}
X = np.arange(24).reshape(2, 3, 4)
X
```

<!--
## Basic Properties of Tensor Arithmetic
-->

## Các thuộc tính cơ bản của phép toán tensor

<!--
Scalars, vectors, matrices, and tensors of an arbitrary number of axes have some nice properties that often come in handy.
For example, you might have noticed from the definition of an elementwise operation that any elementwise unary operation does not change the shape of its operand.
Similarly, given any two tensors with the same shape, the result of any binary elementwise operation will be a tensor of that same shape.
For example, adding two matrices of the same shape performs elementwise addition over these two matrices.
-->

Số vô hướng, vector, ma trận và tensor với một số trục bất kỳ có một vài thuộc tính rất hữu dụng.
Ví dụ, bạn có thể để ý từ định nghĩa của phép toán theo từng phần tử (_elementwise_), bất kỳ phép toán theo từng phần tử một ngôi nào cũng không làm thay đổi kích thước của toán hạng của nó.
Tương tự, cho hai tensor bất kỳ có cùng kích thước, kết quả của bất kỳ phép toán theo từng phần tử hai ngôi sẽ là một tensor có cùng kích thước.
Ví dụ, cộng hai ma trận có cùng kích thước sẽ thực hiện phép cộng theo từng phần tử giữa hai ma trận này.  

```{.python .input}
A = np.arange(20).reshape(5, 4)
B = A.copy()  # Assign a copy of A to B by allocating new memory
A, A + B
```

<!--
Specifically, elementwise multiplication of two matrices is called their *Hadamard product* (math notation $\odot$).
Consider matrix $\mathbf{B} \in \mathbb{R}^{m \times n}$ whose element of row $i$ and column $j$ is $b_{ij}$. 
The Hadamard product of matrices $\mathbf{A}$ (defined in :eqref:`eq_matrix_def`) and $\mathbf{B}$
-->

Đặc biệt, phép nhân theo phần tử của hai ma trận được gọi là *phép nhân Hadamard* (*Hadamard product* -- ký hiệu toán học là $\odot$).
Xét ma trận $\mathbf{B} \in \mathbb{R}^{m \times n}$ có phần tử dòng $i$ và cột $j$ là $b_{ij}$.
Phép nhân Hadamard giữa ma trận $\mathbf{A}$ (khai báo ở :eqref:`eq_matrix_def`) và $\mathbf{B}$ là

$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```{.python .input}
A * B
```

<!--
Multiplying or adding a tensor by a scalar also does not change the shape of the tensor, where each element of the operand tensor will be added or multiplied by the scalar.
-->

Nhân hoặc cộng một tensor với một số vô hướng cũng sẽ không thay đổi kích thước của tensor, mỗi phần tử của tensor sẽ được cộng hoặc nhân cho số vô hướng đó.

```{.python .input}
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

<!-- =================== Kết thúc dịch Phần 5 ==================== -->

<!-- =================== Bắt đầu dịch Phần 6 ==================== -->

<!--
## Reduction
-->

## *dịch tiêu đề phía trên*

<!--
One useful operation that we can perform with arbitrary tensors is to calculate the sum of their elements.
In mathematical notation, we express sums using the $\sum$ symbol.
To express the sum of the elements in a vector $\mathbf{x}$ of length $d$, we write $\sum_{i=1}^d x_i$. In code, we can just call the `sum` function.
-->

*dịch đoạn phía trên*

```{.python .input  n=11}
x = np.arange(4)
x, x.sum()
```

<!--
We can express sums over the elements of tensors of arbitrary shape.
For example, the sum of the elements of an $m \times n$ matrix $\mathbf{A}$ could be written $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$.
-->

*dịch đoạn phía trên*

```{.python .input  n=12}
A.shape, A.sum()
```

<!--
By default, invoking the `sum` function *reduces* a tensor along all its axes to a scalar.
We can also specify the axes along which the tensor is reduced via summation.
Take matrices as an example.
To reduce the row dimension (axis $0$) by summing up elements of all the rows, we specify `axis=0` when invoking `sum`.
Since the input matrix reduces along axis $0$ to generate the output vector, the dimension of axis $0$ of the input is lost in the output shape.
-->

*dịch đoạn phía trên*

```{.python .input}
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

<!--
Specifying `axis=1` will reduce the column dimension (axis $1$) by summing up elements of all the columns.
Thus, the dimension of axis $1$ of the input is lost in the output shape.
-->

*dịch đoạn phía trên*

```{.python .input}
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

<!--
Reducing a matrix along both rows and columns via summation
is equivalent to summing up all the elements of the matrix.
-->

*dịch đoạn phía trên*

```{.python .input}
A.sum(axis=[0, 1])  # Same as A.sum()
```

<!--
A related quantity is the *mean*, which is also called the *average*.
We calculate the mean by dividing the sum by the total number of elements.
In code, we could just call `mean` on tensors of arbitrary shape.
-->

*dịch đoạn phía trên*

```{.python .input  n=13}
A.mean(), A.sum() / A.size
```

<!--
Like `sum`, `mean` can also reduce a tensor along the specified axes.
-->

*dịch đoạn phía trên*

```{.python .input}
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

<!-- =================== Kết thúc dịch Phần 6 ==================== -->

<!-- =================== Bắt đầu dịch Phần 7 ==================== -->

<!--
### Non-Reduction Sum
-->

### *dịch tiêu đề phía trên*

<!--
However, sometimes it can be useful to keep the number of axes unchanged when invoking `sum` or `mean` by setting `keepdims=True`.
-->

*dịch đoạn phía trên*

```{.python .input}
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

<!--
For instance, since `sum_A` still keeps its $2$ axes after summing each row, we can divide `A` by `sum_A` with broadcasting.
-->

*dịch đoạn phía trên*

```{.python .input}
A / sum_A
```

<!--
If we want to calculate the cumulative sum of elements of `A` along some axis, say `axis=0` (row by row), we can call the `cumsum` function. This function will not reduce the input tensor along any axis.
-->

*dịch đoạn phía trên*

```{.python .input}
A.cumsum(axis=0)
```

<!--
## Dot Products
-->

## *dịch tiêu đề phía trên*

<!--
So far, we have only performed elementwise operations, sums, and averages. 
And if this was all we could do, linear algebra probably would not deserve its own section. 
However, one of the most fundamental operations is the dot product. 
Given two vectors $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$, their *dot product* $\mathbf{x}^\top \mathbf{y}$ (or $\langle \mathbf{x}, \mathbf{y}  \rangle$) is a sum over the products of the elements at the same position: $\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$.
-->

*dịch đoạn phía trên*

```{.python .input  n=14}
y = np.ones(4)
x, y, np.dot(x, y)
```

<!--
Note that we can express the dot product of two vectors equivalently by performing an elementwise multiplication and then a sum:
-->

*dịch đoạn phía trên*

```{.python .input  n=15}
np.sum(x * y)
```

<!--
Dot products are useful in a wide range of contexts.
For example, given some set of values, denoted by a vector $\mathbf{x}  \in \mathbb{R}^d$ and a set of weights denoted by $\mathbf{w} \in \mathbb{R}^d$, the weighted sum of the values in $\mathbf{x}$ according to the weights $\mathbf{w}$ could be expressed as the dot product $\mathbf{x}^\top \mathbf{w}$.
When the weights are non-negative and sum to one (i.e., $\left(\sum_{i=1}^{d} {w_i} = 1\right)$), the dot product expresses a *weighted average*.
After normalizing two vectors to have the unit length, the dot products express the cosine of the angle between them.
We will formally introduce this notion of *length* later in this section.
-->

*dịch đoạn phía trên*

<!-- =================== Kết thúc dịch Phần 7 ==================== -->

<!-- =================== Bắt đầu dịch Phần 8 ==================== -->

<!--
## Matrix-Vector Products
-->

## *dịch tiêu đề phía trên*

<!--
Now that we know how to calculate dot products, we can begin to understand *matrix-vector products*.
Recall the matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ and the vector $\mathbf{x} \in \mathbb{R}^n$ defined and visualized in :eqref:`eq_matrix_def` and :eqref:`eq_vec_def` respectively.
Let's start off by visualizing the matrix $\mathbf{A}$ in terms of its row vectors
-->

*dịch đoạn phía trên*

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$
-->

*dịch đoạn phía trên*

<!--
where each $\mathbf{a}^\top_{i} \in \mathbb{R}^n$ is a row vector representing the $i^\mathrm{th}$ row of the matrix $\mathbf{A}$.
The matrix-vector product $\mathbf{A}\mathbf{x}$ is simply a column vector of length $m$, whose $i^\mathrm{th}$ element is the dot product $\mathbf{a}^\top_i \mathbf{x}$:
-->

*dịch đoạn phía trên*

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

<!--
We can think of multiplication by a matrix $\mathbf{A}\in \mathbb{R}^{m \times n}$ as a transformation that projects vectors from $\mathbb{R}^{n}$ to $\mathbb{R}^{m}$.
These transformations turn out to be remarkably useful.
For example, we can represent rotations as multiplications by a square matrix.
As we will see in subsequent chapters, we can also use matrix-vector products to describe the most intensive calculations required when computing each layer in a neural network given the values of the previous layer.
-->

*dịch đoạn phía trên*

<!--
Expressing matrix-vector products in code with `ndarray`s, we use the same `dot` function as for dot products.
When we call `np.dot(A, x)` with a matrix `A` and a vector `x`, the matrix-vector product is performed.
Note that the column dimension of `A` (its length along axis $1$) must be the same as the dimension of `x` (its length).
-->

*dịch đoạn phía trên*

```{.python .input  n=16}
A.shape, x.shape, np.dot(A, x)
```

<!-- =================== Kết thúc dịch Phần 8 ==================== -->

<!-- =================== Bắt đầu dịch Phần 9 ==================== -->

<!--
## Matrix-Matrix Multiplication
-->

## Phép nhân ma trận

<!--
If you have gotten the hang of dot products and matrix-vector products, then *matrix-matrix multiplication* should be straightforward.
-->

Nếu bạn đã quen thuộc với tích của ma trận và vector, tích của hai ma trận cũng tương tự như thế.

<!--
Say that we have two matrices $\mathbf{A} \in \mathbb{R}^{n \times k}$ and $\mathbf{B} \in \mathbb{R}^{k \times m}$:
-->

Giải sử ta có hai ma trận $\mathbf{A} \in \mathbb{R}^{n \times k}$ và $\mathbf{B} \in \mathbb{R}^{k \times m}$:

$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$

<!--
Denote by $\mathbf{a}^\top_{i} \in \mathbb{R}^k$ the row vector representing the $i^\mathrm{th}$ row of the matrix $\mathbf{A}$, and let $\mathbf{b}_{j} \in \mathbb{R}^k$ be the column vector from the $j^\mathrm{th}$ column of the matrix $\mathbf{B}$.
To produce the matrix product $\mathbf{C} = \mathbf{A}\mathbf{B}$, it is easiest to think of $\mathbf{A}$ in terms of its row vectors and $\mathbf{B}$ in terms of its column vectors:
-->

Đặt $\mathbf{a}^\top_{i} \in \mathbb{R}^k$ là vector hàng biểu diễn hàng thứ $$ của ma trận $\mathbf{A}$ và $\mathbf{b}_{j} \in \mathbb{R}^k$ là vector cột thứ $j$ của ma trận $\mathbf{B}$.
Để tính ma trận tích $\mathbf{C} = \mathbf{A}\mathbf{B}$, đơn giản nhất là viết các hàng của ma trận $\mathbf{A}$ các cột của ma trận $\mathbf{B}$:

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$


<!--
Then the matrix product $\mathbf{C} \in \mathbb{R}^{n \times m}$ is produced as we simply compute each element $c_{ij}$ as the dot product $\mathbf{a}^\top_i \mathbf{b}_j$:
-->

Khi đó ma trận tích $\mathbf{C} \in \mathbb{R}^{n \times m}$ được tạo bằng cách tính mỗi phần tử $c_{ij}$ như tích vô hướng  $\mathbf{a}^\top_i \mathbf{b}_j$:

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$

<!--
We can think of the matrix-matrix multiplication $\mathbf{AB}$ as simply performing $m$ matrix-vector products and stitching the results together to form an $n \times m$ matrix. 
Just as with ordinary dot products and matrix-vector products, we can compute matrix-matrix multiplication by using the `dot` function.
In the following snippet, we perform matrix multiplication on `A` and `B`.
Here, `A` is a matrix with $5$ rows and $4$ columns, and `B` is a matrix with $4$ rows and $3$ columns.
After multiplication, we obtain a matrix with $5$ rows and $3$ columns.
-->

Ta có thể coi tích hai ma trận $\mathbf{AB}$ như việc tính $m$ phép nhân ma trận và vector sau đó ghép các kết quả với nhau để tạo ra một ma trận $n \times m$.
Giống như tích vô hướng và phép nhân ma trận và vector, ta có thể tính phép nhân hai ma trận bằng cách sử dụng hàm `dot`.
Trong đoạn mã dưới đây, chúng ta tính phép nhân giữa `A` và `B`.
Ở đây, `A` là một ma trận với $5$ hàng $4$ cột và `B` là một ma trận với `4` hàng `3` cột.
Sau phép nhân này, ta thu được một ma trận với $5$ hàng $3$ cột.

```{.python .input  n=17}
B = np.ones(shape=(4, 3))
np.dot(A, B)
```

<!--
Matrix-matrix multiplication can be simply called *matrix multiplication*, and should not be confused with the Hadamard product.
-->

Phép nhân hai ma trận có thể được gọi đơn giản là *phép nhân ma trận* và không nên nhầm lẫn với phép nhân Hadamard.

<!-- =================== Kết thúc dịch Phần 9 ==================== -->

<!-- =================== Bắt đầu dịch Phần 10 ==================== -->

<!--
## Norms
-->

## Chuẩn

<!--
Some of the most useful operators in linear algebra are *norms*.
Informally, the norm of a vector tells us how *big* a vector is.
The notion of *size* under consideration here concerns not dimensionality but rather the magnitude of the components.
-->

Một trong những toán tử hữu dụng nhất của đại số tuyến tính là *chuẩn* (*norm*).
Nói dân dã thì, các chuẩn của một vector cho ta biết một vector *lớn* tầm nào.
Thuật ngữ *kích thước* đang xét ở đây không nói tới số chiều không gian mà đúng hơn là về độ lớn của các thành phần.

<!--
In linear algebra, a vector norm is a function $f$ that maps a vector to a scalar, satisfying a handful of properties.
Given any vector $\mathbf{x}$, the first property says that if we scale all the elements of a vector by a constant factor $\alpha$, its norm also scales by the *absolute value* of the same constant factor:
-->

Trong đại số tuyến tính, chuẩn của một vector là hàm số $f$ mà ánh xạ vector thành số vô hướng, thỏa mãn các tính chất sau.
Cho vector $\mathbf{x}$ bất kỳ, tính chất đầu tiên phát biểu rằng nếu chúng ta co giãn toàn bộ các phần tử của một vector bằng một hằng số $\alpha$, chuẩn của vector đó cũng co giãn theo *giá trị tuyệt đối* của hằng số đó :

$$f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x}).$$


<!--
The second property is the familiar triangle inequality:
-->

Tính chất thứ hai cũng giống như bất đẳng thức tam giác:

$$f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y}).$$


<!--
The third property simply says that the norm must be non-negative:
-->

Tính chất thứ ba phát biểu rằng một chuẩn phải không âm:

$$f(\mathbf{x}) \geq 0.$$

<!--
That makes sense, as in most contexts the smallest *size* for anything is 0.
The final property requires that the smallest norm is achieved and only achieved by a vector consisting of all zeros.
-->

Điều này nghĩa là, trong hầu hết các trường hợp thì *kích thước* nhỏ nhất cho mọi vector sẽ bằng 0.
Tính chất cuối cùng yêu cầu chuẩn nhỏ nhất thu được khi và chỉ khi toàn bộ thành phần của vector đó là 0.

$$\forall i, [\mathbf{x}]_i = 0 \Leftrightarrow f(\mathbf{x})=0.$$

<!--
You might notice that norms sound a lot like measures of distance.
And if you remember Euclidean distances (think Pythagoras' theorem) from grade school, then the concepts of non-negativity and the triangle inequality might ring a bell.
In fact, the Euclidean distance is a norm: specifically it is the $\ell_2$ norm.
Suppose that the elements in the $n$-dimensional vector $\mathbf{x}$ are $x_1, \ldots, x_n$.
The $\ell_2$ *norm* of $\mathbf{x}$ is the square root of the sum of the squares of the vector elements:
-->

Bạn chắc sẽ để ý là các chuẩn có vẻ giống như một thước đo khoảng cách.
Và nếu còn nhớ khái niệm khoảng cách Euclid (nhớ định lý Pythagoras không) học hồi cấp 3, thì mong khái niệm không âm và bất đẳng thức tam giác có thể gợi nhắc được một chút.
Thực tế là, khoảng cách Euclid cũng là một chuẩn: cụ thể là $\ell_2$.
Giả sử rằng các thành phần trong vector $n$ chiều $\mathbf{x}$ là $x_1, \ldots, x_n$.
*Chuẩn* $\ell_2$ của $\mathbf{x}$ là căn bậc hai của tổng các bình phương của các thành phần trong vector: 

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2},$$

<!--
where the subscript $2$ is often omitted in $\ell_2$ norms, i.e., $\|\mathbf{x}\|$ is equivalent to $\|\mathbf{x}\|_2$. 
In code, we can calculate the $\ell_2$ norm of a vector by calling `linalg.norm`.
-->

Số $2$ nhỏ ở dưới thường được lược đi khi viết chuẩn $\ell_2$, ví dụ, $\|\mathbf{x}\|$ cũng tương đương với $\|\mathbf{x}\|_2$.   

```{.python .input  n=18}
u = np.array([3, -4])
np.linalg.norm(u)
```

<!--
In deep learning, we work more often with the squared $\ell_2$ norm.
You will also frequently encounter the $\ell_1$ *norm*, which is expressed as the sum of the absolute values of the vector elements:
-->

Trong Học sâu, chúng ta thường gặp chuẩn $\ell_2$ bình phương hơn.
Bạn cũng hay gặp *chuẩn* $\ell_1$, chuẩn được biểu diễn bằng tổng các giá trị tuyệt đối của các thành phần trong vector.

$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$

<!--
As compared with the $\ell_2$ norm, it is less influenced by outliers.
To calculate the $\ell_1$ norm, we compose the absolute value function with a sum over the elements.
-->

So với chuẩn $\ell_2$, nó ít bị ảnh ưởng bởi các giá trị ngoại biên hơn.
Để tính chuẩn $\ell_1$, chúng ta dùng hàm giá trị tuyệt đối rồi lấy tổng các thành phần.

```{.python .input  n=19}
np.abs(u).sum()
```

<!--
Both the $\ell_2$ norm and the $\ell_1$ norm
are special cases of the more general $\ell_p$ *norm*:
-->

Cả hai chuẩn $\ell_2$ và $\ell_1$ đều là trường hợp riêng của một chuẩn tổng quát hơn, *chuẩn* $\ell_p$:

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

<!--
Analogous to $\ell_2$ norms of vectors, the *Frobenius norm* of a matrix $\mathbf{X} \in \mathbb{R}^{m \times n}$ is the square root of the sum of the squares of the matrix elements:
-->

Tương tự với chuẩn $\ell_2$ của vector, *chuẩn Frobenius* của một ma trận $\mathbf{X} \in \mathbb{R}^{m \times n}$ là căn bậc hai của tổng các bình phương của các thành phần trong ma trận:

$$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$

<!--
The Frobenius norm satisfies all the properties of vector norms.
It behaves as if it were an $\ell_2$ norm of a matrix-shaped vector. Invoking `linalg.norm` will calculate the Frobenius norm of a matrix.
-->

Chuẩn Frobenius thỏa mãn tất cả các tính chất của một chuẩn vector.
Nó giống như là chuẩn $\ell_2$ của một vector nhưng trong hình dạng của ma trận.
Dùng hàm `linalg.norm` để tính toán chuẩn Frobenius của ma trận.

```{.python .input}
np.linalg.norm(np.ones((4, 9)))
```

<!-- =================== Kết thúc dịch Phần 10 ==================== -->

<!-- =================== Bắt đầu dịch Phần 11 ==================== -->

<!--
### Norms and Objectives
-->

### *dịch tiêu đề phía trên*
:label:`subsec_norms_and_objectives`

<!--
While we do not want to get too far ahead of ourselves, we can plant some intuition already about why these concepts are useful.
In deep learning, we are often trying to solve optimization problems: *maximize* the probability assigned to observed data; *minimize* the distance between predictions and the ground-truth observations.
Assign vector representations to items (like words, products, or news articles) such that the distance between similar items is minimized, and the distance between dissimilar items is maximized.
Oftentimes, the objectives, perhaps the most important components of deep learning algorithms (besides the data), are expressed as norms.
-->

*dịch đoạn phía trên*



<!--
## More on Linear Algebra
-->

## *dịch tiêu đề phía trên*

<!--
In just this section, we have taught you all the linear algebra that you will need to understand a remarkable chunk of modern deep learning.
There is a lot more to linear algebra and a lot of that mathematics is useful for machine learning.
For example, matrices can be decomposed into factors, and these decompositions can reveal low-dimensional structure in real-world datasets.
There are entire subfields of machine learning that focus on using matrix decompositions and their generalizations to high-order tensors to discover structure in datasets and solve prediction problems.
But this book focuses on deep learning.
And we believe you will be much more inclined to learn more mathematics once you have gotten your hands dirty deploying useful machine learning models on real datasets.
So while we reserve the right to introduce more mathematics much later on, we will wrap up this section here.
-->

*dịch đoạn phía trên*

<!--
If you are eager to learn more about linear algebra,
you may refer to either :numref:`sec_geometry-linear-algebric-ops`
or other excellent resources :cite:`Strang.1993,Kolter.2008,Petersen.Pedersen.ea.2008`.
-->

*dịch đoạn phía trên*

<!-- =================== Kết thúc dịch Phần 11 ==================== -->

<!-- =================== Bắt đầu dịch Phần 12 ==================== -->

<!--
## Summary
-->

## *dịch tiêu đề phía trên*

<!--
* Scalars, vectors, matrices, and tensors are basic mathematical objects in linear algebra.
* Vectors generalize scalars, and matrices generalize vectors.
* In the `ndarray` representation, scalars, vectors, matrices, and tensors have 0, 1, 2, and an arbitrary number of axes, respectively.
* A tensor can be reduced along the specified axes by `sum` and `mean`.
* Elementwise multiplication of two matrices is called their Hadamard product. It is different from matrix multiplication.
* In deep learning, we often work with norms such as the $\ell_1$ norm, the $\ell_2$ norm, and the Frobenius norm.
* We can perform a variety of operations over scalars, vectors, matrices, and tensors with `ndarray` functions.
-->

*dịch đoạn phía trên*


<!--
## Exercises
-->

## *dịch tiêu đề phía trên*

<!--
1. Prove that the transpose of a matrix $\mathbf{A}$'s transpose is $\mathbf{A}$: $(\mathbf{A}^\top)^\top = \mathbf{A}$.
2. Given two matrices $\mathbf{A}$ and $\mathbf{B}$, show that the sum of transposes is equal to the transpose of a sum: $\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$.
3. Given any square matrix $\mathbf{A}$, is $\mathbf{A} + \mathbf{A}^\top$ always symmetric? Why?
4. We defined the tensor `X` of shape ($2$, $3$, $4$) in this section. What is the output of `len(X)`?
5. For a tensor `X` of arbitrary shape, does `len(X)` always correspond to the length of a certain axis of `X`? What is that axis?
6. Run `A / A.sum(axis=1)` and see what happens. Can you analyze the reason?
7. When traveling between two points in Manhattan, what is the distance that you need to cover in terms of the coordinates, i.e., in terms of avenues and streets? Can you travel diagonally?
8. Consider a tensor with shape ($2$, $3$, $4$). What are the shapes of the summation outputs along axis $0$, $1$, and $2$?
9. Feed a tensor with 3 or more axes to the `linalg.norm` function and observe its output. What does this function compute for `ndarray`s of arbitrary shape?
-->

*dịch đoạn phía trên*




<!--
## [Discussions](https://discuss.mxnet.io/t/2317)
-->

## *dịch tiêu đề phía trên*

<!--
![](../img/qr_linear-algebra.svg)
-->

![](../img/qr_linear-algebra.svg)

<!-- ===================== Kết thúc dịch Phần 12 ==================== -->

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
* Vũ Hữu Tiệp

<!-- Phần 10 -->
* Mai Sơn Hải

<!-- Phần 11 -->
*

<!-- Phần 12 -->
*
