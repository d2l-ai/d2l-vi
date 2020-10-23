<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Calculus
-->

# Giải tích
:label:`sec_calculus`

<!--
Finding the area of a polygon had remained mysterious
until at least $2,500$ years ago, when ancient Greeks divided a polygon into triangles and summed their areas.
To find the area of curved shapes, such as a circle,
ancient Greeks inscribed polygons in such shapes.
As shown in :numref:`fig_circle_area`,
an inscribed polygon with more sides of equal length better approximates
the circle. This process is also known as the *method of exhaustion*.
-->

Tìm diện tích của một đa giác vẫn là một bí ẩn cho tới ít nhất $2.500$ năm trước, khi người Hy Lạp cổ đại chia đa giác thành các tam giác và cộng diện tích của chúng lại.
Để tìm diện tích của các hình cong, như hình tròn, người Hy Lạp cổ đại đặt các đa giác nội tiếp bên trong các hình cong đó.
Như trong :numref:`fig_circle_area`, một đa giác nội tiếp với càng nhiều cạnh bằng nhau thì càng xấp xỉ đúng hình tròn. 
Quy trình này còn được biết đến như *phương pháp vét kiệt*.

<!--
![Find the area of a circle with the method of exhaustion.](../img/polygon_circle.svg)
-->

![Tìm diện tích hình tròn bằng phương pháp vét kiệt.](../img/polygon_circle.svg)
:label:`fig_circle_area`

<!--
In fact, the method of exhaustion is where *integral calculus* (will be described in :numref:`sec_integral_calculus`) originates from.
More than $2,000$ years later,
the other branch of calculus, *differential calculus*,
was invented.
Among the most critical applications of differential calculus,
optimization problems consider how to do something *the best*.
As discussed in :numref:`subsec_norms_and_objectives`,
such problems are ubiquitous in deep learning.
-->

Phương pháp vét kiệt chính là khởi nguồn của *giải tích tích phân* (sẽ được miêu tả trong :numref:`sec_integral_calculus`).
Hơn $2.000$ năm sau, nhánh còn lại của giải tích, *giải tích vi phân*, ra đời.
Trong những ứng dụng quan trọng nhất của giải tích vi phân, các bài toán tối ưu hoá sẽ tìm *cách tốt nhất* để thực hiện một công việc nào đó.
Như đã bàn đến trong :numref:`subsec_norms_and_objectives`, các bài toán như vậy vô cùng phổ biến trong học sâu.

<!--
In deep learning, we *train* models, updating them successively
so that they get better and better as they see more and more data.
Usually, getting better means minimizing a *loss function*,
a score that answers the question "how *bad* is our model?"
This question is more subtle than it appears.
Ultimately, what we really care about
is producing a model that performs well on data
that we have never seen before.
But we can only fit the model to data that we can actually see.
Thus we can decompose the task of fitting models into two key concerns:
i) *optimization*: the process of fitting our models to observed data;
ii) *generalization*: the mathematical principles and practitioners' wisdom
that guide as to how to produce models whose validity extends
beyond the exact set of data points used to train them.
-->

Trong học sâu, chúng ta *huấn luyện* các mô hình, cập nhật chúng liên tục để chúng ngày càng tốt hơn khi học với nhiều dữ liệu hơn.
Thông thường, trở nên tốt hơn tương đương với cực tiểu hoá một *hàm mất mát*, một điểm số sẽ trả lời câu hỏi "mô hình của ta đang *tệ* tới mức nào?"
Câu hỏi này lắt léo hơn ta tưởng nhiều.
Mục đích cuối cùng mà ta muốn là mô hình sẽ hoạt động tốt trên dữ liệu mà nó chưa từng nhìn thấy. <!-- người dịch tự sửa -->
Nhưng chúng ta chỉ có thể khớp mô hình trên dữ liệu mà ta đang có thể thấy.
Do đó ta có thể chia việc huấn luyện mô hình thành hai vấn đề chính:
i) *tối ưu hoá*: quy trình huấn luyện mô hình trên dữ liệu đã thấy.
ii) *tổng quát hoá*: dựa trên các nguyên tắc toán học và sự uyên thâm của người huấn luyện để tạo ra các mô hình mà tính hiệu quả của nó vượt ra khỏi tập dữ liệu huấn luyện.

<!--
To help you understand
optimization problems and methods in later chapters,
here we give a very brief primer on differential calculus
that is commonly used in deep learning.
-->

Để giúp bạn hiểu các bài toán tối ưu hóa và các phương pháp tối ưu hóa trong các chương sau, ở đây chúng tôi sẽ cung cấp một chương ngắn vỡ lòng về các kĩ thuật giải tích vi phân thông dụng trong học sâu. 

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Derivatives and Differentiation
-->

## Đạo hàm và Vi phân

<!--
We begin by addressing the calculation of derivatives,
a crucial step in nearly all deep learning optimization algorithms.
In deep learning, we typically choose loss functions
that are differentiable with respect to our model's parameters.
Put simply, this means that for each parameter,
we can determine how rapidly the loss would increase or decrease,
were we to *increase* or *decrease* that parameter
by an infinitesimally small amount.
-->

Chúng ta bắt đầu bằng việc đề cập tới khái niệm đạo hàm, một bước quan trọng của hầu hết các thuật toán tối ưu trong học sâu.
Trong học sâu, ta thường chọn những hàm mất mát khả vi theo các tham số của mô hình.
Nói đơn giản, với mỗi tham số, ta có thể xác định hàm mất mát tăng hoặc giảm nhanh như thế nào khi tham số đó *tăng* hoặc *giảm* chỉ một lượng cực nhỏ.

<!--
Suppose that we have a function $f: \mathbb{R} \rightarrow \mathbb{R}$,
whose input and output are both scalars.
The *derivative* of $f$ is defined as
-->

Giả sử ta có một hàm $f: \mathbb{R} \rightarrow \mathbb{R}$ có đầu vào và đầu ra đều là số vô hướng.
*Đạo hàm* của $f$ được định nghĩa là

$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h},$$
:eqlabel:`eq_derivative`

<!--
if this limit exists.
If $f'(a)$ exists,
$f$ is said to be *differentiable* at $a$.
If $f$ is differentiable at every number of an interval,
then this function is differentiable on this interval.
We can interpret the derivative $f'(x)$ in :eqref:`eq_derivative`
as the *instantaneous* rate of change of $f(x)$
with respect to $x$.
The so-called instantaneous rate of change is based on
the variation $h$ in $x$, which approaches $0$.
-->

nếu giới hạn này tồn tại.
Nếu $f'(a)$ tồn tại, $f$ được gọi là *khả vi* (_differentiable_) tại $a$.
Nếu $f$ khả vi tại mọi điểm trong một khoảng, thì hàm này được gọi là khả vi trong khoảng đó.
Ta có thể giải nghĩa đạo hàm $f'(x)$ trong :eqref:`eq_derivative` như là tốc độ thay đổi *tức thời* của hàm $f$ theo biến $x$.
Cái gọi là tốc độ thay đổi tức thời được dựa trên độ biến thiên $h$ trong $x$ khi $h$ tiến về $0$.

<!--
To illustrate derivatives,
let's experiment with an example.
Define $u = f(x) = 3x^2-4x$.
-->

Để minh họa cho khái niệm đạo hàm, hãy thử với một ví dụ.
Định nghĩa $u = f(x) = 3x^2-4x$.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

def f(x):
    return 3 * x ** 2 - 4 * x
```

<!--
By setting $x=1$ and letting $h$ approach $0$,
the numerical result of $\frac{f(x+h) - f(x)}{h}$
in :eqref:`eq_derivative` approaches $2$.
Though this experiment is not a mathematical proof,
we will see later that the derivative $u'$ is $2$ when $x=1$.
-->

Cho $x=1$ và $h$ tiến về $0$, kết quả của phương trình $\frac{f(x+h) - f(x)}{h}$ trong :eqref:`eq_derivative` tiến về $2$.
Dù thử nghiệm này không phải là một dạng chứng minh toán học, lát nữa ta sẽ thấy rằng quả thật đạo hàm của $u'$ là $2$ khi $x=1$.

```{.python .input}
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
for i in range(5):
    print('h=%.5f, numerical limit=%.5f' % (h, numerical_lim(f, 1, h)))
    h *= 0.1
```

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
Let's familiarize ourselves with a few equivalent notations for derivatives.
Given $y = f(x)$, where $x$ and $y$ are the independent variable and the dependent variable of the function $f$, respectively. The following expressions are equivalent:
-->

Hãy làm quen với một vài ký hiệu cùng được dùng để biểu diễn đạo hàm.
Cho $y = f(x)$ với $x$ và $y$ lần lượt là biến độc lập và biến phụ thuộc của hàm $f$. Những biểu diễn sau đây là tương đương nhau:

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$

<!--
where symbols $\frac{d}{dx}$ and $D$ are *differentiation operators* that indicate operation of *differentiation*.
We can use the following rules to differentiate common functions:
-->

với ký hiệu $\frac{d}{dx}$ và $D$ là các *toán tử vi phân* (_differentiation operator_) để chỉ các phép toán *vi phân*.
Ta có thể sử dụng các quy tắc lấy đạo hàm của các hàm thông dụng sau đây:

<!--
* $DC = 0$ ($C$ is a constant),
* $Dx^n = nx^{n-1}$ (the *power rule*, $n$ is any real number),
* $De^x = e^x$,
* $D\ln(x) = 1/x.$
-->

* $DC = 0$ ($C$ là một hằng số),
* $Dx^n = nx^{n-1}$ (*quy tắc lũy thừa*, $n$ là số thực bất kỳ),
* $De^x = e^x$,
* $D\ln(x) = 1/x.$

<!--
To differentiate a function that is formed from a few simpler functions such as the above common functions,
the following rules can be handy for us.
Suppose that functions $f$ and $g$ are both differentiable and $C$ is a constant,
we have the *constant multiple rule*
-->

Để lấy đạo hàm của một hàm được tạo từ vài hàm đơn giản hơn, ví dụ như từ những hàm thông dụng ở trên, có thể dùng các quy tắc hữu dụng dưới đây.
Giả sử hàm $f$ và $g$ đều khả vi và $C$ là một hằng số, ta có *quy tắc nhân hằng số* 

$$\frac{d}{dx} [Cf(x)] = C \frac{d}{dx} f(x),$$

<!--
the *sum rule*
-->

*quy tắc tổng*

$$\frac{d}{dx} [f(x) + g(x)] = \frac{d}{dx} f(x) + \frac{d}{dx} g(x),$$

<!--
the *product rule*
-->

*quy tắc nhân*

$$\frac{d}{dx} [f(x)g(x)] = f(x) \frac{d}{dx} [g(x)] + g(x) \frac{d}{dx} [f(x)],$$

<!--
and the *quotient rule*
-->

và *quy tắc đạo hàm phân thức*

$$\frac{d}{dx} \left[\frac{f(x)}{g(x)}\right] = \frac{g(x) \frac{d}{dx} [f(x)] - f(x) \frac{d}{dx} [g(x)]}{[g(x)]^2}.$$

<!--
Now we can apply a few of the above rules to find
$u' = f'(x) = 3 \frac{d}{dx} x^2-4\frac{d}{dx}x = 6x-4$.
Thus, by setting $x = 1$, we have $u' = 2$:
this is supported by our earlier experiment in this section
where the numerical result approaches $2$.
This derivative is also the slope of the tangent line
to the curve $u = f(x)$ when $x = 1$.
-->

Bây giờ ta có thể áp dụng các quy tắc ở trên để tìm đạo hàm $u' = f'(x) = 3 \frac{d}{dx} x^2-4\frac{d}{dx}x = 6x-4$.
Vậy nên, với $x = 1$, ta có $u' = 2$: điều này đã được kiểm chứng với thử nghiệm lúc trước ở mục này khi kết quả có được cũng tiến tới $2$.
Giá trị đạo hàm này cũng đồng thời là độ dốc của đường tiếp tuyến với đường cong $u = f(x)$ tại $x = 1$.

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
To visualize such an interpretation of derivatives,
we will use `matplotlib`,
a popular plotting library in Python.
To configure properties of the figures produced by `matplotlib`,
we need to define a few functions.
In the following,
the `use_svg_display` function specifies the `matplotlib` package to output the svg figures for sharper images.
-->

Để minh họa cách hiểu này của đạo hàm, ta sẽ dùng `matplotlib`, một thư viện vẽ biểu đồ thông dụng trong Python.
Ta cần định nghĩa một số hàm để cấu hình thuộc tính của các biểu đồ được tạo ra bởi `matplotlib`.
Trong đoạn mã sau, hàm `use_svg_display` chỉ định `matplotlib` tạo các biểu đồ ở dạng svg để có được chất lượng ảnh sắc nét hơn.

```{.python .input}
# Saved in the d2l package for later use
def use_svg_display():
    """Use the svg format to display a plot in Jupyter."""
    display.set_matplotlib_formats('svg')
```

<!--
We define the `set_figsize` function to specify the figure sizes. Note that here we directly use `d2l.plt` since the import statement `from matplotlib import pyplot as plt` has been marked for being saved in the `d2l` package in the preface.
-->

Ta định nghĩa hàm `set_figsize` để chỉ định kích thước của biểu đồ.
Lưu ý rằng ở đây ta đang dùng trực tiếp `d2l.plt` do câu lệnh `from matplotlib import pyplot as plt` đã được đánh dấu để lưu vào gói `d2l` trong phần Lời nói đầu.

```{.python .input}
# Saved in the d2l package for later use
def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
```

<!--
The following `set_axes` function sets properties of axes of figures produced by `matplotlib`.
-->

Hàm `set_axes` sau cấu hình thuộc tính của các trục biểu đồ tạo bởi `matplotlib`.

```{.python .input}
# Saved in the d2l package for later use
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
```

<!--
With these $3$ functions for figure configurations,
we define the `plot` function
to plot multiple curves succinctly
since we will need to visualize many curves throughout the book.
-->

Với ba hàm cấu hình biểu đồ trên, ta định nghĩa hàm `plot` để vẽ nhiều đồ thị một cách nhanh chóng vì ta sẽ cần minh họa khá nhiều đồ thị xuyên suốt cuốn sách.

```{.python .input}
# Saved in the d2l package for later use
def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=['-', 'm--', 'g-.', 'r:'], figsize=(3.5, 2.5), axes=None):
    """Plot data points."""
    d2l.set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # Return True if X (ndarray or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
```

<!--
Now we can plot the function $u = f(x)$ and its tangent line $y = 2x - 3$ at $x=1$, where the coefficient $2$ is the slope of the tangent line.
-->

Giờ ta có thể vẽ đồ thị của hàm số $u = f(x)$ và đường tiếp tuyến của nó $y = 2x - 3$ tại $x=1$, với hệ số $2$ là độ dốc của tiếp tuyến.

```{.python .input}
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 4 - BẮT ĐẦU ===================================-->

<!--
## Partial Derivatives
-->

## Đạo hàm riêng 

<!--
So far we have dealt with the differentiation of functions of just one variable.
In deep learning, functions often depend on *many* variables.
Thus, we need to extend the ideas of differentiation to these *multivariate* functions.
-->

Cho tới giờ, ta đã làm việc với đạo hàm của các hàm một biến.
Trong học sâu, các hàm lại thường phụ thuộc vào *nhiều* biến.
Do đó, ta cần mở rộng ý tưởng của đạo hàm cho các hàm *nhiều biến* đó.

<!--
Let $y = f(x_1, x_2, \ldots, x_n)$ be a function with $n$ variables. The *partial derivative* of $y$ with respect to its $i^\mathrm{th}$  parameter $x_i$ is
-->

Cho $y = f(x_1, x_2, \ldots, x_n)$ là một hàm với $n$ biến.
*Đạo hàm riêng* của $y$ theo tham số thứ $i$, $x_i$, là

$$ \frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$


<!--
To calculate $\frac{\partial y}{\partial x_i}$, we can simply treat $x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$ as constants and calculate the derivative of $y$ with respect to $x_i$.
For notation of partial derivatives, the following are equivalent:
-->

Để tính $\frac{\partial y}{\partial x_i}$, ta chỉ cần coi $x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$ là các hằng số và tính đạo hàm của $y$ theo $x_i$.
Để biểu diễn đạo hàm riêng, các ký hiệu sau đây đều có ý nghĩa tương đương:

$$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = f_{x_i} = f_i = D_i f = D_{x_i} f.$$

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
## Gradients
-->

## Gradient

<!--
We can concatenate partial derivatives of a multivariate function with respect to all its variables to obtain the *gradient* vector of the function.
Suppose that the input of function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is an $n$-dimensional vector $\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$ and the output is a scalar. The gradient of the function $f(\mathbf{x})$ with respect to $\mathbf{x}$ is a vector of $n$ partial derivatives:
-->

Chúng ta có thể ghép các đạo hàm riêng của mọi biến trong một hàm nhiều biến để thu được vector *gradient* của hàm số đó.
Giả sử rằng đầu vào của hàm $f: \mathbb{R}^n \rightarrow \mathbb{R}$ là một vector $n$ chiều $\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$ và đầu ra là một số vô hướng.
Gradient của hàm $f(\mathbf{x})$ theo $\mathbf{x}$ là một vector gồm $n$ đạo hàm riêng đó: 

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_n}\bigg]^\top.$$

<!--
where $\nabla_{\mathbf{x}} f(\mathbf{x})$ is often replaced by $\nabla f(\mathbf{x})$ when there is no ambiguity.
-->

Biểu thức $\nabla_{\mathbf{x}} f(\mathbf{x})$ thường được viết gọn thành $\nabla f(\mathbf{x})$ trong trường hợp không sợ nhầm lẫn.

<!--
Let $\mathbf{x}$ be an $n$-dimensional vector, the following rules are often used when differentiating multivariate functions:
-->

Cho $\mathbf{x}$ là một vector $n$-chiều, các quy tắc sau thường được dùng khi tính vi phân hàm đa biến:

<!--
* For all $\mathbf{A} \in \mathbb{R}^{m \times n}$, $\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$,
* For all  $\mathbf{A} \in \mathbb{R}^{n \times m}$, $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$,
* For all  $\mathbf{A} \in \mathbb{R}^{n \times n}$, $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$,
* $\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$.
-->

* Với mọi $\mathbf{A} \in \mathbb{R}^{m \times n}$, $\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$,
* Với mọi  $\mathbf{A} \in \mathbb{R}^{n \times m}$, $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$,
* Với mọi  $\mathbf{A} \in \mathbb{R}^{n \times n}$, $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$,
* $\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$.

<!--
Similarly, for any matrix $\mathbf{X}$, we have $\nabla_{\mathbf{X}} \|\mathbf{X} \|_F^2 = 2\mathbf{X}$. As we will see later, gradients are useful for designing optimization algorithms in deep learning.
-->

Tương tự, với bất kỳ ma trận $\mathbf{X}$ nào, ta đều có $\nabla_{\mathbf{X}} \|\mathbf{X} \|_F^2 = 2\mathbf{X}$.
Sau này ta sẽ thấy, gradient sẽ rất hữu ích khi thiết kế thuật toán tối ưu trong học sâu.
<!-- kết thúc revise phần 4 -->
<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!-- ========================================= REVISE PHẦN 4 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 5 - BẮT ĐẦU ===================================-->

<!--
## Chain Rule
-->

## Quy tắc dây chuyền

<!--
However, such gradients can be hard to find.
This is because multivariate functions in deep learning are often *composite*,
so we may not apply any of the aforementioned rules to differentiate these functions.
Fortunately, the *chain rule* enables us to differentiate composite functions.
-->

Tuy nhiên, những gradient như thế có thể khó để tính toán.
Đó là bởi vì các hàm nhiều biến trong học sâu đa phần là những *hàm hợp*, nên ta không thể áp dụng các quy tắc đề cập ở trên để lấy vi phân cho những hàm này.
May mắn thay, *quy tắc dây chuyền* cho phép chúng ta lấy vi phân của các hàm hợp.

<!--
Let's first consider functions of a single variable.
Suppose that functions $y=f(u)$ and $u=g(x)$ are both differentiable, then the chain rule states that
-->

Trước tiên, chúng ta hãy xem xét các hàm một biến.
Giả sử hai hàm $y=f(u)$ và $u=g(x)$ đều khả vi, quy tắc dây chuyền được mô tả như sau

$$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$

<!--
Now let's turn our attention to a more general scenario
where functions have an arbitrary number of variables.
Suppose that the differentiable function $y$ has variables
$u_1, u_2, \ldots, u_m$, where each differentiable function $u_i$
has variables $x_1, x_2, \ldots, x_n$.
Note that $y$ is a function of $x_1, x_2, \ldots, x_n$.
Then the chain rule gives
-->

Giờ ta sẽ xét trường hợp tổng quát hơn đối với các hàm nhiều biến.
Giả sử một hàm khả vi $y$ có các biến số $u_1, u_2, \ldots, u_m$, trong đó mỗi biến $u_i$ là một hàm khả vi của các biến $x_1, x_2, \ldots, x_n$.
Lưu ý rằng $y$ cũng là hàm của các biến $x_1, x_2, \ldots, x_n$.
Quy tắc dây chuyền cho ta

$$\frac{dy}{dx_i} = \frac{dy}{du_1} \frac{du_1}{dx_i} + \frac{dy}{du_2} \frac{du_2}{dx_i} + \cdots + \frac{dy}{du_m} \frac{du_m}{dx_i}$$

<!--
for any $i = 1, 2, \ldots, n$.
-->

cho mỗi $i = 1, 2, \ldots, n$.



<!--
## Summary
-->

## Tóm tắt


<!--
* Differential calculus and integral calculus are two branches of calculus, where the former can be applied to the ubiquitous optimization problems in deep learning.
* A derivative can be interpreted as the instantaneous rate of change of a function with respect to its variable. It is also the slope of the tangent line to the curve of the function.
* A gradient is a vector whose components are the partial derivatives of a multivariate function with respect to all its variables.
* The chain rule enables us to differentiate composite functions.
-->

* Vi phân và tích phân là hai nhánh con của giải tích, trong đó vi phân được ứng dụng rộng rãi trong các bài toán tối ưu hóa của học sâu.
* Đạo hàm có thể được hiểu như là tốc độ thay đổi tức thì của một hàm số đối với các biến số. Nó cũng là độ dốc của đường tiếp tuyến với đường cong của hàm.
* Gradient là một vector có các phần tử là đạo hàm riêng của một hàm nhiều biến theo tất cả các biến số của nó.
* Quy tắc dây chuyền cho phép chúng ta lấy vi phân của các hàm hợp.



<!--
## Exercises
-->

## Bài tập

<!--
1. Plot the function $y = f(x) = x^3 - \frac{1}{x}$ and its tangent line when $x = 1$.
1. Find the gradient of the function $f(\mathbf{x}) = 3x_1^2 + 5e^{x_2}$.
1. What is the gradient of the function $f(\mathbf{x}) = \|\mathbf{x}\|_2$?
1. Can you write out the chain rule for the case where $u = f(x, y, z)$ and $x = x(a, b)$, $y = y(a, b)$, and $z = z(a, b)$?
-->

*dịch đoạn phía trên*
1. Vẽ đồ thị của hàm số $y = f(x) = x^3 - \frac{1}{x}$ và đường tiếp tuyến của nó tại $x = 1$.
1. Tìm gradient của hàm số $f(\mathbf{x}) = 3x_1^2 + 5e^{x_2}$.
1. Gradient của hàm $f(\mathbf{x}) = \|\mathbf{x}\|_2$ là gì?
1.  Có thể dùng quy tắc dây chuyền cho trường hợp sau đây không: $u = f(x, y, z)$, với $x = x(a, b)$, $y = y(a, b)$ và $z = z(a, b)$?

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->

<!-- ========================================= REVISE PHẦN 5 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/5008)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/5008)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)


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
* Lê Khắc Hồng Phúc
* Phạm Hồng Vinh
* Vũ Hữu Tiệp
* Nguyễn Cảnh Thướng
* Phạm Minh Đức
* Tạ H. Duy Nguyên
