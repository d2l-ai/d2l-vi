# Calculus (Giải tích)
:label:`sec_calculus`

Tìm diện tích của một đa giác vẫn còn bí ẩn cho đến ít nhất 2.500 năm trước, khi người Hy Lạp cổ đại chia một đa giác thành tam giác và tổng hợp các khu vực của chúng. Để tìm khu vực của các hình dạng cong, chẳng hạn như một vòng tròn, người Hy Lạp cổ đại được ghi đa giác trong các hình dạng như vậy. Như thể hiện trong :numref:`fig_circle_area`, một đa giác được ghi với nhiều cạnh có chiều dài bằng nhau tốt hơn gần đúng vòng tròn. Quá trình này còn được gọi là *phương pháp kiệt thúc*. 

![Find the area of a circle with the method of exhaustion.](../img/polygon-circle.svg)
:label:`fig_circle_area`

Trên thực tế, phương pháp kiệt sức là nơi * tích phân tích* (sẽ được mô tả trong :numref:`sec_integral_calculus`) bắt nguồn từ. Hơn 2.000 năm sau, nhánh khác của giải tích, *vi phân tích*, đã được phát minh. Trong số các ứng dụng quan trọng nhất của phép tính vi phân, các bài toán tối ưu hóa xem xét cách làm một cái gì đó * tốt nhất*. Như đã thảo luận trong :numref:`subsec_norms_and_objectives`, những vấn đề như vậy là phổ biến trong học sâu. 

Trong học sâu, chúng tôi * đào tạo* mô hình, cập nhật chúng liên tiếp để chúng trở nên tốt hơn và tốt hơn khi họ thấy ngày càng nhiều dữ liệu. Thông thường, nhận được tốt hơn có nghĩa là giảm thiểu chức năng *mất *, một điểm số trả lời câu hỏi “làm thế nào * xấu* là mô hình của chúng tôi?” Câu hỏi này tinh tế hơn nó xuất hiện. Cuối cùng, những gì chúng tôi thực sự quan tâm là tạo ra một mô hình hoạt động tốt trên dữ liệu mà chúng tôi chưa từng thấy trước đây. Nhưng chúng ta chỉ có thể phù hợp với mô hình để dữ liệu mà chúng ta thực sự có thể thấy. Do đó, chúng ta có thể phân hủy nhiệm vụ lắp các mô hình thành hai mối quan tâm chính: (i) *tối ưu hóa*: quá trình lắp các mô hình của chúng tôi với dữ liệu quan sát; (ii) *khái quát hóa*: các nguyên tắc toán học và trí tuệ của các học viên hướng dẫn về cách tạo ra các mô hình có giá trị vượt quá bộ dữ liệu chính xác examples ví dụ used to train đào tạo them. 

Để giúp bạn hiểu các vấn đề và phương pháp tối ưu hóa trong các chương sau, ở đây chúng tôi đưa ra một mồi rất ngắn gọn về phép tính vi phân thường được sử dụng trong deep learning. 

## Các dẫn xuất và sự khác biệt

Chúng tôi bắt đầu bằng cách giải quyết việc tính toán các dẫn xuất, một bước quan trọng trong gần như tất cả các thuật toán tối ưu hóa học tập sâu. Trong deep learning, chúng ta thường chọn các chức năng mất mát có thể khác biệt đối với các thông số của mô hình của chúng tôi. Nói một cách đơn giản, điều này có nghĩa là đối với mỗi tham số, chúng ta có thể xác định mức độ tổn thất sẽ tăng hoặc giảm nhanh như thế nào, chúng tôi * tăng* hoặc * giảm tham số đó bằng một lượng nhỏ vô hạn. 

Giả sử rằng chúng ta có một hàm $f: \mathbb{R} \rightarrow \mathbb{R}$, có đầu vào và đầu ra là cả vô hướng. [***phái sinh* của $f$ được định nghĩa là**] 

(**$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h},$$**) :eqlabel:`eq_derivative` 

if this limitgiới hạn exists tồn tại. Nếu $f'(a)$ tồn tại, $f$ được cho là *khác biệt* tại $a$. Nếu $f$ có thể khác biệt ở mọi số của một khoảng thời gian, thì chức năng này có thể phân biệt trong khoảng thời gian này. Chúng ta có thể giải thích đạo hàm $f'(x)$ trong :eqref:`eq_derivative` là tỷ lệ thay đổi tức thường* của $f(x)$ đối với $x$. Cái gọi là tỷ lệ thay đổi tức thời dựa trên sự thay đổi $h$ trong $x$, tiếp cận $0$. 

Để minh họa các dẫn xuất, chúng ta hãy thử nghiệm với một ví dụ. (** Define $u = f(x) = 3x^2-4x$.**)

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from matplotlib_inline import backend_inline
from mxnet import np, npx
npx.set_np()

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from matplotlib_inline import backend_inline
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from matplotlib_inline import backend_inline
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

[**Bằng cách thiết lập $x=1$ và để $h$ tiếp cận $0$, kết quả số của $\frac{f(x+h) - f(x)}{h}$**] trong :eqref:`eq_derivative` (** cách tiếp cận $2$.**) Mặc dù thí nghiệm này không phải là một bằng chứng toán học, chúng ta sẽ thấy sau đó đạo hàm $u'$ là $2$ khi $2$.

```{.python .input}
#@tab all
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1
```

Chúng ta hãy làm quen với một vài ký hiệu tương đương cho các dẫn xuất. Cho $y = f(x)$, trong đó $x$ và $y$ là biến độc lập và biến phụ thuộc của hàm $f$, tương ứng. Các biểu thức sau là tương đương: 

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$

trong đó các ký hiệu $\frac{d}{dx}$ và $D$ là * toán tử khác biệt* cho biết hoạt động của *sự khác biệt*. Chúng ta có thể sử dụng các quy tắc sau để phân biệt các hàm chung: 

* $DC = 0$ ($C$ là một hằng số),
* $Dx^n = nx^{n-1}$ (quy tắc điện*, $n$ là bất kỳ số thực nào),
* $De^x = e^x$,
* $D\ln(x) = 1/x.$

Để phân biệt một hàm được hình thành từ một vài chức năng đơn giản hơn như các chức năng phổ biến trên, các quy tắc sau đây có thể hữu ích cho chúng ta. Giả sử rằng các chức năng $f$ và $g$ đều có thể phân biệt và $C$ là một hằng số, chúng ta có quy tắc nhiều * hằng số* 

$$\frac{d}{dx} [Cf(x)] = C \frac{d}{dx} f(x),$$

quy tắc * sum* 

$$\frac{d}{dx} [f(x) + g(x)] = \frac{d}{dx} f(x) + \frac{d}{dx} g(x),$$

quy tắc *sản phẩm* 

$$\frac{d}{dx} [f(x)g(x)] = f(x) \frac{d}{dx} [g(x)] + g(x) \frac{d}{dx} [f(x)],$$

và quy tắc *thương lượng* 

$$\frac{d}{dx} \left[\frac{f(x)}{g(x)}\right] = \frac{g(x) \frac{d}{dx} [f(x)] - f(x) \frac{d}{dx} [g(x)]}{[g(x)]^2}.$$

Bây giờ chúng ta có thể áp dụng một vài trong số các quy tắc trên để tìm $u' = f'(x) = 3 \frac{d}{dx} x^2-4\frac{d}{dx}x = 6x-4$. Do đó, bằng cách đặt $x = 1$, chúng tôi có $u' = 2$: điều này được hỗ trợ bởi thí nghiệm trước đó của chúng tôi trong phần này, nơi kết quả số tiếp cận $2$. Đạo hàm này cũng là độ dốc của đường tiếp tuyến với đường cong $u = f(x)$ khi $x = 1$. 

[**Để hình dung một cách giải thích các dẫn xuất như vậy, chúng ta sẽ sử dụng `matplotlib`, **] một thư viện vẽ phổ biến trong Python. Để cấu hình các thuộc tính của các số liệu được tạo ra bởi `matplotlib`, chúng ta cần xác định một vài chức năng. Sau đây, hàm `use_svg_display` chỉ định gói `matplotlib` để xuất các số liệu svg cho hình ảnh sắc nét hơn. Lưu ý rằng nhận xét `# @save `là một dấu đặc biệt trong đó hàm, lớp hoặc câu lệnh sau được lưu trong gói `d2l` để sau này chúng có thể được gọi trực tiếp (ví dụ, `d2l.use_svg_display()`) mà không được định nghĩa lại.

```{.python .input}
#@tab all
def use_svg_display():  #@save
    """Use the svg format to display a plot in Jupyter."""
    backend_inline.set_matplotlib_formats('svg')
```

Chúng tôi xác định hàm `set_figsize` để chỉ định kích thước hình. Lưu ý rằng ở đây chúng tôi trực tiếp sử dụng `d2l.plt` vì lệnh nhập khẩu `from matplotlib import pyplot as plt` đã được đánh dấu là được lưu trong gói `d2l` trong lời nói đầu.

```{.python .input}
#@tab all
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """Set the figure size for matplotlib."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
```

Chức năng `set_axes` sau đây đặt các thuộc tính của trục của các con số được tạo ra bởi `matplotlib`.

```{.python .input}
#@tab all
#@save
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

Với ba chức năng này cho cấu hình hình, chúng tôi xác định hàm `plot` để vẽ nhiều đường cong một cách ngắn gọn vì chúng ta sẽ cần hình dung nhiều đường cong trong suốt cuốn sách.

```{.python .input}
#@tab all
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points."""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # Return True if `X` (tensor or list) has 1 axis
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

Bây giờ chúng ta có thể [** vẽ hàm $u = f(x)$ và đường tiếp tuyến của nó $y = 2x - 3$ tại $x=1$**], trong đó hệ số $2$ là độ dốc của đường tiếp tuyến.

```{.python .input}
#@tab all
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

## Phái sinh một phần

Cho đến nay chúng ta đã xử lý sự khác biệt của các chức năng của chỉ một biến. Trong deep learning, các hàm thường phụ thuộc vào biến * nhiều *. Do đó, chúng ta cần mở rộng các ý tưởng khác biệt cho các chức năng * đa lộ* này. 

Hãy để $y = f(x_1, x_2, \ldots, x_n)$ là một hàm với $n$ biến. Phái sinh một phần* của $y$ đối với tham số $i^\mathrm{th}$ $x_i$ của nó là 

$$ \frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$

Để tính toán $\frac{\partial y}{\partial x_i}$, chúng ta chỉ có thể coi $x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$ là hằng số và tính toán đạo hàm của $y$ đối với $x_i$. Đối với ký hiệu của các dẫn xuất từng phần, sau đây là tương đương: 

$$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = f_{x_i} = f_i = D_i f = D_{x_i} f.$$

## Độ dốc
:label:`subsec_calculus-grad`

Chúng ta có thể nối các dẫn xuất từng phần của một hàm đa biến đối với tất cả các biến của nó để có được vectơ *gradient* của hàm. Giả sử rằng đầu vào của hàm $f: \mathbb{R}^n \rightarrow \mathbb{R}$ là một vector $n$ chiều $\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$ và đầu ra là vô hướng. Gradient của hàm $f(\mathbf{x})$ đối với $\mathbf{x}$ là một vectơ của $n$ dẫn xuất một phần: 

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_n}\bigg]^\top,$$

trong đó $\nabla_{\mathbf{x}} f(\mathbf{x})$ thường được thay thế bằng $\nabla f(\mathbf{x})$ khi không có sự mơ hồ. 

Để $\mathbf{x}$ là một vector $n$ chiều, các quy tắc sau thường được sử dụng khi phân biệt các chức năng đa biến: 

* For all $\mathbf{A} \in \mathbb{R}^{m \times n}$, $\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$,
* For all $\mathbf{A} \in \mathbb{R}^{n \times m}$, $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$,
* For all $\mathbf{A} \in \mathbb{R}^{n \times n}$, $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$,
* $\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$.

Tương tự, đối với bất kỳ ma trận $\mathbf{X}$, chúng tôi có $\nabla_{\mathbf{X}} \|\mathbf{X} \|_F^2 = 2\mathbf{X}$. Như chúng ta sẽ thấy sau, gradient rất hữu ích cho việc thiết kế các thuật toán tối ưu hóa trong deep learning. 

## Quy tắc chuỗi

Tuy nhiên, độ dốc như vậy có thể khó tìm. Điều này là do các chức năng đa biến trong học sâu thường là * composite*, vì vậy chúng tôi có thể không áp dụng bất kỳ quy tắc nào nói trên để phân biệt các chức năng này. May mắn thay, quy tắc chuỗi * cho phép chúng tôi phân biệt các chức năng tổng hợp. 

Trước tiên chúng ta hãy xem xét các chức năng của một biến duy nhất. Giả sử rằng các chức năng $y=f(u)$ và $u=g(x)$ đều có thể khác biệt, thì quy tắc chuỗi nói rằng 

$$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$

Bây giờ chúng ta hãy chuyển sự chú ý của chúng tôi sang một kịch bản chung hơn trong đó các chức năng có một số tùy ý của biến. Giả sử rằng hàm phân biệt $y$ có các biến $u_1, u_2, \ldots, u_m$, trong đó mỗi hàm phân biệt $u_i$ có các biến $x_1, x_2, \ldots, x_n$. Lưu ý rằng $y$ là một hàm của $x_1, x_2, \ldots, x_n$. Sau đó, quy tắc chuỗi cho 

$$\frac{dy}{dx_i} = \frac{dy}{du_1} \frac{du_1}{dx_i} + \frac{dy}{du_2} \frac{du_2}{dx_i} + \cdots + \frac{dy}{du_m} \frac{du_m}{dx_i}$$

for any $i = 1, 2, \ldots, n$. 

## Tóm tắt

* Giải tích phân và giải tích phân là hai nhánh của giải tích, nơi mà trước đây có thể được áp dụng cho các bài toán tối ưu hóa phổ biến trong học sâu.
* Một đạo hàm có thể được hiểu là tốc độ thay đổi tức thời của một hàm đối với biến của nó. Nó cũng là độ dốc của đường tiếp tuyến với đường cong của hàm.
* Gradient là một vectơ có thành phần là các dẫn xuất từng phần của một hàm đa biến đối với tất cả các biến của nó.
* Quy tắc chuỗi cho phép chúng ta phân biệt các hàm tổng hợp.

## Bài tập

1. Vẽ chức năng $y = f(x) = x^3 - \frac{1}{x}$ và đường tiếp tuyến của nó khi $x = 1$.
1. Tìm gradient của hàm $f(\mathbf{x}) = 3x_1^2 + 5e^{x_2}$.
1. Gradient của hàm $f(\mathbf{x}) = \|\mathbf{x}\|_2$ là gì?
1. Bạn có thể viết ra quy tắc chuỗi cho trường hợp $u = f(x, y, z)$ và $x = x(a, b)$, $y = y(a, b)$ và $z = z(a, b)$ không?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/32)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/33)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/197)
:end_tab:
