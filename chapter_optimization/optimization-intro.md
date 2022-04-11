# Tối ưu hóa và học sâu

Trong phần này, chúng ta sẽ thảo luận về mối quan hệ giữa tối ưu hóa và học sâu cũng như những thách thức của việc sử dụng tối ưu hóa trong học sâu. Đối với một vấn đề học sâu, chúng ta thường sẽ xác định hàm *loss* trước tiên. Khi chúng tôi có chức năng mất mát, chúng ta có thể sử dụng thuật toán tối ưu hóa để cố gắng giảm thiểu tổn thất. Trong tối ưu hóa, một hàm mất thường được gọi là chức năng mục tiêu * của bài toán tối ưu hóa. Theo truyền thống và quy ước hầu hết các thuật toán tối ưu hóa có liên quan đến *giảm thiểu *. Nếu chúng ta cần tối đa hóa một mục tiêu thì có một giải pháp đơn giản: chỉ cần lật dấu hiệu trên mục tiêu. 

## Mục tiêu tối ưu hóa

Mặc dù tối ưu hóa cung cấp một cách để giảm thiểu chức năng mất mát cho học sâu, nhưng về bản chất, các mục tiêu tối ưu hóa và học sâu về cơ bản là khác nhau. Cái trước chủ yếu quan tâm đến việc giảm thiểu một mục tiêu trong khi sau này liên quan đến việc tìm kiếm một mô hình phù hợp, cho một lượng dữ liệu hữu hạn. Năm :numref:`sec_model_selection`, chúng tôi đã thảo luận chi tiết về sự khác biệt giữa hai mục tiêu này. Ví dụ, lỗi đào tạo và lỗi tổng quát thường khác nhau: vì chức năng khách quan của thuật toán tối ưu hóa thường là hàm mất dựa trên tập dữ liệu đào tạo, mục tiêu tối ưu hóa là giảm lỗi đào tạo. Tuy nhiên, mục tiêu của học sâu (hoặc rộng hơn là suy luận thống kê) là giảm sai số tổng quát hóa. Để thực hiện sau này, chúng ta cần chú ý đến overfitting ngoài việc sử dụng thuật toán tối ưu hóa để giảm lỗi đào tạo.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
from mpl_toolkits import mplot3d
import tensorflow as tf
```

Để minh họa các mục tiêu khác nhau đã nói ở trên, chúng ta hãy xem xét rủi ro thực nghiệm và rủi ro. Như được mô tả trong :numref:`subsec_empirical-risk-and-risk`, rủi ro thực nghiệm là một tổn thất trung bình trên tập dữ liệu đào tạo trong khi rủi ro là sự mất mát dự kiến trên toàn bộ dân số dữ liệu. Dưới đây chúng tôi xác định hai chức năng: hàm rủi ro `f` và hàm rủi ro thực nghiệm `g`. Giả sử rằng chúng ta chỉ có một số lượng hữu hạn của dữ liệu đào tạo. Kết quả là, ở đây `g` ít mịn hơn `f`.

```{.python .input}
#@tab all
def f(x):
    return x * d2l.cos(np.pi * x)

def g(x):
    return f(x) + 0.2 * d2l.cos(5 * np.pi * x)
```

Biểu đồ dưới đây minh họa rằng mức tối thiểu của rủi ro thực nghiệm trên một tập dữ liệu đào tạo có thể ở một vị trí khác với mức tối thiểu của rủi ro (lỗi tổng quát).

```{.python .input}
#@tab all
def annotate(text, xy, xytext):  #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

x = d2l.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
```

## Tối ưu hóa thách thức trong Deep Learning

Trong chương này, chúng ta sẽ tập trung cụ thể vào hiệu suất của các thuật toán tối ưu hóa trong việc giảm thiểu hàm khách quan, chứ không phải là lỗi tổng quát hóa của mô hình. Trong :numref:`sec_linear_regression`, chúng tôi phân biệt giữa các giải pháp phân tích và các giải pháp số trong các vấn đề tối ưu hóa. Trong học sâu, hầu hết các chức năng khách quan đều phức tạp và không có giải pháp phân tích. Thay vào đó, chúng ta phải sử dụng các thuật toán tối ưu hóa số. Các thuật toán tối ưu hóa trong chương này đều thuộc thể loại này. 

Có rất nhiều thách thức trong tối ưu hóa học tập sâu. Một số trong những điều đáng kinh ngạc nhất là minima địa phương, điểm yên ngựa, và độ dốc biến mất. Hãy để chúng tôi có một cái nhìn tại họ. 

### Minima địa phương

Đối với bất kỳ hàm khách quan $f(x)$, nếu giá trị của $f(x)$ tại $x$ nhỏ hơn giá trị của $f(x)$ tại bất kỳ điểm nào khác trong vùng lân cận $x$, thì $f(x)$ có thể là mức tối thiểu địa phương. Nếu giá trị của $f(x)$ tại $x$ là mức tối thiểu của hàm khách quan trên toàn bộ miền, thì $f(x)$ là mức tối thiểu toàn cầu. 

Ví dụ, cho chức năng 

$$f(x) = x \cdot \text{cos}(\pi x) \text{ for } -1.0 \leq x \leq 2.0,$$

chúng ta có thể xấp xỉ tối thiểu địa phương và tối thiểu toàn cầu của chức năng này.

```{.python .input}
#@tab all
x = d2l.arange(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
```

Chức năng khách quan của các mô hình học sâu thường có nhiều optima địa phương. Khi giải pháp số của một bài toán tối ưu hóa ở gần tối ưu cục bộ, giải pháp số thu được bằng cách lặp cuối cùng chỉ có thể giảm thiểu hàm mục tiêu *locally*, thay vì * globally*, khi gradient của các giải pháp của hàm khách quan tiếp cận hoặc trở thành số không. Chỉ một số mức độ tiếng ồn có thể đánh bật tham số ra khỏi mức tối thiểu cục bộ. Trên thực tế, đây là một trong những đặc tính có lợi của dòng gradient ngẫu nhiên minibatch, nơi sự thay đổi tự nhiên của gradient trên minibatches có thể loại bỏ các thông số từ minima cục bộ. 

### Điểm yên

Bên cạnh minima địa phương, điểm yên là một lý do khác để gradient biến mất. Một điểm yên * là bất kỳ vị trí nào mà tất cả các gradient của một hàm biến mất nhưng không phải là một toàn cầu cũng không phải là một mức tối thiểu địa phương. Hãy xem xét chức năng $f(x) = x^3$. Dẫn xuất đầu tiên và thứ hai của nó biến mất cho $x=0$. Tối ưu hóa có thể bị đình trệ vào thời điểm này, mặc dù nó không phải là mức tối thiểu.

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
```

Các điểm yên ở kích thước cao hơn thậm chí còn ngấm ngầm hơn, như ví dụ dưới đây cho thấy. Hãy xem xét chức năng $f(x, y) = x^2 - y^2$. Nó có điểm yên của nó ở $(0, 0)$. Đây là mức tối đa đối với $y$ và mức tối thiểu đối với $x$. Hơn nữa, nó * trông giống như một yên ngựa, đó là nơi tài sản toán học này có tên của nó.

```{.python .input}
#@tab mxnet
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), z.asnumpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

```{.python .input}
#@tab pytorch, tensorflow
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

Chúng ta giả định rằng đầu vào của một hàm là một vectơ $k$ chiều và đầu ra của nó là vô hướng, do đó ma trận Hessian của nó sẽ có $k$ eigenvalues (tham khảo [online appendix on eigendecompositions](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/eigendecomposition.html)). Giải pháp của hàm có thể là một mức tối thiểu cục bộ, tối đa cục bộ, hoặc một điểm yên tại một vị trí mà gradient hàm bằng 0: 

* Khi eigenvalues của ma trận Hessian của hàm tại vị trí zero-gradient đều dương, chúng ta có một mức tối thiểu cục bộ cho hàm.
* Khi eigenvalues của ma trận Hessian của hàm tại vị trí zero-gradient đều âm, chúng ta có một cực đại cục bộ cho hàm.
* Khi eigenvalues của ma trận Hessian của hàm tại vị trí zero-gradient là âm và dương, chúng ta có một điểm yên cho hàm.

Đối với các vấn đề chiều cao, khả năng ít nhất * một số* của eigenvalues là âm là khá cao. Điều này làm cho các điểm yên có nhiều khả năng hơn minima địa phương. Chúng tôi sẽ thảo luận một số ngoại lệ đối với tình huống này trong phần tiếp theo khi giới thiệu lồi. Nói tóm lại, các hàm lồi là những hàm mà các eigenvalues của Hessian không bao giờ là âm. Đáng buồn thay, mặc dù, hầu hết các vấn đề học tập sâu không rơi vào thể loại này. Tuy nhiên, nó là một công cụ tuyệt vời để nghiên cứu các thuật toán tối ưu hóa. 

### Biến mất Gradient

Có lẽ vấn đề ngấm ngầm nhất để gặp phải là gradient biến mất. Nhớ lại các chức năng kích hoạt được sử dụng phổ biến của chúng tôi và các dẫn xuất của chúng trong :numref:`subsec_activation-functions`. Ví dụ, giả sử rằng chúng tôi muốn giảm thiểu chức năng $f(x) = \tanh(x)$ và chúng tôi tình cờ bắt đầu tại $x = 4$. Như chúng ta có thể thấy, gradient của $f$ gần với nil. Cụ thể hơn, $f'(x) = 1 - \tanh^2(x)$ và do đó $f'(4) = 0.0013$. Do đó, tối ưu hóa sẽ bị kẹt trong một thời gian dài trước khi chúng tôi tiến bộ. Đây hóa ra là một trong những lý do khiến việc đào tạo các mô hình học sâu khá khó khăn trước khi giới thiệu chức năng kích hoạt ReLU.

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 5.0, 0.01)
d2l.plot(x, [d2l.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))
```

Như chúng ta đã thấy, tối ưu hóa cho học sâu là đầy thách thức. May mắn thay, tồn tại một loạt các thuật toán mạnh mẽ hoạt động tốt và dễ sử dụng ngay cả đối với người mới bắt đầu. Hơn nữa, nó không phải là thực sự cần thiết để tìm *giải pháp tốt nhất*. Optima địa phương hoặc thậm chí các giải pháp gần đúng của chúng vẫn còn rất hữu ích. 

## Tóm tắt

* Giảm thiểu lỗi đào tạo không * không* đảm bảo rằng chúng tôi tìm thấy bộ thông số tốt nhất để giảm thiểu lỗi tổng quát hóa.
* Các vấn đề tối ưu hóa có thể có nhiều minima cục bộ.
* Vấn đề có thể có nhiều điểm yên hơn, như nói chung các vấn đề không lồi.
* Độ dốc biến mất có thể gây ra tối ưu hóa để gian hàng. Thông thường một reparameterization của vấn đề giúp. Khởi tạo tốt các thông số cũng có thể có lợi.

## Bài tập

1. Hãy xem xét một MLP đơn giản với một lớp ẩn duy nhất của, ví dụ, $d$ kích thước trong lớp ẩn và một đầu ra duy nhất. Cho thấy rằng đối với bất kỳ mức tối thiểu địa phương có ít nhất $ d! $ các giải pháp tương đương hành xử giống hệt nhau.
1. Giả sử rằng chúng ta có một ma trận ngẫu nhiên đối xứng $\mathbf{M}$ nơi các mục $M_{ij} = M_{ji}$ mỗi được rút ra từ một số phân phối xác suất $p_{ij}$. Hơn nữa giả định rằng $p_{ij}(x) = p_{ij}(-x)$, tức là, rằng phân phối là đối xứng (xem ví dụ, :cite:`Wigner.1958` để biết chi tiết).
    1. Chứng minh rằng sự phân bố trên eigenvalues cũng là đối xứng. Đó là, đối với bất kỳ eigenvector $\mathbf{v}$ xác suất mà giá trị eigenvalue liên quan $\lambda$ thỏa mãn $P(\lambda > 0) = P(\lambda < 0)$.
    1. Tại sao *không* ở trên ngụ ý $P(\lambda > 0) = 0.5$?
1. Bạn có thể nghĩ đến những thách thức nào khác liên quan đến tối ưu hóa học tập sâu?
1. Giả sử rằng bạn muốn cân bằng một quả bóng (thực) trên một (thực) yên xe.
    1. Tại sao điều này khó?
    1. Bạn có thể khai thác hiệu ứng này cũng cho các thuật toán tối ưu hóa?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/349)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/487)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/489)
:end_tab:
