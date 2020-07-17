<!--
# Optimization and Deep Learning
-->

# Tối ưu và Học sâu

<!--
In this section, we will discuss the relationship between optimization and deep learning as well as the challenges of using optimization in deep learning.
For a deep learning problem, we will usually define a loss function first.
Once we have the loss function, we can use an optimization algorithm in attempt to minimize the loss.
In optimization, a loss function is often referred to as the objective function of the optimization problem.
By tradition and convention most optimization algorithms are concerned with *minimization*.
If we ever need to maximize an objective there is a simple solution: just flip the sign on the objective.
-->

Trong phần này, ta sẽ thảo luận mối quan hệ giữa tối ưu và học sâu, cũng như những thách thức khi áp dụng các thuật toán tối ưu trong học sâu.
Đối với một bài toán học sâu, đầu tiên chúng ta thường định nghĩa hàm mất mát, 
sau đó sử dụng một thuật toán tối ưu nhằm cực tiểu hóa hàm mất mát đó.
Hàm mất mát trong học sâu thường được xem là hàm mục tiêu của bài toán tối ưu.
Thông thường, đa số các thuật toán tối ưu thường giải quyết bài toán *cực tiểu hóa*.
Tuy nhiên, nếu ta cần cực đại hóa, có một cách khá đơn giản là đổi dấu hàm mục tiêu.

<!--
## Optimization and Estimation
-->

## Tối ưu và Ước lượng

<!--
Although optimization provides a way to minimize the loss function for deep learning, in essence, the goals of optimization and deep learning are fundamentally different.
The former is primarily concerned with minimizing an objective whereas the latter is concerned with finding a suitable model, given a finite amount of data. 
In :numref:`sec_model_selection`, we discussed the difference between these two goals in detail.
For instance, training error and generalization error generally differ: since the objective function of the optimization algorithm is usually a loss function 
based on the training dataset, the goal of optimization is to reduce the training error.
However, the goal of statistical inference (and thus of deep learning) is to reduce the generalization error. 
To accomplish the latter we need to pay attention to overfitting in addition to using the optimization algorithm to reduce the training error.
We begin by importing a few libraries for this chapter.
-->

Mặc dù các phương pháp tối ưu thường được sử dụng để cực tiểu hóa hàm mất mát trong học sâu, nhưng mục đích của tối ưu và học sâu về bản chất là khác nhau.
Mối quan tâm của tối ưu chủ yếu là cực tiểu hóa một mục tiêu nào đó, trong khi đối với học sâu là tìm kiếm một mô hình phù hợp với một lượng dữ liệu hữu hạn.
Trong :numref:`sec_model_selection`, ta đã thảo luận chi tiết về sự khác biệt giữa các mục đích trên.
Chẳng hạn như là sự khác biệt giữa lỗi huấn luyện và lỗi khái quát.
Do hàm mục tiêu của thuật toán tối ưu thường là hàm mất mát trên tập huấn luyện nên mục đích của tối ưu là giảm thiểu lỗi huấn luyện.
Tuy nhiên, mục đích của suy luận thống kê (*statistical inference*) và học sâu nói riêng là giảm thiểu lỗi khái quát.
Để thực hiện điều này, bên cạnh việc giảm thiểu lỗi huấn luyện, ta cần chú ý đến hiện tượng quá khớp.
Hãy bắt đầu bằng việc nhập một số thư viện sử dụng trong chương này.


```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()
```

<!--
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
-->

<!--
Next we define two functions, the expected function $f$ and the empirical function $g$, to illustrate this issue.
Here the $g$ is less smooth than $f$ since we have only a finite amount of data.
-->

Tiếp theo, ta định nghĩa hai hàm: hàm kỳ vọng $f$ và hàm thực nghiệm $g$ để minh họa vấn đề này.
Ở đây, $g$ kém mượt hơn $f$ vì ta chỉ có một lượng dữ liệu hữu hạn.


```{.python .input}
#@tab all
def f(x): return x * d2l.cos(np.pi * x)
def g(x): return f(x) + 0.2 * d2l.cos(5 * np.pi * x)
```


<!--
The graph below illustrates the issue in some more detail.
Since we have only a finite amount of data the minimum of the training error may be at a different location than the minimum of the expected error (or of the test error).
-->

Đồ thị phía dưới mô tả chi tiết hơn về vấn đề trên.
Do ta chỉ có một lượng dữ liệu hữu hạn, cực tiểu của lỗi huấn luyện có thể khác so với cực tiểu kỳ vọng của lỗi (lỗi trên tập kiểm tra).


```{.python .input}
#@tab all
def annotate(text, xy, xytext):  #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

x = d2l.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('empirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('expected risk', (1.1, -1.05), (0.95, -0.5))
```


<!--
## Optimization Challenges in Deep Learning
-->

# Các Thách thức của Tối ưu trong Học sâu

<!--
In this chapter, we are going to focus specifically on the performance of the optimization algorithm in minimizing the objective function, rather than a model's generalization error.
In :numref:`sec_linear_regression` we distinguished between analytical solutions and numerical solutions in optimization problems.
In deep learning, most objective functions are complicated and do not have analytical solutions.
Instead, we must use numerical optimization algorithms.
The optimization algorithms below all fall into this category.
-->

Ở chương này, ta sẽ chỉ tập trung vào chất lượng của thuật toán tối ưu trong việc cực tiểu hóa hàm mục tiêu, thay vì lỗi khái quát của mô hình.
Trong :numref:`sec_linear_regression`, ta đã phân biệt giữa nghiệm theo công thức và nghiệm xấp xỉ trong các bài toán tối ưu.
Trong học sâu, đa số các hàm mục tiêu khá phức tạp và không tính được nghiệm theo công thức.
Thay vào đó, ta phải dùng các thuật toán tối ưu xấp xỉ.
Các thuật toán tối ưu dưới đây được liệt vào loại này.

<!--
There are many challenges in deep learning optimization.
Some of the most vexing ones are local minima, saddle points and vanishing gradients.
Let us have a look at a few of them.
-->

Có rất nhiều thách thức về tối ưu trong học sâu.
Các điểm cực tiểu, điểm yên ngựa, tiêu biến gradient là một số vấn đề gây đau đầu hơn cả.
Hãy cùng tìm hiểu về các vấn đề này.

<!--
### Local Minima
-->

# Các vùng Cực tiểu

<!--
For the objective function $f(x)$, if the value of $f(x)$ at $x$ is smaller than the values of $f(x)$ at any other points in the vicinity of $x$, then $f(x)$ could be a local minimum.
If the value of $f(x)$ at $x$ is the minimum of the objective function over the entire domain, then $f(x)$ is the global minimum.
-->

Cho hàm mục tiêu $f(x)$, nếu giá trị của $f(x)$ tại $x$ nhỏ hơn các giá trị khác của $f(x)$ tại lân cận của $x$ thì $f(x)$ có thể là một *cực tiểu* (*local minimum*).
Nếu giá trị của hàm mục tiêu $f(x)$ tại $x$ là nhỏ nhất trên toàn tập xác định thì $f(x)$ là *giá trị nhỏ nhất* (*global minimum*).

<!--
For example, given the function
-->

Ví dụ, cho hàm

$$f(x) = x \cdot \text{cos}(\pi x) \text{ for } -1.0 \leq x \leq 2.0,$$


<!--
we can approximate the local minimum and global minimum of this function.
-->

ta có thể tính xấp xỉ cực tiểu và giá trị nhỏ nhất của hàm này.


```{.python .input}
#@tab all
x = d2l.arange(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
```


<!--
The objective function of deep learning models usually has many local optima.
When the numerical solution of an optimization problem is near the local optimum, the numerical solution obtained by the final iteration may only minimize the objective function locally,
rather than globally, as the gradient of the objective function's solutions approaches or becomes zero.
Only some degree of noise might knock the parameter out of the local minimum.
In fact, this is one of the beneficial properties of stochastic gradient descent where the natural variation of gradients over minibatches is able to dislodge the parameters from local minima.
-->

Hàm mục tiêu trong các mô hình học sâu thường có nhiều vùng cực trị.
Khi nghiệm xấp xỉ của một bài toán tối ưu đang ở gần giá trị cực tiểu, gradient của hàm mục tiêu tại nghiệm này gần hoặc bằng 0, tuy nhiên nghiệm này có thể chỉ đang cực tiểu hóa hàm mục tiêu một cách cục bộ chứ không phải toàn cục.
Chỉ với một mức độ nhiễu nhất định thì mới có thể đẩy tham số ra khỏi vùng cực tiểu.
Trên thực tế, nhiễu là một trong những tính chất có lợi của hạ gradient ngẫu nhiên khi sự biến động của gradient qua từng minibatch có thể đẩy các tham số ra khỏi các vùng cực tiểu.


<!--
### Saddle Points
-->

# Các điểm Yên ngựa

<!--
Besides local minima, saddle points are another reason for gradients to vanish.
A [saddle point](https://en.wikipedia.org/wiki/Saddle_point) is any location where all gradients of a function vanish but which is neither a global nor a local minimum.
Consider the function $f(x) = x^3$.
Its first and second derivative vanish for $x=0$.
Optimization might stall at the point, even though it is not a minimum.
-->

Ngoài các vùng cực tiểu, các điểm yên ngựa cũng có gradient bằng 0.
Một [điểm yên ngựa](https://en.wikipedia.org/wiki/Saddle_point) là bất cứ điểm nào mà tất cả gradient của một hàm bằng 0, nhưng đó không phải là một điểm cực tiểu hay giá trị nhỏ nhất.
Xét hàm $f(x) = x^3$, đạo hàm bậc một và bậc hai của hàm này bằng 0 tại điểm $x=0$.
Việc tối ưu có thể bị ngưng trệ tại điểm này, cho dù đó không phải là điểm cực tiểu.


```{.python .input}
#@tab all
x = d2l.arange(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
```


<!--
Saddle points in higher dimensions are even more insidious, as the example below shows.
Consider the function $f(x, y) = x^2 - y^2$.
It has its saddle point at $(0, 0)$.
This is a maximum with respect to $y$ and a minimum with respect to $x$.
Moreover, it *looks* like a saddle, which is where this mathematical property got its name.
-->

Các điểm yên ngựa trong không gian nhiều chiều còn khó chịu hơn nhiều, như ví dụ dưới đây.
Xét hàm $f(x, y) = x^2 - y^2$.
Hàm này tồn tại một điểm yên ngựa tại $(0, 0)$.
Đây là một điểm cực đại theo $y$ và là điểm cực tiểu theo $x$.
Tên gọi của tính chất toán học này bắt nguồn từ chính việc đồ thị tại đó có hình dạng giống một cái yên ngựa.


```{.python .input}
#@tab all
x, y = d2l.meshgrid(d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
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


<!--
We assume that the input of a function is a $k$-dimensional vector and its output is a scalar, so its Hessian matrix will have $k$ eigenvalues (refer to :numref:`sec_geometry-linear-algebraic-ops`).
The solution of the function could be a local minimum, a local maximum, or a saddle point at a position where the function gradient is zero:
-->

Giả sử đầu vào của một hàm là một vector $k$ chiều và đầu ra là một số vô hướng; do đó ma trận Hessian của nó có $k$ trị riêng (xem thêm tại :numref:`sec_geometry-linear-algebraic-ops`).
Nghiệm của hàm này có thể là một cực tiểu, cực đại, hoặc một điểm yên ngựa tại vị trí mà gradient của hàm bằng 0.

<!--
* When the eigenvalues of the function's Hessian matrix at the zero-gradient position are all positive, we have a local minimum for the function.
* When the eigenvalues of the function's Hessian matrix at the zero-gradient position are all negative, we have a local maximum for the function.
* When the eigenvalues of the function's Hessian matrix at the zero-gradient position are negative and positive, we have a saddle point for the function.
-->

* Điểm cực tiểu là vị trí mà ở đó gradient bằng 0 và các trị riêng của ma trận Hessian đều dương.
* Điểm cực đại là vị trí mà ở đó graident bằng 0 và các trị riêng của ma trận Hessian đều âm.
* Điểm yên ngựa là vị trí mà ở đó gradient bằng 0 và các trị riêng của ma trận Hessian mang cả giá trị âm lẫn dương.

<!--
For high-dimensional problems the likelihood that at least some of the eigenvalues are negative is quite high.
This makes saddle points more likely than local minima.
We will discuss some exceptions to this situation in the next section when introducing convexity.
In short, convex functions are those where the eigenvalues of the Hessian are never negative.
Sadly, though, most deep learning problems do not fall into this category.
Nonetheless it is a great tool to study optimization algorithms.
-->

Đối với bài toán trong không gian nhiều chiều, khả năng có một vài trị riêng âm là khá cao.
Do đó các điểm yên ngựa có khả năng xuất hiện cao hơn các cực tiểu.
Ta sẽ thảo luận một số ngoại lệ của vấn đề này ở phần tới khi giới thiệu đến tính lồi.
Nói ngắn gọn, các hàm lồi là hàm có các trị riêng của ma trận Hessian không bao giờ âm.
Nhưng không may, đa số bài toán học sâu đều không thuộc loại này.
Dù sao thì tính lồi vẫn là một công cụ tốt để học về các thuật toán tối ưu.


<!--
### Vanishing Gradients
-->

# Tiêu biến Gradient

<!--
Probably the most insidious problem to encounter are vanishing gradients.
For instance, assume that we want to minimize the function $f(x) = \tanh(x)$ and we happen to get started at $x = 4$.
As we can see, the gradient of $f$ is close to nil.
More specifically $f'(x) = 1 - \tanh^2(x)$ and thus $f'(4) = 0.0013$.
Consequently optimization will get stuck for a long time before we make progress.
This turns out to be one of the reasons that training deep learning models was quite tricky prior to the introduction of the ReLU activation function.
-->

Có lẽ vấn đế khó chịu nhất mà ta phải đối mặt là tiêu biến gradient.
Ví dụ, giả sử ta muốn cực tiểu hóa hàm $f(x) = \tanh(x)$ và ta bắt đầu tại $x = 4$.
Như ta có thể thấy, gradient của $f$ gần như là bằng 0.
Cụ thể, $f'(x) = 1 - \tanh^2(x)$ và do đó $f'(4) = 0.0013$.
Hậu quả là quá trình tối ưu sẽ bị trì trệ khá lâu trước khi có tiến triển.
Đây hóa ra lại là lý do tại sao việc huấn luyện các mô hình học sâu khá khó khăn trước khi hàm kích hoạt ReLU xuất hiện.


```{.python .input}
#@tab all
x = d2l.arange(-2.0, 5.0, 0.01)
d2l.plot(x, [d2l.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))
```


<!--
As we saw, optimization for deep learning is full of challenges.
Fortunately there exists a robust range of algorithms that perform well and that are easy to use even for beginners.
Furthermore, it is not really necessary to find *the* best solution.
Local optima or even approximate solutions thereof are still very useful.
-->

Tối ưu trong học sâu mang đầy thử thách.
May mắn thay, có khá nhiều thuật toán hoạt động tốt và dễ sử dụng ngay cả đối với người mới bắt đầu.
Hơn nữa, việc tìm kiếm giải pháp tốt *nhất* là không thật cần thiết.
Các cực tiểu và ngay cả nghiệm xấp xỉ cũng đã rất hữu dụng rồi.

## Tóm tắt

<!--
* Minimizing the training error does *not* guarantee that we find the best set of parameters to minimize the expected error.
* The optimization problems may have many local minima.
* The problem may have even more saddle points, as generally the problems are not convex.
* Vanishing gradients can cause optimization to stall. Often a reparameterization of the problem helps. Good initialization of the parameters can be beneficial, too.
-->

* Cực tiểu hóa lỗi huấn luyện *không* đảm bảo việc ta sẽ tìm ra tập tham số tốt nhất để cực tiểu hóa lỗi khái quát.
* Các bài toán tối ưu thường có nhiều vùng cực tiểu.
* Và do các bài toán thường không có tính lồi, số lượng điểm yên ngựa thậm chí có thể nhiều hơn.
* Tiêu biến gradient có thể khiến cho quá trình tối ưu bị đình trệ. Thường thì việc tái tham số hóa bài toán (*reparameterization*) và khởi tạo tham số cẩn thận cũng sẽ giúp ích.


## Bài tập

<!--
1. Consider a simple multilayer perceptron with a single hidden layer of, say, $d$ dimensions in the hidden layer and a single output.
Show that for any local minimum there are at least $d!$ equivalent solutions that behave identically.
2. Assume that we have a symmetric random matrix $\mathbf{M}$ where the entries $M_{ij} = M_{ji}$ are each drawn from some probability distribution $p_{ij}$.
Furthermore assume that $p_{ij}(x) = p_{ij}(-x)$, i.e., that the distribution is symmetric (see e.g., :cite:`Wigner.1958` for details).
    * Prove that the distribution over eigenvalues is also symmetric. 
    That is, for any eigenvector $\mathbf{v}$ the probability that the associated eigenvalue $\lambda$ satisfies $P(\lambda > 0) = P(\lambda < 0)$.
    * Why does the above *not* imply $P(\lambda > 0) = 0.5$?
3. What other challenges involved in deep learning optimization can you think of?
4. Assume that you want to balance a (real) ball on a (real) saddle.
    * Why is this hard?
    * Can you exploit this effect also for optimization algorithms?
-->

1. Xét một mạng perceptron đa tầng đơn giản với một tầng ẩn $d$ chiều và một đầu ra duy nhất.
Chỉ ra rằng bất kỳ cực tiểu nào cũng có ít nhất $d!$ nghiệm tương đương khiến mạng vận hành giống nhau.
2. Giả sử ta có một ma trận đối xứng $\mathbf{M}$ ngẫu nhiên, trong đó mỗi phần tử $M_{ij} = M_{ji}$ tuân theo phân phối xác suất $p_{ij}$. Ngoài ra, giả sử $p_{ij}(x) = p_{ij}(-x)$, tức phân phối là đối xứng (xem :cite:`Wigner.1958` để biết thêm chi tiết).
    * Chứng minh rằng phân phối của các trị riêng cũng là đối xứng, tức với mọi vector riêng $\mathbf{v}$, trị riêng $\lambda$ tương ứng thoả mãn $P(\lambda > 0) = P(\lambda < 0)$.
    * Tại sao điều trên *không* có nghĩa là $P(\lambda > 0) = 0.5$?
1. Liệu còn thử thách tối ưu nào trong học sâu không?
2. Giả sử bạn muốn cân bằng một quả bóng (thật) trên một chiếc yên ngựa (thật).
    * Tại sao điều này lại khó khăn?
    * Hãy vận dụng kết quả trên vào các thuật toán tối ưu.



## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/349)
* [Tiếng Anh - Pytorch](https://discuss.d2l.ai/t/487)
* [Tiếng Anh - Tensorflow](https://discuss.d2l.ai/t/489)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Đỗ Trường Giang
* Lê Khắc Hồng Phúc
* Nguyễn Văn Quang
* Phạm Minh Đức
* Nguyễn Văn Cường
* Phạm Hồng Vinh
