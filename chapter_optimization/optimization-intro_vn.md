<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Optimization and Deep Learning
-->

# *dịch tiêu đề phía trên*

<!--
In this section, we will discuss the relationship between optimization and deep learning as well as the challenges of using optimization in deep learning.
For a deep learning problem, we will usually define a loss function first.
Once we have the loss function, we can use an optimization algorithm in attempt to minimize the loss.
In optimization, a loss function is often referred to as the objective function of the optimization problem.
By tradition and convention most optimization algorithms are concerned with *minimization*.
If we ever need to maximize an objective there is a simple solution: just flip the sign on the objective.
-->

*dịch đoạn phía trên*


<!--
## Optimization and Estimation
-->

## *dịch tiêu đề phía trên*

<!--
Although optimization provides a way to minimize the loss function for deep learning, in essence, the goals of optimization and deep learning are fundamentally different.
The former is primarily concerned with minimizing an objective whereas the latter is concerned with finding a suitable model, given a finite amount of data. 
In :numref:`sec_model_selection`, we discussed the difference between these two goals in detail.
For instance, training error and generalization error generally differ: since the objective function of the optimization algorithm is usually a loss function 
based on the training dataset, the goal of optimization is to reduce the training error.
However, the goal of statistical inference (and thus of deep learning) is to reduce the generalization error. 
To accomplish the latter we need to pay attention to overfitting in addition to using the optimization algorithm to reduce the training error.
We begin by importing a few libraries with a function to annotate in a figure.
-->

*dịch đoạn phía trên*


```{.python .input  n=1}
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()

#@save
def annotate(text, xy, xytext):
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))
```

<!--
The graph below illustrates the issue in some more detail.
Since we have only a finite amount of data the minimum of the training error may be at a different location than the minimum of the expected error (or of the test error).
-->

*dịch đoạn phía trên*


```{.python .input  n=2}
def f(x): return x * np.cos(np.pi * x)
def g(x): return f(x) + 0.2 * np.cos(5 * np.pi * x)

d2l.set_figsize((4.5, 2.5))
x = np.arange(0.5, 1.5, 0.01)
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('empirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('expected risk', (1.1, -1.05), (0.95, -0.5))
```

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Optimization Challenges in Deep Learning
-->

# *dịch tiêu đề phía trên*

<!--
In this chapter, we are going to focus specifically on the performance of the optimization algorithm in minimizing the objective function, rather than a model's generalization error.
In :numref:`sec_linear_regression` we distinguished between analytical solutions and numerical solutions in optimization problems.
In deep learning, most objective functions are complicated and do not have analytical solutions. Instead, we must use numerical optimization algorithms.
The optimization algorithms below all fall into this category.
-->

*dịch đoạn phía trên*

<!--
There are many challenges in deep learning optimization.
Some of the most vexing ones are local minima, saddle points and vanishing gradients.
Let us have a look at a few of them.
-->

*dịch đoạn phía trên*

<!--
### Local Minima
-->

# *dịch tiêu đề phía trên*

<!--
For the objective function $f(x)$, if the value of $f(x)$ at $x$ is smaller than the values of $f(x)$ at any other points in the vicinity of $x$, then $f(x)$ could be a local minimum.
If the value of $f(x)$ at $x$ is the minimum of the objective function over the entire domain, then $f(x)$ is the global minimum.
-->

*dịch đoạn phía trên*

<!--
For example, given the function
-->

*dịch đoạn phía trên*

$$f(x) = x \cdot \text{cos}(\pi x) \text{ for } -1.0 \leq x \leq 2.0,$$


<!--
we can approximate the local minimum and global minimum of this function.
-->

*dịch đoạn phía trên*


```{.python .input  n=3}
x = np.arange(-1.0, 2.0, 0.01)
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

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
### Saddle Points
-->

# *dịch tiêu đề phía trên*

<!--
Besides local minima, saddle points are another reason for gradients to vanish.
A [saddle point](https://en.wikipedia.org/wiki/Saddle_point) is any location where all gradients of a function vanish but which is neither a global nor a local minimum.
Consider the function $f(x) = x^3$.
Its first and second derivative vanish for $x=0$.
Optimization might stall at the point, even though it is not a minimum.
-->

*dịch đoạn phía trên*


```{.python .input  n=4}
x = np.arange(-2.0, 2.0, 0.01)
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

*dịch đoạn phía trên*


```{.python .input  n=5}
x, y = np.meshgrid(np.linspace(-1, 1, 101), np.linspace(-1, 1, 101),
                   indexing='ij')

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

*dịch đoạn phía trên*

<!--
* When the eigenvalues of the function's Hessian matrix at the zero-gradient position are all positive, we have a local minimum for the function.
* When the eigenvalues of the function's Hessian matrix at the zero-gradient position are all negative, we have a local maximum for the function.
* When the eigenvalues of the function's Hessian matrix at the zero-gradient position are negative and positive, we have a saddle point for the function.
-->

*dịch đoạn phía trên*

<!--
For high-dimensional problems the likelihood that at least some of the eigenvalues are negative is quite high.
This makes saddle points more likely than local minima.
We will discuss some exceptions to this situation in the next section when introducing convexity.
In short, convex functions are those where the eigenvalues of the Hessian are never negative.
Sadly, though, most deep learning problems do not fall into this category.
Nonetheless it is a great tool to study optimization algorithms.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

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

Có lẽ vấn đế quỷ quyệt nhất mà ta phải đối mặt là tiêu biến gradient.
Ví dụ, giả sử ta muốn tối thiểu hàm $f(x) = \tanh(x)$ và ta bắt đầu tại $x = 4$.
Như ta có thể thấy, gradient của $f$ gần như là bằng 0.
Cụ thể, $f'(x) = 1 - \tanh^2(x)$ và do đó $f'(4) = 0.0013$.
Hậu quả là quá trình tối ưu sẽ bị trì trệ khá lâu trước khi có tiến triển.
Đây hoá ra là lý do tại sao huấn luyện các mô hình học sâu khá khó khăn trước khi hàm kích hoạt ReLU được ra mắt.

```{.python .input  n=6}
x = np.arange(-2.0, 5.0, 0.01)
d2l.plot(x, [np.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))
```


<!--
As we saw, optimization for deep learning is full of challenges.
Fortunately there exists a robust range of algorithms that perform well and that are easy to use even for beginners.
Furthermore, it is not really necessary to find *the* best solution.
Local optima or even approximate solutions thereof are still very useful.
-->

Tối ưu trong học sâu mang đầy thử thách.
May mắn thay, có một lượng lớn các thuật toán hoạt động tốt và dễ sử dụng ngay cả đối với người mới bắt đầu.
Hơn nữa, việc tìm kiếm giải pháp tốt *nhất* là không thực sự cần thiết.
Các cực tiểu và ngay cả nghiệm xấp xỉ của nó cũng rất hữu dụng.

## Tóm tắt

<!--
* Minimizing the training error does *not* guarantee that we find the best set of parameters to minimize the expected error.
* The optimization problems may have many local minima.
* The problem may have even more saddle points, as generally the problems are not convex.
* Vanishing gradients can cause optimization to stall. Often a reparameterization of the problem helps. Good initialization of the parameters can be beneficial, too.
-->

* Tối thiểu lỗi huấn luyện *không* đảm bảo việc ta sẽ tìm ra tập tham số tốt nhất để tối thiểu lỗi ta mong muốn.
* Các bài toán tối ưu thường có nhiều vùng cực tiểu.
* Bài toán còn có thể có nhiều điểm yên ngựa hơn nữa, do các bài toán thường không có tính lồi.
* Tiêu biến gradient có thể khiến cho quá trình tối ưu bị đình trệ. Thường thì việc tái tham số hoá bài toán (*reparameterization*) sẽ giúp ích. Việc khởi tạo tốt tập tham số cũng có thể có ích.


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
Chỉ ra rằng bất kì cực tiểu nào cũng có ít nhất $d!$ nghiệm tương đương, vận hành giống nhau.
2. Giả sử ta có một ma trận đối xứng $\mathbf{M}$ ngẫu nhiên, mỗi phần tử $M_{ij} = M_{ji}$ tuân theo phân phối xác suất $p_{ij}$.
Ngoài ra, giả sử $p_{ij}(x) = p_{ij}(-x)$, tức phân phối là đối xứng (xem ví dụ :cite:`Wigner.1958` để biết thêm chi tiết).
    * Chứng minh rằng phân phối của các trị riêng cũng là đối xứng.
    Hay, với mọi vector riêng $\mathbf{v}$, xác suất giá trị riêng $\lambda$ tương ứng thoả mãn $P(\lambda > 0) = P(\lambda < 0)$.
    * Tại sao điều trên *không* ám chỉ $P(\lambda > 0) = 0.5$?
3. Liệu còn thử thách nào tối ưu trong học sâu mà bạn có thể nghĩ tới?
4. Giả sử bạn muốn cân bằng một quả bóng (thật) trên một chiếc yên ngựa (thật).
    * Tại sao điều này lại khó khăn đến vậy?
    * Bạn có thể tận dụng kết quả trên cho các thuật toán tối ưu?

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->
<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->


## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2371)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.

* Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md
-->

* Đoàn Võ Duy Thanh
<!-- Phần 1 -->
* 

<!-- Phần 2 -->
* 

<!-- Phần 3 -->
* 

<!-- Phần 4 -->
* Đỗ Trường Giang
* Lê Khắc Hồng Phúc
