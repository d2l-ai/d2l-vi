<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Momentum
-->

# Động lượng
:label:`sec_momentum`

<!--
In :numref:`sec_sgd` we reviewed what happens when performing stochastic gradient descent, i.e., when performing optimization where only a noisy variant of the gradient is available.
In particular, we noticed that for noisy gradients we need to be extra cautious when it comes to choosing the learning rate in the face of noise.
If we decrease it too rapidly, convergence stalls.
If we are too lenient, we fail to converge to a good enough solution since noise keeps on driving us away from optimality.
-->

Trong :numref:`sec_sgd` chúng ta đã ôn tập về kỹ thuật hạ gradient ngẫu nhiên, ví dụ như tối ưu hoá khi chỉ có một biến thể gây nhiễu gradient.
Cụ thể, cần chú ý với gradient nhiễu chúng ta cần cực kỳ cẩn trọng trong việc chọn tốc độ học khi có mặt của tác nhân gây nhiễu.
Nếu gradient giảm quá nhanh, việc hội tụ sẽ bị chững lại.
Nếu gradient giảm chậm, việc hội tụ tại một kết quả đủ tốt sẽ khó xảy ra bởi vì nhiễu sẽ đẩy điểm hội tụ ra xa điểm tối ưu.

<!--
## Basics
-->

## Kiến thức cơ bản

<!--
In this section, we will explore more effective optimization algorithms, especially for certain types of optimization problems that are common in practice.
-->

Trong phần này, chúng ta sẽ cùng nhau khám phá những thuật toán tối ưu hiệu quả, đặc biệt cho một số loại bài toán tối ưu cụ thể phổ biến trong thực tế.

<!--
### Leaky Averages
-->

### Giá trị trung bình rò rỉ

<!--
The previous section saw us discussing minibatch SGD as a means for accelerating computation.
It also had the nice side-effect that averaging gradients reduced the amount of variance.
-->

Trong phần trước, chúng ta đã thảo luận về hạ gradient ngẫu nhiên minibatch như một cách để tăng tốc độ tính toán.
Đồng thời, kỹ thuật này cũng có một "tác dụng phụ" tốt đó là lấy trung bình gradients giúp giảm phương sai.


$$\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}_{t-1}) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbf{g}_{i, t-1}.
$$


<!--
Here we used $\mathbf{g}_{ii} = \partial_{\mathbf{w}} f(\mathbf{x}_i, \mathbf{w}_t)$ to keep the notation simple.
It would be nice if we could benefit from the effect of variance reduction even beyond averaging gradients on a mini-batch.
One option to accomplish this task is to replace the gradient computation by a "leaky average":
-->

Ở đây chúng ta dùng $\mathbf{g}_{ii} = \partial_{\mathbf{w}} f(\mathbf{x}_i, \mathbf{w}_t)$ để giúp ký hiệu được đơn giản.
Tác động giảm phương sai rất có lợi cho chúng ta, thậm chí hơn cả việc lấy trung bình gradient trên từng minibatch.
Một phương pháp để đạt được điều này đó là thay thế việc tính toán gradient bằng một "trung bình rò rỉ": 


$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}$$


<!--
for some $\beta \in (0, 1)$. This effectively replaces the instantaneous gradient by one that's been averaged over multiple *past* gradients.
$\mathbf{v}$ is called *momentum*.
It accumulates past gradients similar to how a heavy ball rolling down the objective function landscape integrates over past forces.
To see what is happening in more detail let us expand $\mathbf{v}_t$ recursively into
-->

với $\beta \in (0, 1)$. Phương pháp này thay thế gradient tức thời một cách hiệu quả bằng một giá trị được lấy trung bình trên các gradient trước đó.
$\mathbf{v}$ đực gọi là *động lượng*.
Động lượng tích luỹ các gradients trong quá khứ tương tự như cách một quả bóng nặng lăn xuống ngọn đồi sẽ tích hợp hết tất cả các lực tác động từ điểm bắt đầu lăn tới điểm hiện tại.
Để thấy rõ hơn những gì đang diễn ra, chúng ta hãy mở rộng $\mathbf{v}_t$ thành


<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->


$$\begin{aligned}
\mathbf{v}_t = \beta^2 \mathbf{v}_{t-2} + \beta \mathbf{g}_{t-1, t-2} + \mathbf{g}_{t, t-1}
= \ldots, = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}.
\end{aligned}$$


<!--
Large $\beta$ amounts to a long-range average, whereas small $\beta$ amounts to only a slight correction relative to a gradient method.
The new gradient replacement no longer points into the direction of steepest descent on a particular instance any longer but rather in the direction of a weighted average of past gradients.
This allows us to realize most of the benefits of averaging over a batch without the cost of actually computing the gradients on it.
We will revisit this averaging procedure in more detail later.
-->

*dịch đoạn phía trên*

<!--
The above reasoning formed the basis for what is now known as *accelerated* gradient methods, such as gradients with momentum.
They enjoy the additional benefit of being much more effective in cases where the optimization problem is ill-conditioned 
(i.e., where there are some directions where progress is much slower than in others, resembling a narrow canyon).
Furthermore, they allow us to average over subsequent gradients to obtain more stable directions of descent.
Indeed, the aspect of acceleration even for noise-free convex problems is one of the key reasons why momentum works and why it works so well.
-->

*dịch đoạn phía trên*

<!--
As one would expect, due to its efficacy momentum is a well-studied subject in optimization for deep learning and beyond.
See e.g., the beautiful [expository article](https://distill.pub/2017/momentum/) by :cite:`Goh.2017` for an in-depth analysis and interactive animation.
It was proposed by :cite:`Polyak.1964`.
:cite:`Nesterov.2018` has a detailed theoretical discussion in the context of convex optimization.
Momentum in deep learning has been known to be beneficial for a long time.
See e.g., the discussion by :cite:`Sutskever.Martens.Dahl.ea.2013` for details.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
### An Ill-conditioned Problem
-->

### *dịch tiêu đề phía trên*

<!--
To get a better understanding of the geometric properties of the momentum method we revisit gradient descent, albeit with a significantly less pleasant objective function.
Recall that in :numref:`sec_gd` we used $f(\mathbf{x}) = x_1^2 + 2 x_2^2$, i.e., a moderately distorted ellipsoid objective.
We distort this function further by stretching it out in the $x_1$ direction via
-->

*dịch đoạn phía trên*


$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$


<!--
As before $f$ has its minimum at $(0, 0)$. This function is *very* flat in the direction of $x_1$.
Let us see what happens when we perform gradient descent as before on this new function.
We pick a learning rate of $0.4$.
-->

*dịch đoạn phía trên*


```{.python .input  n=3}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```


<!--
By construction, the gradient in the $x_2$ direction is *much* higher and changes much more rapidly than in the horizontal $x_1$ direction.
Thus we are stuck between two undesirable choices: if we pick a small learning rate we ensure that the solution does not diverge in the $x_2$ direction 
but we are saddled with slow convergence in the $x_1$ direction.
Conversely, with a large learning rate we progress rapidly in the $x_1$ direction but diverge in $x_2$.
The example below illustrates what happens even after a slight increase in learning rate from $0.4$ to $0.6$.
Convergence in the $x_1$ direction improves but the overall solution quality is much worse.
-->

*dịch đoạn phía trên*


```{.python .input  n=4}
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
### The Momentum Method
-->

### *dịch tiêu đề phía trên*

<!--
The momentum method allows us to solve the gradient descent problem described above.
Looking at the optimization trace above we might intuit that averaging gradients over the past would work well.
After all, in the $x_1$ direction this will aggregate well-aligned gradients, thus increasing the distance we cover with every step.
Conversely, in the $x_2$ direction where gradients oscillate, an aggregate gradient will reduce step size due to oscillations that cancel each other out.
Using $\mathbf{v}_t$ instead of the gradient $\mathbf{g}_t$ yields the following update equations:
-->

*dịch đoạn phía trên*


$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}, \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{v}_t.
\end{aligned}
$$


<!--
Note that for $\beta = 0$ we recover regular gradient descent.
Before delving deeper into the mathematical properties let us have a quick look at how the algorithm behaves in practice.
-->

*dịch đoạn phía trên*


```{.python .input  n=5}
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```


<!--
As we can see, even with the same learning rate that we used before, momentum still converges well.
Let us see what happens when we decrease the momentum parameter.
Halving it to $\beta = 0.25$ leads to a trajectory that barely converges at all.
Nonetheless, it is a lot better than without momentum (when the solution diverges).
-->

*dịch đoạn phía trên*


```{.python .input  n=11}
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```


<!--
Note that we can combine momentum with SGD and in particular, minibatch-SGD.
The only change is that in that case we replace the gradients $\mathbf{g}_{t, t-1}$ with $\mathbf{g}_t$.
Last, for convenience we initialize $\mathbf{v}_0 = 0$ at time $t=0$.
Let us look at what leaky averaging actually does to the updates.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
### Effective Sample Weight
-->

### *dịch tiêu đề phía trên*

<!--
Recall that $\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}$.
In the limit the terms add up to $\sum_{\tau=0}^\infty \beta^\tau = \frac{1}{1-\beta}$.
In other words, rather than taking a step of size $\eta$ in GD or SGD we take a step of size $\frac{\eta}{1-\beta}$ while at the same time, dealing with a potentially much better behaved descent direction.
These are two benefits in one.
To illustrate how weighting behaves for different choices of $\beta$ consider the diagram below.
-->

*dịch đoạn phía trên*


```{.python .input}
gammas = [0.95, 0.9, 0.6, 0]
d2l.set_figsize((3.5, 2.5))
for gamma in gammas:
    x = np.arange(40).asnumpy()
    d2l.plt.plot(x, gamma ** x, label='gamma = %.2f' % gamma)
d2l.plt.xlabel('time')
d2l.plt.legend();
```

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
## Practical Experiments
-->

## *dịch tiêu đề phía trên*

<!--
Let us see how momentum works in practice, i.e., when used within the context of a proper optimizer.
For this we need a somewhat more scalable implementation.
-->

*dịch đoạn phía trên*

<!--
### Implementation from Scratch
-->

### *dịch tiêu đề phía trên*

<!--
Compared with (minibatch) SGD the momentum method needs to maintain a set of  auxiliary variables, i.e., velocity.
It has the same shape as the gradients (and variables of the optimization problem).
In the implementation below we call these variables `states`.
-->

*dịch đoạn phía trên*


```{.python .input  n=13}
def init_momentum_states(feature_dim):
    v_w = np.zeros((feature_dim, 1))
    v_b = np.zeros(1)
    return (v_w, v_b)

def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum'] * v + p.grad
        p[:] -= hyperparams['lr'] * v
```


<!--
Let us see how this works in practice.
-->

*dịch đoạn phía trên*


```{.python .input  n=15}
def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
train_momentum(0.02, 0.5)
```


<!--
When we increase the momentum hyperparameter `momentum` to 0.9, it amounts to a significantly larger effective sample size of $\frac{1}{1 - 0.9} = 10$.
We reduce the learning rate slightly to $0.01$ to keep matters under control.
-->

*dịch đoạn phía trên*


```{.python .input  n=8}
train_momentum(0.01, 0.9)
```


<!--
Reducing the learning rate further addresses any issue of non-smooth optimization problems.
Setting it to $0.005$ yields good convergence properties.
-->

*dịch đoạn phía trên*


```{.python .input}
train_momentum(0.005, 0.9)
```

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!--
### Concise Implementation
-->

### *dịch tiêu đề phía trên*

<!--
There is very little to do in Gluon since the standard `sgd` solver already had momentum built in.
Setting matching parameters yields a very similar trajectory.
-->

*dịch đoạn phía trên*


```{.python .input  n=9}
d2l.train_gluon_ch11('sgd', {'learning_rate': 0.005, 'momentum': 0.9},
                     data_iter)
```


<!--
## Theoretical Analysis
-->

## *dịch tiêu đề phía trên*

<!--
So far the 2D example of $f(x) = 0.1 x_1^2 + 2 x_2^2$ seemed rather contrived.
We will now see that this is actually quite representative of the types of problem one might encounter, at least in the case of minimizing convex quadratic objective functions.
-->

*dịch đoạn phía trên*

<!--
### Quadratic Convex Functions
-->

### *dịch tiêu đề phía trên*

<!--
Consider the function
-->

*dịch đoạn phía trên*


$$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b.$$


<!--
This is a general quadratic function.
For positive semidefinite matrices $\mathbf{Q} \succ 0$, i.e., for matrices with positive eigenvalues 
this has a minimizer at $\mathbf{x}^* = -\mathbf{Q}^{-1} \mathbf{c}$ with minimum value $b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$.
Hence we can rewrite $h$ as
-->

*dịch đoạn phía trên*


$$h(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})^\top \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c}) + b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}.$$


<!--
The gradient is given by $\partial_{\mathbf{x}} f(\mathbf{x}) = \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$.
That is, it is given by the distance between $\mathbf{x}$ and the minimizer, multiplied by $\mathbf{Q}$.
Consequently also the momentum  is a linear combination of terms $\mathbf{Q} (\mathbf{x}_t - \mathbf{Q}^{-1} \mathbf{c})$.
-->

*dịch đoạn phía trên*

<!--
Since $\mathbf{Q}$ is positive definite it can be decomposed into its eigensystem via 
$\mathbf{Q} = \mathbf{O}^\top \boldsymbol{\Lambda} \mathbf{O}$ for an orthogonal (rotation) matrix $\mathbf{O}$ and a diagonal matrix $\boldsymbol{\Lambda}$ of positive eigenvalues.
This allows us to perform a change of variables from $\mathbf{x}$ to $\mathbf{z} := \mathbf{O} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$ to obtain a much simplified expression:
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 7 ===================== -->


$$h(\mathbf{z}) = \frac{1}{2} \mathbf{z}^\top \boldsymbol{\Lambda} \mathbf{z} + b'.$$


<!--
Here $c' = b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$.
Since $\mathbf{O}$ is only an orthogonal matrix this does not perturb the gradients in a meaningful way.
Expressed in terms of $\mathbf{z}$ gradient descent becomes
-->

*dịch đoạn phía trên*


$$\mathbf{z}_t = \mathbf{z}_{t-1} - \boldsymbol{\Lambda} \mathbf{z}_{t-1} = (\mathbf{I} - \boldsymbol{\Lambda}) \mathbf{z}_{t-1}.$$


<!--
The important fact in this expression is that gradient descent *does not mix* between different eigenspaces.
That is, when expressed in terms of the eigensystem of $\mathbf{Q}$ the optimization problem proceeds in a coordinate-wise manner.
This also holds for momentum.
-->

*dịch đoạn phía trên*


$$\begin{aligned}
\mathbf{v}_t & = \beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1} \\
\mathbf{z}_t & = \mathbf{z}_{t-1} - \eta \left(\beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1}\right) \\
    & = (\mathbf{I} - \eta \boldsymbol{\Lambda}) \mathbf{z}_{t-1} - \eta \beta \mathbf{v}_{t-1}.
\end{aligned}$$


<!--
In doing this we just proved the following theorem: Gradient Descent with and without momentum for a convex quadratic function decomposes 
into coordinate-wise optimization in the direction of the eigenvectors of the quadratic matrix.
-->

*dịch đoạn phía trên*

<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 4 - BẮT ĐẦU ===================================-->

<!--
### Scalar Functions
-->

### *dịch tiêu đề phía trên*

<!--
Given the above result let us see what happens when we minimize the function $f(x) = \frac{\lambda}{2} x^2$. For gradient descent we have
-->

*dịch đoạn phía trên*


$$x_{t+1} = x_t - \eta \lambda x_t = (1 - \eta \lambda) x_t.$$


<!--
Whenever $|1 - \eta \lambda| < 1$ this optimization converges at an exponential rate since after $t$ steps we have $x_t = (1 - \eta \lambda)^t x_0$.
This shows how the rate of convergence improves initially as we increase the learning rate $\eta$ until $\eta \lambda = 1$.
Beyond that things diverge and for $\eta \lambda > 2$ the optimization problem diverges.
-->

*dịch đoạn phía trên*


```{.python .input}
lambdas = [0.1, 1, 10, 19]
eta = 0.1
d2l.set_figsize((6, 4))
for lam in lambdas:
    t = np.arange(20).asnumpy()
    d2l.plt.plot(t, (1 - eta * lam) ** t, label='lambda = %.2f' % lam)
d2l.plt.xlabel('time')
d2l.plt.legend();
```


<!--
To analyze convergence in the case of momentum we begin by rewriting the update equations in terms of two scalars: one for $x$ and one for the momentum $v$. This yields:
-->

*dịch đoạn phía trên*


$$
\begin{bmatrix} v_{t+1} \\ x_{t+1} \end{bmatrix} =
\begin{bmatrix} \beta & \lambda \\ -\eta \beta & (1 - \eta \lambda) \end{bmatrix}
\begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix} = \mathbf{R}(\beta, \eta, \lambda) \begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix}.
$$

<!-- ===================== Kết thúc dịch Phần 7 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 8 ===================== -->

<!--
We used $\mathbf{R}$ to denote the $2 \times 2$ governing convergence behavior.
After $t$ steps the initial choice $[v_0, x_0]$ becomes $\mathbf{R}(\beta, \eta, \lambda)^t [v_0, x_0]$.
Hence, it is up to the eigenvalues of $\mathbf{R}$ to detmine the speed of convergence.
See the [Distill post](https://distill.pub/2017/momentum/) of :cite:`Goh.2017` for a great animation and :cite:`Flammarion.Bach.2015` for a detailed analysis.
One can show that $0 < \eta \lambda < 2 + 2 \beta$ momentum converges.
This is a larger range of feasible parameters when compared to $0 < \eta \lambda < 2$ for gradient descent.
It also suggests that in general large values of $\beta$ are desirable.
Further details require a fair amount of technical detail and we suggest that the interested reader consult the original publications.
-->

*dịch đoạn phía trên*

<!--
## Summary
-->

## Tóm tắt

<!--
* Momentum replaces gradients with a leaky average over past gradients. This accelerates convergence significantly.
* It is desirable for both noise-free gradient descent and (noisy) stochastic gradient descent.
* Momentum prevents stalling of the optimization process that is much more likely to occur for stochastic gradient descent.
* The effective number of gradients is given by $\frac{1}{1-\beta}$ due to exponentiated downweighting of past data.
* In the case of convex quadratic problems this can be analyzed explicitly in detail.
* Implementation is quite straightforward but it requires us to store an additional state vector (momentum $\mathbf{v}$).
-->

*dịch đoạn phía trên*

<!--
## Exercises
-->

## Bài tập

<!--
1. Use other combinations of momentum hyperparameters and learning rates and observe and analyze the different experimental results.
2. Try out GD and momentum for a quadratic problem where you have multiple eigenvalues, i.e., $f(x) = \frac{1}{2} \sum_i \lambda_i x_i^2$, e.g., $\lambda_i = 2^{-i}$.
Plot how the values of $x$ decrease for the initialization $x_i = 1$.
3. Derive minimum value and minimizer for $h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b$.
4. What changes when we perform SGD with momentum? What happens when we use mini-batch SGD with momentum? Experiment with the parameters?
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 8 ===================== -->
<!-- ========================================= REVISE PHẦN 4 - KẾT THÚC ===================================-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2374)
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
* Nguyễn Thanh Hoà

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
