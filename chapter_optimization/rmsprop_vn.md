<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->

<!--
# RMSProp
-->

# RMSProp
:label:`sec_rmsprop`

<!--
One of the key issues in :numref:`sec_adagrad` is that the learning rate decreases at a predefined schedule of effectively $\mathcal{O}(t^{-\frac{1}{2}})$.
While this is generally appropriate for convex problems, it might not be ideal for nonconvex ones, such as those encountered in deep learning.
Yet, the coordinate-wise adaptivity of Adagrad is highly desirable as a preconditioner.
-->

Một trong những vấn đề then chốt trong :numref:`sec_adagrad` là tốc độ học giảm theo một định thời được định nghĩa sẵn $\mathcal{O}(t^{-\frac{1}{2}})$ một cách hiệu quả.
Nhìn chung, cách này thích hợp với các bài toán lồi nhưng có thể không phải giải pháp lý tưởng cho những bài toán không lồi, chẳng hạn những bài toán gặp phải trong học sâu.
Tuy vậy, khả năng thay đổi theo toạ độ của Adagrad là rất mong muốn như một điều kiện tiên quyết (_preconditioner_).

<!--
:cite:`Tieleman.Hinton.2012` proposed the RMSProp algorithm as a simple fix to decouple rate scheduling from coordinate-adaptive learning rates.
The issue is that Adagrad accumulates the squares of the gradient $\mathbf{g}_t$ into a state vector $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$.
As a result $\mathbf{s}_t$ keeps on growing without bound due to the lack of normalization, essentially linarly as the algorithm converges.
-->

:cite:`Tieleman.Hinton.2012` đề xuất thuật toán RMSProp như một bản vá đơn giản để tách rời tốc độ định thời ra khỏi tốc độ học thay đổi theo toạ độ (_coordinate-adaptive_).
Vấn đề ở đây là Adagrad tính tổng bình phương của các gradient $\mathbf{g}_t$ được lưu trong vector trạng thái $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$.
Kết quả là, do không có phép chuẩn hoá, $\mathbf{s}_t$ vẫn tiếp tục tăng tuyến tính không ngừng trong quá trình hội tụ của thuật toán.

<!--
One way of fixing this problem would be to use $\mathbf{s}_t / t$.
For reasonable distributions of $\mathbf{g}_t$ this will converge.
Unfortunately it might take a very long time until the limit behavior starts to matter since the procedure remembers the full trajectory of values.
An alternative is to use a leaky average in the same way we used in the momentum method, i.e., $\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$ for some parameter $\gamma > 0$.
Keeping all other parts unchanged yields RMSProp.
-->

Vấn đề này có thể được giải quyết bằng cách sử dụng $\mathbf{s}_t / t$.
Đối với các phân phối có ý nghĩa của $\mathbf{g}_t$, thuật toán sẽ hội tụ.
Đáng tiếc là có thể mất rất nhiều thời gian cho đến khi giới hạn bắt đầu hoạt động có ý nghĩa vì thuật toán này ghi nhớ toàn bộ quỹ đạo của các giá trị.
Một cách khác là sử dụng trung bình rò tương tự như cách chúng ta sử dụng trong phương pháp động lượng, tức là $\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$ cho các tham số $\gamma > 0$.
Giữ nguyên tất cả các phần khác ta có thuật toán RMSProp.

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## The Algorithm
-->

## *dịch tiêu đề phía trên*

<!--
Let us write out the equations in detail.
-->

*dịch đoạn phía trên*


$$\begin{aligned}
    \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2, \\
    \mathbf{x}_t & \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.
\end{aligned}$$


<!--
The constant $\epsilon > 0$ is typically set to $10^{-6}$ to ensure that we do not suffer from division by zero or overly large step sizes.
Given this expansion we are now free to control the learning rate $\eta$ independently of the scaling that is applied on a per-coordinate basis.
In terms of leaky averages we can apply the same reasoning as previously applied in the case of the momentum method.
Expanding the definition of $\mathbf{s}_t$ yields
-->

*dịch đoạn phía trên*


$$
\begin{aligned}
\mathbf{s}_t & = (1 - \gamma) \mathbf{g}_t^2 + \gamma \mathbf{s}_{t-1} \\
& = (1 - \gamma) \left(\mathbf{g}_t^2 + \gamma \mathbf{g}_{t-1}^2 + \gamma^2 \mathbf{g}_{t-2} + \ldots, \right).
\end{aligned}
$$


<!--
As before in :numref:`sec_momentum` we use $1 + \gamma + \gamma^2 + \ldots, = \frac{1}{1-\gamma}$.
Hence the sum of weights is normalized to $1$ with a half-life time of an observation of $\gamma^{-1}$.
Let us visualize the weights for the past 40 timesteps for various choices of $\gamma$.
-->

*dịch đoạn phía trên*


```{.python .input  n=1}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx

npx.set_np()
d2l.set_figsize((3.5, 2.5))

gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = np.arange(40).asnumpy()
    d2l.plt.plot(x, (1-gamma) * gamma ** x, label='gamma = %.2f' % gamma)
d2l.plt.xlabel('time');
```


<!--
## Implementation from Scratch
-->

## *dịch tiêu đề phía trên*

<!--
As before we use the quadratic function $f(\mathbf{x})=0.1x_1^2+2x_2^2$ to observe the trajectory of RMSProp.
Recall that in :numref:`sec_adagrad`, when we used Adagrad with a learning rate of 0.4, 
the variables moved only very slowly in the later stages of the algorithm since the learning rate decreased too quickly.
Since $\eta$ is controlled separately this does not happen with RMSProp.
-->

*dịch đoạn phía trên*


```{.python .input}
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```


<!--
Next, we implement RMSProp to be used in a deep network.
This is equally straightforward.
-->

*dịch đoạn phía trên*


```{.python .input  n=22}
def init_rmsprop_states(feature_dim):
    s_w = np.zeros((feature_dim, 1))
    s_b = np.zeros(1)
    return (s_w, s_b)

def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s[:] = gamma * s + (1 - gamma) * np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```


<!--
We set the initial learning rate to 0.01 and the weighting term $\gamma$ to 0.9.
That is, $\mathbf{s}$ aggregates on average over the past $1/(1-\gamma) = 10$ observations of the square gradient.
-->

*dịch đoạn phía trên*


```{.python .input  n=24}
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);
```

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
## Concise Implementation
-->

## *dịch tiêu đề phía trên*

<!--
Since RMSProp is a rather popular algorithm it is also available in the `Trainer` instance.
All we need to do is instantiate it using an algorithm named `rmsprop`, assigning $\gamma$ to the parameter `gamma1`.
-->

*dịch đoạn phía trên*


```{.python .input  n=29}
d2l.train_gluon_ch11('rmsprop', {'learning_rate': 0.01, 'gamma1': 0.9},
                     data_iter)
```


<!--
## Summary
-->

## Tóm tắt

<!--
* RMSProp is very similar to Adagrad insofar as both use the square of the gradient to scale coefficients.
* RMSProp shares with momentum the leaky averaging. However, RMSProp uses the technique to adjust the coefficient-wise preconditioner.
* The learning rate needs to be scheduled by the experimenter in practice.
* The coefficient $\gamma$ determines how long the history is when adjusting the per-coordinate scale.
-->

*dịch đoạn phía trên*

<!--
## Exercises
-->

## Bài tập

<!--
1. What happens experimentally if we set $\gamma = 1$? Why?
2. Rotate the optimization problem to minimize $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$. What happens to the convergence?
3. Try out what happens to RMSProp on a real machine learning problem, such as training on Fashion-MNIST. Experiment with different choices for adjusting the learning rate.
4. Would you want to adjust $\gamma$ as optimization progresses? How sensitive is RMSProp to this?
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->


## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2376)
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
* Nguyễn Văn Quang

<!-- Phần 2 -->
* 

<!-- Phần 3 -->
* 
