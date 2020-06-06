<!-- ===================== Bắt đầu dịch Phần  ==================== -->
<!-- ========================================= REVISE PHẦN  - BẮT ĐẦU =================================== -->

<!--
# Adadelta
-->

# *dịch tiêu đề phía trên*
:label:`sec_adadelta`

<!--
Adadelta is yet another variant of AdaGrad. The main difference lies in the fact that it decreases the amount by which the learning rate is adaptive to coordinates. Moreover, traditionally it referred to as not having a learning rate since it uses the amount of change itself as calibration for future change. The algorithm was proposed in :cite:`Zeiler.2012`. It is fairly straightforward, given the discussion of previous algorithms so far.
-->

*dịch đoạn phía trên*

<!--
## The Algorithm
-->

## *dịch tiêu đề phía trên*

<!--
In a nutshell Adadelta uses two state variables, $\mathbf{s}_t$ to store a leaky average of the second moment of the gradient and $\Delta\mathbf{x}_t$ to store a leaky average of the second moment of the change of parameters in the model itself. Note that we use the original notation and naming of the authors for compatibility with other publications and implementations (there is no other real reason why one should use different Greek variables to indicate a parameter serving the same purpose in momentum, Adagrad, RMSProp, and Adadelta). The parameter du jour is $\rho$. We obtain the following leaky updates:
-->

*dịch đoạn phía trên*

$$\begin{aligned}
    \mathbf{s}_t & = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2, \\
    \mathbf{g}_t' & = \sqrt{\frac{\Delta\mathbf{x}_{t-1} + \epsilon}{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t, \\
    \mathbf{x}_t  & = \mathbf{x}_{t-1} - \mathbf{g}_t', \\
    \Delta \mathbf{x}_t & = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) \mathbf{x}_t^2.
\end{aligned}$$
-->

*dịch đoạn phía trên*

<!--
The difference to before is that we perform updates with the rescaled gradient $\mathbf{g}_t'$ which is computed by taking the ratio between the average squared rate of change and the average second moment of the gradient. The use of $\mathbf{g}_t'$ is purely for notational convenience. In practice we can implement this algorithm without the need to use additional temporary space for $\mathbf{g}_t'$. As before $\eta$ is a parameter ensuring nontrivial numerical results, i.e., avoiding zero step size or infinite variance. Typically we set this to $\eta = 10^{-5}$.
-->

*dịch đoạn phía trên*

<!--
## Implementation
-->

## *dịch tiêu đề phía trên*

<!--
Adadelta needs to maintain two state variables for each variable, $\mathbf{s}_t$ and $\Delta\mathbf{x}_t$. This yields the following implementation.
-->

*dịch đoạn phía trên*

```{.python .input  n=11}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adadelta_states(feature_dim):
    s_w, s_b = np.zeros((feature_dim, 1)), np.zeros(1)
    delta_w, delta_b = np.zeros((feature_dim, 1)), np.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        # In-place updates via [:]
        s[:] = rho * s + (1 - rho) * np.square(p.grad)
        g = (np.sqrt(delta + eps) / np.sqrt(s + eps)) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```

<!--
Choosing $\rho = 0.9$ amounts to a half-life time of 10 for each parameter update. This tends to work quite well. We get the following behavior.
-->

*dịch đoạn phía trên*

```{.python .input  n=12}
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```

<!--
For a concise implementation we simply use the `adadelta` algorithm from the `Trainer` class. This yields the following one-liner for a much more compact invocation.
-->

*dịch đoạn phía trên*

```{.python .input  n=9}
d2l.train_gluon_ch11('adadelta', {'rho': 0.9}, data_iter)
```

<!--
## Summary
-->

## *dịch tiêu đề phía trên*

<!--
* Adadelta has no learning rate parameter. Instead, it uses the rate of change in the parameters itself to adapt the learning rate.
* Adadelta requires two state variables to store the second moments of gradient and the change in parameters.
* Adadelta uses leaky averages to keep a running estimate of the appropriate statistics.
-->

*dịch đoạn phía trên*

<!--
## Exercises
-->

## *dịch tiêu đề phía trên*

<!--
1. Adjust the value of $\rho$. What happens?
1. Show how to implement the algorithm without the use of $\mathbf{g}_t'$. Why might this be a good idea?
1. Is Adadelta really learning rate free? Could you find optimization problems that break Adadelta?
1. Compare Adadelta to Adagrad and RMS prop to discuss their convergence behavior.
-->

*dịch đoạn phía trên*

<!--
## [Discussions](https://discuss.mxnet.io/t/2377)
-->

## *dịch tiêu đề phía trên*

<!--
![](../img/qr_adadelta.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/qr_adadelta.svg)

<!-- ===================== Kết thúc dịch Phần  ==================== -->
<!-- ========================================= REVISE PHẦN  - KẾT THÚC ===================================-->

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
