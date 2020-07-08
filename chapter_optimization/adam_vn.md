<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Adam
-->

# Adam
:label:`sec_adam`

<!--
In the discussions leading up to this section we encountered a number of techniques for efficient optimization.
Let us recap them in detail here:
-->

Trong các thảo luận dẫn đến phần này, chúng ta đã làm quen với một số kỹ thuật để tối ưu hóa hiệu quả.
Hãy cùng tóm tắt chi tiết những kỹ thuật này ở đây:

<!--
* We saw that :numref:`sec_sgd` is more effective than Gradient Descent when solving optimization problems, e.g., due to its inherent resilience to redundant data.
* We saw that :numref:`sec_minibatch_sgd` affords significant additional efficiency arising from vectorization, using larger sets of observations in one minibatch. 
This is the key to efficient multi-machine, multi-GPU and overall parallel processing.
* :numref:`sec_momentum` added a mechanism for aggregating a history of past gradients to accelerate convergence.
* :numref:`sec_adagrad` used per-coordinate scaling to allow for a computationally efficient preconditioner.
* :numref:`sec_rmsprop` decoupled per-coordinate scaling from a learning rate adjustment.
-->

* Chúng ta thấy rằng :numref:`sec_sgd` hiệu quả hơn thuật toán hạ gradient khi giải các bài toán tối ưu do nó chịu ít ảnh hưởng xấu gây ra bởi dữ liệu dư thừa.
* Chúng ta thấy rằng :numref:`sec_minibatch_sgd` mang lại hiệu quả đáng kể nhờ việc vector hóa, tức xử lý nhiều mẫu quan sát hơn trong một minibatch.
Đây là chìa khóa để xử lý dữ liệu song song trên nhiều GPU và nhiều máy tính một cách hiệu quả.
* :numref:`sec_momentum` bổ sung cơ chế gộp các gradient quá khứ, giúp quá trình hội tụ diễn ra nhanh hơn.
* :numref:`sec_adagrad` sử dụng phép chuyển đổi giá trị theo từng tọa độ để tạo ra tiền điều kiện hiệu quả về mặt tính toán.
* :numref:`sec_rmsprop` tách rời phép chuyển đổi giá trị theo từng tọa độ và phép điều chỉnh tốc độ học.

<!--
Adam :cite:`Kingma.Ba.2014` combines all these techniques into one efficient learning algorithm.
As expected, this is an algorithm that has become rather popular as one of the more robust and effective optimization algorithms to use in deep learning.
It is not without issues, though.
In particular, :cite:`Reddi.Kale.Kumar.2019` show that there are situations where Adam can diverge due to poor variance control.
In a follow-up work :cite:`Zaheer.Reddi.Sachan.ea.2018` proposed a hotfix to Adam, called Yogi which addresses these issues.
More on this later. For now let us review the Adam algorithm.
-->

Adam :cite:`Kingma.Ba.2014` kết hợp tất cả các kỹ thuật trên thành một thuật toán học hiệu quả.
Như kỳ vọng, đây là một trong những thuật toán tối ưu mạnh mẽ và hiệu quả được sử dụng phổ biến trong học sâu.
Tuy nhiên nó cũng có một vài điểm yếu.
Cụ thể, :cite:`Reddi.Kale.Kumar.2019` đã chỉ ra những trường hợp mà Adam có thể phân kỳ do việc kiểm soát phương sai kém.
Trong một nghiên cứu sau đó, :cite:`Zaheer.Reddi.Sachan.ea.2018` đã đề xuất Yogi, một bản vá nhanh cho Adam để giải quyết các vấn đề này.
Chi tiết về bản vá này sẽ được đề cập sau, còn bây giờ hãy xem xét thuật toán Adam.

<!--
## The Algorithm
-->

## Thuật toán

<!--
One of the key components of Adam is that it uses exponential weighted moving averages (also known as leaky averaging) 
to obtain an estimate of both the momentum and also the second moment of the gradient. 
That is, it uses the state variables
-->

Một trong những thành phần chính của Adam là các trung bình động theo trọng số bậc luỹ thừa (hay còn được gọi là trung bình rò rỉ)
để ước lượng cả động lượng và mô-men bậc hai của gradient.
Cụ thể, nó sử dụng các biến trạng thái

$$\begin{aligned}
    \mathbf{v}_t & \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \\
    \mathbf{s}_t & \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2.
\end{aligned}$$

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
Here $\beta_1$ and $\beta_2$ are nonnegative weighting parameters.
Common choices for them are $\beta_1 = 0.9$ and $\beta_2 = 0.999$.
That is, the variance estimate moves *much more slowly* than the momentum term.
Note that if we initialize $\mathbf{v}_0 = \mathbf{s}_0 = 0$ we have a significant amount of bias initially towards smaller values.
This can be addressed by using the fact that $\sum_{i=0}^t \beta^i = \frac{1 - \beta^t}{1 - \beta}$ to re-normalize terms.
Correspondingly the normalized state variables are given by
-->

*dịch đoạn phía trên*


$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \text{ and } \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}.$$


<!--
Armed with the proper estimates we can now write out the update equations.
First, we rescale the gradient in a manner very much akin to that of RMSProp to obtain
-->

*dịch đoạn phía trên*


$$\mathbf{g}_t' = \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}.$$


<!--
Unlike RMSProp our update uses the momentum $\hat{\mathbf{v}}_t$ rather than the gradient itself.
Moreover, there is a slight cosmetic difference as the rescaling happens using $\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$ instead of $\frac{1}{\sqrt{\hat{\mathbf{s}}_t + \epsilon}}$.
The former works arguably slightly better in practice, hence the deviation from RMSProp.
Typically we pick $\epsilon = 10^{-6}$ for a good trade-off between numerical stability and fidelity.
-->

*dịch đoạn phía trên*

<!--
Now we have all the pieces in place to compute updates.
This is slightly anticlimactic and we have a simple update of the form
-->

*dịch đoạn phía trên*


$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g}_t'.$$


<!--
Reviewing the design of Adam its inspiration is clear.
Momentum and scale are clearly visible in the state variables.
Their rather peculiar definition forces us to debias terms (this could be fixed by a slightly different initialization and update condition).
Second, the combination of both terms is pretty straightforward, given RMSProp.
Last, the explicit learning rate $\eta$ allows us to control the step length to address issues of convergence.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Implementation
-->

## *dịch tiêu đề phía trên*

<!--
Implementing Adam from scratch is not very daunting.
For convenience we store the timestep counter $t$ in the `hyperparams` dictionary.
Beyond that all is straightforward.
-->

*dịch đoạn phía trên*


```{.python .input  n=2}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adam_states(feature_dim):
    v_w, v_b = np.zeros((feature_dim, 1)), np.zeros(1)
    s_w, s_b = np.zeros((feature_dim, 1)), np.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1
```


<!--
We are ready to use Adam to train the model.
We use a learning rate of $\eta = 0.01$.
-->

*dịch đoạn phía trên*


```{.python .input  n=5}
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```


<!--
A more concise implementation is straightforward since `adam` is one of the algorithms provided as part of the Gluon `trainer` optimization library.
Hence we only need to pass configuration parameters for an implementation in Gluon.
-->

*dịch đoạn phía trên*


```{.python .input  n=11}
d2l.train_gluon_ch11('adam', {'learning_rate': 0.01}, data_iter)
```


<!--
## Yogi
-->

## *dịch tiêu đề phía trên*

<!--
One of the problems of Adam is that it can fail to converge even in convex settings when the second moment estimate in $\mathbf{s}_t$ blows up.
As a fix :cite:`Zaheer.Reddi.Sachan.ea.2018` proposed a refined update (and initialization) for $\mathbf{s}_t$.
To understand what's going on, let us rewrite the Adam update as follows:
-->

*dịch đoạn phía trên*


$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \left(\mathbf{g}_t^2 - \mathbf{s}_{t-1}\right).$$


<!--
Whenever $\mathbf{g}_t^2$ has high variance or updates are sparse, $\mathbf{s}_t$ might forget past values too quickly.
A possible fix for this is to replace $\mathbf{g}_t^2 - \mathbf{s}_{t-1}$ by $\mathbf{g}_t^2 \odot \mathop{\mathrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1})$.
Now the magnitude of the update no longer depends on the amount of deviation.
This yields the Yogi updates
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->


$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \odot \mathop{\mathrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1}).$$


<!--
The authors furthermore advise to initialize the momentum on a larger initial batch rather than just initial pointwise estimate.
We omit the details since they are not material to the discussion and since even without this convergence remains pretty good.
-->

*dịch đoạn phía trên*


```{.python .input}
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = s + (1 - beta2) * np.sign(
            np.square(p.grad) - s) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```


<!--
## Summary
-->

## Tóm tắt

<!--
* Adam combines features of many optimization algorithms into a fairly robust update rule.
* Created on the basis of RMSProp, Adam also uses EWMA on the minibatch stochastic gradient
* Adam uses bias correction to adjust for a slow startup when estimating momentum and a second moment.
* For gradients with significant variance we may encounter issues with convergence. 
They can be amended by using larger minibatches or by switching to an improved estimate for $\mathbf{s}_t$. 
Yogi offers such an alternative.
-->

*dịch đoạn phía trên*

<!--
## Exercises
-->

## Bài tập

<!--
1. Adjust the learning rate and observe and analyze the experimental results.
2. Can you rewrite momentum and second moment updates such that it does not require bias correction?
3. Why do you need to reduce the learning rate $\eta$ as we converge?
4. Try to construct a case for which Adam diverges and Yogi converges?
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->
<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2378)
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
* Trần Yến Thy

<!-- Phần 2 -->
* 

<!-- Phần 3 -->
* 

<!-- Phần 4 -->
* 
