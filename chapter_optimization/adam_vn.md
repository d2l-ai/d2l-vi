<!--
# Adam
-->

# Adam
:label:`sec_adam`

<!--
In the discussions leading up to this section we encountered a number of techniques for efficient optimization.
Let us recap them in detail here:
-->

Từ các thảo luận dẫn trước, chúng ta đã làm quen với một số kỹ thuật để tối ưu hóa hiệu quả. 
Hãy cùng tóm tắt chi tiết những kỹ thuật này ở đây: 

<!--
* We saw that :numref:`sec_sgd` is more effective than Gradient Descent when solving optimization problems, e.g., due to its inherent resilience to redundant data.
* We saw that :numref:`sec_minibatch_sgd` affords significant additional efficiency arising from vectorization, using larger sets of observations in one minibatch. 
This is the key to efficient multi-machine, multi-GPU and overall parallel processing.
* :numref:`sec_momentum` added a mechanism for aggregating a history of past gradients to accelerate convergence.
* :numref:`sec_adagrad` used per-coordinate scaling to allow for a computationally efficient preconditioner.
* :numref:`sec_rmsprop` decoupled per-coordinate scaling from a learning rate adjustment.
-->

* Chúng ta thấy rằng SGD trong :numref:`sec_sgd` hiệu quả hơn hạ gradient khi giải các bài toán tối ưu, ví dụ, nó chịu ít ảnh hưởng xấu gây ra bởi dữ liệu dư thừa.
* Chúng ta thấy rằng minibatch SGD trong :numref:`sec_minibatch_sgd` mang lại hiệu quả đáng kể nhờ việc vector hóa, tức xử lý nhiều mẫu quan sát hơn trong một minibatch. 
Đây là chìa khóa để xử lý dữ liệu song song trên nhiều GPU và nhiều máy tính một cách hiệu quả. 
* Phương pháp động lượng trong :numref:`sec_momentum` bổ sung cơ chế gộp các gradient quá khứ, giúp quá trình hội tụ diễn ra nhanh hơn. 
* Adagrad trong :numref:`sec_adagrad` sử dụng phép biến đổi tỉ lệ theo từng tọa độ để tạo ra tiền điều kiện hiệu quả về mặt tính toán. 
* RMSprop trong :numref:`sec_rmsprop` tách rời phép biến đổi tỉ lệ theo từng tọa độ khỏi phép điều chỉnh tốc độ học. 

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

Một trong những thành phần chính của Adam là các trung bình động trọng số mũ (hay còn được gọi là trung bình rò rỉ)
để ước lượng cả động lượng và mô-men bậc hai của gradient.
Cụ thể, nó sử dụng các biến trạng thái

$$\begin{aligned}
    \mathbf{v}_t & \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \\
    \mathbf{s}_t & \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2.
\end{aligned}$$

<!--
Here $\beta_1$ and $\beta_2$ are nonnegative weighting parameters.
Common choices for them are $\beta_1 = 0.9$ and $\beta_2 = 0.999$.
That is, the variance estimate moves *much more slowly* than the momentum term.
Note that if we initialize $\mathbf{v}_0 = \mathbf{s}_0 = 0$ we have a significant amount of bias initially towards smaller values.
This can be addressed by using the fact that $\sum_{i=0}^t \beta^i = \frac{1 - \beta^t}{1 - \beta}$ to re-normalize terms.
Correspondingly the normalized state variables are given by
-->

Ở đây $\beta_1$ và $\beta_2$ là các tham số trọng số không âm. 
Các lựa chọn phổ biến cho chúng là $\beta_1 = 0.9$ và $\beta_2 = 0.999$. 
Điều này có nghĩa là ước lượng phương sai di chuyển *chậm hơn nhiều* so với số hạng động lượng. 
Lưu ý rằng nếu ta khởi tạo $\mathbf{v}_0 = \mathbf{s}_0 = 0$, thuật toán sẽ có độ chệch ban đầu đáng kể về các giá trị nhỏ hơn. 
Vấn đề này có thể được giải quyết bằng cách sử dụng $\sum_{i=0}^t \beta^i = \frac{1 - \beta^t}{1 - \beta}$ để chuẩn hóa lại các số hạng. 
Tương tự, các biến trạng thái được chuẩn hóa như sau 


$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \text{ and } \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}.$$


<!--
Armed with the proper estimates we can now write out the update equations.
First, we rescale the gradient in a manner very much akin to that of RMSProp to obtain
-->

Với các ước lượng thích hợp, bây giờ chúng ta có thể viết ra các phương trình cập nhật. 
Đầu tiên, chúng ta điều chỉnh lại giá trị gradient, tương tự như ở RMSProp để có được 


$$\mathbf{g}_t' = \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}.$$


<!--
Unlike RMSProp our update uses the momentum $\hat{\mathbf{v}}_t$ rather than the gradient itself.
Moreover, there is a slight cosmetic difference as the rescaling happens using $\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$ instead of $\frac{1}{\sqrt{\hat{\mathbf{s}}_t + \epsilon}}$.
The former works arguably slightly better in practice, hence the deviation from RMSProp.
Typically we pick $\epsilon = 10^{-6}$ for a good trade-off between numerical stability and fidelity.
-->

Không giống như RMSProp, phương trình cập nhật sử dụng động lượng $\hat{\mathbf{v}}_t$ thay vì gradient.
Hơn nữa, có một sự khác biệt nhỏ ở đây: phép chuyển đổi được thực hiện bằng cách sử dụng $\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$ thay vì $\frac{1}{\sqrt{\hat{\mathbf{s}}_t + \epsilon}}$. 
Trong thực tế, cách đầu tiên hoạt động tốt hơn một chút, dẫn đến sự khác biệt này so với RMSProp.
Thông thường, ta chọn $\epsilon = 10^{-6}$ để cân bằng giữa tính ổn định số học và độ tin cậy.

<!--
Now we have all the pieces in place to compute updates.
This is slightly anticlimactic and we have a simple update of the form
-->

Bây giờ chúng ta sẽ tổng hợp lại tất cả các điều trên để tính toán bước cập nhật.
Có thể bạn sẽ thấy hơi tụt hứng một chút vì thực ra nó khá đơn giản 


$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g}_t'.$$


<!--
Reviewing the design of Adam its inspiration is clear.
Momentum and scale are clearly visible in the state variables.
Their rather peculiar definition forces us to debias terms (this could be fixed by a slightly different initialization and update condition).
Second, the combination of both terms is pretty straightforward, given RMSProp.
Last, the explicit learning rate $\eta$ allows us to control the step length to address issues of convergence.
-->

Khi xem xét thiết kế của Adam, ta thấy rõ nguồn cảm hứng của thuật toán.
Động lượng và khoảng giá trị được thể hiện rõ ràng trong các biến trạng thái.
Định nghĩa khá kì lạ của chúng đòi hỏi ta phải giảm độ chệch của các số hạng (có thể được thực hiện bằng cách tinh chỉnh một chút phép khởi tạo và điều kiện cập nhật).
Thứ hai, việc kết hợp của cả hai số hạng trên khá đơn giản, dựa trên RMSProp.
Cuối cùng, tốc độ học tường minh $\eta$ cho phép ta kiểm soát độ dài bước cập nhật để giải quyết các vấn đề về hội tụ.

<!--
## Implementation
-->

## Lập trình

<!--
Implementing Adam from scratch is not very daunting.
For convenience we store the timestep counter $t$ in the `hyperparams` dictionary.
Beyond that all is straightforward.
-->

Lập trình Adam từ đầu không quá khó khăn.
Để thuận tiện, chúng ta lưu trữ biến đếm bước thời gian $t$ trong từ điển `hyperparams`.
Ngoài điều đó ra, mọi thứ khác khá đơn giản.


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

Chúng ta đã sẵn sàng sử dụng Adam để huấn luyện mô hình.
Chúng ta sử dụng tốc độ học $\eta = 0.01$.


```{.python .input  n=5}
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```


<!--
A more concise implementation is straightforward since `adam` is one of the algorithms provided as part of the Gluon `trainer` optimization library.
Hence we only need to pass configuration parameters for an implementation in Gluon.
-->

Cách lập trình súc tích hơn là gọi trực tiếp `adam` được cung cấp sẵn trong thư viện tối ưu `trainer` của Gluon.
Do đó ta chỉ cần truyền các tham số cấu hình để lập trình trong Gluon.


```{.python .input  n=11}
d2l.train_concise_ch11('adam', {'learning_rate': 0.01}, data_iter)
```


<!--
## Yogi
-->

## Yogi


<!--
One of the problems of Adam is that it can fail to converge even in convex settings when the second moment estimate in $\mathbf{s}_t$ blows up.
As a fix :cite:`Zaheer.Reddi.Sachan.ea.2018` proposed a refined update (and initialization) for $\mathbf{s}_t$.
To understand what's going on, let us rewrite the Adam update as follows:
-->

Một trong những vấn đề của Adam là nó có thể không hội tụ ngay cả trong các điều kiện lồi khi ước lượng mô-men bậc hai trong $\mathbf{s}_t$ tăng đột biến.
:cite:`Zaheer.Reddi.Sachan.ea.2018` đề xuất phiên bản cải thiện của bước cập nhật (và khởi tạo) $\mathbf{s}_t$ để giải quyết vấn đề này.
Để hiểu rõ hơn, chúng ta hãy viết lại bước cập nhật Adam như sau:


$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \left(\mathbf{g}_t^2 - \mathbf{s}_{t-1}\right).$$


<!--
Whenever $\mathbf{g}_t^2$ has high variance or updates are sparse, $\mathbf{s}_t$ might forget past values too quickly.
A possible fix for this is to replace $\mathbf{g}_t^2 - \mathbf{s}_{t-1}$ by $\mathbf{g}_t^2 \odot \mathop{\mathrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1})$.
Now the magnitude of the update no longer depends on the amount of deviation.
This yields the Yogi updates
-->

Khi $\mathbf{g}_t^2$ có phương sai lớn hay các cập nhật trở nên thưa, $\mathbf{s}_t$ sẽ có thể nhanh chóng quên mất các giá trị quá khứ.
Một cách giải quyết vấn đề trên đó là thay $\mathbf{g}_t^2 - \mathbf{s}_{t-1}$ bằng $\mathbf{g}_t^2 \odot \mathop{\mathrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1})$.
Bây giờ, độ lớn của cập nhật không còn phụ thuộc vào giá trị độ lệch.
Từ đó ta có bước cập nhật Yogi sau: 


$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \odot \mathop{\mathrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1}).$$


<!--
The authors furthermore advise to initialize the momentum on a larger initial batch rather than just initial pointwise estimate.
We omit the details since they are not material to the discussion and since even without this convergence remains pretty good.
-->

Hơn nữa, các tác giả khuyên nên khởi tạo động lượng trên một batch ban đầu có kích thước lớn hơn thay vì ước lượng ban đầu theo điểm.
Chúng ta không đi sâu vào điểm này, vì quá trình hội tụ vẫn diễn ra khá tốt ngay cả khi không áp dụng chúng.


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

* Adam kết hợp các kỹ thuật của nhiều thuật toán tối ưu thành một quy tắc cập nhật khá mạnh mẽ.
* Dựa trên RMSProp, Adam cũng sử dụng trung bình động trọng số mũ cho gradient ngẫu nhiên theo minibatch.
* Adam sử dụng phép hiệu chỉnh độ chệch (_bias correction_) để điều chỉnh cho trường hợp khởi động chậm khi ước lượng động lượng và mô-men bậc hai.
* Đối với gradient có phương sai đáng kể, chúng ta có thể gặp phải những vấn đề liên quan tới hội tụ.
Những vấn đề này có thể được khắc phục bằng cách sử dụng các minibatch có kích thước lớn hơn hoặc bằng cách chuyển sang sử dụng ước lượng được cải tiến cho $\mathbf{s}_t$.
Yogi là một trong nhưng giải pháp như vậy.

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

1. Hãy điều chỉnh tốc độ học, quan sát và phân tích kết quả thực nghiệm.
2. Bạn có thể viết lại các phương trình cập nhật cho động lượng và mô-men bậc hai mà không cần thực hiện phép hiệu chỉnh độ chệch (_bias correction_) không?
3. Tại sao ta cần phải giảm tốc độ học $\eta$ khi quá trình hội tụ diễn ra?
4. Hãy xây dựng một trường hợp mà thuật toán Adam phân kỳ nhưng Yogi lại hội tụ?


## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/358)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Trần Yến Thy
* Nguyễn Lê Quang Nhật
* Nguyễn Văn Quang
* Nguyễn Văn Cường
* Phạm Minh Đức
* Lê Khắc Hồng Phúc
* Phạm Hồng Vinh
