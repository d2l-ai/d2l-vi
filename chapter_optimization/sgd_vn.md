<!--
# Stochastic Gradient Descent
-->

# Hạ Gradient Ngẫu nhiên
:label:`sec_sgd`

<!--
In this section, we are going to introduce the basic principles of stochastic gradient descent.
-->

Trong phần này chúng tôi sẽ giới thiệu các nguyên tắc cơ bản của hạ gradient ngẫu nhiên.


```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx
npx.set_np()
```

<!--
```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
```
-->


<!--
## Stochastic Gradient Updates
-->

## Cập nhật Gradient Ngẫu nhiên

<!--
In deep learning, the objective function is usually the average of the loss functions for each example in the training dataset.
We assume that $f_i(\mathbf{x})$ is the loss function of the training dataset with $n$ examples, an index of $i$, and parameter vector of $\mathbf{x}$, then we have the objective function
-->

Trong học sâu, hàm mục tiêu thường là trung bình của các hàm mất mát cho từng mẫu trong tập huấn luyện.
Giả sử tập huấn luyện có $n$ mẫu, $f_i(\mathbf{x})$ là hàm mất mát của mẫu thứ $i$, và vector tham số là $\mathbf{x}$. Ta có hàm mục tiêu


$$f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x}).$$


<!--
The gradient of the objective function at $\mathbf{x}$ is computed as
-->

Gradient của hàm mục tiêu tại $\mathbf{x}$ được tính như sau


$$\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}).$$


<!--
If gradient descent is used, the computing cost for each independent variable iteration is $\mathcal{O}(n)$, which grows linearly with $n$.
Therefore, when the model training dataset is large, the cost of gradient descent for each iteration will be very high.
-->

Nếu hạ gradient được sử dụng, chi phí tính toán cho mỗi vòng lặp độc lập là $\mathcal{O}(n)$, tăng tuyến tính với $n$.
Do đó, với tập huấn luyện lớn, chi phí của hạ gradient cho mỗi vòng lặp sẽ rất cao.

<!--
Stochastic gradient descent (SGD) reduces computational cost at each iteration.
At each iteration of stochastic gradient descent, we uniformly sample an index $i\in\{1,\ldots, n\}$ for data instances at random, 
and compute the gradient $\nabla f_i(\mathbf{x})$ to update $\mathbf{x}$:
-->

Hạ gradient ngẫu nhiên (*stochastic gradient descent* - SGD) giúp giảm chi phí tính toán ở mỗi vòng lặp.
Ở mỗi vòng lặp, ta lấy ngẫu nhiên một mẫu dữ liệu có chỉ số $i\in\{1,\ldots, n\}$ theo phân phối đều, và chỉ cập nhật $\mathbf{x}$ bằng gradient $\nabla f_i(\mathbf{x})$:


$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x}).$$


<!--
Here, $\eta$ is the learning rate.
We can see that the computing cost for each iteration drops from $\mathcal{O}(n)$ of the gradient descent to the constant $\mathcal{O}(1)$.
We should mention that the stochastic gradient $\nabla f_i(\mathbf{x})$ is the unbiased estimate of gradient $\nabla f(\mathbf{x})$.
-->


Ở đây, $\eta$ là tốc độ học.
Ta có thể thấy rằng chi phí tính toán cho mỗi vòng lặp giảm từ $\mathcal{O}(n)$ của hạ gradient xuống còn hằng số $\mathcal{O}(1)$.
Nên nhớ rằng gradient ngẫu nhiên $\nabla f_i(\mathbf{x})$ là một ước lượng không thiên lệch (*unbiased*) của gradient $\nabla f(\mathbf{x})$.


$$\mathbb{E}_i \nabla f_i(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}).$$


<!--
This means that, on average, the stochastic gradient is a good estimate of the gradient.
-->

Do đó, trên trung bình, gradient ngẫu nhiên là một ước lượng gradient tốt.

<!--
Now, we will compare it to gradient descent by adding random noise with a mean of 0 and a variance of 1 to the gradient to simulate a SGD.
-->

Bây giờ, ta mô phỏng hạ gradient ngẫu nhiên bằng cách thêm nhiễu ngẫu nhiên với trung bình bằng 0 và phương sai bằng 1 vào gradient và so sánh với phương pháp hạ gradient.


```{.python .input}
#@tab all
f = lambda x1, x2: x1 ** 2 + 2 * x2 ** 2  # Objective
gradf = lambda x1, x2: (2 * x1, 4 * x2)  # Gradient

def sgd(x1, x2, s1, s2):
    global lr  # Learning rate scheduler
    (g1, g2) = gradf(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal(0.0, 1, (1,))
    g2 += d2l.normal(0.0, 1, (1,))
    eta_t = eta * lr()  # Learning rate at time t
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)  # Update variables

eta = 0.1
lr = (lambda: 1)  # Constant learning rate
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50))
```


<!--
As we can see, the trajectory of the variables in the SGD is much more noisy than the one we observed in gradient descent in the previous section.
This is due to the stochastic nature of the gradient.
That is, even when we arrive near the minimum, we are still subject to the uncertainty injected by the instantaneous gradient via $\eta \nabla f_i(\mathbf{x})$.
Even after 50 steps the quality is still not so good.
Even worse, it will not improve after additional steps (we encourage the reader to experiment with a larger number of steps to confirm this on his own).
This leaves us with the only alternative---change the learning rate $\eta$.
However, if we pick this too small, we will not make any meaningful progress initially.
On the other hand, if we pick it too large, we will not get a good solution, as seen above.
The only way to resolve these conflicting goals is to reduce the learning rate *dynamically* as optimization progresses.
-->

Như có thể thấy, quỹ đạo của các biến trong SGD dao động mạnh hơn hạ gradient ở phần trước.
Điều này là do bản chất ngẫu nhiên của gradient.
Tức là, ngay cả khi tới gần giá trị cực tiểu, ta vẫn gặp phải sự bất định gây ra bởi gradient ngẫu nhiên $\eta \nabla f_i(\mathbf{x})$.
Thậm chí sau 50 bước thì chất lượng vẫn không tốt lắm.
Tệ hơn, nó vẫn sẽ không cải thiện với nhiều bước hơn (chúng tôi khuyến khích bạn đọc thử nghiệm với số lượng bước lớn hơn để tự xác nhận điều này).
Ta chỉ còn một lựa chọn duy nhất --- thay đổi tốc độ học $\eta$.
Tuy nhiên, nếu chọn giá trị quá nhỏ, ta sẽ không đạt được bất kỳ tiến triển đáng kể nào ở những bước đầu tiên.
Mặt khác, nếu chọn giá trị quá lớn, ta sẽ không thu được nghiệm tốt, như đã thấy ở trên.
Cách duy nhất để giải quyết hai mục tiêu xung đột này là giảm tốc độ học *một cách linh hoạt* trong quá trình tối ưu.

<!--
This is also the reason for adding a learning rate function `lr` into the `sgd` step function.
In the example above any functionality for learning rate scheduling lies dormant as we set the associated `lr` function to be constant, i.e., `lr = (lambda: 1)`.
-->

Đây cũng là lý do cho việc thêm hàm tốc độ học `lr` vào hàm bước `sgd`.
Trong ví dụ trên, chức năng định thời tốc độ học (*learning rate scheduling*) không được kích hoạt vì ta đặt hàm `lr` bằng một hằng số, tức `lr = (lambda: 1)`.

<!--
## Dynamic Learning Rate
-->

## Tốc độ học Linh hoạt

<!--
Replacing $\eta$ with a time-dependent learning rate $\eta(t)$ adds to the complexity of controlling convergence of an optimization algorithm.
In particular, need to figure out how rapidly $\eta$ should decay.
If it is too quick, we will stop optimizing prematurely.
If we decrease it too slowly, we waste too much time on optimization.
There are a few basic strategies that are used in adjusting $\eta$ over time (we will discuss more advanced strategies in a later chapter):
-->

Thay thế $\eta$ bằng tốc độ học phụ thuộc thời gian $\eta(t)$ sẽ khiến việc kiểm soát sự hội tụ của thuật toán tối ưu trở nên phức tạp hơn.
Cụ thể, ta cần tìm ra mức độ suy giảm $\eta$ hợp lý.
Nếu giảm quá nhanh, quá trình tối ưu sẽ ngừng quá sớm.
Nếu giảm quá chậm, ta sẽ lãng phí rất nhiều thời gian cho việc tối ưu.
Có một vài chiến lược cơ bản được sử dụng để điều chỉnh $\eta$ theo thời gian (ta sẽ thảo luận về các chiến lược cao cấp hơn trong chương sau):


$$
\begin{aligned}
    \eta(t) & = \eta_i \text{ if } t_i \leq t \leq t_{i+1}  && \mathrm{~hằng~số~theo~khoảng~} \\
    \eta(t) & = \eta_0 \cdot e^{-\lambda t} && \mathrm{~lũy~thừa~} \\
    \eta(t) & = \eta_0 \cdot (\beta t + 1)^{-\alpha} && \mathrm{~đa~thức~}
\end{aligned}
$$


<!--
In the first scenario we decrease the learning rate, e.g., whenever progress in optimization has stalled.
This is a common strategy for training deep networks.
Alternatively we could decrease it much more aggressively by an exponential decay.
Unfortunately this leads to premature stopping before the algorithm has converged.
A popular choice is polynomial decay with $\alpha = 0.5$.
In the case of convex optimization there are a number of proofs which show that this rate is well behaved.
Let us see what this looks like in practice.
-->

Trong trường hợp đầu tiên, ta giảm tốc độ học bất cứ khi nào tiến trình tối ưu bị đình trệ.
Đây là một chiến lược phổ biến để huấn luyện các mạng sâu.
Ngoài ra, ta có thể làm giảm tốc độ học nhanh hơn bằng suy giảm theo lũy thừa.
Thật không may, phương pháp này dẫn đến việc dừng tối ưu quá sớm trước khi thuật toán hội tụ.
Một lựa chọn phổ biến khác là suy giảm đa thức với $\alpha = 0.5$.
Trong trường hợp tối ưu lồi, có các chứng minh cho thấy giá trị này cho kết quả tốt.
Hãy cùng xem nó hoạt động như thế nào trong thực tế.


```{.python .input}
#@tab all
def exponential():
    global ctr
    ctr += 1
    return math.exp(-0.1 * ctr)

ctr = 1
lr = exponential  # Set up learning rate
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000))
```


<!--
As expected, the variance in the parameters is significantly reduced.
However, this comes at the expense of failing to converge to the optimal solution $\mathbf{x} = (0, 0)$.
Even after 1000 steps are we are still very far away from the optimal solution.
Indeed, the algorithm fails to converge at all.
On the other hand, if we use a polynomial decay where the learning rate decays with the inverse square root of the number of steps convergence is good.
-->

Như dự đoán, giá trị phương sai của các tham số giảm đáng kể.
Tuy nhiên, suy giảm lũy thừa không hội tụ tới nghiệm tối ưu $\mathbf{x} = (0, 0)$.
Thậm chí sau 1000 vòng lặp, nghiệm tìm được vẫn cách nghiệm tối ưu rất xa. 
Trên thực tế, thuật toán này không hội tụ được.
Mặt khác, nếu ta sử dụng suy giảm đa thức trong đó tốc độ học suy giảm tỉ lệ nghịch với căn bình phương thời gian, thuật toán hội tụ tốt.


```{.python .input}
#@tab all
def polynomial():
    global ctr
    ctr += 1
    return (1 + 0.1 * ctr)**(-0.5)

ctr = 1
lr = polynomial  # Set up learning rate
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50))
```


<!--
There exist many more choices for how to set the learning rate.
For instance, we could start with a small rate, then rapidly ramp up and then decrease it again, albeit more slowly.
We could even alternate between smaller and larger learning rates.
There exists a large variety of such schedules.
For now let us focus on learning rate schedules for which a comprehensive theoretical analysis is possible, i.e., on learning rates in a convex setting.
For general nonconvex problems it is very difficult to obtain meaningful convergence guarantees, since in general minimizing nonlinear nonconvex problems is NP hard.
For a survey see e.g., the excellent [lecture notes](https://www.stat.cmu.edu/~ryantibs/convexopt-F15/lectures/26-nonconvex.pdf) of Tibshirani 2015.
-->

Vẫn còn có rất nhiều lựa chọn khác để thiết lập tốc độ học. 
Ví dụ, ta có thể bắt đầu với tốc độ học nhỏ, sau đó tăng nhanh rồi tiếp tục giảm nhưng với tốc độ chậm hơn.
Ta cũng có thể thiết lập tốc độ học tăng và giảm luân phiên.
Có rất nhiều cách khác nhau để định thời tốc độ học.
Bây giờ, chúng ta hãy tập trung vào thiết lập tốc độ học trong điều kiện lồi mà ta có thể phân tích lý thuyết.
Với bài toán không lồi tổng quát, rất khó để đảm bảo được mức hội tụ có ý nghĩa, vì nói chung các bài toán tối ưu phi tuyến không lồi đều thuộc dạng NP-hard.
Để tìm hiểu thêm, tham khảo các ví dụ trong [tập bài giảng](https://www.stat.cmu.edu/~ryantibs/convexopt-F15/lectures/26-nonconvex.pdf) của Tibshirani năm 2015.


<!--
## Convergence Analysis for Convex Objectives
-->

## Phân tích Hội tụ cho Hàm mục tiêu Lồi

<!--
The following is optional and primarily serves to convey more intuition about the problem.
We limit ourselves to one of the simplest proofs, as described by :cite:`Nesterov.Vial.2000`.
Significantly more advanced proof techniques exist, e.g., whenever the objective function is particularly well behaved.
:cite:`Hazan.Rakhlin.Bartlett.2008` show that for strongly convex functions, i.e., for functions that can be bounded from below by $\mathbf{x}^\top \mathbf{Q} \mathbf{x}$, 
it is possible to minimize them in a small number of steps while decreasing the learning rate like $\eta(t) = \eta_0/(\beta t + 1)$.
Unfortunately this case never really occurs in deep learning and we are left with a much more slowly decreasing rate in practice.
-->

Đây là phần đọc thêm để mang lại cái nhìn trực quan hơn về bài toán, 
giới hạn lại trong một cách chứng minh đơn giản được trình bày trong :cite:`Nesterov.Vial.2000`.
Cũng có những cách chứng minh nâng cao hơn, ví dụ như khi hàm mục tiêu được định nghĩa tốt.
:cite: `Hazan.Rakhlin.Bartlett.2008` chỉ ra rằng với các hàm lồi chặt, cụ thể là các hàm có cận dưới là $\mathbf{x}^\top \mathbf{Q} \mathbf{x}$, ta có thể cực tiểu hóa chúng chỉ với một số lượng nhỏ bước lặp trong khi giảm tốc độ học theo $\eta(t) = \eta_0/(\beta t + 1)$.
Thật không may, trường hợp này không xảy ra trong học sâu, trên thực tế mức độ giảm tốc độ học chậm hơn rất nhiều.

<!--
Consider the case where
-->

Xét trường hợp


$$\mathbf{w}_{t+1} = \mathbf{w}_{t} - \eta_t \partial_\mathbf{w} l(\mathbf{x}_t, \mathbf{w}).$$


<!--
In particular, assume that $\mathbf{x}_t$ is drawn from some distribution $P(\mathbf{x})$ and that $l(\mathbf{x}, \mathbf{w})$ is a convex function in $\mathbf{w}$ for all $\mathbf{x}$.
Last denote by
-->

Cụ thể, ta giả sử $\mathbf{x}_t$ được lấy từ phân phối $P(\mathbf{x})$ và $l(\mathbf{x}, \mathbf{w})$ là hàm lồi theo biến $\mathbf{w}$ với mọi $\mathbf{x}$.
Cuối cùng, ta kí hiệu


$$R(\mathbf{w}) = E_{\mathbf{x} \sim P}[l(\mathbf{x}, \mathbf{w})]$$


<!--
the expected risk and by $R^*$ its minimum with regard to $\mathbf{w}$.
Last let $\mathbf{w}^*$ be the minimizer (we assume that it exists within the domain which $\mathbf{w}$ is defined).
In this case we can track the distance between the current parameter $\mathbf{w}_t$ and the risk minimizer $\mathbf{w}^*$ and see whether it improves over time:
-->

là giá trị mất mát kỳ vọng và $R^*$ là cực tiểu của hàm mất mát theo $\mathbf{w}$.
Ta kí hiệu $\mathbf{w}^*$ là nghiệm tại cực tiểu (_minimizer_) với giả định giá trị này tồn tại trong miền xác định.
Trong trường hợp này, chúng ta lần theo khoảng cách giữa tham số hiện tại $\mathbf{w}_t$ và nghiệm cực tiểu $\mathbf{w}^*$, và xem liệu giá trị này có cải thiện theo thời gian không:


$$\begin{aligned}
    \|\mathbf{w}_{t+1} - \mathbf{w}^*\|^2 & = \|\mathbf{w}_{t} - \eta_t \partial_\mathbf{w} l(\mathbf{x}_t, \mathbf{w}) - \mathbf{w}^*\|^2 \\
    & = \|\mathbf{w}_{t} - \mathbf{w}^*\|^2 + \eta_t^2 \|\partial_\mathbf{w} l(\mathbf{x}_t, \mathbf{w})\|^2 - 2 \eta_t
    \left\langle \mathbf{w}_t - \mathbf{w}^*, \partial_\mathbf{w} l(\mathbf{x}_t, \mathbf{w})\right\rangle.
   \end{aligned}
$$


<!--
The gradient $\partial_\mathbf{w} l(\mathbf{x}_t, \mathbf{w})$ can be bounded from above by some Lipschitz constant $L$, hence we have that
-->

Gradient $\partial_\mathbf{w} l(\mathbf{x}_t, \mathbf{w})$ có cận trên là một hằng số Lipschitz $L$, do đó ta có 


$$\eta_t^2 \|\partial_\mathbf{w} l(\mathbf{x}_t, \mathbf{w})\|^2 \leq \eta_t^2 L^2.$$


<!--
We are mostly interested in how the distance between $\mathbf{w}_t$ and $\mathbf{w}^*$ changes *in expectation*.
In fact, for any specific sequence of steps the distance might well increase, depending on whichever $\mathbf{x}_t$ we encounter.
Hence we need to bound the inner product. By convexity we have that
-->

Điều chúng ta thực sự quan tâm là khoảng cách giữa $\mathbf{w}_t$ và $\mathbf{w}^*$ thay đổi như thế nào *theo kỳ vọng*.
Thực tế, với chuỗi các bước bất kỳ, khoảng cách này cũng có thể thể tăng lên, tuỳ thuộc vào giá trị của $\mathbf{x}_t$ mà ta gặp phải.
Do đó cần xác định cận cho tích vô hướng. Từ tính chất lồi, ta có

$$
l(\mathbf{x}_t, \mathbf{w}^*) \geq l(\mathbf{x}_t, \mathbf{w}_t) + \left\langle \mathbf{w}^* - \mathbf{w}_t, \partial_{\mathbf{w}} l(\mathbf{x}_t, \mathbf{w}_t) \right\rangle.
$$


<!--
Using both inequalities and plugging it into the above we obtain a bound on the distance between parameters at time $t+1$ as follows:
-->

Kết hợp hai bất đẳng thức trên, ta tìm được cận cho khoảng cách giữa các tham số tại bước $t+1$ như sau:


$$\|\mathbf{w}_{t} - \mathbf{w}^*\|^2 - \|\mathbf{w}_{t+1} - \mathbf{w}^*\|^2 \geq 2 \eta_t (l(\mathbf{x}_t, \mathbf{w}_t) - l(\mathbf{x}_t, \mathbf{w}^*)) - \eta_t^2 L^2.$$


<!--
This means that we make progress as long as the expected difference between current loss and the optimal loss outweighs $\eta_t L^2$.
Since the former is bound to converge to $0$ it follows that the learning rate $\eta_t$ also needs to vanish.
-->

Điều này có nghĩa quá trình học vẫn sẽ cải thiện miễn là hiệu số giữa hàm mất mát hiện tại và giá trị mất mát tối ưu vẫn lớn hơn $\eta_t L^2$.
Để đảm bảo hàm mất mát hội tụ về $0$, tốc độ học $\eta_t$ cũng cần phải giảm dần.

<!--
Next we take expectations over this expression. This yields
-->

Tiếp theo chúng ta tính giá trị kỳ vọng cho biểu thức trên


$$E_{\mathbf{w}_t}\left[\|\mathbf{w}_{t} - \mathbf{w}^*\|^2\right] - E_{\mathbf{w}_{t+1}\mid \mathbf{w}_t}\left[\|\mathbf{w}_{t+1} - \mathbf{w}^*\|^2\right] \geq 2 \eta_t [E[R[\mathbf{w}_t]] - R^*] -  \eta_t^2 L^2.$$


<!--
The last step involves summing over the inequalities for $t \in \{t, \ldots, T\}$.
Since the sum telescopes and by dropping the lower term we obtain
-->

Ở bước cuối cùng, ta tính tổng các bất đẳng thức trên cho mọi $t \in \{t, \ldots, T\}$. 
Rút gọn tổng và bỏ qua các hạng tử thấp hơn, ta có


$$\|\mathbf{w}_{0} - \mathbf{w}^*\|^2 \geq 2 \sum_{t=1}^T \eta_t [E[R[\mathbf{w}_t]] - R^*] - L^2 \sum_{t=1}^T \eta_t^2.$$


<!--
Note that we exploited that $\mathbf{w}_0$ is given and thus the expectation can be dropped. Last define
-->

Lưu ý rằng ta tận dụng $\mathbf{w}_0$ cho trước và do đó có thể bỏ qua giá trị kỳ vọng. Cuối cùng, ta định nghĩa

$$\bar{\mathbf{w}} := \frac{\sum_{t=1}^T \eta_t \mathbf{w}_t}{\sum_{t=1}^T \eta_t}.$$

<!--
Then by convexity it follows that
-->

Từ đó, theo tính chất lồi, ta có


$$\sum_t \eta_t E[R[\mathbf{w}_t]] \geq \sum \eta_t \cdot \left[E[\bar{\mathbf{w}}]\right].$$


<!--
Plugging this into the above inequality yields the bound
-->

Thay vào bất đẳng thức ở trên, ta tìm được cận



$$
\left[E[\bar{\mathbf{w}}]\right] - R^* \leq \frac{r^2 + L^2 \sum_{t=1}^T \eta_t^2}{2 \sum_{t=1}^T \eta_t}.
$$


<!--
Here $r^2 := \|\mathbf{w}_0 - \mathbf{w}^*\|^2$ is a bound on the distance between the initial choice of parameters and the final outcome.
In short, the speed of convergence depends on how rapidly the loss function changes via the Lipschitz constant $L$ and how far away from optimality the initial value is $r$.
Note that the bound is in terms of $\bar{\mathbf{w}}$ rather than $\mathbf{w}_T$.
This is the case since $\bar{\mathbf{w}}$ is a smoothed version of the optimization path.
Now let us analyze some choices for $\eta_t$.
-->

Trong đó $r^2 := \|\mathbf{w}_0 - \mathbf{w}^*\|^2$ là khoảng cách giới hạn giữa giá trị khởi tạo của các tham số và kết quả cuối cùng.
Nói tóm lại, tốc độ hội tụ phụ thuộc vào tốc độ thay đổi của hàm mất mát thông qua hằng số Lipschitz $L$ và khoảng cách $r$ giữa giá trị ban đầu so với giá trị tối ưu.
Chú ý rằng giới hạn ở trên được kí hiệu bởi $\bar{\mathbf{w}}$ thay vì $\mathbf{w}_T$ do $\bar{\mathbf{w}}$ là quỹ đạo tối ưu được làm mượt.
Hãy cùng phân tích một số cách lựa chọn $\eta_t$.

<!--
* **Known Time Horizon**. 
Whenever $r, L$ and $T$ are known we can pick $\eta = r/L \sqrt{T}$. 
This yields as upper bound $r L (1 + 1/T)/2\sqrt{T} < rL/\sqrt{T}$. 
That is, we converge with rate $\mathcal{O}(1/\sqrt{T})$ to the optimal solution.
* **Unknown Time Horizon**. 
Whenever we want to have a good solution for *any* time $T$ we can pick $\eta = \mathcal{O}(1/\sqrt{T})$. 
This costs us an extra logarithmic factor and it leads to an upper bound of the form $\mathcal{O}(\log T / \sqrt{T})$.
-->

* **Thời điểm xác định**.
Với mỗi $r, L$ và $T$ xác định ta có thể chọn $\eta = r/L \sqrt{T}$.
Biểu thức này dẫn tới giới hạn trên $r L (1 + 1/T)/2\sqrt{T} < rL/\sqrt{T}$.
Điều này nghĩa là hàm sẽ hội tụ đến nghiệm tối ưu với tốc độ $\mathcal{O}(1/\sqrt{T})$.
* **Thời điểm chưa xác định**.
Khi muốn nghiệm tốt cho *bất kì* thời điểm $T$ nào, ta có thể chọn $\eta = \mathcal{O}(1/\sqrt{T})$.
Cách làm trên tốn thêm một thừa số logarit, dẫn tới giới hạn trên có dạng $\mathcal{O}(\log T / \sqrt{T})$.

<!--
Note that for strongly convex losses 
$l(\mathbf{x}, \mathbf{w}') \geq l(\mathbf{x}, \mathbf{w}) + \langle \mathbf{w}'-\mathbf{w}, \partial_\mathbf{w} l(\mathbf{x}, \mathbf{w}) \rangle + \frac{\lambda}{2} \|\mathbf{w}-\mathbf{w}'\|^2$ 
we can design even more rapidly converging optimization schedules. 
In fact, an exponential decay in $\eta$ leads to a bound of the form $\mathcal{O}(\log T / T)$.
-->

Chú ý rằng đối với những hàm mất mát lồi tuyệt đối
$l(\mathbf{x}, \mathbf{w}') \geq l(\mathbf{x}, \mathbf{w}) + \langle \mathbf{w}'-\mathbf{w}, \partial_\mathbf{w} l(\mathbf{x}, \mathbf{w}) \rangle + \frac{\lambda}{2} \|\mathbf{w}-\mathbf{w}'\|^2$
ta có thể thiết kế quy trình tối ưu nhằm tăng tốc độ hội tụ nhanh hơn nữa.
Thực tế, sự suy giảm theo cấp số mũ của $\eta$ dẫn đến giới hạn có dạng $\mathcal{O}(\log T / T)$.


<!--
## Stochastic Gradients and Finite Samples
-->

## Gradient ngẫu nhiên và Mẫu hữu hạn

<!--
So far we have played a bit fast and loose when it comes to talking about stochastic gradient descent.
We posited that we draw instances $x_i$, typically with labels $y_i$ from some distribution $p(x, y)$ and that we use this to update the weights $w$ in some manner.
In particular, for a finite sample size we simply argued that the discrete distribution $p(x, y) = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}(x) \delta_{y_i}(y)$ allows us to perform SGD over it.
-->

Tới phần này, ta đi khá nhanh và chưa chặt chẽ khi thảo luận về hạ gradient ngẫu nhiên.
Ta ngầm định lấy các đối tượng $x_i$, thường là cùng với nhãn $y_i$ từ phân phối $p(x, y)$ nào đó và sử dụng chúng để cập nhật các trọng số $w$ theo cách nào đó.
Cụ thể, với kích thước mẫu hữu hạn, ta đơn giản lập luận rằng phân phối rời rạc $p(x, y) = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}(x) \delta_{y_i}(y)$ cho phép áp dụng SGD.

<!--
However, this is not really what we did.
In the toy examples in the current section we simply added noise to an otherwise non-stochastic gradient, i.e., we pretended to have pairs $(x_i, y_i)$.
It turns out that this is justified here (see the exercises for a detailed discussion).
More troubling is that in all previous discussions we clearly did not do this.
Instead we iterated over all instances exactly once.
To see why this is preferable consider the converse, namely that we are sampling $n$ observations from the discrete distribution with replacement.
The probability of choosing an element $i$ at random is $N^{-1}$. Thus to choose it at least once is
-->

Tuy nhiên, đó thật ra không phải là cách ta đã làm.
Trong các ví dụ đơn giản ở phần này ta chỉ thêm nhiễu vào gradient không ngẫu nhiên, tức giả sử đang có sẵn các cặp giá trị $(x_i, y_i)$.
hóa ra cách làm đó khá hợp lý (xem phần bài tập để thảo luận chi tiết).
Vấn đề là ở tất cả các thảo luận trước, ta không hề làm thế.
Thay vào đó ta duyệt qua tất cả các đối tượng đúng một lần.
Để hiểu tại sao quá trình này được ưa chuộng, hãy xét trường hợp ngược lại khi ta lấy có hoàn lại $N$ mẫu từ một phân phối rời rạc.
Xác suất phần tử $i$ được chọn ngẫu nhiên là $N^{-1}$.
Do đó xác suất chọn $i$ ít nhất một lần là

$$P(\mathrm{~chọn~} i) = 1 - P(\mathrm{~loại~} i) = 1 - (1-N^{-1})^N \approx 1-e^{-1} \approx 0.63.$$

<!--
A similar reasoning shows that the probability of picking a sample exactly once is given by ${N \choose 1} N^{-1} (1-N^{-1})^{N-1} = \frac{N-1}{N} (1-N^{-1})^{N} \approx e^{-1} \approx 0.37$.
This leads to an increased variance and decreased data efficiency relative to sampling without replacement.
Hence, in practice we perform the latter (and this is the default choice throughout this book).
Last note that repeated passes through the dataset traverse it in a *different* random order.
-->

Tương tự, ta có thể chỉ ra rằng xác suất chọn một mẫu đúng một lần là ${N \choose 1} N^{-1} (1-N^{-1})^{N-1} = \frac{N-1}{N} (1-N^{-1})^{N} \approx e^{-1} \approx 0.37$.
Điều này gây tăng phương sai và giảm hiệu quả sử dụng dữ liệu so với lấy mẫu không hoàn lại.
Do đó trong thực tế, ta thực hiện lấy mẫu không hoàn lại (và đây cũng là lựa chọn mặc định trong quyển sách này).
Điều cuối cùng cần chú ý là mỗi lần duyệt lại tập dữ liệu, ta sẽ duyệt theo một thứ tự ngẫu nhiên *khác*.


<!--
## Summary
-->

## Tóm tắt

<!--
* For convex problems we can prove that for a wide choice of learning rates Stochastic Gradient Descent will converge to the optimal solution.
* For deep learning this is generally not the case. However, the analysis of convex problems gives us useful insight into how to approach optimization, 
namely to reduce the learning rate progressively, albeit not too quickly.
* Problems occur when the learning rate is too small or too large. In practice  a suitable learning rate is often found only after multiple experiments.
* When there are more examples in the training dataset, it costs more to compute each iteration for gradient descent, so SGD is preferred in these cases.
* Optimality guarantees for SGD are in general not available in nonconvex cases since the number of local minima that require checking might well be exponential.
-->

* Đối với các bài toán lồi, ta có thể chứng minh rằng Hạ Gradient Ngẫu nhiên sẽ hội tụ về nghiệm tối ưu với nhiều tốc độ học khác nhau.
* Trường hợp trên thường không xảy ra trong học sâu. Tuy nhiên việc phân tích các bài toán lồi cho ta kiến thức hữu ích để tiếp cận bài toán tối ưu, đó là giảm dần tốc độ học, dù không quá nhanh.
* Nhiều vấn đề xuất hiện khi tốc độ học quá lớn hoặc quá nhỏ. Trong thực tế, ta chỉ có thể tìm được tốc độ học thích hợp sau nhiều lần thử nghiệm.
* Khi kích thước tập huấn luyện tăng, chi phí tính toán cho mỗi lần lặp của hạ gradient cũng tăng theo, do đó SGD được ưa chuộng hơn trong trường hợp này.
* Trong SGD, không có sự đảm bảo tối ưu đối với các trường hợp không lồi do số cực tiểu cần phải kiểm tra có thể tăng theo cấp số nhân.


<!--
## Exercises
-->

## Bài tập

<!--
1. Experiment with different learning rate schedules for SGD and with different numbers of iterations.
In particular, plot the distance from the optimal solution $(0, 0)$ as a function of the number of iterations.
2. Prove that for the function $f(x_1, x_2) = x_1^2 + 2 x_2^2$ adding normal noise to the gradient is equivalent to minimizing a loss function $l(\mathbf{x}, \mathbf{w}) = (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$ where $x$ is drawn from a normal distribution.
    * Derive mean and variance of the distribution for $\mathbf{x}$.
    * Show that this property holds in general for objective functions $f(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{\mu})^\top Q (\mathbf{x} - \mathbf{\mu})$ for $Q \succeq 0$.
3. Compare convergence of SGD when you sample from $\{(x_1, y_1), \ldots, (x_m, y_m)\}$ with replacement and when you sample without replacement.
4. How would you change the SGD solver if some gradient (or rather some coordinate associated with it) was consistently larger than all other gradients?
5. Assume that $f(x) = x^2 (1 + \sin x)$. How many local minima does $f$ have? Can you change $f$ in such a way that to minimize it one needs to evaluate all local minima?
-->

1. Hãy thử nghiệm với nhiều bộ định thời tốc độ học khác nhau trong SGD và với số vòng lặp khác nhau.
Cụ thể, hãy vẽ biểu đồ khoảng cách tới nghiệm tối ưu $(0, 0)$ theo số vòng lặp.
2. Chứng minh rằng với hàm $f(x_1, x_2) = x_1^2 + 2 x_2^2$, việc thêm nhiễu Gauss (*normal noise*) vào gradient tương đương với việc cực tiểu hóa hàm mất mát $l(\mathbf{x}, \mathbf{w}) = (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$ trong đó $x$ tuân theo phân phối chuẩn.
    * Suy ra kỳ vọng và phương sai của $\mathbf{x}$.
    * Chỉ ra rằng tính chất này có thể áp dụng tổng quát cho hàm mục tiêu $f(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{\mu})^\top Q (\mathbf{x} - \mathbf{\mu})$ với $Q \succeq 0$.
3. So sánh sự hội tụ của SGD khi lấy mẫu không hoàn lại từ $\{(x_1, y_1), \ldots, (x_m, y_m)\}$ và khi lấy mẫu có hoàn lại.
4. Bạn sẽ thay đổi SGD thế nào nếu như một số gradient (hoặc một số toạ độ liên kết với nó) liên tục lớn hơn tất cả các gradient khác?
5. Giả sử $f(x) = x^2 (1 + \sin x)$. $f$ có bao nhiêu cực tiểu? Thay đổi hàm $f$ sao cho để cực tiểu hóa giá trị hàm này, ta cần xét tất cả các điểm cực tiểu?


## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/352)
* [Tiếng Anh - Pytorch](https://discuss.d2l.ai/t/497)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện

* Đoàn Võ Duy Thanh
* Nguyễn Duy Du
* Phạm Minh Đức
* Nguyễn Văn Quang
* Đỗ Trường Giang
* Lê Khắc Hồng Phúc
* Phạm Hồng Vinh
* Nguyễn Văn Cường
