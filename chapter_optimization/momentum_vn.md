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

Trong :numref:`sec_sgd` chúng ta đã ôn tập về kỹ thuật hạ gradient ngẫu nhiên, là khi tối ưu hoá mà chỉ có thể sử dụng một dạng gradient có nhiễu.
Cụ thể, với gradient nhiễu chúng ta cần cực kỳ cẩn trọng trong việc chọn tốc độ học khi có mặt tác nhân gây nhiễu.
Nếu gradient giảm quá nhanh, việc hội tụ sẽ bị chững lại.
Nếu gradient giảm chậm, việc hội tụ tại một kết quả đủ tốt sẽ khó xảy ra bởi vì nhiễu sẽ đẩy điểm hội tụ ra xa điểm tối ưu.

<!--
## Basics
-->

## Kiến thức cơ bản

<!--
In this section, we will explore more effective optimization algorithms, especially for certain types of optimization problems that are common in practice.
-->

Trong phần này, chúng ta sẽ cùng nhau khám phá những thuật toán tối ưu hiệu quả hơn, cụ thể là cho một số dạng bài toán tối ưu phổ biến trong thực tế.

<!--
### Leaky Averages
-->

### Giá trị trung bình rò rỉ

<!--
The previous section saw us discussing minibatch SGD as a means for accelerating computation.
It also had the nice side-effect that averaging gradients reduced the amount of variance.
-->

Trong phần trước, chúng ta đã thảo luận về hạ gradient ngẫu nhiên theo minibatch như một cách để tăng tốc độ tính toán.
Đồng thời, kỹ thuật lấy trung bình gradients này cũng có một "tác dụng phụ" tốt đó là giúp giảm phương sai.


$$\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}_{t-1}) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbf{g}_{i, t-1}.
$$


<!--
Here we used $\mathbf{g}_{ii} = \partial_{\mathbf{w}} f(\mathbf{x}_i, \mathbf{w}_t)$ to keep the notation simple.
It would be nice if we could benefit from the effect of variance reduction even beyond averaging gradients on a mini-batch.
One option to accomplish this task is to replace the gradient computation by a "leaky average":
-->

Ở đây chúng ta dùng $\mathbf{g}_{ii} = \partial_{\mathbf{w}} f(\mathbf{x}_i, \mathbf{w}_t)$ để giúp ký hiệu được đơn giản.
Sẽ rất tốt nếu ta có khả năng tận dụng được lợi ích từ việc giảm phương sai bên cạnh cách lấy trung bình gradient trên từng minibatch.
Một phương pháp để đạt được điều này đó là thay thế việc tính toán gradient bằng một giá trị "trung bình rò rỉ": 


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
Để thấy rõ hơn những gì đang diễn ra, chúng ta hãy mở rộng $\mathbf{v}_t$ một cách đệ quy thành


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

Giá trị $\beta$ lớn tương ứng với trung bình trong khoảng rộng, trong khi đó giá trị $\beta$ nhỏ có nghĩa là chỉ có một chút chỉnh sửa nhẹ so với phương pháp gradient bình thường.
Gradient mới này không còn trỏ về hướng đi dốc nhất trong từng trường hợp cụ thể nữa mà thay vào đó đi theo hướng trung bình có trọng số của các gradient trước đó.
Điều này cho phép chúng ta nhận được hầu hết lợi ích của việc tính toán trung bình theo batch mà không phải tốn chi phí tính toán gradients theo cả batch.
Chúng ta sẽ xem xét lại quy trình lấy trung bình một cách cụ thể hơn ở những phần sau.

<!--
The above reasoning formed the basis for what is now known as *accelerated* gradient methods, such as gradients with momentum.
They enjoy the additional benefit of being much more effective in cases where the optimization problem is ill-conditioned 
(i.e., where there are some directions where progress is much slower than in others, resembling a narrow canyon).
Furthermore, they allow us to average over subsequent gradients to obtain more stable directions of descent.
Indeed, the aspect of acceleration even for noise-free convex problems is one of the key reasons why momentum works and why it works so well.
-->

Các lập luận là cơ sở đã hình thành nên các phương pháp *tăng tốc* gradient, chẳng hạn như gradient với động lượng.
Một lợi ích phụ là chúng hiệu quả hơn rất nhiều trong các trường hợp bài toán tối ưu có điều kiện xấu (ví dụ: khi một vài hướng có tiến trình chậm hơn rất nhiều so với các hướng khác, giống như ở trong một hẻm núi hẹp).
Hơn nữa, cách này cho phép chúng ta tính trung bình các gradient liên tiếp để đạt được hướng đi xuống ổn định hơn.
Thật vậy, việc tăng tốc ngay cả đối với bài toán hàm lồi không nhiễu là một trong những nguyên nhân chính lý giải vì sao động lượng hoạt động và có hiệu quả rất tốt.


<!--
As one would expect, due to its efficacy momentum is a well-studied subject in optimization for deep learning and beyond.
See e.g., the beautiful [expository article](https://distill.pub/2017/momentum/) by :cite:`Goh.2017` for an in-depth analysis and interactive animation.
It was proposed by :cite:`Polyak.1964`.
:cite:`Nesterov.2018` has a detailed theoretical discussion in the context of convex optimization.
Momentum in deep learning has been known to be beneficial for a long time.
See e.g., the discussion by :cite:`Sutskever.Martens.Dahl.ea.2013` for details.
-->

Do tính hiệu quả của nó, động lượng là một chủ đề đã được nghiên cứu kỹ trong tối ưu hoá cho học sâu và hơn thế nữa.
[Bài báo rất đẹp này](https://distill.pub/2017/momentum/) của :cite:`Goh.2017` có một phân tích chuyên sâu và minh hoạ sinh động về vấn đề này.
Động lượng được đề xuất bởi :cite:`Polyak.1964` và :cite:`Nesterov.2018` đã có một thảo luận học thuật chi tiết trong ngữ cảnh tối ưu hoá lồi.
Động lượng trong học sâu đã được biết đến từ lâu vì lợi ích mà nó mang lại.
Xem thảo luận của :cite:`Sutskever.Martens.Dahl.ea.2013` để có thêm chi tiết.

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
### An Ill-conditioned Problem
-->

### Bài toán với Điều kiện Xấu

<!--
To get a better understanding of the geometric properties of the momentum method we revisit gradient descent, albeit with a significantly less pleasant objective function.
Recall that in :numref:`sec_gd` we used $f(\mathbf{x}) = x_1^2 + 2 x_2^2$, i.e., a moderately distorted ellipsoid objective.
We distort this function further by stretching it out in the $x_1$ direction via
-->

Để hiểu hơn về các tính chất hình học của phương pháp động lượng, chúng ta hãy ôn lại thuật toán hạ gradient với hàm mục tiêu khó chịu hơn.
Hãy nhớ lại trong :numref:`sec_gd` chúng ta sử dụng $f(\mathbf{x}) = x_1^2 + 2 x_2^2$ là hàm mục tiêu dạng elip.
Chúng ta sẽ sửa đổi hàm này một chút để kéo dài thêm hình dạng theo hướng $x_1$ như sau:


$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$


<!--
As before $f$ has its minimum at $(0, 0)$. This function is *very* flat in the direction of $x_1$.
Let us see what happens when we perform gradient descent as before on this new function.
We pick a learning rate of $0.4$.
-->

Như trước đây, $f$ đạt cực tiểu tại điểm $(0, 0)$. Hàm này *rất* phẳng theo hướng $x_1$.
Chúng ta hãy xem điều gì sẽ xảy ra khi thực hiện hạ gradient tương tự như trước trên hàm mới định nghĩa.
Chúng ta đặt tốc độ học bằng $0.4$.


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

Ta thấy, gradient theo hướng $x_2$ có giá trị *lớn hơn nhiều* và thay đổi nhanh hơn nhiều so với gradient theo hướng ngang $x_1$.
Vì thế, chúng ta bị mắc kẹt giữa hai lựa chọn không mong muốn: Nếu chọn tốc độ học nhỏ, các nghiệm sẽ không phân kỳ theo hướng $x_2$
nhưng tốc độ hội tụ sẽ chậm theo hướng $x_1$.
Ngược lại, với một tốc độ học lớn mô hình sẽ hội tụ nhanh chóng theo hướng $x_1$ nhưng phân kỳ theo hướng $x_2$.
Ví dụ dưới đây minh họa những gì xảy ra khi tăng nhẹ tốc độ học từ $0.4$ đến $0.6$.
Hội tụ theo hướng $x_1$ cải thiện nhưng kết quả cuối cùng tệ hơn rất nhiều.


```{.python .input  n=4}
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
### The Momentum Method
-->

### Phương pháp Động lượng

<!--
The momentum method allows us to solve the gradient descent problem described above.
Looking at the optimization trace above we might intuit that averaging gradients over the past would work well.
After all, in the $x_1$ direction this will aggregate well-aligned gradients, thus increasing the distance we cover with every step.
Conversely, in the $x_2$ direction where gradients oscillate, an aggregate gradient will reduce step size due to oscillations that cancel each other out.
Using $\mathbf{v}_t$ instead of the gradient $\mathbf{g}_t$ yields the following update equations:
-->

Phương pháp động lượng cho phép chúng ta giải quyết bài toán hạ gradient mô tả ở trên.
Nhìn vào các vết tối ưu trên, chúng ta có thể tưởng tượng trực quan rằng sẽ tốt hơn nếu lấy trung bình gradient của các bước trước.
Sau cùng, với chiều $x_1$ các gradient là cùng hướng, cách làm này sẽ đơn thuần lấy tổng, từ đó tăng khoảng cách di chuyển ở từng bước.
Ngược lại, gradient dao động mạnh theo hướng $x_2$, từ đó kết hợp các gradient sẽ làm giảm kích thước bước do dao động triệt tiêu lẫn nhau.
Sử dụng $\mathbf{v}_t$ thay vì gradient $\mathbf{g}_t$, ta có các phương trình cập nhật sau:


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

Lưu ý rằng với $\beta = 0$, phương pháp này trở thành thuật toán hạ gradient thông thường.
Trước khi nghiên cứu sâu hơn các tính chất toán học, chúng ta hãy nhìn lướt qua cách thuật toán này hoạt động trong thực tế.


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

Như chúng ta có thể thấy, ngay cả với tốc độ học tương tự như trước đó, phương pháp động lượng vẫn hội tụ tốt.
Chúng ta hãy xem điều gì sẽ xảy ra khi giảm tham số động lượng.
Giảm một nửa giá trị động lượng $\beta = 0.25$ sẽ dẫn đến một quỹ đạo mà rất khó hội tụ.
Tuy nhiên, điều này vẫn tốt hơn rất nhiều so với trường hợp không sử dụng động lượng (khi nghiệm phân kì).


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

Lưu ý rằng chúng ta có thể kết hợp động lượng với SGD và đặc biệt là SGD theo minibatch.
Thay đổi duy nhất trong trường hợp đó là chúng ta sẽ thay thế các gradient $\mathbf{g}_{t, t-1}$ bằng $\mathbf{g}_t$.
Cuối cùng, để thuận tiện chúng ta khởi tạo $\mathbf{v}_0 = 0$ tại thời điểm $t=0$.
Chúng ta hãy xem phép trung bình rò rỉ (leaky average) thực sự làm gì ở các phiên cập nhật.

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
### Effective Sample Weight
-->

### Trọng số mẫu thực sự

<!--
Recall that $\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}$.
In the limit the terms add up to $\sum_{\tau=0}^\infty \beta^\tau = \frac{1}{1-\beta}$.
In other words, rather than taking a step of size $\eta$ in GD or SGD we take a step of size $\frac{\eta}{1-\beta}$ while at the same time, dealing with a potentially much better behaved descent direction.
These are two benefits in one.
To illustrate how weighting behaves for different choices of $\beta$ consider the diagram below.
-->

Hãy nhớ lại rằng $\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}$.
Tại giới hạn, tổng các số hạng là $\sum_{\tau=0}^\infty \beta^\tau = \frac{1}{1-\beta}$.
Nói cách khác, thay vì thực hiện bước có kích thước $\eta$ trong GD hoặc SGD, ta thực hiện bước có kích thước $$\frac{\eta}{1-\beta}$, đồng thời hướng hạ gradient nhiều khả năng cũng ổn định hơn.
Đây là hai lợi ích trong một.
Để minh họa cách trọng số thực hiện cho các lựa chọn khác nhau của $\beta$, hãy xem xét sơ đồ bên dưới.


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

## Thực nghiệm

<!--
Let us see how momentum works in practice, i.e., when used within the context of a proper optimizer.
For this we need a somewhat more scalable implementation.
-->

Hãy xem động lượng hoạt động như thế nào trong thực tế, khi được sử dụng cùng một bộ tối ưu.
Để làm điều này, ta cần các đoạn mã dễ mở rộng hơn.

<!--
### Implementation from Scratch
-->

### Lập trình từ đầu

<!--
Compared with (minibatch) SGD the momentum method needs to maintain a set of  auxiliary variables, i.e., velocity.
It has the same shape as the gradients (and variables of the optimization problem).
In the implementation below we call these variables `states`.
-->

So với SGD hoặc SGD theo minibatch, phương pháp động lượng cần duy trì các biến phụ trợ, chính là vận tốc.
Nó có kích thước giống gradient (và các biến khác trong bài toán tối ưu).
Trong đoạn mã bên dưới, ta gọi các biến vận tốc này là `states`.

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

Hãy xem đoạn mã hoạt động như thế nào trong thực tế.


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

Khi tăng siêu tham số động lượng `momentum` lên 0,9, kích thước mẫu hiệu dụng sẽ tăng lên đáng kể thành $\frac{1}{1 - 0,9} = 10$.
Ta giảm tốc độ học xuống còn $0,01$ để kiểm soát vấn đề.


```{.python .input  n=8}
train_momentum(0.01, 0.9)
```


<!--
Reducing the learning rate further addresses any issue of non-smooth optimization problems.
Setting it to $0.005$ yields good convergence properties.
-->

Tiếp tục giảm tốc độ học sẽ giải quyết bất kỳ vấn đề tối ưu không trơn tru nào.
Giảm còn $0,005$ đem lại các đặc tính hội tụ tốt.


```{.python .input}
train_momentum(0.005, 0.9)
```

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!--
### Concise Implementation
-->

### Lập trình súc tích

<!--
There is very little to do in Gluon since the standard `sgd` solver already had momentum built in.
Setting matching parameters yields a very similar trajectory.
-->

Rất đơn giản nếu sử dụng Gluon vì bộ tối ưu `sgd` tiêu chuẩn đã tích hợp sẵn phương pháp động lượng.
Với cùng các tham số, ta có quỹ đạo rất giống khi lập trình từ đầu.


```{.python .input  n=9}
d2l.train_gluon_ch11('sgd', {'learning_rate': 0.005, 'momentum': 0.9},
                     data_iter)
```


<!--
## Theoretical Analysis
-->

## Phân tích lý thuyết

<!--
So far the 2D example of $f(x) = 0.1 x_1^2 + 2 x_2^2$ seemed rather contrived.
We will now see that this is actually quite representative of the types of problem one might encounter, at least in the case of minimizing convex quadratic objective functions.
-->

Cho đến nay, ví dụ hai chiều $f(x) = 0.1 x_1^2 + 2 x_2^2$ dường như không thực tế cho lắm.
Thực tế, hàm này khá tiêu biểu cho các dạng bài toán có thể gặp phải, ít nhất trong trường hợp cực tiểu hóa các hàm mục tiêu lồi bậc hai.


<!--
### Quadratic Convex Functions
-->

### Hàm lồi bậc hai

<!--
Consider the function
-->

Xét hàm số 


$$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b.$$


<!--
This is a general quadratic function.
For positive semidefinite matrices $\mathbf{Q} \succ 0$, i.e., for matrices with positive eigenvalues 
this has a minimizer at $\mathbf{x}^* = -\mathbf{Q}^{-1} \mathbf{c}$ with minimum value $b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$.
Hence we can rewrite $h$ as
-->

Đây là một hàm bậc hai tổng quát.
Với ma trận xác định dương $\mathbf{Q} \succ 0$, tức ma trận có trị riêng dương, <!-- chỗ này phải là `positive definite` thì trị riêng mới dương và có nghịch đảo -->
hàm có nghiệm cực tiểu tại $\mathbf{x}^* = -\mathbf{Q}^{-1} \mathbf{c}$ với giá trị cực tiểu $b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$. 
Do đó ta có thể viết lại $h$ như sau


$$h(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})^\top \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c}) + b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}.$$


<!--
The gradient is given by $\partial_{\mathbf{x}} f(\mathbf{x}) = \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$.
That is, it is given by the distance between $\mathbf{x}$ and the minimizer, multiplied by $\mathbf{Q}$.
Consequently also the momentum  is a linear combination of terms $\mathbf{Q} (\mathbf{x}_t - \mathbf{Q}^{-1} \mathbf{c})$.
-->

Gradient được tính bởi $\partial_{\mathbf{x}} f(\mathbf{x}) = \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$.
Nghĩa là bằng khoảng cách giữa $\mathbf{x}$ và nghiệm cực tiểu nhân với $\mathbf{Q}$.
Do đó, động lượng cũng là tổ hợp tuyến tính của $\mathbf{Q} (\mathbf{x}_t - \mathbf{Q}^{-1} \mathbf{c})$.

<!--
Since $\mathbf{Q}$ is positive definite it can be decomposed into its eigensystem via 
$\mathbf{Q} = \mathbf{O}^\top \boldsymbol{\Lambda} \mathbf{O}$ for an orthogonal (rotation) matrix $\mathbf{O}$ and a diagonal matrix $\boldsymbol{\Lambda}$ of positive eigenvalues.
This allows us to perform a change of variables from $\mathbf{x}$ to $\mathbf{z} := \mathbf{O} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$ to obtain a much simplified expression:
-->

Vì $\mathbf{Q}$ là xác định dương nên nó có thể được phân tích thành hệ riêng thông qua $\mathbf{Q} = \mathbf{O}^\top \boldsymbol{\Lambda} \mathbf{O}$, với $\mathbf{O}$ là ma trận trực giao (xoay vòng) và $\boldsymbol{\Lambda}$ là ma trận đường chéo của các trị riêng dương.
Điều này cho phép ta đổi biến $\mathbf{x}$ thành $\mathbf{z} := \mathbf{O} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$ để có biểu thức đơn giản hơn nhiều:

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 7 ===================== -->


$$h(\mathbf{z}) = \frac{1}{2} \mathbf{z}^\top \boldsymbol{\Lambda} \mathbf{z} + b'.$$


<!--
Here $c' = b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$.
Since $\mathbf{O}$ is only an orthogonal matrix this does not perturb the gradients in a meaningful way.
Expressed in terms of $\mathbf{z}$ gradient descent becomes
-->

Ở đây $c' = b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$.
Vì $\mathbf{O}$ chỉ là một ma trận trực giao nên điều này không làm nhiễu các gradient một cách có ý nghĩa.
Biểu diễn theo $\mathbf{z}$, ta có hạ gradient


$$\mathbf{z}_t = \mathbf{z}_{t-1} - \boldsymbol{\Lambda} \mathbf{z}_{t-1} = (\mathbf{I} - \boldsymbol{\Lambda}) \mathbf{z}_{t-1}.$$


<!--
The important fact in this expression is that gradient descent *does not mix* between different eigenspaces.
That is, when expressed in terms of the eigensystem of $\mathbf{Q}$ the optimization problem proceeds in a coordinate-wise manner.
This also holds for momentum.
-->

Một điểm quan trọng trong biểu thức này là hạ gradient *không trộn lẫn* các không gian riêng khác nhau.
Nghĩa là, khi được biểu diễn dưới dạng hệ riêng của $\mathbf{Q}$, việc tối ưu được thực hiện theo từng trục tọa độ.
Điều này cũng đúng với phương pháp động lượng.


$$\begin{aligned}
\mathbf{v}_t & = \beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1} \\
\mathbf{z}_t & = \mathbf{z}_{t-1} - \eta \left(\beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1}\right) \\
    & = (\mathbf{I} - \eta \boldsymbol{\Lambda}) \mathbf{z}_{t-1} - \eta \beta \mathbf{v}_{t-1}.
\end{aligned}$$


<!--
In doing this we just proved the following theorem: Gradient Descent with and without momentum for a convex quadratic function decomposes 
into coordinate-wise optimization in the direction of the eigenvectors of the quadratic matrix.
-->

Khi thực hiện điều này, ta đã chứng minh định lý sau: Hạ Gradient có và không có động lượng cho hàm lồi bậc hai có thể được phân tích thành bài toán tối ưu theo hướng các vector riêng của ma trận bậc hai theo từng trục tọa độ.

<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 4 - BẮT ĐẦU ===================================-->

<!--
### Scalar Functions
-->

### Hàm vô hướng

<!--
Given the above result let us see what happens when we minimize the function $f(x) = \frac{\lambda}{2} x^2$. For gradient descent we have
-->

Với kết quả trên hãy xem điều gì xảy ra khi cực tiểu hóa hàm $f(x) = \frac{\lambda}{2} x^2$. Đối với hạ gradient, ta có


$$x_{t+1} = x_t - \eta \lambda x_t = (1 - \eta \lambda) x_t.$$


<!--
Whenever $|1 - \eta \lambda| < 1$ this optimization converges at an exponential rate since after $t$ steps we have $x_t = (1 - \eta \lambda)^t x_0$.
This shows how the rate of convergence improves initially as we increase the learning rate $\eta$ until $\eta \lambda = 1$.
Beyond that things diverge and for $\eta \lambda > 2$ the optimization problem diverges.
-->

Với $|1 - \eta \lambda| < 1$ việc tối ưu này hội tụ theo hàm mũ vì sau $t$ bước ta có $x_t = (1 - \eta \lambda)^t x_0$.
Điều này cho thấy cách tốc độ hội tụ được cải thiện lúc bắt đầu khi tăng tốc độ học $\eta$ cho đến khi $\eta \lambda = 1$.
Vượt qua giá trị đó, khi $\eta \lambda > 2$, bài toán tối ưu hóa sẽ phân kỳ.


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

Để phân tích sự hội tụ khi sử dụng động lượng, chúng ta bắt đầu với việc viết lại các phương trình cập nhật theo hai số vô hướng: một cho $x$ và một cho động lượng $v$.
Ta sẽ có:

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

Chúng ta sử dụng $\mathbf{R}$ kích thước $2 \times 2$ để biểu diễn ma trận chi phối hội tụ.
Sau $t$ bước thì lựa chọn ban đầu $[v_0, x_0]$ trở thành $\mathbf{R}(\ beta, \eta, \lambda)^t [v_0, x_0]$.
Do đó, các trị riêng của $\mathbf{R}$ sẽ quyết định tốc độ hội tụ.
Xem hình ảnh động tại [Distill post](https://distill.pub/2017/momentum/) của :cite:`Goh.2017` và đọc :cite:`Flammarion.Bach.2015` để biết phân tích chi tiết.
Có thể chỉ ra rằng phương pháp động lượng hội tụ với $0 < \eta \lambda < 2 + 2 \beta$.
Khoảng tham số khả thi lớn hơn khi so sánh với $0 < \eta \lambda <2$ của hạ gradient.
Nó cũng gợi ý rằng nhìn chung ta mong muốn $\beta$ có giá trị lớn.
Thông tin chi tiết hơn đòi hỏi nền tảng kiến thức sâu hơn, bạn đọc quan tâm có thể tham khảo các bài báo gốc.

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

* Phương pháp động lượng thay thế gradient bằng trung bình rò rỉ của gradient trong quá khứ. Điều này tăng tốc độ hội tụ đáng kể.
* Phương pháp này có thể sử dụng cho cả hạ gradient không nhiễu và hạ gradient ngẫu nhiên (có nhiễu). 
* Phương pháp động lượng ngăn chặn việc tối ưu hóa bị dừng, điều nhiều khả năng xảy ra đối với hạ gradient ngẫu nhiên.
* Số lượng gradient thực sự là $\frac{1}{1-\beta}$, tính từ tổng trọng số suy giảm theo hàm mũ của dữ liệu trong quá khứ.
* Trong trường hợp các bài toán bậc hai lồi, điều này có thể được phân tích chi tiết một cách tường minh.
* Việc lập trình khá đơn giản nhưng nó yêu cầu lưu trữ thêm một vector trạng thái (động lượng $\mathbf{v}$).

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

1. Sử dụng các kết hợp khác của siêu tham số động lượng và tốc độ học, quan sát và phân tích các kết quả thí nghiệm khác nhau.
2. Hãy thử GD và phương pháp động lượng cho một bài toán bậc hai trong đó có nhiều trị riêng, ví dụ: $f(x) = \frac{1}{2} \sum_i \lambda_i x_i^2$, e.g., $\lambda_i = 2^{-i}$.
Vẽ đồ thị biểu diễn sự giảm xuống của $x$ khi khởi tạo $x_i = 1$.
3. Lấy giá trị và nghiệm cực tiểu cho $h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b$.
4. Điều gì thay đổi khi chúng ta thực hiện SGD và SGD theo mini-batch với phương pháp động lượng? Thử nghiệm với các tham số.

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
* Nguyễn Thanh Hoà

<!-- Phần 3 -->
* Nguyễn Văn Quang
* Nguyễn Văn Cường

<!-- Phần 4 -->
* Nguyễn Văn Quang
* Nguyễn Văn Cường

<!-- Phần 5 -->
* Trần Yến Thy

<!-- Phần 6 -->
* Trần Yến Thy

<!-- Phần 7 -->
* Trần Yến Thy

<!-- Phần 8 -->
* Trần Yến Thy
