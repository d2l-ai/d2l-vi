<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Learning Rate Scheduling
-->

# Định thời Tốc độ Học 
:label:`sec_scheduler`

<!--
So far we primarily focused on optimization *algorithms* for how to update the weight vectors rather than on the *rate* at which they are being updated.
Nonetheless, adjusting the learning rate is often just as important as the actual algorithm.
There are a number of aspects to consider:
-->

Cho đến nay ta tập trung chủ yếu vào *thuật toán* tối ưu ở cách cập nhật các vector trọng số thay vì *tốc độ* cập nhật các vector đó.
Tuy nhiên, thường thì điều chỉnh tốc độ học cũng quan trọng như thuật toán.
Có một vài khía cạnh để chúng ta xem xét:

<!--
* Most obviously the *magnitude* of the learning rate matters. 
If it is too large, optimization diverges, if it is too small, it takes too long to train or we end up with a suboptimal result. 
We saw previously that the condition number of the problem matters (see e.g., :numref:`sec_momentum` for details). 
Intuitively it is the ratio of the amount of change in the least sensitive direction vs. the most sensitive one.
* Secondly, the rate of decay is just as important. 
If the learning rate remains large we may simply end up bouncing around the minimum and thus not reach optimality. 
:numref:`sec_minibatch_sgd` discussed this in some detail and we analyzed performance guarantees in :numref:`sec_sgd`. 
In short, we want the rate to decay, but probably more slowly than $\mathcal{O}(t^{-\frac{1}{2}})$ which would be a good choice for convex problems.
* Another aspect that is equally important is *initialization*.
This pertains both to how the parameters are set initially (review :numref:`sec_numerical_stability` for details) and also how they evolve initially.
This goes under the moniker of *warmup*, i.e., how rapidly we start moving towards the solution initially.
Large steps in the beginning might not be beneficial, in particular since the initial set of parameters is random.
The initial update directions might be quite meaningless, too.
* Lastly, there are a number of optimization variants that perform cyclical learning rate adjustment. 
This is beyond the scope of the current chapter. 
We recommend the reader to review details in :cite:`Izmailov.Podoprikhin.Garipov.ea.2018`, e.g., how to obtain better solutions by averaging over an entire *path* of parameters.
-->

* Vấn đề rõ nhất là *độ lớn* của tốc độ học.
Nếu quá lớn thì tối ưu phân kỳ, nếu quá nhỏ thì việc huấn luyện mất quá nhiều thời gian hoặc kết quả cuối cùng không đủ tốt.
Trước đây ta đã thấy rằng số điều kiện (*condition number*) của bài toán rất quan trọng (xem :numref:`sec_momentum` để biết thêm chi tiết).
Theo trực giác, nó là tỷ lệ giữa mức độ thay đổi theo hướng ít nhạy cảm nhất và hướng nhạy cảm nhất.
* Thứ hai, tốc độ suy giảm cũng quan trọng tương đương.
Nếu tốc độ học còn lớn, ta có thể chỉ chạy xung quanh cực tiểu và do đó không đạt được nghiệm tối ưu.
:numref:`sec_minibatch_sgd` đã thảo luận một số chi tiết về vấn đề này và :numref:`sec_sgd` đã phân tích các đảm bảo hội tụ. 
Nói tóm lại, chúng ta muốn suy giảm tốc độ hội tụ, có thể chậm hơn $\mathcal{O}(t^{-\frac{1}{2}})$, một lựa chọn tốt cho các bài toán hàm lồi.
* Một khía cạnh khác cũng quan trọng không kém là *khởi tạo*.
Điều này liên quan đến cả cách thức các tham số được đặt (xem lại :numref:`sec_numerical_stability`) và cả cách chúng thay đổi ban đầu.
Có thể gọi đây là *khởi động (warmup)*, tức ta bắt đầu tối ưu nhanh như thế nào.
Bước tối ưu lớn khi bắt đầu có thể không có lợi, cụ thể vì bộ tham số ban đầu là ngẫu nhiên.
Các hướng cập nhật ban đầu cũng có thể không quan trọng.
* Cuối cùng, có một số biến thể tối ưu hóa thực hiện điều chỉnh tốc độ học theo chu kỳ.
Điều này nằm ngoài phạm vi của chương hiện tại.
Độc giả có thể đọc thêm tại :cite:`Izmailov.Podoprikhin.Garipov.ea.2018`, ví dụ về làm thế nào để có các giải pháp tốt hơn bằng cách lấy trung bình trên toàn bộ *đường đi* của các tham số.

<!--
Given the fact that there is a lot of detail needed to manage learning rates, most deep learning frameworks have tools to deal with this automatically.
In the current chapter we will review the effects that different schedules have on accuracy and also show how this can be managed efficiently via a *learning rate scheduler*.
-->

Vì việc quản lý tốc độ học khá vất vả, hầu hết các framework học sâu đều có các công cụ tự động giải quyết điều này.
Trong phần này ta sẽ xem xét ảnh hưởng của các định thời khác nhau lên độ chính xác, cũng như xem cách quản lý hiệu quả tốc độ học thông qua một *bộ định thời tốc độ học*.

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Toy Problem
-->

## Ví dụ Đơn giản

<!--
We begin with a toy problem that is cheap enough to compute easily, yet sufficiently nontrivial to illustrate some of the key aspects.
For that we pick a slightly modernized version of LeNet (`relu` instead of `sigmoid` activation, MaxPooling rather than AveragePooling), as applied to Fashion-MNIST.
Moreover, we hybridize the network for performance.
Since most of the code is standard we just introduce the basics without further detailed discussion.
See :numref:`chap_cnn` for a refresher as needed.
-->

Hãy bắt đầu với một ví dụ đơn giản với ít chi phí tính toán nhưng đủ để minh họa một vài điểm cốt lõi.
Ta sử dụng LeNet cải tiến (thay thế hàm kích hoạt `sigmoid` bằng `relu` và hàm gộp trung bình bằng hàm gộp cực đại) và áp dụng trên tập dữ liệu Fashion-MNIST.
Hơn nữa, để có hiệu năng tốt, ta lai hoá mạng.
Vì hầu hết mã nguồn tương tự như trước, ta sẽ không thảo luận chi tiết.
Xem lại :numref:`chap_cnn` để biết thêm chi tiết nếu cần.


```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, lr_scheduler, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.HybridSequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10))
net.hybridize()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
ctx = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to "d2l.train_ch6" that defined in the lenet
# section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, ctx):
    net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            X, y = X.as_in_ctx(ctx), y.as_in_ctx(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            train_loss, train_acc = metric[0]/metric[2], metric[1]/metric[2]
            if (i+1) % 50 == 0:
                animator.add(epoch + i/len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))
    print('train loss %.3f, train acc %.3f, test acc %.3f' % (
        train_loss, train_acc, test_acc))
```


<!--
Let us have a look at what happens if we invoke this algorithm with default settings, such as a learning rate of $0.3$ and train for $30$ iterations.
Note how the training accuracy keeps on increasing while progress in terms of test accuracy stalls beyond a point.
The gap between both curves indicates overfitting.
-->

Ta hãy xem điều gì sẽ xảy ra khi ta gọi thuật toán với các thiết lập mặc định, chẳng hạn tốc độ học bằng $0.3$ và huấn luyện với $30$ epoch.
Lưu ý rằng độ chính xác trên tập huấn luyện vẫn tiếp tục tăng trong khi độ chính xác trên tập kiểm tra không tăng thêm khi đạt giá trị nào đó.
Khoảng cách giữa hai đường cong cho thấy độ quá khớp của thuật toán.


```{.python .input}
lr, num_epochs = 0.3, 30
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, num_epochs, loss, trainer, ctx)
```


<!--
## Schedulers
-->

## Bộ Định thời

<!--
One way of adjusting the learning rate is to set it explicitly at each step.
This is conveniently achieved by the `set_learning_rate` method.
We could adjust it downward after every epoch (or even after every minibatch), e.g., in a dynamic manner in response to how optimization is progressing.
-->

Một cách để điều chỉnh tốc độ học là thiết lập giá trị của tốc độ học tường minh ở mỗi bước lặp.
Điều này có thể đạt được bằng phương thức `set_learning_rate`.
Ta có thể hạ giá trị tốc độ học xuống sau mỗi epoch (hay thậm chí sau mỗi minibatch) như là một cách phản hồi khi quá trình tối ưu đang diễn ra.


```{.python .input}
trainer.set_learning_rate(0.1)
print('Learning rate is now %.2f' % trainer.learning_rate)
```


<!--
More generally we want to define a scheduler.
When invoked with the number of updates it returns the appropriate value of the learning rate.
Let us define a simple one that sets the learning rate to $\eta = \eta_0 (t + 1)^{-\frac{1}{2}}$.
-->

Tổng quát hơn, ta muốn định nghĩa một bộ định thời.
Khi được gọi bằng cách truyền số bước cập nhật, bộ định thời trả về giá trị tương ứng của tốc độ học.
Ta hãy định nghĩa một bộ định thời đơn giản có tốc độ học $\eta = \eta_0 (t + 1)^{-\frac{1}{2}}$.


```{.python .input}
class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)
```

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
Let us plot its behavior over a range of values.
-->

Chúng ta hãy vẽ hành vi của bộ định thời trên một dải giá trị. 


```{.python .input}
scheduler = SquareRootScheduler(lr=1.0)
d2l.plot(np.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```


<!--
Now let us see how this plays out for training on Fashion-MNIST.
We simply provide the scheduler as an additional argument to the training algorithm.
-->

Giờ hãy xem bộ định thời này hoạt động thế nào khi huấn luyện trên Fashion-MNIST.
Chúng ta đơn giản đưa bộ định thời vào giải thuật huấn luyện như một đối số bổ sung.


```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, ctx)
```


<!--
This worked quite a bit better than previously.
Two things stand out: the curve was rather more smooth than previously.
Secondly, there was less overfitting.
Unfortunately it is not a well-resolved question as to why certain strategies lead to less overfitting in *theory*.
There is some argument that a smaller stepsize will lead to parameters that are closer to zero and thus simpler.
However, this does not explain the phenomenon entirely since we do not really stop early but simply reduce the learning rate gently.
-->

Phương pháp này làm việc tốt hơn một chút so với phương pháp trước. Nổi bật hơn là đồ thị quá trình học mượt hơn và ít quá khớp hơn.
Không may là chưa có lời giải thích ổn thỏa nào cho câu hỏi liên quan tới việc tại sao những chiến lược như vậy lại dẫn đến việc giảm quá khớp về mặt lý thuyết.
Có một số nhận định rằng kích thước bước nhỏ hơn sẽ đưa các tham số tới gần giá trị không hơn và do đó đơn giản hơn.
Tuy nhiên, điều này không giải thích hoàn toàn hiện tượng này vì chúng ta thật sự không hề dừng giải thuật sớm mà đơn giản chỉ giảm từ từ tốc độ học. 

<!--
## Policies
-->

## Những chính sách

<!--
While we cannot possibly cover the entire variety of learning rate schedulers, we attempt to give a brief overview of popular policies below.
Common choices are polynomial decay and piecewise constant schedules.
Beyond that, cosine learning rate schedules have been found to work well empirically on some problems.
Lastly, on some problems it is beneficial to warm up the optimizer prior to using large learning rates.
-->

Vì không đủ khả năng xem xét toàn bộ các loại bộ định thời tốc độ học, chúng tôi cố gắng để đưa ra một bản tóm lược khái quát về các chiến lược phổ biến dưới đây.
Những lựa chọn thông thường là định thời suy giảm theo đa thức và định thời hằng số theo từng khoảng. 
Xa hơn nữa, thực nghiệm cho thấy các bộ định thời theo hàm cô-sin làm việc tốt đối với một số bài toán.
Sau cùng, với một số bài toán sẽ có lợi khi ta từ từ nâng dần tốc độ học cho bộ tối ưu trước khi sử dụng các tốc độ học lớn. 

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
### Factor Scheduler
-->

### Định thời Thừa số

<!--
One alternative to a polynomial decay would be a multiplicative one, that is $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$ for $\alpha \in (0, 1)$.
To prevent the learning rate from decaying beyond a reasonable lower bound the update equation is often modified to $\eta_{t+1} \leftarrow \mathop{\mathrm{max}}(\eta_{\mathrm{min}}, \eta_t \cdot \alpha)$.
-->

Một giải pháp thay thế cho suy giảm đa thức đó là sử dụng thừa số nhân $\alpha \in (0, 1)$, lúc này $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$.
Để tránh trường hợp tốc độ học suy giảm thấp hơn cả biên chặn dưới, phương trình cập nhật thường được sửa lại thành $\eta_{t+1} \leftarrow \mathop{\mathrm{max}}(\eta_{\mathrm{min}}, \eta_t \cdot \alpha)$.


```{.python .input}
class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
d2l.plot(np.arange(50), [scheduler(t) for t in range(50)])
```


<!--
This can also be accomplished by a built-in scheduler in MXNet via the `lr_scheduler.FactorScheduler` object.
It takes a few more parameters, such as warmup period, warmup mode (linear or constant), the maximum number of desired updates, etc.
Going forward we will use the built-in schedulers as appropriate and only explain their functionality here.
As illustrated, it is fairly straightforward to build your own scheduler if needed.
-->

Cách trên cũng có thể được thực hiện bằng một bộ định thời có sẵn trong MXNet `lr_scheduler.FactorScheduler`.
Cách này yêu cầu nhiều tham số hơn một chút, ví dụ như thời gian khởi động (_warmup period_), chế độ khởi động (_warmup mode_), số bước cập nhật tối đa, v.v.
Ở các phần tiếp theo, chúng ta sẽ sử dụng các bộ định thời tốc độ học được lập trình sẵn, ở đây chỉ giải thích cách thức hoạt động của chúng.
Như minh họa, khá đơn giản để xây dựng một định thời của riêng bạn nếu cần thiết.

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
### Multi Factor Scheduler
-->

### Định thời Đa Thừa số

<!--
A common strategy for training deep networks is to keep the learning rate piecewise constant and to decrease it by a given amount every so often.
That is, given a set of times when to decrease the rate, such as $s = \{5, 10, 20\}$ decrease $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$ whenever $t \in s$.
Assuming that the values are halved at each step we can implement this as follows.
-->

Một chiến lược chung để huấn luyện các mạng nơ-ron sâu là giữ cho tốc độ học không đổi theo từng khoảng và thường xuyên giảm tốc độ học đi một lượng cho trước sau mỗi khoảng.
Cụ thể, với một tập thời điểm giảm tốc độ học, ví dụ như với $s = \{15, 30\}$, ta giảm $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$ khi $t \in s$.
Giả sử rằng tốc độ học được giảm một nửa tại mỗi bước thời gian trên, ta có thể lập trình như sau.


```{.python .input}
scheduler = lr_scheduler.MultiFactorScheduler(step=[15, 30], factor=0.5,
                                              base_lr=0.5)
d2l.plot(np.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```


<!--
The intuition behind this piecewise constant learning rate schedule is that one lets optimization proceed until a stationary point has been reached in terms of the distribution of weight vectors.
Then (and only then) do we decrease the rate such as to obtain a higher quality proxy to a good local minimum.
The example below shows how this can produce ever slightly better solutions.
-->

Ý tưởng trực quan đằng sau định thời tốc độ học không đổi theo khoảng đó là phương pháp này cho phép quá trình tối ưu xảy ra cho tới khi thuật toán đạt tới điểm ổn định về phân phối của các vector trọng số.
Khi và chỉ khi đạt được trạng thái đó, chúng ta mới giảm tốc độ học hướng tới điểm cực tiểu chất lượng hơn.
Ví dụ dưới đây cho ta thấy cách phương pháp này giúp tìm được nghiệm tốt hơn đôi chút.


```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, ctx)
```

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
### Cosine Scheduler
-->

### Định thời Cô-sin

<!--
A rather perplexing heuristic was proposed by :cite:`Loshchilov.Hutter.2016`.
It relies on the observation that we might not want to decrease the learning rate too drastically in the beginning and moreover, 
that we might want to "refine" the solution in the end using a very small learning rate.
This results in a cosine-like schedule with the following functional form for learning rates in the range $t \in [0, T]$.
-->

Đây là một phương pháp khá phức tạp dựa trên thực nghiệm được đề xuất bởi :cite:`Loshchilov.Hutter.2016`.
Phương pháp dựa trên quan sát rằng ta có thể không muốn giảm tốc độ học quá nhanh ở giai đoạn đầu và hơn nữa ta muốn làm mịn nghiệm thu được ở giai đoạn cuối của quá trình tối ưu bằng cách sử dụng tốc độ học rất nhỏ.
Từ đó ta thu được một định thời giống cô-sin với tốc độ học trong khoảng $t \in [0, T]$ có công thức như sau.


$$\eta_t = \eta_T + \frac{\eta_0 - \eta_T}{2} \left(1 + \cos(\pi t/T)\right)$$


<!--
Here $\eta_0$ is the initial learning rate, $\eta_T$ is the target rate at time $T$.
Furthermore, for $t > T$ we simply pin the value to $\eta_T$ without increasing it again.
In the following example, we set the max update step $T = 20$.
-->

Trong đó $\eta_0$ là tốc độ học ban đầu, $\eta_T$ được tốc độ học đích tại thời điểm $T$.
Hơn nữa, với $t > T$ ta không tăng giá trị tốc độ học mà đơn giản gán nó bằng $\eta_T$.
Trong ví dụ sau, chúng ta thiết lập số bước cập nhật tối đa $T = 20$.


```{.python .input}
scheduler = lr_scheduler.CosineScheduler(max_update=20, base_lr=0.5,
                                         final_lr=0.01)
d2l.plot(np.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```


<!--
In the context of computer vision this schedule *can* lead to improved results. 
Note, though, that such improvements are not guaranteed (as can be seen below).
-->

Trong ngữ cảnh thị giác máy tính, cách định thời này *có thể* cải thiện kết quả thu được.
Tuy nhiên, chú ý rằng những cải thiện này không chắc chắn được đảm đảo (có thể thấy qua ví dụ dưới đây).


```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, ctx)
```

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!--
### Warmup
-->

### Khởi động

<!--
In some cases initializing the parameters is not sufficient to guarantee a good solution.
This particularly a problem for some advanced network designs that may lead to unstable optimization problems.
We could address this by choosing a sufficiently small learning rate to prevent divergence in the beginning.
Unfortunately this means that progress is slow.
Conversely, a large learning rate initially leads to divergence.
-->

Trong một số trường hợp, khởi tạo tham số không đảm bảo sẽ có kết quả tốt. 
Đặc biệt đối với các mạng phức tạp, nó có thể làm việc tối ưu hóa không ổn định. 
Chúng ta có thể giải quyết việc này bằng cách chọn tốc độ học đủ nhỏ để ngăn phân kỳ vào lúc bắt đầu. Tuy nhiên, tiến trình học sẽ chậm hơn. 
Ngược lại, tốc độ học lớn ban đầu cũng gây ra phân kỳ.

<!--
A rather simple fix for this dilemma is to use a warmup period during which the learning rate *increases* to its initial maximum and to cool down the rate until the end of the optimization process.
For simplicity one typically uses a linear increase for this purpose.
This leads to a schedule of the form indicated below.
-->
Một giải pháp đơn giản cho vấn đề trên là dùng quá trình khởi động (*warmup*), trong thời gian đó tốc độ học *tăng* tới giá trị lớn nhất, sau đó giảm dần tới khi kết thúc quá trình tối ưu.
Để đơn giản, ta có thể dụng hàm tăng tuyến tính để khởi động. 
Kết quả là ta có một bộ định thời dưới đây.


```{.python .input}
scheduler = lr_scheduler.CosineScheduler(20, warmup_steps=5, base_lr=0.5,
                                         final_lr=0.01)
d2l.plot(np.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```


<!--
Note that the network converges better initially (in particular observe the performance during the first 5 epochs).
-->

Có thể thấy rằng ban đầu, mạng hội tụ tốt hơn (cụ thể, hãy quan sát quá trình tối ưu trong 5 epoch đầu tiên).

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, ctx)
```


<!--
Warmup can be applied to any scheduler (not just cosine).
For a more detailed discussion of learning rate schedules and many more experiments see also :cite:`Gotmare.Keskar.Xiong.ea.2018`.
In particular they find that a warmup phase limits the amount of divergence of parameters in very deep networks.
This makes intuitively sense since we would expect significant divergence due to random initialization in those parts of the network that take the most time to make progress in the beginning.
-->

Phép khởi động có thể sử dụng trong bất kỳ bộ định thời nào (không chỉ là cosine).
Để biết thêm chi tiết thảo luận và các thí nghiệm về định thời tốc độ học, có thể đọc thêm :cite:`Gotmare.Keskar.Xiong.ea.2018`.
Đáng chú ý là các tác giả thấy rằng quá trình khởi động làm giảm lượng phân kì của tham số trong các mạng rất sâu. 
Điều này hợp lý về trực giác, vì ta thấy rằng phân kỳ mạnh là do khởi tạo tham số ngẫu nhiên ở những phần mạng học lâu nhất vào lúc đầu.

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 7 ===================== -->

<!--
## Summary
-->

## Tóm tắt

<!--
* Decreasing the learning rate during training can lead to improved accuracy and (most perplexingly) reduced overfitting of the model.
* A piecewise decrease of the learning rate whenever progress has plateaued is effective in practice. 
Essentially this ensures that we converge efficiently to a suitable solution and only then reduce the inherent variance of the parameters by reducing the learning rate.
* Cosine schedulers are popular for some computer vision problems. See e.g., [GluonCV](http://gluon-cv.mxnet.io) for details of such a scheduler.
* A warmup period before optimization can prevent divergence.
* Optimization serves multiple purposes in deep learning. Besides minimizing the training objective, 
different choices of optimization algorithms and learning rate scheduling can lead to rather different amounts of generalization and overfitting on the test set (for the same amount of training error).
-->

* Giảm tốc độ học trong huấn luyện có thể cải thiện độ chính xác và giảm tính quá khớp của mô hình.
* Một cách rất hiệu quả trong thực tế đó là giảm tốc độ học theo khoảng bất cứ khi nào quá trình tối ưu không có tiến bộ đáng kể (_plateau_).
Về cơ bản, định thời trên đảm bảo quá trình tối ưu sẽ hội tụ đến nghiệm phù hợp và chỉ sau đó mới giảm phương sai vốn có của các tham số bằng cách giảm tốc độ học.
* Định thời cô-sin khá phổ biến trong các bài toán thị giác máy tính. Xem ví dụ [GluonCV](http://gluon-cv.mxnet.io) để biết thêm chi tiết về định thời này.
* Quá trình khởi động trước khi tối ưu có thể giúp tránh phân kỳ.
* Tối ưu hóa phục vụ nhiều mục đích trong việc học sâu. Bên cạnh việc cực tiểu hoá hàm mục tiêu trên tập huấn luyện, các thuật toán tối ưu và các định thời tốc độ học khác nhau có thể thay đổi tính khái quát hoá và tính quá khớp trên tập kiểm tra (đối với cùng một giá trị lỗi trên tập huấn luyện).

<!--
## Exercises
-->

## Bài tập

<!--
1. Experiment with the optimization behavior for a given fixed learning rate. What is the best model you can obtain this way?
2. How does convergence change if you change the exponent of the decrease in the learning rate? Use `PolyScheduler` for your convenience in the experiments.
3. Apply the cosine scheduler to large computer vision problems, e.g., training ImageNet. How does it affect performance relative to other schedulers?
4. How long should warmup last?
5. Can you connect optimization and sampling? Start by using results from :cite:`Welling.Teh.2011` on Stochastic Gradient Langevin Dynamics.
-->

1. Hãy thí nghiệm về cách hoạt động của thuật toán tối ưu với một tốc độ học cố định cho trước. Hãy cho biết mô hình tốt nhất mà bạn có thể có được theo cách này?
2. Quá trình hội tụ thay đổi như thế nào nếu bạn thay đổi lũy thừa giảm trong tốc độ học? Để thuận tiện, hãy sử dụng `PolyScheduler`.
3. Hãy áp dụng định thời cô-sin cho nhiều bài toán thị giác máy tính, ví dụ, huấn luyện trên tập ImageNet. Hãy chỉ ra những ảnh hưởng của phương pháp này tới chất lượng của mô hình thu được so với các định thời khác.
4. Quá trình khởi động nên kéo dài bao lâu?
5. Bạn có thể liên hệ tối ưu hoá và phép lấy mẫu được không? Hãy bắt đầu bằng cách sử dụng kết quả từ  :cite:`Welling.Teh.2011` về động lực học Langevin của Gradient ngẫu nghiên (_Stochastic Gradient Langevin Dynamics_).

<!-- ===================== Kết thúc dịch Phần 7 ===================== -->
<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->


## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/5183)
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
* Nguyễn Văn Cường

<!-- Phần 2 -->
* Nguyễn Văn Quang
* Nguyễn Văn Cường

<!-- Phần 3 -->
* Nguyễn Mai Hoàng Long

<!-- Phần 4 -->
* Nguyễn Văn Quang
* Nguyễn Văn Cường


<!-- Phần 5 -->
* Nguyễn Văn Quang

<!-- Phần 6 -->
* Hoang Van-Tien

<!-- Phần 7 -->
* Nguyễn Văn Quang
