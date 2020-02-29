<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Concise Implementation of Softmax Regression
-->

# Triển khai súc tích của Hồi quy Softmax
:label:`sec_softmax_gluon`

<!--
Just as Gluon made it much easier to implement linear regression in :numref:`sec_linear_gluon`, 
we will find it similarly (or possibly more) convenient for implementing classification models.
Again, we begin with our import ritual.
-->

Giống như cách Gluon giúp việc cài đặt hồi quy tuyến tính ở :numref:`sec_linear_gluon` trở nên dễ dàng hơn,
chúng ta sẽ thấy nó cũng sẽ mang đến sự tiện lợi tương tự (hoặc có thể hơn) cho việc triển khai các mô hình phân loại.
Một lần nữa, chúng ta bắt đầu bằng việc import.

```{.python .input  n=1}
import d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

<!--
Let's stick with the Fashion-MNIST dataset and keep the batch size at $256$ as in the last section.
-->

Chúng ta tiếp tục làm việc với bộ dữ liệu Fashion-MNIST và giữ kích cỡ batch bằng $256$ như ở phần trước.

```{.python .input  n=2}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

<!--
## Initializing Model Parameters
-->

## Khởi tạo tham số mô hình

<!--
As mentioned in :numref:`sec_softmax`, the output layer of softmax regression is a fully-connected (`Dense`) layer.
Therefore, to implement our model, we just need to add one `Dense` layer with 10 outputs to our `Sequential`.
Again, here, the `Sequential` is not really necessary, but we might as well form the habit since it will be ubiquitous when implementing deep models.
Again, we initialize the weights at random with zero mean and standard deviation $0.01$.
-->

Như đã đề cập trong :numref:`sec_softmax`, tầng output của hồi quy softmax là một tầng kết nối đầy đủ (`Dense`).
Do đó, để triển khai mô hình, chúng ta chỉ cần thêm một tầng `Dense` với 10 output vào đối tượng `Sequential`.
Ở đây, việc sử dụng `Sequential` không thực sự cần thiết, nhưng ta nên hình thành thói quen sử dụng vì nó sẽ luôn hiện diện khi ta cài đặt các mô hình học sâu.
Một lần nữa, chúng ta khởi tạo các trọng số một cách ngẫu nhiên với trung bình bằng không và độ lệch chuẩn bằng $0.01$.

```{.python .input  n=3}
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## The Softmax
-->

## *dịch tiêu đề phía trên*

<!--
In the previous example, we calculated our model's output and then ran this output through the cross-entropy loss.
Mathematically, that is a perfectly reasonable thing to do.
However, from a computational perspective, exponentiation can be a source of numerical stability issues (as discussed  in :numref:`sec_naive_bayes`).
Recall that the softmax function calculates $\hat y_j = \frac{e^{z_j}}{\sum_{i=1}^{n} e^{z_i}}$, 
where $\hat y_j$ is the $j^\mathrm{th}$ element of ``yhat`` and $z_j$ is the $j^\mathrm{th}$ element of the input ``y_linear`` variable, as computed by the softmax.
-->

*dịch đoạn phía trên*

<!--
If some of the $z_i$ are very large (i.e., very positive), then $e^{z_i}$ might be larger than the largest number we can have for certain types of ``float`` (i.e., overflow).
This would make the denominator (and/or numerator) ``inf`` and we wind up encountering either $0$, ``inf``, or ``nan`` for $\hat y_j$.
In these situations we do not get a well-defined return value for ``cross_entropy``.
One trick to get around this is to first subtract $\text{max}(z_i)$ from all $z_i$ before proceeding with the ``softmax`` calculation.
You can verify that this shifting of each $z_i$ by constant factor does not change the return value of ``softmax``.
-->

*dịch đoạn phía trên*

<!--
After the subtraction and normalization step, it might be that possible that some $z_j$ have large negative values and thus that the corresponding $e^{z_j}$ will take values close to zero.
These might be rounded to zero due to finite precision (i.e underflow), making $\hat y_j$ zero and giving us ``-inf`` for $\text{log}(\hat y_j)$.
A few steps down the road in backpropagation, we might find ourselves faced with a screenful of the dreaded not-a-number (``nan``) results.
-->

*dịch đoạn phía trên*

<!--
Fortunately, we are saved by the fact that even though we are computing exponential functions, we ultimately intend to take their log (when calculating the cross-entropy loss).
By combining these two operators (``softmax`` and ``cross_entropy``) together, we can escape the numerical stability issues that might otherwise plague us during backpropagation.
As shown in the equation below, we avoided calculating $e^{z_j}$ and can instead $z_j$ directly due to the canceling in $\log(\exp(\cdot))$.
-->

*dịch đoạn phía trên*

$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{e^{z_j}}{\sum_{i=1}^{n} e^{z_i}}\right) \\
& = \log{(e^{z_j})}-\text{log}{\left( \sum_{i=1}^{n} e^{z_i} \right)} \\
& = z_j -\log{\left( \sum_{i=1}^{n} e^{z_i} \right)}.
\end{aligned}
$$

<!--
We will want to keep the conventional softmax function handy in case we ever want to evaluate the probabilities output by our model.
But instead of passing softmax probabilities into our new loss function, we will just pass the logits and compute the softmax and its log all at once inside the softmax_cross_entropy loss function, 
which does smart things like the log-sum-exp trick ([see on Wikipedia](https://en.wikipedia.org/wiki/LogSumExp)).
-->

*dịch đoạn phía trên*

```{.python .input  n=4}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Optimization Algorithm
-->

## *dịch tiêu đề phía trên*

<!--
Here, we use minibatch stochastic gradient descent with a learning rate of $0.1$ as the optimization algorithm.
Note that this is the same as we applied in the linear regression example and it illustrates the general applicability of the optimizers.
-->

*dịch đoạn phía trên*

```{.python .input  n=5}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

<!--
## Training
-->

## *dịch tiêu đề phía trên*

<!--
Next we call the training function defined in the last section to train a model.
-->

*dịch đoạn phía trên*

```{.python .input  n=6}
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

<!--
As before, this algorithm converges to a solution that achieves an accuracy of 83.7%, albeit this time with fewer lines of code than before.
Note that in many cases, Gluon takes additional precautions beyond these most well-known tricks to ensure numerical stability, 
saving us from even more pitfalls that we would encounter if we tried to code all of our models from scratch in practice.
-->

*dịch đoạn phía trên*

<!--
## Exercises
-->

## *dịch tiêu đề phía trên*

<!--
1. Try adjusting the hyper-parameters, such as batch size, epoch, and learning rate, to see what the results are.
2. Why might the test accuracy decrease again after a while? How could we fix this?
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2337)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2337)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

<!--
![](../img/qr_softmax-regression-gluon.svg)
-->


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.

* Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md.
-->

* Đoàn Võ Duy Thanh
<!-- Phần 1 -->
* Nguyễn Duy Du

<!-- Phần 2 -->
*

<!-- Phần 3 -->
*
