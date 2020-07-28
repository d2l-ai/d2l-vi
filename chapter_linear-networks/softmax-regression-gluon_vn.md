<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Concise Implementation of Softmax Regression
-->

# Cách lập trình súc tích Hồi quy Softmax
:label:`sec_softmax_gluon`

<!--
Just as Gluon made it much easier to implement linear regression in :numref:`sec_linear_gluon`, 
we will find it similarly (or possibly more) convenient for implementing classification models.
Again, we begin with our import ritual.
-->

Giống như cách Gluon giúp việc lập trình hồi quy tuyến tính ở :numref:`sec_linear_gluon` trở nên dễ dàng hơn, ta sẽ thấy nó cũng mang đến sự tiện lợi tương tự (hoặc có thể hơn) khi lập trình các mô hình phân loại.
Một lần nữa, chúng ta bắt đầu bằng việc nhập các gói thư viện.

```{.python .input  n=1}
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

<!--
Let's stick with the Fashion-MNIST dataset and keep the batch size at $256$ as in the last section.
-->

Chúng ta sẽ tiếp tục làm việc với bộ dữ liệu Fashion-MNIST và giữ kích thước batch bằng $256$ như ở mục trước.

```{.python .input  n=2}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

<!--
## Initializing Model Parameters
-->

## Khởi tạo Tham số Mô hình

<!--
As mentioned in :numref:`sec_softmax`, the output layer of softmax regression is a fully-connected (`Dense`) layer.
Therefore, to implement our model, we just need to add one `Dense` layer with 10 outputs to our `Sequential`.
Again, here, the `Sequential` is not really necessary, but we might as well form the habit since it will be ubiquitous when implementing deep models.
Again, we initialize the weights at random with zero mean and standard deviation $0.01$.
-->

Như đã đề cập trong :numref:`sec_softmax`, tầng đầu ra của hồi quy softmax là một tầng kết nối đầy đủ (`Dense`).
Do đó, để xây dựng mô hình, ta chỉ cần thêm một tầng `Dense` với 10 đầu ra vào đối tượng `Sequential`.
Việc sử dụng `Sequential` ở đây không thực sự cần thiết, nhưng ta nên hình thành thói quen này vì nó sẽ luôn hiện diện khi ta xây dựng các mô hình sâu.
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

## Hàm Softmax

<!--
In the previous example, we calculated our model's output and then ran this output through the cross-entropy loss.
Mathematically, that is a perfectly reasonable thing to do.
However, from a computational perspective, exponentiation can be a source of numerical stability issues (as discussed  in :numref:`sec_naive_bayes`).
Recall that the softmax function calculates $\hat y_j = \frac{e^{z_j}}{\sum_{i=1}^{n} e^{z_i}}$, 
where $\hat y_j$ is the $j^\mathrm{th}$ element of ``yhat`` and $z_j$ is the $j^\mathrm{th}$ element of the input ``y_linear`` variable, as computed by the softmax.
-->

Ở ví dụ trước, ta đã tính toán kết quả đầu ra của mô hình và sau đó đưa các kết quả này vào hàm mất mát entropy chéo.
Về mặt toán học, cách làm này hoàn toàn có lý.
Tuy nhiên, từ góc độ điện toán, sử dụng hàm mũ có thể là nguồn gốc của các vấn đề về ổn định số học (được bàn trong :numref:`sec_naive_bayes`).
Hãy nhớ rằng, hàm softmax tính $\hat y_j = \frac{e^{z_j}}{\sum_{i=1}^{n} e^{z_i}}$, trong đó $\hat y_j$ là phần tử thứ $j^\mathrm{th}$ của ``yhat`` và $z_j$ là phần tử thứ $j^\mathrm{th}$ của biến đầu vào ``y_linear``.

<!--
If some of the $z_i$ are very large (i.e., very positive), then $e^{z_i}$ might be larger than the largest number we can have for certain types of ``float`` (i.e., overflow).
This would make the denominator (and/or numerator) ``inf`` and we wind up encountering either $0$, ``inf``, or ``nan`` for $\hat y_j$.
In these situations we do not get a well-defined return value for ``cross_entropy``.
One trick to get around this is to first subtract $\text{max}(z_i)$ from all $z_i$ before proceeding with the ``softmax`` calculation.
You can verify that this shifting of each $z_i$ by constant factor does not change the return value of ``softmax``.
-->

Nếu một phần tử $z_i$ quá lớn, $e^{z_i}$ có thể sẽ lớn hơn giá trị cực đại mà kiểu ``float`` có thể biểu diễn được (đây là hiện tượng tràn số trên).
Lúc này mẫu số hoặc tử số (hoặc cả hai) sẽ tiến tới ``inf`` và ta gặp phải trường hợp $\hat y_i$ bằng $0$, ``inf`` hoặc ``nan``.
Trong những tình huống này, giá trị trả về của ``cross_entropy`` có thể không được xác định một cách rõ ràng.
Một mẹo để khắc phục việc này là: đầu tiên ta trừ tất cả các $z_i$ đi $\text{max}(z_i)$, sau đó mới đưa chúng vào hàm ``softmax``.
Bạn có thể nhận thấy rằng việc tịnh tiến mỗi $z_i$ theo một hệ số không đổi sẽ không làm ảnh hưởng đến giá trị trả về của hàm ``softmax``.

<!--
After the subtraction and normalization step, it might be that possible that some $z_j$ have large negative values and thus that the corresponding $e^{z_j}$ will take values close to zero.
These might be rounded to zero due to finite precision (i.e underflow), making $\hat y_j$ zero and giving us ``-inf`` for $\text{log}(\hat y_j)$.
A few steps down the road in backpropagation, we might find ourselves faced with a screenful of the dreaded not-a-number (``nan``) results.
-->

Sau khi thực hiện bước trừ và chuẩn hóa, một vài $z_j$ có thể có giá trị âm lớn và do đó $e^{z_j}$ sẽ xấp xỉ 0.
Điều này có thể dẫn đến việc chúng bị làm tròn thành 0 do khả năng biễu diễn chính xác là hữu hạn (tức tràn số dưới), khiến $\hat y_j$ tiến về không và giá trị $\text{log}(\hat y_j)$ tiến về ``-inf``.
Thực hiện vài bước lan truyền ngược với lỗi trên, ta có thể sẽ đối mặt với một loạt giá trị `nan` (*not-a-number*: *không phải số*) đáng sợ.

<!--
Fortunately, we are saved by the fact that even though we are computing exponential functions, we ultimately intend to take their log (when calculating the cross-entropy loss).
By combining these two operators (``softmax`` and ``cross_entropy``) together, we can escape the numerical stability issues that might otherwise plague us during backpropagation.
As shown in the equation below, we avoided calculating $e^{z_j}$ and can instead $z_j$ directly due to the canceling in $\log(\exp(\cdot))$.
-->

May mắn thay, mặc dù ta đang thực hiện tính toán với các hàm mũ, kết quả cuối cùng ta muốn là giá trị log của nó (khi tính hàm mất mát entropy chéo).
Bằng cách kết hợp cả hai hàm (``softmax`` và ``cross-entropy``) lại với nhau, ta có thể khắc phục vấn đề về ổn định số học và tránh gặp khó khăn trong quá trình lan truyền ngược.
Trong phương trình bên dưới, ta đã không tính $e^{z_j}$ mà thay vào đó, ta tính trực tiếp $z_j$ do việc rút gọn $\log(\exp(\cdot))$.

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

Ta vẫn muốn giữ lại hàm softmax gốc để sử dụng khi muốn tính đầu ra của mô hình dưới dạng xác suất.
Nhưng thay vì truyền xác suất softmax vào hàm mất mát mới, ta sẽ chỉ truyền các giá trị logit (các giá trị khi chưa qua softmax) và tính softmax cùng log của nó trong hàm mất mát `softmax_cross_entropy`.
Hàm này cũng sẽ tự động thực hiện các mẹo thông minh như log-sum-exp ([xem thêm Wikipedia](https://en.wikipedia.org/wiki/LogSumExp)).

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

## Thuật toán Tối ưu

<!--
Here, we use minibatch stochastic gradient descent with a learning rate of $0.1$ as the optimization algorithm.
Note that this is the same as we applied in the linear regression example and it illustrates the general applicability of the optimizers.
-->

Ở đây, chúng ta sử dụng thuật toán tối ưu hạ gradient ngẫu nhiên theo minibatch với tốc độ học bằng $0.1$.
Lưu ý rằng cách làm này giống hệt cách làm ở ví dụ về hồi quy tuyến tính, minh chứng cho tính khái quát của bộ tối ưu hạ gradient ngẫu nhiên.

```{.python .input  n=5}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

<!--
## Training
-->

## Huấn luyện

<!--
Next we call the training function defined in the last section to train a model.
-->

Tiếp theo, chúng ta sẽ gọi hàm huấn luyện đã được khai báo ở mục trước để huấn luyện mô hình.

```{.python .input  n=6}
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

<!--
As before, this algorithm converges to a solution that achieves an accuracy of 83.7%, albeit this time with fewer lines of code than before.
Note that in many cases, Gluon takes additional precautions beyond these most well-known tricks to ensure numerical stability, 
saving us from even more pitfalls that we would encounter if we tried to code all of our models from scratch in practice.
-->

Giống lần trước, thuật toán hội tụ tới một mô hình có độ chính xác 83.7% nhưng chỉ khác là cần ít dòng lệnh hơn.
Lưu ý rằng trong nhiều trường hợp, Gluon không chỉ dùng các mánh phổ biến mà còn sử dụng các kỹ thuật khác để tránh các lỗi kĩ thuật tính toán mà ta dễ gặp phải nếu tự lập trình mô hình từ đầu.

<!--
## Exercises
-->

## Bài tập

<!--
1. Try adjusting the hyper-parameters, such as batch size, epoch, and learning rate, to see what the results are.
2. Why might the test accuracy decrease again after a while? How could we fix this?
-->

1. Thử thay đổi các siêu tham số như kích thước batch, số epoch và tốc độ học. Theo dõi kết quả sau khi thay đổi.
2. Tại sao độ chính xác trên tập kiểm tra lại giảm sau một khoảng thời gian? Chúng ta giải quyết việc này thế nào?

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
* Nguyễn Duy Du
* Vũ Hữu Tiệp
* Lê Khắc Hồng Phúc
* Lý Phi Long
* Phạm Minh Đức
* Dương Nhật Tân
* Nguyễn Văn Tâm
* Phạm Hồng Vinh
