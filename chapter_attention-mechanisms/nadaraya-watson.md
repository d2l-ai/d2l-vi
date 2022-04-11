# Chú ý Pooling: Hồi quy hạt nhân Nadaraya-Watson
:label:`sec_nadaraya-watson`

Bây giờ bạn đã biết các thành phần chính của các cơ chế chú ý theo khuôn khổ trong :numref:`fig_qkv`. Để tái lập lại, các tương tác giữa các truy vấn (tín hiệu ý chí) và các phím (tín hiệu không có ý định) dẫn đến *chú ý cùng*. Sự chú ý tập hợp chọn lọc các giá trị (đầu vào cảm giác) để tạo ra đầu ra. Trong phần này, chúng tôi sẽ mô tả chi tiết hơn để cung cấp cho bạn cái nhìn cấp cao về cách các cơ chế chú ý hoạt động trong thực tế. Cụ thể, mô hình hồi quy hạt nhân Nadaraya-Watson đề xuất năm 1964 là một ví dụ đơn giản nhưng đầy đủ để chứng minh máy học với các cơ chế chú ý.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
tf.random.set_seed(seed=1322)
```

## [**Tạo ra bộ dữ liệu**]

Để giữ cho mọi thứ đơn giản, chúng ta hãy xem xét vấn đề hồi quy sau: đưa ra một tập dữ liệu của cặp đầu vào-đầu ra $\{(x_1, y_1), \ldots, (x_n, y_n)\}$, làm thế nào để tìm hiểu $f$ để dự đoán đầu ra $\hat{y} = f(x)$ cho bất kỳ đầu vào mới $x$? 

Ở đây chúng ta tạo ra một bộ dữ liệu nhân tạo theo hàm phi tuyến sau với thuật ngữ nhiễu $\epsilon$: 

$$y_i = 2\sin(x_i) + x_i^{0.8} + \epsilon,$$

trong đó $\epsilon$ tuân theo một phân phối bình thường với 0 trung bình và độ lệch chuẩn 0,5. Cả 50 ví dụ đào tạo và 50 ví dụ thử nghiệm được tạo ra. Để hình dung rõ hơn mô hình chú ý sau này, các đầu vào đào tạo được sắp xếp.

```{.python .input}
n_train = 50  # No. of training examples
x_train = np.sort(d2l.rand(n_train) * 5)   # Training inputs
```

```{.python .input}
#@tab pytorch
n_train = 50  # No. of training examples
x_train, _ = torch.sort(d2l.rand(n_train) * 5)   # Training inputs
```

```{.python .input}
#@tab tensorflow
n_train = 50
x_train = tf.sort(tf.random.uniform(shape=(n_train,), maxval=5))
```

```{.python .input}
def f(x):
    return 2 * d2l.sin(x) + x**0.8

y_train = f(x_train) + d2l.normal(0.0, 0.5, (n_train,))  # Training outputs
x_test = d2l.arange(0, 5, 0.1)  # Testing examples
y_truth = f(x_test)  # Ground-truth outputs for the testing examples
n_test = len(x_test)  # No. of testing examples
n_test
```

```{.python .input}
#@tab pytorch
def f(x):
    return 2 * d2l.sin(x) + x**0.8

y_train = f(x_train) + d2l.normal(0.0, 0.5, (n_train,))  # Training outputs
x_test = d2l.arange(0, 5, 0.1)  # Testing examples
y_truth = f(x_test)  # Ground-truth outputs for the testing examples
n_test = len(x_test)  # No. of testing examples
n_test
```

```{.python .input}
#@tab tensorflow
def f(x):
    return 2 * d2l.sin(x) + x**0.8

y_train = f(x_train) + d2l.normal((n_train,), 0.0, 0.5)  # Training outputs
x_test = d2l.arange(0, 5, 0.1)  # Testing examples
y_truth = f(x_test)  # Ground-truth outputs for the testing examples
n_test = len(x_test)  # No. of testing examples
n_test
```

Hàm sau vẽ tất cả các ví dụ đào tạo (được biểu diễn bằng các vòng tròn), hàm tạo dữ liệu đất-chân lý `f` không có thuật ngữ nhiễu (được dán nhãn bởi “Truth”), và hàm dự đoán đã học (được dán nhãn bởi “Pred”).

```{.python .input}
#@tab all
def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5);
```

## Pooling trung bình

Chúng ta bắt đầu với sự ước tính “ngu ngốc nhất” của thế giới cho vấn đề hồi quy này: sử dụng tổng hợp trung bình đến trung bình trên tất cả các đầu ra đào tạo: 

$$f(x) = \frac{1}{n}\sum_{i=1}^n y_i,$$
:eqlabel:`eq_avg-pooling`

được vẽ dưới đây. Như chúng ta có thể thấy, ước tính này thực sự không quá thông minh.

```{.python .input}
y_hat = y_train.mean().repeat(n_test)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab tensorflow
y_hat = tf.repeat(tf.reduce_mean(y_train), repeats=n_test)
plot_kernel_reg(y_hat)
```

## [**Không tham số Chú ý Pooling**]

Rõ ràng, tổng hợp trung bình bỏ qua các đầu vào $x_i$. Một ý tưởng tốt hơn đã được Nadaraya :cite:`Nadaraya.1964` và Watson :cite:`Watson.1964` đề xuất để cân nhắc các đầu ra $y_i$ theo vị trí đầu vào của chúng: 

$$f(x) = \sum_{i=1}^n \frac{K(x - x_i)}{\sum_{j=1}^n K(x - x_j)} y_i,$$
:eqlabel:`eq_nadaraya-watson`

trong đó $K$ là một * kernel*. Các ước tính trong :eqref:`eq_nadaraya-watson` được gọi là * Nadaraya-Watson kernel regression*. Ở đây chúng tôi sẽ không đi sâu vào chi tiết của hạt nhân. Nhớ lại khuôn khổ của các cơ chế chú ý trong :numref:`fig_qkv`. Từ quan điểm của sự chú ý, chúng ta có thể viết lại :eqref:`eq_nadaraya-watson` dưới dạng tổng quát hơn* chú ý cùng*: 

$$f(x) = \sum_{i=1}^n \alpha(x, x_i) y_i,$$
:eqlabel:`eq_attn-pooling`

trong đó $x$ là truy vấn và $(x_i, y_i)$ là cặp giá trị khóa-giá trị. So sánh :eqref:`eq_attn-pooling` và :eqref:`eq_avg-pooling`, sự chú ý tập hợp ở đây là trung bình có trọng số của các giá trị $y_i$. Trọng lượng chú ý*$\alpha(x, x_i)$ trong :eqref:`eq_attn-pooling` được gán cho giá trị tương ứng $y_i$ dựa trên sự tương tác giữa truy vấn $x$ và khóa $x_i$ được mô hình hóa bởi $\alpha$. Đối với bất kỳ truy vấn nào, trọng lượng chú ý của nó đối với tất cả các cặp khóa-giá trị là một phân phối xác suất hợp lệ: chúng không âm và tổng hợp lên đến một. 

Để đạt được trực giác của sự chú ý, chỉ cần xem xét một * Gaussian kernel* được định nghĩa là 

$$
K(u) = \frac{1}{\sqrt{2\pi}} \exp(-\frac{u^2}{2}).
$$

Cắm hạt nhân Gaussian vào :eqref:`eq_attn-pooling` và :eqref:`eq_nadaraya-watson` cho 

$$\begin{aligned} f(x) &=\sum_{i=1}^n \alpha(x, x_i) y_i\\ &= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}(x - x_i)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}(x - x_j)^2\right)} y_i \\&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}(x - x_i)^2\right) y_i. \end{aligned}$$
:eqlabel:`eq_nadaraya-watson-gaussian`

Trong :eqref:`eq_nadaraya-watson-gaussian`, một khóa $x_i$ gần với truy vấn đã cho $x$ sẽ nhận được
*chú ý hơn* thông qua trọng lượng chú ý * lớn hơn* được gán cho giá trị tương ứng của khóa $y_i$.

Đáng chú ý, hồi quy hạt nhân Nadaraya-Watson là một mô hình không tham số; do đó :eqref:`eq_nadaraya-watson-gaussian` là một ví dụ về *không tham số attention pooling*. Sau đây, chúng tôi vẽ dự đoán dựa trên mô hình chú ý không tham số này. Đường dự đoán là trơn tru và gần với sự thật mặt đất hơn so với sự thật được tạo ra bởi tổng hợp trung bình.

```{.python .input}
# Shape of `X_repeat`: (`n_test`, `n_train`), where each row contains the
# same testing inputs (i.e., same queries)
X_repeat = d2l.reshape(x_test.repeat(n_train), (-1, n_train))
# Note that `x_train` contains the keys. Shape of `attention_weights`:
# (`n_test`, `n_train`), where each row contains attention weights to be
# assigned among the values (`y_train`) given each query
attention_weights = npx.softmax(-(X_repeat - x_train)**2 / 2)
# Each element of `y_hat` is weighted average of values, where weights are
# attention weights
y_hat = d2l.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
# Shape of `X_repeat`: (`n_test`, `n_train`), where each row contains the
# same testing inputs (i.e., same queries)
X_repeat = d2l.reshape(x_test.repeat_interleave(n_train), (-1, n_train))
# Note that `x_train` contains the keys. Shape of `attention_weights`:
# (`n_test`, `n_train`), where each row contains attention weights to be
# assigned among the values (`y_train`) given each query
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# Each element of `y_hat` is weighted average of values, where weights are
# attention weights
y_hat = d2l.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab tensorflow
# Shape of `X_repeat`: (`n_test`, `n_train`), where each row contains the 
# same testing inputs (i.e., same queries)
X_repeat = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_train, axis=0)
# Note that `x_train` contains the keys. Shape of `attention_weights`:
# (`n_test`, `n_train`), where each row contains attention weights to be
# assigned among the values (`y_train`) given each query
attention_weights = tf.nn.softmax(-(X_repeat - tf.expand_dims(x_train, axis=1))**2/2, axis=1)
# Each element of `y_hat` is weighted average of values, where weights are attention weights
y_hat = tf.matmul(attention_weights, tf.expand_dims(y_train, axis=1))
plot_kernel_reg(y_hat)
```

Bây giờ chúng ta hãy nhìn vào [** trọng lượng chú ý**]. Ở đây thử nghiệm đầu vào là các truy vấn trong khi đầu vào đào tạo là chìa khóa. Vì cả hai đầu vào được sắp xếp, chúng ta có thể thấy rằng cặp khóa truy vấn càng gần, trọng lượng chú ý cao hơn nằm trong sự chú ý.

```{.python .input}
d2l.show_heatmaps(np.expand_dims(np.expand_dims(attention_weights, 0), 0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab pytorch
d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab tensorflow
d2l.show_heatmaps(tf.expand_dims(tf.expand_dims(attention_weights, axis=0), axis=0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

## ** Tham số chú ý Pooling**

Hồi quy hạt nhân Nadaraya-Watson không tham số được hưởng lợi ích *nhất quán *: cung cấp đủ dữ liệu mô hình này hội tụ với giải pháp tối ưu. Tuy nhiên, chúng ta có thể dễ dàng tích hợp các thông số có thể học được vào tập hợp sự chú ý. 

Ví dụ, hơi khác so với :eqref:`eq_nadaraya-watson-gaussian`, trong khoảng cách sau giữa truy vấn $x$ và khóa $x_i$ được nhân với một tham số có thể học được $w$: 

$$\begin{aligned}f(x) &= \sum_{i=1}^n \alpha(x, x_i) y_i \\&= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}((x - x_i)w)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}((x - x_j)w)^2\right)} y_i \\&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}((x - x_i)w)^2\right) y_i.\end{aligned}$$
:eqlabel:`eq_nadaraya-watson-gaussian-para`

Trong phần còn lại của phần này, chúng tôi sẽ đào tạo mô hình này bằng cách học tham số của sự chú ý trong :eqref:`eq_nadaraya-watson-gaussian-para`. 

### Phép nhân ma trận hàng loạt
:label:`subsec_batch_dot`

Để tính toán sự chú ý hiệu quả hơn cho các minibatches, chúng ta có thể tận dụng các tiện ích nhân ma trận hàng loạt được cung cấp bởi các framework deep learning. 

Giả sử rằng minibatch đầu tiên chứa $n$ ma trận $\mathbf{X}_1, \ldots, \mathbf{X}_n$ của hình dạng $a\times b$, và minibatch thứ hai chứa $n$ ma trận $\mathbf{Y}_1, \ldots, \mathbf{Y}_n$ hình dạng $b\times c$. Phép nhân ma trận lô của chúng dẫn đến ma trận $n$ $\mathbf{X}_1\mathbf{Y}_1, \ldots, \mathbf{X}_n\mathbf{Y}_n$ của hình dạng $a\times c$. Do đó, [** cho hai hàng chục hình dạng ($n$, $a$, $b$) và ($n$, $b$, $c$), hình dạng của sản lượng nhân ma trận lô của chúng là ($n$, $a$, $c$) .**]

```{.python .input}
X = d2l.ones((2, 1, 4))
Y = d2l.ones((2, 4, 6))
npx.batch_dot(X, Y).shape
```

```{.python .input}
#@tab pytorch
X = d2l.ones((2, 1, 4))
Y = d2l.ones((2, 4, 6))
torch.bmm(X, Y).shape
```

```{.python .input}
#@tab tensorflow
X = tf.ones((2, 1, 4))
Y = tf.ones((2, 4, 6))
tf.matmul(X, Y).shape
```

Trong bối cảnh của các cơ chế chú ý, chúng ta có thể [** sử dụng phép nhân ma trận minibatch để tính toán trung bình có trọng số của các giá trị trong một minibatch.**]

```{.python .input}
weights = d2l.ones((2, 10)) * 0.1
values = d2l.reshape(d2l.arange(20), (2, 10))
npx.batch_dot(np.expand_dims(weights, 1), np.expand_dims(values, -1))
```

```{.python .input}
#@tab pytorch
weights = d2l.ones((2, 10)) * 0.1
values = d2l.reshape(d2l.arange(20.0), (2, 10))
torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))
```

```{.python .input}
#@tab tensorflow
weights = tf.ones((2, 10)) * 0.1
values = tf.reshape(tf.range(20.0), shape = (2, 10))
tf.matmul(tf.expand_dims(weights, axis=1), tf.expand_dims(values, axis=-1)).numpy()
```

### Xác định mô hình

Sử dụng phép nhân ma trận minibatch, bên dưới chúng ta xác định phiên bản tham số của hồi quy hạt nhân Nadaraya-Watson dựa trên [** parametric attention pooling**] trong :eqref:`eq_nadaraya-watson-gaussian-para`.

```{.python .input}
class NWKernelRegression(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = self.params.get('w', shape=(1,))

    def forward(self, queries, keys, values):
        # Shape of the output `queries` and `attention_weights`:
        # (no. of queries, no. of key-value pairs)
        queries = d2l.reshape(
            queries.repeat(keys.shape[1]), (-1, keys.shape[1]))
        self.attention_weights = npx.softmax(
            -((queries - keys) * self.w.data())**2 / 2)
        # Shape of `values`: (no. of queries, no. of key-value pairs)
        return npx.batch_dot(np.expand_dims(self.attention_weights, 1),
                             np.expand_dims(values, -1)).reshape(-1)
```

```{.python .input}
#@tab pytorch
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # Shape of the output `queries` and `attention_weights`:
        # (no. of queries, no. of key-value pairs)
        queries = d2l.reshape(
            queries.repeat_interleave(keys.shape[1]), (-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # Shape of `values`: (no. of queries, no. of key-value pairs)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)
```

```{.python .input}
#@tab tensorflow
class NWKernelRegression(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(initial_value=tf.random.uniform(shape=(1,)))
        
    def call(self, queries, keys, values, **kwargs):
        # For training queries are `x_train`. Keys are distance of taining data for each point. Values are `y_train`.
        # Shape of the output `queries` and `attention_weights`: (no. of queries, no. of key-value pairs)
        queries = tf.repeat(tf.expand_dims(queries, axis=1), repeats=keys.shape[1], axis=1)
        self.attention_weights = tf.nn.softmax(-((queries - keys) * self.w)**2 /2, axis =1)
        # Shape of `values`: (no. of queries, no. of key-value pairs)
        return tf.squeeze(tf.matmul(tf.expand_dims(self.attention_weights, axis=1), tf.expand_dims(values, axis=-1)))
```

### Đào tạo

Sau đây, chúng tôi [** chuyển đổi tập dữ liệu đào tạo thành các khóa và giá trị**] để đào tạo mô hình chú ý. Trong tập hợp sự chú ý tham số, bất kỳ đầu vào đào tạo nào lấy các cặp giá trị khóa từ tất cả các ví dụ đào tạo ngoại trừ chính nó để dự đoán đầu ra của nó.

```{.python .input}
# Shape of `X_tile`: (`n_train`, `n_train`), where each column contains the
# same training inputs
X_tile = np.tile(x_train, (n_train, 1))
# Shape of `Y_tile`: (`n_train`, `n_train`), where each column contains the
# same training outputs
Y_tile = np.tile(y_train, (n_train, 1))
# Shape of `keys`: ('n_train', 'n_train' - 1)
keys = d2l.reshape(X_tile[(1 - d2l.eye(n_train)).astype('bool')],
                   (n_train, -1))
# Shape of `values`: ('n_train', 'n_train' - 1)
values = d2l.reshape(Y_tile[(1 - d2l.eye(n_train)).astype('bool')],
                     (n_train, -1))
```

```{.python .input}
#@tab pytorch
# Shape of `X_tile`: (`n_train`, `n_train`), where each column contains the
# same training inputs
X_tile = x_train.repeat((n_train, 1))
# Shape of `Y_tile`: (`n_train`, `n_train`), where each column contains the
# same training outputs
Y_tile = y_train.repeat((n_train, 1))
# Shape of `keys`: ('n_train', 'n_train' - 1)
keys = d2l.reshape(X_tile[(1 - d2l.eye(n_train)).type(torch.bool)],
                   (n_train, -1))
# Shape of `values`: ('n_train', 'n_train' - 1)
values = d2l.reshape(Y_tile[(1 - d2l.eye(n_train)).type(torch.bool)],
                     (n_train, -1))
```

```{.python .input}
#@tab tensorflow
# Shape of `X_tile`: (`n_train`, `n_train`), where each column contains the
# same training inputs
X_tile = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_train, axis=0)
# Shape of `Y_tile`: (`n_train`, `n_train`), where each column contains the
# same training outputs
Y_tile = tf.repeat(tf.expand_dims(y_train, axis=0), repeats=n_train, axis=0)
# Shape of `keys`: ('n_train', 'n_train' - 1)
keys = tf.reshape(X_tile[tf.cast(1 - tf.eye(n_train), dtype=tf.bool)], shape=(n_train, -1))
# Shape of `values`: ('n_train', 'n_train' - 1)
values = tf.reshape(Y_tile[tf.cast(1 - tf.eye(n_train), dtype=tf.bool)], shape=(n_train, -1))
```

Sử dụng sự mất mát bình phương và gốc gradient ngẫu nhiên, chúng tôi [** đào tạo mô hình chú ý tham số**].

```{.python .input}
net = NWKernelRegression()
net.initialize()
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    with autograd.record():
        l = loss(net(x_train, keys, values), y_train)
    l.backward()
    trainer.step(1)
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
```

```{.python .input}
#@tab pytorch
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
```

```{.python .input}
#@tab tensorflow
net = NWKernelRegression()
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])


for epoch in range(5):
    with tf.GradientTape() as t:
        loss = loss_object(y_train, net(x_train, keys, values)) * len(y_train)
    grads = t.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    print(f'epoch {epoch + 1}, loss {float(loss):.6f}')
    animator.add(epoch + 1, float(loss))
```

Sau khi đào tạo mô hình chú ý tham số, chúng ta có thể [** vẽ giá của nó**]. Cố gắng để phù hợp với tập dữ liệu đào tạo với tiếng ồn, dòng dự đoán là ít trơn tru hơn so với đối tác không tham số của nó đã được vẽ trước đó.

```{.python .input}
# Shape of `keys`: (`n_test`, `n_train`), where each column contains the same
# training inputs (i.e., same keys)
keys = np.tile(x_train, (n_test, 1))
# Shape of `value`: (`n_test`, `n_train`)
values = np.tile(y_train, (n_test, 1))
y_hat = net(x_test, keys, values)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
# Shape of `keys`: (`n_test`, `n_train`), where each column contains the same
# training inputs (i.e., same keys)
keys = x_train.repeat((n_test, 1))
# Shape of `value`: (`n_test`, `n_train`)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab tensorflow
# Shape of `keys`: (`n_test`, `n_train`), where each column contains the same
# training inputs (i.e., same keys)
keys = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_test, axis=0)
# Shape of `value`: (`n_test`, `n_train`)
values = tf.repeat(tf.expand_dims(y_train, axis=0), repeats=n_test, axis=0)
y_hat = net(x_test, keys, values)
plot_kernel_reg(y_hat)
```

So sánh với sự chú ý không tham số, [** khu vực có trọng lượng chú ý lớn trở nên sắc bén hơn **] trong cài đặt có thể học được và tham số.

```{.python .input}
d2l.show_heatmaps(np.expand_dims(np.expand_dims(net.attention_weights, 0), 0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab pytorch
d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab tensorflow
d2l.show_heatmaps(tf.expand_dims(tf.expand_dims(net.attention_weights, axis=0), axis=0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

## Tóm tắt

* Hồi quy hạt nhân Nadaraya-Watson là một ví dụ về máy học với các cơ chế chú ý.
* Sự chú ý của hồi quy hạt nhân Nadaraya-Watson là một trung bình trọng số của các đầu ra đào tạo. Từ góc độ chú ý, trọng lượng chú ý được gán cho một giá trị dựa trên một hàm của truy vấn và khóa được ghép nối với giá trị.
* Chú ý pooling có thể là một trong hai không tham số hoặc tham số.

## Bài tập

1. Tăng số lượng các ví dụ đào tạo. Bạn có thể học hồi quy hạt nhân Nadaraya-Watson không tham số tốt hơn?
1. Giá trị của $w$ đã học được của chúng tôi trong thí nghiệm tập hợp chú ý tham số là gì? Tại sao nó làm cho vùng có trọng số sắc nét hơn khi hình dung trọng lượng chú ý?
1. Làm thế nào chúng ta có thể thêm các siêu tham số vào hồi quy hạt nhân Nadaraya-Watson không tham số để dự đoán tốt hơn?
1. Thiết kế một tập hợp chú ý tham số khác cho hồi quy hạt nhân của phần này. Đào tạo mô hình mới này và hình dung trọng lượng chú ý của nó.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1598)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1599)
:end_tab:
