# Thực hiện hồi quy Softmax từ đầu
:label:`sec_softmax_scratch`

(** Cũng giống như chúng tôi thực hiện hồi quy tuyến tính từ đầu, chúng tôi tin rằng that**) hồi quy softmax là tương tự cơ bản và (** bạn nên biết các chi tiết gory của**) (~~ softmax regression~~) và làm thế nào để thực hiện nó cho mình. Chúng tôi sẽ làm việc với bộ dữ liệu Fashion-MNIST, vừa được giới thiệu trong :numref:`sec_fashion_mnist`, thiết lập một bộ lặp dữ liệu với kích thước lô 256.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
from IPython import display
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from IPython import display
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from IPython import display
```

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## Khởi tạo các tham số mô hình

Như trong ví dụ hồi quy tuyến tính của chúng ta, mỗi ví dụ ở đây sẽ được biểu diễn bằng một vectơ có độ dài cố định. Mỗi ví dụ trong tập dữ liệu thô là một hình ảnh $28 \times 28$. Trong phần này, [** chúng ta sẽ làm phẳng từng hình ảnh, coi chúng là vectơ có độ dài 784.**] Trong tương lai, chúng ta sẽ nói về các chiến lược phức tạp hơn để khai thác cấu trúc không gian trong hình ảnh, nhưng bây giờ chúng ta coi từng vị trí pixel như một tính năng khác. 

Nhớ lại rằng trong hồi quy softmax, chúng ta có nhiều đầu ra như có các lớp. (**Vì tập dữ liệu của chúng tôi có 10 lớp, mạng của chúng tôi sẽ có kích thước đầu ra là 10.**) Do đó, trọng lượng của chúng tôi sẽ tạo thành ma trận $784 \times 10$ và các thành kiến sẽ tạo thành một vector hàng $1 \times 10$. Như với hồi quy tuyến tính, chúng tôi sẽ khởi tạo trọng lượng của chúng tôi `W` với tiếng ồn Gaussian và thành kiến của chúng tôi để lấy giá trị ban đầu 0.

```{.python .input}
num_inputs = 784
num_outputs = 10

W = np.random.normal(0, 0.01, (num_inputs, num_outputs))
b = np.zeros(num_outputs)
W.attach_grad()
b.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

```{.python .input}
#@tab tensorflow
num_inputs = 784
num_outputs = 10

W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs),
                                 mean=0, stddev=0.01))
b = tf.Variable(tf.zeros(num_outputs))
```

## Xác định hoạt động Softmax

Trước khi thực hiện mô hình hồi quy softmax, chúng ta hãy xem xét ngắn gọn cách toán tử tổng hoạt động dọc theo các kích thước cụ thể trong một tensor, như đã thảo luận trong :numref:`subseq_lin-alg-reduction` và :numref:`subseq_lin-alg-non-reduction`. [**Cho một ma trận `X` chúng ta có thể tổng hợp tất cả các phần tử (theo mặc định) hoặc chỉ trên các phần tử trong cùng một trục, **] tức là, cùng một cột (trục 0) hoặc cùng một hàng (trục 1). Lưu ý rằng nếu `X` là một tensor với hình dạng (2, 3) và chúng ta tổng hợp trên các cột, kết quả sẽ là một vectơ có hình dạng (3,). Khi gọi toán tử tổng, chúng ta có thể chỉ định để giữ số trục trong tensor ban đầu, thay vì thu gọn kích thước mà chúng ta tóm tắt. Điều này sẽ dẫn đến một tensor hai chiều với hình dạng (1, 3).

```{.python .input}
#@tab pytorch
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdim=True), d2l.reduce_sum(X, 1, keepdim=True)
```

```{.python .input}
#@tab mxnet, tensorflow
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdims=True), d2l.reduce_sum(X, 1, keepdims=True)
```

Bây giờ chúng tôi đã sẵn sàng để (** thực hiện các hoạt động softmax**). Nhớ lại rằng softmax bao gồm ba bước: (i) chúng ta cấp mũ mỗi thuật ngữ (sử dụng `exp`); (ii) chúng ta tổng hợp trên mỗi hàng (chúng ta có một hàng cho mỗi ví dụ trong lô) để lấy hằng số bình thường hóa cho mỗi ví dụ; (iii) chúng ta chia mỗi hàng bằng hằng số bình thường hóa của nó, đảm bảo rằng kết quả tổng thành 1. Trước khi nhìn vào mã, chúng ta hãy nhớ lại cách này trông thể hiện như một phương trình: 

(** $$\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.$$
**)

Mẫu số, hoặc hằng số chuẩn hóa, đôi khi cũng được gọi là hàm phân vùng * (và logarit của nó được gọi là hàm logarit). Nguồn gốc của tên đó nằm trong [vật lý thống kê](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics)) trong đó một phương trình liên quan mô hình sự phân bố trên một nhóm các hạt.

```{.python .input}
#@tab mxnet, tensorflow
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

```{.python .input}
#@tab pytorch
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdim=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

Như bạn có thể thấy, đối với bất kỳ đầu vào ngẫu nhiên nào, [**chúng tôi biến mỗi phần tử thành một số không âm. Hơn nữa, mỗi hàng tổng cộng lên đến 1, **] như là cần thiết cho một xác suất.

```{.python .input}
#@tab mxnet, pytorch
X = d2l.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
#@tab tensorflow
X = tf.random.normal((2, 5), 0, 1)
X_prob = softmax(X)
X_prob, tf.reduce_sum(X_prob, 1)
```

Lưu ý rằng mặc dù điều này có vẻ chính xác về mặt toán học, chúng tôi hơi cẩu thả trong việc thực hiện bởi vì chúng tôi không thực hiện các biện pháp phòng ngừa chống lại tràn số hoặc tràn do các yếu tố lớn hoặc rất nhỏ của ma trận. 

## Xác định mô hình

Bây giờ chúng ta đã xác định hoạt động softmax, chúng ta có thể [** triển khai mô hình hồi quy softmax.**] Đoạn mã dưới đây định nghĩa cách nhập liệu được ánh xạ đến đầu ra thông qua mạng. Lưu ý rằng chúng ta làm phẳng từng ảnh gốc trong lô thành một vectơ bằng hàm `reshape` trước khi truyền dữ liệu qua mô hình của chúng ta.

```{.python .input}
#@tab all
def net(X):
    return softmax(d2l.matmul(d2l.reshape(X, (-1, W.shape[0])), W) + b)
```

## Xác định chức năng mất

Tiếp theo, chúng ta cần thực hiện hàm mất chéo entropy, như được giới thiệu trong :numref:`sec_softmax`. Đây có thể là chức năng mất mát phổ biến nhất trong tất cả các học sâu bởi vì, tại thời điểm này, các vấn đề phân loại vượt xa vấn đề hồi quy. 

Nhớ lại rằng cross-entropy lấy khả năng log âm của xác suất dự đoán được gán cho nhãn thật. Thay vì lặp qua các dự đoán bằng Python for-loop (có xu hướng không hiệu quả), chúng ta có thể chọn tất cả các phần tử bằng một toán tử duy nhất. Dưới đây, chúng ta [** tạo dữ liệu mẫu `y_hat` với 2 ví dụ về xác suất dự đoán trên 3 lớp và nhãn tương ứng của chúng `y`.**] Với `y` chúng ta biết rằng trong ví dụ đầu tiên, lớp đầu tiên là dự đoán chính xác và trong ví dụ thứ hai, lớp thứ ba là sự thật. [**Sử dụng `y` làm chỉ số xác suất trong `y_hat`, **] chúng ta chọn xác suất của lớp đầu tiên trong ví dụ đầu tiên và xác suất của lớp thứ ba trong ví dụ thứ hai.

```{.python .input}
#@tab mxnet, pytorch
y = d2l.tensor([0, 2])
y_hat = d2l.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

```{.python .input}
#@tab tensorflow
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
```

Bây giờ chúng ta có thể (** triển khai hàm mất chéo entropy**) một cách hiệu quả chỉ với một dòng mã.

```{.python .input}
#@tab mxnet, pytorch
def cross_entropy(y_hat, y):
    return - d2l.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)
```

```{.python .input}
#@tab tensorflow
def cross_entropy(y_hat, y):
    return -tf.math.log(tf.boolean_mask(
        y_hat, tf.one_hot(y, depth=y_hat.shape[-1])))

cross_entropy(y_hat, y)
```

## Phân loại chính xác

Với phân phối xác suất dự đoán `y_hat`, chúng ta thường chọn lớp có xác suất dự đoán cao nhất bất cứ khi nào chúng ta phải đưa ra một dự đoán cứng. Thật vậy, nhiều ứng dụng yêu cầu chúng tôi đưa ra lựa chọn. Gmail phải phân loại email thành “Chính”, “Xã hội”, “Cập nhật” hoặc “Diễn đàn”. Nó có thể ước tính xác suất trong nội bộ, nhưng vào cuối ngày, nó phải chọn một trong số các lớp học. 

Khi dự đoán phù hợp với lớp nhãn `y`, chúng là chính xác. Độ chính xác phân loại là phần nhỏ của tất cả các dự đoán là chính xác. Mặc dù có thể khó tối ưu hóa độ chính xác trực tiếp (nó không phân biệt được), nhưng nó thường là biện pháp hiệu suất mà chúng tôi quan tâm nhất và gần như chúng tôi sẽ luôn báo cáo khi đào tạo phân loại. 

Để tính toán độ chính xác, chúng tôi làm như sau. Đầu tiên, nếu `y_hat` là một ma trận, chúng ta giả định rằng chiều thứ hai lưu trữ điểm dự đoán cho mỗi lớp. Chúng tôi sử dụng `argmax` để có được lớp dự đoán bằng chỉ số cho mục nhập lớn nhất trong mỗi hàng. Sau đó, chúng ta [** so sánh lớp dự đoán với `y` elementwise.**] Vì toán tử bình đẳng `==` nhạy cảm với các kiểu dữ liệu, chúng tôi chuyển đổi kiểu dữ liệu của `y_hat` để khớp với `y`. Kết quả là một tensor chứa các mục của 0 (false) và 1 (true). Lấy tổng sản lượng số dự đoán chính xác.

```{.python .input}
#@tab all
def accuracy(y_hat, y):  #@save
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))
```

Chúng tôi sẽ tiếp tục sử dụng các biến `y_hat` và `y` được xác định trước đó như là phân phối xác suất dự đoán và nhãn, tương ứng. Chúng ta có thể thấy lớp dự đoán của ví dụ đầu tiên là 2 (phần tử lớn nhất của hàng là 0,6 với chỉ số 2), không phù hợp với nhãn thực tế, 0. Lớp dự đoán của ví dụ thứ hai là 2 (phần tử lớn nhất của hàng là 0,5 với chỉ số 2), phù hợp với nhãn thực tế, 2. Do đó, tỷ lệ chính xác phân loại cho hai ví dụ này là 0,5.

```{.python .input}
#@tab all
accuracy(y_hat, y) / len(y)
```

[**Tương tự, chúng ta có thể đánh giá độ chính xác cho bất kỳ model `net` nào trên một bộ dữ liệu**] được truy cập thông qua bộ lặp dữ liệu `data_iter`.

```{.python .input}
#@tab mxnet, tensorflow
def evaluate_accuracy(net, data_iter):  #@save
    """Compute the accuracy for a model on a dataset."""
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

```{.python .input}
#@tab pytorch
def evaluate_accuracy(net, data_iter):  #@save
    """Compute the accuracy for a model on a dataset."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

Ở đây `Accumulator` là một lớp tiện ích để tích lũy tổng trên nhiều biến. Trong hàm `evaluate_accuracy` trên, chúng ta tạo ra 2 biến trong phiên bản `Accumulator` để lưu trữ cả số dự đoán đúng và số dự đoán, tương ứng. Cả hai sẽ được tích lũy theo thời gian khi chúng ta lặp lại tập dữ liệu.

```{.python .input}
#@tab all
class Accumulator:  #@save
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

[**Bởi vì chúng tôi đã khởi tạo mô hình `net` với trọng lượng ngẫu nhiên, độ chính xác của mô hình này phải gần với đoán ngẫu nhiên, **] tức là 0,1 cho 10 lớp.

```{.python .input}
#@tab all
evaluate_accuracy(net, test_iter)
```

## Đào tạo

[** Vòng đào tạo**] cho hồi quy softmax sẽ trông nổi bật quen thuộc nếu bạn đọc thông qua việc thực hiện hồi quy tuyến tính của chúng tôi trong :numref:`sec_linear_scratch`. Ở đây chúng tôi tái cấu trúc việc thực hiện để làm cho nó có thể tái sử dụng. Đầu tiên, chúng ta định nghĩa một hàm để đào tạo cho một kỷ nguyên. Lưu ý rằng `updater` là một hàm chung để cập nhật các tham số mô hình, chấp nhận kích thước lô làm đối số. Nó có thể là một trình bao bọc của hàm `d2l.sgd` hoặc chức năng tối ưu hóa tích hợp của khung.

```{.python .input}
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """Train a model within one epoch (defined in Chapter 3)."""
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    if isinstance(updater, gluon.Trainer):
        updater = updater.step
    for X, y in train_iter:
        # Compute gradients and update parameters
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.size)
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

```{.python .input}
#@tab pytorch
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """The training loop defined in Chapter 3."""
    # Set the model to training mode
    if isinstance(net, torch.nn.Module):
        net.train()
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # Using PyTorch in-built optimizer & loss criterion
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # Using custom built optimizer & loss criterion
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

```{.python .input}
#@tab tensorflow
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """The training loop defined in Chapter 3."""
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        with tf.GradientTape() as tape:
            y_hat = net(X)
            # Keras implementations for loss takes (labels, predictions)
            # instead of (predictions, labels) that users might implement
            # in this book, e.g. `cross_entropy` that we implemented above
            if isinstance(loss, tf.keras.losses.Loss):
                l = loss(y, y_hat)
            else:
                l = loss(y_hat, y)
        if isinstance(updater, tf.keras.optimizers.Optimizer):
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            updater.apply_gradients(zip(grads, params))
        else:
            updater(X.shape[0], tape.gradient(l, updater.params))
        # Keras loss by default returns the average loss in a batch
        l_sum = l * float(tf.size(y)) if isinstance(
            loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l_sum, accuracy(y_hat, y), tf.size(y))
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

Trước khi hiển thị việc thực hiện hàm đào tạo, chúng ta định nghĩa [** một lớp tiện ích vẽ dữ liệu trong hoạt hình.**] Một lần nữa, nó nhằm mục đích đơn giản hóa mã trong phần còn lại của cuốn sách.

```{.python .input}
#@tab all
class Animator:  #@save
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
```

[~~Chức năng đào tạo ~~] Chức năng đào tạo sau đây sau đó đào tạo một mô hình `net` trên một tập dữ liệu đào tạo được truy cập qua `train_iter` cho nhiều kỷ nguyên, được chỉ định bởi `num_epochs`. Vào cuối mỗi kỷ nguyên, mô hình được đánh giá trên một tập dữ liệu thử nghiệm truy cập qua `test_iter`. Chúng tôi sẽ tận dụng lớp `Animator` để hình dung tiến độ đào tạo.

```{.python .input}
#@tab all
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """Train a model (defined in Chapter 3)."""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
```

Là một triển khai từ đầu, chúng tôi [** sử dụng minibatch stochastic gradient descent**] được định nghĩa trong :numref:`sec_linear_scratch` để tối ưu hóa chức năng mất mát của mô hình với tốc độ học tập 0.1.

```{.python .input}
#@tab mxnet, pytorch
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
```

```{.python .input}
#@tab tensorflow
class Updater():  #@save
    """For updating parameters using minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def __call__(self, batch_size, grads):
        d2l.sgd(self.params, grads, self.lr, batch_size)

updater = Updater([W, b], lr=0.1)
```

Bây giờ chúng ta [** đào tạo mô hình với 10 kỷ nguyên **] Lưu ý rằng cả số epochs (`num_epochs`) và tốc độ học tập (`lr`) đều có thể điều chỉnh các siêu tham số. Bằng cách thay đổi giá trị của chúng, chúng ta có thể tăng độ chính xác phân loại của mô hình.

```{.python .input}
#@tab all
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```

## Prediction

Bây giờ việc đào tạo đã hoàn tất, mô hình của chúng tôi đã sẵn sàng để [** phân loại một số hình ảnh**] Với một loạt các hình ảnh, chúng tôi sẽ so sánh các nhãn thực tế của chúng (dòng đầu ra văn bản đầu tiên) và dự đoán từ mô hình (dòng thứ hai của đầu ra văn bản).

```{.python .input}
#@tab all
def predict_ch3(net, test_iter, n=6):  #@save
    """Predict labels (defined in Chapter 3)."""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(d2l.argmax(net(X), axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        d2l.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
```

## Tóm tắt

* Với hồi quy softmax, chúng ta có thể đào tạo các mô hình để phân loại đa lớp.
* Vòng đào tạo của hồi quy softmax rất giống với trong hồi quy tuyến tính: lấy và đọc dữ liệu, xác định mô hình và chức năng mất mát, sau đó đào tạo các mô hình sử dụng các thuật toán tối ưu hóa. Như bạn sẽ sớm tìm ra, hầu hết các mô hình học sâu phổ biến đều có các quy trình đào tạo tương tự.

## Bài tập

1. Trong phần này, chúng tôi trực tiếp triển khai hàm softmax dựa trên định nghĩa toán học của phép toán softmax. Những vấn đề này có thể gây ra? Gợi ý: cố gắng tính toán kích thước của $\exp(50)$.
1. Hàm `cross_entropy` trong phần này được thực hiện theo định nghĩa của hàm mất chéo entropy. Điều gì có thể là vấn đề với việc thực hiện này? Gợi ý: xem xét tên miền của logarit.
1. Những giải pháp bạn có thể nghĩ đến để khắc phục hai vấn đề ở trên?
1. Có phải luôn luôn là một ý tưởng tốt để trả lại nhãn có khả năng nhất? Ví dụ, bạn sẽ làm điều này để chẩn đoán y tế?
1. Giả sử rằng chúng ta muốn sử dụng hồi quy softmax để dự đoán từ tiếp theo dựa trên một số tính năng. Một số vấn đề có thể phát sinh từ một từ vựng lớn là gì?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/50)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/51)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/225)
:end_tab:
