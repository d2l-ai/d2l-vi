# Mạng thần kinh phức tạp (LeNet)
:label:`sec_lenet`

Bây giờ chúng tôi có tất cả các thành phần cần thiết để lắp ráp một CNN đầy đủ chức năng. Trong cuộc gặp gỡ trước đó của chúng tôi với dữ liệu hình ảnh, chúng tôi đã áp dụng mô hình hồi quy softmax (:numref:`sec_softmax_scratch`) và mô hình MLP (:numref:`sec_mlp_scratch`) cho hình ảnh quần áo trong bộ dữ liệu Fashion-MNIST. Để làm cho dữ liệu như vậy tuân theo hồi quy softmax và MLP s, trước tiên chúng ta làm phẳng mỗi hình ảnh từ một ma trận $28\times28$ thành một vector $784$ chiều dài cố định, và sau đó xử lý chúng với các lớp được kết nối hoàn toàn. Bây giờ chúng ta có một tay cầm trên các lớp phức tạp, chúng ta có thể giữ lại cấu trúc không gian trong hình ảnh của chúng ta. Là một lợi ích bổ sung của việc thay thế các lớp được kết nối hoàn toàn bằng các lớp kết nối, chúng ta sẽ tận hưởng các mô hình phân tích hơn đòi hỏi ít tham số hơn nhiều. 

Trong phần này, chúng tôi sẽ giới thiệu *LeNet*, trong số các CNN được xuất bản đầu tiên để thu hút sự chú ý rộng rãi về hiệu suất của nó trên các tác vụ thị giác máy tính. Mô hình được giới thiệu bởi (và đặt tên cho) Yann LeCun, sau đó là một nhà nghiên cứu tại AT&T Bell Labs, với mục đích nhận dạng chữ số viết tay trong hình ảnh :cite:`LeCun.Bottou.Bengio.ea.1998`. Công trình này đại diện cho đỉnh cao của một thập kỷ nghiên cứu phát triển công nghệ. Năm 1989, LeCun công bố nghiên cứu đầu tiên để đào tạo thành công CNN thông qua truyền ngược. 

Vào thời điểm đó LeNet đạt được kết quả xuất sắc phù hợp với hiệu suất của các máy vector hỗ trợ, sau đó là một cách tiếp cận thống trị trong việc học có giám sát. LeNet cuối cùng đã được điều chỉnh để nhận ra các chữ số để xử lý tiền gửi trong máy ATM. Cho đến ngày nay, một số máy ATM vẫn chạy mã mà Yann và đồng nghiệp Leon Bottou đã viết vào những năm 1990! 

## LeNet

Ở cấp độ cao, (**LeNet (LeNet-5) bao gồm hai phần: (i) một bộ mã hóa phức tạp bao gồm hai lớp phức tạp; và (ii) một khối dày đặc bao gồm ba lớp kết nối đầy đủ**); Kiến trúc được tóm tắt trong :numref:`img_lenet`. 

![Data flow in LeNet. The input is a handwritten digit, the output a probability over 10 possible outcomes.](../img/lenet.svg)
:label:`img_lenet`

Các đơn vị cơ bản trong mỗi khối phức tạp là một lớp phức tạp, một chức năng kích hoạt sigmoid, và một hoạt động tổng hợp trung bình tiếp theo. Lưu ý rằng trong khi Relus và max-pooling hoạt động tốt hơn, những khám phá này vẫn chưa được thực hiện trong những năm 1990. Mỗi lớp phức hợp sử dụng một hạt nhân $5\times 5$ và một hàm kích hoạt sigmoid. Các lớp này ánh xạ các đầu vào sắp xếp không gian cho một số bản đồ tính năng hai chiều, thường làm tăng số lượng kênh. Lớp phức tạp đầu tiên có 6 kênh đầu ra, trong khi thứ hai có 16 kênh. Mỗi hoạt động tổng hợp $2\times2$ (sải chân 2) làm giảm kích thước bằng hệ số $4$ thông qua lấy mẫu xuống không gian. Khối phức tạp phát ra một đầu ra với hình dạng được đưa ra bởi (kích thước lô, số kênh, chiều cao, chiều rộng). 

Để truyền đầu ra từ khối phức tạp đến khối dày đặc, chúng ta phải làm phẳng từng ví dụ trong minibatch. Nói cách khác, chúng ta lấy đầu vào bốn chiều này và biến nó thành đầu vào hai chiều được mong đợi bởi các lớp được kết nối hoàn toàn: như một lời nhắc nhở, biểu diễn hai chiều mà chúng ta mong muốn sử dụng kích thước đầu tiên để lập chỉ mục các ví dụ trong minibatch và thứ hai để đưa ra biểu diễn vector phẳng of each mỗi example thí dụ. Khối dày đặc của LeNet có ba lớp kết nối hoàn toàn, tương ứng với 120, 84 và 10 đầu ra. Bởi vì chúng ta vẫn đang thực hiện phân loại, lớp đầu ra 10 chiều tương ứng với số lượng lớp đầu ra có thể. 

Trong khi đi đến mức bạn thực sự hiểu những gì đang xảy ra bên trong LeNet có thể đã thực hiện một chút công việc, hy vọng đoạn mã sau đây sẽ thuyết phục bạn rằng việc thực hiện các mô hình như vậy với các khuôn khổ học sâu hiện đại rất đơn giản. Chúng ta chỉ cần khởi tạo một khối `Sequential` và chuỗi với nhau các lớp thích hợp.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        # `Dense` will transform an input of the shape (batch size, number of
        # channels, height, width) into an input of the shape (batch size,
        # number of channels * height * width) automatically by default
        nn.Dense(120, activation='sigmoid'),
        nn.Dense(84, activation='sigmoid'),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='sigmoid'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])
```

Chúng tôi đã có một sự tự do nhỏ với mô hình ban đầu, loại bỏ kích hoạt Gaussian trong lớp cuối cùng. Ngoài ra, mạng này phù hợp với kiến trúc LeNet-5 ban đầu. 

Bằng cách truyền một hình ảnh $28 \times 28$ một kênh (đen và trắng) qua mạng và in hình dạng đầu ra ở mỗi lớp, chúng ta có thể [** kiểm tra mô hình**] để đảm bảo rằng các hoạt động của nó phù hợp với những gì chúng ta mong đợi từ :numref:`img_lenet_vert`. 

![Compressed notation for LeNet-5.](../img/lenet-vert.svg)
:label:`img_lenet_vert`

```{.python .input}
X = np.random.uniform(size=(1, 1, 28, 28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 28, 28, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
```

Lưu ý rằng chiều cao và chiều rộng của biểu diễn tại mỗi lớp trong suốt khối phức tạp bị giảm (so với lớp trước đó). Lớp kết hợp đầu tiên sử dụng 2 pixel đệm để bù đắp cho việc giảm chiều cao và chiều rộng mà nếu không sẽ là kết quả của việc sử dụng một hạt nhân $5 \times 5$. Ngược lại, lớp phức tạp thứ hai bỏ đệm, và do đó chiều cao và chiều rộng đều giảm 4 pixel. Khi chúng ta đi lên ngăn xếp các lớp, số lượng kênh tăng lớp trên lớp từ 1 trong đầu vào lên 6 sau lớp phức tạp đầu tiên và 16 sau lớp phức tạp thứ hai. Tuy nhiên, mỗi lớp tổng hợp giảm một nửa chiều cao và chiều rộng. Cuối cùng, mỗi lớp được kết nối hoàn toàn làm giảm chiều, cuối cùng phát ra một đầu ra có kích thước phù hợp với số lượng lớp. 

## Đào tạo

Bây giờ chúng tôi đã triển khai mô hình, chúng ta hãy [** chạy một thử nghiệm để xem LeNet giá vé trên Fashion-MNIST**] như thế nào.

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
```

Trong khi CNN có ít tham số hơn, chúng vẫn có thể tốn kém hơn để tính toán so với MLP sâu tương tự vì mỗi tham số tham gia vào nhiều phép nhân hơn. Nếu bạn có quyền truy cập vào GPU, đây có thể là thời điểm tốt để đưa nó vào hành động để tăng tốc độ đào tạo.

:begin_tab:`mxnet, pytorch`
Để đánh giá, chúng ta cần [** thực hiện một sửa đổi nhỏ đối với hàm `evaluate_accuracy`**] mà chúng tôi đã mô tả trong :numref:`sec_softmax_scratch`. Vì tập dữ liệu đầy đủ nằm trong bộ nhớ chính, chúng ta cần sao chép nó vào bộ nhớ GPU trước khi mô hình sử dụng GPU để tính toán với bộ dữ liệu.
:end_tab:

```{.python .input}
def evaluate_accuracy_gpu(net, data_iter, device=None):  #@save
    """Compute the accuracy for a model on a dataset using a GPU."""
    if not device:  # Query the first device where the first parameter is on
        device = list(net.collect_params().values())[0].list_ctx()[0]
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        X, y = X.as_in_ctx(device), y.as_in_ctx(device)
        metric.add(d2l.accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

```{.python .input}
#@tab pytorch
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """Compute the accuracy for a model on a dataset using a GPU."""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

Chúng tôi cũng cần [** cập nhật chức năng đào tạo của chúng tôi để đối phó với GPU. **] Không giống như `train_epoch_ch3` được xác định trong :numref:`sec_softmax_scratch`, bây giờ chúng ta cần di chuyển từng minibatch dữ liệu đến thiết bị được chỉ định của chúng tôi (hy vọng là GPU) trước khi thực hiện tuyên truyền chuyển tiếp và lùi. 

Chức năng huấn luyện `train_ch6` cũng tương tự như `train_ch3` được định nghĩa trong :numref:`sec_softmax_scratch`. Vì chúng tôi sẽ triển khai các mạng với nhiều lớp trong tương lai, chúng tôi sẽ chủ yếu dựa vào các API cấp cao. Hàm đào tạo sau đây giả định một mô hình được tạo từ API cấp cao làm đầu vào và được tối ưu hóa cho phù hợp. Chúng tôi khởi tạo các tham số mô hình trên thiết bị được chỉ định bởi đối số `device`, sử dụng khởi tạo Xavier như được giới thiệu trong :numref:`subsec_xavier`. Cũng giống như với MLP, chức năng mất mát của chúng tôi là cross-entropy, và chúng tôi giảm thiểu nó thông qua minibatch stochastic gradient descent. Vì mỗi kỷ nguyên mất hàng chục giây để chạy, chúng tôi hình dung sự mất mát đào tạo thường xuyên hơn.

```{.python .input}
#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6)."""
    net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(),
                            'sgd', {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            # Here is the major difference from `d2l.train_epoch_ch3`
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6)."""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```

```{.python .input}
#@tab tensorflow
class TrainCallback(tf.keras.callbacks.Callback):  #@save
    """A callback to visiualize the training progress."""
    def __init__(self, net, train_iter, test_iter, num_epochs, device_name):
        self.timer = d2l.Timer()
        self.animator = d2l.Animator(
            xlabel='epoch', xlim=[1, num_epochs], legend=[
                'train loss', 'train acc', 'test acc'])
        self.net = net
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.num_epochs = num_epochs
        self.device_name = device_name
    def on_epoch_begin(self, epoch, logs=None):
        self.timer.start()
    def on_epoch_end(self, epoch, logs):
        self.timer.stop()
        test_acc = self.net.evaluate(
            self.test_iter, verbose=0, return_dict=True)['accuracy']
        metrics = (logs['loss'], logs['accuracy'], test_acc)
        self.animator.add(epoch + 1, metrics)
        if epoch == self.num_epochs - 1:
            batch_size = next(iter(self.train_iter))[0].shape[0]
            num_examples = batch_size * tf.data.experimental.cardinality(
                self.train_iter).numpy()
            print(f'loss {metrics[0]:.3f}, train acc {metrics[1]:.3f}, '
                  f'test acc {metrics[2]:.3f}')
            print(f'{num_examples / self.timer.avg():.1f} examples/sec on '
                  f'{str(self.device_name)}')

#@save
def train_ch6(net_fn, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6)."""
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    net.fit(train_iter, epochs=num_epochs, verbose=0, callbacks=[callback])
    return net
```

[**Bây giờ chúng ta hãy đào tạo và đánh giá mô hình LeNet-5. **]

```{.python .input}
#@tab all
lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## Tóm tắt

* CNN là một mạng sử dụng các lớp phức tạp.
* Trong một CNN, chúng tôi xen kẽ các phức tạp, phi tuyến tính, và (thường) các hoạt động tập hợp.
* Trong một CNN, các lớp phức hợp thường được sắp xếp sao cho chúng giảm dần độ phân giải không gian của các biểu diễn, đồng thời tăng số lượng kênh.
* Trong CNN truyền thống, các biểu diễn được mã hóa bởi các khối phức tạp được xử lý bởi một hoặc nhiều lớp kết nối đầy đủ trước khi phát ra đầu ra.
* LeNet được cho là triển khai thành công đầu tiên của một mạng như vậy.

## Bài tập

1. Thay thế các pooling trung bình với tổng hợp tối đa. Điều gì xảy ra?
1. Cố gắng xây dựng một mạng phức tạp hơn dựa trên LeNet để cải thiện độ chính xác của nó.
    1. Điều chỉnh kích thước cửa sổ covolution.
    1. Điều chỉnh số lượng kênh đầu ra.
    1. Điều chỉnh chức năng kích hoạt (ví dụ: ReLU).
    1. Điều chỉnh số lượng lớp covolution.
    1. Điều chỉnh số lượng lớp được kết nối hoàn toàn.
    1. Điều chỉnh tỷ lệ học tập và các chi tiết đào tạo khác (ví dụ: khởi tạo và số kỷ nguyên.)
1. Hãy thử mạng cải tiến trên tập dữ liệu MNIST ban đầu.
1. Hiển thị các kích hoạt của lớp đầu tiên và thứ hai của LeNet cho các đầu vào khác nhau (ví dụ, áo len và áo khoác).

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/73)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/74)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/275)
:end_tab:
