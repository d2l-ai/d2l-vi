# Lập kế hoạch tỷ lệ học tập
:label:`sec_scheduler`

Cho đến nay chúng tôi chủ yếu tập trung vào tối ưu hóa các thuật toán * để làm thế nào để cập nhật các vectơ trọng lượng thay vì trên * tỷ lệ* tại đó chúng đang được cập nhật. Tuy nhiên, việc điều chỉnh tốc độ học tập thường cũng quan trọng như thuật toán thực tế. Có một số khía cạnh cần xem xét: 

* Rõ ràng nhất là *độ lượng* của tỷ lệ học tập quan trọng. Nếu nó quá lớn, tối ưu hóa phân kỳ, nếu nó quá nhỏ, phải mất quá nhiều thời gian để đào tạo hoặc chúng ta kết thúc với một kết quả tối ưu. Trước đây chúng tôi đã thấy rằng số điều kiện của vấn đề quan trọng (xem ví dụ, :numref:`sec_momentum` để biết chi tiết). Trực giác đó là tỷ lệ giữa số lượng thay đổi theo hướng ít nhạy cảm nhất so với một nhạy cảm nhất.
* Thứ hai, tỷ lệ phân rã cũng quan trọng như vậy. Nếu tỷ lệ học tập vẫn lớn, chúng tôi có thể chỉ đơn giản là kết thúc nảy xung quanh mức tối thiểu và do đó không đạt được sự tối ưu. :numref:`sec_minibatch_sgd` đã thảo luận chi tiết về điều này và chúng tôi đã phân tích đảm bảo hiệu suất trong :numref:`sec_sgd`. Nói tóm lại, chúng tôi muốn tỷ lệ phân rã, nhưng có lẽ chậm hơn $\mathcal{O}(t^{-\frac{1}{2}})$, đây sẽ là một lựa chọn tốt cho các vấn đề lồi.
* Một khía cạnh khác quan trọng không kém là * khởi hóa*. Điều này liên quan đến cả cách các tham số được đặt ban đầu (xem lại :numref:`sec_numerical_stability` để biết chi tiết) và cũng như cách chúng phát triển ban đầu. Điều này đi theo moniker của * warmup*, tức là, nhanh chóng như thế nào chúng ta bắt đầu di chuyển về phía giải pháp ban đầu. Các bước lớn ngay từ đầu có thể không có lợi, đặc biệt là vì tập hợp các tham số ban đầu là ngẫu nhiên. Các hướng cập nhật ban đầu cũng có thể khá vô nghĩa.
* Cuối cùng, có một số biến thể tối ưu hóa thực hiện điều chỉnh tỷ lệ học tập theo chu kỳ. Điều này nằm ngoài phạm vi của chương hiện tại. Chúng tôi khuyên người đọc xem lại chi tiết trong :cite:`Izmailov.Podoprikhin.Garipov.ea.2018`, ví dụ, làm thế nào để có được các giải pháp tốt hơn bằng cách trung bình trên toàn bộ một * path* của các tham số.

Với thực tế là có rất nhiều chi tiết cần thiết để quản lý tỷ lệ học tập, hầu hết các khuôn khổ học sâu đều có các công cụ để xử lý điều này một cách tự động. Trong chương hiện tại, chúng tôi sẽ xem xét các hiệu ứng mà các lịch trình khác nhau có về độ chính xác và cũng cho thấy cách điều này có thể được quản lý hiệu quả thông qua một *học tỷ lệ lịch học*. 

## Vấn đề Toy

Chúng tôi bắt đầu với một vấn đề đồ chơi đủ rẻ để tính toán dễ dàng, nhưng đủ không tầm thường để minh họa một số khía cạnh chính. Đối với điều đó, chúng tôi chọn một phiên bản hơi hiện đại hóa của LeNet (`relu` thay vì kích hoạt `sigmoid`, MaxPooling thay vì AveragePooling), như áp dụng cho Fashion-MNIST. Hơn nữa, chúng tôi lai mạng để thực hiện. Vì hầu hết các mã là tiêu chuẩn, chúng tôi chỉ giới thiệu những điều cơ bản mà không cần thảo luận chi tiết thêm. Xin xem :numref:`chap_cnn` để được làm mới khi cần thiết.

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
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the 
# lenet section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device):
    net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.optim import lr_scheduler

def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))

    return model

loss = nn.CrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the 
# lenet section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
          scheduler=None):
    net.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            net.train()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))
    
        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # Using PyTorch In-Built scheduler
                scheduler.step()
            else:
                # Using custom defined scheduler
                for param_group in trainer.param_groups:
                    param_group['lr'] = scheduler(epoch)

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import math
from tensorflow.keras.callbacks import LearningRateScheduler

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='relu'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the 
# lenet section of chapter convolutional neural networks
def train(net_fn, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu(), custom_callback = False):
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = d2l.TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    if custom_callback is False:
        net.fit(train_iter, epochs=num_epochs, verbose=0, 
                callbacks=[callback])
    else:
         net.fit(train_iter, epochs=num_epochs, verbose=0,
                 callbacks=[callback, custom_callback])
    return net
```

Chúng ta hãy xem những gì sẽ xảy ra nếu chúng ta gọi thuật toán này với các cài đặt mặc định, chẳng hạn như tốc độ học tập $0.3$ và đào tạo cho $30$ lặp lại. Lưu ý độ chính xác của đào tạo tiếp tục tăng lên trong khi tiến bộ về các quầy hàng chính xác thử nghiệm vượt ra ngoài một điểm. Khoảng cách giữa cả hai đường cong cho thấy quá mức.

```{.python .input}
lr, num_epochs = 0.3, 30
net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.3, 30
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab tensorflow
lr, num_epochs = 0.3, 30
train(net, train_iter, test_iter, num_epochs, lr)
```

## Người lập lịch

Một cách để điều chỉnh tốc độ học tập là thiết lập nó một cách rõ ràng ở mỗi bước. Điều này đạt được thuận tiện bằng phương pháp `set_learning_rate`. Chúng ta có thể điều chỉnh nó xuống sau mỗi kỷ nguyên (hoặc thậm chí sau mỗi minibatch), ví dụ, một cách năng động để đáp ứng cách tối ưu hóa đang tiến triển.

```{.python .input}
trainer.set_learning_rate(0.1)
print(f'learning rate is now {trainer.learning_rate:.2f}')
```

```{.python .input}
#@tab pytorch
lr = 0.1
trainer.param_groups[0]["lr"] = lr
print(f'learning rate is now {trainer.param_groups[0]["lr"]:.2f}')
```

```{.python .input}
#@tab tensorflow
lr = 0.1
dummy_model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
dummy_model.compile(tf.keras.optimizers.SGD(learning_rate=lr), loss='mse')
print(f'learning rate is now ,', dummy_model.optimizer.lr.numpy())
```

Nói chung hơn chúng tôi muốn xác định một trình lập lịch. Khi được gọi với số lượng cập nhật nó trả về giá trị thích hợp của tốc độ học tập. Hãy để chúng tôi xác định một cái đơn giản đặt tỷ lệ học tập thành $\eta = \eta_0 (t + 1)^{-\frac{1}{2}}$.

```{.python .input}
#@tab all
class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)
```

Hãy để chúng tôi vẽ hành vi của nó trên một loạt các giá trị.

```{.python .input}
#@tab all
scheduler = SquareRootScheduler(lr=0.1)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Bây giờ chúng ta hãy xem cách điều này diễn ra để đào tạo về thời trang-MNIST. Chúng tôi chỉ cần cung cấp trình lập lịch như một đối số bổ sung cho thuật toán đào tạo.

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

Điều này làm việc khá tốt hơn một chút so với trước đây. Hai điều nổi bật: đường cong khá trơn tru hơn trước đây. Thứ hai, có ít quá nhiều hơn. Thật không may, nó không phải là một câu hỏi được giải quyết tốt về lý do tại sao một số chiến lược nhất định dẫn đến ít quá mức trong * lý đề*. Có một số lập luận rằng kích thước bước nhỏ hơn sẽ dẫn đến các tham số gần bằng 0 và do đó đơn giản hơn. Tuy nhiên, điều này không giải thích hoàn toàn hiện tượng vì chúng ta không thực sự dừng lại sớm mà chỉ đơn giản là giảm tốc độ học tập nhẹ nhàng. 

## Chính sách

Mặc dù chúng tôi không thể bao gồm toàn bộ các lập lịch trình tỷ lệ học tập, chúng tôi cố gắng đưa ra một cái nhìn tổng quan ngắn gọn về các chính sách phổ biến dưới đây. Các lựa chọn phổ biến là phân rã đa thức và từng mảnh lịch trình liên tục. Ngoài ra, lịch trình học cosine đã được tìm thấy để làm việc tốt theo kinh nghiệm về một số vấn đề. Cuối cùng, về một số vấn đề, có lợi khi làm nóng trình tối ưu hóa trước khi sử dụng tỷ lệ học tập lớn. 

### Bộ lập lịch yếu tố

Một thay thế cho phân rã đa thức sẽ là một phân rã nhân, đó là $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$ cho $\alpha \in (0, 1)$. Để ngăn chặn tỷ lệ học tập phân rã vượt quá giới hạn thấp hơn hợp lý, phương trình cập nhật thường được sửa đổi thành $\eta_{t+1} \leftarrow \mathop{\mathrm{max}}(\eta_{\mathrm{min}}, \eta_t \cdot \alpha)$.

```{.python .input}
#@tab all
class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
d2l.plot(d2l.arange(50), [scheduler(t) for t in range(50)])
```

Điều này cũng có thể được thực hiện bằng trình lập lịch tích hợp trong MXNet thông qua đối tượng `lr_scheduler.FactorScheduler`. Phải mất thêm một vài tham số, chẳng hạn như thời gian khởi động, chế độ khởi động (tuyến tính hoặc không đổi), số lượng cập nhật mong muốn tối đa, v.v.; Về phía trước, chúng tôi sẽ sử dụng các bộ lập lịch tích hợp phù hợp và chỉ giải thích chức năng của chúng ở đây. Như minh họa, nó là khá đơn giản để xây dựng lịch trình của riêng bạn nếu cần thiết. 

### Bộ lập lịch đa yếu tố

Một chiến lược phổ biến để đào tạo các mạng sâu là giữ cho tỷ lệ học tập không đổi và giảm nó bằng một số tiền nhất định thường xuyên. Đó là, đưa ra một tập hợp thời gian khi giảm tỷ lệ, chẳng hạn như $s = \{5, 10, 20\}$ giảm $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$ bất cứ khi nào $t \in s$. Giả sử rằng các giá trị giảm một nửa ở mỗi bước chúng ta có thể thực hiện điều này như sau.

```{.python .input}
scheduler = lr_scheduler.MultiFactorScheduler(step=[15, 30], factor=0.5,
                                              base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)

def get_lr(trainer, scheduler):
    lr = scheduler.get_last_lr()[0]
    trainer.step()
    scheduler.step()
    return lr

d2l.plot(d2l.arange(num_epochs), [get_lr(trainer, scheduler) 
                                  for t in range(num_epochs)])
```

```{.python .input}
#@tab tensorflow
class MultiFactorScheduler:
    def __init__(self, step, factor, base_lr):
        self.step = step
        self.factor = factor
        self.base_lr = base_lr
  
    def __call__(self, epoch):
        if epoch in self.step:
            self.base_lr = self.base_lr * self.factor
            return self.base_lr
        else:
            return self.base_lr

scheduler = MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Trực giác đằng sau lịch trình tỷ lệ học tập liên tục piecewise này là một cho phép tối ưu hóa tiến hành cho đến khi đạt được một điểm cố định về sự phân bố của vectơ trọng lượng. Sau đó (và chỉ sau đó) chúng ta có giảm tỷ lệ như để có được một proxy chất lượng cao hơn đến mức tối thiểu địa phương tốt. Ví dụ dưới đây cho thấy làm thế nào điều này có thể tạo ra các giải pháp tốt hơn bao giờ hết.

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### Bộ lập lịch cosine

Một heuristic khá bối rối đã được đề xuất bởi :cite:`Loshchilov.Hutter.2016`. Nó dựa vào quan sát rằng chúng ta có thể không muốn giảm tốc độ học tập quá đáng kể ngay từ đầu và hơn nữa, rằng chúng ta có thể muốn “tinh chỉnh” giải pháp cuối cùng bằng cách sử dụng một tốc độ học tập rất nhỏ. Điều này dẫn đến một lịch trình giống như cosin với dạng chức năng sau đây cho tỷ lệ học tập trong khoảng $t \in [0, T]$. 

$$\eta_t = \eta_T + \frac{\eta_0 - \eta_T}{2} \left(1 + \cos(\pi t/T)\right)$$

Ở đây $\eta_0$ là tỷ lệ học tập ban đầu, $\eta_T$ là tỷ lệ mục tiêu tại thời điểm $T$. Hơn nữa, đối với $t > T$, chúng tôi chỉ cần ghim giá trị lên $\eta_T$ mà không tăng lại. Trong ví dụ sau, chúng tôi đặt bước cập nhật tối đa $T = 20$.

```{.python .input}
scheduler = lr_scheduler.CosineScheduler(max_update=20, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps
  
    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Trong bối cảnh tầm nhìn máy tính, lịch trình này * có thể* dẫn đến kết quả được cải thiện. Tuy nhiên, lưu ý rằng những cải tiến như vậy không được đảm bảo (như có thể thấy dưới đây).

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### Khởi động

Trong một số trường hợp khởi tạo các tham số là không đủ để đảm bảo một giải pháp tốt. Điều này đặc biệt là một vấn đề đối với một số thiết kế mạng tiên tiến có thể dẫn đến các vấn đề tối ưu hóa không ổn định. Chúng tôi có thể giải quyết điều này bằng cách chọn một tốc độ học tập đủ nhỏ để ngăn chặn sự phân kỳ ngay từ đầu. Thật không may, điều này có nghĩa là tiến bộ chậm. Ngược lại, một tỷ lệ học tập lớn ban đầu dẫn đến sự phân kỳ. 

Một khắc phục khá đơn giản cho tình huống khó xử này là sử dụng thời gian khởi động trong đó tốc độ học tập * tăng* lên mức tối đa ban đầu và để hạ nhiệt tỷ lệ cho đến khi kết thúc quá trình tối ưu hóa. Để đơn giản, người ta thường sử dụng sự gia tăng tuyến tính cho mục đích này. Điều này dẫn đến một lịch trình của biểu mẫu được chỉ định dưới đây.

```{.python .input}
scheduler = lr_scheduler.CosineScheduler(20, warmup_steps=5, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(np.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Lưu ý rằng mạng hội tụ tốt hơn ban đầu (đặc biệt quan sát hiệu suất trong 5 thời đại đầu tiên).

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

Khởi động có thể được áp dụng cho bất kỳ bộ lập lịch nào (không chỉ cosine). Để có một cuộc thảo luận chi tiết hơn về lịch trình học tập và nhiều thí nghiệm khác xem thêm :cite:`Gotmare.Keskar.Xiong.ea.2018`. Đặc biệt, họ thấy rằng một giai đoạn khởi động giới hạn số lượng phân kỳ của các tham số trong các mạng rất sâu. Điều này có ý nghĩa trực giác vì chúng ta sẽ mong đợi sự phân kỳ đáng kể do khởi tạo ngẫu nhiên trong những phần của mạng mất nhiều thời gian nhất để đạt được tiến bộ ngay từ đầu. 

## Tóm tắt

* Giảm tỷ lệ học tập trong quá trình đào tạo có thể dẫn đến cải thiện độ chính xác và (khó chịu nhất) giảm quá mức của mô hình.
* Giảm từng phần tỷ lệ học tập bất cứ khi nào tiến bộ đã đạt được hiệu quả trong thực tế. Về cơ bản, điều này đảm bảo rằng chúng tôi hội tụ hiệu quả với một giải pháp phù hợp và chỉ sau đó giảm phương sai vốn có của các tham số bằng cách giảm tỷ lệ học tập.
* Cosine scheulers là phổ biến cho một số vấn đề tầm nhìn máy tính. Xem ví dụ, [GluonCV](http://gluon-cv.mxnet.io) để biết chi tiết về trình lập lịch như vậy.
* Một thời gian khởi động trước khi tối ưu hóa có thể ngăn chặn sự phân kỳ.
* Tối ưu hóa phục vụ nhiều mục đích trong học sâu. Bên cạnh việc giảm thiểu mục tiêu đào tạo, các lựa chọn khác nhau về thuật toán tối ưu hóa và lập kế hoạch tỷ lệ học tập có thể dẫn đến số lượng tổng quát hóa và quá mức khác nhau trên bộ thử nghiệm (đối với cùng một lượng lỗi đào tạo).

## Bài tập

1. Thử nghiệm với hành vi tối ưu hóa cho một tỷ lệ học tập cố định nhất định. Mô hình tốt nhất bạn có thể có được theo cách này là gì?
1. Làm thế nào để hội tụ thay đổi nếu bạn thay đổi số mũ của sự giảm tỷ lệ học tập? Sử dụng `PolyScheduler` để thuận tiện cho bạn trong các thí nghiệm.
1. Áp dụng bộ lập lịch cosine cho các vấn đề tầm nhìn máy tính lớn, ví dụ: đào tạo ImageNet. Làm thế nào để nó ảnh hưởng đến hiệu suất so với các lập lịch khác?
1. Thời gian khởi động nên kéo dài bao lâu?
1. Bạn có thể kết nối tối ưu hóa và lấy mẫu? Bắt đầu bằng cách sử dụng kết quả từ :cite:`Welling.Teh.2011` trên Stochastic Gradient Langevin Dynamics.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/359)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1080)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1081)
:end_tab:
