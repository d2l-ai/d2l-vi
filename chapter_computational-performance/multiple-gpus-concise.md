# Triển khai ngắn gọn cho nhiều GPU
:label:`sec_multi_gpu_concise`

Thực hiện song song từ đầu cho mỗi mô hình mới là không thú vị. Hơn nữa, có lợi ích đáng kể trong việc tối ưu hóa các công cụ đồng bộ hóa cho hiệu suất cao. Trong phần sau đây, chúng tôi sẽ chỉ ra cách thực hiện việc này bằng cách sử dụng API cấp cao của các framework deep learning. Toán học và các thuật toán giống như trong :numref:`sec_multi_gpu`. Khá không ngạc nhiên khi bạn sẽ cần ít nhất hai GPU để chạy mã của phần này.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## [**Một mạng đồ chơi**]

Hãy để chúng tôi sử dụng một mạng có ý nghĩa hơn một chút so với LeNet từ :numref:`sec_multi_gpu` mà vẫn đủ dễ dàng và nhanh chóng để đào tạo. Chúng tôi chọn một biến thể ResNet-18 :cite:`He.Zhang.Ren.ea.2016`. Vì hình ảnh đầu vào rất nhỏ, chúng tôi sửa đổi nó một chút. Đặc biệt, sự khác biệt so với :numref:`sec_resnet` là chúng ta sử dụng một hạt nhân phức tạp nhỏ hơn, sải chân và đệm ở đầu. Hơn nữa, chúng tôi loại bỏ lớp tổng hợp tối đa.

```{.python .input}
#@save
def resnet18(num_classes):
    """A slightly modified ResNet-18 model."""
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(d2l.Residual(
                    num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk

    net = nn.Sequential()
    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the maximum pooling layer
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

```{.python .input}
#@tab pytorch
#@save
def resnet18(num_classes, in_channels=1):
    """A slightly modified ResNet-18 model."""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(in_channels, out_channels,
                                        use_1x1conv=True, strides=2))
            else:
                blk.append(d2l.Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the maximum pooling layer
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net
```

## Khởi tạo mạng

:begin_tab:`mxnet`
Chức năng `initialize` cho phép chúng tôi khởi tạo các tham số trên một thiết bị mà chúng tôi lựa chọn. Đối với một bồi dưỡng về các phương pháp khởi tạo xem :numref:`sec_numerical_stability`. Điều đặc biệt thuận tiện là nó cũng cho phép chúng tôi khởi tạo mạng trên các thiết bị * nhiều* cùng một lúc. Hãy để chúng tôi thử làm thế nào điều này hoạt động trong thực tế.
:end_tab:

:begin_tab:`pytorch`
Chúng tôi sẽ khởi tạo mạng bên trong vòng đào tạo. Đối với một bồi dưỡng về các phương pháp khởi tạo xem :numref:`sec_numerical_stability`.
:end_tab:

```{.python .input}
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# Initialize all the parameters of the network
net.initialize(init=init.Normal(sigma=0.01), ctx=devices)
```

```{.python .input}
#@tab pytorch
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# We will initialize the network inside the training loop
```

:begin_tab:`mxnet`
Sử dụng chức năng `split_and_load` được giới thiệu trong :numref:`sec_multi_gpu`, chúng ta có thể chia một minibatch dữ liệu và sao chép các phần vào danh sách các thiết bị được cung cấp bởi biến `devices`. Phiên bản mạng* automatically* sử dụng GPU thích hợp để tính toán giá trị của sự lan truyền chuyển tiếp. Ở đây chúng tôi tạo ra 4 quan sát và chia chúng qua GPU.
:end_tab:

```{.python .input}
x = np.random.uniform(size=(4, 1, 28, 28))
x_shards = gluon.utils.split_and_load(x, devices)
net(x_shards[0]), net(x_shards[1])
```

:begin_tab:`mxnet`
Khi dữ liệu đi qua mạng, các tham số tương ứng được khởi tạo * trên thiết bị dữ liệu được truyền qua*. Điều này có nghĩa là khởi tạo xảy ra trên cơ sở mỗi thiết bị. Vì chúng tôi đã chọn GPU 0 và GPU 1 để khởi tạo, mạng chỉ được khởi tạo ở đó chứ không phải trên CPU. Trong thực tế, các tham số thậm chí không tồn tại trên CPU. Chúng tôi có thể xác minh điều này bằng cách in ra các tham số và quan sát bất kỳ lỗi nào có thể phát sinh.
:end_tab:

```{.python .input}
weight = net[0].params.get('weight')

try:
    weight.data()
except RuntimeError:
    print('not initialized on cpu')
weight.data(devices[0])[0], weight.data(devices[1])[0]
```

:begin_tab:`mxnet`
Tiếp theo, chúng ta hãy thay thế mã thành [**đánh giá độ chính xác**] bằng một mã hoạt động (** song song trên nhiều thiết bị**). Điều này phục vụ như là một sự thay thế của chức năng `evaluate_accuracy_gpu` từ :numref:`sec_lenet`. Sự khác biệt chính là chúng tôi chia nhỏ một minibatch trước khi gọi mạng. Tất cả những thứ khác về cơ bản là giống hệt nhau.
:end_tab:

```{.python .input}
#@save
def evaluate_accuracy_gpus(net, data_iter, split_f=d2l.split_batch):
    """Compute the accuracy for a model on a dataset using multiple GPUs."""
    # Query the list of devices
    devices = list(net.collect_params().values())[0].list_ctx()
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for features, labels in data_iter:
        X_shards, y_shards = split_f(features, labels, devices)
        # Run in parallel
        pred_shards = [net(X_shard) for X_shard in X_shards]
        metric.add(sum(float(d2l.accuracy(pred_shard, y_shard)) for
                       pred_shard, y_shard in zip(
                           pred_shards, y_shards)), labels.size)
    return metric[0] / metric[1]
```

## [**Đào tạo**]

Như trước đây, mã đào tạo cần thực hiện một số chức năng cơ bản để song song hiệu quả: 

* Các tham số mạng cần được khởi tạo trên tất cả các thiết bị.
* Trong khi lặp lại các minibatches tập dữ liệu sẽ được chia trên tất cả các thiết bị.
* Chúng tôi tính toán sự mất mát và độ dốc của nó song song trên các thiết bị.
* Gradient được tổng hợp và các tham số được cập nhật cho phù hợp.

Cuối cùng, chúng tôi tính toán độ chính xác (một lần nữa song song) để báo cáo hiệu suất cuối cùng của mạng. Thói quen đào tạo khá giống với việc triển khai trong các chương trước, ngoại trừ việc chúng ta cần phân chia và tổng hợp dữ liệu.

```{.python .input}
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx = [d2l.try_gpu(i) for i in range(num_gpus)]
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        timer.start()
        for features, labels in train_iter:
            X_shards, y_shards = d2l.split_batch(features, labels, ctx)
            with autograd.record():
                ls = [loss(net(X_shard), y_shard) for X_shard, y_shard
                      in zip(X_shards, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        npx.waitall()
        timer.stop()
        animator.add(epoch + 1, (evaluate_accuracy_gpus(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(ctx)}')
```

```{.python .input}
#@tab pytorch
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)
    # Set the model on multiple GPUs
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

Hãy để chúng tôi xem làm thế nào điều này hoạt động trong thực tế. Như một khởi động, chúng tôi [** đào tạo mạng trên một GPU.**]

```{.python .input}
train(num_gpus=1, batch_size=256, lr=0.1)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=1, batch_size=256, lr=0.1)
```

Tiếp theo chúng ta [**sử dụng 2 GPU để đào tạo**]. So với LeNet được đánh giá vào năm :numref:`sec_multi_gpu`, mô hình cho ResNet-18 phức tạp hơn đáng kể. Đây là nơi song song cho thấy lợi thế của nó. Thời gian tính toán lớn hơn một cách có ý nghĩa so với thời gian đồng bộ hóa các tham số. Điều này cải thiện khả năng mở rộng vì chi phí cho song song ít liên quan hơn.

```{.python .input}
train(num_gpus=2, batch_size=512, lr=0.2)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=2, batch_size=512, lr=0.2)
```

## Tóm tắt

:begin_tab:`mxnet`
* Gluon cung cấp nguyên thủy để khởi tạo mô hình trên nhiều thiết bị bằng cách cung cấp một danh sách ngữ cảnh.
:end_tab:

* Dữ liệu được tự động đánh giá trên các thiết bị nơi dữ liệu có thể được tìm thấy.
* Hãy cẩn thận để khởi tạo các mạng trên mỗi thiết bị trước khi cố gắng truy cập các tham số trên thiết bị đó. Nếu không bạn sẽ gặp phải một lỗi.
* Các thuật toán tối ưu hóa tự động tổng hợp trên nhiều GPU.

## Bài tập

:begin_tab:`mxnet`
1. Phần này sử dụng ResNet-18. Hãy thử các thời đại khác nhau, quy mô hàng loạt và tỷ lệ học tập. Sử dụng nhiều GPU hơn để tính toán. Điều gì xảy ra nếu bạn dùng thử điều này với 16 GPU (ví dụ: trên phiên bản AWS p2.16xlarge)?
1. Đôi khi, các thiết bị khác nhau cung cấp sức mạnh tính toán khác nhau. Chúng ta có thể sử dụng GPU và CPU cùng một lúc. Làm thế nào chúng ta nên chia công việc? Nó có đáng để nỗ lực không? Tại sao? Tại sao không?
1. Điều gì sẽ xảy ra nếu chúng ta thả `npx.waitall()`? Làm thế nào bạn sẽ sửa đổi đào tạo sao cho bạn có một chồng chéo lên đến hai bước cho song song?
:end_tab:

:begin_tab:`pytorch`
1. Phần này sử dụng ResNet-18. Hãy thử các thời đại khác nhau, quy mô hàng loạt và tỷ lệ học tập. Sử dụng nhiều GPU hơn để tính toán. Điều gì xảy ra nếu bạn dùng thử điều này với 16 GPU (ví dụ: trên phiên bản AWS p2.16xlarge)?
1. Đôi khi, các thiết bị khác nhau cung cấp sức mạnh tính toán khác nhau. Chúng ta có thể sử dụng GPU và CPU cùng một lúc. Làm thế nào chúng ta nên chia công việc? Nó có đáng để nỗ lực không? Tại sao? Tại sao không?
:end_tab:

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/365)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1403)
:end_tab:
