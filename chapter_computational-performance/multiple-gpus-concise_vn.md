<!--
# Concise Implementation for Multiple GPUs
-->

# Cách lập trình Súc tích đa GPU
:label:`sec_multi_gpu_gluon`


<!--
Implementing parallelism from scratch for every new model is no fun.
Moreover, there is significant benefit in optimizing synchronization tools for high performance.
In the following we will show how to do this using Gluon.
The math and the algorithms are the same as in :numref:`sec_multi_gpu`.
As before we begin by importing the required modules (quite unsurprisingly you will need at least two GPUs to run this notebook).
-->

Lập trình từ đầu việc song song hóa cho từng mô hình mới khá mất công.
Hơn nữa, việc tối ưu các công cụ đồng bộ hóa sẽ cho hiệu suất cao.
Sau đây chúng tôi sẽ giới thiệu cách thực hiện điều này bằng Gluon.
Phần lý thuyết toán và các thuật toán giống trong :numref:`sec_multi_gpu`.
Như trước đây, ta bắt đầu bằng cách nhập các mô-đun cần thiết (tất nhiên là ta sẽ cần ít nhất hai GPU để chạy notebook này).



```{.python .input  n=1}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```


<!--
## A Toy Network
-->

## Ví dụ Đơn giản


<!--
Let us use a slightly more meaningful network than LeNet from the previous section that's still sufficiently easy and quick to train.
We pick a ResNet-18 variant :cite:`He.Zhang.Ren.ea.2016`.
Since the input images are tiny we modify it slightly.
In particular, the difference to :numref:`sec_resnet` is that we use a smaller convolution kernel, stride, and padding at the beginning.
Moreover, we remove the max-pooling layer.
-->

Hãy sử dụng một mạng có ý nghĩa hơn một chút so với LeNet ở phần trước mà vẫn có thể huấn luyện dễ dàng và nhanh chóng.
Chúng tôi chọn một biến thể của ResNet-18 :cite:`He.Zhang.Ren.ea.2016`.
Vì hình ảnh đầu vào rất nhỏ nên ta sửa đổi nó một chút.
Cụ thể, điểm khác biệt so với ở :numref:`sec_resnet` là ở phần đầu, ta sử dụng hạt nhân tích chập có kích thước, sải bước và đệm nhỏ hơn, và cũng loại bỏ đi tầng gộp cực đại.



```{.python .input  n=2}
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


<!--
## Parameter Initialization and Logistics
-->

## Khởi tạo Tham số và Công việc phụ trợ


<!--
The `initialize` method allows us to set initial defaults for parameters on a device of our choice.
For a refresher see :numref:`sec_numerical_stability`.
What is particularly convenient is that it also lets us initialize the network on *multiple* devices simultaneously.
Let us try how this works in practice.
-->

Phương thức `initialize` cho phép ta thiết lập giá trị mặc định ban đầu cho các tham số trên thiết bị được chọn.
Với độc giả mới, có thể tham khảo :numref:`sec_numerical_stability`.
Một điều rất thuận tiện là nó cũng cho phép ta khởi tạo mạng trên *nhiều* thiết bị cùng một lúc.
Hãy thử xem cách nó hoạt động trong thực tế.


```{.python .input  n=3}
net = resnet18(10)
# get a list of GPUs
ctx = d2l.try_all_gpus()
# initialize the network on all of them 
net.initialize(init=init.Normal(sigma=0.01), ctx=ctx)
```


<!--
Using the `split_and_load` function introduced in the previous section we can divide a minibatch of data and copy portions to the list of devices provided by the context variable.
The network object *automatically* uses the appropriate GPU to compute the value of the forward pass.
As before we generate 4 observations and split them over the GPUs.
-->

Sử dụng hàm `split_and_load` được giới thiệu trong phần trước, chúng ta có thể phân chia một minibatch dữ liệu và sao chép các phần dữ liệu vào danh sách các thiết bị được cung cấp bởi biến ngữ cảnh.
Mạng sẽ *tự động* sử dụng GPU thích hợp để tính giá trị của lượt truyền xuôi.
Ta tạo ra 4 mẫu dữ liệu và phân chia chúng trên các GPU như trước đây.


```{.python .input  n=4}
x = np.random.uniform(size=(4, 1, 28, 28))
x_shards = gluon.utils.split_and_load(x, ctx)
net(x_shards[0]), net(x_shards[1])
```


<!--
Once data passes through the network, the corresponding parameters are initialized *on the device the data passed through*.
This means that initialization happens on a per-device basis.
Since we picked GPU 0 and GPU 1 for initialization, the network is initialized only there, and not on the CPU.
In fact, the parameters do not even exist on the device.
We can verify this by printing out the parameters and observing any errors that might arise.
-->

Khi dữ liệu được truyền qua mạng, các tham số tương ứng sẽ được khởi tạo *trên thiết bị mà dữ liệu được truyền qua*.
Điều này có nghĩa là việc khởi tạo xảy ra theo từng thiết bị.
Do ta lựa chọn việc khởi tạo trên GPU 0 và GPU 1, mạng chỉ được khởi tạo trên hai thiết bị này chứ trên CPU thì không.
Trong thực tế, các tham số này thậm chí còn không tồn tại trên CPU.
Ta có thể kiểm chứng điều này bằng cách in các tham số ra và theo dõi xem liệu có lỗi nào xảy ra hay không.


```{.python .input  n=5}
weight = net[0].params.get('weight')

try:
    weight.data()
except RuntimeError:
    print('not initialized on cpu')
weight.data(ctx[0])[0], weight.data(ctx[1])[0]
```


<!--
Lastly let us replace the code to evaluate the accuracy by one that works in parallel across multiple devices.
This serves as a replacement of the `evaluate_accuracy_gpu` function from :numref:`sec_lenet`.
The main difference is that we split a batch before invoking the network.
All else is essentially identical.
-->

Cuối cùng, hãy cùng thay đổi đoạn mã đánh giá độ chính xác để có thể chạy song song trên nhiều thiết bị.
Hàm này được viết lại từ hàm `evaluate_accuracy_gpu` ở :numref:`sec_lenet`.
Điểm khác biệt lớn nhất nằm ở việc ta tách một batch ra trước khi truyền vào mạng.
Các phần còn lại gần như là giống hệt.


```{.python .input  n=6}
#@save
def evaluate_accuracy_gpus(net, data_iter, split_f=d2l.split_batch):
    # Query the list of devices
    ctx = list(net.collect_params().values())[0].list_ctx()
    metric = d2l.Accumulator(2)  # num_corrected_examples, num_examples
    for features, labels in data_iter:
        X_shards, y_shards = split_f(features, labels, ctx)
        # Run in parallel
        pred_shards = [net(X_shard) for X_shard in X_shards]
        metric.add(sum(float(d2l.accuracy(pred_shard, y_shard)) for
                       pred_shard, y_shard in zip(
                           pred_shards, y_shards)), labels.size)
    return metric[0] / metric[1]
```


<!--
## Training
-->

## Huấn luyện


<!--
As before, the training code needs to perform a number of basic functions for efficient parallelism:
-->

Như phần trên, đoạn mã huấn luyện cần thực hiện một số hàm cơ bản để quá trình song song hóa đạt hiệu quả:


<!--
* Network parameters need to be initialized across all devices.
* While iterating over the dataset minibatches are to be divided across all devices.
* We compute the loss and its gradient in parallel across devices. 
* Losses are aggregated (by the trainer method) and parameters are updated accordingly. 
-->

* Các tham số của mạng cần được khởi tạo trên tất cả các thiết bị.
* Trong suốt quá trình lặp trên tập dữ liệu, các minibatch được chia nhỏ cho tất cả các thiết bị.
* Ta tính toán song song hàm mất mát và gradient của nó trên tất cả các thiết bị.
* Mất mát được tích luỹ (bởi phương thức huấn luyện `trainer`) và các tham số được cập nhật tương ứng.


<!--
In the end we compute the accuracy (again in parallel) to report the final value of the network.
The training routine is quite similar to implementations in previous chapters, except that we need to split and aggregate data.
-->

Cuối cùng ta tính toán (vẫn song song) độ chính xác và báo cáo giá trị cuối cùng của mạng.
Quá trình huấn luyện ở đây khá giống với chương trước, trừ việc ta cần chia nhỏ và tổng hợp lại dữ liệu.



```{.python .input  n=7}
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
                losses = [loss(net(X_shard), y_shard) for X_shard, y_shard
                          in zip(X_shards, y_shards)]
            for l in losses:
                l.backward()
            trainer.step(batch_size)
        npx.waitall()
        timer.stop()
        animator.add(epoch + 1, (evaluate_accuracy_gpus(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(ctx)}')
```


<!--
## Experiments
-->

## Thử nghiệm


<!--
Let us see how this works in practice. As a warmup we train the network on a single GPU.
-->

Hãy cùng xem cách hoạt động trong thực tế. Để khởi động, ta huấn luyện mạng này trên một GPU đơn.


```{.python .input  n=8}
train(num_gpus=1, batch_size=256, lr=0.1)
```

<!--
Next we use 2 GPUs for training. Compared to LeNet the model for ResNet-18 is considerably more complex.
This is where parallelization shows its advantage.
The time for computation is meaningfully larger than the time for synchronizing parameters.
This improves scalability since the overhead for parallelization is less relevant.
-->

Tiếp theo, ta sử dụng 2 GPU để huấn luyện. Mô hình ResNet-18 phức tạp hơn đáng kể so với LeNet.
Đây chính là cơ hội để song song hóa chứng tỏ lợi thế của nó,
vì thời gian dành cho việc tính toán lớn hơn đáng kể so với thời gian đồng bộ hóa các tham số.
Điều này giúp cải thiện khả năng mở rộng do tổng chi phí song song hóa không quá đáng kể.


```{.python .input  n=9}
train(num_gpus=2, batch_size=512, lr=0.2)
```


## Tóm tắt

<!--
* Gluon provides primitives for model initialization across multiple devices by providing a context list.
* Data is automatically evaluated on the devices where the data can be found.
* Take care to initialize the networks on each device before trying to access the parameters on that device. Otherwise you will encounter an error.
* The optimization algorithms automatically aggregate over multiple GPUs.
-->

* Gluon cung cấp các hàm để khởi tạo mô hình trên nhiều thiết bị bằng cách cung cấp một danh sách ngữ cảnh.
* Dữ liệu được tự động đánh giá trên các thiết bị mà nó được lưu trữ.
* Chú ý việc khởi tạo mạng trên mỗi thiết bị trước khi thử truy cập vào các tham số trên thiết bị đó. Nếu không khả năng cao sẽ có lỗi xảy ra.
* Các thuật toán tối ưu tự động tổng hợp kết quả trên nhiều GPU.


## Bài tập

<!--
1. This section uses ResNet-18. Try different epochs, batch sizes, and learning rates. Use more GPUs for computation. What happens if you try this on a p2.16xlarge instance with 16 GPUs? 
2. Sometimes, different devices provide different computing power. We could use the GPUs and the CPU at the same time. How should we divide the work? Is it worth the effort? Why? Why not?
3. What happens if we drop `npx.waitall()`? How would you modify training such that you have an overlap of up to two steps for parallelism? 
-->

1. Phần này ta sử dụng ResNet-18. Hãy thử với số epoch, kích thước batch và tốc độ học khác. Thử sử dụng nhiều GPU hơn để tính toán.
Chuyện gì sẽ xảy ra nếu ta chạy mô hình này trên máy chủ p2.16xlarge với 16 GPU?
2. Đôi khi mỗi thiết bị khác nhau cung cấp khả năng tính toán khác nhau. Ta có thể sử dụng GPU và CPU cùng lúc.
Vậy ta nên phân chia công việc thế nào? Liệu việc phân chia có đáng hay không? Tại sao?
3. Chuyện gì sẽ xảy ra nếu ta bỏ hàm `npx.waitall()`? Bạn sẽ thay đổi quá trình huấn luyện thế nào để có thể xử lý song song tối đa 2 bước cùng lúc?



## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/365)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Trần Yến Thy
* Lê Khắc Hồng Phúc
* Nguyễn Văn Cường
* Đỗ Trường Giang
* Nguyễn Lê Quang Nhật
* Phạm Hồng Vinh
