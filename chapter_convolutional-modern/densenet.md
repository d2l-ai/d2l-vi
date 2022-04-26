# Mạng kết nối mật độ (DenseNet)

ResNet thay đổi đáng kể quan điểm về cách tham số hóa các chức năng trong các mạng sâu. * DenseNet* (mạng phức tạp dày đặc) ở một mức độ nào đó là phần mở rộng hợp lý của :cite:`Huang.Liu.Van-Der-Maaten.ea.2017` này. Để hiểu làm thế nào để đến nó, chúng ta hãy đi một đường vòng nhỏ đến toán học. 

## Từ ResNet đến DenseNet

Nhớ lại bản mở rộng Taylor cho các chức năng. Đối với điểm $x = 0$ nó có thể được viết là 

$$f(x) = f(0) + f'(0) x + \frac{f''(0)}{2!}  x^2 + \frac{f'''(0)}{3!}  x^3 + \ldots.$$

Điểm mấu chốt là nó phân hủy một hàm thành các thuật ngữ thứ tự ngày càng cao hơn. Trong một tĩnh mạch tương tự, ResNet phân hủy các chức năng thành 

$$f(\mathbf{x}) = \mathbf{x} + g(\mathbf{x}).$$

Đó là, ResNet phân hủy $f$ thành một thuật ngữ tuyến tính đơn giản và một thuật ngữ phi tuyến phức tạp hơn. Điều gì sẽ xảy ra nếu chúng ta muốn nắm bắt (không nhất thiết phải thêm) thông tin ngoài hai thuật ngữ? Một giải pháp là DenseNet :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`. 

![The main difference between ResNet (left) and DenseNet (right) in cross-layer connections: use of addition and use of concatenation. ](../img/densenet-block.svg)
:label:`fig_densenet_block`

Như thể hiện trong :numref:`fig_densenet_block`, sự khác biệt chính giữa ResNet và DenseNet là trong trường hợp đầu ra sau là * concatenated* (được biểu thị bởi $[,]$) thay vì được thêm vào. Do đó, chúng tôi thực hiện ánh xạ từ $\mathbf{x}$ đến các giá trị của nó sau khi áp dụng một chuỗi hàm ngày càng phức tạp: 

$$\mathbf{x} \to \left[
\mathbf{x},
f_1(\mathbf{x}),
f_2([\mathbf{x}, f_1(\mathbf{x})]), f_3([\mathbf{x}, f_1(\mathbf{x}), f_2([\mathbf{x}, f_1(\mathbf{x})])]), \ldots\right].$$

Cuối cùng, tất cả các chức năng này được kết hợp trong MLP để giảm số lượng tính năng một lần nữa. Về mặt thực hiện, điều này khá đơn giản: thay vì thêm các thuật ngữ, chúng tôi nối chúng. Tên DenseNet phát sinh từ thực tế là đồ thị phụ thuộc giữa các biến trở nên khá dày đặc. Lớp cuối cùng của một chuỗi như vậy được kết nối mật độ với tất cả các lớp trước đó. Các kết nối dày đặc được hiển thị trong :numref:`fig_densenet`. 

![Dense connections in DenseNet.](../img/densenet.svg)
:label:`fig_densenet`

Các thành phần chính tạo ra DenseNet là các khối * dày đặc * và lớp chuyển thuyền*. Cái trước xác định cách các đầu vào và đầu ra được nối, trong khi cái sau kiểm soát số lượng kênh để nó không quá lớn. 

## [** Khối dày dàng**]

DenseNet sử dụng cấu trúc “chuẩn hóa hàng loạt, kích hoạt và phức tạp” được sửa đổi của ResNet (xem bài tập trong :numref:`sec_resnet`). Đầu tiên, chúng tôi thực hiện cấu trúc khối phức tạp này.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(
            filters=num_channels, kernel_size=(3, 3), padding='same')

        self.listLayers = [self.bn, self.relu, self.conv]

    def call(self, x):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x,y], axis=-1)
        return y
```

Một khối * dày dàng* bao gồm nhiều khối phức tạp, mỗi khối sử dụng cùng một số kênh đầu ra. Tuy nhiên, trong quá trình lan truyền về phía trước, chúng tôi nối đầu vào và đầu ra của mỗi khối phức tạp trên kích thước kênh.

```{.python .input}
class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate the input and output of each block on the channel
            # dimension
            X = np.concatenate((X, Y), axis=1)
        return X
```

```{.python .input}
#@tab pytorch
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate the input and output of each block on the channel
            # dimension
            X = torch.cat((X, Y), dim=1)
        return X
```

```{.python .input}
#@tab tensorflow
class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        self.listLayers = []
        for _ in range(num_convs):
            self.listLayers.append(ConvBlock(num_channels))

    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x
```

Trong ví dụ sau, chúng ta [**define a `DenseBlock` instance**] với 2 khối phức tạp của 10 kênh đầu ra. Khi sử dụng đầu vào với 3 kênh, chúng tôi sẽ nhận được đầu ra với $3+2\times 10=23$ kênh. Số lượng các kênh khối phức tạp kiểm soát sự tăng trưởng về số lượng kênh đầu ra so với số lượng kênh đầu vào. Đây cũng được gọi là * tốc độ tăng trưởng*.

```{.python .input}
blk = DenseBlock(2, 10)
blk.initialize()
X = np.random.uniform(size=(4, 3, 8, 8))
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab pytorch
blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab tensorflow
blk = DenseBlock(2, 10)
X = tf.random.uniform((4, 8, 8, 3))
Y = blk(X)
Y.shape
```

## [**Layers Transition**]

Vì mỗi khối dày đặc sẽ làm tăng số lượng kênh, việc thêm quá nhiều trong số chúng sẽ dẫn đến một mô hình quá phức tạp. Một lớp chuyển đổi* được sử dụng để kiểm soát độ phức tạp của mô hình. Nó làm giảm số lượng kênh bằng cách sử dụng lớp ghép $1\times 1$ và giảm một nửa chiều cao và chiều rộng của lớp tổng hợp trung bình với một bước tiến là 2, làm giảm thêm độ phức tạp của mô hình.

```{.python .input}
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
#@tab pytorch
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
```

```{.python .input}
#@tab tensorflow
class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(num_channels, kernel_size=1)
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=2, strides=2)

    def call(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)
```

[**Áp dụng một lớp chuyển tiếp**] với 10 kênh đến đầu ra của khối dày đặc trong ví dụ trước. Điều này làm giảm số lượng kênh đầu ra xuống 10 và giảm một nửa chiều cao và chiều rộng.

```{.python .input}
blk = transition_block(10)
blk.initialize()
blk(Y).shape
```

```{.python .input}
#@tab pytorch
blk = transition_block(23, 10)
blk(Y).shape
```

```{.python .input}
#@tab tensorflow
blk = TransitionBlock(10)
blk(Y).shape
```

## [**Mô hình DenseNet**]

Tiếp theo, chúng ta sẽ xây dựng một mô hình DenseNet. DenseNet lần đầu tiên sử dụng cùng một lớp tích hợp đơn và lớp tổng hợp tối đa như trong ResNet.

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def block_1():
    return tf.keras.Sequential([
       tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
       tf.keras.layers.BatchNormalization(),
       tf.keras.layers.ReLU(),
       tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

Sau đó, tương tự như bốn mô-đun được tạo thành từ các khối còn lại mà ResNet sử dụng, DenseNet sử dụng bốn khối dày đặc. Tương tự như ResNet, chúng ta có thể đặt số lớp phức tạp được sử dụng trong mỗi khối dày đặc. Ở đây, chúng tôi đặt nó thành 4, phù hợp với mô hình ResNet-18 trong :numref:`sec_resnet`. Hơn nữa, chúng tôi đặt số lượng kênh (tức là tốc độ tăng trưởng) cho các lớp phức tạp trong khối dày đặc thành 32, vì vậy 128 kênh sẽ được thêm vào mỗi khối dày đặc. 

Trong ResNet, chiều cao và chiều rộng được giảm giữa mỗi mô-đun bởi một khối còn lại với một sải chân là 2. Ở đây, chúng ta sử dụng layer transition để giảm một nửa chiều cao và chiều rộng và giảm một nửa số kênh.

```{.python .input}
# `num_channels`: the current number of channels
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    net.add(DenseBlock(num_convs, growth_rate))
    # This is the number of output channels in the previous dense block
    num_channels += num_convs * growth_rate
    # A transition layer that halves the number of channels is added between
    # the dense blocks
    if i != len(num_convs_in_dense_blocks) - 1:
        num_channels //= 2
        net.add(transition_block(num_channels))
```

```{.python .input}
#@tab pytorch
# `num_channels`: the current number of channels
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # This is the number of output channels in the previous dense block
    num_channels += num_convs * growth_rate
    # A transition layer that halves the number of channels is added between
    # the dense blocks
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2
```

```{.python .input}
#@tab tensorflow
def block_2():
    net = block_1()
    # `num_channels`: the current number of channels
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        net.add(DenseBlock(num_convs, growth_rate))
        # This is the number of output channels in the previous dense block
        num_channels += num_convs * growth_rate
        # A transition layer that halves the number of channels is added
        # between the dense blocks
        if i != len(num_convs_in_dense_blocks) - 1:
            num_channels //= 2
            net.add(TransitionBlock(num_channels))
    return net
```

Tương tự như ResNet, một lớp tổng hợp toàn cầu và một lớp kết nối hoàn toàn được kết nối ở cuối để tạo ra đầu ra.

```{.python .input}
net.add(nn.BatchNorm(),
        nn.Activation('relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10))
```

```{.python .input}
#@tab tensorflow
def net():
    net = block_2()
    net.add(tf.keras.layers.BatchNormalization())
    net.add(tf.keras.layers.ReLU())
    net.add(tf.keras.layers.GlobalAvgPool2D())
    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(10))
    return net
```

## [**Đào tạo**]

Vì chúng ta đang sử dụng mạng sâu hơn ở đây, trong phần này, chúng ta sẽ giảm chiều cao và chiều rộng đầu vào từ 224 xuống 96 để đơn giản hóa tính toán.

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## Tóm tắt

* Về kết nối chéo lớp, không giống như ResNet, nơi đầu vào và đầu ra được thêm vào với nhau, DenseNet nối đầu vào và đầu ra trên kích thước kênh.
* Các thành phần chính sáng tác DenseNet là các khối dày đặc và các lớp chuyển tiếp.
* Chúng ta cần phải kiểm soát kích thước khi soạn mạng bằng cách thêm các lớp chuyển tiếp thu nhỏ lại số kênh.

## Bài tập

1. Tại sao chúng ta sử dụng tổng hợp trung bình chứ không phải là tổng hợp tối đa trong layer chuyển tiếp?
1. Một trong những lợi thế được đề cập trong giấy DenseNet là các thông số mô hình của nó nhỏ hơn so với các thông số của ResNet. Tại sao lại là trường hợp này?
1. Một vấn đề mà DenseNet đã bị chỉ trích là mức tiêu thụ bộ nhớ cao.
    1. Đây có thực sự là trường hợp? Cố gắng thay đổi hình dạng đầu vào thành $224\times 224$ để xem mức tiêu thụ bộ nhớ GPU thực tế.
    1. Bạn có thể nghĩ về một phương tiện thay thế để giảm tiêu thụ bộ nhớ? Làm thế nào bạn sẽ cần phải thay đổi khuôn khổ?
1. Thực hiện các phiên bản DenseNet khác nhau được trình bày trong Bảng 1 của giấy DenseNet :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`.
1. Thiết kế mô hình dựa trên MLP bằng cách áp dụng ý tưởng DenseNet. Áp dụng nó cho nhiệm vụ dự đoán giá nhà ở trong :numref:`sec_kaggle_house`.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/87)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/88)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/331)
:end_tab:
