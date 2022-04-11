# Mạng sử dụng khối (VGG)
:label:`sec_vgg`

Trong khi AlexNet đưa ra bằng chứng thực nghiệm rằng CNN sâu có thể đạt được kết quả tốt, nó không cung cấp một mẫu chung để hướng dẫn các nhà nghiên cứu tiếp theo trong việc thiết kế các mạng mới. Trong các phần sau, chúng tôi sẽ giới thiệu một số khái niệm heuristic thường được sử dụng để thiết kế các mạng sâu. 

Tiến bộ trong lĩnh vực này phản ánh rằng trong thiết kế chip nơi các kỹ sư đã đi từ đặt bóng bán dẫn đến các yếu tố logic đến các khối logic. Tương tự, thiết kế của các kiến trúc mạng thần kinh đã phát triển dần dần trừu tượng hơn, với các nhà nghiên cứu chuyển từ suy nghĩ về các tế bào thần kinh riêng lẻ sang toàn bộ các lớp, và bây giờ đến các khối, lặp lại các mô hình của các lớp. 

Ý tưởng sử dụng các khối đầu tiên xuất hiện từ [Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/) (VGG) tại Đại học Oxford, trong mạng *VGG* eponymously-name của họ. Thật dễ dàng để thực hiện các cấu trúc lặp đi lặp lại này trong code với bất kỳ khuôn khổ học sâu hiện đại nào bằng cách sử dụng các vòng lặp và chương trình con. 

## (** VGG Blocks**)
:label:`subsec_vgg-blocks`

Khối xây dựng cơ bản của CNN cổ điển là một chuỗi các như sau: (i) một lớp tích hợp với đệm để duy trì độ phân giải, (ii) một phi tuyến tính như một ReLU, (iii) một lớp tổng hợp như một lớp tổng hợp tối đa. Một khối VGG bao gồm một chuỗi các lớp phức tạp, tiếp theo là một lớp tổng hợp tối đa để lấy mẫu không gian. Trong giấy VGG gốc :cite:`Simonyan.Zisserman.2014`, các tác giả đã sử dụng các phức hợp với $3\times3$ hạt nhân với lớp đệm 1 (giữ chiều cao và chiều rộng) và $2 \times 2$ tổng hợp tối đa với sải chân là 2 (giảm một nửa độ phân giải sau mỗi khối). Trong đoạn code dưới đây, ta định nghĩa một hàm gọi là `vgg_block` để thực hiện một khối VGG.

:begin_tab:`mxnet,tensorflow`
Hàm này có hai đối số tương ứng với số lớp phức tạp `num_convs` và số kênh đầu ra `num_channels`.
:end_tab:

:begin_tab:`pytorch`
Hàm này có ba đối số tương ứng với số lớp phức tạp `num_convs`, số kênh đầu vào `in_channels` và số kênh đầu ra `out_channels`.
:end_tab:

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3,
                          padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(tf.keras.layers.Conv2D(num_channels,kernel_size=3,
                                    padding='same',activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk
```

## [**VGG mạng**]

Giống như AlexNet và LeNet, Mạng VGG có thể được phân chia thành hai phần: phần đầu tiên bao gồm chủ yếu là các lớp phức tạp và tập hợp và thứ hai bao gồm các lớp được kết nối hoàn toàn. Điều này được miêu tả năm :numref:`fig_vgg`. 

![From AlexNet to VGG that is designed from building blocks.](../img/vgg.svg)
:width:`400px`
:label:`fig_vgg`

Phần tích tụ của mạng kết nối một số khối VGG từ :numref:`fig_vgg` (cũng được định nghĩa trong hàm `vgg_block`) liên tiếp. Biến sau đây `conv_arch` bao gồm một danh sách các tuples (một cho mỗi khối), trong đó mỗi khối chứa hai giá trị: số lớp phức tạp và số kênh đầu ra, đó chính xác là các đối số cần thiết để gọi hàm `vgg_block`. Phần được kết nối hoàn toàn của mạng VGG giống hệt với phần được bao phủ trong AlexNet. 

Mạng VGG ban đầu có 5 khối phức tạp, trong đó hai khối đầu tiên có một lớp ghép mỗi lớp và ba khối sau chứa hai lớp ghép mỗi lớp. Khối đầu tiên có 64 kênh đầu ra và mỗi khối tiếp theo tăng gấp đôi số kênh đầu ra, cho đến khi con số đó đạt 512. Vì mạng này sử dụng 8 lớp kết nối và 3 lớp kết nối hoàn toàn nên nó thường được gọi là VGG-11.

```{.python .input}
#@tab all
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
```

Mã sau đây thực hiện VGG-11. Đây là một vấn đề đơn giản của việc thực hiện một for-loop trên `conv_arch`.

```{.python .input}
def vgg(conv_arch):
    net = nn.Sequential()
    # The convolutional part
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # The fully-connected part
    net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(10))
    return net

net = vgg(conv_arch)
```

```{.python .input}
#@tab pytorch
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # The convolutional part
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # The fully-connected part
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)
```

```{.python .input}
#@tab tensorflow
def vgg(conv_arch):
    net = tf.keras.models.Sequential()
    # The convulational part
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # The fully-connected part
    net.add(tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10)]))
    return net

net = vgg(conv_arch)
```

Tiếp theo, chúng ta sẽ xây dựng một ví dụ dữ liệu đơn kênh với chiều cao và chiều rộng từ 224 đến [** quan sát hình dạng đầu ra của mỗi lớp**].

```{.python .input}
net.initialize()
X = np.random.uniform(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 224, 224, 1))
for blk in net.layers:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t', X.shape)
```

Như bạn có thể thấy, chúng ta giảm một nửa chiều cao và chiều rộng tại mỗi khối, cuối cùng đạt đến chiều cao và chiều rộng 7 trước khi làm phẳng các biểu diễn để xử lý bởi phần kết nối hoàn toàn của mạng. 

## Đào tạo

[**Vì VGG-11 nặng tính toán hơn AlexNet, chúng tôi xây dựng một mạng với số lượng kênh nhỏ hơn.**] Điều này là quá đủ để đào tạo về Fashion-MNIST.

```{.python .input}
#@tab mxnet, pytorch
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
```

```{.python .input}
#@tab tensorflow
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
# Recall that this has to be a function that will be passed to
# `d2l.train_ch6()` so that model building/compiling need to be within
# `strategy.scope()` in order to utilize the CPU/GPU devices that we have
net = lambda: vgg(small_conv_arch)
```

Ngoài việc sử dụng tốc độ học tập lớn hơn một chút, quá trình [** model training**] cũng tương tự như của AlexNet trong :numref:`sec_alexnet`.

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## Tóm tắt

* VGG-11 xây dựng một mạng bằng cách sử dụng các khối phức tạp có thể tái sử dụng. Các mô hình VGG khác nhau có thể được xác định bởi sự khác biệt về số lượng lớp phức tạp và các kênh đầu ra trong mỗi khối.
* Việc sử dụng các khối dẫn đến các đại diện rất nhỏ gọn của định nghĩa mạng. Nó cho phép thiết kế hiệu quả các mạng phức tạp.
* Trong giấy VGG của họ, Simonyan và Ziserman đã thử nghiệm với nhiều kiến trúc khác nhau. Đặc biệt, họ phát hiện ra rằng một số lớp phức tạp sâu và hẹp (tức là $3 \times 3$) có hiệu quả hơn so với ít lớp phức tạp rộng hơn.

## Bài tập

1. Khi in ra kích thước của các lớp chúng ta chỉ thấy 8 kết quả hơn là 11. Thông tin 3 lớp còn lại đã đi đâu?
1. So với AlexNet, VGG chậm hơn nhiều về mặt tính toán, và nó cũng cần nhiều bộ nhớ GPU hơn. Phân tích lý do cho việc này.
1. Hãy thử thay đổi chiều cao và chiều rộng của hình ảnh trong Fashion-MNIST từ 224 đến 96. Điều này có ảnh hưởng gì đến các thí nghiệm?
1. Tham khảo Bảng 1 trong giấy VGG :cite:`Simonyan.Zisserman.2014` để xây dựng các mô hình phổ biến khác, chẳng hạn như VGG-16 hoặc VGG-19.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/77)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/78)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/277)
:end_tab:
