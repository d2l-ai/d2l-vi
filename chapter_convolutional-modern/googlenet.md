# Mạng có kết nối song song (GoogLeNet)
:label:`sec_googlenet`

Năm 2014, *GoogLeNet* giành chiến thắng trong ImageNet Challenge, đề xuất một cấu trúc kết hợp các thế mạnh của Nin và mô hình của các khối lặp lại :cite:`Szegedy.Liu.Jia.ea.2015`. Một trọng tâm của bài báo là giải quyết câu hỏi về hạt nhân có kích thước nào là tốt nhất. Rốt cuộc, các mạng phổ biến trước đây sử dụng các lựa chọn nhỏ tới $1 \times 1$ và lớn tới $11 \times 11$. Một cái nhìn sâu sắc trong bài báo này là đôi khi nó có thể thuận lợi để sử dụng một sự kết hợp của các hạt nhân có kích thước đa dạng. Trong phần này, chúng tôi sẽ giới thiệu GoogLeNet, trình bày một phiên bản đơn giản hóa một chút của mô hình gốc: chúng tôi bỏ qua một vài tính năng đặc biệt đã được thêm vào để ổn định đào tạo nhưng bây giờ không cần thiết với các thuật toán đào tạo tốt hơn có sẵn. 

## (**Chống** Bắt nối**)

Khối phức tạp cơ bản trong GoogLeNet được gọi là khối *Inception *, có thể được đặt tên do một trích dẫn từ bộ phim *Inception* (“Chúng ta cần phải đi sâu hơn”), đã đưa ra một meme virus. 

![Structure of the Inception block.](../img/inception.svg)
:label:`fig_inception`

Như được mô tả trong :numref:`fig_inception`, khối khởi đầu bao gồm bốn con đường song song. Ba đường dẫn đầu tiên sử dụng các lớp phức tạp với kích thước cửa sổ $1\times 1$, $3\times 3$ và $5\times 5$ để trích xuất thông tin từ các kích thước không gian khác nhau. Hai đường dẫn giữa thực hiện sự phức tạp $1\times 1$ trên đầu vào để giảm số lượng kênh, giảm độ phức tạp của mô hình. Đường dẫn thứ tư sử dụng một lớp tổng hợp tối đa $3\times 3$, tiếp theo là một lớp ghép $1\times 1$ để thay đổi số kênh. Bốn đường dẫn đều sử dụng đệm thích hợp để cung cấp cho đầu vào và đầu ra cùng chiều cao và chiều rộng. Cuối cùng, các đầu ra dọc theo mỗi đường dẫn được nối dọc theo kích thước kênh và bao gồm đầu ra của khối. Các siêu tham số được điều chỉnh chung của khối Inception là số kênh đầu ra trên mỗi lớp.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

class Inception(nn.Block):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Path 1 is a single 1 x 1 convolutional layer
        self.p1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        # convolutional layer
        self.p2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.p2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1,
                              activation='relu')
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        # convolutional layer
        self.p3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
        self.p3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2,
                              activation='relu')
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        # convolutional layer
        self.p4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.p4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')

    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # Concatenate the outputs on the channel dimension
        return np.concatenate((p1, p2, p3, p4), axis=1)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

class Inception(nn.Module):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Path 1 is a single 1 x 1 convolutional layer
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        # convolutional layer
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        # convolutional layer
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        # convolutional layer
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # Concatenate the outputs on the channel dimension
        return torch.cat((p1, p2, p3, p4), dim=1)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

class Inception(tf.keras.Model):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        # Path 1 is a single 1 x 1 convolutional layer
        self.p1_1 = tf.keras.layers.Conv2D(c1, 1, activation='relu')
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        # convolutional layer
        self.p2_1 = tf.keras.layers.Conv2D(c2[0], 1, activation='relu')
        self.p2_2 = tf.keras.layers.Conv2D(c2[1], 3, padding='same',
                                           activation='relu')
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        # convolutional layer
        self.p3_1 = tf.keras.layers.Conv2D(c3[0], 1, activation='relu')
        self.p3_2 = tf.keras.layers.Conv2D(c3[1], 5, padding='same',
                                           activation='relu')
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        # convolutional layer
        self.p4_1 = tf.keras.layers.MaxPool2D(3, 1, padding='same')
        self.p4_2 = tf.keras.layers.Conv2D(c4, 1, activation='relu')


    def call(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # Concatenate the outputs on the channel dimension
        return tf.keras.layers.Concatenate()([p1, p2, p3, p4])
```

Để đạt được một số trực giác về lý do tại sao mạng này hoạt động rất tốt, hãy xem xét sự kết hợp của các bộ lọc. Họ khám phá hình ảnh trong một loạt các kích cỡ bộ lọc. Điều này có nghĩa là các chi tiết ở các phạm vi khác nhau có thể được nhận dạng hiệu quả bởi các bộ lọc có kích thước khác nhau. Đồng thời, chúng ta có thể phân bổ lượng tham số khác nhau cho các bộ lọc khác nhau. 

## [**Mô hình GoogLeNet**]

Như thể hiện trong :numref:`fig_inception_full`, GoogLeNet sử dụng một chồng tổng cộng 9 khối khởi đầu và tổng hợp trung bình toàn cầu để tạo ra ước tính của nó. Tổng hợp tối đa giữa các khối khởi đầu làm giảm kích thước. mô-đun đầu tiên tương tự như AlexNet và LeNet. Ngăn xếp các khối được kế thừa từ VGG và tổng hợp trung bình toàn cầu tránh được một chồng các lớp được kết nối hoàn toàn ở cuối. 

![The GoogLeNet architecture.](../img/inception-full.svg)
:label:`fig_inception_full`

Bây giờ chúng ta có thể thực hiện GoogLeNet từng mảnh. mô-đun đầu tiên sử dụng lớp ghép $7\times 7$ 64 kênh.

```{.python .input}
b1 = nn.Sequential()
b1.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b1():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 7, strides=2, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

mô-đun thứ hai sử dụng hai lớp phức tạp: thứ nhất, một lớp ghép $1\times 1$ 64 kênh, sau đó là một lớp ghép $3\times 3$ tăng gấp ba lần số kênh. Điều này tương ứng với đường dẫn thứ hai trong khối Inception.

```{.python .input}
b2 = nn.Sequential()
b2.add(nn.Conv2D(64, kernel_size=1, activation='relu'),
       nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b2():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 1, activation='relu'),
        tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

mô-đun thứ ba kết nối hai khối Inception hoàn chỉnh trong chuỗi. Số kênh đầu ra của khối Inception đầu tiên là $64+128+32+32=256$ và tỷ lệ số kênh đầu ra giữa bốn đường dẫn là $64:128:32:32=2:4:1:1$. Các đường dẫn thứ hai và thứ ba đầu tiên làm giảm số lượng kênh đầu vào xuống $96/192=1/2$ và $16/192=1/12$, và sau đó kết nối lớp phức tạp thứ hai. Số lượng kênh đầu ra của khối Inception thứ hai được tăng lên $128+192+96+64=480$ và tỷ lệ số kênh đầu ra trong bốn đường dẫn là $128:192:96:64 = 4:6:3:2$. Các đường dẫn thứ hai và thứ ba đầu tiên làm giảm số lượng kênh đầu vào xuống $128/256=1/2$ và $32/256=1/8$, tương ứng.

```{.python .input}
b3 = nn.Sequential()
b3.add(Inception(64, (96, 128), (16, 32), 32),
       Inception(128, (128, 192), (32, 96), 64),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b3():
    return tf.keras.models.Sequential([
        Inception(64, (96, 128), (16, 32), 32),
        Inception(128, (128, 192), (32, 96), 64),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

mô-đun thứ tư phức tạp hơn. Nó kết nối năm khối Inception trong loạt, và họ có $192+208+48+64=512$, $160+224+64+64=512$, $128+256+64+64=512$, $112+288+64+64=528$, và $256+320+128+128=832$ kênh đầu ra, tương ứng. Số lượng kênh được gán cho các đường dẫn này tương tự như trong mô-đun thứ ba: đường dẫn thứ hai với lớp cuộn $3\times 3$ xuất ra số lượng kênh lớn nhất, tiếp theo là đường dẫn đầu tiên chỉ có lớp ghép $1\times 1$, đường dẫn thứ ba với lớp biên độ $5\times 5$ và con đường thứ tư với lớp tổng hợp tối đa $3\times 3$. Các đường dẫn thứ hai và thứ ba trước tiên sẽ giảm số lượng kênh theo tỷ lệ. Các tỷ lệ này hơi khác nhau trong các khối Inception khác nhau.

```{.python .input}
b4 = nn.Sequential()
b4.add(Inception(192, (96, 208), (16, 48), 64),
       Inception(160, (112, 224), (24, 64), 64),
       Inception(128, (128, 256), (24, 64), 64),
       Inception(112, (144, 288), (32, 64), 64),
       Inception(256, (160, 320), (32, 128), 128),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b4():
    return tf.keras.Sequential([
        Inception(192, (96, 208), (16, 48), 64),
        Inception(160, (112, 224), (24, 64), 64),
        Inception(128, (128, 256), (24, 64), 64),
        Inception(112, (144, 288), (32, 64), 64),
        Inception(256, (160, 320), (32, 128), 128),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

mô-đun thứ năm có hai khối Inception với các kênh đầu ra $256+320+128+128=832$ và $384+384+128+128=1024$. Số lượng kênh được gán cho mỗi đường dẫn giống như trong các mô-đun thứ ba và thứ tư, nhưng khác nhau về các giá trị cụ thể. Cần lưu ý rằng khối thứ năm được theo sau bởi lớp đầu ra. Khối này sử dụng lớp tổng hợp trung bình toàn cầu để thay đổi chiều cao và chiều rộng của mỗi kênh thành 1, giống như trong Nin. Cuối cùng, chúng ta biến đầu ra thành một mảng hai chiều theo sau là một lớp kết nối hoàn toàn có số lượng đầu ra là số lớp nhãn.

```{.python .input}
b5 = nn.Sequential()
b5.add(Inception(256, (160, 320), (32, 128), 128),
       Inception(384, (192, 384), (48, 128), 128),
       nn.GlobalAvgPool2D())

net = nn.Sequential()
net.add(b1, b2, b3, b4, b5, nn.Dense(10))
```

```{.python .input}
#@tab pytorch
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
```

```{.python .input}
#@tab tensorflow
def b5():
    return tf.keras.Sequential([
        Inception(256, (160, 320), (32, 128), 128),
        Inception(384, (192, 384), (48, 128), 128),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Flatten()
    ])
# Recall that this has to be a function that will be passed to
# `d2l.train_ch6()` so that model building/compiling need to be within
# `strategy.scope()` in order to utilize the CPU/GPU devices that we have
def net():
    return tf.keras.Sequential([b1(), b2(), b3(), b4(), b5(),
                                tf.keras.layers.Dense(10)])
```

Mô hình GoogLeNet phức tạp về mặt tính toán, vì vậy không dễ dàng để sửa đổi số lượng kênh như trong VGG. [**Để có thời gian đào tạo hợp lý về Fashion-MNIST, chúng tôi giảm chiều cao và chiều rộng đầu vào từ 224 xuống 96.**] Điều này đơn giản hóa việc tính toán. Những thay đổi về hình dạng của đầu ra giữa các mô-đun khác nhau được thể hiện dưới đây.

```{.python .input}
X = np.random.uniform(size=(1, 1, 96, 96))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform(shape=(1, 96, 96, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
```

## [**Đào tạo**]

Như trước đây, chúng tôi đào tạo mô hình của mình bằng cách sử dụng bộ dữ liệu Fashion-MNIST. Chúng tôi chuyển đổi nó thành độ phân giải $96 \times 96$ pixel trước khi gọi quy trình đào tạo.

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## Tóm tắt

* Khối Inception tương đương với một mạng con với bốn đường dẫn. Nó trích xuất thông tin song song thông qua các lớp phức tạp của các hình dạng cửa sổ khác nhau và các lớp tổng hợp tối đa. $1 \times 1$ sự covolutions làm giảm kích thước kênh ở mức mỗi pixel. Tổng hợp tối đa làm giảm độ phân giải.
* GoogLeNet kết nối nhiều khối Inception được thiết kế tốt với các lớp khác trong loạt. Tỷ lệ số kênh được gán trong khối Inception thu được thông qua một số lượng lớn các thí nghiệm trên tập dữ liệu ImageNet.
* GoogLeNet, cũng như các phiên bản thành công của nó, là một trong những mô hình hiệu quả nhất trên ImageNet, cung cấp độ chính xác thử nghiệm tương tự với độ phức tạp tính toán thấp hơn.

## Bài tập

1. Có một số lần lặp lại của GoogLeNet. Cố gắng thực hiện và chạy chúng. Một số trong số họ bao gồm những điều sau đây:
    * Thêm một lớp chuẩn hóa hàng loạt :cite:`Ioffe.Szegedy.2015`, như được mô tả sau trong :numref:`sec_batch_norm`.
    * Thực hiện các điều chỉnh đối với khối Inception :cite:`Szegedy.Vanhoucke.Ioffe.ea.2016`.
    * Sử dụng làm mịn nhãn cho mô hình regarization :cite:`Szegedy.Vanhoucke.Ioffe.ea.2016`.
    * Bao gồm nó trong kết nối còn lại :cite:`Szegedy.Ioffe.Vanhoucke.ea.2017`, như được mô tả sau trong :numref:`sec_resnet`.
1. Kích thước hình ảnh tối thiểu để GoogLeNet hoạt động là bao nhiêu?
1. So sánh kích thước tham số mô hình của AlexNet, VGG và Nin với GoogLeNet. Làm thế nào để hai kiến trúc mạng sau làm giảm đáng kể kích thước tham số mô hình?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/81)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/82)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/316)
:end_tab:
