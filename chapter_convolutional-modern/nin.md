# Mạng trong mạng (Nin)
:label:`sec_nin`

LeNet, AlexNet và VGG đều chia sẻ một mẫu thiết kế chung: trích xuất các tính năng khai thác cấu trúc *không gian* thông qua một chuỗi các lớp phức tạp và tập hợp và sau đó xử lý các biểu diễn thông qua các lớp được kết nối hoàn toàn. Những cải tiến trên LeNet của AlexNet và VGG chủ yếu nằm ở cách các mạng sau này mở rộng và làm sâu sắc thêm hai mô-đun này. Ngoài ra, người ta có thể tưởng tượng sử dụng các lớp được kết nối hoàn toàn trước đó trong quá trình này. Tuy nhiên, việc sử dụng bất cẩn các lớp dày đặc có thể từ bỏ cấu trúc không gian của biểu diễn hoàn toàn,
*mạng trong mạng* (* NiN*) khối cung cấp một giải pháp thay thế.
Chúng được đề xuất dựa trên một cái nhìn sâu sắc rất đơn giản: sử dụng MLP trên các kênh cho mỗi pixel riêng biệt :cite:`Lin.Chen.Yan.2013`. 

## (** NiN Blocks**)

Nhớ lại rằng các đầu vào và đầu ra của các lớp phức tạp bao gồm các hàng chục bốn chiều với các trục tương ứng với ví dụ, kênh, chiều cao và chiều rộng. Cũng nhớ lại rằng các đầu vào và đầu ra của các lớp được kết nối hoàn toàn thường là các hàng chục hai chiều tương ứng với ví dụ và tính năng. Ý tưởng đằng sau Nin là áp dụng một lớp được kết nối hoàn toàn tại mỗi vị trí pixel (cho mỗi chiều cao và chiều rộng). Nếu chúng ta buộc trọng lượng trên mỗi vị trí không gian, chúng ta có thể nghĩ đây là một lớp ghép $1\times 1$ (như được mô tả trong :numref:`sec_channels`) hoặc như một lớp được kết nối hoàn toàn hoạt động độc lập trên mỗi vị trí pixel. Một cách khác để xem điều này là nghĩ về từng phần tử trong chiều không gian (chiều cao và chiều rộng) tương đương với một ví dụ và một kênh tương đương với một đối tượng. 

:numref:`fig_nin` minh họa sự khác biệt về cấu trúc chính giữa VGG và Nin, và các khối của chúng. Khối Nin bao gồm một lớp phức tạp tiếp theo là hai lớp ghép $1\times 1$ hoạt động như các lớp được kết nối hoàn toàn trên mỗi pixel với các kích hoạt ReLU. Hình dạng cửa sổ phức tạp của lớp đầu tiên thường được đặt bởi người dùng. Các hình dạng cửa sổ tiếp theo được cố định thành $1 \times 1$. 

![Comparing architectures of VGG and NiN, and their blocks.](../img/nin.svg)
:width:`600px`
:label:`fig_nin`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernel_size, strides, padding,
                      activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
    return blk
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def nin_block(num_channels, kernel_size, strides, padding):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(num_channels, kernel_size, strides=strides,
                               padding=padding, activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                               activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                               activation='relu')])
```

## [**Mô hình NiN**]

Mạng Nin ban đầu được đề xuất ngay sau AlexNet và rõ ràng rút ra một số cảm hứng. Nin sử dụng các lớp phức tạp với hình dạng cửa sổ $11\times 11$, $5\times 5$, và $3\times 3$, và các số kênh đầu ra tương ứng giống như trong AlexNet. Mỗi khối Nin được theo sau bởi một lớp tổng hợp tối đa với một sải chân là 2 và một hình dạng cửa sổ là $3\times 3$. 

Một điểm khác biệt đáng kể giữa Nin và AlexNet là Nin tránh hoàn toàn các lớp được kết nối hoàn toàn. Thay vào đó, Nin sử dụng một khối Nin với một số kênh đầu ra bằng số lớp nhãn, tiếp theo là một lớp tổng hợp trung bình *global*, mang lại một vectơ của logits. Một ưu điểm của thiết kế của Nin là nó làm giảm đáng kể số lượng các thông số mô hình cần thiết. Tuy nhiên, trong thực tế, thiết kế này đôi khi đòi hỏi thời gian đào tạo mô hình tăng lên.

```{.python .input}
net = nn.Sequential()
net.add(nin_block(96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Dropout(0.5),
        # There are 10 label classes
        nin_block(10, kernel_size=3, strides=1, padding=1),
        # The global average pooling layer automatically sets the window shape
        # to the height and width of the input
        nn.GlobalAvgPool2D(),
        # Transform the four-dimensional output into two-dimensional output
        # with a shape of (batch size, 10)
        nn.Flatten())
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # There are 10 label classes
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # Transform the four-dimensional output into two-dimensional output with a
    # shape of (batch size, 10)
    nn.Flatten())
```

```{.python .input}
#@tab tensorflow
def net():
    return tf.keras.models.Sequential([
        nin_block(96, kernel_size=11, strides=4, padding='valid'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        nin_block(256, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        nin_block(384, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Dropout(0.5),
        # There are 10 label classes
        nin_block(10, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Reshape((1, 1, 10)),
        # Transform the four-dimensional output into two-dimensional output
        # with a shape of (batch size, 10)
        tf.keras.layers.Flatten(),
        ])
```

Chúng ta tạo ra một ví dụ dữ liệu để xem [** hình dạng đầu ra của mỗi block**].

```{.python .input}
X = np.random.uniform(size=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

## [**Đào tạo**]

Như trước đây chúng ta sử dụng Fashion-MNIST để đào tạo mô hình. Đào tạo của Nin tương tự như cho AlexNet và VGG.

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## Tóm tắt

* Nin sử dụng các khối bao gồm một lớp phức tạp và nhiều lớp ghép $1\times 1$. Điều này có thể được sử dụng trong ngăn xếp phức tạp để cho phép tính phi tuyến trên mỗi pixel hơn.
* Nin loại bỏ các lớp được kết nối hoàn toàn và thay thế chúng bằng tổng hợp trung bình toàn cầu (tức là tổng hợp tất cả các vị trí) sau khi giảm số lượng kênh xuống số lượng đầu ra mong muốn (ví dụ: 10 cho Fashion-MNIST).
* Loại bỏ các lớp được kết nối hoàn toàn làm giảm quá mức. Nin có ít tham số hơn đáng kể.
* Thiết kế Nin ảnh hưởng đến nhiều thiết kế CNN tiếp theo.

## Bài tập

1. Điều chỉnh các siêu tham số để cải thiện độ chính xác phân loại.
1. Tại sao có hai lớp ghép $1\times 1$ trong khối Nin? Loại bỏ một trong số chúng, sau đó quan sát và phân tích các hiện tượng thử nghiệm.
1. Tính toán mức sử dụng tài nguyên cho Nin.
    1. Số lượng tham số là bao nhiêu?
    1. Số lượng tính toán là bao nhiêu?
    1. Dung lượng bộ nhớ cần thiết trong quá trình đào tạo là bao nhiêu?
    1. Dung lượng bộ nhớ cần thiết trong quá trình dự đoán là bao nhiêu?
1. Những vấn đề có thể xảy ra với việc giảm đại diện $384 \times 5 \times 5$ thành đại diện $10 \times 5 \times 5$ trong một bước là gì?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/79)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/80)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/332)
:end_tab:
