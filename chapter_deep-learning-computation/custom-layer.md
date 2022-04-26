# Layers tùy chỉnh

Một yếu tố đằng sau sự thành công của deep learning là sự sẵn có của một loạt các lớp có thể được sáng tác theo những cách sáng tạo để thiết kế kiến trúc phù hợp với nhiều nhiệm vụ khác nhau. Ví dụ, các nhà nghiên cứu đã phát minh ra các lớp đặc biệt để xử lý hình ảnh, văn bản, lặp lại dữ liệu tuần tự và thực hiện lập trình động. Sớm hay muộn, bạn sẽ gặp hoặc phát minh ra một lớp chưa tồn tại trong khuôn khổ học sâu. Trong những trường hợp này, bạn phải xây dựng một lớp tùy chỉnh. Trong phần này, chúng tôi chỉ cho bạn như thế nào. 

## (** Các lớp không có tham số**)

Để bắt đầu, chúng tôi xây dựng một lớp tùy chỉnh không có bất kỳ tham số nào của riêng nó. Điều này sẽ trông quen thuộc nếu bạn nhớ lại giới thiệu của chúng tôi để chặn trong :numref:`sec_model_construction`. Lớp `CenteredLayer` sau chỉ đơn giản là trừ trung bình từ đầu vào của nó. Để xây dựng nó, chúng ta chỉ cần kế thừa từ lớp lớp cơ sở và thực hiện hàm tuyên truyền chuyển tiếp.

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs)
```

Hãy để chúng tôi xác minh rằng lớp của chúng tôi hoạt động như dự định bằng cách cung cấp một số dữ liệu thông qua nó.

```{.python .input}
layer = CenteredLayer()
layer(np.array([1, 2, 3, 4, 5]))
```

```{.python .input}
#@tab pytorch
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
```

```{.python .input}
#@tab tensorflow
layer = CenteredLayer()
layer(tf.constant([1, 2, 3, 4, 5]))
```

Bây giờ chúng ta có thể [** kết hợp layer của chúng ta như một component trong việc xây dựng các mô hình phức tạp hơn.**]

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
```

```{.python .input}
#@tab tensorflow
net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()])
```

Là một kiểm tra thêm sự tỉnh táo, chúng tôi có thể gửi dữ liệu ngẫu nhiên thông qua mạng và kiểm tra xem trung bình có thực tế 0 không. Bởi vì chúng ta đang xử lý các số điểm nổi, chúng ta vẫn có thể thấy một số nonzero rất nhỏ do lượng tử hóa.

```{.python .input}
Y = net(np.random.uniform(size=(4, 8)))
Y.mean()
```

```{.python .input}
#@tab pytorch
Y = net(torch.rand(4, 8))
Y.mean()
```

```{.python .input}
#@tab tensorflow
Y = net(tf.random.uniform((4, 8)))
tf.reduce_mean(Y)
```

## [**Các lớp có tham số**]

Bây giờ chúng ta đã biết cách xác định các lớp đơn giản, chúng ta hãy chuyển sang xác định các lớp với các tham số có thể được điều chỉnh thông qua đào tạo. Chúng ta có thể sử dụng các chức năng tích hợp để tạo các tham số, cung cấp một số chức năng dọn phòng cơ bản. Đặc biệt, họ chi phối quyền truy cập, khởi tạo, chia sẻ, lưu và tải các tham số mô hình. Bằng cách này, trong số các lợi ích khác, chúng tôi sẽ không cần phải viết các thói quen serialization tùy chỉnh cho mọi lớp tùy chỉnh. 

Bây giờ chúng ta hãy thực hiện phiên bản riêng của chúng ta của lớp được kết nối hoàn toàn. Nhớ lại rằng lớp này yêu cầu hai tham số, một để đại diện cho trọng lượng và một cho sự thiên vị. Trong triển khai này, chúng tôi nướng trong kích hoạt ReLU dưới dạng mặc định. Lớp này yêu cầu nhập đối số: `in_units` và `units`, biểu thị số lượng đầu vào và đầu ra, tương ứng.

```{.python .input}
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = np.dot(x, self.weight.data(ctx=x.ctx)) + self.bias.data(
            ctx=x.ctx)
        return npx.relu(linear)
```

```{.python .input}
#@tab pytorch
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

```{.python .input}
#@tab tensorflow
class MyDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, X_shape):
        self.weight = self.add_weight(name='weight',
            shape=[X_shape[-1], self.units],
            initializer=tf.random_normal_initializer())
        self.bias = self.add_weight(
            name='bias', shape=[self.units],
            initializer=tf.zeros_initializer())

    def call(self, X):
        linear = tf.matmul(X, self.weight) + self.bias
        return tf.nn.relu(linear)
```

:begin_tab:`mxnet, tensorflow`
Tiếp theo, chúng tôi khởi tạo lớp `MyDense` và truy cập các tham số mô hình của nó.
:end_tab:

:begin_tab:`pytorch`
Tiếp theo, chúng tôi khởi tạo lớp `MyLinear` và truy cập các tham số mô hình của nó.
:end_tab:

```{.python .input}
dense = MyDense(units=3, in_units=5)
dense.params
```

```{.python .input}
#@tab pytorch
linear = MyLinear(5, 3)
linear.weight
```

```{.python .input}
#@tab tensorflow
dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
dense.get_weights()
```

Chúng ta có thể [** trực tiếp thực hiện các phép tính tuyên truyền chuyển tiếp bằng cách sử dụng các lớp tùy chỉnh**]

```{.python .input}
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

```{.python .input}
#@tab pytorch
linear(torch.rand(2, 5))
```

```{.python .input}
#@tab tensorflow
dense(tf.random.uniform((2, 5)))
```

Chúng ta cũng có thể (** xây dựng các mô hình bằng cách sử dụng lớp tùy chỉnh**) Khi chúng ta có rằng chúng ta có thể sử dụng nó giống như lớp được kết nối hoàn toàn tích hợp.

```{.python .input}
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(np.random.uniform(size=(2, 64)))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
net(tf.random.uniform((2, 64)))
```

## Tóm tắt

* Chúng ta có thể thiết kế các lớp tùy chỉnh thông qua lớp lớp cơ bản. Điều này cho phép chúng ta xác định các lớp mới linh hoạt hoạt hoạt khác với bất kỳ lớp hiện có nào trong thư viện.
* Sau khi được xác định, các lớp tùy chỉnh có thể được gọi trong bối cảnh và kiến trúc tùy ý.
* Các lớp có thể có các tham số cục bộ, có thể được tạo thông qua các chức năng tích hợp.

## Bài tập

1. Thiết kế một lớp lấy đầu vào và tính toán giảm tensor, tức là nó trả về $y_k = \sum_{i, j} W_{ijk} x_i x_j$.
1. Thiết kế một lớp trả về một nửa hàng đầu của hệ số Fourier của dữ liệu.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/58)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/59)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/279)
:end_tab:
