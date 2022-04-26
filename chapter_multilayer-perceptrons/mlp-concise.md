# Thực hiện ngắn gọn của Multilayer Perceptrons
:label:`sec_mlp_concise`

Như bạn có thể mong đợi, bằng cách (** dựa vào các API cấp cao, chúng tôi có thể triển khai MLP thậm chí còn ngắn gọn hơn.**)

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## Mô hình

So với việc thực hiện súc tích của chúng tôi về triển khai hồi quy softmax (:numref:`sec_softmax_concise`), sự khác biệt duy nhất là chúng tôi thêm
*hai* lớp được kết nối hoàn toàn
(trước đây, chúng tôi đã thêm * một*). Đầu tiên là [** lớp ẩn của chúng tôi**], (** chứa 256 đơn vị ẩn và áp dụng chức năng kích hoạt ReLU**). Thứ hai là lớp đầu ra của chúng tôi.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10)])
```

[**Vòng đào tạo**] chính xác giống như khi chúng tôi thực hiện hồi quy softmax. Mô đun này cho phép chúng ta tách các vấn đề liên quan đến kiến trúc mô hình khỏi các cân nhắc trực giao.

```{.python .input}
batch_size, lr, num_epochs = 256, 0.1, 10
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
```

```{.python .input}
#@tab pytorch
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)
```

```{.python .input}
#@tab tensorflow
batch_size, lr, num_epochs = 256, 0.1, 10
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
trainer = tf.keras.optimizers.SGD(learning_rate=lr)
```

```{.python .input}
#@tab all
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## Tóm tắt

* Sử dụng API cấp cao, chúng ta có thể triển khai MLP chính xác hơn nhiều.
* Đối với bài toán phân loại tương tự, việc thực hiện một MLP giống như của hồi quy softmax ngoại trừ các lớp ẩn bổ sung có hàm kích hoạt.

## Bài tập

1. Hãy thử thêm các số lớp ẩn khác nhau (bạn cũng có thể sửa đổi tốc độ học tập). Cài đặt nào hoạt động tốt nhất?
1. Hãy thử các chức năng kích hoạt khác nhau. Cái nào hoạt động tốt nhất?
1. Hãy thử các sơ đồ khác nhau để khởi tạo các trọng lượng. Phương pháp nào hoạt động tốt nhất?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/94)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/95)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/262)
:end_tab:
