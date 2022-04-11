# Thực hiện ngắn gọn các mạng nơ-ron tái phát
:label:`sec_rnn-concise`

Trong khi :numref:`sec_rnn_scratch` được hướng dẫn để xem RNN được thực hiện như thế nào, điều này không thuận tiện hay nhanh chóng. Phần này sẽ chỉ ra cách triển khai cùng một mô hình ngôn ngữ hiệu quả hơn bằng cách sử dụng các hàm được cung cấp bởi các API cấp cao của một khuôn khổ học sâu. Chúng tôi bắt đầu như trước bằng cách đọc tập dữ liệu máy thời gian.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn, rnn
npx.set_np()

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

## [**Định nghĩa mẫu**]

API cấp cao cung cấp triển khai các mạng thần kinh định kỳ. Chúng tôi xây dựng lớp mạng thần kinh tái phát `rnn_layer` với một lớp ẩn duy nhất và 256 đơn vị ẩn. Trên thực tế, chúng tôi thậm chí còn chưa thảo luận về ý nghĩa của việc có nhiều lớp — điều này sẽ xảy ra trong :numref:`sec_deep_rnn`. Hiện tại, đủ để nói rằng nhiều lớp chỉ đơn giản là lên đến đầu ra của một lớp RNN được sử dụng làm đầu vào cho lớp tiếp theo của RNN.

```{.python .input}
num_hiddens = 256
rnn_layer = rnn.RNN(num_hiddens)
rnn_layer.initialize()
```

```{.python .input}
#@tab pytorch
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)
```

```{.python .input}
#@tab tensorflow
num_hiddens = 256
rnn_cell = tf.keras.layers.SimpleRNNCell(num_hiddens,
    kernel_initializer='glorot_uniform')
rnn_layer = tf.keras.layers.RNN(rnn_cell, time_major=True,
    return_sequences=True, return_state=True)
```

:begin_tab:`mxnet`
Khởi tạo trạng thái ẩn là đơn giản. Chúng tôi gọi hàm thành viên `begin_state`. Điều này trả về một danh sách (`state`) chứa một trạng thái ẩn ban đầu cho mỗi ví dụ trong minibatch, có hình dạng là (số lớp ẩn, kích thước lô, số đơn vị ẩn). Đối với một số mô hình được giới thiệu sau (ví dụ: bộ nhớ ngắn hạn dài), một danh sách như vậy cũng chứa các thông tin khác.
:end_tab:

:begin_tab:`pytorch`
Chúng tôi (** sử dụng một tensor để khởi tạo trạng thái ẩn**), có hình dạng là (số lớp ẩn, kích thước lô, số đơn vị ẩn).
:end_tab:

```{.python .input}
state = rnn_layer.begin_state(batch_size=batch_size)
len(state), state[0].shape
```

```{.python .input}
#@tab pytorch
state = torch.zeros((1, batch_size, num_hiddens))
state.shape
```

```{.python .input}
#@tab tensorflow
state = rnn_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
state.shape
```

[**Với trạng thái ẩn và đầu vào, chúng ta có thể tính toán đầu ra với trạng thái ẩn cập nhật.**] Cần nhấn mạnh rằng “đầu ra” (`Y`) của `rnn_layer` không * không* liên quan đến tính toán các lớp đầu ra: nó đề cập đến trạng thái ẩn ở bước thời gian * mỗi* và chúng có thể được sử dụng làm đầu vào cho lớp đầu ra tiếp theo.

:begin_tab:`mxnet`
Bên cạnh đó, trạng thái ẩn được cập nhật (`state_new`) trả về bởi `rnn_layer` đề cập đến trạng thái ẩn ở bước thời gian *last* của minibatch. Nó có thể được sử dụng để khởi tạo trạng thái ẩn cho minibatch tiếp theo trong một kỷ nguyên trong phân vùng tuần tự. Đối với nhiều lớp ẩn, trạng thái ẩn của mỗi lớp sẽ được lưu trữ trong biến này (`state_new`). Đối với một số mô hình được giới thiệu sau (ví dụ: bộ nhớ ngắn hạn dài), biến này cũng chứa các thông tin khác.
:end_tab:

```{.python .input}
X = np.random.uniform(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, len(state_new), state_new[0].shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, state_new.shape
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, len(state_new), state_new[0].shape
```

Tương tự như :numref:`sec_rnn_scratch`, [**chúng tôi định nghĩa một lớp `RNNModel` cho một mô hình RNN hoàn chỉnh**] Lưu ý rằng `rnn_layer` chỉ chứa các lớp tái phát ẩn, chúng ta cần tạo một lớp đầu ra riêng biệt.

```{.python .input}
#@save
class RNNModel(nn.Block):
    """The RNN model."""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        X = npx.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        # The fully-connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `vocab_size`).
        output = self.dense(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```

```{.python .input}
#@tab pytorch
#@save
class RNNModel(nn.Module):
    """The RNN model."""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # If the RNN is bidirectional (to be introduced later),
        # `num_directions` should be 2, else it should be 1.
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # The fully connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `vocab_size`).
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # `nn.GRU` takes a tensor as hidden state
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # `nn.LSTM` takes a tuple of hidden states
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))
```

```{.python .input}
#@tab tensorflow
#@save
class RNNModel(tf.keras.layers.Layer):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, state):
        X = tf.one_hot(tf.transpose(inputs), self.vocab_size)
        # Later RNN like `tf.keras.layers.LSTMCell` return more than two values
        Y, *state = self.rnn(X, state)
        output = self.dense(tf.reshape(Y, (-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.cell.get_initial_state(*args, **kwargs)
```

## Đào tạo và dự đoán

Trước khi đào tạo mô hình, chúng ta hãy [** đưa ra dự đoán với một mô hình có trọng lượng ngẫu nhiên.**]

```{.python .input}
device = d2l.try_gpu()
net = RNNModel(rnn_layer, len(vocab))
net.initialize(force_reinit=True, ctx=device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)
```

```{.python .input}
#@tab pytorch
device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)
```

```{.python .input}
#@tab tensorflow
device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)
with strategy.scope():
    net = RNNModel(rnn_layer, vocab_size=len(vocab))

d2l.predict_ch8('time traveller', 10, net, vocab)
```

Như là khá rõ ràng, mô hình này hoàn toàn không hoạt động. Tiếp theo, chúng tôi gọi `train_ch8` với các siêu tham số tương tự được xác định trong :numref:`sec_rnn_scratch` và [** đào tạo mô hình của chúng tôi với APIs** cấp cao**].

```{.python .input}
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab pytorch
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, strategy)
```

So với phần cuối, mô hình này đạt được sự bối rối tương đương, mặc dù trong một khoảng thời gian ngắn hơn, do mã được tối ưu hóa hơn bởi các API cấp cao của khung học sâu. 

## Tóm tắt

* API cấp cao của khung học sâu cung cấp một triển khai của lớp RNN.
* Lớp RNN của API cấp cao trả về một đầu ra và trạng thái ẩn cập nhật, trong đó đầu ra không liên quan đến tính toán lớp đầu ra.
* Sử dụng API cấp cao dẫn đến đào tạo RNN nhanh hơn so với sử dụng triển khai từ đầu.

## Bài tập

1. Bạn có thể làm cho mô hình RNN overfit bằng cách sử dụng các API cấp cao không?
1. Điều gì sẽ xảy ra nếu bạn tăng số lượng các lớp ẩn trong mô hình RNN? Bạn có thể làm cho mô hình hoạt động?
1. Thực hiện mô hình autoregressive của :numref:`sec_sequence` bằng cách sử dụng một RNN.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/335)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1053)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2211)
:end_tab:
