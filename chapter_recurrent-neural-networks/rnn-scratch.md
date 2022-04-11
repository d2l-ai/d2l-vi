# Thực hiện các mạng nơ-ron tái phát từ đầu
:label:`sec_rnn_scratch`

Trong phần này, chúng tôi sẽ triển khai RNN từ đầu cho mô hình ngôn ngữ cấp ký tự, theo mô tả của chúng tôi trong :numref:`sec_rnn`. Một mô hình như vậy sẽ được đào tạo trên H Gwells' * The Time Machine*. Như trước đây, chúng ta bắt đầu bằng cách đọc tập dữ liệu đầu tiên, được giới thiệu trong :numref:`sec_language_model`.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input}
#@tab all
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab tensorflow
train_random_iter, vocab_random_iter = d2l.load_data_time_machine(
    batch_size, num_steps, use_random_iter=True)
```

## [**Mã hóa Một-Hot**]

Nhớ lại rằng mỗi mã thông báo được biểu diễn dưới dạng chỉ số số trong `train_iter`. Việc cung cấp các chỉ số này trực tiếp vào mạng thần kinh có thể khiến bạn khó học. Chúng ta thường đại diện cho mỗi token như một vector tính năng biểu cảm hơn. Đại diện dễ nhất được gọi là * mã hóa một nóng*, được giới thiệu trong :numref:`subsec_classification-problem`. 

Tóm lại, chúng ta ánh xạ mỗi chỉ mục đến một vector đơn vị khác nhau: giả sử rằng số lượng token khác nhau trong từ vựng là $N$ (`len(vocab)`) và các chỉ số token dao động từ $0$ đến $N-1$. Nếu chỉ số của một mã thông báo là số nguyên $i$, thì ta tạo một vectơ của tất cả 0s với độ dài $N$ và đặt phần tử ở vị trí $i$ thành 1. Vector này là vectơ một nóng của mã thông báo gốc. Các vectơ một nóng với chỉ số 0 và 2 được hiển thị bên dưới.

```{.python .input}
npx.one_hot(np.array([0, 2]), len(vocab))
```

```{.python .input}
#@tab pytorch
F.one_hot(torch.tensor([0, 2]), len(vocab))
```

```{.python .input}
#@tab tensorflow
tf.one_hot(tf.constant([0, 2]), len(vocab))
```

(**Hình dạng của minibatch**) mà chúng tôi lấy mẫu mỗi lần (** là (kích thước lô, số bước thời gian). Hàm `one_hot` biến một minibatch như vậy thành một tensor ba chiều với chiều cuối cùng bằng với kích thước từ vựng (`len(vocab)`) .**) Chúng ta thường chuyển đổi đầu vào để chúng ta sẽ có được một đầu ra của hình dạng (số bước thời gian, kích thước lô, kích thước từ vựng). Điều này sẽ cho phép chúng tôi vòng lặp thuận tiện hơn thông qua kích thước ngoài cùng để cập nhật các trạng thái ẩn của một minibatch, từng bước từng bước.

```{.python .input}
X = d2l.reshape(d2l.arange(10), (2, 5))
npx.one_hot(X.T, 28).shape
```

```{.python .input}
#@tab pytorch
X = d2l.reshape(d2l.arange(10), (2, 5))
F.one_hot(X.T, 28).shape
```

```{.python .input}
#@tab tensorflow
X = d2l.reshape(d2l.arange(10), (2, 5))
tf.one_hot(tf.transpose(X), 28).shape
```

## Khởi tạo các tham số mô hình

Tiếp theo, chúng ta [** khởi tạo các tham số mô hình cho mô hình RNN**]. Số lượng các đơn vị ẩn `num_hiddens` là một siêu tham số có thể điều chỉnh. Khi đào tạo mô hình ngôn ngữ, các đầu vào và đầu ra là từ cùng một từ vựng. Do đó, chúng có cùng chiều, tương đương với kích thước từ vựng.

```{.python .input}
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape, ctx=device)

    # Hidden layer parameters
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = d2l.zeros(num_hiddens, ctx=device)
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, ctx=device)
    # Attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params
```

```{.python .input}
#@tab pytorch
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # Hidden layer parameters
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = d2l.zeros(num_hiddens, device=device)
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

```{.python .input}
#@tab tensorflow
def get_params(vocab_size, num_hiddens):
    num_inputs = num_outputs = vocab_size
    
    def normal(shape):
        return d2l.normal(shape=shape,stddev=0.01,mean=0,dtype=tf.float32)

    # Hidden layer parameters
    W_xh = tf.Variable(normal((num_inputs, num_hiddens)), dtype=tf.float32)
    W_hh = tf.Variable(normal((num_hiddens, num_hiddens)), dtype=tf.float32)
    b_h = tf.Variable(d2l.zeros(num_hiddens), dtype=tf.float32)
    # Output layer parameters
    W_hq = tf.Variable(normal((num_hiddens, num_outputs)), dtype=tf.float32)
    b_q = tf.Variable(d2l.zeros(num_outputs), dtype=tf.float32)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    return params
```

## Mô hình RNN

Để định nghĩa một mô hình RNN, trước tiên chúng ta cần [** một hàm `init_rnn_state` để trả về trạng thái ẩn lúc khởi hóa.**] Nó trả về một tensor chứa đầy 0 và với một hình dạng của (batch size, số đơn vị ẩn). Sử dụng các tuples làm cho nó dễ dàng hơn để xử lý các tình huống mà trạng thái ẩn chứa nhiều biến, mà chúng ta sẽ gặp phải trong các phần sau.

```{.python .input}
def init_rnn_state(batch_size, num_hiddens, device):
    return (d2l.zeros((batch_size, num_hiddens), ctx=device), )
```

```{.python .input}
#@tab pytorch
def init_rnn_state(batch_size, num_hiddens, device):
    return (d2l.zeros((batch_size, num_hiddens), device=device), )
```

```{.python .input}
#@tab tensorflow
def init_rnn_state(batch_size, num_hiddens):
    return (d2l.zeros((batch_size, num_hiddens)), )
```

[**Hàm `rnn` sau định nghĩa cách tính trạng thái ẩn và đầu ra tại một bước thời gian**] Lưu ý rằng mô hình RNN vòng qua chiều ngoài cùng của `inputs` để nó cập nhật các trạng thái ẩn `H` của một minibatch, từng bước từng bước thời gian. Bên cạnh đó, chức năng kích hoạt ở đây sử dụng chức năng $\tanh$. Như được mô tả trong :numref:`sec_mlp`, giá trị trung bình của hàm $\tanh$ là 0, khi các phần tử được phân bố đồng đều trên các số thực.

```{.python .input}
def rnn(inputs, state, params):
    # Shape of `inputs`: (`num_steps`, `batch_size`, `vocab_size`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of `X`: (`batch_size`, `vocab_size`)
    for X in inputs:
        H = np.tanh(np.dot(X, W_xh) + np.dot(H, W_hh) + b_h)
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H,)
```

```{.python .input}
#@tab pytorch
def rnn(inputs, state, params):
    # Here `inputs` shape: (`num_steps`, `batch_size`, `vocab_size`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of `X`: (`batch_size`, `vocab_size`)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
```

```{.python .input}
#@tab tensorflow
def rnn(inputs, state, params):
    # Here `inputs` shape: (`num_steps`, `batch_size`, `vocab_size`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of `X`: (`batch_size`, `vocab_size`)
    for X in inputs:
        X = tf.reshape(X,[-1,W_xh.shape[0]])
        H = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(H, W_hh) + b_h)
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return d2l.concat(outputs, axis=0), (H,)
```

Với tất cả các hàm cần thiết được định nghĩa, tiếp theo chúng ta [**create a class để bọc các hàm này và lưu trữ các tham số**] cho một mô hình RNN được thực hiện từ đầu.

```{.python .input}
class RNNModelScratch:  #@save
    """An RNN Model implemented from scratch."""
    def __init__(self, vocab_size, num_hiddens, device, get_params,
                 init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = npx.one_hot(X.T, self.vocab_size)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, ctx):
        return self.init_state(batch_size, self.num_hiddens, ctx)
```

```{.python .input}
#@tab pytorch
class RNNModelScratch: #@save
    """A RNN Model implemented from scratch."""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
```

```{.python .input}
#@tab tensorflow
class RNNModelScratch: #@save
    """A RNN Model implemented from scratch."""
    def __init__(self, vocab_size, num_hiddens,
                 init_state, forward_fn, get_params):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.init_state, self.forward_fn = init_state, forward_fn
        self.trainable_variables = get_params(vocab_size, num_hiddens)

    def __call__(self, X, state):
        X = tf.one_hot(tf.transpose(X), self.vocab_size)
        X = tf.cast(X, tf.float32)
        return self.forward_fn(X, state, self.trainable_variables)

    def begin_state(self, batch_size, *args, **kwargs):
        return self.init_state(batch_size, self.num_hiddens)
```

Hãy để chúng tôi [** kiểm tra xem các đầu ra có hình dạng chính xác**], ví dụ, để đảm bảo rằng chiều của trạng thái ẩn vẫn không thay đổi.

```{.python .input}
#@tab mxnet
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.as_in_context(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape
```

```{.python .input}
#@tab pytorch
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape
```

```{.python .input}
#@tab tensorflow
# defining tensorflow training strategy
device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)

num_hiddens = 512
with strategy.scope():
    net = RNNModelScratch(len(vocab), num_hiddens, init_rnn_state, rnn,
                          get_params)
state = net.begin_state(X.shape[0])
Y, new_state = net(X, state)
Y.shape, len(new_state), new_state[0].shape
```

Chúng ta có thể thấy rằng hình dạng đầu ra là (số bước thời gian $\times$ kích thước lô, kích thước từ vựng), trong khi hình dạng trạng thái ẩn vẫn giữ nguyên, tức là, (kích thước lô, số đơn vị ẩn). 

## Prediction

Hãy để chúng tôi [** đầu tiên xác định hàm dự đoán để tạo ra các ký tự mới sau `prefix`** người dùng cung cấp, đó là một chuỗi chứa một số ký tự. Khi lặp qua các ký tự bắt đầu này trong `prefix`, chúng tôi tiếp tục chuyển trạng thái ẩn sang bước thời gian tiếp theo mà không tạo ra bất kỳ đầu ra nào. Đây được gọi là khoảng thời gian *warm-up*, trong đó mô hình tự cập nhật (ví dụ: cập nhật trạng thái ẩn) nhưng không đưa ra dự đoán. Sau thời gian khởi động, trạng thái ẩn thường tốt hơn giá trị khởi tạo của nó ở đầu. Vì vậy, chúng tôi tạo ra các nhân vật dự đoán và phát ra chúng.

```{.python .input}
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """Generate new characters following the `prefix`."""
    state = net.begin_state(batch_size=1, ctx=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(
        d2l.tensor([outputs[-1]], ctx=device), (1, 1))
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input}
#@tab pytorch
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """Generate new characters following the `prefix`."""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(d2l.tensor(
        [outputs[-1]], device=device), (1, 1))
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input}
#@tab tensorflow
def predict_ch8(prefix, num_preds, net, vocab):  #@save
    """Generate new characters following the `prefix`."""
    state = net.begin_state(batch_size=1, dtype=tf.float32)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(d2l.tensor([outputs[-1]]), (1, 1)).numpy()
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        outputs.append(int(y.numpy().argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

Bây giờ chúng ta có thể kiểm tra hàm `predict_ch8`. Chúng tôi chỉ định tiền tố là `time traveller ` và có nó tạo ra 10 ký tự bổ sung. Cho rằng chúng tôi chưa đào tạo mạng, nó sẽ tạo ra những dự đoán vô nghĩa.

```{.python .input}
#@tab mxnet,pytorch
predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())
```

```{.python .input}
#@tab tensorflow
predict_ch8('time traveller ', 10, net, vocab)
```

## [**Clipping Gradient**]

Đối với một chuỗi chiều dài $T$, chúng tôi tính toán độ dốc trên các bước thời gian $T$ này trong một lần lặp lại, dẫn đến một chuỗi các sản phẩm ma trận có chiều dài $\mathcal{O}(T)$ trong quá trình truyền ngược. Như đã đề cập trong :numref:`sec_numerical_stability`, nó có thể dẫn đến sự bất ổn số, ví dụ, các gradient có thể phát nổ hoặc biến mất, khi $T$ lớn. Do đó, các mô hình RNN thường cần trợ giúp thêm để ổn định việc đào tạo. 

Nói chung, khi giải quyết vấn đề tối ưu hóa, chúng tôi thực hiện các bước cập nhật cho tham số mô hình, nói ở dạng vector $\mathbf{x}$, theo hướng gradient âm $\mathbf{g}$ trên một minibatch. Ví dụ: với $\eta > 0$ là tốc độ học tập, trong một lần lặp lại, chúng tôi cập nhật $\mathbf{x}$ là $\mathbf{x} - \eta \mathbf{g}$. Chúng ta hãy giả định thêm rằng chức năng khách quan $f$ được cư xử tốt, giả sử, * Lipschitz liên tục* với hằng số $L$. Điều đó có nghĩa là, đối với bất kỳ $\mathbf{x}$ và $\mathbf{y}$ nào chúng tôi có 

$$|f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|.$$

Trong trường hợp này, chúng ta có thể giả định một cách an toàn rằng nếu chúng ta cập nhật vector tham số bởi $\eta \mathbf{g}$, thì 

$$|f(\mathbf{x}) - f(\mathbf{x} - \eta\mathbf{g})| \leq L \eta\|\mathbf{g}\|,$$

điều đó có nghĩa là chúng ta sẽ không quan sát một sự thay đổi hơn $L \eta \|\mathbf{g}\|$. Đây vừa là một lời nguyền vừa là một phước lành. Về phía lời nguyền, nó giới hạn tốc độ tiến bộ; trong khi về phía phước lành, nó giới hạn mức độ mà mọi thứ có thể đi sai nếu chúng ta di chuyển sai hướng. 

Đôi khi các gradient có thể khá lớn và thuật toán tối ưu hóa có thể không hội tụ. Chúng tôi có thể giải quyết điều này bằng cách giảm tỷ lệ học tập $\eta$. Nhưng nếu chúng ta chỉ * hiếm có * có được gradient lớn thì sao? Trong trường hợp này một cách tiếp cận như vậy có thể xuất hiện hoàn toàn không chính đáng. Một lựa chọn phổ biến là cắt gradient $\mathbf{g}$ bằng cách chiếu chúng trở lại một quả bóng có bán kính nhất định, giả sử $\theta$ qua 

(**$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$**) 

Bằng cách đó, chúng tôi biết rằng định mức gradient không bao giờ vượt quá $\theta$ và gradient được cập nhật hoàn toàn phù hợp với hướng ban đầu là $\mathbf{g}$. Nó cũng có tác dụng phụ mong muốn của việc hạn chế ảnh hưởng bất kỳ minibatch nhất định nào (và trong đó bất kỳ mẫu nhất định nào) có thể tác dụng lên vectơ tham số. Điều này ban cho một mức độ mạnh mẽ nhất định cho mô hình. Gradient clipping cung cấp một sửa chữa nhanh chóng cho sự bùng nổ gradient. Mặc dù nó không hoàn toàn giải quyết vấn đề, nhưng nó là một trong nhiều kỹ thuật để giảm bớt nó. 

Dưới đây chúng ta định nghĩa một hàm để cắt các gradient của một mô hình được triển khai từ đầu hoặc một mô hình được xây dựng bởi các API cấp cao. Cũng lưu ý rằng chúng tôi tính toán định mức gradient trên tất cả các tham số mô hình.

```{.python .input}
def grad_clipping(net, theta):  #@save
    """Clip the gradient."""
    if isinstance(net, gluon.Block):
        params = [p.data() for p in net.collect_params().values()]
    else:
        params = net.params
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

```{.python .input}
#@tab pytorch
def grad_clipping(net, theta):  #@save
    """Clip the gradient."""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

```{.python .input}
#@tab tensorflow
def grad_clipping(grads, theta):  #@save
    """Clip the gradient."""
    theta = tf.constant(theta, dtype=tf.float32)
    new_grad = []
    for grad in grads:
        if isinstance(grad, tf.IndexedSlices):
            new_grad.append(tf.convert_to_tensor(grad))
        else:
            new_grad.append(grad)
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)).numpy()
                        for grad in new_grad))
    norm = tf.cast(norm, tf.float32)
    if tf.greater(norm, theta):
        for i, grad in enumerate(new_grad):
            new_grad[i] = grad * theta / norm
    else:
        new_grad = new_grad
    return new_grad
```

## Đào tạo

Trước khi đào tạo mô hình, chúng ta hãy [** xác định một hàm để đào tạo mô hình trong một kỷ chức**]. Nó khác với cách chúng tôi đào tạo mô hình :numref:`sec_softmax_scratch` ở ba nơi: 

1. Các phương pháp lấy mẫu khác nhau cho dữ liệu tuần tự (lấy mẫu ngẫu nhiên và phân vùng tuần tự) sẽ dẫn đến sự khác biệt trong việc khởi tạo các trạng thái ẩn.
1. Chúng tôi kẹp các gradient trước khi cập nhật các tham số mô hình. Điều này đảm bảo rằng mô hình không phân kỳ ngay cả khi gradient thổi lên tại một số điểm trong quá trình đào tạo.
1. Chúng tôi sử dụng sự bối rối để đánh giá mô hình. Như đã thảo luận trong :numref:`subsec_perplexity`, điều này đảm bảo rằng các chuỗi có độ dài khác nhau có thể so sánh được.

Cụ thể, khi phân vùng tuần tự được sử dụng, chúng ta chỉ khởi tạo trạng thái ẩn ở đầu mỗi kỷ nguyên. Vì ví dụ dãy con $i^\mathrm{th}$ trong minibatch tiếp theo liền kề với ví dụ dãy thứ tự $i^\mathrm{th}$ hiện tại, trạng thái ẩn ở cuối minibatch hiện tại sẽ được sử dụng để khởi tạo trạng thái ẩn ở đầu minibatch tiếp theo. Bằng cách này, thông tin lịch sử của dãy được lưu trữ trong trạng thái ẩn có thể chảy qua các dãy tiếp giáp bên trong một kỷ nguyên. Tuy nhiên, việc tính toán trạng thái ẩn tại bất kỳ điểm nào phụ thuộc vào tất cả các minibatches trước đó trong cùng một kỷ nguyên, làm phức tạp tính toán gradient. Để giảm chi phí tính toán, chúng tôi tách gradient trước khi xử lý bất kỳ minibatch nào để tính toán gradient của trạng thái ẩn luôn bị giới hạn ở các bước thời gian trong một minibatch.  

Khi sử dụng lấy mẫu ngẫu nhiên, chúng ta cần khởi tạo lại trạng thái ẩn cho mỗi lần lặp lại vì mỗi ví dụ được lấy mẫu với một vị trí ngẫu nhiên. Tương tự như hàm `train_epoch_ch3` trong :numref:`sec_softmax_scratch`, `updater` là một hàm chung để cập nhật các tham số mô hình. Nó có thể là chức năng `d2l.sgd` được triển khai từ đầu hoặc chức năng tối ưu hóa tích hợp trong một khuôn khổ học sâu.

```{.python .input}
#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """Train a model within one epoch (defined in Chapter 8)."""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = net.begin_state(batch_size=X.shape[0], ctx=device)
        else:
            for s in state:
                s.detach()
        y = Y.T.reshape(-1)
        X, y = X.as_in_ctx(device), y.as_in_ctx(device)
        with autograd.record():
            y_hat, state = net(X, state)
            l = loss(y_hat, y).mean()
        l.backward()
        grad_clipping(net, 1)
        updater(batch_size=1)  # Since the `mean` function has been invoked
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

```{.python .input}
#@tab pytorch
#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """Train a net within one epoch (defined in Chapter 8)."""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # `state` is a tensor for `nn.GRU`
                state.detach_()
            else:
                # `state` is a tuple of tensors for `nn.LSTM` and
                # for our custom scratch implementation 
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # Since the `mean` function has been invoked
            updater(batch_size=1)
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

```{.python .input}
#@tab tensorflow
#@save
def train_epoch_ch8(net, train_iter, loss, updater, use_random_iter):
    """Train a model within one epoch (defined in Chapter 8)."""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = net.begin_state(batch_size=X.shape[0], dtype=tf.float32)
        with tf.GradientTape(persistent=True) as g:
            y_hat, state = net(X, state)
            y = d2l.reshape(tf.transpose(Y), (-1))
            l = loss(y, y_hat)
        params = net.trainable_variables
        grads = g.gradient(l, params)
        grads = grad_clipping(grads, 1)
        updater.apply_gradients(zip(grads, params))

        # Keras loss by default returns the average loss in a batch
        # l_sum = l * float(d2l.size(y)) if isinstance(
        #     loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

[**Chức năng đào tạo hỗ trợ một mô hình RNN được thực hiện từ đầu hoặc sử dụng APIs cấp cao.**]

```{.python .input}
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,  #@save
              use_random_iter=False):
    """Train a model (defined in Chapter 8)."""
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # Initialize
    if isinstance(net, gluon.Block):
        net.initialize(ctx=device, force_reinit=True,
                         init=init.Normal(0.01))
        trainer = gluon.Trainer(net.collect_params(),
                                'sgd', {'learning_rate': lr})
        updater = lambda batch_size: trainer.step(batch_size)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, [ppl])
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

```{.python .input}
#@tab pytorch
#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """Train a model (defined in Chapter 8)."""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # Initialize
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

```{.python .input}
#@tab tensorflow
#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, strategy,
              use_random_iter=False):
    """Train a model (defined in Chapter 8)."""
    with strategy.scope():
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        updater = tf.keras.optimizers.SGD(lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater,
                                     use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    device = d2l.try_gpu()._device_name
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

[**Bây giờ chúng ta có thể đào tạo mô hình RNN**] Vì chúng ta chỉ sử dụng 10000 mã thông báo trong tập dữ liệu, mô hình cần nhiều kỷ nguyên hơn để hội tụ tốt hơn.

```{.python .input}
#@tab mxnet,pytorch
num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
```

```{.python .input}
#@tab tensorflow
num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, strategy)
```

[**Cuối cùng, chúng ta hãy kiểm tra kết quả của việc sử dụng phương pháp lấy mẫu ngẫu nhiên.**]

```{.python .input}
#@tab mxnet,pytorch
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)
```

```{.python .input}
#@tab tensorflow
with strategy.scope():
    net = RNNModelScratch(len(vocab), num_hiddens, init_rnn_state, rnn,
                          get_params)
train_ch8(net, train_iter, vocab_random_iter, lr, num_epochs, strategy,
          use_random_iter=True)
```

Trong khi thực hiện mô hình RNN trên từ đầu là hướng dẫn, nó không thuận tiện. Trong phần tiếp theo, chúng ta sẽ thấy cách cải thiện mô hình RNN, chẳng hạn như cách thực hiện dễ dàng hơn và làm cho nó chạy nhanh hơn. 

## Tóm tắt

* Chúng ta có thể đào tạo mô hình ngôn ngữ cấp ký tự dựa trên RNN để tạo văn bản theo tiền tố văn bản do người dùng cung cấp.
* Một mô hình ngôn ngữ RNN đơn giản bao gồm mã hóa đầu vào, mô hình RNN và tạo ra đầu ra.
* Các mô hình RNN cần khởi tạo trạng thái để đào tạo, mặc dù lấy mẫu ngẫu nhiên và phân vùng tuần tự sử dụng các cách khác nhau.
* Khi sử dụng phân vùng tuần tự, chúng ta cần tách gradient để giảm chi phí tính toán.
* Thời gian khởi động cho phép một mô hình tự cập nhật (ví dụ: có được trạng thái ẩn tốt hơn giá trị khởi tạo của nó) trước khi đưa ra bất kỳ dự đoán nào.
* Gradient clipping ngăn chặn sự bùng nổ gradient, nhưng nó không thể sửa chữa độ dốc biến mất.

## Bài tập

1. Cho thấy rằng mã hóa một nóng tương đương với việc chọn một nhúng khác nhau cho mỗi đối tượng.
1. Điều chỉnh các siêu tham số (ví dụ: số kỷ nguyên, số lượng đơn vị ẩn, số bước thời gian trong một minibatch và tốc độ học tập) để cải thiện sự bối rối.
    * Làm thế nào thấp bạn có thể đi?
    * Thay thế mã hóa một nóng bằng các embeddings có thể học được. Điều này có dẫn đến hiệu suất tốt hơn?
    * Nó sẽ hoạt động tốt như thế nào trên các cuốn sách khác của H Gwells, ví dụ, [*The War of the Worlds*](http://www.gutenberg.org/ebooks/36)?
1. Sửa đổi chức năng dự đoán như sử dụng lấy mẫu thay vì chọn ký tự tiếp theo có khả năng cao nhất.
    * Điều gì xảy ra?
    * Thiên vị mô hình hướng tới các đầu ra có khả năng cao hơn, ví dụ, bằng cách lấy mẫu từ $q(x_t \mid x_{t-1}, \ldots, x_1) \propto P(x_t \mid x_{t-1}, \ldots, x_1)^\alpha$ cho $\alpha > 1$.
1. Chạy mã trong phần này mà không cần cắt gradient. Điều gì xảy ra?
1. Thay đổi phân vùng tuần tự để nó không tách các trạng thái ẩn khỏi biểu đồ tính toán. Thời gian chạy có thay đổi không? Làm thế nào về sự bối rối?
1. Thay thế chức năng kích hoạt được sử dụng trong phần này bằng ReLU và lặp lại các thí nghiệm trong phần này. Chúng ta vẫn cần cắt gradient? Tại sao?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/336)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/486)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1052)
:end_tab:
