# Sự phức tạp cho hình ảnh
:label:`sec_conv_layer`

Bây giờ chúng ta đã hiểu cách các lớp phức tạp hoạt động theo lý thuyết, chúng ta đã sẵn sàng để xem chúng hoạt động như thế nào trong thực tế. Dựa trên động lực của chúng tôi về các mạng thần kinh phức tạp như kiến trúc hiệu quả để khám phá cấu trúc trong dữ liệu hình ảnh, chúng tôi gắn bó với hình ảnh làm ví dụ chạy của chúng tôi. 

## Hoạt động tương quan chéo

Nhớ lại rằng nói đúng, các lớp phức tạp là một sự nhầm lẫn, vì các hoạt động mà chúng thể hiện được mô tả chính xác hơn là tương quan chéo. Dựa trên mô tả của chúng tôi về các lớp phức tạp trong :numref:`sec_why-conv`, trong một lớp như vậy, một tensor đầu vào và một tensor hạt nhân được kết hợp để tạo ra một tensor đầu ra thông qua một (** cross-correlation operation.**) 

Chúng ta hãy bỏ qua các kênh ngay bây giờ và xem cách điều này hoạt động với dữ liệu hai chiều và biểu diễn ẩn. Trong :numref:`fig_correlation`, đầu vào là tensor hai chiều với chiều cao 3 và chiều rộng 3. Chúng tôi đánh dấu hình dạng của tensor là $3 \times 3$ hoặc ($3$, $3$). Chiều cao và chiều rộng của hạt nhân đều là 2. Hình dạng của cửa sổ *kernel * (hoặc cửa sổ *convolution*) được cho bởi chiều cao và chiều rộng của hạt nhân (ở đây nó là $2 \times 2$). 

![Two-dimensional cross-correlation operation. The shaded portions are the first output element as well as the input and kernel tensor elements used for the output computation: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
:label:`fig_correlation`

Trong hoạt động tương quan chéo hai chiều, chúng ta bắt đầu với cửa sổ covolution được đặt ở góc trên bên trái của tensor đầu vào và trượt nó qua tensor đầu vào, cả từ trái sang phải và trên xuống dưới. Khi cửa sổ covolution trượt đến một vị trí nhất định, subtensor đầu vào chứa trong cửa sổ đó và tensor kernel được nhân elementwise và tensor kết quả được tổng hợp lên mang lại một giá trị vô hướng duy nhất. Kết quả này cho giá trị của tensor đầu ra tại vị trí tương ứng. Ở đây, tensor đầu ra có chiều cao 2 và chiều rộng 2 và bốn phần tử có nguồn gốc từ hoạt động tương quan chéo hai chiều: 

$$
0\times0+1\times1+3\times2+4\times3=19,\\
1\times0+2\times1+4\times2+5\times3=25,\\
3\times0+4\times1+6\times2+7\times3=37,\\
4\times0+5\times1+7\times2+8\times3=43.
$$

Lưu ý rằng dọc theo mỗi trục, kích thước đầu ra nhỏ hơn một chút so với kích thước đầu vào. Bởi vì hạt nhân có chiều rộng và chiều cao lớn hơn một, chúng ta chỉ có thể tính toán tương quan chéo cho các vị trí mà hạt nhân vừa vặn hoàn toàn trong hình ảnh, kích thước đầu ra được đưa ra bởi kích thước đầu vào $n_h \times n_w$ trừ đi kích thước của hạt nhân phức tạp $k_h \times k_w$ qua 

$$(n_h-k_h+1) \times (n_w-k_w+1).$$

Đây là trường hợp vì chúng ta cần đủ không gian để “thay đổi” hạt nhân phức tạp trên hình ảnh. Sau đó chúng ta sẽ thấy làm thế nào để giữ cho kích thước không thay đổi bằng cách đệm hình ảnh với các số không xung quanh ranh giới của nó để có đủ không gian để thay đổi hạt nhân. Tiếp theo, chúng tôi triển khai quá trình này trong hàm `corr2d`, chấp nhận tensor đầu vào `X` và tensor kernel `K` và trả về tensor đầu ra `Y`.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
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
#@tab mxnet, pytorch
def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.reduce_sum(
                X[i: i + h, j: j + w] * K))
    return Y
```

Chúng ta có thể xây dựng tensor đầu vào `X` và tensor kernel `K` từ :numref:`fig_correlation` đến [** xác nhận đầu ra của việc triển khai ở trên**] của hoạt động tương quan chéo hai chiều.

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)
```

## Layers phức tạp

Một lớp phức hợp chéo tương quan đầu vào và hạt nhân và thêm một thiên vị vô hướng để tạo ra một đầu ra. Hai tham số của một lớp tổ hợp là hạt nhân và thiên vị vô hướng. Khi đào tạo các mô hình dựa trên các lớp phức tạp, chúng ta thường khởi tạo các hạt nhân một cách ngẫu nhiên, giống như chúng ta sẽ với một lớp được kết nối hoàn toàn. 

Bây giờ chúng ta đã sẵn sàng để [**triển khai một lớp ghép hai chiều**] dựa trên hàm `corr2d` được xác định ở trên. Trong hàm xây dựng `__init__`, chúng ta khai báo `weight` và `bias` là hai tham số model. Hàm tuyên truyền chuyển tiếp gọi hàm `corr2d` và thêm sự thiên vị.

```{.python .input}
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()
```

```{.python .input}
#@tab pytorch
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

```{.python .input}
#@tab tensorflow
class Conv2D(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, kernel_size):
        initializer = tf.random_normal_initializer()
        self.weight = self.add_weight(name='w', shape=kernel_size,
                                      initializer=initializer)
        self.bias = self.add_weight(name='b', shape=(1, ),
                                    initializer=initializer)

    def call(self, inputs):
        return corr2d(inputs, self.weight) + self.bias
```

Trong $h \times w$ sự phức tạp hoặc một hạt nhân phức tạp $h \times w$, chiều cao và chiều rộng của hạt nhân covolution lần lượt là $h$ và $w$. Chúng tôi cũng đề cập đến một lớp phức tạp với một hạt nhân phức tạp $h \times w$ đơn giản là một lớp ghép $h \times w$. 

## Phát hiện cạnh đối tượng trong hình ảnh

Chúng ta hãy dành một chút thời gian để phân tích cú pháp [** một ứng dụng đơn giản của một lớp phức tạp: phát hiện cạnh của một đối tượng trong một hình ảnh**] bằng cách tìm vị trí của sự thay đổi điểm ảnh. Đầu tiên, chúng tôi xây dựng một “hình ảnh” của $6\times 8$ pixel. Bốn cột giữa có màu đen (0) và phần còn lại có màu trắng (1).

```{.python .input}
#@tab mxnet, pytorch
X = d2l.ones((6, 8))
X[:, 2:6] = 0
X
```

```{.python .input}
#@tab tensorflow
X = tf.Variable(tf.ones((6, 8)))
X[:, 2:6].assign(tf.zeros(X[:, 2:6].shape))
X
```

Tiếp theo, chúng ta xây dựng một hạt nhân `K` với chiều cao 1 và chiều rộng là 2. Khi chúng ta thực hiện thao tác tương quan chéo với đầu vào, nếu các phần tử liền kề theo chiều ngang giống nhau, đầu ra là 0. Nếu không, đầu ra không phải là không.

```{.python .input}
#@tab all
K = d2l.tensor([[1.0, -1.0]])
```

Chúng tôi đã sẵn sàng để thực hiện các hoạt động tương quan chéo với các đối số `X` (đầu vào của chúng tôi) và `K` (hạt nhân của chúng tôi). Như bạn có thể thấy, [** chúng tôi phát hiện 1 cho cạnh từ trắng sang đen và -1 cho cạnh từ đen sang trắng.**] Tất cả các đầu ra khác đều có giá trị 0.

```{.python .input}
#@tab all
Y = corr2d(X, K)
Y
```

Bây giờ chúng ta có thể áp dụng hạt nhân vào hình ảnh chuyển tiếp. Đúng như dự đoán, nó biến mất. [**Hạt nhân `K` chỉ phát hiện các viền dọc. **]

```{.python .input}
#@tab all
corr2d(d2l.transpose(X), K)
```

## Học một hạt nhân

Thiết kế một máy dò cạnh bằng sự khác biệt hữu hạn `[1, -1]` là gọn gàng nếu chúng ta biết đây chính xác là những gì chúng tôi đang tìm kiếm. Tuy nhiên, khi chúng ta nhìn vào các hạt nhân lớn hơn và xem xét các lớp phức tạp liên tiếp, có thể không thể xác định chính xác những gì mỗi bộ lọc nên làm thủ công. 

Bây giờ chúng ta hãy xem liệu chúng ta có thể [** tìm hiểu hạt nhân tạo ra `Y` từ `X`**] bằng cách xem các cặp đầu vào-đầu ra chỉ. Đầu tiên chúng ta xây dựng một lớp phức tạp và khởi tạo hạt nhân của nó như một tensor ngẫu nhiên. Tiếp theo, trong mỗi lần lặp lại, chúng ta sẽ sử dụng lỗi bình phương để so sánh `Y` với đầu ra của lớp ghép. Sau đó chúng ta có thể tính toán gradient để cập nhật hạt nhân. Vì lợi ích của sự đơn giản, sau đây chúng ta sử dụng lớp tích hợp cho các lớp phức tạp hai chiều và bỏ qua sự thiên vị.

```{.python .input}
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.Conv2D(1, kernel_size=(1, 2), use_bias=False)
conv2d.initialize()

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape(1, 1, 6, 8)
Y = Y.reshape(1, 1, 6, 7)
lr = 3e-2  # Learning rate

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    # Update the kernel
    conv2d.weight.data()[:] -= lr * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {float(l.sum()):.3f}')
```

```{.python .input}
#@tab pytorch
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # Learning rate

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # Update the kernel
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')
```

```{.python .input}
#@tab tensorflow
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = tf.keras.layers.Conv2D(1, (1, 2), use_bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, height, width, channel), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = tf.reshape(X, (1, 6, 8, 1))
Y = tf.reshape(Y, (1, 6, 7, 1))
lr = 3e-2  # Learning rate

Y_hat = conv2d(X)
for i in range(10):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(conv2d.weights[0])
        Y_hat = conv2d(X)
        l = (abs(Y_hat - Y)) ** 2
        # Update the kernel
        update = tf.multiply(lr, g.gradient(l, conv2d.weights[0]))
        weights = conv2d.get_weights()
        weights[0] = conv2d.weights[0] - update
        conv2d.set_weights(weights)
        if (i + 1) % 2 == 0:
            print(f'epoch {i + 1}, loss {tf.reduce_sum(l):.3f}')
```

Lưu ý rằng lỗi đã giảm xuống một giá trị nhỏ sau 10 lần lặp lại. Bây giờ chúng ta sẽ [** hãy xem tensor kernel mà chúng ta đã học được.**]

```{.python .input}
d2l.reshape(conv2d.weight.data(), (1, 2))
```

```{.python .input}
#@tab pytorch
d2l.reshape(conv2d.weight.data, (1, 2))
```

```{.python .input}
#@tab tensorflow
d2l.reshape(conv2d.get_weights()[0], (1, 2))
```

Thật vậy, tensor kernel đã học được rất gần với tensor kernel `K` mà chúng tôi đã xác định trước đó. 

## Tương quan chéo và phức tạp

Nhớ lại quan sát của chúng tôi từ :numref:`sec_why-conv` về sự tương ứng giữa các hoạt động tương quan và phức tạp chéo. Ở đây chúng ta hãy tiếp tục xem xét các lớp phức tạp hai chiều. Điều gì sẽ xảy ra nếu các lớp như vậy thực hiện các hoạt động phức tạp nghiêm ngặt như được định nghĩa trong :eqref:`eq_2d-conv-discrete` thay vì tương quan chéo? Để có được đầu ra của thao tác * convolution* nghiêm ngặt, chúng ta chỉ cần lật tensor hạt nhân hai chiều theo chiều ngang và chiều dọc, sau đó thực hiện thao tác *cross-correlation* với tensor đầu vào. 

Đáng chú ý là vì hạt nhân được học từ dữ liệu trong học sâu, các đầu ra của các lớp phức tạp vẫn không bị ảnh hưởng bất kể các lớp như vậy thực hiện các hoạt động phức tạp nghiêm ngặt hoặc các hoạt động tương quan chéo. 

Để minh họa điều này, giả sử rằng một lớp phức hợp thực hiện *tương quan chéo * và học hạt nhân trong :numref:`fig_correlation`, được ký hiệu là ma trận $\mathbf{K}$ ở đây. Giả sử rằng các điều kiện khác vẫn không thay đổi, khi lớp này thực hiện nghiêm ngặt *convolution* thay vào đó, hạt nhân đã học $\mathbf{K}'$ sẽ giống như $\mathbf{K}$ sau khi $\mathbf{K}'$ được lật cả hai chiều ngang và chiều dọc. Điều đó có nghĩa là, khi lớp phức tạp thực hiện nghiêm ngặt * độ lượng* cho đầu vào trong :numref:`fig_correlation` và $\mathbf{K}'$, cùng một đầu ra trong :numref:`fig_correlation` (tương quan chéo của đầu vào và $\mathbf{K}$) sẽ thu được. 

Để phù hợp với thuật ngữ tiêu chuẩn với tài liệu học sâu, chúng ta sẽ tiếp tục đề cập đến hoạt động tương quan chéo như một sự phức tạp mặc dù, nói nghiêm ngặt, nó hơi khác nhau. Bên cạnh đó, chúng ta sử dụng thuật ngữ *element* để chỉ một mục nhập (hoặc component) của bất kỳ tensor đại diện cho một biểu diễn lớp hoặc một hạt nhân phức tạp. 

## Bản đồ tính năng và trường tiếp nhận

Như được mô tả trong :numref:`subsec_why-conv-channels`, đầu ra lớp phức tạp trong :numref:`fig_correlation` đôi khi được gọi là bản đồ *tính năng*, vì nó có thể được coi là biểu diễn (tính năng) đã học trong các kích thước không gian (ví dụ, chiều rộng và chiều cao) cho lớp tiếp theo. Trong CNN, đối với bất kỳ yếu tố $x$ nào của một số lớp, trường tiếp thu* của nóđề cập đến tất cả các yếu tố (từ tất cả các lớp trước) có thể ảnh hưởng đến việc tính toán $x$ trong quá trình lan truyền về phía trước. Lưu ý rằng trường tiếp nhận có thể lớn hơn kích thước thực tế của đầu vào. 

Chúng ta hãy tiếp tục sử dụng :numref:`fig_correlation` để giải thích lĩnh vực tiếp nhận. Với hạt nhân phức tạp $2 \times 2$, trường tiếp nhận của phần tử đầu ra bóng mờ (có giá trị $19$) là bốn phần tử trong phần bóng mờ của đầu vào. Bây giờ chúng ta hãy biểu thị đầu ra $2 \times 2$ là $\mathbf{Y}$ và xem xét một CNN sâu hơn với một lớp phức tạp $2 \times 2$ bổ sung lấy $\mathbf{Y}$ làm đầu vào của nó, xuất ra một phần tử duy nhất $z$. Trong trường hợp này, trường tiếp nhận $z$ trên $\mathbf{Y}$ bao gồm tất cả bốn yếu tố của $\mathbf{Y}$, trong khi trường tiếp nhận trên đầu vào bao gồm tất cả chín yếu tố đầu vào. Do đó, khi bất kỳ yếu tố nào trong bản đồ tính năng cần một trường tiếp nhận lớn hơn để phát hiện các tính năng đầu vào trên một khu vực rộng hơn, chúng ta có thể xây dựng một mạng lưới sâu hơn. 

## Tóm tắt

* Tính toán cốt lõi của một lớp phức hợp hai chiều là một hoạt động tương quan chéo hai chiều. Ở dạng đơn giản nhất của nó, điều này thực hiện một hoạt động tương quan chéo trên dữ liệu đầu vào hai chiều và hạt nhân, và sau đó thêm một thiên vị.
* Chúng ta có thể thiết kế một hạt nhân để phát hiện các cạnh trong hình ảnh.
* Chúng ta có thể tìm hiểu các tham số của hạt nhân từ dữ liệu.
* Với hạt nhân học được từ dữ liệu, các đầu ra của các lớp phức tạp vẫn không bị ảnh hưởng bất kể các hoạt động được thực hiện của các lớp như vậy (hoặc là sự phức tạp nghiêm ngặt hoặc tương quan chéo).
* Khi bất kỳ yếu tố nào trong bản đồ tính năng cần một trường tiếp nhận lớn hơn để phát hiện các tính năng rộng hơn trên đầu vào, một mạng sâu hơn có thể được xem xét.

## Bài tập

1. Xây dựng một hình ảnh `X` với các cạnh chéo.
    1. Điều gì xảy ra nếu bạn áp dụng hạt nhân `K` trong phần này cho nó?
    1. Điều gì xảy ra nếu bạn chuyển `X`?
    1. Điều gì xảy ra nếu bạn chuyển `K`?
1. Khi bạn cố gắng tự động tìm gradient cho lớp `Conv2D` mà chúng tôi đã tạo, bạn thấy loại thông báo lỗi nào?
1. Làm thế nào để bạn đại diện cho một hoạt động tương quan chéo như một phép nhân ma trận bằng cách thay đổi các hàng chục đầu vào và hạt nhân?
1. Thiết kế một số hạt nhân bằng tay.
    1. Hình thức của một hạt nhân cho đạo hàm thứ hai là gì?
    1. Hạt nhân cho một tích phân là gì?
    1. Kích thước tối thiểu của một hạt nhân để có được một đạo hàm của độ $d$ là bao nhiêu?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/65)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/66)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/271)
:end_tab:
