# Pooling
:label:`sec_pooling`

Thông thường, khi chúng tôi xử lý hình ảnh, chúng tôi muốn giảm dần độ phân giải không gian của các biểu diễn ẩn của chúng tôi, tổng hợp thông tin để chúng ta đi lên cao hơn trong mạng, trường tiếp nhận càng lớn (trong đầu vào) mà mỗi nút ẩn nhạy cảm. 

Thường thì nhiệm vụ cuối cùng của chúng tôi hỏi một số câu hỏi toàn cầu về hình ảnh, ví dụ: * nó có chứa một con mèo không? * Vì vậy, thông thường các đơn vị của lớp cuối cùng của chúng ta phải nhạy cảm với toàn bộ đầu vào. Bằng cách dần dần tổng hợp thông tin, mang lại bản đồ thô hơn và thô hơn, chúng tôi hoàn thành mục tiêu này cuối cùng là học một đại diện toàn cầu, đồng thời giữ tất cả các lợi thế của các lớp phức tạp ở các lớp xử lý trung gian. 

Hơn nữa, khi phát hiện các tính năng cấp thấp hơn, chẳng hạn như các cạnh (như đã thảo luận trong :numref:`sec_conv_layer`), chúng ta thường muốn các đại diện của mình có phần bất biến với bản dịch. Ví dụ, nếu chúng ta chụp ảnh `X` với sự phân định sắc nét giữa đen và trắng và thay đổi toàn bộ hình ảnh theo một pixel sang phải, tức là `Z[i, j] = X[i, j + 1]`, thì đầu ra cho hình ảnh mới `Z` có thể rất khác nhau. Các cạnh sẽ thay đổi bởi một pixel. Trong thực tế, các vật thể hầu như không bao giờ xảy ra chính xác ở cùng một nơi. Trên thực tế, ngay cả với một chân máy và một đối tượng đứng yên, rung động của máy ảnh do chuyển động của màn trập có thể thay đổi mọi thứ bằng một pixel hoặc lâu hơn (máy ảnh cao cấp được tải với các tính năng đặc biệt để giải quyết vấn đề này). 

Phần này giới thiệu các lớp * pooling*, phục vụ các mục đích kép của việc giảm thiểu độ nhạy của các lớp phức tạp đến vị trí và các biểu diễn lấy mẫu không gian. 

## Pooling tối đa và Pooling trung bình

Giống như các lớp phức tạp, các toán tử * bộ* bao gồm một cửa sổ hình dạng cố định được trượt trên tất cả các vùng trong đầu vào theo sải chân của nó, tính toán một đầu ra duy nhất cho mỗi vị trí đi qua bởi cửa sổ hình dạng cố định (đôi khi được gọi là cửa sổ * gộp lọc*). Tuy nhiên, không giống như tính toán tương quan chéo của các đầu vào và hạt nhân trong lớp phức tạp, lớp tổng hợp không chứa tham số (không có * kernel*). Thay vào đó, toán tử tổng hợp là xác định, thường tính toán giá trị tối đa hoặc giá trị trung bình của các phần tử trong cửa sổ tổng hợp. Các hoạt động này được gọi là * bể tối đa* (* tổng hợp tối đa* cho ngắn hạn) và * trung bình*, tương ứng. 

Trong cả hai trường hợp, như với toán tử tương quan chéo, chúng ta có thể nghĩ về cửa sổ tổng hợp như bắt đầu từ phía trên bên trái của tensor đầu vào và trượt qua tensor đầu vào từ trái sang phải và trên xuống dưới. Tại mỗi vị trí mà cửa sổ pooling nhấn, nó tính toán giá trị lớn nhất hoặc trung bình của subtensor đầu vào trong cửa sổ, tùy thuộc vào việc tổng hợp tối đa hay trung bình được sử dụng. 

![Maximum pooling with a pooling window shape of $2\times 2$. The shaded portions are the first output element as well as the input tensor elements used for the output computation: $\max(0, 1, 3, 4)=4$.](../img/pooling.svg)
:label:`fig_pooling`

Tensor đầu ra trong :numref:`fig_pooling` có chiều cao 2 và chiều rộng là 2. Bốn phần tử được bắt nguồn từ giá trị lớn nhất trong mỗi cửa sổ pooling: 

$$
\max(0, 1, 3, 4)=4,\\
\max(1, 2, 4, 5)=5,\\
\max(3, 4, 6, 7)=7,\\
\max(4, 5, 7, 8)=8.\\
$$

Một lớp tổng hợp với hình dạng cửa sổ tổng hợp là $p \times q$ được gọi là một lớp tổng hợp $p \times q$. Các hoạt động pooling được gọi là $p \times q$ gộp. 

Chúng ta hãy quay lại ví dụ phát hiện cạnh đối tượng được đề cập ở đầu phần này. Bây giờ chúng ta sẽ sử dụng đầu ra của lớp phức tạp làm đầu vào cho tổng hợp tối đa $2\times 2$. Đặt đầu vào lớp phức tạp là `X` và đầu ra lớp pooling là `Y`. Các giá trị của `X[i, j]` và `X[i, j + 1]` có khác nhau hay không, hoặc `X[i, j + 1]` và `X[i, j + 2]` có khác nhau hay không, lớp tổng hợp luôn xuất ra `Y[i, j] = 1`. Có nghĩa là, sử dụng lớp tổng hợp tối đa $2\times 2$, chúng ta vẫn có thể phát hiện xem mẫu được lớp phức tạp nhận ra không di chuyển không quá một phần tử về chiều cao hoặc chiều rộng. 

Trong đoạn code dưới đây, chúng ta (**thực hiện sự lan truyền chuyển tiếp của lớp gộp **) trong hàm `pool2d`. Chức năng này tương tự như hàm `corr2d` trong :numref:`sec_conv_layer`. Tuy nhiên, ở đây chúng tôi không có hạt nhân, tính toán đầu ra là mức tối đa hoặc trung bình của từng vùng trong đầu vào.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
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
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = d2l.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = tf.Variable(tf.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w +1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j].assign(tf.reduce_max(X[i: i + p_h, j: j + p_w]))
            elif mode =='avg':
                Y[i, j].assign(tf.reduce_mean(X[i: i + p_h, j: j + p_w]))
    return Y
```

Chúng ta có thể xây dựng tensor đầu vào `X` trong :numref:`fig_pooling` để [** xác nhận đầu ra của lớp tổng hợp tối đa hai chiều**].

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
```

Ngoài ra, chúng tôi thử nghiệm với (** lớp tổng hợp trung bình**).

```{.python .input}
#@tab all
pool2d(X, (2, 2), 'avg')
```

## [**Padding và Stride**]

Như với các lớp phức tạp, các lớp gộp cũng có thể thay đổi hình dạng đầu ra. Và như trước đây, chúng ta có thể thay đổi hoạt động để đạt được hình dạng đầu ra mong muốn bằng cách đệm đầu vào và điều chỉnh sải chân. Chúng ta có thể chứng minh việc sử dụng đệm và bước tiến trong các lớp tổng hợp thông qua lớp tổng hợp tối đa hai chiều tích hợp từ khung học sâu. Đầu tiên chúng ta xây dựng một tensor đầu vào `X` có hình dạng có bốn chiều, trong đó số lượng ví dụ (kích thước lô) và số kênh đều là 1.

:begin_tab:`tensorflow`
Điều quan trọng cần lưu ý là tensorflow thích và được tối ưu hóa cho * channels-last* đầu vào.
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 1, 4, 4))
X
```

```{.python .input}
#@tab tensorflow
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 4, 4, 1))
X
```

Theo mặc định, (** sải chân và cửa sổ tổng hợp trong ví dụ từ lớp tích hợp của framework có cùng hình dạng.**) Dưới đây, chúng ta sử dụng một cửa sổ tổng hợp hình dạng `(3, 3)`, vì vậy chúng ta có được một hình dạng sải chân của `(3, 3)` theo mặc định.

```{.python .input}
pool2d = nn.MaxPool2D(3)
# Because there are no model parameters in the pooling layer, we do not need
# to call the parameter initialization function
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3])
pool2d(X)
```

[**Sải chân và đệm có thể được chỉ định thủ công.**]

```{.python .input}
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)
```

:begin_tab:`mxnet`
Tất nhiên, chúng ta có thể chỉ định một cửa sổ tổng hợp hình chữ nhật tùy ý và chỉ định đệm và sải chân cho chiều cao và chiều rộng, tương ứng.
:end_tab:

:begin_tab:`pytorch`
Tất nhiên, chúng ta có thể (** chỉ định một cửa sổ tổng hợp hình chữ nhật tùy ý và chỉ định padding và sải chân cho chiều cao và chiều rộng**), tương ứng.
:end_tab:

:begin_tab:`tensorflow`
Tất nhiên, chúng ta có thể chỉ định một cửa sổ tổng hợp hình chữ nhật tùy ý và chỉ định đệm và sải chân cho chiều cao và chiều rộng, tương ứng.
:end_tab:

```{.python .input}
pool2d = nn.MaxPool2D((2, 3), padding=(0, 1), strides=(2, 3))
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
pool2d(X)
```

```{.python .input}
#@tab tensorflow
paddings = tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]])
X_padded = tf.pad(X, paddings, "CONSTANT")

pool2d = tf.keras.layers.MaxPool2D(pool_size=[2, 3], padding='valid',
                                   strides=(2, 3))
pool2d(X_padded)
```

## Nhiều kênh

Khi xử lý dữ liệu đầu vào đa kênh, [** lớp tổng hợp nhóm mỗi kênh đầu vào riêng biệt**], thay vì tổng hợp các đầu vào lên trên các kênh như trong một lớp phức tạp. Điều này có nghĩa là số kênh đầu ra cho lớp tổng hợp giống như số kênh đầu vào. Dưới đây, chúng tôi sẽ nối hàng chục `X` và `X + 1` trên kích thước kênh để xây dựng một đầu vào với 2 kênh.

:begin_tab:`tensorflow`
Lưu ý rằng điều này sẽ yêu cầu một nối dọc theo chiều cuối cùng cho TensorFlow do cú pháp kênh-cuối cùng.
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
X = d2l.concat((X, X + 1), 1)
X
```

```{.python .input}
#@tab tensorflow
X = tf.concat([X, X + 1], 3)  # Concatenate along `dim=3` due to channels-last syntax
```

Như chúng ta có thể thấy, số lượng kênh đầu ra vẫn còn 2 sau khi tổng hợp.

```{.python .input}
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)
```

:begin_tab:`tensorflow`
Lưu ý rằng đầu ra cho các tập hợp tensorflow xuất hiện ở cái nhìn đầu tiên là khác nhau, tuy nhiên số lượng các kết quả tương tự được trình bày như MXNet và PyTorch. Sự khác biệt nằm ở chiều, và đọc đầu ra theo chiều dọc mang lại cùng một đầu ra như các triển khai khác.
:end_tab:

## Tóm tắt

* Lấy các phần tử đầu vào trong cửa sổ tổng hợp, thao tác tổng hợp tối đa gán giá trị lớn nhất làm đầu ra và hoạt động tổng hợp trung bình gán giá trị trung bình làm đầu ra.
* Một trong những lợi ích chính của một lớp tổng hợp là làm giảm bớt độ nhạy quá mức của lớp phức tạp với vị trí.
* Chúng ta có thể chỉ định padding và sải chân cho layer pooling.
* Tổng hợp tối đa, kết hợp với một sải chân lớn hơn 1 có thể được sử dụng để giảm kích thước không gian (ví dụ, chiều rộng và chiều cao).
* Số kênh đầu ra của lớp tổng hợp giống như số kênh đầu vào.

## Bài tập

1. Bạn có thể thực hiện tổng hợp trung bình như một trường hợp đặc biệt của một lớp phức tạp không? Nếu vậy, hãy làm điều đó.
1. Bạn có thể thực hiện pooling tối đa như một trường hợp đặc biệt của một lớp phức tạp? Nếu vậy, hãy làm điều đó.
1. Chi phí tính toán của lớp pooling là bao nhiêu? Giả sử rằng đầu vào của lớp pooling có kích thước $c\times h\times w$, cửa sổ pooling có hình dạng $p_h\times p_w$ với một padding là $(p_h, p_w)$ và một sải chân là $(s_h, s_w)$.
1. Tại sao bạn mong đợi tổng hợp tối đa và trung bình để làm việc khác nhau?
1. Chúng ta có cần một lớp tổng hợp tối thiểu riêng biệt không? Bạn có thể thay thế nó bằng một hoạt động khác không?
1. Có một hoạt động khác giữa tổng hợp trung bình và tối đa mà bạn có thể xem xét (gợi ý: nhớ lại softmax)? Tại sao nó có thể không được phổ biến như vậy?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/71)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/72)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/274)
:end_tab:
