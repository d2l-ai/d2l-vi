# Thực hiện ngắn gọn về hồi quy Softmax
:label:`sec_softmax_concise`

(** Cũng giống như APIs cấp cao**) của các khuôn khổ học sâu (** làm cho nó dễ dàng hơn nhiều để thực hiện hồi quy tuyến tính**) trong :numref:`sec_linear_concise`, (** chúng tôi sẽ tìm thấy nó tương tự **) (~~here ~ ~) (hoặc có thể nhiều hơn) thuận tiện cho việc thực hiện các mô hình phân loại. Hãy để chúng tôi gắn bó với tập dữ liệu Fashion-MNIST và giữ kích thước lô ở mức 256 như trong :numref:`sec_softmax_scratch`.

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

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## Khởi tạo các tham số mô hình

Như đã đề cập trong :numref:`sec_softmax`, [** lớp đầu ra của hồi quy softmax là một lớp kết nối đầy đủ**] Do đó, để thực hiện mô hình của chúng tôi, chúng ta chỉ cần thêm một lớp kết nối hoàn toàn với 10 đầu ra vào `Sequential` của chúng tôi. Một lần nữa, ở đây, `Sequential` không thực sự cần thiết, nhưng chúng ta cũng có thể hình thành thói quen vì nó sẽ phổ biến khi thực hiện các mô hình sâu. Một lần nữa, chúng tôi khởi tạo các trọng lượng một cách ngẫu nhiên với 0 trung bình và độ lệch chuẩn 0,01.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
# PyTorch does not implicitly reshape the inputs. Thus we define the flatten
# layer to reshape the inputs before the linear layer in our network
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
net.add(tf.keras.layers.Dense(10, kernel_initializer=weight_initializer))
```

## Triển khai Softmax Revisited
:label:`subsec_softmax-implementation-revisited`

Trong ví dụ trước của :numref:`sec_softmax_scratch`, chúng tôi đã tính toán đầu ra của mô hình và sau đó chạy đầu ra này thông qua tổn thất ngẫu nhiên chéo. Về mặt toán học, đó là một điều hoàn toàn hợp lý để làm. Tuy nhiên, từ góc độ tính toán, số mũ có thể là một nguồn của các vấn đề ổn định số. 

Nhớ lại rằng hàm softmax tính toán $\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$, trong đó $\hat y_j$ là phần tử $j^\mathrm{th}$ của phân phối xác suất dự đoán $\hat{\mathbf{y}}$ và $o_j$ là phần tử $j^\mathrm{th}$ của logits $\mathbf{o}$. Nếu một số $o_k$ rất lớn (tức là rất tích cực), thì $\exp(o_k)$ có thể lớn hơn số lớn nhất chúng ta có thể có cho một số loại dữ liệu nhất định (tức là * đầu*). Điều này sẽ làm cho mẫu số (và/hoặc tử số) `inf` (vô cùng) và chúng tôi gặp phải 0, `inf` hoặc `nan` (không phải là một số) cho $\hat y_j$. Trong những tình huống này, chúng ta không nhận được giá trị trả về được xác định rõ cho cross-entropy. 

Một mẹo để giải quyết vấn đề này là lần đầu tiên trừ $\max(o_k)$ khỏi tất cả $o_k$ trước khi tiến hành tính toán softmax. Bạn có thể thấy rằng sự dịch chuyển này của mỗi $o_k$ theo hệ số không đổi không thay đổi giá trị trả về của softmax: 

$$
\begin{aligned}
\hat y_j & =  \frac{\exp(o_j - \max(o_k))\exp(\max(o_k))}{\sum_k \exp(o_k - \max(o_k))\exp(\max(o_k))} \\
& = \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}.
\end{aligned}
$$

Sau bước trừ và bình thường hóa, có thể một số $o_j - \max(o_k)$ có giá trị âm lớn và do đó $\exp(o_j - \max(o_k))$ tương ứng sẽ lấy giá trị gần bằng 0. Chúng có thể được làm tròn thành 0 do độ chính xác hữu hạn (ví dụ: * underflow*), làm cho $\hat y_j$ không và cho chúng tôi `-inf` cho $\log(\hat y_j)$. Một vài bước xuống con đường trong backpropagation, chúng ta có thể thấy mình phải đối mặt với một màn hình của đáng sợ `nan` kết quả. 

May mắn thay, chúng tôi được lưu bởi thực tế là mặc dù chúng tôi đang tính toán các hàm mũ, cuối cùng chúng tôi có ý định lấy nhật ký của chúng (khi tính toán mất chéo entropy). Bằng cách kết hợp hai toán tử softmax và cross-entropy với nhau, chúng ta có thể thoát khỏi các vấn đề ổn định số mà nếu không có thể gây ra chúng ta trong quá trình truyền ngược. Như thể hiện trong phương trình dưới đây, chúng ta tránh tính toán $\exp(o_j - \max(o_k))$ và có thể sử dụng thay thế $o_j - \max(o_k)$ trực tiếp do việc hủy bỏ trong $\log(\exp(\cdot))$: 

$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}\right) \\
& = \log{(\exp(o_j - \max(o_k)))}-\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)} \\
& = o_j - \max(o_k) -\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)}.
\end{aligned}
$$

Chúng tôi sẽ muốn giữ chức năng softmax thông thường tiện dụng trong trường hợp chúng tôi muốn đánh giá xác suất đầu ra theo mô hình của chúng tôi. Nhưng thay vì truyền xác suất softmax vào chức năng mất mát mới của chúng ta, chúng ta sẽ chỉ [** vượt qua các logits và tính toán softmax và nhật ký của nó cùng một lúc bên trong hàm loss cross-entropy, **] mà thực hiện những điều thông minh như ["LogSumExp trick"](https://en.wikipedia.org/wiki/LogSumExp).

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss(reduction='none')
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

## Thuật toán tối ưu hóa

Ở đây, chúng ta (**sử dụng minibatch stochastic gradient descent**) với tốc độ học tập là 0.1 làm thuật toán tối ưu hóa. Lưu ý rằng điều này giống như chúng ta đã áp dụng trong ví dụ hồi quy tuyến tính và nó minh họa khả năng ứng dụng chung của các trình tối ưu hóa.

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=.1)
```

## Đào tạo

Tiếp theo chúng ta [**gọi hàm đào tạo**](~~earlier~~) trong :numref:`sec_softmax_scratch` để đào tạo mô hình.

```{.python .input}
#@tab all
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

Như trước đây, thuật toán này hội tụ thành một giải pháp đạt được độ chính xác khá, mặc dù lần này với ít dòng mã hơn trước. 

## Tóm tắt

* Sử dụng API cấp cao, chúng ta có thể thực hiện hồi quy softmax chính xác hơn nhiều.
* Từ góc độ tính toán, việc thực hiện hồi quy softmax có những phức tạp. Lưu ý rằng trong nhiều trường hợp, một khuôn khổ học sâu thực hiện các biện pháp phòng ngừa bổ sung ngoài các thủ thuật nổi tiếng nhất này để đảm bảo sự ổn định về số, cứu chúng ta khỏi những cạm bẫy hơn nữa mà chúng ta sẽ gặp phải nếu chúng ta cố gắng mã hóa tất cả các mô hình của mình từ đầu trong thực tế.

## Bài tập

1. Hãy thử điều chỉnh các siêu tham số, chẳng hạn như kích thước lô, số kỷ nguyên và tốc độ học tập, để xem kết quả là gì.
1. Tăng số lượng kỷ nguyên để đào tạo. Tại sao độ chính xác thử nghiệm có thể giảm sau một thời gian? Làm thế nào chúng ta có thể sửa chữa điều này?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/52)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/53)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/260)
:end_tab:
