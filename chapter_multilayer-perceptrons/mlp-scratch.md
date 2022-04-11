# Thực hiện các Perceptrons đa lớp từ đầu
:label:`sec_mlp_scratch`

Bây giờ chúng ta đã đặc trưng các nhận thức đa lớp (MLPs) về mặt toán học, chúng ta hãy cố gắng thực hiện một chính mình. Để so sánh với các kết quả trước đây của chúng tôi đạt được với hồi quy softmax (:numref:`sec_softmax_scratch`), chúng tôi sẽ tiếp tục làm việc với tập dữ liệu phân loại hình ảnh Fashion-MNIST (:numref:`sec_fashion_mnist`).

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
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

Nhớ lại rằng Fashion-MNIST chứa 10 lớp, và rằng mỗi hình ảnh bao gồm một $28 \times 28 = 784$ lưới giá trị điểm ảnh xám. Một lần nữa, chúng ta sẽ bỏ qua cấu trúc không gian giữa các pixel cho bây giờ, vì vậy chúng ta có thể nghĩ về điều này chỉ đơn giản là một tập dữ liệu phân loại với 784 tính năng đầu vào và 10 lớp. Để bắt đầu, chúng ta sẽ [** triển khai MLP với một lớp ẩn và 256 đơn vị ẩn. **] Lưu ý rằng chúng ta có thể coi cả hai đại lượng này là siêu tham số. Thông thường, chúng ta chọn độ rộng lớp trong quyền hạn của 2, mà có xu hướng được tính toán hiệu quả vì cách bộ nhớ được phân bổ và giải quyết trong phần cứng. 

Một lần nữa, chúng tôi sẽ đại diện cho các thông số của chúng tôi với một số hàng chục. Lưu ý rằng * cho mỗi lớp*, chúng ta phải theo dõi một ma trận trọng lượng và một vector thiên vị. Như mọi khi, chúng tôi phân bổ bộ nhớ cho gradient của sự mất mát đối với các tham số này.

```{.python .input}
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens))
b1 = np.zeros(num_hiddens)
W2 = np.random.normal(scale=0.01, size=(num_hiddens, num_outputs))
b2 = np.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
```

```{.python .input}
#@tab tensorflow
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = tf.Variable(tf.random.normal(
    shape=(num_inputs, num_hiddens), mean=0, stddev=0.01))
b1 = tf.Variable(tf.zeros(num_hiddens))
W2 = tf.Variable(tf.random.normal(
    shape=(num_hiddens, num_outputs), mean=0, stddev=0.01))
b2 = tf.Variable(tf.random.normal([num_outputs], stddev=.01))

params = [W1, b1, W2, b2]
```

## Chức năng kích hoạt

Để đảm bảo rằng chúng ta biết mọi thứ hoạt động như thế nào, chúng ta sẽ [** triển khai kích hoạt ReLU**] bằng cách sử dụng hàm tối đa thay vì gọi trực tiếp hàm `relu` tích hợp sẵn.

```{.python .input}
def relu(X):
    return np.maximum(X, 0)
```

```{.python .input}
#@tab pytorch
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

```{.python .input}
#@tab tensorflow
def relu(X):
    return tf.math.maximum(X, 0)
```

## Mô hình

Bởi vì chúng ta đang bỏ qua cấu trúc không gian, chúng ta `reshape` mỗi hình ảnh hai chiều thành một vector phẳng có chiều dài `num_inputs`. Cuối cùng, chúng tôi (**triển khai mô hình của chúng tôi**) chỉ với một vài dòng code.

```{.python .input}
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(np.dot(X, W1) + b1)
    return np.dot(H, W2) + b2
```

```{.python .input}
#@tab pytorch
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(X@W1 + b1)  # Here '@' stands for matrix multiplication
    return (H@W2 + b2)
```

```{.python .input}
#@tab tensorflow
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(tf.matmul(X, W1) + b1)
    return tf.matmul(H, W2) + b2
```

## Chức năng mất

Để đảm bảo tính ổn định số, và bởi vì chúng tôi đã triển khai chức năng softmax từ đầu (:numref:`sec_softmax_scratch`), chúng tôi tận dụng chức năng tích hợp từ các API cấp cao để tính toán sự mất mát softmax và cross-entropy. Nhớ lại cuộc thảo luận trước đó của chúng tôi về những phức tạp này trong :numref:`subsec_softmax-implementation-revisited`. Chúng tôi khuyến khích người đọc quan tâm kiểm tra mã nguồn cho chức năng mất mát để đào sâu kiến thức của họ về chi tiết thực hiện.

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss(reduction='none')
```

```{.python .input}
#@tab tensorflow
def loss(y_hat, y):
    return tf.losses.sparse_categorical_crossentropy(
        y, y_hat, from_logits=True)
```

## Đào tạo

May mắn thay, [** vòng đào tạo cho MLP hoàn toàn giống như đối với hồi quy softmax.**] Tận dụng gói `d2l` một lần nữa, chúng tôi gọi hàm `train_ch3` (xem :numref:`sec_softmax_scratch`), đặt số epochs thành 10 và tỷ lệ học tập là 0.1.

```{.python .input}
num_epochs, lr = 10, 0.1
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              lambda batch_size: d2l.sgd(params, lr, batch_size))
```

```{.python .input}
#@tab pytorch
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr = 10, 0.1
updater = d2l.Updater([W1, W2, b1, b2], lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

Để đánh giá mô hình đã học, chúng tôi [** áp dụng nó trên một số dữ liệu thử nghiệm**].

```{.python .input}
#@tab all
d2l.predict_ch3(net, test_iter)
```

## Tóm tắt

* Chúng tôi thấy rằng việc thực hiện một MLP đơn giản là dễ dàng, ngay cả khi thực hiện thủ công.
* Tuy nhiên, với một số lượng lớn các lớp, việc thực hiện MLP từ đầu vẫn có thể trở nên lộn xộn (ví dụ, đặt tên và theo dõi các thông số của mô hình của chúng tôi).

## Bài tập

1. Thay đổi giá trị của hyperparameters `num_hiddens` và xem siêu tham số này ảnh hưởng như thế nào đến kết quả của bạn. Xác định giá trị tốt nhất của siêu tham số này, giữ cho tất cả những người khác không đổi.
1. Hãy thử thêm một layer ẩn bổ sung để xem nó ảnh hưởng đến kết quả như thế nào.
1. Làm thế nào để thay đổi tỷ lệ học tập làm thay đổi kết quả của bạn? Sửa chữa kiến trúc mô hình và các siêu tham số khác (bao gồm số thời đại), tỷ lệ học tập nào mang lại cho bạn kết quả tốt nhất?
1. Kết quả tốt nhất bạn có thể nhận được bằng cách tối ưu hóa tất cả các siêu tham số (tốc độ học tập, số lượng kỷ nguyên, số lớp ẩn, số lượng đơn vị ẩn trên mỗi lớp) cùng nhau?
1. Mô tả lý do tại sao việc đối phó với nhiều siêu tham số khó khăn hơn nhiều.
1. Chiến lược thông minh nhất bạn có thể nghĩ đến để cấu trúc tìm kiếm trên nhiều siêu tham số là gì?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/92)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/93)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/227)
:end_tab:
