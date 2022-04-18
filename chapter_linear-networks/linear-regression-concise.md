# Thực hiện ngắn gọn của hồi quy tuyến tính
:label:`sec_linear_concise`

Sự quan tâm rộng rãi và mãnh liệt trong việc học sâu trong nhiều năm qua đã truyền cảm hứng cho các công ty, học giả và những người có sở thích phát triển một loạt các khuôn khổ nguồn mở trưởng thành để tự động hóa công việc lặp đi lặp lại của việc thực hiện các thuật toán học tập dựa trên độ dốc. Năm :numref:`sec_linear_scratch`, chúng tôi chỉ dựa vào (i) hàng chục để lưu trữ dữ liệu và đại số tuyến tính; và (ii) tự động phân biệt để tính toán độ dốc. Trong thực tế, bởi vì các bộ lặp dữ liệu, chức năng mất mát, tối ưu hóa và các lớp mạng thần kinh rất phổ biến, các thư viện hiện đại cũng triển khai các thành phần này cho chúng ta. 

Trong phần này, (**chúng tôi sẽ chỉ cho bạn cách thực hiện mô hình hồi quy tuyến tính**) từ :numref:`sec_linear_scratch` (**chính xác bằng cách sử dụng APIs cấp cao**) của các framework học sâu. 

## Tạo tập dữ liệu

Để bắt đầu, chúng ta sẽ tạo ra cùng một tập dữ liệu như trong :numref:`sec_linear_scratch`.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch.utils import data
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
true_w = d2l.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```

## Đọc tập dữ liệu

Thay vì lăn iterator của riêng mình, chúng ta có thể [** gọi API hiện có trong một framework để đọc dữ liệu**] Chúng tôi vượt qua `features` và `labels` làm đối số và chỉ định `batch_size` khi khởi tạo một đối tượng lặp dữ liệu. Bên cạnh đó, giá trị boolean `is_train` cho biết liệu chúng ta có muốn đối tượng lặp dữ liệu xáo trộn dữ liệu trên mỗi kỷ nguyên hay không (đi qua bộ dữ liệu).

```{.python .input}
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a Gluon data iterator."""
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)
```

```{.python .input}
#@tab pytorch
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
```

```{.python .input}
#@tab tensorflow
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a TensorFlow data iterator."""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset
```

```{.python .input}
#@tab all
batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

Bây giờ chúng ta có thể sử dụng `data_iter` theo cách tương tự như chúng ta gọi hàm `data_iter` trong :numref:`sec_linear_scratch`. Để xác minh rằng nó đang hoạt động, chúng ta có thể đọc và in minibatch đầu tiên của các ví dụ. So sánh với :numref:`sec_linear_scratch`, ở đây chúng ta sử dụng `iter` để xây dựng một iterator Python và sử dụng `next` để lấy mục đầu tiên từ iterator.

```{.python .input}
#@tab all
next(iter(data_iter))
```

## Xác định mô hình

Khi chúng tôi thực hiện hồi quy tuyến tính từ đầu vào năm :numref:`sec_linear_scratch`, chúng tôi đã xác định các tham số mô hình của mình một cách rõ ràng và mã hóa các phép tính để tạo ra đầu ra bằng cách sử dụng các phép toán đại số tuyến tính cơ bản. Bạn * nên biết làm thế nào để làm điều này. Nhưng một khi mô hình của bạn trở nên phức tạp hơn, và một khi bạn phải làm điều này gần như mỗi ngày, bạn sẽ rất vui vì được hỗ trợ. Tình hình tương tự như mã hóa blog của riêng bạn từ đầu. Làm điều đó một hoặc hai lần là bổ ích và hướng dẫn, nhưng bạn sẽ là một nhà phát triển web tệ hại nếu mỗi khi bạn cần một blog bạn đã dành một tháng để phát minh lại bánh xe. 

Đối với các phép toán chuẩn, chúng ta có thể [** sử dụng các lớp được xác định trước của framework, **] cho phép chúng ta tập trung đặc biệt vào các layer dùng để xây dựng mô hình chứ không phải tập trung vào việc triển khai. Trước tiên chúng ta sẽ xác định một biến mô hình `net`, sẽ đề cập đến một phiên bản của lớp `Sequential`. Lớp `Sequential` định nghĩa một container cho nhiều lớp sẽ được xích lại với nhau. Cho dữ liệu đầu vào, một trường hợp `Sequential` truyền nó qua lớp đầu tiên, lần lượt đi qua đầu ra dưới dạng đầu vào của lớp thứ hai và vân vân. Trong ví dụ sau, mô hình của chúng tôi chỉ bao gồm một lớp, vì vậy chúng tôi không thực sự cần `Sequential`. Nhưng vì gần như tất cả các mô hình trong tương lai của chúng tôi sẽ liên quan đến nhiều lớp, dù sao chúng tôi sẽ sử dụng nó chỉ để làm quen với bạn với quy trình làm việc tiêu chuẩn nhất. 

Nhớ lại kiến trúc của một mạng một lớp như thể hiện trong :numref:`fig_single_neuron`. Lớp được cho là * kết nối đầy đủ* bởi vì mỗi đầu vào của nó được kết nối với mỗi đầu ra của nó bằng phương pháp nhân ma thuật-vector.

:begin_tab:`mxnet`
Trong Gluon, lớp kết nối hoàn toàn được định nghĩa trong lớp `Dense`. Vì chúng ta chỉ muốn tạo ra một đầu ra vô hướng duy nhất, chúng ta đặt số đó thành 1. 

Điều đáng chú ý là, để thuận tiện, Gluon không yêu cầu chúng ta chỉ định hình dạng đầu vào cho mỗi lớp. Vì vậy, ở đây, chúng ta không cần phải nói với Gluon có bao nhiêu đầu vào đi vào lớp tuyến tính này. Khi lần đầu tiên chúng ta cố gắng truyền dữ liệu thông qua mô hình của mình, ví dụ, khi chúng ta thực thi `net(X)` sau đó, Gluon sẽ tự động suy ra số lượng đầu vào cho mỗi lớp. Chúng tôi sẽ mô tả cách thức hoạt động chi tiết hơn sau.
:end_tab:

:begin_tab:`pytorch`
Trong PyTorch, lớp kết nối hoàn toàn được định nghĩa trong lớp `Linear`. Lưu ý rằng chúng tôi đã thông qua hai đối số vào `nn.Linear`. Cái đầu tiên xác định kích thước đối tượng đầu vào, đó là 2, và thứ hai là kích thước tính năng đầu ra, đó là một vô hướng duy nhất và do đó 1.
:end_tab:

:begin_tab:`tensorflow`
Trong Keras, lớp kết nối hoàn toàn được định nghĩa trong lớp `Dense`. Vì chúng ta chỉ muốn tạo ra một đầu ra vô hướng duy nhất, chúng ta đặt số đó thành 1. 

Điều đáng chú ý là, để thuận tiện, Keras không yêu cầu chúng ta chỉ định hình dạng đầu vào cho mỗi lớp. Vì vậy, ở đây, chúng ta không cần phải nói Keras có bao nhiêu đầu vào đi vào lớp tuyến tính này. Khi lần đầu tiên chúng ta cố gắng truyền dữ liệu thông qua mô hình của mình, ví dụ, khi chúng ta thực thi `net(X)` sau đó, Keras sẽ tự động suy ra số lượng đầu vào cho mỗi lớp. Chúng tôi sẽ mô tả cách thức hoạt động chi tiết hơn sau.
:end_tab:

```{.python .input}
# `nn` is an abbreviation for neural networks
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1))
```

```{.python .input}
#@tab pytorch
# `nn` is an abbreviation for neural networks
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
```

```{.python .input}
#@tab tensorflow
# `keras` is the high-level API for TensorFlow
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1))
```

## Khởi tạo các tham số mô hình

Trước khi sử dụng `net`, chúng ta cần (**khởi tạo các tham số model**) chẳng hạn như trọng lượng và thiên vị trong mô hình hồi quy tuyến tính. Các khuôn khổ học sâu thường có cách xác định trước để khởi tạo các tham số. Ở đây chúng tôi chỉ định rằng mỗi tham số trọng lượng nên được lấy mẫu ngẫu nhiên từ phân phối bình thường với 0 trung bình và độ lệch chuẩn 0,01. Tham số thiên vị sẽ được khởi tạo thành 0.

:begin_tab:`mxnet`
Chúng tôi sẽ nhập mô-đun `initializer` từ MXNet. mô-đun này cung cấp các phương pháp khác nhau để khởi tạo tham số mô hình. Gluon làm cho `init` có sẵn dưới dạng phím tắt (viết tắt) để truy cập gói `initializer`. Chúng tôi chỉ chỉ định cách khởi tạo trọng lượng bằng cách gọi `init.Normal(sigma=0.01)`. Các tham số thiên vị được khởi tạo thành 0 theo mặc định.
:end_tab:

:begin_tab:`pytorch`
Như chúng ta đã chỉ định kích thước đầu vào và đầu ra khi xây dựng `nn.Linear`, bây giờ chúng ta có thể truy cập trực tiếp các tham số để chỉ định các giá trị ban đầu của chúng. Đầu tiên chúng tôi xác định vị trí lớp `net[0]`, là lớp đầu tiên trong mạng, sau đó sử dụng các phương thức `weight.data` và `bias.data` để truy cập các tham số. Tiếp theo chúng ta sử dụng các phương pháp thay thế `normal_` và `fill_` để ghi đè lên các giá trị tham số.
:end_tab:

:begin_tab:`tensorflow`
mô-đun `initializers` trong TensorFlow cung cấp các phương pháp khác nhau để khởi tạo tham số mô hình. Cách dễ nhất để chỉ định phương thức khởi tạo trong Keras là khi tạo lớp bằng cách chỉ định `kernel_initializer`. Ở đây chúng tôi tái tạo lại `net` một lần nữa.
:end_tab:

```{.python .input}
from mxnet import init
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```

```{.python .input}
#@tab tensorflow
initializer = tf.initializers.RandomNormal(stddev=0.01)
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))
```

:begin_tab:`mxnet`
Mã ở trên có thể trông đơn giản nhưng bạn nên lưu ý rằng một cái gì đó kỳ lạ đang xảy ra ở đây. Chúng tôi đang khởi tạo các tham số cho một mạng mặc dù Gluon chưa biết đầu vào sẽ có bao nhiêu kích thước! Nó có thể là 2 như trong ví dụ của chúng tôi hoặc nó có thể là 2000. Gluon cho phép chúng tôi thoát khỏi điều này bởi vì đằng sau hiện trường, việc khởi tạo thực sự là * hoãn lại*. Việc khởi tạo thực sự sẽ chỉ diễn ra khi chúng tôi lần đầu tiên cố gắng truyền dữ liệu qua mạng. Chỉ cần cẩn thận để nhớ rằng vì các tham số chưa được khởi tạo, chúng tôi không thể truy cập hoặc thao tác chúng.
:end_tab:

:begin_tab:`pytorch`

:end_tab:

:begin_tab:`tensorflow`
Mã ở trên có thể trông đơn giản nhưng bạn nên lưu ý rằng một cái gì đó kỳ lạ đang xảy ra ở đây. Chúng tôi đang khởi tạo các tham số cho một mạng mặc dù Keras chưa biết có bao nhiêu kích thước đầu vào sẽ có! Nó có thể là 2 như trong ví dụ của chúng tôi hoặc nó có thể là 2000. Keras cho phép chúng tôi thoát khỏi điều này bởi vì đằng sau hậu trường, việc khởi tạo thực sự là * hoãn lại*. Việc khởi tạo thực sự sẽ chỉ diễn ra khi chúng tôi lần đầu tiên cố gắng truyền dữ liệu qua mạng. Chỉ cần cẩn thận để nhớ rằng vì các tham số chưa được khởi tạo, chúng tôi không thể truy cập hoặc thao tác chúng.
:end_tab:

## Xác định chức năng mất

:begin_tab:`mxnet`
Trong Gluon, mô-đun `loss` xác định các chức năng mất mát khác nhau. Trong ví dụ này, chúng ta sẽ sử dụng việc thực hiện Gluon về tổn thất bình phương (`L2Loss`).
:end_tab:

:begin_tab:`pytorch`
[**Lớp `MSELoss` tính toán lỗi bình phương trung bình (không có hệ số $1/2$ trong :eqref:`eq_mse`) .**] Theo mặc định, nó trả về mức lỗ trung bình so với các ví dụ.
:end_tab:

:begin_tab:`tensorflow`
Lớp `MeanSquaredError` tính toán sai số bình phương trung bình (không có hệ số $1/2$ trong :eqref:`eq_mse`). Theo mặc định, nó trả về mức lỗ trung bình so với các ví dụ.
:end_tab:

```{.python .input}
loss = gluon.loss.L2Loss()
```

```{.python .input}
#@tab pytorch
loss = nn.MSELoss()
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.MeanSquaredError()
```

## Xác định thuật toán tối ưu hóa

:begin_tab:`mxnet`
Minibatch stochastic gradient descent là một công cụ tiêu chuẩn để tối ưu hóa các mạng thần kinh và do đó Gluon hỗ trợ nó cùng với một số biến thể trên thuật toán này thông qua lớp `Trainer` của nó. Khi chúng tôi khởi tạo `Trainer`, chúng tôi sẽ chỉ định các tham số để tối ưu hóa (có thể đạt được từ mô hình `net` của chúng tôi thông qua `net.collect_params()`), thuật toán tối ưu hóa chúng tôi muốn sử dụng (`sgd`) và từ điển các siêu tham số theo yêu cầu của thuật toán tối ưu hóa của chúng tôi. Minibatch stochastic gradient gốc chỉ yêu cầu chúng ta đặt giá trị `learning_rate`, được đặt thành 0,03 ở đây.
:end_tab:

:begin_tab:`pytorch`
Minibatch stochastic gradient descent là một công cụ tiêu chuẩn để tối ưu hóa các mạng thần kinh và do đó PyTorch hỗ trợ nó cùng với một số biến thể trên thuật toán này trong mô-đun `optim`. Khi chúng ta (**khởi tạo một phiên bản `SGD`**) chúng ta sẽ chỉ định các tham số để tối ưu hóa (có thể đạt được từ mạng của chúng tôi thông qua `net.parameters()`), với một từ điển các siêu tham số theo yêu cầu của thuật toán tối ưu hóa của chúng tôi. Minibatch stochastic gradient gốc chỉ yêu cầu chúng ta đặt giá trị `lr`, được đặt thành 0,03 ở đây.
:end_tab:

:begin_tab:`tensorflow`
Minibatch stochastic gradient descent là một công cụ tiêu chuẩn để tối ưu hóa mạng thần kinh và do đó Keras hỗ trợ nó cùng với một số biến thể trên thuật toán này trong mô-đun `optimizers`. Minibatch stochastic gradient descent chỉ yêu cầu chúng ta đặt giá trị `learning_rate`, được đặt thành 0,03 ở đây.
:end_tab:

```{.python .input}
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=0.03)
```

## Đào tạo

Bạn có thể nhận thấy rằng thể hiện mô hình của chúng tôi thông qua các API cấp cao của một khuôn khổ học sâu đòi hỏi tương đối ít dòng mã. Chúng tôi không phải phân bổ các thông số riêng lẻ, xác định chức năng mất mát của chúng tôi hoặc thực hiện minibatch stochastic gradient descent. Khi chúng tôi bắt đầu làm việc với các mô hình phức tạp hơn nhiều, lợi thế của API cấp cao sẽ tăng lên đáng kể. Tuy nhiên, một khi chúng tôi có tất cả các phần cơ bản tại chỗ, [** bản thân vòng đào tạo rất giống với những gì chúng tôi đã làm khi thực hiện mọi thứ từ vết xước.**] 

Để làm mới bộ nhớ của bạn: đối với một số kỷ nguyên, chúng tôi sẽ thực hiện một vượt qua toàn bộ dữ liệu (`train_data`), lặp đi lặp lại lấy một minibatch đầu vào và các nhãn chân lý mặt đất tương ứng. Đối với mỗi minibatch, chúng tôi trải qua các nghi thức sau: 

* Tạo ra dự đoán bằng cách gọi `net(X)` và tính toán tổn thất `l` (sự lan truyền về phía trước).
* Tính toán gradient bằng cách chạy backpropagation.
* Cập nhật các tham số mô hình bằng cách gọi trình tối ưu hóa của chúng tôi.

Để có biện pháp tốt, chúng tôi tính toán tổn thất sau mỗi kỷ nguyên và in nó để theo dõi tiến độ.

```{.python .input}
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l.mean().asnumpy():f}')
```

```{.python .input}
#@tab pytorch
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

```{.python .input}
#@tab tensorflow
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with tf.GradientTape() as tape:
            l = loss(net(X, training=True), y)
        grads = tape.gradient(l, net.trainable_variables)
        trainer.apply_gradients(zip(grads, net.trainable_variables))
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

Dưới đây, chúng ta [**so sánh các tham số mô hình học được bằng cách đào tạo về dữ liệu hữu hạn và các tham số thực tế**] đã tạo ra tập dữ liệu của chúng tôi. Để truy cập các tham số, trước tiên chúng ta truy cập vào lớp mà chúng ta cần từ `net` và sau đó truy cập vào trọng lượng và thiên vị của lớp đó. Như trong triển khai từ đầu của chúng tôi, lưu ý rằng các thông số ước tính của chúng tôi gần với các đối tác thực tế mặt đất của chúng.

```{.python .input}
w = net[0].weight.data()
print(f'error in estimating w: {true_w - d2l.reshape(w, true_w.shape)}')
b = net[0].bias.data()
print(f'error in estimating b: {true_b - b}')
```

```{.python .input}
#@tab pytorch
w = net[0].weight.data
print('error in estimating w:', true_w - d2l.reshape(w, true_w.shape))
b = net[0].bias.data
print('error in estimating b:', true_b - b)
```

```{.python .input}
#@tab tensorflow
w = net.get_weights()[0]
print('error in estimating w', true_w - d2l.reshape(w, true_w.shape))
b = net.get_weights()[1]
print('error in estimating b', true_b - b)
```

## Tóm tắt

:begin_tab:`mxnet`
* Sử dụng Gluon, chúng ta có thể thực hiện các mô hình chính xác hơn nhiều.
* Trong Gluon, mô-đun `data` cung cấp các công cụ để xử lý dữ liệu, mô-đun `nn` định nghĩa một số lượng lớn các lớp mạng thần kinh và mô-đun `loss` định nghĩa nhiều chức năng mất mát phổ biến.
* mô-đun của MXNet `initializer` cung cấp các phương pháp khác nhau để khởi tạo tham số mô hình.
* Kích thước và lưu trữ được tự động suy ra, nhưng hãy cẩn thận không cố gắng truy cập các tham số trước khi chúng được khởi tạo.
:end_tab:

:begin_tab:`pytorch`
* Sử dụng API cấp cao của PyTorch, chúng ta có thể triển khai các mô hình chính xác hơn nhiều.
* Trong PyTorch, mô-đun `data` cung cấp các công cụ để xử lý dữ liệu, mô-đun `nn` định nghĩa một số lượng lớn các lớp mạng thần kinh và các chức năng mất phổ biến.
* Chúng ta có thể khởi tạo các tham số bằng cách thay thế các giá trị của chúng bằng các phương thức kết thúc bằng `_`.
:end_tab:

:begin_tab:`tensorflow`
* Sử dụng API cấp cao của TensorFlow, chúng ta có thể triển khai các mô hình chính xác hơn nhiều.
* Trong TensorFlow, mô-đun `data` cung cấp các công cụ để xử lý dữ liệu, mô-đun `keras` định nghĩa một số lượng lớn các lớp mạng thần kinh và các chức năng mất phổ biến.
* mô-đun của TensorFlow `initializers` cung cấp các phương pháp khác nhau để khởi tạo tham số mô hình.
* Kích thước và lưu trữ được tự động suy ra (nhưng hãy cẩn thận không cố gắng truy cập các tham số trước khi chúng được khởi tạo).
:end_tab:

## Bài tập

:begin_tab:`mxnet`
1. Nếu chúng ta thay thế `l = loss(output, y)` bằng `l = loss(output, y).mean()`, chúng ta cần thay đổi `trainer.step(batch_size)` thành `trainer.step(1)` để mã hoạt động giống hệt nhau. Tại sao?
1. Xem lại tài liệu MXNet để xem những chức năng mất mát và phương pháp khởi tạo được cung cấp trong các mô-đun `gluon.loss` và `init`. Thay thế tổn thất bằng mất mát của Huber.
1. Làm thế nào để bạn truy cập gradient của `dense.weight`?

[Discussions](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
1. Nếu chúng ta thay thế `nn.MSELoss(reduction='sum')` bằng `nn.MSELoss()`, làm thế nào chúng ta có thể thay đổi tốc độ học tập để mã hoạt động giống hệt nhau. Tại sao?
1. Xem lại tài liệu PyTorch để xem các chức năng mất mát và phương pháp khởi tạo được cung cấp. Thay thế tổn thất bằng mất mát của Huber.
1. Làm thế nào để bạn truy cập gradient của `net[0].weight`?

[Discussions](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
1. Xem lại tài liệu TensorFlow để xem các chức năng mất mát và phương pháp khởi tạo nào được cung cấp. Thay thế tổn thất bằng mất mát của Huber.

[Discussions](https://discuss.d2l.ai/t/204)
:end_tab:
