# Quản lý tham số

Khi chúng tôi đã chọn một kiến trúc và đặt các siêu tham số của mình, chúng tôi tiến hành vòng đào tạo, nơi mục tiêu của chúng tôi là tìm các giá trị tham số giảm thiểu chức năng mất mát của chúng tôi. Sau khi đào tạo, chúng tôi sẽ cần các thông số này để đưa ra dự đoán trong tương lai. Ngoài ra, đôi khi chúng tôi sẽ muốn trích xuất các tham số để sử dụng lại chúng trong một số bối cảnh khác, để lưu mô hình của chúng tôi vào đĩa để nó có thể được thực thi trong phần mềm khác hoặc để kiểm tra với hy vọng đạt được sự hiểu biết khoa học. 

Hầu hết thời gian, chúng ta sẽ có thể bỏ qua các chi tiết nitty-gritty về cách các tham số được khai báo và thao tác, dựa vào các khuôn khổ học sâu để thực hiện việc nâng nặng. Tuy nhiên, khi chúng ta di chuyển ra khỏi các kiến trúc xếp chồng lên nhau với các lớp tiêu chuẩn, đôi khi chúng ta sẽ cần phải vào cỏ dại khai báo và thao tác các tham số. Trong phần này, chúng tôi đề cập đến những điều sau: 

* Truy cập các tham số để gỡ lỗi, chẩn đoán và trực quan hóa.
* Khởi tạo tham số.
* Chia sẻ các thông số trên các thành phần mô hình khác nhau.

(**Chúng tôi bắt đầu bằng cách tập trung vào MLP với một lớp ẩn. **)

```{.python .input}
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Use the default initialization method

X = np.random.uniform(size=(2, 4))
net(X)  # Forward computation
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X)
```

## [**Truy cập tham số**]

Hãy để chúng tôi bắt đầu với cách truy cập các tham số từ các mô hình mà bạn đã biết. Khi một mô hình được định nghĩa thông qua lớp `Sequential`, trước tiên chúng ta có thể truy cập bất kỳ lớp nào bằng cách lập chỉ mục vào mô hình như thể nó là một danh sách. Các tham số của mỗi lớp được đặt thuận tiện trong thuộc tính của nó. Chúng ta có thể kiểm tra các tham số của lớp được kết nối hoàn toàn thứ hai như sau.

```{.python .input}
print(net[1].params)
```

```{.python .input}
#@tab pytorch
print(net[2].state_dict())
```

```{.python .input}
#@tab tensorflow
print(net.layers[2].weights)
```

Đầu ra cho chúng ta biết một vài điều quan trọng. Đầu tiên, lớp kết nối hoàn toàn này chứa hai tham số, tương ứng với trọng lượng và thành kiến của lớp đó, tương ứng. Cả hai đều được lưu trữ dưới dạng phao chính xác đơn (float32). Lưu ý rằng tên của các tham số cho phép chúng ta xác định duy nhất các tham số của từng lớp, ngay cả trong một mạng chứa hàng trăm lớp. 

### [**Tham số được nhắm mục tiêu**]

Lưu ý rằng mỗi tham số được biểu diễn dưới dạng một đối tượng của lớp tham số. Để làm bất cứ điều gì hữu ích với các tham số, trước tiên chúng ta cần truy cập các giá trị số cơ bản. Có một số cách để làm điều này. Một số đơn giản hơn trong khi những người khác nói chung hơn. Đoạn code sau trích xuất sự thiên vị từ lớp mạng nơ-ron thứ hai, nó trả về một đối tượng lớp tham số, và truy cập thêm giá trị của tham số đó.

```{.python .input}
print(type(net[1].bias))
print(net[1].bias)
print(net[1].bias.data())
```

```{.python .input}
#@tab pytorch
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
```

```{.python .input}
#@tab tensorflow
print(type(net.layers[2].weights[1]))
print(net.layers[2].weights[1])
print(tf.convert_to_tensor(net.layers[2].weights[1]))
```

:begin_tab:`mxnet,pytorch`
Tham số là các đối tượng phức tạp, chứa các giá trị, độ dốc và thông tin bổ sung. Đó là lý do tại sao chúng ta cần yêu cầu giá trị một cách rõ ràng. 

Ngoài giá trị, mỗi tham số cũng cho phép chúng ta truy cập gradient. Bởi vì chúng tôi chưa gọi backpropagation cho mạng này được nêu ra, nó ở trạng thái ban đầu của nó.
:end_tab:

```{.python .input}
net[1].weight.grad()
```

```{.python .input}
#@tab pytorch
net[2].weight.grad == None
```

### [**Tất cả các thông số tại một lần**]

Khi chúng ta cần thực hiện các thao tác trên tất cả các tham số, truy cập chúng từng cái một có thể phát triển tẻ nhạt. Tình hình có thể phát triển đặc biệt khó sử dụng khi chúng ta làm việc với các khối phức tạp hơn (ví dụ, các khối lồng nhau), vì chúng ta cần đệ quy qua toàn bộ cây để trích xuất các tham số của từng khối phụ. Dưới đây chúng tôi chứng minh việc truy cập các tham số của lớp được kết nối hoàn toàn đầu tiên so với truy cập tất cả các lớp.

```{.python .input}
print(net[0].collect_params())
print(net.collect_params())
```

```{.python .input}
#@tab pytorch
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
```

```{.python .input}
#@tab tensorflow
print(net.layers[1].weights)
print(net.get_weights())
```

Điều này cung cấp cho chúng tôi một cách khác để truy cập các tham số của mạng như sau.

```{.python .input}
net.collect_params()['dense1_bias'].data()
```

```{.python .input}
#@tab pytorch
net.state_dict()['2.bias'].data
```

```{.python .input}
#@tab tensorflow
net.get_weights()[1]
```

### [**Thu thập các thông số từ các khối lồng nhựa**]

Chúng ta hãy xem làm thế nào các quy ước đặt tên tham số hoạt động nếu chúng ta tổ nhiều khối bên trong nhau. Đối với điều đó đầu tiên chúng ta xác định một hàm tạo ra các khối (một nhà máy khối, có thể nói) và sau đó kết hợp các khối bên trong nhưng lớn hơn.

```{.python .input}
def block1():
    net = nn.Sequential()
    net.add(nn.Dense(32, activation='relu'))
    net.add(nn.Dense(16, activation='relu'))
    return net

def block2():
    net = nn.Sequential()
    for _ in range(4):
        # Nested here
        net.add(block1())
    return net

rgnet = nn.Sequential()
rgnet.add(block2())
rgnet.add(nn.Dense(10))
rgnet.initialize()
rgnet(X)
```

```{.python .input}
#@tab pytorch
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # Nested here
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
```

```{.python .input}
#@tab tensorflow
def block1(name):
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4, activation=tf.nn.relu)],
        name=name)

def block2():
    net = tf.keras.Sequential()
    for i in range(4):
        # Nested here
        net.add(block1(name=f'block-{i}'))
    return net

rgnet = tf.keras.Sequential()
rgnet.add(block2())
rgnet.add(tf.keras.layers.Dense(1))
rgnet(X)
```

Bây giờ [** chúng tôi đã thiết kế mạng, chúng ta hãy xem nó được tổ chức như thế nào.**]

```{.python .input}
print(rgnet.collect_params)
print(rgnet.collect_params())
```

```{.python .input}
#@tab pytorch
print(rgnet)
```

```{.python .input}
#@tab tensorflow
print(rgnet.summary())
```

Kể từ khi các lớp được phân cấp lồng nhau, chúng ta cũng có thể truy cập chúng như thể lập chỉ mục thông qua các danh sách lồng nhau. Ví dụ, chúng ta có thể truy cập khối chính đầu tiên, bên trong đó khối phụ thứ hai và trong đó sự thiên vị của lớp đầu tiên, với như sau.

```{.python .input}
rgnet[0][1][0].bias.data()
```

```{.python .input}
#@tab pytorch
rgnet[0][1][0].bias.data
```

```{.python .input}
#@tab tensorflow
rgnet.layers[0].layers[1].layers[1].weights[1]
```

## Khởi tạo tham số

Bây giờ chúng ta đã biết cách truy cập các tham số, chúng ta hãy xem cách khởi tạo chúng đúng cách. Chúng tôi đã thảo luận về sự cần thiết phải khởi tạo thích hợp trong :numref:`sec_numerical_stability`. Khung học sâu cung cấp các khởi tạo ngẫu nhiên mặc định cho các lớp của nó. Tuy nhiên, chúng tôi thường muốn khởi tạo trọng lượng của mình theo các giao thức khác nhau. Khung cung cấp các giao thức được sử dụng phổ biến nhất và cũng cho phép tạo một trình khởi tạo tùy chỉnh.

:begin_tab:`mxnet`
Theo mặc định, MXNet khởi tạo các tham số trọng lượng bằng cách vẽ ngẫu nhiên từ một phân phối thống nhất $U(-0.07, 0.07)$, xóa các tham số thiên vị về 0. mô-đun `init` của MXNet cung cấp nhiều phương pháp khởi tạo cài sẵn.
:end_tab:

:begin_tab:`pytorch`
Theo mặc định, PyTorch khởi tạo các ma trận trọng lượng và thiên vị đồng đều bằng cách vẽ từ một phạm vi được tính toán theo kích thước đầu vào và đầu ra. mô-đun `nn.init` của PyTorch cung cấp một loạt các phương pháp khởi tạo cài sẵn.
:end_tab:

:begin_tab:`tensorflow`
Theo mặc định, Keras khởi tạo các ma trận trọng lượng đồng đều bằng cách vẽ từ một phạm vi được tính theo kích thước đầu vào và đầu ra, và các tham số thiên vị đều được đặt thành 0. TensorFlow cung cấp một loạt các phương pháp khởi tạo cả trong mô-đun gốc và mô-đun `keras.initializers`.
:end_tab:

### [** Khởi tạo tích hợp**]

Hãy để chúng tôi bắt đầu bằng cách gọi trên built-in initializers. Mã dưới đây khởi tạo tất cả các tham số trọng lượng dưới dạng biến ngẫu nhiên Gaussian với độ lệch chuẩn 0,01, trong khi các tham số thiên vị bị xóa về 0.

```{.python .input}
# Here `force_reinit` ensures that parameters are freshly initialized even if
# they were already initialized previously
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
#@tab pytorch
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1)])

net(X)
net.weights[0], net.weights[1]
```

Chúng ta cũng có thể khởi tạo tất cả các tham số thành một giá trị không đổi cho trước (ví dụ, 1).

```{.python .input}
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
#@tab pytorch
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.Constant(1),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1),
])

net(X)
net.weights[0], net.weights[1]
```

[**Chúng tôi cũng có thể áp dụng các trình khởi tạo khác nhau cho các khối nhất định**] Ví dụ, bên dưới chúng ta khởi tạo lớp đầu tiên với trình khởi tạo Xavier và khởi tạo lớp thứ hai thành giá trị không đổi là 42.

```{.python .input}
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
print(net[0].weight.data()[0])
print(net[1].weight.data())
```

```{.python .input}
#@tab pytorch
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.Constant(42)),
])

net(X)
print(net.layers[1].weights[0])
print(net.layers[2].weights[0])
```

### [** Initializationtùy chỉnh**]

Đôi khi, các phương pháp khởi tạo chúng ta cần không được cung cấp bởi khung học sâu. Trong ví dụ dưới đây, chúng ta định nghĩa một bộ khởi tạo cho bất kỳ tham số trọng lượng nào $w$ bằng cách sử dụng phân phối lạ sau: 

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \text{ with probability } \frac{1}{4} \\
            0    & \text{ with probability } \frac{1}{2} \\
        U(-10, -5) & \text{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

:begin_tab:`mxnet`
Ở đây chúng ta định nghĩa một lớp con của lớp `Initializer`. Thông thường, chúng ta chỉ cần thực hiện hàm `_init_weight` mà lấy một đối số tensor (`data`) và gán cho nó các giá trị khởi tạo mong muốn.
:end_tab:

:begin_tab:`pytorch`
Một lần nữa, chúng tôi triển khai một hàm `my_init` để áp dụng cho `net`.
:end_tab:

:begin_tab:`tensorflow`
Ở đây chúng ta định nghĩa một lớp con của `Initializer` và thực hiện hàm `__call__` trả về một tensor mong muốn cho hình dạng và kiểu dữ liệu.
:end_tab:

```{.python .input}
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[:2]
```

```{.python .input}
#@tab pytorch
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) 
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

```{.python .input}
#@tab tensorflow
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        data=tf.random.uniform(shape, -10, 10, dtype=dtype)
        factor=(tf.abs(data) >= 5)
        factor=tf.cast(factor, tf.float32)
        return data * factor        

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1),
])

net(X)
print(net.layers[1].weights[0])
```

Lưu ý rằng chúng tôi luôn có tùy chọn đặt tham số trực tiếp.

```{.python .input}
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

```{.python .input}
#@tab pytorch
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

```{.python .input}
#@tab tensorflow
net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
net.layers[1].weights[0][0, 0].assign(42)
net.layers[1].weights[0]
```

:begin_tab:`mxnet`
Lưu ý cho người dùng nâng cao: nếu bạn muốn điều chỉnh các tham số trong phạm vi `autograd`, bạn cần sử dụng `set_data` để tránh nhầm lẫn cơ chế phân biệt tự động.
:end_tab:

## [**Tham số Tied**]

Thông thường, chúng tôi muốn chia sẻ các tham số trên nhiều lớp. Hãy để chúng tôi xem làm thế nào để làm điều này một cách thanh lịch. Sau đây, chúng tôi phân bổ một lớp dày đặc và sau đó sử dụng các tham số của nó đặc biệt để đặt các thông số của một lớp khác.

```{.python .input}
net = nn.Sequential()
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X)

# Check whether the parameters are the same
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

```{.python .input}
#@tab pytorch
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# Check whether the parameters are the same
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[2].weight.data[0] == net[4].weight.data[0])
```

```{.python .input}
#@tab tensorflow
# tf.keras behaves a bit differently. It removes the duplicate layer
# automatically
shared = tf.keras.layers.Dense(4, activation=tf.nn.relu)
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    shared,
    shared,
    tf.keras.layers.Dense(1),
])

net(X)
# Check whether the parameters are different
print(len(net.layers) == 3)
```

:begin_tab:`mxnet,pytorch`
Ví dụ này cho thấy các tham số của lớp thứ hai và thứ ba được gắn. Chúng không chỉ bằng nhau, chúng được đại diện bởi cùng một tensor chính xác. Do đó, nếu chúng ta thay đổi một trong các tham số, cái kia cũng thay đổi. Bạn có thể tự hỏi, khi tham số được gắn những gì sẽ xảy ra với gradient? Vì các tham số mô hình chứa gradient, các gradient của lớp ẩn thứ hai và lớp ẩn thứ ba được thêm vào với nhau trong quá trình truyền ngược.
:end_tab:

## Tóm tắt

* Chúng tôi có một số cách để truy cập, khởi tạo và buộc các tham số mô hình.
* Chúng ta có thể sử dụng khởi tạo tùy chỉnh.

## Bài tập

1. Sử dụng mô hình `FancyMLP` được xác định trong :numref:`sec_model_construction` và truy cập các tham số của các lớp khác nhau.
1. Nhìn vào tài liệu mô-đun khởi tạo để khám phá các trình khởi tạo khác nhau.
1. Xây dựng một MLP chứa một lớp tham số được chia sẻ và đào tạo nó. Trong quá trình đào tạo, quan sát các thông số mô hình và gradient của mỗi lớp.
1. Tại sao chia sẻ các thông số là một ý tưởng tốt?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/56)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/57)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/269)
:end_tab:
