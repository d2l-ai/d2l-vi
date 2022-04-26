# Trình biên dịch và phiên dịch
:label:`sec_hybridize`

Cho đến nay, cuốn sách này đã tập trung vào lập trình bắt buộc, làm cho việc sử dụng các tuyên bố như `print`, `+`, và `if` để thay đổi trạng thái của một chương trình. Hãy xem xét ví dụ sau đây của một chương trình bắt buộc đơn giản.

```{.python .input}
#@tab all
def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

print(fancy_func(1, 2, 3, 4))
```

Python là một ngôn ngữ*được giải thị*. Khi đánh giá hàm `fancy_func` trên, nó thực hiện các thao tác tạo nên cơ thể của hàm * theo trình tự *. Đó là, nó sẽ đánh giá `e = add(a, b)` và lưu trữ các kết quả dưới dạng biến `e`, do đó thay đổi trạng thái của chương trình. Hai câu lệnh tiếp theo `f = add(c, d)` và `g = add(e, f)` sẽ được thực thi tương tự, thực hiện bổ sung và lưu trữ các kết quả dưới dạng biến. :numref:`fig_compute_graph` minh họa luồng dữ liệu. 

![Data flow in an imperative program.](../img/computegraph.svg)
:label:`fig_compute_graph`

Mặc dù lập trình bắt buộc là thuận tiện, nó có thể không hiệu quả. Một mặt, ngay cả khi hàm `add` được gọi nhiều lần trong suốt `fancy_func`, Python sẽ thực hiện ba cuộc gọi hàm riêng lẻ. Nếu chúng được thực thi, giả sử, trên GPU (hoặc thậm chí trên nhiều GPU), chi phí phát sinh từ trình thông dịch Python có thể trở nên áp đảo. Hơn nữa, nó sẽ cần phải lưu các giá trị biến của `e` và `f` cho đến khi tất cả các câu lệnh trong `fancy_func` đã được thực thi. Điều này là do chúng ta không biết liệu các biến `e` và `f` sẽ được sử dụng bởi các phần khác của chương trình sau khi các câu lệnh `e = add(a, b)` và `f = add(c, d)` được thực thi. 

## Lập trình tượng trưng

Hãy xem xét phương pháp thay thế, *lập trình tượng trưng, trong đó tính toán thường chỉ được thực hiện một khi quá trình đã được xác định đầy đủ. Chiến lược này được sử dụng bởi nhiều khuôn khổ học sâu, bao gồm Theano và TensorFlow (sau này đã có được các phần mở rộng bắt buộc). Nó thường bao gồm các bước sau: 

1. Xác định các hoạt động sẽ được thực thi.
1. Biên dịch các hoạt động thành một chương trình thực thi.
1. Cung cấp các đầu vào cần thiết và gọi chương trình được biên dịch để thực hiện.

Điều này cho phép tối ưu hóa một lượng đáng kể. Đầu tiên, chúng ta có thể bỏ qua trình thông dịch Python trong nhiều trường hợp, do đó loại bỏ một nút cổ chai hiệu suất có thể trở nên đáng kể trên nhiều GPU nhanh được ghép nối với một luồng Python duy nhất trên CPU. Thứ hai, một trình biên dịch có thể tối ưu hóa và viết lại mã trên vào `print((1 + 2) + (3 + 4))` hoặc thậm chí `print(10)`. Điều này là có thể kể từ khi một trình biên dịch được để xem mã đầy đủ trước khi biến nó thành hướng dẫn máy. Ví dụ, nó có thể giải phóng bộ nhớ (hoặc không bao giờ phân bổ nó) bất cứ khi nào một biến không còn cần thiết nữa. Hoặc nó có thể biến đổi mã hoàn toàn thành một phần tương đương. Để có được một ý tưởng tốt hơn, hãy xem xét mô phỏng sau đây của lập trình bắt buộc (nó là Python sau tất cả) dưới đây.

```{.python .input}
#@tab all
def add_():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_():
    return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'

prog = evoke_()
print(prog)
y = compile(prog, '', 'exec')
exec(y)
```

Sự khác biệt giữa lập trình bắt buộc (giải thích) và lập trình tượng trưng như sau: 

* Lập trình bắt buộc dễ dàng hơn. Khi lập trình bắt buộc được sử dụng trong Python, phần lớn mã là đơn giản và dễ viết. Nó cũng dễ dàng hơn để gỡ lỗi mã lập trình bắt buộc. Điều này là do dễ dàng hơn để có được và in tất cả các giá trị biến trung gian có liên quan hoặc sử dụng các công cụ gỡ lỗi tích hợp sẵn của Python.
* Lập trình tượng trưng hiệu quả hơn và dễ dàng hơn để cổng. Lập trình tượng trưng giúp tối ưu hóa mã trong quá trình biên dịch dễ dàng hơn, đồng thời có khả năng chuyển chương trình thành một định dạng độc lập với Python. Điều này cho phép chương trình được chạy trong môi trường không phải Python, do đó tránh mọi vấn đề hiệu suất tiềm ẩn liên quan đến trình thông dịch Python.

## Lập trình lai

Trong lịch sử hầu hết các khuôn khổ học sâu lựa chọn giữa một cách tiếp cận bắt buộc hoặc mang tính biểu tượng. Ví dụ, Theano, TensorFlow (lấy cảm hứng từ trước đây), Keras và CNTK xây dựng các mô hình tượng trưng. Ngược lại, Chainer và PyTorch có một cách tiếp cận bắt buộc. Một chế độ bắt buộc đã được thêm vào TensorFlow 2.0 và Keras trong các phiên bản sau này.

:begin_tab:`mxnet`
Khi thiết kế Gluon, các nhà phát triển đã xem xét liệu có thể kết hợp những lợi ích của cả hai mô hình lập trình hay không. Điều này dẫn đến một mô hình lai cho phép người dùng phát triển và gỡ lỗi với lập trình bắt buộc thuần túy, trong khi có khả năng chuyển đổi hầu hết các chương trình thành các chương trình tượng trưng để chạy khi hiệu suất tính toán cấp sản phẩm và triển khai được yêu cầu. 

Trong thực tế, điều này có nghĩa là chúng tôi xây dựng các mô hình bằng cách sử dụng lớp `HybridBlock` hoặc `HybridSequential`. Theo mặc định, một trong số chúng được thực thi theo cùng một cách lớp `Block` hoặc `Sequential` được thực thi trong lập trình bắt buộc. Lớp `HybridSequential` là một lớp con của `HybridBlock` (giống như `Sequential` lớp con `Block`). Khi hàm `hybridize` được gọi, Gluon biên dịch mô hình thành dạng được sử dụng trong lập trình tượng trưng. Điều này cho phép người ta tối ưu hóa các thành phần chuyên sâu tính toán mà không phải hy sinh theo cách thực hiện mô hình. Chúng tôi sẽ minh họa những lợi ích dưới đây, tập trung vào các mô hình và khối tuần tự.
:end_tab:

:begin_tab:`pytorch`
Như đã đề cập ở trên, PyTorch dựa trên lập trình bắt buộc và sử dụng đồ thị tính toán động. Trong nỗ lực tận dụng tính di động và hiệu quả của lập trình tượng trưng, các nhà phát triển đã xem xét liệu có thể kết hợp lợi ích của cả hai mô hình lập trình hay không. Điều này dẫn đến một torchscript cho phép người dùng phát triển và gỡ lỗi bằng cách sử dụng lập trình bắt buộc thuần túy, trong khi có khả năng chuyển đổi hầu hết các chương trình thành các chương trình tượng trưng để chạy khi hiệu suất và triển khai tính toán cấp sản phẩm được yêu cầu.
:end_tab:

:begin_tab:`tensorflow`
Mô hình lập trình bắt buộc bây giờ là mặc định trong Tensorflow 2, một thay đổi chào đón cho những người mới đến ngôn ngữ. Tuy nhiên, các kỹ thuật lập trình tượng trưng tương tự và đồ thị tính toán tiếp theo vẫn tồn tại trong TensorFlow, và có thể được truy cập bởi trình trang trí `tf.function` dễ sử dụng. Điều này đã mang lại mô hình lập trình bắt buộc cho TensorFlow, cho phép người dùng xác định các hàm trực quan hơn, sau đó gói chúng và biên dịch chúng thành đồ thị tính toán tự động bằng cách sử dụng một tính năng mà nhóm TensorFlow đề cập đến là [autograph](https://www.tensorflow.org/api_docs/python/tf/autograph).
:end_tab:

## Lai tạo lớp `Sequential`

Cách dễ nhất để có được cảm nhận về cách thức hoạt động của lai là xem xét các mạng sâu với nhiều lớp. Thông thường, trình thông dịch Python sẽ cần thực thi mã cho tất cả các lớp để tạo ra một lệnh sau đó có thể được chuyển tiếp đến CPU hoặc GPU. Đối với một thiết bị điện toán (nhanh) duy nhất, điều này không gây ra bất kỳ vấn đề lớn nào. Mặt khác, nếu chúng ta sử dụng một máy chủ 8 GPU tiên tiến như một phiên bản AWS P3dn.24xlarge Python sẽ phải vật lộn để giữ cho tất cả các GPU bận rộn. Trình thông dịch Python đơn luồng trở thành nút cổ chai ở đây. Hãy để chúng tôi xem cách chúng tôi có thể giải quyết vấn đề này cho các phần quan trọng của mã bằng cách thay thế `Sequential` bằng `HybridSequential`. Chúng tôi bắt đầu bằng cách xác định một MLP đơn giản.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

# Factory for networks
def get_net():
    net = nn.HybridSequential()  
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net

x = np.random.normal(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

# Factory for networks
def get_net():
    net = nn.Sequential(nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2))
    return net

x = torch.randn(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from tensorflow.keras.layers import Dense

# Factory for networks
def get_net():
    net = tf.keras.Sequential()
    net.add(Dense(256, input_shape = (512,), activation = "relu"))
    net.add(Dense(128, activation = "relu"))
    net.add(Dense(2, activation = "linear"))
    return net

x = tf.random.normal([1,512])
net = get_net()
net(x)
```

:begin_tab:`mxnet`
Bằng cách gọi hàm `hybridize`, chúng ta có thể biên dịch và tối ưu hóa tính toán trong MLP. Kết quả tính toán của mô hình vẫn không thay đổi.
:end_tab:

:begin_tab:`pytorch`
Bằng cách chuyển đổi mô hình bằng hàm `torch.jit.script`, chúng ta có thể biên dịch và tối ưu hóa tính toán trong MLP. Kết quả tính toán của mô hình vẫn không thay đổi.
:end_tab:

:begin_tab:`tensorflow`
Trước đây, tất cả các chức năng được xây dựng trong TensorFlow được xây dựng như một biểu đồ tính toán, và do đó JIT biên dịch theo mặc định. Tuy nhiên, với việc phát hành TensorFlow 2.X và EagerTensor, đây không còn là behavor mặc định. Chúng ta cen kích hoạt lại chức năng này với tf.function. tf.function thường được sử dụng như một trình trang trí hàm, tuy nhiên có thể gọi nó trực tiếp như một hàm python bình thường, được hiển thị bên dưới. Kết quả tính toán của mô hình vẫn không thay đổi.
:end_tab:

```{.python .input}
net.hybridize()
net(x)
```

```{.python .input}
#@tab pytorch
net = torch.jit.script(net)
net(x)
```

```{.python .input}
#@tab tensorflow
net = tf.function(net)
net(x)
```

:begin_tab:`mxnet`
Điều này có vẻ gần như quá tốt để đúng: chỉ cần chỉ định một khối là `HybridSequential`, viết cùng một mã như trước và gọi `hybridize`. Khi điều này xảy ra, mạng được tối ưu hóa (chúng tôi sẽ chuẩn hóa hiệu suất bên dưới). Thật không may, điều này không hoạt động kỳ diệu cho mỗi lớp. Điều đó nói rằng, một layer sẽ không được tối ưu hóa nếu nó kế thừa từ lớp `Block` thay vì lớp `HybridBlock`.
:end_tab:

:begin_tab:`pytorch`
Điều này có vẻ gần như quá tốt là đúng: viết cùng một mã như trước và chỉ cần chuyển đổi mô hình bằng `torch.jit.script`. Khi điều này xảy ra, mạng được tối ưu hóa (chúng tôi sẽ chuẩn hóa hiệu suất bên dưới).
:end_tab:

:begin_tab:`tensorflow`
Điều này có vẻ gần như quá tốt để đúng: viết cùng một mã như trước và chỉ cần chuyển đổi mô hình bằng `tf.function`. Khi điều này xảy ra, mạng được xây dựng như một biểu đồ tính toán trong đại diện trung gian MLIR của TensorFlow và được tối ưu hóa rất nhiều ở cấp trình biên dịch để thực hiện nhanh chóng (chúng tôi sẽ đánh dấu hiệu suất bên dưới). Thêm cờ `jit_compile = True` một cách rõ ràng vào cuộc gọi `tf.function()` cho phép XLA (Accelerated Linear Algebra) chức năng trong TensorFlow. XLA có thể tối ưu hóa hơn nữa mã được biên dịch JIT trong một số trường hợp nhất định. Thực thi chế độ đồ thị được bật mà không có định nghĩa rõ ràng này, tuy nhiên XLA có thể thực hiện một số hoạt động đại số tuyến tính lớn nhất định (trong tĩnh mạch của những người chúng ta thấy trong các ứng dụng học sâu) nhanh hơn nhiều, đặc biệt là trong môi trường GPU.
:end_tab:

### Tăng tốc bằng cách lai

Để chứng minh sự cải thiện hiệu suất đạt được bằng cách biên dịch, chúng tôi so sánh thời gian cần thiết để đánh giá `net(x)` trước và sau khi lai. Hãy để chúng tôi xác định một lớp để đo lần này đầu tiên. Nó sẽ có ích trong suốt chương khi chúng tôi đặt ra để đo lường (và cải thiện) hiệu suất.

```{.python .input}
#@tab all
#@save
class Benchmark:
    """For measuring running time."""
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')
```

:begin_tab:`mxnet`
Bây giờ chúng ta có thể gọi mạng hai lần, một lần với và một lần mà không cần lai tạo.
:end_tab:

:begin_tab:`pytorch`
Bây giờ chúng ta có thể gọi mạng hai lần, một lần với và một lần không có torchscript.
:end_tab:

:begin_tab:`tensorflow`
Bây giờ chúng ta có thể gọi mạng ba lần, một lần được thực hiện háo hức, một lần với thực thi chế độ đồ thị, và một lần nữa sử dụng JIT biên dịch XLA.
:end_tab:

```{.python .input}
net = get_net()
with Benchmark('Without hybridization'):
    for i in range(1000): net(x)
    npx.waitall()

net.hybridize()
with Benchmark('With hybridization'):
    for i in range(1000): net(x)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
net = get_net()
with Benchmark('Without torchscript'):
    for i in range(1000): net(x)

net = torch.jit.script(net)
with Benchmark('With torchscript'):
    for i in range(1000): net(x)
```

```{.python .input}
#@tab tensorflow
net = get_net()
with Benchmark('Eager Mode'):
    for i in range(1000): net(x)

net = tf.function(net)
with Benchmark('Graph Mode'):
    for i in range(1000): net(x)
```

:begin_tab:`mxnet`
Như đã quan sát thấy trong các kết quả trên, sau khi một phiên bản `HybridSequential` gọi hàm `hybridize`, hiệu suất tính toán được cải thiện thông qua việc sử dụng lập trình tượng trưng.
:end_tab:

:begin_tab:`pytorch`
Như đã quan sát thấy trong các kết quả trên, sau khi một phiên bản `nn.Sequential` được viết kịch bản bằng hàm `torch.jit.script`, hiệu suất tính toán được cải thiện thông qua việc sử dụng lập trình tượng trưng.
:end_tab:

:begin_tab:`tensorflow`
Như đã quan sát thấy trong các kết quả trên, sau khi một phiên bản `tf.keras.Sequential` được viết kịch bản bằng hàm `tf.function`, hiệu suất tính toán được cải thiện thông qua việc sử dụng lập trình tượng trưng thông qua thực thi chế độ đồ thị trong tensorflow.
:end_tab:

### Lập số sê-ri

:begin_tab:`mxnet`
Một trong những lợi ích của việc biên dịch các mô hình là chúng ta có thể serialize (lưu) mô hình và các tham số của nó vào đĩa. Điều này cho phép chúng tôi lưu trữ một mô hình theo cách độc lập với ngôn ngữ front-end của sự lựa chọn. Điều này cho phép chúng tôi triển khai các mô hình được đào tạo cho các thiết bị khác và dễ dàng sử dụng các ngôn ngữ lập trình front-end khác. Đồng thời mã thường nhanh hơn những gì có thể đạt được trong lập trình bắt buộc. Chúng ta hãy xem chức năng `export` đang hoạt động.
:end_tab:

:begin_tab:`pytorch`
Một trong những lợi ích của việc biên dịch các mô hình là chúng ta có thể serialize (lưu) mô hình và các tham số của nó vào đĩa. Điều này cho phép chúng tôi lưu trữ một mô hình theo cách độc lập với ngôn ngữ front-end của sự lựa chọn. Điều này cho phép chúng tôi triển khai các mô hình được đào tạo cho các thiết bị khác và dễ dàng sử dụng các ngôn ngữ lập trình front-end khác. Đồng thời mã thường nhanh hơn những gì có thể đạt được trong lập trình bắt buộc. Chúng ta hãy xem chức năng `save` đang hoạt động.
:end_tab:

:begin_tab:`tensorflow`
Một trong những lợi ích của việc biên dịch các mô hình là chúng ta có thể serialize (lưu) mô hình và các tham số của nó vào đĩa. Điều này cho phép chúng tôi lưu trữ một mô hình theo cách độc lập với ngôn ngữ front-end của sự lựa chọn. Điều này cho phép chúng tôi triển khai các mô hình được đào tạo cho các thiết bị khác và dễ dàng sử dụng các ngôn ngữ lập trình front-end khác hoặc thực hiện một mô hình được đào tạo trên máy chủ. Đồng thời mã thường nhanh hơn những gì có thể đạt được trong lập trình bắt buộc. API cấp thấp cho phép chúng ta lưu trong tensorflow là `tf.saved_model`. Hãy xem phiên bản `saved_model` đang hoạt động.
:end_tab:

```{.python .input}
net.export('my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab pytorch
net.save('my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab tensorflow
net = get_net()
tf.saved_model.save(net, 'my_mlp')
!ls -lh my_mlp*
```

:begin_tab:`mxnet`
Mô hình được phân hủy thành một tệp tham số (nhị phân lớn) và mô tả JSON của chương trình cần thiết để thực thi tính toán mô hình. Các tập tin có thể được đọc bởi các ngôn ngữ front-end khác được hỗ trợ bởi Python hoặc MXNet, chẳng hạn như C ++, R, Scala, và Perl. Chúng ta hãy xem xét một vài dòng đầu tiên trong mô tả mô hình.
:end_tab:

```{.python .input}
!head my_mlp-symbol.json
```

:begin_tab:`mxnet`
Trước đó, chúng tôi đã chứng minh rằng, sau khi gọi hàm `hybridize`, mô hình này có thể đạt được hiệu suất tính toán và tính di động vượt trội. Lưu ý, mặc dù việc lai tạo có thể ảnh hưởng đến tính linh hoạt của mô hình, đặc biệt là về dòng chảy điều khiển.  

Bên cạnh đó, trái với phiên bản `Block`, cần sử dụng hàm `forward`, đối với một ví dụ `HybridBlock` chúng ta cần sử dụng hàm `hybrid_forward`.
:end_tab:

```{.python .input}
class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(4)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x):
        print('module F: ', F)
        print('value  x: ', x)
        x = F.npx.relu(self.hidden(x))
        print('result  : ', x)
        return self.output(x)
```

:begin_tab:`mxnet`
Mã trên thực hiện một mạng đơn giản với 4 đơn vị ẩn và 2 đầu ra. Hàm `hybrid_forward` mất một đối số bổ sung `F`. Điều này là cần thiết vì, tùy thuộc vào việc mã đã được lai hay không, nó sẽ sử dụng một thư viện hơi khác (`ndarray` hoặc `symbol`) để xử lý. Cả hai lớp đều thực hiện các hàm rất giống nhau và MXNet tự động xác định đối số. Để hiểu những gì đang xảy ra, chúng ta in các đối số như một phần của lệnh gọi hàm.
:end_tab:

```{.python .input}
net = HybridNet()
net.initialize()
x = np.random.normal(size=(1, 3))
net(x)
```

:begin_tab:`mxnet`
Lặp lại tính toán chuyển tiếp sẽ dẫn đến cùng một đầu ra (chúng tôi bỏ qua chi tiết). Bây giờ chúng ta hãy xem những gì sẽ xảy ra nếu chúng ta gọi hàm `hybridize`.
:end_tab:

```{.python .input}
net.hybridize()
net(x)
```

:begin_tab:`mxnet`
Thay vì sử dụng `ndarray`, bây giờ chúng tôi sử dụng mô-đun `symbol` cho `F`. Hơn nữa, mặc dù đầu vào là loại `ndarray`, dữ liệu chảy qua mạng hiện được chuyển đổi thành loại `symbol` như một phần của quá trình biên dịch. Lặp lại cuộc gọi hàm dẫn đến một kết quả đáng ngạc nhiên:
:end_tab:

```{.python .input}
net(x)
```

:begin_tab:`mxnet`
Điều này khá khác với những gì chúng ta đã thấy trước đây. Tất cả các câu lệnh in, như được định nghĩa trong `hybrid_forward`, đều bị bỏ qua. Thật vậy, sau khi lai, việc thực hiện `net(x)` không liên quan đến thông dịch viên Python nữa. Điều này có nghĩa là bất kỳ mã Python giả nào đều bị bỏ qua (chẳng hạn như các câu lệnh in) có lợi cho việc thực thi hợp lý hơn nhiều và hiệu suất tốt hơn. Thay vào đó, MXNet trực tiếp gọi phụ trợ C ++. Cũng lưu ý rằng một số chức năng không được hỗ trợ trong mô-đun `symbol` (ví dụ, `asnumpy`) và các hoạt động tại chỗ như `a += b` và `a[:] = a + b` phải được viết lại là `a = a + b`. Tuy nhiên, việc tổng hợp các mô hình đáng để nỗ lực bất cứ khi nào tốc độ quan trọng. Lợi ích có thể dao động từ các điểm phần trăm nhỏ đến hơn gấp đôi tốc độ, tùy thuộc vào độ phức tạp của mô hình, tốc độ của CPU và tốc độ và số lượng GPU.
:end_tab:

## Tóm tắt

* Lập trình bắt buộc giúp bạn dễ dàng thiết kế các mô hình mới vì có thể viết mã với luồng điều khiển và khả năng sử dụng một lượng lớn hệ sinh thái phần mềm Python.
* Lập trình tượng trưng yêu cầu chúng ta chỉ định chương trình và biên dịch nó trước khi thực hiện nó. Lợi ích là cải thiện hiệu suất.

:begin_tab:`mxnet`
* MXNet có thể kết hợp những lợi thế của cả hai cách tiếp cận khi cần thiết.
* Các mô hình được xây dựng bởi các lớp `HybridSequential` và `HybridBlock` có thể chuyển đổi các chương trình bắt buộc thành các chương trình tượng trưng bằng cách gọi hàm `hybridize`.
:end_tab:

## Bài tập

:begin_tab:`mxnet`
1. Thêm `x.asnumpy()` vào dòng đầu tiên của hàm `hybrid_forward` của lớp `HybridNet` trong phần này. Thực hiện mã và quan sát các lỗi bạn gặp phải. Tại sao họ lại xảy ra?
1. Điều gì xảy ra nếu chúng ta thêm luồng điều khiển, tức là, các câu lệnh Python `if` và `for` trong hàm `hybrid_forward`?
1. Xem lại các mô hình mà bạn quan tâm trong các chương trước. Bạn có thể cải thiện hiệu suất tính toán của họ bằng cách triển khai lại chúng không?
:end_tab:

:begin_tab:`pytorch,tensorflow`
1. Xem lại các mô hình mà bạn quan tâm trong các chương trước. Bạn có thể cải thiện hiệu suất tính toán của họ bằng cách triển khai lại chúng không?
:end_tab:

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/360)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2490)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2492)
:end_tab:
