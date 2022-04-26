# Lớp và khối
:label:`sec_model_construction`

Khi chúng tôi lần đầu tiên giới thiệu các mạng thần kinh, chúng tôi tập trung vào các mô hình tuyến tính với một đầu ra duy nhất. Ở đây, toàn bộ mô hình chỉ bao gồm một tế bào thần kinh duy nhất. Lưu ý rằng một tế bào thần kinh đơn (i) lấy một số tập hợp các đầu vào; (ii) tạo ra một đầu ra vô hướng tương ứng; và (iii) có một tập hợp các tham số liên quan có thể được cập nhật để tối ưu hóa một số chức năng quan tâm. Sau đó, một khi chúng tôi bắt đầu suy nghĩ về các mạng có nhiều đầu ra, chúng tôi đã tận dụng số học vector hóa để mô tả toàn bộ một lớp tế bào thần kinh. Cũng giống như các tế bào thần kinh riêng lẻ, các lớp (i) lấy một tập hợp các đầu vào, (ii) tạo ra các đầu ra tương ứng, và (iii) được mô tả bởi một tập hợp các tham số có thể điều chỉnh. Khi chúng tôi làm việc thông qua hồi quy softmax, một lớp duy nhất chính nó là mô hình. Tuy nhiên, ngay cả khi chúng tôi sau đó giới thiệu MLP, chúng ta vẫn có thể nghĩ về mô hình là giữ lại cấu trúc cơ bản tương tự này. 

Điều thú vị là đối với MLP, cả toàn bộ mô hình và các lớp cấu thành của nó đều chia sẻ cấu trúc này. Toàn bộ mô hình lấy đầu vào thô (các tính năng), tạo ra các đầu ra (dự đoán) và sở hữu các tham số (các tham số kết hợp từ tất cả các lớp cấu thành). Tương tự như vậy, mỗi lớp riêng lẻ ăn vào đầu vào (được cung cấp bởi lớp trước đó) tạo ra các đầu ra (đầu vào cho lớp tiếp theo) và sở hữu một tập hợp các tham số có thể điều chỉnh được cập nhật theo tín hiệu chảy ngược từ lớp tiếp theo. 

Mặc dù bạn có thể nghĩ rằng tế bào thần kinh, lớp và mô hình cung cấp cho chúng ta đủ trừu tượng để đi về kinh doanh của chúng tôi, hóa ra chúng ta thường thấy thuận tiện khi nói về các thành phần lớn hơn một lớp riêng lẻ nhưng nhỏ hơn toàn bộ mô hình. Ví dụ, kiến trúc ResNet-152, rất phổ biến trong tầm nhìn máy tính, sở hữu hàng trăm lớp. Các lớp này bao gồm các mô hình lặp lại của * nhóm lớp*. Thực hiện một mạng như vậy một lớp tại một thời điểm có thể phát triển tẻ nhạt. Mối quan tâm này không chỉ là giả thuyết, các mẫu thiết kế như vậy là phổ biến trong thực tế. Kiến trúc ResNet được đề cập ở trên đã giành được các cuộc thi tầm nhìn máy tính ImageNet và COCO 2015 cho cả công nhận và phát hiện :cite:`He.Zhang.Ren.ea.2016` và vẫn là một kiến trúc đi đến cho nhiều nhiệm vụ thị giác. Các kiến trúc tương tự trong đó các lớp được sắp xếp theo nhiều mẫu lặp lại khác nhau hiện đang phổ biến trong các lĩnh vực khác, bao gồm xử lý ngôn ngữ tự nhiên và lời nói. 

Để thực hiện các mạng phức tạp này, chúng tôi giới thiệu khái niệm về một mạng nơ-ron* block*. Một khối có thể mô tả một lớp duy nhất, một thành phần bao gồm nhiều lớp, hoặc toàn bộ mô hình chính nó! Một lợi ích của việc làm việc với trừu tượng khối là chúng có thể được kết hợp thành các hiện vật lớn hơn, thường đệ quy. Điều này được minh họa trong :numref:`fig_blocks`. Bằng cách xác định mã để tạo ra các khối phức tạp tùy ý theo yêu cầu, chúng ta có thể viết mã nhỏ gọn đáng ngạc nhiên và vẫn triển khai các mạng nơ-ron phức tạp. 

![Multiple layers are combined into blocks, forming repeating patterns of larger models.](../img/blocks.svg)
:label:`fig_blocks`

Từ quan điểm lập trình, một khối được biểu diễn bằng một lớp * *. Bất kỳ lớp con nào của nó phải xác định một hàm tuyên truyền chuyển tiếp biến đổi đầu vào của nó thành đầu ra và phải lưu trữ bất kỳ tham số cần thiết nào. Lưu ý rằng một số khối không yêu cầu bất kỳ tham số nào cả. Cuối cùng một khối phải sở hữu một hàm backpropagation, cho mục đích tính toán gradient. May mắn thay, do một số phép thuật hậu trường được cung cấp bởi sự khác biệt tự động (được giới thiệu trong :numref:`sec_autograd`) khi xác định khối của riêng mình, chúng ta chỉ cần lo lắng về các thông số và chức năng lan truyền chuyển tiếp. 

[**Để bắt đầu, chúng tôi xem lại mã mà chúng tôi đã sử dụng để thực hiện MLPs**](:numref:`sec_mlp_concise`). Mã sau tạo ra một mạng với một lớp ẩn được kết nối hoàn toàn với 256 đơn vị và kích hoạt ReLU, tiếp theo là lớp đầu ra được kết nối hoàn toàn với 10 đơn vị (không có chức năng kích hoạt).

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
net(X)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])

X = tf.random.uniform((2, 20))
net(X)
```

:begin_tab:`mxnet`
Trong ví dụ này, chúng tôi xây dựng mô hình của mình bằng cách khởi tạo một `nn.Sequential`, gán đối tượng trả về cho biến `net`. Tiếp theo, chúng ta liên tục gọi hàm `add` của nó, nối các lớp theo thứ tự chúng nên được thực thi. Tóm lại, `nn.Sequential` định nghĩa một loại đặc biệt `Block`, lớp trình bày một khối trong Gluon. Nó duy trì một danh sách đặt hàng của `Block`s cấu thành. Chức năng `add` chỉ đơn giản là tạo điều kiện cho việc bổ sung mỗi `Block` liên tiếp vào danh sách. Lưu ý rằng mỗi lớp là một đối tượng của lớp `Dense` mà chính nó là một lớp con của `Block`. Chức năng lan truyền chuyển tiếp (`forward`) cũng rất đơn giản: nó chuỗi mỗi `Block` trong danh sách với nhau, truyền đầu ra của mỗi chức năng làm đầu vào cho tiếp theo. Lưu ý rằng cho đến bây giờ, chúng tôi đã gọi các mô hình của chúng tôi thông qua xây dựng `net(X)` để có được đầu ra của chúng. Điều này thực sự chỉ là viết tắt cho `net.forward(X)`, một thủ thuật Python slick đạt được thông qua chức năng `Block` của lớp `__call__`.
:end_tab:

:begin_tab:`pytorch`
Trong ví dụ này, chúng tôi xây dựng mô hình của mình bằng cách khởi tạo một `nn.Sequential`, với các lớp theo thứ tự mà chúng nên được thực hiện thông qua dưới dạng đối số. Nói tóm lại, (**`nn.Sequential` định nghĩa một loại đặc biệt của `Module`**), lớp trình bày một khối trong PyTorch. Nó duy trì một danh sách đặt hàng của `Module`s cấu thành. Lưu ý rằng mỗi lớp trong hai lớp được kết nối hoàn toàn là một thể hiện của lớp `Linear` mà chính nó là một lớp con của `Module`. Chức năng lan truyền chuyển tiếp (`forward`) cũng rất đơn giản: nó chuỗi mỗi khối trong danh sách với nhau, truyền đầu ra của mỗi như đầu vào cho tiếp theo. Lưu ý rằng cho đến bây giờ, chúng tôi đã gọi các mô hình của chúng tôi thông qua xây dựng `net(X)` để có được kết quả đầu ra của chúng. Điều này thực sự chỉ là viết tắt cho `net.__call__(X)`.
:end_tab:

:begin_tab:`tensorflow`
Trong ví dụ này, chúng tôi xây dựng mô hình của mình bằng cách khởi tạo một `keras.models.Sequential`, với các lớp theo thứ tự mà chúng nên được thực hiện thông qua dưới dạng đối số. Tóm lại, `Sequential` định nghĩa một loại đặc biệt của `keras.Model`, lớp trình bày một khối trong Keras. Nó duy trì một danh sách đặt hàng của `Model`s cấu thành. Lưu ý rằng mỗi lớp trong hai lớp được kết nối hoàn toàn là một thể hiện của lớp `Dense` mà chính nó là một lớp con của `Model`. Chức năng lan truyền chuyển tiếp (`call`) cũng rất đơn giản: nó chuỗi mỗi khối trong danh sách với nhau, truyền đầu ra của mỗi như đầu vào cho tiếp theo. Lưu ý rằng cho đến bây giờ, chúng tôi đã gọi các mô hình của chúng tôi thông qua xây dựng `net(X)` để có được đầu ra của chúng. Đây thực sự chỉ là viết tắt cho `net.call(X)`, một thủ thuật Python slick đạt được thông qua hàm `__call__` của lớp Block.
:end_tab:

## [** Một khối tùy chỉnh**]

Có lẽ cách dễ nhất để phát triển trực giác về cách thức hoạt động của một khối là tự thực hiện một. Trước khi chúng tôi triển khai khối tùy chỉnh của riêng mình, chúng tôi tóm tắt ngắn gọn các chức năng cơ bản mà mỗi khối phải cung cấp:

:begin_tab:`mxnet, tensorflow`
1. Nhập dữ liệu đầu vào dưới dạng đối số cho chức năng tuyên truyền chuyển tiếp của nó.
1. Tạo ra một đầu ra bằng cách có hàm tuyên truyền chuyển tiếp trả về một giá trị. Lưu ý rằng đầu ra có thể có một hình dạng khác với đầu vào. Ví dụ, lớp kết nối hoàn toàn đầu tiên trong mô hình của chúng ta ở trên sẽ nhập một đầu vào của chiều tùy ý nhưng trả về đầu ra của kích thước 256.
1. Tính gradient của đầu ra của nó đối với đầu vào của nó, có thể được truy cập thông qua chức năng truyền ngược của nó. Thông thường điều này xảy ra tự động.
1. Lưu trữ và cung cấp quyền truy cập vào các tham số cần thiết để thực hiện tính toán tuyên truyền chuyển tiếp.
1. Khởi tạo các tham số mô hình khi cần thiết.
:end_tab:

:begin_tab:`pytorch`
1. Nhập dữ liệu đầu vào dưới dạng đối số cho chức năng tuyên truyền chuyển tiếp của nó.
1. Tạo ra một đầu ra bằng cách có hàm tuyên truyền chuyển tiếp trả về một giá trị. Lưu ý rằng đầu ra có thể có một hình dạng khác với đầu vào. Ví dụ: lớp kết nối hoàn toàn đầu tiên trong mô hình của chúng ta ở trên sẽ nhập một đầu vào của kích thước 20 nhưng trả về đầu ra của kích thước 256.
1. Tính gradient của đầu ra của nó đối với đầu vào của nó, có thể được truy cập thông qua chức năng truyền ngược của nó. Thông thường điều này xảy ra tự động.
1. Lưu trữ và cung cấp quyền truy cập vào các tham số cần thiết để thực hiện tính toán tuyên truyền chuyển tiếp.
1. Khởi tạo các tham số mô hình khi cần thiết.
:end_tab:

Trong đoạn mã sau, chúng tôi mã hóa một khối từ đầu tương ứng với một MLP với một lớp ẩn với 256 đơn vị ẩn, và một lớp đầu ra 10 chiều. Lưu ý rằng lớp `MLP` bên dưới kế thừa lớp đại diện cho một khối. Chúng ta sẽ dựa rất nhiều vào các hàm của lớp cha, chỉ cung cấp hàm tạo riêng của chúng ta (hàm `__init__` trong Python) và hàm tuyên truyền chuyển tiếp.

```{.python .input}
class MLP(nn.Block):
    # Declare a layer with model parameters. Here, we declare two
    # fully-connected layers
    def __init__(self, **kwargs):
        # Call the constructor of the `MLP` parent class `Block` to perform
        # the necessary initialization. In this way, other function arguments
        # can also be specified during class instantiation, such as the model
        # parameters, `params` (to be described later)
        super().__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # Hidden layer
        self.out = nn.Dense(10)  # Output layer

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input `X`
    def forward(self, X):
        return self.out(self.hidden(X))
```

```{.python .input}
#@tab pytorch
class MLP(nn.Module):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self):
        # Call the constructor of the `MLP` parent class `Module` to perform
        # the necessary initialization. In this way, other function arguments
        # can also be specified during class instantiation, such as the model
        # parameters, `params` (to be described later)
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # Hidden layer
        self.out = nn.Linear(256, 10)  # Output layer

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input `X`
    def forward(self, X):
        # Note here we use the funtional version of ReLU defined in the
        # nn.functional module.
        return self.out(F.relu(self.hidden(X)))
```

```{.python .input}
#@tab tensorflow
class MLP(tf.keras.Model):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self):
        # Call the constructor of the `MLP` parent class `Model` to perform
        # the necessary initialization. In this way, other function arguments
        # can also be specified during class instantiation, such as the model
        # parameters, `params` (to be described later)
        super().__init__()
        # Hidden layer
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)  # Output layer

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input `X`
    def call(self, X):
        return self.out(self.hidden((X)))
```

Trước tiên chúng ta hãy tập trung vào chức năng lan truyền về phía trước. Lưu ý rằng phải mất `X` làm đầu vào, tính toán biểu diễn ẩn với chức năng kích hoạt được áp dụng và đầu ra các bản ghi của nó. Trong triển khai `MLP` này, cả hai lớp đều là các biến thể. Để xem lý do tại sao điều này là hợp lý, hãy tưởng tượng khởi tạo hai MLP s, `net1` và `net2`, và đào tạo chúng trên các dữ liệu khác nhau. Đương nhiên, chúng tôi mong đợi họ đại diện cho hai mô hình đã học khác nhau. 

Chúng ta [**khởi tạo các lớp MLP**] trong hàm tạo (** và sau đó gọi các lớp này**) trên mỗi lần gọi đến hàm tuyên truyền chuyển tiếp. Lưu ý một vài chi tiết chính. Đầu tiên, chức năng `__init__` tùy chỉnh của chúng tôi gọi chức năng `__init__` của lớp mẹ thông qua `super().__init__()` tiết kiệm cho chúng tôi nỗi đau của việc đặt lại mã boilerplate áp dụng cho hầu hết các khối. Sau đó, chúng tôi khởi tạo hai lớp được kết nối hoàn toàn của chúng tôi, gán chúng cho `self.hidden` và `self.out`. Lưu ý rằng trừ khi chúng ta thực hiện một toán tử mới, chúng ta không cần phải lo lắng về hàm backpropagation hoặc khởi tạo tham số. Hệ thống sẽ tự động tạo ra các chức năng này. Hãy để chúng tôi thử điều này ra.

```{.python .input}
net = MLP()
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch
net = MLP()
net(X)
```

```{.python .input}
#@tab tensorflow
net = MLP()
net(X)
```

Một đức tính chính của trừu tượng khối là tính linh hoạt của nó. Chúng ta có thể phân lớp một khối để tạo ra các lớp (chẳng hạn như lớp lớp lớp được kết nối hoàn toàn), toàn bộ mô hình (chẳng hạn như lớp `MLP` ở trên) hoặc các thành phần khác nhau có độ phức tạp trung gian. Chúng tôi khai thác tính linh hoạt này trong suốt các chương sau, chẳng hạn như khi giải quyết các mạng thần kinh phức tạp. 

## [** The Sequential Block**]

Bây giờ chúng ta có thể xem xét kỹ hơn cách thức hoạt động của lớp `Sequential`. Nhớ lại rằng `Sequential` được thiết kế để chuỗi các khối khác với nhau. Để xây dựng `MySequential` đơn giản hóa của riêng mình, chúng ta chỉ cần định nghĩa hai hàm chính:
1. Một hàm để nối các khối từng cái một vào một danh sách.
2. Một chức năng tuyên truyền chuyển tiếp để truyền một đầu vào thông qua chuỗi các khối, theo thứ tự như chúng được nối thêm.

Lớp `MySequential` sau đây cung cấp chức năng tương tự của lớp `Sequential` mặc định.

```{.python .input}
class MySequential(nn.Block):
    def add(self, block):
        # Here, `block` is an instance of a `Block` subclass, and we assume 
        # that it has a unique name. We save it in the member variable
        # `_children` of the `Block` class, and its type is OrderedDict. When
        # the `MySequential` instance calls the `initialize` function, the
        # system automatically initializes all members of `_children`
        self._children[block.name] = block

    def forward(self, X):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._children.values():
            X = block(X)
        return X
```

```{.python .input}
#@tab pytorch
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # Here, `module` is an instance of a `Module` subclass. We save it
            # in the member variable `_modules` of the `Module` class, and its
            # type is OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._modules.values():
            X = block(X)
        return X
```

```{.python .input}
#@tab tensorflow
class MySequential(tf.keras.Model):
    def __init__(self, *args):
        super().__init__()
        self.modules = []
        for block in args:
            # Here, `block` is an instance of a `tf.keras.layers.Layer`
            # subclass
            self.modules.append(block)

    def call(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

:begin_tab:`mxnet`
Hàm `add` thêm một khối duy nhất vào từ điển có thứ tự `_children`. Bạn có thể tự hỏi tại sao mỗi Gluon `Block` sở hữu một thuộc tính `_children` và tại sao chúng tôi sử dụng nó hơn là chỉ định một danh sách Python chính mình. Tóm lại, lợi thế chính của `_children` là trong quá trình khởi tạo tham số khối của chúng tôi, Gluon biết nhìn vào bên trong từ điển `_children` để tìm các khối phụ có tham số cũng cần được khởi tạo.
:end_tab:

:begin_tab:`pytorch`
Trong phương pháp `__init__`, chúng tôi thêm từng mô-đun vào từ điển được đặt hàng `_modules` từng cái một. Bạn có thể tự hỏi tại sao mỗi `Module` đều sở hữu một thuộc tính `_modules` và tại sao chúng ta sử dụng nó thay vì chỉ tự xác định một danh sách Python. Tóm lại, lợi thế chính của `_modules` là trong quá trình khởi tạo tham số mô-đun của chúng tôi, hệ thống biết nhìn vào bên trong từ điển `_modules` để tìm các mô-đun phụ có tham số cũng cần được khởi tạo.
:end_tab:

Khi hàm tuyên truyền chuyển tiếp `MySequential` của chúng tôi được gọi, mỗi khối được thêm vào được thực hiện theo thứ tự mà chúng được thêm vào. Bây giờ chúng ta có thể triển khai lại MLP bằng cách sử dụng lớp `MySequential` của chúng tôi.

```{.python .input}
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
```

```{.python .input}
#@tab tensorflow
net = MySequential(
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10))
net(X)
```

Lưu ý rằng việc sử dụng `MySequential` này giống với mã mà chúng tôi đã viết trước đây cho lớp `Sequential` (như được mô tả trong :numref:`sec_mlp_concise`). 

## [**Thihành mã trong chức năng tuyên truyền chuyển tiếp**]

Lớp `Sequential` giúp việc xây dựng mô hình dễ dàng, cho phép chúng tôi lắp ráp các kiến trúc mới mà không cần phải xác định lớp học của riêng mình. Tuy nhiên, không phải tất cả các kiến trúc đều là chuỗi daisy đơn giản. Khi cần tính linh hoạt cao hơn, chúng tôi sẽ muốn xác định các khối của riêng mình. Ví dụ, chúng ta có thể muốn thực thi luồng điều khiển của Python trong hàm tuyên truyền chuyển tiếp. Hơn nữa, chúng ta có thể muốn thực hiện các phép toán tùy ý, không chỉ dựa vào các lớp mạng thần kinh được xác định trước. 

Bạn có thể nhận thấy rằng cho đến bây giờ, tất cả các hoạt động trong mạng của chúng tôi đã hoạt động dựa trên các kích hoạt mạng và các thông số của mạng của chúng tôi. Tuy nhiên, đôi khi, chúng ta có thể muốn kết hợp các thuật ngữ không phải là kết quả của các lớp trước đó cũng như các tham số có thể cập nhật. Chúng tôi gọi các tham số * không đổi này*. Ví dụ, chúng ta muốn một lớp tính hàm $f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$, trong đó $\mathbf{x}$ là đầu vào, $\mathbf{w}$ là tham số của chúng ta và $c$ là một số hằng số được chỉ định không được cập nhật trong quá trình tối ưu hóa. Vì vậy, chúng tôi thực hiện một lớp `FixedHiddenMLP` như sau.

```{.python .input}
class FixedHiddenMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Random weight parameters created with the `get_constant` function
        # are not updated during training (i.e., constant parameters)
        self.rand_weight = self.params.get_constant(
            'rand_weight', np.random.uniform(size=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, X):
        X = self.dense(X)
        # Use the created constant parameters, as well as the `relu` and `dot`
        # functions
        X = npx.relu(np.dot(X, self.rand_weight.data()) + 1)
        # Reuse the fully-connected layer. This is equivalent to sharing
        # parameters with two fully-connected layers
        X = self.dense(X)
        # Control flow
        while np.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
#@tab pytorch
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Random weight parameters that will not compute gradients and
        # therefore keep constant during training
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # Use the created constant parameters, as well as the `relu` and `mm`
        # functions
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # Reuse the fully-connected layer. This is equivalent to sharing
        # parameters with two fully-connected layers
        X = self.linear(X)
        # Control flow
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
#@tab tensorflow
class FixedHiddenMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        # Random weight parameters created with `tf.constant` are not updated
        # during training (i.e., constant parameters)
        self.rand_weight = tf.constant(tf.random.uniform((20, 20)))
        self.dense = tf.keras.layers.Dense(20, activation=tf.nn.relu)

    def call(self, inputs):
        X = self.flatten(inputs)
        # Use the created constant parameters, as well as the `relu` and
        # `matmul` functions
        X = tf.nn.relu(tf.matmul(X, self.rand_weight) + 1)
        # Reuse the fully-connected layer. This is equivalent to sharing
        # parameters with two fully-connected layers
        X = self.dense(X)
        # Control flow
        while tf.reduce_sum(tf.math.abs(X)) > 1:
            X /= 2
        return tf.reduce_sum(X)
```

Trong mô hình `FixedHiddenMLP` này, chúng tôi triển khai một lớp ẩn có trọng lượng (`self.rand_weight`) được khởi tạo ngẫu nhiên khi khởi tạo và sau đó là hằng số. Trọng lượng này không phải là một tham số mô hình và do đó nó không bao giờ được cập nhật bằng cách truyền ngược. Sau đó, mạng truyền đầu ra của lớp “cố định” này thông qua một lớp được kết nối hoàn toàn. 

Lưu ý rằng trước khi trả lại đầu ra, mô hình của chúng tôi đã làm một cái gì đó bất thường. Chúng tôi chạy một vòng lặp trong khi, thử nghiệm với điều kiện định mức $L_1$ của nó lớn hơn $1$ và chia vector đầu ra của chúng tôi cho $2$ cho đến khi nó thỏa mãn điều kiện. Cuối cùng, chúng tôi trả lại tổng các mục trong `X`. Theo kiến thức của chúng tôi, không có mạng thần kinh tiêu chuẩn nào thực hiện hoạt động này. Lưu ý rằng hoạt động cụ thể này có thể không hữu ích trong bất kỳ nhiệm vụ thực tế nào. Quan điểm của chúng tôi chỉ là chỉ cho bạn cách tích hợp mã tùy ý vào luồng tính toán mạng thần kinh của bạn.

```{.python .input}
net = FixedHiddenMLP()
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch, tensorflow
net = FixedHiddenMLP()
net(X)
```

Chúng ta có thể [** trộn và kết hợp các cách khác nhau để lắp ráp các khối với nhau.**] Trong ví dụ sau, chúng ta tổ các khối theo một số cách sáng tạo.

```{.python .input}
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, X):
        return self.dense(self.net(X))

chimera = nn.Sequential()
chimera.add(NestMLP(), nn.Dense(20), FixedHiddenMLP())
chimera.initialize()
chimera(X)
```

```{.python .input}
#@tab pytorch
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
```

```{.python .input}
#@tab tensorflow
class NestMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        self.net.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        self.dense = tf.keras.layers.Dense(16, activation=tf.nn.relu)

    def call(self, inputs):
        return self.dense(self.net(inputs))

chimera = tf.keras.Sequential()
chimera.add(NestMLP())
chimera.add(tf.keras.layers.Dense(20))
chimera.add(FixedHiddenMLP())
chimera(X)
```

## Hiệu quả

:begin_tab:`mxnet`
Người đọc avid có thể bắt đầu lo lắng về hiệu quả của một số hoạt động này. Rốt cuộc, chúng ta có rất nhiều tra cứu từ điển, thực thi mã và rất nhiều thứ Pythonic khác diễn ra trong những gì được cho là một thư viện deep learning hiệu suất cao. Các vấn đề về [global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock) của Python được biết đến nhiều. Trong bối cảnh học sâu, chúng ta có thể lo lắng rằng (các) GPU cực nhanh của chúng ta có thể phải đợi cho đến khi một CPU nhỏ chạy mã Python trước khi nó có được một công việc khác để chạy. Cách tốt nhất để tăng tốc Python là tránh nó hoàn toàn. 

Một cách mà Gluon làm điều này là bằng cách cho phép
*hybridization*, sẽ được mô tả sau.
Ở đây, trình thông dịch Python thực thi một khối lần đầu tiên nó được gọi. Thời gian chạy Gluon ghi lại những gì đang xảy ra và lần sau xung quanh nó ngắn mạch gọi đến Python. Điều này có thể đẩy nhanh mọi thứ đáng kể trong một số trường hợp nhưng cần phải cẩn thận khi dòng chảy kiểm soát (như trên) dẫn xuống các nhánh khác nhau trên các đường đi khác nhau qua mạng. Chúng tôi khuyên người đọc quan tâm kiểm tra phần lai (:numref:`sec_hybridize`) để tìm hiểu về việc biên soạn sau khi kết thúc chương hiện tại.
:end_tab:

:begin_tab:`pytorch`
Người đọc avid có thể bắt đầu lo lắng về hiệu quả của một số hoạt động này. Rốt cuộc, chúng ta có rất nhiều tra cứu từ điển, thực thi mã và rất nhiều thứ Pythonic khác diễn ra trong những gì được cho là một thư viện deep learning hiệu suất cao. Các vấn đề về [global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock) của Python được biết đến nhiều. Trong bối cảnh học sâu, chúng ta có thể lo lắng rằng (các) GPU cực nhanh của chúng ta có thể phải đợi cho đến khi một CPU nhỏ chạy mã Python trước khi nó có được một công việc khác để chạy.
:end_tab:

:begin_tab:`tensorflow`
Người đọc avid có thể bắt đầu lo lắng về hiệu quả của một số hoạt động này. Rốt cuộc, chúng ta có rất nhiều tra cứu từ điển, thực thi mã và rất nhiều thứ Pythonic khác diễn ra trong những gì được cho là một thư viện deep learning hiệu suất cao. Các vấn đề về [global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock) của Python được biết đến nhiều. Trong bối cảnh học sâu, chúng ta có thể lo lắng rằng (các) GPU cực nhanh của chúng ta có thể phải đợi cho đến khi một CPU nhỏ chạy mã Python trước khi nó có được một công việc khác để chạy. Cách tốt nhất để tăng tốc Python là tránh nó hoàn toàn.
:end_tab:

## Tóm tắt

* Lớp là các khối.
* Nhiều lớp có thể bao gồm một khối.
* Nhiều khối có thể bao gồm một khối.
* Một khối có thể chứa mã.
* Các khối chăm sóc rất nhiều dịch vụ vệ sinh, bao gồm khởi tạo tham số và truyền ngược.
* Các kết nối tuần tự của các lớp và khối được xử lý bởi khối `Sequential`.

## Bài tập

1. Những loại vấn đề sẽ xảy ra nếu bạn thay đổi `MySequential` để lưu trữ các khối trong một danh sách Python?
1. Thực hiện một khối lấy hai khối làm đối số, giả sử `net1` và `net2` và trả về đầu ra nối của cả hai mạng trong tuyên truyền chuyển tiếp. Đây còn được gọi là một khối song song.
1. Giả sử rằng bạn muốn nối nhiều phiên bản của cùng một mạng. Triển khai một hàm factory tạo ra nhiều trường hợp của cùng một khối và xây dựng một mạng lớn hơn từ nó.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/54)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/55)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/264)
:end_tab:
