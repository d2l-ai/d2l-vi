# Tệp I/O

Cho đến nay chúng tôi đã thảo luận về cách xử lý dữ liệu và cách xây dựng, đào tạo và kiểm tra các mô hình học sâu. Tuy nhiên, tại một số điểm, chúng tôi hy vọng sẽ đủ hạnh phúc với các mô hình đã học mà chúng tôi sẽ muốn lưu kết quả để sử dụng sau này trong các bối cảnh khác nhau (thậm chí có thể đưa ra dự đoán trong triển khai). Ngoài ra, khi chạy một quá trình đào tạo dài, thực hành tốt nhất là lưu định kỳ kết quả trung gian (checkpointing) để đảm bảo rằng chúng tôi không mất vài ngày giá trị tính toán nếu chúng ta đi qua dây nguồn của máy chủ của chúng tôi. Vì vậy, đã đến lúc học cách tải và lưu trữ cả vectơ trọng lượng riêng lẻ và toàn bộ mô hình. Phần này giải quyết cả hai vấn đề. 

## (**Tải và tiết kiệm Tensors**)

Đối với hàng chục cá nhân, chúng ta có thể gọi trực tiếp các hàm `load` và `save` để đọc và viết chúng tương ứng. Cả hai chức năng yêu cầu chúng tôi cung cấp một tên, và `save` yêu cầu như là đầu vào biến được lưu.

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

x = np.arange(4)
npx.save('x-file', x)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
import numpy as np

x = tf.range(4)
np.save('x-file.npy', x)
```

Bây giờ chúng ta có thể đọc dữ liệu từ tệp được lưu trữ trở lại vào bộ nhớ.

```{.python .input}
x2 = npx.load('x-file')
x2
```

```{.python .input}
#@tab pytorch
x2 = torch.load('x-file')
x2
```

```{.python .input}
#@tab tensorflow
x2 = np.load('x-file.npy', allow_pickle=True)
x2
```

Chúng ta có thể [** lưu trữ một danh sách các hàng chục và đọc lại vào bộ nhớ. **]

```{.python .input}
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

```{.python .input}
#@tab pytorch
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

```{.python .input}
#@tab tensorflow
y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

Chúng ta thậm chí có thể [** viết và đọc một từ điển bản đồ từ chuỗi đến tensors.**] Điều này thuận tiện khi chúng ta muốn đọc hoặc viết tất cả các trọng lượng trong một mô hình.

```{.python .input}
mydict = {'x': x, 'y': y}
npx.save('mydict', mydict)
mydict2 = npx.load('mydict')
mydict2
```

```{.python .input}
#@tab pytorch
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

```{.python .input}
#@tab tensorflow
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
mydict2
```

## [**Tải và lưu thông số mô hình**]

Tiết kiệm vectơ trọng lượng riêng lẻ (hoặc các hàng chục khác) rất hữu ích, nhưng nó trở nên rất tẻ nhạt nếu chúng ta muốn lưu (và sau đó tải) toàn bộ mô hình. Rốt cuộc, chúng ta có thể có hàng trăm nhóm tham số rắc khắp. Vì lý do này, khung học sâu cung cấp các chức năng tích hợp để tải và lưu toàn bộ mạng. Một chi tiết quan trọng cần lưu ý là điều này lưu mô hình *tham số* chứ không phải toàn bộ mô hình. Ví dụ: nếu chúng ta có MLP 3 lớp, chúng ta cần chỉ định kiến trúc riêng biệt. Lý do cho điều này là bản thân các mô hình có thể chứa mã tùy ý, do đó chúng không thể được serialized như một cách tự nhiên. Do đó, để khôi phục lại một mô hình, chúng ta cần tạo kiến trúc trong mã và sau đó tải các tham số từ đĩa. (** Hãy để chúng tôi bắt đầu với MLP.quen thuộc của chúng tôi**)

```{.python .input}
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = np.random.uniform(size=(2, 20))
Y = net(X)
```

```{.python .input}
#@tab pytorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

```{.python .input}
#@tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)
```

Tiếp theo, chúng ta [** lưu trữ các tham số của mô hình dưới dạng tệp**] với tên “mlp.params”.

```{.python .input}
net.save_parameters('mlp.params')
```

```{.python .input}
#@tab pytorch
torch.save(net.state_dict(), 'mlp.params')
```

```{.python .input}
#@tab tensorflow
net.save_weights('mlp.params')
```

Để khôi phục mô hình, chúng tôi khởi tạo một bản sao của mô hình MLP ban đầu. Thay vì khởi tạo ngẫu nhiên các tham số model, chúng ta [**đọc các tham số được lưu trữ trong tập tin trực tiếp**].

```{.python .input}
clone = MLP()
clone.load_parameters('mlp.params')
```

```{.python .input}
#@tab pytorch
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

```{.python .input}
#@tab tensorflow
clone = MLP()
clone.load_weights('mlp.params')
```

Vì cả hai trường hợp đều có cùng tham số mô hình, kết quả tính toán của cùng một đầu vào `X` phải giống nhau. Hãy để chúng tôi xác minh điều này.

```{.python .input}
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
#@tab pytorch
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
#@tab tensorflow
Y_clone = clone(X)
Y_clone == Y
```

## Tóm tắt

* Các chức năng `save` và `load` có thể được sử dụng để thực hiện I/O tệp cho các đối tượng tensor.
* Chúng ta có thể lưu và tải toàn bộ bộ tham số cho một mạng thông qua một từ điển tham số.
* Lưu kiến trúc phải được thực hiện trong mã chứ không phải trong các tham số.

## Bài tập

1. Ngay cả khi không cần triển khai các mô hình được đào tạo cho một thiết bị khác, lợi ích thiết thực của việc lưu trữ các thông số mô hình là gì?
1. Giả sử rằng chúng ta chỉ muốn sử dụng lại các phần của mạng để được kết hợp vào một mạng của một kiến trúc khác nhau. Làm thế nào bạn sẽ đi về sử dụng, nói hai lớp đầu tiên từ một mạng trước đó trong một mạng mới?
1. Làm thế nào bạn sẽ đi về lưu kiến trúc mạng và các tham số? Bạn sẽ áp đặt những hạn chế nào đối với kiến trúc?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/60)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/61)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/327)
:end_tab:
