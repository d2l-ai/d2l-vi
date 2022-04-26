# Mạng dư (ResNet)
:label:`sec_resnet`

Khi chúng ta thiết kế các mạng ngày càng sâu hơn, bắt buộc phải hiểu cách thêm các lớp có thể làm tăng độ phức tạp và biểu cảm của mạng. Thậm chí quan trọng hơn nữa là khả năng thiết kế mạng mà việc thêm các lớp làm cho các mạng trở nên biểu cảm nghiêm ngặt hơn là chỉ khác biệt. Để đạt được một số tiến bộ, chúng ta cần một chút toán học. 

## Các lớp hàm

Xem xét $\mathcal{F}$, lớp hàm mà một kiến trúc mạng cụ thể (cùng với tốc độ học tập và các cài đặt siêu tham số khác) có thể đạt được. Đó là, đối với tất cả $f \in \mathcal{F}$ có tồn tại một số tập hợp các tham số (ví dụ, trọng lượng và thành kiến) có thể thu được thông qua đào tạo trên một tập dữ liệu phù hợp. Chúng ta hãy giả sử rằng $f^*$ là chức năng “sự thật” mà chúng ta thực sự muốn tìm thấy. Nếu nó là trong $\mathcal{F}$, chúng tôi đang trong tình trạng tốt nhưng thông thường chúng tôi sẽ không hoàn toàn may mắn như vậy. Thay vào đó, chúng tôi sẽ cố gắng tìm một số $f^*_\mathcal{F}$ là đặt cược tốt nhất của chúng tôi trong vòng $\mathcal{F}$. Ví dụ, với một tập dữ liệu với các tính năng $\mathbf{X}$ và nhãn $\mathbf{y}$, chúng ta có thể thử tìm nó bằng cách giải quyết vấn đề tối ưu hóa sau: 

$$f^*_\mathcal{F} \stackrel{\mathrm{def}}{=} \mathop{\mathrm{argmin}}_f L(\mathbf{X}, \mathbf{y}, f) \text{ subject to } f \in \mathcal{F}.$$

Nó chỉ là hợp lý để giả định rằng nếu chúng ta thiết kế một kiến trúc khác và mạnh mẽ hơn $\mathcal{F}'$ chúng ta nên đi đến một kết quả tốt hơn. Nói cách khác, chúng tôi hy vọng rằng $f^*_{\mathcal{F}'}$ là “tốt hơn” so với $f^*_{\mathcal{F}}$. Tuy nhiên, nếu $\mathcal{F} \not\subseteq \mathcal{F}'$ không có gì đảm bảo rằng điều này thậm chí sẽ xảy ra. Trong thực tế, $f^*_{\mathcal{F}'}$ cũng có thể tồi tệ hơn. Như minh họa bởi :numref:`fig_functionclasses`, đối với các lớp hàm không lồng nhau, một lớp hàm lớn hơn không phải lúc nào cũng di chuyển gần hơn với hàm “truth” $f^*$. Ví dụ, ở bên trái của :numref:`fig_functionclasses`, mặc dù $\mathcal{F}_3$ gần $f^*$ hơn $\mathcal{F}_1$, $\mathcal{F}_6$ di chuyển đi và không có gì đảm bảo rằng việc tăng thêm độ phức tạp có thể làm giảm khoảng cách từ $f^*$. Với các lớp hàm lồng nhau trong đó $\mathcal{F}_1 \subseteq \ldots \subseteq \mathcal{F}_6$ ở bên phải của :numref:`fig_functionclasses`, chúng ta có thể tránh được vấn đề nói trên từ các lớp hàm không lồng nhau. 

![For non-nested function classes, a larger (indicated by area) function class does not guarantee to get closer to the "truth" function ($f^*$). This does not happen in nested function classes.](../img/functionclasses.svg)
:label:`fig_functionclasses`

Do đó, chỉ khi các lớp hàm lớn hơn chứa các lớp nhỏ hơn, chúng tôi đảm bảo rằng việc tăng chúng làm tăng nghiêm ngặt sức mạnh biểu cảm của mạng. Đối với các mạng thần kinh sâu, nếu chúng ta có thể đào tạo lớp mới được thêm vào một hàm nhận dạng $f(\mathbf{x}) = \mathbf{x}$, mô hình mới sẽ hiệu quả như mô hình ban đầu. Vì mô hình mới có thể nhận được một giải pháp tốt hơn để phù hợp với tập dữ liệu đào tạo, lớp được thêm vào có thể giúp giảm lỗi đào tạo dễ dàng hơn. 

Đây là câu hỏi mà Ông et al. xem xét khi làm việc trên các mô hình tầm nhìn máy tính rất sâu :cite:`He.Zhang.Ren.ea.2016`. Trọng tâm của họ* mạng dư * (* ResNet*) được đề xuất là ý tưởng rằng mỗi lớp bổ sung nên dễ dàng chứa chức năng nhận dạng như một trong những yếu tố của nó. Những cân nhắc này khá sâu sắc nhưng chúng dẫn đến một giải pháp đơn giản đáng ngạc nhiên, một khối * dư*. Với nó, ResNet đã giành chiến thắng trong ImageNet Large Scale Visual Recognition Challenge vào năm 2015. Thiết kế có ảnh hưởng sâu sắc đến cách xây dựng các mạng thần kinh sâu. 

## (** Khối** Residual**)

Chúng ta hãy tập trung vào một phần cục bộ của mạng thần kinh, như được mô tả trong :numref:`fig_residual_block`. Biểu thị đầu vào bằng $\mathbf{x}$. Chúng tôi giả định rằng bản đồ cơ bản mong muốn mà chúng tôi muốn có được bằng cách học là $f(\mathbf{x})$, được sử dụng làm đầu vào cho hàm kích hoạt ở trên cùng. Ở bên trái của :numref:`fig_residual_block`, phần trong hộp dòng dotted-line phải trực tiếp tìm hiểu bản đồ $f(\mathbf{x})$. Ở bên phải, phần trong hộp dòng dotted-line cần tìm hiểu ánh xạ dư* $f(\mathbf{x}) - \mathbf{x}$, đó là cách khối còn lại lấy tên của nó. Nếu ánh xạ nhận dạng $f(\mathbf{x}) = \mathbf{x}$ là ánh xạ cơ bản mong muốn, ánh xạ còn lại sẽ dễ học hơn: chúng ta chỉ cần đẩy trọng lượng và thành kiến của lớp trọng lượng trên (ví dụ: lớp kết nối hoàn toàn và lớp kết nối) trong hộp dòng dotted-line về 0. Con số bên phải trong :numref:`fig_residual_block` minh họa khối * dư* của ResNet, trong đó đường rắn mang đầu vào lớp $\mathbf{x}$ đến toán tử bổ sung được gọi là kết nối dư* (hoặc kết nối phím tắt *). Với các khối còn lại, đầu vào có thể chuyển tiếp tuyên truyền nhanh hơn thông qua các kết nối còn lại trên các lớp. 

![A regular block (left) and a residual block (right).](../img/residual-block.svg)
:label:`fig_residual_block`

ResNet tuân theo thiết kế lớp phức tạp $3\times 3$ đầy đủ của VGG. Khối còn lại có hai lớp ghép $3\times 3$ với cùng số kênh đầu ra. Mỗi lớp phức tạp được theo sau bởi một lớp chuẩn hóa hàng loạt và một chức năng kích hoạt ReLU. Sau đó, chúng tôi bỏ qua hai thao tác phức tạp này và thêm đầu vào trực tiếp trước chức năng kích hoạt ReLU cuối cùng. Kiểu thiết kế này đòi hỏi đầu ra của hai lớp ghép phải có cùng hình dạng với đầu vào, để chúng có thể được thêm vào lại với nhau. Nếu chúng ta muốn thay đổi số lượng kênh, chúng ta cần giới thiệu thêm một lớp ghép $1\times 1$ để chuyển đổi đầu vào thành hình dạng mong muốn cho thao tác bổ sung. Hãy để chúng tôi có một cái nhìn tại mã dưới đây.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

class Residual(nn.Block):  #@save
    """The Residual block of ResNet."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = npx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return npx.relu(Y + X)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):  #@save
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

class Residual(tf.keras.Model):  #@save
    """The Residual block of ResNet."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            num_channels, padding='same', kernel_size=3, strides=strides)
        self.conv2 = tf.keras.layers.Conv2D(
            num_channels, kernel_size=3, padding='same')
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(
                num_channels, kernel_size=1, strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)
```

Mã này tạo ra hai loại mạng: một trong đó chúng tôi thêm đầu vào vào đầu ra trước khi áp dụng tính phi tuyến ReLU bất cứ khi nào `use_1x1conv=False` và một loại mà chúng tôi điều chỉnh các kênh và độ phân giải bằng cách ghép $1 \times 1$ trước khi thêm. :numref:`fig_resnet_block` minh họa điều này: 

![ResNet block with and without $1 \times 1$ convolution.](../img/resnet-block.svg)
:label:`fig_resnet_block`

Bây giờ chúng ta hãy nhìn vào [** một tình huống mà đầu vào và đầu ra có cùng hình dạng**].

```{.python .input}
blk = Residual(3)
blk.initialize()
X = np.random.uniform(size=(4, 3, 6, 6))
blk(X).shape
```

```{.python .input}
#@tab pytorch
blk = Residual(3,3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab tensorflow
blk = Residual(3)
X = tf.random.uniform((4, 6, 6, 3))
Y = blk(X)
Y.shape
```

Chúng tôi cũng có tùy chọn để [** giảm một nửa chiều cao và chiều rộng đầu ra trong khi tăng số lượng kênh đầu ra**].

```{.python .input}
blk = Residual(6, use_1x1conv=True, strides=2)
blk.initialize()
blk(X).shape
```

```{.python .input}
#@tab pytorch
blk = Residual(3,6, use_1x1conv=True, strides=2)
blk(X).shape
```

```{.python .input}
#@tab tensorflow
blk = Residual(6, use_1x1conv=True, strides=2)
blk(X).shape
```

## [**Mô hình ResNet**]

Hai lớp ResNet đầu tiên giống như của GoogLeNet mà chúng tôi đã mô tả trước đây: lớp ghép $7\times 7$ với 64 kênh đầu ra và bước tiến 2 được theo sau bởi lớp tổng hợp tối đa $3\times 3$ với một sải chân là 2. Sự khác biệt là lớp chuẩn hóa hàng loạt được thêm vào sau mỗi lớp phức tạp trong ResNet.

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
b1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

GoogLeNet sử dụng bốn mô-đun được tạo thành từ các khối Inception. Tuy nhiên, ResNet sử dụng bốn mô-đun được tạo thành từ các khối còn lại, mỗi khối sử dụng một số khối còn lại với cùng một số kênh đầu ra. Số lượng kênh trong mô-đun đầu tiên giống như số kênh đầu vào. Kể từ khi một lớp tổng hợp tối đa với một sải chân là 2 đã được sử dụng, nó không phải là cần thiết để giảm chiều cao và chiều rộng. Trong khối dư đầu tiên cho mỗi mô-đun tiếp theo, số lượng kênh được tăng gấp đôi so với mô-đun trước đó và chiều cao và chiều rộng giảm một nửa. 

Bây giờ, chúng tôi thực hiện mô-đun này. Lưu ý rằng xử lý đặc biệt đã được thực hiện trên mô-đun đầu tiên.

```{.python .input}
def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk
```

```{.python .input}
#@tab pytorch
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
```

```{.python .input}
#@tab tensorflow
class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False,
                 **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residual_layers.append(
                    Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.residual_layers.append(Residual(num_channels))

    def call(self, X):
        for layer in self.residual_layers.layers:
            X = layer(X)
        return X
```

Sau đó, chúng tôi thêm tất cả các mô-đun vào ResNet. Ở đây, hai khối còn lại được sử dụng cho mỗi mô-đun.

```{.python .input}
net.add(resnet_block(64, 2, first_block=True),
        resnet_block(128, 2),
        resnet_block(256, 2),
        resnet_block(512, 2))
```

```{.python .input}
#@tab pytorch
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
```

```{.python .input}
#@tab tensorflow
b2 = ResnetBlock(64, 2, first_block=True)
b3 = ResnetBlock(128, 2)
b4 = ResnetBlock(256, 2)
b5 = ResnetBlock(512, 2)
```

Cuối cùng, giống như GoogLeNet, chúng ta thêm một lớp tổng hợp trung bình toàn cầu, tiếp theo là đầu ra lớp được kết nối hoàn toàn.

```{.python .input}
net.add(nn.GlobalAvgPool2D(), nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))
```

```{.python .input}
#@tab tensorflow
# Recall that we define this as a function so we can reuse later and run it
# within `tf.distribute.MirroredStrategy`'s scope to utilize various
# computational resources, e.g. GPUs. Also note that even though we have
# created b1, b2, b3, b4, b5 but we will recreate them inside this function's
# scope instead
def net():
    return tf.keras.Sequential([
        # The following layers are the same as b1 that we created earlier
        tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        # The following layers are the same as b2, b3, b4, and b5 that we
        # created earlier
        ResnetBlock(64, 2, first_block=True),
        ResnetBlock(128, 2),
        ResnetBlock(256, 2),
        ResnetBlock(512, 2),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dense(units=10)])
```

Có 4 lớp phức tạp trong mỗi mô-đun (không bao gồm lớp ghép $1\times 1$). Cùng với lớp kết nối $7\times 7$ đầu tiên và lớp kết nối hoàn toàn cuối cùng, tổng cộng có 18 lớp. Do đó, mô hình này thường được gọi là ResNet-18. Bằng cách định cấu hình các số kênh và khối còn lại khác nhau trong mô-đun, chúng ta có thể tạo các mô hình ResNet khác nhau, chẳng hạn như ResNet-152 lớp sâu hơn 152. Mặc dù kiến trúc chính của ResNet tương tự như kiến trúc của GoogLeNet, cấu trúc của ResNet đơn giản và dễ sửa đổi hơn. Tất cả những yếu tố này đã dẫn đến việc sử dụng ResNet nhanh chóng và rộng rãi. :numref:`fig_resnet18` mô tả đầy đủ ResNet-18. 

![The ResNet-18 architecture.](../img/resnet18.svg)
:label:`fig_resnet18`

Trước khi đào tạo ResNet, chúng ta hãy [** quan sát cách hình dạng đầu vào thay đổi trên các mô-đun khác nhau trong ResNet**]. Như trong tất cả các kiến trúc trước đó, độ phân giải giảm trong khi số lượng kênh tăng lên cho đến khi điểm mà một lớp tổng hợp trung bình toàn cầu tổng hợp tất cả các tính năng.

```{.python .input}
X = np.random.uniform(size=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform(shape=(1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

## [**Đào tạo**]

Chúng tôi đào tạo ResNet trên bộ dữ liệu Fashion-MNIST, giống như trước đây.

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## Tóm tắt

* Các lớp hàm lồng nhau là mong muốn. Học thêm một lớp trong các mạng thần kinh sâu như một chức năng nhận dạng (mặc dù đây là một trường hợp cực đoan) nên được thực hiện dễ dàng.
* Ánh xạ còn lại có thể tìm hiểu hàm nhận dạng dễ dàng hơn, chẳng hạn như đẩy các tham số trong lớp trọng lượng xuống 0.
* Chúng ta có thể đào tạo một mạng lưới thần kinh sâu hiệu quả bằng cách có các khối còn lại. Đầu vào có thể chuyển tiếp tuyên truyền nhanh hơn thông qua các kết nối còn lại trên các lớp.
* ResNet có ảnh hưởng lớn đến việc thiết kế các mạng thần kinh sâu tiếp theo, cả về tính chất phức tạp và tuần tự.

## Bài tập

1. Sự khác biệt lớn giữa khối Inception trong :numref:`fig_inception` và khối còn lại là gì? Sau khi loại bỏ một số đường dẫn trong khối Inception, chúng liên quan đến nhau như thế nào?
1. Tham khảo Bảng 1 trong giấy ResNet :cite:`He.Zhang.Ren.ea.2016` để thực hiện các biến thể khác nhau.
1. Đối với các mạng sâu hơn, ResNet giới thiệu kiến trúc “nút cổ chai” để giảm độ phức tạp của mô hình. Cố gắng thực hiện nó.
1. Trong các phiên bản tiếp theo của ResNet, các tác giả đã thay đổi cấu trúc “phức tạp, bình thường hóa hàng loạt và kích hoạt” thành cấu trúc “bình thường hóa hàng loạt, kích hoạt và phức tạp”. Tự cải thiện này. Xem Hình 1 trong :cite:`He.Zhang.Ren.ea.2016*1` để biết chi tiết.
1. Tại sao chúng ta không thể tăng độ phức tạp của hàm mà không bị ràng buộc, ngay cả khi các lớp hàm được lồng nhau?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/85)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/86)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/333)
:end_tab:
