<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Densely Connected Networks (DenseNet)
-->

# Mạng Tích chập Kết nối Dày đặc (_Densely Connected Networks - DenseNet_)

<!--
ResNet significantly changed the view of how to parametrize the functions in deep networks.
DenseNet is to some extent the logical extension of this.
To understand how to arrive at it, let us take a small detour to theory.
Recall the Taylor expansion for functions. For scalars it can be written as
-->

ResNet đã làm thay đổi đáng kể quan điểm về cách tham số hóa các hàm số trong mạng học sâu.
Ở một mức độ nào đó, DenseNet có thể được coi là một mở rộng hợp lý của ResNet.
Để hiểu cách đi đến kết luận đó, ta cần tìm hiểu một chút lý thuyết.
Hãy nhớ lại công thức khai triển Taylor cho hàm một biến vô hướng

$$f(x) = f(0) + f'(x) x + \frac{1}{2} f''(x) x^2 + \frac{1}{6} f'''(x) x^3 + o(x^3).$$

<!--
## Function Decomposition
-->

## Sự Phân tách Hàm số

<!--
The key point is that it decomposes the function into increasingly higher order terms.
In a similar vein, ResNet decomposes functions into
-->

Điểm mấu chốt là khai triển Taylor phân tách hàm số thành các số hạng có bậc tăng dần.
Tương tự, ResNet phân tách các hàm số thành

$$f(\mathbf{x}) = \mathbf{x} + g(\mathbf{x}).$$

<!--
That is, ResNet decomposes $f$ into a simple linear term and a more complex nonlinear one.
What if we want to go beyond two terms? A solution was proposed by :cite:`Huang.Liu.Van-Der-Maaten.ea.2017` in the form of DenseNet, 
an architecture that reported record performance on the ImageNet dataset.
-->

Cụ thể là, ResNet tách hàm số $f$ thành một số hạng tuyến tính đơn giản và một số hạng phi tuyến phức tạp hơn.
Nếu ta muốn tách ra nhiều hơn hai số hạng thì sao? Một giải pháp đã được đề xuất bởi :cite:`Huang.Liu.Van-Der-Maaten.ea.2017` trong kiến trúc DenseNet. Kiến trúc này đạt được hiệu suất kỉ lục trên tập dữ liệu ImageNet.

<!--
![The main difference between ResNet (left) and DenseNet (right) in cross-layer connections: use of addition and use of concatenation. ](../img/densenet-block.svg)
-->

![Sự khác biệt chính giữa ResNet (bên trái) và DenseNet (bên phải) trong các kết nối liên tầng: sử dụng phép cộng và sử dụng phép nối.](../img/densenet-block.svg)
:label:`fig_densenet_block`

<!--
As shown in :numref:`fig_densenet_block`, the key difference between ResNet and DenseNet is that in the latter case outputs are *concatenated* rather than added.
As a result we perform a mapping from $\mathbf{x}$ to its values after applying an increasingly complex sequence of functions.
-->

Như được biểu diễn trong :numref:`fig_densenet_block`, điểm khác biệt chính giữa ResNet và DenseNet là trong kiến trúc DenseNet, đầu ra được *nối* với nhau thay vì được cộng lại.
Kết quả là ta thực hiện một ánh xạ từ $\mathbf{x}$ đến các giá trị của nó sau khi áp dụng một chuỗi các hàm với độ phức tạp tăng dần.

$$\mathbf{x} \to \left[\mathbf{x}, f_1(\mathbf{x}), f_2(\mathbf{x}, f_1(\mathbf{x})), f_3(\mathbf{x}, f_1(\mathbf{x}), f_2(\mathbf{x}, f_1(\mathbf{x})), \ldots\right].$$

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
In the end, all these functions are combined in an MLP to reduce the number of features again.
In terms of implementation this is quite simple---rather than adding terms, we concatenate them.
The name DenseNet arises from the fact that the dependency graph between variables becomes quite dense.
The last layer of such a chain is densely connected to all previous layers.
The main components that compose a DenseNet are dense blocks and transition layers.
The former defines how the inputs and outputs are concatenated, while the latter controls the number of channels so that it is not too large.
The dense connections are shown in :numref:`fig_densenet`.
-->

*dịch đoạn phía trên*

<!--
![Dense connections in DenseNet](../img/densenet.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/densenet.svg)
:label:`fig_densenet`

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Dense Blocks
-->

## *dịch tiêu đề phía trên*

<!--
DenseNet uses the modified "batch normalization, activation, and convolution" architecture of ResNet (see the exercise in :numref:`sec_resnet`).
First, we implement this architecture in the `conv_block` function.
-->

*dịch đoạn phía trên*

```{.python .input  n=1}
import d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk
```

<!--
A dense block consists of multiple `conv_block` units, each using the same number of output channels.
In the forward computation, however, we concatenate the input and output of each block on the channel dimension.
-->

*dịch đoạn phía trên*

```{.python .input  n=2}
class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate the input and output of each block on the channel
            # dimension
            X = np.concatenate((X, Y), axis=1)
        return X
```

<!--
In the following example, we define a convolution block with two blocks of 10 output channels.
When using an input with 3 channels, we will get an output with the $3+2\times 10=23$ channels.
The number of convolution block channels controls the increase in the number of output channels relative to the number of input channels.
This is also referred to as the growth rate.
-->

*dịch đoạn phía trên*

```{.python .input  n=8}
blk = DenseBlock(2, 10)
blk.initialize()
X = np.random.uniform(size=(4, 3, 8, 8))
Y = blk(X)
Y.shape
```

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
## Transition Layers
-->

## *dịch tiêu đề phía trên*

<!--
Since each dense block will increase the number of channels, adding too many of them will lead to an excessively complex model.
A transition layer is used to control the complexity of the model.
It reduces the number of channels by using the $1\times 1$ convolutional layer and halves the height 
and width of the average pooling layer with a stride of 2, further reducing the complexity of the model.
-->

*dịch đoạn phía trên*

```{.python .input  n=3}
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk
```

<!--
Apply a transition layer with 10 channels to the output of the dense block in the previous example.
This reduces the number of output channels to 10, and halves the height and width.
-->

*dịch đoạn phía trên*

```{.python .input}
blk = transition_block(10)
blk.initialize()
blk(Y).shape
```

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
## DenseNet Model
-->

## *dịch tiêu đề phía trên*

<!--
Next, we will construct a DenseNet model.
DenseNet first uses the same single convolutional layer and maximum pooling layer as ResNet.
-->

*dịch đoạn phía trên*

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
Then, similar to the four residual blocks that ResNet uses, DenseNet uses four dense blocks.
Similar to ResNet, we can set the number of convolutional layers used in each dense block.
Here, we set it to 4, consistent with the ResNet-18 in the previous section.
Furthermore, we set the number of channels (i.e., growth rate) for the convolutional layers in the dense block to 32, so 128 channels will be added to each dense block.
-->

*dịch đoạn phía trên*

<!--
In ResNet, the height and width are reduced between each module by a residual block with a stride of 2.
Here, we use the transition layer to halve the height and width and halve the number of channels.
-->

*dịch đoạn phía trên*

```{.python .input  n=5}
# Num_channels: the current number of channels
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    net.add(DenseBlock(num_convs, growth_rate))
    # This is the number of output channels in the previous dense block
    num_channels += num_convs * growth_rate
    # A transition layer that haves the number of channels is added between
    # the dense blocks
    if i != len(num_convs_in_dense_blocks) - 1:
        num_channels //= 2
        net.add(transition_block(num_channels))
```

<!--
Similar to ResNet, a global pooling layer and fully connected layer are connected at the end to produce the output.
-->

*dịch đoạn phía trên*

```{.python .input}
net.add(nn.BatchNorm(),
        nn.Activation('relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(10))
```

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
## Data Acquisition and Training
-->

## *dịch tiêu đề phía trên*

<!--
Since we are using a deeper network here, in this section, we will reduce the input height and width from 224 to 96 to simplify the computation.
-->

*dịch đoạn phía trên*

```{.python .input}
lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

<!--
## Summary
-->

## Tóm tắt

<!--
* In terms of cross-layer connections, unlike ResNet, where inputs and outputs are added together, DenseNet concatenates inputs and outputs on the channel dimension.
* The main units that compose DenseNet are dense blocks and transition layers.
* We need to keep the dimensionality under control when composing the network by adding transition layers that shrink the number of channels again.
-->

*dịch đoạn phía trên*

<!--
## Exercises
-->

## Bài tập

<!--
1. Why do we use average pooling rather than max-pooling in the transition layer?
2. One of the advantages mentioned in the DenseNet paper is that its model parameters are smaller than those of ResNet. Why is this the case?
3. One problem for which DenseNet has been criticized is its high memory consumption.
    * Is this really the case? Try to change the input shape to $224\times 224$ to see the actual (GPU) memory consumption.
    * Can you think of an alternative means of reducing the memory consumption? How would you need to change the framework?
4. Implement the various DenseNet versions presented in Table 1 of :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`.
5. Why do we not need to concatenate terms if we are just interested in $\mathbf{x}$ and $f(\mathbf{x})$ for ResNet? Why do we need this for more than two layers in DenseNet?
6. Design a DenseNet for fully connected networks and apply it to the Housing Price prediction task.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->
<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2360)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2360)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.

* Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md
-->

* Đoàn Võ Duy Thanh
<!-- Phần 1 -->
* Nguyễn Duy Du
* Nguyễn Văn Cường

<!-- Phần 2 -->
*

<!-- Phần 3 -->
*

<!-- Phần 4 -->
*

<!-- Phần 5 -->
*
