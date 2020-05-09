<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Residual Networks (ResNet)
-->

# Mạng thặng dư (ResNet)
:label:`sec_resnet`

<!--
As we design increasingly deeper networks it becomes imperative to understand how adding layers can increase the complexity and expressiveness of the network.
Even more important is the ability to design networks where adding layers makes networks strictly more expressive rather than just different.
To make some progress we need a bit of theory.
-->

Khi thiết kế các mạng ngày càng sâu hơn, chúng ta cần hiểu rằng việc thêm các tầng vào mạng sẽ khiến tính phức tạp và biểu diễn của mạng tăng lên.
Điều quan trọng hơn nữa là khả năng thiết kế các mạng trong đó việc thêm các tầng vào mạng khiến chúng có tính biểu diễn cao hơn thay vì chỉ khác nhau.
Để đạt được một số tiến bộ, chúng ta cần một chút lý thuyết.

<!--
## Function Classes
-->

## Các Lớp Hàm Số

<!--
Consider $\mathcal{F}$, the class of functions that a specific network architecture (together with learning rates and other hyperparameter settings) can reach.
That is, for all $f \in \mathcal{F}$ there exists some set of parameters $W$ that can be obtained through training on a suitable dataset.
Let us assume that $f^*$ is the function that we really would like to find.
If it is in $\mathcal{F}$, we are in good shape but typically we will not be quite so lucky.
Instead, we will try to find some $f^*_\mathcal{F}$ which is our best bet within $\mathcal{F}$.
For instance, we might try finding it by solving the following optimization problem:
-->

Hãy xem xét $ \ mathcal {F} $, một lớp các hàm số mà một kiến trúc mạng cụ thể (cùng với tốc độ học và các siêu tham số khác) có thể biểu diễn được.
Đó là, luôn tồn tại một số tập tham số $W$ có thể tìm được thông qua việc huấn luyện trên một tập dữ liệu phù hợp, cho mọi hàm số $f \in \mathcal{F}$.
Giả sử $f^*$ là hàm số chúng ta đang muốn tìm kiếm.
Nếu hàm số này thuộc tập $\mathcal{F}$, thì việc tìm kiếm sẽ thuận lợi nhưng thường thì chúng ta sẽ không may mắn như vậy. 
Thay vào đó, chúng ta sẽ cố gắng tìm các hàm số $f^*_\mathcal{F}$ mà chúng ta tin rằng chúng nằm trong tập $\mathcal{F}$.  
Ví dụ, chúng ta có thể thử tìm các hàm số này bằng cách giải bài toán tối ưu sau đây:

$$f^*_\mathcal{F} := \mathop{\mathrm{argmin}}_f L(X, Y, f) \text{ subject to } f \in \mathcal{F}.$$

<!--
It is only reasonable to assume that if we design a different and more powerful architecture $\mathcal{F}'$ we should arrive at a better outcome.
In other words, we would expect that $f^*_{\mathcal{F}'}$ is "better" than $f^*_{\mathcal{F}}$.
However, if $\mathcal{F} \not\subseteq \mathcal{F}'$ there is no guarantee that this should even happen.
In fact, $f^*_{\mathcal{F}'}$ might well be worse.
This is a situation that we often encounter in practice---adding layers does not only make the network more expressive, it also changes it in sometimes not quite so predictable ways. :numref:`fig_functionclasses`illustrates this in slightly abstract terms.
-->

Chỉ có lý khi giả sử rằng nếu chúng ta thiết kế một kiến trúc $\mathcal{F}'$ khác biệt và mạnh mẽ hơn thì chúng ta mới đạt được kết quả tốt hơn.
Nói cách khác, chúng ta kỳ vọng rằng hàm số $f^*_{\mathcal{F}'}$ sẽ "tốt hơn" $f^*_{\mathcal{F}}$.
Tuy nhiên, nếu $\mathcal{F} \not\subseteq \mathcal{F}'$, thì sẽ không đảm bảo rằng điều này có thể xảy ra.
Trên thực tế, $f^*_{\mathcal{F}'}$ có thể còn tệ hơn.
Đây là tình huống hay xảy ra trong thực tế--- việc thêm các tầng không chỉ khiến cho một mạng có tính biểu diễn cao hơn, mà nó còn mang lại những thay đổi mà đôi khi rất khó lường.

<!--
![Left: non-nested function classes. The distance may in fact increase as the complexity increases. Right: with nested function classes this does not happen.](../img/functionclasses.svg)
-->

![Trái: Các lớp hàm số không lồng nhau. Khoảng cách này, trên thực tế, có thể tăng khi độ phức tạp tăng lên. Phải: với các lớp hàm số lồng nhau, điều này không xảy ra.](../img/functionclasses.svg)
:label:`fig_functionclasses`

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
Only if larger function classes contain the smaller ones are we guaranteed that increasing them strictly increases the expressive power of the network.
This is the question that He et al, 2016 considered when working on very deep computer vision models.
At the heart of ResNet is the idea that every additional layer should contain the identity function as one of its elements.
This means that if we can train the newly-added layer into an identity mapping $f(\mathbf{x}) = \mathbf{x}$, the new model will be as effective as the original model.
As the new model may get a better solution to fit the training dataset, the added layer might make it easier to reduce training errors.
Even better, the identity function rather than the null $f(\mathbf{x}) = 0$ should be the simplest function within a layer.
-->

*dịch đoạn phía trên*

<!--
These considerations are rather profound but they led to a surprisingly simple solution, a residual block.
With it, :cite:`He.Zhang.Ren.ea.2016` won the ImageNet Visual Recognition Challenge in 2015.
The design had a profound influence on how to build deep neural networks.
-->

*dịch đoạn phía trên*


<!--
## Residual Blocks
-->

## *dịch tiêu đề phía trên*

<!--
Let us focus on a local neural network, as depicted below.
Denote the input by $\mathbf{x}$.
We assume that the ideal mapping we want to obtain by learning is $f(\mathbf{x})$, to be used as the input to the activation function.
The portion within the dotted-line box in the left image must directly fit the mapping $f(\mathbf{x})$.
This can be tricky if we do not need that particular layer and we would much rather retain the input $\mathbf{x}$.
The portion within the dotted-line box in the right image now only needs to parametrize the *deviation* from the identity, since we return $\mathbf{x} + f(\mathbf{x})$.
In practice, the residual mapping is often easier to optimize.
We only need to set $f(\mathbf{x}) = 0$.
The right image in :numref:`fig_residual_block` illustrates the basic Residual Block of ResNet.
Similar architectures were later proposed for sequence models which we will study later.
-->

*dịch đoạn phía trên*

<!--
![The difference between a regular block (left) and a residual block (right). In the latter case, we can short-circuit the convolutions.](../img/residual-block.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/residual-block.svg)
:label:`fig_residual_block`

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
ResNet follows VGG's full $3\times 3$ convolutional layer design.
The residual block has two $3\times 3$ convolutional layers with the same number of output channels.
Each convolutional layer is followed by a batch normalization layer and a ReLU activation function.
Then, we skip these two convolution operations and add the input directly before the final ReLU activation function.
This kind of design requires that the output of the two convolutional layers be of the same shape as the input, so that they can be added together.
If we want to change the number of channels or the stride, we need to introduce an additional $1\times 1$ convolutional layer to transform the input into the desired shape for the addition operation.
Let us have a look at the code below.
-->

*dịch đoạn phía trên*

```{.python .input  n=1}
import d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

# Saved in the d2l package for later use
class Residual(nn.Block):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
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

<!--
This code generates two types of networks: one where we add the input to the output before applying the ReLU nonlinearity, 
and whenever `use_1x1conv=True`, one where we adjust channels and resolution by means of a $1 \times 1$ convolution before adding.
:numref:`fig_resnet_block` illustrates this:
-->

*dịch đoạn phía trên*

<!--
![Left: regular ResNet block; Right: ResNet block with 1x1 convolution](../img/resnet-block.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/resnet-block.svg)
:label:`fig_resnet_block`

<!--
Now let us look at a situation where the input and output are of the same shape.
-->

*dịch đoạn phía trên*

```{.python .input  n=2}
blk = Residual(3)
blk.initialize()
X = np.random.uniform(size=(4, 3, 6, 6))
blk(X).shape
```

<!--
We also have the option to halve the output height and width while increasing the number of output channels.
-->

*dịch đoạn phía trên*

```{.python .input  n=3}
blk = Residual(6, use_1x1conv=True, strides=2)
blk.initialize()
blk(X).shape
```

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## ResNet Model
-->

## *dịch tiêu đề phía trên*

<!--
The first two layers of ResNet are the same as those of the GoogLeNet we described before: 
the $7\times 7$ convolutional layer with 64 output channels and a stride of 2 is followed by the $3\times 3$ maximum pooling layer with a stride of 2.
The difference is the batch normalization layer added after each convolutional layer in ResNet.
-->

*dịch đoạn phía trên*

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

<!--
GoogLeNet uses four blocks made up of Inception blocks.
However, ResNet uses four modules made up of residual blocks, each of which uses several residual blocks with the same number of output channels.
The number of channels in the first module is the same as the number of input channels.
Since a maximum pooling layer with a stride of 2 has already been used, it is not necessary to reduce the height and width.
In the first residual block for each of the subsequent modules, the number of channels is doubled compared with that of the previous module, and the height and width are halved.
-->

*dịch đoạn phía trên*

<!--
Now, we implement this module.
Note that special processing has been performed on the first module.
-->

*dịch đoạn phía trên*

```{.python .input  n=4}
def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk
```

<!--
Then, we add all the residual blocks to ResNet.
Here, two residual blocks are used for each module.
-->

*dịch đoạn phía trên*

```{.python .input  n=5}
net.add(resnet_block(64, 2, first_block=True),
        resnet_block(128, 2),
        resnet_block(256, 2),
        resnet_block(512, 2))
```

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
Finally, just like GoogLeNet, we add a global average pooling layer, followed by the fully connected layer output.
-->

*dịch đoạn phía trên*

```{.python .input}
net.add(nn.GlobalAvgPool2D(), nn.Dense(10))
```

<!--
There are 4 convolutional layers in each module (excluding the $1\times 1$ convolutional layer).
Together with the first convolutional layer and the final fully connected layer, there are 18 layers in total.
Therefore, this model is commonly known as ResNet-18.
By configuring different numbers of channels and residual blocks in the module, we can create different ResNet models, such as the deeper 152-layer ResNet-152.
Although the main architecture of ResNet is similar to that of GoogLeNet, ResNet's structure is simpler and easier to modify.
All these factors have resulted in the rapid and widespread use of ResNet.
:numref:`fig_ResNetFull` is a diagram of the full ResNet-18.
-->

*dịch đoạn phía trên*

<!--
![ResNet 18](../img/ResNetFull.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/ResNetFull.svg)
:label:`fig_ResNetFull`

<!--
Before training ResNet, let us observe how the input shape changes between different modules in ResNet.
As in all previous architectures, the resolution decreases while the number of channels increases up until the point where a global average pooling layer aggregates all features.
-->

*dịch đoạn phía trên*

```{.python .input  n=6}
X = np.random.uniform(size=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!--
## Data Acquisition and Training
-->

## *dịch tiêu đề phía trên*

<!--
We train ResNet on the Fashion-MNIST dataset, just like before.
The only thing that has changed is the learning rate that decreased again, due to the more complex architecture.
-->

*dịch đoạn phía trên*

```{.python .input}
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

<!--
## Summary
-->

## Tóm tắt

<!--
* Residual blocks allow for a parametrization relative to the identity function $f(\mathbf{x}) = \mathbf{x}$.
* Adding residual blocks increases the function complexity in a well-defined manner.
* We can train an effective deep neural network by having residual blocks pass through cross-layer data channels.
* ResNet had a major influence on the design of subsequent deep neural networks, both for convolutional and sequential nature.
-->

*dịch đoạn phía trên*


<!--
## Exercises
-->

## Bài tập

<!--
1. Refer to Table 1 in the :cite:`He.Zhang.Ren.ea.2016` to implement different variants.
2. For deeper networks, ResNet introduces a "bottleneck" architecture to reduce model complexity. Try to implement it.
3. In subsequent versions of ResNet, the author changed the "convolution, batch normalization, and activation" architecture to the "batch normalization,
   activation, and convolution" architecture. Make this improvement yourself. See Figure 1 in :cite:`He.Zhang.Ren.ea.2016*1` for details.
4. Prove that if $\mathbf{x}$ is generated by a ReLU, the ResNet block does indeed include the identity function.
5. Why cannot we just increase the complexity of functions without bound, even if the function classes are nested?
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->
<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->


<!--
## [Discussions](https://discuss.mxnet.io/t/2359)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2359)
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
* Nguyễn Văn Quang

<!-- Phần 2 -->
*

<!-- Phần 3 -->
*

<!-- Phần 4 -->
*

<!-- Phần 5 -->
*

<!-- Phần 6 -->
*
