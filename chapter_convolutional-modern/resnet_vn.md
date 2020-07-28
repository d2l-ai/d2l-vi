<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Residual Networks (ResNet)
-->

# Mạng phần dư (ResNet)
:label:`sec_resnet`

<!--
As we design increasingly deeper networks it becomes imperative to understand how adding layers can increase the complexity and expressiveness of the network.
Even more important is the ability to design networks where adding layers makes networks strictly more expressive rather than just different.
To make some progress we need a bit of theory.
-->

Khi thiết kế các mạng ngày càng sâu, ta cần hiểu việc thêm các tầng sẽ tăng độ phức tạp và khả năng biểu diễn của mạng như thế nào.
Quan trọng hơn là khả năng thiết kế các mạng trong đó việc thêm các tầng vào mạng chắc chắn sẽ làm tăng tính biểu diễn thay vì chỉ tạo ra một chút khác biệt.
Để làm được điều này, chúng ta cần một chút lý thuyết.

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

Coi $\mathcal{F}$ là một lớp các hàm mà một kiến trúc mạng cụ thể (cùng với tốc độ học và các siêu tham số khác) có thể đạt được.
Nói cách khác, với mọi hàm số $f \in \mathcal{F}$, luôn tồn tại một số tập tham số $W$ có thể tìm được bằng việc huấn luyện trên một tập dữ liệu phù hợp.
Giả sử $f^*$ là hàm cần tìm.
Sẽ rất thuận lợi nếu hàm này thuộc tập $\mathcal{F}$, nhưng thường không may mắn như vậy. 
Thay vào đó, ta sẽ cố gắng tìm các hàm số $f^*_\mathcal{F}$ tốt nhất có thể trong tập $\mathcal{F}$.  
Ví dụ, có thể thử tìm $f^*_\mathcal{F}$ bằng cách giải bài toán tối ưu sau:

$$f^*_\mathcal{F} := \mathop{\mathrm{argmin}}_f L(X, Y, f) \text{ đối~tượng~thoả~mãn } f \in \mathcal{F}.$$

<!--
It is only reasonable to assume that if we design a different and more powerful architecture $\mathcal{F}'$ we should arrive at a better outcome.
In other words, we would expect that $f^*_{\mathcal{F}'}$ is "better" than $f^*_{\mathcal{F}}$.
However, if $\mathcal{F} \not\subseteq \mathcal{F}'$ there is no guarantee that this should even happen.
In fact, $f^*_{\mathcal{F}'}$ might well be worse.
This is a situation that we often encounter in practice---adding layers does not only make the network more expressive, it also changes it in sometimes not quite so predictable ways. :numref:`fig_functionclasses`illustrates this in slightly abstract terms.
-->

Khá hợp lý khi giả sử rằng nếu thiết kế một kiến trúc khác $\mathcal{F}'$ mạnh mẽ hơn thì sẽ đạt được kết quả tốt hơn.
Nói cách khác, ta kỳ vọng hàm số $f^*_{\mathcal{F}'}$ sẽ "tốt hơn" $f^*_{\mathcal{F}}$.
Tuy nhiên, nếu $\mathcal{F} \not\subseteq \mathcal{F}'$, thì không khẳng định được $f^*_{\mathcal{F}'}$ "tốt hơn" $f^*_{\mathcal{F}}$.
Trên thực tế, $f^*_{\mathcal{F}'}$ có thể còn tệ hơn.
Và đây là trường hợp thường xuyên xảy ra --- việc thêm các tầng không phải lúc nào cũng tăng tính biểu diễn của mạng mà đôi khi còn tạo ra những thay đổi rất khó lường.
:numref:`fig_functionclasses` minh hoạ rõ hơn điều này.

<!--
![Left: non-nested function classes. The distance may in fact increase as the complexity increases. Right: with nested function classes this does not happen.](../img/functionclasses.svg)
-->

![Hình trái: Các lớp hàm số tổng quát. Khoảng cách đến hàm cần tìm $f^*$ (ngôi sao), trên thực tế có thể tăng khi độ phức tạp tăng lên. Hình phải: với các lớp hàm số lồng nhau, điều này không xảy ra.](../img/functionclasses.svg)
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

Chỉ khi các lớp hàm lớn hơn chứa các lớp nhỏ hơn, thì mới đảm bảo rằng việc tăng thêm các tầng sẽ tăng khả năng biểu diễn của mạng.
Đây là câu hỏi mà He và các cộng sự đã suy nghĩ khi nghiên cứu các mô hình thị giác sâu năm 2016.
Ý tưởng trọng tâm của ResNet là mỗi tầng được thêm vào nên có một thành phần là hàm số đồng nhất. 
Điều này có nghĩa rằng, nếu ta huấn luyện tầng mới được thêm vào thành một ánh xạ đồng nhất $f(\mathbf{x}) = \mathbf{x}$, thì mô hình mới sẽ hiệu quả ít nhất bằng mô hình ban đầu.
Vì tầng được thêm vào có thể khớp dữ liệu huấn luyện tốt hơn, dẫn đến sai số huấn luyện cũng nhỏ hơn.
Tốt hơn nữa, hàm số đồng nhất nên là hàm đơn giản nhất trong một tầng thay vì hàm null $f(\mathbf{x}) = 0$.

<!--
These considerations are rather profound but they led to a surprisingly simple solution, a residual block.
With it, :cite:`He.Zhang.Ren.ea.2016` won the ImageNet Visual Recognition Challenge in 2015.
The design had a profound influence on how to build deep neural networks.
-->

Cách suy nghĩ này khá trừu tượng nhưng lại dẫn đến một lời giải đơn giản đáng ngạc nhiên, một khối phần dư (*residual block*).
Với ý tưởng này, :cite:`He.Zhang.Ren.ea.2016` đã chiến thắng cuộc thi Nhận dạng Ảnh ImageNet năm 2015.
Thiết kế này có ảnh hưởng sâu sắc tới việc xây dựng các mạng nơ-ron sâu.

<!--
## Residual Blocks
-->

## Khối phần dư

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

Bây giờ, hãy tập trung vào mạng nơ-ron dưới đây.
Ký hiệu đầu vào là $\mathbf{x}$.
Giả sử ánh xạ lý tưởng muốn học được là $f(\mathbf{x})$, và được dùng làm đầu vào của hàm kích hoạt.
Phần nằm trong viền nét đứt bên trái phải khớp trực tiếp với ánh xạ $f(\mathbf{x})$.
Điều này có thể không đơn giản nếu chúng ta không cần khối đó và muốn giữ lại đầu vào $\mathbf{x}$.
Khi đó, phần nằm trong viền nét đứt bên phải chỉ cần tham số hoá *độ lệch* khỏi giá trị $\mathbf{x}$, bởi vì ta đã trả về $\mathbf{x} + f(\mathbf{x})$.
Trên thực tế, ánh xạ phần dư thường dễ tối ưu hơn,
vì chỉ cần đặt $f(\mathbf{x}) = 0$.
Nửa bên phải :numref:`fig_residual_block` mô tả khối phần dư cơ bản của ResNet.
Về sau, những kiến trúc tương tự đã được đề xuất cho các mô hình chuỗi (*sequence model*), sẽ đề cập ở chương sau.

<!--
![The difference between a regular block (left) and a residual block (right). In the latter case, we can short-circuit the convolutions.](../img/residual-block.svg)
-->

![Sự khác biệt giữa một khối thông thường (trái) và một khối phần dư (phải). Trong khối phần dư, ta có thể nối tắt các tích chập.](../img/residual-block.svg)
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

ResNet có thiết kế tầng tích chập $3\times 3$ giống VGG.
Khối phần dư có hai tầng tích chập $3\times 3$ với cùng số kênh đầu ra.
Mỗi tầng tích chập được theo sau bởi một tầng chuẩn hóa theo batch và một hàm kích hoạt ReLU.
Ta đưa đầu vào qua khối phần dư rồi cộng với chính nó trước hàm kích hoạt ReLU cuối cùng.
Thiết kế này đòi hỏi đầu ra của hai tầng tích chập phải có cùng kích thước với đầu vào, để có thể cộng lại với nhau.
Nếu muốn thay đổi số lượng kênh hoặc sải bước trong khối phần dư, cần thêm một tầng tích chập $1\times 1$ để thay đổi kích thước đầu vào tương ứng ở nhánh ngoài.
Hãy cùng xem đoạn mã bên dưới.

```{.python .input  n=1}
from d2l import mxnet as d2l
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

Đoạn mã này tạo ra hai loại mạng: một loại cộng đầu vào vào đầu ra trước khi áp dụng hàm phi tuyến ReLU (khi `use_1x1conv=True`), 
còn ở loại thứ hai chúng ta thay đổi số kênh và độ phân giải bằng một tầng tích chập $1 \times 1$ trước khi thực hiện phép cộng.
:numref:`fig_resnet_block` minh họa điều này:

<!--
![Left: regular ResNet block; Right: ResNet block with 1x1 convolution](../img/resnet-block.svg)
-->

![Trái: khối ResNet thông thường; Phải: Khối ResNet với tầng tích chập 1x1](../img/resnet-block.svg)
:label:`fig_resnet_block`

<!--
Now let us look at a situation where the input and output are of the same shape.
-->

Giờ hãy xem xét tình huống khi cả đầu vào và đầu ra có cùng kích thước.

```{.python .input  n=2}
blk = Residual(3)
blk.initialize()
X = np.random.uniform(size=(4, 3, 6, 6))
blk(X).shape
```

<!--
We also have the option to halve the output height and width while increasing the number of output channels.
-->

Chúng ta cũng có thể giảm một nửa kích thước chiều cao và chiều rộng của đầu ra trong khi tăng số kênh.

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

## Mô hình ResNet

<!--
The first two layers of ResNet are the same as those of the GoogLeNet we described before: 
the $7\times 7$ convolutional layer with 64 output channels and a stride of 2 is followed by the $3\times 3$ maximum pooling layer with a stride of 2.
The difference is the batch normalization layer added after each convolutional layer in ResNet.
-->

Hai tầng đầu tiên của ResNet giống hai tầng đầu tiên của GoogLeNet:
tầng tích chập $7\times 7$ với 64 kênh đầu ra và sải bước 2, theo sau bởi tầng gộp cực đại $3 \times 3$ với sải bước 2.
Sự khác biệt là trong ResNet, mỗi tầng tích chập theo sau bởi tầng chuẩn hóa theo batch.

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

GoogLeNet sử dụng bốn mô-đun được tạo thành từ các khối Inception.
ResNet sử dụng bốn mô-đun được tạo thành từ các khối phần dư có cùng số kênh đầu ra.
Mô-đun đầu tiên có số kênh bằng số kênh đầu vào.
Vì trước đó đã sử dụng tầng gộp cực đại với sải bước 2, nên không cần phải giảm chiều cao và chiều rộng ở mô-đun này.
Trong các mô-đun sau, khối phần dư đầu tiên nhân đôi số kênh, đồng thời giảm một nửa chiều cao và chiều rộng.

<!--
Now, we implement this module.
Note that special processing has been performed on the first module.
-->

Bây giờ ta sẽ lập trình mô-đun này.
Chú ý rằng mô-đun đầu tiên được xử lý khác một chút.

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

Sau đó, chúng ta thêm các khối phần dư vào ResNet.
Ở đây, mỗi mô-đun có hai khối phần dư.

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

Cuối cùng, giống như GoogLeNet, ta thêm một tầng gộp trung bình toàn cục và một tầng kết nối đầy đủ.

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

Có 4 tầng tích chập trong mỗi mô-đun (không tính tầng tích chập $1 \times 1$).
Cộng thêm tầng tích chập đầu tiên và tầng kết nối đầy đủ cuối cùng, mô hình có tổng cộng 18 tầng.
Do đó, mô hình này thường được gọi là ResNet-18.
Có thể thay đổi số kênh và các khối phần dư trong mô-đun để tạo ra các mô hình ResNet khác nhau, 
ví dụ mô hình 152 tầng của ResNet-152.
Mặc dù có kiến trúc lõi tương tự như GoogLeNet, cấu trúc của ResNet đơn giản và dễ sửa đổi hơn.
Tất cả các yếu tố này dẫn đến sự phổ cập nhanh chóng và rộng rãi của ResNet.
:numref:`fig_ResNetFull` là sơ đồ đầy đủ của ResNet-18.

<!--
![ResNet 18](../img/ResNetFull.svg)
-->

![ResNet-18](../img/ResNetFull.svg)
:label:`fig_ResNetFull`

<!--
Before training ResNet, let us observe how the input shape changes between different modules in ResNet.
As in all previous architectures, the resolution decreases while the number of channels increases up until the point where a global average pooling layer aggregates all features.
-->

Trước khi huấn luyện, hãy quan sát thay đổi của kích thước đầu vào qua các mô-đun khác nhau trong ResNet.
Như trong tất cả các kiến trúc trước, độ phân giải giảm trong khi số lượng kênh tăng đến khi tầng gộp trung bình toàn cục tổng hợp tất cả các đặc trưng.

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

## Thu thập dữ liệu và Huấn luyện 

<!--
We train ResNet on the Fashion-MNIST dataset, just like before.
The only thing that has changed is the learning rate that decreased again, due to the more complex architecture.
-->

Giống như các phần trước, chúng ta huấn luyện ResNet trên bộ dữ liệu Fashion-MNIST.
Thay đổi duy nhất là giảm tốc độ học lại do kiến trúc mạng phức tạp hơn.

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

* Khối phần dư cho phép tham số hóa đến hàm đồng nhất $f(\mathbf{x}) = \mathbf{x}$.
* Thêm các khối phần dư làm tăng độ phức tạp của hàm số theo một cách chủ đích.
* Chúng ta có thể huấn luyện hiệu quả mạng nơ-ron sâu nhờ khối phần dư chuyển dữ liệu liên tầng.
* ResNet có ảnh hưởng lớn đến thiết kế sau này của các mạng nơ-ron sâu, cả tích chập và tuần tự.

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

1. Tham khảo Bảng 1 trong :cite:`He.Zhang.Ren.ea.2016` để lập trình các biến thể khác nhau.
2. Đối với các mạng sâu hơn, ResNet giới thiệu kiến trúc "thắt cổ chai" để giảm độ phức tạp của mô hình. Hãy thử lập trình kiến trúc đó.
3. Trong các phiên bản sau của ResNet, tác giả đã thay đổi kiến trúc "tích chập, chuẩn hóa theo batch, và hàm kích hoạt" thành "chuẩn hóa theo batch, hàm kích hoạt, và tích chập". Hãy tự lập trình kiến trúc này. Xem hình 1 trong :cite:`He.Zhang.Ren.ea.2016*1` để biết chi tiết.
4. Chứng minh rằng nếu $\mathbf{x}$ được tạo ra bởi ReLU thì khối ResNet sẽ bao gồm hàm số đồng nhất.
5. Tại sao không thể tăng không giới hạn độ phức tạp của các hàm số, ngay cả với các lớp hàm lồng nhau?

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->
<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2359)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Văn Quang
* Nguyễn Cảnh Thướng
* Lê Khắc Hồng Phúc
* Nguyễn Văn Cường
* Nguyễn Đình Nam
* Phạm Minh Đức
* Phạm Hồng Vinh
