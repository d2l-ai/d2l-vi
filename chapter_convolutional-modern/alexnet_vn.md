<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Deep Convolutional Neural Networks (AlexNet)
-->

# *dịch tiêu đề phía trên*
:label:`sec_alexnet`


<!--
Although convolutional neural networks were well known in the computer vision and machine learning communities following the introduction of LeNet, they did not immediately dominate the field.
Although LeNet achieved good results on early small datasets, the performance and feasability of training convolutional networks on larger, more realistic datasets had yet to be established.
In fact, for much of the intervening time between the early 1990s and the watershed results of 2012, neural networks were often surpassed by other machine learning methods, such as support vector machines.
-->

*dịch đoạn phía trên*


<!--
For computer vision, this comparison is perhaps not fair.
That is although the inputs to convolutional networks consist of raw or lightly-processed (e.g., by centering) pixel values, practitioners would never feed raw pixels into traditional models.
Instead, typical computer vision pipelines consisted of manually engineering feature extraction pipelines.
Rather than *learn the features*, the features were *crafted*.
Most of the progress came from having more clever ideas for features, and the learning algorithm was often relegated to an afterthought.
-->

*dịch đoạn phía trên*

<!--
Although some neural network accelerators were available in the 1990s, they were not yet sufficiently powerful to make deep multichannel, 
multilayer convolutional neural networks with a large number of parameters.
Moreover, datasets were still relatively small.
Added to these obstacles, key tricks for training neural networks including parameter initialization heuristics, clever variants of stochastic gradient descent,
non-squashing activation functions, and effective regularization techniques were still missing.
-->

*dịch đoạn phía trên*

<!--
Thus, rather than training *end-to-end* (pixel to classification) systems, classical pipelines looked more like this:
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
1. Obtain an interesting dataset. In early days, these datasets required expensive sensors (at the time, 1 megapixel images were state of the art).
2. Preprocess the dataset with hand-crafted features based on some knowledge of optics, geometry, other analytic tools, 
and occasionally on the serendipitous discoveries of lucky graduate students.
3. Feed the data through a standard set of feature extractors such as [SIFT](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform), the Scale-Invariant Feature Transform, 
or [SURF](https://en.wikipedia.org/wiki/Speeded_up_robust_features), the Speeded-Up Robust Features, or any number of other hand-tuned pipelines.
4. Dump the resulting representations into your favorite classifier, likely a linear model or kernel method, to learn a classifier.
-->

*dịch đoạn phía trên*

<!--
If you spoke to machine learning researchers, they believed that machine learning was both important and beautiful.
Elegant theories proved the properties of various classifiers.
The field of machine learning was thriving, rigorous and eminently useful.
However, if you spoke to a computer vision researcher, you’d hear a very different story.
The dirty truth of image recognition, they’d tell you, is that features, not learning algorithms, drove progress.
Computer vision researchers justifiably believed that a slightly bigger or cleaner dataset
or a slightly improved feature-extraction pipeline mattered far more to the final accuracy than any learning algorithm.
-->

*dịch đoạn phía trên*

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Learning Feature Representation
-->

## *dịch tiêu đề phía trên*

<!--
Another way to cast the state of affairs is that the most important part of the pipeline was the representation.
And up until 2012 the representation was calculated mechanically.
In fact, engineering a new set of feature functions, improving results, and writing up the method was a prominent genre of paper.
[SIFT](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform),
[SURF](https://en.wikipedia.org/wiki/Speeded_up_robust_features),
[HOG](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients),
[Bags of visual words](https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision)
and similar feature extractors ruled the roost.
-->

*dịch đoạn phía trên*

<!--
Another group of researchers, including Yann LeCun, Geoff Hinton, Yoshua Bengio,
Andrew Ng, Shun-ichi Amari, and Juergen Schmidhuber, had different plans.
They believed that features themselves ought to be learned.
Moreover, they believed that to be reasonably complex, the features ought to be hierarchically composed with multiple jointly learned layers, each with learnable parameters.
In the case of an image, the lowest layers might come to detect edges, colors, and textures.
Indeed, :cite:`Krizhevsky.Sutskever.Hinton.2012` proposed a new variant of a convolutional neural network which achieved excellent performance in the ImageNet challenge.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
Interestingly in the lowest layers of the network, the model learned feature extractors that resembled some traditional filters.
:numref:`fig_filters` is reproduced from this paper and describes lower-level image descriptors.
-->

*dịch đoạn phía trên*

<!--
![Image filters learned by the first layer of AlexNet](../img/filters.png)
-->

![*dịch chú thích ảnh phía trên*](../img/filters.png)
:width:`400px`
:label:`fig_filters`

<!--
Higher layers in the network might build upon these representations to represent larger structures, like eyes, noses, blades of grass, etc.
Even higher layers might represent whole objects like people, airplanes, dogs, or frisbees.
Ultimately, the final hidden state learns a compact representation of the image that summarizes its contents such that data belonging to different categories be separated easily.
-->

*dịch đoạn phía trên*


<!--
While the ultimate breakthrough for many-layered convolutional networks came in 2012, 
a core group of researchers had dedicated themselves to this idea, attempting to learn hierarchical representations of visual data for many years.
The ultimate breakthrough in 2012 can be attributed to two key factors.
-->

*dịch đoạn phía trên*

<!--
### Missing Ingredient - Data
-->

### *dịch tiêu đề phía trên*

<!--
Deep models with many layers require large amounts of data in order to enter the regime where they significantly outperform traditional methods based on convex optimizations (e.g., linear and kernel methods).
However, given the limited storage capacity of computers, the relative expense of sensors, and the comparatively tighter research budgets in the 1990s, most research relied on tiny datasets.
Numerous papers addressed the UCI collection of datasets, many of which contained only hundreds or (a few) thousands of images captured in unnatural settings with low resolution.
-->

*dịch đoạn phía trên*

<!--
In 2009, the ImageNet dataset was released, challenging researchers to learn models from 1 million examples, 1,000 each from 1,000 distinct categories of objects.
The researchers, led by Fei-Fei Li, who introduced this dataset leveraged Google Image Search to prefilter large candidate sets for each category 
and employed the Amazon Mechanical Turk crowdsourcing pipeline to confirm for each image whether it belonged to the associated category.
This scale was unprecedented.
The associated competition, dubbed the ImageNet Challenge pushed computer vision and machine learning research forward,
challenging researchers to identify which models performed best at a greater scale than academics had previously considered.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
### Missing Ingredient - Hardware
-->

### Yếu tố bị thiếu - Phần cứng

<!--
Deep learning models are voracious consumers of compute cycles.
Training can take hundreds of epochs, and each iteration requires passing data through many layers of computationally-expensive linear algebra operations.
This is one of the main reasons why in the 90s and early 2000s, simple algorithms based on the more-efficiently optimized convex objectives were preferred.
-->

Các mô hình học sâu cần rất nhiều chu kì tính toán. 
Quá trình huấn luyện sẽ cần hàng trăm epoch và mỗi vòng lặp yêu cầu đưa dữ liệu qua rất nhiều tầng các phép toán đại số tuyến tính cồng kềnh. 
Đây là một trong những lý do chính tại sao vào những năm 90 và đầu những năm 2000, những thuật toán đơn giản được tối ưu hiệu quả dựa trên các hàm mục tiêu lồi  lại được ưa chuộng hơn.

<!--
Graphical processing units (GPUs) proved to be a game changer in make deep learning feasible.
These chips had long been developed for accelerating graphics processing to benefit computer games.
In particular, they were optimized for high throughput 4x4 matrix-vector products, which are needed for many computer graphics tasks.
Fortunately, this math is strikingly similar to that required to calculate convolutional layers.
Around that time, NVIDIA and ATI had begun optimizing GPUs for general compute operations, going as far as to market them as General Purpose GPUs (GPGPU).
-->

Bộ xử lý đồ hoạ (GPUs) đóng vai trò thay đổi hoàn toàn cuộc chơi khi làm cho việc học sâu trở nên khả thi. 
Những vi xử lý này đã được phát triển một thời gian dài để tăng tốc độ xử lý đồ họa dành cho các trò chơi máy tính. 
Cụ thể, chúng được tối ưu hoá cho các phép nhân ma trận - vector 4x4 thông lượng cao, cần thiết cho nhiều tác vụ đồ hoạ. 
May mắn thay, phép toán này rất giống với phép tính cần thiết cho việc tính toán các tầng chập. 
Trong khoảng thời gian này, các công ty NVIDIA và ATI đã bắt đầu tối ưu GPU cho các mục đích tính toán tổng quát, còn tới mức tiếp thị chúng dưới dạng GPU đa dụng (*General Purpose GPUs - GPGPU*).

<!--
To provide some intuition, consider the cores of a modern microprocessor (CPU).
Each of the cores is fairly powerful running at a high clock frequency and sporting large caches (up to several MB of L3).
Each core is well-suited to executing a wide range of instructions, with branch predictors, a deep pipeline, and other bells and whistles that enable it to run a large variety of programs.
This apparent strength, however, is also its Achilles heel: general purpose cores are very expensive to build.
They require lots of chip area, a sophisticated support structure (memory interfaces, caching logic between cores, high speed interconnects, etc.),
and they are comparatively bad at any single task.
Modern laptops have up to 4 cores, and even high end servers rarely exceed 64 cores, simply because it is not cost effective.
-->

Để hình dung rõ hơn, hãy cùng xem lại các nhân của bộ vi xử lý ngày nay (CPU). Mỗi nhân thì khá mạnh khi chạy ở tần số xung nhịp cao với và với bộ nhớ đệm lớn (lên đến vài MB ở bộ nhớ đệm L3). 
Mỗi nhân phù hợp với việc thực hiện hàng loạt các lệnh, với các bộ dự báo rẽ nhánh, một đường ống lệnh dài, và những tính năng phụ trợ khác cho phép nó có khả năng chạy nhiều chương trình lớn khác nhau. 
Tuy nhiên, sức mạnh rõ rệt này cũng có điểm yếu: sản xuất các nhân đa dụng rất đắt đỏ. 
Chúng đòi hỏi nhiều diện tích cho chip, cùng cấu trúc hỗ trợ phức tạp (giao diện bộ nhớ, logic bộ nhớ đệm giữa các nhân, kết nối tốc độ cao, v.v.), và chúng tương đối tệ ở bất kỳ tác vụ đơn lẻ nào. 
Những máy tính xách tay ngày nay có tới 4 nhân, và thậm chí những máy chủ cao cấp hiếm khi vượt quá 64 nhân, đơn giản bởi không hiệu quả về chi phí.

<!--
By comparison, GPUs consist of 100-1000 small processing elements (the details differ somewhat between NVIDIA, ATI, ARM and other chip vendors), 
often grouped into larger groups (NVIDIA calls them warps).
While each core is relatively weak, sometimes even running at sub-1GHz clock frequency, 
it is the total number of such cores that makes GPUs orders of magnitude faster than CPUs.
For instance, NVIDIA's latest Volta generation offers up to 120 TFlops per chip for specialized instructions (and up to 24 TFlops for more general purpose ones), 
while floating point performance of CPUs has not exceeded 1 TFlop to date.
The reason for why this is possible is actually quite simple: first, power consumption tends to grow *quadratically* with clock frequency.
Hence, for the power budget of a CPU core that runs 4x faster (a typical number), you can use 16 GPU cores at 1/4 the speed, which yields 16 x 1/4 = 4x the performance.
Furthermore, GPU cores are much simpler (in fact, for a long time they were not even *able* to execute general purpose code), which makes them more energy efficient.
Last, many operations in deep learning require high memory bandwidth.
Again, GPUs shine here with buses that are at least 10x as wide as many CPUs.
-->

Để so sánh, GPUs bao gồm 100-1000 các phần tử xử lý nhỏ (các chi tiết khác nhau đôi chút giữ NVIDIA, ATI, ARM và các nhà sản xuất chip khác), 
thường được gộp thành các nhóm lớn hơn (NVIDIA gọi các nhóm này là luồng (*warp*). 
Mặc dù mỗi nhân thì tương đối yếu, đôi khi thậm chí chạy ở tần số xung nhịp dưới 1GHZ,
nhưng số lượng của những nhân này làm cho GPUs có tốc độ nhanh hơn so với CPUs hàng chục, trăm hoặc hàng nghìn lần. 
Chẳng hạn, thế hệ Volta mới nhất của NVIDIA có thể thực hiện tới 120 nghìn tỷ phép toán dấu phẩy động (TFlop) cho mỗi chip cho những lệnh chuyên biệt (và lên tới 24 TFlop cho các lệnh có mục đích chung), 
trong khi hiệu năng của CPU trong việc thực hiện tính toán với các số thực dấu phẩy động  không vượt quá 1 TFlop cho đến nay. 
Lý do khá đơn giản: thứ nhất, mức độ tiêu thụ năng lượng có xu hướng tăng theo hàm bậc hai so với tần số xung nhịp. 
Do đó, với cùng lượng năng lượng để một nhân CPU chạy nhanh gấp 4 lần tốc độ hiện tại (mức tăng thường gặp), chúng ta có thể thay bằng 16 nhân GPU với tốc độ mỗi nhân giảm còn 1/4, cũng sẽ cho kết quả là 16 x 1/4 = 4 lần tốc độ hiện tại.
Hơn nữa, các nhân của GPU thì đơn giản hơn nhiều (trên thực tế, trong một khoảng thời gian dài, những nhân này thậm chí *không thể* thực thi được mã lệnh dành cho những mục đích cơ bản), điều này giúp chúng tiết kiệm năng lượng hơn. 
Cuối cùng, nhiều phép tính trong quá trình học sâu đòi hỏi bộ nhớ băng thông cao. 
Và một lần nữa, GPUs vượt trội khi độ rộng đường bus của nó lớn hơn ít nhất 10 lần so với nhiều loại CPUs.

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
Back to 2012. A major breakthrough came when Alex Krizhevsky and Ilya Sutskever implemented a deep convolutional neural network that could run on GPU hardware.
They realized that the computational bottlenecks in CNNs (convolutions and matrix multiplications) are all operations that could be parallelized in hardware.
Using two NVIDIA GTX 580s with 3GB of memory, they implemented fast convolutions.
The code [cuda-convnet](https://code.google.com/archive/p/cuda-convnet/) was good enough that for several years it was the industry standard and powered the first couple years of the deep learning boom.
-->

*dịch đoạn phía trên*

<!--
## AlexNet
-->

## *dịch tiêu đề phía trên*

<!--
AlexNet was introduced in 2012, named after Alex Krizhevsky, the first author of the breakthrough ImageNet classification paper :cite:`Krizhevsky.Sutskever.Hinton.2012`.
AlexNet, which employed an 8-layer convolutional neural network, won the ImageNet Large Scale Visual Recognition Challenge 2012 by a phenomenally large margin.
This network proved, for the first time, that the features obtained by learning can transcend manually-design features, breaking the previous paradigm in computer vision.
The architectures of AlexNet and LeNet are *very similar*, as :numref:`fig_alexnet` illustrates.
Note that we provide a slightly streamlined version of AlexNet removing some of the design quirks that were needed in 2012 to make the model fit on two small GPUs.
-->

*dịch đoạn phía trên*

<!--
![LeNet (left) and AlexNet (right)](../img/alexnet.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/alexnet.svg)
:label:`fig_alexnet`

<!--
The design philosophies of AlexNet and LeNet are very similar, but there are also significant differences.
First, AlexNet is much deeper than the comparatively small LeNet5.
AlexNet consists of eight layers: five convolutional layers, two fully-connected hidden layers, and one fully-connected output layer.
Second, AlexNet used the ReLU instead of the sigmoid as its activation function.
Let us delve into the details below.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 4 - BẮT ĐẦU ===================================-->

<!--
### Architecture
-->

### *dịch tiêu đề phía trên*

<!--
In AlexNet's first layer, the convolution window shape is $11\times11$.
Since most images in ImageNet are more than ten times higher and wider than the MNIST images, objects in ImageNet data tend to occupy more pixels.
Consequently, a larger convolution window is needed to capture the object.
The convolution window shape in the second layer is reduced to $5\times5$, followed by $3\times3$.
In addition, after the first, second, and fifth convolutional layers, the network adds maximum pooling layers with a window shape of $3\times3$ and a stride of 2.
Moreover, AlexNet has ten times more convolution channels than LeNet.
-->

*dịch đoạn phía trên*

<!--
After the last convolutional layer are two fully-connected layers with 4096 outputs.
These two huge fully-connected layers produce model parameters of nearly 1 GB.
Due to the limited memory in early GPUs, the original AlexNet used a dual data stream design, so that each of their two GPUs could be responsible for storing and computing only its half of the model.
Fortunately, GPU memory is comparatively abundant now, so we rarely need to break up models across GPUs these days (our version of the AlexNet model deviates from the original paper in this aspect).
-->

*dịch đoạn phía trên*

<!--
### Activation Functions
-->

### *dịch tiêu đề phía trên*

<!--
Second, AlexNet changed the sigmoid activation function to a simpler ReLU activation function.
On the one hand, the computation of the ReLU activation function is simpler.
For example, it does not have the exponentiation operation found in the sigmoid activation function.
On the other hand, the ReLU activation function makes model training easier when using different parameter initialization methods.
This is because, when the output of the sigmoid activation function is very close to 0 or 1, the gradient of these regions is almost 0, 
so that back propagation cannot continue to update some of the model parameters.
In contrast, the gradient of the ReLU activation function in the positive interval is always 1.
Therefore, if the model parameters are not properly initialized, the sigmoid function may obtain a gradient of almost 0 in the positive interval, 
so that the model cannot be effectively trained.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 7 ===================== -->

<!--
### Capacity Control and Preprocessing
-->

### *dịch tiêu đề phía trên*

<!--
AlexNet controls the model complexity of the fully-connected layer by dropout (:numref:`sec_dropout`), while LeNet only uses weight decay.
To augment the data even further, the training loop of AlexNet added a great deal of image augmentation, such as flipping, clipping, and color changes.
This makes the model more robust and the larger sample size effectively reduces overfitting.
We will discuss data augmentation in greater detail in :numref:`sec_image_augmentation`.
-->

*dịch đoạn phía trên*

```{.python .input  n=1}
import d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
# Here, we use a larger 11 x 11 window to capture objects. At the same time,
# we use a stride of 4 to greatly reduce the height and width of the output.
# Here, the number of output channels is much larger than that in LeNet
net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Make the convolution window smaller, set padding to 2 for consistent
        # height and width across the input and output, and increase the
        # number of output channels
        nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Use three successive convolutional layers and a smaller convolution
        # window. Except for the final convolutional layer, the number of
        # output channels is further increased. Pooling layers are not used to
        # reduce the height and width of input after the first two
        # convolutional layers
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Here, the number of outputs of the fully connected layer is several
        # times larger than that in LeNet. Use the dropout layer to mitigate
        # overfitting
        nn.Dense(4096, activation="relu"), nn.Dropout(0.5),
        nn.Dense(4096, activation="relu"), nn.Dropout(0.5),
        # Output layer. Since we are using Fashion-MNIST, the number of
        # classes is 10, instead of 1000 as in the paper
        nn.Dense(10))
```

<!--
We construct a single-channel data instance with both height and width of 224 to observe the output shape of each layer.
It matches our diagram above.
-->

*dịch đoạn phía trên*

```{.python .input  n=2}
X = np.random.uniform(size=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

<!-- ========================================= REVISE PHẦN 4 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 5 - BẮT ĐẦU ===================================-->

<!--
## Reading the Dataset
-->

## *dịch tiêu đề phía trên*

<!--
Although AlexNet uses ImageNet in the paper, we use Fashion-MNIST here since training an ImageNet model to convergence could take hours or days even on a modern GPU.
One of the problems with applying AlexNet directly on Fashion-MNIST is that our images are lower resolution ($28 \times 28$ pixels) than ImageNet images.
To make things work, we upsample them to $244 \times 244$ (generally not a smart practice, but we do it here to be faithful to the AlexNet architecture).
We perform this resizing with the `resize` argument in `load_data_fashion_mnist`.
-->

*dịch đoạn phía trên*

```{.python .input  n=3}
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
```

<!-- ===================== Kết thúc dịch Phần 7 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 8 ===================== -->

<!--
## Training
-->

## *dịch tiêu đề phía trên*

<!--
Now, we can start training AlexNet.
Compared to LeNet in the previous section, the main change here is the use of a smaller learning rate and much slower training due to the deeper and wider network, 
the higher image resolution and the more costly convolutions.
-->

*dịch đoạn phía trên*

```{.python .input  n=5}
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

<!--
## Summary
-->

## Tóm tắt

<!--
* AlexNet has a similar structure to that of LeNet, but uses more convolutional layers and a larger parameter space to fit the large-scale dataset ImageNet.
* Today AlexNet has been surpassed by much more effective architectures but it is a key step from shallow to deep networks that are used nowadays.
* Although it seems that there are only a few more lines in AlexNet's implementation than in LeNet, it took the academic community many years to embrace this conceptual change and take advantage of its excellent experimental results. This was also due to the lack of efficient computational tools.
* Dropout, ReLU and preprocessing were the other key steps in achieving excellent performance in computer vision tasks.
-->

*dịch đoạn phía trên*

<!--
## Exercises
-->

## Bài tập

<!--
1. Try increasing the number of epochs. Compared with LeNet, how are the results different? Why?
2. AlexNet may be too complex for the Fashion-MNIST dataset.
    * Try to simplify the model to make the training faster, while ensuring that the accuracy does not drop significantly.
    * Can you design a better model that works directly on $28 \times 28$ images.
3. Modify the batch size, and observe the changes in accuracy and GPU memory.
4. Rooflines:
    * What is the dominant part for the memory footprint of AlexNet?
    * What is the dominant part for computation in AlexNet?
    * How about memory bandwidth when computing the results?
5. Apply dropout and ReLU to LeNet5. Does it improve? How about preprocessing?
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 8 ===================== -->
<!-- ========================================= REVISE PHẦN 5 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2354)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2354)
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
*

<!-- Phần 2 -->
*

<!-- Phần 3 -->
*

<!-- Phần 4 -->
* Dac Dinh 

<!-- Phần 5 -->
*

<!-- Phần 6 -->
*

<!-- Phần 7 -->
*

<!-- Phần 8 -->
*
