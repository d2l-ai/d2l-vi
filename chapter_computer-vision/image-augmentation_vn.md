<!--
# Image Augmentation
-->

# Tăng cường Ảnh
:label:`sec_image_augmentation`



<!--
We mentioned that large-scale datasets are prerequisites for the successful application of deep neural networks in :numref:`sec_alexnet`.
Image augmentation technology expands the scale of training datasets by making a series of random changes to the training images to produce similar, but different, training examples.
Another way to explain image augmentation is that randomly changing training examples can reduce a model's dependence on certain properties, thereby improving its capability for generalization.
For example, we can crop the images in different ways, so that the objects of interest appear in different positions, reducing the model's dependence on the position where objects appear.
We can also adjust the brightness, color, and other factors to reduce model's sensitivity to color.
It can be said that image augmentation technology contributed greatly to the success of AlexNet.
In this section, we will discuss this technology, which is widely used in computer vision.
-->

Trong :numref:`sec_alexnet` chúng ta có đề cập đến việc các bộ dữ liệu lớn là điều kiện tiên quyết cho sự thành công của các mạng nơ-ron sâu.
Kỹ thuật tăng cường ảnh giúp mở rộng kích thước của tập dữ liệu huấn luyện thông qua việc áp dụng một loạt thay đổi ngẫu nhiên trên các mẫu ảnh,
từ đó tạo ra các mẫu huấn luyện tuy tương tự nhưng vẫn có sự khác biệt.
Cũng có thể giải thích tác dụng của tăng cường ảnh là việc thay đổi ngẫu nhiên các mẫu dùng cho huấn luyện, làm giảm sự phụ thuộc của mô hình vào một số thuộc tính nhất định. Do đó giúp cải thiện năng lực khái quát hóa của mô hình.

Chẳng hạn, ta có thể cắt tập ảnh theo các cách khác nhau, để các đối tượng ta quan tâm xuất hiện ở các vị trí khác nhau, vì vậy giảm sự phụ thuộc của mô hình vào vị trí xuất hiện của đối tượng.
Ta cũng có thể điều chỉnh độ sáng, màu sắc, và các yếu tố khác để giảm độ nhạy màu sắc của mô hình. 
Có thể khẳng định rằng kỹ thuật tăng cường ảnh đóng góp rất lớn cho sự thành công của mạng AlexNet.
Tới đây, chúng ta sẽ thảo luận về kỹ thuật mà được sử dụng rộng rãi trong lĩnh vực thị giác máy tính này. 

<!--
First, import the packages or modules required for the experiment in this section.
-->

Trước tiên, thực hiện nhập các gói và mô-đun cần thiết.


```{.python .input  n=1}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```


<!--
## Common Image Augmentation Method
-->

## Phương pháp Tăng cường Ảnh Thông dụng


<!--
In this experiment, we will use an image with a shape of $400\times 500$ as an example.
-->

Trong phần thử nghiệm này, ta sẽ dùng một ảnh có kích thước $400\times 500$ làm ví dụ.


```{.python .input  n=2}
d2l.set_figsize()
img = image.imread('../img/cat1.jpg')
d2l.plt.imshow(img.asnumpy());
```


<!--
Most image augmentation methods have a certain degree of randomness.
To make it easier for us to observe the effect of image augmentation, we next define the auxiliary function `apply`.
This function runs the image augmentation method `aug` multiple times on the input image `img` and shows all results.
-->

Hầu hết các phương pháp tăng cường ảnh có một độ ngẫu nhiên nhất định.
Để giúp việc quan sát tính hiệu quả của nó dễ hơn, ta sẽ định nghĩa hàm bổ trợ `apply`.
Hàm này thực hiện phương thức tăng cường ảnh `aug` nhiều lần từ ảnh đầu vào `img` và hiển thị tất cả kết quả.


```{.python .input  n=3}
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
```


<!--
### Flipping and Cropping
-->

### Lật và Cắt ảnh


<!--
Flipping the image left and right usually does not change the category of the object.
This is one of the earliest and most widely used methods of image augmentation.
Next, we use the `transforms` module to create the `RandomFlipLeftRight` instance, which introduces a 50% chance that the image is flipped left and right.
-->

Lật hình ảnh sang trái và phải thường không thay đổi thể loại đối tượng.
Đây là một trong những phương pháp tăng cường ảnh được sử dụng sớm nhất và rộng rãi nhất.
Tiếp theo, chúng ta sử dụng mô-đun `transforms` để tạo thực thể `RandomFlipLeftRight`, ngẫu nhiên lật hình ảnh sang trái hoặc phải với xác suất 50%.



```{.python .input  n=4}
apply(img, gluon.data.vision.transforms.RandomFlipLeftRight())
```


<!--
Flipping up and down is not as commonly used as flipping left and right.
However, at least for this example image, flipping up and down does not hinder recognition.
Next, we create a `RandomFlipTopBottom` instance for a 50% chance of flipping the image up and down.
-->

Lật lên và xuống không được sử dụng phổ biến như lật trái và phải.
Tuy nhiên, ít nhất là đối với hình ảnh ví dụ này, lật lên xuống không gây trở ngại cho việc nhận dạng.
Tiếp theo, chúng tôi tạo thực thể `RandomFlipTopBottom` để lật hình ảnh lên và xuống với xác suất 50%.


```{.python .input  n=5}
apply(img, gluon.data.vision.transforms.RandomFlipTopBottom())
```


<!--
In the example image we used, the cat is in the middle of the image, but this may not be the case for all images.
In :numref:`sec_pooling`, we explained that the pooling layer can reduce the sensitivity of the convolutional layer to the target location.
In addition, we can make objects appear at different positions in the image in different proportions by randomly cropping the image.
This can also reduce the sensitivity of the model to the target position.
-->

Trong ví dụ chúng ta sử dụng, con mèo nằm ở giữa ảnh, nhưng không phải tất cả các ảnh mèo khác đều sẽ như vậy.
:numref:`sec_pooling` có đề cập rằng tầng gộp có thể làm giảm độ nhạy của tầng tích chập với vị trí mục tiêu.
Ngoài ra, chúng ta có thể làm cho các đối tượng xuất hiện ở các vị trí khác nhau trong ảnh theo tỷ lệ khác nhau bằng cách cắt ngẫu nhiên hình ảnh.
Điều này cũng có thể làm giảm độ nhạy của mô hình với vị trí mục tiêu.


<!--
In the following code, we randomly crop a region with an area of 10% to 100% of the original area, and the ratio of width to height of the region is randomly selected from between 0.5 and 2.
Then, the width and height of the region are both scaled to 200 pixels.
Unless otherwise stated, the random number between $a$ and $b$ in this section refers to a continuous value obtained by uniform sampling in the interval $[a, b]$.
-->

Trong đoạn mã sau, chúng tôi cắt ngẫu nhiên một vùng có diện tích từ 10% đến 100% diện tích ban đầu và tỷ lệ giữa chiều rộng và chiều cao của vùng được chọn ngẫu nhiên trong khoảng từ 0.5 đến 2.
Sau đó, cả chiều rộng và chiều cao của vùng đều được biến đổi tỷ lệ thành 200 pixel.
Trừ khi có quy định khác, giá trị ngẫu nhiên liên tục giữa $a$ và $b$ thu được bằng cách lấy mẫu đồng nhất trong khoảng $[a, b]$.



```{.python .input  n=6}
shape_aug = gluon.data.vision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```


<!--
### Changing the Color
-->

### Đổi màu


<!--
Another augmentation method is changing colors.
We can change four aspects of the image color: brightness, contrast, saturation, and hue.
In the example below, we randomly change the brightness of the image to a value between 50% ($1-0.5$) and 150% ($1+0.5$) of the original image.
-->

Một phương pháp tăng cường khác là thay đổi màu sắc.
Chúng ta có thể thay đổi bốn khía cạnh màu sắc của hình ảnh: độ sáng, độ tương phản, độ bão hòa và tông màu.
Trong ví dụ dưới đây, chúng tôi thay đổi ngẫu nhiên độ sáng của hình ảnh với giá trị trong khoảng từ 50% ($1-0.5$) đến 150% ($1+0.5$) độ sáng của ảnh gốc.



```{.python .input  n=7}
apply(img, gluon.data.vision.transforms.RandomBrightness(0.5))
```


<!--
Similarly, we can randomly change the hue of the image.
-->

Tương tự vậy, ta có thể ngẫu nhiên thay đổi tông màu của ảnh.


```{.python .input  n=8}
apply(img, gluon.data.vision.transforms.RandomHue(0.5))
```


<!--
We can also create a `RandomColorJitter` instance and set how to randomly change the `brightness`, `contrast`, `saturation`, and `hue` of the image at the same time.
-->

Ta cũng có thể tạo một thực thể `RandomColorJitter` và thiết lập để ngẫu nhiên thay đổi `brightness` (độ sáng), `contrast` (độ tương phản), `saturation` (độ bão hòa), và `hue` (tông màu) của ảnh cùng một lúc. 
 


```{.python .input  n=9}
color_aug = gluon.data.vision.transforms.RandomColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```


<!--
### Overlying Multiple Image Augmentation Methods
-->

### Kết hợp nhiều Phương pháp Tăng cường Ảnh


<!--
In practice, we will overlay multiple image augmentation methods.
We can overlay the different image augmentation methods defined above and apply them to each image by using a `Compose` instance.
-->

Trong thực tế, chúng ta sẽ kết hợp nhiều phương pháp tăng cường ảnh.
Ta có thể kết hợp các phương pháp trên và áp dụng chúng cho từng hình ảnh bằng cách sử dụng thực thể `Compose`.



```{.python .input  n=10}
augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(), color_aug, shape_aug])
apply(img, augs)
```


<!--
## Using an Image Augmentation Training Model
-->

## Huấn luyện Mô hình dùng Tăng cường Ảnh


<!--
Next, we will look at how to apply image augmentation in actual training.
Here, we use the CIFAR-10 dataset, instead of the Fashion-MNIST dataset we have been using.
This is because the position and size of the objects in the Fashion-MNIST dataset have been normalized, and the differences in color and size of the objects in CIFAR-10 dataset are more significant.
The first 32 training images in the CIFAR-10 dataset are shown below.
-->

Tiếp theo, ta sẽ xem xét làm thế nào để áp dụng tăng cường hình ảnh trong huấn luyện thực tế.
Ở đây, ta sử dụng bộ dữ liệu CIFAR-10, thay vì Fashion-MNIST trước đây.
Điều này là do vị trí và kích thước của các đối tượng trong bộ dữ liệu Fashion-MNIST đã được chuẩn hóa và sự khác biệt về màu sắc và kích thước của các đối tượng trong bộ dữ liệu CIFAR-10 là đáng kể hơn.
32 hình ảnh huấn luyện đầu tiên trong bộ dữ liệu CIFAR-10 được hiển thị bên dưới.



```{.python .input  n=11}
d2l.show_images(gluon.data.vision.CIFAR10(
    train=True)[0:32][0], 4, 8, scale=0.8);
```


<!--
In order to obtain definitive results during prediction, we usually only apply image augmentation to the training example, and do not use image augmentation with random operations during prediction.
Here, we only use the simplest random left-right flipping method.
In addition, we use a `ToTensor` instance to convert minibatch images into the format required by MXNet, 
i.e., 32-bit floating point numbers with the shape of (batch size, number of channels, height, width) and value range between 0 and 1.
-->

Để có được kết quả cuối cùng trong dự đoán, ta thường chỉ áp dụng tăng cường ảnh khi huấn luyện nhưng không sử dụng các biến đổi ngẫu nhiên trong dự đoán.
Ở đây, chúng ta chỉ sử dụng phương pháp lật ngẫu nhiên trái phải đơn giản nhất.
Ngoài ra, chúng ta sử dụng một thực thể `ToTensor` để chuyển đổi minibatch hình ảnh thành định dạng theo yêu cầu của MXNet,
tức là, tensor số thực dấu phẩy động 32-bit có kích thước (kích thước batch, số kênh, chiều cao, chiều rộng) và phạm vi giá trị trong khoảng từ 0 đến 1.



```{.python .input  n=12}
train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor()])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor()])
```


<!--
Next, we define an auxiliary function to make it easier to read the image and apply image augmentation.
The `transform_first` function provided by Gluon's dataset applies image augmentation to the first element of each training example (image and label), i.e., the element at the top of the image.
For detailed descriptions of `DataLoader`, refer to :numref:`sec_fashion_mnist`.
-->

Tiếp theo, ta định nghĩa một chức năng phụ trợ để giúp đọc hình ảnh và áp dụng tăng cường ảnh dễ dàng hơn.
Hàm `transform_first` được cung cấp bởi Gluon giúp thực thi tăng cường ảnh cho phần tử đầu tiên của mỗi mẫu huấn luyện (hình ảnh và nhãn), tức là chỉ áp dụng lên phần ảnh.
Để biết thêm chi tiết về `DataLoader`, hãy tham khảo :numref:`sec_fashion_mnist`.



```{.python .input  n=13}
def load_cifar10(is_train, augs, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=is_train).transform_first(augs),
        batch_size=batch_size, shuffle=is_train,
        num_workers=d2l.get_dataloader_workers())
```


<!--
### Using a Multi-GPU Training Model
-->

### Sử dụng Mô hình Huấn luyện Đa GPU


<!--
We train the ResNet-18 model described in :numref:`sec_resnet` on the
CIFAR-10 dataset. We will also apply the methods described in
:numref:`sec_multi_gpu_concise` and use a multi-GPU training model.
-->

Ta huấn luyện mô hình ResNet-18 như mô tả ở :numref:`sec_resnet` trên tập dữ liệu CIFAR-10.
Cùng với đó ta áp dụng các phương pháp được mô tả trong :numref:`sec_multi_gpu_concise` và sử dụng mô hình huấn luyện đa GPU.


<!--
Next, we define the training function to train and evaluate the model using multiple GPUs.
-->

Tiếp theo, ta định nghĩa hàm huấn luyện để huấn luyện và đánh giá mô hình sử dụng nhiều GPU.


```{.python .input  n=14}
#@save
def train_batch_ch13(net, features, labels, loss, trainer, devices,
                     split_f=d2l.split_batch):
    X_shards, y_shards = split_f(features, labels, devices)
    with autograd.record():
        pred_shards = [net(X_shard) for X_shard in X_shards]
        ls = [loss(pred_shard, y_shard) for pred_shard, y_shard
              in zip(pred_shards, y_shards)]
    for l in ls:
        l.backward()
    # The True flag allows parameters with stale gradients, which is useful
    # later (e.g., in fine-tuning BERT)
    trainer.step(labels.shape[0], ignore_stale_grad=True)
    train_loss_sum = sum([float(l.sum()) for l in ls])
    train_acc_sum = sum(d2l.accuracy(pred_shard, y_shard)
                        for pred_shard, y_shard in zip(pred_shards, y_shards))
    return train_loss_sum, train_acc_sum
```

```{.python .input  n=16}
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus(), split_f=d2l.split_batch):
    num_batches, timer = len(train_iter), d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        # Store training_loss, training_accuracy, num_examples, num_features
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices, split_f)
            metric.add(l, acc, labels.shape[0], labels.size)
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0:
                animator.add(epoch + i / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpus(net, test_iter, split_f)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```




<!--
Now, we can define the `train_with_data_aug` function to use image augmentation to train the model.
This function obtains all available GPUs and uses Adam as the optimization algorithm for training.
It then applies image augmentation to the training dataset, and finally calls the `train_ch13` function just defined to train and evaluate the model.
-->

Giờ ta có thể định nghĩa hàm `train_with_data_aug` để áp dụng tăng cường ảnh vào huấn luyện mô hình.
Hàm này tìm tất cả các GPU có sẵn và sử dụng Adam làm thuật toán tối ưu cho quá trình huấn luyện.
Sau đó nó áp dụng tăng cường ảnh vào tập huấn luyện, và cuối cùng gọi đến hàm `train_ch13` được định nghĩa ở trên để huấn luyện và đánh giá mô hình.


```{.python .input  n=18}
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10)
net.initialize(init=init.Xavier(), ctx=devices)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```


<!-- Now we train the model using image augmentation of random flipping left and right. -->

Giờ ta huấn luyện mô hình áp dụng tăng cường ảnh qua phép lật ngẫu nhiên trái và phải. 


```{.python .input  n=19}
train_with_data_aug(train_augs, test_augs, net)
```


## Tóm tắt

<!--
* Image augmentation generates random images based on existing training data to cope with overfitting.
* In order to obtain definitive results during prediction, we usually only apply image augmentation to the training example, and do not use image augmentation with random operations during prediction.
* We can obtain classes related to image augmentation from Gluon's `transforms` module.
-->

* Tăng cường ảnh sản sinh ra những ảnh ngẫu nhiên dựa vào dữ liệu có sẵn trong tập huấn luyện để đối phó với hiện tượng quá khớp.
* Để có thể thu được kết quả tin cậy trong quá trình dự đoán, thường thì ta chỉ áp dụng tăng cường ảnh lên mẫu huấn luyện, không áp dụng các biến đổi tăng cường ảnh ngẫu nhiên trong quá trình dự đoán.
* Mô-đun `transforms` của Gluon có các lớp thực hiện tăng cường ảnh.


## Bài tập

<!--
1. Train the model without using image augmentation: `train_with_data_aug(no_aug, no_aug)`.
Compare training and testing accuracy when using and not using image augmentation.
Can this comparative experiment support the argument that image augmentation can mitigate overfitting? Why?
2. Add different image augmentation methods in model training based on the CIFAR-10 dataset. Observe the implementation results.
3. With reference to the MXNet documentation, what other image augmentation methods are provided in Gluon's `transforms` module?
-->

1. Huấn luyện mô hình mà không áp dụng tăng cường ảnh: `train_with_data_aug(no_aug, no_aug)`.
So sánh độ chính xác trong huấn luyện và kiểm tra khi áp dụng và không áp dụng tăng cường ảnh.
Liệu thí nghiệm so sánh này có thể hỗ trợ cho luận điểm rằng tăng cường ảnh có thể làm giảm hiện tượng quá khớp? Tại sao?
2. Sử dụng thêm các phương thức tăng cường ảnh khác trên tập dữ liệu CIFAR-10 khi huấn luyện mô hình. Theo dõi kết quả.
3. Tham khảo tài liệu của MXNet và cho biết mô-đun `transforms` của Gluon còn cung cấp các phương thức tăng cường ảnh nào khác?


## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/367)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Mai Hoàng Long
* Trần Yến Thy
* Lê Khắc Hồng Phúc
* Nguyễn Văn Cường
* Phạm Hồng Vinh
* Đỗ Trường Giang
* Nguyễn Lê Quang Nhật
