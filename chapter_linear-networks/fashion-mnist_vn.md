<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# The Image Classification Dataset (Fashion-MNIST)
-->

# Bộ dữ liệu Phân loại Ảnh (Fashion-MNIST)
:label:`sec_fashion_mnist`

<!--
In :numref:`sec_naive_bayes`, we trained a naive Bayes classifier, using the MNIST dataset introduced in 1998 :cite:`LeCun.Bottou.Bengio.ea.1998`.
While MNIST had a good run as a benchmark dataset, even simple models by today's standards achieve classification accuracy over 95%. 
Making it unsuitable for distinguishing between stronger models and weaker ones.
Today, MNIST serves as more of sanity checks than as a benchmark.
To up the ante just a bit, we will focus our discussion in the coming sections on the qualitatively similar, 
but comparatively complex Fashion-MNIST dataset :cite:`Xiao.Rasul.Vollgraf.2017`, which was released in 2017.
-->

Ở :numref:`sec_naive_bayes`, chúng ta đã huấn luyện bộ phân loại Naive Bayes, sử dụng bộ dữ liệu MNIST được giới thiệu vào năm 1998 :cite:`LeCun.Bottou.Bengio.ea.1998`.
Mặc dù MNIST từng là một bộ dữ liệu tốt để đánh giá xếp hạng (_benchmark_), các mô hình đơn giản theo tiêu chuẩn ngày nay cũng có thể đạt được độ chính xác phân loại lên tới 95%.
Điều này khiến nó không phù hợp cho việc phân biệt độ mạnh yếu của các mô hình.
Ngày nay, MNIST được dùng trong các phép kiểm tra sơ bộ hơn là dùng để đánh giá xếp hạng.
Để cải thiện vấn đề này, chúng ta sẽ tập trung thảo luận trong các mục tiếp theo về một bộ dữ liệu tương tự nhưng phức tạp hơn, đó là bộ dữ liệu Fashion-MNIST :cite:`Xiao.Rasul.Vollgraf.2017` được giới thiệu vào năm 2017.

```{.python .input  n=7}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon
import sys

d2l.use_svg_display()
```

<!--
## Getting the Dataset
-->

## Tải về Bộ dữ liệu

<!--
Just as with MNIST, Gluon makes it easy to download and load the FashionMNIST dataset into memory via the `FashionMNIST` class contained in `gluon.data.vision`.
We briefly work through the mechanics of loading and exploring the dataset below.
Please refer to :numref:`sec_naive_bayes` for more details on loading data.
-->

Cũng giống như với MNIST, Gluon giúp việc tải và nạp bộ dữ liệu FashionMNIST vào bộ nhớ trở nên dễ dàng với lớp `FashionMNIST` trong `gluon.data.vision`.
Các cơ chế của việc nạp và khám phá bộ dữ liệu sẽ được hướng dẫn ngắn gọn bên dưới.
Vui lòng tham khảo :numref:`sec_naive_bayes` để biết thêm chi tiết về việc nạp dữ liệu.

```{.python .input  n=23}
mnist_train = gluon.data.vision.FashionMNIST(train=True)
mnist_test = gluon.data.vision.FashionMNIST(train=False)
```

<!--
FashionMNIST consists of images from 10 categories, each represented by 6k images in the training set and by 1k in the test set.
Consequently the training set and the test set contain 60k and 10k images, respectively.
-->

FashionMNIST chứa các hình ảnh thuộc 10 lớp, mỗi lớp có 6000 ảnh trong tập huấn luyện và 1000 ảnh trong tập kiểm tra.
Do đó, tập huấn luyện và tập kiểm tra sẽ chứa tổng cộng lần lượt 60000 và 10000 ảnh.

```{.python .input}
len(mnist_train), len(mnist_test)
```

<!--
The images in Fashion-MNIST are associated with the following categories: t-shirt, trousers, pullover, dress, coat, sandal, shirt, sneaker, bag and ankle boot.
The following function converts between numeric label indices and their names in text.
-->

Các ảnh trong Fashion-MNIST tương ứng với các lớp: áo phông, quần dài, áo thun, váy, áo khoác, dép, áo sơ-mi, giày thể thao, túi và giày cao gót.
Hàm dưới đây giúp chuyển đổi các nhãn giá trị số thành tên của từng lớp.

```{.python .input  n=25}
# Saved in the d2l package for later use
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
```

<!--
We can now create a function to visualize these examples.
-->

Chúng ta có thể tạo một hàm để minh hoạ các mẫu này.

```{.python .input}
# Saved in the d2l package for later use
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.asnumpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

<!--
Here are the images and their corresponding labels (in text) for the first few examples in the training dataset.
-->

Dưới đây là các hình ảnh và nhãn tương ứng của chúng (ở dạng chữ) từ một vài mẫu đầu tiên trong tập huấn luyện.

```{.python .input}
X, y = mnist_train[:18]
show_images(X.squeeze(axis=-1), 2, 9, titles=get_fashion_mnist_labels(y));
```

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Reading a Minibatch
-->

## Đọc một Minibatch

<!--
To make our life easier when reading from the training and test sets, we use a `DataLoader` rather than creating one from scratch, as we did in :numref:`sec_linear_scratch`.
Recall that at each iteration, a `DataLoader` reads a minibatch of data with size `batch_size` each time.
-->

Để đọc dữ liệu từ tập huấn luyện và tập kiểm tra một cách dễ dàng hơn, chúng ta sử dụng một `DataLoader` có sẵn thay vì tạo từ đầu như đã làm ở :numref:`sec_linear_scratch`.
Nhắc lại là ở mỗi vòng lặp, một `DataLoader` sẽ đọc một minibatch của tập dữ liệu với kích thước `batch_size`.

<!--
During training, reading data can be a significant performance bottleneck, especially when our model is simple or when our computer is fast.
A handy feature of Gluon's `DataLoader` is the ability to use multiple processes to speed up data reading.
For instance, we can set aside 4 processes to read the data (via `num_workers`).
Because this feature is not currently supported on Windows the following code checks the platform to make sure that we do not saddle our Windows-using friends with error messages later on.
-->

Trong quá trình huấn luyện, việc đọc dữ liệu có thể gây ra hiện tượng nghẽn cổ chai hiệu năng đáng kể, trừ khi mô hình đơn giản hoặc máy tính rất nhanh.
Một tính năng tiện dụng của `DataLoader` là khả năng sử dụng đa tiến trình (_multiple processes_) để tăng tốc việc đọc dữ liệu.
Ví dụ, chúng ta có thể dùng 4 tiến trình để đọc dữ liệu (thông qua `num_workers`).
Vì tính năng này hiện tại không được hỗ trợ trên Windows, đoạn mã lập trình dưới đây sẽ kiểm tra nền tảng hệ điều hành để đảm bảo rằng chúng ta không làm phiền những người dùng Windows với các thông báo lỗi sau này.

```{.python .input}
# Saved in the d2l package for later use
def get_dataloader_workers(num_workers=4):
    # 0 means no additional process is used to speed up the reading of data.
    if sys.platform.startswith('win'):
        return 0
    else:
        return num_workers
```

<!--
Below, we convert the image data from uint8 to 32-bit floating point numbers using the `ToTensor` class.
Additionally, the transformer will divide all numbers by 255 so that all pixels have values between 0 and 1.
The `ToTensor` class also moves the image channel from the last dimension to the first dimension to facilitate the convolutional neural network calculations introduced later.
Through the `transform_first` function of the dataset, we apply the transformation of `ToTensor` to the first element of each instance (image and label).
-->

Dưới đây, chúng ta chuyển đổi dữ liệu hình ảnh từ uint8 sang số thực dấu phẩy động (_floating point number_) 32 bit với lớp `ToTensor`.
Ngoài ra, bộ chuyển đổi sẽ chia tất cả các số cho 255 để các điểm ảnh có giá trị từ 0 đến 1.
Lớp `ToTensor` cũng chuyển kênh hình ảnh từ chiều cuối cùng sang chiều thứ nhất để tạo điều kiện cho các tính toán của mạng nơ-ron tích chập được giới thiệu sau này.
Thông qua hàm `transform_first` của tập dữ liệu, chúng ta có thể áp dụng phép biến đổi `ToTensor` cho phần tử đầu tiên của mỗi ví dụ (một ví dụ chứa hai phần tử là ảnh và nhãn).

```{.python .input  n=28}
batch_size = 256
transformer = gluon.data.vision.transforms.ToTensor()
train_iter = gluon.data.DataLoader(mnist_train.transform_first(transformer),
                                   batch_size, shuffle=True,
                                   num_workers=get_dataloader_workers())
```

<!--
Let's look at the time it takes to read the training data.
-->

Hãy cùng xem thời gian cần thiết để hoàn tất việc đọc dữ liệu huấn luyện.

```{.python .input}
timer = d2l.Timer()
for X, y in train_iter:
    continue
'%.2f sec' % timer.stop()
```

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Putting All Things Together
-->

## Kết hợp Tất cả lại với nhau

<!--
Now we define the `load_data_fashion_mnist` function that obtains and reads the Fashion-MNIST dataset.
It returns the data iterators for both the training set and validation set.
In addition, it accepts an optional argument to resize images to another shape.
-->

Bây giờ, chúng ta sẽ định nghĩa hàm `load_data_fashion_mnist` để nạp và đọc bộ dữ liệu Fashion-MNIST.
Hàm này sẽ trả về các iterator cho dữ liệu của cả tập huấn luyện và tập kiểm định.
Thêm nữa, nó chấp nhận một tham số tùy chọn để thay đổi kích thước hình ảnh đầu vào.

```{.python .input  n=4}
# Saved in the d2l package for later use
def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load into memory."""
    dataset = gluon.data.vision
    trans = [dataset.transforms.Resize(resize)] if resize else []
    trans.append(dataset.transforms.ToTensor())
    trans = dataset.transforms.Compose(trans)
    mnist_train = dataset.FashionMNIST(train=True).transform_first(trans)
    mnist_test = dataset.FashionMNIST(train=False).transform_first(trans)
    return (gluon.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                  num_workers=get_dataloader_workers()),
            gluon.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                  num_workers=get_dataloader_workers()))
```

<!--
Below, we verify that image resizing works.
-->

Dưới đây, chúng ta xác nhận rằng kích thước hình ảnh đã được thay đổi.

```{.python .input  n=5}
train_iter, test_iter = load_data_fashion_mnist(32, (64, 64))
for X, y in train_iter:
    print(X.shape)
    break
```

<!--
We are now ready to work with the FashionMNIST dataset in the sections that follow.
-->

Giờ chúng ta đã sẵn sàng để làm việc với bộ dữ liệu FashionMNIST trong các mục tiếp theo.

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
## Summary
-->

## Tóm tắt

<!--
* Fashion-MNIST is an apparel classification dataset consisting of images representing 10 categories.
* We will use this dataset in subsequent sections and chapters to evaluate various classification algorithms.
* We store the shape of each image with height $h$ width $w$ pixels as $h \times w$ or `(h, w)`.
* Data iterators are a key component for efficient performance. Rely on well-implemented iterators that exploit multi-threading to avoid slowing down your training loop.
-->

* Fashion-MNIST là một tập dữ liệu phân loại trang phục bao gồm các hình ảnh đại diện cho 10 lớp.
* Chúng ta sẽ sử dụng tập dữ liệu này trong các mục và chương tiếp theo để đánh giá các thuật toán phân loại khác nhau.
* Chúng ta lưu trữ kích thước của mỗi hình ảnh với chiều cao $h$ chiều rộng $w$ điểm ảnh dưới dạng $h \times w$ hoặc `(h, w)`.
* Iterator cho dữ liệu là nhân tố chính để đạt được hiệu suất cao. Hãy sử dụng các iterator được lập trình tốt để tận dụng khả năng chạy đa tiến trình, tránh làm chậm vòng lặp huấn luyện.

<!--
## Exercises
-->

## Bài tập

<!--
1. Does reducing the `batch_size` (for instance, to 1) affect read performance?
2. For non-Windows users, try modifying `num_workers` to see how it affects read performance. Plot the performance against the number of works employed.
3. Use the MXNet documentation to see which other datasets are available in `mxnet.gluon.data.vision`.
4. Use the MXNet documentation to see which other transformations are available in `mxnet.gluon.data.vision.transforms`.
-->

1. Việc giảm `batch_size` (ví dụ xuống 1) có ảnh hưởng tới tốc độ đọc dữ liệu hay không?
2. Với người dùng không sử dụng Windows, hãy thử thay đổi `num_workers` để xem nó ảnh hưởng đến hiệu năng đọc dữ liệu như thế nào. Vẽ đồ thị hiệu năng tương ứng với số tiến trình được sử dụng.
3. Sử dụng tài liệu MXNet để xem các bộ dữ liệu có sẵn khác trong `mxnet.gluon.data.vision`.
4. Sử dụng tài liệu MXNet để xem những phép biến đổi nào có sẵn trong `mxnet.gluon.data.vision.transforms`.

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2335)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2335)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.

* Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md.
-->

* Đoàn Võ Duy Thanh
* Nguyễn Lê Quang Nhật
* Vũ Hữu Tiệp
* Lê Khắc Hồng Phúc
* Phạm Hồng Vinh
* Phạm Minh Đức
