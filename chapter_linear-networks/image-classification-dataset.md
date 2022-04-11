# Tập dữ liệu phân loại hình ảnh
:label:`sec_fashion_mnist`

(~~Tập dữ liệu MNIST là một trong những tập dữ liệu được sử dụng rộng rãi để phân loại hình ảnh, trong khi nó quá đơn giản như một tập dữ liệu chuẩn. Chúng tôi sẽ sử dụng bộ dữ liệu Fashion-MNIST tương tự, nhưng phức tạp hơn ~ ~) 

Một trong những tập dữ liệu được sử dụng rộng rãi để phân loại hình ảnh là tập dữ liệu MNIST :cite:`LeCun.Bottou.Bengio.ea.1998`. Mặc dù nó có một chạy tốt như một tập dữ liệu chuẩn, thậm chí các mô hình đơn giản theo tiêu chuẩn ngày nay cũng đạt được độ chính xác phân loại trên 95%, khiến nó không phù hợp để phân biệt giữa các mô hình mạnh hơn và các mô hình yếu hơn. Ngày nay, MNIST phục vụ như là kiểm tra sự tỉnh táo hơn là một chuẩn mực. Để lên ante chỉ một chút, chúng tôi sẽ tập trung thảo luận của chúng tôi trong các phần sắp tới về chất lượng tương tự, nhưng tương đối phức tạp Fashion-MNIST dataset :cite:`Xiao.Rasul.Vollgraf.2017`, được phát hành vào năm 2017.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon
import sys

d2l.use_svg_display()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torchvision import transforms
from torch.utils import data

d2l.use_svg_display()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

d2l.use_svg_display()
```

## Đọc tập dữ liệu

Chúng ta có thể [** tải xuống và đọc tập dữ liệu Fashion-MNIST vào bộ nhớ thông qua các chức năng tích hợp trong khuôn việt.**]

```{.python .input}
mnist_train = gluon.data.vision.FashionMNIST(train=True)
mnist_test = gluon.data.vision.FashionMNIST(train=False)
```

```{.python .input}
#@tab pytorch
# `ToTensor` converts the image data from PIL type to 32-bit floating point
# tensors. It divides all numbers by 255 so that all pixel values are between
# 0 and 1
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
```

```{.python .input}
#@tab tensorflow
mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
```

Fashion-MNIST bao gồm các hình ảnh từ 10 loại, mỗi loại được đại diện bởi 6000 hình ảnh trong tập dữ liệu đào tạo và 1000 trong tập dữ liệu thử nghiệm. Một *test dataset* (hoặc * test set*) được sử dụng để đánh giá hiệu suất mô hình và không cho đào tạo. Do đó, bộ đào tạo và bộ thử nghiệm chứa 60000 và 10000 hình ảnh, tương ứng.

```{.python .input}
#@tab mxnet, pytorch
len(mnist_train), len(mnist_test)
```

```{.python .input}
#@tab tensorflow
len(mnist_train[0]), len(mnist_test[0])
```

Chiều cao và chiều rộng của mỗi hình ảnh đầu vào đều là 28 pixel. Lưu ý rằng tập dữ liệu bao gồm các hình ảnh thang màu xám, có số kênh là 1. Đối với ngắn gọn, trong suốt cuốn sách này, chúng tôi lưu trữ hình dạng của bất kỳ hình ảnh nào có chiều cao $h$ chiều rộng $w$ pixel là $h \times w$ hoặc ($h$, $w$).

```{.python .input}
#@tab all
mnist_train[0][0].shape
```

[~~Hai hàm tiện ích để hình dung tập dữ liệu ~~] 

Các hình ảnh trong Fashion-MNIST được liên kết với các loại sau: áo phông, quần tây, áo thun, váy, áo khoác, sandal, áo sơ mi, giày thể thao, túi xách và khởi động mắt cá chân. Hàm sau chuyển đổi giữa các chỉ số nhãn số và tên của chúng trong văn bản.

```{.python .input}
#@tab all
def get_fashion_mnist_labels(labels):  #@save
    """Return text labels for the Fashion-MNIST dataset."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
```

Bây giờ chúng ta có thể tạo ra một hàm để hình dung các ví dụ này.

```{.python .input}
#@tab mxnet, tensorflow
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(d2l.numpy(img))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

```{.python .input}
#@tab pytorch
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            ax.imshow(img.numpy())
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

Dưới đây là [** hình ảnh và nhãn tương ứng của chúng**](trong văn bản) cho một vài ví dụ đầu tiên trong tập dữ liệu đào tạo.

```{.python .input}
X, y = mnist_train[:18]

print(X.shape)
show_images(X.squeeze(axis=-1), 2, 9, titles=get_fashion_mnist_labels(y));
```

```{.python .input}
#@tab pytorch
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
```

```{.python .input}
#@tab tensorflow
X = tf.constant(mnist_train[0][:18])
y = tf.constant(mnist_train[1][:18])
show_images(X, 2, 9, titles=get_fashion_mnist_labels(y));
```

## Đọc một Minibatch

Để làm cho cuộc sống của chúng ta dễ dàng hơn khi đọc từ các bộ đào tạo và kiểm tra, chúng tôi sử dụng bộ lặp dữ liệu tích hợp hơn là tạo một từ đầu. Nhớ lại rằng tại mỗi lần lặp lại, một bộ lặp dữ liệu [** đọc một minibatch dữ liệu với kích thước `batch_size` mỗi lần.**] Chúng tôi cũng ngẫu nhiên xáo trộn các ví dụ cho bộ lặp dữ liệu đào tạo.

```{.python .input}
batch_size = 256

def get_dataloader_workers():  #@save
    """Use 4 processes to read the data except for Windows."""
    return 0 if sys.platform.startswith('win') else 4

# `ToTensor` converts the image data from uint8 to 32-bit floating point. It
# divides all numbers by 255 so that all pixel values are between 0 and 1
transformer = gluon.data.vision.transforms.ToTensor()
train_iter = gluon.data.DataLoader(mnist_train.transform_first(transformer),
                                   batch_size, shuffle=True,
                                   num_workers=get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
batch_size = 256

def get_dataloader_workers():  #@save
    """Use 4 processes to read the data."""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
```

```{.python .input}
#@tab tensorflow
batch_size = 256
train_iter = tf.data.Dataset.from_tensor_slices(
    mnist_train).batch(batch_size).shuffle(len(mnist_train[0]))
```

Chúng ta hãy nhìn vào thời gian cần thiết để đọc dữ liệu đào tạo.

```{.python .input}
#@tab all
timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'
```

## Đặt tất cả mọi thứ lại với nhau

Bây giờ chúng ta định nghĩa [** hàm `load_data_fashion_mnist` có được và đọc bộ dữ liệu Fashion-MNIST.**] Nó trả về các bộ lặp dữ liệu cho cả bộ đào tạo và bộ xác nhận. Ngoài ra, nó chấp nhận một đối số tùy chọn để thay đổi kích thước hình ảnh thành một hình dạng khác.

```{.python .input}
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    dataset = gluon.data.vision
    trans = [dataset.transforms.ToTensor()]
    if resize:
        trans.insert(0, dataset.transforms.Resize(resize))
    trans = dataset.transforms.Compose(trans)
    mnist_train = dataset.FashionMNIST(train=True).transform_first(trans)
    mnist_test = dataset.FashionMNIST(train=False).transform_first(trans)
    return (gluon.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                  num_workers=get_dataloader_workers()),
            gluon.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                  num_workers=get_dataloader_workers()))
```

```{.python .input}
#@tab pytorch
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))
```

```{.python .input}
#@tab tensorflow
def load_data_fashion_mnist(batch_size, resize=None):   #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
    # Divide all numbers by 255 so that all pixel values are between
    # 0 and 1, add a batch dimension at the last. And cast label to int32
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (
        tf.image.resize_with_pad(X, resize, resize) if resize else X, y)
    return (
        tf.data.Dataset.from_tensor_slices(process(*mnist_train)).batch(
            batch_size).shuffle(len(mnist_train[0])).map(resize_fn),
        tf.data.Dataset.from_tensor_slices(process(*mnist_test)).batch(
            batch_size).map(resize_fn))
```

Dưới đây chúng tôi kiểm tra tính năng thay đổi kích thước hình ảnh của hàm `load_data_fashion_mnist` bằng cách chỉ định đối số `resize`.

```{.python .input}
#@tab all
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
```

Bây giờ chúng tôi đã sẵn sàng để làm việc với tập dữ liệu Fashion-MNIST trong các phần tiếp theo. 

## Tóm tắt

* Fashion-MNIST là một tập dữ liệu phân loại trang phục bao gồm các hình ảnh đại diện cho 10 loại. Chúng tôi sẽ sử dụng tập dữ liệu này trong các phần và chương tiếp theo để đánh giá các thuật toán phân loại khác nhau.
* Chúng tôi lưu trữ hình dạng của bất kỳ hình ảnh nào với chiều cao $h$ chiều rộng $w$ pixel là $h \times w$ hoặc ($h$, $w$).
* Bộ lặp dữ liệu là một thành phần quan trọng cho hiệu suất hiệu quả. Dựa vào các bộ lặp dữ liệu được triển khai tốt khai thác tính toán hiệu suất cao để tránh làm chậm vòng đào tạo của bạn.

## Bài tập

1. Việc giảm `batch_size` (ví dụ, xuống 1) có ảnh hưởng đến hiệu suất đọc không?
1. Hiệu suất lặp dữ liệu rất quan trọng. Bạn có nghĩ rằng việc thực hiện hiện tại là đủ nhanh? Khám phá các tùy chọn khác nhau để cải thiện nó.
1. Kiểm tra tài liệu API trực tuyến của framework. Những bộ dữ liệu nào khác có sẵn?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/48)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/49)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/224)
:end_tab:
