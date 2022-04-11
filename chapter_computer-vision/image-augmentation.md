# Hình ảnh Augmentation
:label:`sec_image_augmentation`

Trong :numref:`sec_alexnet`, chúng tôi đã đề cập rằng các tập dữ liệu lớn là điều kiện tiên quyết cho sự thành công của các mạng thần kinh sâu trong các ứng dụng khác nhau.
*Tăng cường hình ảnh* 
tạo ra các ví dụ đào tạo tương tự nhưng khác biệt sau một loạt các thay đổi ngẫu nhiên đối với hình ảnh đào tạo, từ đó mở rộng quy mô của bộ đào tạo. Ngoài ra, tăng cường hình ảnh có thể được thúc đẩy bởi thực tế là các tinh chỉnh ngẫu nhiên của các ví dụ đào tạo cho phép các mô hình ít dựa vào các thuộc tính nhất định, do đó cải thiện khả năng tổng quát của chúng. Ví dụ, chúng ta có thể cắt một hình ảnh theo những cách khác nhau để làm cho đối tượng quan tâm xuất hiện ở các vị trí khác nhau, do đó làm giảm sự phụ thuộc của một mô hình vào vị trí của đối tượng. Chúng tôi cũng có thể điều chỉnh các yếu tố như độ sáng và màu sắc để giảm độ nhạy của mô hình với màu sắc. Có lẽ đúng là nâng hình ảnh là không thể thiếu cho sự thành công của AlexNet vào thời điểm đó. Trong phần này, chúng tôi sẽ thảo luận về kỹ thuật được sử dụng rộng rãi này trong tầm nhìn máy tính.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
```

## Phương pháp Augmentation hình ảnh phổ biến

Trong cuộc điều tra của chúng tôi về các phương pháp nâng hình ảnh phổ biến, chúng tôi sẽ sử dụng hình ảnh $400\times 500$ sau đây một ví dụ.

```{.python .input}
d2l.set_figsize()
img = image.imread('../img/cat1.jpg')
d2l.plt.imshow(img.asnumpy());
```

```{.python .input}
#@tab pytorch
d2l.set_figsize()
img = d2l.Image.open('../img/cat1.jpg')
d2l.plt.imshow(img);
```

Hầu hết các phương pháp nâng hình ảnh đều có một mức độ ngẫu nhiên nhất định. Để giúp chúng ta dễ dàng quan sát hiệu ứng của việc nâng hình ảnh hơn, tiếp theo chúng ta xác định một hàm phụ `apply`. Chức năng này chạy phương pháp augmentation hình ảnh `aug` nhiều lần trên hình ảnh đầu vào `img` và hiển thị tất cả các kết quả.

```{.python .input}
#@tab all
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
```

### Lật và cắt xén

:begin_tab:`mxnet`
[**Lật lại hình ảnh bên trái và phải**] thường không thay đổi danh mục của đối tượng. Đây là một trong những phương pháp nâng hình ảnh sớm nhất và được sử dụng rộng rãi nhất. Tiếp theo, chúng ta sử dụng mô-đun `transforms` để tạo phiên bản `RandomFlipLeftRight`, nó lật một hình ảnh sang trái và phải với cơ hội 50%.
:end_tab:

:begin_tab:`pytorch`
[**Lật lại hình ảnh bên trái và phải**] thường không thay đổi danh mục của đối tượng. Đây là một trong những phương pháp nâng hình ảnh sớm nhất và được sử dụng rộng rãi nhất. Tiếp theo, chúng ta sử dụng mô-đun `transforms` để tạo phiên bản `RandomHorizontalFlip`, nó lật một hình ảnh sang trái và phải với cơ hội 50%.
:end_tab:

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomFlipLeftRight())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomHorizontalFlip())
```

:begin_tab:`mxnet`
[**Flipping lên và xuống**] không phổ biến như lật trái và phải. Nhưng ít nhất đối với hình ảnh ví dụ này, lật lên xuống không cản trở sự nhận dạng. Tiếp theo, chúng ta tạo một phiên bản `RandomFlipTopBottom` để lật ảnh lên xuống với 50% cơ hội.
:end_tab:

:begin_tab:`pytorch`
[**Flipping lên và xuống**] không phổ biến như lật trái và phải. Nhưng ít nhất đối với hình ảnh ví dụ này, lật lên xuống không cản trở sự nhận dạng. Tiếp theo, chúng ta tạo một phiên bản `RandomVerticalFlip` để lật ảnh lên xuống với 50% cơ hội.
:end_tab:

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomFlipTopBottom())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomVerticalFlip())
```

Trong hình ảnh ví dụ chúng tôi đã sử dụng, con mèo nằm ở giữa hình ảnh, nhưng điều này có thể không phải là trường hợp nói chung. Trong :numref:`sec_pooling`, chúng tôi giải thích rằng lớp tổng hợp có thể làm giảm độ nhạy của một lớp phức tạp đến vị trí mục tiêu. Ngoài ra, chúng ta cũng có thể cắt ngẫu nhiên hình ảnh để làm cho các đối tượng xuất hiện ở các vị trí khác nhau trong ảnh ở các thang đo khác nhau, điều này cũng có thể làm giảm độ nhạy của một mô hình đến vị trí mục tiêu. 

Trong mã dưới đây, chúng ta [** ngẫu nhiên crop**] một khu vực có diện tích $10\%\ sim 100\ %$ of the original area each time, and the ratio of width to height of this area is randomly selected from $0.5\ sim 2$. Sau đó, chiều rộng và chiều cao của vùng đều được thu nhỏ thành 200 pixel. Trừ khi có quy định khác, số ngẫu nhiên giữa $a$ và $b$ trong phần này đề cập đến một giá trị liên tục thu được bằng cách lấy mẫu ngẫu nhiên và thống nhất từ khoảng $[a, b]$.

```{.python .input}
shape_aug = gluon.data.vision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

```{.python .input}
#@tab pytorch
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

### Thay đổi màu sắc

Một phương pháp nâng cao khác là thay đổi màu sắc. Chúng ta có thể thay đổi bốn khía cạnh của màu hình ảnh: độ sáng, độ tương phản, độ bão hòa và màu sắc. Trong ví dụ dưới đây, chúng ta [**thay đổi ngẫu nhiên độ sáng**] của hình ảnh thành giá trị giữa 50% ($1-0.5$) và 150% ($1+0.5$) của ảnh gốc.

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomBrightness(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))
```

Tương tự, chúng ta có thể [** ngẫu nhiên thay đổi hue**] của hình ảnh.

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomHue(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5))
```

Chúng ta cũng có thể tạo một phiên bản `RandomColorJitter` và đặt cách [** thay đổi ngẫu nhiên `brightness`, `contrast`, `saturation` và `hue` của ảnh cùng một lúc**].

```{.python .input}
color_aug = gluon.data.vision.transforms.RandomColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

```{.python .input}
#@tab pytorch
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

### Kết hợp nhiều phương pháp Augmentation ảnh

Trong thực tế, chúng ta sẽ [** kết hợp nhiều phương pháp tăng cường hình ảnh**]. Ví dụ: chúng ta có thể kết hợp các phương thức augmentation hình ảnh khác nhau được xác định ở trên và áp dụng chúng cho mỗi hình ảnh thông qua một phiên bản `Compose`.

```{.python .input}
augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(), color_aug, shape_aug])
apply(img, augs)
```

```{.python .input}
#@tab pytorch
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
```

## [**Đào tạo với hình ảnh Augmentation**]

Hãy để chúng tôi đào tạo một mô hình với nâng hình ảnh. Ở đây chúng tôi sử dụng bộ dữ liệu CIFAR-10 thay vì tập dữ liệu Fashion-MNIST mà chúng tôi đã sử dụng trước đây. Điều này là do vị trí và kích thước của các đối tượng trong bộ dữ liệu Fashion-MNIST đã được chuẩn hóa, trong khi màu sắc và kích thước của các đối tượng trong tập dữ liệu CIFAR-10 có sự khác biệt đáng kể hơn. 32 hình ảnh đào tạo đầu tiên trong tập dữ liệu CIFAR-10 được hiển thị bên dưới.

```{.python .input}
d2l.show_images(gluon.data.vision.CIFAR10(
    train=True)[0:32][0], 4, 8, scale=0.8);
```

```{.python .input}
#@tab pytorch
all_images = torchvision.datasets.CIFAR10(train=True, root="../data",
                                          download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8);
```

Để có được kết quả dứt khoát trong quá trình dự đoán, chúng ta thường chỉ áp dụng nâng hình ảnh cho các ví dụ đào tạo và không sử dụng nâng hình ảnh với các phép toán ngẫu nhiên trong quá trình dự đoán. [**Ở đây chúng tôi chỉ sử dụng phương pháp lật trái phải ngẫu nhiên đơn giản nhất**]. Ngoài ra, chúng ta sử dụng một phiên bản `ToTensor` để chuyển đổi một minibatch ảnh thành định dạng theo yêu cầu của khung học sâu, tức là các số điểm nổi 32 bit giữa 0 đến 1 với hình dạng (kích thước lô, số kênh, chiều cao, chiều rộng).

```{.python .input}
train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor()])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor()])
```

```{.python .input}
#@tab pytorch
train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])
```

:begin_tab:`mxnet`
Tiếp theo, chúng tôi xác định một chức năng phụ trợ để tạo điều kiện đọc hình ảnh và áp dụng nâng hình ảnh. Hàm `transform_first` được cung cấp bởi các bộ dữ liệu của Gluon áp dụng nâng hình ảnh cho phần tử đầu tiên của mỗi ví dụ đào tạo (hình ảnh và nhãn), tức là hình ảnh. Để biết giới thiệu chi tiết về `DataLoader`, vui lòng tham khảo :numref:`sec_fashion_mnist`.
:end_tab:

:begin_tab:`pytorch`
Tiếp theo, chúng ta [** xác định một hàm phụ trợ để tạo điều kiện đọc hình ảnh và áp dụng tăng cường hình ảnh**]. Đối số `transform` được cung cấp bởi tập dữ liệu của PyTorch áp dụng tăng cường để biến đổi hình ảnh. Để biết giới thiệu chi tiết về `DataLoader`, vui lòng tham khảo :numref:`sec_fashion_mnist`.
:end_tab:

```{.python .input}
def load_cifar10(is_train, augs, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=is_train).transform_first(augs),
        batch_size=batch_size, shuffle=is_train,
        num_workers=d2l.get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader
```

### Đào tạo Multi-GPU

Chúng tôi đào tạo mô hình ResNet-18 từ :numref:`sec_resnet` trên bộ dữ liệu CIFAR-10. Nhớ lại phần giới thiệu về đào tạo đa GPU trong :numref:`sec_multi_gpu_concise`. Sau đây, [**chúng tôi định nghĩa một hàm để đào tạo và đánh giá mô hình bằng cách sử dụng nhiều GPU **].

```{.python .input}
#@save
def train_batch_ch13(net, features, labels, loss, trainer, devices,
                     split_f=d2l.split_batch):
    """Train for a minibatch with mutiple GPUs (defined in Chapter 13)."""
    X_shards, y_shards = split_f(features, labels, devices)
    with autograd.record():
        pred_shards = [net(X_shard) for X_shard in X_shards]
        ls = [loss(pred_shard, y_shard) for pred_shard, y_shard
              in zip(pred_shards, y_shards)]
    for l in ls:
        l.backward()
    # The `True` flag allows parameters with stale gradients, which is useful
    # later (e.g., in fine-tuning BERT)
    trainer.step(labels.shape[0], ignore_stale_grad=True)
    train_loss_sum = sum([float(l.sum()) for l in ls])
    train_acc_sum = sum(d2l.accuracy(pred_shard, y_shard)
                        for pred_shard, y_shard in zip(pred_shards, y_shards))
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@tab pytorch
#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """Train for a minibatch with mutiple GPUs (defined in Chapter 13)."""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus(), split_f=d2l.split_batch):
    """Train a model with mutiple GPUs (defined in Chapter 13)."""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices, split_f)
            metric.add(l, acc, labels.shape[0], labels.size)
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpus(net, test_iter, split_f)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """Train a model with mutiple GPUs (defined in Chapter 13)."""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

Bây giờ chúng ta có thể [** xác định hàm `train_with_data_aug` để đào tạo mô hình với khả năng tăng cường hình ảnh**]. Chức năng này nhận được tất cả các GPU có sẵn, sử dụng Adam làm thuật toán tối ưu hóa, áp dụng nâng hình ảnh cho tập dữ liệu đào tạo và cuối cùng gọi hàm `train_ch13` vừa được định nghĩa để đào tạo và đánh giá mô hình.

```{.python .input}
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

```{.python .input}
#@tab pytorch
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

Hãy để chúng tôi [** đào tạo mô hình**] bằng cách sử dụng nâng hình ảnh dựa trên lật trái phải ngẫu nhiên.

```{.python .input}
#@tab all
train_with_data_aug(train_augs, test_augs, net)
```

## Tóm tắt

* Nâng hình ảnh tạo ra hình ảnh ngẫu nhiên dựa trên dữ liệu đào tạo hiện có để cải thiện khả năng tổng quát hóa của các mô hình.
* Để có được kết quả dứt khoát trong quá trình dự đoán, chúng ta thường chỉ áp dụng nâng hình ảnh cho các ví dụ đào tạo và không sử dụng nâng hình ảnh với các phép toán ngẫu nhiên trong quá trình dự đoán.
* Khung học sâu cung cấp nhiều phương pháp nâng hình ảnh khác nhau, có thể được áp dụng đồng thời.

## Bài tập

1. Đào tạo mô hình mà không cần sử dụng nâng hình ảnh: `train_with_data_aug(test_augs, test_augs)`. So sánh độ chính xác của đào tạo và kiểm tra khi sử dụng và không sử dụng nâng hình ảnh. Thí nghiệm so sánh này có thể hỗ trợ lập luận rằng việc tăng cường hình ảnh có thể giảm thiểu quá mức không? Tại sao?
1. Kết hợp nhiều phương pháp nâng hình ảnh khác nhau trong đào tạo mô hình trên tập dữ liệu CIFAR-10. Nó có cải thiện độ chính xác của thử nghiệm không? 
1. Tham khảo tài liệu trực tuyến của khung học sâu. Những phương pháp nâng hình ảnh nào khác mà nó cũng cung cấp?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/367)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1404)
:end_tab:
