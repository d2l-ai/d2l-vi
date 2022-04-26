# Nhận dạng giống chó (Chó ImageNet) trên Kaggle

Trong phần này, chúng tôi sẽ thực hành vấn đề nhận dạng giống chó trên Kaggle. (**Địa chỉ web của cuộc thi này là https://www.kaggle.com/c/dog-breed-identification **) 

Trong cuộc thi này, 120 giống chó khác nhau sẽ được công nhận. Trên thực tế, tập dữ liệu cho cuộc thi này là một tập hợp con của tập dữ liệu ImageNet. Không giống như các hình ảnh trong tập dữ liệu CIFAR-10 trong :numref:`sec_kaggle_cifar10`, các hình ảnh trong tập dữ liệu ImageNet đều cao hơn và rộng hơn ở các kích thước khác nhau. :numref:`fig_kaggle_dog` hiển thị thông tin trên trang web của đối thủ. Bạn cần một tài khoản Kaggle để gửi kết quả của mình. 

![The dog breed identification competition website. The competition dataset can be obtained by clicking the "Data" tab.](../img/kaggle-dog.jpg)
:width:`400px`
:label:`fig_kaggle_dog`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
import os
```

## Lấy và tổ chức tập dữ liệu

Tập dữ liệu cạnh tranh được chia thành một bộ đào tạo và một bộ thử nghiệm, trong đó có 10222 và 10357 hình ảnh JPEG của ba kênh RGB (màu), tương ứng. Trong số các tập dữ liệu huấn luyện, có 120 giống chó như Labradors, Poodles, Dachshunds, Samoyeds, Huskies, Chihuahua, và Yorkshire Terriers. 

### Tải xuống tập dữ liệu

Sau khi đăng nhập vào Kaggle, bạn có thể nhấp vào tab “Dữ liệu” trên trang web cạnh tranh được hiển thị trong :numref:`fig_kaggle_dog` và tải xuống tập dữ liệu bằng cách nhấp vào nút “Tải xuống tất cả”. Sau khi giải nén tệp đã tải xuống trong `../data`, bạn sẽ tìm thấy toàn bộ tập dữ liệu trong các đường dẫn sau: 

* .. /data/dog-breed-identification/labels.csv
* .. /data/dog-breed-identification/sample_submission.csv
* .. /data/dog-breed-identification/tàu
* .. /data/dog-breed-identification/test

Bạn có thể nhận thấy rằng cấu trúc trên tương tự như cấu trúc của cuộc thi CIFAR-10 trong :numref:`sec_kaggle_cifar10`, trong đó các thư mục `train/` và `test/` chứa các hình ảnh huấn luyện và thử nghiệm hình ảnh chó, và `labels.csv` chứa nhãn cho hình ảnh đào tạo. Tương tự, để bắt đầu dễ dàng hơn, [**chúng tôi cung cấp một mẫu nhỏ của bộ dữ liệu**] được đề cập ở trên: `train_valid_test_tiny.zip`. Nếu bạn định sử dụng bộ dữ liệu đầy đủ cho cuộc thi Kaggle, bạn cần thay đổi biến `demo` bên dưới thành `False`.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                            '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')

# If you use the full dataset downloaded for the Kaggle competition, change
# the variable below to `False`
demo = True
if demo:
    data_dir = d2l.download_extract('dog_tiny')
else:
    data_dir = os.path.join('..', 'data', 'dog-breed-identification')
```

### [**Torganizing the Dataset**]

Chúng ta có thể sắp xếp tập dữ liệu tương tự như những gì chúng ta đã làm trong :numref:`sec_kaggle_cifar10`, cụ thể là tách ra một bộ xác thực từ bộ đào tạo ban đầu và di chuyển hình ảnh vào các thư mục con được nhóm theo nhãn. 

Chức năng `reorg_dog_data` dưới đây đọc các nhãn dữ liệu đào tạo, tách bộ xác thực và tổ chức bộ đào tạo.

```{.python .input}
#@tab all
def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)


batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)
```

## [**Image Augmentation**]

Nhớ lại rằng tập dữ liệu giống chó này là một tập hợp con của tập dữ liệu ImageNet, có hình ảnh lớn hơn của tập dữ liệu CIFAR-10 vào năm :numref:`sec_kaggle_cifar10`. Sau đây liệt kê một vài thao tác nâng hình ảnh có thể hữu ích cho các hình ảnh tương đối lớn hơn.

```{.python .input}
transform_train = gluon.data.vision.transforms.Compose([
    # Randomly crop the image to obtain an image with an area of 0.08 to 1 of
    # the original area and height-to-width ratio between 3/4 and 4/3. Then,
    # scale the image to create a new 224 x 224 image
    gluon.data.vision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                                   ratio=(3.0/4.0, 4.0/3.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    # Randomly change the brightness, contrast, and saturation
    gluon.data.vision.transforms.RandomColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4),
    # Add random noise
    gluon.data.vision.transforms.RandomLighting(0.1),
    gluon.data.vision.transforms.ToTensor(),
    # Standardize each channel of the image
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```

```{.python .input}
#@tab pytorch
transform_train = torchvision.transforms.Compose([
    # Randomly crop the image to obtain an image with an area of 0.08 to 1 of
    # the original area and height-to-width ratio between 3/4 and 4/3. Then,
    # scale the image to create a new 224 x 224 image
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             ratio=(3.0/4.0, 4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # Randomly change the brightness, contrast, and saturation
    torchvision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),
    # Add random noise
    torchvision.transforms.ToTensor(),
    # Standardize each channel of the image
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

Trong quá trình dự đoán, chúng tôi chỉ sử dụng các thao tác tiền xử lý hình ảnh mà không có tính ngẫu nhiên.

```{.python .input}
transform_test = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    # Crop a 224 x 224 square area from the center of the image
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```

```{.python .input}
#@tab pytorch
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    # Crop a 224 x 224 square area from the center of the image
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

## [**Đọc dữ liệu**]

Như trong :numref:`sec_kaggle_cifar10`, chúng ta có thể đọc tập dữ liệu có tổ chức bao gồm các tệp hình ảnh thô.

```{.python .input}
train_ds, valid_ds, train_valid_ds, test_ds = [
    gluon.data.vision.ImageFolderDataset(
        os.path.join(data_dir, 'train_valid_test', folder))
    for folder in ('train', 'valid', 'train_valid', 'test')]
```

```{.python .input}
#@tab pytorch
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]
```

Dưới đây chúng ta tạo các trường hợp lặp dữ liệu giống như trong :numref:`sec_kaggle_cifar10`.

```{.python .input}
train_iter, train_valid_iter = [gluon.data.DataLoader(
    dataset.transform_first(transform_train), batch_size, shuffle=True,
    last_batch='discard') for dataset in (train_ds, train_valid_ds)]

valid_iter = gluon.data.DataLoader(
    valid_ds.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='discard')

test_iter = gluon.data.DataLoader(
    test_ds.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='keep')
```

```{.python .input}
#@tab pytorch
train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)
```

## [**Tinh chỉnh một mô hình Pretrained Model**]

Một lần nữa, tập dữ liệu cho cuộc thi này là một tập hợp con của tập dữ liệu ImageNet. Do đó, chúng ta có thể sử dụng phương pháp được thảo luận trong :numref:`sec_fine_tuning` để chọn một mô hình được đào tạo trước trên tập dữ liệu ImageNet đầy đủ và sử dụng nó để trích xuất các tính năng hình ảnh được đưa vào mạng đầu ra quy mô nhỏ tùy chỉnh. API cấp cao của các framework deep learning cung cấp một loạt các mô hình được đào tạo trước trên tập dữ liệu ImageNet. Ở đây, chúng tôi chọn một mô hình ResNet-34 được đào tạo trước, nơi chúng tôi chỉ cần sử dụng lại đầu vào của lớp đầu ra của mô hình này (tức là các tính năng được trích xuất). Sau đó, chúng ta có thể thay thế lớp đầu ra ban đầu bằng một mạng đầu ra tùy chỉnh nhỏ có thể được đào tạo, chẳng hạn như xếp chồng hai lớp được kết nối hoàn toàn. Khác với thí nghiệm trong :numref:`sec_fine_tuning`, những điều sau đây không đào tạo lại mô hình được đào tạo trước được sử dụng để trích xuất tính năng. Điều này làm giảm thời gian đào tạo và bộ nhớ để lưu trữ gradient. 

Nhớ lại rằng chúng tôi chuẩn hóa hình ảnh bằng cách sử dụng phương tiện và độ lệch chuẩn của ba kênh RGB cho tập dữ liệu ImageNet đầy đủ. Trên thực tế, điều này cũng phù hợp với hoạt động tiêu chuẩn hóa bởi mô hình được đào tạo trước trên ImageNet.

```{.python .input}
def get_net(devices):
    finetune_net = gluon.model_zoo.vision.resnet34_v2(pretrained=True)
    # Define a new output network
    finetune_net.output_new = nn.HybridSequential(prefix='')
    finetune_net.output_new.add(nn.Dense(256, activation='relu'))
    # There are 120 output categories
    finetune_net.output_new.add(nn.Dense(120))
    # Initialize the output network
    finetune_net.output_new.initialize(init.Xavier(), ctx=devices)
    # Distribute the model parameters to the CPUs or GPUs used for computation
    finetune_net.collect_params().reset_ctx(devices)
    return finetune_net
```

```{.python .input}
#@tab pytorch
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # Define a new output network (there are 120 output categories)
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # Move the model to devices
    finetune_net = finetune_net.to(devices[0])
    # Freeze parameters of feature layers
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net
```

Trước khi [** tính lấu**], trước tiên chúng ta có được đầu vào của lớp đầu ra của mô hình được đào tạo sẵn, tức là tính năng trích xuất. Sau đó, chúng tôi sử dụng tính năng này làm đầu vào cho mạng đầu ra tùy chỉnh nhỏ của chúng tôi để tính toán tổn thất.

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        X_shards, y_shards = d2l.split_batch(features, labels, devices)
        output_features = [net.features(X_shard) for X_shard in X_shards]
        outputs = [net.output_new(feature) for feature in output_features]
        ls = [loss(output, y_shard).sum() for output, y_shard
              in zip(outputs, y_shards)]
        l_sum += sum([float(l.sum()) for l in ls])
        n += labels.size
    return l_sum / n
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss(reduction='none')

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return l_sum / n
```

## Xác định [**chức năng đào tạo**]

Chúng tôi sẽ chọn mô hình và điều chỉnh các siêu tham số theo hiệu suất của mô hình trên bộ xác thực. Chức năng đào tạo mô hình `train` chỉ lặp lại các thông số của mạng đầu ra tùy chỉnh nhỏ.

```{.python .input}
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # Only train the small custom output network
    trainer = gluon.Trainer(net.output_new.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            X_shards, y_shards = d2l.split_batch(features, labels, devices)
            output_features = [net.features(X_shard) for X_shard in X_shards]
            with autograd.record():
                outputs = [net.output_new(feature)
                           for feature in output_features]
                ls = [loss(output, y_shard).sum() for output, y_shard
                      in zip(outputs, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            metric.add(sum([float(l.sum()) for l in ls]), labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss))
    measures = f'train loss {metric[0] / metric[1]:.3f}'
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # Only train the small custom output network
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=lr,
                              momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss.detach().cpu()))
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

## [**Đào tạo và xác thực mô hình**]

Bây giờ chúng ta có thể đào tạo và xác thực mô hình. Các siêu tham số sau đây đều có thể điều chỉnh được. Ví dụ, số lượng kỷ nguyên có thể được tăng lên. Bởi vì `lr_period` và `lr_decay` được đặt thành 2 và 0,9, tương ứng, tỷ lệ học tập của thuật toán tối ưu hóa sẽ được nhân với 0,9 sau mỗi 2 kỷ nguyên.

```{.python .input}
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 5e-3, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
net.hybridize()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

```{.python .input}
#@tab pytorch
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 1e-4, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

## [**Phân loại bộ thử nghiệm**] và gửi kết quả trên Kaggle

Tương tự như bước cuối cùng trong :numref:`sec_kaggle_cifar10`, cuối cùng tất cả dữ liệu được dán nhãn (bao gồm cả bộ xác thực) được sử dụng để đào tạo mô hình và phân loại bộ thử nghiệm. Chúng tôi sẽ sử dụng mạng đầu ra tùy chỉnh được đào tạo để phân loại.

```{.python .input}
net = get_net(devices)
net.hybridize()
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output_features = net.features(data.as_in_ctx(devices[0]))
    output = npx.softmax(net.output_new(output_features))
    preds.extend(output.asnumpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.synsets) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

```{.python .input}
#@tab pytorch
net = get_net(devices)
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output = torch.nn.functional.softmax(net(data.to(devices[0])), dim=0)
    preds.extend(output.cpu().detach().numpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

Đoạn mã trên sẽ tạo ra một tập tin `submission.csv` để gửi cho Kaggle theo cách tương tự được mô tả trong :numref:`sec_kaggle_house`. 

## Tóm tắt

* Hình ảnh trong tập dữ liệu ImageNet lớn hơn (với các kích thước khác nhau) so với hình ảnh CIFAR-10. Chúng tôi có thể sửa đổi các thao tác nâng hình ảnh cho các tác vụ trên một tập dữ liệu khác.
* Để phân loại một tập hợp con của tập dữ liệu ImageNet, chúng ta có thể tận dụng các mô hình được đào tạo trước trên tập dữ liệu ImageNet đầy đủ để trích xuất các tính năng và chỉ đào tạo mạng đầu ra quy mô nhỏ tùy chỉnh. Điều này sẽ dẫn đến thời gian tính toán và chi phí bộ nhớ ít hơn.

## Bài tập

1. Khi sử dụng bộ dữ liệu cạnh tranh Kaggle đầy đủ, bạn có thể đạt được kết quả gì khi tăng `batch_size` (kích thước lô) và `num_epochs` (số thời đại) trong khi đặt một số siêu tham số khác là `lr = 0.01`, `lr_period = 10` và `lr_decay = 0.1`?
1. Bạn có nhận được kết quả tốt hơn nếu bạn sử dụng một mô hình được đào tạo sâu hơn? Làm thế nào để bạn điều chỉnh các siêu tham số? Bạn có thể cải thiện hơn nữa kết quả?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/380)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1481)
:end_tab:
