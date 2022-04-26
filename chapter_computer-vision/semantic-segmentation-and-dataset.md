# Phân đoạn ngữ nghĩa và tập dữ liệu
:label:`sec_semantic_segmentation`

Khi thảo luận về các tác vụ phát hiện đối tượng trong :numref:`sec_bbox`—:numref:`sec_rcnn`, các hộp giới hạn hình chữ nhật được sử dụng để dán nhãn và dự đoán các đối tượng trong ảnh. Phần này sẽ thảo luận về vấn đề phân đoạn *ngữ nghĩa *, tập trung vào cách chia hình ảnh thành các vùng thuộc các lớp ngữ nghĩa khác nhau. Khác với phát hiện đối tượng, phân đoạn ngữ nghĩa nhận ra và hiểu những gì có trong hình ảnh ở cấp độ pixel: ghi nhãn và dự đoán các vùng ngữ nghĩa của nó ở mức pixel. :numref:`fig_segmentation` hiển thị nhãn của chó, mèo và nền của hình ảnh trong phân khúc ngữ nghĩa. So với trong phát hiện đối tượng, các đường viền cấp pixel được dán nhãn trong phân đoạn ngữ nghĩa rõ ràng là hạt mịn hơn. 

![Labels of the dog, cat, and background of the image in semantic segmentation.](../img/segmentation.svg)
:label:`fig_segmentation`

## Phân đoạn hình ảnh và Phân đoạn phiên bản

Ngoài ra còn có hai nhiệm vụ quan trọng trong lĩnh vực thị giác máy tính tương tự như phân đoạn ngữ nghĩa, đó là phân đoạn hình ảnh và phân đoạn phiên bản. Chúng tôi sẽ phân biệt ngắn gọn chúng với phân khúc ngữ nghĩa như sau. 

* *Phân khúc hình ảnh* chia một hình ảnh thành nhiều vùng cấu thành. Các phương pháp cho loại vấn đề này thường sử dụng mối tương quan giữa các pixel trong hình ảnh. Nó không cần thông tin nhãn về pixel hình ảnh trong quá trình đào tạo và nó không thể đảm bảo rằng các vùng được phân đoạn sẽ có ngữ nghĩa mà chúng tôi hy vọng sẽ có được trong quá trình dự đoán. Chụp ảnh trong :numref:`fig_segmentation` làm đầu vào, phân đoạn hình ảnh có thể chia chó thành hai vùng: một vùng che miệng và mắt chủ yếu là màu đen, và phần còn lại bao phủ phần còn lại của cơ thể chủ yếu là màu vàng.
* * Phân khúc Instance* còn được gọi là * phát hiện và phân đoạn đồng thời*. Nó nghiên cứu làm thế nào để nhận ra các vùng cấp pixel của mỗi đối tượng trong một hình ảnh. Khác với phân đoạn ngữ nghĩa, phân đoạn phiên bản cần phân biệt không chỉ ngữ nghĩa, mà còn các trường hợp đối tượng khác nhau. Ví dụ, nếu có hai chú chó trong hình ảnh, phân đoạn ví dụ cần phân biệt một pixel thuộc về cái nào trong hai chú chó.

## Tập dữ liệu phân đoạn ngữ nghĩa Pascal VOC2012

[**Trên tập dữ liệu phân đoạn ngữ nghĩa quan trọng nhất là [Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).**] Sau đây, chúng ta sẽ xem tập dữ liệu này.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
```

Tệp tar của tập dữ liệu khoảng 2 GB, vì vậy có thể mất một thời gian để tải xuống tệp. Tập dữ liệu được trích xuất nằm ở `../data/VOCdevkit/VOC2012`.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
```

Sau khi nhập đường dẫn `../data/VOCdevkit/VOC2012`, chúng ta có thể thấy các thành phần khác nhau của tập dữ liệu. Đường dẫn `ImageSets/Segmentation` chứa các tệp văn bản chỉ định các mẫu đào tạo và thử nghiệm, trong khi các đường dẫn `JPEGImages` và `SegmentationClass` lưu trữ hình ảnh đầu vào và nhãn cho mỗi ví dụ tương ứng. Nhãn ở đây cũng ở định dạng hình ảnh, có cùng kích thước với hình ảnh đầu vào được dán nhãn của nó. Bên cạnh đó, các pixel có cùng màu trong bất kỳ hình ảnh nhãn nào thuộc cùng một lớp ngữ nghĩa. Sau đây xác định hàm `read_voc_images` thành [** đọc tất cả các hình ảnh đầu vào và nhãn vào bộ nhớ**].

```{.python .input}
#@save
def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images."""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(image.imread(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(image.imread(os.path.join(
            voc_dir, 'SegmentationClass', f'{fname}.png')))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
```

```{.python .input}
#@tab pytorch
#@save
def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images."""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
```

Chúng tôi [** vẽ năm hình ảnh đầu vào đầu tiên và nhãn của chúng**]. Trong hình ảnh nhãn, màu trắng và đen đại diện cho đường viền và nền tương ứng, trong khi các màu khác tương ứng với các lớp khác nhau.

```{.python .input}
n = 5
imgs = train_features[0:n] + train_labels[0:n]
d2l.show_images(imgs, 2, n);
```

```{.python .input}
#@tab pytorch
n = 5
imgs = train_features[0:n] + train_labels[0:n]
imgs = [img.permute(1,2,0) for img in imgs]
d2l.show_images(imgs, 2, n);
```

Tiếp theo, chúng ta [** liệt kê các giá trị màu RGB và tên lớp **] cho tất cả các nhãn trong tập dữ liệu này.

```{.python .input}
#@tab all
#@save
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

#@save
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
```

Với hai hằng số được định nghĩa ở trên, chúng ta có thể thuận tiện [** tìm chỉ mục lớp cho mỗi pixel trong một nhãy**]. Chúng tôi xác định hàm `voc_colormap2label` để xây dựng ánh xạ từ các giá trị màu RGB ở trên đến các chỉ số lớp và hàm `voc_label_indices` để ánh xạ bất kỳ giá trị RGB nào với chỉ số lớp của chúng trong bộ dữ liệu Pascal VOC2012 này.

```{.python .input}
#@save
def voc_colormap2label():
    """Build the mapping from RGB to class indices for VOC labels."""
    colormap2label = np.zeros(256 ** 3)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """Map any RGB values in VOC labels to their class indices."""
    colormap = colormap.astype(np.int32)
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
```

```{.python .input}
#@tab pytorch
#@save
def voc_colormap2label():
    """Build the mapping from RGB to class indices for VOC labels."""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """Map any RGB values in VOC labels to their class indices."""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
```

[**Ví dụ**], trong ảnh ví dụ đầu tiên, chỉ mục lớp cho phần trước của máy bay là 1, trong khi chỉ mục nền là 0.

```{.python .input}
#@tab all
y = voc_label_indices(train_labels[0], voc_colormap2label())
y[105:115, 130:140], VOC_CLASSES[1]
```

### Xử lý sơ bộ dữ liệu

Trong các thí nghiệm trước đó như trong :numref:`sec_alexnet`—:numref:`sec_googlenet`, hình ảnh được thay đổi lại để phù hợp với hình dạng đầu vào yêu cầu của mô hình. Tuy nhiên, trong phân đoạn ngữ nghĩa, làm như vậy đòi hỏi phải thay đổi lại các lớp pixel dự đoán trở lại hình dạng ban đầu của hình ảnh đầu vào. Việc tái cặn như vậy có thể không chính xác, đặc biệt là đối với các vùng được phân đoạn với các lớp khác nhau. Để tránh sự cố này, chúng tôi cắt hình ảnh thành một hình dạng *fixed* thay vì rescaling. Cụ thể, [** sử dụng cắt xén ngẫu nhiên từ nâng hình ảnh, chúng ta cắt cùng một vùng của hình ảnh đầu vào và nhãn**].

```{.python .input}
#@save
def voc_rand_crop(feature, label, height, width):
    """Randomly crop both feature and label images."""
    feature, rect = image.random_crop(feature, (width, height))
    label = image.fixed_crop(label, *rect)
    return feature, label
```

```{.python .input}
#@tab pytorch
#@save
def voc_rand_crop(feature, label, height, width):
    """Randomly crop both feature and label images."""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label
```

```{.python .input}
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
```

```{.python .input}
#@tab pytorch
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)

imgs = [img.permute(1, 2, 0) for img in imgs]
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
```

### [**Custom Semantic Segmentation Dataset Class**]

Chúng tôi xác định một tập dữ liệu phân đoạn ngữ nghĩa tùy chỉnh lớp `VOCSegDataset` bằng cách kế thừa lớp `Dataset` được cung cấp bởi các API cấp cao. Bằng cách thực hiện hàm `__getitem__`, chúng ta có thể tùy ý truy cập vào hình ảnh đầu vào được lập chỉ mục là `idx` trong tập dữ liệu và chỉ số lớp của mỗi pixel trong hình ảnh này. Vì một số hình ảnh trong tập dữ liệu có kích thước nhỏ hơn kích thước đầu ra của cắt ngẫu nhiên, các ví dụ này được lọc ra bởi một hàm `filter` tùy chỉnh. Ngoài ra, chúng tôi cũng xác định hàm `normalize_image` để chuẩn hóa các giá trị của ba kênh RGB của hình ảnh đầu vào.

```{.python .input}
#@save
class VOCSegDataset(gluon.data.Dataset):
    """A customized dataset to load the VOC dataset."""
    def __init__(self, is_train, crop_size, voc_dir):
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return (img.astype('float32') / 255 - self.rgb_mean) / self.rgb_std

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[0] >= self.crop_size[0] and
            img.shape[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature.transpose(2, 0, 1),
                voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class VOCSegDataset(torch.utils.data.Dataset):
    """A customized dataset to load the VOC dataset."""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
```

### [**Đọc dữ liệu**]

Chúng tôi sử dụng lớp `VOCSegDatase`t tùy chỉnh để tạo các phiên bản của bộ đào tạo và bộ kiểm tra, tương ứng. Giả sử rằng chúng ta chỉ định rằng hình dạng đầu ra của hình ảnh được cắt ngẫu nhiên là $320\times 480$. Dưới đây chúng ta có thể xem số ví dụ được giữ lại trong bộ đào tạo và bộ kiểm tra.

```{.python .input}
#@tab all
crop_size = (320, 480)
voc_train = VOCSegDataset(True, crop_size, voc_dir)
voc_test = VOCSegDataset(False, crop_size, voc_dir)
```

Đặt kích thước lô thành 64, chúng tôi xác định bộ lặp dữ liệu cho bộ đào tạo. Hãy để chúng tôi in hình dạng của minibatch đầu tiên. Khác với phân loại hình ảnh hoặc phát hiện đối tượng, nhãn ở đây là hàng chục ba chiều.

```{.python .input}
batch_size = 64
train_iter = gluon.data.DataLoader(voc_train, batch_size, shuffle=True,
                                   last_batch='discard',
                                   num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
```

```{.python .input}
#@tab pytorch
batch_size = 64
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                                    drop_last=True,
                                    num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
```

### [** Putting All Things Together**]

Cuối cùng, chúng tôi xác định hàm `load_data_voc` sau để tải xuống và đọc tập dữ liệu phân đoạn ngữ nghĩa Pascal VOC2012. Nó trả về bộ lặp dữ liệu cho cả tập dữ liệu đào tạo và kiểm tra.

```{.python .input}
#@save
def load_data_voc(batch_size, crop_size):
    """Load the VOC semantic segmentation dataset."""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = gluon.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, last_batch='discard', num_workers=num_workers)
    test_iter = gluon.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        last_batch='discard', num_workers=num_workers)
    return train_iter, test_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_voc(batch_size, crop_size):
    """Load the VOC semantic segmentation dataset."""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter
```

## Tóm tắt

* Phân đoạn ngữ nghĩa nhận ra và hiểu những gì trong một hình ảnh ở mức pixel bằng cách chia hình ảnh thành các vùng thuộc các lớp ngữ nghĩa khác nhau.
* Trên của tập dữ liệu phân khúc ngữ nghĩa quan trọng nhất là Pascal VOC2012.
* Trong phân đoạn ngữ nghĩa, vì hình ảnh đầu vào và nhãn tương ứng một-một trên pixel, hình ảnh đầu vào được cắt ngẫu nhiên thành một hình dạng cố định chứ không phải rescaled.

## Bài tập

1. Làm thế nào phân khúc ngữ nghĩa có thể được áp dụng trong các phương tiện tự trị và chẩn đoán hình ảnh y tế? Bạn có thể nghĩ về các ứng dụng khác?
1. Nhớ lại các mô tả về tăng cường dữ liệu trong :numref:`sec_image_augmentation`. Phương pháp nâng hình ảnh nào được sử dụng trong phân loại hình ảnh sẽ không khả thi được áp dụng trong phân khúc ngữ nghĩa?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/375)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1480)
:end_tab:
