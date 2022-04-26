# Bộ dữ liệu phát hiện đối tượng
:label:`sec_object-detection-dataset`

Không có bộ dữ liệu nhỏ như MNIST và Fashion-MNIST trong lĩnh vực phát hiện đối tượng. Để nhanh chóng chứng minh các mô hình phát hiện đối tượng, [**chúng tôi đã thu thập và dán nhãn một tập dữ liệu nhỏ**]. Đầu tiên, chúng tôi chụp ảnh chuối miễn phí từ văn phòng của chúng tôi và tạo ra 1000 hình ảnh chuối với các vòng quay và kích cỡ khác nhau. Sau đó, chúng tôi đặt mỗi hình ảnh chuối ở một vị trí ngẫu nhiên trên một số hình nền. Cuối cùng, chúng tôi dán nhãn các hộp giới hạn cho những quả chuối trên hình ảnh. 

## [**Tải xuống dữ liệu**]

Bộ dữ liệu phát hiện chuối với tất cả các tệp nhãn hình ảnh và csv có thể được tải xuống trực tiếp từ Internet.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os
import pandas as pd

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
import pandas as pd
```

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')
```

## Đọc tập dữ liệu

Chúng ta sẽ [**đọc dữ liệu phát hiện chuối **] trong hàm `read_data_bananas` bên dưới. Tập dữ liệu bao gồm một tệp csv cho nhãn lớp đối tượng và tọa độ hộp giới hạn đất-chân lý ở góc trên bên trái và dưới bên phải.

```{.python .input}
#@save
def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(image.imread(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, np.expand_dims(np.array(targets), 1) / 256
```

```{.python .input}
#@tab pytorch
#@save
def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256
```

Bằng cách sử dụng chức năng `read_data_bananas` để đọc hình ảnh và nhãn, lớp `BananasDataset` sau sẽ cho phép chúng ta [** tạo một phiên bản `Dataset` tùy chỉnh**] để tải bộ dữ liệu phát hiện chuối.

```{.python .input}
#@save
class BananasDataset(gluon.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].astype('float32').transpose(2, 0, 1),
                self.labels[idx])

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class BananasDataset(torch.utils.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
```

Cuối cùng, chúng ta định nghĩa hàm `load_data_bananas` thành [** trả về hai phiên bản lặp dữ liệu cho cả bộ đào tạo và thử nghiệm.**] Đối với tập dữ liệu thử nghiệm, không cần phải đọc nó theo thứ tự ngẫu nhiên.

```{.python .input}
#@save
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = gluon.data.DataLoader(BananasDataset(is_train=True),
                                       batch_size, shuffle=True)
    val_iter = gluon.data.DataLoader(BananasDataset(is_train=False),
                                     batch_size)
    return train_iter, val_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter
```

Hãy để chúng tôi [** đọc một minibatch và in hình dạng của cả hình ảnh và nhãn**] trong minibatch này. Hình dạng của hình ảnh minibatch, (kích thước lô, số kênh, chiều cao, chiều rộng), trông quen thuộc: nó giống như trong các tác vụ phân loại hình ảnh trước đó của chúng tôi. Hình dạng của minibatch nhãn là (kích thước lô, $m$, 5), trong đó $m$ là số hộp giới hạn lớn nhất có thể có mà bất kỳ hình ảnh nào có trong tập dữ liệu. 

Mặc dù tính toán trong minibatches hiệu quả hơn, nhưng nó đòi hỏi tất cả các ví dụ hình ảnh chứa cùng một số hộp giới hạn để tạo thành một minibatch thông qua nối. Nói chung, hình ảnh có thể có một số hộp giới hạn khác nhau; do đó, hình ảnh có ít hơn $m$ hộp giới hạn sẽ được đệm bằng các hộp giới hạn bất hợp pháp cho đến khi đạt được $m$. Sau đó, nhãn của mỗi hộp giới hạn được biểu diễn bằng một mảng có độ dài 5. Phần tử đầu tiên trong mảng là lớp của đối tượng trong hộp giới hạn, trong đó -1 chỉ ra một hộp giới hạn bất hợp pháp cho padding. Bốn phần tử còn lại của mảng là các giá trị tọa độ ($x$, $y$) -tọa độ của góc trên bên trái và góc dưới bên phải của hộp giới hạn (phạm vi nằm trong khoảng từ 0 đến 1). Đối với tập dữ liệu chuối, vì chỉ có một hộp giới hạn trên mỗi hình ảnh, chúng tôi có $m=1$.

```{.python .input}
#@tab all
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
batch[0].shape, batch[1].shape
```

## [**Demonstration**]

Hãy để chúng tôi chứng minh mười hình ảnh với các hộp giới hạn mặt đất được dán nhãn của họ. Chúng ta có thể thấy rằng các vòng quay, kích thước và vị trí của chuối khác nhau trên tất cả các hình ảnh này. Tất nhiên, đây chỉ là một tập dữ liệu nhân tạo đơn giản. Trong thực tế, các bộ dữ liệu trong thế giới thực thường phức tạp hơn nhiều.

```{.python .input}
imgs = (batch[0][0:10].transpose(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

```{.python .input}
#@tab pytorch
imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

## Tóm tắt

* Bộ dữ liệu phát hiện chuối mà chúng tôi thu thập có thể được sử dụng để chứng minh các mô hình phát hiện đối tượng.
* Việc tải dữ liệu để phát hiện đối tượng tương tự như dữ liệu để phân loại hình ảnh. Tuy nhiên, trong việc phát hiện đối tượng, nhãn cũng chứa thông tin của các hộp giới hạn đất-chân lý, bị thiếu trong phân loại hình ảnh.

## Bài tập

1. Thể hiện các hình ảnh khác với các hộp giới hạn sự thật mặt đất trong bộ dữ liệu phát hiện chuối. Làm thế nào để chúng khác nhau đối với các hộp giới hạn và các đối tượng?
1. Giả sử rằng chúng ta muốn áp dụng tăng cường dữ liệu, chẳng hạn như cắt xén ngẫu nhiên, để phát hiện đối tượng. Làm thế nào nó có thể khác với điều đó trong phân loại hình ảnh? Gợi ý: điều gì sẽ xảy ra nếu một hình ảnh cắt chỉ chứa một phần nhỏ của một đối tượng?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/372)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1608)
:end_tab:
