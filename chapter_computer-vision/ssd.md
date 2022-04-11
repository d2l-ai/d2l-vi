# Phát hiện Multibox Shot đơn
:label:`sec_ssd`

Trong :numref:`sec_bbox`—:numref:`sec_object-detection-dataset`, chúng tôi giới thiệu các hộp giới hạn, hộp neo, phát hiện đối tượng đa quy mô và bộ dữ liệu để phát hiện đối tượng. Bây giờ chúng tôi đã sẵn sàng để sử dụng kiến thức nền như vậy để thiết kế một mô hình phát hiện đối tượng: phát hiện multibox shot đơn (SSD) :cite:`Liu.Anguelov.Erhan.ea.2016`. Mô hình này đơn giản, nhanh chóng và được sử dụng rộng rãi. Mặc dù đây chỉ là một trong số lượng lớn các mô hình phát hiện đối tượng, một số nguyên tắc thiết kế và chi tiết triển khai trong phần này cũng được áp dụng cho các mô hình khác. 

## Mô hình

:numref:`fig_ssd` cung cấp một cái nhìn tổng quan về thiết kế phát hiện multibox một shot. Mô hình này chủ yếu bao gồm một mạng cơ sở theo sau là một số khối bản đồ tính năng đa thang. Mạng cơ sở là để trích xuất các tính năng từ hình ảnh đầu vào, vì vậy nó có thể sử dụng CNN sâu. Ví dụ, giấy phát hiện multibox một shot ban đầu sử dụng một mạng VGG cắt ngắn trước lớp phân loại :cite:`Liu.Anguelov.Erhan.ea.2016`, trong khi ResNet cũng đã được sử dụng phổ biến. Thông qua thiết kế của chúng tôi, chúng tôi có thể làm cho các bản đồ tính năng đầu ra mạng cơ sở lớn hơn để tạo ra nhiều hộp neo hơn để phát hiện các đối tượng nhỏ hơn. Sau đó, mỗi khối bản đồ tính năng đa thang giảm (ví dụ, bằng một nửa) chiều cao và chiều rộng của bản đồ tính năng từ khối trước đó và cho phép mỗi đơn vị của bản đồ tính năng tăng trường tiếp nhận của nó trên hình ảnh đầu vào. 

Nhớ lại thiết kế phát hiện đối tượng đa quy mô thông qua các biểu diễn theo lớp của hình ảnh bằng các mạng thần kinh sâu trong :numref:`sec_multiscale-object-detection`. Vì bản đồ tính năng đa quy mô gần với đỉnh :numref:`fig_ssd` nhỏ hơn nhưng có các trường tiếp nhận lớn hơn, chúng phù hợp để phát hiện các vật thể ít hơn nhưng lớn hơn. 

Tóm lại, thông qua mạng cơ sở của nó và một số khối bản đồ tính năng đa quy mô, phát hiện multibox single-shot tạo ra một số lượng khác nhau của các hộp neo với kích thước khác nhau, và phát hiện các đối tượng kích thước khác nhau bằng cách dự đoán các lớp và bù đắp của các hộp neo (do đó các hộp giới hạn); do đó, đây là một mô hình phát hiện đối tượng đa quy mô. 

![As a multiscale object detection model, single-shot multibox detection mainly consists of a base network followed by several multiscale feature map blocks.](../img/ssd.svg)
:label:`fig_ssd`

Sau đây, chúng tôi sẽ mô tả các chi tiết triển khai của các khối khác nhau trong :numref:`fig_ssd`. Để bắt đầu, chúng tôi thảo luận về cách thực hiện dự đoán lớp và hộp giới hạn. 

### [** Lớp dự đoán lớp **]

Hãy để số lượng các lớp đối tượng là $q$. Sau đó, hộp neo có $q+1$ lớp, trong đó lớp 0 là nền. Ở một số quy mô, giả sử rằng chiều cao và chiều rộng của bản đồ tính năng lần lượt là $h$ và $w$. Khi $a$ hộp neo được tạo ra với mỗi vị trí không gian của các bản đồ tính năng này làm trung tâm của chúng, tổng cộng $hwa$ hộp neo cần được phân loại. Điều này thường làm cho việc phân loại với các lớp kết nối hoàn toàn không khả thi do chi phí tham số hóa nặng. Nhớ lại cách chúng ta sử dụng các kênh của các lớp phức tạp để dự đoán các lớp trong :numref:`sec_nin`. Phát hiện multibox Single-shot sử dụng cùng một kỹ thuật để giảm độ phức tạp của mô hình. 

Cụ thể, lớp dự đoán lớp sử dụng một lớp phức tạp mà không làm thay đổi chiều rộng hoặc chiều cao của bản đồ đối tượng. Bằng cách này, có thể có sự tương ứng một-một giữa các đầu ra và đầu vào ở cùng một kích thước không gian (chiều rộng và chiều cao) của bản đồ tính năng. Cụ thể hơn, các kênh của bản đồ tính năng đầu ra ở bất kỳ vị trí không gian nào ($x$, $y$) đại diện cho các dự đoán lớp cho tất cả các hộp neo tập trung vào ($x$, $y$) của bản đồ tính năng đầu vào. Để tạo ra các dự đoán hợp lệ, phải có $a(q+1)$ kênh đầu ra, trong đó cho cùng một vị trí không gian kênh đầu ra với chỉ số $i(q+1) + j$ đại diện cho dự đoán của lớp $j$ ($0 \leq j \leq q$) cho hộp neo $i$ ($0 \leq i < a$). 

Dưới đây chúng ta xác định một lớp dự đoán lớp như vậy, chỉ định $a$ và $q$ thông qua các đối số `num_anchors` và `num_classes`, tương ứng. Lớp này sử dụng một lớp ghép $3\times3$ với lớp đệm là 1. Chiều rộng và chiều cao của đầu vào và đầu ra của lớp phức tạp này vẫn không thay đổi.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()

def cls_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3,
                     padding=1)
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)
```

### (** Lớp dự đoán hộp giới hộp**)

Thiết kế của lớp dự đoán hộp giới hạn tương tự như của lớp dự đoán lớp. Sự khác biệt duy nhất nằm ở số lượng đầu ra cho mỗi hộp neo: ở đây chúng ta cần dự đoán bốn bù đắp hơn là $q+1$ lớp.

```{.python .input}
def bbox_predictor(num_anchors):
    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)
```

```{.python .input}
#@tab pytorch
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)
```

### [**Concatenating dự đoán cho nhiều quy mô**]

Như chúng tôi đã đề cập, phát hiện multibox single shot sử dụng bản đồ tính năng đa thang để tạo ra các hộp neo và dự đoán các lớp và bù đắp của chúng. Ở các thang đo khác nhau, hình dạng của bản đồ tính năng hoặc số lượng hộp neo tập trung vào cùng một đơn vị có thể khác nhau. Do đó, hình dạng của đầu ra dự đoán ở các thang đo khác nhau có thể khác nhau. 

Trong ví dụ sau, chúng tôi xây dựng các bản đồ tính năng ở hai thang khác nhau, `Y1` và `Y2`, cho cùng một minibatch, trong đó chiều cao và chiều rộng của `Y2` là một nửa so với `Y1`. Hãy để chúng tôi lấy dự đoán lớp học như một ví dụ. Giả sử rằng 5 và 3 hộp neo được tạo ra cho mỗi đơn vị trong `Y1` và `Y2`, tương ứng. Giả sử thêm rằng số lượng lớp đối tượng là 10. Đối với bản đồ tính năng `Y1` và `Y2` số kênh trong đầu ra dự đoán lớp là $5\times(10+1)=55$ và $3\times(10+1)=33$, tương ứng, trong đó một trong hai hình dạng đầu ra là (kích thước lô, số kênh, chiều cao, chiều rộng).

```{.python .input}
def forward(x, block):
    block.initialize()
    return block(x)

Y1 = forward(np.zeros((2, 8, 20, 20)), cls_predictor(5, 10))
Y2 = forward(np.zeros((2, 16, 10, 10)), cls_predictor(3, 10))
Y1.shape, Y2.shape
```

```{.python .input}
#@tab pytorch
def forward(x, block):
    return block(x)

Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
Y1.shape, Y2.shape
```

Như chúng ta có thể thấy, ngoại trừ kích thước kích thước lô, ba chiều còn lại đều có kích thước khác nhau. Để nối hai đầu ra dự đoán này để tính toán hiệu quả hơn, chúng tôi sẽ chuyển đổi các tenors này thành một định dạng nhất quán hơn. 

Lưu ý rằng kích thước kênh giữ các dự đoán cho các hộp neo có cùng tâm. Đầu tiên chúng ta di chuyển kích thước này đến trong cùng. Vì kích thước lô vẫn giữ nguyên đối với các thang đo khác nhau, chúng ta có thể chuyển đổi đầu ra dự đoán thành tensor hai chiều với hình dạng (kích thước lô, chiều cao $\times$ chiều rộng $\times$ số kênh). Sau đó, chúng ta có thể nối các đầu ra như vậy ở các quy mô khác nhau dọc theo kích thước 1.

```{.python .input}
def flatten_pred(pred):
    return npx.batch_flatten(pred.transpose(0, 2, 3, 1))

def concat_preds(preds):
    return np.concatenate([flatten_pred(p) for p in preds], axis=1)
```

```{.python .input}
#@tab pytorch
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)
```

Bằng cách này, mặc dù `Y1` và `Y2` có kích thước khác nhau về các kênh, chiều cao và chiều rộng, chúng ta vẫn có thể nối hai đầu ra dự đoán này ở hai thang đo khác nhau cho cùng một minibatch.

```{.python .input}
#@tab all
concat_preds([Y1, Y2]).shape
```

### [** Khối lấy mẫu xuống**]

Để phát hiện các đối tượng ở nhiều thang đo, chúng tôi xác định khối lấy mẫu xuống sau `down_sample_blk` làm giảm một nửa chiều cao và chiều rộng của bản đồ tính năng đầu vào. Trên thực tế, khối này áp dụng thiết kế các khối VGG trong :numref:`subsec_vgg-blocks`. Cụ thể hơn, mỗi khối lấy mẫu downsampling bao gồm hai $3\times3$ lớp ghép với lớp đệm 1 tiếp theo là một lớp tổng hợp tối đa $2\times2$ với sải chân là 2. Như chúng ta đã biết, $3\times3$ lớp ghép với lớp đệm 1 không thay đổi hình dạng của bản đồ tính năng. Tuy nhiên, tổng hợp tối đa $2\times2$ tiếp theo làm giảm chiều cao và chiều rộng của bản đồ tính năng đầu vào xuống một nửa. Đối với cả bản đồ tính năng đầu vào và đầu ra của khối lấy mẫu xuống này, vì $1\times 2+(3-1)+(3-1)=6$, mỗi đơn vị trong đầu ra có một trường tiếp nhận $6\times6$ trên đầu vào. Do đó, khối lấy mẫu xuống mở rộng trường tiếp nhận của mỗi đơn vị trong bản đồ tính năng đầu ra của nó.

```{.python .input}
def down_sample_blk(num_channels):
    blk = nn.Sequential()
    for _ in range(2):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                nn.BatchNorm(in_channels=num_channels),
                nn.Activation('relu'))
    blk.add(nn.MaxPool2D(2))
    return blk
```

```{.python .input}
#@tab pytorch
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)
```

Trong ví dụ sau, khối lấy mẫu downsampling xây dựng của chúng tôi thay đổi số kênh đầu vào và giảm một nửa chiều cao và chiều rộng của bản đồ tính năng đầu vào.

```{.python .input}
forward(np.zeros((2, 3, 20, 20)), down_sample_blk(10)).shape
```

```{.python .input}
#@tab pytorch
forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape
```

### [** Khối mạng cơ sở**]

Khối mạng cơ sở được sử dụng để trích xuất các tính năng từ hình ảnh đầu vào. Để đơn giản, chúng tôi xây dựng một mạng cơ sở nhỏ bao gồm ba khối lấy mẫu giảm gấp đôi số kênh tại mỗi khối. Với hình ảnh đầu vào $256\times256$, khối mạng cơ sở này xuất bản đồ tính năng $32 \times 32$ ($256/2^3=32$).

```{.python .input}
def base_net():
    blk = nn.Sequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
    return blk

forward(np.zeros((2, 3, 256, 256)), base_net()).shape
```

```{.python .input}
#@tab pytorch
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

forward(torch.zeros((2, 3, 256, 256)), base_net()).shape
```

### Mô hình hoàn chỉnh

[**Mô hình phát hiện multibox bắn hoàn chỉnh bao gồm năm khối.**] Các bản đồ tính năng được tạo ra bởi mỗi khối được sử dụng cho cả (i) tạo hộp neo và (ii) dự đoán các lớp và bù đắp của các hộp neo này. Trong số năm khối này, khối đầu tiên là khối mạng cơ sở, khối thứ hai đến thứ tư là các khối lấy mẫu xuống và khối cuối cùng sử dụng tổng hợp tối đa toàn cầu để giảm cả chiều cao và chiều rộng xuống 1. Về mặt kỹ thuật, khối thứ hai đến thứ năm là tất cả các khối bản đồ tính năng đa quy mô trong :numref:`fig_ssd`.

```{.python .input}
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 4:
        blk = nn.GlobalMaxPool2D()
    else:
        blk = down_sample_blk(128)
    return blk
```

```{.python .input}
#@tab pytorch
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk
```

Bây giờ chúng ta [**xác định tuyên truyền chuyển tiếp**] cho mỗi khối. Khác với các tác vụ phân loại hình ảnh, đầu ra ở đây bao gồm (i) bản đồ tính năng CNN `Y`, (ii) hộp neo được tạo ra bằng cách sử dụng `Y` ở thang đo hiện tại, và (iii) lớp và bù dự đoán (dựa trên `Y`) cho các hộp neo này.

```{.python .input}
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

```{.python .input}
#@tab pytorch
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

Nhớ lại rằng trong :numref:`fig_ssd` một khối bản đồ tính năng đa quy mô gần với đỉnh là để phát hiện các đối tượng lớn hơn; do đó, nó cần phải tạo ra các hộp neo lớn hơn. Trong tuyên truyền chuyển tiếp ở trên, tại mỗi khối bản đồ tính năng đa thang chúng ta truyền trong một danh sách hai giá trị tỷ lệ thông qua đối số `sizes` của hàm `multibox_prior` được gọi (được mô tả trong :numref:`sec_anchor`). Sau đây, khoảng cách giữa 0,2 và 1,05 được chia đều thành năm phần để xác định các giá trị quy mô nhỏ hơn tại năm khối: 0.2, 0,37, 0,54, 0,71 và 0,88. Sau đó, các giá trị quy mô lớn hơn của chúng được đưa ra bởi $\sqrt{0.2 \times 0.37} = 0.272$, $\sqrt{0.37 \times 0.54} = 0.447$, v.v. 

[~~Siêu tham số cho mỗi khối ~~]

```{.python .input}
#@tab all
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1
```

Bây giờ chúng ta có thể [** xác định mô hình hoàn chỉnh**] `TinySSD` như sau.

```{.python .input}
class TinySSD(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            # Equivalent to the assignment statement `self.blk_i = get_blk(i)`
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # Here `getattr(self, 'blk_%d' % i)` accesses `self.blk_i`
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = np.concatenate(anchors, axis=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

```{.python .input}
#@tab pytorch
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # Equivalent to the assignment statement `self.blk_i = get_blk(i)`
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # Here `getattr(self, 'blk_%d' % i)` accesses `self.blk_i`
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

Chúng tôi [** tạo ra một ví dụ mô hình và sử dụng nó để thực hiện tuyên truyền chuyển tiếp**] trên một minibatch $256 \times 256$ hình ảnh `X`. 

Như được hiển thị trước đó trong phần này, khối đầu tiên xuất bản đồ tính năng $32 \times 32$. Nhớ lại rằng khối lấy mẫu thứ hai đến thứ tư giảm một nửa chiều cao và chiều rộng và khối thứ năm sử dụng tổng hợp toàn cầu. Vì 4 hộp neo được tạo ra cho mỗi đơn vị dọc theo kích thước không gian của bản đồ tính năng, tại tất cả năm thang đo tổng cộng $(32^2 + 16^2 + 8^2 + 4^2 + 1)\times 4 = 5444$ hộp neo được tạo ra cho mỗi hình ảnh.

```{.python .input}
net = TinySSD(num_classes=1)
net.initialize()
X = np.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

```{.python .input}
#@tab pytorch
net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

## Đào tạo

Bây giờ chúng tôi sẽ giải thích cách huấn luyện mô hình phát hiện multibox shot đơn để phát hiện đối tượng. 

### Đọc tập dữ liệu và khởi tạo mô hình

Để bắt đầu, chúng ta hãy [** đọc dữ liệu phát hiện chuối **] được mô tả trong :numref:`sec_object-detection-dataset`.

```{.python .input}
#@tab all
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)
```

Chỉ có một lớp trong bộ dữ liệu phát hiện chuối. Sau khi xác định mô hình, chúng ta cần phải (** khởi tạo các tham số của nó và xác định thuật toán tối ưu hoạt**).

```{.python .input}
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
net.initialize(init=init.Xavier(), ctx=device)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.2, 'wd': 5e-4})
```

```{.python .input}
#@tab pytorch
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
```

### [**Xác định mất và đánh giá chức năng**]

Phát hiện đối tượng có hai loại tổn thất. Sự mất mát đầu tiên liên quan đến các lớp của hộp neo: tính toán của nó có thể chỉ đơn giản là tái sử dụng hàm mất chéo entropy mà chúng ta sử dụng để phân loại hình ảnh. Tổn thất thứ hai liên quan đến bù đắp các hộp neo tích cực (không nền): đây là một vấn đề hồi quy. Tuy nhiên, đối với vấn đề hồi quy này, ở đây chúng tôi không sử dụng tổn thất bình phương được mô tả trong :numref:`subsec_normal_distribution_and_squared_loss`. Thay vào đó, chúng ta sử dụng mức mất định mức $L_1$, giá trị tuyệt đối của sự khác biệt giữa dự đoán và sự thật mặt đất. Biến mặt nạ `bbox_masks` lọc ra các hộp neo âm và các hộp neo bất hợp pháp (đệm) trong tính toán tổn thất. Cuối cùng, chúng tôi tổng hợp mất lớp hộp neo và tổn thất bù đắp hộp neo để có được chức năng mất mát cho mô hình.

```{.python .input}
cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
bbox_loss = gluon.loss.L1Loss()

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls = cls_loss(cls_preds, cls_labels)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls + bbox
```

```{.python .input}
#@tab pytorch
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox
```

Chúng ta có thể sử dụng độ chính xác để đánh giá kết quả phân loại. Do tổn thất định mức $L_1$ đã sử dụng cho các bù trừ, chúng tôi sử dụng lỗi tuyệt đối *trung bình để đánh giá các hộp giới hạn dự đoán. Những kết quả dự đoán này thu được từ các hộp neo được tạo ra và các bù đắp dự đoán cho chúng.

```{.python .input}
def cls_eval(cls_preds, cls_labels):
    # Because the class prediction results are on the final dimension,
    # `argmax` needs to specify this dimension
    return float((cls_preds.argmax(axis=-1).astype(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((np.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

```{.python .input}
#@tab pytorch
def cls_eval(cls_preds, cls_labels):
    # Because the class prediction results are on the final dimension,
    # `argmax` needs to specify this dimension
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

### [**Đào tạo mô hình**]

Khi đào tạo mô hình, chúng ta cần tạo ra các hộp neo đa quy mô (`anchors`) và dự đoán các lớp của chúng (`cls_preds`) và bù đắp (`bbox_preds`) trong tuyên truyền về phía trước. Sau đó, chúng tôi dán nhãn các lớp (`cls_labels`) và bù đắp (`bbox_labels`) của các hộp neo được tạo ra như vậy dựa trên thông tin nhãn `Y`. Cuối cùng, chúng ta tính toán hàm mất bằng cách sử dụng các giá trị dự đoán và dán nhãn của các lớp và bù đắp. Để triển khai ngắn gọn, đánh giá tập dữ liệu thử nghiệm được bỏ qua ở đây.

```{.python .input}
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
for epoch in range(num_epochs):
    # Sum of training accuracy, no. of examples in sum of training accuracy,
    # Sum of absolute error, no. of examples in sum of absolute error
    metric = d2l.Accumulator(4)
    for features, target in train_iter:
        timer.start()
        X = features.as_in_ctx(device)
        Y = target.as_in_ctx(device)
        with autograd.record():
            # Generate multiscale anchor boxes and predict their classes and
            # offsets
            anchors, cls_preds, bbox_preds = net(X)
            # Label the classes and offsets of these anchor boxes
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors,
                                                                      Y)
            # Calculate the loss function using the predicted and labeled
            # values of the classes and offsets
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                          bbox_masks)
        l.backward()
        trainer.step(batch_size)
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.size,
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.size)
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter._dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

```{.python .input}
#@tab pytorch
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # Sum of training accuracy, no. of examples in sum of training accuracy,
    # Sum of absolute error, no. of examples in sum of absolute error
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # Generate multiscale anchor boxes and predict their classes and
        # offsets
        anchors, cls_preds, bbox_preds = net(X)
        # Label the classes and offsets of these anchor boxes
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # Calculate the loss function using the predicted and labeled values
        # of the classes and offsets
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

## [**Prediction**]

Trong quá trình dự đoán, mục tiêu là phát hiện tất cả các đối tượng quan tâm trên hình ảnh. Dưới đây chúng tôi đọc và thay đổi kích thước một hình ảnh thử nghiệm, chuyển đổi nó thành tensor bốn chiều được yêu cầu bởi các lớp phức tạp.

```{.python .input}
img = image.imread('../img/banana.jpg')
feature = image.imresize(img, 256, 256).astype('float32')
X = np.expand_dims(feature.transpose(2, 0, 1), axis=0)
```

```{.python .input}
#@tab pytorch
X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()
```

Sử dụng hàm `multibox_detection` bên dưới, các hộp giới hạn dự đoán được lấy từ các hộp neo và bù đắp dự đoán của chúng. Sau đó, ức chế không tối đa được sử dụng để loại bỏ các hộp giới hạn dự đoán tương tự.

```{.python .input}
def predict(X):
    anchors, cls_preds, bbox_preds = net(X.as_in_ctx(device))
    cls_probs = npx.softmax(cls_preds).transpose(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

```{.python .input}
#@tab pytorch
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

Cuối cùng, chúng ta [** hiển thị tất cả các hộp giới hạn dự đoán với sự tự tin 0.9 trở lên**] làm đầu ra.

```{.python .input}
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img.asnumpy())
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * np.array((w, h, w, h), ctx=row.ctx)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output, threshold=0.9)
```

```{.python .input}
#@tab pytorch
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), threshold=0.9)
```

## Tóm tắt

* Phát hiện multibox shot đơn là một mô hình phát hiện đối tượng đa quy mô. Thông qua mạng cơ sở của nó và một số khối bản đồ tính năng đa quy mô, phát hiện multibox single shot tạo ra một số lượng khác nhau của các hộp neo với kích thước khác nhau, và phát hiện các đối tượng kích thước khác nhau bằng cách dự đoán các lớp và bù đắp của các hộp neo (do đó các hộp giới hạn).
* Khi đào tạo mô hình phát hiện multibox single shot, chức năng mất mát được tính toán dựa trên các giá trị dự đoán và gắn nhãn của các lớp và bù đắp hộp neo.

## Bài tập

1. Bạn có thể cải thiện phát hiện multibox một lần bằng cách cải thiện chức năng mất mát? Ví dụ, thay thế mất tiêu chuẩn $L_1$ với tổn thất định mức $L_1$ trơn tru cho các bù đắp dự đoán. Hàm mất này sử dụng một hàm vuông khoảng 0 cho độ mịn, được điều khiển bởi siêu tham số $\sigma$:

$$
f(x) =
    \begin{cases}
    (\sigma x)^2/2,& \text{if }|x| < 1/\sigma^2\\
    |x|-0.5/\sigma^2,& \text{otherwise}
    \end{cases}
$$

Khi $\sigma$ rất lớn, tổn thất này tương tự như mức mất định mức $L_1$. Khi giá trị của nó nhỏ hơn, chức năng mất mát mượt mà hơn.

```{.python .input}
sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = np.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = npx.smooth_l1(x, scalar=s)
    d2l.plt.plot(x.asnumpy(), y.asnumpy(), l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
def smooth_l1(data, scalar):
    out = []
    for i in data:
        if abs(i) < 1 / (scalar ** 2):
            out.append(((scalar * i) ** 2) / 2)
        else:
            out.append(abs(i) - 0.5 / (scalar ** 2))
    return torch.tensor(out)

sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = torch.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = smooth_l1(x, scalar=s)
    d2l.plt.plot(x, y, l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

Bên cạnh đó, trong thí nghiệm, chúng tôi đã sử dụng mất chéo entropy cho dự đoán lớp: biểu thị bằng $p_j$ xác suất dự đoán cho lớp chân lý mặt đất $j$, tổn thất chéo entropy là $-\log p_j$. Chúng ta cũng có thể sử dụng tổn thất tiêu cự :cite:`Lin.Goyal.Girshick.ea.2017`: cho các siêu tham số $\gamma > 0$ và $\alpha > 0$, sự mất mát này được định nghĩa là: 

$$ - \alpha (1-p_j)^{\gamma} \log p_j.$$

Như chúng ta có thể thấy, tăng $\gamma$ có thể làm giảm hiệu quả tổn thất tương đối cho các ví dụ được phân loại tốt (ví dụ, $p_j > 0.5$) để đào tạo có thể tập trung nhiều hơn vào những ví dụ khó khăn bị phân loại sai.

```{.python .input}
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * np.log(x)

x = np.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x.asnumpy(), focal_loss(gamma, x).asnumpy(), l,
                     label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * torch.log(x)

x = torch.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x, focal_loss(gamma, x), l, label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

2. Do hạn chế về không gian, chúng tôi đã bỏ qua một số chi tiết triển khai của mô hình phát hiện multibox một shot trong phần này. Bạn có thể cải thiện thêm mô hình trong các khía cạnh sau:
    1. Khi một đối tượng nhỏ hơn nhiều so với hình ảnh, mô hình có thể thay đổi kích thước hình ảnh đầu vào lớn hơn.
    1. Thường có một số lượng lớn các hộp neo âm. Để làm cho phân phối lớp cân bằng hơn, chúng ta có thể hạ mẫu các hộp neo âm.
    1. Trong chức năng mất, gán các siêu tham số trọng lượng khác nhau cho việc mất lớp và mất bù.
    1. Sử dụng các phương pháp khác để đánh giá mô hình phát hiện đối tượng, chẳng hạn như các phương pháp trong một shot multibox phát hiện giấy :cite:`Liu.Anguelov.Erhan.ea.2016`.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/373)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1604)
:end_tab:
