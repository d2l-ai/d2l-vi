# Tinh chỉnh
:label:`sec_fine_tuning`

Trong các chương trước đó, chúng tôi đã thảo luận về cách đào tạo các mô hình trên tập dữ liệu đào tạo Fashion-MNIST chỉ với 60000 hình ảnh. Chúng tôi cũng mô tả ImageNet, tập dữ liệu hình ảnh quy mô lớn được sử dụng rộng rãi nhất trong học viện, có hơn 10 triệu hình ảnh và 1000 đối tượng. Tuy nhiên, kích thước của tập dữ liệu mà chúng ta thường gặp phải là giữa các tập dữ liệu của hai tập dữ liệu. 

Giả sử rằng chúng tôi muốn nhận ra các loại ghế khác nhau từ hình ảnh, và sau đó khuyên bạn nên mua liên kết cho người dùng. Một phương pháp có thể là đầu tiên xác định 100 ghế thông thường, chụp 1000 hình ảnh của các góc khác nhau cho mỗi ghế, và sau đó đào tạo một mô hình phân loại trên tập dữ liệu hình ảnh thu thập được. Mặc dù bộ dữ liệu ghế này có thể lớn hơn tập dữ liệu Fashion-MNIST, số lượng ví dụ vẫn ít hơn một phần mười trong ImageNet. Điều này có thể dẫn đến quá mức các mô hình phức tạp phù hợp với ImageNet trên bộ dữ liệu ghế này. Bên cạnh đó, do số lượng ví dụ đào tạo hạn chế, độ chính xác của mô hình được đào tạo có thể không đáp ứng các yêu cầu thực tế. 

Để giải quyết các vấn đề trên, một giải pháp rõ ràng là thu thập thêm dữ liệu. Tuy nhiên, việc thu thập và ghi nhãn dữ liệu có thể mất rất nhiều thời gian và tiền bạc. Ví dụ, để thu thập tập dữ liệu ImageNet, các nhà nghiên cứu đã chi hàng triệu đô la từ tài trợ nghiên cứu. Mặc dù chi phí thu thập dữ liệu hiện tại đã giảm đáng kể, chi phí này vẫn không thể bỏ qua. 

Một giải pháp khác là áp dụng *transfer learning* để chuyển các kiến thức học được từ *sourcedataset* sang tập dữ liệu đích *. Ví dụ, mặc dù hầu hết các hình ảnh trong tập dữ liệu ImageNet không liên quan gì đến ghế, mô hình được đào tạo trên bộ dữ liệu này có thể trích xuất các tính năng hình ảnh chung hơn, có thể giúp xác định các cạnh, kết cấu, hình dạng và bố cục đối tượng. Những tính năng tương tự này cũng có thể có hiệu quả để nhận ra ghế. 

## Các bước

Trong phần này, chúng tôi sẽ giới thiệu một kỹ thuật phổ biến trong chuyển learning: *fine-tuning*. As shown in :numref:`fig_finetune`, tinh chỉnh bao gồm bốn bước sau: 

1. Pretrain một mô hình mạng thần kinh, tức là mô hình *source*, trên một tập dữ liệu nguồn (ví dụ, tập dữ liệu ImageNet).
1. Tạo một mô hình mạng thần kinh mới, tức là mô hình mục tiêu *. Điều này sao chép tất cả các thiết kế mô hình và các tham số của chúng trên mô hình nguồn ngoại trừ lớp đầu ra. Chúng tôi giả định rằng các tham số mô hình này chứa kiến thức học được từ tập dữ liệu nguồn và kiến thức này cũng sẽ được áp dụng cho tập dữ liệu đích. Chúng tôi cũng giả định rằng lớp đầu ra của mô hình nguồn có liên quan chặt chẽ đến các nhãn của tập dữ liệu nguồn; do đó nó không được sử dụng trong mô hình đích.
1. Thêm một lớp đầu ra vào mô hình đích, có số lượng đầu ra là số lượng danh mục trong tập dữ liệu đích. Sau đó khởi tạo ngẫu nhiên các tham số mô hình của lớp này.
1. Đào tạo mô hình mục tiêu trên tập dữ liệu mục tiêu, chẳng hạn như bộ dữ liệu ghế. Lớp đầu ra sẽ được đào tạo từ đầu, trong khi các tham số của tất cả các lớp khác được tinh chỉnh dựa trên các thông số của mô hình nguồn.

![Fine tuning.](../img/finetune.svg)
:label:`fig_finetune`

Khi các bộ dữ liệu đích nhỏ hơn nhiều so với các bộ dữ liệu nguồn, tinh chỉnh giúp cải thiện khả năng tổng quát hóa của các mô hình. 

## Nhận dạng chó nóng

Hãy để chúng tôi chứng minh tinh chỉnh thông qua một trường hợp cụ thể: nhận dạng chó nóng. Chúng tôi sẽ tinh chỉnh một mô hình ResNet trên một tập dữ liệu nhỏ, được đào tạo trước trên tập dữ liệu ImageNet. Tập dữ liệu nhỏ này bao gồm hàng ngàn hình ảnh có và không có xúc xích. Chúng tôi sẽ sử dụng mô hình tinh chỉnh để nhận ra xúc xích từ hình ảnh.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from torch import nn
import torch
import torchvision
import os
```

### Đọc tập dữ liệu

[**Bộ dữ liệu hot dog chúng tôi sử dụng được lấy từ hình ảnh trực tuyến**]. Tập dữ liệu này bao gồm 1400 hình ảnh đẳng cấp tích cực chứa xúc xích, và càng nhiều hình ảnh đẳng cấp tiêu cực chứa các loại thực phẩm khác. 1000 hình ảnh của cả hai lớp được sử dụng để đào tạo và phần còn lại là để thử nghiệm. 

Sau khi giải nén tập dữ liệu đã tải xuống, chúng tôi nhận được hai thư mục `hotdog/train` và `hotdog/test`. Cả hai thư mục đều có các thư mục con `hotdog` và `not-hotdog`, trong đó có chứa hình ảnh của lớp tương ứng.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')

data_dir = d2l.download_extract('hotdog')
```

Chúng tôi tạo hai trường hợp để đọc tất cả các tệp hình ảnh trong các tập dữ liệu đào tạo và thử nghiệm, tương ứng.

```{.python .input}
train_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'train'))
test_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'test'))
```

```{.python .input}
#@tab pytorch
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))
```

8 ví dụ tích cực đầu tiên và 8 hình ảnh tiêu cực cuối cùng được hiển thị bên dưới. Như bạn có thể thấy, [** hình ảnh khác nhau về kích thước và tỷ lệ khung hình**].

```{.python .input}
#@tab all
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);
```

Trong quá trình đào tạo, trước tiên chúng ta cắt một vùng ngẫu nhiên có kích thước ngẫu nhiên và tỷ lệ khung hình ngẫu nhiên từ hình ảnh, sau đó mở rộng khu vực này thành hình ảnh đầu vào $224 \times 224$. Trong quá trình thử nghiệm, chúng tôi chia tỷ lệ cả chiều cao và chiều rộng của hình ảnh lên 256 pixel, và sau đó cắt một khu vực trung tâm $224 \times 224$ làm đầu vào. Ngoài ra, đối với ba kênh màu RGB (đỏ, xanh lá cây và xanh dương), chúng tôi * tiêu chuẩn* giá trị của chúng theo kênh. Cụ thể, giá trị trung bình của một kênh được trừ đi từ mỗi giá trị của kênh đó và sau đó kết quả được chia cho độ lệch chuẩn của kênh đó. 

[~~ Tăng dữ liệu ~~]

```{.python .input}
# Specify the means and standard deviations of the three RGB channels to
# standardize each channel
normalize = gluon.data.vision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomResizedCrop(224),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    normalize])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    normalize])
```

```{.python .input}
#@tab pytorch
# Specify the means and standard deviations of the three RGB channels to
# standardize each channel
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])
```

### [**Xác định và khởi tạo mô hình**]

Chúng tôi sử dụng ResNet-18, được đào tạo trước trên tập dữ liệu ImageNet, làm mô hình nguồn. Ở đây, chúng tôi chỉ định `pretrained=True` để tự động tải xuống các tham số mô hình được đào tạo trước. Nếu mô hình này được sử dụng lần đầu tiên, cần có kết nối Internet để tải xuống.

```{.python .input}
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.resnet18(pretrained=True)
```

:begin_tab:`mxnet`
Ví dụ mô hình nguồn được đào tạo trước chứa hai biến thành viên: `features` và `output`. Cái trước chứa tất cả các lớp của mô hình ngoại trừ lớp đầu ra và lớp thứ hai là lớp đầu ra của mô hình. Mục đích chính của bộ phận này là tạo điều kiện thuận lợi cho việc tinh chỉnh các tham số mô hình của tất cả các lớp nhưng lớp đầu ra. Biến thành viên `output` của mô hình nguồn được hiển thị dưới đây.
:end_tab:

:begin_tab:`pytorch`
Ví dụ mô hình nguồn được đào tạo sẵn chứa một số lớp đối tượng và một lớp đầu ra `fc`. Mục đích chính của bộ phận này là tạo điều kiện thuận lợi cho việc tinh chỉnh các tham số mô hình của tất cả các lớp nhưng lớp đầu ra. Biến thành viên `fc` của mô hình nguồn được đưa ra dưới đây.
:end_tab:

```{.python .input}
pretrained_net.output
```

```{.python .input}
#@tab pytorch
pretrained_net.fc
```

Là một lớp được kết nối hoàn toàn, nó biến đổi các đầu ra tổng hợp trung bình toàn cầu cuối cùng của ResNet thành 1000 đầu ra lớp của tập dữ liệu ImageNet. Sau đó, chúng tôi xây dựng một mạng nơ-ron mới làm mô hình mục tiêu. Nó được định nghĩa theo cách tương tự như mô hình nguồn được đào tạo trước ngoại trừ số lượng đầu ra của nó trong lớp cuối cùng được đặt thành số lớp trong tập dữ liệu đích (chứ không phải 1000). 

Trong đoạn code sau, các tham số mô hình trong các đối tượng biến thành viên của đối tượng mô hình đích finetune_net được khởi tạo thành các tham số mô hình của lớp tương ứng của mô hình nguồn. Vì các thông số mô hình trong các tính năng được đào tạo trước trên tập dữ liệu ImageNet và đủ tốt, nói chung chỉ cần một tốc độ học tập nhỏ là cần thiết để tinh chỉnh các tham số này.  

Các tham số mô hình trong đầu ra biến thành viên được khởi tạo ngẫu nhiên và thường yêu cầu tốc độ học tập lớn hơn để đào tạo từ đầu. Giả sử rằng tốc độ học tập trong trường hợp Trainer là η, chúng ta đặt tốc độ học tập của các tham số mô hình trong đầu ra biến thành viên là 10η trong lần lặp. 

Trong đoạn mã dưới đây, các tham số mô hình trước lớp đầu ra của đối tượng mô hình đích `finetune_net` được khởi tạo thành các tham số mô hình của các lớp tương ứng từ mô hình nguồn. Vì các thông số mô hình này thu được thông qua đào tạo trước trên ImageNet, chúng có hiệu quả. Do đó, chúng ta chỉ có thể sử dụng một tốc độ học tập nhỏ để * tốt-tune* các thông số được đào tạo trước đó. Ngược lại, các tham số mô hình trong lớp đầu ra được khởi tạo ngẫu nhiên và thường yêu cầu một tỷ lệ học tập lớn hơn để được học từ đầu. Hãy để tốc độ học cơ bản là $\eta$, tốc độ học tập là $10\eta$ sẽ được sử dụng để lặp lại các tham số mô hình trong lớp đầu ra.

```{.python .input}
finetune_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier())
# The model parameters in the output layer will be iterated using a learning
# rate ten times greater
finetune_net.output.collect_params().setattr('lr_mult', 10)
```

```{.python .input}
#@tab pytorch
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight);
```

### [**Tinh chỉnh mô hình**]

Đầu tiên, chúng tôi xác định một hàm đào tạo `train_fine_tuning` sử dụng tinh chỉnh để nó có thể được gọi nhiều lần.

```{.python .input}
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5):
    train_iter = gluon.data.DataLoader(
        train_imgs.transform_first(train_augs), batch_size, shuffle=True)
    test_iter = gluon.data.DataLoader(
        test_imgs.transform_first(test_augs), batch_size)
    devices = d2l.try_all_gpus()
    net.collect_params().reset_ctx(devices)
    net.hybridize()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': 0.001})
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
```

```{.python .input}
#@tab pytorch
# If `param_group=True`, the model parameters in the output layer will be
# updated using a learning rate ten times greater
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)    
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
```

Chúng tôi [** đặt tỷ lệ học tập cơ bản thành một giá trị nhỏ**] để * tốt-tune* các thông số mô hình thu được thông qua pretraining. Dựa trên các cài đặt trước đó, chúng tôi sẽ đào tạo các tham số lớp đầu ra của mô hình mục tiêu từ đầu bằng cách sử dụng tốc độ học tập lớn hơn mười lần.

```{.python .input}
train_fine_tuning(finetune_net, 0.01)
```

```{.python .input}
#@tab pytorch
train_fine_tuning(finetune_net, 5e-5)
```

[**Để so sánh, **] chúng ta định nghĩa một mô hình giống hệt nhau, nhưng (** khởi tạo tất cả các tham số mô hình của nó thành các giá trị ngẫu nhiên**). Vì toàn bộ mô hình cần được đào tạo từ đầu, chúng ta có thể sử dụng tốc độ học tập lớn hơn.

```{.python .input}
scratch_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
scratch_net.initialize(init=init.Xavier())
train_fine_tuning(scratch_net, 0.1)
```

```{.python .input}
#@tab pytorch
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)
```

Như chúng ta có thể thấy, mô hình tinh chỉnh có xu hướng hoạt động tốt hơn cho cùng một kỷ nguyên vì các giá trị tham số ban đầu của nó hiệu quả hơn. 

## Tóm tắt

* Chuyển học tập chuyển kiến thức học được từ tập dữ liệu nguồn sang tập dữ liệu đích. Tinh chỉnh là một kỹ thuật phổ biến để học chuyển giao.
* Mô hình đích sao chép tất cả các thiết kế mô hình với các tham số của chúng từ mô hình nguồn ngoại trừ lớp đầu ra và tinh chỉnh các tham số này dựa trên tập dữ liệu đích. Ngược lại, lớp đầu ra của mô hình mục tiêu cần được đào tạo từ đầu.
* Nói chung, các thông số tinh chỉnh sử dụng tốc độ học tập nhỏ hơn, trong khi đào tạo lớp đầu ra từ đầu có thể sử dụng tốc độ học tập lớn hơn.

## Bài tập

1. Tiếp tục tăng tỷ lệ học tập của `finetune_net`. Làm thế nào để độ chính xác của mô hình thay đổi?
2. Điều chỉnh thêm các siêu tham số của `finetune_net` và `scratch_net` trong thí nghiệm so sánh. Họ vẫn khác nhau về độ chính xác?
3. Đặt các tham số trước lớp đầu ra của `finetune_net` cho các tham số của mô hình nguồn và làm * không* cập nhật chúng trong quá trình đào tạo. Làm thế nào để độ chính xác của mô hình thay đổi? Bạn có thể sử dụng mã sau.

```{.python .input}
finetune_net.features.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
for param in finetune_net.parameters():
    param.requires_grad = False
```

4. Trên thực tế, có một lớp “hotdog” trong tập dữ liệu `ImageNet`. Tham số trọng lượng tương ứng của nó trong lớp đầu ra có thể thu được thông qua mã sau. Làm thế nào chúng ta có thể tận dụng thông số trọng lượng này?

```{.python .input}
weight = pretrained_net.output.weight
hotdog_w = np.split(weight.data(), 1000, axis=0)[713]
hotdog_w.shape
```

```{.python .input}
#@tab pytorch
weight = pretrained_net.fc.weight
hotdog_w = torch.split(weight.data, 1, dim=0)[934]
hotdog_w.shape
```

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/368)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1439)
:end_tab:
