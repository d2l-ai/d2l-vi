# Sâu Factorization Máy móc

Học kết hợp tính năng hiệu quả là rất quan trọng đối với sự thành công của nhiệm vụ dự đoán tỷ lệ nhấp qua. Máy factorization mô hình tính năng tương tác trong một mô hình tuyến tính (ví dụ, tương tác song tuyến). Điều này thường không đủ đối với dữ liệu trong thế giới thực, nơi các cấu trúc chéo tính năng vốn có thường rất phức tạp và phi tuyến. Điều tồi tệ hơn, tương tác tính năng thứ hai thường được sử dụng trong các máy factorization trong thực tế. Mô hình hóa mức độ kết hợp tính năng cao hơn với các máy factorization là có thể về mặt lý thuyết nhưng nó thường không được áp dụng do sự bất ổn số và độ phức tạp tính toán cao. 

Một giải pháp hiệu quả là sử dụng mạng thần kinh sâu. Các mạng thần kinh sâu mạnh mẽ trong việc học đại diện tính năng và có tiềm năng học các tương tác tính năng phức tạp. Như vậy, nó là tự nhiên để tích hợp các mạng thần kinh sâu vào các máy factorization. Thêm các lớp chuyển đổi phi tuyến vào các máy factorization cho phép nó khả năng mô hình hóa cả kết hợp tính năng bậc thấp và kết hợp tính năng bậc cao. Hơn nữa, các cấu trúc vốn có phi tuyến tính từ đầu vào cũng có thể được chụp bằng các mạng thần kinh sâu. Trong phần này, chúng tôi sẽ giới thiệu một mô hình đại diện có tên là máy factorization sâu (DeepFM) :cite:`Guo.Tang.Ye.ea.2017` kết hợp FM và mạng thần kinh sâu. 

## Model Architectures

DeepFM bao gồm một thành phần FM và một thành phần sâu được tích hợp trong một cấu trúc song song. Các thành phần FM là giống như các máy factorization 2 chiều được sử dụng để mô hình hóa các tương tác tính năng bậc thấp. Thành phần sâu là một MLP được sử dụng để nắm bắt các tương tác tính năng bậc cao và phi tuyến tính. Hai thành phần này chia sẻ cùng một đầu vào/nhúng và đầu ra của chúng được tóm tắt là dự đoán cuối cùng. Điều đáng để chỉ ra rằng tinh thần của DeepFM giống với kiến trúc Wide\ & Deep có thể nắm bắt cả ghi nhớ và khái quát hóa. Ưu điểm của DeepFM so với mô hình Wide\ & Deep là nó làm giảm nỗ lực của kỹ thuật tính năng thủ công bằng cách xác định các kết hợp tính năng tự động. 

Chúng tôi bỏ qua mô tả của thành phần FM cho ngắn gọn và biểu thị đầu ra là $\hat{y}^{(FM)}$. Độc giả được giới thiệu đến phần cuối cùng để biết thêm chi tiết. Hãy để $\mathbf{e}_i \in \mathbb{R}^{k}$ biểu thị vector tính năng tiềm ẩn của trường $i^\mathrm{th}$. Đầu vào của thành phần sâu là sự nối của các nhúng dày đặc của tất cả các trường được nhìn lên với đầu vào tính năng phân loại thưa thớt, được ký hiệu là: 

$$
\mathbf{z}^{(0)}  = [\mathbf{e}_1, \mathbf{e}_2, ..., \mathbf{e}_f],
$$

trong đó $f$ là số trường. Sau đó, nó được đưa vào mạng thần kinh sau: 

$$
\mathbf{z}^{(l)}  = \alpha(\mathbf{W}^{(l)}\mathbf{z}^{(l-1)} + \mathbf{b}^{(l)}),
$$

trong đó $\alpha$ là chức năng kích hoạt. $\mathbf{W}_{l}$ và $\mathbf{b}_{l}$ là trọng lượng và thiên vị ở lớp $l^\mathrm{th}$. Hãy để $y_{DNN}$ biểu thị đầu ra của dự đoán. Dự đoán cuối cùng của DeepFM là tổng kết các đầu ra từ cả FM và DNN. Vì vậy, chúng tôi có: 

$$
\hat{y} = \sigma(\hat{y}^{(FM)} + \hat{y}^{(DNN)}),
$$

trong đó $\sigma$ là chức năng sigmoid. Kiến trúc của DeepFM được minh họa dưới đây.! [Illustration of the DeepFM model](../img/rec-deepfm.svg) 

Điều đáng chú ý là DeepFM không phải là cách duy nhất để kết hợp các mạng thần kinh sâu với FM. Chúng ta cũng có thể thêm các lớp phi tuyến trên các tương tác tính năng :cite:`He.Chua.2017`.

```{.python .input  n=2}
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

## Implemenation của DeepFM Việc thực hiện DeepFM tương tự như FM. Chúng tôi giữ phần FM không thay đổi và sử dụng khối MLP với `relu` làm chức năng kích hoạt. Dropout cũng được sử dụng để thường xuyên hóa mô hình. Số lượng tế bào thần kinh của MLP có thể được điều chỉnh với siêu tham số `mlp_dims`.

```{.python .input  n=2}
class DeepFM(nn.Block):
    def __init__(self, field_dims, num_factors, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)
        input_dim = self.embed_output_dim = len(field_dims) * num_factors
        self.mlp = nn.Sequential()
        for dim in mlp_dims:
            self.mlp.add(nn.Dense(dim, 'relu', True, in_units=input_dim))
            self.mlp.add(nn.Dropout(rate=drop_rate))
            input_dim = dim
        self.mlp.add(nn.Dense(in_units=input_dim, units=1))

    def forward(self, x):
        embed_x = self.embedding(x)
        square_of_sum = np.sum(embed_x, axis=1) ** 2
        sum_of_square = np.sum(embed_x ** 2, axis=1)
        inputs = np.reshape(embed_x, (-1, self.embed_output_dim))
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True) \
            + self.mlp(inputs)
        x = npx.sigmoid(x)
        return x
```

## Đào tạo và Đánh giá mô hình Quá trình tải dữ liệu giống như của FM. Chúng tôi đặt thành phần MLP của DeepFM thành một mạng dày đặc ba lớp với cấu trúc kim tự tháp (30-20-10). Tất cả các siêu tham số khác vẫn giống như FM.

```{.python .input  n=4}
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
field_dims = train_data.field_dims
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
devices = d2l.try_all_gpus()
net = DeepFM(field_dims, num_factors=10, mlp_dims=[30, 20, 10])
net.initialize(init.Xavier(), ctx=devices)
lr, num_epochs, optimizer = 0.01, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

So với FM, DeepFM hội tụ nhanh hơn và đạt được hiệu suất tốt hơn. 

## Tóm tắt

* Tích hợp các mạng thần kinh với FM cho phép nó mô hình hóa các tương tác phức tạp và bậc cao.
* DeepFM vượt trội hơn FM ban đầu trên tập dữ liệu quảng cáo.

## Bài tập

* Thay đổi cấu trúc của MLP để kiểm tra tác động của nó đến hiệu suất mô hình.
* Thay đổi tập dữ liệu thành Criteo và so sánh nó với mô hình FM gốc.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/407)
:end_tab:
