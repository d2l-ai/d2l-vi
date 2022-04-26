# Máy Factorization

Máy factorization (FM) :cite:`Rendle.2010`, được đề xuất bởi Steffen Rendle năm 2010, là một thuật toán được giám sát có thể được sử dụng để phân loại, hồi quy, và xếp hạng nhiệm vụ. Nó nhanh chóng chú ý và trở thành một phương pháp phổ biến và có tác động để đưa ra dự đoán và khuyến nghị. Đặc biệt, nó là một khái quát hóa của mô hình hồi quy tuyến tính và mô hình factorization ma trận. Hơn nữa, nó gợi nhớ đến các máy vector hỗ trợ với một hạt nhân đa thức. Điểm mạnh của máy factorization so với hồi quy tuyến tính và tính toán ma trận là: (1) nó có thể mô hình tương tác biến $\chi$-chiều, trong đó $\chi$ là số thứ tự đa thức và thường được đặt thành hai. (2) Một thuật toán tối ưu hóa nhanh liên quan đến máy factorization có thể làm giảm thời gian tính toán đa thức đến độ phức tạp tuyến tính, làm cho nó cực kỳ hiệu quả đặc biệt là đối với các đầu vào thưa thớt chiều cao. Vì những lý do này, máy factorization được sử dụng rộng rãi trong các khuyến nghị quảng cáo và sản phẩm hiện đại. Các chi tiết kỹ thuật và triển khai được mô tả dưới đây. 

## 2-Way Factorization Máy móc

Chính thức, hãy để $x \in \mathbb{R}^d$ biểu thị các vectơ tính năng của một mẫu và $y$ biểu thị nhãn tương ứng có thể là nhãn có giá trị thực hoặc nhãn lớp như lớp nhị phân “click/non-click”. Mô hình cho một máy factorization độ hai được định nghĩa là: 

$$
\hat{y}(x) = \mathbf{w}_0 + \sum_{i=1}^d \mathbf{w}_i x_i + \sum_{i=1}^d\sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j
$$

trong đó $\mathbf{w}_0 \in \mathbb{R}$ là thiên vị toàn cầu; $\mathbf{w} \in \mathbb{R}^d$ biểu thị trọng lượng của biến thứ i; $\mathbf{V} \in \mathbb{R}^{d\times k}$ đại diện cho các tính năng nhúng; $\mathbf{v}_i$ đại diện cho hàng $i^\mathrm{th}$ của $\mathbf{V}$; $k$ là chiều của các yếu tố tiềm ẩn; $\langle\cdot, \cdot \rangle$ là tích chấm của hai vectơ. 3620 mô hình sự tương tác giữa tính năng $i^\mathrm{th}$ và $j^\mathrm{th}$. Một số tương tác tính năng có thể dễ hiểu để chúng có thể được thiết kế bởi các chuyên gia. Tuy nhiên, hầu hết các tương tác tính năng khác đều ẩn trong dữ liệu và khó xác định. Vì vậy, mô hình hóa tính năng tương tác tự động có thể làm giảm đáng kể những nỗ lực trong kỹ thuật tính năng. Rõ ràng là hai thuật ngữ đầu tiên tương ứng với mô hình hồi quy tuyến tính và thuật ngữ cuối cùng là một phần mở rộng của mô hình factorization ma trận. Nếu tính năng $i$ đại diện cho một mục và tính năng $j$ đại diện cho người dùng, thuật ngữ thứ ba chính xác là sản phẩm chấm giữa nhúng người dùng và mục. Điều đáng chú ý là FM cũng có thể khái quát hóa với các đơn đặt hàng cao hơn (độ> 2). Tuy nhiên, sự ổn định số có thể làm suy yếu sự tổng quát hóa. 

## Một tiêu chí tối ưu hóa hiệu quả

Tối ưu hóa các máy factorization trong một phương pháp thẳng về phía trước dẫn đến độ phức tạp của $\mathcal{O}(kd^2)$ vì tất cả các tương tác cặp yêu cầu phải được tính toán. Để giải quyết vấn đề kém hiệu quả này, chúng ta có thể tổ chức lại nhiệm kỳ thứ ba của FM có thể làm giảm đáng kể chi phí tính toán, dẫn đến độ phức tạp thời gian tuyến tính ($\mathcal{O}(kd)$). Việc cải cách thuật ngữ tương tác cặp như sau: 

$$
\begin{aligned}
&\sum_{i=1}^d \sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j \\
 &= \frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d\langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j - \frac{1}{2}\sum_{i=1}^d \langle\mathbf{v}_i, \mathbf{v}_i\rangle x_i x_i \\
 &= \frac{1}{2} \big (\sum_{i=1}^d \sum_{j=1}^d \sum_{l=1}^k\mathbf{v}_{i, l} \mathbf{v}_{j, l} x_i x_j - \sum_{i=1}^d \sum_{l=1}^k \mathbf{v}_{i, l} \mathbf{v}_{i, l} x_i x_i \big)\\
 &=  \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i) (\sum_{j=1}^d \mathbf{v}_{j, l}x_j) - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2 \big ) \\
 &= \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i)^2 - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2)
 \end{aligned}
$$

Với sự cải cách này, độ phức tạp của mô hình được giảm đáng kể. Hơn nữa, đối với các tính năng thưa thớt, chỉ các phần tử không phải bằng không cần được tính toán sao cho độ phức tạp tổng thể là tuyến tính với số lượng các tính năng không phải bằng không. 

Để tìm hiểu mô hình FM, chúng ta có thể sử dụng mất MSE cho nhiệm vụ hồi quy, mất chéo entropy cho các nhiệm vụ phân loại và mất BPR cho nhiệm vụ xếp hạng. Các trình tối ưu hóa tiêu chuẩn như gốc gradient ngẫu nhiên và Adam là khả thi để tối ưu hóa.

```{.python .input  n=2}
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

## Mô hình triển khai Mã sau đây thực hiện các máy tính factorization. Rõ ràng là thấy rằng FM bao gồm một khối hồi quy tuyến tính và một khối tương tác tính năng hiệu quả. Chúng tôi áp dụng một hàm sigmoid trên điểm số cuối cùng vì chúng tôi coi dự đoán CTR như một nhiệm vụ phân loại.

```{.python .input  n=2}
class FM(nn.Block):
    def __init__(self, field_dims, num_factors):
        super(FM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)

    def forward(self, x):
        square_of_sum = np.sum(self.embedding(x), axis=1) ** 2
        sum_of_square = np.sum(self.embedding(x) ** 2, axis=1)
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True)
        x = npx.sigmoid(x)
        return x
```

## Tải tập dữ liệu quảng cáo Chúng tôi sử dụng trình bao bọc dữ liệu CTR từ phần cuối để tải tập dữ liệu quảng cáo trực tuyến.

```{.python .input  n=3}
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
```

## Đào tạo mô hình Sau đó, chúng tôi đào tạo mô hình. Tỷ lệ học tập được đặt thành 0,02 và kích thước nhúng được đặt thành 20 theo mặc định. Trình tối ưu hóa `Adam` và tổn thất `SigmoidBinaryCrossEntropyLoss` được sử dụng để đào tạo mô hình.

```{.python .input  n=5}
devices = d2l.try_all_gpus()
net = FM(train_data.field_dims, num_factors=20)
net.initialize(init.Xavier(), ctx=devices)
lr, num_epochs, optimizer = 0.02, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## Tóm tắt

* FM là một khuôn khổ chung có thể được áp dụng trên nhiều nhiệm vụ khác nhau như hồi quy, phân loại và xếp hạng.
* Tương tự/vượt qua tính năng rất quan trọng đối với các nhiệm vụ dự đoán và tương tác 2 chiều có thể được mô hình hóa hiệu quả với FM.

## Bài tập

* Bạn có thể kiểm tra FM trên các tập dữ liệu khác như Avazu, MovieLens và Criteo bộ dữ liệu không?
* Thay đổi kích thước nhúng để kiểm tra tác động của nó đối với hiệu suất, bạn có thể quan sát một mô hình tương tự như của factorization ma trận?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/406)
:end_tab:
