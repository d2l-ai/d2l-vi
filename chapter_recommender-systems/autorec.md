# AutoRec: Dự đoán xếp hạng với Autoencoders

Mặc dù mô hình factorization ma trận đạt được hiệu suất tốt trên nhiệm vụ dự đoán xếp hạng, về cơ bản nó là một mô hình tuyến tính. Do đó, các mô hình như vậy không có khả năng nắm bắt các mối quan hệ phi tuyến và phức tạp phức tạp có thể dự đoán được sở thích của người dùng. Trong phần này, chúng tôi giới thiệu một mô hình lọc hợp tác mạng thần kinh phi tuyến, AutoRec :cite:`Sedhain.Menon.Sanner.ea.2015`. Nó xác định bộ lọc hợp tác (CF) với kiến trúc tự động mã hóa và nhằm mục đích tích hợp các biến đổi phi tuyến vào CF trên cơ sở phản hồi rõ ràng. Các mạng thần kinh đã được chứng minh là có khả năng xấp xỉ bất kỳ chức năng liên tục nào, làm cho nó phù hợp để giải quyết giới hạn của factorization ma trận và làm phong phú thêm tính biểu cảm của factorization ma trận. 

Một mặt, AutoRec có cấu trúc tương tự như một bộ mã hóa tự động bao gồm một lớp đầu vào, một lớp ẩn và một lớp tái thiết (đầu ra). Một bộ mã hóa tự động là một mạng thần kinh học cách sao chép đầu vào của nó vào đầu ra của nó để mã hóa các đầu vào vào các biểu diễn ẩn (và thường là chiều thấp). Trong AutoRec, thay vì nhúng rõ ràng người dùng/mục vào không gian chiều thấp, nó sử dụng cột/hàng của ma trận tương tác làm đầu vào, sau đó tái tạo lại ma trận tương tác trong lớp đầu ra. 

Mặt khác, AutoRec khác với một autoencoder truyền thống: thay vì học các biểu diễn ẩn, AutoRec tập trung vào việc học/tái tạo lại lớp đầu ra. Nó sử dụng ma trận tương tác quan sát một phần làm đầu vào, nhằm tái tạo lại một ma trận đánh giá đã hoàn thành. Trong khi đó, các mục còn thiếu của đầu vào được điền vào lớp đầu ra thông qua tái thiết cho mục đích khuyến nghị.  

Có hai biến thể của AutoRec: dựa trên người dùng và dựa trên mặt hàng. Đối với ngắn gọn, ở đây chúng tôi chỉ giới thiệu AutoRec dựa trên mặt hàng. AutoRec dựa trên người dùng có thể được bắt nguồn cho phù hợp. 

## Mô hình

Hãy để $\mathbf{R}_{*i}$ biểu thị cột $i^\mathrm{th}$ của ma trận xếp hạng, trong đó xếp hạng không xác định được đặt thành số không theo mặc định. Kiến trúc thần kinh được định nghĩa là: 

$$
h(\mathbf{R}_{*i}) = f(\mathbf{W} \cdot g(\mathbf{V} \mathbf{R}_{*i} + \mu) + b)
$$

trong đó $f(\cdot)$ và $g(\cdot)$ đại diện cho các chức năng kích hoạt, $\mathbf{W}$ và $\mathbf{V}$ là ma trận trọng lượng, $\mu$ và $b$ là những thiên vị. Hãy để $h( \cdot )$ biểu thị toàn bộ mạng của AutoRec. Đầu ra $h(\mathbf{R}_{*i})$ là việc tái thiết cột $i^\mathrm{th}$ của ma trận đánh giá. 

Chức năng mục tiêu sau đây nhằm giảm thiểu lỗi tái thiết: 

$$
\underset{\mathbf{W},\mathbf{V},\mu, b}{\mathrm{argmin}} \sum_{i=1}^M{\parallel \mathbf{R}_{*i} - h(\mathbf{R}_{*i})\parallel_{\mathcal{O}}^2} +\lambda(\| \mathbf{W} \|_F^2 + \| \mathbf{V}\|_F^2)
$$

trong đó $\| \cdot \|_{\mathcal{O}}$ có nghĩa là chỉ có sự đóng góp của xếp hạng quan sát được xem xét, nghĩa là, chỉ có trọng lượng được liên kết với đầu vào quan sát được cập nhật trong quá trình lan truyền ngược.

```{.python .input  n=3}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx

npx.set_np()
```

## Triển khai mô hình

Một autoencoder điển hình bao gồm một bộ mã hóa và một bộ giải mã. Bộ mã hóa chiếu đầu vào để biểu diễn ẩn và bộ giải mã ánh xạ lớp ẩn vào lớp tái thiết. Chúng tôi làm theo thực tế này và tạo bộ mã hóa và bộ giải mã với các lớp dày đặc. Việc kích hoạt bộ mã hóa được đặt thành `sigmoid` theo mặc định và không có kích hoạt nào được áp dụng cho bộ giải mã. Dropout được bao gồm sau khi chuyển đổi mã hóa để giảm quá phù hợp. Các gradient của các đầu vào không quan sát được che giấu để đảm bảo rằng chỉ có xếp hạng quan sát đóng góp vào quá trình học tập mô hình.

```{.python .input  n=2}
class AutoRec(nn.Block):
    def __init__(self, num_hidden, num_users, dropout=0.05):
        super(AutoRec, self).__init__()
        self.encoder = nn.Dense(num_hidden, activation='sigmoid',
                                use_bias=True)
        self.decoder = nn.Dense(num_users, use_bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        hidden = self.dropout(self.encoder(input))
        pred = self.decoder(hidden)
        if autograd.is_training():  # Mask the gradient during training
            return pred * np.sign(input)
        else:
            return pred
```

## Triển khai lại Người đánh giá

Vì đầu vào và đầu ra đã được thay đổi, chúng ta cần thực hiện lại hàm đánh giá, trong khi chúng ta vẫn sử dụng RMSE làm thước đo chính xác.

```{.python .input  n=3}
def evaluator(network, inter_matrix, test_data, devices):
    scores = []
    for values in inter_matrix:
        feat = gluon.utils.split_and_load(values, devices, even_split=False)
        scores.extend([network(i).asnumpy() for i in feat])
    recons = np.array([item for sublist in scores for item in sublist])
    # Calculate the test RMSE
    rmse = np.sqrt(np.sum(np.square(test_data - np.sign(test_data) * recons))
                   / np.sum(np.sign(test_data)))
    return float(rmse)
```

## Đào tạo và đánh giá mô hình

Bây giờ, chúng ta hãy đào tạo và đánh giá AutoRec trên bộ dữ liệu MovieLens. Chúng ta có thể thấy rõ rằng thử nghiệm RMSE thấp hơn mô hình factorization ma trận, xác nhận hiệu quả của các mạng thần kinh trong nhiệm vụ dự đoán xếp hạng.

```{.python .input  n=4}
devices = d2l.try_all_gpus()
# Load the MovieLens 100K dataset
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items)
_, _, _, train_inter_mat = d2l.load_data_ml100k(train_data, num_users,
                                                num_items)
_, _, _, test_inter_mat = d2l.load_data_ml100k(test_data, num_users,
                                               num_items)
train_iter = gluon.data.DataLoader(train_inter_mat, shuffle=True,
                                   last_batch="rollover", batch_size=256,
                                   num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(np.array(train_inter_mat), shuffle=False,
                                  last_batch="keep", batch_size=1024,
                                  num_workers=d2l.get_dataloader_workers())
# Model initialization, training, and evaluation
net = AutoRec(500, num_users)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 25, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
d2l.train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices, evaluator, inter_mat=test_inter_mat)
```

## Tóm tắt

* Chúng ta có thể đóng khung thuật toán factorization ma trận với các bộ mã hóa tự động, trong khi tích hợp các lớp phi tuyến tính và định kỳ bỏ học. 
* Các thí nghiệm trên bộ dữ liệu MovieLens 100K cho thấy AutoRec đạt được hiệu suất vượt trội so với tính toán ma trận.

## Bài tập

* Thay đổi kích thước ẩn của AutoRec để xem tác động của nó đến hiệu suất mô hình.
* Cố gắng thêm nhiều lớp ẩn hơn. Có hữu ích để cải thiện hiệu suất mô hình?
* Bạn có thể tìm thấy một sự kết hợp tốt hơn của bộ giải mã và chức năng kích hoạt mã hóa?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/401)
:end_tab:
