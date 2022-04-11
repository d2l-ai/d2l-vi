# Matrận Factorization

Matrix Factorization :cite:`Koren.Bell.Volinsky.2009` là một thuật toán được thiết lập tốt trong tài liệu hệ thống giới thiệu. Phiên bản đầu tiên của mô hình bao thanh toán ma trận được đề xuất bởi Simon Funk trong một [bài đăng blog] nổi tiếng (https://sifter.org/~simon/journal/20061211.html) trong đó ông mô tả ý tưởng bao thanh toán ma trận tương tác. Sau đó nó trở nên nổi tiếng rộng rãi do cuộc thi Netflix được tổ chức vào năm 2006. Vào thời điểm đó, Netflix, một công ty phát trực tuyến truyền thông và cho thuê video, đã công bố một cuộc thi để cải thiện hiệu suất hệ thống giới thiệu của mình. Đội ngũ tốt nhất có thể cải thiện trên đường cơ sở Netflix, tức là Cinematch), 10 phần trăm sẽ giành được giải thưởng một triệu USD. Như vậy, cuộc thi này đã thu hút rất nhiều sự chú ý đến lĩnh vực nghiên cứu hệ thống giới thiệu. Sau đó, giải thưởng lớn đã giành được bởi đội Pragmatic Chaos của BellKor, một nhóm kết hợp của BellKor, Lý thuyết thực dụng và BigChaos (bạn không cần phải lo lắng về các thuật toán này bây giờ). Mặc dù điểm cuối cùng là kết quả của một giải pháp ensemble (tức là sự kết hợp của nhiều thuật toán), thuật toán factorization ma trận đóng một vai trò quan trọng trong sự pha trộn cuối cùng. Báo cáo kỹ thuật của giải pháp Netflix Grand Prize :cite:`Toscher.Jahrer.Bell.2009` cung cấp một giới thiệu chi tiết về mô hình được thông qua. Trong phần này, chúng ta sẽ đi sâu vào các chi tiết của mô hình factorization ma trận và việc thực hiện nó. 

## Mô hình Factorization Matrix

Matrận factorization là một lớp các mô hình lọc hợp tác. Cụ thể, mô hình factorizes ma trận tương tác mục người dùng (ví dụ, ma trận đánh giá) vào tích của hai ma trận cấp thấp hơn, nắm bắt cấu trúc cấp thấp của các tương tác mục người dùng. 

Hãy để $\mathbf{R} \in \mathbb{R}^{m \times n}$ biểu thị ma trận tương tác với $m$ người dùng và $n$ mục, và các giá trị của $\mathbf{R}$ đại diện cho xếp hạng rõ ràng. Tương tác mục người dùng sẽ được bao thanh toán thành ma trận tiềm ẩn người dùng $\mathbf{P} \in \mathbb{R}^{m \times k}$ và một ma trận tiềm ẩn mục $\mathbf{Q} \in \mathbb{R}^{n \times k}$, trong đó $k \ll m, n$, là kích thước yếu tố tiềm ẩn. Hãy để $\mathbf{p}_u$ biểu thị hàng $u^\mathrm{th}$ của $\mathbf{P}$ và $\mathbf{q}_i$ biểu thị hàng $i^\mathrm{th}$ của $\mathbf{Q}$. Đối với một mục nhất định $i$, các yếu tố của $\mathbf{q}_i$ đo mức độ mà vật phẩm sở hữu những đặc điểm đó như thể loại và ngôn ngữ của một bộ phim. Đối với một người dùng nhất định $u$, các yếu tố của $\mathbf{p}_u$ đo mức độ quan tâm mà người dùng có trong các đặc điểm tương ứng của các mặt hàng. Những yếu tố tiềm ẩn này có thể đo lường các kích thước rõ ràng như đã đề cập trong các ví dụ đó hoặc hoàn toàn không thể hiểu được. Các xếp hạng dự đoán có thể được ước tính bởi 

$$\hat{\mathbf{R}} = \mathbf{PQ}^\top$$

trong đó $\hat{\mathbf{R}}\in \mathbb{R}^{m \times n}$ là ma trận đánh giá dự đoán có hình dạng giống như $\mathbf{R}$. Một vấn đề lớn của quy tắc dự đoán này là thành kiến của người dùng/mục không thể được mô hình hóa. Ví dụ, một số người dùng có xu hướng đưa ra xếp hạng cao hơn hoặc một số mặt hàng luôn nhận được xếp hạng thấp hơn do chất lượng kém hơn. Những thành kiến này là phổ biến trong các ứng dụng trong thế giới thực. Để nắm bắt những thành kiến này, các thuật ngữ thiên vị cụ thể và mục cụ thể của người dùng được giới thiệu. Cụ thể, người dùng đánh giá dự đoán $u$ cung cấp cho mục $i$ được tính bằng 

$$
\hat{\mathbf{R}}_{ui} = \mathbf{p}_u\mathbf{q}^\top_i + b_u + b_i
$$

Sau đó, chúng tôi đào tạo mô hình factorization ma trận bằng cách giảm thiểu sai số bình phương trung bình giữa điểm đánh giá dự đoán và điểm đánh giá thực. Hàm mục tiêu được định nghĩa như sau: 

$$
\underset{\mathbf{P}, \mathbf{Q}, b}{\mathrm{argmin}} \sum_{(u, i) \in \mathcal{K}} \| \mathbf{R}_{ui} -
\hat{\mathbf{R}}_{ui} \|^2 + \lambda (\| \mathbf{P} \|^2_F + \| \mathbf{Q}
\|^2_F + b_u^2 + b_i^2 )
$$

trong đó $\lambda$ biểu thị tỷ lệ đều đặn. Thuật ngữ chính quy hóa $\ lambda (\ |\ mathbf {P}\ |^2_F +\ |\ mathbf {Q}\ |^2_F + b_u^2 + b_i^2) $ is used to avoid over-fitting by penalizing the magnitude of the parameters. The $ (u, i) $ pairs for which $\ mathbf {R} _ {ui} $ được biết đến được lưu trữ trong bộ 732 293616. Các tham số mô hình có thể được học với một thuật toán tối ưu hóa, chẳng hạn như Stochastic Gradient Descent và Adam. 

Một minh họa trực quan của mô hình factorization ma trận được hiển thị dưới đây: 

![Illustration of matrix factorization model](../img/rec-mf.svg)

Trong phần còn lại của phần này, chúng tôi sẽ giải thích việc triển khai tính toán ma trận và đào tạo mô hình trên bộ dữ liệu MovieLens.

```{.python .input  n=2}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
npx.set_np()
```

## Thực hiện mô hình

Đầu tiên, chúng tôi thực hiện mô hình factorization ma trận được mô tả ở trên. Các yếu tố tiềm ẩn của người dùng và mục có thể được tạo ra với `nn.Embedding`. `input_dim` là số mục/người dùng và (`output_dim`) là kích thước của các yếu tố tiềm ẩn ($k$). Chúng tôi cũng có thể sử dụng `nn.Embedding` để tạo thành kiến người dùng/mục bằng cách đặt `output_dim` thành một. Trong chức năng `forward`, id người dùng và mục được sử dụng để tra cứu các embeddings.

```{.python .input  n=4}
class MF(nn.Block):
    def __init__(self, num_factors, num_users, num_items, **kwargs):
        super(MF, self).__init__(**kwargs)
        self.P = nn.Embedding(input_dim=num_users, output_dim=num_factors)
        self.Q = nn.Embedding(input_dim=num_items, output_dim=num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_id, item_id):
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        b_u = self.user_bias(user_id)
        b_i = self.item_bias(item_id)
        outputs = (P_u * Q_i).sum(axis=1) + np.squeeze(b_u) + np.squeeze(b_i)
        return outputs.flatten()
```

## Các biện pháp đánh giá

Sau đó, chúng tôi thực hiện đo RMSE (lỗi gốc có nghĩa là vuông), thường được sử dụng để đo lường sự khác biệt giữa điểm xếp hạng được dự đoán bởi mô hình và xếp hạng thực sự quan sát (sự thật mặt đất) :cite:`Gunawardana.Shani.2015`. RMSE được định nghĩa là: 

$$
\mathrm{RMSE} = \sqrt{\frac{1}{|\mathcal{T}|}\sum_{(u, i) \in \mathcal{T}}(\mathbf{R}_{ui} -\hat{\mathbf{R}}_{ui})^2}
$$

trong đó $\mathcal{T}$ là bộ bao gồm các cặp người dùng và các mặt hàng mà bạn muốn đánh giá. $|\mathcal{T}|$ là kích thước của bộ này. Chúng ta có thể sử dụng hàm RMSE được cung cấp bởi `mx.metric`.

```{.python .input  n=3}
def evaluator(net, test_iter, devices):
    rmse = mx.metric.RMSE()  # Get the RMSE
    rmse_list = []
    for idx, (users, items, ratings) in enumerate(test_iter):
        u = gluon.utils.split_and_load(users, devices, even_split=False)
        i = gluon.utils.split_and_load(items, devices, even_split=False)
        r_ui = gluon.utils.split_and_load(ratings, devices, even_split=False)
        r_hat = [net(u, i) for u, i in zip(u, i)]
        rmse.update(labels=r_ui, preds=r_hat)
        rmse_list.append(rmse.get()[1])
    return float(np.mean(np.array(rmse_list)))
```

## Đào tạo và đánh giá mô hình

Trong chức năng đào tạo, chúng tôi áp dụng giảm $L_2$ với giảm cân. Cơ chế phân rã trọng lượng có tác dụng tương tự như quy định hóa $L_2$.

```{.python .input  n=4}
#@save
def train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices=d2l.try_all_gpus(), evaluator=None,
                        **kwargs):
    timer = d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 2],
                            legend=['train loss', 'test RMSE'])
    for epoch in range(num_epochs):
        metric, l = d2l.Accumulator(3), 0.
        for i, values in enumerate(train_iter):
            timer.start()
            input_data = []
            values = values if isinstance(values, list) else [values]
            for v in values:
                input_data.append(gluon.utils.split_and_load(v, devices))
            train_feat = input_data[0:-1] if len(values) > 1 else input_data
            train_label = input_data[-1]
            with autograd.record():
                preds = [net(*t) for t in zip(*train_feat)]
                ls = [loss(p, s) for p, s in zip(preds, train_label)]
            [l.backward() for l in ls]
            l += sum([l.asnumpy() for l in ls]).mean() / len(devices)
            trainer.step(values[0].shape[0])
            metric.add(l, values[0].shape[0], values[0].size)
            timer.stop()
        if len(kwargs) > 0:  # It will be used in section AutoRec
            test_rmse = evaluator(net, test_iter, kwargs['inter_mat'],
                                  devices)
        else:
            test_rmse = evaluator(net, test_iter, devices)
        train_l = l / (i + 1)
        animator.add(epoch + 1, (train_l, test_rmse))
    print(f'train loss {metric[0] / metric[1]:.3f}, '
          f'test RMSE {test_rmse:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(devices)}')
```

Cuối cùng, chúng ta hãy đặt tất cả mọi thứ lại với nhau và đào tạo mô hình. Ở đây, chúng ta đặt kích thước yếu tố tiềm ẩn là 30.

```{.python .input  n=5}
devices = d2l.try_all_gpus()
num_users, num_items, train_iter, test_iter = d2l.split_and_load_ml100k(
    test_ratio=0.1, batch_size=512)
net = MF(30, num_users, num_items)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 20, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                    devices, evaluator)
```

Dưới đây, chúng tôi sử dụng mô hình được đào tạo để dự đoán xếp hạng mà người dùng (ID 20) có thể cung cấp cho một mục (ID 30).

```{.python .input  n=6}
scores = net(np.array([20], dtype='int', ctx=devices[0]),
             np.array([30], dtype='int', ctx=devices[0]))
scores
```

## Tóm tắt

* Mô hình factorization ma trận được sử dụng rộng rãi trong các hệ thống recommender. Nó có thể được sử dụng để dự đoán xếp hạng mà người dùng có thể cung cấp cho một mục.
* Chúng tôi có thể thực hiện và đào tạo ma trận factorization cho các hệ thống recommender.

## Bài tập

* Thay đổi kích thước của các yếu tố tiềm ẩn. Làm thế nào để kích thước của các yếu tố tiềm ẩn ảnh hưởng đến hiệu suất mô hình?
* Hãy thử các trình tối ưu hóa khác nhau, tỷ lệ học tập và tỷ lệ phân rã cân nặng.
* Kiểm tra điểm xếp hạng dự đoán của những người dùng khác cho một bộ phim cụ thể.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/400)
:end_tab:
