<!--
# Matrix Factorization
-->

# Phân rã Ma trận


<!--
Matrix Factorization :cite:`Koren.Bell.Volinsky.2009` is a well-established algorithm in the recommender systems literature.
The first version of matrix factorization model is proposed by Simon Funk in a famous [blog post](https://sifter.org/~simon/journal/20061211.html) 
in which he described the idea of factorizing the interaction matrix.
It then became widely known due to the Netflix contest which was held in 2006.
At that time, Netflix, a media-streaming and video-rental company, announced a contest to improve its recommender system performance.
The best team that can improve on the Netflix baseline, i.e., Cinematch), by 10 percent would win a one million USD prize.
As such, this contest attracted a lot of attention to the field of recommender system research.
Subsequently, the grand prize was won by the BellKor's Pragmatic Chaos team, a combined team of BellKor, Pragmatic Theory, and BigChaos (you do not need to worry about these algorithms now).
Although the final score was the result of an ensemble solution (i.e., a combination of many algorithms), 
the matrix factorization algorithm played a critical role in the final blend.
The technical report of the Netflix Grand Prize solution :cite:`Toscher.Jahrer.Bell.2009` provides a detailed introduction to the adopted model.
In this section, we will dive into the details of the matrix factorization model and its implementation.
-->

Phân rã ma trận (*Matrix Factorization*) :cite:`Koren.Bell.Volinsky.2009` là một trong những thuật toán lâu đời trong các tài liệu về hệ thống đề xuất.
Phiên bản đầu tiên của mô hình phân rã ma trận được đề xuất bởi Simon Funk trong một [bài blog nổi tiếng](https://sifter.org/~simon/journal/20061211.html), trong đó anh đã mô tả ý tưởng phân rã ma trận tương tác thành các nhân tử.
Phân rã ma trận sau đó trở nên phổ biến nhờ cuộc thi Netflix tổ chức năm 2006.
Tại thời điểm đó, Netflix, một công ty truyền thông đa phương tiện và cho thuê phim, công bố một cuộc thi nhằm cải thiện hiệu năng hệ thống đề xuất của họ.
Đội xuất sắc nhất giúp cải thiện Netflix ở mức cơ sở (*baseline*) như thuật toán Cinematch lên 10 phần trăm sẽ đoạt giải thưởng là một triệu USD.
Do đó, cuộc thi này thu hút rất nhiều sự chú ý trong ngành nghiên cứu hệ thống đề xuất.
Cuối cùng, giải thưởng chung cuộc đã thuộc về đội Pragmatic Chaos của BellKor, là đội đã kết hợp của BellKor, Pragmatic Theory và BigChaos (hiện tại bạn chưa cần phải quan tâm đến các thuật toán này). 
Dù kết quả cuối cùng là một giải pháp kết hợp (tức phối hợp nhiều thuật toán với nhau), 
thuật toán phân rã ma trận đóng vai trò chủ đạo trong thuật toán kết hợp cuối cùng.
Báo cáo kỹ thuật của Giải thưởng Netflix :cite:`Toscher.Jahrer.Bell.2009` cung cấp giới thiệu chi tiết về mô hình được chấp thuận.
Trong phần này, ta sẽ đi sâu vào chi tiết mô hình phân rã ma trận và cách lập trình nó.


<!--
## The Matrix Factorization Model
-->

## Mô hình Phân rã Ma trận


<!--
Matrix factorization is a class of collaborative filtering models.
Specifically, the model factorizes the user-item interaction matrix (e.g., rating matrix)
into the product of two lower-rank matrices, capturing the low-rank structure of the user-item interactions.
-->

Phân rã ma trận là một lớp trong các mô hình lọc cộng tác.
Cụ thể, mô hình này phân tích ma trận tương tác giữa người dùng - sản phẩm (ví dụ như ma trận đánh giá)
thành tích hai ma trận có hạng thấp hơn, nhằm nắm bắt cấu trúc hạng thấp trong tương tác người dùng - sản phẩm.


<!--
Let $\mathbf{R} \in \mathbb{R}^{m \times n}$ denote the interaction matrix with $m$ users and $n$ items,
and the values of $\mathbf{R}$ represent explicit ratings.
The user-item interaction will be factorized into a user latent matrix $\mathbf{P} \in \mathbb{R}^{m \times k}$ 
and an item latent matrix $\mathbf{Q} \in \mathbb{R}^{n \times k}$, where $k \ll m, n$, is the latent factor size.
Let $\mathbf{p}_u$ denote the $u^\mathrm{th}$ row of $\mathbf{P}$ and $\mathbf{q}_i$ denote the $i^\mathrm{th}$ row of $\mathbf{Q}$.
For a given item $i$, the elements of $\mathbf{q}_i$ measure the extent to which the item possesses those characteristics such as the genres and languages of a movie.
For a given user $u$, the elements of $\mathbf{p}_u$ measure the extent of interest the user has in items' corresponding characteristics.
These latent factors might measure obvious dimensions as mentioned in those examples or are completely uninterpretable.
The predicted ratings can be estimated by
-->

Gọi $\mathbf{R} \in \mathbb{R}^{m \times n}$ ký hiệu ma trận tương tác với $m$ người dùng và $n$ sản phẩm,
và các giá trị $\mathbf{R}$ biểu diễn đánh giá trực tiếp.
Tương tác người dùng - sản phẩm được phân tích thành ma trận người dùng tiềm ẩn $\mathbf{P} \in \mathbb{R}^{m \times k}$
và ma trận sản phẩm tiềm ẩn $\mathbf{Q} \in \mathbb{R}^{n \times k}$, trong đó $k \ll m, n$, là kích thước nhân tố tiềm ẩn.
Gọi $\mathbf{p}_u$ ký hiệu hàng thứ $u$ và của $\mathbf{P}$ và $\mathbf{q}_i$ ký hiệu hàng thứ $i$ của $\mathbf{Q}$. 
Với một sản phẩm $i$ cho trước, các phần tử trong $\mathbf{q}_i$ đo lường mức độ mà sản phẩm đó sở hữu các đặc trưng, ví dụ như thể loại hay ngôn ngữ của một bộ phim.
Với một người dùng $u$ cho trước, các phần tử trong $\mathbf{p}_u$ đo mức độ ưa thích của người dùng này đối với các đặc trưng tương ứng của các sản phẩm.
Các nhân tố tiềm ẩn này có thể là các đặc trưng rõ ràng như đã đề cập trong các ví dụ trên, hoặc hoàn toàn không thể giải thích được.
Đánh giá dự đoán có thể được ước lượng như sau


$$\hat{\mathbf{R}} = \mathbf{PQ}^\top$$


<!--
where $\hat{\mathbf{R}}\in \mathbb{R}^{m \times n}$ is the predicted rating matrix which has the same shape as $\mathbf{R}$.
One major problem of this prediction rule is that users/items biases can not be modeled.
For example, some users tend to give higher ratings or some items always get lower ratings due to poorer quality.
These biases are commonplace in real-world applications.
To capture these biases, user specific and item specific bias terms are introduced.
Specifically, the predicted rating user $u$ gives to item $i$ is calculated by
-->

trong đó $\hat{\mathbf{R}}\in \mathbb{R}^{m \times n}$ là ma trận đánh giá dự đoán và có cùng kích thước với $\mathbf{R}$. 
Một vấn đề lớn của cách dự đoán này là độ chệch (*bias*) của người dùng/sản phẩm không được mô hình hóa.
Ví dụ, một số người dùng có thiên hướng đánh giá cao hơn, hoặc một số sản phẩm luôn bị đánh giá thấp hơn bởi chất lượng kém.
Các độ chệch này là rất phổ biến trong những ứng dụng thực tế.
Để thu được các độ chệch này, số hạng độ chệch riêng biệt cho từng người dùng và sản phẩm được sử dụng.
Cụ thể, đánh giá dự đoán của người dùng $u$ cho sản phẩm $i$ được tính theo công thức


$$
\hat{\mathbf{R}}_{ui} = \mathbf{p}_u\mathbf{q}^\top_i + b_u + b_i
$$


<!--
Then, we train the matrix factorization model by minimizing the mean squared error between predicted rating scores and real rating scores.
The objective function is defined as follows:
-->

Sau đó, ta huấn luyện mô hình phân rã ma trận bằng cách cực tiểu hóa trung bình bình phương sai số giữa đánh giá dự đoán và đánh giá thực.
Hàm mục tiêu được định nghĩa như sau:


$$
\underset{\mathbf{P}, \mathbf{Q}, b}{\mathrm{argmin}} \sum_{(u, i) \in \mathcal{K}} \| \mathbf{R}_{ui} -
\hat{\mathbf{R}}_{ui} \|^2 + \lambda (\| \mathbf{P} \|^2_F + \| \mathbf{Q}
\|^2_F + b_u^2 + b_i^2 )
$$


<!--
where $\lambda$ denotes the regularization rate.
The regularizing term $\lambda (\| \mathbf{P} \|^2_F + \| \mathbf{Q}\|^2_F + b_u^2 + b_i^2 )$ is used to avoid over-fitting by penalizing the magnitude of the parameters.
The $(u, i)$ pairs for which $\mathbf{R}_{ui}$ is known are stored in the set $\mathcal{K}=\{(u, i) \mid \mathbf{R}_{ui} \text{ is known}\}$.
The model parameters can be learned with an optimization algorithm, such as Stochastic Gradient Descent and Adam.
-->

trong đó $\lambda$ là tỷ lệ điều chuẩn. 
Số hạng điều chuẩn $\lambda (\| \mathbf{P} \|^2_F + \| \mathbf{Q}\|^2_F + b_u^2 + b_i^2 )$ được sử dụng để tránh hiện tượng quá khớp bằng cách phạt độ lớn của các tham số.
Cặp $(u, i)$ với $\mathbf{R}_{ui}$ đã biết được lưu trong tập $\mathcal{K}=\{(u, i) \mid \mathbf{R}_{ui} \text{đã biết}\}$.
Các tham số mô hình có thể được học thông qua một thuật toán tối ưu, ví dụ như hạ gradient ngẫu nhiên hay Adam. 


<!--
An intuitive illustration of the matrix factorization model is shown below:
-->

Ảnh minh họa trực quan cho mô hình phân rã ma trận được đưa ra như hình dưới:


<!--
![Illustration of matrix factorization model](../img/rec-mf.svg)
-->

![Minh họa mô hình phân rã ma trận](../img/rec-mf.svg)


<!--
In the rest of this section, we will explain the implementation of matrix factorization and train the model on the MovieLens dataset.
-->

Trong phần còn lại, chúng tôi sẽ giải thích cách lập trình cho phân rã ma trận và huấn luyện mô hình trên tập dữ liệu MovieLens.


```{.python .input  n=2}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
npx.set_np()
```


<!--
## Model Implementation
-->

## Cách lập trình Mô hình


<!--
First, we implement the matrix factorization model described above.
The user and item latent factors can be created with the `nn.Embedding`.
The `input_dim` is the number of items/users and the (`output_dim`) is the dimension of the latent factors ($k$).
We can also use `nn.Embedding` to create the user/item biases by setting the `output_dim` to one.
In the `forward` function, user and item ids are used to look up the embeddings.
-->

Đầu tiên, ta lập trình mô hình phân rã ma trận như mô tả trên.
Các nhân tố tiềm ẩn của người dùng và sản phẩm được tạo bằng `nn.Embedding`.
Tham số `input_dim` là số sản phẩm/người dùng và `output_dim` là kích thước nhân tố tiềm ẩn ($k$).
Ta cũng có thể sử dụng `nn.Embedding` để tạo độ chệch cho người dùng/sản phẩm bằng cách gán `output_dim` bằng một.
Trong hàm `forward`, id người dùng và sản phẩm được sử dụng để truy vấn tới embedding tương ứng.


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


<!--
## Evaluation Measures
-->

## Phương pháp Đánh giá


<!--
We then implement the RMSE (root-mean-square error) measure, which is commonly used to measure the differences between rating scores 
predicted by the model and the actually observed ratings (ground truth) :cite:`Gunawardana.Shani.2015`. RMSE is defined as:
-->

Tiếp theo, ta lập trình phép đo RMSE (*root-mean-square error - căn bậc hai trung bình bình phương sai số*), 
phương pháp này được sử dụng rộng rãi nhằm đo lường sự khác nhau giữa giá trị đánh giá dự đoán và giá trị đánh giá thực tế (nhãn gốc) :cite:`Gunawardana.Shani.2015`.
RMSE được định nghĩa như sau:


$$
\mathrm{RMSE} = \sqrt{\frac{1}{|\mathcal{T}|}\sum_{(u, i) \in \mathcal{T}}(\mathbf{R}_{ui} -\hat{\mathbf{R}}_{ui})^2}
$$


<!--
where $\mathcal{T}$ is the set consisting of pairs of users and items that you want to evaluate on.
$|\mathcal{T}|$ is the size of this set. We can use the RMSE function provided by `mx.metric`.
-->

trong đó $\mathcal{T}$ là tập bao gồm các cặp người dùng và sản phẩm mà ta sử dụng để đánh giá.
$|\mathcal{T}|$ là kích thước tập này. Ta có thể sử dụng hàm RMSE được cung cấp sẵn trong `mx.metric`.


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


<!--
## Training and Evaluating the Model
-->

## Huấn luyện và Đánh giá Mô hình


<!--
In the training function, we adopt the $L_2$ loss with weight decay.
The weight decay mechanism has the same effect as the $L_2$ regularization.
-->

Trong hàm huấn luyện, ta áp dụng mất mát $L_2$ với suy giảm trọng số.
Phương thức suy giảm trọng số có tác dụng giống như điều chuẩn $L_2$.


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


<!--
Finally, let us put all things together and train the model.
Here, we set the latent factor dimension to 30.
-->

Cuối cùng, hãy kết hợp tất cả với nhau và huấn luyện mô hình.
Trường hợp này, ta đặt kích thước nhân tố tiềm ẩn bằng 30.


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


<!--
Below, we use the trained model to predict the rating that a user (ID 20) might give to an item (ID 30).
-->

Ở dưới, ta sử dụng mô hình đã được huấn luyện để dự đoán đánh giá mà một người dùng (ID 20) có thể gán cho một sản phẩm (ID 30).


```{.python .input  n=6}
scores = net(np.array([20], dtype='int', ctx=devices[0]),
             np.array([30], dtype='int', ctx=devices[0]))
scores
```

## Tóm tắt

<!--
* The matrix factorization model is widely used in recommender systems.  It can be used to predict ratings that a user might give to an item.
* We can implement and train matrix factorization for recommender systems.
-->

* Mô hình phân rã ma trận được sử dụng rộng rãi trong hệ thống đề xuất. Nó có thể được sử dụng để dự đoán đánh giá của một người dùng cho một sản phẩm.
* Ta có thể lập trình và huấn luyện mô hình phân rã ma trận cho hệ thống đề xuất.


## Bài tập

<!--
* Vary the size of latent factors. How does the size of latent factors influence the model performance?
* Try different optimizers, learning rates, and weight decay rates.
* Check the predicted rating scores of other users for a specific movie.
-->

* Thay đổi kích thước của nhân tố tiềm ẩn. Kích thước của nhân tố tiềm ẩn ảnh hướng thế nào đến hiệu năng của mô hình?
* Thử các bộ tối ưu, tốc độ học và tốc độ suy giảm trọng số khác nhau.
* Kiểm tra giá trị đánh giá dự đoán của những người dùng khác cho một bộ phim cụ thể.


## Thảo luận
* Tiếng Anh: [MXNet](https://discuss.d2l.ai/t/400)
* Tiếng Việt: [Diễn đàn Machine Learning Cơ Bản](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Đỗ Trường Giang
* Nguyễn Văn Cường
* Nguyễn Lê Quang Nhật
* Lê Khắc Hồng Phúc

*Cập nhật lần cuối: 06/10/2020. (Cập nhật lần cuối từ nội dung gốc: 12/09/2020)*
