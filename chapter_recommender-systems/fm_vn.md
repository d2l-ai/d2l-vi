<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->

<!--
# Factorization Machines
-->

# Máy Phân rã ma trận


<!--
Factorization machines (FM) :cite:`Rendle.2010`, proposed by Steffen Rendle in 2010, 
is a supervised algorithm that can be used for classification, regression, and ranking tasks.
It quickly took notice and became a popular and impactful method for making predictions and recommendations.
Particularly, it is a generalization of the linear regression model and the matrix factorization model.
Moreover, it is reminiscent of support vector machines with a polynomial kernel.
The strengths of factorization machines over the linear regression and matrix factorization are:
(1) it can model $\chi$-way variable interactions, where $\chi$ is the number of polynomial order and is usually set to two.
(2) A fast optimization algorithm associated with factorization machines can reduce the polynomial computation time to linear complexity, 
making it extremely efficient especially for high dimensional sparse inputs.
For these reasons, factorization machines are widely employed in modern advertisement and products recommendations.
The technical details and implementations are described below.
-->

Máy phân rã ma trận (*Factorization machines - FM*) :cite:`Rendle.2010`, được đề xuất bởi Steffen Rendle vào năm 2010,
là một thuật toán học có giám sát, có thể được sử dụng trong các tác vụ phân loại, hồi quy hay xếp hạng.
Nó nhanh chóng nhận được sự chú ý và trở thành một phương pháp nổi tiếng và có sức ảnh hưởng trong việc đưa ra dự đoán và gợi ý.
Cụ thể, nó là kết quả khái quát hoá mô hình hồi quy tuyến tính và mô hình phân rã ma trận.
Hơn nữa, nó còn gợi nhớ đến máy vector hỗ trợ với hạt nhân đa thức.
Điểm mạnh của máy phân rã ma trận so với hồi quy tuyến tính và phân ra mã trận là:
(1) Nó có thể mô hình hoá biến tương tác $\chi$ chiều, với $\chi$ là bậc của đa thức và thường được đặt bằng hai.
(2) Một thuật toán tối ưu tốc độ cao đi kèm với máy phân rã ma trận có thể giảm thời gian tính toán đa thức về độ phức tạp tuyến tính,
giúp nó đạt hiệu quả cực tốt đặc biệt với đầu vào thưa thớt nhiều chiều.
Với các lý do trên, máy phân rã được áp dụng rộng rãi trong quảng cáo hiện đại và gợi ý sản phẩm.
Chi tiết kỹ thuật cũng như cách lập trình được mô tả trong các phần dưới.


<!--
## 2-Way Factorization Machines
-->

## Máy Phân rã 2 Chiều.


<!--
Formally, let $x \in \mathbb{R}^d$ denote the feature vectors of one sample, and $y$ denote the corresponding label 
which can be real-valued label or class label such as binary class "click/non-click".
The model for a factorization machine of degree two is defined as:
-->

Gọi $x \in \mathbb{R}^d$ ký hiệu vector đặc trưng của một mẫu, và $y$ ký hiệu nhãn tương ứng,
nhãn này có thể mang giá trị thực hoặc là nhãn hạng mục ví dụ như hạng mục nhị phân "nhấp/chưa nhấp".
Mô hình của máy phân rã ma trận bậc hai được định nghĩa như sau:


$$
\hat{y}(x) = \mathbf{w}_0 + \sum_{i=1}^d \mathbf{w}_i x_i + \sum_{i=1}^d\sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j
$$


<!--
where $\mathbf{w}_0 \in \mathbb{R}$ is the global bias;
$\mathbf{w} \in \mathbb{R}^d$ denotes the weights of the i-th variable;
$\mathbf{V} \in \mathbb{R}^{d\times k}$ represents the feature embeddings;
$\mathbf{v}_i$ represents the $i^\mathrm{th}$ row of $\mathbf{V}$; $k$ is the dimensionality of latent factors;
$\langle\cdot, \cdot \rangle$ is the dot product of two vectors.
$\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ model the interaction between the $i^\mathrm{th}$ and $j^\mathrm{th}$ feature.
Some feature interactions can be easily understood so they can be designed by experts.
However, most other feature interactions are hidden in data and difficult to identify.
So modeling feature interactions automatically can greatly reduce the efforts in feature engineering.
It is obvious that the first two terms correspond to the linear regression model and the last term is an extension of the matrix factorization model.
If the feature $i$ represents a item and the feature $j$ represents a user, the third term is exactly the dot product between user and item embeddings.
It is worth noting that FM can also generalize to higher orders (degree > 2).
Nevertheless, the numerical stability might weaken the generalization.
-->

trong đó $\mathbf{w}_0 \in \mathbb{R}$ là độ chệch toàn cục;
$\mathbf{w} \in \mathbb{R}^d$ ký hiệu trọng số của biến thứ $i$;
$\mathbf{V} \in \mathbb{R}^{d\times k}$ biểu diễn embedding đặc trưng;
$\mathbf{v}_i$ biểu diễn hàng thứ $i$ của $\mathbf{V}$; $k$ là số chiều của nhân tố ẩn;
$\langle\cdot, \cdot \rangle$ là tích vô hướng của hai vector.
$\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ mô hình hoá sự tương tác giữa đặc trưng thứ $i$ và thứ $j$.
Một số tương tác đặc trưng có thể dễ dàng hiểu được cho nên chúng có thể được thiết kế bởi các chuyên gia.
Tuy nhiên, đa số các tương tác đặc trưng khác thường bị ẩn giấu trong dữ liệu và khó để có thể nhận biết.
Do đó việc tự động mô hình hoá tương tác đặc trưng có thể giảm đáng kể công sức trong việc thiết kế đặc trưng (*feature engineering*).
Ta có thể thấy rõ ràng rằng hai số hạng đầu tiên tương ứng với mô hình hồi quy tuyến tính và số hạng cuối cùng là dạng mở rộng của mô hình phân rã ma trận.
Nếu đặc trưng $i$ biểu diễn một sản phẩm và đặc trưng $j$ biểu diễn một người dùng, số hạng thứ ba chính là tích vô hướng giữa embedding người dùng và sản phẩm.
Đáng chú ý là FM cũng có thể khái quát hoá với bậc cao hơn (bậc > 2).
Tuy vậy, tính ổn định số có thể khiến sự khái quát bị suy giảm.

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## An Efficient Optimization Criterion
-->

## *dịch tiêu đề trên*


<!--
Optimizing the factorization machines in a straight forward method leads to a complexity of $\mathcal{O}(kd^2)$ as all pairwise interactions require to be computed.
To solve this inefficiency problem, we can reorganize the third term of FM which could greatly reduce the computation cost, leading to a linear time complexity ($\mathcal{O}(kd)$).
The reformulation of the pairwise interaction term is as follows:
-->

*dịch đoạn phía trên*


$$
\begin{aligned}
&\sum_{i=1}^d \sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j \\
 &= \frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d\langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j - \frac{1}{2}\sum_{i=1}^d \langle\mathbf{v}_i, \mathbf{v}_i\rangle x_i x_i \\
 &= \frac{1}{2} \big (\sum_{i=1}^d \sum_{j=1}^d \sum_{l=1}^k\mathbf{v}_{i, l} \mathbf{v}_{j, l} x_i x_j - \sum_{i=1}^d \sum_{l=1}^k \mathbf{v}_{i, l} \mathbf{v}_{i, l} x_i x_i \big)\\
 &=  \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i) (\sum_{j=1}^d \mathbf{v}_{j, l}x_j) - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2 \big ) \\
 &= \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i)^2 - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2)
 \end{aligned}
$$


<!--
With this reformulation, the model complexity are decreased greatly.
Moreover, for sparse features, only non-zero elements needs to be computed so that the overall complexity is linear to the number of non-zero features.
-->

*dịch đoạn phía trên*


<!--
To learn the FM model, we can use the MSE loss for regression task, the cross entropy loss for classification tasks, and the BPR loss for ranking task.
Standard optimizers such as SGD and Adam are viable for optimization.
-->

*dịch đoạn phía trên*


```{.python .input  n=2}
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os
import sys
npx.set_np()
```


<!--
## Model Implementation
-->

## *dịch tiêu đề trên*


<!--
The following code implement the factorization machines.
It is clear to see that FM consists a linear regression block and an efficient feature interaction block.
We apply a sigmoid function over the final score since we treat the CTR prediction as a classification task.
-->

*dịch đoạn phía trên*


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


<!--
## Load the Advertising Dataset
-->

## *dịch tiêu đề trên*


<!--
We use the CTR data wrapper from the last section to load the online advertising dataset.
-->

*dịch đoạn phía trên*


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


<!--
## Train the Model
-->

## *dịch tiêu đề trên*


<!--
Afterwards, we train the model. The learning rate is set to 0.01 and the embedding size is set to 20 by default.
The `Adam` optimizer and the `SigmoidBinaryCrossEntropyLoss` loss are used for model training.
-->

*dịch đoạn phía trên*


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

<!--
* FM is a general framework that can be applied on a variety of tasks such as regression, classification, and ranking. 
* Feature interaction/crossing is important for prediction tasks and the 2-way interaction can be efficiently modeled with FM.
-->

*dịch đoạn phía trên*


## Bài tập

<!--
* Can you test FM on other dataset such as Avazu, MovieLens, and Criteo datasets?
* Vary the embedding size to check its impact on performance, can you observe a similar pattern as that of matrix factorization?
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/406)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md
-->

* Đoàn Võ Duy Thanh
<!-- Phần 1 -->
* Đỗ Trường Giang

<!-- Phần 2 -->
* 

*Cập nhật lần cuối: 03/09/2020. (Cập nhật lần cuối từ nội dung gốc: 21/07/2020)*
