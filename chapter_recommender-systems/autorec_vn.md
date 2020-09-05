<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->

<!--
# AutoRec: Rating Prediction with Autoencoders
-->

# AutoRec: Dự đoán Đánh giá với Bộ tự mã hoá


<!--
Although the matrix factorization model achieves decent performance on the rating prediction task, it is essentially a linear model.
Thus, such models are not capable of capturing complex nonlinear and intricate relationships that may be predictive of users' preferences.
In this section, we introduce a nonlinear neural network collaborative filtering model, AutoRec :cite:`Sedhain.Menon.Sanner.ea.2015`.
It identifies collaborative filtering (CF) with an autoencoder architecture and aims to integrate nonlinear transformations into CF on the basis of explicit feedback.
Neural networks have been proven to be capable of approximating any continuous function, 
making it suitable to address the limitation of matrix factorization and enrich the expressiveness of matrix factorization.
-->

Mặc dù mô hình phân rã ma trận đạt được hiệu năng tương đối ổn trong bài toán dự đoán đánh giá, về căn bản nó là một mô hình tuyến tính.
Do đó, những mô hình như này không có khả năng nắm bắt được quan hệ phi tuyến phức tạp và rắc rối mà có thể dự đoán được sở thích của người dùng.
Trong phần này, chúng tôi sẽ giới thiệu một mô hình mạng nơ-ron lọc cộng tác phi tuyến, AutoRec :cite:`Sedhain.Menon.Sanner.ea.2015`.
Nó đồng nhất so với với lọc cộng tác (CF) với kiến trúc của một bộ tự mã hoá (*autoencoder*) và nhằm mục đích tích hợp biến đổi phi tuyến vào CF dựa theo phản hồi trực tiếp.
Mạng nơ-ron đã được chứng minh rằng có khả năng xấp xỉ bất kì hàm liên tục nào,
điều này khiến nó phù hợp để khắc phục hạn chế của mô hình phân rã ma trận và tăng cường khả năng biển diễn ý nghĩa của phân rã ma trận.


<!--
On one hand, AutoRec has the same structure as an autoencoder which consists of an input layer, a hidden layer, and a reconstruction (output) layer.
An autoencoder is a neural network that learns to copy its input to its output in order to code the inputs into the hidden (and usually low-dimensional) representations.
In AutoRec, instead of explicitly embedding users/items into low-dimensional space, 
it uses the column/row of the interaction matrix as the input, then reconstructs the interaction matrix in the output layer.
-->

Một mặt, AutoRec có cùng cấu trúc với một bộ tự mã hoá gồm có một tầng đầu vào, một tầng ẩn và một tầng khôi phục (đầu ra).
Bộ tự mã hoá là một mạng nơ-ron học cách sao chép đầu vào sang đầu ra nhằm mã hoá đầu vào thành dạng biểu diễn ẩn (và thường có kích thước nhỏ).
Trong AutoRec, thay vì nhúng trực tiếp người dùng/sản phẩm vào không gian kích thước nhỏ hơn,
nó sử dụng các cột/hàng của ma trận tương tác làm đầu vào, sau đó khôi phục lại ma trận tương tác ở đầu ra.


<!--
On the other hand, AutoRec differs from a traditional autoencoder: rather than learning the hidden representations, AutoRec focuses on learning/reconstructing the output layer.
It uses a partially observed interaction matrix as the input, aiming to reconstruct a completed rating matrix.
In the meantime, the missing entries of the input are filled in the output layer via reconstruction for the purpose of recommendation.
-->

Mặt khác, AutoRec khác với bộ tự mã hoá truyền thống ở chỗ: thay vì học dạng biểu diễn ẩn, AutoRec tập trung vào học/khôi phục tầng đầu ra.
Nó sử dụng từng phần của ma trận tương tác đã biết làm đầu vào, nhắm tới việc khôi mục một ma trận đánh giá hoàn chỉnh.
Trong khi đó, các phần tử còn thiếu được thêm vào trong tầng đầu ra thông qua quá trình khôi phục với mục đích gợi ý.


<!--
There are two variants of AutoRec: user-based and item-based.
For brevity, here we only introduce the item-based AutoRec.
User-based AutoRec can be derived accordingly.
-->

Có hai dạng AutoRec: dựa vào người dùng (*user-based*) và dựa vào sản phẩm (*item-based*).
Để ngắn gọn, ở đây chúng tôi chỉ giới thiệu AutoRec dựa vào sản phẩm.
AutoRec dựa vào người dùng có thể được suy ra một cách tương tự.


<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Model
-->

## *dịch tiêu đề trên*


<!--
Let $\mathbf{R}_{*i}$ denote the $i^\mathrm{th}$ column of the rating matrix, 
where unknown ratings are set to zeros by default.
The neural architecture is defined as:
-->

*dịch đoạn phía trên*


$$
h(\mathbf{R}_{*i}) = f(\mathbf{W} \cdot g(\mathbf{V} \mathbf{R}_{*i} + \mu) + b)
$$


<!--
where $f(\cdot)$ and $g(\cdot)$ represent activation functions, $\mathbf{W}$ and $\mathbf{V}$ are weight matrices, $\mu$ and $b$ are biases.
Let $h( \cdot )$ denote the whole network of AutoRec.
The output $h(\mathbf{R}_{*i})$ is the reconstruction of the $i^\mathrm{th}$ column of the rating matrix.
-->

*dịch đoạn phía trên*


<!--
The following objective function aims to minimize the reconstruction error:
-->

*dịch đoạn phía trên*


$$
\underset{\mathbf{W},\mathbf{V},\mu, b}{\mathrm{argmin}} \sum_{i=1}^M{\parallel \mathbf{R}_{*i} - h(\mathbf{R}_{*i})\parallel_{\mathcal{O}}^2} +\lambda(\| \mathbf{W} \|_F^2 + \| \mathbf{V}\|_F^2)
$$


<!--
where $\| \cdot \|_{\mathcal{O}}$ means only the contribution of observed ratings are considered, 
that is, only weights that are associated with observed inputs are updated during back-propagation.
-->

*dịch đoạn phía trên*


```{.python .input  n=3}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
import sys
npx.set_np()
```


<!--
## Implementing the Model
-->

## *dịch tiêu đề trên*


<!--
A typical autoencoder consists of an encoder and a decoder.
The encoder projects the input to hidden representations and the decoder maps the hidden layer to the reconstruction layer.
We follow this practice and create the encoder and decoder with dense layers.
The activation of encoder is set to `sigmoid` by default and no activation is applied for decoder.
Dropout is included after the encoding transformation to reduce over-fitting.
The gradients of unobserved inputs are masked out to ensure that only observed ratings contribute to the model learning process.
-->

*dịch đoạn phía trên*


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

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
## Reimplementing the Evaluator
-->

## *dịch tiêu đề trên*


<!--
Since the input and output have been changed, we need to reimplement the evaluation function, while we still use RMSE as the accuracy measure.
-->

*dịch đoạn phía trên*


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


<!--
## Training and Evaluating the Model
-->

## *dịch tiêu đề trên*


<!--
Now, let us train and evaluate AutoRec on the MovieLens dataset.
We can clearly see that the test RMSE is lower than the matrix factorization model,
confirming the effectiveness of neural networks in the rating prediction task.
-->

*dịch đoạn phía trên*


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

<!--
* We can frame the matrix factorization algorithm with autoencoders, while integrating non-linear layers and dropout regularization. 
* Experiments on the MovieLens 100K dataset show that AutoRec achieves superior performance than matrix factorization.
-->

*dịch đoạn phía trên*


## Bài tập

<!--
* Vary the hidden dimension of AutoRec to see its impact on the model performance.
* Try to add more hidden layers. Is it helpful to improve the model performance?
* Can you find a better combination of decoder and encoder activation functions?
-->

*dịch đoạn phía trên*


<!-- ===================== Kết thúc dịch Phần 3 ===================== -->


## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/401)
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

<!-- Phần 3 -->
* 

*Cập nhật lần cuối: 03/09/2020. (Cập nhật lần cuối từ nội dung gốc: 21/07/2020)*
