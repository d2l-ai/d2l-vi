# Chuẩn hóa hàng loạt
:label:`sec_batch_norm`

Đào tạo mạng thần kinh sâu là khó khăn. Và khiến họ hội tụ trong một khoảng thời gian hợp lý có thể khó khăn. Trong phần này, chúng tôi mô tả * chuẩn hóa theo lô*, một kỹ thuật phổ biến và hiệu quả liên tục tăng tốc sự hội tụ của các mạng sâu :cite:`Ioffe.Szegedy.2015`. Cùng với các khối còn lại - được bao phủ sau này trong :numref:`sec_resnet` — bình thường hóa hàng loạt đã giúp các học viên có thể thường xuyên đào tạo mạng với hơn 100 lớp. 

## Đào tạo mạng sâu

Để thúc đẩy bình thường hóa hàng loạt, chúng ta hãy xem xét một vài thách thức thực tế phát sinh khi đào tạo các mô hình học máy và mạng thần kinh nói riêng. 

Đầu tiên, các lựa chọn liên quan đến tiền xử lý dữ liệu thường tạo ra sự khác biệt rất lớn trong kết quả cuối cùng. Nhớ lại ứng dụng MLP của chúng tôi để dự đoán giá nhà (:numref:`sec_kaggle_house`). Bước đầu tiên của chúng tôi khi làm việc với dữ liệu thực là tiêu chuẩn hóa các tính năng đầu vào của chúng tôi cho mỗi tính năng có trung bình bằng 0 và phương sai của một. Trực giác, tiêu chuẩn hóa này chơi độc đáo với các trình tối ưu hóa của chúng tôi vì nó đặt các thông số* một ưu tiết* ở quy mô tương tự. 

Thứ hai, đối với MLP hoặc CNN điển hình, khi chúng ta đào tạo, các biến (ví dụ, đầu ra chuyển đổi affine trong MLP) trong các lớp trung gian có thể lấy các giá trị với kích thước khác nhau rộng rãi: cả dọc theo các lớp từ đầu vào đến đầu ra, trên các đơn vị trong cùng một lớp và theo thời gian do các bản cập nhật của chúng tôi cho mô hình tham số. Các nhà phát minh của bình thường hóa hàng loạt đưa ra một cách không chính thức rằng sự trôi dạt này trong việc phân phối các biến như vậy có thể cản trở sự hội tụ của mạng. Bằng trực giác, chúng ta có thể phỏng đoán rằng nếu một lớp có giá trị biến gấp 100 lần so với một lớp khác, điều này có thể đòi hỏi phải điều chỉnh bù trong tỷ lệ học tập. 

Thứ ba, các mạng sâu hơn rất phức tạp và dễ dàng có khả năng vượt trội. Điều này có nghĩa là sự chính quy hóa trở nên quan trọng hơn. 

Chuẩn hóa hàng loạt được áp dụng cho các lớp riêng lẻ (tùy chọn, cho tất cả chúng) và hoạt động như sau: Trong mỗi lần lặp đào tạo, trước tiên chúng ta bình thường hóa các đầu vào (bình thường hóa hàng loạt) bằng cách trừ trung bình của chúng và chia cho độ lệch chuẩn của chúng, trong đó cả hai được ước tính dựa trên số liệu thống kê của minibatch hiện tại. Tiếp theo, chúng tôi áp dụng một hệ số tỷ lệ và bù tỷ lệ. Chính xác là do *bình thường hóa* dựa trên số liệu thống kê * lô* mà * chuẩn hóa lông* bắt nguồn tên của nó. 

Lưu ý rằng nếu chúng tôi cố gắng áp dụng bình thường hóa hàng loạt với minibatches kích thước 1, chúng tôi sẽ không thể học bất cứ điều gì. Đó là bởi vì sau khi trừ các phương tiện, mỗi đơn vị ẩn sẽ lấy giá trị 0! Như bạn có thể đoán, vì chúng tôi đang dành cả một phần để bình thường hóa hàng loạt, với các minibatches đủ lớn, cách tiếp cận chứng minh hiệu quả và ổn định. Một takeaway ở đây là khi áp dụng bình thường hóa hàng loạt, sự lựa chọn kích thước lô có thể còn quan trọng hơn so với không bình thường hóa hàng loạt. 

Chính thức, biểu thị bằng $\mathbf{x} \in \mathcal{B}$ một đầu vào để bình thường hóa hàng loạt ($\mathrm{BN}$) đó là từ một minibatch $\mathcal{B}$, bình thường hóa hàng loạt biến đổi $\mathbf{x}$ theo biểu thức sau: 

$$\mathrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$
:eqlabel:`eq_batchnorm`

Trong :eqref:`eq_batchnorm`, $\hat{\boldsymbol{\mu}}_\mathcal{B}$ là trung bình mẫu và $\hat{\boldsymbol{\sigma}}_\mathcal{B}$ là độ lệch chuẩn mẫu của minibatch $\mathcal{B}$. Sau khi áp dụng tiêu chuẩn hóa, minibatch kết quả có không trung bình và phương sai đơn vị. Bởi vì sự lựa chọn phương sai đơn vị (so với một số số phép thuật khác) là một lựa chọn tùy ý, chúng ta thường bao gồm elementwise
*tham số quy mô* $\boldsymbol{\gamma}$ và *thay đổi* $\boldsymbol{\beta}$
that have the sametương tự shape hình dạng as $\mathbf{x}$. Lưu ý rằng $\boldsymbol{\gamma}$ và $\boldsymbol{\beta}$ là các thông số cần được học chung với các tham số mô hình khác. 

Do đó, các cường độ biến đổi cho các lớp trung gian không thể phân kỳ trong quá trình đào tạo vì bình thường hóa hàng loạt tích cực tập trung và rescales chúng trở lại một trung bình và kích thước nhất định (thông qua $\hat{\boldsymbol{\mu}}_\mathcal{B}$ và ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$). Một phần của trực giác hoặc trí tuệ của học viên là bình thường hóa hàng loạt dường như cho phép tỷ lệ học tập tích cực hơn. 

Chính thức, ta tính $\hat{\boldsymbol{\mu}}_\mathcal{B}$ và ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ trong :eqref:`eq_batchnorm` như sau: 

$$\begin{aligned} \hat{\boldsymbol{\mu}}_\mathcal{B} &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x},\\
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.\end{aligned}$$

Lưu ý rằng chúng ta thêm một hằng số nhỏ $\epsilon > 0$ vào ước tính phương sai để đảm bảo rằng chúng ta không bao giờ cố gắng chia bằng 0, ngay cả trong trường hợp ước tính phương sai thực nghiệm có thể biến mất. Các ước tính $\hat{\boldsymbol{\mu}}_\mathcal{B}$ và ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ chống lại vấn đề mở rộng quy mô bằng cách sử dụng các ước tính ồn ào về trung bình và phương sai. Bạn có thể nghĩ rằng tiếng ồn này nên là một vấn đề. Hóa ra, điều này thực sự có lợi. 

Điều này hóa ra là một chủ đề định kỳ trong học sâu. Vì những lý do chưa được đặc trưng tốt về mặt lý thuyết, các nguồn tiếng ồn khác nhau trong tối ưu hóa thường dẫn đến đào tạo nhanh hơn và ít quá mức hơn: biến thể này dường như hoạt động như một hình thức chính quy hóa. Trong một số nghiên cứu sơ bộ, :cite:`Teye.Azizpour.Smith.2018` và :cite:`Luo.Wang.Shao.ea.2018` liên quan đến các tính chất của bình thường hóa hàng loạt với các ưu tiên Bayesian và hình phạt tương ứng. Đặc biệt, điều này làm sáng tỏ câu đố tại sao bình thường hóa hàng loạt hoạt động tốt nhất cho kích thước minibatches vừa phải trong phạm vi $50 \sim 100$. 

Sửa một mô hình được đào tạo, bạn có thể nghĩ rằng chúng tôi muốn sử dụng toàn bộ tập dữ liệu để ước tính trung bình và phương sai. Sau khi đào tạo hoàn tất, tại sao chúng ta lại muốn cùng một hình ảnh được phân loại khác nhau, tùy thuộc vào lô mà nó xảy ra cư trú? Trong quá trình đào tạo, tính toán chính xác như vậy là không khả thi vì các biến trung gian cho tất cả các ví dụ dữ liệu thay đổi mỗi khi chúng tôi cập nhật mô hình của chúng tôi. Tuy nhiên, một khi mô hình được đào tạo, chúng ta có thể tính toán các phương tiện và phương sai của các biến của mỗi lớp dựa trên toàn bộ tập dữ liệu. Thật vậy đây là thực hành tiêu chuẩn cho các mô hình sử dụng bình thường hóa hàng loạt và do đó các lớp bình thường hóa hàng loạt hoạt động khác nhau trong chế độ đào tạo* (bình thường hóa bằng thống kê minibatch) và trong chế độ dự đoán * (bình thường hóa theo thống kê tập dữ liệu). 

Bây giờ chúng tôi đã sẵn sàng để xem cách bình thường hóa hàng loạt hoạt động trong thực tế. 

## Các lớp chuẩn hóa hàng loạt

Triển khai bình thường hóa hàng loạt cho các lớp được kết nối hoàn toàn và các lớp phức tạp hơi khác nhau. Chúng tôi thảo luận về cả hai trường hợp dưới đây. Nhớ lại rằng một sự khác biệt chính giữa chuẩn hóa hàng loạt và các lớp khác là vì việc chuẩn hóa hàng loạt hoạt động trên một minibatch đầy đủ tại một thời điểm, chúng ta không thể bỏ qua kích thước lô như chúng ta đã làm trước khi giới thiệu các layer khác. 

### Các lớp được kết nối hoàn toàn

Khi áp dụng bình thường hóa hàng loạt cho các lớp được kết nối hoàn toàn, giấy ban đầu sẽ chèn bình thường hóa hàng loạt sau khi chuyển đổi affine và trước chức năng kích hoạt phi tuyến (các ứng dụng sau này có thể chèn bình thường hóa hàng loạt ngay sau khi kích hoạt chức năng) :cite:`Ioffe.Szegedy.2015`. Biểu thị đầu vào cho lớp được kết nối hoàn toàn bằng $\mathbf{x}$, chuyển đổi affine bởi $\mathbf{W}\mathbf{x} + \mathbf{b}$ (với tham số trọng lượng $\mathbf{W}$ và tham số thiên vị $\mathbf{b}$) và chức năng kích hoạt bởi $\phi$, chúng ta có thể thể hiện tính toán của đầu ra lớp được kích hoạt theo lô, được kết nối đầy đủ $\mathbf{h}$ như sau: 

$$\mathbf{h} = \phi(\mathrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

Nhớ lại rằng trung bình và phương sai được tính toán trên minibatch * same* mà việc chuyển đổi được áp dụng. 

### Layers phức tạp

Tương tự, với các lớp phức tạp, chúng ta có thể áp dụng bình thường hóa hàng loạt sau khi kết hợp và trước hàm kích hoạt phi tuyến. Khi sự phức tạp có nhiều kênh đầu ra, chúng ta cần thực hiện bình thường hóa hàng loạt cho * mỗi* đầu ra của các kênh này và mỗi kênh có các thông số quy mô và thay đổi riêng, cả hai đều là vô hướng. Giả sử rằng minibatches của chúng tôi chứa $m$ ví dụ và đối với mỗi kênh, đầu ra của sự kết hợp có chiều cao $p$ và chiều rộng $q$. Đối với các lớp phức tạp, chúng tôi thực hiện mỗi đợt bình thường hóa trên $m \cdot p \cdot q$ các phần tử trên mỗi kênh đầu ra cùng một lúc. Do đó, chúng tôi thu thập các giá trị trên tất cả các vị trí không gian khi tính toán trung bình và phương sai và do đó áp dụng cùng một trung bình và phương sai trong một kênh nhất định để bình thường hóa giá trị tại mỗi vị trí không gian. 

### Chuẩn hóa hàng loạt trong dự đoán

Như chúng tôi đã đề cập trước đó, bình thường hóa hàng loạt thường hoạt động khác nhau trong chế độ đào tạo và chế độ dự đoán. Đầu tiên, tiếng ồn trong trung bình mẫu và phương sai mẫu phát sinh từ việc ước tính từng chiếc trên minibatches không còn mong muốn một khi chúng tôi đã đào tạo mô hình. Thứ hai, chúng ta có thể không có sự sang trọng của tính toán số liệu thống kê bình thường hóa mỗi lô. Ví dụ, chúng ta có thể cần áp dụng mô hình của mình để đưa ra một dự đoán tại một thời điểm. 

Thông thường, sau khi đào tạo, chúng tôi sử dụng toàn bộ tập dữ liệu để tính toán các ước tính ổn định của số liệu thống kê biến và sau đó sửa chúng vào thời điểm dự đoán. Do đó, bình thường hóa hàng loạt hoạt động khác nhau trong quá trình đào tạo và tại thời điểm thử nghiệm. Nhớ lại rằng bỏ học cũng thể hiện đặc điểm này. 

## (** Implementation from Scratch**)

Dưới đây, chúng tôi thực hiện một lớp bình thường hóa hàng loạt với hàng chục từ đầu.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, init
from mxnet.gluon import nn
npx.set_np()

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use `autograd` to determine whether the current mode is training mode or
    # prediction mode
    if not autograd.is_training():
        # If it is prediction mode, directly use the mean and variance
        # obtained by moving average
        X_hat = (X - moving_mean) / np.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully-connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / np.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean, moving_var
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use `is_grad_enabled` to determine whether the current mode is training
    # mode or prediction mode
    if not torch.is_grad_enabled():
        # If it is prediction mode, directly use the mean and variance
        # obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully-connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # Compute reciprocal of square root of the moving variance elementwise
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # Scale and shift
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y
```

Bây giờ chúng ta có thể [** tạo một lớp `BatchNorm` thích hợp.**] Lớp của chúng tôi sẽ duy trì các thông số thích hợp cho quy mô `gamma` và thay đổi `beta`, cả hai sẽ được cập nhật trong quá trình đào tạo. Ngoài ra, lớp của chúng ta sẽ duy trì đường trung bình động của phương tiện và phương sai để sử dụng tiếp theo trong quá trình dự đoán mô hình. 

Gạt các chi tiết thuật toán sang một bên, lưu ý mô hình thiết kế bên dưới việc thực hiện lớp của chúng ta. Thông thường, chúng tôi xác định toán học trong một hàm riêng biệt, nói `batch_norm`. Sau đó, chúng tôi tích hợp chức năng này vào một lớp tùy chỉnh, có mã chủ yếu đề cập đến các vấn đề kế toán, chẳng hạn như di chuyển dữ liệu sang bối cảnh thiết bị phù hợp, phân bổ và khởi tạo bất kỳ biến nào cần thiết, theo dõi các đường trung bình động (ở đây cho trung bình và phương sai), v.v. Mô hình này cho phép tách toán học sạch khỏi mã boilerplate. Cũng lưu ý rằng vì lợi ích của sự tiện lợi, chúng tôi đã không lo lắng về việc tự động suy ra hình dạng đầu vào ở đây, do đó chúng tôi cần chỉ định số lượng các tính năng trong suốt. Đừng lo lắng, các API chuẩn hóa hàng loạt cấp cao trong khuôn khổ học tập sâu sẽ quan tâm đến điều này cho chúng tôi và chúng tôi sẽ chứng minh rằng sau này.

```{.python .input}
class BatchNorm(nn.Block):
    # `num_features`: the number of outputs for a fully-connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully-connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims, **kwargs):
        super().__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # The variables that are not model parameters are initialized to 0 and 1
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.ones(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.ctx != X.ctx:
            self.moving_mean = self.moving_mean.copyto(X.ctx)
            self.moving_var = self.moving_var.copyto(X.ctx)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-12, momentum=0.9)
        return Y
```

```{.python .input}
#@tab pytorch
class BatchNorm(nn.Module):
    # `num_features`: the number of outputs for a fully-connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully-connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
```

```{.python .input}
#@tab tensorflow
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # The variables that are not model parameters are initialized to 0
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.ones,
            trainable=False)
        super(BatchNorm, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        momentum = 0.9
        delta = variable * momentum + value * (1 - momentum)
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, training):
        if training:
            axes = list(range(len(inputs.shape) - 1))
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            mean_update = self.assign_moving_average(
                self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(
                self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output
```

## [**Áp dụng Bình thường hóa hàng loạt trong LeNet**]

Để xem cách áp dụng `BatchNorm` trong bối cảnh, bên dưới chúng tôi áp dụng nó cho một mô hình LeNet truyền thống (:numref:`sec_lenet`). Nhớ lại rằng việc chuẩn hóa hàng loạt được áp dụng sau các lớp phức tạp hoặc các lớp được kết nối hoàn toàn nhưng trước các chức năng kích hoạt tương ứng.

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        BatchNorm(6, num_dims=4),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        BatchNorm(16, num_dims=4),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        BatchNorm(120, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        BatchNorm(84, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
# Recall that this has to be a function that will be passed to `d2l.train_ch6`
# so that model building or compiling need to be within `strategy.scope()` in
# order to utilize the CPU/GPU devices that we have
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10)]
    )
```

Như trước đây, chúng tôi sẽ [** đào tạo mạng của chúng tôi trên dữ liệu Fashion-MNIST**]. Mã này hầu như giống hệt với điều đó khi chúng tôi lần đầu tiên đào tạo LeNet (:numref:`sec_lenet`). Sự khác biệt chính là tỷ lệ học tập lớn hơn.

```{.python .input}
#@tab mxnet, pytorch
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

```{.python .input}
#@tab tensorflow
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
net = d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

Hãy để chúng tôi [** có một cái nhìn tại tham số quy mô `gamma` và tham số shift `beta`**] học được từ lớp chuẩn hóa hàng loạt đầu tiên.

```{.python .input}
net[1].gamma.data().reshape(-1,), net[1].beta.data().reshape(-1,)
```

```{.python .input}
#@tab pytorch
net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,))
```

```{.python .input}
#@tab tensorflow
tf.reshape(net.layers[1].gamma, (-1,)), tf.reshape(net.layers[1].beta, (-1,))
```

## [**Thiết tập**]

So với lớp `BatchNorm`, mà chúng ta vừa định nghĩa, chúng ta có thể sử dụng lớp `BatchNorm` được định nghĩa trong API cấp cao từ khung học sâu trực tiếp. Mã trông hầu như giống hệt với việc thực hiện của chúng tôi ở trên.

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10),
    ])
```

Dưới đây, chúng ta [** sử dụng cùng một siêu tham số để đào tạo mô hình của chúng tôi.**] Lưu ý rằng như thường lệ, biến thể API cấp cao chạy nhanh hơn nhiều vì mã của nó đã được biên dịch thành C++ hoặc CDA trong khi triển khai tùy chỉnh của chúng tôi phải được giải thích bởi Python.

```{.python .input}
#@tab all
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## Tranh cãi

Trực giác, bình thường hóa hàng loạt được cho là làm cho cảnh quan tối ưu hóa mượt mà hơn. Tuy nhiên, chúng ta phải cẩn thận để phân biệt giữa trực giác đầu cơ và giải thích thực sự cho các hiện tượng mà chúng ta quan sát khi đào tạo các mô hình sâu. Nhớ lại rằng chúng ta thậm chí không biết tại sao các mạng thần kinh sâu đơn giản hơn (MLP và CNN thông thường) khái quát tốt ngay từ đầu. Ngay cả khi bỏ học và phân rã trọng lượng, chúng vẫn linh hoạt đến mức khả năng khái quát hóa với dữ liệu vô hình của họ không thể được giải thích thông qua các đảm bảo tổng quát học hỏi thông thường. 

Trong bài viết gốc đề xuất bình thường hóa hàng loạt, các tác giả, ngoài việc giới thiệu một công cụ mạnh mẽ và hữu ích, đã đưa ra một lời giải thích cho lý do tại sao nó hoạt động: bằng cách giảm sự thay đổi giao hợp nội bộ*. Có lẽ bởi * chuyển giao hợp nội bộ* các tác giả có nghĩa là một cái gì đó giống như trực giác thể hiện ở trên - khái niệm rằng sự phân bố các giá trị biến thay đổi trong quá trình đào tạo. Tuy nhiên, đã có hai vấn đề với lời giải thích này: i) Sự trôi dạt này rất khác so với * covariate shift*, khiến tên một tên sai. ii) Lời giải thích cung cấp một trực giác dưới quy định nhưng để lại câu hỏi * tại sao chính xác kỹ thuật này hoạt động* một câu hỏi mở muốn giải thích nghiêm ngặt . Trong suốt cuốn sách này, chúng tôi đặt mục tiêu truyền đạt những trực giác mà các học viên sử dụng để hướng dẫn sự phát triển của họ về các mạng thần kinh sâu. Tuy nhiên, chúng tôi tin rằng điều quan trọng là phải tách các trực giác hướng dẫn này khỏi thực tế khoa học đã được thiết lập. Cuối cùng, khi bạn nắm vững tài liệu này và bắt đầu viết các bài báo nghiên cứu của riêng bạn, bạn sẽ muốn rõ ràng để phân định giữa các tuyên bố kỹ thuật và hunches. 

Sau sự thành công của việc bình thường hóa hàng loạt, lời giải thích của nó về * chuyển giao hợp nội bộ* đã nhiều lần nổi lên trong các cuộc tranh luận trong tài liệu kỹ thuật và diễn ngôn rộng hơn về cách trình bày nghiên cứu machine learning. Trong một bài phát biểu đáng nhớ được đưa ra trong khi chấp nhận Giải thưởng Thử nghiệm Thời gian tại hội nghị NeurIPS 2017, Ali Rahimi đã sử dụng dịch chuyển giao hợp nội bộ * làm tâm điểm trong một cuộc tranh luận giống như thực hành học sâu hiện đại với giả kim thuật. Sau đó, ví dụ đã được xem xét lại chi tiết trong một bài báo vị trí phác thảo xu hướng gây rắc rối trong máy học :cite:`Lipton.Steinhardt.2018`. Các tác giả khác đã đề xuất các giải thích thay thế cho sự thành công của việc bình thường hóa hàng loạt, một số người tuyên bố rằng thành công của bình thường hóa hàng loạt đến mặc dù thể hiện hành vi theo một số cách trái ngược với những người được tuyên bố trong bài báo gốc :cite:`Santurkar.Tsipras.Ilyas.ea.2018`. 

Chúng tôi lưu ý rằng sự thay đổi nội bộ* không xứng đáng với những lời chỉ trích hơn bất kỳ trong số hàng ngàn tuyên bố mơ hồ tương tự được đưa ra mỗi năm trong tài liệu học máy kỹ thuật. Có khả năng, sự cộng hưởng của nó như là một tâm điểm của các cuộc tranh luận này nợ khả năng nhận biết rộng rãi của nó đối với đối tượng mục tiêu. Chuẩn hóa hàng loạt đã chứng minh một phương pháp không thể thiếu, áp dụng trong gần như tất cả các phân loại hình ảnh được triển khai, kiếm được bài báo giới thiệu kỹ thuật hàng chục ngàn trích dẫn. 

## Tóm tắt

* Trong quá trình đào tạo mô hình, bình thường hóa hàng loạt liên tục điều chỉnh đầu ra trung gian của mạng thần kinh bằng cách sử dụng độ lệch trung bình và chuẩn của minibatch, do đó các giá trị của đầu ra trung gian trong mỗi lớp trong suốt mạng thần kinh ổn định hơn.
* Các phương pháp bình thường hóa hàng loạt cho các lớp được kết nối hoàn toàn và các lớp phức tạp hơi khác nhau.
* Giống như một lớp bỏ học, các lớp chuẩn hóa hàng loạt có kết quả tính toán khác nhau trong chế độ đào tạo và chế độ dự đoán.
* Bình thường hóa hàng loạt có nhiều tác dụng phụ có lợi, chủ yếu là thường xuyên hóa. Mặt khác, động lực ban đầu của việc giảm sự thay đổi đồng biến nội bộ dường như không phải là một lời giải thích hợp lệ.

## Bài tập

1. Chúng ta có thể loại bỏ tham số thiên vị khỏi lớp được kết nối hoàn toàn hoặc lớp phức tạp trước khi bình thường hóa hàng loạt không? Tại sao?
1. So sánh tỷ lệ học tập cho LeNet có và không có bình thường hóa hàng loạt.
    1. Vẽ sự gia tăng trong đào tạo và kiểm tra độ chính xác.
    1. Làm thế nào lớn bạn có thể làm cho tỷ lệ học tập?
1. Chúng ta có cần bình thường hóa hàng loạt trong mỗi lớp không? Thử nghiệm với nó?
1. Bạn có thể thay thế bỏ học bằng cách bình thường hóa hàng loạt không? Hành vi thay đổi như thế nào?
1. Khắc phục các thông số `beta` và `gamma`, và quan sát và phân tích kết quả.
1. Xem lại tài liệu trực tuyến cho `BatchNorm` từ API cấp cao để xem các ứng dụng khác để bình thường hóa hàng loạt.
1. Ý tưởng nghiên cứu: nghĩ về các biến đổi bình thường hóa khác mà bạn có thể áp dụng? Bạn có thể áp dụng biến đổi tích phân xác suất? Làm thế nào về một ước tính hiệp phương sai cấp đầy đủ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/83)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/84)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/330)
:end_tab:
