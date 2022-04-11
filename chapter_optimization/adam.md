# Ađam
:label:`sec_adam`

Trong các cuộc thảo luận dẫn đến phần này, chúng tôi đã gặp phải một số kỹ thuật để tối ưu hóa hiệu quả. Hãy để chúng tôi tóm tắt chi tiết chúng ở đây: 

* Chúng tôi thấy rằng :numref:`sec_sgd` có hiệu quả hơn Gradient Descent khi giải quyết các vấn đề tối ưu hóa, ví dụ, do khả năng phục hồi vốn có của nó đối với dữ liệu dư thừa. 
* Chúng tôi thấy rằng :numref:`sec_minibatch_sgd` mang lại hiệu quả bổ sung đáng kể phát sinh từ vectơ hóa, sử dụng các bộ quan sát lớn hơn trong một minibatch. Đây là chìa khóa để đa máy hiệu quả, đa GPU và xử lý song song tổng thể. 
* :numref:`sec_momentum` đã thêm một cơ chế để tổng hợp lịch sử của các gradient trong quá khứ để đẩy nhanh sự hội tụ.
* :numref:`sec_adagrad` sử dụng tỷ lệ cho mỗi tọa độ để cho phép một preconditioner hiệu quả tính toán. 
* :numref:`sec_rmsprop` tách tỷ lệ cho mỗi tọa độ từ một điều chỉnh tốc độ học tập. 

Adam :cite:`Kingma.Ba.2014` kết hợp tất cả các kỹ thuật này thành một thuật toán học tập hiệu quả. Đúng như dự đoán, đây là một thuật toán đã trở nên khá phổ biến như một trong những thuật toán tối ưu hóa mạnh mẽ và hiệu quả hơn để sử dụng trong học sâu. Nó không phải là không có vấn đề, mặc dù. Đặc biệt, :cite:`Reddi.Kale.Kumar.2019` cho thấy có những tình huống mà Adam có thể phân kỳ do kiểm soát phương sai kém. Trong một công việc tiếp theo :cite:`Zaheer.Reddi.Sachan.ea.2018` đã đề xuất một hotfix cho Adam, được gọi là Yogi giải quyết những vấn đề này. Thêm về điều này sau. Bây giờ chúng ta hãy xem xét thuật toán Adam.  

## Các thuật toán

Một trong những thành phần chính của Adam là nó sử dụng các đường trung bình động có trọng số mũ (còn được gọi là tính trung bình rò rỉ) để có được ước tính cả động lượng và cũng là khoảnh khắc thứ hai của gradient. Đó là, nó sử dụng các biến trạng thái 

$$\begin{aligned}
    \mathbf{v}_t & \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \\
    \mathbf{s}_t & \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2.
\end{aligned}$$

Ở đây $\beta_1$ và $\beta_2$ là các thông số trọng số không âm. Các lựa chọn phổ biến cho họ là $\beta_1 = 0.9$ và $\beta_2 = 0.999$. Đó là, ước tính phương sai di chuyển * chậm hơn* so với thuật ngữ động lượng. Lưu ý rằng nếu chúng ta khởi tạo $\mathbf{v}_0 = \mathbf{s}_0 = 0$, chúng ta có một lượng thiên vị đáng kể ban đầu hướng tới các giá trị nhỏ hơn. Điều này có thể được giải quyết bằng cách sử dụng thực tế là $\sum_{i=0}^t \beta^i = \frac{1 - \beta^t}{1 - \beta}$ để bình thường hóa các điều khoản. Tương ứng với các biến trạng thái bình thường được đưa ra bởi  

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \text{ and } \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}.$$

Được trang bị các ước tính thích hợp bây giờ chúng ta có thể viết ra các phương trình cập nhật. Đầu tiên, chúng tôi phát lại gradient theo cách rất giống với RMSProp để có được 

$$\mathbf{g}_t' = \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}.$$

Không giống như RMSProp, bản cập nhật của chúng tôi sử dụng động lượng $\hat{\mathbf{v}}_t$ thay vì chính gradient. Hơn nữa, có một sự khác biệt nhỏ về mỹ phẩm khi việc tái cặn xảy ra bằng cách sử dụng $\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$ thay vì $\frac{1}{\sqrt{\hat{\mathbf{s}}_t + \epsilon}}$. Các tác phẩm trước đây được cho là tốt hơn một chút trong thực tế, do đó độ lệch so với RMSProp. Thông thường, chúng tôi chọn $\epsilon = 10^{-6}$ để đánh đổi tốt giữa độ ổn định số và độ trung thực.  

Bây giờ chúng tôi có tất cả các phần tại chỗ để tính toán các bản cập nhật. Đây là một chút anticlimactic và chúng tôi có một bản cập nhật đơn giản của biểu mẫu 

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g}_t'.$$

Xem xét thiết kế của Adam cảm hứng của nó là rõ ràng. Động lượng và quy mô có thể nhìn thấy rõ trong các biến trạng thái. Định nghĩa khá đặc biệt của họ buộc chúng ta phải debias thuật ngữ (điều này có thể được khắc phục bởi một điều kiện khởi tạo và cập nhật hơi khác). Thứ hai, sự kết hợp của cả hai thuật ngữ là khá đơn giản, cho RMSProp. Cuối cùng, tỷ lệ học tập rõ ràng $\eta$ cho phép chúng ta kiểm soát độ dài bước để giải quyết các vấn đề hội tụ.  

## Thực hiện 

Thực hiện Adam từ đầu không phải là rất khó khăn. Để thuận tiện, chúng tôi lưu trữ thời gian bước truy cập $t$ trong từ điển `hyperparams`. Ngoài ra tất cả là đơn giản.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adam_states(feature_dim):
    v_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return ((v_w, s_w), (v_b, s_b))

def adam(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(beta2 * s + (1 - beta2) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
```

Chúng tôi đã sẵn sàng sử dụng Adam để đào tạo mô hình. Chúng tôi sử dụng tỷ lệ học tập là $\eta = 0.01$.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

Một triển khai ngắn gọn hơn là đơn giản vì `adam` là một trong những thuật toán được cung cấp như một phần của thư viện tối ưu hóa Gluon `trainer`. Do đó chúng ta chỉ cần vượt qua các tham số cấu hình để thực hiện trong Gluon.

```{.python .input}
d2l.train_concise_ch11('adam', {'learning_rate': 0.01}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adam
d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adam
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01}, data_iter)
```

## Yogi

Một trong những vấn đề của Adam là nó có thể không hội tụ ngay cả trong các cài đặt lồi khi ước tính khoảnh khắc thứ hai trong $\mathbf{s}_t$ nổ tung. Như một sửa chữa :cite:`Zaheer.Reddi.Sachan.ea.2018` đề xuất một bản cập nhật tinh chế (và khởi tạo) cho $\mathbf{s}_t$. Để hiểu những gì đang xảy ra, chúng ta hãy viết lại bản cập nhật Adam như sau: 

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \left(\mathbf{g}_t^2 - \mathbf{s}_{t-1}\right).$$

Bất cứ khi nào $\mathbf{g}_t^2$ có phương sai cao hoặc cập nhật thưa thớt, $\mathbf{s}_t$ có thể quên các giá trị quá khứ quá nhanh. Một sửa chữa có thể cho điều này là thay thế $\mathbf{g}_t^2 - \mathbf{s}_{t-1}$ bởi $\mathbf{g}_t^2 \odot \mathop{\mathrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1})$. Bây giờ độ lớn của bản cập nhật không còn phụ thuộc vào số lượng độ lệch. Điều này mang lại các bản cập nhật Yogi 

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \odot \mathop{\mathrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1}).$$

Các tác giả hơn nữa khuyên nên khởi tạo động lượng trên một lô ban đầu lớn hơn là chỉ ước tính theo chiều ngang ban đầu. Chúng tôi bỏ qua các chi tiết vì chúng không phải là vật chất đối với cuộc thảo luận và vì ngay cả khi không có sự hội tụ này vẫn còn khá tốt.

```{.python .input}
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = s + (1 - beta2) * np.sign(
            np.square(p.grad) - s) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab pytorch
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * torch.sign(
                torch.square(p.grad) - s) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab tensorflow
def yogi(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(s + (1 - beta2) * tf.math.sign(
                   tf.math.square(grad) - s) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

## Tóm tắt

* Adam kết hợp các tính năng của nhiều thuật toán tối ưu hóa thành một quy tắc cập nhật khá mạnh mẽ. 
* Được tạo ra trên cơ sở RMSProp, Adam cũng sử dụng EWMA trên gradient stochastic minibatch.
* Adam sử dụng hiệu chỉnh thiên vị để điều chỉnh cho một khởi động chậm khi ước tính động lượng và khoảnh khắc thứ hai. 
* Đối với gradient có phương sai đáng kể, chúng ta có thể gặp phải các vấn đề với sự hội tụ. Chúng có thể được sửa đổi bằng cách sử dụng minibatches lớn hơn hoặc bằng cách chuyển sang ước tính được cải thiện cho $\mathbf{s}_t$. Yogi cung cấp một sự thay thế như vậy. 

## Bài tập

1. Điều chỉnh tốc độ học tập và quan sát và phân tích kết quả thử nghiệm.
1. Bạn có thể viết lại đà và cập nhật khoảnh khắc thứ hai sao cho nó không yêu cầu điều chỉnh thiên vị?
1. Tại sao bạn cần giảm tỷ lệ học tập $\eta$ khi chúng ta hội tụ?
1. Cố gắng xây dựng một trường hợp mà Adam phân kỳ và Yogi hội tụ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/358)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1078)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1079)
:end_tab:
