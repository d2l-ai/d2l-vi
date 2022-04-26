# Adadelta
:label:`sec_adadelta`

Adadelta là một biến thể khác của AdaGrad (:numref:`sec_adagrad`). Sự khác biệt chính nằm ở thực tế là nó làm giảm số tiền mà tốc độ học tập thích ứng với tọa độ. Hơn nữa, theo truyền thống, nó được gọi là không có tỷ lệ học tập vì nó sử dụng số lượng thay đổi chính nó làm hiệu chuẩn cho sự thay đổi trong tương lai. Thuật toán được đề xuất vào năm :cite:`Zeiler.2012`. Nó khá đơn giản, được thảo luận về các thuật toán trước đó cho đến nay.  

## Các thuật toán

Tóm lại, Adadelta sử dụng hai biến trạng thái, $\mathbf{s}_t$ để lưu trữ trung bình bị rò rỉ của khoảnh khắc thứ hai của gradient và $\Delta\mathbf{x}_t$ để lưu trữ trung bình bị rò rỉ của khoảnh khắc thứ hai của sự thay đổi các tham số trong chính mô hình. Lưu ý rằng chúng tôi sử dụng ký hiệu ban đầu và đặt tên của các tác giả để tương thích với các ấn phẩm và triển khai khác (không có lý do thực sự nào khác tại sao người ta nên sử dụng các biến Hy Lạp khác nhau để chỉ ra một tham số phục vụ cùng một mục đích trong đà, Adagrad, RMSProp và Adadelta).  

Dưới đây là các chi tiết kỹ thuật của Adadelta. Với tham số du jour là $\rho$, chúng tôi có được các bản cập nhật rò rỉ sau tương tự như :numref:`sec_rmsprop`: 

$$\begin{aligned}
    \mathbf{s}_t & = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2.
\end{aligned}$$

Sự khác biệt đối với :numref:`sec_rmsprop` là chúng tôi thực hiện các bản cập nhật với gradient $\mathbf{g}_t'$ được thay đổi lại, tức là, 

$$\begin{aligned}
    \mathbf{x}_t  & = \mathbf{x}_{t-1} - \mathbf{g}_t'. \\
\end{aligned}$$

Vậy gradient được rescaled $\mathbf{g}_t'$ là gì? Chúng ta có thể tính toán nó như sau: 

$$\begin{aligned}
    \mathbf{g}_t' & = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{{\mathbf{s}_t + \epsilon}}} \odot \mathbf{g}_t, \\
\end{aligned}$$

trong đó $\Delta \mathbf{x}_{t-1}$ là mức trung bình bị rò rỉ của gradient được định lại bình phương $\mathbf{g}_t'$. Chúng tôi khởi tạo $\Delta \mathbf{x}_{0}$ là $0$ và cập nhật nó ở mỗi bước với $\mathbf{g}_t'$, tức là, 

$$\begin{aligned}
    \Delta \mathbf{x}_t & = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) {\mathbf{g}_t'}^2,
\end{aligned}$$

và $\epsilon$ (một giá trị nhỏ như $10^{-5}$) được thêm vào để duy trì sự ổn định số. 

## Thực hiện

Adadelta cần duy trì hai biến trạng thái cho mỗi biến, $\mathbf{s}_t$ và $\Delta\mathbf{x}_t$. Điều này mang lại việc thực hiện sau đây.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        # In-place updates via [:]
        s[:] = rho * s + (1 - rho) * np.square(p.grad)
        g = (np.sqrt(delta + eps) / np.sqrt(s + eps)) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # In-place updates via [:]
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adadelta_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    delta_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    delta_b = tf.Variable(d2l.zeros(1))
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, grads, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta), grad in zip(params, states, grads):
        s[:].assign(rho * s + (1 - rho) * tf.math.square(grad))
        g = (tf.math.sqrt(delta + eps) / tf.math.sqrt(s + eps)) * grad
        p[:].assign(p - g)
        delta[:].assign(rho * delta + (1 - rho) * g * g)
```

Chọn $\rho = 0.9$ lên tới thời gian bán hủy là 10 cho mỗi bản cập nhật tham số. Điều này có xu hướng hoạt động khá tốt. Chúng tôi nhận được hành vi sau đây.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```

Để thực hiện ngắn gọn, chúng tôi chỉ cần sử dụng thuật toán `adadelta` từ lớp `Trainer`. Điều này mang lại một lớp lót sau đây cho một lời gọi nhỏ gọn hơn nhiều.

```{.python .input}
d2l.train_concise_ch11('adadelta', {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
# adadelta is not converging at default learning rate
# but it's converging at lr = 5.0
trainer = tf.keras.optimizers.Adadelta
d2l.train_concise_ch11(trainer, {'learning_rate':5.0, 'rho': 0.9}, data_iter)
```

## Tóm tắt

* Adadelta không có tham số tỷ lệ học tập. Thay vào đó, nó sử dụng tốc độ thay đổi trong chính các thông số để điều chỉnh tốc độ học tập. 
* Adadelta yêu cầu hai biến trạng thái để lưu trữ những khoảnh khắc thứ hai của gradient và sự thay đổi trong các tham số. 
* Adadelta sử dụng trung bình rò rỉ để giữ ước tính chạy của các số liệu thống kê thích hợp. 

## Bài tập

1. Điều chỉnh giá trị của $\rho$. Điều gì xảy ra?
1. Hiển thị cách thực hiện thuật toán mà không cần sử dụng $\mathbf{g}_t'$. Tại sao đây có thể là một ý tưởng tốt?
1. Adadelta có thực sự học tỷ lệ miễn phí không? Bạn có thể tìm thấy các vấn đề tối ưu hóa phá vỡ Adadelta?
1. So sánh Adadelta với Adagrad và RMS prop để thảo luận về hành vi hội tụ của họ.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/357)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1076)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1077)
:end_tab:
