# RMSProp
:label:`sec_rmsprop`

Một trong những vấn đề chính trong :numref:`sec_adagrad` là tốc độ học tập giảm theo lịch trình xác định trước là $\mathcal{O}(t^{-\frac{1}{2}})$ hiệu quả. Mặc dù điều này thường thích hợp cho các vấn đề lồi, nhưng nó có thể không lý tưởng cho những vấn đề không lồi, chẳng hạn như những vấn đề gặp phải trong học sâu. Tuy nhiên, sự thích nghi phối hợp khôn ngoan của Adagrad là rất mong muốn như một preconditioner. 

:cite:`Tieleman.Hinton.2012` đề xuất thuật toán RMSProp như một sửa chữa đơn giản để tách lập kế hoạch tỷ lệ từ tốc độ học tập thích ứng phối hợp. Vấn đề là Adagrad tích lũy các ô vuông của gradient $\mathbf{g}_t$ thành một vector trạng thái $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$. Kết quả là $\mathbf{s}_t$ tiếp tục phát triển mà không bị ràng buộc do thiếu bình thường hóa, về cơ bản tuyến tính khi thuật toán hội tụ. 

Một cách để khắc phục vấn đề này là sử dụng $\mathbf{s}_t / t$. Đối với các phân phối hợp lý của $\mathbf{g}_t$, điều này sẽ hội tụ. Thật không may, nó có thể mất một thời gian rất dài cho đến khi hành vi giới hạn bắt đầu quan trọng vì thủ tục ghi nhớ quỹ đạo đầy đủ của các giá trị. Một cách khác là sử dụng trung bình rò rỉ theo cùng một cách chúng ta đã sử dụng trong phương pháp động lượng, tức là $\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$ cho một số tham số $\gamma > 0$. Giữ tất cả các bộ phận khác không thay đổi năng suất RMSProp. 

## Các thuật toán

Hãy để chúng tôi viết ra các phương trình một cách chi tiết. 

$$\begin{aligned}
    \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2, \\
    \mathbf{x}_t & \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.
\end{aligned}$$

Hằng số $\epsilon > 0$ thường được đặt thành $10^{-6}$ để đảm bảo rằng chúng tôi không bị phân chia bằng 0 hoặc quá lớn kích thước bước. Với sự mở rộng này, chúng tôi hiện có thể tự do kiểm soát tốc độ học tập $\eta$ độc lập với tỷ lệ được áp dụng trên cơ sở mỗi tọa độ. Về mặt trung bình rò rỉ, chúng ta có thể áp dụng lý luận tương tự như được áp dụng trước đây trong trường hợp phương pháp động lượng. Mở rộng định nghĩa về sản lượng $\mathbf{s}_t$ 

$$
\begin{aligned}
\mathbf{s}_t & = (1 - \gamma) \mathbf{g}_t^2 + \gamma \mathbf{s}_{t-1} \\
& = (1 - \gamma) \left(\mathbf{g}_t^2 + \gamma \mathbf{g}_{t-1}^2 + \gamma^2 \mathbf{g}_{t-2} + \ldots, \right).
\end{aligned}
$$

Như trước đây trong :numref:`sec_momentum` chúng tôi sử dụng $1 + \gamma + \gamma^2 + \ldots, = \frac{1}{1-\gamma}$. Do đó tổng trọng lượng được chuẩn hóa thành $1$ với thời gian bán hủy của một quan sát $\gamma^{-1}$. Chúng ta hãy hình dung trọng lượng trong 40 bước thời gian qua cho các lựa chọn khác nhau của $\gamma$.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import math
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import math
```

```{.python .input}
#@tab all
d2l.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, (1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')
d2l.plt.xlabel('time');
```

## Thực hiện từ đầu

Như trước đây chúng ta sử dụng hàm bậc hai $f(\mathbf{x})=0.1x_1^2+2x_2^2$ để quan sát quỹ đạo của RMSProp. Nhớ lại rằng trong :numref:`sec_adagrad`, khi chúng ta sử dụng Adagrad với tốc độ học 0.4, các biến chỉ di chuyển rất chậm trong các giai đoạn sau của thuật toán vì tốc độ học tập giảm quá nhanh. Vì $\eta$ được kiểm soát riêng, điều này không xảy ra với RMSProp.

```{.python .input}
#@tab all
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```

Tiếp theo, chúng tôi triển khai RMSProp để được sử dụng trong một mạng sâu. Điều này cũng đơn giản như nhau.

```{.python .input}
#@tab mxnet,pytorch
def init_rmsprop_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)
```

```{.python .input}
#@tab tensorflow
def init_rmsprop_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)
```

```{.python .input}
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s[:] = gamma * s + (1 - gamma) * np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def rmsprop(params, grads, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(gamma * s + (1 - gamma) * tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

Chúng tôi đặt tỷ lệ học tập ban đầu là 0,01 và thuật ngữ trọng số $\gamma$ thành 0.9. Đó là, $\mathbf{s}$ tổng hợp trung bình trong quá khứ $1/(1-\gamma) = 10$ quan sát của gradient vuông.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);
```

## Thực hiện ngắn gọn

Vì RMSProp là một thuật toán khá phổ biến, nó cũng có sẵn trong phiên bản `Trainer`. Tất cả những gì chúng ta cần làm là khởi tạo nó bằng cách sử dụng một thuật toán có tên `rmsprop`, gán $\gamma$ cho tham số `gamma1`.

```{.python .input}
d2l.train_concise_ch11('rmsprop', {'learning_rate': 0.01, 'gamma1': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.RMSprop
d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9},
                       data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.RMSprop
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01, 'rho': 0.9},
                       data_iter)
```

## Tóm tắt

* RMSProp rất giống với Adagrad trong chừng mực vì cả hai đều sử dụng hình vuông của gradient để quy mô các hệ số.
* RMSProp chia sẻ với động lượng trung bình bị rò rỉ. Tuy nhiên, RMSProp sử dụng kỹ thuật này để điều chỉnh tiền điều hòa hệ số khôn ngoan.
* Tỷ lệ học tập cần được lên lịch bởi các thí nghiệm trong thực tế.
* Hệ số $\gamma$ xác định lịch sử trong bao lâu khi điều chỉnh thang đo mỗi tọa độ.

## Bài tập

1. Điều gì xảy ra bằng thực nghiệm nếu chúng ta đặt $\gamma = 1$? Tại sao?
1. Xoay vấn đề tối ưu hóa để giảm thiểu $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$. Điều gì xảy ra với sự hội tụ?
1. Hãy thử những gì xảy ra với RMSProp về một vấn đề máy học thực sự, chẳng hạn như đào tạo về Fashion-MNIST. Thử nghiệm với các lựa chọn khác nhau để điều chỉnh tốc độ học tập.
1. Bạn có muốn điều chỉnh $\gamma$ khi tối ưu hóa tiến triển không? RMSProp nhạy cảm như thế nào đối với điều này?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/356)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1074)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1075)
:end_tab:
