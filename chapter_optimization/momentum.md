# Đà
:label:`sec_momentum`

Trong :numref:`sec_sgd`, chúng tôi đã xem xét những gì xảy ra khi thực hiện gốc gradient ngẫu nhiên, tức là khi thực hiện tối ưu hóa, nơi chỉ có một biến thể ồn ào của gradient có sẵn. Đặc biệt, chúng tôi nhận thấy rằng đối với độ dốc ồn ào, chúng ta cần phải thận trọng hơn khi chọn tốc độ học tập khi đối mặt với tiếng ồn. Nếu chúng ta giảm nó quá nhanh, hội tụ quầy hàng. Nếu chúng ta quá khoan dung, chúng ta không hội tụ thành một giải pháp đủ tốt vì tiếng ồn tiếp tục khiến chúng ta tránh xa sự tối ưu. 

## Khái niệm cơ bản

Trong phần này, chúng ta sẽ khám phá các thuật toán tối ưu hóa hiệu quả hơn, đặc biệt là đối với một số loại vấn đề tối ưu hóa phổ biến trong thực tế. 

### Đường trung bình bị rò rỉ

Phần trước đã thấy chúng tôi thảo luận về minibatch SGD như một phương tiện để tăng tốc tính toán. Nó cũng có tác dụng phụ tốt đẹp mà độ dốc trung bình làm giảm lượng phương sai. Các minibatch stochastic gradient descent có thể được tính bằng cách: 

$$\mathbf{g}_{t, t-1} = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}_{t-1}) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbf{h}_{i, t-1}.
$$

Để giữ cho ký hiệu đơn giản, ở đây chúng tôi đã sử dụng $\mathbf{h}_{i, t-1} = \partial_{\mathbf{w}} f(\mathbf{x}_i, \mathbf{w}_{t-1})$ làm gốc gradient ngẫu nhiên cho mẫu $i$ bằng cách sử dụng các trọng lượng được cập nhật tại thời điểm $t-1$. Sẽ thật tuyệt nếu chúng ta có thể hưởng lợi từ ảnh hưởng của việc giảm phương sai ngay cả ngoài độ dốc trung bình trên một minibatch. Một lựa chọn để thực hiện nhiệm vụ này là thay thế tính toán gradient bằng một “trung bình rò rỉ”: 

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}$$

for some $\beta \in (0, 1)$. Điều này thay thế hiệu quả gradient tức thời bằng một gradient được trung bình trên nhiều độ dốc * past*. $\mathbf{v}$ được gọi là *momentum*. Nó tích lũy gradient trong quá khứ tương tự như cách một quả bóng nặng lăn xuống cảnh quan chức năng mục tiêu tích hợp trên các lực trong quá khứ. Để xem những gì đang xảy ra chi tiết hơn, chúng ta hãy mở rộng $\mathbf{v}_t$ đệ quy vào 

$$\begin{aligned}
\mathbf{v}_t = \beta^2 \mathbf{v}_{t-2} + \beta \mathbf{g}_{t-1, t-2} + \mathbf{g}_{t, t-1}
= \ldots, = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}.
\end{aligned}$$

$\beta$ lớn lên tới mức trung bình tầm xa, trong khi $\beta$ nhỏ chỉ có một sự điều chỉnh nhẹ so với phương pháp gradient. Sự thay thế gradient mới không còn trỏ vào hướng dốc nhất trên một trường hợp cụ thể nữa mà theo hướng trung bình có trọng số của các gradient trong quá khứ. Điều này cho phép chúng tôi nhận ra hầu hết các lợi ích của việc trung bình trong một lô mà không có chi phí thực sự tính toán độ dốc trên đó. Chúng tôi sẽ xem lại quy trình trung bình này chi tiết hơn sau. 

Lý luận trên hình thành cơ sở cho những gì bây giờ được gọi là phương pháp gradient *accelerated*, chẳng hạn như gradient với đà. Họ tận hưởng lợi ích bổ sung là hiệu quả hơn nhiều trong trường hợp vấn đề tối ưu hóa là không điều kiện (tức là, nơi có một số hướng mà sự tiến bộ chậm hơn nhiều so với những người khác, giống như một hẻm núi hẹp). Hơn nữa, chúng cho phép chúng ta trung bình trên các gradient tiếp theo để có được hướng xuống ổn định hơn. Thật vậy, khía cạnh của khả năng tăng tốc ngay cả đối với các vấn đề lồi không ồn là một trong những lý do chính khiến động lượng hoạt động và tại sao nó hoạt động tốt như vậy. 

Như người ta mong đợi, do động lực hiệu quả của nó là một môn học được nghiên cứu tốt trong tối ưu hóa cho học sâu và hơn thế nữa. Xem ví dụ, đẹp [bài viết expository](https://distill.pub/2017/momentum/) by :cite:`Goh.2017` để phân tích chuyên sâu và hoạt hình tương tác. Nó được đề xuất bởi :cite:`Polyak.1964`. :cite:`Nesterov.2018` có một cuộc thảo luận lý thuyết chi tiết trong bối cảnh tối ưu hóa lồi. Đà trong học sâu đã được biết đến là có lợi trong một thời gian dài. Xem ví dụ, cuộc thảo luận của :cite:`Sutskever.Martens.Dahl.ea.2013` để biết chi tiết. 

### Một vấn đề Ill-conditioned

Để hiểu rõ hơn về các thuộc tính hình học của phương pháp động lượng, chúng tôi xem lại gradient descent, mặc dù với một chức năng khách quan ít dễ chịu hơn đáng kể. Nhớ lại rằng trong :numref:`sec_gd`, chúng tôi đã sử dụng $f(\mathbf{x}) = x_1^2 + 2 x_2^2$, tức là, một mục tiêu ellipsoid biến dạng vừa phải. Chúng tôi làm biến dạng chức năng này hơn nữa bằng cách kéo dài nó theo hướng $x_1$ thông qua 

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

Như trước $f$ có mức tối thiểu là $(0, 0)$. Chức năng này là * rấy* phẳng theo hướng $x_1$. Chúng ta hãy xem những gì sẽ xảy ra khi chúng ta thực hiện gradient gốc như trước đây trên chức năng mới này. Chúng tôi chọn một tỷ lệ học tập là $0.4$.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

Bằng cách xây dựng, gradient theo hướng $x_2$ là * nhiều* cao hơn và thay đổi nhanh hơn nhiều so với theo chiều ngang $x_1$. Do đó, chúng tôi đang bị mắc kẹt giữa hai lựa chọn không mong muốn: nếu chúng tôi chọn một tốc độ học tập nhỏ, chúng tôi đảm bảo rằng giải pháp không phân kỳ theo hướng $x_2$ nhưng chúng tôi đang yên tâm với sự hội tụ chậm theo hướng $x_1$. Ngược lại, với tốc độ học tập lớn, chúng tôi tiến bộ nhanh chóng theo hướng $x_1$ nhưng phân kỳ trong $x_2$. Ví dụ dưới đây minh họa những gì xảy ra ngay cả sau khi tăng nhẹ tỷ lệ học tập từ $0.4$ lên $0.6$. Sự hội tụ theo hướng $x_1$ được cải thiện nhưng chất lượng giải pháp tổng thể tồi tệ hơn nhiều.

```{.python .input}
#@tab all
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

### Phương pháp Momentum

Phương pháp động lượng cho phép chúng ta giải quyết vấn đề gốc gradient được mô tả ở trên. Nhìn vào dấu vết tối ưu hóa ở trên, chúng ta có thể cảm thấy rằng độ dốc trung bình trong quá khứ sẽ hoạt động tốt. Rốt cuộc, theo hướng $x_1$, điều này sẽ tổng hợp các gradient được căn chỉnh tốt, do đó làm tăng khoảng cách chúng ta bao gồm mỗi bước. Ngược lại, theo hướng $x_2$ nơi gradient dao động, một gradient tổng hợp sẽ làm giảm kích thước bước do dao động hủy bỏ lẫn nhau ra. Sử dụng $\mathbf{v}_t$ thay vì gradient $\mathbf{g}_t$ mang lại các phương trình cập nhật sau: 

$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}, \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{v}_t.
\end{aligned}
$$

Lưu ý rằng đối với $\beta = 0$, chúng tôi phục hồi độ dốc thường xuyên. Trước khi nghiên cứu sâu hơn vào các thuộc tính toán học, chúng ta hãy nhìn nhanh cách thuật toán hoạt động trong thực tế.

```{.python .input}
#@tab all
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

Như chúng ta có thể thấy, ngay cả với cùng tốc độ học tập mà chúng ta đã sử dụng trước đây, đà vẫn hội tụ tốt. Hãy để chúng tôi xem những gì xảy ra khi chúng ta giảm tham số động lượng. Giảm một nửa nó xuống $\beta = 0.25$ dẫn đến một quỹ đạo hầu như không hội tụ ở tất cả. Tuy nhiên, nó tốt hơn rất nhiều so với không có đà (khi giải pháp phân kỳ).

```{.python .input}
#@tab all
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

Lưu ý rằng chúng ta có thể kết hợp động lượng với gốc gradient ngẫu nhiên và đặc biệt, minibatch stochastic gradient descent. Thay đổi duy nhất là trong trường hợp đó, chúng tôi thay thế gradient $\mathbf{g}_{t, t-1}$ bằng $\mathbf{g}_t$. Cuối cùng, để thuận tiện, chúng tôi khởi tạo $\mathbf{v}_0 = 0$ tại thời điểm $t=0$. Chúng ta hãy nhìn vào những gì rò rỉ trung bình thực sự làm cho các bản cập nhật. 

### Trọng lượng mẫu hiệu quả

Nhớ lại rằng $\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}$. Trong giới hạn, các điều khoản thêm lên đến $\sum_{\tau=0}^\infty \beta^\tau = \frac{1}{1-\beta}$. Nói cách khác, thay vì thực hiện một bước có kích thước $\eta$ trong gradient gốc hoặc gốc gradient ngẫu nhiên, chúng tôi thực hiện một bước có kích thước $\frac{\eta}{1-\beta}$ trong khi đồng thời, đối phó với một hướng gốc có khả năng tốt hơn nhiều. Đây là hai lợi ích trong một. Để minh họa cách thức hoạt động của trọng số cho các lựa chọn khác nhau của $\beta$ hãy xem xét sơ đồ dưới đây.

```{.python .input}
#@tab all
d2l.set_figsize()
betas = [0.95, 0.9, 0.6, 0]
for beta in betas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

## Các thí nghiệm thực tế

Chúng ta hãy xem động lượng hoạt động như thế nào trong thực tế, tức là, khi được sử dụng trong bối cảnh của một trình tối ưu hóa thích hợp. Đối với điều này, chúng ta cần một triển khai có thể mở rộng hơn một chút. 

### Thực hiện từ đầu

So với (minibatch) stochastic gradient gốc phương pháp động lượng cần duy trì một tập hợp các biến phụ trợ, tức là vận tốc. Nó có hình dạng tương tự như gradient (và các biến của bài toán tối ưu hóa). Trong việc thực hiện dưới đây, chúng tôi gọi các biến này là `states`.

```{.python .input}
#@tab mxnet,pytorch
def init_momentum_states(feature_dim):
    v_w = d2l.zeros((feature_dim, 1))
    v_b = d2l.zeros(1)
    return (v_w, v_b)
```

```{.python .input}
#@tab tensorflow
def init_momentum_states(features_dim):
    v_w = tf.Variable(d2l.zeros((features_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    return (v_w, v_b)
```

```{.python .input}
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum'] * v + p.grad
        p[:] -= hyperparams['lr'] * v
```

```{.python .input}
#@tab pytorch
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd_momentum(params, grads, states, hyperparams):
    for p, v, g in zip(params, states, grads):
            v[:].assign(hyperparams['momentum'] * v + g)
            p[:].assign(p - hyperparams['lr'] * v)
```

Hãy để chúng tôi xem làm thế nào điều này hoạt động trong thực tế.

```{.python .input}
#@tab all
def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
train_momentum(0.02, 0.5)
```

Khi chúng ta tăng siêu tham số động lượng `momentum` lên 0,9, nó sẽ lên tới kích thước mẫu hiệu quả lớn hơn đáng kể là $\frac{1}{1 - 0.9} = 10$. Chúng tôi giảm tỷ lệ học tập một chút xuống $0.01$ để kiểm soát các vấn đề.

```{.python .input}
#@tab all
train_momentum(0.01, 0.9)
```

Giảm tỷ lệ học tập tiếp tục giải quyết bất kỳ vấn đề nào về các vấn đề tối ưu hóa không trơn tru. Đặt nó thành $0.005$ mang lại các đặc tính hội tụ tốt.

```{.python .input}
#@tab all
train_momentum(0.005, 0.9)
```

### Thực hiện ngắn gọn

Có rất ít việc phải làm trong Gluon kể từ khi bộ giải `sgd` tiêu chuẩn đã có động lực tích hợp. Thiết lập các tham số phù hợp mang lại một quỹ đạo rất giống nhau.

```{.python .input}
d2l.train_concise_ch11('sgd', {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD
d2l.train_concise_ch11(trainer, {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

## Phân tích lý thuyết

Cho đến nay, ví dụ 2D của $f(x) = 0.1 x_1^2 + 2 x_2^2$ dường như khá contrived. Bây giờ chúng ta sẽ thấy rằng điều này thực sự khá đại diện cho các loại vấn đề mà người ta có thể gặp phải, ít nhất là trong trường hợp giảm thiểu các hàm khách quan bậc hai lồi. 

### Hàm lồi bậc hai

Xem xét các chức năng 

$$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b.$$

Đây là một hàm bậc hai chung. Đối với ma trận xác định dương $\mathbf{Q} \succ 0$, tức là, đối với ma trận có giá trị eigenvalues dương, điều này có bộ giảm thiểu ở $\mathbf{x}^* = -\mathbf{Q}^{-1} \mathbf{c}$ với giá trị tối thiểu $b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$. Do đó chúng tôi có thể viết lại $h$ như 

$$h(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})^\top \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c}) + b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}.$$

Gradient được đưa ra bởi $\partial_{\mathbf{x}} f(\mathbf{x}) = \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$. Đó là, nó được đưa ra bởi khoảng cách giữa $\mathbf{x}$ và bộ thu nhỏ, nhân với $\mathbf{Q}$. Do đó, động lượng cũng là sự kết hợp tuyến tính của các thuật ngữ $\mathbf{Q} (\mathbf{x}_t - \mathbf{Q}^{-1} \mathbf{c})$. 

Kể từ $\mathbf{Q}$ là xác định dương, nó có thể được phân hủy thành hệ thống eigencủa nó thông qua $\mathbf{Q} = \mathbf{O}^\top \boldsymbol{\Lambda} \mathbf{O}$ cho một ma trận trực giao (xoay) $\mathbf{O}$ và một ma trận chéo $\boldsymbol{\Lambda}$ của eigenvalues dương. Điều này cho phép chúng ta thực hiện thay đổi các biến từ $\mathbf{x}$ thành $\mathbf{z} := \mathbf{O} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$ để có được một biểu thức đơn giản hóa nhiều: 

$$h(\mathbf{z}) = \frac{1}{2} \mathbf{z}^\top \boldsymbol{\Lambda} \mathbf{z} + b'.$$

đây $b' = b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$. Vì $\mathbf{O}$ chỉ là một ma trận trực giao, điều này không làm xáo trộn độ dốc một cách có ý nghĩa. Thể hiện trong điều khoản của $\mathbf{z}$ gradient gốc trở thành 

$$\mathbf{z}_t = \mathbf{z}_{t-1} - \boldsymbol{\Lambda} \mathbf{z}_{t-1} = (\mathbf{I} - \boldsymbol{\Lambda}) \mathbf{z}_{t-1}.$$

Thực tế quan trọng trong biểu thức này là gradient descent * không trộn xúc* giữa các eigenspace khác nhau. Đó là, khi được thể hiện theo hệ thống eigensystem của $\mathbf{Q}$, vấn đề tối ưu hóa tiến hành theo cách phối hợp khôn ngoan. Điều này cũng giữ cho đà. 

$$\begin{aligned}
\mathbf{v}_t & = \beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1} \\
\mathbf{z}_t & = \mathbf{z}_{t-1} - \eta \left(\beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1}\right) \\
    & = (\mathbf{I} - \eta \boldsymbol{\Lambda}) \mathbf{z}_{t-1} - \eta \beta \mathbf{v}_{t-1}.
\end{aligned}$$

Khi làm điều này chúng ta chỉ chứng minh định lý sau: Gradient Descent có và không có động lượng cho một hàm bậc hai lồi phân hủy thành tối ưu hóa phối hợp khôn ngoan theo hướng của eigenvectors của ma trận bậc hai. 

### Hàm vô hướng

Với kết quả trên chúng ta hãy xem những gì sẽ xảy ra khi chúng ta giảm thiểu chức năng $f(x) = \frac{\lambda}{2} x^2$. Đối với gradient descent chúng tôi có 

$$x_{t+1} = x_t - \eta \lambda x_t = (1 - \eta \lambda) x_t.$$

Bất cứ khi nào $|1 - \eta \lambda| < 1$ tối ưu hóa này hội tụ với tốc độ theo cấp số nhân kể từ sau $t$ bước, chúng tôi có $x_t = (1 - \eta \lambda)^t x_0$. Điều này cho thấy tốc độ hội tụ được cải thiện ban đầu như thế nào khi chúng ta tăng tỷ lệ học tập $\eta$ cho đến $\eta \lambda = 1$. Ngoài ra, mọi thứ phân kỳ và cho $\eta \lambda > 2$ vấn đề tối ưu hóa phân kỳ.

```{.python .input}
#@tab all
lambdas = [0.1, 1, 10, 19]
eta = 0.1
d2l.set_figsize((6, 4))
for lam in lambdas:
    t = d2l.numpy(d2l.arange(20))
    d2l.plt.plot(t, (1 - eta * lam) ** t, label=f'lambda = {lam:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

Để phân tích sự hội tụ trong trường hợp động lượng, chúng ta bắt đầu bằng cách viết lại phương trình cập nhật theo hai vô hướng: một cho $x$ và một cho động lượng $v$. Điều này mang lại: 

$$
\begin{bmatrix} v_{t+1} \\ x_{t+1} \end{bmatrix} =
\begin{bmatrix} \beta & \lambda \\ -\eta \beta & (1 - \eta \lambda) \end{bmatrix}
\begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix} = \mathbf{R}(\beta, \eta, \lambda) \begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix}.
$$

Chúng tôi đã sử dụng $\mathbf{R}$ để biểu thị $2 \times 2$ điều chỉnh hành vi hội tụ. Sau $t$ bước sự lựa chọn ban đầu $[v_0, x_0]$ trở thành $\mathbf{R}(\beta, \eta, \lambda)^t [v_0, x_0]$. Do đó, tùy thuộc vào giá trị eigenvalues của $\mathbf{R}$ để xác định tốc độ hội tụ. Xem [Distill post](https://distill.pub/2017/momentum/) of :cite:`Goh.2017` cho một hình ảnh động tuyệt vời và :cite:`Flammarion.Bach.2015` để phân tích chi tiết. Người ta có thể cho thấy đà $0 < \eta \lambda < 2 + 2 \beta$ hội tụ. Đây là một phạm vi lớn hơn của các thông số khả thi khi so sánh với $0 < \eta \lambda < 2$ cho gradient gốc. Nó cũng cho thấy rằng nói chung các giá trị lớn của $\beta$ là mong muốn. Thêm chi tiết yêu cầu một số lượng hợp lý của chi tiết kỹ thuật và chúng tôi đề nghị người đọc quan tâm tham khảo ý kiến các ấn phẩm gốc. 

## Tóm tắt

* Momentum thay thế gradient với một trung bình bị rò rỉ trên gradient trong quá khứ. Điều này tăng tốc sự hội tụ đáng kể.
* Đó là mong muốn cho cả hai gốc gradient không có tiếng ồn và (ồn ào) stochastic gradient gốc.
* Momentum ngăn chặn sự trì trệ của quá trình tối ưu hóa có nhiều khả năng xảy ra cho gốc gradient ngẫu nhiên.
* Số gradient hiệu quả được đưa ra bởi $\frac{1}{1-\beta}$ do giảm trọng lượng theo cấp số của dữ liệu trong quá khứ.
* Trong trường hợp các bài toán bậc hai lồi, điều này có thể được phân tích một cách rõ ràng một cách chi tiết.
* Việc thực hiện khá đơn giản nhưng nó đòi hỏi chúng ta phải lưu trữ một vector trạng thái bổ sung (đà $\mathbf{v}$).

## Bài tập

1. Sử dụng các kết hợp khác của các siêu tham số động lượng và tỷ lệ học tập và quan sát và phân tích các kết quả thử nghiệm khác nhau.
1. Hãy thử GD và đà cho một vấn đề bậc hai nơi bạn có nhiều eigenvalues, tức là, $f(x) = \frac{1}{2} \sum_i \lambda_i x_i^2$, ví dụ, $\lambda_i = 2^{-i}$. Vẽ cách các giá trị của $x$ giảm cho khởi tạo $x_i = 1$.
1. Lấy giá trị tối thiểu và minimizer cho $h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b$.
1. Điều gì thay đổi khi chúng ta thực hiện chuyển đổi ngẫu nhiên gradient với đà? Điều gì xảy ra khi chúng ta sử dụng minibatch stochastic gradient descent với đà? Thử nghiệm với các thông số?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/354)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1070)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1071)
:end_tab:
