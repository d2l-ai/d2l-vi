# Adagrad
:label:`sec_adagrad`

Chúng ta hãy bắt đầu bằng cách xem xét các vấn đề học tập với các tính năng xảy ra không thường xuyên. 

## Tính năng thưa thớt và tỷ lệ học tập

Hãy tưởng tượng rằng chúng ta đang đào tạo một mô hình ngôn ngữ. Để có được độ chính xác tốt, chúng tôi thường muốn giảm tỷ lệ học tập khi chúng tôi tiếp tục đào tạo, thường ở tốc độ $\mathcal{O}(t^{-\frac{1}{2}})$ hoặc chậm hơn. Bây giờ hãy xem xét một đào tạo mô hình về các tính năng thưa thớt, tức là, các tính năng chỉ xảy ra không thường xuyên. Điều này là phổ biến cho ngôn ngữ tự nhiên, ví dụ, nó là rất ít khả năng rằng chúng ta sẽ thấy từ * preconditioning* so với *learning*. Tuy nhiên, nó cũng phổ biến ở các lĩnh vực khác như quảng cáo tính toán và lọc hợp tác được cá nhân hóa. Rốt cuộc, có rất nhiều điều chỉ được quan tâm đối với một số lượng nhỏ người. 

Các tham số liên quan đến các tính năng không thường xuyên chỉ nhận được cập nhật có ý nghĩa bất cứ khi nào các tính năng này xảy ra. Với tốc độ học tập giảm, chúng ta có thể kết thúc trong một tình huống mà các thông số cho các tính năng chung hội tụ khá nhanh đến các giá trị tối ưu của chúng, trong khi đối với các tính năng không thường xuyên, chúng ta vẫn thiếu quan sát chúng đủ thường xuyên trước khi các giá trị tối ưu của chúng có thể được xác định. Nói cách khác, tỷ lệ học tập giảm quá chậm đối với các tính năng thường xuyên hoặc quá nhanh đối với những người không thường xuyên. 

Một hack có thể để khắc phục vấn đề này sẽ là đếm số lần chúng ta thấy một tính năng cụ thể và sử dụng điều này làm đồng hồ để điều chỉnh tỷ lệ học tập. Đó là, thay vì chọn một tỷ lệ học tập của mẫu $\eta = \frac{\eta_0}{\sqrt{t + c}}$ chúng ta có thể sử dụng $\eta_i = \frac{\eta_0}{\sqrt{s(i, t) + c}}$. Ở đây $s(i, t)$ đếm số lượng nonzeros cho tính năng $i$ mà chúng tôi đã quan sát đến thời gian $t$. Điều này thực sự khá dễ thực hiện tại không có chi phí có ý nghĩa. Tuy nhiên, nó thất bại bất cứ khi nào chúng ta không hoàn toàn có độ thưa thớt mà chỉ là dữ liệu mà các gradient thường rất nhỏ và chỉ hiếm khi lớn. Rốt cuộc, không rõ nơi người ta sẽ vẽ đường giữa một cái gì đó đủ điều kiện là một tính năng quan sát hay không. 

Adagrad bởi :cite:`Duchi.Hazan.Singer.2011` giải quyết điều này bằng cách thay thế bộ đếm khá thô $s(i, t)$ bằng tổng hợp các ô vuông của gradient quan sát trước đó. Đặc biệt, nó sử dụng $s(i, t+1) = s(i, t) + \left(\partial_i f(\mathbf{x})\right)^2$ như một phương tiện để điều chỉnh tốc độ học tập. Điều này có hai lợi ích: đầu tiên, chúng ta không còn cần phải quyết định chỉ khi một gradient đủ lớn. Thứ hai, nó tự động quy mô với độ lớn của gradient. Các tọa độ thường xuyên tương ứng với các gradient lớn được thu nhỏ đáng kể, trong khi các tọa độ khác có độ dốc nhỏ sẽ được điều trị nhẹ nhàng hơn nhiều. Trong thực tế, điều này dẫn đến một thủ tục tối ưu hóa rất hiệu quả cho quảng cáo tính toán và các vấn đề liên quan. Nhưng điều này che giấu một số lợi ích bổ sung vốn có trong Adagrad được hiểu rõ nhất trong bối cảnh điều kiện tiên quyết. 

## Điều hòa trước

Vấn đề tối ưu hóa lồi là tốt cho việc phân tích các đặc tính của thuật toán. Rốt cuộc, đối với hầu hết các vấn đề không lồi, rất khó để lấy được những đảm bảo lý thuyết có ý nghĩa, nhưng * trực tuyến* và * cái nhìn sâu thắng* thường tiếp tục. Chúng ta hãy nhìn vào vấn đề giảm thiểu $f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x} + b$. 

Như chúng ta đã thấy trong :numref:`sec_momentum`, có thể viết lại vấn đề này về khả năng phân hủy $\mathbf{Q} = \mathbf{U}^\top \boldsymbol{\Lambda} \mathbf{U}$ để đến một vấn đề đơn giản hóa nhiều trong đó mỗi tọa độ có thể được giải quyết riêng lẻ: 

$$f(\mathbf{x}) = \bar{f}(\bar{\mathbf{x}}) = \frac{1}{2} \bar{\mathbf{x}}^\top \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}}^\top \bar{\mathbf{x}} + b.$$

Ở đây chúng tôi đã sử dụng $\mathbf{x} = \mathbf{U} \mathbf{x}$ và do đó $\mathbf{c} = \mathbf{U} \mathbf{c}$. Vấn đề sửa đổi có như minimizer của nó $\bar{\mathbf{x}} = -\boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}}$ và giá trị tối thiểu $-\frac{1}{2} \bar{\mathbf{c}}^\top \boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}} + b$. Điều này dễ tính toán hơn nhiều vì $\boldsymbol{\Lambda}$ là một ma trận chéo chứa các giá trị eigenvalues của $\mathbf{Q}$. 

Nếu chúng ta làm hỏng $\mathbf{c}$ một chút, chúng tôi hy vọng sẽ chỉ tìm thấy những thay đổi nhỏ trong bộ giảm thiểu $f$. Thật không may đây không phải là trường hợp. Trong khi những thay đổi nhỏ trong $\mathbf{c}$ dẫn đến những thay đổi nhỏ như nhau trong $\bar{\mathbf{c}}$, đây không phải là trường hợp cho minimizer $f$ (và $\bar{f}$ tương ứng). Bất cứ khi nào giá trị eigenvalues $\boldsymbol{\Lambda}_i$ lớn, chúng ta sẽ chỉ thấy những thay đổi nhỏ trong $\bar{x}_i$ và tối thiểu là $\bar{f}$. Ngược lại, đối với những thay đổi nhỏ $\boldsymbol{\Lambda}_i$ trong $\bar{x}_i$ có thể là kịch tính. Tỷ lệ giữa eigenvalue lớn nhất và nhỏ nhất được gọi là số điều kiện của một bài toán tối ưu hóa. 

$$\kappa = \frac{\boldsymbol{\Lambda}_1}{\boldsymbol{\Lambda}_d}.$$

Nếu điều kiện số $\kappa$ lớn, rất khó để giải quyết vấn đề tối ưu hóa một cách chính xác. Chúng ta cần đảm bảo rằng chúng ta cẩn thận trong việc có được một phạm vi động lớn các giá trị đúng. Phân tích của chúng tôi dẫn đến một câu hỏi rõ ràng, mặc dù hơi ngây thơ: chúng ta không thể chỉ đơn giản là “khắc phục” vấn đề bằng cách làm biến dạng không gian sao cho tất cả các giá trị eigenvalues là $1$. Về lý thuyết, điều này khá dễ dàng: chúng ta chỉ cần eigenvalues và eigenvectors của $\mathbf{Q}$ để giải phóng vấn đề từ $\mathbf{x}$ thành một trong $\mathbf{z} := \boldsymbol{\Lambda}^{\frac{1}{2}} \mathbf{U} \mathbf{x}$. Trong hệ tọa độ mới $\mathbf{x}^\top \mathbf{Q} \mathbf{x}$ có thể được đơn giản hóa thành $\|\mathbf{z}\|^2$. Than ôi, đây là một gợi ý khá không thực tế. Tính toán eigenvalues và eigenvectors nói chung là * nhiều hơn* đắt hơn so với giải quyết vấn đề thực tế. 

Trong khi tính toán eigenvalues chính xác có thể đắt tiền, đoán chúng và tính toán chúng thậm chí có phần xấp xỉ có thể đã tốt hơn rất nhiều so với không làm bất cứ điều gì cả. Đặc biệt, chúng ta có thể sử dụng các mục chéo của $\mathbf{Q}$ và giải thích nó cho phù hợp. Đây là * nhiều* rẻ hơn so với tính toán eigenvalues. 

$$\tilde{\mathbf{Q}} = \mathrm{diag}^{-\frac{1}{2}}(\mathbf{Q}) \mathbf{Q} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{Q}).$$

Trong trường hợp này, chúng tôi có $\tilde{\mathbf{Q}}_{ij} = \mathbf{Q}_{ij} / \sqrt{\mathbf{Q}_{ii} \mathbf{Q}_{jj}}$ và cụ thể là $\tilde{\mathbf{Q}}_{ii} = 1$ cho tất cả $i$. Trong hầu hết các trường hợp, điều này đơn giản hóa số điều kiện đáng kể. Ví dụ, các trường hợp chúng ta đã thảo luận trước đây, điều này sẽ loại bỏ hoàn toàn vấn đề trong tầm tay vì vấn đề được căn chỉnh trục. 

Thật không may, chúng ta phải đối mặt với một vấn đề khác: trong học sâu, chúng ta thường thậm chí không có quyền truy cập vào đạo hàm thứ hai của hàm khách quan: cho $\mathbf{x} \in \mathbb{R}^d$, phái sinh thứ hai ngay cả trên một minibatch có thể yêu cầu $\mathcal{O}(d^2)$ không gian và hoạt động để tính toán, do đó làm cho nó thực tế không khả thi. Ý tưởng khéo léo của Adagrad là sử dụng một proxy cho đường chéo khó nắm bắt của Hessian vừa tương đối rẻ để tính toán và hiệu quả—độ lớn của bản thân gradient. 

Để xem lý do tại sao điều này hoạt động, chúng ta hãy nhìn vào $\bar{f}(\bar{\mathbf{x}})$. Chúng tôi có điều đó 

$$\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}}) = \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}} = \boldsymbol{\Lambda} \left(\bar{\mathbf{x}} - \bar{\mathbf{x}}_0\right),$$

trong đó $\bar{\mathbf{x}}_0$ là minimizer của $\bar{f}$. Do đó độ lớn của gradient phụ thuộc cả vào $\boldsymbol{\Lambda}$ và khoảng cách từ độ tối ưu. Nếu $\bar{\mathbf{x}} - \bar{\mathbf{x}}_0$ không thay đổi, đây sẽ là tất cả những gì cần thiết. Rốt cuộc, trong trường hợp này độ lớn của gradient $\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}})$ đủ. Vì AdaGrad là một thuật toán gốc gradient ngẫu nhiên, chúng ta sẽ thấy gradient với phương sai nonzero ngay cả khi tối ưu. Kết quả là chúng ta có thể sử dụng phương sai của gradient một cách an toàn như một proxy giá rẻ cho thang đo của Hessian. Một phân tích kỹ lưỡng nằm ngoài phạm vi của phần này (nó sẽ là một số trang). Chúng tôi giới thiệu người đọc đến :cite:`Duchi.Hazan.Singer.2011` để biết chi tiết. 

## Các thuật toán

Hãy để chúng tôi chính thức hóa các cuộc thảo luận từ trên cao. Chúng ta sử dụng biến $\mathbf{s}_t$ để tích lũy phương sai gradient quá khứ như sau. 

$$\begin{aligned}
    \mathbf{g}_t & = \partial_{\mathbf{w}} l(y_t, f(\mathbf{x}_t, \mathbf{w})), \\
    \mathbf{s}_t & = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \\
    \mathbf{w}_t & = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \mathbf{g}_t.
\end{aligned}$$

Ở đây hoạt động được áp dụng tọa độ khôn ngoan. Đó là, $\mathbf{v}^2$ có mục $v_i^2$. Tương tự như vậy $\frac{1}{\sqrt{v}}$ có mục $\frac{1}{\sqrt{v_i}}$ và $\mathbf{u} \cdot \mathbf{v}$ có mục $u_i v_i$. Như trước $\eta$ là tốc độ học tập và $\epsilon$ là một hằng số phụ gia đảm bảo rằng chúng ta không chia cho $0$. Cuối cùng, chúng tôi khởi tạo $\mathbf{s}_0 = \mathbf{0}$. 

Cũng giống như trong trường hợp động lượng chúng ta cần phải theo dõi một biến phụ trợ, trong trường hợp này để cho phép một tỷ lệ học tập cá nhân trên mỗi tọa độ. Điều này không làm tăng chi phí của Adagrad đáng kể so với SGD, đơn giản vì chi phí chính thường là tính toán $l(y_t, f(\mathbf{x}_t, \mathbf{w}))$ và phái sinh của nó. 

Lưu ý rằng tích lũy gradient bình phương trong $\mathbf{s}_t$ có nghĩa là $\mathbf{s}_t$ phát triển cơ bản ở tốc độ tuyến tính (hơi chậm hơn tuyến tính trong thực tế, vì độ dốc ban đầu giảm dần). Điều này dẫn đến tỷ lệ học tập $\mathcal{O}(t^{-\frac{1}{2}})$, mặc dù được điều chỉnh trên cơ sở tọa độ. Đối với các vấn đề lồi, điều này là hoàn toàn đầy đủ. Tuy nhiên, trong học sâu, chúng ta có thể muốn giảm tốc độ học tập khá chậm hơn. Điều này dẫn đến một số biến thể Adagrad mà chúng ta sẽ thảo luận trong các chương tiếp theo. Bây giờ chúng ta hãy xem nó hoạt động như thế nào trong một bài toán lồi bậc hai. Chúng tôi sử dụng cùng một vấn đề như trước đây: 

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

Chúng tôi sẽ thực hiện Adagrad bằng cách sử dụng cùng một tốc độ học tập trước đây, tức là, $\eta = 0.4$. Như chúng ta có thể thấy, quỹ đạo lặp đi lặp lại của biến độc lập là mượt mà hơn. Tuy nhiên, do hiệu ứng tích lũy của $\boldsymbol{s}_t$, tốc độ học tập liên tục phân rã, do đó biến độc lập không di chuyển nhiều trong các giai đoạn lặp sau này.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input}
#@tab all
def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

Khi chúng ta tăng tỷ lệ học tập lên $2$, chúng ta thấy hành vi tốt hơn nhiều. Điều này đã chỉ ra rằng tỷ lệ học tập giảm có thể khá tích cực, ngay cả trong trường hợp không có tiếng ồn và chúng ta cần đảm bảo rằng các tham số hội tụ một cách thích hợp.

```{.python .input}
#@tab all
eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

## Thực hiện từ đầu

Cũng giống như phương pháp động lượng, Adagrad cần duy trì một biến trạng thái có hình dạng giống như các tham số.

```{.python .input}
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] += np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def init_adagrad_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)

def adagrad(params, grads, states, hyperparams):
    eps = 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(s + tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

So với thí nghiệm trong :numref:`sec_minibatch_sgd`, chúng tôi sử dụng tốc độ học tập lớn hơn để đào tạo mô hình.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
               {'lr': 0.1}, data_iter, feature_dim);
```

## Thực hiện ngắn gọn

Sử dụng phiên bản `Trainer` của thuật toán `adagrad`, chúng ta có thể gọi thuật toán Adagrad trong Gluon.

```{.python .input}
d2l.train_concise_ch11('adagrad', {'learning_rate': 0.1}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adagrad
d2l.train_concise_ch11(trainer, {'lr': 0.1}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adagrad
d2l.train_concise_ch11(trainer, {'learning_rate' : 0.1}, data_iter)
```

## Tóm tắt

* Adagrad giảm tốc độ học tập động trên cơ sở mỗi tọa độ.
* Nó sử dụng độ lớn của gradient như một phương tiện để điều chỉnh mức độ nhanh chóng đạt được tiến bộ - tọa độ với độ dốc lớn được bù với tốc độ học tập nhỏ hơn.
* Tính toán đạo hàm thứ hai chính xác thường không khả thi trong các bài toán học sâu do các ràng buộc về trí nhớ và tính toán. Gradient có thể là một proxy hữu ích.
* Nếu vấn đề tối ưu hóa có cấu trúc khá không đồng đều Adagrad có thể giúp giảm thiểu sự biến dạng.
* Adagrad đặc biệt hiệu quả đối với các tính năng thưa thớt, nơi tỷ lệ học tập cần giảm chậm hơn cho các thuật ngữ không thường xuyên xảy ra.
* Về các vấn đề học sâu Adagrad đôi khi có thể quá tích cực trong việc giảm tỷ lệ học tập. Chúng tôi sẽ thảo luận về các chiến lược để giảm thiểu điều này trong bối cảnh :numref:`sec_adam`.

## Bài tập

1. Chứng minh rằng đối với một ma trận trực giao $\mathbf{U}$ và một vector $\mathbf{c}$ các tổ chức sau đây: $\|\mathbf{c} - \mathbf{\delta}\|_2 = \|\mathbf{U} \mathbf{c} - \mathbf{U} \mathbf{\delta}\|_2$. Tại sao điều này có nghĩa là độ lớn của nhiễu loạn không thay đổi sau khi thay đổi trực giao của các biến?
1. Hãy thử Adagrad cho $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ và cũng cho chức năng mục tiêu đã được xoay 45 độ, tức là, $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$. Nó có cư xử khác nhau không?
1. Chứng minh [định lý vòng tròn Gerschgorin](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem) trong đó nói rằng eigenvalues $\lambda_i$ của một ma trận $\mathbf{M}$ thỏa mãn $|\lambda_i - \mathbf{M}_{jj}| \leq \sum_{k \neq j} |\mathbf{M}_{jk}|$ cho ít nhất một lựa chọn $j$.
1. Định lý Gerschgorin cho chúng ta biết gì về giá trị eigenvalues của ma trận điều hòa trước theo đường chéo $\mathrm{diag}^{-\frac{1}{2}}(\mathbf{M}) \mathbf{M} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{M})$?
1. Hãy thử Adagrad cho một mạng sâu thích hợp, chẳng hạn như :numref:`sec_lenet` khi áp dụng cho Fashion MNIST.
1. Làm thế nào bạn cần sửa đổi Adagrad để đạt được một sự phân rã ít hung hăng trong tốc độ học tập?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/355)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1072)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1073)
:end_tab:
