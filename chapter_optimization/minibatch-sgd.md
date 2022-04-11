# Minibatch Stochastic Gradient Descent
:label:`sec_minibatch_sgd`

Cho đến nay chúng ta đã gặp phải hai thái cực trong cách tiếp cận để gradient dựa learning: :numref:`sec_gd` sử dụng bộ dữ liệu đầy đủ để tính toán gradient và để cập nhật các tham số, một lần vượt qua tại một thời điểm. Ngược lại :numref:`sec_sgd` xử lý một quan sát tại một thời điểm để đạt được tiến bộ. Mỗi người trong số họ có nhược điểm riêng. Gradient Descent không đặc biệt* hiệu quả dữ liệu* bất cứ khi nào dữ liệu rất giống nhau. Stochastic Gradient Descent không đặc biệt* hiệu quả tính tế* vì CPU và GPU không thể khai thác toàn bộ sức mạnh của vectơ hóa. Điều này cho thấy rằng có thể có một phương tiện hạnh phúc, và trên thực tế, đó là những gì chúng tôi đã sử dụng cho đến nay trong các ví dụ chúng tôi đã thảo luận. 

## Vector hóa và bộ nhớ cache

Trọng tâm của quyết định sử dụng minibatches là hiệu quả tính toán. Điều này dễ hiểu nhất khi xem xét song song với nhiều GPU và nhiều máy chủ. Trong trường hợp này, chúng ta cần gửi ít nhất một hình ảnh cho mỗi GPU. Với 8 GPU trên mỗi máy chủ và 16 máy chủ, chúng tôi đã có kích thước minibatch là 128. 

Mọi thứ tinh tế hơn một chút khi nói đến GPU đơn lẻ hoặc thậm chí CPU. Các thiết bị này có nhiều loại bộ nhớ, thường là nhiều loại đơn vị tính toán và hạn chế băng thông khác nhau giữa chúng. Ví dụ, CPU có một số lượng nhỏ các thanh ghi và sau đó là L1, L2 và trong một số trường hợp thậm chí bộ nhớ cache L3 (được chia sẻ giữa các lõi bộ xử lý khác nhau). Các bộ nhớ đệm này có kích thước và độ trễ ngày càng tăng (đồng thời chúng giảm băng thông). Nó đủ để nói, bộ xử lý có khả năng thực hiện nhiều hoạt động hơn so với những gì giao diện bộ nhớ chính có thể cung cấp. 

* CPU 2GHz với 16 lõi và vectorization AVX-512 có thể xử lý lên đến $2 \cdot 10^9 \cdot 16 \cdot 32 = 10^{12}$ byte mỗi giây. Khả năng của GPU dễ dàng vượt quá con số này theo hệ số 100. Mặt khác, một bộ xử lý máy chủ tầm trung có thể không có nhiều hơn 100 Gb/s băng thông, tức là, ít hơn một phần mười những gì sẽ được yêu cầu để giữ cho bộ xử lý ăn. Để làm cho vấn đề tồi tệ hơn, không phải tất cả truy cập bộ nhớ được tạo ra bằng nhau: đầu tiên, giao diện bộ nhớ thường rộng 64 bit hoặc rộng hơn (ví dụ, trên GPU lên đến 384 bit), do đó đọc một byte duy nhất phải chịu chi phí truy cập rộng hơn nhiều.
* Có chi phí đáng kể cho truy cập đầu tiên trong khi truy cập tuần tự là tương đối rẻ (điều này thường được gọi là một lần đọc liên tục). Có rất nhiều điều cần lưu ý, chẳng hạn như bộ nhớ đệm khi chúng ta có nhiều ổ cắm, chiplet và các cấu trúc khác. Một cuộc thảo luận chi tiết về điều này nằm ngoài phạm vi của phần này. Xem ví dụ, [Wikipedia article](https://en.wikipedia.org/wiki/Cache_hierarchy) này để có một cuộc thảo luận chuyên sâu hơn.

Cách để giảm bớt những hạn chế này là sử dụng một hệ thống phân cấp của bộ nhớ cache CPU thực sự đủ nhanh để cung cấp cho bộ xử lý dữ liệu. Đây là * động lực là* đằng sau việc phân mẻ trong học sâu. Để giữ cho vấn đề đơn giản, hãy xem xét phép nhân ma trận ma trận, nói $\mathbf{A} = \mathbf{B}\mathbf{C}$. Chúng tôi có một số tùy chọn để tính toán $\mathbf{A}$. Ví dụ, chúng tôi có thể thử như sau: 

1. Chúng tôi có thể tính toán $\mathbf{A}_{ij} = \mathbf{B}_{i,:} \mathbf{C}_{:,j}^\top$, tức là, chúng tôi có thể tính toán nó elementwise bằng phương tiện của các sản phẩm chấm.
1. Chúng ta có thể tính toán $\mathbf{A}_{:,j} = \mathbf{B} \mathbf{C}_{:,j}^\top$, tức là, chúng ta có thể tính toán nó một cột tại một thời điểm. Tương tự như vậy chúng ta có thể tính $\mathbf{A}$ một hàng $\mathbf{A}_{i,:}$ tại một thời điểm.
1. We could simply đơn giản compute tính toán $\mathbf{A} = \mathbf{B} \mathbf{C}$.
1. Chúng ta có thể phá vỡ $\mathbf{B}$ và $\mathbf{C}$ thành các ma trận khối nhỏ hơn và tính toán $\mathbf{A}$ một khối tại một thời điểm.

Nếu chúng ta làm theo tùy chọn đầu tiên, chúng ta sẽ cần sao chép một hàng và một vectơ cột vào CPU mỗi khi chúng ta muốn tính một phần tử $\mathbf{A}_{ij}$. Thậm chí tệ hơn, do thực tế là các yếu tố ma trận được căn chỉnh tuần tự, do đó, chúng ta được yêu cầu truy cập nhiều vị trí tách rời cho một trong hai vectơ khi chúng ta đọc chúng từ bộ nhớ. Tùy chọn thứ hai thuận lợi hơn nhiều. Trong đó, chúng ta có thể giữ vector cột $\mathbf{C}_{:,j}$ trong bộ nhớ cache CPU trong khi chúng tôi tiếp tục đi qua $B$. Điều này giảm một nửa yêu cầu băng thông bộ nhớ với truy cập nhanh hơn tương ứng. Tất nhiên, tùy chọn 3 là mong muốn nhất. Thật không may, hầu hết các ma trận có thể không hoàn toàn phù hợp với bộ nhớ cache (đây là những gì chúng ta đang thảo luận sau khi tất cả). Tuy nhiên, tùy chọn 4 cung cấp một giải pháp thay thế thực tế hữu ích: chúng ta có thể di chuyển các khối ma trận vào bộ nhớ cache và nhân chúng cục bộ. Tối ưu hóa thư viện chăm sóc này cho chúng tôi. Chúng ta hãy xem các hoạt động này hiệu quả như thế nào trong thực tế. 

Ngoài hiệu quả tính toán, chi phí được giới thiệu bởi Python và bởi chính khuôn khổ học tập sâu là đáng kể. Nhớ lại rằng mỗi lần chúng ta thực hiện một lệnh, trình thông dịch Python sẽ gửi một lệnh đến công cụ MXNet cần chèn nó vào biểu đồ tính toán và xử lý nó trong quá trình lên lịch. Chi phí như vậy có thể khá bất lợi. Nói tóm lại, rất nên sử dụng vector hóa (và ma trận) bất cứ khi nào có thể.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

timer = d2l.Timer()
A = np.zeros((256, 256))
B = np.random.normal(0, 1, (256, 256))
C = np.random.normal(0, 1, (256, 256))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
import numpy as np

timer = d2l.Timer()
A = torch.zeros(256, 256)
B = torch.randn(256, 256)
C = torch.randn(256, 256)
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np

timer = d2l.Timer()
A = tf.Variable(d2l.zeros((256, 256)))
B = tf.Variable(d2l.normal([256, 256], 0, 1))
C = tf.Variable(d2l.normal([256, 256], 0, 1))
```

Nhiệm vụ yếu tố khôn ngoan chỉ đơn giản là lặp lại trên tất cả các hàng và cột của $\mathbf{B}$ và $\mathbf{C}$ tương ứng để gán giá trị cho $\mathbf{A}$.

```{.python .input}
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = np.dot(B[i, :], C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = torch.dot(B[i, :], C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j].assign(tf.tensordot(B[i, :], C[:, j], axes=1))
timer.stop()
```

Một chiến lược nhanh hơn là thực hiện gán cột khôn ngoan.

```{.python .input}
# Compute A = BC one column at a time
timer.start()
for j in range(256):
    A[:, j] = np.dot(B, C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Compute A = BC one column at a time
timer.start()
for j in range(256):
    A[:, j] = torch.mv(B, C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(256):
    A[:, j].assign(tf.tensordot(B, C[:, j], axes=1))
timer.stop()
```

Cuối cùng, cách hiệu quả nhất là thực hiện toàn bộ hoạt động trong một khối. Hãy để chúng tôi xem tốc độ tương ứng của các hoạt động là bao nhiêu.

```{.python .input}
# Compute A = BC in one go
timer.start()
A = np.dot(B, C)
A.wait_to_read()
timer.stop()

# Multiply and add count as separate operations (fused in practice)
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab pytorch
# Compute A = BC in one go
timer.start()
A = torch.mm(B, C)
timer.stop()

# Multiply and add count as separate operations (fused in practice)
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
A.assign(tf.tensordot(B, C, axes=1))
timer.stop()

# Multiply and add count as separate operations (fused in practice)
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

## Minibatches

:label:`sec_minibatches` 

Trong quá khứ, chúng tôi đã cho rằng chúng tôi sẽ đọc * minibatches* của dữ liệu chứ không phải là quan sát duy nhất để cập nhật các tham số. Bây giờ chúng tôi đưa ra một lời biện minh ngắn gọn cho nó. Xử lý các quan sát đơn yêu cầu chúng ta thực hiện nhiều phép nhân ma thuật-vector (hoặc thậm chí là vector-vector) đơn lẻ, khá tốn kém và phát sinh một chi phí đáng kể thay mặt cho khuôn khổ học sâu cơ bản. Điều này áp dụng cả để đánh giá một mạng khi áp dụng cho dữ liệu (thường được gọi là suy luận) và khi tính toán gradient để cập nhật các tham số. Đó là, điều này áp dụng bất cứ khi nào chúng tôi thực hiện $\mathbf{w} \leftarrow \mathbf{w} - \eta_t \mathbf{g}_t$ ở đâu 

$$\mathbf{g}_t = \partial_{\mathbf{w}} f(\mathbf{x}_{t}, \mathbf{w})$$

Chúng ta có thể tăng hiệu quả * tính tị* của thao tác này bằng cách áp dụng nó vào một loạt các quan sát tại một thời điểm. Đó là, chúng tôi thay thế gradient $\mathbf{g}_t$ trong một quan sát duy nhất bằng một trong một lô nhỏ 

$$\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w})$$

Chúng ta hãy xem điều này làm gì với các thuộc tính thống kê của $\mathbf{g}_t$: vì cả $\mathbf{x}_t$ và tất cả các yếu tố của minibatch $\mathcal{B}_t$ được vẽ đồng đều ngẫu nhiên từ bộ đào tạo, kỳ vọng của gradient vẫn không thay đổi. Mặt khác, phương sai được giảm đáng kể. Kể từ khi gradient minibatch bao gồm $b := |\mathcal{B}_t|$ gradient độc lập đang được trung bình, độ lệch chuẩn của nó được giảm bởi một hệ số $b^{-\frac{1}{2}}$. Bản thân nó, điều này là một điều tốt, vì nó có nghĩa là các bản cập nhật được liên kết đáng tin cậy hơn với gradient đầy đủ. 

Ngây thơ điều này sẽ chỉ ra rằng việc lựa chọn một minibatch lớn $\mathcal{B}_t$ sẽ là mong muốn phổ biến. Than ôi, sau một số điểm, việc giảm thêm độ lệch chuẩn là tối thiểu khi so sánh với sự gia tăng tuyến tính trong chi phí tính toán. Trong thực tế, chúng tôi chọn một minibatch đủ lớn để mang lại hiệu quả tính toán tốt trong khi vẫn phù hợp với bộ nhớ của GPU. Để minh họa tiết kiệm chúng ta hãy xem xét một số mã. Trong đó chúng ta thực hiện cùng một phép nhân ma trận ma trận, nhưng lần này chia thành “minibatches” của 64 cột tại một thời điểm.

```{.python .input}
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = np.dot(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab pytorch
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64].assign(tf.tensordot(B, C[:, j:j+64], axes=1))
timer.stop()
print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')
```

Như chúng ta có thể thấy, tính toán trên minibatch về cơ bản là hiệu quả như trên ma trận đầy đủ. Một lời thận trọng là theo thứ tự. Trong :numref:`sec_batch_norm`, chúng tôi đã sử dụng một loại chính quy hóa phụ thuộc nhiều vào số lượng phương sai trong một minibatch. Khi chúng ta tăng sau này, phương sai giảm và cùng với nó là lợi ích của việc phun tiếng ồn do bình thường hóa hàng loạt. Xem ví dụ, :cite:`Ioffe.2017` để biết chi tiết về cách giải thích và tính toán các điều khoản thích hợp. 

## Đọc tập dữ liệu

Chúng ta hãy xem cách minibatches được tạo ra hiệu quả từ dữ liệu. Sau đây chúng tôi sử dụng một tập dữ liệu do NASA phát triển để kiểm tra cánh [noise from different aircraft](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise) để so sánh các thuật toán tối ưu hóa này. Để thuận tiện, chúng tôi chỉ sử dụng các ví dụ $1,500$ đầu tiên. Dữ liệu được làm trắng để xử lý trước, tức là, chúng tôi loại bỏ trung bình và giải thích phương sai thành $1$ cho mỗi tọa độ.

```{.python .input}
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array(
        (data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab pytorch
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab tensorflow
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

## Thực hiện từ đầu

Nhớ lại việc thực hiện giảm dần dần ngẫu nhiên minibatch từ :numref:`sec_linear_scratch`. Trong phần sau đây, chúng tôi cung cấp một thực hiện tổng quát hơn một chút. Để thuận tiện, nó có chữ ký cuộc gọi giống như các thuật toán tối ưu hóa khác được giới thiệu sau trong chương này. Cụ thể, chúng tôi thêm đầu vào trạng thái `states` và đặt siêu tham số trong từ điển `hyperparams`. Ngoài ra, chúng tôi sẽ trung bình mất mỗi ví dụ minibatch trong chức năng đào tạo, do đó gradient trong thuật toán tối ưu hóa không cần phải chia cho kích thước lô.

```{.python .input}
def sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad
```

```{.python .input}
#@tab pytorch
def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, states, hyperparams):
    for param, grad in zip(params, grads):
        param.assign_sub(hyperparams['lr']*grad)
```

Tiếp theo, chúng tôi thực hiện một chức năng đào tạo chung để tạo điều kiện cho việc sử dụng các thuật toán tối ưu hóa khác được giới thiệu sau này trong chương này. Nó khởi tạo một mô hình hồi quy tuyến tính và có thể được sử dụng để đào tạo mô hình với minibatch stochastic gradient gốc và các thuật toán khác được giới thiệu sau đó.

```{.python .input}
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = np.random.normal(scale=0.01, size=(feature_dim, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab pytorch
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab tensorflow
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = tf.Variable(tf.random.normal(shape=(feature_dim, 1),
                                   mean=0, stddev=0.01),trainable=True)
    b = tf.Variable(tf.zeros(1), trainable=True)

    # Train
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()

    for _ in range(num_epochs):
        for X, y in data_iter:
          with tf.GradientTape() as g:
            l = tf.math.reduce_mean(loss(net(X), y))

          dw, db = g.gradient(l, [w, b])
          trainer_fn([w, b], [dw, db], states, hyperparams)
          n += X.shape[0]
          if n % 200 == 0:
              timer.stop()
              p = n/X.shape[0]
              q = p/tf.data.experimental.cardinality(data_iter).numpy()
              r = (d2l.evaluate_loss(net, data_iter, loss),)
              animator.add(q, r)
              timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

Hãy để chúng tôi xem tối ưu hóa tiến hành như thế nào để giảm gradient hàng loạt. Điều này có thể đạt được bằng cách đặt kích thước minibatch thành 1500 (tức là tổng số ví dụ). Kết quả là các thông số mô hình chỉ được cập nhật một lần cho mỗi kỷ nguyên. Có rất ít tiến bộ. Trong thực tế, sau 6 bước tiến trình quầy hàng.

```{.python .input}
#@tab all
def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

gd_res = train_sgd(1, 1500, 10)
```

Khi kích thước lô bằng 1, chúng ta sử dụng stochastic gradient descent để tối ưu hóa. Để đơn giản thực hiện, chúng tôi đã chọn một tỷ lệ học tập liên tục (mặc dù nhỏ). Trong dòng dốc ngẫu nhiên, các tham số mô hình được cập nhật bất cứ khi nào một ví dụ được xử lý. Trong trường hợp của chúng tôi, điều này lên tới 1500 bản cập nhật mỗi kỷ nguyên. Như chúng ta có thể thấy, sự suy giảm giá trị của hàm mục tiêu chậm lại sau một kỷ nguyên. Mặc dù cả hai quy trình đã xử lý 1500 ví dụ trong một kỷ nguyên, dòng dốc ngẫu nhiên tiêu thụ nhiều thời gian hơn so với gradient gốc trong thí nghiệm của chúng tôi. Điều này là do stochastic gradient gốc cập nhật các thông số thường xuyên hơn và vì nó kém hiệu quả hơn để xử lý các quan sát duy nhất tại một thời điểm.

```{.python .input}
#@tab all
sgd_res = train_sgd(0.005, 1)
```

Cuối cùng, khi kích thước lô bằng 100, chúng ta sử dụng minibatch stochastic gradient descent để tối ưu hóa. Thời gian cần thiết cho mỗi kỷ nguyên ngắn hơn thời gian cần thiết cho dòng gradient stochastic và thời gian để giảm gradient hàng loạt.

```{.python .input}
#@tab all
mini1_res = train_sgd(.4, 100)
```

Giảm kích thước lô xuống còn 10, thời gian cho mỗi kỷ nguyên tăng vì khối lượng công việc cho mỗi lô kém hiệu quả hơn để thực hiện.

```{.python .input}
#@tab all
mini2_res = train_sgd(.05, 10)
```

Bây giờ chúng ta có thể so sánh thời gian so với mất mát cho bốn thí nghiệm trước đó. Như có thể thấy, mặc dù stochastic gradient descent hội tụ nhanh hơn GD về số lượng ví dụ được xử lý, nó sử dụng nhiều thời gian hơn để đạt được tổn thất tương tự so với GD vì tính toán ví dụ gradient bằng ví dụ không hiệu quả như vậy. Minibatch stochastic gradient gốc có thể đánh đổi tốc độ hội tụ và hiệu quả tính toán. Kích thước minibatch là 10 hiệu quả hơn so với dòng gradient stochastic; kích thước minibatch 100 thậm chí vượt trội hơn GD về thời gian chạy.

```{.python .input}
#@tab all
d2l.set_figsize([6, 3])
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10],
         legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
d2l.plt.gca().set_xscale('log')
```

## Thực hiện ngắn gọn

Trong Gluon, chúng ta có thể sử dụng lớp `Trainer` để gọi các thuật toán tối ưu hóa. Điều này được sử dụng để thực hiện một chức năng đào tạo chung. Chúng tôi sẽ sử dụng điều này trong suốt chương hiện tại.

```{.python .input}
#@save
def train_concise_ch11(tr_name, hyperparams, data_iter, num_epochs=2):
    # Initialization
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    trainer = gluon.Trainer(net.collect_params(), tr_name, hyperparams)
    loss = gluon.loss.L2Loss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(X.shape[0])
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
```

```{.python .input}
#@tab pytorch
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    # Initialization
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                # `MSELoss` computes squared error without the 1/2 factor
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
```

```{.python .input}
#@tab tensorflow
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=2):
    # Initialization
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    optimizer = trainer_fn(**hyperparams)
    loss = tf.keras.losses.MeanSquaredError()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with tf.GradientTape() as g:
                out = net(X)
                l = loss(y, out)
                params = net.trainable_variables
                grads = g.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                p = n/X.shape[0]
                q = p/tf.data.experimental.cardinality(data_iter).numpy()
                # `MeanSquaredError` computes squared error without the 1/2
                # factor
                r = (d2l.evaluate_loss(net, data_iter, loss) / 2,)
                animator.add(q, r)
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
```

Sử dụng Gluon để lặp lại thí nghiệm cuối cùng cho thấy hành vi giống hệt nhau.

```{.python .input}
data_iter, _ = get_data_ch11(10)
train_concise_ch11('sgd', {'learning_rate': 0.05}, data_iter)
```

```{.python .input}
#@tab pytorch
data_iter, _ = get_data_ch11(10)
trainer = torch.optim.SGD
train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
data_iter, _ = get_data_ch11(10)
trainer = tf.keras.optimizers.SGD
train_concise_ch11(trainer, {'learning_rate': 0.05}, data_iter)
```

## Tóm tắt

* Vectorization làm cho mã hiệu quả hơn do giảm chi phí phát sinh từ khung học sâu và do địa phương bộ nhớ tốt hơn và bộ nhớ đệm trên CPU và GPU.
* Có một sự đánh đổi giữa hiệu quả thống kê phát sinh từ gốc gradient ngẫu nhiên và hiệu quả tính toán phát sinh từ việc xử lý các lô dữ liệu lớn tại một thời điểm.
* Minibatch stochastic gradient descent cung cấp tốt nhất của cả hai thế giới: tính toán và hiệu quả thống kê.
* Trong minibatch stochastic gradient descent, chúng tôi xử lý các lô dữ liệu thu được bằng một hoán vị ngẫu nhiên của dữ liệu đào tạo (tức là, mỗi quan sát chỉ được xử lý một lần cho mỗi kỷ nguyên, mặc dù theo thứ tự ngẫu nhiên).
* Đó là khuyến khích để phân rã tỷ lệ học tập trong quá trình đào tạo.
* Nói chung, minibatch stochastic gradient gốc nhanh hơn so với stochastic gradient descent và gradient descent để hội tụ đến một rủi ro nhỏ hơn, khi được đo về thời gian đồng hồ.

## Bài tập

1. Sửa đổi kích thước lô và tỷ lệ học tập và quan sát tốc độ suy giảm đối với giá trị của hàm khách quan và thời gian tiêu thụ trong mỗi kỷ nguyên.
1. Đọc tài liệu MXNet và sử dụng chức năng `Trainer` lớp `set_learning_rate` để giảm tốc độ học tập của gradient ngẫu nhiên minibatch xuống 1/10 giá trị trước đó của nó sau mỗi kỷ nguyên.
1. So sánh minibatch stochastic gradient descent với một biến thể mà thực sự * mẫu với thay thế* từ bộ đào tạo. Điều gì xảy ra?
1. Một vị thần ác sao chép tập dữ liệu của bạn mà không nói với bạn (tức là, mỗi quan sát xảy ra hai lần và tập dữ liệu của bạn phát triển lên gấp đôi kích thước ban đầu của nó, nhưng không ai nói với bạn). Làm thế nào để hành vi của stochastic gradient gốc, minibatch stochastic gradient descent và của gradient gốc thay đổi?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/353)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1068)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1069)
:end_tab:
