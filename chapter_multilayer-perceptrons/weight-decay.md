# Trọng lượng phân rã
:label:`sec_weight_decay`

Bây giờ chúng tôi đã đặc trưng vấn đề overfitting, chúng tôi có thể giới thiệu một số kỹ thuật tiêu chuẩn để điều chỉnh các mô hình. Nhớ lại rằng chúng ta luôn có thể giảm thiểu quá mức bằng cách đi ra ngoài và thu thập thêm dữ liệu đào tạo. Điều đó có thể tốn kém, tốn thời gian hoặc hoàn toàn nằm ngoài tầm kiểm soát của chúng tôi, khiến nó không thể xảy ra trong thời gian ngắn. Hiện tại, chúng ta có thể giả định rằng chúng ta đã có nhiều dữ liệu chất lượng cao như tài nguyên của chúng tôi cho phép và tập trung vào các kỹ thuật chính quy hóa. 

Nhớ lại rằng trong ví dụ hồi quy đa thức của chúng tôi (:numref:`sec_model_selection`), chúng ta có thể giới hạn năng lực của mô hình của chúng tôi chỉ đơn giản bằng cách điều chỉnh mức độ của đa thức được trang bị. Thật vậy, hạn chế số lượng các tính năng là một kỹ thuật phổ biến để giảm thiểu quá mức. Tuy nhiên, chỉ cần ném các tính năng sang một bên có thể quá cùn một công cụ cho công việc. Gắn bó với ví dụ hồi quy đa thức, xem xét những gì có thể xảy ra với các đầu vào chiều cao. Các phần mở rộng tự nhiên của đa thức cho dữ liệu đa biến được gọi là * monomials*, đơn giản là sản phẩm của quyền hạn của các biến. Mức độ của một monomial là tổng của các cường quốc. Ví dụ, $x_1^2 x_2$, và $x_3 x_5^2$ đều là nguyên khối của độ 3. 

Lưu ý rằng số lượng thuật ngữ với mức độ $d$ thổi lên nhanh chóng khi $d$ phát triển lớn hơn. Với $k$ biến số, số lượng monomials của độ $d$ (tức là $k$ multichoose $d$) là ${k - 1 + d} \choose {k - 1}$. Ngay cả những thay đổi nhỏ về mức độ, nói từ $2$ đến $3$, làm tăng đáng kể độ phức tạp của mô hình của chúng tôi. Vì vậy chúng ta thường cần một công cụ hạt mịn hơn để điều chỉnh độ phức tạp của chức năng. 

## Định mức và phân rã trọng lượng

Chúng tôi đã mô tả cả định mức $L_2$ và định mức $L_1$, đây là những trường hợp đặc biệt của định mức $L_p$ chung hơn trong :numref:`subsec_lin-algebra-norms`. (*** Phân rã trọng lượng* (thường được gọi là $L_2$ regarization), có thể là kỹ thuật được sử dụng rộng rãi nhất để điều chỉnh các mô hình học máy tham số.**) Kỹ thuật này được thúc đẩy bởi trực giác cơ bản trong số tất cả các chức năng $f$, chức năng $f = 0$ (gán giá trị $0$ cho tất cả các đầu vào) trong một số nghĩa là * đơn giản nhất*, và rằng chúng ta có thể đo lường sự phức tạp của một hàm bằng khoảng cách của nó từ 0. Nhưng chính xác như thế nào chúng ta nên đo khoảng cách giữa một hàm và không? Không có câu trả lời đúng duy nhất. Trên thực tế, toàn bộ nhánh toán học, bao gồm các phần phân tích chức năng và lý thuyết về không gian Banach, được dành để trả lời vấn đề này. 

Một cách giải thích đơn giản có thể là đo độ phức tạp của một hàm tuyến tính $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$ theo một số định mức của vectơ trọng lượng của nó, ví dụ, $\| \mathbf{w} \|^2$. Phương pháp phổ biến nhất để đảm bảo một vector trọng lượng nhỏ là thêm định mức của nó như một thuật ngữ phạt cho vấn đề giảm thiểu tổn thất. Do đó chúng tôi thay thế mục tiêu ban đầu của chúng tôi,
*giảm thiểu tổn thất dự đoán trên nhãn đào tạo*,
with new Mới objective mục tiêu,
*giảm thiểu tổng số tổn thất dự đoán và thuật ngữ phạt phút*.
Bây giờ, nếu vector trọng lượng của chúng ta phát triển quá lớn, thuật toán học tập của chúng ta có thể tập trung vào việc giảm thiểu định mức trọng lượng $\| \mathbf{w} \|^2$ so với giảm thiểu lỗi đào tạo. Đó chính xác là những gì chúng tôi muốn. Để minh họa những điều trong code, chúng ta hãy hồi sinh ví dụ trước đây của chúng tôi từ :numref:`sec_linear_regression` cho hồi quy tuyến tính. Ở đó, sự mất mát của chúng tôi đã được đưa ra bởi 

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

Nhớ lại rằng $\mathbf{x}^{(i)}$ là các tính năng, $y^{(i)}$ là nhãn cho tất cả các ví dụ dữ liệu $i$ và $(\mathbf{w}, b)$ là các thông số trọng lượng và thiên vị, tương ứng. Để phạt kích thước của vector trọng lượng, bằng cách nào đó chúng ta phải thêm $\| \mathbf{w} \|^2$ vào chức năng mất mát, nhưng làm thế nào mô hình nên thương mại mất tiêu chuẩn cho hình phạt phụ gia mới này? Trong thực tế, chúng tôi mô tả sự cân bằng này thông qua hằng số * thường xuyệt* $\lambda$, một siêu tham số không âm mà chúng tôi phù hợp bằng dữ liệu xác thực: 

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2,$$

Đối với $\lambda = 0$, chúng tôi phục hồi chức năng mất mát ban đầu của chúng tôi. Đối với $\lambda > 0$, chúng tôi hạn chế kích thước của $\| \mathbf{w} \|$. Chúng tôi chia cho $2$ theo quy ước: khi chúng ta lấy đạo hàm của một hàm bậc hai, $2$ và $1/2$ hủy bỏ, đảm bảo rằng biểu thức cho bản cập nhật trông đẹp và đơn giản. Người đọc tinh xảo có thể tự hỏi tại sao chúng ta làm việc với định mức bình phương chứ không phải định mức tiêu chuẩn (tức là khoảng cách Euclide). Chúng tôi làm điều này để thuận tiện tính toán. Bằng cách bình phương định mức $L_2$, chúng ta loại bỏ căn bậc hai, để lại tổng hình vuông của mỗi thành phần của vector trọng lượng. Điều này làm cho đạo hàm của hình phạt dễ tính toán: tổng các dẫn xuất bằng đạo hàm của tổng. 

Hơn nữa, bạn có thể hỏi tại sao chúng tôi làm việc với định mức $L_2$ ở nơi đầu tiên và không, nói, định mức $L_1$. Trên thực tế, các lựa chọn khác là hợp lệ và phổ biến trong suốt số liệu thống kê. Trong khi $L_2$-mô hình tuyến tính được điều chỉnh tạo thành thuật toán * hồi quy sườn núi cổ điển*, hồi quy tuyến tính $L_1$ là một mô hình cơ bản tương tự trong thống kê, thường được gọi là hồi quy lasso*. 

Một lý do để làm việc với định mức $L_2$ là nó đặt một hình phạt lớn trên các thành phần lớn của vector trọng lượng. Điều này làm thiên vị thuật toán học tập của chúng tôi đối với các mô hình phân phối trọng lượng đều trên một số lượng lớn các tính năng. Trong thực tế, điều này có thể làm cho chúng mạnh mẽ hơn với lỗi đo lường trong một biến duy nhất. Ngược lại, $L_1$ hình phạt dẫn đến các mô hình tập trung trọng lượng trên một tập hợp nhỏ các tính năng bằng cách xóa các trọng lượng khác về 0. Điều này được gọi là lựa chọn tính năng*, có thể mong muốn vì các lý do khác. 

Sử dụng cùng một ký hiệu trong :eqref:`eq_linreg_batch_update`, các bản cập nhật xuống dốc ngẫu nhiên minibatch cho hồi quy $L_2$-regarized theo sau: 

$$
\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}
$$

Như trước đây, chúng tôi cập nhật $\mathbf{w}$ dựa trên số tiền mà ước tính của chúng tôi khác với quan sát. Tuy nhiên, chúng tôi cũng thu nhỏ kích thước $\mathbf{w}$ về phía không. Đó là lý do tại sao phương pháp đôi khi được gọi là “phân rã trọng lượng”: chỉ cho thuật ngữ phạt, thuật toán tối ưu hóa của chúng tôi* phân tác* trọng lượng ở mỗi bước tập luyện. Trái ngược với lựa chọn tính năng, trọng lượng phân rã cung cấp cho chúng ta một cơ chế liên tục để điều chỉnh độ phức tạp của một hàm. Các giá trị nhỏ hơn của $\lambda$ tương ứng với ít bị hạn chế $\mathbf{w}$, trong khi các giá trị lớn hơn của $\lambda$ hạn chế $\mathbf{w}$ đáng kể hơn. 

Cho dù chúng ta bao gồm một hình phạt thiên vị tương ứng $b^2$ có thể thay đổi giữa các triển khai, và có thể thay đổi giữa các lớp của mạng thần kinh. Thông thường, chúng ta không thường xuyên hóa thuật ngữ thiên vị của lớp đầu ra của mạng. 

## High-Dimensional Linear Regression

Chúng ta có thể minh họa những lợi ích của phân rã trọng lượng thông qua một ví dụ tổng hợp đơn giản.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

Đầu tiên, chúng tôi [** tạo ra một số dữ liệu như trước**] 

(**$$y = 0.05 +\ sum_ {i = 1} ^d 0.01 x_i +\ epsilon\ text {where}\ epsilon\ sim\ mathcal {N} (0, 0.01^2) .$$**) 

Chúng tôi chọn nhãn của chúng tôi là một chức năng tuyến tính của đầu vào của chúng tôi, bị hỏng bởi tiếng ồn Gaussian với 0 trung bình và độ lệch chuẩn 0,01. Để làm cho các hiệu ứng của overfitting rõ rệt, chúng ta có thể tăng chiều của vấn đề của chúng tôi lên $d = 200$ và làm việc với một bộ đào tạo nhỏ chỉ chứa 20 ví dụ.

```{.python .input}
#@tab all
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
```

## Thực hiện từ đầu

Sau đây, chúng tôi sẽ thực hiện phân rã trọng lượng từ đầu, chỉ bằng cách thêm hình phạt $L_2$ bình phương vào hàm mục tiêu ban đầu. 

### [**Initializing Model Parameters**]

Đầu tiên, chúng ta sẽ định nghĩa một hàm để khởi tạo ngẫu nhiên các tham số model của chúng ta.

```{.python .input}
def init_params():
    w = np.random.normal(scale=1, size=(num_inputs, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    return [w, b]
```

```{.python .input}
#@tab pytorch
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]
```

```{.python .input}
#@tab tensorflow
def init_params():
    w = tf.Variable(tf.random.normal(mean=1, shape=(num_inputs, 1)))
    b = tf.Variable(tf.zeros(shape=(1, )))
    return [w, b]
```

### (**Định nghĩa $L_2$ Hình phạt Norm**)

Có lẽ cách thuận tiện nhất để thực hiện hình phạt này là để vuông tất cả các điều khoản tại chỗ và tổng hợp chúng lên.

```{.python .input}
def l2_penalty(w):
    return (w**2).sum() / 2
```

```{.python .input}
#@tab pytorch
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2
```

```{.python .input}
#@tab tensorflow
def l2_penalty(w):
    return tf.reduce_sum(tf.pow(w, 2)) / 2
```

### [**Xác định vòng đào tạo**]

Mã sau phù hợp với một mô hình trên bộ đào tạo và đánh giá nó trên bộ thử nghiệm. Mạng tuyến tính và tổn thất bình phương không thay đổi kể từ :numref:`chap_linear`, vì vậy chúng tôi sẽ chỉ nhập chúng qua `d2l.linreg` và `d2l.squared_loss`. Thay đổi duy nhất ở đây là mất mát của chúng tôi bây giờ bao gồm thời hạn phạt.

```{.python .input}
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                # The L2 norm penalty term has been added, and broadcasting
                # makes `l2_penalty(w)` a vector whose length is `batch_size`
                l = loss(net(X), y) + lambd * l2_penalty(w)
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', np.linalg.norm(w))
```

```{.python .input}
#@tab pytorch
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # The L2 norm penalty term has been added, and broadcasting
            # makes `l2_penalty(w)` a vector whose length is `batch_size`
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', torch.norm(w).item())
```

```{.python .input}
#@tab tensorflow
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                # The L2 norm penalty term has been added, and broadcasting
                # makes `l2_penalty(w)` a vector whose length is `batch_size`
                l = loss(net(X), y) + lambd * l2_penalty(w)
            grads = tape.gradient(l, [w, b])
            d2l.sgd([w, b], grads, lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', tf.norm(w).numpy())
```

### [**Đào tạo mà không cần Regularization**]

Bây giờ chúng ta chạy mã này với `lambd = 0`, vô hiệu hóa phân rã trọng lượng. Lưu ý rằng chúng tôi quá phù hợp xấu, giảm lỗi đào tạo nhưng không phải lỗi kiểm tra - một trường hợp sách giáo khoa của overfitting.

```{.python .input}
#@tab all
train(lambd=0)
```

### [**Sử dụng trọng lượng**]

Dưới đây, chúng tôi chạy với sự phân rã trọng lượng đáng kể. Lưu ý rằng lỗi đào tạo tăng nhưng lỗi thử nghiệm giảm. Đây chính xác là hiệu quả mà chúng ta mong đợi từ việc chính quy hóa.

```{.python .input}
#@tab all
train(lambd=3)
```

## [**Thiết tập**]

Bởi vì sự phân rã trọng lượng có mặt khắp nơi trong việc tối ưu hóa mạng thần kinh, khung học sâu làm cho nó đặc biệt thuận tiện, tích hợp phân rã trọng lượng vào thuật toán tối ưu hóa để dễ sử dụng kết hợp với bất kỳ chức năng mất mát nào. Hơn nữa, tích hợp này phục vụ một lợi ích tính toán, cho phép các thủ thuật thực hiện thêm phân rã trọng lượng vào thuật toán, mà không cần bất kỳ chi phí tính toán bổ sung nào. Vì phần phân rã trọng lượng của bản cập nhật chỉ phụ thuộc vào giá trị hiện tại của mỗi tham số, trình tối ưu hóa phải chạm vào từng tham số một lần.

:begin_tab:`mxnet`
Trong mã sau, chúng tôi chỉ định siêu tham số phân rã trọng lượng trực tiếp thông qua `wd` khi khởi tạo `Trainer` của chúng tôi. Theo mặc định, Gluon phân rã cả trọng lượng và thành kiến cùng một lúc. Lưu ý rằng siêu tham số `wd` sẽ được nhân với `wd_mult` khi cập nhật các tham số mô hình. Do đó, nếu chúng ta đặt `wd_mult` thành 0, tham số thiên vị $b$ sẽ không phân rã.
:end_tab:

:begin_tab:`pytorch`
Trong mã sau, chúng tôi chỉ định siêu tham số phân rã trọng lượng trực tiếp thông qua `weight_decay` khi khởi tạo trình tối ưu hóa của chúng tôi. Theo mặc định, PyTorch phân rã cả trọng lượng và thành kiến cùng một lúc. Ở đây chúng tôi chỉ đặt `weight_decay` cho trọng lượng, vì vậy tham số thiên vị $b$ sẽ không phân rã.
:end_tab:

:begin_tab:`tensorflow`
Trong đoạn code sau, chúng ta tạo ra một bộ điều chỉnh $L_2$ với siêu tham số phân rã trọng lượng `wd` và áp dụng nó vào lớp thông qua đối số `kernel_regularizer`.
:end_tab:

```{.python .input}
def train_concise(wd):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=1))
    loss = gluon.loss.L2Loss()
    num_epochs, lr = 100, 0.003
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'wd': wd})
    # The bias parameter has not decayed. Bias names generally end with "bias"
    net.collect_params('.*bias').setattr('wd_mult', 0)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', np.linalg.norm(net[0].weight.data()))
```

```{.python .input}
#@tab pytorch
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # The bias parameter has not decayed
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', net[0].weight.norm().item())
```

```{.python .input}
#@tab tensorflow
def train_concise(wd):
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Dense(
        1, kernel_regularizer=tf.keras.regularizers.l2(wd)))
    net.build(input_shape=(1, num_inputs))
    w, b = net.trainable_variables
    loss = tf.keras.losses.MeanSquaredError()
    num_epochs, lr = 100, 0.003
    trainer = tf.keras.optimizers.SGD(learning_rate=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                # `tf.keras` requires retrieving and adding the losses from
                # layers manually for custom training loop.
                l = loss(net(X), y) + net.losses
            grads = tape.gradient(l, net.trainable_variables)
            trainer.apply_gradients(zip(grads, net.trainable_variables))
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', tf.norm(net.get_weights()[0]).numpy())
```

[**Các ô trông giống hệt với những lô khi chúng tôi thực hiện phân rã trọng lượng từ vết trầy xước**]. Tuy nhiên, chúng chạy nhanh hơn đáng kể và dễ thực hiện hơn, một lợi ích sẽ trở nên rõ rệt hơn đối với các vấn đề lớn hơn.

```{.python .input}
#@tab all
train_concise(0)
```

```{.python .input}
#@tab all
train_concise(3)
```

Cho đến nay, chúng ta chỉ chạm vào một khái niệm về những gì tạo thành một hàm tuyến tính đơn giản. Hơn nữa, những gì tạo thành một hàm phi tuyến đơn giản có thể là một câu hỏi thậm chí còn phức tạp hơn. Ví dụ, [tái tạo hạt nhân Hilbert space (RKHS)](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space) cho phép người ta áp dụng các công cụ được giới thiệu cho các hàm tuyến tính trong bối cảnh phi tuyến tính. Thật không may, các thuật toán dựa trên RKHS có xu hướng mở rộng quy mô dữ liệu lớn, chiều cao. Trong cuốn sách này, chúng tôi sẽ mặc định để heuristic đơn giản của việc áp dụng phân rã trọng lượng trên tất cả các lớp của một mạng sâu. 

## Tóm tắt

* Thường xuyên hóa là một phương pháp phổ biến để đối phó với overfitting. Nó bổ sung một thuật ngữ phạt cho chức năng mất trên bộ đào tạo để giảm độ phức tạp của mô hình đã học.
* Một lựa chọn đặc biệt để giữ cho mô hình đơn giản là giảm cân bằng cách sử dụng hình phạt $L_2$. Điều này dẫn đến phân rã trọng lượng trong các bước cập nhật của thuật toán học tập.
* Chức năng phân rã trọng lượng được cung cấp trong các trình tối ưu hóa từ các khuôn khổ học sâu.
* Các bộ tham số khác nhau có thể có các hành vi cập nhật khác nhau trong cùng một vòng đào tạo.

## Bài tập

1. Thử nghiệm với giá trị của $\lambda$ trong bài toán ước lượng trong phần này. Đào tạo cốt truyện và độ chính xác kiểm tra như một chức năng của $\lambda$. Bạn quan sát điều gì?
1. Sử dụng bộ xác thực để tìm giá trị tối ưu là $\lambda$. Nó có thực sự là giá trị tối ưu? Điều này có quan trọng không?
1. Các phương trình cập nhật sẽ trông như thế nào nếu thay vì $\|\mathbf{w}\|^2$ chúng tôi sử dụng $\sum_i |w_i|$ làm hình phạt của sự lựa chọn của chúng tôi ($L_1$ chính quy hóa)?
1. Chúng ta biết rằng $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$. Bạn có thể tìm thấy một phương trình tương tự cho ma trận (xem định mức Frobenius trong :numref:`subsec_lin-algebra-norms`)?
1. Xem lại mối quan hệ giữa lỗi đào tạo và lỗi tổng quát hóa. Ngoài phân rã trọng lượng, tăng cường đào tạo, và sử dụng một mô hình phức tạp phù hợp, những cách khác bạn có thể nghĩ đến để đối phó với overfitting?
1. Trong thống kê Bayesian, chúng tôi sử dụng sản phẩm trước và khả năng để đến một phía sau thông qua $P(w \mid x) \propto P(x \mid w) P(w)$. Làm thế nào bạn có thể xác định $P(w)$ với quy định hóa?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/98)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/99)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/236)
:end_tab:
