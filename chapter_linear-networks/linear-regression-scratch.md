# Thực hiện hồi quy tuyến tính từ đầu
:label:`sec_linear_scratch`

Bây giờ bạn đã hiểu những ý tưởng quan trọng đằng sau hồi quy tuyến tính, chúng ta có thể bắt đầu làm việc thông qua việc thực hiện thực hành trong mã. Trong phần này, (**chúng tôi sẽ thực hiện toàn bộ phương pháp từ đầu, bao gồm đường ống dữ liệu, mô hình, chức năng mất mát và trình tối ưu hóa giảm dần dần ngẫu nhiên minibatch. **) Trong khi các khuôn khổ học sâu hiện đại có thể tự động hóa gần như tất cả công việc này, thực hiện mọi thứ từ đầu là cách duy nhất để đảm bảo rằng bạn thực sự biết những gì bạn đang làm. Hơn nữa, khi đến lúc tùy chỉnh các mô hình, xác định các lớp hoặc chức năng mất mát của riêng chúng ta, hiểu cách mọi thứ hoạt động dưới mui xe sẽ chứng minh tiện dụng. Trong phần này, chúng tôi sẽ chỉ dựa vào hàng chục và sự khác biệt tự động. Sau đó, chúng tôi sẽ giới thiệu một triển khai ngắn gọn hơn, tận dụng chuông và còi của các khuôn khổ học sâu.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

## Tạo tập dữ liệu

Để giữ cho mọi thứ đơn giản, chúng ta sẽ [** xây dựng một tập dữ liệu nhân tạo theo một mô hình tuyến tính với tiếng ồn phụ gia.**] Nhiệm vụ của chúng ta là khôi phục các tham số của mô hình này bằng cách sử dụng tập hợp các ví dụ hữu hạn có trong tập dữ liệu của chúng ta. Chúng tôi sẽ giữ dữ liệu chiều thấp để chúng tôi có thể hình dung nó một cách dễ dàng. Trong đoạn mã sau, chúng ta tạo ra một tập dữ liệu chứa 1000 ví dụ, mỗi ví dụ gồm 2 tính năng được lấy mẫu từ một phân phối bình thường chuẩn. Do đó tập dữ liệu tổng hợp của chúng tôi sẽ là một ma trận $\mathbf{X}\in \mathbb{R}^{1000 \times 2}$. 

(** Các thông số thực sự tạo ra bộ dữ liệu của chúng tôi sẽ là $\mathbf{w} = [2, -3.4]^\top$ và $b = 4.2$ và**) nhãn tổng hợp của chúng tôi sẽ được gán theo mô hình tuyến tính sau với thuật ngữ nhiễu $\epsilon$: 

(**$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \mathbf\epsilon.$$**) 

Bạn có thể nghĩ về $\epsilon$ là nắm bắt các lỗi đo lường tiềm ẩn trên các tính năng và nhãn. Chúng tôi sẽ giả định rằng các giả định tiêu chuẩn giữ và do đó $\epsilon$ tuân theo một phân phối bình thường với trung bình 0. Để làm cho vấn đề của chúng tôi dễ dàng, chúng tôi sẽ đặt độ lệch chuẩn của nó thành 0,01. Mã sau đây tạo ra tập dữ liệu tổng hợp của chúng tôi.

```{.python .input}
#@tab mxnet, pytorch
def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = d2l.normal(0, 1, (num_examples, len(w)))
    y = d2l.matmul(X, w) + b
    y += d2l.normal(0, 0.01, y.shape)
    return X, d2l.reshape(y, (-1, 1))
```

```{.python .input}
#@tab tensorflow
def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = d2l.zeros((num_examples, w.shape[0]))
    X += tf.random.normal(shape=X.shape)
    y = d2l.matmul(X, tf.reshape(w, (-1, 1))) + b
    y += tf.random.normal(shape=y.shape, stddev=0.01)
    y = d2l.reshape(y, (-1, 1))
    return X, y
```

```{.python .input}
#@tab all
true_w = d2l.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

Lưu ý rằng [** mỗi hàng trong `features` bao gồm một ví dụ dữ liệu 2 chiều và mỗi hàng trong `labels` bao gồm một giá trị nhãn 1 chiều (vô hướng) .**]

```{.python .input}
#@tab all
print('features:', features[0],'\nlabel:', labels[0])
```

Bằng cách tạo ra một âm mưu phân tán bằng cách sử dụng tính năng thứ hai `features[:, 1]` và `labels`, chúng ta có thể quan sát rõ mối tương quan tuyến tính giữa hai.

```{.python .input}
#@tab all
d2l.set_figsize()
# The semicolon is for displaying the plot only
d2l.plt.scatter(d2l.numpy(features[:, 1]), d2l.numpy(labels), 1);
```

## Đọc tập dữ liệu

Nhớ lại rằng các mô hình đào tạo bao gồm thực hiện nhiều lần vượt qua tập dữ liệu, lấy một minibatch ví dụ tại một thời điểm và sử dụng chúng để cập nhật mô hình của chúng tôi. Vì quá trình này rất cơ bản để đào tạo các thuật toán học máy, nên nó đáng để xác định một chức năng tiện ích để xáo trộn bộ dữ liệu và truy cập nó trong các minibatches. 

Trong đoạn code sau, chúng ta [**define the `data_iter` function**](~~that~~) để chứng minh một thực hiện có thể thực hiện chức năng này. Chức năng (** có kích thước lô, một ma trận của các tính năng và một vector nhãn, mang lại minibatches kích thước `batch_size`.**) Mỗi minibatch bao gồm một loạt các tính năng và nhãn.

```{.python .input}
#@tab mxnet, pytorch
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = d2l.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
```

```{.python .input}
#@tab tensorflow
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = tf.constant(indices[i: min(i + batch_size, num_examples)])
        yield tf.gather(features, j), tf.gather(labels, j)
```

Nói chung, lưu ý rằng chúng tôi muốn sử dụng minibatches có kích thước hợp lý để tận dụng lợi thế của phần cứng GPU, vượt trội ở các hoạt động song song. Bởi vì mỗi ví dụ có thể được cung cấp thông qua các mô hình của chúng tôi song song và gradient của hàm mất cho mỗi ví dụ cũng có thể được thực hiện song song, GPU cho phép chúng tôi xử lý hàng trăm ví dụ trong thời gian ít hơn so với việc xử lý chỉ là một ví dụ duy nhất. 

Để xây dựng một số trực giác, chúng ta hãy đọc và in hàng loạt ví dụ dữ liệu nhỏ đầu tiên. Hình dạng của các tính năng trong mỗi minibatch cho chúng ta biết cả kích thước minibatch và số lượng tính năng đầu vào. Tương tự như vậy, minibatch nhãn của chúng tôi sẽ có một hình dạng được đưa ra bởi `batch_size`.

```{.python .input}
#@tab all
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
```

Khi chúng ta chạy lặp lại, chúng ta có được các minibatches riêng biệt liên tiếp cho đến khi toàn bộ bộ dữ liệu đã cạn kiệt (hãy thử điều này). Mặc dù việc lặp lại được thực hiện ở trên là tốt cho mục đích giáo khoa, nhưng nó không hiệu quả theo những cách có thể khiến chúng ta gặp rắc rối về các vấn đề thực sự. Ví dụ: nó yêu cầu chúng tôi tải tất cả dữ liệu trong bộ nhớ và chúng tôi thực hiện nhiều truy cập bộ nhớ ngẫu nhiên. Các bộ lặp tích hợp được triển khai trong một khuôn khổ học sâu hiệu quả hơn đáng kể và chúng có thể xử lý cả dữ liệu được lưu trữ trong tệp và dữ liệu được cung cấp thông qua các luồng dữ liệu. 

## Khởi tạo các tham số mô hình

[**Trước khi chúng ta có thể bắt đầu tối ưu hóa các tham số của mô hình của mình**] bằng cách hạ xuống gradient ngẫu nhiên minibatch, (** chúng ta cần có một số tham số ở vị trí đầu tiên.**) Trong mã sau, chúng ta khởi tạo trọng lượng bằng cách lấy mẫu các số ngẫu nhiên từ phân phối bình thường với 0 trung bình và độ lệch chuẩn 0,01, và thiết lập sự thiên vị thành 0.

```{.python .input}
w = np.random.normal(0, 0.01, (2, 1))
b = np.zeros(1)
w.attach_grad()
b.attach_grad()
```

```{.python .input}
#@tab pytorch
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

```{.python .input}
#@tab tensorflow
w = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.01),
                trainable=True)
b = tf.Variable(tf.zeros(1), trainable=True)
```

Sau khi khởi tạo các tham số của chúng tôi, nhiệm vụ tiếp theo của chúng tôi là cập nhật chúng cho đến khi chúng phù hợp với dữ liệu của chúng tôi đủ tốt. Mỗi bản cập nhật yêu cầu lấy gradient của hàm mất của chúng tôi đối với các tham số. Với gradient này, chúng ta có thể cập nhật từng tham số theo hướng có thể làm giảm tổn thất. 

Vì không ai muốn tính toán độ dốc một cách rõ ràng (điều này là tẻ nhạt và dễ bị lỗi), chúng tôi sử dụng sự khác biệt tự động, như được giới thiệu trong :numref:`sec_autograd`, để tính toán gradient. 

## Xác định mô hình

Tiếp theo, chúng ta phải [** xác định mô hình của chúng tôi, liên quan đến các đầu vào và tham số của nó với đầu racủa nó.**] Nhớ lại rằng để tính toán đầu ra của mô hình tuyến tính, chúng ta chỉ cần lấy sản phẩm chấm ma thuật-vector của các tính năng đầu vào $\mathbf{X}$ và trọng lượng mô hình $\mathbf{w}$ và thêm $b$ bù vào mỗi ví dụ. Lưu ý rằng dưới $\mathbf{Xw}$ là một vectơ và $b$ là vô hướng. Nhớ lại cơ chế phát sóng như được mô tả trong :numref:`subsec_broadcasting`. Khi chúng ta thêm một vectơ và vô hướng, vô hướng được thêm vào mỗi thành phần của vectơ.

```{.python .input}
#@tab all
def linreg(X, w, b):  #@save
    """The linear regression model."""
    return d2l.matmul(X, w) + b
```

## Xác định chức năng mất

Vì [**update model của chúng tôi yêu cầu dùng gradient của hàm mất của chúng ta, **] chúng ta nên (**define the loss function first.**) Ở đây chúng ta sẽ sử dụng hàm mất bình phương như mô tả trong :numref:`sec_linear_regression`. Trong quá trình thực hiện, chúng ta cần chuyển đổi giá trị thực `y` thành hình dạng của giá trị dự đoán `y_hat`. Kết quả được trả về bởi hàm sau cũng sẽ có hình dạng giống như `y_hat`.

```{.python .input}
#@tab all
def squared_loss(y_hat, y):  #@save
    """Squared loss."""
    return (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
```

## Xác định thuật toán tối ưu hóa

Như chúng ta đã thảo luận trong :numref:`sec_linear_regression`, hồi quy tuyến tính có một giải pháp dạng kín. Tuy nhiên, đây không phải là một cuốn sách về hồi quy tuyến tính: nó là một cuốn sách về học sâu. Vì không có mô hình nào khác mà cuốn sách này giới thiệu có thể được giải quyết một cách phân tích, chúng tôi sẽ nhân cơ hội này để giới thiệu ví dụ làm việc đầu tiên của bạn về dòng dốc ngẫu nhiên minibatch. [~~ Mặc dù hồi quy tuyến tính có một giải pháp dạng kín, các mô hình khác trong cuốn sách này thì không. Ở đây chúng tôi giới thiệu minibatch stochastic gradient descent.~ ~] 

Ở mỗi bước, sử dụng một minibatch được vẽ ngẫu nhiên từ tập dữ liệu của chúng tôi, chúng tôi sẽ ước tính độ dốc của sự mất mát đối với các tham số của chúng tôi. Tiếp theo, chúng tôi sẽ cập nhật các thông số của chúng tôi theo hướng có thể làm giảm tổn thất. Mã sau áp dụng bản cập nhật gradient gốc minibatch stochastic, cho một tập hợp các tham số, tốc độ học tập và kích thước lô. Kích thước của bước cập nhật được xác định bởi tỷ lệ học tập `lr`. Bởi vì tổn thất của chúng tôi được tính như một tổng so với các ví dụ nhỏ, chúng tôi bình thường hóa kích thước bước của chúng tôi theo kích thước lô (`batch_size`), do đó độ lớn của một kích thước bước điển hình không phụ thuộc nhiều vào sự lựa chọn của chúng tôi về kích thước lô.

```{.python .input}
def sgd(params, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param in params:
        param[:] = param - lr * param.grad / batch_size
```

```{.python .input}
#@tab pytorch
def sgd(params, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param, grad in zip(params, grads):
        param.assign_sub(lr*grad/batch_size)
```

## Đào tạo

Bây giờ chúng tôi đã có tất cả các phần tại chỗ, chúng tôi đã sẵn sàng để [** thực hiện vòng lặp đào tạo chính**] Điều quan trọng là bạn phải hiểu mã này bởi vì bạn sẽ thấy các vòng đào tạo gần giống hệt nhau hơn và hơn nữa trong suốt sự nghiệp của bạn trong học sâu. 

Trong mỗi lần lặp lại, chúng ta sẽ lấy một loạt các ví dụ đào tạo và chuyển chúng thông qua mô hình của chúng tôi để có được một tập hợp các dự đoán. Sau khi tính toán tổn thất, chúng tôi bắt đầu đi ngược qua mạng, lưu trữ các gradient đối với mỗi tham số. Cuối cùng, chúng ta sẽ gọi thuật toán tối ưu hóa `sgd` để cập nhật các tham số mô hình. 

Tóm lại, chúng tôi sẽ thực hiện vòng lặp sau: 

* Khởi tạo tham số $(\mathbf{w}, b)$
* Lặp lại cho đến khi xong
    * Tính gradient $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
    * Cập nhật thông số $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$

Trong mỗi *epoch*, chúng ta sẽ lặp qua toàn bộ tập dữ liệu (sử dụng hàm `data_iter`) một lần đi qua mọi ví dụ trong tập dữ liệu đào tạo (giả sử rằng số ví dụ được chia hết cho kích thước lô). Số epochs `num_epochs` và tỷ lệ học tập `lr` là cả hai siêu tham số, mà chúng tôi đặt ở đây là 3 và 0,03, tương ứng. Thật không may, việc thiết lập các siêu tham số là khó khăn và yêu cầu một số điều chỉnh bằng cách dùng thử và lỗi. Chúng tôi elide những chi tiết này cho bây giờ nhưng sửa đổi chúng sau này trong :numref:`chap_optimization`.

```{.python .input}
#@tab all
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
```

```{.python .input}
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Because `l` has a shape (`batch_size`, 1) and is not a scalar
        # variable, the elements in `l` are added together to obtain a new
        # variable, on which gradients with respect to [`w`, `b`] are computed
        l.backward()
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

```{.python .input}
#@tab pytorch
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Compute gradient on `l` with respect to [`w`, `b`]
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

```{.python .input}
#@tab tensorflow
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with tf.GradientTape() as g:
            l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Compute gradient on l with respect to [`w`, `b`]
        dw, db = g.gradient(l, [w, b])
        # Update parameters using their gradient
        sgd([w, b], [dw, db], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')
```

Trong trường hợp này, bởi vì chúng tôi tự tổng hợp bộ dữ liệu, chúng tôi biết chính xác các tham số thực sự là gì. Do đó, chúng ta có thể [** đánh giá thành công của mình trong đào tạo bằng cách so sánh các thông số thực sự với những thông số mà chúng tôi đã học **] thông qua vòng đào tạo của chúng tôi. Thật vậy, họ hóa ra rất gần nhau.

```{.python .input}
#@tab all
print(f'error in estimating w: {true_w - d2l.reshape(w, true_w.shape)}')
print(f'error in estimating b: {true_b - b}')
```

Lưu ý rằng chúng ta không nên coi là điều hiển nhiên rằng chúng ta có thể khôi phục các tham số một cách hoàn hảo. Tuy nhiên, trong machine learning, chúng ta thường ít quan tâm đến việc khôi phục các tham số cơ bản thực sự và quan tâm nhiều hơn đến các tham số dẫn đến dự đoán chính xác cao. May mắn thay, ngay cả trên các vấn đề tối ưu hóa khó khăn, gốc gradient stochastic thường có thể tìm thấy các giải pháp tốt đáng kể, một phần do thực tế là, đối với các mạng sâu, tồn tại nhiều cấu hình của các thông số dẫn đến dự đoán chính xác cao. 

## Tóm tắt

* Chúng tôi đã thấy cách một mạng sâu có thể được triển khai và tối ưu hóa từ đầu, chỉ sử dụng hàng chục và sự khác biệt tự động, mà không cần xác định các lớp hoặc tối ưu hóa ưa thích.
* Phần này chỉ làm trầy xước bề mặt của những gì có thể. Trong các phần sau, chúng tôi sẽ mô tả các mô hình bổ sung dựa trên các khái niệm mà chúng tôi vừa giới thiệu và tìm hiểu cách thực hiện chúng một cách chính xác hơn.

## Bài tập

1. Điều gì sẽ xảy ra nếu chúng ta khởi tạo trọng lượng bằng không. Liệu thuật toán vẫn hoạt động?
1. Giả sử rằng bạn đang [Georg Simon Ohm](https://en.wikipedia.org/wiki/Georg_Ohm) đang cố gắng đưa ra một mô hình giữa điện áp và dòng điện. Bạn có thể sử dụng sự khác biệt tự động để tìm hiểu các thông số của mô hình của bạn?
1. Bạn có thể sử dụng [Định luật Planck](https://en.wikipedia.org/wiki/Planck%27s_law) để xác định nhiệt độ của một vật thể sử dụng mật độ năng lượng quang phổ không?
1. Các vấn đề bạn có thể gặp phải là gì nếu bạn muốn tính toán các dẫn xuất thứ hai? Bạn sẽ sửa chúng như thế nào?
1.  Tại sao chức năng `reshape` cần thiết trong hàm `squared_loss`?
1. Thử nghiệm sử dụng các tỷ lệ học tập khác nhau để tìm hiểu giá trị hàm mất giảm nhanh như thế nào.
1. Nếu số lượng ví dụ không thể chia cho kích thước lô, điều gì sẽ xảy ra với hành vi của hàm `data_iter`?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/42)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/43)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/201)
:end_tab:
