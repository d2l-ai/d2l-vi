# Bayes ngây thơ
:label:`sec_naive_bayes`

Trong suốt các phần trước, chúng ta đã học về lý thuyết xác suất và các biến ngẫu nhiên. Để đưa lý thuyết này vào hoạt động, chúng ta hãy giới thiệu bộ phân loại Bayes* ngây thơ. Điều này không sử dụng gì ngoài các nguyên tắc cơ bản xác suất để cho phép chúng tôi thực hiện phân loại các chữ số. 

Học tập là tất cả về việc đưa ra các giả định. Nếu chúng ta muốn phân loại một ví dụ dữ liệu mới mà chúng ta chưa bao giờ thấy trước đây chúng ta phải đưa ra một số giả định về ví dụ dữ liệu nào tương tự nhau. Phân loại Bayes ngây thơ, một thuật toán phổ biến và rõ ràng đáng kể, giả định tất cả các tính năng đều độc lập với nhau để đơn giản hóa việc tính toán. Trong phần này, chúng tôi sẽ áp dụng mô hình này để nhận dạng các ký tự trong hình ảnh.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import gluon, np, npx
npx.set_np()
d2l.use_svg_display()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import torchvision
d2l.use_svg_display()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
d2l.use_svg_display()
```

## Nhận dạng ký tự quang

MNIST :cite:`LeCun.Bottou.Bengio.ea.1998` là một trong những bộ dữ liệu được sử dụng rộng rãi. Nó chứa 60.000 hình ảnh để đào tạo và 10.000 hình ảnh để xác nhận. Mỗi hình ảnh chứa một chữ số viết tay từ 0 đến 9. Nhiệm vụ là phân loại từng hình ảnh thành chữ số tương ứng. 

Gluon cung cấp lớp `MNIST` trong mô-đun `data.vision` để tự động lấy tập dữ liệu từ Internet. Sau đó, Gluon sẽ sử dụng bản sao cục bộ đã tải xuống. Chúng tôi chỉ định xem chúng tôi đang yêu cầu bộ đào tạo hay bộ thử nghiệm bằng cách đặt giá trị của tham số `train` thành `True` hoặc `False` tương ứng. Mỗi ảnh là một hình ảnh thang màu xám với cả chiều rộng và chiều cao $28$ với hình dạng ($28$,$28$,$1$). Chúng tôi sử dụng một chuyển đổi tùy chỉnh để loại bỏ kích thước kênh cuối cùng. Ngoài ra, tập dữ liệu đại diện cho mỗi pixel bằng một số nguyên $8$-bit không dấu. Chúng tôi định lượng chúng thành các tính năng nhị phân để đơn giản hóa vấn đề.

```{.python .input}
def transform(data, label):
    return np.floor(data.astype('float32') / 128).squeeze(axis=-1), label

mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)
```

```{.python .input}
#@tab pytorch
data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    lambda x: torch.floor(x * 255 / 128).squeeze(dim=0)
])

mnist_train = torchvision.datasets.MNIST(
    root='./temp', train=True, transform=data_transform, download=True)
mnist_test = torchvision.datasets.MNIST(
    root='./temp', train=False, transform=data_transform, download=True)
```

```{.python .input}
#@tab tensorflow
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.mnist.load_data()

# Original pixel values of MNIST range from 0-255 (as the digits are stored as
# uint8). For this section, pixel values that are greater than 128 (in the
# original image) are converted to 1 and values that are less than 128 are
# converted to 0. See section 18.9.2 and 18.9.3 for why
train_images = tf.floor(tf.constant(train_images / 128, dtype = tf.float32))
test_images = tf.floor(tf.constant(test_images / 128, dtype = tf.float32))

train_labels = tf.constant(train_labels, dtype = tf.int32)
test_labels = tf.constant(test_labels, dtype = tf.int32)
```

Chúng ta có thể truy cập một ví dụ cụ thể, chứa hình ảnh và nhãn tương ứng.

```{.python .input}
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab pytorch
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab tensorflow
image, label = train_images[2], train_labels[2]
image.shape, label.numpy()
```

Ví dụ của chúng tôi, được lưu trữ ở đây trong biến `image`, tương ứng với một hình ảnh có chiều cao và chiều rộng $28$ pixel.

```{.python .input}
#@tab all
image.shape, image.dtype
```

Mã của chúng tôi lưu trữ nhãn của mỗi hình ảnh dưới dạng vô hướng. Loại của nó là một số nguyên $32$-bit.

```{.python .input}
label, type(label), label.dtype
```

```{.python .input}
#@tab pytorch
label, type(label)
```

```{.python .input}
#@tab tensorflow
label.numpy(), label.dtype
```

Chúng tôi cũng có thể truy cập nhiều ví dụ cùng một lúc.

```{.python .input}
images, labels = mnist_train[10:38]
images.shape, labels.shape
```

```{.python .input}
#@tab pytorch
images = torch.stack([mnist_train[i][0] for i in range(10, 38)], dim=0)
labels = torch.tensor([mnist_train[i][1] for i in range(10, 38)])
images.shape, labels.shape
```

```{.python .input}
#@tab tensorflow
images = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
labels = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
images.shape, labels.shape
```

Hãy để chúng tôi hình dung những ví dụ này.

```{.python .input}
#@tab all
d2l.show_images(images, 2, 9);
```

## Mô hình xác suất để phân loại

Trong một nhiệm vụ phân loại, chúng tôi ánh xạ một ví dụ vào một danh mục. Ở đây một ví dụ là một hình ảnh $28\times 28$ xám và một danh mục là một chữ số. (Tham khảo :numref:`sec_softmax` để được giải thích chi tiết hơn.) Một cách tự nhiên để thể hiện nhiệm vụ phân loại là thông qua câu hỏi xác suất: nhãn có khả năng nhất được đưa ra các tính năng (tức là pixel hình ảnh) là gì? Biểu thị bởi $\mathbf x\in\mathbb R^d$ các tính năng của ví dụ và $y\in\mathbb R$ nhãn. Ở đây các tính năng là pixel hình ảnh, nơi chúng ta có thể định hình lại hình ảnh $2$ chiều thành một vectơ sao cho $d=28^2=784$ và nhãn là chữ số. Xác suất của nhãn cho các tính năng là $p(y  \mid  \mathbf{x})$. Nếu chúng ta có thể tính toán các xác suất này, đó là $p(y  \mid  \mathbf{x})$ cho $y=0, \ldots,9$ trong ví dụ của chúng tôi, thì phân loại sẽ đưa ra dự đoán $\hat{y}$ được đưa ra bởi biểu thức: 

$$\hat{y} = \mathrm{argmax} \> p(y  \mid  \mathbf{x}).$$

Thật không may, điều này đòi hỏi chúng tôi ước tính $p(y  \mid  \mathbf{x})$ cho mọi giá trị của $\mathbf{x} = x_1, ..., x_d$. Hãy tưởng tượng rằng mỗi tính năng có thể lấy một trong $2$ giá trị. Ví dụ: tính năng $x_1 = 1$ có thể biểu thị rằng từ apple xuất hiện trong một tài liệu nhất định và $x_1 = 0$ sẽ biểu thị rằng nó không. Nếu chúng ta có $30$ các tính năng nhị phân như vậy, điều đó có nghĩa là chúng ta cần phải chuẩn bị để phân loại bất kỳ $2^{30}$ nào (hơn 1 tỷ!) giá trị có thể của vector đầu vào $\mathbf{x}$. 

Hơn nữa, việc học ở đâu? Nếu chúng ta cần xem mọi ví dụ có thể để dự đoán nhãn tương ứng thì chúng ta không thực sự học một mẫu mà chỉ ghi nhớ tập dữ liệu. 

## Phân loại Bayes ngây thơ

May mắn thay, bằng cách đưa ra một số giả định về tính độc lập có điều kiện, chúng ta có thể giới thiệu một số thiên vị quy nạp và xây dựng một mô hình có khả năng khái quát hóa từ một lựa chọn tương đối khiêm tốn các ví dụ đào tạo. Để bắt đầu, chúng ta hãy sử dụng định lý Bayes, để thể hiện phân loại như 

$$\hat{y} = \mathrm{argmax}_y \> p(y  \mid  \mathbf{x}) = \mathrm{argmax}_y \> \frac{p( \mathbf{x}  \mid  y) p(y)}{p(\mathbf{x})}.$$

Lưu ý rằng mẫu số là thuật ngữ bình thường hóa $p(\mathbf{x})$ mà không phụ thuộc vào giá trị của nhãn $y$. Kết quả là, chúng ta chỉ cần lo lắng về việc so sánh tử số trên các giá trị khác nhau của $y$. Ngay cả khi tính toán mẫu số hóa ra là khó chữa, chúng ta có thể thoát khỏi việc bỏ qua nó, miễn là chúng ta có thể đánh giá tử số. May mắn thay, ngay cả khi chúng ta muốn phục hồi hằng số bình thường hóa, chúng ta có thể. Chúng tôi luôn có thể phục hồi thời hạn bình thường hóa kể từ $\sum_y p(y  \mid  \mathbf{x}) = 1$. 

Bây giờ, chúng ta hãy tập trung vào $p( \mathbf{x}  \mid  y)$. Sử dụng quy tắc chuỗi xác suất, chúng ta có thể thể hiện thuật ngữ $p( \mathbf{x}  \mid  y)$ như 

$$p(x_1  \mid y) \cdot p(x_2  \mid  x_1, y) \cdot ... \cdot p( x_d  \mid  x_1, ..., x_{d-1}, y).$$

Bản thân nó, biểu thức này không giúp chúng ta xa hơn. Chúng ta vẫn phải ước tính khoảng $2^d$ tham số. Tuy nhiên, nếu chúng ta giả định rằng * các tính năng độc lập có điều kiện với nhau, với nhãn*, thì đột nhiên chúng ta có hình dạng tốt hơn nhiều, vì thuật ngữ này đơn giản hóa thành $\prod_i p(x_i  \mid  y)$, cho chúng ta bộ dự đoán 

$$\hat{y} = \mathrm{argmax}_y \> \prod_{i=1}^d p(x_i  \mid  y) p(y).$$

Nếu chúng ta có thể ước tính $p(x_i=1  \mid  y)$ cho mỗi $i$ và $y$ và lưu giá trị của nó trong $P_{xy}[i, y]$, ở đây $P_{xy}$ là ma trận $d\times n$ với $n$ là số lớp và $y\in\{1, \ldots, n\}$, thì chúng ta cũng có thể sử dụng điều này để ước tính $p(x_i = 0 \mid y)$, tức là, 

$$ 
p(x_i = t_i \mid y) = 
\begin{cases}
    P_{xy}[i, y] & \text{for } t_i=1 ;\\
    1 - P_{xy}[i, y] & \text{for } t_i = 0 .
\end{cases}
$$

Ngoài ra, chúng tôi ước tính $p(y)$ cho mỗi $y$ và lưu nó trong $P_y[y]$, với $P_y$ một vector chiều dài $n$. Then, for any newMới example thí dụ $\mathbf t = (t_1, t_2, \ldots, t_d)$, we could computetính toán 

$$\begin{aligned}\hat{y} &= \mathrm{argmax}_ y \ p(y)\prod_{i=1}^d   p(x_t = t_i \mid y) \\ &= \mathrm{argmax}_y \ P_y[y]\prod_{i=1}^d \ P_{xy}[i, y]^{t_i}\, \left(1 - P_{xy}[i, y]\right)^{1-t_i}\end{aligned}$$
:eqlabel:`eq_naive_bayes_estimation`

for any $y$. Vì vậy, giả định của chúng tôi về sự độc lập có điều kiện đã lấy sự phức tạp của mô hình của chúng tôi từ sự phụ thuộc theo cấp số nhân vào số tính năng $\mathcal{O}(2^dn)$ đến một sự phụ thuộc tuyến tính, đó là $\mathcal{O}(dn)$. 

## Đào tạo

Vấn đề bây giờ là chúng ta không biết $P_{xy}$ và $P_y$. Vì vậy, chúng ta cần phải ước tính giá trị của họ cho một số dữ liệu đào tạo đầu tiên. Đây là * đào tạo* mô hình. Ước tính $P_y$ không quá khó. Vì chúng ta chỉ xử lý các lớp $10$, chúng ta có thể đếm số lần xuất hiện $n_y$ cho mỗi chữ số và chia nó cho tổng lượng dữ liệu $n$. Ví dụ: nếu chữ số 8 xảy ra $n_8 = 5,800$ lần và chúng ta có tổng cộng $n = 60,000$ hình ảnh, ước tính xác suất là $p(y=8) = 0.0967$.

```{.python .input}
X, Y = mnist_train[:]  # All training examples

n_y = np.zeros((10))
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_train[i][0] for i in range(len(mnist_train))], dim=0)
Y = torch.tensor([mnist_train[i][1] for i in range(len(mnist_train))])

n_y = torch.zeros(10)
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab tensorflow
X = train_images
Y = train_labels

n_y = tf.Variable(tf.zeros(10))
for y in range(10):
    n_y[y].assign(tf.reduce_sum(tf.cast(Y == y, tf.float32)))
P_y = n_y / tf.reduce_sum(n_y)
P_y
```

Bây giờ để điều hơi khó khăn hơn $P_{xy}$. Vì chúng tôi chọn hình ảnh đen trắng, $p(x_i  \mid  y)$ biểu thị xác suất pixel $i$ được bật cho lớp $y$. Giống như trước khi chúng ta có thể đi và đếm số lần $n_{iy}$ sao cho một sự kiện xảy ra và chia nó cho tổng số lần xuất hiện $y$, tức là $n_y$. Nhưng có một cái gì đó hơi rắc rối: một số pixel nhất định có thể không bao giờ có màu đen (ví dụ: đối với hình ảnh được cắt tốt, các pixel góc có thể luôn có màu trắng). Một cách thuận tiện để các nhà thống kê giải quyết vấn đề này là thêm số giả cho tất cả các lần xuất hiện. Do đó, thay vì $n_{iy}$ chúng tôi sử dụng $n_{iy}+1$ và thay vì $n_y$ chúng tôi sử dụng $n_{y} + 1$. Điều này còn được gọi là *Laplace Smoothing*. Nó có vẻ đặc biệt, tuy nhiên nó có thể được thúc đẩy tốt từ một quan điểm Bayesian.

```{.python .input}
n_x = np.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = np.array(X.asnumpy()[Y.asnumpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 1).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab pytorch
n_x = torch.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = torch.tensor(X.numpy()[Y.numpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 1).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab tensorflow
n_x = tf.Variable(tf.zeros((10, 28, 28)))
for y in range(10):
    n_x[y].assign(tf.cast(tf.reduce_sum(
        X.numpy()[Y.numpy() == y], axis=0), tf.float32))
P_xy = (n_x + 1) / tf.reshape((n_y + 1), (10, 1, 1))

d2l.show_images(P_xy, 2, 5);
```

Bằng cách hình dung các xác suất $10\times 28\times 28$ này (đối với mỗi pixel cho mỗi lớp), chúng ta có thể nhận được một số chữ số có vẻ trung bình. 

Bây giờ chúng ta có thể sử dụng :eqref:`eq_naive_bayes_estimation` để dự đoán một hình ảnh mới. Cho $\mathbf x$, các chức năng sau tính $p(\mathbf x \mid y)p(y)$ cho mỗi $y$.

```{.python .input}
def bayes_pred(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(axis=1)  # p(x|y)
    return np.array(p_xy) * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab pytorch
def bayes_pred(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(dim=1)  # p(x|y)
    return p_xy * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab tensorflow
def bayes_pred(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = tf.math.reduce_prod(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy * P_y

image, label = train_images[0], train_labels[0]
bayes_pred(image)
```

Điều này đã đi sai khủng khiếp! Để tìm hiểu lý do tại sao, chúng ta hãy nhìn vào xác suất trên mỗi pixel. Chúng thường là những con số giữa $0.001$ và $1$. Chúng tôi đang nhân $784$ trong số họ. Tại thời điểm này, điều đáng nói là chúng ta đang tính toán những con số này trên máy tính, do đó với một phạm vi cố định cho số mũ. Điều gì xảy ra là chúng ta trải nghiệm * underflow số*, tức là, nhân tất cả các số nhỏ dẫn đến một cái gì đó thậm chí còn nhỏ hơn cho đến khi nó được làm tròn xuống 0. Chúng tôi đã thảo luận về điều này như một vấn đề lý thuyết trong :numref:`sec_maximum_likelihood`, nhưng chúng ta thấy rõ các hiện tượng ở đây trong thực tế. 

Như đã thảo luận trong phần đó, chúng tôi khắc phục điều này bằng cách sử dụng thực tế là $\log a b = \log a + \log b$, tức là, chúng tôi chuyển sang tổng hợp logarit. Ngay cả khi cả $a$ và $b$ đều là những con số nhỏ, các giá trị logarit phải nằm trong một phạm vi thích hợp.

```{.python .input}
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```{.python .input}
#@tab pytorch
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```{.python .input}
#@tab tensorflow
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*tf.math.log(a).numpy())
```

Vì logarit là một hàm ngày càng tăng, chúng ta có thể viết lại :eqref:`eq_naive_bayes_estimation` như 

$$ \hat{y} = \mathrm{argmax}_y \ \log P_y[y] + \sum_{i=1}^d \Big[t_i\log P_{xy}[x_i, y] + (1-t_i) \log (1 - P_{xy}[x_i, y]) \Big].$$

Chúng ta có thể thực hiện phiên bản ổn định sau:

```{.python .input}
log_P_xy = np.log(P_xy)
log_P_xy_neg = np.log(1 - P_xy)
log_P_y = np.log(P_y)

def bayes_pred_stable(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab pytorch
log_P_xy = torch.log(P_xy)
log_P_xy_neg = torch.log(1 - P_xy)
log_P_y = torch.log(P_y)

def bayes_pred_stable(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab tensorflow
log_P_xy = tf.math.log(P_xy)
log_P_xy_neg = tf.math.log(1 - P_xy)
log_P_y = tf.math.log(P_y)

def bayes_pred_stable(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = tf.math.reduce_sum(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

Bây giờ chúng ta có thể kiểm tra xem dự đoán có đúng không.

```{.python .input}
# Convert label which is a scalar tensor of int32 dtype to a Python scalar
# integer for comparison
py.argmax(axis=0) == int(label)
```

```{.python .input}
#@tab pytorch
py.argmax(dim=0) == label
```

```{.python .input}
#@tab tensorflow
tf.argmax(py, axis=0, output_type = tf.int32) == label
```

Nếu bây giờ chúng ta dự đoán một vài ví dụ xác nhận, chúng ta có thể thấy bộ phân loại Bayes hoạt động khá tốt.

```{.python .input}
def predict(X):
    return [bayes_pred_stable(x).argmax(axis=0).astype(np.int32) for x in X]

X, y = mnist_test[:18]
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab pytorch
def predict(X):
    return [bayes_pred_stable(x).argmax(dim=0).type(torch.int32).item() 
            for x in X]

X = torch.stack([mnist_test[i][0] for i in range(18)], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(18)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab tensorflow
def predict(X):
    return [tf.argmax(
        bayes_pred_stable(x), axis=0, output_type = tf.int32).numpy()
            for x in X]

X = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
y = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

Cuối cùng, chúng ta hãy tính toán độ chính xác tổng thể của bộ phân loại.

```{.python .input}
X, y = mnist_test[:]
preds = np.array(predict(X), dtype=np.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_test[i][0] for i in range(len(mnist_test))], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(len(mnist_test))])
preds = torch.tensor(predict(X), dtype=torch.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

```{.python .input}
#@tab tensorflow
X = test_images
y = test_labels
preds = tf.constant(predict(X), dtype=tf.int32)
# Validation accuracy
tf.reduce_sum(tf.cast(preds == y, tf.float32)).numpy() / len(y)
```

Các mạng sâu hiện đại đạt được tỷ lệ lỗi dưới $0.01$. Hiệu suất tương đối kém là do các giả định thống kê không chính xác mà chúng tôi đã thực hiện trong mô hình của mình: chúng tôi giả định rằng mỗi điểm ảnh được tạo ra * độc lập*, chỉ tùy thuộc vào nhãn. Đây rõ ràng không phải là cách con người viết chữ số, và giả định sai lầm này dẫn đến sự sụp đổ của phân loại quá ngây thơ (Bayes) của chúng ta. 

## Tóm tắt * Sử dụng quy tắc của Bayes', một phân loại có thể được thực hiện bằng cách giả sử tất cả các tính năng quan sát là độc lập. * Phân loại này có thể được đào tạo trên một tập dữ liệu bằng cách đếm số lần xuất hiện của các kết hợp của nhãn và giá trị pixel * Phân loại này là tiêu chuẩn vàng trong nhiều thập kỷ cho các tác vụ như thư rác phát hiện. 

## Bài tập 1. Xem xét tập dữ liệu $[[0,0], [0,1], [1,0], [1,1]]$ với các nhãn được đưa ra bởi XOR của hai phần tử $[0,1,1,0]$. Xác suất cho một phân loại ngây thơ Bayes được xây dựng trên bộ dữ liệu này là gì. Nó có phân loại thành công điểm của chúng tôi không? Nếu không, những giả định nào bị vi phạm? 1. Giả sử rằng chúng tôi đã không sử dụng Laplace làm mịn khi ước tính xác suất và một ví dụ dữ liệu đến lúc thử nghiệm trong đó có một giá trị không bao giờ quan sát thấy trong đào tạo. Sản lượng mô hình sẽ là gì? 1. Phân loại Bayes ngây thơ là một ví dụ cụ thể của một mạng Bayesian, trong đó sự phụ thuộc của các biến ngẫu nhiên được mã hóa bằng cấu trúc đồ thị. Trong khi lý thuyết đầy đủ nằm ngoài phạm vi của phần này (xem :cite:`Koller.Friedman.2009` để biết chi tiết đầy đủ), giải thích tại sao cho phép phụ thuộc rõ ràng giữa hai biến đầu vào trong mô hình XOR cho phép tạo ra một phân loại thành công.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/418)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1100)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1101)
:end_tab:
