# Bỏ học
:label:`sec_dropout`

Năm :numref:`sec_weight_decay`, chúng tôi đã giới thiệu cách tiếp cận cổ điển để điều chỉnh các mô hình thống kê bằng cách phạt định mức $L_2$ của trọng lượng. Về xác suất, chúng ta có thể biện minh cho kỹ thuật này bằng cách lập luận rằng chúng ta đã giả định một niềm tin trước đó rằng trọng lượng lấy giá trị từ một phân phối Gaussian với trung bình 0. Trực quan hơn, chúng ta có thể lập luận rằng chúng tôi khuyến khích mô hình trải ra trọng lượng của nó giữa nhiều tính năng thay vì phụ thuộc quá nhiều vào một số lượng nhỏ các hiệp hội có khả năng giả mạo. 

## Overfitting Revisited

Đối mặt với nhiều tính năng hơn ví dụ, các mô hình tuyến tính có xu hướng overfit. Nhưng đưa ra nhiều ví dụ hơn các tính năng, chúng ta thường có thể tin tưởng vào các mô hình tuyến tính không quá mức. Thật không may, độ tin cậy mà các mô hình tuyến tính khái quát hóa đi kèm với chi phí. Các mô hình tuyến tính được áp dụng một cách ngây thơ không tính đến các tương tác giữa các tính năng. Đối với mọi tính năng, một mô hình tuyến tính phải gán một trọng lượng dương hoặc âm, bỏ qua bối cảnh. 

Trong các văn bản truyền thống, sự căng thẳng cơ bản này giữa tính tổng quát và tính linh hoạt được mô tả là sự cân bằng phương sai *bias-variance*. Các mô hình tuyến tính có thiên vị cao: chúng chỉ có thể đại diện cho một lớp hàm nhỏ. Tuy nhiên, các mô hình này có phương sai thấp: chúng cho kết quả tương tự trên các mẫu ngẫu nhiên khác nhau của dữ liệu. 

Các mạng thần kinh sâu sống ở đầu đối diện của phổ phương sai thiên vị. Không giống như các mô hình tuyến tính, mạng thần kinh không bị giới hạn để nhìn vào từng tính năng riêng lẻ. Họ có thể học tương tác giữa các nhóm tính năng. Ví dụ, họ có thể suy ra rằng “Nigeria” và “Western Union” xuất hiện cùng nhau trong một email chỉ ra thư rác nhưng điều đó riêng biệt họ không. 

Ngay cả khi chúng ta có nhiều ví dụ hơn nhiều so với các tính năng, các mạng thần kinh sâu có khả năng vượt trội. Năm 2017, một nhóm các nhà nghiên cứu đã chứng minh tính linh hoạt cực đoan của mạng thần kinh bằng cách đào tạo các mạng lưới sâu trên các hình ảnh được dán nhãn ngẫu nhiên. Mặc dù không có bất kỳ mô hình thực sự nào liên kết đầu vào với đầu ra, họ phát hiện ra rằng mạng thần kinh được tối ưu hóa bởi gốc gradient ngẫu nhiên có thể gắn nhãn mọi hình ảnh trong bộ đào tạo một cách hoàn hảo. Hãy xem xét điều này có nghĩa là gì. Nếu các nhãn được gán thống nhất một cách ngẫu nhiên và có 10 lớp, thì không có bộ phân loại nào có thể làm tốt hơn 10% độ chính xác trên dữ liệu holdout. Khoảng cách tổng quát ở đây là một con số khổng lồ 90%. Nếu các mô hình của chúng tôi rất biểu cảm đến mức chúng có thể quá mức phù hợp với điều này, thì khi nào chúng ta nên mong đợi chúng không quá mức? 

Các nền tảng toán học cho các thuộc tính tổng quát khó hiểu của các mạng sâu vẫn là các câu hỏi nghiên cứu mở, và chúng tôi khuyến khích người đọc định hướng lý thuyết để đào sâu hơn vào chủ đề. Hiện tại, chúng tôi chuyển sang điều tra các công cụ thực tế có xu hướng cải thiện thực nghiệm sự tổng quát của lưới sâu. 

## Mạnh mẽ thông qua nhiễu loạn

Chúng ta hãy suy nghĩ ngắn gọn về những gì chúng ta mong đợi từ một mô hình dự đoán tốt. Chúng tôi muốn nó để peform tốt trên dữ liệu không nhìn thấy. Lý thuyết tổng quát hóa cổ điển cho thấy rằng để thu hẹp khoảng cách giữa tàu và hiệu suất thử nghiệm, chúng ta nên hướng đến một mô hình đơn giản. Sự đơn giản có thể đến dưới dạng một số lượng nhỏ kích thước. Chúng tôi đã khám phá điều này khi thảo luận về các hàm cơ sở đơn nguyên của các mô hình tuyến tính trong :numref:`sec_model_selection`. Ngoài ra, như chúng ta đã thấy khi thảo luận về phân rã trọng lượng ($L_2$ chính quy hóa) trong :numref:`sec_weight_decay`, định mức (nghịch đảo) của các tham số cũng đại diện cho một thước đo đơn giản hữu ích. Một khái niệm hữu ích khác về sự đơn giản là sự trơn tru, tức là chức năng không nên nhạy cảm với những thay đổi nhỏ đối với đầu vào của nó. Ví dụ: khi chúng tôi phân loại hình ảnh, chúng tôi hy vọng rằng việc thêm một số nhiễu ngẫu nhiên vào các pixel chủ yếu là vô hại. 

Năm 1995, Christopher Bishop chính thức hóa ý tưởng này khi ông chứng minh rằng việc đào tạo với tiếng ồn đầu vào tương đương với việc chính quy hóa Tikhonov :cite:`Bishop.1995`. Công việc này đã vẽ một kết nối toán học rõ ràng giữa yêu cầu rằng một hàm được trơn tru (và do đó đơn giản), và yêu cầu rằng nó có khả năng phục hồi với nhiễu loạn trong đầu vào. 

Sau đó, vào năm 2014, Srivastava et al. :cite:`Srivastava.Hinton.Krizhevsky.ea.2014` đã phát triển một ý tưởng thông minh về cách áp dụng ý tưởng của Giám mục vào các lớp nội bộ của một mạng, quá. Cụ thể, họ đề xuất tiêm tiếng ồn vào mỗi lớp của mạng trước khi tính toán lớp tiếp theo trong quá trình đào tạo. Họ nhận ra rằng khi đào tạo một mạng lưới sâu với nhiều lớp, việc tiêm tiếng ồn sẽ thực hiện sự trơn tru chỉ trên bản đồ đầu vào-đầu ra. 

Ý tưởng của họ, được gọi là * dropout*, liên quan đến việc tiêm tiếng ồn trong khi tính toán từng lớp bên trong trong quá trình truyền chuyển tiếp và nó đã trở thành một kỹ thuật tiêu chuẩn để đào tạo mạng thần kinh. Phương pháp này được gọi là *dropout* bởi vì chúng tôi theo nghĩa đen
*thả ra* một số tế bào thần kinh trong quá trình đào tạo.
Trong suốt quá trình đào tạo, trên mỗi lần lặp lại, bỏ học tiêu chuẩn bao gồm zeroing ra một số phần của các nút trong mỗi lớp trước khi tính toán lớp tiếp theo. 

Để rõ ràng, chúng tôi đang áp đặt câu chuyện của riêng mình với liên kết với Giám mục. Bài báo gốc về bỏ học cung cấp trực giác thông qua một sự tương tự đáng ngạc nhiên với sinh sản tình dục. Các tác giả cho rằng mạng thần kinh overfitting được đặc trưng bởi một trạng thái trong đó mỗi lớp dựa vào một mô hình kích hoạt specifc trong lớp trước, gọi điều kiện này * đồng thích nghi*. Bỏ học, họ tuyên bố, phá vỡ sự đồng thích nghi giống như sinh sản tình dục được lập luận là phá vỡ các gen đồng thích nghi. 

Thách thức quan trọng sau đó là làm thế nào để tiêm tiếng ồn này. Một ý tưởng là tiêm nhiễu theo cách *unbiased* sao cho giá trị kỳ vọng của mỗi lớp — trong khi sửa chữa các lớp khác - bằng với giá trị mà nó sẽ mất tiếng ồn vắng mặt. 

Trong tác phẩm của Bishop, ông đã thêm tiếng ồn Gaussian vào các đầu vào cho một mô hình tuyến tính. Tại mỗi lần lặp lại đào tạo, ông đã thêm tiếng ồn lấy mẫu từ một phân phối với trung bình 0 $\epsilon \sim \mathcal{N}(0,\sigma^2)$ đến đầu vào $\mathbf{x}$, mang lại một điểm nhiễu loạn $\mathbf{x}' = \mathbf{x} + \epsilon$. Trong kỳ vọng, $E[\mathbf{x}'] = \mathbf{x}$. 

Trong quy định bỏ học tiêu chuẩn, người ta làm giảm từng lớp bằng cách bình thường hóa bởi phần nhỏ của các nút được giữ lại (không bị loại bỏ). Nói cách khác, với xác suất bỏ lọc* $p$, mỗi kích hoạt trung gian $h$ được thay thế bằng một biến ngẫu nhiên $h'$ như sau: 

$$
\begin{aligned}
h' =
\begin{cases}
    0 & \text{ with probability } p \\
    \frac{h}{1-p} & \text{ otherwise}
\end{cases}
\end{aligned}
$$

Theo thiết kế, kỳ vọng vẫn không thay đổi, tức là $E[h'] = h$. 

## Bỏ học trong thực hành

Nhớ lại MLP với một lớp ẩn và 5 đơn vị ẩn trong :numref:`fig_mlp`. Khi chúng ta áp dụng dropout cho một lớp ẩn, zeroing ra từng đơn vị ẩn với xác suất $p$, kết quả có thể được xem như là một mạng chỉ chứa một tập hợp con của các tế bào thần kinh ban đầu. Trong :numref:`fig_dropout2`, $h_2$ và $h_5$ bị loại bỏ. Do đó, việc tính toán các đầu ra không còn phụ thuộc vào $h_2$ hoặc $h_5$ và gradient tương ứng của chúng cũng biến mất khi thực hiện truyền ngược. Bằng cách này, việc tính toán lớp đầu ra không thể phụ thuộc quá mức vào bất kỳ một phần tử nào của $h_1, \ldots, h_5$. 

![MLP before and after dropout.](../img/dropout2.svg)
:label:`fig_dropout2`

Thông thường, chúng tôi vô hiệu hóa bỏ học tại thời điểm thử nghiệm. Với một mô hình được đào tạo và một ví dụ mới, chúng tôi không bỏ bất kỳ nút nào và do đó không cần phải bình thường hóa. Tuy nhiên, có một số ngoại lệ: một số nhà nghiên cứu sử dụng bỏ học tại thời điểm thử nghiệm như một heuristic để ước tính * không chắc chắn* của dự đoán mạng thần kinh: nếu các dự đoán đồng ý trên nhiều mặt nạ bỏ học khác nhau, thì chúng ta có thể nói rằng mạng tự tin hơn. 

## Thực hiện từ đầu

Để thực hiện hàm dropout cho một lớp duy nhất, chúng ta phải vẽ nhiều mẫu từ một biến ngẫu nhiên Bernoulli (nhị phân) như lớp của chúng ta có kích thước, trong đó biến ngẫu nhiên có giá trị $1$ (giữ) với xác suất $1-p$ và $0$ (thả) với xác suất $p$. Một cách dễ dàng để thực hiện điều này là lần đầu tiên vẽ các mẫu từ phân phối thống nhất $U[0, 1]$. Sau đó, chúng ta có thể giữ các nút đó mà mẫu tương ứng lớn hơn $p$, thả phần còn lại. 

Trong đoạn code sau đây, ta (**thực hiện hàm `dropout_layer` loại bỏ các phần tử trong đầu vào tensor `X` với xác suất `dropout`**), rescaling phần còn lại như mô tả ở trên: chia những người sống sót cho `1.0-dropout`.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # In this case, all elements are dropped out
    if dropout == 1:
        return np.zeros_like(X)
    # In this case, all elements are kept
    if dropout == 0:
        return X
    mask = np.random.uniform(0, 1, X.shape) > dropout
    return mask.astype(np.float32) * X / (1.0 - dropout)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # In this case, all elements are dropped out
    if dropout == 1:
        return torch.zeros_like(X)
    # In this case, all elements are kept
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # In this case, all elements are dropped out
    if dropout == 1:
        return tf.zeros_like(X)
    # In this case, all elements are kept
    if dropout == 0:
        return X
    mask = tf.random.uniform(
        shape=tf.shape(X), minval=0, maxval=1) < 1 - dropout
    return tf.cast(mask, dtype=tf.float32) * X / (1.0 - dropout)
```

Chúng ta có thể [** test out the `dropout_layer` function on a few examples**]. Trong các dòng mã sau đây, chúng tôi vượt qua đầu vào `X` thông qua thao tác bỏ học, với xác suất lần lượt là 0, 0.5 và 1.

```{.python .input}
X = np.arange(16).reshape(2, 8)
print(dropout_layer(X, 0))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1))
```

```{.python .input}
#@tab pytorch
X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(16, dtype=tf.float32), (2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))
```

### Xác định các tham số mô hình

Một lần nữa, chúng tôi làm việc với bộ dữ liệu Fashion-MNIST được giới thiệu trong :numref:`sec_fashion_mnist`. Chúng ta [**xác định một MLP với hai lớp ẩn chứa 256 đơn vị mỗi hột.**]

```{.python .input}
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens1))
b1 = np.zeros(num_hiddens1)
W2 = np.random.normal(scale=0.01, size=(num_hiddens1, num_hiddens2))
b2 = np.zeros(num_hiddens2)
W3 = np.random.normal(scale=0.01, size=(num_hiddens2, num_outputs))
b3 = np.zeros(num_outputs)

params = [W1, b1, W2, b2, W3, b3]
for param in params:
    param.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
```

```{.python .input}
#@tab tensorflow
num_outputs, num_hiddens1, num_hiddens2 = 10, 256, 256
```

### Xác định mô hình

Mô hình dưới đây áp dụng dropout cho đầu ra của mỗi lớp ẩn (theo chức năng kích hoạt). Chúng ta có thể đặt xác suất bỏ học cho từng lớp riêng biệt. Một xu hướng phổ biến là đặt xác suất bỏ học thấp hơn gần với lớp đầu vào. Dưới đây chúng tôi đặt nó thành 0,2 và 0,5 cho các lớp ẩn đầu tiên và thứ hai, tương ứng. Chúng tôi đảm bảo rằng việc bỏ học chỉ hoạt động trong quá trình đào tạo.

```{.python .input}
dropout1, dropout2 = 0.2, 0.5

def net(X):
    X = X.reshape(-1, num_inputs)
    H1 = npx.relu(np.dot(X, W1) + b1)
    # Use dropout only when training the model
    if autograd.is_training():
        # Add a dropout layer after the first fully connected layer
        H1 = dropout_layer(H1, dropout1)
    H2 = npx.relu(np.dot(H1, W2) + b2)
    if autograd.is_training():
        # Add a dropout layer after the second fully connected layer
        H2 = dropout_layer(H2, dropout2)
    return np.dot(H2, W3) + b3
```

```{.python .input}
#@tab pytorch
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # Use dropout only when training the model
        if self.training == True:
            # Add a dropout layer after the first fully connected layer
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # Add a dropout layer after the second fully connected layer
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
```

```{.python .input}
#@tab tensorflow
dropout1, dropout2 = 0.2, 0.5

class Net(tf.keras.Model):
    def __init__(self, num_outputs, num_hiddens1, num_hiddens2):
        super().__init__()
        self.input_layer = tf.keras.layers.Flatten()
        self.hidden1 = tf.keras.layers.Dense(num_hiddens1, activation='relu')
        self.hidden2 = tf.keras.layers.Dense(num_hiddens2, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_outputs)

    def call(self, inputs, training=None):
        x = self.input_layer(inputs)
        x = self.hidden1(x)
        if training:
            x = dropout_layer(x, dropout1)
        x = self.hidden2(x)
        if training:
            x = dropout_layer(x, dropout2)
        x = self.output_layer(x)
        return x

net = Net(num_outputs, num_hiddens1, num_hiddens2)
```

### [**Đào tạo và kiểm tra**]

Điều này tương tự như đào tạo và thử nghiệm MLP s được mô tả trước đó.

```{.python .input}
num_epochs, lr, batch_size = 10, 0.5, 256
loss = gluon.loss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              lambda batch_size: d2l.sgd(params, lr, batch_size))
```

```{.python .input}
#@tab pytorch
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr, batch_size = 10, 0.5, 256
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = tf.keras.optimizers.SGD(learning_rate=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## [**Thiết tập**]

Với API cấp cao, tất cả những gì chúng ta cần làm là thêm một lớp `Dropout` sau mỗi lớp được kết nối hoàn toàn, truyền trong xác suất bỏ học làm đối số duy nhất cho hàm tạo của nó. Trong quá trình đào tạo, lớp `Dropout` sẽ thả ngẫu nhiên các đầu ra của lớp trước đó (hoặc tương đương, các đầu vào cho lớp tiếp theo) theo xác suất bỏ học được chỉ định. Khi không ở chế độ đào tạo, lớp `Dropout` chỉ đơn giản là truyền dữ liệu qua trong quá trình thử nghiệm.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, activation="relu"),
        # Add a dropout layer after the first fully connected layer
        nn.Dropout(dropout1),
        nn.Dense(256, activation="relu"),
        # Add a dropout layer after the second fully connected layer
        nn.Dropout(dropout2),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # Add a dropout layer after the first fully connected layer
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # Add a dropout layer after the second fully connected layer
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    # Add a dropout layer after the first fully connected layer
    tf.keras.layers.Dropout(dropout1),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    # Add a dropout layer after the second fully connected layer
    tf.keras.layers.Dropout(dropout2),
    tf.keras.layers.Dense(10),
])
```

Tiếp theo, chúng tôi [** đào tạo và kiểm tra mô hình**].

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## Tóm tắt

* Ngoài việc kiểm soát số lượng kích thước và kích thước của vectơ trọng lượng, bỏ học là một công cụ khác để tránh quá mức. Thường thì chúng được sử dụng cùng nhau.
* Dropout thay thế một kích hoạt $h$ bằng một biến ngẫu nhiên với giá trị kỳ vọng $h$.
* Dropout chỉ được sử dụng trong quá trình đào tạo.

## Bài tập

1. Điều gì sẽ xảy ra nếu bạn thay đổi xác suất bỏ học cho các lớp thứ nhất và thứ hai? Đặc biệt, điều gì sẽ xảy ra nếu bạn chuyển đổi những cái cho cả hai lớp? Thiết kế một thử nghiệm để trả lời những câu hỏi này, mô tả kết quả của bạn một cách định lượng và tóm tắt các takeaways định tính.
1. Tăng số lượng thời đại và so sánh kết quả thu được khi sử dụng bỏ học với những người khi không sử dụng nó.
1. Phương sai của các kích hoạt trong mỗi lớp ẩn khi bỏ học và không được áp dụng là gì? Vẽ một âm mưu để hiển thị số lượng này phát triển như thế nào theo thời gian cho cả hai mô hình.
1. Tại sao bỏ học thường không được sử dụng tại thời điểm thử nghiệm?
1. Sử dụng mô hình trong phần này làm ví dụ, so sánh các hiệu ứng của việc sử dụng bỏ học và phân rã trọng lượng. Điều gì xảy ra khi bỏ học và phân rã trọng lượng được sử dụng cùng một lúc? Có phụ gia kết quả không? Có lợi nhuận giảm (hoặc tệ hơn)? Họ có hủy bỏ nhau không?
1. Điều gì sẽ xảy ra nếu chúng ta áp dụng bỏ học cho các trọng lượng riêng lẻ của ma trận trọng lượng chứ không phải là kích hoạt?
1. Phát minh ra một kỹ thuật khác để tiêm nhiễu ngẫu nhiên ở mỗi lớp khác với kỹ thuật bỏ học tiêu chuẩn. Bạn có thể phát triển một phương pháp vượt trội hơn việc bỏ học trên tập dữ liệu Fashion-MNIST (cho một kiến trúc cố định)?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/100)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/101)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/261)
:end_tab:
