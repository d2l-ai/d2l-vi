# Phân tích tình cảm: Sử dụng mạng thần kinh phức tạp 
:label:`sec_sentiment_cnn`

Trong :numref:`chap_cnn`, chúng tôi đã điều tra các cơ chế xử lý dữ liệu hình ảnh hai chiều với CNN hai chiều, được áp dụng cho các tính năng cục bộ như pixel liền kề. Mặc dù ban đầu được thiết kế cho tầm nhìn máy tính, CNN cũng được sử dụng rộng rãi để xử lý ngôn ngữ tự nhiên. Nói một cách đơn giản, chỉ cần nghĩ về bất kỳ chuỗi văn bản nào như một hình ảnh một chiều. Bằng cách này, CNN một chiều có thể xử lý các tính năng địa phương như $n$-gram trong văn bản. 

Trong phần này, chúng ta sẽ sử dụng mô hình *textCNN* để chứng minh cách thiết kế kiến trúc CNN để đại diện cho văn bản duy nhất :cite:`Kim.2014`. So với :numref:`fig_nlp-map-sa-rnn` sử dụng một kiến trúc RNN với Glove pretraining để phân tích tình cảm, sự khác biệt duy nhất trong :numref:`fig_nlp-map-sa-cnn` nằm ở sự lựa chọn của kiến trúc. 

![This section feeds pretrained GloVe to a CNN-based architecture for sentiment analysis.](../img/nlp-map-sa-cnn.svg)
:label:`fig_nlp-map-sa-cnn`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

## Một chiều Convolutions

Trước khi giới thiệu mô hình, chúng ta hãy xem cách hoạt động phức tạp một chiều. Hãy nhớ rằng nó chỉ là một trường hợp đặc biệt của một sự phức tạp hai chiều dựa trên hoạt động tương quan chéo. 

![One-dimensional cross-correlation operation. The shaded portions are the first output element as well as the input and kernel tensor elements used for the output computation: $0\times1+1\times2=2$.](../img/conv1d.svg)
:label:`fig_conv1d`

Như thể hiện trong :numref:`fig_conv1d`, trong trường hợp một chiều, cửa sổ phức tạp trượt từ trái sang phải qua tensor đầu vào. Trong quá trình trượt, subtensor đầu vào (ví dụ, $0$ và $1$ trong :numref:`fig_conv1d`) chứa trong cửa sổ covolution ở một vị trí nhất định và tensor hạt nhân (ví dụ, $1$ và $2$ trong :numref:`fig_conv1d`) được nhân lên elementwise. Tổng các phép nhân này cho giá trị vô hướng đơn lẻ (ví dụ, $0\times1+1\times2=2$ trong :numref:`fig_conv1d`) tại vị trí tương ứng của tensor đầu ra. 

Chúng tôi thực hiện tương quan chéo một chiều trong hàm `corr1d` sau đây. Với tensor đầu vào `X` và tensor hạt nhân `K`, nó trả về tensor đầu ra `Y`.

```{.python .input}
#@tab all
def corr1d(X, K):
    w = K.shape[0]
    Y = d2l.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y
```

Chúng ta có thể xây dựng tensor đầu vào `X` và tensor kernel `K` từ :numref:`fig_conv1d` để xác nhận đầu ra của việc thực hiện tương quan chéo một chiều ở trên.

```{.python .input}
#@tab all
X, K = d2l.tensor([0, 1, 2, 3, 4, 5, 6]), d2l.tensor([1, 2])
corr1d(X, K)
```

Đối với bất kỳ đầu vào một chiều nào có nhiều kênh, hạt nhân tích tụ cần phải có cùng số kênh đầu vào. Sau đó, đối với mỗi kênh, thực hiện thao tác tương quan chéo trên tensor một chiều của đầu vào và tensor một chiều của hạt nhân covolution, tổng kết kết quả trên tất cả các kênh để tạo ra tensor đầu ra một chiều. :numref:`fig_conv1d_channel` cho thấy một hoạt động tương quan chéo một chiều với 3 kênh đầu vào. 

![One-dimensional cross-correlation operation with 3 input channels. The shaded portions are the first output element as well as the input and kernel tensor elements used for the output computation: $0\times1+1\times2+1\times3+2\times4+2\times(-1)+3\times(-3)=2$.](../img/conv1d-channel.svg)
:label:`fig_conv1d_channel`

Chúng tôi có thể thực hiện hoạt động tương quan chéo một chiều cho nhiều kênh đầu vào và xác nhận kết quả trong :numref:`fig_conv1d_channel`.

```{.python .input}
#@tab all
def corr1d_multi_in(X, K):
    # First, iterate through the 0th dimension (channel dimension) of `X` and
    # `K`. Then, add them together
    return sum(corr1d(x, k) for x, k in zip(X, K))

X = d2l.tensor([[0, 1, 2, 3, 4, 5, 6],
              [1, 2, 3, 4, 5, 6, 7],
              [2, 3, 4, 5, 6, 7, 8]])
K = d2l.tensor([[1, 2], [3, 4], [-1, -3]])
corr1d_multi_in(X, K)
```

Lưu ý rằng các mối tương quan chéo một chiều đa đầu vào kênh tương đương với tương quan chéo hai chiều đơn vào-kênh. Để minh họa, một dạng tương đương của tương quan chéo một chiều đa đầu vào kênh trong :numref:`fig_conv1d_channel` là tương quan chéo hai chiều kênh đầu vào đơn trong :numref:`fig_conv1d_2d`, trong đó chiều cao của hạt nhân phức tạp phải giống như của tensor đầu vào. 

![Two-dimensional cross-correlation operation with a single input channel. The shaded portions are the first output element as well as the input and kernel tensor elements used for the output computation: $2\times(-1)+3\times(-3)+1\times3+2\times4+0\times1+1\times2=2$.](../img/conv1d-2d.svg)
:label:`fig_conv1d_2d`

Cả hai đầu ra trong :numref:`fig_conv1d` và :numref:`fig_conv1d_channel` chỉ có một kênh. Tương tự như các kết hợp hai chiều với nhiều kênh đầu ra được mô tả trong :numref:`subsec_multi-output-channels`, chúng ta cũng có thể chỉ định nhiều kênh đầu ra cho các phức tạp một chiều. 

## Max-Over-Time Pooling

Tương tự, chúng ta có thể sử dụng pooling để trích xuất giá trị cao nhất từ biểu diễn trình tự như là tính năng quan trọng nhất qua các bước thời gian. * max-overtime pooling* được sử dụng trong textCNN hoạt động giống như tổng hợp tối đa toàn cầu một chiều :cite:`Collobert.Weston.Bottou.ea.2011`. Đối với đầu vào đa kênh trong đó mỗi kênh lưu trữ các giá trị ở các bước thời gian khác nhau, đầu ra tại mỗi kênh là giá trị tối đa cho kênh đó. Lưu ý rằng tổng hợp tối đa thời gian cho phép các số bước thời gian khác nhau tại các kênh khác nhau. 

## Mô hình textCNN

Sử dụng sự kết hợp một chiều và tổng hợp tối đa thời gian, mô hình textCNN lấy các biểu diễn token được đào tạo trước riêng lẻ làm đầu vào, sau đó lấy và chuyển đổi biểu diễn trình tự cho ứng dụng hạ lưu. 

Đối với một chuỗi văn bản duy nhất với mã thông báo $n$ được đại diện bởi các vectơ $d$ chiều, chiều rộng, chiều cao và số kênh của tensor đầu vào là $n$, $1$ và $d$, tương ứng. Mô hình textCNN biến đổi đầu vào thành đầu ra như sau: 

1. Xác định nhiều hạt nhân phức tạp một chiều và thực hiện các thao tác phức tạp riêng biệt trên các đầu vào. Các hạt nhân có độ rộng khác nhau có thể nắm bắt các tính năng cục bộ giữa các số lượng khác nhau của các mã thông báo liền kề.
1. Thực hiện tổng hợp tối đa thời gian trên tất cả các kênh đầu ra, và sau đó nối tất cả các đầu ra tổng hợp vô hướng như một vectơ.
1. Chuyển đổi vectơ nối thành các loại đầu ra bằng cách sử dụng lớp được kết nối hoàn toàn. Dropout có thể được sử dụng để giảm overfitting.

![The model architecture of textCNN.](../img/textcnn.svg)
:label:`fig_conv1d_textcnn`

:numref:`fig_conv1d_textcnn` minh họa kiến trúc mô hình của textCNN với một ví dụ cụ thể. Đầu vào là một câu với 11 mã thông báo, trong đó mỗi mã thông báo được biểu diễn bởi một vectơ 6 chiều. Vì vậy, chúng tôi có một đầu vào 6 kênh với chiều rộng 11. Xác định hai hạt nhân phức tạp một chiều có chiều rộng 2 và 4, với 4 và 5 kênh đầu ra, tương ứng. Họ sản xuất 4 kênh đầu ra với chiều rộng $11-2+1=10$ và 5 kênh đầu ra với chiều rộng $11-4+1=8$. Mặc dù chiều rộng khác nhau của 9 kênh này, tổng hợp tối đa thời gian cho một vectơ 9 chiều nối, cuối cùng được chuyển thành vectơ đầu ra 2 chiều cho các dự đoán tình cảm nhị phân. 

### Xác định mô hình

Chúng tôi thực hiện mô hình textCNN trong lớp sau. So với mô hình RNN hai chiều trong :numref:`sec_sentiment_rnn`, bên cạnh việc thay thế các lớp tái phát bằng các lớp phức tạp, chúng tôi cũng sử dụng hai lớp nhúng: một có trọng lượng có thể huấn luyện và một có trọng lượng cố định.

```{.python .input}
class TextCNN(nn.Block):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # The embedding layer not to be trained
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)
        # The max-over-time pooling layer has no parameters, so this instance
        # can be shared
        self.pool = nn.GlobalMaxPool1D()
        # Create multiple one-dimensional convolutional layers
        self.convs = nn.Sequential()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))

    def forward(self, inputs):
        # Concatenate two embedding layer outputs with shape (batch size, no.
        # of tokens, token vector dimension) along vectors
        embeddings = np.concatenate((
            self.embedding(inputs), self.constant_embedding(inputs)), axis=2)
        # Per the input format of one-dimensional convolutional layers,
        # rearrange the tensor so that the second dimension stores channels
        embeddings = embeddings.transpose(0, 2, 1)
        # For each one-dimensional convolutional layer, after max-over-time
        # pooling, a tensor of shape (batch size, no. of channels, 1) is
        # obtained. Remove the last dimension and concatenate along channels
        encoding = np.concatenate([
            np.squeeze(self.pool(conv(embeddings)), axis=-1)
            for conv in self.convs], axis=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

```{.python .input}
#@tab pytorch
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # The embedding layer not to be trained
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # The max-over-time pooling layer has no parameters, so this instance
        # can be shared
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        # Create multiple one-dimensional convolutional layers
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # Concatenate two embedding layer outputs with shape (batch size, no.
        # of tokens, token vector dimension) along vectors
        embeddings = torch.cat((
            self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # Per the input format of one-dimensional convolutional layers,
        # rearrange the tensor so that the second dimension stores channels
        embeddings = embeddings.permute(0, 2, 1)
        # For each one-dimensional convolutional layer, after max-over-time
        # pooling, a tensor of shape (batch size, no. of channels, 1) is
        # obtained. Remove the last dimension and concatenate along channels
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

Hãy để chúng tôi tạo một ví dụ textCNN. Nó có 3 lớp phức tạp với độ rộng hạt nhân là 3, 4 và 5, tất cả đều có 100 kênh đầu ra.

```{.python .input}
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)
net.initialize(init.Xavier(), ctx=devices)
```

```{.python .input}
#@tab pytorch
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)

def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights);
```

### Đang tải Pretrained Word Vectơ

Tương tự như :numref:`sec_sentiment_rnn`, chúng tôi tải các nhúng Glove 100 chiều được đào tạo trước dưới dạng các biểu diễn mã thông báo được khởi tạo. Các đại diện token này (trọng lượng nhúng) sẽ được đào tạo trong `embedding` và cố định vào `constant_embedding`.

```{.python .input}
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.set_data(embeds)
net.constant_embedding.weight.set_data(embeds)
net.constant_embedding.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.requires_grad = False
```

### Đào tạo và đánh giá mô hình

Bây giờ chúng ta có thể đào tạo mô hình textCNN để phân tích tình cảm.

```{.python .input}
lr, num_epochs = 0.001, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.001, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

Dưới đây chúng tôi sử dụng mô hình được đào tạo để dự đoán tình cảm cho hai câu đơn giản.

```{.python .input}
#@tab all
d2l.predict_sentiment(net, vocab, 'this movie is so great')
```

```{.python .input}
#@tab all
d2l.predict_sentiment(net, vocab, 'this movie is so bad')
```

## Tóm tắt

* CNN một chiều có thể xử lý các tính năng cục bộ như $n$-gram trong văn bản.
* Tương quan chéo một chiều đa đầu vào kênh tương đương với tương quan chéo hai chiều đơn vào-kênh.
* Tổng hợp tối đa thời gian cho phép các số bước thời gian khác nhau tại các kênh khác nhau.
* Mô hình textCNN biến các biểu diễn token riêng lẻ thành các đầu ra ứng dụng hạ lưu bằng cách sử dụng các lớp phức tạp một chiều và các lớp tổng hợp tối đa thời gian.

## Bài tập

1. Điều chỉnh các siêu tham số và so sánh hai kiến trúc để phân tích tình cảm trong :numref:`sec_sentiment_rnn` và trong phần này, chẳng hạn như độ chính xác phân loại và hiệu quả tính toán.
1. Bạn có thể cải thiện thêm độ chính xác phân loại của mô hình bằng cách sử dụng các phương pháp được giới thiệu trong các bài tập của :numref:`sec_sentiment_rnn` không?
1. Thêm mã hóa vị trí trong các biểu diễn đầu vào. Nó có cải thiện độ chính xác phân loại không?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/393)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1425)
:end_tab:
