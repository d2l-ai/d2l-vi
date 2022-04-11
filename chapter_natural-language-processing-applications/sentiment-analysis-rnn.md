# Phân tích tình cảm: Sử dụng mạng nơ-ron tái phát
:label:`sec_sentiment_rnn`

Giống như sự tương đồng từ và các nhiệm vụ tương tự, chúng ta cũng có thể áp dụng các vectơ từ được đào tạo trước vào phân tích tình cảm. Vì tập dữ liệu đánh giá IMDb trong :numref:`sec_sentiment` không lớn lắm, sử dụng các đại diện văn bản đã được đào tạo trước trên corpora quy mô lớn có thể làm giảm quá mức của mô hình. Như một ví dụ cụ thể được minh họa trong :numref:`fig_nlp-map-sa-rnn`, chúng tôi sẽ đại diện cho mỗi mã thông báo bằng cách sử dụng mô hình Glove được đào tạo trước và đưa các biểu diễn token này thành RNN hai chiều nhiều lớp để có được biểu diễn chuỗi văn bản, sẽ được chuyển thành đầu ra phân tích tình cảm :cite:`Maas.Daly.Pham.ea.2011`. Đối với cùng một ứng dụng hạ lưu, chúng tôi sẽ xem xét một sự lựa chọn kiến trúc khác nhau sau này. 

![This section feeds pretrained GloVe to an RNN-based architecture for sentiment analysis.](../img/nlp-map-sa-rnn.svg)
:label:`fig_nlp-map-sa-rnn`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn, rnn
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

## Đại diện cho văn bản đơn với RNNs

Trong các tác vụ phân loại văn bản, chẳng hạn như phân tích tình cảm, một chuỗi văn bản có độ dài khác nhau sẽ được chuyển đổi thành các loại có độ dài cố định. Trong lớp `BiRNN` sau, trong khi mỗi token của một chuỗi văn bản được biểu diễn Glove trước riêng lẻ của nó thông qua lớp nhúng (`self.embedding`), toàn bộ chuỗi được mã hóa bởi một RNN hai chiều (`self.encoder`). Cụ thể hơn, các trạng thái ẩn (ở lớp cuối cùng) của LSTM hai chiều ở cả hai bước thời gian ban đầu và cuối cùng được nối như là biểu diễn của chuỗi văn bản. Biểu diễn văn bản đơn này sau đó được chuyển đổi thành các loại đầu ra bởi một lớp kết nối hoàn toàn (`self.decoder`) với hai đầu ra (“dương” và “âm”).

```{.python .input}
class BiRNN(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Set `bidirectional` to True to get a bidirectional RNN
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # The shape of `inputs` is (batch size, no. of time steps). Because
        # LSTM requires its input's first dimension to be the temporal
        # dimension, the input is transposed before obtaining token
        # representations. The output shape is (no. of time steps, batch size,
        # word vector dimension)
        embeddings = self.embedding(inputs.T)
        # Returns hidden states of the last hidden layer at different time
        # steps. The shape of `outputs` is (no. of time steps, batch size,
        # 2 * no. of hidden units)
        outputs = self.encoder(embeddings)
        # Concatenate the hidden states at the initial and final time steps as
        # the input of the fully-connected layer. Its shape is (batch size,
        # 4 * no. of hidden units)
        encoding = np.concatenate((outputs[0], outputs[-1]), axis=1)
        outs = self.decoder(encoding)
        return outs
```

```{.python .input}
#@tab pytorch
class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Set `bidirectional` to True to get a bidirectional RNN
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                                bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # The shape of `inputs` is (batch size, no. of time steps). Because
        # LSTM requires its input's first dimension to be the temporal
        # dimension, the input is transposed before obtaining token
        # representations. The output shape is (no. of time steps, batch size,
        # word vector dimension)
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # Returns hidden states of the last hidden layer at different time
        # steps. The shape of `outputs` is (no. of time steps, batch size,
        # 2 * no. of hidden units)
        outputs, _ = self.encoder(embeddings)
        # Concatenate the hidden states of the initial time step and final
        # time step to use as the input of the fully connected layer. Its
        # shape is (batch size, 4 * no. of hidden units)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        # Concatenate the hidden states at the initial and final time steps as
        # the input of the fully-connected layer. Its shape is (batch size,
        # 4 * no. of hidden units)
        outs = self.decoder(encoding)
        return outs
```

Chúng ta hãy xây dựng một RNN hai chiều với hai lớp ẩn để đại diện cho văn bản duy nhất để phân tích tình cảm.

```{.python .input}
#@tab all
embed_size, num_hiddens, num_layers, devices = 100, 100, 2, d2l.try_all_gpus()
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
```

```{.python .input}
net.initialize(init.Xavier(), ctx=devices)
```

```{.python .input}
#@tab pytorch
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])
net.apply(init_weights);
```

## Đang tải Pretrained Word Vectơ

Dưới đây chúng tôi tải các bản nhúng Glove 100 chiều được đào tạo trước (cần phải phù hợp với `embed_size`) Glove cho các mã thông báo trong từ vựng.

```{.python .input}
#@tab all
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
```

In hình dạng của các vectơ cho tất cả các mã thông báo trong từ vựng.

```{.python .input}
#@tab all
embeds = glove_embedding[vocab.idx_to_token]
embeds.shape
```

Chúng tôi sử dụng các vectơ từ được đào tạo trước này để đại diện cho các mã thông báo trong các đánh giá và sẽ không cập nhật các vectơ này trong quá trình đào tạo.

```{.python .input}
net.embedding.weight.set_data(embeds)
net.embedding.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False
```

## Đào tạo và đánh giá mô hình

Bây giờ chúng ta có thể đào tạo RNN hai chiều để phân tích tình cảm.

```{.python .input}
lr, num_epochs = 0.01, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

Chúng tôi xác định hàm sau để dự đoán tình cảm của một chuỗi văn bản bằng cách sử dụng mô hình được đào tạo `net`.

```{.python .input}
#@save
def predict_sentiment(net, vocab, sequence):
    """Predict the sentiment of a text sequence."""
    sequence = np.array(vocab[sequence.split()], ctx=d2l.try_gpu())
    label = np.argmax(net(sequence.reshape(1, -1)), axis=1)
    return 'positive' if label == 1 else 'negative'
```

```{.python .input}
#@tab pytorch
#@save
def predict_sentiment(net, vocab, sequence):
    """Predict the sentiment of a text sequence."""
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'
```

Cuối cùng, chúng ta hãy sử dụng mô hình được đào tạo để dự đoán tình cảm cho hai câu đơn giản.

```{.python .input}
#@tab all
predict_sentiment(net, vocab, 'this movie is so great')
```

```{.python .input}
#@tab all
predict_sentiment(net, vocab, 'this movie is so bad')
```

## Tóm tắt

* Vectơ từ được đào tạo trước có thể đại diện cho các mã thông báo riêng lẻ trong một chuỗi văn bản.
* Các RNN hai chiều có thể đại diện cho một chuỗi văn bản, chẳng hạn như thông qua việc nối các trạng thái ẩn của nó ở các bước thời gian ban đầu và cuối cùng. Biểu diễn văn bản đơn này có thể được chuyển đổi thành các danh mục bằng cách sử dụng một lớp được kết nối hoàn toàn.

## Bài tập

1. Tăng số lượng kỷ nguyên. Bạn có thể cải thiện việc đào tạo và kiểm tra độ chính xác? Làm thế nào về điều chỉnh các siêu tham số khác?
1. Sử dụng các vectơ từ được đào tạo trước lớn hơn, chẳng hạn như nhúng găng tay 300 chiều. Nó có cải thiện độ chính xác phân loại không?
1. Chúng ta có thể cải thiện độ chính xác phân loại bằng cách sử dụng token hóa spacy không? Bạn cần cài đặt spacy (`pip install spacy`) và cài đặt gói tiếng Anh (`python -m spacy download en`). Trong mã, đầu tiên, nhập khẩu spacy (`import spacy`). Sau đó, tải gói spacy tiếng Anh (`spacy_en = spacy.load('en')`). Cuối cùng, xác định hàm `def tokenizer(text): return [tok.text for tok in spacy_en.tokenizer(text)]` và thay thế hàm `tokenizer` ban đầu. Lưu ý các dạng mã thông báo cụm từ khác nhau trong Glove và spacy. Ví dụ, cụm từ token “new york” có dạng “new-york” trong Glove và dạng “new york” sau khi mã hóa spacy.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/392)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1424)
:end_tab:
