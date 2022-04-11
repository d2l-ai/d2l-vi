# Bahdanau Chú ý

:label:`sec_seq2seq_attention` 

Chúng tôi đã nghiên cứu vấn đề dịch máy trong :numref:`sec_seq2seq`, nơi chúng tôi thiết kế một kiến trúc bộ mã hóa-giải mã dựa trên hai RNNs cho trình tự để học trình tự. Cụ thể, bộ mã hóa RNN biến một chuỗi có độ dài biến đổi thành một biến ngữ cảnh hình dạng cố định, sau đó bộ giải mã RNN tạo ra mã thông báo chuỗi đầu ra (mục tiêu) theo mã thông báo dựa trên các token được tạo ra và biến ngữ cảnh. Tuy nhiên, mặc dù không phải tất cả các mã thông báo đầu vào (nguồn) đều hữu ích cho việc giải mã một mã thông báo nhất định, biến ngữ cảnh *same* mã hóa toàn bộ chuỗi đầu vào vẫn được sử dụng ở mỗi bước giải mã. 

Trong một thách thức riêng biệt nhưng liên quan về thế hệ chữ viết tay cho một chuỗi văn bản nhất định, Graves đã thiết kế một mô hình chú ý khác biệt để căn chỉnh các ký tự văn bản với dấu vết bút dài hơn nhiều, trong đó căn chỉnh chỉ di chuyển theo một hướng :cite:`Graves.2013`. Lấy cảm hứng từ ý tưởng học cách căn chỉnh, Bahdanau et al. đề xuất một mô hình chú ý khác biệt mà không có giới hạn liên kết một chiều nghiêm trọng :cite:`Bahdanau.Cho.Bengio.2014`. Khi dự đoán một mã thông báo, nếu không phải tất cả các token đầu vào đều có liên quan, mô hình sẽ căn chỉnh (hoặc tham dự) chỉ với các phần của chuỗi đầu vào có liên quan đến dự đoán hiện tại. Điều này đạt được bằng cách coi biến ngữ cảnh như một đầu ra của sự chú ý pooling. 

## Mô hình

Khi mô tả sự chú ý của Bahdanau cho bộ giải mã RNN bên dưới, chúng tôi sẽ làm theo cùng một ký hiệu trong :numref:`sec_seq2seq`. Mô hình dựa trên sự chú ý mới giống như trong :numref:`sec_seq2seq` ngoại trừ biến ngữ cảnh $\mathbf{c}$ trong :eqref:`eq_seq2seq_s_t` được thay thế bằng $\mathbf{c}_{t'}$ tại bất kỳ bước thời gian giải mã nào $t'$. Giả sử có $T$ token trong chuỗi đầu vào, biến ngữ cảnh tại bước thời gian giải mã $t'$ là đầu ra của sự chú ý pooling: 

$$\mathbf{c}_{t'} = \sum_{t=1}^T \alpha(\mathbf{s}_{t' - 1}, \mathbf{h}_t) \mathbf{h}_t,$$

trong đó bộ giải mã ẩn trạng thái $\mathbf{s}_{t' - 1}$ tại thời điểm bước $t' - 1$ là truy vấn và các trạng thái ẩn mã hóa $\mathbf{h}_t$ là cả các phím và giá trị, và trọng lượng chú ý $\alpha$ được tính như trong :eqref:`eq_attn-scoring-alpha` bằng cách sử dụng chức năng chấm điểm chú ý phụ gia được xác định bởi :eqref:`eq_additive-attn`. 

Hơi khác so với kiến trúc mã hóa giải mã RNN vani trong :numref:`fig_seq2seq_details`, kiến trúc tương tự với sự chú ý Bahdanau được mô tả trong :numref:`fig_s2s_attention_details`. 

![Layers in an RNN encoder-decoder model with Bahdanau attention.](../img/seq2seq-attention-details.svg)
:label:`fig_s2s_attention_details`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import rnn, nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## Xác định bộ giải mã với sự chú ý

Để thực hiện bộ giải mã mã RNN với sự chú ý của Bahdanau, chúng ta chỉ cần xác định lại bộ giải mã. Để hình dung các trọng lượng chú ý đã học được thuận tiện hơn, lớp `AttentionDecoder` sau định nghĩa [** giao diện cơ bản cho bộ giải mã với cơ chế chú ý**].

```{.python .input}
#@tab all
#@save
class AttentionDecoder(d2l.Decoder):
    """The base attention-based decoder interface."""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError
```

Bây giờ chúng ta hãy [** triển khai bộ giải mã RNN với sự chú ý Bahdanau**] trong lớp `Seq2SeqAttentionDecoder` sau. Trạng thái của bộ giải mã được khởi tạo với (i) các trạng thái ẩn lớp cuối của bộ mã hóa ở tất cả các bước thời gian (như các phím và giá trị của sự chú ý); (ii) bộ mã hóa tất cả các lớp ẩn trạng thái ở bước thời gian cuối cùng (để khởi tạo trạng thái ẩn của bộ giải mã); và (iii) bộ mã hóa độ dài hợp lệ (để loại trừ các thẻ đệm trong tập hợp chú ý). Tại mỗi bước thời gian giải mã, trạng thái ẩn lớp cuối của bộ giải mã ở bước thời gian trước đó được sử dụng làm truy vấn của sự chú ý. Kết quả là, cả đầu ra chú ý và nhúng đầu vào được nối làm đầu vào của bộ giải mã RNN.

```{.python .input}
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # Shape of `outputs`: (`num_steps`, `batch_size`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`,
        # `num_hiddens`)
        outputs, hidden_state = enc_outputs
        return (outputs.swapaxes(0, 1), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # Shape of `enc_outputs`: (`batch_size`, `num_steps`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`,
        # `num_hiddens`)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output `X`: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X).swapaxes(0, 1)
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of `query`: (`batch_size`, 1, `num_hiddens`)
            query = np.expand_dims(hidden_state[0][-1], axis=1)
            # Shape of `context`: (`batch_size`, 1, `num_hiddens`)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # Concatenate on the feature dimension
            x = np.concatenate((context, np.expand_dims(x, axis=1)), axis=-1)
            # Reshape `x` as (1, `batch_size`, `embed_size` + `num_hiddens`)
            out, hidden_state = self.rnn(x.swapaxes(0, 1), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # After fully-connected layer transformation, shape of `outputs`:
        # (`num_steps`, `batch_size`, `vocab_size`)
        outputs = self.dense(np.concatenate(outputs, axis=0))
        return outputs.swapaxes(0, 1), [enc_outputs, hidden_state,
                                        enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
#@tab pytorch
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # Shape of `outputs`: (`num_steps`, `batch_size`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`,
        # `num_hiddens`)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # Shape of `enc_outputs`: (`batch_size`, `num_steps`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`,
        # `num_hiddens`)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output `X`: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of `query`: (`batch_size`, 1, `num_hiddens`)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # Shape of `context`: (`batch_size`, 1, `num_hiddens`)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # Concatenate on the feature dimension
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # Reshape `x` as (1, `batch_size`, `embed_size` + `num_hiddens`)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # After fully-connected layer transformation, shape of `outputs`:
        # (`num_steps`, `batch_size`, `vocab_size`)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
#@tab tensorflow
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(num_hiddens, num_hiddens,
                                               num_hiddens, dropout)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout)
             for _ in range(num_layers)]),
                                      return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # Shape of `outputs`: (`batch_size`, `num_steps`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`, `num_hiddens`)
        outputs, hidden_state = enc_outputs
        return (outputs, hidden_state, enc_valid_lens)

    def call(self, X, state, **kwargs):
        # Shape of `enc_outputs`: (`batch_size`, `num_steps`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`, `num_hiddens`)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output `X`: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X) # Input `X` has shape: (`batch_size`, `num_steps`)
        X = tf.transpose(X, perm=(1, 0, 2))
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of `query`: (`batch_size`, 1, `num_hiddens`)
            query = tf.expand_dims(hidden_state[-1], axis=1)
            # Shape of `context`: (`batch_size, 1, `num_hiddens`)
            context = self.attention(query, enc_outputs, enc_outputs,
                                     enc_valid_lens, **kwargs)
            # Concatenate on the feature dimension
            x = tf.concat((context, tf.expand_dims(x, axis=1)), axis=-1)
            out = self.rnn(x, hidden_state, **kwargs)
            hidden_state = out[1:]
            outputs.append(out[0])
            self._attention_weights.append(self.attention.attention_weights)
        # After fully-connected layer transformation, shape of `outputs`:
        # (`batch_size`, `num_steps`, `vocab_size`)
        outputs = self.dense(tf.concat(outputs, axis=1))
        return outputs, [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

Sau đây, chúng tôi [** test the implemented decoder**] với Bahdanau chú ý sử dụng một minibatch gồm 4 chuỗi đầu vào của 7 bước thời gian.

```{.python .input}
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
encoder.initialize()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                  num_layers=2)
decoder.initialize()
X = d2l.zeros((4, 7))  # (`batch_size`, `num_steps`)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape
```

```{.python .input}
#@tab pytorch
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
encoder.eval()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                  num_layers=2)
decoder.eval()
X = d2l.zeros((4, 7), dtype=torch.long)  # (`batch_size`, `num_steps`)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape
```

```{.python .input}
#@tab tensorflow
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                  num_layers=2)
X = tf.zeros((4, 7))
state = decoder.init_state(encoder(X, training=False), None)
output, state = decoder(X, state, training=False)
output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape
```

## [**Đào tạo**]

Tương tự như :numref:`sec_seq2seq_training`, ở đây chúng tôi chỉ định hyperparemeters, khởi tạo bộ mã hóa và bộ giải mã với sự chú ý của Bahdanau và đào tạo mô hình này để dịch máy. Do cơ chế chú ý mới được thêm vào, việc đào tạo này chậm hơn nhiều so với năm :numref:`sec_seq2seq_training` mà không có cơ chế chú ý.

```{.python .input}
#@tab all
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 250, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = d2l.Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
```

Sau khi mô hình được đào tạo, chúng tôi sử dụng nó để [** dịch một vài câu tiếng Anh**] sang tiếng Pháp và tính điểm BLEU của họ.

```{.python .input}
#@tab mxnet, pytorch
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
```

```{.python .input}
#@tab tensorflow
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
```

```{.python .input}
#@tab all
attention_weights = d2l.reshape(
    d2l.concat([step[0][0][0] for step in dec_attention_weight_seq], 0),
    (1, 1, -1, num_steps))
```

Bằng cách [**visualizing the attention weights**] khi dịch câu tiếng Anh cuối cùng, chúng ta có thể thấy rằng mỗi truy vấn gán trọng lượng không thống nhất trên các cặp key-value. Nó cho thấy ở mỗi bước giải mã, các phần khác nhau của chuỗi đầu vào được tổng hợp một cách chọn lọc trong tập hợp sự chú ý.

```{.python .input}
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1],
    xlabel='Key positions', ylabel='Query positions')
```

```{.python .input}
#@tab pytorch
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
    xlabel='Key positions', ylabel='Query positions')
```

```{.python .input}
#@tab tensorflow
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1],
                  xlabel='Key posistions', ylabel='Query posistions')
```

## Tóm tắt

* Khi dự đoán một mã thông báo, nếu không phải tất cả các mã thông báo đầu vào đều có liên quan, bộ giải mã RNN với sự chú ý Bahdanau tập hợp chọn lọc các phần khác nhau của chuỗi đầu vào. Điều này đạt được bằng cách coi biến ngữ cảnh như một đầu ra của sự chú ý phụ gia tập hợp.
* Trong bộ giải mã RNN, sự chú ý Bahdanau xử lý trạng thái ẩn bộ giải mã ở bước thời gian trước như truy vấn và các trạng thái ẩn mã hóa ở tất cả các bước thời gian như cả các phím và giá trị.

## Bài tập

1. Thay thế GRU bằng LSTM trong thí nghiệm.
1. Sửa đổi thí nghiệm để thay thế chức năng chấm điểm sự chú ý phụ gia bằng sản phẩm điểm thu nhỏ. Làm thế nào để nó ảnh hưởng đến hiệu quả đào tạo?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/347)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1065)
:end_tab:
