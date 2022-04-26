# Máy biến áp
:label:`sec_transformer`

Chúng tôi đã so sánh CNN, RNN và sự tự chú ý trong :numref:`subsec_cnn-rnn-self-attention`. Đáng chú ý, sự tự chú ý thích cả tính toán song song và độ dài đường tối đa ngắn nhất. Do đó về mặt tự nhiên, nó là hấp dẫn để thiết kế kiến trúc sâu sắc bằng cách sử dụng sự tự chú ý. Không giống như các mô hình tự chú ý trước đó vẫn dựa vào RNN để biểu diễn đầu vào :cite:`Cheng.Dong.Lapata.2016,Lin.Feng.Santos.ea.2017,Paulus.Xiong.Socher.2017`, mô hình máy biến áp chỉ dựa trên các cơ chế chú ý mà không có bất kỳ lớp phức tạp hoặc tái phát nào :cite:`Vaswani.Shazeer.Parmar.ea.2017`. Mặc dù ban đầu được đề xuất cho trình tự để học trình tự về dữ liệu văn bản, các máy biến áp đã phổ biến trong một loạt các ứng dụng học sâu hiện đại, chẳng hạn như trong các lĩnh vực ngôn ngữ, tầm nhìn, lời nói và học tập củng cố. 

## Mô hình

Là một ví dụ của kiến trúc bộ mã hóa-giải mã, kiến trúc tổng thể của máy biến áp được trình bày trong :numref:`fig_transformer`. Như chúng ta có thể thấy, máy biến áp bao gồm một bộ mã hóa và bộ giải mã. Khác với sự chú ý của Bahdanau cho trình tự để học trình tự trong :numref:`fig_s2s_attention_details`, các nhúng chuỗi đầu vào (nguồn) và đầu ra (mục tiêu) được thêm vào với mã hóa vị trí trước khi được đưa vào bộ mã hóa và bộ giải mã ngăn xếp các mô-đun dựa trên sự tự chú ý. 

![The transformer architecture.](../img/transformer.svg)
:width:`500px`
:label:`fig_transformer`

Bây giờ chúng tôi cung cấp một cái nhìn tổng quan về kiến trúc biến áp trong :numref:`fig_transformer`. Ở mức độ cao, bộ mã hóa biến áp là một chồng của nhiều lớp giống hệt nhau, trong đó mỗi lớp có hai lớp con (hoặc được ký hiệu là $\mathrm{sublayer}$). Đầu tiên là một tập hợp tự chú ý nhiều đầu và thứ hai là một mạng lưới chuyển tiếp nguồn cấp dữ liệu theo định vị. Cụ thể, trong bộ mã hóa tự chú ý, truy vấn, phím và giá trị đều từ đầu ra của lớp mã hóa trước đó. Lấy cảm hứng từ thiết kế ResNet trong :numref:`sec_resnet`, một kết nối còn lại được sử dụng xung quanh cả hai lớp con. Trong máy biến áp, đối với bất kỳ đầu vào $\mathbf{x} \in \mathbb{R}^d$ nào ở bất kỳ vị trí nào của chuỗi, chúng tôi yêu cầu $\mathrm{sublayer}(\mathbf{x}) \in \mathbb{R}^d$ để kết nối còn lại $\mathbf{x} + \mathrm{sublayer}(\mathbf{x}) \in \mathbb{R}^d$ là khả thi. Bổ sung này từ kết nối còn lại ngay lập tức được theo sau bởi chuẩn hóa lớp :cite:`Ba.Kiros.Hinton.2016`. Kết quả là, bộ mã hóa biến áp xuất ra một biểu diễn vector $d$ chiều cho mỗi vị trí của chuỗi đầu vào. 

Bộ giải mã biến áp cũng là một chồng của nhiều lớp giống hệt nhau với các kết nối còn lại và chuẩn hóa lớp. Bên cạnh hai sublayers được mô tả trong bộ mã hóa, bộ giải mã chèn một sublayer thứ ba, được gọi là sự chú ý của bộ mã hóa-giải mã, giữa hai bộ giải mã này. Trong sự chú ý của bộ mã hóa giải mã, các truy vấn là từ đầu ra của lớp giải mã trước đó và các phím và giá trị là từ đầu ra bộ mã hóa biến áp. Trong bộ giải mã tự chú ý, truy vấn, khóa và giá trị đều từ đầu ra của lớp giải mã trước đó. Tuy nhiên, mỗi vị trí trong bộ giải mã chỉ được phép tham dự tất cả các vị trí trong bộ giải mã cho đến vị trí đó. Chú ý *masked* này bảo tồn thuộc tính tự động hồi quy, đảm bảo rằng dự đoán chỉ phụ thuộc vào các token đầu ra đã được tạo ra. 

Chúng tôi đã mô tả và thực hiện sự chú ý nhiều đầu dựa trên các sản phẩm dot-thu nhỏ trong :numref:`sec_multihead-attention` và mã hóa vị trí trong :numref:`subsec_positional-encoding`. Sau đây, chúng tôi sẽ thực hiện phần còn lại của mô hình máy biến áp.

```{.python .input}
from d2l import mxnet as d2l
import math
from mxnet import autograd, np, npx
from mxnet.gluon import nn
import pandas as pd
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import pandas as pd
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import pandas as pd
import tensorflow as tf
```

## [**Các mạng thức ăn chuyển tiếp vị trí**]

Mạng chuyển tiếp nguồn cấp dữ liệu định vị biến đổi biểu diễn ở tất cả các vị trí trình tự bằng cách sử dụng cùng một MLP. Đây là lý do tại sao chúng tôi gọi nó là *positionwise*. Trong phần thực hiện dưới đây, đầu vào `X` với hình dạng (kích thước lô, số bước thời gian hoặc độ dài chuỗi trong thẻ, số đơn vị ẩn hoặc kích thước tính năng) sẽ được chuyển đổi bởi một MLP hai lớp thành một tensor đầu ra của hình dạng (kích thước lô, số bước thời gian, `ffn_num_outputs`).

```{.python .input}
#@save
class PositionWiseFFN(nn.Block):
    """Positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Dense(ffn_num_hiddens, flatten=False,
                               activation='relu')
        self.dense2 = nn.Dense(ffn_num_outputs, flatten=False)

    def forward(self, X):
        return self.dense2(self.dense1(X))
```

```{.python .input}
#@tab pytorch
#@save
class PositionWiseFFN(nn.Module):
    """Positionwise feed-forward network."""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
```

```{.python .input}
#@tab tensorflow
#@save
class PositionWiseFFN(tf.keras.layers.Layer):
    """Positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super().__init__(*kwargs)
        self.dense1 = tf.keras.layers.Dense(ffn_num_hiddens)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(ffn_num_outputs)

    def call(self, X):
        return self.dense2(self.relu(self.dense1(X)))
```

Ví dụ sau đây cho thấy [** chiều trong cùng của một tensor thay đổi**] thành số lượng đầu ra trong mạng chuyển tiếp theo vị trí. Vì cùng một MLP biến đổi ở tất cả các vị trí, khi các đầu vào ở tất cả các vị trí này giống nhau, đầu ra của chúng cũng giống hệt nhau.

```{.python .input}
ffn = PositionWiseFFN(4, 8)
ffn.initialize()
ffn(np.ones((2, 3, 4)))[0]
```

```{.python .input}
#@tab pytorch
ffn = PositionWiseFFN(4, 4, 8)
ffn.eval()
ffn(d2l.ones((2, 3, 4)))[0]
```

```{.python .input}
#@tab tensorflow
ffn = PositionWiseFFN(4, 8)
ffn(tf.ones((2, 3, 4)))[0]
```

## Kết nối còn lại và chuẩn hóa lớp

Bây giờ chúng ta hãy tập trung vào thành phần “add & norm” trong :numref:`fig_transformer`. Như chúng tôi đã mô tả ở đầu phần này, đây là một kết nối còn lại ngay lập tức theo sau là chuẩn hóa lớp. Cả hai đều là chìa khóa cho các kiến trúc sâu hiệu quả. 

Trong :numref:`sec_batch_norm`, chúng tôi đã giải thích cách bình thường hóa hàng loạt lại và rescales trên các ví dụ trong một minibatch. Chuẩn hóa lớp giống như bình thường hóa hàng loạt ngoại trừ việc trước đây bình thường hóa trên kích thước tính năng. Mặc dù các ứng dụng phổ biến của nó trong tầm nhìn máy tính, việc bình thường hóa hàng loạt thường kém hiệu quả hơn so với bình thường hóa lớp trong các tác vụ xử lý ngôn ngữ tự nhiên, có đầu vào thường là chuỗi độ dài thay đổi. 

Đoạn mã sau đây [** so sánh bình thường hóa trên các kích thước khác nhau theo chuẩn hóa lớp và chuẩn hóa hàng lạnh**].

```{.python .input}
ln = nn.LayerNorm()
ln.initialize()
bn = nn.BatchNorm()
bn.initialize()
X = d2l.tensor([[1, 2], [2, 3]])
# Compute mean and variance from `X` in the training mode
with autograd.record():
    print('layer norm:', ln(X), '\nbatch norm:', bn(X))
```

```{.python .input}
#@tab pytorch
ln = nn.LayerNorm(2)
bn = nn.BatchNorm1d(2)
X = d2l.tensor([[1, 2], [2, 3]], dtype=torch.float32)
# Compute mean and variance from `X` in the training mode
print('layer norm:', ln(X), '\nbatch norm:', bn(X))
```

```{.python .input}
#@tab tensorflow
ln = tf.keras.layers.LayerNormalization()
bn = tf.keras.layers.BatchNormalization()
X = tf.constant([[1, 2], [2, 3]], dtype=tf.float32)
print('layer norm:', ln(X), '\nbatch norm:', bn(X, training=True))
```

Bây giờ chúng ta có thể thực hiện lớp `AddNorm` [** sử dụng một kết nối còn lại theo sau là chuẩn hóa lớp **]. Dropout cũng được áp dụng để thường xuyên hóa.

```{.python .input}
#@save
class AddNorm(nn.Block):
    """Residual connection followed by layer normalization."""
    def __init__(self, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm()

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
```

```{.python .input}
#@tab pytorch
#@save
class AddNorm(nn.Module):
    """Residual connection followed by layer normalization."""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
```

```{.python .input}
#@tab tensorflow
#@save
class AddNorm(tf.keras.layers.Layer):
    """Residual connection followed by layer normalization."""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(normalized_shape)
        
    def call(self, X, Y, **kwargs):
        return self.ln(self.dropout(Y, **kwargs) + X)
```

Kết nối còn lại yêu cầu hai đầu vào có cùng hình dạng sao cho [**tensor đầu ra cũng có hình dạng giống nhau sau khi hoạt động bổ sung**].

```{.python .input}
add_norm = AddNorm(0.5)
add_norm.initialize()
add_norm(d2l.ones((2, 3, 4)), d2l.ones((2, 3, 4))).shape
```

```{.python .input}
#@tab pytorch
add_norm = AddNorm([3, 4], 0.5) # Normalized_shape is input.size()[1:]
add_norm.eval()
add_norm(d2l.ones((2, 3, 4)), d2l.ones((2, 3, 4))).shape
```

```{.python .input}
#@tab tensorflow
add_norm = AddNorm([1, 2], 0.5) # Normalized_shape is: [i for i in range(len(input.shape))][1:]
add_norm(tf.ones((2, 3, 4)), tf.ones((2, 3, 4)), training=False).shape
```

## Bộ mã hóa

Với tất cả các thành phần thiết yếu để lắp ráp bộ mã hóa biến áp, chúng ta hãy bắt đầu bằng cách thực hiện [** một lớp duy nhất trong bộ mã hóa**]. Lớp `EncoderBlock` sau chứa hai lớp con: nhiều đầu tự chú ý và các mạng chuyển tiếp theo định vị, trong đó một kết nối còn lại tiếp theo là chuẩn hóa lớp được sử dụng xung quanh cả hai lớp con.

```{.python .input}
#@save
class EncoderBlock(nn.Block):
    """Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(
            num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
```

```{.python .input}
#@tab pytorch
#@save
class EncoderBlock(nn.Module):
    """Transformer encoder block."""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
```

```{.python .input}
#@tab tensorflow
#@save
class EncoderBlock(tf.keras.layers.Layer):
    """Transformer encoder block."""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens,
                                                num_heads, dropout, bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        
    def call(self, X, valid_lens, **kwargs):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens, **kwargs), **kwargs)
        return self.addnorm2(Y, self.ffn(Y), **kwargs)
```

Như chúng ta có thể thấy, [** bất kỳ lớp nào trong bộ mã hóa biến áp không thay đổi hình dạng của đầu vào của nó.**]

```{.python .input}
X = d2l.ones((2, 100, 24))
valid_lens = d2l.tensor([3, 2])
encoder_blk = EncoderBlock(24, 48, 8, 0.5)
encoder_blk.initialize()
encoder_blk(X, valid_lens).shape
```

```{.python .input}
#@tab pytorch
X = d2l.ones((2, 100, 24))
valid_lens = d2l.tensor([3, 2])
encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
encoder_blk.eval()
encoder_blk(X, valid_lens).shape
```

```{.python .input}
#@tab tensorflow
X = tf.ones((2, 100, 24))
valid_lens = tf.constant([3, 2])
norm_shape = [i for i in range(len(X.shape))][1:]
encoder_blk = EncoderBlock(24, 24, 24, 24, norm_shape, 48, 8, 0.5)
encoder_blk(X, valid_lens, training=False).shape
```

Trong thực hiện [**transformer encoder**] sau đây, chúng tôi xếp chồng `num_layers` phiên bản của `EncoderBlock` lớp trên. Vì chúng ta sử dụng mã hóa vị trí cố định có giá trị luôn nằm trong khoảng từ -1 đến 1, chúng ta nhân các giá trị của các nhúng đầu vào có thể học được bằng căn bậc hai của kích thước nhúng để giải phóng trước khi tổng hợp nhúng đầu vào và mã hóa vị trí.

```{.python .input}
#@save
class TransformerEncoder(d2l.Encoder):
    """Transformer encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for _ in range(num_layers):
            self.blks.add(
                EncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout,
                             use_bias))

    def forward(self, X, valid_lens, *args):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
```

```{.python .input}
#@tab pytorch
#@save
class TransformerEncoder(d2l.Encoder):
    """Transformer encoder."""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
```

```{.python .input}
#@tab tensorflow
#@save
class TransformerEncoder(d2l.Encoder):
    """Transformer encoder."""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_hiddens, num_heads,
                 num_layers, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = [EncoderBlock(
            key_size, query_size, value_size, num_hiddens, norm_shape,
            ffn_num_hiddens, num_heads, dropout, bias) for _ in range(
            num_layers)]
        
    def call(self, X, valid_lens, **kwargs):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(self.embedding(X) * tf.math.sqrt(
            tf.cast(self.num_hiddens, dtype=tf.float32)), **kwargs)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens, **kwargs)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
```

Dưới đây chúng tôi chỉ định các siêu tham số để [** tạo một bộ mã hóa biến áp hai lớp**]. Hình dạng của đầu ra bộ mã hóa biến áp là (kích thước lô, số bước thời gian, `num_hiddens`).

```{.python .input}
encoder = TransformerEncoder(200, 24, 48, 8, 2, 0.5)
encoder.initialize()
encoder(np.ones((2, 100)), valid_lens).shape
```

```{.python .input}
#@tab pytorch
encoder = TransformerEncoder(
    200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
encoder.eval()
encoder(d2l.ones((2, 100), dtype=torch.long), valid_lens).shape
```

```{.python .input}
#@tab tensorflow
encoder = TransformerEncoder(200, 24, 24, 24, 24, [1, 2], 48, 8, 2, 0.5)
encoder(tf.ones((2, 100)), valid_lens, training=False).shape
```

## Bộ giải mã

Như thể hiện trong :numref:`fig_transformer`, [** bộ giải mã biến áp bao gồm nhiều lớp giống hệt nhau**]. Mỗi lớp được triển khai trong lớp `DecoderBlock` sau, chứa ba lớp con: bộ giải mã tự chú ý, chú ý bộ mã hóa-giải mã, và các mạng chuyển tiếp nguồn cấp dữ liệu định vị. Các lớp con này sử dụng một kết nối còn lại xung quanh chúng theo sau là chuẩn hóa lớp. 

Như chúng ta đã mô tả trước đó trong phần này, trong bộ giải mã đa đầu được đeo mặt nạ tự chú ý (con đầu tiên), truy vấn, khóa và giá trị đều đến từ đầu ra của lớp giải mã trước đó. Khi đào tạo các mô hình trình tự theo trình tự, mã thông báo ở tất cả các vị trí (bước thời gian) của chuỗi đầu ra được biết đến. Tuy nhiên, trong quá trình dự đoán chuỗi đầu ra được tạo ra token bằng mã thông báo; do đó, tại bất kỳ bước thời gian giải mã nào chỉ có các token được tạo ra có thể được sử dụng trong bộ giải mã tự chú ý. Để duy trì hồi quy tự động trong bộ giải mã, sự tự chú ý đeo mặt nạ của nó chỉ định `dec_valid_lens` để bất kỳ truy vấn nào chỉ tham quan đến tất cả các vị trí trong bộ giải mã cho đến vị trí truy vấn.

```{.python .input}
class DecoderBlock(nn.Block):
    # The `i`-th block in the decoder
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm1 = AddNorm(dropout)
        self.attention2 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm2 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # During training, all the tokens of any output sequence are processed
        # at the same time, so `state[2][self.i]` is `None` as initialized.
        # When decoding any output sequence token by token during prediction,
        # `state[2][self.i]` contains representations of the decoded output at
        # the `i`-th block up to the current time step
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = np.concatenate((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values

        if autograd.is_training():
            batch_size, num_steps, _ = X.shape
            # Shape of `dec_valid_lens`: (`batch_size`, `num_steps`), where
            # every row is [1, 2, ..., `num_steps`]
            dec_valid_lens = np.tile(np.arange(1, num_steps + 1, ctx=X.ctx),
                                     (batch_size, 1))
        else:
            dec_valid_lens = None

        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # Encoder-decoder attention. Shape of `enc_outputs`:
        # (`batch_size`, `num_steps`, `num_hiddens`)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state
```

```{.python .input}
#@tab pytorch
class DecoderBlock(nn.Module):
    # The `i`-th block in the decoder
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # During training, all the tokens of any output sequence are processed
        # at the same time, so `state[2][self.i]` is `None` as initialized.
        # When decoding any output sequence token by token during prediction,
        # `state[2][self.i]` contains representations of the decoded output at
        # the `i`-th block up to the current time step
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # Shape of `dec_valid_lens`: (`batch_size`, `num_steps`), where
            # every row is [1, 2, ..., `num_steps`]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # Encoder-decoder attention. Shape of `enc_outputs`:
        # (`batch_size`, `num_steps`, `num_hiddens`)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state
```

```{.python .input}
#@tab tensorflow
class DecoderBlock(tf.keras.layers.Layer):
    # The `i`-th block in the decoder
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_hiddens, num_heads, dropout, i, **kwargs):
        super().__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)
        
    def call(self, X, state, **kwargs):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # During training, all the tokens of any output sequence are processed
        # at the same time, so `state[2][self.i]` is `None` as initialized.
        # When decoding any output sequence token by token during prediction,
        # `state[2][self.i]` contains representations of the decoded output at
        # the `i`-th block up to the current time step
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = tf.concat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if kwargs["training"]:
            batch_size, num_steps, _ = X.shape
            # Shape of `dec_valid_lens`: (`batch_size`, `num_steps`), where
            # every row is [1, 2, ..., `num_steps`]
            dec_valid_lens = tf.repeat(tf.reshape(tf.range(1, num_steps + 1),
                                                 shape=(-1, num_steps)), repeats=batch_size, axis=0)

        else:
            dec_valid_lens = None
            
        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens, **kwargs)
        Y = self.addnorm1(X, X2, **kwargs)
        # Encoder-decoder attention. Shape of `enc_outputs`: (`batch_size`, `num_steps`, `num_hiddens`)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens, **kwargs)
        Z = self.addnorm2(Y, Y2, **kwargs)
        return self.addnorm3(Z, self.ffn(Z), **kwargs), state
```

Để tạo điều kiện cho các hoạt động sản phẩm điểm thu nhỏ trong bộ giải mã mã hóa chú ý và các hoạt động bổ sung trong các kết nối còn lại, [** kích thước tính năng (`num_hiddens`) của bộ giải mã giống như của bộ mã hóa.**]

```{.python .input}
decoder_blk = DecoderBlock(24, 48, 8, 0.5, 0)
decoder_blk.initialize()
X = np.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
decoder_blk(X, state)[0].shape
```

```{.python .input}
#@tab pytorch
decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
decoder_blk.eval()
X = d2l.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
decoder_blk(X, state)[0].shape
```

```{.python .input}
#@tab tensorflow
decoder_blk = DecoderBlock(24, 24, 24, 24, [1, 2], 48, 8, 0.5, 0)
X = tf.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
decoder_blk(X, state, training=False)[0].shape
```

Bây giờ chúng ta [** xây dựng toàn bộ bộ giải mã transformer **] bao gồm `num_layers` trường hợp của `DecoderBlock`. Cuối cùng, một lớp kết nối hoàn toàn tính toán dự đoán cho tất cả các token đầu ra có thể có `vocab_size`. Cả hai trọng lượng tự chú ý của bộ giải mã và trọng lượng chú ý bộ giải mã hóa được lưu trữ để trực quan hóa sau này.

```{.python .input}
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add(
                DecoderBlock(num_hiddens, ffn_num_hiddens, num_heads,
                             dropout, i))
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # Decoder self-attention weights
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
#@tab pytorch
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # Decoder self-attention weights
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
#@tab tensorflow
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_hidens, num_heads, num_layers, dropout, **kwargs):
        super().__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = [DecoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape,
                                  ffn_num_hiddens, num_heads, dropout, i) for i in range(num_layers)]
        self.dense = tf.keras.layers.Dense(vocab_size)
        
    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]
    
    def call(self, X, state, **kwargs):
        X = self.pos_encoding(self.embedding(X) * tf.math.sqrt(tf.cast(self.num_hiddens, dtype=tf.float32)), **kwargs)
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]  # 2 Attention layers in decoder
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state, **kwargs)
            # Decoder self-attention weights
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state
    
    @property
    def attention_weights(self):
        return self._attention_weights
```

## [**Đào tạo**]

Hãy để chúng tôi khởi tạo một mô hình bộ mã hóa-giải mã bằng cách làm theo kiến trúc biến áp. Ở đây chúng tôi chỉ định rằng cả bộ mã hóa biến áp và bộ giải mã biến áp có 2 lớp sử dụng sự chú ý 4 đầu. Tương tự như :numref:`sec_seq2seq_training`, chúng tôi đào tạo mô hình biến áp cho trình tự để học trình tự trên tập dữ liệu dịch máy tiếng Anh-Pháp.

```{.python .input}
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_hiddens, num_heads = 64, 4

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

encoder = TransformerEncoder(
    len(src_vocab), num_hiddens, ffn_num_hiddens, num_heads, num_layers,
    dropout)
decoder = TransformerDecoder(
    len(tgt_vocab), num_hiddens, ffn_num_hiddens, num_heads, num_layers,
    dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
```

```{.python .input}
#@tab pytorch
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

encoder = TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
decoder = TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
```

```{.python .input}
#@tab tensorflow
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_hiddens, num_heads = 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [2]

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens, norm_shape,
    ffn_num_hiddens, num_heads, num_layers, dropout)
decoder = TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens, norm_shape,
    ffn_num_hiddens, num_heads, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
```

Sau khi đào tạo, chúng tôi sử dụng mô hình biến áp để [** dịch một vài câu tiếng Anh**] sang tiếng Pháp và tính điểm BLEU của họ.

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

Hãy để chúng tôi [** hình dung trọng lượng chú ý của máy biến áp **] khi dịch câu tiếng Anh cuối cùng sang tiếng Pháp. Hình dạng của các quả cân tự chú ý của bộ mã hóa là (số lớp mã hóa, số lượng đầu chú ý, `num_steps` hoặc số truy vấn, `num_steps` hoặc số cặp giá trị khóa).

```{.python .input}
#@tab all
enc_attention_weights = d2l.reshape(
    d2l.concat(net.encoder.attention_weights, 0),
    (num_layers, num_heads, -1, num_steps))
enc_attention_weights.shape
```

Trong bộ mã hóa tự chú ý, cả truy vấn và phím đều đến từ cùng một chuỗi đầu vào. Vì thẻ đệm không mang ý nghĩa, với độ dài hợp lệ được chỉ định của chuỗi đầu vào, không có truy vấn nào tham dự các vị trí của thẻ đệm. Sau đây, hai lớp trọng lượng chú ý nhiều đầu được trình bày từng hàng. Mỗi đầu độc lập tham dự dựa trên một không gian con đại diện riêng biệt của các truy vấn, khóa và giá trị.

```{.python .input}
#@tab mxnet, tensorflow
d2l.show_heatmaps(
    enc_attention_weights, xlabel='Key positions', ylabel='Query positions',
    titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
```

```{.python .input}
#@tab pytorch
d2l.show_heatmaps(
    enc_attention_weights.cpu(), xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))
```

[**Để hình dung cả trọng lượng tự chú ý của bộ giải mã và trọng lượng chú ý bộ giải mã hóa, chúng ta cần nhiều thao tác dữ liệu**] Ví dụ, chúng tôi lấp đầy trọng lượng chú ý đeo mặt nạ bằng 0. Lưu ý rằng trọng lượng tự chú ý của bộ giải mã và trọng lượng chú ý bộ giải mã hóa cả hai đều có cùng một truy vấn: mã thông báo bắt đầu theo sau là mã thông báo đầu ra.

```{.python .input}
dec_attention_weights_2d = [d2l.tensor(head[0]).tolist()
                            for step in dec_attention_weight_seq
                            for attn in step for blk in attn for head in blk]
dec_attention_weights_filled = d2l.tensor(
    pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
dec_attention_weights = d2l.reshape(dec_attention_weights_filled,
                                    (-1, 2, num_layers, num_heads, num_steps))
dec_self_attention_weights, dec_inter_attention_weights = \
    dec_attention_weights.transpose(1, 2, 3, 0, 4)
dec_self_attention_weights.shape, dec_inter_attention_weights.shape
```

```{.python .input}
#@tab pytorch
dec_attention_weights_2d = [head[0].tolist()
                            for step in dec_attention_weight_seq
                            for attn in step for blk in attn for head in blk]
dec_attention_weights_filled = d2l.tensor(
    pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
dec_attention_weights = d2l.reshape(dec_attention_weights_filled,
                                    (-1, 2, num_layers, num_heads, num_steps))
dec_self_attention_weights, dec_inter_attention_weights = \
    dec_attention_weights.permute(1, 2, 3, 0, 4)
dec_self_attention_weights.shape, dec_inter_attention_weights.shape
```

```{.python .input}
#@tab tensorflow
dec_attention_weights_2d = [head[0] for step in dec_attention_weight_seq
                            for attn in step 
                            for blk in attn for head in blk]
dec_attention_weights_filled = tf.convert_to_tensor(
    np.asarray(pd.DataFrame(dec_attention_weights_2d).fillna(
        0.0).values).astype(np.float32))
dec_attention_weights = tf.reshape(dec_attention_weights_filled, shape=(
    -1, 2, num_layers, num_heads, num_steps))
dec_self_attention_weights, dec_inter_attention_weights = tf.transpose(
    dec_attention_weights, perm=(1, 2, 3, 0, 4))
print(dec_self_attention_weights.shape, dec_inter_attention_weights.shape)
```

Do thuộc tính tự động hồi quy của bộ giải mã tự chú ý, không có truy vấn nào tham dự các cặp khóa-giá trị sau vị trí truy vấn.

```{.python .input}
#@tab all
# Plus one to include the beginning-of-sequence token
d2l.show_heatmaps(
    dec_self_attention_weights[:, :, :, :len(translation.split()) + 1],
    xlabel='Key positions', ylabel='Query positions',
    titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
```

Tương tự như trường hợp trong bộ mã hóa tự chú ý, thông qua độ dài hợp lệ được chỉ định của chuỗi đầu vào, [** không có truy vấn từ chuỗi đầu ra tham quan đến các thẻ đệm đó từ chuỗi đầu vào. **]

```{.python .input}
#@tab all
d2l.show_heatmaps(
    dec_inter_attention_weights, xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))
```

Mặc dù kiến trúc biến áp ban đầu được đề xuất để học theo trình tự, vì chúng ta sẽ khám phá sau trong cuốn sách, bộ mã hóa biến áp hoặc bộ giải mã biến áp thường được sử dụng riêng cho các nhiệm vụ học sâu khác nhau. 

## Tóm tắt

* Máy biến áp là một thể hiện của kiến trúc bộ mã hóa-giải mã, mặc dù bộ mã hóa hoặc bộ giải mã có thể được sử dụng riêng lẻ trong thực tế.
* Trong máy biến áp, tự chú ý nhiều đầu được sử dụng để biểu diễn trình tự đầu vào và chuỗi đầu ra, mặc dù bộ giải mã phải bảo toàn thuộc tính hồi quy tự động thông qua một phiên bản đeo mặt nạ.
* Cả hai kết nối còn lại và bình thường hóa lớp trong máy biến áp đều quan trọng để đào tạo một mô hình rất sâu.
* Mạng chuyển tiếp nguồn cấp dữ liệu định vị trong mô hình máy biến áp biến đổi biểu diễn ở tất cả các vị trí trình tự bằng cách sử dụng cùng một MLP.

## Bài tập

1. Đào tạo một máy biến áp sâu hơn trong các thí nghiệm. Làm thế nào để nó ảnh hưởng đến tốc độ đào tạo và hiệu suất dịch thuật?
1. Có phải là một ý tưởng tốt để thay thế sự chú ý của sản phẩm điểm thu nhỏ bằng sự chú ý phụ gia trong máy biến áp? Tại sao?
1. Đối với mô hình hóa ngôn ngữ, chúng ta có nên sử dụng bộ mã hóa biến áp, bộ giải mã hoặc cả hai? Làm thế nào để thiết kế phương pháp này?
1. Điều gì có thể là thách thức đối với máy biến áp nếu chuỗi đầu vào rất dài? Tại sao?
1. Làm thế nào để cải thiện tính toán và hiệu quả bộ nhớ của máy biến áp? Hint: you may refer to the survey paper by Tay et al. :cite:`Tay.Dehghani.Bahri.ea.2020`.
1. Làm thế nào chúng ta có thể thiết kế các mô hình dựa trên máy biến áp cho các tác vụ phân loại hình ảnh mà không cần sử dụng CNN? Hint: you may refer to the vision transformer :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/348)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1066)
:end_tab:
