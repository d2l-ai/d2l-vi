# Kiến trúc mã hóa-Decoder
:label:`sec_encoder-decoder`

Như chúng ta đã thảo luận trong :numref:`sec_machine_translation`, dịch máy là một tên miền vấn đề lớn cho các mô hình chuyển tải chuỗi, có đầu vào và đầu ra đều là chuỗi độ dài biến đổi. Để xử lý loại đầu vào và đầu ra này, chúng ta có thể thiết kế một kiến trúc với hai thành phần chính. Thành phần đầu tiên là bộ mã hóa*: nó lấy một chuỗi có độ dài thay đổi làm đầu vào và biến nó thành trạng thái có hình dạng cố định. Thành phần thứ hai là một *decoder*: nó ánh xạ trạng thái mã hóa của một hình dạng cố định đến một chuỗi có chiều dài biến đổi. Đây được gọi là kiến trúc *encoder-decoder*, được mô tả trong :numref:`fig_encoder_decoder`. 

![The encoder-decoder architecture.](../img/encoder-decoder.svg)
:label:`fig_encoder_decoder`

Hãy để chúng tôi lấy bản dịch máy từ tiếng Anh sang tiếng Pháp làm ví dụ. Cho một chuỗi đầu vào bằng tiếng Anh: “They”, “are”, “watching”,” . “, kiến trúc bộ mã hóa giải mã này đầu tiên mã hóa đầu vào có độ dài biến đổi thành một trạng thái, sau đó giải mã trạng thái để tạo ra token chuỗi dịch bằng token làm đầu ra: “Ils”, “regardent”, “.”. Vì kiến trúc bộ mã hóa-giải mã tạo thành nền tảng của các mô hình chuyển đổi chuỗi khác nhau trong các phần tiếp theo, phần này sẽ chuyển đổi kiến trúc này thành một giao diện sẽ được triển khai sau này. 

## (**Bộ mã hóa**)

Trong giao diện bộ mã hóa, chúng tôi chỉ xác định rằng bộ mã hóa lấy các chuỗi có độ dài thay đổi làm đầu vào `X`. Việc triển khai sẽ được cung cấp bởi bất kỳ mô hình nào kế thừa lớp cơ sở `Encoder` này.

```{.python .input}
from mxnet.gluon import nn

#@save
class Encoder(nn.Block):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
#@tab pytorch
from torch import nn

#@save
class Encoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

#@save
class Encoder(tf.keras.layers.Layer):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def call(self, X, *args, **kwargs):
        raise NotImplementedError
```

## [**Decoder**]

Trong giao diện giải mã sau, chúng tôi thêm một chức năng `init_state` bổ sung để chuyển đổi đầu ra bộ mã hóa (`enc_outputs`) vào trạng thái mã hóa. Lưu ý rằng bước này có thể cần thêm các đầu vào như độ dài hợp lệ của đầu vào, được giải thích trong :numref:`subsec_mt_data_loading`. Để tạo ra một mã thông báo chuỗi có độ dài biến đổi theo mã thông báo, mỗi khi bộ giải mã có thể ánh xạ một đầu vào (ví dụ, mã thông báo được tạo ra ở bước thời gian trước) và trạng thái mã hóa thành một mã thông báo đầu ra ở bước thời gian hiện tại.

```{.python .input}
#@save
class Decoder(nn.Block):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
#@tab pytorch
#@save
class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
#@tab tensorflow
#@save
class Decoder(tf.keras.layers.Layer):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def call(self, X, state, **kwargs):
        raise NotImplementedError
```

## [**Đút bộ mã hóa và bộ giải mã cùng nhau**]

Cuối cùng, kiến trúc bộ mã hóa-giải mã chứa cả bộ mã hóa và bộ giải mã, với các đối số bổ sung tùy chọn. Trong tuyên truyền chuyển tiếp, đầu ra của bộ mã hóa được sử dụng để tạo ra trạng thái được mã hóa và trạng thái này sẽ được bộ giải mã sử dụng thêm như một trong những đầu vào của nó.

```{.python .input}
#@save
class EncoderDecoder(nn.Block):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

```{.python .input}
#@tab pytorch
#@save
class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

```{.python .input}
#@tab tensorflow
#@save
class EncoderDecoder(tf.keras.Model):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, enc_X, dec_X, *args, **kwargs):
        enc_outputs = self.encoder(enc_X, *args, **kwargs)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state, **kwargs)
```

Thuật ngữ “trạng thái” trong kiến trúc bộ mã hóa-giải mã có lẽ đã truyền cảm hứng cho bạn thực hiện kiến trúc này bằng cách sử dụng các mạng thần kinh với các trạng thái. Trong phần tiếp theo, chúng ta sẽ thấy cách áp dụng RNNs để thiết kế các mô hình chuyển tải chuỗi dựa trên kiến trúc bộ mã hóa-giải mã này. 

## Tóm tắt

* Kiến trúc bộ mã hóa-giải mã có thể xử lý các đầu vào và đầu ra vừa là chuỗi độ dài biến đổi, do đó phù hợp với các vấn đề chuyển tải trình tự như dịch máy.
* Bộ mã hóa lấy một chuỗi chiều dài biến đổi làm đầu vào và biến nó thành trạng thái có hình dạng cố định.
* Bộ giải mã ánh xạ trạng thái được mã hóa của một hình dạng cố định đến một chuỗi có độ dài thay đổi.

## Bài tập

1. Giả sử rằng chúng ta sử dụng mạng nơ-ron để thực hiện kiến trúc bộ mã hóa-giải mã. Bộ mã hóa và bộ giải mã có phải là cùng một loại mạng thần kinh không?  
1. Bên cạnh dịch máy, bạn có thể nghĩ đến một ứng dụng khác mà kiến trúc bộ mã hóa-giải mã có thể được áp dụng không?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/341)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1061)
:end_tab:
