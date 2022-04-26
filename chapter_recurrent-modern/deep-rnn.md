# Mạng thần kinh tái phát sâu

:label:`sec_deep_rnn` 

Cho đến nay, chúng ta chỉ thảo luận về RNNs với một layer ẩn một chiều duy nhất. Trong đó hình thức chức năng cụ thể của cách các biến và quan sát tiềm ẩn tương tác là khá tùy ý. Đây không phải là một vấn đề lớn miễn là chúng ta có đủ linh hoạt để mô hình hóa các loại tương tác khác nhau. Tuy nhiên, với một lớp duy nhất, điều này có thể khá khó khăn. Trong trường hợp của các mô hình tuyến tính, chúng tôi đã khắc phục vấn đề này bằng cách thêm nhiều lớp hơn. Trong RNNs, điều này phức tạp hơn một chút, vì trước tiên chúng ta cần quyết định cách thức và ở đâu để thêm tính phi tuyến thêm. 

Trên thực tế, chúng ta có thể xếp chồng nhiều lớp RNNlên nhau. Điều này dẫn đến một cơ chế linh hoạt, do sự kết hợp của một số lớp đơn giản. Đặc biệt, dữ liệu có thể có liên quan ở các cấp độ khác nhau của ngăn xếp. Ví dụ: chúng tôi có thể muốn giữ dữ liệu cấp cao về các điều kiện thị trường tài chính (thị trường gấu hoặc thị trường tăng giá) có sẵn, trong khi ở mức thấp hơn, chúng tôi chỉ ghi lại động thái thời gian ngắn hạn. 

Ngoài tất cả các cuộc thảo luận trừu tượng ở trên, có lẽ dễ dàng nhất để hiểu được gia đình của các mô hình mà chúng tôi quan tâm bằng cách xem xét :numref:`fig_deep_rnn`. Nó mô tả một RNN sâu với $L$ lớp ẩn. Mỗi trạng thái ẩn liên tục được truyền cho cả bước thời gian tiếp theo của lớp hiện tại và bước thời gian hiện tại của lớp tiếp theo. 

![Architecture of a deep RNN.](../img/deep-rnn.svg)
:label:`fig_deep_rnn`

## Phụ thuộc chức năng

Chúng ta có thể chính thức hóa các phụ thuộc chức năng trong kiến trúc sâu của $L$ các lớp ẩn được mô tả trong :numref:`fig_deep_rnn`. Cuộc thảo luận sau đây của chúng tôi tập trung chủ yếu vào mô hình vanilla RNN, nhưng nó cũng áp dụng cho các mô hình trình tự khác. 

Giả sử rằng chúng ta có một đầu vào minibatch $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (số ví dụ: $n$, số lượng đầu vào trong mỗi ví dụ: $d$) tại bước thời gian $t$. Đồng thời bước, hãy để trạng thái ẩn của lớp ẩn $l^\mathrm{th}$ ($l=1,\ldots,L$) là $\mathbf{H}_t^{(l)}  \in \mathbb{R}^{n \times h}$ (số đơn vị ẩn: $h$) và biến lớp đầu ra là $\mathbf{O}_t \in \mathbb{R}^{n \times q}$ (số lượng đầu ra: $q$). Đặt $\mathbf{H}_t^{(0)} = \mathbf{X}_t$, trạng thái ẩn của lớp ẩn $l^\mathrm{th}$ sử dụng hàm kích hoạt $\phi_l$ được thể hiện như sau: 

$$\mathbf{H}_t^{(l)} = \phi_l(\mathbf{H}_t^{(l-1)} \mathbf{W}_{xh}^{(l)} + \mathbf{H}_{t-1}^{(l)} \mathbf{W}_{hh}^{(l)}  + \mathbf{b}_h^{(l)}),$$
:eqlabel:`eq_deep_rnn_H`

trong đó trọng lượng $\mathbf{W}_{xh}^{(l)} \in \mathbb{R}^{h \times h}$ và $\mathbf{W}_{hh}^{(l)} \in \mathbb{R}^{h \times h}$, cùng với sự thiên vị $\mathbf{b}_h^{(l)} \in \mathbb{R}^{1 \times h}$, là các thông số mô hình của lớp ẩn $l^\mathrm{th}$. 

Cuối cùng, việc tính toán lớp đầu ra chỉ dựa trên trạng thái ẩn của lớp ẩn cuối cùng $L^\mathrm{th}$: 

$$\mathbf{O}_t = \mathbf{H}_t^{(L)} \mathbf{W}_{hq} + \mathbf{b}_q,$$

trong đó trọng lượng $\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$ và thiên vị $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$ là các thông số mô hình của lớp đầu ra. 

Cũng giống như với MLP, số lượng lớp ẩn $L$ và số đơn vị ẩn $h$ là các siêu tham số. Nói cách khác, chúng có thể được điều chỉnh hoặc chỉ định bởi chúng tôi. Ngoài ra, chúng ta có thể dễ dàng có được một RNN có cổng sâu bằng cách thay thế tính toán trạng thái ẩn trong :eqref:`eq_deep_rnn_H` với nó từ GRU hoặc LSTM. 

## Thực hiện ngắn gọn

May mắn thay, nhiều chi tiết hậu cần cần thiết để triển khai nhiều lớp của một RNN có sẵn trong các API cấp cao. Để giữ cho mọi thứ đơn giản, chúng tôi chỉ minh họa việc thực hiện bằng cách sử dụng các chức năng tích hợp như vậy. Hãy để chúng tôi lấy một mô hình LSTM làm ví dụ. Mã này rất giống với mã chúng tôi đã sử dụng trước đây trong :numref:`sec_lstm`. Trên thực tế, sự khác biệt duy nhất là chúng ta chỉ định số lớp một cách rõ ràng hơn là chọn mặc định của một lớp duy nhất. Như thường lệ, chúng ta bắt đầu bằng cách tải tập dữ liệu.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import npx
from mxnet.gluon import rnn
npx.set_np()

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

Các quyết định kiến trúc như chọn siêu tham số rất giống với các quyết định của :numref:`sec_lstm`. Chúng tôi chọn cùng một số đầu vào và đầu ra như chúng tôi có mã thông báo riêng biệt, tức là `vocab_size`. Số lượng đơn vị ẩn vẫn là 256. Sự khác biệt duy nhất là chúng ta bây giờ (** chọn một số không tầm thường của các lớp ẩn bằng cách xác định giá trị của `num_layers`.**)

```{.python .input}
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
device = d2l.try_gpu()
lstm_layer = rnn.LSTM(num_hiddens, num_layers)
model = d2l.RNNModel(lstm_layer, len(vocab))
```

```{.python .input}
#@tab pytorch
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
```

```{.python .input}
#@tab tensorflow
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)
rnn_cells = [tf.keras.layers.LSTMCell(num_hiddens) for _ in range(num_layers)]
stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
lstm_layer = tf.keras.layers.RNN(stacked_lstm, time_major=True,
                                 return_sequences=True, return_state=True)
with strategy.scope():
    model = d2l.RNNModel(lstm_layer, len(vocab))
```

## [**Đào tạo**] và Dự đoán

Kể từ bây giờ chúng tôi khởi tạo hai lớp với mô hình LSTM, kiến trúc khá phức tạp hơn này làm chậm quá trình đào tạo đáng kể.

```{.python .input}
#@tab mxnet, pytorch
num_epochs, lr = 500, 2
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr = 500, 2
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, strategy)
```

## Tóm tắt

* Trong RNNsâu, thông tin trạng thái ẩn được chuyển sang bước thời gian tiếp theo của lớp hiện tại và bước thời gian hiện tại của lớp tiếp theo.
* Có tồn tại nhiều hương vị khác nhau của RNNsâu, chẳng hạn như LSTMs, Grus, hoặc RNN vani. Thuận tiện, các mô hình này đều có sẵn như là một phần của API cấp cao của các khuôn khổ deep learning.
* Khởi tạo các mô hình đòi hỏi sự chăm sóc. Nhìn chung, RNN sâu đòi hỏi lượng công việc đáng kể (chẳng hạn như tốc độ học tập và cắt) để đảm bảo sự hội tụ thích hợp.

## Bài tập

1. Cố gắng thực hiện RNN hai lớp từ đầu bằng cách sử dụng triển khai lớp duy nhất mà chúng tôi đã thảo luận trong :numref:`sec_rnn_scratch`.
2. Thay thế LSTM bằng GRU và so sánh độ chính xác và tốc độ đào tạo.
3. Tăng dữ liệu đào tạo để bao gồm nhiều cuốn sách. Làm thế nào thấp bạn có thể đi trên quy mô bối rối?
4. Bạn có muốn kết hợp các nguồn của các tác giả khác nhau khi mô hình hóa văn bản? Tại sao đây là một ý tưởng tốt? Điều gì có thể đi sai?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/340)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1058)
:end_tab:
