# Mạng nơ-ron định kỳ hai chiều
:label:`sec_bi_rnn`

Trong học trình tự, cho đến nay chúng tôi giả định rằng mục tiêu của chúng tôi là mô hình hóa đầu ra tiếp theo cho những gì chúng ta đã thấy cho đến nay, ví dụ, trong bối cảnh của một chuỗi thời gian hoặc trong bối cảnh của một mô hình ngôn ngữ. Mặc dù đây là một kịch bản điển hình, nhưng nó không phải là kịch bản duy nhất chúng ta có thể gặp phải. Để minh họa vấn đề, hãy xem xét ba nhiệm vụ sau đây để điền vào chỗ trống trong một chuỗi văn bản: 

* Tôi là `___`.
* Tôi `___` đói.
* Tôi `___` đói, và tôi có thể ăn nửa con lợn.

Tùy thuộc vào lượng thông tin có sẵn, chúng tôi có thể điền vào các khoảng trống với các từ rất khác nhau như “hạnh phúc”, “không” và “rất”. Rõ ràng phần cuối của cụm từ (nếu có) truyền tải thông tin quan trọng về việc chọn từ nào. Một mô hình trình tự không có khả năng tận dụng điều này sẽ thực hiện kém trên các nhiệm vụ liên quan. Ví dụ, để làm tốt trong việc nhận dạng thực thể được đặt tên (ví dụ: để nhận ra liệu “Green” có đề cập đến “Mr. Green” hay màu) bối cảnh tầm xa cũng quan trọng không kém. Để có được một số nguồn cảm hứng để giải quyết vấn đề, chúng ta hãy đi đường bộ đến các mô hình đồ họa xác suất. 

## Lập trình động trong các mô hình Markov ẩn

Tiểu mục này phục vụ để minh họa bài toán lập trình động. Các chi tiết kỹ thuật cụ thể không quan trọng để hiểu các mô hình học sâu nhưng chúng giúp thúc đẩy lý do tại sao người ta có thể sử dụng deep learning và tại sao người ta có thể chọn kiến trúc cụ thể. 

Nếu chúng ta muốn giải quyết vấn đề bằng cách sử dụng mô hình đồ họa xác suất, chúng ta có thể ví dụ thiết kế một mô hình biến tiềm ẩn như sau. Bất cứ lúc nào bước $t$, chúng tôi giả định rằng có tồn tại một số biến tiềm ẩn $h_t$ chi phối phát thải quan sát của chúng tôi $x_t$ thông qua $P(x_t \mid h_t)$. Hơn nữa, bất kỳ quá trình chuyển đổi $h_t \to h_{t+1}$ được đưa ra bởi một số xác suất chuyển đổi trạng thái $P(h_{t+1} \mid h_{t})$. Mô hình đồ họa xác suất này sau đó là mô hình Markov * ẩn* như trong :numref:`fig_hmm`. 

![A hidden Markov model.](../img/hmm.svg)
:label:`fig_hmm`

Do đó, đối với một chuỗi $T$ quan sát chúng ta có sự phân bố xác suất chung sau đây trên các trạng thái quan sát và ẩn: 

$$P(x_1, \ldots, x_T, h_1, \ldots, h_T) = \prod_{t=1}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t), \text{ where } P(h_1 \mid h_0) = P(h_1).$$
:eqlabel:`eq_hmm_jointP`

Bây giờ giả định rằng chúng ta quan sát tất cả $x_i$ ngoại trừ một số $x_j$ và đó là mục tiêu của chúng tôi để tính toán $P(x_j \mid x_{-j})$, trong đó $x_{-j} = (x_1, \ldots, x_{j-1}, x_{j+1}, \ldots, x_{T})$. Vì không có biến tiềm ẩn trong $P(x_j \mid x_{-j})$, chúng tôi xem xét tổng hợp tất cả các kết hợp có thể có của các lựa chọn cho $h_1, \ldots, h_T$. Trong trường hợp bất kỳ $h_i$ nào cũng có thể đảm nhận $k$ các giá trị riêng biệt (một số trạng thái hữu hạn), điều này có nghĩa là chúng ta cần tổng hợp hơn $k^T$ điều khoản — thường là nhiệm vụ không thể! May mắn thay, có một giải pháp thanh lịch cho việc này: * lập trình động *. 

Để xem nó hoạt động như thế nào, hãy xem xét tổng hợp các biến tiềm ẩn $h_1, \ldots, h_T$ lần lượt. Theo :eqref:`eq_hmm_jointP`, sản lượng này: 

$$\begin{aligned}
    &P(x_1, \ldots, x_T) \\
    =& \sum_{h_1, \ldots, h_T} P(x_1, \ldots, x_T, h_1, \ldots, h_T) \\
    =& \sum_{h_1, \ldots, h_T} \prod_{t=1}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t) \\
    =& \sum_{h_2, \ldots, h_T} \underbrace{\left[\sum_{h_1} P(h_1) P(x_1 \mid h_1) P(h_2 \mid h_1)\right]}_{\pi_2(h_2) \stackrel{\mathrm{def}}{=}}
    P(x_2 \mid h_2) \prod_{t=3}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t) \\
    =& \sum_{h_3, \ldots, h_T} \underbrace{\left[\sum_{h_2} \pi_2(h_2) P(x_2 \mid h_2) P(h_3 \mid h_2)\right]}_{\pi_3(h_3)\stackrel{\mathrm{def}}{=}}
    P(x_3 \mid h_3) \prod_{t=4}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t)\\
    =& \dots \\
    =& \sum_{h_T} \pi_T(h_T) P(x_T \mid h_T).
\end{aligned}$$

Nói chung chúng ta có đệ quy* forward* như 

$$\pi_{t+1}(h_{t+1}) = \sum_{h_t} \pi_t(h_t) P(x_t \mid h_t) P(h_{t+1} \mid h_t).$$

Đệ quy được khởi tạo là $\pi_1(h_1) = P(h_1)$. Trong thuật ngữ trừu tượng, điều này có thể được viết là $\pi_{t+1} = f(\pi_t, x_t)$, trong đó $f$ là một số hàm có thể học được. Điều này trông rất giống phương trình cập nhật trong các mô hình biến tiềm ẩn mà chúng ta đã thảo luận cho đến nay trong bối cảnh RNN!  

Hoàn toàn tương tự như đệ quy về phía trước, chúng ta cũng có thể tổng hợp cùng một tập hợp các biến tiềm ẩn với đệ quy ngược. Điều này mang lại: 

$$\begin{aligned}
    & P(x_1, \ldots, x_T) \\
     =& \sum_{h_1, \ldots, h_T} P(x_1, \ldots, x_T, h_1, \ldots, h_T) \\
    =& \sum_{h_1, \ldots, h_T} \prod_{t=1}^{T-1} P(h_t \mid h_{t-1}) P(x_t \mid h_t) \cdot P(h_T \mid h_{T-1}) P(x_T \mid h_T) \\
    =& \sum_{h_1, \ldots, h_{T-1}} \prod_{t=1}^{T-1} P(h_t \mid h_{t-1}) P(x_t \mid h_t) \cdot
    \underbrace{\left[\sum_{h_T} P(h_T \mid h_{T-1}) P(x_T \mid h_T)\right]}_{\rho_{T-1}(h_{T-1})\stackrel{\mathrm{def}}{=}} \\
    =& \sum_{h_1, \ldots, h_{T-2}} \prod_{t=1}^{T-2} P(h_t \mid h_{t-1}) P(x_t \mid h_t) \cdot
    \underbrace{\left[\sum_{h_{T-1}} P(h_{T-1} \mid h_{T-2}) P(x_{T-1} \mid h_{T-1}) \rho_{T-1}(h_{T-1}) \right]}_{\rho_{T-2}(h_{T-2})\stackrel{\mathrm{def}}{=}} \\
    =& \ldots \\
    =& \sum_{h_1} P(h_1) P(x_1 \mid h_1)\rho_{1}(h_{1}).
\end{aligned}$$

Do đó, chúng ta có thể viết các * backward recursion* như 

$$\rho_{t-1}(h_{t-1})= \sum_{h_{t}} P(h_{t} \mid h_{t-1}) P(x_{t} \mid h_{t}) \rho_{t}(h_{t}),$$

với khởi tạo $\rho_T(h_T) = 1$. Cả đệ quy tiến và lùi đều cho phép chúng ta tổng hợp hơn $T$ các biến tiềm ẩn trong thời gian $\mathcal{O}(kT)$ (tuyến tính) trên tất cả các giá trị của $(h_1, \ldots, h_T)$ chứ không phải trong thời gian hàm mũ. Đây là một trong những lợi ích lớn của suy luận xác suất với các mô hình đồ họa. Nó cũng là một trường hợp rất đặc biệt của một thông điệp chung vượt qua thuật toán :cite:`Aji.McEliece.2000`. Kết hợp cả đệ quy tiến và lùi, chúng ta có thể tính toán 

$$P(x_j \mid x_{-j}) \propto \sum_{h_j} \pi_j(h_j) \rho_j(h_j) P(x_j \mid h_j).$$

Lưu ý rằng trong thuật ngữ trừu tượng đệ quy ngược có thể được viết là $\rho_{t-1} = g(\rho_t, x_t)$, trong đó $g$ là một hàm có thể học được. Một lần nữa, điều này trông rất giống như một phương trình cập nhật, chỉ chạy ngược không giống như những gì chúng ta đã thấy cho đến nay trong RNNs. Thật vậy, các mô hình Markov ẩn được hưởng lợi từ việc biết dữ liệu trong tương lai khi có sẵn. Các nhà khoa học xử lý tín hiệu phân biệt giữa hai trường hợp biết và không biết các quan sát trong tương lai như nội suy v.s. ngoại suy. Xem chương giới thiệu của cuốn sách về thuật toán Monte Carlo tuần tự để biết thêm chi tiết :cite:`Doucet.De-Freitas.Gordon.2001`. 

## Mô hình hai chiều

Nếu chúng ta muốn có một cơ chế trong RNN cung cấp khả năng nhìn trước tương đương như trong các mô hình Markov ẩn, chúng ta cần sửa đổi thiết kế RNN mà chúng ta đã thấy cho đến nay. May mắn thay, điều này là dễ dàng về mặt khái niệm. Thay vì chạy RNN chỉ ở chế độ chuyển tiếp bắt đầu từ token đầu tiên, chúng ta bắt đầu một mã thông báo khác từ token cuối cùng chạy từ sau ra trước. 
*Hai chiều RNNs* thêm một lớp ẩn truyền thông tin theo hướng ngược để xử lý thông tin đó linh hoạt hơn. :numref:`fig_birnn` minh họa kiến trúc của RNN hai chiều với một lớp ẩn duy nhất.

![Architecture of a bidirectional RNN.](../img/birnn.svg)
:label:`fig_birnn`

Trên thực tế, điều này không quá khác nhau với các đệ quy về phía trước và lạc hậu trong việc lập trình động của các mô hình Markov ẩn. Sự khác biệt chính là trong trường hợp trước các phương trình này có ý nghĩa thống kê cụ thể. Bây giờ chúng không có những giải thích dễ tiếp cận như vậy và chúng ta chỉ có thể coi chúng như là các chức năng chung và có thể học được. Quá trình chuyển đổi này mô tả nhiều nguyên tắc hướng dẫn thiết kế các mạng sâu hiện đại: đầu tiên, sử dụng loại phụ thuộc chức năng của các mô hình thống kê cổ điển, và sau đó tham số hóa chúng trong một hình thức chung chung. 

### Definition

Các RNN hai chiều được giới thiệu bởi :cite:`Schuster.Paliwal.1997`. Đối với một cuộc thảo luận chi tiết về các kiến trúc khác nhau xem thêm bài báo :cite:`Graves.Schmidhuber.2005`. Chúng ta hãy nhìn vào các chi tiết cụ thể của một mạng như vậy. 

Đối với bất kỳ bước thời gian $t$, đưa ra một đầu vào minibatch $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (số ví dụ: $n$, số đầu vào trong mỗi ví dụ: $d$) và để cho chức năng kích hoạt lớp ẩn là $\phi$. Trong kiến trúc hai chiều, chúng ta giả định rằng các trạng thái ẩn về phía trước và lạc hậu cho bước thời gian này là $\overrightarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$ và $\overleftarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$, tương ứng, trong đó $h$ là số đơn vị ẩn. Các bản cập nhật trạng thái ẩn về phía trước và lạc hậu như sau: 

$$
\begin{aligned}
\overrightarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(f)} + \overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{hh}^{(f)}  + \mathbf{b}_h^{(f)}),\\
\overleftarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(b)} + \overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{hh}^{(b)}  + \mathbf{b}_h^{(b)}),
\end{aligned}
$$

trong đó trọng lượng $\mathbf{W}_{xh}^{(f)} \in \mathbb{R}^{d \times h}, \mathbf{W}_{hh}^{(f)} \in \mathbb{R}^{h \times h}, \mathbf{W}_{xh}^{(b)} \in \mathbb{R}^{d \times h}, \text{ and } \mathbf{W}_{hh}^{(b)} \in \mathbb{R}^{h \times h}$, và thiên vị $\mathbf{b}_h^{(f)} \in \mathbb{R}^{1 \times h} \text{ and } \mathbf{b}_h^{(b)} \in \mathbb{R}^{1 \times h}$ là tất cả các thông số mô hình. 

Tiếp theo, chúng tôi nối các trạng thái ẩn về phía trước và lạc hậu $\overrightarrow{\mathbf{H}}_t$ và $\overleftarrow{\mathbf{H}}_t$ để có được trạng thái ẩn $\mathbf{H}_t \in \mathbb{R}^{n \times 2h}$ được đưa vào lớp đầu ra. Trong các RNN hai chiều sâu với nhiều lớp ẩn, thông tin như vậy được truyền dưới dạng * đầu vào* sang lớp hai chiều tiếp theo. Cuối cùng, lớp đầu ra tính toán đầu ra $\mathbf{O}_t \in \mathbb{R}^{n \times q}$ (số lượng đầu ra: $q$): 

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.$$

Ở đây, ma trận trọng lượng $\mathbf{W}_{hq} \in \mathbb{R}^{2h \times q}$ và thiên vị $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$ là các tham số mô hình của lớp đầu ra. Trên thực tế, hai hướng có thể có số lượng khác nhau của các đơn vị ẩn. 

### Chi phí tính toán và các ứng dụng

Một trong những tính năng chính của RNN hai chiều là thông tin từ cả hai đầu của chuỗi được sử dụng để ước tính đầu ra. Đó là, chúng tôi sử dụng thông tin từ cả quan sát trong tương lai và trong quá khứ để dự đoán thông tin hiện tại. Trong trường hợp dự đoán token tiếp theo, đây không hoàn toàn là những gì chúng ta muốn. Rốt cuộc, chúng ta không có sự xa xỉ khi biết mã thông báo tiếp theo khi dự đoán mã tiếp theo. Do đó, nếu chúng ta sử dụng RNN hai chiều một cách ngây thơ, chúng ta sẽ không có được độ chính xác rất tốt: trong quá trình đào tạo, chúng ta có dữ liệu trong quá khứ và tương lai để ước tính hiện tại. Trong thời gian thử nghiệm, chúng tôi chỉ có dữ liệu trong quá khứ và do đó độ chính xác kém. Chúng tôi sẽ minh họa điều này trong một thí nghiệm dưới đây. 

Để thêm xúc phạm đến chấn thương, RNN hai chiều cũng cực kỳ chậm. Những lý do chính cho điều này là sự lan truyền về phía trước đòi hỏi cả đệ quy về phía trước và ngược trong các lớp hai chiều và sự lan truyền ngược phụ thuộc vào kết quả của sự lan truyền về phía trước. Do đó, gradient sẽ có một chuỗi phụ thuộc rất dài. 

Trong thực tế các lớp hai chiều được sử dụng rất ít và chỉ cho một tập hợp các ứng dụng hẹp, chẳng hạn như điền vào các từ bị thiếu, mã thông báo chú thích (ví dụ, để nhận dạng thực thể được đặt tên) và mã hóa chuỗi bán buôn như một bước trong một đường ống xử lý trình tự (ví dụ, cho dịch máy). Trong :numref:`sec_bert` và :numref:`sec_sentiment_rnn`, chúng tôi sẽ giới thiệu cách sử dụng RNN hai chiều để mã hóa chuỗi văn bản. 

## (**Đào tạo một RNN hai chiều cho một ứng dụng sai **)

Nếu chúng ta bỏ qua tất cả lời khuyên liên quan đến thực tế là RNN hai chiều sử dụng dữ liệu trong quá khứ và tương lai và chỉ cần áp dụng nó cho các mô hình ngôn ngữ, chúng tôi sẽ nhận được ước tính với sự bối rối chấp nhận được. Tuy nhiên, khả năng của mô hình dự đoán token trong tương lai bị tổn hại nghiêm trọng như thí nghiệm dưới đây minh họa. Mặc dù bối rối hợp lý, nó chỉ tạo ra sự lúng túng ngay cả sau nhiều lần lặp lại. Chúng tôi bao gồm mã dưới đây như một ví dụ cảnh báo chống lại việc sử dụng chúng trong bối cảnh sai.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import npx
from mxnet.gluon import rnn
npx.set_np()

# Load data
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# Define the bidirectional LSTM model by setting `bidirectional=True`
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
lstm_layer = rnn.LSTM(num_hiddens, num_layers, bidirectional=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
# Train the model
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

# Load data
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# Define the bidirectional LSTM model by setting `bidirectional=True`
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
# Train the model
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

Đầu ra rõ ràng là không đạt yêu cầu vì những lý do được mô tả ở trên. Để thảo luận về việc sử dụng RNN hai chiều hiệu quả hơn, vui lòng xem ứng dụng phân tích tâm lý trong :numref:`sec_sentiment_rnn`. 

## Tóm tắt

* Trong RNN hai chiều, trạng thái ẩn cho mỗi bước thời gian được xác định đồng thời bởi dữ liệu trước và sau bước thời gian hiện tại.
* RNN hai chiều mang một sự tương đồng nổi bật với thuật toán về phía trước về phía sau trong các mô hình đồ họa xác suất.
* Các RNN hai chiều chủ yếu hữu ích cho mã hóa trình tự và ước lượng các quan sát được đưa ra bối cảnh hai chiều.
* RNNhai chiều rất tốn kém để đào tạo do chuỗi gradient dài.

## Bài tập

1. Nếu các hướng khác nhau sử dụng một số đơn vị ẩn khác nhau, hình dạng của $\mathbf{H}_t$ sẽ thay đổi như thế nào?
1. Thiết kế một RNN hai chiều với nhiều lớp ẩn.
1. Polysemy phổ biến trong các ngôn ngữ tự nhiên. Ví dụ, từ “ngân hàng” có ý nghĩa khác nhau trong bối cảnh “tôi đã đến ngân hàng để gửi tiền mặt” và “tôi đã đến ngân hàng để ngồi xuống”. Làm thế nào chúng ta có thể thiết kế một mô hình mạng thần kinh như vậy mà đưa ra một chuỗi ngữ cảnh và một từ, một biểu diễn vector của từ trong ngữ cảnh sẽ được trả về? Loại kiến trúc thần kinh nào được ưa thích để xử lý polysemy?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/339)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1059)
:end_tab:
