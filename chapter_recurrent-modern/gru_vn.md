<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Gated Recurrent Units (GRU)
-->

# Nút Truy hồi có Cổng (GRU)
:label:`sec_gru`

<!--
In the previous section, we discussed how gradients are calculated in a recurrent neural network.
In particular we found that long products of matrices can lead to vanishing or divergent gradients.
Let us briefly think about what such gradient anomalies mean in practice:
-->

Trong phần trước, chúng ta đã thảo luận về cách gradient được tính toán trong mạng nơ-ron truy hồi.
Cụ thể ta đã phát hiện rằng tích của một chuỗi dài các ma trận có thể dẫn đến việc gradient tiêu biến hoặc bùng nổ.
Bây giờ hãy suy nghĩ nhanh về ý nghĩa của những gradient bất thường như vậy trong thực tế:

<!--
* We might encounter a situation where an early observation is highly significant for predicting all future observations.
Consider the somewhat contrived case where the first observation contains a checksum and the goal is to discern whether the checksum is correct at the end of the sequence.
In this case, the influence of the first token is vital.
We would like to have some mechanisms for storing vital early information in a *memory cell*.
Without such a mechanism, we will have to assign a very large gradient to this observation, since it affects all subsequent observations.
* We might encounter situations where some symbols carry no pertinent observation.
For instance, when parsing a web page there might be auxiliary HTML code that is irrelevant for the purpose of assessing the sentiment conveyed on the page.
We would like to have some mechanism for *skipping such symbols* in the latent state representation.
* We might encounter situations where there is a logical break between parts of a sequence.
For instance, there might be a transition between chapters in a book, or a transition between a bear and a bull market for securities.
In this case it would be nice to have a means of *resetting* our internal state representation.
-->

* Chúng ta có thể gặp phải tình huống mà những quan sát xuất hiện sớm có ảnh hưởng đáng kể đến việc dự doán toàn bộ những quan sát ở tương lai.
Xét một ví dụ hơi không thực tế, trong đó ta có quan sát đầu tiên chứa giá trị tổng kiểm (_checksum_) và mục tiêu ở đây là phân biệt liệu giá trị tổng kiểm đó có đúng hay không tại cuối chuỗi.
Trong trường hợp này, ảnh hưởng của token đầu tiên là tối quan trọng.
Chúng ta muốn có một vài cơ chế cho việc lưu trữ những thông tin ban đầu quan trọng trong *ô nhớ*.
Nếu không có cơ chế như vậy, ta phải gán một giá trị gradient cực lớn cho quan sát này, vì nó ảnh hưởng đến toàn bộ các quan sát tiếp theo.
* Chúng ta có thể gặp phải tình huống mà một vài ký hiệu không chứa thông tin phù hợp.
Ví dụ, khi phân tích một trang web, có thể sẽ có mã HTML phụ trợ không liên quan đến mục tiêu xác định thông tin được truyền tải trên trang web.
Chúng ta sẽ muốn có một số cơ chế để *bỏ qua những ký hiệu như vậy* trong việc biểu diễn trạng thái tiềm ẩn.
* Chúng ta có thể gặp phải tình huống trong đó tồn tại những khoảng ngắt giữa các phần của một chuỗi.
Ví dụ, có thể sẽ có những đoạn chuyển tiếp giữa các chương của một quyển sách, hay chuyển biến giữa thị trường giá lên và thị trường giá xuống trong chứng khoán.
Trong trường hợp này, sẽ tốt hơn nếu có một cách để *xoá* hay *đặt lại* các biểu diễn trạng thái ẩn về trạng thái ban đầu.

<!--
A number of methods have been proposed to address this.
One of the earliest is Long Short Term Memory (LSTM) :cite:`Hochreiter.Schmidhuber.1997` which we will discuss in :numref:`sec_lstm`.
Gated Recurrent Unit (GRU) :cite:`Cho.Van-Merrienboer.Bahdanau.ea.2014` is a slightly more streamlined variant that often offers comparable performance and is significantly faster to compute.
See also :cite:`Chung.Gulcehre.Cho.ea.2014` for more details.
Due to its simplicity, let us start with the GRU.
-->

Nhiều phương pháp đã được đề xuất để giải quyết những vấn đề này.
Một trong những phương pháp sớm nhất chính là Bộ nhớ ngắn hạn dài (Long Short Term Memory - LSTM) :cite:`Hochreiter.Schmidhuber.1997`, sẽ được thảo luận ở :numref:`sec_lstm`.
Nút Truy hồi có Cổng (*Gated Recurrent Unit - GRU*) :cite:`Cho.Van-Merrienboer.Bahdanau.ea.2014` là một biến thể gọn hơn của LSTM, thường mang lại chất lượng tương đương và nhanh hơn đáng kể về mặt tính toán.
Hãy đọc :cite:`Chung.Gulcehre.Cho.ea.2014` để biết thêm chi tiết.
Do GRU đơn giản hơn nên chúng ta sẽ bắt đầu với nó trước.

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Gating the Hidden State
-->

## Kiểm soát Trạng thái ẩn

<!--
The key distinction between regular RNNs and GRUs is that the latter support gating of the hidden state.
This means that we have dedicated mechanisms for when a hidden state should be updated and also when it should be reset.
These mechanisms are learned and they address the concerns listed above.
For instance, if the first symbol is of great importance we will learn not to update the hidden state after the first observation.
Likewise, we will learn to skip irrelevant temporary observations.
Last, we will learn to reset the latent state whenever needed.
We discuss this in detail below.
-->

Sự khác biệt chính giữa RNN thông thường và GRU là GRU hỗ trợ việc kiểm soát trạng thái ẩn.
Điều này có nghĩa là ta có các cơ chế chuyên dụng để quyết định khi nào nên cập nhật và khi nào nên xóa trạng thái ẩn.
Các cơ chế này được học để giải quyết các vấn đề được đề cập ở bên trên.
Ví dụ, nếu ký tự đầu tiên có mức độ quan trọng cao, ta sẽ học để không cập nhật trạng thái ẩn sau lần quan sát đầu tiên.
Tương tự như vậy, ta sẽ học để bỏ qua những quan sát tạm thời không liên quan.
Cuối cùng, ta sẽ học để xóa trạng thái tiềm ẩn bất cứ khi nào cần thiết.
Ta sẽ thảo luận vấn đề này một cách chi tiết dưới đây.

<!--
### Reset Gates and Update Gates
-->

### Cổng Xóa và Cổng Cập Nhật

<!--
The first thing we need to introduce are reset and update gates.
We engineer them to be vectors with entries in $(0, 1)$ such that we can perform convex combinations.
For instance, a reset variable would allow us to control how much of the previous state we might still want to remember.
Likewise, an update variable would allow us to control how much of the new state is just a copy of the old state.
-->

Điều đầu tiên ta cần giới thiệu là cổng xóa và cổng cập nhật.
Ta thiết kế chúng thành các vector với mỗi phần tử có giá trị trong khoảng $(0, 1)$ để ta có thể thực hiện các tổ hợp lồi.
Chẳng hạn, một biến xóa sẽ cho phép ta kiểm soát bao nhiêu phần của trạng thái trước đây mà ta muốn ghi nhớ.
Tương tự, một biến cập nhật sẽ cho phép ta kiểm soát bao nhiêu phần của trạng thái mới sẽ chỉ là một bản sao của trạng thái cũ.

<!--
We begin by engineering gates to generate these variables.
:numref:`fig_gru_1` illustrates the inputs for both reset and update gates in a GRU, given the current timestep input $\mathbf{X}_t$ and the hidden state of the previous timestep $\mathbf{H}_{t-1}$.
The output is given by a fully connected layer with a sigmoid as its activation function.
-->

Ta bắt đầu bằng việc thiết kế các cổng để tạo ra các biến này.
:numref:`fig_gru_1` minh họa các đầu vào cho cả cổng xóa và cổng cập nhật trong GRU, với đầu vào là bước thời gian hiện tại $\mathbf{X}_t$ và trạng thái ẩn của bước thời gian trước đó $\mathbf{H}_{t-1}$.
Đầu ra được tạo ra bởi một tầng kết nối đầy đủ với hàm kích hoạt sigmoid.

<!--
![ Reset and update gate in a GRU. ](../img/gru_1.svg)
-->

![Cổng xóa và cổng cập nhật trong GRU.](../img/gru_1.svg)
:label:`fig_gru_1`


<!--
For a given timestep $t$, the minibatch input is $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (number of examples: $n$, number of inputs: $d$) 
and the hidden state of the last timestep is $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$ (number of hidden states: $h$).
Then, the reset gate $\mathbf{R}_t \in \mathbb{R}^{n \times h}$ and update gate $\mathbf{Z}_t \in \mathbb{R}^{n \times h}$ are computed as follows:
-->

Với một bước thời gian nhất định $t$, đầu vào minibatch là $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (số lượng mẫu: $n$, số lượng đầu vào: $d$) và trạng thái ẩn của bước thời gian gần nhất là $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$ (số lượng trạng thái ẩn: $h$), thì cổng xoá $\mathbf{R}_t \in \mathbb{R}^{n \times h}$ và cổng cập nhật $\mathbf{Z}_t \in \mathbb{R}^{n \times h}$ được tính như sau:


$$
\begin{aligned}
\mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xr} + \mathbf{H}_{t-1} \mathbf{W}_{hr} + \mathbf{b}_r),\\
\mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xz} + \mathbf{H}_{t-1} \mathbf{W}_{hz} + \mathbf{b}_z).
\end{aligned}
$$


<!--
Here, $\mathbf{W}_{xr}, \mathbf{W}_{xz} \in \mathbb{R}^{d \times h}$ and
$\mathbf{W}_{hr}, \mathbf{W}_{hz} \in \mathbb{R}^{h \times h}$ are weight parameters 
and $\mathbf{b}_r, \mathbf{b}_z \in \mathbb{R}^{1 \times h}$ are biases.
We use a sigmoid function (as introduced in :numref:`sec_mlp`) to transform input values to the interval $(0, 1)$.
-->

Ở đây, $\mathbf{W}_{xr}, \mathbf{W}_{xz} \in \mathbb{R}^{d \times h}$ và $\mathbf{W}_{hr}, \mathbf{W}_{hz} \in \mathbb{R}^{h \times h}$ là các tham số trọng số và $\mathbf{b}_r, \mathbf{b}_z \in \mathbb{R}^{1 \times h}$ là các hệ số điều chỉnh.
Ta sẽ sử dụng hàm sigmoid (như được giới thiệu trong :numref:`sec_mlp`) để biến đổi các giá trị đầu vào thành các giá trị trong khoảng $(0, 1)$.

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
### Reset Gates in Action
-->

### Cổng Xóa

<!--
We begin by integrating the reset gate with a regular latent state updating mechanism.
In a conventional RNN, we would have an hidden state update of the form
-->

Ta bắt đầu bằng cách tích hợp cổng xóa với một cơ chế cập nhật trạng thái tiềm ẩn thông thường.
Trong một RNN thông thường, ta sẽ cập nhật trạng thái ẩn bằng công thức


$$\mathbf{H}_t = \tanh(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1}\mathbf{W}_{hh} + \mathbf{b}_h).$$


<!--
This is essentially identical to the discussion of the previous section, albeit with a nonlinearity in the form of $\tanh$ to ensure that the values of the hidden states remain in the interval $(-1, 1)$.
If we want to be able to reduce the influence of the previous states we can multiply $\mathbf{H}_{t-1}$ with $\mathbf{R}_t$ elementwise.
Whenever the entries in the reset gate $\mathbf{R}_t$ are close to $1$, we recover a conventional RNN.
For all entries of the reset gate $\mathbf{R}_t$ that are close to $0$, the hidden state is the result of an MLP with $\mathbf{X}_t$ as input.
Any pre-existing hidden state is thus reset to defaults.
This leads to the following *candidate hidden state* (it is a *candidate* since we still need to incorporate the action of the update gate).
-->

Điều này về cơ bản giống với những gì ta đã thảo luận ở phần trước, mặc dù có thêm tính phi tuyến dưới dạng hàm $\tanh$ để đảm bảo rằng các giá trị trạng thái ẩn nằm trong khoảng $(-1, 1)$.
Nếu muốn giảm ảnh hưởng của các trạng thái trước đó, ta có thể nhân $\mathbf{H}_{t-1}$ với $\mathbf{R}_t$ theo từng phần tử.
Bất cứ khi nào các phần tử trong cổng xóa $\mathbf{R}_t$ có giá trị gần với $1$, kết quả sẽ giống RNN thông thường.
Đối với tất cả các phần tử của cổng xóa $\mathbf{R}_t$ có giá trị gần với $0$, trạng thái ẩn sẽ là kết quả của MLP với $\mathbf{X}_t$ làm đầu vào.
Bất kỳ trạng thái ẩn nào tồn tại trước đó đều được đặt lại về giá trị mặc định.
Điều này dẫn đến *trạng thái ẩn tiềm năng* sau đây (nó là *tiềm năng* vì ta vẫn cần kết hợp đầu ra của cổng cập nhật).


$$\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{xh} + \left(\mathbf{R}_t \odot \mathbf{H}_{t-1}\right) \mathbf{W}_{hh} + \mathbf{b}_h).$$


<!--
:numref:`fig_gru_2` illustrates the computational flow after applying the reset gate.
The symbol $\odot$ indicates pointwise multiplication between tensors.
-->

:numref:`fig_gru_2` minh họa luồng tính toán sau khi áp dụng cổng xóa.
Ký hiệu $\odot$ biểu thị phép nhân theo từng phần tử giữa các tensor.

<!--
![ Candidate hidden state computation in a GRU. The multiplication is carried out elementwise. ](../img/gru_2.svg)
-->

![Tính toán của trạng thái ẩn tiềm năng trong một GRU. Phép nhân được tính theo từng phần tử.](../img/gru_2.svg)
:label:`fig_gru_2`

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
### Update Gates in Action
-->

### Cổng Cập nhật

<!--
Next we need to incorporate the effect of the update gate $\mathbf{Z}_t$, as shown in :numref:`fig_gru_3`.
This determines the extent to which the new state $\mathbf{H}_t$ is just the old state $\mathbf{H}_{t-1}$ and by how much the new candidate state $\tilde{\mathbf{H}}_t$ is used.
The gating variable $\mathbf{Z}_t$ can be used for this purpose, simply by taking elementwise convex combinations between both candidates.
This leads to the final update equation for the GRU.
-->

Tiếp đến chúng ta cần kết hợp hiệu ứng của cổng cập nhật $\mathbf{Z}_t$, như trong :numref:`fig_gru_3`.
Điều này xác định mức độ trạng thái mới $\mathbf{H}_t$ giống trạng thái cũ $\mathbf{H}_{t-1}$ và mức độ trạng thái ẩn tiềm năng $\tilde{\mathbf{H}}_t$ được sử dụng.
Biến cổng (_gating variable_) $\mathbf{Z}_t$ được sử dụng cho mục đích này, chỉ đơn giản bằng cách áp dụng tổ hợp lồi theo từng phần tử giữa trạng thái cũ và trạng thái tiềm năng.
Điều này dẫn đến phương trình cập nhật cuối cùng cho GRU.


$$\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1}  + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t.$$


<!--
![ Hidden state computation in a GRU. As before, the multiplication is carried out elementwise. ](../img/gru_3.svg)
-->

![Tính toán trạng thái ẩn trong GRU. Như trước đây, phép nhân được thực hiện theo từng phần tử.](../img/gru_3.svg)
:label:`fig_gru_3`

<!--
Whenever the update gate $\mathbf{Z}_t$ is close to $1$, we simply retain the old state.
In this case the information from $\mathbf{X}_t$ is essentially ignored, effectively skipping timestep $t$ in the dependency chain.
In contrast, whenever $\mathbf{Z}_t$ is close to $0$, the new latent state $\mathbf{H}_t$ approaches the candidate latent state $\tilde{\mathbf{H}}_t$.
These designs can help us cope with the vanishing gradient problem in RNNs and better capture dependencies for time series with large timestep distances.
In summary, GRUs have the following two distinguishing features:
-->

Bất cứ khi nào cổng cập nhật $\mathbf{Z}_t$ gần tới giá trị $1$, chúng ta chỉ đơn giản là giữ lại trạng thái cũ.
Trong trường hợp này, thông tin từ $\mathbf{X}_t$ về cơ bản được bỏ qua, dẫn đến việc bỏ qua bước thời gian $t$ trong chuỗi phụ thuộc một cách hiệu quả.
Ngược lại, bất cứ khi nào $\mathbf{Z}_t$ gần tới giá trị $0$, trạng thái tiềm ẩn $\mathbf{H}_t$ tiến gần tới trạng thái ẩn tiềm năng $\tilde{\mathbf{H}}_t$.
Những thiết kế trên có thể giúp chúng ta giải quyết vấn đề tiêu biến gradient trong các mạng RNN và bắt được những phụ thuộc có khoảng cách bước thời gian lớn trong chuỗi thời gian tốt hơn.
Nói tóm lại, các mạng GRU có hai tính chất nổi bật như sau:

<!--
* Reset gates help capture short-term dependencies in time series.
* Update gates help capture long-term dependencies in time series.
-->

* Cổng xoá giúp nắm bắt các phụ thuộc ngắn hạn trong chuỗi thời gian.
* Cổng cập nhật giúp nắm bắt các phụ thuộc dài hạn trong chuỗi thời gian.

<!--
## Implementation from Scratch
-->

## Lập trình từ đầu

<!--
To gain a better understanding of the model, let us implement a GRU from scratch.
-->

Để có cái nhìn rõ nét hơn về mô hình, chúng ta hãy lập trình một mô hình GRU từ đầu.

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
### Reading the Dataset
-->

### Đọc Dữ liệu

<!--
We begin by reading *The Time Machine* corpus that we used in :numref:`sec_rnn_scratch`.
The code for reading the dataset is given below:
-->

Chúng ta bắt đầu bằng cách đọc kho ngữ liệu *Cỗ máy Thời gian* mà chúng ta đã sử dụng trong :numref:`sec_rnn_scratch`.
Mã nguồn để đọc dữ liệu được cho dưới đây:

```{.python .input  n=1}
import d2l
from mxnet import np, npx
from mxnet.gluon import rnn
npx.set_np()

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

<!--
### Initializing Model Parameters
-->

### Khởi tạo Tham số Mô hình

<!--
The next step is to initialize the model parameters.
We draw the weights from a Gaussian with variance to be $0.01$ and set the bias to $0$.
The hyperparameter `num_hiddens` defines the number of hidden units.
We instantiate all weights and biases relating to the update gate, the reset gate, and the candidate hidden state itself.
Subsequently, we attach gradients to all the parameters.
-->

Bước tiếp theo là để khởi tạo các tham số mô hình.
Chúng ta khởi tạo giá trị của các trọng số từ phân phối Gauss với phương sai là $0.01$ và thiết lập các hệ số điều chỉnh bằng $0$.
Siêu tham số `num_hiddens` xác định số lượng các đơn vị ẩn.
Chúng ta khởi tạo tất cả các trọng số và các hệ số điều chỉnh ​​liên quan đến cổng cập nhật, cổng xoá, và các trạng thái ẩn tiềm năng.
Sau đó, chúng ta gắn gradient cho tất cả các tham số.


```{.python .input  n=2}
def get_params(vocab_size, num_hiddens, ctx):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape, ctx=ctx)

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                np.zeros(num_hiddens, ctx=ctx))

    W_xz, W_hz, b_z = three()  # Update gate parameter
    W_xr, W_hr, b_r = three()  # Reset gate parameter
    W_xh, W_hh, b_h = three()  # Candidate hidden state parameter
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = np.zeros(num_outputs, ctx=ctx)
    # Attach gradients
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params
```

<!--
### Defining the Model
-->

### Định nghĩa Mô hình

<!--
Now we will define the hidden state initialization function `init_gru_state`.
Just like the `init_rnn_state` function defined in :numref:`sec_rnn_scratch`, this function returns an `ndarray` with a shape (batch size, number of hidden units) whose values are all zeros.
-->

Bây giờ chúng ta sẽ định nghĩa hàm khởi tạo trạng thái ẩn `init_gru_state`.
Cũng giống như hàm `init_rnn_state` được định nghĩa tại :numref:`sec_rnn_scratch`, hàm này trả về một mảng `ndarray` với kích thước là (kích thước batch, số lượng đơn vị ẩn) có giá trị bằng không.


```{.python .input  n=3}
def init_gru_state(batch_size, num_hiddens, ctx):
    return (np.zeros(shape=(batch_size, num_hiddens), ctx=ctx), )
```

<!--
Now we are ready to define the GRU model.
Its structure is the same as the basic RNN cell, except that the update equations are more complex.
-->

Bây giờ chúng ta đã sẵn sàng để định nghĩa mô hình GRU.
Cấu trúc GRU cũng giống một khối RNN cơ bản nhưng với phương trình cập nhật phức tạp hơn.


```{.python .input  n=4}
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = npx.sigmoid(np.dot(X, W_xz) + np.dot(H, W_hz) + b_z)
        R = npx.sigmoid(np.dot(X, W_xr) + np.dot(H, W_hr) + b_r)
        H_tilda = np.tanh(np.dot(X, W_xh) + np.dot(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H,)
```

<!--
### Training and Prediction
-->

### Huấn luyện và Dự đoán

<!--
Training and prediction work in exactly the same manner as before.
After training for one epoch, the perplexity and the output sentence will be like the following.
-->

Việc huấn luyện và dự đoán hoạt động theo cách tương tự như trước đây.
Sau khi huấn luyện cho một epoch, độ hỗn độn (_perplexity_) và câu đầu ra sẽ giống dưới đây.


```{.python .input  n=3}
vocab_size, num_hiddens, ctx = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, ctx, get_params,
                            init_gru_state, gru)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, ctx)
```

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!--
## Concise Implementation
-->

## *dịch tiêu đề phía trên*

<!--
In Gluon, we can directly call the `GRU` class in the `rnn` module.
This encapsulates all the configuration detail that we made explicit above.
The code is significantly faster as it uses compiled operators rather than Python for many details that we spelled out in detail before.
-->

*dịch đoạn phía trên*


```{.python .input  n=9}
gru_layer = rnn.GRU(num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, ctx)
```

<!--
## Summary
-->

## Tóm tắt

<!--
* Gated recurrent neural networks are better at capturing dependencies for time series with large timestep distances.
* Reset gates help capture short-term dependencies in time series.
* Update gates help capture long-term dependencies in time series.
* GRUs contain basic RNNs as their extreme case whenever the reset gate is switched on. They can ignore sequences as needed.
-->

*dịch đoạn phía trên*


<!--
## Exercises
-->

## Bài tập

<!--
1. Compare runtime, perplexity, and the output strings for `rnn.RNN` and `rnn.GRU` implementations with each other.
2. Assume that we only want to use the input for timestep $t'$ to predict the output at timestep $t > t'$. What are the best values for the reset and update gates for each timestep?
3. Adjust the hyperparameters and observe and analyze the impact on running time, perplexity, and the written lyrics.
4. What happens if you implement only parts of a GRU? That is, implement a recurrent cell that only has a reset gate. Likewise, implement a recurrent cell only with an update gate.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->
<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2367)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.

* Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md
-->

* Đoàn Võ Duy Thanh
<!-- Phần 1 -->
* Võ Tấn Phát
* Lê Khắc Hồng Phúc

<!-- Phần 2 -->
* Nguyễn Duy Du
* Nguyễn Văn Cường

<!-- Phần 3 -->
* Nguyễn Duy Du

<!-- Phần 4 -->
* Nguyễn Văn Quang

<!-- Phần 5 -->
* Nguyễn Văn Quang

<!-- Phần 6 -->
*
